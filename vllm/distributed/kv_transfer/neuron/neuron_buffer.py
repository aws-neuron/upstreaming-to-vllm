# SPDX-License-Identifier: Apache-2.0
import os
import queue
import threading
import time
from enum import Enum

import torch

from vllm.distributed.kv_transfer.neuron.neuron_transfer_engine import (
    init_transfer_engine)
from vllm.distributed.kv_transfer.neuron.nxdi_kv_map_utils import (
    generate_kv_transfer_sequences_identical_sharding_block_kv,
    setup_transfer_scheme, validate_and_load_kv_map)
from vllm.distributed.kv_transfer.neuron.zmq_util import Dealer, Router
from vllm.logger import init_logger

logger = init_logger(__name__)


class Status(Enum):
    NOT_STARTED = 0
    PENDING = 1
    DONE = 2


class LookupEntry:

    def __init__(self,
                 request_id,
                 block_ids,
                 output_token=None,
                 completion_token=None,
                 completion_count=0):
        self.request_id = request_id
        self.output_token = output_token
        self.block_ids = block_ids
        self.block_ids_in_peer_device = None
        self.token_transfer_status = Status.NOT_STARTED
        self.completion_token = completion_token
        self.completion_count = completion_count

    def set_block_ids_in_peer_device(self, block_ids_in_peer_device):
        self.block_ids_in_peer_device = block_ids_in_peer_device


class LookupTask:

    def __init__(self, request_type, request_id, entry, priority=1):
        self.priority = priority
        self.request_type = request_type
        self.request_id = request_id
        self.entry = entry

    def __str__(self):
        return f"Task(request_type={self.request_type}, \
            request_id={self.request_id})"

    def __lt__(self, other):
        return self.priority < other.priority


class NeuronBuffer:

    def __init__(self,
                 kv_caches,
                 zmq_context,
                 remote_ip,
                 zmq_ip,
                 zmq_port,
                 kv_map_path,
                 is_block_kv_layout,
                 local_ip,
                 local_port,
                 nc_offset=0,
                 per_layer_kv_transfer=False,
                 send=True):
        logger.info(
            "initialize %s buffer, " \
            "with server zmq %s:%s, transfer to remote_ip %s ",
            'send' if send else 'recv', zmq_ip, zmq_port, remote_ip,
        )
        self.is_block_kv_layout = is_block_kv_layout
        self.nc_offset = nc_offset

        self.kv_caches = kv_caches
        self.buffer_id = f"{local_ip}:{local_port}"
        self.zmq_context = zmq_context
        # Note: this remote_ip should only valid for RecvBuffer
        self.remote_ip = remote_ip
        self.zmq_ip = zmq_ip
        self.zmq_port = zmq_port
        self.per_layer_kv_transfer = per_layer_kv_transfer
        self.send = send

        self.lock = threading.Lock()

        self.kv_map = None

        self.lookup_dict = {}
        self.lookup_queue: queue.PriorityQueue = queue.PriorityQueue()

        # Try to load kv_map_path from kv_transfer_config if specified
        if kv_map_path:
            self.kv_map = validate_and_load_kv_map(kv_map_path)

        self.transfer_engines = {}
        self.transfer_sequences = {}

        logger.info(
            "initializing %s buffer with local ip:port %s, " \
            "with zmq server %s:%s, transfer to remote_ip %s ",
            'send' if self.send else 'recv', self.buffer_id, self.zmq_ip,
            self.zmq_port, self.remote_ip,
        )

        self.init_signal_plane()

    def init_signal_plane(self):
        raise NotImplementedError("init_signal_plane is not implemented")

    def setup_kv_scheme_and_transfer_engine(self,
                                            kv_map,
                                            peer_kv_map,
                                            remote_id,
                                            local_nc_offset=0,
                                            peer_nc_offset=0):
        logger.info(
            "Setting up kv_scheme with peer <ip:port>: %s, " \
            "local kv_map:%s, peer_kv_map: %s, " \
            "local nc_offset %s, peer nc_offset %s",
            remote_id, kv_map, peer_kv_map,
            local_nc_offset, peer_nc_offset)
        # Try to load kv_map_path from kv_transfer_config if specified
        assert self.kv_caches is not None
        max_num_seqs = self.kv_caches[0].shape[0]

        producer_kv_map = kv_map if self.send else peer_kv_map
        consumer_kv_map = peer_kv_map if self.send else kv_map
        transfer_sequences, transfer_device_pairs = setup_transfer_scheme(
            self.kv_caches,
            producer_kv_map=producer_kv_map,
            consumer_kv_map=consumer_kv_map,
            max_num_seqs=max_num_seqs,
            is_producer=self.send)

        self.transfer_sequences[remote_id] = transfer_sequences
        remote_ip = remote_id.split(":")[0] if self.send else self.remote_ip
        self.transfer_engines[remote_id] = init_transfer_engine(
            transfer_device_pairs,
            remote_ip,
            self.send,
            per_layer_kv_transfer=self.per_layer_kv_transfer,
            local_nc_offset=local_nc_offset,
            peer_nc_offset=peer_nc_offset)
        logger.info("Done setting up KV transfer scheme")

    def get_transfer_engine(self, remote_id=""):
        return self.transfer_engines[remote_id]

    def generate_transfer_sequences(self, entry, remote_id=""):
        """
        return generated transfer sequence by indexing with seq_id.

        for contiguous kv cache layout, the transfer sequence can be
        pre-calculated.

        for blockwise kv cache layout, the transfer sequence (particularly
        offsets) needs to be generated based on the block ids of each sequence.
        Node the offsets can be different between the prefill node and the
        decodenode for the same sequence, depending on what physical blocks
        are allocated for the sequence on the two nodes.

        returns
            kv_caches: list of KV cache tensors
            offsets: list of ints, offset of data to transfer in each KV cache
            lengths: list of ints, length of data to transfer in each KV cache
            peer_devices: list of ints, peer device index
        """
        logger.debug("generate_transfer_sequences with block_ids %s",
                     entry.block_ids)
        if self.is_block_kv_layout:
            return generate_kv_transfer_sequences_identical_sharding_block_kv(
                self.kv_caches, entry.block_ids,
                entry.block_ids_in_peer_device)
        else:
            seq_id = entry.block_ids[0]
            return self.transfer_sequences[remote_id][seq_id]

    def get_output_token(self, request_id):
        assert request_id in self.lookup_dict, f"Cannot find \
            request_id {request_id} in lookup_dict"

        return self.lookup_dict[request_id].output_token

    def check_transfer_done(self, request_id, remove=False):
        entry = self.lookup_dict[request_id]
        completion_token = entry.completion_token
        done = completion_token.is_done()

        if not done:
            return False

        if done and self.per_layer_kv_transfer:

            if self.send:
                if entry.token_transfer_status != Status.DONE:
                    return False
            else:
                # initiate token lookup with higher priority
                # as the token should be ready soon once KV cache
                # are all transferred. And new KV cache transfer shouldn't
                # start before that.
                if entry.token_transfer_status == Status.NOT_STARTED:
                    self.lookup_queue.put(
                        LookupTask("lookup_token",
                                   request_id,
                                   entry,
                                   priority=0))
                    entry.token_transfer_status = Status.PENDING
                    return False

                elif entry.token_transfer_status == Status.PENDING:
                    return False
                else:
                    assert entry.output_token is not None

        if remove and done:
            del self.lookup_dict[request_id]

        return True


class SendBuffer(NeuronBuffer):

    def init_signal_plane(self):
        logger.info("[SendBuffer %s] init signal plane", self.buffer_id)
        self.router = Router(f"tcp://*:{self.zmq_port}")

        self.send_handler_thread = threading.Thread(target=self.send_handler,
                                                    daemon=True)
        self.send_handler_thread.start()

    def set_output_token(self, request_id, output_token):
        logger.debug("[SendBuffer %s] set output token for request_id %s",
                     self.buffer_id, request_id)
        self.lookup_dict[request_id].output_token = output_token

    def async_send_kv_caches(self,
                             request_id,
                             block_ids,
                             output_tokens,
                             completion_count=0):
        self.lookup_dict[request_id] = LookupEntry(
            request_id,
            block_ids,
            output_tokens,
            completion_count=completion_count,
            completion_token=torch.classes.neuron.CompletionToken())

    def _process_lookup_all(self, identity, request):
        identity_str = identity.decode('utf-8')
        request_id = request["request_id"]
        block_ids_in_peer_device = request["block_ids_in_peer_device"]

        if request_id not in self.lookup_dict:
            self.router.send_json(identity, {"success": False})
            return

        entry = self.lookup_dict[request_id]

        self.router.send_json(identity, {
            "success": True,
            "output_token": entry.output_token.item() \
                if entry.output_token else None,
            "block_ids_in_peer_device": entry.block_ids,
        })

        entry.set_block_ids_in_peer_device(block_ids_in_peer_device)
        kv_caches, offsets, lengths, peer_devices = \
            self.generate_transfer_sequences(entry,
                                            remote_id=identity_str)

        self.get_transfer_engine(
            remote_id=identity_str).transfer_neuron_tensors(
                kv_caches,
                offsets,
                lengths,
                peer_devices,
                completion_token=entry.completion_token,
                completion_count=entry.completion_count \
                    if self.per_layer_kv_transfer else 0,
                is_block_kv_layout=self.is_block_kv_layout)

    def _process_lookup_token(self, identity, request):
        request_id = request["request_id"]

        if request_id not in self.lookup_dict or \
            self.lookup_dict[request_id].output_token is None:
            self.router.send_json(identity, {"success": False})
            return

        entry = self.lookup_dict[request_id]

        self.router.send_json(identity, {
            "success": True,
            "output_token": entry.output_token.item()
        })

        entry.token_transfer_status = Status.DONE

    def send_handler(self):
        logger.info("[SendBuffer %s] start send_handler thread",
                    self.buffer_id)
        try:
            while True:
                # Receive all message parts
                identity, request = self.router.recv_json()

                identity_str = identity.decode('utf-8')

                logger.debug("[SendBuffer %s] get message: %s %s",
                             self.buffer_id, identity_str, request)

                if request["type"] == "handshake":
                    self.router.send_json(identity, {
                        "status": "ok",
                        "timestamp": time.time()
                    })
                    logger.info(
                        "[SendBuffer %s] Get handshake request " \
                        "from receiver: %s",
                        self.buffer_id, identity_str)
                    continue

                if request["type"] == "kv_map_init":
                    peer_kv_map = request["kv_map"]
                    peer_nc_offset = request["nc_offset"]
                    self.router.send_json(identity, {
                        "kv_map": self.kv_map,
                        "nc_offset": self.nc_offset
                    })
                    logger.info(
                        "[SendBuffer %s] Get kv_map_init request " \
                        "from receiver: %s, with peer kv_map %s "
                        "and nc_offset %s",
                        self.buffer_id, identity_str,
                        peer_kv_map, peer_nc_offset)

                    self.setup_kv_scheme_and_transfer_engine(
                        self.kv_map,
                        peer_kv_map,
                        remote_id=identity_str,
                        local_nc_offset=self.nc_offset,
                        peer_nc_offset=peer_nc_offset)
                    continue

                if request["type"] == "lookup_all":
                    self._process_lookup_all(identity, request)
                    continue

                if request["type"] == "lookup_token":
                    self._process_lookup_token(identity, request)
                    continue

                raise Exception(f"invalid request type {request['type']}")

        except RuntimeError as e:
            if 'Connection closed by peer' not in str(e):
                raise e

        logger.info("[SendBuffer %s] Closing send_handler thread",
                    self.buffer_id)


class RecvBuffer(NeuronBuffer):

    def init_signal_plane(self):
        logger.info("[RecvBuffer %s] init signal plane", self.buffer_id)

        self.remote_zmq_server = f"{self.zmq_ip}:{self.zmq_port}"
        self.dealer = Dealer(self.buffer_id, f"tcp://{self.remote_zmq_server}")

        self.dealer.send_json({"type": "handshake"})

        response = self.dealer.recv_json()
        logger.info("[RecvBuffer %s] get response %s", self.buffer_id,
                    response)

        # Exchange KV maps
        peer_kv_map = None
        self.dealer.send_json({
            "type": "kv_map_init",
            "kv_map": self.kv_map,
            "nc_offset": self.nc_offset
        })
        response = self.dealer.recv_json()
        peer_kv_map = response["kv_map"]
        peer_nc_offset = response["nc_offset"]

        logger.info("[RecvBuffer %s] sent kv_map: %s and nc_offset %s",
                    self.buffer_id, self.kv_map, self.nc_offset)
        logger.info("[RecvBuffer %s] received peer_kv_map: %s" \
                    " and peer_nc_offset %s", self.buffer_id,
                    peer_kv_map, peer_nc_offset)
        self.setup_kv_scheme_and_transfer_engine(
            self.kv_map,
            peer_kv_map,
            remote_id="",
            local_nc_offset=self.nc_offset,
            peer_nc_offset=peer_nc_offset)

        self.recv_handler_thread = threading.Thread(target=self.recv_handler,
                                                    daemon=True)
        self.recv_handler_thread.start()

    def async_recv_kv_caches(self, request_id, block_ids):
        entry = LookupEntry(
            request_id,
            block_ids,
            completion_token=torch.classes.neuron.CompletionToken())
        self.lookup_dict[request_id] = entry
        self.lookup_queue.put(
            LookupTask("lookup_all", request_id, entry, priority=1))

    def _process_lookup_all(self, entry, response):
        logger.debug(
            "[Recv Buffer %s] got success lookup_all response %s " \
            "from peer %s:%s",
            self.buffer_id, response, self.zmq_ip, self.zmq_port)
        if response["output_token"] is None:
            # this should only happen at per layer transfer
            # TODO: add proper error handling
            if not self.per_layer_kv_transfer:
                logger.error("Get None output token from peer with " \
                    "per layer transfer disable")
        else:
            entry.output_token = torch.tensor(
                response["output_token"]).unsqueeze(0)
        entry.set_block_ids_in_peer_device(
            response["block_ids_in_peer_device"])

        kv_caches, offsets, lengths, peer_devices = \
            self.generate_transfer_sequences(entry)

        # no need to wait on completion for recv buffer
        self.get_transfer_engine().transfer_neuron_tensors(
            kv_caches,
            offsets,
            lengths,
            peer_devices,
            completion_count=0,
            completion_token=entry.completion_token,
            is_block_kv_layout=self.is_block_kv_layout)

    def _process_lookup_token(self, entry, response):
        logger.debug(
            "[RecvBuffer %s] got success lookup token response %s " \
            "from peer %s:%s",
            self.buffer_id, response, self.zmq_ip, self.zmq_port)
        entry.output_token = torch.tensor(
            response["output_token"]).unsqueeze(0)
        entry.token_transfer_status = Status.DONE

    def recv_handler(self):
        logger.info("[RecvBuffer %s] start recv_hanlder thread",
                    self.buffer_id)
        try:
            while True:
                task = self.lookup_queue.get()
                assert task.request_type in ["lookup_all", "lookup_token"]

                logger.debug(
                    "[RecvBuffer %s] send request to " \
                    "remote_zmq_server %s: %s",
                    self.buffer_id, self.remote_zmq_server, task)
                self.dealer.send_json({
                    "request_id": task.request_id,
                    "block_ids_in_peer_device": task.entry.block_ids,
                    "type": task.request_type,
                })

                response = self.dealer.recv_json()

                logger.debug(
                    "[RecvBuffer %s] get response from " \
                    "remote_zmq_server %s: %s",
                    self.buffer_id, self.remote_zmq_server, response)

                if response["success"]:
                    logger.debug(
                        "[RecvBuffer %s] lookup [%s] with request_id [%s]"
                        " is successful", self.buffer_id, task.request_type,
                        task.request_id)
                    if task.request_type == "lookup_all":
                        self._process_lookup_all(task.entry, response)
                    else:
                        self._process_lookup_token(task.entry, response)
                else:
                    logger.debug(
                        "[RecvBuffer %s] lookup [%s] with request_id [%s]"
                        " failed", self.buffer_id, task.request_type,
                        task.request_id)
                    self.lookup_queue.put(task)
                    if os.environ.get("DI_DEBUG_FAILED_SLEEP", None):
                        t = int(os.environ.get("DI_DEBUG_FAILED_SLEEP", "1"))
                        logger.debug("[RecvBuffer %s] sleep for %s s",
                                     self.buffer_id, t)
                        time.sleep(t)

                    continue

        except RuntimeError as e:
            if 'Connection closed by peer' not in str(e):
                raise e

        logger.info("[RecvBuffer %s] Closing recv_handler thread",
                    self.buffer_id)
