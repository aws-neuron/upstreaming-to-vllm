# SPDX-License-Identifier: Apache-2.0
import queue
import threading
import time
from typing import Optional

import torch

from vllm.distributed.kv_transfer.neuron.neuron_transfer_engine import (
    init_transfer_engine)
from vllm.distributed.kv_transfer.neuron.nxdi_kv_map_utils import (
    generate_kv_transfer_sequences_identical_sharding_block_kv,
    setup_transfer_scheme, validate_and_load_kv_map)
from vllm.distributed.kv_transfer.neuron.zmq_util import Dealer, Router
from vllm.logger import init_logger

logger = init_logger(__name__)


class LookupEntry:

    def __init__(self, request_id, block_ids, output_token=None, token=None):
        self.request_id = request_id
        self.output_token = output_token
        self.block_ids = block_ids
        self.transfer_done = False
        self.token = token


class NeuronBuffer:

    def __init__(self,
                 kv_caches,
                 zmq_context,
                 remote_ip,
                 zmq_ip,
                 zmq_port,
                 kv_map_path,
                 is_block_kv_layout,
                 nc_offset,
                 local_ip,
                 local_port,
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
        self.send = send

        self.lock = threading.Lock()

        self.kv_map = None

        self.lookup_dict = {}
        self.lookup_queue: queue.Queue = queue.Queue()

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
            "local kv_map:%s, peer_kv_map: %s",
            remote_id, kv_map, peer_kv_map)
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
            local_nc_offset=local_nc_offset,
            peer_nc_offset=peer_nc_offset)
        logger.info("Done setting up KV transfer scheme")

    def get_transfer_engine(self, remote_id=""):
        return self.transfer_engines[remote_id]

    def generate_transfer_sequences(self, block_ids, remote_id=""):
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
                     block_ids)
        if self.is_block_kv_layout:
            return generate_kv_transfer_sequences_identical_sharding_block_kv(
                self.kv_caches, block_ids)
        else:
            seq_id = block_ids[0]
            return self.transfer_sequences[remote_id][seq_id]

    def get_output_token(self, request_id):
        assert request_id in self.lookup_dict, f"Cannot find \
            request_id {request_id} in lookup_dict"

        return self.lookup_dict[request_id].output_token

    def check_transfer_done(self, request_id, remove=False):
        token = self.lookup_dict[request_id].token
        done = token.is_done()
        if remove and done:
            del self.lookup_dict[request_id]

        return done


class SendBuffer(NeuronBuffer):

    def init_signal_plane(self):
        logger.info("SendBuffer %s init signal plane", self.buffer_id)
        self.router = Router(f"tcp://*:{self.zmq_port}")

        self.send_handler_thread = threading.Thread(target=self.send_handler, daemon=True)
        self.send_handler_thread.start()

    def async_send_kv_caches(self, request_id, block_ids, output_tokens):
        self.lookup_dict[request_id] = LookupEntry(
            request_id,
            block_ids,
            output_tokens,
            token=torch.classes.neuron.CompletionToken())

    def send_handler(self):
        logger.info("start send_handler thread on %s", self.buffer_id)
        try:
            while True:
                # Receive all message parts
                identity, request = self.router.recv_json()

                identity_str = identity.decode('utf-8')

                logger.debug("Buffer %s get message: %s %s", self.buffer_id,
                             identity_str, request)

                if request["type"] == "handshake":
                    self.router.send_json(identity, {
                        "status": "ok",
                        "timestamp": time.time()
                    })
                    logger.info("Get handshake request from receiver: %s",
                                identity_str)
                    continue

                if request["type"] == "kv_map_init":
                    peer_kv_map = request["kv_map"]
                    peer_nc_offset = request["nc_offset"]
                    self.router.send_json(identity, {
                        "kv_map": self.kv_map,
                        "nc_offset": self.nc_offset
                    })
                    logger.info("Get kv_map_init request from receiver: %s",
                                identity_str)

                    self.setup_kv_scheme_and_transfer_engine(
                        self.kv_map,
                        peer_kv_map,
                        remote_id=identity_str,
                        local_nc_offset=self.nc_offset,
                        peer_nc_offset=peer_nc_offset)
                    continue

                assert request["type"] == "lookup", \
                    f"invalid request type {request['type']}"

                request_id = request["request_id"]

                look_up_success = True
                entry: Optional[LookupEntry] = None

                if request_id in self.lookup_dict:
                    entry = self.lookup_dict[request_id]
                else:
                    look_up_success = False

                if not look_up_success:
                    self.router.send_json(identity, {"success": False})
                    continue

                assert entry is not None
                self.router.send_json(identity, {
                    "success": True,
                    "output_token": entry.output_token.item()
                })

                kv_caches, offsets, lengths, peer_devices = \
                    self.generate_transfer_sequences(entry.block_ids,
                                                     remote_id=identity_str)

                self.get_transfer_engine(
                    remote_id=identity_str).transfer_neuron_tensors(
                        kv_caches,
                        offsets,
                        lengths,
                        peer_devices,
                        send=True,
                        token=entry.token,
                        is_block_kv_layout=self.is_block_kv_layout)

        except RuntimeError as e:
            if 'Connection closed by peer' not in str(e):
                raise e

        logger.info("Closing send_handler thread")


class RecvBuffer(NeuronBuffer):

    def init_signal_plane(self):
        logger.info("RecvBuffer %s init signal plane", self.buffer_id)
        self.dealer = Dealer(self.buffer_id,
                             f"tcp://{self.zmq_ip}:{self.zmq_port}")

        self.dealer.send_json({"type": "handshake"})

        response = self.dealer.recv_json()
        logger.info(
            "RecvBuffer get response from SendBuffer, " \
            "self.buffer_id %s: response %s",
            self.buffer_id, response)

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

        logger.info("RecvBuffer %s sent kv_map: %s", self.buffer_id,
                    self.kv_map)
        logger.info("RecvBuffer %s received peer_kv_map: %s", self.buffer_id,
                    peer_kv_map)
        self.setup_kv_scheme_and_transfer_engine(
            self.kv_map,
            peer_kv_map,
            remote_id="",
            local_nc_offset=self.nc_offset,
            peer_nc_offset=peer_nc_offset)

        self.recv_handler_thread = threading.Thread(target=self.recv_handler, daemon=True)
        self.recv_handler_thread.start()

    def async_recv_kv_caches(self, request_id, block_ids):
        entry = LookupEntry(request_id,
                            block_ids,
                            token=torch.classes.neuron.CompletionToken())
        self.lookup_dict[request_id] = entry
        self.lookup_queue.put((request_id, entry))

    def recv_handler(self):
        logger.info("start recv_hanlder thread on %s", self.buffer_id)
        try:
            while True:
                request_id, entry = self.lookup_queue.get()

                logger.debug("Try lookup with request_id %s", request_id)

                self.dealer.send_json({
                    "type": "lookup",
                    "request_id": request_id
                })

                response = self.dealer.recv_json()

                if not response["success"]:
                    self.lookup_queue.put((request_id, entry))
                    continue

                logger.debug("Lookup with request_id %s is successful",
                             request_id)

                entry.output_token = torch.tensor(
                    response["output_token"]).unsqueeze(0)

                kv_caches, offsets, lengths, peer_devices = \
                    self.generate_transfer_sequences(entry.block_ids)

                self.get_transfer_engine().transfer_neuron_tensors(
                    kv_caches,
                    offsets,
                    lengths,
                    peer_devices,
                    send=False,
                    token=entry.token,
                    is_block_kv_layout=self.is_block_kv_layout)

        except RuntimeError as e:
            if 'Connection closed by peer' not in str(e):
                raise e

        logger.info("Closing recv_handler thread")
