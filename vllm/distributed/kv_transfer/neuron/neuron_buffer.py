# SPDX-License-Identifier: Apache-2.0
import queue
import threading
import time
from typing import Optional

import torch
import zmq

from vllm.distributed.kv_transfer.neuron.neuron_transfer_engine import (
    NeuronTransferEngine)
from vllm.distributed.kv_transfer.neuron.nxdi_kv_map_utils import (
    generate_kv_transfer_sequences_identical_sharding_block_kv,
    setup_transfer_scheme, validate_and_load_kv_map)
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
                 zmq_context,
                 remote_ip,
                 zmq_ip,
                 zmq_port,
                 kv_map_path,
                 is_block_kv_layout,
                 nc_offset,
                 send=True):
        logger.info(
            "initialize %s buffer, " \
            "with server zmq %s:%s, transfer to remote_ip %s ",
            'send' if send else 'recv', zmq_ip, zmq_port, remote_ip,
        )
        self.is_block_kv_layout = is_block_kv_layout
        if send:
            self.socket = zmq_context.socket(zmq.REP)
            self.socket.bind(f"tcp://*:{zmq_port}")
            # First handshake
            response = self.socket.recv_string()
            logger.info("Get handshake msg %s from client", response)
            self.socket.send_string("hello from server")
            # Second exchange for nc_offset
            peer_nc_offset = self.socket.recv_string()
            logger.info("Get peer nc offset %s from client", peer_nc_offset)
            self.socket.send_string(str(nc_offset))
        else:
            self.socket = zmq_context.socket(zmq.REQ)
            self.socket.connect(f"tcp://{zmq_ip}:{zmq_port}")
            # First handshake
            self.socket.send_string("hello from client")
            response = self.socket.recv_string()
            logger.info("Get handshake msg %s from server", response)
            # Second exchange for nc_offset
            self.socket.send_string(str(nc_offset))
            peer_nc_offset = self.socket.recv_string()
            logger.info("Get peer nc offset %s from server", peer_nc_offset)

        # Try to load kv_map_path from kv_transfer_config if specified
        # and also exchange KV maps
        if kv_map_path:
            kv_map = validate_and_load_kv_map(kv_map_path)
            # Exchange KV maps
            if send:
                # on sender receive first and send later due to ZMQ rules
                peer_kv_map = self.socket.recv_pyobj()
                self.socket.send_pyobj(kv_map)
            else:
                self.socket.send_pyobj(kv_map)
                peer_kv_map = self.socket.recv_pyobj()
            logger.info("sent kv_map: %s", kv_map)
            logger.info("received peer_kv_map: %s", peer_kv_map)
        else:
            # If KV map path is not specified set up default scheme by setting
            # to None.
            # Note we assume that sender and receiver either both have or
            # both don't have a KV map.
            kv_map = None
            peer_kv_map = None
        logger.info("Done exchanging KV maps")
        self.producer_kv_map = kv_map if send else peer_kv_map
        self.consumer_kv_map = peer_kv_map if send else kv_map
        self.is_kv_producer = send

        self.lookup_dict = {}
        self.lookup_queue: queue.Queue = queue.Queue()

        nc_offset = int(nc_offset)
        peer_nc_offset = int(peer_nc_offset)
        visible_core_count = torch.classes.neuron.Runtime(
        ).get_visible_nc_count()
        logger.info("Initializing async send recv on %s cores",
                    visible_core_count)
        for i in range(visible_core_count):
            torch.ops.neuron._nrt_async_init_neuron(i + nc_offset)
        # make device to communicator map given scheme
        # for now just one to one
        device_to_communicator_map = {}
        if send:
            comm_create_func = torch.ops.neuron._nrt_create_send_communicator
        else:
            comm_create_func = torch.ops.neuron._nrt_create_recv_communicator

        # TODO support mapping when producer and consumer kv map are not None
        for i in range(visible_core_count):
            device_to_communicator_map[i] = comm_create_func(
                remote_ip, i + peer_nc_offset, i + nc_offset)

        for i, comm in device_to_communicator_map.items():
            iters = 0
            sleep_time = 0.1
            while True:
                if iters * sleep_time == 60:
                    raise TimeoutError(
                        "Communicator establishment timed out after 1 minute.")
                time.sleep(sleep_time)
                if torch.ops.neuron._nrt_test_communicator(comm):
                    break
                iters += 1

        self.transfer_engine = NeuronTransferEngine(
            remote_ip,
            device_to_communicator_map,
            send,
            nc_offset,
            default_batch=visible_core_count * 128)

        # TODO: capture thread error and clean up buffer properly
        if send:
            self.send_handler_thread = threading.Thread(
                target=self.send_handler, daemon=True)
            self.send_handler_thread.start()
        else:
            self.recv_handler_thread = threading.Thread(
                target=self.recv_handler, daemon=True)
            self.recv_handler_thread.start()

        self.kv_caches = None

    def setup_kv_scheme(self):
        # Try to load kv_map_path from kv_transfer_config if specified
        assert self.kv_caches is not None
        max_num_seqs = self.kv_caches[0].shape[0]
        self.transfer_sequences = setup_transfer_scheme(
            self.kv_caches,
            producer_kv_map=self.producer_kv_map,
            consumer_kv_map=self.consumer_kv_map,
            max_num_seqs=max_num_seqs,
            is_producer=self.is_kv_producer)
        logger.info("Done setting up KV transfer scheme")

    def register_kv_caches(self, kv_caches):
        self.kv_caches = kv_caches
        # TODO: move setup_kv_scheme to init. Currently it relies on kv_caches
        # being available to get the shape information and also it populates the
        # tensors in the transfer scheme. We can consider storing a indexing
        # instead of storing the actual tensors in the transfer sequence list.
        self.setup_kv_scheme()

    def generate_transfer_sequences(self, block_ids):
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
            return self.transfer_sequences[seq_id]

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

#  =============== send buffer implementation ===========================

    def async_send_kv_caches(self, request_id, block_ids, output_tokens):
        self.lookup_dict[request_id] = LookupEntry(
            request_id,
            block_ids,
            output_tokens,
            token=torch.classes.neuron.CompletionToken())

    def send_handler(self):
        logger.info("start send_handler thread")
        try:
            while True:
                request_json = self.socket.recv_json()

                request_id = request_json["request_id"]

                look_up_success = True
                entry: Optional[LookupEntry] = None

                if request_id in self.lookup_dict:
                    entry = self.lookup_dict[request_id]
                else:
                    look_up_success = False

                if not look_up_success:
                    self.socket.send_json({"success": False})
                    continue

                assert entry is not None
                self.socket.send_json({
                    "success": True,
                    "output_token": entry.output_token.item()
                })

                kv_caches, offsets, lengths, peer_devices = \
                    self.generate_transfer_sequences(entry.block_ids)

                self.transfer_engine.transfer_neuron_tensors(kv_caches,
                                                             offsets,
                                                             lengths,
                                                             peer_devices,
                                                             send=True,
                                                             token=entry.token)

        except RuntimeError as e:
            if 'Connection closed by peer' not in str(e):
                raise e

        logger.info("Closing send_handler thread")


# =============== recv buffer implementation ================

    def async_recv_kv_caches(self, request_id, block_ids):
        entry = LookupEntry(request_id,
                            block_ids,
                            token=torch.classes.neuron.CompletionToken())
        self.lookup_dict[request_id] = entry
        self.lookup_queue.put((request_id, entry))

    def recv_handler(self):
        try:
            while True:
                request_id, entry = self.lookup_queue.get()

                logger.debug("try lookup with request_id %s", request_id)

                self.socket.send_json({"request_id": request_id})

                response = self.socket.recv_json()

                if not response["success"]:
                    self.lookup_queue.put((request_id, entry))
                    continue

                entry.output_token = torch.tensor(
                    response["output_token"]).unsqueeze(0)

                kv_caches, offsets, lengths, peer_devices = \
                    self.generate_transfer_sequences(entry.block_ids)

                self.transfer_engine.transfer_neuron_tensors(kv_caches,
                                                             offsets,
                                                             lengths,
                                                             peer_devices,
                                                             send=False,
                                                             token=entry.token)

        except RuntimeError as e:
            if 'Connection closed by peer' not in str(e):
                raise e

        logger.info("Closing recv_handler thread")
