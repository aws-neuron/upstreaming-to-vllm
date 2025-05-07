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
    setup_transfer_scheme, validate_and_load_kv_map)
from vllm.logger import init_logger

logger = init_logger(__name__)


class LookupEntry:

    def __init__(self, request_id, block_ids, output_token=None):
        self.request_id = request_id
        self.output_token = output_token
        self.block_ids = block_ids
        self.transfer_done = False


class NeuronBuffer:

    def __init__(self,
                 zmq_context,
                 remote_ip,
                 zmq_ip,
                 zmq_port,
                 kv_map_path,
                 send=True):
        logger.info(
            "initialize %s buffer, " \
            "with server zmq %s:%s, transfer to remote_ip %s ",
            'send' if send else 'recv', zmq_ip, zmq_port, remote_ip,
        )
        if send:
            self.socket = zmq_context.socket(zmq.REP)
            self.socket.bind(f"tcp://*:{zmq_port}")
            response = self.socket.recv_string()
            logger.info("Get handshake msg %s from client", response)
            self.socket.send_json("hello from server")
        else:
            self.socket = zmq_context.socket(zmq.REQ)
            self.socket.connect(f"tcp://{zmq_ip}:{zmq_port}")
            self.socket.send_string("hello from client")
            response = self.socket.recv_string()
            logger.info("Get handshake msg %s from server", response)

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

        self.buffer_lock = threading.Lock()
        self.lookup_dict = {}
        self.lookup_queue: queue.Queue = queue.Queue()

        self.transfer_engine = NeuronTransferEngine(remote_ip)

        # TODO: capture thread error and clean up buffer properly
        if send:
            self.send_handler_thread = threading.Thread(
                target=self.send_handler)
            self.send_handler_thread.start()
        else:
            self.recv_handler_thread = threading.Thread(
                target=self.recv_handler)
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
        return pre-generated transfer sequence by indexing with seq_id

        TODO: modify once we support block-wise kv cache
        """
        seq_id = block_ids[0]
        return self.transfer_sequences[seq_id]

    def get_output_token(self, request_id):
        assert request_id in self.lookup_dict, f"Cannot find \
            request_id {request_id} in lookup_dict"

        return self.lookup_dict[request_id].output_token

    def check_transfer_done(self, request_id, remove=False):
        if request_id not in self.lookup_dict:
            return False

        done = self.lookup_dict[request_id].transfer_done

        if remove and done:
            with self.buffer_lock:
                del self.lookup_dict[request_id]

        return done

#  =============== send buffer implementation ===========================

    def async_send_kv_caches(self, request_id, block_ids, output_tokens):
        with self.buffer_lock:
            self.lookup_dict[request_id] = LookupEntry(request_id, block_ids,
                                                       output_tokens)

    def send_handler(self):
        logger.info("start send_handler thread")
        try:
            while True:
                request_json = self.socket.recv_json()

                request_id = request_json["request_id"]

                look_up_success = True
                entry: Optional[LookupEntry] = None
                with self.buffer_lock:

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
                                                             send=True)

                entry.transfer_done = True
        except RuntimeError as e:
            if 'Connection closed by peer' not in str(e):
                raise e

        logger.info("Closing send_handler thread")


# =============== recv buffer implementation ================

    def async_recv_kv_caches(self, request_id, block_ids):
        entry = LookupEntry(request_id, block_ids)
        self.lookup_dict[request_id] = entry
        self.lookup_queue.put((request_id, entry))

    def recv_handler(self):
        try:
            while True:
                try:
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
                                                                 send=False)

                    entry.transfer_done = True
                except queue.Empty:
                    # sleep for 1 ms to avoid contention buffer_lock
                    time.sleep(0.001)
                    pass

        except RuntimeError as e:
            if 'Connection closed by peer' not in str(e):
                raise e

        logger.info("Closing recv_handler thread")
