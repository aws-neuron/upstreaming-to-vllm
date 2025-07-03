# SPDX-License-Identifier: Apache-2.0
"""
    Lookup buffer for Neuron
"""
import os
import time

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)


def init_transfer_engine(device_pairs,
                         remote_ip,
                         send,
                         per_layer_kv_transfer=False,
                         local_nc_offset=0,
                         peer_nc_offset=0):
    # make device to communicator map given scheme
    # for now just one to one
    device_to_communicator_map = {}  # local_lnc: communicator_id
    if send:
        comm_create_func = torch.ops.neuron._nrt_create_send_communicator
    else:
        comm_create_func = torch.ops.neuron._nrt_create_recv_communicator

    device_pairs = sorted(device_pairs)

    # TODO support mapping when producer and consumer kv map are not None
    logger.info(
        "Creating %s communicators to remote ip %s with device_pairs %s "
        " (offset %s,%s) ...", "send" if send else "recv", remote_ip,
        device_pairs, local_nc_offset, peer_nc_offset)
    for peer_lnc, local_lnc in device_pairs:
        if local_lnc not in device_to_communicator_map:
            device_to_communicator_map[local_lnc] = comm_create_func(
                remote_ip, peer_lnc + peer_nc_offset,
                local_lnc + local_nc_offset)

    logger.info("Testing communicators...")
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
    logger.info("Communicators tested successfully")
    transfer_engine = NeuronTransferEngine(
        remote_ip,
        device_to_communicator_map,
        send,
        nc_offset=local_nc_offset,
        per_layer_kv_transfer=per_layer_kv_transfer,
        default_batch=len(set(device_to_communicator_map)) * 128)
    return transfer_engine


class NeuronTransferEngine:

    def __init__(self,
                 remote_ip,
                 device_to_communicator_map,
                 send,
                 nc_offset=0,
                 per_layer_kv_transfer=False,
                 default_batch=64):
        logger.info("Setting up Neuron Transfer Engine")
        batch_transfer_size = int(
            os.environ.get("BATCH_TRANSFER", default_batch))

        self.engine = torch.classes.neuron.NeuronTransferEngine(
            batch_transfer_size, remote_ip, send, nc_offset)
        self.current_task_id = None
        self.send = send
        self.device_to_communicator_map = device_to_communicator_map
        self.local_devices = None
        self.comm_ids = None
        self.per_layer_kv_transfer = per_layer_kv_transfer

    def transfer_neuron_tensors(self,
                                tensors,
                                offsets,
                                lengths,
                                peer_devices,
                                completion_token=None,
                                is_block_kv_layout=False,
                                completion_count=0):
        start_time = time.time()
        # TODO: this assume the devices and commids are identitical
        #   across sequences, which might not be true with data parallel
        if is_block_kv_layout:
            # In block kv cache layout, the input tensor list can change
            # (specifically the total number of the tensors can change with
            # the number of block ids drawn from each tensor), so we need to
            # recompute the local devices and comm ids
            self.local_devices = [tensor.device.index for tensor in tensors]
            self.comm_ids = [
                self.device_to_communicator_map[i] for i in self.local_devices
            ]
        else:
            if self.local_devices is None:
                self.local_devices = [
                    tensor.device.index for tensor in tensors
                ]

            if self.comm_ids is None:
                self.comm_ids = [
                    self.device_to_communicator_map[i]
                    for i in self.local_devices
                ]
        logger.debug("try initiating transfer with completion count %s",
                     completion_count)
        use_queue = os.environ.get("DI_USE_QUEUE", None) == "1"
        completion_time_out = int(
            os.environ.get("DI_COMPLETION_TIMEOUT", 50))
        logger.debug("use queue based transfer: %s", use_queue)
        self.engine.queue_transfer_with_token(
            tensors, offsets, lengths, peer_devices, self.local_devices,
            self.comm_ids, completion_count, completion_token, use_queue,
            completion_time_out)
        _duration = time.time() - start_time
        logger.debug(
            "initiated %s %s tensors takes %s ms (still transferring)",
            'sending' if self.send else 'recving', len(tensors),
            _duration * 1000)

        # force completion for per layer debug purpose
        if os.environ.get("DI_FORCE_COMPLETION", None) == "1":
            while not completion_token.is_done():
                time.sleep(0.005)
            _duration = time.time() - start_time
            logger.debug("transfer %s %s tensors takes %s ms",
                         'sending' if self.send else 'recving', len(tensors),
                         _duration * 1000)
