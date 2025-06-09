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
        "Creating %s communicators to remote ip %s with device_pairs %s ...",
        "send" if send else "recv", remote_ip, device_pairs)
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
        default_batch=len(set(device_to_communicator_map)) * 128)
    return transfer_engine


class NeuronTransferEngine:

    def __init__(self,
                 remote_ip,
                 device_to_communicator_map,
                 send,
                 nc_offset=0,
                 default_batch=64):
        logger.info("Setting up Neuron Transfer Engine")
        batch_transfer_size = int(
            os.environ.get("BATCH_TRANSFER", default_batch))
        self.engine = torch.classes.neuron.NeuronTransferEngine(
            batch_transfer_size, remote_ip, send, True, nc_offset)
        self.current_task_id = None
        self.send = send
        self.device_to_communicator_map = device_to_communicator_map

        self.local_devices = None
        self.comm_ids = None

    def transfer_neuron_tensors(self,
                                tensors,
                                offsets,
                                lengths,
                                peer_devices,
                                send=True,
                                token=None,
                                is_block_kv_layout=False):
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

        self.engine.queue_transfer_with_token(tensors, offsets, lengths,
                                              peer_devices, self.local_devices,
                                              self.comm_ids, token)
        _duration = time.time() - start_time
        logger.debug("Finished %s %s tensors takes %s ms",
                     'sending' if self.send else 'recving', len(tensors),
                     _duration * 1000)
