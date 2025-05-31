# SPDX-License-Identifier: Apache-2.0
"""
    Lookup buffer for Neuron
"""
import os
import time

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)


class NeuronTransferEngine:

    def __init__(self,
                 remote_ip,
                 device_to_communicator_map,
                 send,
                 nc_offset,
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
                                token=None):
        start_time = time.time()
        # TODO: this assume the devices and commids are identitical
        #   across sequences, which might not be true with data parallel
        if self.local_devices is None:
            self.local_devices = [tensor.device.index for tensor in tensors]

        if self.comm_ids is None:
            self.comm_ids = [
                self.device_to_communicator_map[i] for i in self.local_devices
            ]

        self.engine.queue_transfer_with_token(tensors, offsets, lengths,
                                              peer_devices, self.local_devices,
                                              self.comm_ids, token)
        _duration = time.time() - start_time
        logger.debug("Finished %s %s tensors takes %s ms",
                     'sending' if self.send else 'recving', len(tensors),
                     _duration * 1000)
