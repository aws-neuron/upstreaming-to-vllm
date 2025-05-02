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

    def __init__(self, remote_ip):
        self.remote_ip = remote_ip

    def transfer_neuron_tensors(self,
                                tensors,
                                offsets,
                                lengths,
                                peer_devices,
                                send=True):
        logger.debug("Start %s %s tensors on Neuron",
                     'sending' if send else 'recving', len(tensors))
        i = 0
        num_tensors = len(tensors)
        start_time = time.time()

        batch_transfer_size = int(os.environ.get("BATCH_TRANSFER", "64"))
        devices = [tensor.device.index for tensor in tensors]
        while i < num_tensors:
            logger.debug("transfer with C++ batch")
            start = i
            end = min(i + batch_transfer_size, num_tensors)
            s = time.time()
            if send:
                torch.ops.neuron._nrt_batch_transfer(tensors[start:end],
                                                     offsets[start:end],
                                                     lengths[start:end],
                                                     self.remote_ip,
                                                     peer_devices[start:end],
                                                     devices[start:end], send)
            else:
                torch.ops.neuron._nrt_batch_transfer(tensors[start:end],
                                                     offsets[start:end],
                                                     lengths[start:end],
                                                     self.remote_ip,
                                                     peer_devices[start:end],
                                                     devices[start:end], send)
            _d = time.time() - s
            _total_bytes = sum(lengths[start:end])
            logger.debug("Transfer %s GB, taking %s ms, throughput: %s Gbps",
                         _total_bytes / 1e9, _d * 1000,
                         (_total_bytes / 1e9) / _d * 8)
            i += batch_transfer_size
        _duration = time.time() - start_time
        logger.debug("Finished %s %s tensors takes %s ms",
                     'sending' if send else 'recving', len(tensors),
                     _duration * 1000)
