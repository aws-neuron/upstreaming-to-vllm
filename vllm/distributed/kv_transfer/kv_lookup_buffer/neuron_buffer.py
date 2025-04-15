# SPDX-License-Identifier: Apache-2.0
"""
    Lookup buffer for Neuron
"""
import os
import queue
import threading
import time
from collections import defaultdict
from typing import List, Optional, Union

import torch

from vllm.distributed.kv_transfer.kv_lookup_buffer.base import (
    KVLookupBufferBase)
from vllm.distributed.kv_transfer.kv_pipe.base import KVPipeBase
from vllm.logger import init_logger

logger = init_logger(__name__)


class DropSelectRequest:

    def __init__(self, input_tokens, roi):
        self.input_tokens = input_tokens
        self.roi = roi

    def __repr__(self):
        return (f"DropSelectRequest: {self.input_tokens}, "
                f"{self.input_tokens.dtype}")


class BufferEntry:

    def __init__(self,
                 input_tokens,
                 roi,
                 kv_caches,
                 hidden,
                 kv_offset=0,
                 kv_length=None):
        self.input_tokens = input_tokens
        self.roi = roi
        self.kv_caches = kv_caches
        self.hidden = hidden
        self.kv_offset = kv_offset
        self.kv_length = kv_length
        self.done_transfer = False


class NeuronBuffer(KVLookupBufferBase):

    def __init__(self, signal_pipe: KVPipeBase, data_pipe: KVPipeBase,
                 buffer_size_thresh: float):
        """
        signal_pipe: on CPU

        NOTE: on-device recv will block all threads in the process, making the
        KV cache producer unable to listen to new request while transmitting
        KV cache. Luckily CPU recv only blocks the current thread so we use
        CPU recv to listen to new request.

        data_pipe: on device (e.g. GPU)
        """

        # for prefill
        #   - add to buffer when a new prefill finishes
        #   - remove from buffer when transfer is done
        # for decode
        #   - add to buffer when request a prefill kv cache transfer
        #   - remove from the buffer when transfer is done
        self.buffer: dict[int, BufferEntry] = {}  # seq_id: {BufferEntry}
        self.input_ids_table: dict[tuple[int], list[int]] = defaultdict(
            list)  # input_ids: {seq_id}
        self.buffer_size = 0
        self.buffer_lock = threading.Lock()
        self.consumer_drop_select_thread: Optional[threading.Thread] = None

        self.signal_pipe = signal_pipe
        self.data_pipe = data_pipe
        self.request_handling_thread: Optional[threading.Thread] = None
        self.drop_select_queue: queue.Queue = queue.Queue()
        self.normal_signal = torch.tensor([0], device="cpu")
        self.end_signal = None

        self.neuron_send_start_rank = int(
            os.environ.get("NEURON_SEND_START_RANK", "0"))
        self.neuron_recv_start_rank = int(
            os.environ.get("NEURON_RECV_START_RANK", "0"))
        self.neuron_send_ip = os.environ.get("NEURON_SEND_IP", "127.0.0.1")
        self.neuron_recv_ip = os.environ.get("NEURON_RECV_IP", "127.0.0.1")

        logger.info("Neuron Send IP: %s, Neuron Recv IP: %s",
                    self.neuron_send_ip, self.neuron_recv_ip)

    def _batch_transfer_neuron(self,
                               tensors,
                               is_send=True,
                               offset=0,
                               length=None):
        logger.debug("batch transferring tensors %s with offset %s length %s",
                     len(tensors), offset, length)
        # currently unused as we do the transfer loop in c++
        requests = []

        total_bytes = 0
        start = time.time()
        for tensor, L, o in zip(tensors, length, offset):
            logger.debug("transferring tensor %s with offset %s length %s",
                         tensor.shape, o, L)
            # get device id based on tensor allocation
            device_id = tensor.device.index
            length = length if length else tensor.nbytes

            total_bytes += L

            # assume 1-1 mapping
            if is_send:
                request_id = torch.ops.neuron._nrt_async_send_tensor(
                    tensor, o, L, self.neuron_recv_ip, device_id, device_id)
            else:
                request_id = torch.ops.neuron._nrt_async_recv_tensor(
                    tensor, o, L, self.neuron_send_ip, device_id, device_id)

            requests.append((request_id, device_id))

        for request_id, device_id in requests:
            while True:
                done, size = torch.ops.neuron._nrt_async_test(
                    request_id, device_id)
                # TODO; check transfer size
                if done:
                    break

        duration = time.time() - start

        logger.debug("Transfer %s GB, taking %s ms, throughput: %s Gbps",
                     total_bytes / 1e9, duration * 1000,
                     (total_bytes / 1e9) / duration * 8)

    def _transfer_neuron_tensors(self, tensors, is_send, offset, length):
        logger.debug("Start %s %s tensors on Neuron",
                     'sending' if is_send else 'recving', len(tensors))
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
            if is_send:
                torch.ops.neuron._nrt_batch_transfer(
                    tensors[start:end], offset[start:end], length[start:end],
                    self.neuron_recv_ip, devices[start:end],
                    devices[start:end], is_send)
            else:
                torch.ops.neuron._nrt_batch_transfer(
                    tensors[start:end], offset[start:end], length[start:end],
                    self.neuron_send_ip, devices[start:end],
                    devices[start:end], is_send)
            _d = time.time() - s
            _total_bytes = sum(length[start:end])
            logger.debug("Transfer %s GB, taking %s ms, throughput: %s Gbps",
                         _total_bytes / 1e9, _d * 1000,
                         (_total_bytes / 1e9) / _d * 8)
            i += batch_transfer_size
        _duration = time.time() - start_time
        logger.debug("Finished %s %s tensors takes %s ms",
                     'sending' if is_send else 'recving', len(tensors),
                     _duration * 1000)

    def _send_tensor_and_dec_size(self,
                                  tensor: Optional[torch.Tensor]) -> None:

        assert tensor is not None, "Use self.data_pipe.send(None) instead"
        if tensor.dtype == torch.bool:
            tensor = tensor.float()
        self.data_pipe.send_tensor(tensor)

    def _get_element_size(self, data: Optional[Union[List, torch.Tensor]]):

        if isinstance(data, torch.Tensor):
            return data.element_size() * data.numel()
        if not data:
            # cannot perform `not data` on a tensor
            # so this check needs to go after the check above
            return 0

        raise AssertionError(f"Unknown data type {type(data)}")

    def _add_to_buffer(self,
                       seq_id: int,
                       input_tokens: torch.Tensor,
                       roi: torch.Tensor,
                       kv_caches: List[torch.Tensor],
                       hidden: torch.Tensor,
                       kv_offset=0,
                       kv_length=None):
        logger.debug("adding seq_id %s to producer buffer %s, %s", seq_id,
                     input_tokens, input_tokens.dtype)
        if isinstance(input_tokens, torch.Tensor):
            input_tokens = input_tokens.clone()
        if isinstance(roi, torch.Tensor):
            roi = roi.clone()
        # done clone key and value
        if isinstance(hidden, torch.Tensor):
            hidden = hidden.clone()

        entry = BufferEntry(input_tokens=input_tokens,
                            roi=roi,
                            kv_caches=kv_caches,
                            kv_offset=kv_offset,
                            kv_length=kv_length,
                            hidden=hidden)

        with self.buffer_lock:
            input_tokens_key = tuple(input_tokens.tolist())
            # simply overwrites if there is already another kv cache with
            # same input_tokens
            # not needed for decode
            self.input_ids_table[input_tokens_key].append(seq_id)
            self.buffer[seq_id] = entry

    def _remove_from_buffer(self, seq_id, input_tokens):
        with self.buffer_lock:
            input_tokens_key = tuple(input_tokens.tolist())
            # simply overwrites if there is already another kv cache with
            # same input_tokens
            seq_ids = self.input_ids_table[input_tokens_key]
            seq_ids.remove(seq_id)
            del self.buffer[seq_id]

    def _is_end_signal(self, signal):
        return signal is None

    def drop_select_requester(self):

        try:
            while True:
                try:
                    seq_id, entry = self.drop_select_queue.get()

                    logger.debug(
                        "start drop_select_request on seq_id %s with "
                        "input_tokens %s", seq_id, entry.input_tokens)

                    input_tokens = entry.input_tokens.clone()
                    roi = entry.roi.clone().float()

                    # request for match
                    self.signal_pipe.send_tensor(self.normal_signal)
                    self.data_pipe.send_tensor(input_tokens)
                    self.data_pipe.send_tensor(roi)

                    roi = self.data_pipe.recv_tensor()
                    if roi is None:
                        logger.debug(
                            "received nothing for seq_id %s with "
                            "input_tokens %s", seq_id, entry.input_tokens)

                        # # receive nothing
                        # with self.buffer_lock:
                        #     self.buffer[tuple(input_tokens.tolist())]\
                        #       .done_transfer = False

                        # put request back to queue for next execution
                        self.drop_select_queue.put((seq_id, entry))
                    else:
                        # convert from float tensor to bool tensor
                        # as PyNccl does not support sending bool tensor
                        roi = (roi > 0.5)

                        # FIXME: okay to leave it unprotected by lock?
                        # if locks the the done_transfer check will gets hang
                        # due to transfer - need a better locking machnism
                        entry.hidden = self.data_pipe.recv_tensor()
                        self._transfer_neuron_tensors(entry.kv_caches,
                                                      is_send=False,
                                                      offset=entry.kv_offset,
                                                      length=entry.kv_length)
                        entry.done_transfer = True

                except queue.Empty:
                    # sleep for 1 ms to avoid contention buffer_lock
                    time.sleep(0.001)
                    pass

        except RuntimeError as e:
            if 'Connection closed by peer' not in str(e):
                raise e

        logger.debug("Closing drop_select_requester")

    def check_transfer_status(self, seq_id, input_tokens, remove=False):
        entry = self.buffer[seq_id]
        assert torch.equal(
            entry.input_tokens, input_tokens
        ), f"not getting same tensor input {input_tokens} vs expected "
        f"{entry.input_tokens} for entry under {seq_id}"
        done = entry.done_transfer

        if done and remove:
            self._remove_from_buffer(seq_id, input_tokens)

        return done, entry

    def drop_select(self, input_tokens, roi):
        raise NotImplementedError()

    def async_drop_select(self, seq_id: int, input_tokens: torch.Tensor,
                          roi: torch.Tensor, kv_caches: List[torch.Tensor],
                          kv_offset: List, kv_length: List):

        logger.debug("start async_drop_select for %s", input_tokens)
        # if the request is already in buffer, the skip
        with self.buffer_lock:
            if input_tokens in self.buffer:
                return

        with self.buffer_lock:
            entry = BufferEntry(input_tokens,
                                roi,
                                kv_caches,
                                None,
                                kv_offset=kv_offset,
                                kv_length=kv_length)
            input_tokens_key = tuple(input_tokens.tolist())
            self.input_ids_table[input_tokens_key].append(seq_id)
            self.buffer[seq_id] = entry
            self.drop_select_queue.put((seq_id, entry))

        # calling the thread to fetch KV cache data
        if self.consumer_drop_select_thread is None:
            self.consumer_drop_select_thread = threading.Thread(
                target=self.drop_select_requester)
            self.consumer_drop_select_thread.start()

    def drop_select_handler(self):
        logger.debug("start drop_select_handler")
        try:

            while True:
                signal = self.signal_pipe.recv_tensor(is_signal_pipe=True)

                if self._is_end_signal(signal):
                    logger.debug("Received end signal!")
                    break

                input_tokens = self.data_pipe.recv_tensor()

                roi = self.data_pipe.recv_tensor()
                assert roi is not None, "Please provide the roi when sending "\
                    "drop-select request"
                roi = (roi > 0.5)

                # perform input tokens and roi matching
                # FIXME: this matching is O(n), ideally it should be O(1)
                # but this buffer size won't (and shouldn't) be too large so
                # the fix is not urgent.
                assert input_tokens is not None
                logger.debug("drop_select_handler receive: %s, %s",
                             input_tokens, input_tokens.dtype)
                look_up_key = tuple(input_tokens.tolist())
                entry = None
                with self.buffer_lock:
                    seq_ids = self.input_ids_table[look_up_key]
                    if len(seq_ids) == 0:
                        pass
                    else:
                        for seq_id in seq_ids:
                            entry_candidate = self.buffer[seq_id]
                            if not entry_candidate.done_transfer:
                                entry = entry_candidate
                                break
                        if entry is None:
                            pass
                            logger.debug(
                                "drop_select_handler find matched entry but "
                                "doesn't find untransferred entry: %s",
                                input_tokens)

                if entry:
                    # skip sending input_tokens because there is a match
                    self._send_tensor_and_dec_size(entry.roi)
                    self._send_tensor_and_dec_size(entry.hidden)
                    self._transfer_neuron_tensors(entry.kv_caches,
                                                  is_send=True,
                                                  offset=entry.kv_offset,
                                                  length=entry.kv_length)
                    entry.done_transfer = True
                else:
                    # no match, just send None
                    logger.debug(
                        "drop_select_handler doesn't find match for: %s",
                        input_tokens)
                    self.data_pipe.send_tensor(None)

        except RuntimeError as e:
            if 'Connection closed by peer' not in str(e):
                raise e

        logger.debug("Closing drop_select_handler")

    def full_handler(self):
        time.sleep(0.001)

    def insert(self, input_tokens: torch.Tensor, roi: torch.Tensor,
               key: torch.Tensor, value: torch.Tensor,
               hidden: torch.Tensor) -> None:
        raise NotImplementedError("Use insert_neuron_buffer instead")

    def insert_neuron_buffer(self,
                             seq_id: int,
                             input_tokens: torch.Tensor,
                             roi: torch.Tensor,
                             kv_caches: List[torch.Tensor],
                             hidden: torch.Tensor,
                             kv_offset=0,
                             kv_length=None) -> None:

        self._add_to_buffer(seq_id,
                            input_tokens,
                            roi,
                            kv_caches,
                            hidden,
                            kv_offset=kv_offset,
                            kv_length=kv_length)

        # when calling the insert, the current process is a sender
        # need to launch the request handler and start listening to request.
        if self.request_handling_thread is None:
            self.request_handling_thread = threading.Thread(
                target=self.drop_select_handler)
            self.request_handling_thread.start()

    def close(self):

        if hasattr(self, "request_handling_thread"
                   ) and self.request_handling_thread is not None:
            self.request_handling_thread.join()

        else:
            # TODO: have a explicit close signal and have a explicit way to
            # check if it's requester
            self.signal_pipe.send_tensor(self.end_signal)
