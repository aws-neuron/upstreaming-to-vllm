# SPDX-License-Identifier: Apache-2.0
"""
Neuron KV Cache Connector for Disaggregated Inference

"""

import os
from typing import List, Tuple, Union

import torch
import zmq

from vllm.distributed.kv_transfer.kv_connector.base import KVConnectorBase
from vllm.distributed.kv_transfer.neuron.neuron_buffer import NeuronBuffer
from vllm.logger import init_logger
from vllm.sequence import IntermediateTensors
from vllm.worker.neuron_model_runner import ModelInputForNeuron

logger = init_logger(__name__)


class NeuronConnector(KVConnectorBase):

    def __init__(self, rank, local_rank, vllm_config):
        self.zmq_context = zmq.Context()
        self.config = vllm_config.kv_transfer_config

        self.block_size = vllm_config.cache_config.block_size
        self.max_model_len = vllm_config.scheduler_config.max_model_len
        self.is_block_kv_layout = self.block_size != self.max_model_len

    def initialize_buffer(self):

        # TODO: create buffer dynamically based on input requests

        self.neuron_send_ip = os.environ.get("NEURON_SEND_IP", "127.0.0.1")
        self.neuron_recv_ip = os.environ.get("NEURON_RECV_IP", "127.0.0.1")

        if self.config.is_kv_producer:
            self.buffer = NeuronBuffer(self.zmq_context,
                                       self.neuron_recv_ip,
                                       self.config.kv_ip,
                                       self.config.kv_port,
                                       self.config.kv_map_path,
                                       self.is_block_kv_layout,
                                       self.config.neuron_core_offset,
                                       send=True)
        else:
            self.buffer = NeuronBuffer(self.zmq_context,
                                       self.neuron_send_ip,
                                       self.config.kv_ip,
                                       self.config.kv_port,
                                       self.config.kv_map_path,
                                       self.is_block_kv_layout,
                                       self.config.neuron_core_offset,
                                       send=False)

    def close(self):
        try:
            self.zmq_context.sockets_map.clear()
            self.zmq_context.term()
        except Exception as e:
            print(f"Error closing ZMQ context: {e}")

    def register_kv_caches(self, kv_caches):
        self.buffer.register_kv_caches(kv_caches)

    def async_send_kv_caches(self, request_id, block_ids, output_token):
        self.buffer.async_send_kv_caches(request_id, block_ids, output_token)

    def async_recv_kv_caches(self, request_id, block_ids):
        self.buffer.async_recv_kv_caches(request_id, block_ids)

    def check_transfer_done(self, request_id, remove=False):
        return self.buffer.check_transfer_done(request_id, remove=remove)

    def get_output_token(self, request_id):
        return self.buffer.get_output_token(request_id)

    # ===================== API for v0 compatibility ==================

    def send_kv_caches_and_hidden_states(
        self,
        model_executable: torch.nn.Module,
        model_input: ModelInputForNeuron,
        kv_caches: List[torch.Tensor],
        hidden_or_intermediate_states: Union[torch.Tensor,
                                             IntermediateTensors],
    ) -> None:

        for i in range(len(model_input.request_ids)):
            request_id = model_input.request_ids[i]
            if self.is_block_kv_layout:
                block_ids = model_input.input_block_tables[i]
                num_effective_blocks = model_input.full_context_lens[
                    i] // self.block_size
                block_ids = block_ids[:num_effective_blocks + 1].tolist()
            else:
                block_ids = model_input.input_block_ids[i]
                if len(block_ids.shape) == 0:
                    block_ids = block_ids.unsqueeze(0)
                block_ids = block_ids.tolist()

            output_token = hidden_or_intermediate_states[i]

            self.async_send_kv_caches(request_id, block_ids, output_token)

    def recv_kv_caches_and_hidden_states(
        self, model_executable: torch.nn.Module,
        model_input: ModelInputForNeuron, kv_caches: List[torch.Tensor]
    ) -> Tuple[Union[torch.Tensor, IntermediateTensors], bool,
               ModelInputForNeuron]:

        output_tokens = []
        for request_id in model_input.request_ids:
            output_token = self.get_output_token(request_id).unsqueeze(0)
            assert self.check_transfer_done(request_id, remove=True)
            output_tokens.append(output_token)
        output_tokens = torch.cat(output_tokens, dim=0)

        return output_tokens, True, model_input
