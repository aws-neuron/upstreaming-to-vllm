# SPDX-License-Identifier: Apache-2.0
"""
Neuron KV Cache Connector for Disaggregated Inference

"""
import math
from typing import TYPE_CHECKING, List, Optional, Tuple, Union

import torch

from vllm.distributed.kv_transfer.kv_connector.base import KVConnectorBase
from vllm.distributed.kv_transfer.kv_lookup_buffer.neuron_buffer import (
    NeuronBuffer)
from vllm.logger import init_logger
from vllm.sequence import IntermediateTensors

if TYPE_CHECKING:
    from vllm.worker.model_runner import ModelInputForGPUWithSamplingMetadata

logger = init_logger(__name__)


class NeuronConnector(KVConnectorBase):

    def __init__(
        self,
        rank: int,
        local_rank: int,
        vllm_config,
    ):

        self.config = vllm_config.kv_transfer_config
        # FIXME: currently hard coded due to config mismatch between
        # 0.6.x vs 0.7
        self.tp_size = 2

        from vllm.distributed.kv_transfer.kv_pipe.neuron_pipe import NeuronPipe
        logger.info("Initializing PyNcclConfig under "
                    "kv_transfer_config %s", self.config)

        self.lookup_buffer_size = self.config.kv_buffer_size

        self.producer_buffer: Optional[NeuronBuffer] = None
        self.consumer_buffer: Optional[NeuronBuffer] = None

        self.producer_data_pipe: Union[NeuronPipe]
        self.consumer_data_pipe: Union[NeuronPipe]
        self.producer_signal_pipe: Union[NeuronPipe]
        self.consumer_signal_pipe: Union[NeuronPipe]

        # 2 pipes for every rank in the world
        # FIXME: a bit confusing here for rank vs local_rank
        # port_offset_base = 2 * rank
        port_offset_base = 0

        # In disaggregated prefill, the prefill vLLM only uses send pipe
        # and the decode vLLM only uses recv pipe
        if self.config.is_kv_producer:
            self.producer_data_pipe = NeuronPipe(
                local_rank=local_rank,
                config=self.config,
                port_offset=port_offset_base,
            )
            self.producer_signal_pipe = NeuronPipe(
                local_rank=local_rank,
                config=self.config,
                port_offset=port_offset_base + 1,
                device="cpu",
            )
            self.producer_buffer = NeuronBuffer(self.producer_signal_pipe,
                                                self.producer_data_pipe,
                                                self.config.kv_buffer_size)

        else:
            self.consumer_data_pipe = NeuronPipe(
                local_rank=local_rank,
                config=self.config,
                port_offset=port_offset_base,
            )
            self.consumer_signal_pipe = NeuronPipe(
                local_rank=local_rank,
                config=self.config,
                port_offset=port_offset_base + 1,
                device="cpu",
            )

            self.consumer_buffer = NeuronBuffer(
                self.consumer_signal_pipe,
                self.consumer_data_pipe,
                self.config.kv_buffer_size,
            )

        logger.info("Done - Initializing %s", self.config.kv_connector)

    def select(self, input_tokens: Optional[torch.Tensor],
               roi: Optional[torch.Tensor]) -> List[Optional[torch.Tensor]]:

        raise NotImplementedError(
            "Use async_recv_kv_caches_and_hidden_states instead")

    def recv_kv_caches_and_hidden_states(
        self, model_executable: torch.nn.Module,
        model_input: "ModelInputForGPUWithSamplingMetadata",
        kv_caches: List[torch.Tensor]
    ) -> Tuple[Union[torch.Tensor, IntermediateTensors], bool,
               "ModelInputForGPUWithSamplingMetadata"]:
        raise NotImplementedError()

    def send_kv_caches_and_hidden_states(
        self,
        model_executable: torch.nn.Module,
        model_input: "ModelInputForGPUWithSamplingMetadata",
        kv_caches: List[torch.Tensor],
        hidden_or_intermediate_states: Union[torch.Tensor,
                                             IntermediateTensors],
    ) -> None:

        for idx, seq_id in enumerate(model_input.input_block_ids.tolist()):

            current_tokens = model_input.input_tokens[idx]
            input_positions = model_input.input_positions[idx]

            mask = (input_positions == torch.arange(input_positions.shape[-1]))

            current_tokens = current_tokens[mask]

            kv_length = [
                math.prod(cache.shape[1:]) * cache.element_size()
                for cache in kv_caches
            ]
            kv_offset = [L * seq_id for L in kv_length]
            logger.debug("kv_length[0] : %s", kv_length[0])
            logger.debug("kv_offset[0]: %s", kv_offset[0])

            logger.debug("connector: send %s on seq_id %s", current_tokens,
                         seq_id)
            assert self.producer_buffer is not None
            self.producer_buffer.insert_neuron_buffer(
                seq_id,
                current_tokens,
                torch.ones_like(current_tokens, dtype=bool),
                kv_caches,
                hidden_or_intermediate_states[idx],
                kv_offset=kv_offset,
                kv_length=kv_length)

        logger.debug("[rank%d]: KV send DONE.", torch.distributed.get_rank())

    def async_recv_kv_caches_and_hidden_states(
        self,
        model_executable: torch.nn.Module,
        input_ids,
        seq_id,
    ):
        kv_caches = model_executable.get_kv_cache()
        kv_length = [
            math.prod(cache.shape[1:]) * cache.element_size()
            for cache in kv_caches
        ]
        kv_offset = [L * seq_id for L in kv_length]

        logger.debug("connector: recv %s on %s with kv_offset %s length %s",
                     input_ids, seq_id, kv_offset, kv_length)
        assert self.consumer_buffer is not None
        self.consumer_buffer.async_drop_select(seq_id,
                                               input_ids,
                                               torch.ones_like(input_ids,
                                                               dtype=bool),
                                               kv_caches,
                                               kv_offset=kv_offset,
                                               kv_length=kv_length)

    def recv_kv_caches_and_hidden_states_from_local_buffer(
        self, model_executable: torch.nn.Module,
        model_input: "ModelInputForGPUWithSamplingMetadata",
        kv_caches: List[torch.Tensor]
    ) -> Tuple[Union[torch.Tensor, IntermediateTensors], bool,
               "ModelInputForGPUWithSamplingMetadata"]:

        # When bypass_model_exec is set to False, it means that at least for one
        # request its corresponding KV cache or hidden state is missing.
        # In this case we need to do prefilling to recompute missing KV cache
        # and hidden states. This should only happen on when we intentionally
        # forward the request only to the decode instance.
        bypass_model_exec = True

        hidden_or_intermediate_states_for_one_req = []

        for idx, seq_id in enumerate(model_input.input_block_ids.tolist()):

            current_tokens = model_input.input_tokens[idx]
            input_positions = model_input.input_positions[idx]

            mask = (input_positions == torch.arange(input_positions.shape[-1]))

            current_tokens = current_tokens[mask]

            assert self.consumer_buffer is not None
            status, entry = self.consumer_buffer.check_transfer_status(
                seq_id, current_tokens, remove=True)

            assert status, f"transfer for input: {current_tokens} is not done"

            assert entry.roi is not None, \
                f"cannot retrieve KV cache for input: {current_tokens}"

            roi: torch.Tensor = entry.roi

            hidden: torch.Tensor = entry.hidden
            hidden = hidden.unsqueeze(0)
            num_computed_tokens = roi.shape[0]

            # check if both KV cache and the hidden states are received
            # If not, need to redo the forwarding to compute missing states
            if not all([(num_computed_tokens == current_tokens.shape[0]),
                        hidden is not None]):
                bypass_model_exec = False

            # no need to update KV cache explicly for RDMA

            hidden_or_intermediate_states_for_one_req.append(hidden)

        assert bypass_model_exec, (f"[rank{torch.distributed.get_rank()}]: "
                                   "Failed to receive all KVs and hidden "
                                   "states, redo model forwarding.")
        logger.debug(
            "[rank%d]: Successfully received all KVs and hidden "
            "states, skip model forwarding.", torch.distributed.get_rank())

        hidden_or_intermediate_states = torch.cat(
            hidden_or_intermediate_states_for_one_req, dim=0)

        return hidden_or_intermediate_states, bypass_model_exec, model_input

    def close(self):
        self.producer_data_pipe.close()
        self.consumer_data_pipe.close()
        if self.config.kv_connector == "NeuronConnector":
            self.producer_signal_pipe.close()
            self.consumer_signal_pipe.close()
        elif self.config.kv_connector == "MooncakeConnector":
            # MooncakePipe reuses data_pipe for signal_pipe, so we only have to
            # close the data_pipe.
            pass
