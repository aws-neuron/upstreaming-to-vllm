# SPDX-License-Identifier: Apache-2.0
from typing import List, Optional

import torch

from vllm.config import VllmConfig
from vllm.distributed import get_kv_transfer_group
from vllm.logger import init_logger
from vllm.model_executor.layers.sampler import SamplerOutput
from vllm.multimodal import MultiModalKwargs
from vllm.sequence import IntermediateTensors
from vllm.worker.neuronx_distributed_model_runner import (
    NeuronxDistributedModelRunner)

logger = init_logger(__name__)


class MultiStepNeuronxDistributedModelRunner(NeuronxDistributedModelRunner):
    """A model runner for multi-step decoding using the
    neuronx-distributed-inference framework"""

    def __init__(
        self,
        vllm_config: VllmConfig,
    ):
        super().__init__(vllm_config)

    def load_model(self) -> None:
        from vllm.model_executor.model_loader.neuronx_distributed import (
            get_neuron_speculation_model)
        self.model = get_neuron_speculation_model(
            self.model_config,
            cache_config=self.cache_config,
            parallel_config=self.parallel_config,
            scheduler_config=self.scheduler_config,
            speculation_config=self.speculative_config)
        self.is_block_kv_layout = self.model.neuron_config.is_block_kv_layout
        self.is_prefix_caching = self.model.neuron_config.is_prefix_caching
        self.model.is_reorder_needed = not self.is_block_kv_layout

    @torch.inference_mode()
    def execute_model(
        self,
        model_input,
        kv_caches: Optional[List[torch.Tensor]] = None,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        num_steps: int = 1,
    ) -> Optional[List[SamplerOutput]]:
        sampling_params = torch.tensor([[
            seq_group.sampling_params.top_k,
            seq_group.sampling_params.top_p,
            seq_group.sampling_params.temperature,
        ] for seq_group in model_input.sampling_metadata.seq_groups])

        model_executable = self.model
        bypass_model_exec = False
        kv_caches = []
        if self.need_recv_kv(model_input):
            # It doesn't trigger KV cache transfer here which
            # could block decode, transfer was trigger during scheduler
            # and completed at this point, so here we directly
            # get hidden_states (output tokens with on-device sampling)
            # from connector
            logits, bypass_model_exec, model_input = \
                get_kv_transfer_group().recv_kv_caches_and_hidden_states(
                    model_executable,
                    model_input,
                    kv_caches=kv_caches,
                )
            assert bypass_model_exec
        logger.debug("bypass_model_exec: %s", bypass_model_exec)
        if not bypass_model_exec:
            logger.debug(
                "Chose to not bypass execution. Running normal inference.")
            logits = self.model(
                input_ids=model_input.input_tokens,
                positions=model_input.input_positions,
                input_block_ids=model_input.input_block_ids,
                sampling_params=sampling_params,
                slot_mapping=model_input.slot_mapping,
                input_block_tables=model_input.input_block_tables,
                full_context_lens=model_input.full_context_lens,
                computed_context_lens=model_input.computed_context_lens,
                **MultiModalKwargs.as_kwargs(model_input.multi_modal_kwargs
                                             or {},
                                             device=self.device),
            )
        if self.need_send_kv(model_input):
            logger.debug(
                "Sending KV cache, model output, and hidden_states (if EAGLE)."
            )
            get_kv_transfer_group().send_kv_caches_and_hidden_states(
                # model_executable is used to know which layer the current
                # worker is working on, so that we can send KV for only those
                # layers.
                model_executable,
                model_input,
                kv_caches,
                logits,
            )

        output = self.model.sample(
            logits=logits,
            sampling_metadata=model_input.sampling_metadata,
        )
        return output
