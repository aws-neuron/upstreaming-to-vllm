"""Utilities for selecting and loading neuron models."""
import importlib
import os
from typing import Dict, Optional, Tuple
import copy

import torch
import torch.nn as nn
import transformers
from transformers import PretrainedConfig

from vllm.config import ModelConfig, ParallelConfig, SchedulerConfig
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.sampler import Sampler
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.sequence import SamplerOutput, SequenceOutput, CompletionSequenceGroupOutput, Logprob

TORCH_DTYPE_TO_NEURON_AMP = {
    "auto": "f32",
    "half": "f16",
    "float16": "f16",
    "bfloat16": "bf16",
    "float": "f32",
    "float32": "f32",
    torch.float16: "f16",
    torch.bfloat16: "bf16",
    torch.float32: "f32",
}

# Models supported by Neuron.
_NEURON_SUPPORTED_MODELS: Dict[str, Tuple[str, str, str]] = {
    "LlamaForCausalLM": ("transformers_neuronx.llama.model",
                         "LlamaForSampling", "LlamaForCausalLM"),
    "MistralForCausalLM": ("transformers_neuronx.mistral.model",
                           "MistralForSampling", "MistralForCausalLM")
}


class NeuronCasualLM(nn.Module):

    def __init__(
        self,
        config: PretrainedConfig,
    ) -> None:
        super().__init__()
        self.config = config
        self.logits_processor = LogitsProcessor(config.vocab_size,
                                                logits_as_input=True)
        self.sampler = Sampler()

        # Lazy initialized
        self.model: nn.Module

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        input_block_ids: torch.Tensor,
    ) -> torch.Tensor:
        logits = self.model(input_ids,
                            cache_ids=positions,
                            start_ids=input_block_ids)
        return logits

    def compute_logits(self, hidden_states: torch.Tensor,
                       sampling_metadata: SamplingMetadata) -> torch.Tensor:
        logits = self.logits_processor(None, hidden_states, sampling_metadata)
        return logits

    def sample(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[SamplerOutput]:
        hidden_states = logits.flatten()
        next_tokens = []
        sample_idx = 0
        #print(f"sampling_metadata: {sampling_metadata}")
        for seq_group in sampling_metadata.seq_groups:
            samples = []
            for seq_id in seq_group.seq_ids:
                token_id = hidden_states[sample_idx].item()
                samples.append(SequenceOutput(parent_seq_id=seq_id, output_token=token_id,
                                                logprobs={token_id: Logprob(token_id)}))
                sample_idx += 1
            next_tokens.append(CompletionSequenceGroupOutput(samples=samples, prompt_logprobs=None))
        returned_output = SamplerOutput(outputs=next_tokens)
        return returned_output

    def load_weights(self, model_name_or_path: str, **kwargs):
        arch = _get_model_architecture(self.config)
        neuronx_module_path, neuronx_model_cls_name, hf_model_cls_name = (
            _NEURON_SUPPORTED_MODELS[arch])
        neuronx_module = importlib.import_module(neuronx_module_path)
        neuronx_model_cls = getattr(neuronx_module, neuronx_model_cls_name)
        self.model = neuronx_model_cls.from_pretrained(model_name_or_path,
                                                       **kwargs)
        self.model.to_neuron()



def _get_model_architecture(config: PretrainedConfig) -> str:
    architectures = getattr(config, "architectures", [])
    for arch in architectures:
        if arch in _NEURON_SUPPORTED_MODELS:
            return arch
    raise ValueError(
        f"Model architectures {architectures} are not supported on Neuron "
        f"for now. Supported architectures: "
        f"{list(_NEURON_SUPPORTED_MODELS.keys())}")


def get_neuron_model(model_config: ModelConfig,
                     parallel_config: ParallelConfig,
                     scheduler_config: SchedulerConfig) -> nn.Module:
    from transformers_neuronx.config import (ContinuousBatchingConfig,
                                             NeuronConfig)

    # Create a model instance.
    model = NeuronCasualLM(model_config.hf_config)

    on_dev_sampling_config = copy.deepcopy(model_config.generation_config)

    continuous_batching_config = ContinuousBatchingConfig(
        batch_size_for_shared_caches=scheduler_config.max_num_seqs)

    neuron_config = NeuronConfig(
        on_device_generation=on_dev_sampling_config,
        continuous_batching=continuous_batching_config)

    # Load the weights from the cached or downloaded files.
    model.load_weights(
        model_config.model,
        tp_degree=parallel_config.tensor_parallel_size,
        amp=TORCH_DTYPE_TO_NEURON_AMP[model_config.dtype],
        neuron_config=neuron_config,
        context_length_estimate=[scheduler_config.max_model_len],
        n_positions=[scheduler_config.max_model_len],
        batch_size=scheduler_config.max_num_seqs)

    return model.eval()
