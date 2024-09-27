import importlib
import hashlib
import os
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from transformers import PretrainedConfig

from vllm.config import ModelConfig, ParallelConfig, SchedulerConfig
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.sampler import Sampler, SamplerOutput
from vllm.model_executor.sampling_metadata import SamplingMetadata

from neuronx_distributed_inference.utils.hf_adapter import load_pretrained_config

TORCH_DTYPE_TO_NEURON_AMP = {
    "auto": "float32",
    "half": "float16",
    "float16": "float16",
    "bfloat16": "bfloat16",
    "float": "float32",
    "float32": "float32",
    torch.float16: "float16",
    torch.bfloat16: "bfloat16",
    torch.float32: "float32",
}


# Models supported by Neuronx distributed for inference.
_NEURON_SUPPORTED_MODELS: Dict[str, Tuple[str, str]] = {
    "LlamaForCausalLM": ("neuronx_distributed_inference.models.llama.modeling_llama",
                         "NeuronLlamaForCausalLM"),
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
        output = self.model(input_ids,
                            attention_mask=None,
                            position_ids=positions,
                            seq_ids=input_block_ids)
        return output.logits[:, -1, :]

    def compute_logits(self, hidden_states: torch.Tensor,
                       sampling_metadata: SamplingMetadata) -> torch.Tensor:
        logits = self.logits_processor(None, hidden_states, sampling_metadata)
        return logits

    def sample(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[SamplerOutput]:
        next_tokens = self.sampler(logits, sampling_metadata)
        return next_tokens

    def load_weights(self, model_name_or_path: str, **kwargs):
        arch = _get_model_architecture(self.config)
        neuronx_module_path, neuronx_model_cls_name = (
            _NEURON_SUPPORTED_MODELS[arch])
        neuronx_module = importlib.import_module(neuronx_module_path)
        neuronx_model_cls = getattr(neuronx_module, neuronx_model_cls_name)
        neuron_config = kwargs['neuron_config']
        self.config.neuron_config = neuron_config
        config = neuronx_model_cls.get_config_cls()(
            neuron_config, load_config=load_pretrained_config(model_name_or_path)
        )
        self.model = neuronx_model_cls(model_name_or_path, config)
        compiled_model_path = os.path.join(model_name_or_path,
            f"neuron-compiled-artifacts/{hashlib.md5(config.to_json_string().encode('utf-8')).hexdigest()}/")
        try:
            self.model.load(compiled_model_path)
        except ValueError:
            self.model.compile(compiled_model_path)
            self.model.load(compiled_model_path)


def _get_model_architecture(config: PretrainedConfig) -> str:
    architectures = getattr(config, "architectures", [])
    for arch in architectures:
        if arch in _NEURON_SUPPORTED_MODELS:
            return arch
    raise ValueError(
        f"Model architectures {architectures} are not supported on Neuron "
        f"for now. Supported architectures: "
        f"{list(_NEURON_SUPPORTED_MODELS.keys())}")

def _get_default_neuron_config(model_config: ModelConfig,
                               parallel_config: ParallelConfig,
                               scheduler_config: SchedulerConfig):
    from neuronx_distributed_inference.models.config import NeuronConfig
    neuron_config = NeuronConfig(
        tp_degree=parallel_config.tensor_parallel_size,
        ctx_batch_size=1,
        batch_size=scheduler_config.max_num_seqs,
        max_context_length=scheduler_config.max_model_len,
        seq_len=scheduler_config.max_model_len,
        enable_bucketing=True,
        is_continuous_batching=True,
        quantized=False,
        torch_dtype=TORCH_DTYPE_TO_NEURON_AMP[model_config.dtype],
        padding_side="right"
    )
    return neuron_config

def get_neuron_model(model_config: ModelConfig,
                     parallel_config: ParallelConfig,
                     scheduler_config: SchedulerConfig) -> nn.Module:
    model = NeuronCasualLM(model_config.hf_config)
    default_neuron_config_args = _get_default_neuron_config(
        model_config, parallel_config, scheduler_config)
    model.load_weights(model_config.model,
                       neuron_config=default_neuron_config_args,)
    return model.eval()