import importlib
import hashlib
import os
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, PretrainedConfig

from vllm.config import ModelConfig, ParallelConfig, SchedulerConfig
from vllm.logger import init_logger
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.sampler import Sampler, SamplerOutput
from vllm.model_executor.sampling_metadata import SamplingMetadata

from neuronx_distributed_inference.utils.hf_adapter import load_pretrained_config

logger = init_logger(__name__)

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
    "DbrxForCausalLM": ("neuronx_distributed_inference.models.dbrx.modeling_dbrx",
                         "NeuronDbrxForCausalLM"),
    "MixtralForCausalLM": ("neuronx_distributed_inference.models.modeling_mixtral",
                         "NeuronMixtralForCausalLM"),
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
        neuron_config = neuronx_model_cls.get_neuron_config_cls()(**kwargs['neuron_config'])
        self.config.neuron_config = neuron_config
        config = neuronx_model_cls.get_config_cls()(
            neuron_config, load_config=load_pretrained_config(model_name_or_path)
        )
        if os.getenv("NEURON_COMPILED_ARTIFACTS") is not None:
            compiled_model_path = os.getenv("NEURON_COMPILED_ARTIFACTS")
        elif os.path.exists(model_name_or_path):
            compiled_model_path = os.path.join(model_name_or_path,
                f"neuron-compiled-artifacts/{hashlib.md5(config.to_json_string().encode('utf-8')).hexdigest()}/")
        else:
            compiled_model_path = os.path.join("local-models", model_name_or_path,
                f"neuron-compiled-artifacts/{hashlib.md5(config.to_json_string().encode('utf-8')).hexdigest()}/")
        try:
            self.model = neuronx_model_cls(compiled_model_path)
            self.model.load(compiled_model_path)
            return
        except (FileNotFoundError, ValueError):
            logger.warning(f"Failed to load the model from {compiled_model_path}, Recompiling...")
        if not os.path.exists(model_name_or_path):
            hf_model = AutoModelForCausalLM.from_pretrained(model_name_or_path)
            saved_path = os.path.join("local-models", model_name_or_path)
            hf_model.save_pretrained(saved_path)
            model_name_or_path = saved_path
        self.model = neuronx_model_cls(model_name_or_path, config)
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
    neuron_config = dict(
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

def _get_neuron_config_after_override(default_neuron_config,
                                      overridden_neuron_config):
    overridden_neuron_config = overridden_neuron_config or {}
    default_neuron_config.update(overridden_neuron_config)
    return default_neuron_config

def get_neuron_model(model_config: ModelConfig,
                     parallel_config: ParallelConfig,
                     scheduler_config: SchedulerConfig) -> nn.Module:
    model = NeuronCasualLM(model_config.hf_config)
    default_neuron_config_args = _get_default_neuron_config(
        model_config, parallel_config, scheduler_config)
    neuron_config = _get_neuron_config_after_override(default_neuron_config_args,
        model_config.override_neuron_config)
    model.load_weights(model_config.model,
                       neuron_config=neuron_config,)
    return model.eval()

