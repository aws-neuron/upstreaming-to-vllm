"""Utilities for selecting and loading models."""
from typing import Type

import torch
import torch.nn as nn
from transformers import PretrainedConfig

from vllm.config import ModelConfig, DeviceConfig
from vllm.model_executor.models import ModelRegistry

TORCH_DTYPE_TO_NEURON_AMP = {
    "auto": torch.float32,
    "half": torch.float16,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float": torch.float32,
    "float32": torch.float32,
    "bf16": torch.bfloat16,
    "f16": torch.float16,
    "f32": torch.float32,
}


def _get_model_architecture(config: PretrainedConfig) -> Type[nn.Module]:
    architectures = getattr(config, "architectures", [])
    for arch in architectures:
        model_cls = ModelRegistry.load_model_cls(arch)
        if model_cls is not None:
            return model_cls
    raise ValueError(
        f"Model architectures {architectures} are not supported for now. "
        f"Supported architectures: {ModelRegistry.get_supported_archs()}")


def get_model(model_config: ModelConfig, device_config: DeviceConfig,
              **kwargs) -> nn.Module:

    parallel_config = kwargs.get("parallel_config")
    scheduler_config = kwargs.get("scheduler_config")

    model_class = _get_model_architecture(model_config.hf_config)
    linear_method = None

    # Create a model instance.
    model = model_class(model_config.hf_config, linear_method)


    def dtype_converter(dtype):
        if isinstance(dtype, str):
            return TORCH_DTYPE_TO_NEURON_AMP[model_config.dtype]
        else:
            return dtype

    # Load the weights from the cached or downloaded files.
    model.load_weights(
        model_config.model,
        model_config.download_dir,
        model_config.load_format,
        model_config.revision,
        tp_degree=parallel_config.neuron_tp_degree,
        amp=dtype_converter(model_config.dtype),
        context_length_estimate=[scheduler_config.max_model_len // 2],
        n_positions=[scheduler_config.max_model_len],
        batch_size=scheduler_config.max_num_seqs)

    return model.eval()
