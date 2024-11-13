import copy
import importlib
import hashlib
import os
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, PretrainedConfig

from vllm.config import ModelConfig, ParallelConfig, SchedulerConfig, SpeculativeConfig
from vllm.logger import init_logger
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.sampler import Sampler, SamplerOutput
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.sequence import (CompletionSequenceGroupOutput, Logprob,
                           SequenceOutput)

from neuronx_distributed_inference.utils.hf_adapter import load_pretrained_config
from neuronx_distributed_inference.models.config import FusedSpecNeuronConfig

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
    "MixtralForCausalLM": ("neuronx_distributed_inference.models.mixtral.modeling_mixtral",
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
        sampling_params: torch.Tensor,
    ) -> torch.Tensor:
        output = self.model(input_ids,
                            attention_mask=None,
                            position_ids=positions,
                            seq_ids=input_block_ids,
                            sampling_params=sampling_params)
        # on-device sampling
        if self.config.neuron_config.on_device_sampling_config:
            return output.hidden_states
        else:
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
        # on-device sampling
        if self.config.neuron_config.on_device_sampling_config:
            batch_size = logits.shape
            seq_ids = [seq_id for sg in sampling_metadata.seq_groups for seq_id in sg.seq_ids]
            assert len(seq_ids) == list(batch_size)[0], "batch size mismatch"
            # Organize input tensors by step instead of by sequence.
            accepted_token_ids_by_step = logits.flatten()
            accepted_token_ids_by_step = accepted_token_ids_by_step.tolist()

            step_output_token_ids = []
            for i, seq_id in enumerate(seq_ids):
                token_id = accepted_token_ids_by_step[i]
                step_output_token_ids.append(CompletionSequenceGroupOutput(samples=[SequenceOutput(parent_seq_id=seq_id, output_token=token_id, logprobs={token_id: Logprob(token_id)})], prompt_logprobs=None))
            return SamplerOutput(outputs=step_output_token_ids)
        else:
            return self.sampler(logits, sampling_metadata)

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


class NeuronSpeculationCasualLM(nn.Module):
    def __init__(
        self,
        config: PretrainedConfig,
    ) -> None:
        super().__init__()
        self.config = config
        self.logits_processor = LogitsProcessor(config.vocab_size,
                                                logits_as_input=True)
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
        if output.fused_outputs[1].shape[-1] == 1:
            # CTX encoding
            return output.fused_outputs[1].view(1, -1)
        draft_new_tokens = output.fused_outputs[0].view(1, -1)
        target_tokens = output.fused_outputs[1].view(1, -1)
        if self.config.neuron_config.enable_eagle_speculation:
            candidate_new_tokens = draft_new_tokens[:, 1:]
        else:
            candidate_new_tokens = draft_new_tokens[:,:-1]
        selected_tokens = target_tokens[:,:-1]
        n_matches = ((~(candidate_new_tokens == selected_tokens)).cumsum(dim=-1) < 1).sum()
        accepted_tokens = target_tokens[:,:n_matches+1]
        return accepted_tokens

    def sample(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[List[SamplerOutput]]:
        batch_size, num_steps = logits.shape
        seq_ids = [seq_id for sg in sampling_metadata.seq_groups for seq_id in sg.seq_ids]
        # Organize input tensors by step instead of by sequence.
        accepted_token_ids_by_step = logits.transpose(0, 1)
        accepted_token_ids_by_step = accepted_token_ids_by_step.tolist()

        sampler_output_list = []
        for step_index in range(num_steps):
            if all(token_id == -1 for token_id in accepted_token_ids_by_step[step_index]):
                break
            step_output_token_ids = []
            for sequence_index in range(batch_size):
                token_id = accepted_token_ids_by_step[step_index][sequence_index]
                step_output_token_ids.append(CompletionSequenceGroupOutput(samples=[SequenceOutput(parent_seq_id=seq_ids[sequence_index], output_token=token_id, logprobs={token_id: Logprob(token_id)})], prompt_logprobs=None))
            sampler_output_list.append(
                SamplerOutput(outputs=step_output_token_ids))
        return sampler_output_list

    def load_weights(self, model_name_or_path: str, draft_model_name_or_path: str, **kwargs):
        arch = _get_model_architecture(self.config)
        neuronx_module_path, neuronx_model_cls_name = (
            _NEURON_SUPPORTED_MODELS[arch])
        neuronx_module = importlib.import_module(neuronx_module_path)
        neuronx_model_cls = getattr(neuronx_module, neuronx_model_cls_name)
        neuron_config = neuronx_model_cls.get_neuron_config_cls()(**kwargs['neuron_config'])
        config = neuronx_model_cls.get_config_cls()(
            neuron_config, load_config=load_pretrained_config(model_name_or_path)
        )

        draft_neuron_config = copy.deepcopy(config.neuron_config)
        if not config.neuron_config.enable_eagle_speculation:
            draft_neuron_config.speculation_length = 0
        draft_neuron_config.trace_tokengen_model = True
        draft_neuron_config.enable_fused_speculation = False
        if config.neuron_config.enable_eagle_speculation:
            draft_neuron_config.is_eagle_draft = True
            draft_neuron_config.sequence_parallel_enabled = False
        draft_config = neuronx_model_cls.get_config_cls()(
            draft_neuron_config, load_config=load_pretrained_config(draft_model_name_or_path)
        )
        fused_spec_config = FusedSpecNeuronConfig(neuronx_model_cls._model_cls, draft_config=draft_config, draft_model_path=draft_model_name_or_path)
        config.fused_spec_config = fused_spec_config
        self.config.neuron_config = neuron_config

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
        if draft_model_name_or_path == model_name_or_path:
            draft_checkpoint_download = False
        if not os.path.exists(model_name_or_path):
            hf_model = AutoModelForCausalLM.from_pretrained(model_name_or_path)
            saved_path = os.path.join("local-models", model_name_or_path)
            hf_model.save_pretrained(saved_path)
            model_name_or_path = saved_path
        if not os.path.exists(draft_model_name_or_path):
            if draft_checkpoint_download:
                hf_model = AutoModelForCausalLM.from_pretrained(draft_model_name_or_path)
                saved_path = os.path.join("local-models", draft_model_name_or_path)
                hf_model.save_pretrained(saved_path)
                draft_model_name_or_path = saved_path
            else:
                draft_model_name_or_path = model_name_or_path
            config.fused_spec_config.draft_model_path = draft_model_name_or_path
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

def _get_default_neuron_speculation_config(model_config: ModelConfig,
                                           parallel_config: ParallelConfig,
                                           scheduler_config: SchedulerConfig,
                                           speculation_config: SpeculativeConfig):
    neuron_config = dict(
        tp_degree=parallel_config.tensor_parallel_size,
        batch_size=scheduler_config.max_num_seqs,
        max_context_length=scheduler_config.max_model_len,
        seq_len=scheduler_config.max_model_len,
        speculation_length=speculation_config.num_speculative_tokens,
        trace_tokengen_model=False,
        enable_fused_speculation=True,
        enable_bucketing=True,
        quantized=False,
        torch_dtype=TORCH_DTYPE_TO_NEURON_AMP[model_config.dtype],
        on_device_sampling_config= dict(top_k=1, do_sample=False,)
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

def get_neuron_speculation_model(model_config: ModelConfig,
                                 parallel_config: ParallelConfig,
                                 scheduler_config: SchedulerConfig,
                                 speculation_config: SpeculativeConfig):
    model = NeuronSpeculationCasualLM(model_config.hf_config)
    default_neuron_config_args = _get_default_neuron_speculation_config(
        model_config, parallel_config, scheduler_config, speculation_config)
    neuron_config = _get_neuron_config_after_override(default_neuron_config_args,
        model_config.override_neuron_config)
    model.load_weights(model_config.model,
                       speculation_config.draft_model_config.model,
                       neuron_config=neuron_config,)
    return model.eval()
