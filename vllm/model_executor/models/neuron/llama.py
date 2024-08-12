"""Inference-only LLaMA model compatible with HuggingFace weights."""
import os
from typing import List, Optional, Tuple

import torch
from torch import nn
from transformers import LlamaConfig

from vllm.model_executor.input_metadata import InputMetadata
from vllm.model_executor.layers.sampler import Sampler
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.sequence import SamplerOutput

KVCache = Tuple[torch.Tensor, torch.Tensor]


class LlamaForCausalLM(nn.Module):

    def __init__(
        self,
        config: LlamaConfig,
        linear_method=None,
    ) -> None:
        super().__init__()
        self.config = config
        self.linear_method = linear_method
        self.model = None
        self.sampler = Sampler(config.vocab_size)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[KVCache],
        input_metadata: InputMetadata,
    ) -> torch.Tensor:


        with torch.inference_mode():
            block_size = self.model.config.n_positions
            if input_metadata.is_prompt:
                seq_ids = input_metadata.slot_mapping[:, 0] // block_size
            else:
                seq_ids = input_metadata.block_tables


            output = self.model(input_ids,
                                attention_mask=None,
                                position_ids=positions,
                                seq_ids=seq_ids.flatten() - 1)
        return output.logits[:, -1, :]

    def sample(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[SamplerOutput]:
        next_tokens = self.sampler(None,
                                   hidden_states, sampling_metadata)
        return next_tokens

    def load_weights(self,
                     model_name_or_path: str,
                     cache_dir: Optional[str] = None,
                     load_format: str = "auto",
                     revision: Optional[str] = None,
                     **kwargs):

        from llama2.neuron_modeling_llama import NeuronLlamaForCausalLM, NeuronLlamaConfig, NeuronLlamaModel, preshard_hook_fn
        from neuronx_distributed.parallel_layers.checkpointing import _invoke_preshard_hook


        from transformers import LlamaForCausalLM as LlamaForCausalLMHF
        from transformers import AutoConfig
        config = NeuronLlamaConfig.from_pretrained(model_name_or_path)

        config.tp_degree = kwargs["tp_degree"]
        config.max_batch_size = kwargs["batch_size"]
        config.torch_dtype = kwargs["amp"]
        config.n_positions = kwargs["n_positions"][-1]
        config.max_length = config.n_positions
        config.max_context_length = kwargs["context_length_estimate"][-1]
        config.max_new_tokens = config.max_length - config.max_context_length
        config.buckets = [config.n_positions]
        config.tkg_batch_size = kwargs["batch_size"]
        config.ctx_batch_size = 1
        config.attn_cls = 'NeuronLlamaAttention'
        config.padding_side = "right"
        config.is_continuous_batching = True

        print(config)

        cpu_mode = os.environ.get("NXD_CPU", None)

        if os.environ.get("NXD_DEBUG", None):
            from imp import reload

            import logging

            reload(logging)
            logging.basicConfig(level=logging.DEBUG)


        # need to save to local if the model paht doesn't exist
        if not os.path.exists(model_name_or_path):

            model = LlamaForCausalLMHF.from_pretrained(model_name_or_path)

            saved_path = os.path.join("local-models", model_name_or_path)
            model.save_pretrained(saved_path)

            model_name_or_path = saved_path

        if cpu_mode is not None:
            config.tp_degree = 1


            hf_model =  LlamaForCausalLMHF.from_pretrained(model_name_or_path)

            model_sd = hf_model.model.state_dict()
            lm_head_sd = hf_model.lm_head.state_dict()
            model_sd["lm_head.weight"] = lm_head_sd["weight"]
            state_dict = model_sd

            llama_model = NeuronLlamaModel(config)
            _invoke_preshard_hook(preshard_hook_fn, llama_model, state_dict)

            self.model = NeuronLlamaForCausalLM("", config)
            config.batch_size = config.ctx_batch_size
            config.n_active_tokens = config.n_positions


            llama_model_ctx = NeuronLlamaModel.from_pretrained(None, config=config, state_dict=state_dict)
            llama_model_ctx.lm_head = hf_model.lm_head

            config.batch_size = config.tkg_batch_size
            config.n_active_tokens = 1
            llama_model_tkg = NeuronLlamaModel.from_pretrained(None, config=config, state_dict=state_dict)
            llama_model_tkg.lm_head = hf_model.lm_head

            self.model.context_encoding_model.model = llama_model_ctx
            self.model.token_generation_model.model = llama_model_tkg
        else:
            self.model = NeuronLlamaForCausalLM.from_pretrained(model_name_or_path, config)
            self.model.to_neuron()
