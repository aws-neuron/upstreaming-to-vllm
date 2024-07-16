"""Inference-only DBRX model compatible with HuggingFace weights."""
import os
from typing import List, Optional, Tuple

import torch
from torch import nn
from transformers import DbrxConfig

from vllm.model_executor.input_metadata import InputMetadata
from vllm.model_executor.layers.sampler import Sampler
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.sequence import SamplerOutput
import neuronx_distributed as nxd


KVCache = Tuple[torch.Tensor, torch.Tensor]


class DbrxForCausalLM(nn.Module):

    def __init__(
        self,
        config: DbrxConfig,
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
        # Need to add path of NeuronxDistributed/examples/inference to the PYTHONPATH for a successful import
        from dbrx.neuron_modeling_dbrx import NeuronDbrxForCausalLM, NeuronDbrxConfig, NeuronDbrxModel, preshard_hook_fn
        from neuronx_distributed.parallel_layers.checkpointing import _invoke_preshard_hook
        from transformers import DbrxForCausalLM as DbrxForCausalLMHF

        config = NeuronDbrxConfig.from_pretrained(model_name_or_path)
        config.tp_degree = kwargs["tp_degree"]
        config.max_batch_size = kwargs["batch_size"]
        config.torch_dtype = kwargs["amp"]
        config.n_positions = kwargs["n_positions"][-1]
        config.buckets = [config.n_positions]
        config.tkg_batch_size = kwargs["batch_size"]
        config.ctx_batch_size = 1
        config.attn_cls = 'NeuronLlamaAttention'
        config.padding_side = "right"
        config.is_continuous_batching = True
        config.do_sample = True
        config.top_k = 1
        config.quantized = False

        print(config)

        if os.environ.get("NXD_DEBUG", None):
            from imp import reload
            import logging

            reload(logging)
            logging.basicConfig(level=logging.DEBUG)

        # need to save to local if the model path doesn't exist
        if not os.path.exists(model_name_or_path):

            model = DbrxForCausalLMHF.from_pretrained(model_name_or_path)

            saved_path = os.path.join("local-models", model_name_or_path)
            model.save_pretrained(saved_path)

            model_name_or_path = saved_path

        cpu_mode = os.environ.get("NXD_CPU", None)
        if cpu_mode is not None:
            config.tp_degree = 1

            self.init_ditributed_env()
            dbrx_model = NeuronDbrxModel(config)
            state_dict = NeuronDbrxForCausalLM.get_state_dict(model_name_or_path, config)
            _invoke_preshard_hook(dbrx_model, state_dict)
            dbrx_model.load_state_dict(state_dict, strict=False)

            config.torch_dtype = torch.float32

            self.model = NeuronDbrxForCausalLM("", config)
            config.batch_size = config.ctx_batch_size
            config.n_active_tokens = config.n_positions
            dbrx_model_ctx = NeuronDbrxModel.from_pretrained(None, config=config, state_dict=state_dict)

            config.batch_size = config.tkg_batch_size
            config.n_active_tokens = 1
            dbrx_model_tkg = NeuronDbrxModel.from_pretrained(None, config=config, state_dict=state_dict)

            self.model.context_encoding_model.model = dbrx_model_ctx
            self.model.token_generation_model.model = dbrx_model_tkg
        else:
            self.model = NeuronDbrxForCausalLM.from_pretrained(model_name_or_path, config)
            self.model.to_neuron()


    def init_ditributed_env(self):
        """
        Initialize a simple neuronx distributed (Tensor Parallelism) environment, where there TP degree is 1.

        This function is just for running NeuronxDistributed models on CPU to validate correctness.
        """
        os.environ["RANK"] = str(0)
        os.environ["WORLD_SIZE"] = str(1)
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "2024"

        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(backend="xla")

        nxd.parallel_layers.parallel_state.destroy_model_parallel()
        nxd.parallel_layers.parallel_state.initialize_model_parallel(tensor_model_parallel_size=1)