# SPDX-License-Identifier: Apache-2.0

import os
from dataclasses import dataclass
from itertools import chain
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set, Tuple, Union

import torch
from torch import nn

from vllm.config import DeviceConfig, VllmConfig
from vllm.distributed import get_kv_transfer_group
from vllm.logger import init_logger
from vllm.lora.layers import LoRAMapping
from vllm.lora.request import LoRARequest
from vllm.model_executor import SamplingMetadata
from vllm.model_executor.layers.sampler import SamplerOutput
from vllm.model_executor.model_loader.neuron import get_neuron_model
from vllm.multimodal import (MULTIMODAL_REGISTRY, BatchedTensorInputs,
                             MultiModalKwargs)
from vllm.sampling_params import SamplingParams
from vllm.sequence import IntermediateTensors, SequenceGroupMetadata
from vllm.utils import is_pin_memory_available, make_tensor_with_pad
from vllm.worker.model_runner_base import ModelRunnerBase, ModelRunnerInputBase
from vllm.worker.utils import use_neuronx_distributed, use_transformers_neuronx

if TYPE_CHECKING:
    from vllm.attention.backends.abstract import AttentionBackend

logger = init_logger(__name__)


@dataclass(frozen=True)
class ModelInputForNeuron(ModelRunnerInputBase):
    """
    Used by the NeuronModelRunner.
    """
    request_ids: Optional[List[str]] = None
    input_tokens: Optional[torch.Tensor] = None
    input_positions: Optional[torch.Tensor] = None
    input_block_ids: Optional[torch.Tensor] = None
    slot_mapping: Optional[torch.Tensor] = None
    input_block_tables: Optional[torch.Tensor] = None
    full_context_lens: Optional[torch.Tensor] = None
    computed_context_lens: Optional[torch.Tensor] = None
    sampling_metadata: SamplingMetadata = None
    multi_modal_kwargs: BatchedTensorInputs = None
    adapter_ids: Optional[str] = None

    def as_broadcastable_tensor_dict(
            self) -> Dict[str, Union[int, torch.Tensor]]:
        return {
            "request_ids": self.request_ids,
            "input_tokens": self.input_tokens,
            "input_positions": self.input_positions,
            "input_block_ids": self.input_block_ids,
            "sampling_metadata": self.sampling_metadata,
            "multi_modal_kwargs": self.multi_modal_kwargs,
        }

    @classmethod
    def from_broadcasted_tensor_dict(
        cls,
        tensor_dict: Dict[str, Any],
        attn_backend: Optional["AttentionBackend"] = None,
    ) -> "ModelInputForNeuron":
        return ModelInputForNeuron(
            request_ids=tensor_dict["request_ids"],
            input_tokens=tensor_dict["input_tokens"],
            input_positions=tensor_dict["input_positions"],
            input_block_ids=tensor_dict["input_block_ids"],
            sampling_metadata=tensor_dict["sampling_metadata"],
            multi_modal_kwargs=tensor_dict["multi_modal_kwargs"],
        )


class NeuronModelRunner(ModelRunnerBase[ModelInputForNeuron]):
    """A model runner for AWS Neuron hardware"""

    # NEURON has an upper limit on the top_k
    _MAX_NEURON_SAMPLING_TOP_K = 256

    # NOTE: Padding table id for slot mapping, note that this will be
    # used as the block index to update KV cache, so we need to make
    # sure no real tokens are mapped to this block_id, we current
    # assume that block 0 will never be used.
    _SLOT_MAPPING_PAD = -1
    _BLOCK_TABLE_PAD = 0

    def __init__(
        self,
        vllm_config: VllmConfig,
    ):
        ModelRunnerBase.__init__(self, vllm_config)

        if (self.model_config is not None
                and self.model_config.get_sliding_window()):
            logger.warning("Sliding window is not supported on Neuron. "
                           "The model will run without sliding window.")
        self.device_config = (self.device_config if self.device_config
                              is not None else DeviceConfig())

        self.lora_config = vllm_config.lora_config
        self.device = self.device_config.device
        self.pin_memory = is_pin_memory_available()

        self.is_block_kv_layout = self.cache_config.block_size \
                                != self.scheduler_config.max_model_len

        # Multi-modal data support
        self.multi_modal_input_mapper = MULTIMODAL_REGISTRY \
            .create_input_mapper(self.model_config)

        # Lazy initialization.
        self.model: nn.Module  # initialize after load_model.

        # Once NEURON_ON_DEVICE_SAMPLING_DISABLED is set to a non-zero value,
        # turn off on-device sampling.
        self._on_device_sampling_disabled = int(
            os.getenv("NEURON_ON_DEVICE_SAMPLING_DISABLED", "0"))

        # NEURON needs to update sampling parameters when request IDs change
        # across batches. This variable stores the previous batch's request IDs
        # to determine if an update is needed.
        self._previous_batch_request_ids: List[str] = []

        if not self._on_device_sampling_disabled:
            self._init_neuron_sampling()

    def _init_neuron_sampling(self) -> None:
        if use_transformers_neuronx():
            from transformers_neuronx.config import GenerationConfig
        else:
            from transformers import GenerationConfig
        logger.warning(
            "On-device sampling is turned on in Neuron by default, only "
            "top_k, top_p, and temperature are current supported sampling "
            "parameters. To turn off the on-device sampling, please set "
            "the environment variable NEURON_ON_DEVICE_SAMPLING_DISABLED=1.")
        self.model_config.neuron_sampling_params = GenerationConfig(
            max_length=self.scheduler_config.max_model_len,
            do_sample=True,
            per_batch_line=True,
            top_k=[self._MAX_NEURON_SAMPLING_TOP_K] \
                  * self.scheduler_config.max_num_seqs,
            top_p=[1.0] * self.scheduler_config.max_num_seqs,
            temperature=[1.0] * self.scheduler_config.max_num_seqs,
            dynamic=True,
            global_top_k=self._MAX_NEURON_SAMPLING_TOP_K)

    def load_model(self) -> None:
        self.model = get_neuron_model(self.model_config,
                                      parallel_config=self.parallel_config,
                                      scheduler_config=self.scheduler_config)

        # Disable prefix caching and chunked prefill by default
        self.is_prefix_caching = False
        self.is_chunked_prefill = False

    def get_model(self) -> nn.Module:
        return self.model

    def _prepare_prompt(
        self,
        seq_group_metadata_list: List[SequenceGroupMetadata],
    ) -> Tuple[List[str], torch.Tensor, torch.Tensor, torch.Tensor,
               torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
               List[int], BatchedTensorInputs]:
        assert len(seq_group_metadata_list) > 0
        request_ids: List[str] = []
        input_tokens: List[List[int]] = []
        input_positions: List[List[int]] = []
        input_block_ids: List[int] = []
        slot_mapping: List[List[int]] = []
        input_block_tables: List[List[int]] = []
        computed_context_lens: List[int] = []
        max_blocks_per_seq = self.scheduler_config.max_model_len \
                             // self.cache_config.block_size

        seq_lens: List[int] = []
        multi_modal_inputs_list: List[MultiModalKwargs] = []
        for seq_group_metadata in seq_group_metadata_list:

            request_ids.append(seq_group_metadata.request_id)
            assert seq_group_metadata.is_prompt
            seq_ids = list(seq_group_metadata.seq_data.keys())
            assert len(seq_ids) == 1
            seq_id = seq_ids[0]

            seq_data = seq_group_metadata.seq_data[seq_id]
            prompt_tokens = seq_data.get_token_ids()
            seq_len = len(prompt_tokens)
            seq_lens.append(seq_len)

            input_tokens.append(prompt_tokens)
            input_positions.append(list(range(seq_len)))

            assert seq_group_metadata.block_tables is not None
            block_table = seq_group_metadata.block_tables[seq_id]
            if self.is_prefix_caching:
                # pad the block_table to have the length of num_gpu_blocks
                padded_block_table = [self._BLOCK_TABLE_PAD
                                      ] * max_blocks_per_seq
                padded_block_table[:len(block_table)] = block_table[:]
                input_block_tables.append(padded_block_table)
                computed_tokens = len(seq_group_metadata.computed_block_nums
                                      ) * self.cache_config.block_size
                computed_context_lens.append(computed_tokens)

                slot_mapping_for_cur_seq = []
                for i in range(self.scheduler_config.max_model_len):
                    if i < seq_len:
                        block_number = block_table[
                            i // self.cache_config.block_size]
                        block_offset = i % self.cache_config.block_size
                        slot = block_number * self.cache_config.block_size \
                                + block_offset
                        slot_mapping_for_cur_seq.append(slot)
                    else:
                        slot_mapping_for_cur_seq.append(self._SLOT_MAPPING_PAD)
                # skip the computed_tokens
                slot_mapping.append(slot_mapping_for_cur_seq[computed_tokens:])
                input_tokens[-1] = input_tokens[-1]
                input_positions[-1] = input_positions[-1]

                input_block_ids.append(seq_id)
            else:
                assert len(block_table) == 1
                input_block_ids.append(block_table[0])

            mm_data = seq_group_metadata.multi_modal_data
            if mm_data:
                # Process multi-modal data
                mm_data = self.process_multi_modal_data_neuron(mm_data)
                multi_modal_inputs_list.append(mm_data)

        max_seq_len = max(seq_lens)
        assert max_seq_len > 0
        input_tokens = make_tensor_with_pad(input_tokens,
                                            pad=0,
                                            max_len=max_seq_len,
                                            dtype=torch.long,
                                            device=self.device)
        input_positions = make_tensor_with_pad(input_positions,
                                               pad=0,
                                               max_len=max_seq_len,
                                               dtype=torch.long,
                                               device=self.device)
        input_block_ids = torch.tensor(input_block_ids,
                                       dtype=torch.long,
                                       device=self.device)
        slot_mapping = make_tensor_with_pad(
            slot_mapping,
            pad=self._SLOT_MAPPING_PAD,
            max_len=self.scheduler_config.max_model_len,
            dtype=torch.long,
            device=self.device)
        input_block_tables = torch.tensor(input_block_tables,
                                          dtype=torch.long,
                                          device=self.device)
        full_context_lens = torch.tensor(seq_lens,
                                         dtype=torch.long,
                                         device=self.device).reshape(-1, 1)
        computed_context_lens = torch.tensor(computed_context_lens,
                                             dtype=torch.long,
                                             device=self.device).reshape(
                                                 -1, 1)

        multi_modal_kwargs = MultiModalKwargs.batch(multi_modal_inputs_list)

        return (request_ids, input_tokens, input_positions, input_block_ids,
                slot_mapping, input_block_tables, full_context_lens,
                computed_context_lens, seq_lens, multi_modal_kwargs)

    def _get_slots_for_speculation(
        self,
        position,
        block_table,
        block_size,
        speculation_length,
    ) -> List[int]:
        seq_slots = [
            range(block_idx * block_size, (block_idx + 1) * block_size)
            for block_idx in block_table
        ]
        flattened_seq_slots = list(chain(*seq_slots))
        return flattened_seq_slots[position:position + speculation_length]

    def _prepare_decode(
        self,
        seq_group_metadata_list: List[SequenceGroupMetadata],
    ) -> Tuple[List[str], torch.Tensor, torch.Tensor, torch.Tensor,
               torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        assert len(seq_group_metadata_list) > 0
        request_ids: List[str] = []
        input_tokens: List[List[int]] = []
        input_positions: List[List[int]] = []
        input_block_ids: List[int] = []
        full_context_lens: List[int] = []
        slot_mapping: List[List[int]] = []
        input_block_tables: List[List[int]] = []
        max_blocks_per_seq = self.scheduler_config.max_model_len \
                             // self.cache_config.block_size

        for seq_group_metadata in seq_group_metadata_list:
            assert not seq_group_metadata.is_prompt

            seq_ids = list(seq_group_metadata.seq_data.keys())

            for seq_id in seq_ids:
                request_ids.append(seq_group_metadata.request_id)
                seq_data = seq_group_metadata.seq_data[seq_id]
                generation_token = seq_data.get_last_token_id()
                input_tokens.append([generation_token])

                seq_len = seq_data.get_len()
                position = seq_len - 1
                input_positions.append([position])
                full_context_lens.append(seq_len)

                assert seq_group_metadata.block_tables is not None
                block_table = seq_group_metadata.block_tables[seq_id]
                if self.is_prefix_caching:
                    # pad the block_table to have the length of num_gpu_blocks
                    padded_block_table = [self._BLOCK_TABLE_PAD
                                          ] * max_blocks_per_seq
                    padded_block_table[:len(block_table)] = block_table[:]
                    input_block_tables.append(padded_block_table)

                    block_number = block_table[position //
                                               self.cache_config.block_size]
                    block_offset = position % self.cache_config.block_size
                    slot = block_number * self.cache_config.block_size \
                            + block_offset

                    if getattr(self, 'speculative_config', None) is not None:
                        slot_mapping_for_cur_seq = \
                            self._get_slots_for_speculation(
                            position, block_table,
                            self.cache_config.block_size,
                            self.speculative_config.num_speculative_tokens)
                    else:
                        slot_mapping_for_cur_seq = [slot]
                    slot_mapping.append(slot_mapping_for_cur_seq)

                    input_block_ids.append(seq_id)
                else:
                    assert len(block_table) == 1
                    input_block_ids.append(block_table[0])

        input_tokens = make_tensor_with_pad(input_tokens,
                                            pad=0,
                                            max_len=1,
                                            dtype=torch.long,
                                            device=self.device)
        input_positions = make_tensor_with_pad(input_positions,
                                               pad=0,
                                               max_len=1,
                                               dtype=torch.long,
                                               device=self.device)

        input_block_ids = torch.tensor(input_block_ids,
                                       dtype=torch.long,
                                       device=self.device)
        slot_mapping = torch.tensor(slot_mapping,
                                    dtype=torch.long,
                                    device=self.device)
        input_block_tables = torch.tensor(input_block_tables,
                                          dtype=torch.long,
                                          device=self.device)

        computed_context_lens_list = [
            length - 1 for length in full_context_lens
        ]

        full_context_lens = torch.tensor(full_context_lens,
                                         dtype=torch.long,
                                         device=self.device).reshape(-1, 1)
        # Convert computed_context_lens to tensor
        computed_context_lens = torch.tensor(computed_context_lens_list,
                                             dtype=torch.long,
                                             device=self.device).reshape(
                                                 -1, 1)

        return (request_ids, input_tokens, input_positions, input_block_ids,
                slot_mapping, input_block_tables, full_context_lens,
                computed_context_lens)

    def make_model_input_from_broadcasted_tensor_dict(
            self, tensor_dict: Dict[str, Any]) -> ModelInputForNeuron:
        return ModelInputForNeuron.from_broadcasted_tensor_dict(tensor_dict)

    def prepare_model_input(
        self,
        seq_group_metadata_list: List[SequenceGroupMetadata],
        virtual_engine: int = 0,
        finished_requests_ids: Optional[List[str]] = None
    ) -> ModelInputForNeuron:
        multi_modal_kwargs = None
        # NOTE: We assume that all sequences in the group are all prompts or
        # all decodes.
        is_prompt = seq_group_metadata_list[0].is_prompt
        # Prepare input tensors.
        if is_prompt:
            (request_ids, input_tokens, input_positions, input_block_ids,
             slot_mapping, input_block_tables, full_context_lens,
             computed_context_lens, seq_lens, multi_modal_kwargs
             ) = self._prepare_prompt(seq_group_metadata_list)
        else:
            (request_ids, input_tokens, input_positions, input_block_ids,
             slot_mapping, input_block_tables, full_context_lens,
             computed_context_lens
             ) = self._prepare_decode(seq_group_metadata_list)
            seq_lens = None

        if not self._on_device_sampling_disabled:
            for seq_group_metadata in seq_group_metadata_list:
                sampling_params = seq_group_metadata.sampling_params
                top_k, top_p, temperature = (
                    self._convert_to_neuron_sampling_params(sampling_params))
                sampling_params.top_k = top_k
                sampling_params.top_p = top_p
                sampling_params.temperature = temperature

        # we need multi_modal_data for later tokens as well
        multi_modal_inputs_list: List[MultiModalKwargs] = []
        for seq_group_metadata in seq_group_metadata_list:
            mm_data = seq_group_metadata.multi_modal_data
            if mm_data:
                multi_modal_inputs_list.append(mm_data)
        multi_modal_kwargs = MultiModalKwargs.batch(multi_modal_inputs_list)

        sampling_metadata = SamplingMetadata.prepare(
            seq_group_metadata_list,
            seq_lens,
            # query_lens is not needed if chunked prefill is not
            # supported. Since neuron worker doesn't support chunked prefill
            # just use seq_lens instead.
            seq_lens,
            self.device,
            self.pin_memory,
            generators=self.get_generators(finished_requests_ids))

        if use_transformers_neuronx(
        ) and not self._on_device_sampling_disabled:
            # Once the request IDs are changed in current iteration, we will
            # update the on-device sampling parameters.
            current_batch_request_ids = [
                seq_group_meta_data.request_id
                for seq_group_meta_data in seq_group_metadata_list
            ]
            if current_batch_request_ids != self._previous_batch_request_ids:
                self._update_neuron_sampling_params(seq_group_metadata_list)
                self._previous_batch_request_ids = current_batch_request_ids

        return ModelInputForNeuron(request_ids=request_ids,
                                   input_tokens=input_tokens,
                                   input_positions=input_positions,
                                   slot_mapping=slot_mapping,
                                   input_block_tables=input_block_tables,
                                   full_context_lens=full_context_lens,
                                   computed_context_lens=computed_context_lens,
                                   input_block_ids=input_block_ids,
                                   sampling_metadata=sampling_metadata,
                                   multi_modal_kwargs=multi_modal_kwargs)

    def _update_neuron_sampling_params(
            self, seq_group_metadata_list: List[SequenceGroupMetadata]):
        # Update Neuron sampling parameters (GenerationConfig in Neuron)
        current_sampling_params = self.model_config.neuron_sampling_params
        assert current_sampling_params is not None, (
            f"Failed to update sampling_params, "
            f"current sampling params is {current_sampling_params}")

        is_update_needed = False

        top_k = current_sampling_params.top_k
        top_p = current_sampling_params.top_p
        temperature = current_sampling_params.temperature

        # The index of a sequence's sampling parameters in neuron is equal to
        # its index in `input_block_ids`.
        for seq_group_metadata in seq_group_metadata_list:
            seq_ids = list(seq_group_metadata.seq_data.keys())
            sampling_params = seq_group_metadata.sampling_params

            seq_group_top_k = sampling_params.top_k
            seq_group_top_p = sampling_params.top_p
            seq_group_temperature = sampling_params.temperature

            for seq_id in seq_ids:
                index = seq_group_metadata.block_tables[seq_id][0]
                if (top_k[index] != seq_group_top_k
                        or top_p[index] != seq_group_top_p
                        or temperature[index] != seq_group_temperature):
                    is_update_needed = True

                top_k[index] = seq_group_top_k
                top_p[index] = seq_group_top_p
                temperature[index] = seq_group_temperature

        # update_generation_config is only available in transformers-neuronx
        if is_update_needed and use_transformers_neuronx():
            self.model.model.update_generation_config(current_sampling_params)

    def _convert_to_neuron_sampling_params(
            self, sampling_params: SamplingParams) -> Tuple[int, float, float]:
        # Returns the top_k, top_p and temperature parameters for neuron.
        top_k = sampling_params.top_k
        top_p = sampling_params.top_p
        temperature = sampling_params.temperature

        if temperature == 0.0:
            # Enable greedy sampling on zero temperature
            return (1, 1.0, 1.0)
        if top_k < 0 or top_k > self._MAX_NEURON_SAMPLING_TOP_K:
            top_k = self._MAX_NEURON_SAMPLING_TOP_K

        return (top_k, top_p, temperature)

    @torch.inference_mode()
    def execute_model(
        self,
        model_input: ModelInputForNeuron,
        kv_caches: Optional[List[torch.Tensor]] = None,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        num_steps: int = 1,
    ) -> Optional[List[SamplerOutput]]:
        if num_steps > 1:
            raise ValueError(
                "NeuronModelRunner does not support multi-step execution.")

        # extract top_k, top_p and temperature from model_input for neuron
        # forward call
        sampling_params = (torch.tensor([[
            seq_group.sampling_params.top_k, seq_group.sampling_params.top_p,
            seq_group.sampling_params.temperature
        ] for seq_group in model_input.sampling_metadata.seq_groups]))

        if use_neuronx_distributed():
            bypass_model_exec = False
            if self.need_recv_kv(model_input):
                # It doesn't trigger KV cache transfer here which
                # could block decode, transfer was trigger during scheduler
                # and completed at this point, so here we directly
                # get hidden_states (output tokens with on-device sampling)
                # from connector
                hidden_states = get_kv_transfer_group(
                ).recv_kv_caches_and_hidden_states(
                    None,
                    model_input,
                    None,
                )

                assert bypass_model_exec, (
                    "bypass_model_exec is not True "
                    "for decode worker when processing prompt")

            logger.debug("bypass_model_exec: %s", bypass_model_exec)

            if not bypass_model_exec:
                hidden_states = self.model(
                    input_ids=model_input.input_tokens,
                    positions=model_input.input_positions,
                    input_block_ids=model_input.input_block_ids,
                    slot_mapping=model_input.slot_mapping,
                    input_block_tables=model_input.input_block_tables,
                    full_context_lens=model_input.full_context_lens,
                    computed_context_lens=model_input.computed_context_lens,
                    sampling_params=sampling_params,
                    adapter_ids=model_input.adapter_ids,
                    **MultiModalKwargs.as_kwargs(model_input.multi_modal_kwargs
                                                 or {},
                                                 device=self.device),
                )

            if self.need_send_kv(model_input):
                logger.debug("Sending KV cache")

                get_kv_transfer_group().async_send_kv_caches(
                    None, model_input, None, hidden_states)

        elif use_transformers_neuronx():
            # [TODO] validate on-device sampling
            # The model signature may need change for on-device sampling
            hidden_states = self.model(
                input_ids=model_input.input_tokens,
                positions=model_input.input_positions,
                input_block_ids=model_input.input_block_ids,
                **MultiModalKwargs.as_kwargs(model_input.multi_modal_kwargs
                                             or {},
                                             device=self.device),
            )

        # Compute the logits only if the on-device sampling is turned off as
        # on-device sampling outputs the token ids.
        if self._on_device_sampling_disabled:
            logits = self.model.compute_logits(hidden_states,
                                               model_input.sampling_metadata)
        else:
            logits = hidden_states

        # Sample the next token.
        output = self.model.sample(
            logits=logits,
            sampling_metadata=model_input.sampling_metadata,
        )
        return [output]

    @property
    def vocab_size(self) -> int:
        return self.model_config.get_vocab_size()

    def process_multi_modal_data_neuron(self, mm_data):
        # this is a no-op for NeuronModelRunner
        return mm_data

    def need_recv_kv(self, model_input) -> bool:
        if self.vllm_config.kv_transfer_config is None:
            return False

        is_prefill_run = (model_input.input_positions[:, 0]).sum().item() == 0

        return self.vllm_config.kv_transfer_config.is_kv_consumer \
            and is_prefill_run

    def need_send_kv(self, model_input) -> bool:
        if self.vllm_config.kv_transfer_config is None:
            return False

        is_prefill_run = (model_input.input_positions[:, 0]).sum().item() == 0

        return self.vllm_config.kv_transfer_config.is_kv_producer \
            and is_prefill_run

    def remove_all_loras(self):
        raise NotImplementedError(
            "LoRAs are not supported for Transformers NeuronX framework")

    def set_active_loras(self, lora_requests: Set[LoRARequest],
                         lora_mapping: LoRAMapping) -> None:
        raise NotImplementedError(
            "LoRAs are not supported for Transformers NeuronX framework")

    def add_lora(self, lora_request: LoRARequest):
        raise NotImplementedError(
            "LoRAs are not supported for Transformers NeuronX framework")

    def remove_lora(self, lora_id: int) -> bool:
        raise NotImplementedError(
            "LoRAs are not supported for Transformers NeuronX framework")

    def pin_lora(self, lora_id: int) -> bool:
        raise NotImplementedError(
            "LoRAs are not supported for Transformers NeuronX framework")

    def list_loras(self) -> Set[int]:
        raise NotImplementedError(
            "LoRAs are not supported for Transformers NeuronX framework")
