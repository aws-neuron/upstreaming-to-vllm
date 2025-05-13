# SPDX-License-Identifier: Apache-2.0
import copy
from typing import List, Optional, Set, Tuple

import torch
from neuronx_distributed_inference.models.mllama.image_transform import (
    custom_image_preprocessing)
from neuronx_distributed_inference.modules.generation.sampling import (
    prepare_sampling_params)
from neuronx_distributed_inference.modules.lora_serving import (
    LoraCheckpoint, LoraServingConfig)
from PIL.Image import Image as PILImage

from vllm.config import VllmConfig
from vllm.distributed import get_kv_transfer_group
from vllm.entrypoints.openai.serving_models import LoRAModulePath
from vllm.logger import init_logger
from vllm.lora.layers import LoRAMapping
from vllm.lora.request import LoRARequest
from vllm.model_executor import SamplingMetadata
from vllm.model_executor.layers.sampler import SamplerOutput
from vllm.model_executor.model_loader.neuronx_distributed import (
    _get_model_architecture, get_neuron_model)
from vllm.multimodal import MultiModalKwargs
from vllm.sequence import IntermediateTensors, SequenceGroupMetadata
from vllm.worker.neuron_model_runner import (ModelInputForNeuron,
                                             NeuronModelRunner)

logger = init_logger(__name__)


class NeuronxDistributedModelRunner(NeuronModelRunner):

    def __init__(
        self,
        vllm_config: VllmConfig,
    ):
        super().__init__(vllm_config)
        self.lora_checkpoint = None
        self.model = None
        self.lora_serving_config = None

        # Init attributes for features on block kv, and they will be
        # overridden in self.load_model()
        block_size = vllm_config.cache_config.block_size
        max_model_len = vllm_config.scheduler_config.max_model_len
        self.is_block_kv_layout = block_size != max_model_len
        self.is_prefix_caching = False
        self.is_chunked_prefill = False

    @staticmethod
    def _get_lora_paths_strings(lora_modules: List[LoRAModulePath]):
        if not lora_modules:
            return None
        return {_.name: _.path for _ in lora_modules}

    def _get_nxdi_lora_config(self):
        override_neuron_config = self.model_config.override_neuron_config
        lora_modules = override_neuron_config.pop("lora_modules", None)
        target_modules = override_neuron_config.pop("target_modules", None)
        lora_ckpt_paths = self._get_lora_paths_strings(lora_modules)
        if self.lora_config.max_loras < len(lora_ckpt_paths):
            raise ValueError(
                "Number of LoRAs (%s) exceeds maximum "
                "allowed (%s)", len(lora_ckpt_paths),
                self.lora_config.max_loras)

        return LoraServingConfig(
            max_loras=self.lora_config.max_loras,
            max_lora_rank=self.lora_config.max_lora_rank,
            target_modules=target_modules,
            lora_ckpt_paths=lora_ckpt_paths,
        )

    def load_model(self) -> None:
        # Update LoRA config
        if self.lora_config is not None:
            self.lora_serving_config = self._get_nxdi_lora_config()
            self.lora_checkpoint = LoraCheckpoint(self.lora_serving_config)
        self.model = get_neuron_model(
            self.model_config,
            cache_config=self.cache_config,
            parallel_config=self.parallel_config,
            scheduler_config=self.scheduler_config,
            lora_serving_config=self.lora_serving_config)
        self.is_block_kv_layout = self.model.neuron_config.is_block_kv_layout
        self.is_prefix_caching = self.model.neuron_config.is_prefix_caching
        self.is_chunked_prefill = \
            self.model.neuron_config.chunked_prefill_config is not None
        self.model.is_reorder_needed = not self.is_block_kv_layout

    def get_nxd_sampling_params(self, sampling_metadata):
        if self.model.config.neuron_config.on_device_sampling_config:
            max_topk = (self.model.config.neuron_config.
                        on_device_sampling_config.global_topk)
        else:
            max_topk = self.model.config.vocab_size

        top_k = [1] * self.scheduler_config.max_num_seqs
        top_p = [1.0] * self.scheduler_config.max_num_seqs
        temperature = [1.0] * self.scheduler_config.max_num_seqs

        for index, sequenceGroupToSample in enumerate(
                sampling_metadata.seq_groups):
            top_k[index] = (sequenceGroupToSample.sampling_params.top_k
                            if sequenceGroupToSample.sampling_params.top_k > 0
                            else max_topk)
            top_p[index] = sequenceGroupToSample.sampling_params.top_p
            temperature[index] = (
                sequenceGroupToSample.sampling_params.temperature)

        sampling_params = prepare_sampling_params(
            batch_size=self.scheduler_config.max_num_seqs,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature)

        # TODO: support different sampling params for chunked prefill
        if self.is_chunked_prefill:
            sampling_params = sampling_params[:1, :]
        return sampling_params

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

        is_mllama = _get_model_architecture(
            self.model.config) == "MllamaForConditionalGeneration"

        if is_mllama:
            return self.execute_model_for_mllama(
                model_input,
                kv_caches,
                intermediate_tensors,
                num_steps,
            )
        else:
            return self.execute_model_for_text(
                model_input,
                kv_caches,
                intermediate_tensors,
                num_steps,
            )

    def execute_model_for_text(
        self,
        model_input: ModelInputForNeuron,
        kv_caches: Optional[List[torch.Tensor]] = None,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        num_steps: int = 1,
    ) -> Optional[List[SamplerOutput]]:

        sampling_params = self.get_nxd_sampling_params(
            model_input.sampling_metadata)

        model_executable = self.model

        bypass_model_exec = False
        if self.need_recv_kv(model_input):
            # It doesn't trigger KV cache transfer here which
            # could block decode, transfer was trigger during scheduler
            # and completed at this point, so here we directly
            # get hidden_states (output tokens with on-device sampling)
            # from connector
            hidden_states, bypass_model_exec, model_input = \
            get_kv_transfer_group().recv_kv_caches_and_hidden_states(
                model_executable,
                model_input,
                kv_caches,
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
                prefill_completion_state=model_input.prefill_completion_state,
                **MultiModalKwargs.as_kwargs(model_input.multi_modal_kwargs
                                             or {},
                                             device=self.device),
            )

        if self.need_send_kv(model_input):
            logger.debug(
                "Sending KV cache, model output, and hidden_states (if "
                "EAGLE).")
            get_kv_transfer_group().send_kv_caches_and_hidden_states(
                # model_executable is used to know which layer the current
                # worker is working on, so that we can send KV for only
                # those layers.
                model_executable,
                model_input,
                kv_caches,
                hidden_states,
            )

        sampled_output = self._sample(hidden_states, model_input)
        return [sampled_output]

    def execute_model_for_mllama(
        self,
        model_input: ModelInputForNeuron,
        kv_caches: Optional[List[torch.Tensor]] = None,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        num_steps: int = 1,
    ) -> Optional[List[SamplerOutput]]:

        sampling_params = self.get_nxd_sampling_params(
            model_input.sampling_metadata)

        if model_input.multi_modal_kwargs.get('image') is not None:
            hidden_states = self.model(
                input_ids=model_input.input_tokens,
                positions=model_input.input_positions,
                seq_ids=model_input.input_block_ids,
                pixel_values=model_input.multi_modal_kwargs.get(
                    'pixel_values'),
                aspect_ratios=model_input.multi_modal_kwargs.get(
                    'aspect_ratios'),
                sampling_params=sampling_params,
                num_chunks=model_input.multi_modal_kwargs.get('num_chunks'),
                has_image=model_input.multi_modal_kwargs.get('has_image'),
            )
        else:
            empty_pixel_values = torch.zeros([1, 1, 4, 3, 560, 560],
                                             dtype=torch.bfloat16)
            empty_aspect_ratios = torch.ones([1, 1, 2], dtype=torch.int64)
            num_chunks = torch.tensor([[1]
                                       ])  # dummy num_chunks, will not be used
            has_image = torch.tensor([0])
            hidden_states = self.model(
                input_ids=model_input.input_tokens,
                positions=model_input.input_positions,
                seq_ids=model_input.input_block_ids,
                pixel_values=empty_pixel_values,
                aspect_ratios=empty_aspect_ratios,
                sampling_params=sampling_params,
                num_chunks=num_chunks,
                has_image=has_image,
            )

        output = self.model.sample(
            hidden_states=hidden_states,
            sampling_metadata=model_input.sampling_metadata,
        )

        return [output]

    def process_multi_modal_data_neuron(self, mm_data):
        pixel_values_list = []
        aspect_ratios_list = []
        num_chunks_list = []
        has_image_list = []
        images = mm_data.get('image')

        if isinstance(images, PILImage):
            images = [images]
        if isinstance(images, torch.Tensor):
            images = [images]

        for input_image in images:
            if isinstance(input_image, PILImage):
                pixel_values, aspect_ratios, num_chunks = \
                    custom_image_preprocessing(self.model.config, \
                                               [[input_image]])
                has_image = torch.tensor([1])
                image_tensors = [pixel_values.bfloat16().clone().detach(), \
                                 aspect_ratios, num_chunks, has_image]
            else:
                empty_pixel_values = torch.zeros([1, 1, 4, 3, 560, 560], \
                                                 dtype=torch.bfloat16)
                empty_aspect_ratios = torch.ones([1, 1, 2], dtype=torch.int64)
                # dummy num_chunks, will not be used
                num_chunks = torch.tensor([[1]])
                has_image = torch.tensor([0])
                image_tensors = [empty_pixel_values, empty_aspect_ratios, \
                                 num_chunks, has_image ]

            pixel_values_list.append(image_tensors[0])
            aspect_ratios_list.append(image_tensors[1])
            num_chunks_list.append(image_tensors[2])
            has_image_list.append(image_tensors[3])

        mm_data["pixel_values"] = torch.cat(pixel_values_list,
                                            dim=0).squeeze(0)
        mm_data["aspect_ratios"] = torch.cat(aspect_ratios_list, dim=0)\
            .squeeze(0)
        mm_data["num_chunks"] = torch.cat(num_chunks_list, dim=0).squeeze(0)
        mm_data["has_image"] = torch.cat(has_image_list, dim=0).squeeze(0)

        return mm_data

    def _get_lora_adapter_ids(self, seq_group_metadata_list):
        # set LoRA adapter IDs for multi-lora serving
        batch_size = len(seq_group_metadata_list)
        if self.lora_checkpoint is not None:
            # "0" indicates NxDI to use the base model for inference
            adapter_ids = ["0"] * batch_size
            for idx, seq_group_metadata in enumerate(seq_group_metadata_list):
                if seq_group_metadata.lora_request is not None:
                    adapter_ids[
                        idx] = seq_group_metadata.lora_request.lora_name

            # convert adapter_ids from strings to integers
            adapter_ids = self.lora_checkpoint.convert_adapter_ids_to_indices(
                adapter_ids, batch_size)
        else:
            adapter_ids = torch.zeros((batch_size), dtype=torch.int32)

        return adapter_ids

    def _prepare_chunked_prefill_inputs(
        self,
        seq_group_metadata_list: List[SequenceGroupMetadata],
    ) -> Tuple[List[str], torch.Tensor, torch.Tensor, torch.Tensor, torch.
               Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        request_ids: List[str] = []
        # prompt tokens for each request
        input_tokens: List[int] = []
        # token position for each request
        input_positions: List[int] = []
        # seq id for each request
        input_block_ids: List[int] = []
        # full context, including all previous context and current context
        full_context_lens: List[int] = []
        # previous context len
        computed_context_lens: List[int] = []
        # slots for each token
        slot_mapping: List[int] = []
        # block tables for all request
        input_block_tables: List[List[int]] = []
        # ready for sampling or not
        prefill_completion_state: List[bool] = []

        for seq_group_metadata in seq_group_metadata_list:
            seq_ids = list(seq_group_metadata.seq_data.keys())
            request_ids.append(seq_group_metadata.request_id)

            if seq_group_metadata.is_prompt:
                assert len(seq_ids) == 1
                seq_id = seq_ids[0]
                seq_data = seq_group_metadata.seq_data[seq_id]
                block_table = copy.deepcopy(
                    seq_group_metadata.block_tables[seq_id])

                cur_prompt_start_idx = seq_data.get_num_computed_tokens()
                cur_prompt_len = seq_group_metadata.token_chunk_size
                cur_prompt_end_idx = cur_prompt_start_idx + cur_prompt_len
                # seq_data always contains the full prompt
                cur_prompt = seq_data.prompt_token_ids[
                    cur_prompt_start_idx:cur_prompt_end_idx]
                input_tokens.extend(cur_prompt)

                input_positions.extend(
                    list(range(cur_prompt_start_idx, cur_prompt_end_idx)))

                input_block_ids.append(seq_id)

                # always get slot mapping from block table and don't
                # assume slot mapping is always continuous
                for i in range(cur_prompt_start_idx, cur_prompt_end_idx):
                    block_number = block_table[i //
                                               self.cache_config.block_size]
                    block_offset = i % self.cache_config.block_size
                    slot = block_number * self.cache_config.block_size + \
                        block_offset
                    slot_mapping.append(slot)

                input_block_tables.append(block_table)

                full_context_lens.append(cur_prompt_end_idx)
                computed_context_lens.append(cur_prompt_start_idx)
                # start sampling only if it prefill all the context
                prefill_completion_state.append(
                    cur_prompt_end_idx >= len(seq_data.prompt_token_ids))
            else:
                for seq_id in seq_ids:
                    seq_data = seq_group_metadata.seq_data[seq_id]
                    generation_token = seq_data.get_last_token_id()
                    input_tokens.append(generation_token)

                    seq_len = seq_data.get_len()
                    position = seq_len - 1
                    input_positions.append(position)

                    input_block_ids.append(seq_id)

                    block_table = copy.deepcopy(
                        seq_group_metadata.block_tables[seq_id])
                    block_number = block_table[position //
                                               self.cache_config.block_size]
                    block_offset = position % self.cache_config.block_size
                    slot = block_number * self.cache_config.block_size + \
                        block_offset
                    slot_mapping.append(slot)

                    input_block_tables.append(block_table)

                    full_context_lens.append(seq_len)

                    computed_context_lens.append(position)

                    prefill_completion_state.append(True)

        input_tokens = torch.tensor(input_tokens,
                                    dtype=torch.long,
                                    device=self.device).reshape(1, -1)
        input_positions = torch.tensor(input_positions,
                                       dtype=torch.long,
                                       device=self.device).reshape(1, -1)
        # see all requests as one prompt due to concatenation
        input_block_ids = torch.tensor(input_block_ids[:1],
                                       dtype=torch.long,
                                       device=self.device)
        slot_mapping = torch.tensor(slot_mapping,
                                    dtype=torch.long,
                                    device=self.device)
        # Need to pad to max
        max_num_blocks = max([len(x) for x in input_block_tables])
        for i in range(len(input_block_tables)):
            block_id_list = input_block_tables[i]
            current_num_blocks = len(block_id_list)
            pad_len = max_num_blocks - current_num_blocks
            input_block_tables[i].extend([self._BLOCK_TABLE_PAD] * pad_len)

        input_block_tables = torch.tensor(input_block_tables,
                                          dtype=torch.long,
                                          device=self.device)
        full_context_lens = torch.tensor(full_context_lens,
                                         dtype=torch.long,
                                         device=self.device)
        computed_context_lens = torch.tensor(computed_context_lens,
                                             dtype=torch.long,
                                             device=self.device)
        prefill_completion_state = torch.tensor(prefill_completion_state,
                                                dtype=torch.bool,
                                                device=self.device)
        # TODO: support chunked prefill for multimodal
        return (request_ids, input_tokens, input_positions, input_block_ids,
                slot_mapping, input_block_tables, full_context_lens,
                computed_context_lens, prefill_completion_state)

    def prepare_model_input(
        self,
        seq_group_metadata_list: List[SequenceGroupMetadata],
        virtual_engine: int = 0,
        finished_requests_ids: Optional[List[str]] = None
    ) -> ModelInputForNeuron:
        multi_modal_kwargs = None

        # Free slots of finished requests
        if finished_requests_ids and self.is_prefix_caching:
            for req_id in finished_requests_ids:
                if req_id in self.vllm_req_to_neuron_seq_id_mapping:
                    freed_slot = self.vllm_req_to_neuron_seq_id_mapping.pop(req_id)
                    self.free_seq_ids.add(freed_slot)

        if self.is_chunked_prefill:
            # For chunked prefill, inputs can be a mix of prefill requests and
            # decoding requests
            (request_ids, input_tokens, input_positions, input_block_ids,
             slot_mapping, input_block_tables, full_context_lens,
             computed_context_lens, prefill_completion_state) = \
                self._prepare_chunked_prefill_inputs(seq_group_metadata_list)

            seq_lens = full_context_lens
            query_lens = full_context_lens - computed_context_lens
            multi_modal_kwargs = None
        else:
            # NOTE: We assume that all sequences in the group are all prompts
            # or all decodes.
            is_prompt = seq_group_metadata_list[0].is_prompt
            prefill_completion_state = None

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

            query_lens = seq_lens

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

        lora_adapter_ids = self._get_lora_adapter_ids(seq_group_metadata_list)

        sampling_metadata = SamplingMetadata.prepare(
            seq_group_metadata_list,
            seq_lens,
            query_lens,
            self.device,
            self.pin_memory,
            generators=self.get_generators(finished_requests_ids))

        return ModelInputForNeuron(
            request_ids=request_ids,
            input_tokens=input_tokens,
            input_positions=input_positions,
            input_block_ids=input_block_ids,
            slot_mapping=slot_mapping,
            input_block_tables=input_block_tables,
            full_context_lens=full_context_lens,
            computed_context_lens=computed_context_lens,
            sampling_metadata=sampling_metadata,
            multi_modal_kwargs=multi_modal_kwargs,
            adapter_ids=lora_adapter_ids,
            prefill_completion_state=prefill_completion_state,
        )

    def remove_all_loras(self):
        raise NotImplementedError(
            "Managing LoRAs is only supported through the "
            "lora_modules parameter to the LLM Class"
            " or --lora-modules while using online server)")

    def set_active_loras(self, lora_requests: Set[LoRARequest],
                         lora_mapping: LoRAMapping) -> None:
        raise NotImplementedError(
            "Managing LoRAs is only supported through the "
            "lora_modules parameter to the LLM Class"
            " or --lora-modules while using online server)")

    def add_lora(self, lora_request: LoRARequest):
        logger.warning(
            "Adding LoRAs is only supported through the "
            "lora_modules parameter to the LLM Class"
            " or --lora-modules while using online server. If you supplied "
            "the parameter, you can ignore this warning. Ignoring"
            "lora request: ", lora_request)

    def remove_lora(self, lora_id: int) -> bool:
        raise NotImplementedError(
            "Managing LoRAs is only supported through the "
            "lora_modules parameter to the LLM Class"
            " or --lora-modules while using online server)")

    def pin_lora(self, lora_id: int) -> bool:
        raise NotImplementedError(
            "Managing LoRAs is only supported through the "
            "lora_modules parameter to the LLM Class"
            " or --lora-modules while using online server)")

    def list_loras(self) -> Set[int]:
        raise NotImplementedError(
            "Managing LoRAs is only supported through the "
            "lora_modules parameter to the LLM Class"
            " or --lora-modules while using online server)")
