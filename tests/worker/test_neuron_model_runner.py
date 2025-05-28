# SPDX-License-Identifier: Apache-2.0
import copy
import os
from unittest.mock import MagicMock

import pytest

from vllm.config import VllmConfig
from vllm.engine.arg_utils import EngineArgs
from vllm.sampling_params import SamplingParams
from vllm.sequence import SequenceData, SequenceGroupMetadata
from vllm.worker.neuron_model_runner import NeuronModelRunner
from vllm.worker.utils import NeuronFramework, use_transformers_neuronx

os.environ[
    'VLLM_NEURON_FRAMEWORK'] = NeuronFramework.TRANSFORMERS_NEURONX.value


def _create_neuron_model_runner(model: str, *args,
                                **kwargs) -> NeuronModelRunner:
    engine_args = EngineArgs(model, *args, **kwargs)
    engine_config = engine_args.create_engine_config()
    vllm_config = VllmConfig(
        model_config=engine_config.model_config,
        parallel_config=engine_config.parallel_config,
        cache_config=engine_config.cache_config,
        scheduler_config=engine_config.scheduler_config,
        device_config=engine_config.device_config,
    )
    neuron_model_runner = NeuronModelRunner(vllm_config=vllm_config)
    return neuron_model_runner


def test_update_neuron_sampling_params_not_full_batch():
    os.environ["NEURON_ON_DEVICE_SAMPLING_DISABLED"] = "0"
    model_runner = _create_neuron_model_runner(
        "facebook/opt-125m",
        seed=0,
        dtype="float16",
        max_num_seqs=2,
    )
    assert not model_runner._on_device_sampling_disabled
    # Test sampling param updating only when TNx is framework
    # NxDI handles sampling parameter updating inside model
    if use_transformers_neuronx():
        model_mock = MagicMock()
        model_runner.model = model_mock

        seq_group_metadata_list = [
            SequenceGroupMetadata(
                request_id="test_0",
                is_prompt=True,
                seq_data={0: SequenceData.from_seqs([1, 2, 3])},
                sampling_params=SamplingParams(temperature=0.5,
                                               top_k=1,
                                               top_p=0.5),
                block_tables={0: [1]},
            )
        ]

        model_runner.prepare_model_input(seq_group_metadata_list)

        # Index neuron sampling parameters based on block_tables indices.
        # The first block_id of the sequence 0 is 1, so its parameters are
        # placed at index 1. So the sampling parameters will be:
        # Index 0: default sampling parameters
        # Index 1: sequecne 0's sampling parameters.
        neuron_sampling_params = (
            model_runner.model_config.neuron_sampling_params)
        assert neuron_sampling_params.temperature == [1.0, 0.5]
        assert neuron_sampling_params.top_k == [
            model_runner._MAX_NEURON_SAMPLING_TOP_K, 1
        ]
        assert neuron_sampling_params.top_p == [1.0, 0.5]
        model_mock.model.update_generation_config.assert_called_once_with(
            neuron_sampling_params)


def test_update_neuron_sampling_params_full_batch():
    os.environ["NEURON_ON_DEVICE_SAMPLING_DISABLED"] = "0"
    model_runner = _create_neuron_model_runner(
        "facebook/opt-125m",
        seed=0,
        dtype="float16",
        max_num_seqs=2,
    )
    assert not model_runner._on_device_sampling_disabled

    # Test sampling param updating only when TNx is framework
    # NxDI handles sampling parameter updating inside model
    if use_transformers_neuronx():
        model_mock = MagicMock()
        model_runner.model = model_mock

        seq_group_metadata_list = [
            SequenceGroupMetadata(
                request_id="test_0",
                is_prompt=True,
                seq_data={0: SequenceData.from_seqs([1, 2, 3])},
                sampling_params=SamplingParams(temperature=0.5,
                                               top_k=1,
                                               top_p=0.5),
                block_tables={0: [1]},
            ),
            SequenceGroupMetadata(
                request_id="test_0",
                is_prompt=True,
                seq_data={1: SequenceData.from_seqs([4, 5, 6])},
                sampling_params=SamplingParams(temperature=0.2,
                                               top_k=2,
                                               top_p=0.2),
                block_tables={1: [0]},
            )
        ]

        model_runner.prepare_model_input(seq_group_metadata_list)

        # Index neuron sampling parameters based on block_tables indices.
        # The first block_id of the sequence 0 is 1, so its parameters are
        # placed at index 1. So the sampling parameters will be:
        # Index 0: sequence 1's sampling parameters
        # Index 1: sequecne 0's sampling parameters.
        neuron_sampling_params = (
            model_runner.model_config.neuron_sampling_params)
        assert neuron_sampling_params.temperature == [0.2, 0.5]
        assert neuron_sampling_params.top_k == [2, 1]
        assert neuron_sampling_params.top_p == [0.2, 0.5]
        model_mock.model.update_generation_config.assert_called_once_with(
            neuron_sampling_params)


def test_req_id_to_neuron_seq_id_mapping_for_prefill_without_finished():
    model_runner = _create_neuron_model_runner(
        "facebook/opt-125m",
        seed=0,
        dtype="float16",
        max_num_seqs=4,
        block_size=16,
    )

    model_runner.is_prefix_caching = True
    model_runner.use_custom_seq_id_mapping = True
    prior_free_seq_ids = copy.copy(model_runner.free_seq_ids)

    model_mock = MagicMock()
    model_runner.model = model_mock

    # 3 context encoding requests
    seq_group_metadata_list = [
        SequenceGroupMetadata(request_id="test_0",
                              is_prompt=True,
                              seq_data={0: SequenceData.from_seqs([1, 2, 3])},
                              sampling_params=SamplingParams(temperature=0.5,
                                                             top_k=1,
                                                             top_p=0.5),
                              block_tables={0: [10]},
                              computed_block_nums=[]),
        SequenceGroupMetadata(request_id="test_1",
                              is_prompt=True,
                              seq_data={1: SequenceData.from_seqs([4, 5, 6])},
                              sampling_params=SamplingParams(temperature=0.2,
                                                             top_k=2,
                                                             top_p=0.2),
                              block_tables={1: [11]},
                              computed_block_nums=[]),
        SequenceGroupMetadata(request_id="test_2",
                              is_prompt=True,
                              seq_data={2: SequenceData.from_seqs([4, 5, 6])},
                              sampling_params=SamplingParams(temperature=0.2,
                                                             top_k=2,
                                                             top_p=0.2),
                              block_tables={2: [12]},
                              computed_block_nums=[])
    ]

    model_runner.prepare_model_input(seq_group_metadata_list,
                                     finished_requests_ids=None)

    for seq_grp_metadata in seq_group_metadata_list:
        assert seq_grp_metadata.request_id in \
        model_runner.vllm_req_to_neuron_seq_id_mapping
    active_allocated_seq_ids = set(
        model_runner.vllm_req_to_neuron_seq_id_mapping[
            seq_grp_metadata.request_id]
        for seq_grp_metadata in seq_group_metadata_list)
    assert len(seq_group_metadata_list) == len(active_allocated_seq_ids)
    assert len(prior_free_seq_ids) == len(active_allocated_seq_ids) + len(
        model_runner.free_seq_ids)
    assert prior_free_seq_ids == set(
        model_runner.vllm_req_to_neuron_seq_id_mapping[
            seq_grp_metadata.request_id] for seq_grp_metadata in
        seq_group_metadata_list) | model_runner.free_seq_ids


def test_req_id_to_neuron_seq_id_mapping_for_prefill_with_finished():
    model_runner = _create_neuron_model_runner(
        "facebook/opt-125m",
        seed=0,
        dtype="float16",
        max_num_seqs=4,
        block_size=16,
    )

    model_runner.is_prefix_caching = True
    model_runner.use_custom_seq_id_mapping = True
    prior_free_seq_ids = copy.copy(model_runner.free_seq_ids)

    model_mock = MagicMock()
    model_runner.model = model_mock

    # 1 context encoding request that will be finished in subsequent request.
    seq_group_metadata_list = [
        SequenceGroupMetadata(request_id="test_0",
                              is_prompt=True,
                              seq_data={0: SequenceData.from_seqs([1, 2, 3])},
                              sampling_params=SamplingParams(temperature=0.5,
                                                             top_k=1,
                                                             top_p=0.5),
                              block_tables={0: [10]},
                              computed_block_nums=[])
    ]
    model_runner.prepare_model_input(seq_group_metadata_list,
                                     finished_requests_ids=None)
    # 3 context encoding requests with test_0 marked as finished.
    seq_group_metadata_list = [
        SequenceGroupMetadata(request_id="test_1",
                              is_prompt=True,
                              seq_data={1: SequenceData.from_seqs([4, 5, 6])},
                              sampling_params=SamplingParams(temperature=0.2,
                                                             top_k=2,
                                                             top_p=0.2),
                              block_tables={1: [11]},
                              computed_block_nums=[]),
        SequenceGroupMetadata(request_id="test_2",
                              is_prompt=True,
                              seq_data={2: SequenceData.from_seqs([7, 8, 9])},
                              sampling_params=SamplingParams(temperature=0.2,
                                                             top_k=2,
                                                             top_p=0.2),
                              block_tables={2: [12]},
                              computed_block_nums=[]),
        SequenceGroupMetadata(
            request_id="test_3",
            is_prompt=True,
            seq_data={3: SequenceData.from_seqs([10, 11, 12])},
            sampling_params=SamplingParams(temperature=0.5, top_k=1,
                                           top_p=0.5),
            block_tables={3: [13]},
            computed_block_nums=[]),
        SequenceGroupMetadata(
            request_id="test_4",
            is_prompt=True,
            seq_data={4: SequenceData.from_seqs([13, 14, 15])},
            sampling_params=SamplingParams(temperature=0.5, top_k=1,
                                           top_p=0.5),
            block_tables={4: [14]},
            computed_block_nums=[]),
    ]

    model_runner.prepare_model_input(seq_group_metadata_list,
                                     finished_requests_ids=['test_0'])

    for seq_grp_metadata in seq_group_metadata_list:
        assert seq_grp_metadata.request_id in \
            model_runner.vllm_req_to_neuron_seq_id_mapping
    active_allocated_seq_ids = set(
        model_runner.vllm_req_to_neuron_seq_id_mapping[
            seq_grp_metadata.request_id]
        for seq_grp_metadata in seq_group_metadata_list)
    assert len(seq_group_metadata_list) == len(active_allocated_seq_ids)
    assert len(prior_free_seq_ids) == len(active_allocated_seq_ids) + len(
        model_runner.free_seq_ids)
    assert prior_free_seq_ids == set(
        model_runner.vllm_req_to_neuron_seq_id_mapping[
            seq_grp_metadata.request_id] for seq_grp_metadata in
        seq_group_metadata_list) | model_runner.free_seq_ids


def test_req_id_to_neuron_seq_id_mapping_for_prefill_with_overflow():
    model_runner = _create_neuron_model_runner(
        "facebook/opt-125m",
        seed=0,
        dtype="float16",
        max_num_seqs=4,
        block_size=16,
    )

    model_runner.is_prefix_caching = True
    model_runner.use_custom_seq_id_mapping = True
    model_mock = MagicMock()
    model_runner.model = model_mock

    # 1 context encoding request that will be finished in subsequent request.
    seq_group_metadata_list = [
        SequenceGroupMetadata(request_id="test_0",
                              is_prompt=True,
                              seq_data={0: SequenceData.from_seqs([1, 2, 3])},
                              sampling_params=SamplingParams(temperature=0.5,
                                                             top_k=1,
                                                             top_p=0.5),
                              block_tables={0: [10]},
                              computed_block_nums=[])
    ]
    model_runner.prepare_model_input(seq_group_metadata_list,
                                     finished_requests_ids=None)
    # 3 context encoding requests with test_0 marked as finished.
    seq_group_metadata_list = [
        SequenceGroupMetadata(request_id="test_1",
                              is_prompt=True,
                              seq_data={1: SequenceData.from_seqs([4, 5, 6])},
                              sampling_params=SamplingParams(temperature=0.2,
                                                             top_k=2,
                                                             top_p=0.2),
                              block_tables={1: [11]},
                              computed_block_nums=[]),
        SequenceGroupMetadata(request_id="test_2",
                              is_prompt=True,
                              seq_data={2: SequenceData.from_seqs([7, 8, 9])},
                              sampling_params=SamplingParams(temperature=0.2,
                                                             top_k=2,
                                                             top_p=0.2),
                              block_tables={2: [12]},
                              computed_block_nums=[]),
        SequenceGroupMetadata(
            request_id="test_3",
            is_prompt=True,
            seq_data={3: SequenceData.from_seqs([10, 11, 12])},
            sampling_params=SamplingParams(temperature=0.5, top_k=1,
                                           top_p=0.5),
            block_tables={3: [13]},
            computed_block_nums=[]),
        SequenceGroupMetadata(
            request_id="test_4",
            is_prompt=True,
            seq_data={4: SequenceData.from_seqs([13, 14, 15])},
            sampling_params=SamplingParams(temperature=0.5, top_k=1,
                                           top_p=0.5),
            block_tables={4: [14]},
            computed_block_nums=[]),
    ]
    with pytest.raises(AssertionError):
        model_runner.prepare_model_input(seq_group_metadata_list,
                                         finished_requests_ids=None)


def test_block_table_padding_with_neuron_kernel_enabled():
    model_runner = _create_neuron_model_runner(
        "facebook/opt-125m",
        seed=0,
        dtype="float16",
        max_num_seqs=1,
        block_size=16,
    )
    model_runner.is_prefix_caching = True
    model_runner.use_custom_seq_id_mapping = True
    model_runner.vllm_req_to_neuron_seq_id_mapping = {"test_0": 0}
    model_mock = MagicMock()
    model_runner.model = model_mock
    model_runner.model.neuron_config.attn_tkg_nki_kernel_enabled = True
    model_runner.model.neuron_config.attn_block_tkg_nki_kernel_enabled = True
    seq_group_metadata_list = [
        SequenceGroupMetadata(request_id="test_0",
                              is_prompt=False,
                              seq_data={0: SequenceData.from_seqs([1, 2, 3])},
                              sampling_params=SamplingParams(temperature=0.5,
                                                             top_k=1,
                                                             top_p=0.5),
                              block_tables={0: [10]},
                              computed_block_nums=[])
    ]
    out = model_runner.prepare_model_input(seq_group_metadata_list,
                                           finished_requests_ids=None)
    assert out.input_block_tables[:, -1] == -1


def test_block_table_padding_with_neuron_kernel_disabled():
    model_runner = _create_neuron_model_runner(
        "facebook/opt-125m",
        seed=0,
        dtype="float16",
        max_num_seqs=1,
        block_size=16,
    )
    model_runner.is_prefix_caching = True
    model_runner.use_custom_seq_id_mapping = True
    model_runner.vllm_req_to_neuron_seq_id_mapping = {"test_0": 0}
    model_mock = MagicMock()
    model_runner.model = model_mock
    model_runner.model.neuron_config.attn_tkg_nki_kernel_enabled = False
    model_runner.model.neuron_config.attn_block_tkg_nki_kernel_enabled = False
    seq_group_metadata_list = [
        SequenceGroupMetadata(request_id="test_0",
                              is_prompt=False,
                              seq_data={0: SequenceData.from_seqs([1, 2, 3])},
                              sampling_params=SamplingParams(temperature=0.5,
                                                             top_k=1,
                                                             top_p=0.5),
                              block_tables={0: [10]},
                              computed_block_nums=[])
    ]
    out = model_runner.prepare_model_input(seq_group_metadata_list,
                                           finished_requests_ids=None)
    assert out.input_block_tables[:, -1] == 0


def test_block_table_padding_with_neuron_kernel_enabled_prefill():
    model_runner = _create_neuron_model_runner(
        "facebook/opt-125m",
        seed=0,
        dtype="float16",
        max_num_seqs=1,
        block_size=16,
    )
    model_runner.is_prefix_caching = True
    model_runner.use_custom_seq_id_mapping = True
    model_mock = MagicMock()
    model_runner.model = model_mock
    model_runner.model.neuron_config.attn_tkg_nki_kernel_enabled = True
    model_runner.model.neuron_config.attn_block_tkg_nki_kernel_enabled = True
    seq_group_metadata_list = [
        SequenceGroupMetadata(request_id="test_0",
                              is_prompt=True,
                              seq_data={0: SequenceData.from_seqs([1, 2, 3])},
                              sampling_params=SamplingParams(temperature=0.5,
                                                             top_k=1,
                                                             top_p=0.5),
                              block_tables={0: [10]},
                              computed_block_nums=[])
    ]
    out = model_runner.prepare_model_input(seq_group_metadata_list,
                                           finished_requests_ids=None)
    assert out.input_block_tables[:, -1] == 0
