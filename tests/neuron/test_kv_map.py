# SPDX-License-Identifier: Apache-2.0
"""
Tests for KV transfer logic for Neuron, particularly for supporting different
sharding on prefill and decode.
"""

import copy

from vllm.distributed.kv_transfer.neuron.nxdi_kv_map_utils import (
    KV_MAP_VERSION, HiddenTransferScheme, HiddenTransferSchemeElement,
    KVTransferScheme, KVTransferSchemeElement, extend_kv_map_cp,
    extend_kv_map_dp, generate_kv_hidden_transfer_scheme, get_kv_map_tp)


def test_kv_map_generation():
    """
    Test the KV map generation utility function
    """
    # CASE 1: PADDED

    tp_degree = 4
    num_kv_heads = 9
    num_attn_heads = 9

    kv_map_expected = {
        "heads": {
            0: [0, 1, 2],
            1: [3, 4, 5],
            2: [6, 7, 8],
            3: [-1, -1, -1]  # padding
        },
        "version": KV_MAP_VERSION
    }
    kv_map = get_kv_map_tp(tp_degree, num_kv_heads, num_attn_heads)
    assert kv_map == kv_map_expected

    kv_map_copy = copy.deepcopy(kv_map)
    world_size = 8
    cp_degree = 2
    kv_map_cp_expected = {
        "heads": {
            0: [0, 1, 2],
            1: [3, 4, 5],
            2: [6, 7, 8],
            3: [-1, -1, -1],  # padding
            4: [0, 1, 2],
            5: [3, 4, 5],
            6: [6, 7, 8],
            7: [-1, -1, -1]  # padding
        },
        "version": KV_MAP_VERSION
    }
    kv_map_cp = extend_kv_map_cp(kv_map_copy, world_size, cp_degree)
    assert kv_map_cp == kv_map_cp_expected

    kv_map_copy = copy.deepcopy(kv_map)
    world_size = 8
    dp_degree = 2
    total_batch_size = 4
    kv_map_dp_expected = {
        "heads": {
            0: [0, 1, 2],
            1: [3, 4, 5],
            2: [6, 7, 8],
            3: [-1, -1, -1],  # padding
            4: [0, 1, 2],
            5: [3, 4, 5],
            6: [6, 7, 8],
            7: [-1, -1, -1]  # padding
        },
        "seq_ids": {
            0: [0, 1],
            1: [0, 1],
            2: [0, 1],
            3: [0, 1],
            4: [2, 3],
            5: [2, 3],
            6: [2, 3],
            7: [2, 3],
        },
        "version": KV_MAP_VERSION
    }
    kv_map_dp = extend_kv_map_dp(kv_map_copy, world_size, dp_degree,
                                 total_batch_size)
    assert kv_map_dp == kv_map_dp_expected

    # CASE 2: GQA REPLICATION
    tp_degree = 4
    num_kv_heads = 2
    num_attn_heads = 4

    kv_map_expected = {
        "heads": {
            0: [0],
            1: [0],
            2: [1],
            3: [1]
        },
        "version": KV_MAP_VERSION
    }
    kv_map = get_kv_map_tp(tp_degree, num_kv_heads, num_attn_heads)

    assert kv_map == kv_map_expected

    kv_map_copy = copy.deepcopy(kv_map)
    world_size = 8
    cp_degree = 2
    kv_map_cp_expected = {
        "heads": {
            0: [0],
            1: [0],
            2: [1],
            3: [1],
            4: [0],
            5: [0],
            6: [1],
            7: [1]
        },
        "version": KV_MAP_VERSION
    }
    kv_map_cp = extend_kv_map_cp(kv_map_copy, world_size, cp_degree)
    assert kv_map_cp == kv_map_cp_expected

    kv_map_copy = copy.deepcopy(kv_map)
    world_size = 8
    dp_degree = 2
    total_batch_size = 4
    kv_map_dp_expected = {
        "heads": {
            0: [0],
            1: [0],
            2: [1],
            3: [1],
            4: [0],
            5: [0],
            6: [1],
            7: [1]
        },
        "seq_ids": {
            0: [0, 1],
            1: [0, 1],
            2: [0, 1],
            3: [0, 1],
            4: [2, 3],
            5: [2, 3],
            6: [2, 3],
            7: [2, 3],
        },
        "version": KV_MAP_VERSION
    }
    kv_map_dp = extend_kv_map_dp(kv_map_copy, world_size, dp_degree,
                                 total_batch_size)
    assert kv_map_dp == kv_map_dp_expected


def test_kv_transfer_scheme_generation():
    # Case I: Padding
    prefill_kv_map = {"heads": {0: [0, 1], 1: [2, 3], 2: [-1, -1]}}
    decode_kv_map = {"heads": {0: [0, 1, 2], 1: [3, -1, -1]}}
    batch_size = 1
    kv_transfer_scheme_expected = KVTransferScheme(
        elements={
            0: [
                KVTransferSchemeElement(0, 0, 0, 0, 0, 0),  # head 0
                KVTransferSchemeElement(0, 0, 1, 0, 0, 1),  # head 1
                KVTransferSchemeElement(1, 0, 0, 0, 0, 2),  # head 2
                KVTransferSchemeElement(1, 0, 1, 1, 0, 0),  # head 3
            ],
        },
        num_ranks_sender=3,
        num_ranks_receiver=2,
        num_heads_per_rank_sender=2,
        num_heads_per_rank_receiver=3,
    )
    hidden_transfer_scheme_expected = HiddenTransferScheme(
        elements={
            0: [
                HiddenTransferSchemeElement(
                    0,
                    0,
                    0,
                    0,
                ),  # 0 -> 0
                HiddenTransferSchemeElement(
                    1,
                    0,
                    1,
                    0,
                ),  # 1 -> 1
            ],
        })
    kv_transfer_scheme, hidden_transfer_scheme = \
        generate_kv_hidden_transfer_scheme(
        prefill_kv_map, decode_kv_map, batch_size)
    print(f"{hidden_transfer_scheme=}")
    assert kv_transfer_scheme_expected == kv_transfer_scheme
    assert hidden_transfer_scheme_expected == hidden_transfer_scheme

    # Case IIA: Replication
    prefill_kv_map = {
        "heads": {
            0: [0],
            1: [0],
            2: [0],
            3: [0],
            4: [1],
            5: [1]
        }
    }
    decode_kv_map = {
        "heads": {
            0: [0, 1],
            1: [0, 1],
        }
    }
    batch_size = 1
    kv_transfer_scheme_expected = KVTransferScheme(
        elements={
            0: [
                KVTransferSchemeElement(0, 0, 0, 0, 0,
                                        0),  # recv rank 0, head 0
                KVTransferSchemeElement(2, 0, 0, 1, 0,
                                        0),  # recv rank 1, head 0
                KVTransferSchemeElement(4, 0, 0, 0, 0,
                                        1),  # recv rank 0, head 1
                KVTransferSchemeElement(5, 0, 0, 1, 0,
                                        1),  # recv rank 1, head 1
            ],
        },
        num_ranks_sender=6,
        num_ranks_receiver=2,
        num_heads_per_rank_sender=1,
        num_heads_per_rank_receiver=2,
    )
    hidden_transfer_scheme_expected = HiddenTransferScheme(
        elements={
            0: [
                HiddenTransferSchemeElement(
                    0,
                    0,
                    0,
                    0,
                ),  # rank 0 -> 0
                HiddenTransferSchemeElement(
                    3,
                    0,
                    1,
                    0,
                ),  # rank 3 -> 1
            ],
        }, )
    kv_transfer_scheme, hidden_transfer_scheme = \
        generate_kv_hidden_transfer_scheme(
        prefill_kv_map, decode_kv_map, batch_size)
    assert kv_transfer_scheme_expected == kv_transfer_scheme
    assert hidden_transfer_scheme_expected == hidden_transfer_scheme

    # Case IIB: Replication
    prefill_kv_map = {"heads": {0: [0], 1: [0], 2: [1], 3: [1]}}
    decode_kv_map = {
        "heads": {
            0: [0],
            1: [1],
        }
    }
    batch_size = 2
    kv_transfer_scheme_expected = KVTransferScheme(
        elements={
            0: [
                KVTransferSchemeElement(0, 0, 0, 0, 0, 0),  # head 0
                KVTransferSchemeElement(2, 0, 0, 1, 0, 0),  # head 1
            ],
            1: [
                KVTransferSchemeElement(0, 1, 0, 0, 1, 0),  # head 0
                KVTransferSchemeElement(2, 1, 0, 1, 1, 0),  # head 1
            ]
        },
        num_ranks_sender=4,
        num_ranks_receiver=2,
        num_heads_per_rank_sender=1,
        num_heads_per_rank_receiver=1,
    )
    hidden_transfer_scheme_expected = HiddenTransferScheme(
        elements={
            0: [
                HiddenTransferSchemeElement(
                    0,
                    0,
                    0,
                    0,
                ),  # seq_id 0, rank 0 -> 0
                HiddenTransferSchemeElement(
                    2,
                    0,
                    1,
                    0,
                ),  # seq_id 0, rank 2 -> 1
            ],
            1: [
                HiddenTransferSchemeElement(
                    0,
                    1,
                    0,
                    1,
                ),  # seq_id 1, rank 0 -> 0
                HiddenTransferSchemeElement(
                    2,
                    1,
                    1,
                    1,
                ),  # seq_id 1, rank 2 -> 1
            ],
        }, )
    kv_transfer_scheme, hidden_transfer_scheme = \
        generate_kv_hidden_transfer_scheme(
        prefill_kv_map, decode_kv_map, batch_size)
    assert kv_transfer_scheme_expected == kv_transfer_scheme
    assert hidden_transfer_scheme_expected == hidden_transfer_scheme

    # CP4TP2 to DP2TP1 with batch 2 with 2 kv heads
    prefill_kv_map = {
        "heads": {
            0: [0],
            1: [0],
            2: [1],
            3: [1],
            4: [0],
            5: [0],
            6: [1],
            7: [1],
        },
        "seq_ids": {
            0: [0, 1],
            1: [0, 1],
            2: [0, 1],
            3: [0, 1],
            4: [0, 1],
            5: [0, 1],
            6: [0, 1],
            7: [0, 1],
        }
    }
    decode_kv_map = {
        "heads": {
            0: [0, 1],
            1: [0, 1],
        },
        "seq_ids": {
            0: [0],
            1: [1],
        }
    }
    batch_size = 2
    kv_transfer_scheme_expected = KVTransferScheme(
        elements={
            0: [
                KVTransferSchemeElement(0, 0, 0, 0, 0, 0),  # head 0
                KVTransferSchemeElement(2, 0, 0, 0, 0, 1),  # head 1
            ],
            1: [
                KVTransferSchemeElement(0, 1, 0, 1, 0, 0),  # head 0
                KVTransferSchemeElement(2, 1, 0, 1, 0, 1),  # head 1
            ]
        },
        num_ranks_sender=8,
        num_ranks_receiver=2,
        num_heads_per_rank_sender=1,
        num_heads_per_rank_receiver=2,
    )
    hidden_transfer_scheme_expected = HiddenTransferScheme(
        elements={
            0: [
                HiddenTransferSchemeElement(
                    0,
                    0,
                    0,
                    0,
                ),  # seq_id 0, rank 0 -> 0
            ],
            1: [
                HiddenTransferSchemeElement(
                    0,
                    1,
                    1,
                    0,
                ),  # seq_id 1, rank 0 -> 1
            ],
        }, )
    kv_transfer_scheme, hidden_transfer_scheme = \
        generate_kv_hidden_transfer_scheme(
        prefill_kv_map, decode_kv_map, batch_size)
    print(f"{kv_transfer_scheme=}")
    print(f"{hidden_transfer_scheme=}")
    assert kv_transfer_scheme_expected == kv_transfer_scheme
    assert hidden_transfer_scheme_expected == hidden_transfer_scheme
