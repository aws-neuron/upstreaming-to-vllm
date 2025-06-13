# SPDX-License-Identifier: Apache-2.0
import pytest

from vllm.distributed.kv_transfer.neuron.kv_block_aggregator import (
    aggregate_kv_blocks)


def test_example_1():
    a = [3, 4, 5, 1, 6, 7, 8, 10, 9]
    b = [3, 4, 5, 6, 7, 8, 9, 11, 10]
    expected_a = [[3, 4, 5], [1], [6, 7, 8, 9, 10]]
    expected_b = [[3, 4, 5], [6], [7, 8, 9, 10, 11]]
    out_a, out_b = aggregate_kv_blocks(a, b)
    assert out_a == expected_a
    assert out_b == expected_b


def test_commutativity():
    a = [3, 4, 5, 6, 7, 8, 9, 11, 10]
    b = [3, 4, 5, 1, 6, 7, 8, 10, 9]
    expected_a = [[3, 4, 5], [6], [7, 8, 9, 10, 11]]
    expected_b = [[3, 4, 5], [1], [6, 7, 8, 9, 10]]
    out_a, out_b = aggregate_kv_blocks(a, b)
    assert out_a == expected_a
    assert out_b == expected_b


def test_example_2():
    a = [3, 4, 5, 1, 6, 7, 8, 10, 9]
    b = [3, 2, 5, 6, 7, 8, 9, 11, 10]
    expected_a = [[3], [4], [5], [1], [6, 7, 8, 9, 10]]
    expected_b = [[3], [2], [5], [6], [7, 8, 9, 10, 11]]
    out_a, out_b = aggregate_kv_blocks(a, b)
    print(out_a, out_b)
    assert out_a == expected_a
    assert out_b == expected_b


def test_length_mismatch():
    a = [1, 2, 3]
    b = [1, 2]
    with pytest.raises(ValueError,
                       match="Block id lists must be the same length"):
        aggregate_kv_blocks(a, b)


def test_duplicate_in_lst1():
    a = [1, 2, 2, 3]
    b = [4, 5, 6, 7]
    with pytest.raises(ValueError,
                       match="Input lists must not contain duplicates."):
        aggregate_kv_blocks(a, b)


def test_duplicate_in_lst2():
    a = [1, 2, 3, 4]
    b = [5, 6, 6, 7]
    with pytest.raises(ValueError,
                       match="Input lists must not contain duplicates."):
        aggregate_kv_blocks(a, b)


def test_single_element_lists():
    a = [42]
    b = [99]
    out_a, out_b = aggregate_kv_blocks(a, b)
    assert out_a == [[42]]
    assert out_b == [[99]]


def test_all_consecutive_increasing():
    a = [1, 2, 3, 4]
    b = [5, 6, 7, 8]
    out_a, out_b = aggregate_kv_blocks(a, b)
    assert out_a == [[1, 2, 3, 4]]
    assert out_b == [[5, 6, 7, 8]]


def test_all_consecutive_decreasing():
    a = [4, 3, 2, 1]
    b = [8, 7, 6, 5]
    out_a, out_b = aggregate_kv_blocks(a, b)
    assert out_a == [[1, 2, 3, 4]]
    assert out_b == [[5, 6, 7, 8]]
