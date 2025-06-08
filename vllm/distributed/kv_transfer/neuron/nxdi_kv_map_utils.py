# SPDX-License-Identifier: Apache-2.0
"""

This file includes:
1. Functions to generate the KV transfer scheme based on sharding used on
prefill and decode.

2. Utility functions to generate KV map based on tp degree and number of heads. 
Also provides functionality to extend KV map for CP or DP cases. The KV map is 
passed to Vllm KV Transfer config and is required for supporting different
sharding on prefill vs. decode nodes. This is temporary utility until long-term 
solution of exporting KV map from NxDI is implemented.
"""

import json
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
from neuronx_distributed_inference.modules.attention.gqa import (
    GQA, determine_sharding_strategy, get_shardable_head_counts, replicate_kv,
    should_pad_scale)
from torch.nn import functional as F

from vllm.logger import init_logger

logger = init_logger(__name__)

KV_MAP_VERSION = "0.1"


@dataclass
class KVTransferSchemeElement:
    """
    Single element of KV transfer scheme. Defines transfer of a KV head for a
    seq_id from send_rank to recv_rank, including information about position of
    the head and the seq_id among all heads/seq_ids within the rank.
    """
    send_rank: int
    send_seq_id_pos_in_rank: int
    send_head_pos_in_rank: int
    recv_rank: int
    recv_seq_id_pos_in_rank: int
    recv_head_pos_in_rank: int


@dataclass
class KVTransferScheme:
    """
    Scheme for KV transfer. For each seq_id includes a list of 
    KVTransferSchemeElement's. In addition includes some metadata such as
    the number of ranks and the number of heads per rank to enable computation
    of the length and offset of KV transfers.
    """
    # elements: map seq_id to list of transfers for that seq_id
    elements: Dict[int, List[KVTransferSchemeElement]]
    num_ranks_sender: int
    num_ranks_receiver: int
    num_heads_per_rank_sender: int
    num_heads_per_rank_receiver: int


@dataclass
class HiddenTransferSchemeElement:
    """
    Single element of Eagle hidden state transfer scheme. Defines transfer of
    seq_id from send_rank to recv_rank, including information about position of
    the seq_id among all heads/seq_ids within the rank.
    """
    send_rank: int
    send_seq_id_pos_in_rank: int
    recv_rank: int
    recv_seq_id_pos_in_rank: int


@dataclass
class HiddenTransferScheme:
    """
    Scheme for hidden transfer. For each seq_id includes a list of 
    HiddenTransferSchemeElement's. Note that hidden is replicated on the cores 
    so the transfer is easier to work through compared to KV.
    """
    # elements: map seq_id to list of transfers for that seq_id
    elements: Dict[int, List[HiddenTransferSchemeElement]]


# Note that kv_caches obtained from nxdi model state is a flattened list of
# [
#   # device 0
#   device_0_layer_0_k, device_0_layer_0_v,
#   device_0_layer_1_k, device_0_layer_1_v,
#   ....
#   device_0_layer_0_draft_k, device_0_layer_0_draft_v, # if (EAGLE) speculation
#   device_0_layer_1_draft_k, device_0_layer_1_draft_v, # if speculation
#   ...
#   device_0_hidden_states # if EAGLE speculation
#   # device 1
#   device_1_layer_0_k, device_1_layer_0_v,
#   ......
# ]

# in the `get_kv_caches`` call this is transformed by interleaving the devices
# to better utilize network throughput by distributing payloads across cores so
# it becomes like
# [
#  device_0_layer_0_k, device_1_layer_0_k, ...,
#  device_0_layer_0_v, device_1_layer_0_v, ...,
#  device_0_layer_1_k, device_1_layer_1_k, ...,
#  ...,
#  device_0_hidden_states, device_1_hidden_states, .. # if EAGLE speculation
# ]


def validate_and_load_kv_map(kv_map_path: str) -> Dict:
    """
    Perform validation of KV map (used to support different sharding on 
    prefill/decode):

    1. Check version of KV map 
    2. Return the KV map after changing key type to int (json keys are str)
    """

    with open(kv_map_path) as f:
        kv_map = json.load(f)
    assert kv_map["version"] == "0.1",\
        f"Unidentified KV map version found in {kv_map_path}"

    # Map has heads information (mandatory) and seq_ids information (optional)
    # (seq_ids needed in DP cases)
    kv_map["heads"] = {int(k): v for k, v in kv_map["heads"].items()}

    if "seq_ids" in kv_map:
        kv_map["seq_ids"] = {int(k): v for k, v in kv_map["seq_ids"].items()}

    return kv_map


def invert_kv_hidden_map(kv_map_head: Dict,
                         kv_map_seq_ids: Dict) -> Tuple[Dict, Dict]:
    """
    Invert KV map originally from rank -> heads and seq_ids per rank 
    to a mapping from seq_ids -> {head -> list of ranks containing head and 
    seq_ids}. Also returns a second dict for hidden state mapping seq_ids ->
    list of ranks containing that seq_id.

    Args:
        kv_map_head:    rank -> list(kv heads)
        kv_map_seq_ids: rank -> list(seq_ids)

    Returns:
        Map from seq_id -> 
            {kv_head -> list((rank, seq_id_pos_in_rank, head_pos_in_rank))}
        Map from seq_id -> 
            list((rank, seq_id_pos_in_rank))}        
    """
    inverted_kv_map: Dict = {}
    inverted_hidden_map: Dict = {}
    assert kv_map_head.keys() == kv_map_seq_ids.keys(
    ), "head and seq_ids maps should have same set of ranks"
    for rank in kv_map_head:
        for p_s, seq_id in enumerate(kv_map_seq_ids[rank]):
            for p_h, kv_head in enumerate(kv_map_head[rank]):
                if kv_head == -1:
                    continue

                if seq_id in inverted_kv_map:
                    if kv_head in inverted_kv_map[seq_id]:
                        inverted_kv_map[seq_id][kv_head].append(
                            (rank, p_s, p_h))
                    else:
                        inverted_kv_map[seq_id][kv_head] = [(rank, p_s, p_h)]
                else:
                    inverted_kv_map[seq_id] = {kv_head: [(rank, p_s, p_h)]}
            # hidden map
            if seq_id in inverted_hidden_map:
                inverted_hidden_map[seq_id].append((rank, p_s))
            else:
                inverted_hidden_map[seq_id] = [(rank, p_s)]

    return inverted_kv_map, inverted_hidden_map


def generate_kv_hidden_transfer_scheme(
        sender_kv_map: Dict, receiver_kv_map: Dict,
        max_num_seqs: int) -> Tuple[KVTransferScheme, HiddenTransferScheme]:
    """
    Generate KV map and hidden transfer scheme based on sender and receiver kv 
    maps.
    Used when prefill and decode use different sharding strategy.
    """

    # First initialize seq_ids map if not provided
    # (typically not provided except for DP cases)
    if "seq_ids" not in sender_kv_map:
        sender_tp_degree = len(sender_kv_map["heads"])
        sender_kv_map["seq_ids"] = {
            i: list(range(max_num_seqs))
            for i in range(sender_tp_degree)
        }

    if "seq_ids" not in receiver_kv_map:
        receiver_tp_degree = len(receiver_kv_map["heads"])
        receiver_kv_map["seq_ids"] = {
            i: list(range(max_num_seqs))
            for i in range(receiver_tp_degree)
        }

    assert sender_kv_map["seq_ids"] is not None
    assert receiver_kv_map["seq_ids"] is not None
    # first translate to inverted version where kv_head maps to
    # list((rank, pos in rank)) (per seq_id)
    sender_kv_map_inverted, sender_hidden_map_inverted = invert_kv_hidden_map(
        sender_kv_map["heads"], sender_kv_map["seq_ids"])
    receiver_kv_map_inverted, receiver_hidden_map_inverted = \
        invert_kv_hidden_map(
        receiver_kv_map["heads"], receiver_kv_map["seq_ids"])

    # now go through each required kv on receiver end, and create a list of
    # KVTransferSchemeElement's

    # Use round robin selection for sender rank selection to enable balanced
    # transfer traffic. We can consider other more complex schemes that take
    # into account the EFA BW per chip into account
    kv_transfer_scheme_elements: Dict[int, List[KVTransferSchemeElement]] = {}
    hidden_transfer_scheme_elements: Dict[
        int, List[HiddenTransferSchemeElement]] = {}
    for seq_id in range(max_num_seqs):
        kv_transfer_scheme_elements[seq_id] = []
        # For each head on receiver side, assign to corresponding rank etc. on
        # sender
        for kv_head in receiver_kv_map_inverted[seq_id]:
            receiver_ranks = receiver_kv_map_inverted[seq_id][kv_head]
            sender_candidate_ranks = sender_kv_map_inverted[seq_id][kv_head]
            L_r = len(receiver_ranks)
            L_s = len(sender_candidate_ranks)
            # if receiver has fewer ranks than sender with this seq_id/head then
            # we choose  the sender ranks more uniformly distributed in the full
            # range instead of sequentially. E.g., instead of sending from rank
            # 0,1,2,3 we prefer to send from 0,4,8,16). This can help since
            # nearby ranks share EFA bandwidth.
            step = L_s // L_r if L_r < L_s else 1
            for i, recv_rank_info in enumerate(receiver_ranks):
                kv_transfer_scheme_elements[seq_id].append(
                    KVTransferSchemeElement(
                        *sender_candidate_ranks[(i * step) % L_s],
                        *recv_rank_info))

        # also populate hidden transfer scheme (this is simpler since no head
        # just seq_id needed for hidden which is simply replicated and sharded)
        hidden_transfer_scheme_elements[seq_id] = []
        receiver_ranks = receiver_hidden_map_inverted[seq_id]
        sender_candidate_ranks = sender_hidden_map_inverted[seq_id]
        L_r = len(receiver_ranks)
        L_s = len(sender_candidate_ranks)
        # if receiver has fewer ranks than sender for this seq_id then we choose
        # the sender ranks more uniformly distributed in the full range instead
        # of sequentially. E.g., instead of sending from rank 0,1,2,3 we prefer
        # to send from 0,4,8,16). This can help since nearby ranks share EFA
        # bandwidth.
        step = L_s // L_r if L_r < L_s else 1
        for i, recv_rank_info in enumerate(receiver_ranks):
            hidden_transfer_scheme_elements[seq_id].append(
                HiddenTransferSchemeElement(
                    *sender_candidate_ranks[(i * step) % L_s],
                    *recv_rank_info))

    def check_all_elem_lens_equal(L):
        return all(len(L[i]) == len(L[0]) for i in range(len(L)))

    assert check_all_elem_lens_equal(
        sender_kv_map["heads"]
    ), "Expect all sender ranks to have same number of heads"
    num_heads_per_rank_sender = len(sender_kv_map["heads"][0])

    assert check_all_elem_lens_equal(
        receiver_kv_map["heads"]
    ), "Expect all receiver ranks to have same number of heads"
    num_heads_per_rank_receiver = len(receiver_kv_map["heads"][0])

    num_ranks_sender = len(sender_kv_map["heads"])
    num_ranks_receiver = len(receiver_kv_map["heads"])

    kv_transfer_scheme = KVTransferScheme(kv_transfer_scheme_elements,
                                          num_ranks_sender, num_ranks_receiver,
                                          num_heads_per_rank_sender,
                                          num_heads_per_rank_receiver)
    hidden_transfer_scheme = HiddenTransferScheme(
        hidden_transfer_scheme_elements)
    return kv_transfer_scheme, hidden_transfer_scheme


def generate_kv_transfer_sequences_identical_sharding(kv_caches, max_num_seqs):
    """
    transfer scheme that support only same sharding
    and contiguous KV cache layout, block_ids in this
    case is a 0-dim tensor as seq_id

    Returns: transfer_sequences which is a mapping from seq_id to 
    (tensors, offsets, lengths, peer_devices)
    """

    # In this case we can use same logic to transform both KV cache and
    # hidden_states for EAGLE
    transfer_sequences: Dict[int, Tuple[List[torch.Tensor], List[int],
                                        List[int], List[int]]] = {}

    for seq_id in range(max_num_seqs):
        tensors = []
        peer_devices = []
        lengths = []
        offsets = []
        for tensor in kv_caches:
            tensors.append(tensor)
            peer_devices.append(tensor.device.index)
            length = math.prod(list(tensor.shape[1:])) * tensor.element_size()
            offset = length * seq_id
            lengths.append(length)
            offsets.append(offset)

        transfer_sequences[seq_id] = (tensors, offsets, lengths, peer_devices)

    return transfer_sequences


def generate_kv_transfer_sequences_different_sharding(kv_caches, max_num_seqs,
                                                      producer_kv_map,
                                                      consumer_kv_map,
                                                      is_producer):
    """
    transfer scheme that support different sharding on prefill/decode
    and contiguous KV cache layout, block_ids in this
    case is a 0-dim tensor as seq_id

    Returns: transfer_sequences which is a mapping from seq_id to 
    (tensors, offsets, lengths, peer_devices)
    """

    transfer_sequences: Dict[int, Tuple[List[torch.Tensor], List[int],
                                        List[int], List[int]]] = {}

    assert producer_kv_map is not None
    assert consumer_kv_map is not None
    kv_transfer_scheme, hidden_transfer_scheme = \
        generate_kv_hidden_transfer_scheme(
        producer_kv_map, consumer_kv_map, max_num_seqs)

    logger.info("kv_transfer_scheme: %s", kv_transfer_scheme)
    logger.info("hidden_transfer_scheme: %s", hidden_transfer_scheme)

    for seq_id in range(max_num_seqs):
        transformed_kv_caches = []
        original_kv_length = [
            math.prod(cache.shape[1:]) * cache.element_size()
            for cache in kv_caches
        ]
        kv_length = []
        kv_offset = []
        peer_devices = []
        num_ranks = kv_transfer_scheme.num_ranks_sender if is_producer \
                    else kv_transfer_scheme.num_ranks_receiver
        num_layers = len(kv_caches) // (2 * num_ranks)  # 2 for K+V
        num_heads_per_rank = kv_transfer_scheme.num_heads_per_rank_sender \
            if is_producer \
            else kv_transfer_scheme.num_heads_per_rank_receiver
        for layer_num in range(num_layers):
            for i in range(2):  # for K/V
                # for each layer and K/V iterate over the transfer scheme
                for t in kv_transfer_scheme.elements[seq_id]:
                    self_rank = t.send_rank if is_producer else t.recv_rank
                    peer_rank = t.recv_rank if is_producer else t.send_rank
                    self_head_pos_in_rank = t.send_head_pos_in_rank \
                        if is_producer else t.recv_head_pos_in_rank
                    self_seq_id_pos_in_rank = t.send_seq_id_pos_in_rank \
                        if is_producer else t.recv_seq_id_pos_in_rank
                    # TODO: this assumes knowledge of how the kv caches are
                    # ordered in the state - can we avoid assuming this?
                    # TODO: we also assume known layout of KV cache for
                    # computing offset.
                    # TODO: once we support blockwise KV cache, we will need to
                    # track head_dim and update logic accordingly
                    idx = 2 * num_ranks * layer_num + num_ranks * i \
                            + self_rank
                    transformed_kv_caches.append(kv_caches[idx])
                    # we always send one head per transfer, we offset
                    # according to seq_id and head positions within rank
                    len_to_transfer = original_kv_length[
                        idx] // num_heads_per_rank
                    offset_to_transfer = original_kv_length[idx] \
                                * self_seq_id_pos_in_rank \
                                + self_head_pos_in_rank * len_to_transfer
                    kv_length.append(len_to_transfer)
                    kv_offset.append(offset_to_transfer)
                    peer_devices.append(peer_rank)

        # check if we have hidden remaining to send - note that the KV caches
        # are 2 (for KV) * num_layers * num_ranks and if we have hidden that is
        # just num_ranks (see the detailed layout of the kv_caches list above)
        if len(kv_caches) > num_layers * 2 * num_ranks:
            assert len(kv_caches) - num_layers * 2 * num_ranks == num_ranks
            for h_t in hidden_transfer_scheme.elements[seq_id]:
                self_rank = h_t.send_rank if is_producer else h_t.recv_rank
                peer_rank = h_t.recv_rank if is_producer else h_t.send_rank
                self_seq_id_pos_in_rank = h_t.send_seq_id_pos_in_rank \
                    if is_producer else h_t.recv_seq_id_pos_in_rank
                idx = num_layers * 2 * num_ranks + self_rank
                transformed_kv_caches.append(kv_caches[idx])
                # we offset according to seq_id position within rank
                len_to_transfer = original_kv_length[idx]
                offset_to_transfer = original_kv_length[idx] \
                            * self_seq_id_pos_in_rank
                kv_length.append(len_to_transfer)
                kv_offset.append(offset_to_transfer)
                peer_devices.append(peer_rank)

        transfer_sequences[seq_id] = (transformed_kv_caches, kv_offset,
                                      kv_length, peer_devices)

    return transfer_sequences


def generate_kv_transfer_sequences_identical_sharding_block_kv(
        kv_caches, block_ids):
    """
    transfer scheme that support only same sharding
    and blockwise KV cache layout
    
    Returns: transfer_sequences which is a mapping from seq_id to 
    (tensors, offsets, lengths, peer_devices)
    """
    # TODO we can define a new data class for transfer sequences
    tensors = []
    peer_devices = []
    lengths = []
    offsets = []

    # One sequence can occupy multiple kv cache blocks, which means each kv
    # cache tensor may be accessed multiple times in different offsets for
    # transferring the KV cache of the sequence.
    # block_ids needs to be the outer loop to interleave the transfers
    # across neuron devices, otherwise the transfer may hit the maximum
    # number of pending requests limit, which is 128.
    for block_id in block_ids:
        for tensor in kv_caches:
            length = math.prod(list(tensor.shape[1:])) * tensor.element_size()
            tensors.append(tensor)
            peer_devices.append(tensor.device.index)
            offset = length * block_id
            lengths.append(length)
            offsets.append(offset)

    return tensors, offsets, lengths, peer_devices


def generate_kv_transfer_sequences_different_sharding_block_kv(
        kv_caches, max_num_seqs):
    """
    transfer scheme that support different sharding
    and blockwise KV cache layout
    """
    raise NotImplementedError(
        "Blockwise KV cache for different sharding is not supported")


def setup_transfer_scheme(kv_caches: List, producer_kv_map: Optional[Dict],
                          consumer_kv_map: Optional[Dict], max_num_seqs: int,
                          is_producer: bool):
    """
    Set up KV transfer scheme based on KV maps on prefill and decode.
    This is used to support different sharding strategies.
    
    returns:
        transfer_sequences: mapping from seq_id to the transfer sequence which 
        is a list of (tensors, offsets, lengths, peer_devices).

    Currently we generate a mapping from seq_id to the transfer sequence which 
    is a list of (tensors, offsets, lengths, peer_devices). 
    
    TODO: For blockwise cache consider generating only single layer here and
    later expand to full layers. This can help reduce the size of this map.
    """

    # Default case - when prefill/decode have same KV map and also
    # we don't have DP (we assume presence of seq_ids means we have DP). For
    # DP we do not wish to transfer KV cache from all ranks.
    is_default_case = True
    if producer_kv_map != consumer_kv_map:
        is_default_case = False
    if producer_kv_map is not None and "seq_ids" in producer_kv_map:
        is_default_case = False
    if consumer_kv_map is not None and "seq_ids" in consumer_kv_map:
        is_default_case = False

    if is_default_case:
        return generate_kv_transfer_sequences_identical_sharding(
            kv_caches, max_num_seqs)
    else:
        return generate_kv_transfer_sequences_different_sharding(
            kv_caches, max_num_seqs, producer_kv_map, consumer_kv_map,
            is_producer)


# ----------------------------------
# Utility Functions to generate KV map for TP/CP/DP cases based on the
# sharding/replication/padding strategy used in NxDI


def maybe_pad_tail(tensor,
                   source_heads: int,
                   target_heads: int,
                   pad_dim: int,
                   tensor_scale=None,
                   pad_value=0):
    """
    Copied from NxDI with additional `pad_value` argument.
    """
    tensor = _maybe_pad_tail(tensor, source_heads, target_heads, pad_dim,
                             pad_value)
    if should_pad_scale(tensor_scale=tensor_scale, pad_dim=pad_dim):
        tensor_scale = _maybe_pad_tail(tensor_scale, source_heads,
                                       target_heads, pad_dim, pad_value)
    return tensor, tensor_scale


def _maybe_pad_tail(tensor,
                    source_heads: int,
                    target_heads: int,
                    pad_dim: int,
                    pad_value=0):
    """
    Copied from NxDI with additional `pad_value` argument.
    """
    if tensor is None:
        return tensor
    size_to_pad = int((tensor.shape[pad_dim] // source_heads) * target_heads -
                      tensor.shape[pad_dim])

    dims_after_pad_dim = len(tensor.size()) - pad_dim
    pad_length = dims_after_pad_dim * 2
    pad = (0, ) * (pad_length - 1) + (size_to_pad, )

    return F.pad(tensor, pad, value=pad_value)


def get_kv_map_tp(tp_degree, num_key_value_heads, num_attention_heads):
    """
    Generate KV map for TP sharded case based on number of KV and attn heads.
    """

    _src_num_key_value_heads = num_key_value_heads
    _src_num_attention_heads = num_attention_heads
    sharding_strategy = determine_sharding_strategy(
        tp_degree,
        _src_num_key_value_heads,
        desired_sharding_strategy=None,
    )
    num_attention_heads, num_key_value_heads = get_shardable_head_counts(
        tp_degree,
        _src_num_attention_heads,
        _src_num_key_value_heads,
        sharding_strategy,
    )
    kv_map = torch.arange(_src_num_key_value_heads, dtype=torch.int32)

    if num_key_value_heads != _src_num_key_value_heads:
        if sharding_strategy == GQA.REPLICATE_TO_TP_DEGREE:
            repeats = tp_degree // _src_num_key_value_heads
        elif sharding_strategy == GQA.CONVERT_TO_MHA:
            repeats = _src_num_attention_heads // _src_num_key_value_heads

        kv_map, _ = replicate_kv(kv_map,
                                 source_heads=_src_num_key_value_heads,
                                 repeats=repeats,
                                 head_dim=0)

    if sharding_strategy == GQA.CONVERT_TO_MHA:
        kv_map, _ = maybe_pad_tail(kv_map,
                                   source_heads=_src_num_key_value_heads,
                                   target_heads=num_key_value_heads,
                                   pad_dim=0,
                                   pad_value=-1)

    num_kv_per_rank = num_key_value_heads // tp_degree

    kv_map = {
        i: kv_map[i * num_kv_per_rank:(i + 1) * num_kv_per_rank].tolist()
        for i in range(tp_degree)
    }
    kv_map = {"heads": kv_map, "version": KV_MAP_VERSION}
    return kv_map


def extend_kv_map_cp(kv_map, world_size, cp_degree):
    """
    Extend KV map with TP into CP+TP case by replicating.
    """
    tp_degree = len(kv_map["heads"])
    assert cp_degree * tp_degree == world_size
    for cp_rank in range(1, cp_degree):
        for tp_rank in range(tp_degree):
            kv_map["heads"][cp_rank * tp_degree +
                            tp_rank] = kv_map["heads"][tp_rank]

    return kv_map


def extend_kv_map_dp(kv_map, world_size, dp_degree, total_batch_size):
    """
    Extend KV map with TP into DP+TP case by replicating and adding the seq_ids.
    """
    tp_degree = len(kv_map["heads"])
    assert dp_degree * tp_degree == world_size
    assert total_batch_size % dp_degree == 0
    batch_per_dp_rank = total_batch_size // dp_degree

    # First call extend_kv_map_cp to update the kv_map["heads"]
    kv_map = extend_kv_map_cp(kv_map, world_size, dp_degree)

    # Now add the seq_ids
    kv_map["seq_ids"] = {}
    for dp_rank in range(dp_degree):
        for tp_rank in range(tp_degree):
            kv_map["seq_ids"][dp_rank * tp_degree + tp_rank] = list(
                range(dp_rank * batch_per_dp_rank,
                      (dp_rank + 1) * batch_per_dp_rank))

    return kv_map
