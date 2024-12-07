import math
import random
import time

import pytest
import torch
import torch.nn.functional as F
from typing import Optional

# from transformers_neuronx import ContinuousBatchingConfig, NeuronConfig, Layout
# from transformers_neuronx import hlo
# from transformers_neuronx.layers import attention
# from transformers_neuronx.layers import attention_utils
# from transformers_neuronx_test.unit.validation import validate, verify


class BlockDiagonalCausalFromBottomRightMask:
    @staticmethod
    def from_seqlens(query_lens, seq_lens):
        from torch import logical_and, logical_or, logical_not

        n_queries = sum(query_lens)
        n_keys = sum(seq_lens)
        prior_mask = torch.zeros(n_queries, n_keys)
        num_seqs = len(query_lens)

        a = torch.arange(n_queries).reshape(n_queries, 1).expand(n_queries, n_keys)
        b = torch.arange(n_keys).reshape(1, n_keys).expand(n_queries, n_keys)
        q_cumsum = torch.tensor([0]+query_lens).cumsum(dim=0)
        k_cumsum = torch.tensor([0]+seq_lens).cumsum(dim=0)

        for seq_id in range(num_seqs):
            ri = q_cumsum[seq_id]
            ci = k_cumsum[seq_id]
            nr = query_lens[seq_id]
            nc = seq_lens[seq_id]

            a_offset = ci + nc - ri - nr
            new_mask = (a + a_offset) >= b

            left_mask = b >= ci
            top_mask = a >= ri
            bottom_mask = a < (ri+nr)

            new_mask = logical_and(logical_and(logical_and(new_mask, left_mask), top_mask), bottom_mask)
            prior_mask = logical_or(prior_mask, new_mask)

        # convert binary mask to -inf values
        prior_mask = logical_not(prior_mask)
        prior_mask = prior_mask.float()*-30000
        return prior_mask


def ref_masked_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    scale: float,
    attn_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    attn_weights = scale * torch.einsum("qhd,khd->hqk", query, key).float()
    if attn_mask is not None:
        attn_weights = attn_weights + attn_mask.float()
    attn_weights = torch.softmax(attn_weights, dim=-1).to(value.dtype)
    out = torch.einsum("hqk,khd->qhd", attn_weights, value)
    return out


def ref_context_attention(query, key, value, query_lens, seq_lens, head_size, num_kv_heads, num_heads, num_queries_per_kv):
    scale = float(1.0 / (head_size**0.5))
    if num_queries_per_kv > 1:
        # Handle MQA and GQA
        key = torch.repeat_interleave(key, num_queries_per_kv, dim=1)
        value = torch.repeat_interleave(value, num_queries_per_kv, dim=1)

    attn_mask = BlockDiagonalCausalFromBottomRightMask.from_seqlens(query_lens, seq_lens)
    output = ref_masked_attention(query, key, value, scale, attn_mask)

    return output.unsqueeze(1)


@pytest.mark.parametrize("num_heads,num_queries_per_kv,head_size", [
    (4, 2, 8),
])
@torch.inference_mode()
def test_contexted_kv_attention(
    num_heads: int,
    num_queries_per_kv: int,
    head_size: int,
) -> None:
    import os
    os.environ["NEURON_CC_FLAGS"]= " -O1 --internal-hlo2tensorizer-options='--verify-hlo' --retry_failed_compilation "

    random.seed(0)
    torch.manual_seed(0)

    max_ctx_len = 64
    max_query_len = 64
    prefill_batch_size = 1
    decode_batch_size = 7
    batch_size = prefill_batch_size + decode_batch_size
    block_size = 32
    max_model_len = (max_query_len + max_ctx_len) * 4
    tp_degree = 2

    max_block_per_request = max_model_len // block_size
    dtype = torch.float32
    cache_size = (batch_size * max_block_per_request) + 2
    ctx_lens = [random.randint(2, max_ctx_len) for _ in range(prefill_batch_size)] + \
        [random.randint(2, max_ctx_len) for _ in range(decode_batch_size)]
    query_lens = [random.randint(2, max_query_len) for _ in range(prefill_batch_size)] + \
        [1 for _ in range(decode_batch_size)]
    seq_lens = [a + b for a, b in zip(query_lens, ctx_lens)]
    num_kv_heads = num_heads // num_queries_per_kv

    num_tokens = sum(query_lens)
    query = torch.empty(num_tokens, num_heads, head_size, dtype=dtype)
    query.uniform_(-1, 1)
    output = torch.empty(num_tokens, num_heads, head_size, dtype=dtype)

    kv = torch.empty(sum(seq_lens), 2, num_kv_heads, head_size, dtype=dtype)
    kv.uniform_(-1, 1)
    key, value = kv.unbind(dim=1)

    k_cache = torch.zeros(cache_size,
                          block_size,
                          num_kv_heads,
                          head_size,
                          dtype=dtype)
    v_cache = torch.zeros(cache_size,
                          block_size,
                          num_kv_heads,
                          head_size,
                          dtype=dtype)
    k = torch.zeros(sum(query_lens), num_kv_heads, head_size, dtype=dtype)
    v = torch.zeros(sum(query_lens), num_kv_heads, head_size, dtype=dtype)
    values = torch.arange(0, cache_size, dtype=torch.long)
    values = values[torch.randperm(cache_size)]
    block_table = values[:batch_size * max_block_per_request].view(
        batch_size, max_block_per_request)
    b_seq_len = torch.tensor(seq_lens, dtype=torch.long)
    b_ctx_len = torch.tensor(ctx_lens, dtype=torch.long)
    b_start_loc = torch.cumsum(torch.tensor([0] + query_lens[:-1],
                                            dtype=torch.long),
                               dim=0)
    # copy kv to cache
    b_seq_start_loc = torch.cumsum(torch.tensor([0] + seq_lens[:-1],
                                                dtype=torch.long),
                                   dim=0)
    for i in range(batch_size):
        for j in range(query_lens[i]):
            k[b_start_loc[i] + j].copy_(key[b_seq_start_loc[i] + b_ctx_len[i] + j])
            v[b_start_loc[i] + j].copy_(value[b_seq_start_loc[i] + b_ctx_len[i] + j])
        cur_ctx = 0
        block_id = 0
        while cur_ctx < b_ctx_len[i]:
            start_loc = b_seq_start_loc[i] + cur_ctx
            if cur_ctx + block_size > b_ctx_len[i]:
                end_loc = b_seq_start_loc[i] + b_ctx_len[i]
            else:
                end_loc = start_loc + block_size
            start_slot = block_table[i, block_id] * block_size
            end_slot = start_slot + end_loc - start_loc
            k_cache.view(-1, num_kv_heads,
                         head_size)[start_slot:end_slot].copy_(
                             key[start_loc:end_loc])
            v_cache.view(-1, num_kv_heads,
                         head_size)[start_slot:end_slot].copy_(
                             value[start_loc:end_loc])
            cur_ctx += block_size
            block_id += 1

    output_ref = ref_context_attention(query, key, value, query_lens, seq_lens, head_size,
                                       num_kv_heads, num_heads, num_queries_per_kv)

    # build neuron program
    max_num_seqs = batch_size
    n_blocks = (max_model_len * max_num_seqs) // block_size
    max_num_queries = ((sum(query_lens) + block_size - 1) // block_size) * block_size
    max_num_keys = n_blocks * block_size

    def context_attention_fwd(query, key, value, key_cache, value_cache, block_table, attn_mask):
        import numpy as np
        import neuronxcc.nki as nki
        from vllm.attention.ops.nki_flash_attn import flash_paged_attention

        # grid_x = a_input.shape[0] // 128
        # grid_y = a_input.shape[1] // 512
        # c_output = np.zeros(a_input.shape, dtype=a_input.dtype)
        # nki_tensor_add_kernel_baremetal = nki.baremetal(nki_tensor_add_kernel_)
        # nki_tensor_add_kernel_baremetal[grid_x, grid_y](a_input, b_input, c_output)

        o = flash_paged_attention[1, 1](
            query, key, value, key_cache, value_cache,
            block_table, attn_mask,
            softmax_scale=None, config=None,
            mixed_precision=True,
            return_softmax_max_sum=False
        )

        return o

    def get_active_block_tables(block_tables, query_lens, seq_lens, block_size, num_blocks):
        context_lens = seq_lens - query_lens
        blocks_per_seq = (context_lens + block_size - 1) // block_size
        num_seqs = len(seq_lens)
        active_blocks = []
        for seq_id in range(num_seqs):
            active_blocks = active_blocks + block_tables[seq_id, :blocks_per_seq[seq_id]].tolist()
        return F.pad(torch.tensor(active_blocks), (0, num_blocks-len(active_blocks)), "constant", 0)

    def shift_bit_length(x):
        return 1<<(x-1).bit_length()

    B_P_SIZE = 128
    max_num_queries_padded = shift_bit_length(max_num_queries) * 2
    head_size_padded = B_P_SIZE
    context_lens = torch.tensor(seq_lens) - torch.tensor(query_lens)
    num_active_blocks = shift_bit_length(((context_lens + block_size - 1) // block_size).sum().item()) * 4
    pad_dims = (
        0, head_size_padded-query.shape[2],
        0, 0,
        0, max_num_queries_padded-query.shape[0]
    )
    query = F.pad(query, pad_dims, "constant", 0)
    k = F.pad(k, pad_dims, "constant", 0)
    v = F.pad(v, pad_dims, "constant", 0)
    output_ref_padded = F.pad(output_ref, (0, 0, 0, 0, 0, 0, 0, max_num_queries_padded-output_ref.shape[0]), "constant", 0)
    active_block_table = get_active_block_tables(
        block_table, torch.tensor(query_lens), torch.tensor(seq_lens), block_size, num_active_blocks)
    k_cache = F.pad(k_cache, (0, head_size_padded-head_size), "constant", 0)
    v_cache = F.pad(v_cache, (0, head_size_padded-head_size), "constant", 0)

    import torch_xla.core.xla_model as xm
    device = xm.xla_device()
    attn_mask = BlockDiagonalCausalFromBottomRightMask.from_seqlens(query_lens, seq_lens)
    context_kv_len = num_active_blocks * block_size
    attn_mask = F.pad(attn_mask, (
        0, context_kv_len+B_P_SIZE-attn_mask.shape[1],
        0, B_P_SIZE-attn_mask.shape[0]
    ), "constant", 0).bool()

    example = (
        # query: shape (1, n_heads, d, seq_q)
        # key:   shape (1, n_kv_heads, d, seq_k)
        # value: shape (1, n_kv_heads, d, seq_v) if config.should_transpose_v else (1, n_kv_heads, seq_v, d)
        query.unsqueeze(1).transpose(0,1).permute(0, 2, 3, 1).contiguous().to(device=device),
        k.unsqueeze(1).transpose(0,1).permute(0, 2, 3, 1).contiguous().to(device=device),
        v.unsqueeze(1).transpose(0,1).permute(0, 2, 3, 1).contiguous().to(device=device),
        # transpose K_cache[num_blocks, block_size, num_kv_heads, head_size]
        # to K_cache[num_blocks, num_kv_heads, head_size, block_size]
        # k_cache.permute(0, 2, 3, 1).contiguous().to(device=device),
        k_cache.to(device=device),
        # transpose V_cache[num_blocks, block_size, num_kv_heads, head_size]
        # to V_cache[num_blocks, num_kv_heads, block_size, head_size]
        # v_cache.permute(0, 2, 1, 3).contiguous().to(device=device),
        v_cache.to(device=device),
        active_block_table.to(torch.int32).to(device=device),
        # torch.tensor(query_lens).to(device=device),
        # torch.tensor(seq_lens).to(device=device),
        attn_mask.to(device=device),
    )
    output_nki = context_attention_fwd(*example)
    # - o: shape (bs, n_heads, seq_q, d) -> (bs, seq_q, n_heads, d)
    output_nki = output_nki.permute(0, 2, 1, 3)[:,:,:,:head_size].cpu()
    output_ref_padded = output_ref_padded.transpose(0, 1)
    torch.testing.assert_close(output_nki, output_ref_padded, atol=1e-2, rtol=0)
