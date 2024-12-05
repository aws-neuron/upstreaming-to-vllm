"""
Copyright (c) 2023, Amazon.com. All Rights Reserved

kernels - Builtin high performance attention kernels

"""
import numpy as np

import neuronxcc.nki.isa as nisa
import neuronxcc.nki.language as nl
from neuronxcc import nki

from neuronxcc.nki.language import par_dim
from dataclasses import dataclass
from functools import reduce as functools_reduce
from operator import mul as operator_mul


def n_elts(shape):
  return functools_reduce(operator_mul, shape, 1)


def linearize(shape, indices):
  return sum(i * (n_elts(shape[dim + 1:]))
             for dim, i in enumerate(indices))


def div_ceil(n, d):
  return (n + d - 1) // d


@dataclass(frozen=True)
class FlashConfig:
  """
    Config class for flash attention with default values
  """
  seq_tile_size:int = 2048
  training:bool = True
  should_transpose_v:bool = False

  __annotations__ = {
    'seq_tile_size': int,
    'training': bool,
    'should_transpose_v': bool
  }

@nki.jit
def transpose_p_local(p_local_transposed, p_local, LARGE_TILE_SZ, forward_mask, B_F_SIZE=512):
  for i in nl.affine_range(LARGE_TILE_SZ // B_F_SIZE):
    if nisa.get_nc_version() == nisa.nc_version.gen3:
      p_local_t_tmp = nl.ndarray((par_dim(128), B_F_SIZE), buffer=nl.sbuf, dtype=p_local.dtype)
    else:
      p_local_t_tmp = nl.ndarray((par_dim(128), B_F_SIZE), buffer=nl.psum, dtype=np.float32)

    for j in nl.affine_range(B_F_SIZE // 128):
      j_128_slice = nl.ds(j * 128, 128)
      i_j_128_slice = nl.ds(i * B_F_SIZE + j * 128, 128)

      if nisa.get_nc_version() == nisa.nc_version.gen3:
        p_local_t_tmp[:, j_128_slice] = nisa.dma_transpose(
          p_local[:, i_j_128_slice], mask=forward_mask)
      else:
        p_local_t_tmp[:, j_128_slice] = nisa.nc_transpose(
          p_local[:, i_j_128_slice], mask=forward_mask)

    p_local_transposed[:, nl.ds(i * B_F_SIZE, B_F_SIZE)] = nl.copy(
      p_local_t_tmp, dtype=p_local_transposed.dtype, mask=forward_mask)


@nki.jit
def dropout_p_local(p_local, dropout_p, dropout_p_tensor, seed_tensor,
                    seed_offset_base, k_r_i, REDUCTION_TILE, forward_mask):
  B_F_SIZE = 512
  for k_d_i in nl.sequential_range(REDUCTION_TILE // B_F_SIZE):
    p_local_f_slice = nl.ds(k_r_i * REDUCTION_TILE + k_d_i * B_F_SIZE, B_F_SIZE)

    offset = k_d_i + seed_offset_base
    offset_seed = nl.add(seed_tensor[0, 0], offset, mask=forward_mask)
    nl.random_seed(seed=offset_seed, mask=forward_mask)
    softmax_dropout = nl.dropout(p_local[:, p_local_f_slice],
                                 rate=dropout_p_tensor[:, 0],
                                 mask=forward_mask)
    p_local[:, p_local_f_slice] = nl.multiply(
      softmax_dropout, 1 / (1 - dropout_p), mask=forward_mask)


@nki.jit
def _flash_attention_core(q_local_tile, k, v,
                          q_h_per_k_h, seqlen_q, nheads,
                          o_buffer, l_buffer, m_buffer,
                          batch_id, head_id, gqa_head_idx, q_tile_idx,
                          local_k_large_tile_idx,
                          kernel_dtype, acc_type,
                          flash_config: FlashConfig,
                          use_causal_mask=False, continuous_batching_mask=None, initialize=False,
                          B_P_SIZE=128, B_F_SIZE=512, B_D_SIZE=128,
                          dropout_p=0.0, dropout_p_tensor=None, seed_tensor=None, logit_bias_tile=None
                          ):
  """
  The flash attention core function to calcualte self attention between a tile of q and a block of K and V.
  The q_local_tile has (B_P_SIZE, B_F_SIZE), which is loaded into the SBUF already. The block size of K and V
  is defined in the seq_tile_size of the flash_config. The results are stored in the following three buffers
  o_buffer: (B_P_SIZE, d)
  l_buffer: (B_P_SIZE, 1)
  m_buffer: (B_P_SIZE, 1)
  """
  LARGE_TILE_SZ = flash_config.seq_tile_size
  num_k_tile_per_large_tile = LARGE_TILE_SZ // B_F_SIZE
  seqlen_k = k.shape[-1]
  seq_q_num_tiles = seqlen_q // B_P_SIZE
  seq_k_num_tiles = seqlen_k // B_F_SIZE

  # TODO: support logit_bias with continuous_batching_mask
  if continuous_batching_mask is not None:
    assert logit_bias_tile is None, "continuous_batching_mask does not support logit_bias!"

  # mask are used to only apply computation to the lower half of the matrix,
  # which reduce the arthimetic intensity by half
  forward_mask = q_tile_idx * B_P_SIZE >= local_k_large_tile_idx * LARGE_TILE_SZ if use_causal_mask else None

  qk_res_buf = nl.ndarray((par_dim(B_P_SIZE), LARGE_TILE_SZ), buffer=nl.sbuf, dtype=acc_type)
  max_local = nl.ndarray((par_dim(B_P_SIZE), num_k_tile_per_large_tile), dtype=acc_type)
  for k_i in nl.affine_range(num_k_tile_per_large_tile):
    k_i_b_f_slice = nl.ds(k_i * B_F_SIZE, B_F_SIZE)

    qk_psum = nl.zeros((par_dim(B_P_SIZE), B_F_SIZE),
                        dtype=np.float32, buffer=nl.psum)  # (128, 512)
    multiplication_required_selection = local_k_large_tile_idx * LARGE_TILE_SZ + k_i * B_F_SIZE <= q_tile_idx * B_P_SIZE if use_causal_mask else None
    qk_psum[:, :] = nl.matmul(q_local_tile, k[:, k_i_b_f_slice], transpose_x=True,
                              mask=multiplication_required_selection) # (p(128), 512)

    if use_causal_mask:
      left_diagonal_selection = q_tile_idx * B_P_SIZE >= local_k_large_tile_idx * LARGE_TILE_SZ + (k_i + 1) * B_F_SIZE
      diagonal_and_right_selection = (q_tile_idx * B_P_SIZE < local_k_large_tile_idx * LARGE_TILE_SZ + (k_i + 1) * B_F_SIZE) & forward_mask
      diagonal_and_left_selection = ((q_tile_idx + 1) * B_P_SIZE > local_k_large_tile_idx * LARGE_TILE_SZ + k_i * B_F_SIZE)
      right_diagonal_selection = ((q_tile_idx + 1) * B_P_SIZE <= local_k_large_tile_idx * LARGE_TILE_SZ + k_i * B_F_SIZE) & forward_mask
      diagonal = (q_tile_idx * B_P_SIZE < local_k_large_tile_idx * LARGE_TILE_SZ + (k_i + 1) * B_F_SIZE) & \
          ((q_tile_idx + 1) * B_P_SIZE > local_k_large_tile_idx * LARGE_TILE_SZ + k_i * B_F_SIZE)

      i_q_p, i_q_f = nl.mgrid[0:B_P_SIZE, 0:B_F_SIZE]
      q_pos = q_tile_idx * B_P_SIZE + i_q_p
      k_pos = local_k_large_tile_idx * LARGE_TILE_SZ + k_i * B_F_SIZE + i_q_f
      pred = q_pos >= k_pos

      if logit_bias_tile is not None:
        # For tiles to the right of the diagonal, do affine_select.
        # Magic number -9984.0 to replace -inf similar to what Tensorizer uses
        qk_res_buf[:, k_i_b_f_slice] = nisa.affine_select(
            pred=pred,
            on_true_tile=qk_psum, on_false_value=-9984.0, dtype=acc_type,
            mask=right_diagonal_selection)

        # For tiles on the diagonal, add logit bias and need to do affine_select.
        intermediate = \
            nl.add(qk_psum, logit_bias_tile[:, k_i_b_f_slice],
                   dtype=acc_type, mask=diagonal)
        qk_res_buf[:, k_i_b_f_slice] = nisa.affine_select(
            pred=pred,
            on_true_tile=intermediate, on_false_value=-9984.0, dtype=acc_type,
            mask=diagonal)

        # For tiles on the left of the diagonal, just add logit bias, no select required.
        qk_res_buf[:, k_i_b_f_slice] = \
            nl.add(qk_psum, logit_bias_tile[:, k_i_b_f_slice],
                   dtype=acc_type, mask=left_diagonal_selection)
      else:
        # For tiles on and to the right of the diagonal, need to do affine_select.
        # Magic number -9984.0 to replace -inf similar to what Tensorizer uses
        qk_res_buf[:, k_i_b_f_slice] = nisa.affine_select(
          pred=pred,
          on_true_tile=qk_psum, on_false_value=-9984.0, dtype=acc_type,
          mask=diagonal_and_right_selection)

        # For tiles on the left of the diagonal, direct copy, no select required.
        qk_res_buf[:, k_i_b_f_slice] = \
          nl.copy(qk_psum, dtype=acc_type, mask=left_diagonal_selection)
    elif continuous_batching_mask is not None:
        qk_res_buf[:, k_i_b_f_slice] = nl.where(
          continuous_batching_mask[:, k_i_b_f_slice], qk_psum[:, nl.ds(0, B_F_SIZE)], -9984.0, dtype=acc_type)
    else:
      if logit_bias_tile is not None:
        # Simply add logit bias which copies back to sbuf at the same time
        qk_res_buf[:, k_i_b_f_slice] = \
            nl.add(qk_psum, logit_bias_tile[:, k_i_b_f_slice], dtype=acc_type)
      else:
        # Simply send psum result back to sbuf
        qk_res_buf[:, k_i_b_f_slice] = nl.copy(qk_psum, dtype=acc_type)

    # Calculate max of the current tile
    max_local[:, k_i] = nisa.tensor_reduce(
      np.max, qk_res_buf[:, k_i_b_f_slice], axis=(1,), dtype=acc_type,
      negate=False, mask=forward_mask)

  max_ = nisa.tensor_reduce(np.max, max_local[:, :], axis=(1, ),
                            dtype=acc_type, negate=False, mask=forward_mask)

  o_previous_scaled = nl.ndarray((par_dim(B_P_SIZE), B_D_SIZE), dtype=o_buffer.dtype)

  if initialize:
    m_buffer[:, 0] = nl.copy(max_)
    m_current = max_
  else:
    m_previous = nl.copy(m_buffer[:, 0])
    m_buffer[:, 0] = nl.maximum(m_previous, max_, mask=forward_mask) # (128,1)

    m_current = m_buffer[:, 0]
    # Compute scaling factor
    alpha = nisa.activation(np.exp, m_previous, bias=-1*m_current, scale=1.0, mask=forward_mask)
    o_previous_scaled[...] = nl.multiply(o_buffer[:, :], alpha, mask=forward_mask)

  p_local = nl.ndarray((par_dim(B_P_SIZE), LARGE_TILE_SZ), dtype=kernel_dtype)
  REDUCTION_TILE = min(2048, LARGE_TILE_SZ // 2)

  p_partial_sum = nl.ndarray((par_dim(B_P_SIZE), LARGE_TILE_SZ // REDUCTION_TILE), dtype=acc_type)


  for k_r_i in nl.affine_range(LARGE_TILE_SZ // REDUCTION_TILE):
    k_r_i_reduce_slice = nl.ds(k_r_i * REDUCTION_TILE, REDUCTION_TILE)

    # dropout
    if dropout_p > 0.0:
      # compute exp(qk-max)
      p_local[:, k_r_i_reduce_slice] = \
        nisa.activation(np.exp, qk_res_buf[:, k_r_i_reduce_slice],
                        bias=-1 * m_current, scale=1.0,
                        dtype=kernel_dtype, mask=forward_mask)

      seed_offset_base = k_r_i * (REDUCTION_TILE // B_F_SIZE) \
                         + local_k_large_tile_idx * (LARGE_TILE_SZ // B_F_SIZE) \
                         + q_tile_idx * seq_k_num_tiles \
                         + (head_id * q_h_per_k_h + gqa_head_idx) * seq_k_num_tiles * seq_q_num_tiles \
                         + batch_id * nheads * seq_k_num_tiles * seq_q_num_tiles

      dropout_p_local(p_local=p_local, dropout_p=dropout_p,
                      dropout_p_tensor=dropout_p_tensor, seed_tensor=seed_tensor,
                      seed_offset_base=seed_offset_base, k_r_i=k_r_i,
                      REDUCTION_TILE=REDUCTION_TILE, forward_mask=forward_mask)

      # Compute partial row-tile sum of exp(qk-max))
      # FIXME: Use activation accumulate and accumulate over k_r_i loop?
      p_partial_sum[:, k_r_i] = nl.sum(p_local[:, k_r_i_reduce_slice],
                                       axis=1, dtype=acc_type, mask=forward_mask)
    else:
      # compute exp(qk-max)
      # Compute partial row-tile sum of exp(qk-max))
      # FIXME: Use activation accumulate to accumulate over k_r_i loop?
      p_local[:, k_r_i_reduce_slice] = \
        nisa.activation_reduce(np.exp, qk_res_buf[:, k_r_i_reduce_slice],
                               bias=-1 * m_current, scale=1.0,
                               reduce_op=nl.add, reduce_res=p_partial_sum[:, k_r_i],
                               dtype=kernel_dtype, mask=forward_mask)

  ps = nl.sum(p_partial_sum, axis=1, dtype=acc_type, mask=forward_mask)

  p_local_transposed = nl.ndarray((par_dim(B_P_SIZE), LARGE_TILE_SZ), dtype=kernel_dtype)
  transpose_p_local(p_local_transposed=p_local_transposed, p_local=p_local,
                    LARGE_TILE_SZ=LARGE_TILE_SZ, forward_mask=forward_mask, B_F_SIZE=B_F_SIZE)

  pv_psum = nl.zeros((par_dim(B_P_SIZE), B_D_SIZE), dtype=np.float32, buffer=nl.psum)
  for k_i in nl.affine_range(LARGE_TILE_SZ // B_P_SIZE):
    pv_psum[:, :] += nl.matmul(p_local_transposed[:, nl.ds(k_i * B_P_SIZE, B_P_SIZE)],
                               v[k_i, :, :], transpose_x=True, mask=forward_mask) # (128, 128) (p(Br), d)

  if initialize:
    o_buffer[:, :] = nl.copy(pv_psum[:, :])
    l_buffer[:, 0] = nl.add(nl.log(ps), max_)
  else:
    o_buffer[:, :] = nl.add(o_previous_scaled, pv_psum, mask=forward_mask)

    l_prev = l_buffer[:, 0]
    l_exp = nl.add(nl.exp(nl.subtract(l_prev, m_current, mask=forward_mask), mask=forward_mask), ps, mask=forward_mask)
    l_buffer[:, 0] = nl.add(m_current, nl.log(l_exp, mask=forward_mask), mask=forward_mask)


@nki.jit
def load_v_tile(v_hbm_tile, cur_v_tile, j, v_i, config):
  LARGE_TILE_SZ = config.seq_tile_size
  B_P_SIZE = 128

  if not config.should_transpose_v:
    cur_v_tile[v_i, :, :] = nl.load(
      v_hbm_tile[nl.ds(j * LARGE_TILE_SZ + B_P_SIZE * v_i, B_P_SIZE), :],
      dtype=cur_v_tile.dtype)
    return

  if nisa.get_nc_version() == nisa.nc_version.gen3:
    cur_v_tile_transposed = nisa.dma_transpose(
      v_hbm_tile[:, nl.ds(j * LARGE_TILE_SZ + B_P_SIZE * v_i, B_P_SIZE)])
    cur_v_tile[v_i, :, :] = nisa.tensor_copy(cur_v_tile_transposed,
                                             dtype=cur_v_tile.dtype)
    return

  cur_v_tile[v_i, :, :] = nl.load_transpose2d(
    v_hbm_tile[:, nl.ds(j * LARGE_TILE_SZ + B_P_SIZE * v_i, B_P_SIZE)],
    dtype=cur_v_tile.dtype)



@nki.jit
def flash_fwd(q, k, v, seed, logit_bias=None,
              softmax_scale=None,
              use_causal_mask=True,
              mixed_precision=True,
              dropout_p=0.0, config=None):
  """
  Flash Attention Forward kernel

  IO tensor layouts:
    - q: shape   (bs, n_heads, d, seq_q)
    - k: shape   (bs, nk_heads, d, seq_k)
    - v: shape   (bs, nv_heads, d, seq_v) if config.should_transpose_v  else (bs, nv_heads, seq_v, d)
    - seed: shape (1,)
    - logit_bias: shape (bs, n_heads, seq_q, seq_k)
    - o: shape (bs, n_heads, seq_q, d)
    - lse: shape (bs, n_heads, nl.tile_size.pmax, seq // nl.tile_size.pmax) if training else None
    - This kernel requires seq_k == seq_v

  IO tensor dtypes:
    - This kernel assumes all IO tensors have the same dtype
    - If mixed_percision is True, then all Tensor Engine operation will be performed in
      bfloat16 and accumulation will be performed in float32. Otherwise the intermediates
      will be in the same type as the inputs.

  Compile-time Constants:
    - softmax_scale: scaling for softmax, is None, default is `1.0/(d**0.5)`
    - mixed_precision: flag to set non-matmul ops in fp32 precision, defualt is set to `true`, if false, we use same precision as input types
    - causal_mask: flag to set causal masking
    - config: Instance of dataclass :class:`nki.kernels.attention.FlashConfig` with Performance config parameters for flash attention with default values
        seq_tile_size: `default=2048`, size of the kv tile size for attention computation reduction
        training: bool to indicate training vs inference `default=True`

  Performance Notes:
    For better performance, the kernel is tiled to be of size `LARGE_TILE_SZ`, and Flash attention math techniques are applied in unit
    of `LARGE_TILE_SZ`. Seqlen that is not divisible by `LARGE_TILE_SZ` is not supported at the moment.

  GQA support Notes:
    the spmd kernel for launching kernel should be on kv_heads instead of nheads

  Example usage:
    MHA: q: [b, h, d, s], k: [b, h, d, s], v: [b, h, s, d]
      usage: `flash_fwd[b, h](q, k, v, ...)`
    GQA: q: [b, h, d, s], k: [b, kv_h, d, s], v: [b, kv_h, s, d]
      usage: `flash_fwd[b, kv_h](q, k, v, ...)`
  """
  config = config or FlashConfig()
  B_F_SIZE=512
  B_P_SIZE=128
  b, h, d, seqlen_q  = q.shape
  B_D_SIZE = d
  _, k_h, _, seqlen_k = k.shape
  if config.should_transpose_v:
    assert tuple(v.shape) == (b, k_h, d, seqlen_k), f"Expect shape of V to be {(b, k_h, d, seqlen_k)} (batch, heads, d_head, seqlen_k) but got {v.shape}"
    assert tuple(k.shape) == (b, k_h, d, seqlen_k), f"Expect shape of K to be {(b, k_h, d, seqlen_k)} (batch, heads, d_head, seqlen_k) but got {k.shape}"
  else:
    assert tuple(v.shape) == (b, k_h, seqlen_k, d), f"Expect shape of V to be {(b, k_h, seqlen_k, d)} (batch, heads, seqlen_k, d_head) but got {v.shape}"
    assert tuple(k.shape) == (b, k_h, d, seqlen_k), f"Expect shape of K to be {(b, k_h, d, seqlen_k)} (batch, heads, d_head, seqlen_k) but got {k.shape}"
  assert d <= 128, f" we do not support head_dim > 128, got head dim {d}"
  kernel_dtype = nl.bfloat16 if mixed_precision else q.dtype
  acc_type = np.dtype(np.float32) if mixed_precision else kernel_dtype

  o = nl.ndarray((b, h, seqlen_q, d), dtype=q.dtype, buffer=nl.shared_hbm)
  if config.training:
    lse = nl.ndarray((b, h, nl.tile_size.pmax, seqlen_q // nl.tile_size.pmax),
                     dtype=acc_type, buffer=nl.shared_hbm)
  else:
    lse = None

  assert nl.program_ndim() == 2,\
    f'Expect spmd grid with 2 dimensions, got {nl.program_ndim()} instead!'
  batch_id = nl.program_id(axis=0)
  head_id = nl.program_id(axis=1)

  softmax_scale = softmax_scale or (1.0 / (d ** 0.5))

  n_tile_q = seqlen_q // B_P_SIZE # since q will be loaded on tensor engine

  LARGE_TILE_SZ = config.seq_tile_size
  # FIXME: Add masking for different seqlen values.
  assert config.seq_tile_size >= 512, f" seq tile_size {config.seq_tile_size} cannot be less than 512"
  assert seqlen_k % LARGE_TILE_SZ == 0, f"Need seqlen_k to be divisible by {LARGE_TILE_SZ} but got {seqlen_k}"
  num_large_k_tile = seqlen_k // LARGE_TILE_SZ

  # inference flag, check if lse is none
  inference = not config.training
  if inference:
    assert lse is None, "lse should be none for inference"
    assert seed is None, f"seed should be None for inference, but got {seed}"
    assert dropout_p==0.0, f"dropout should be 0.0 for inference but got {dropout_p}"
  else:
    assert lse is not None, "lse should not be none for training"
  q_h_per_k_h = h // k_h

  if dropout_p > 0.0 and not inference:
    seed_local = nl.load(seed[0])
    # TODO: Remove this once the dropout supports scale prob
    dropout_p_tensor = nl.full((B_P_SIZE, 1), fill_value=dropout_p, dtype=np.float32)
  else:
    dropout_p_tensor = None
    seed_local = None

  if logit_bias is not None:
    b_logit_bias, h_logit_bias, _, _ = logit_bias.shape
    assert b_logit_bias == 1 and h_logit_bias == 1, "only support broadcasting logit_bias with batch 1, n_heads 1"

  for i_q_h in nl.affine_range(q_h_per_k_h):

    # =============== Global Flash Attention accumulators ====================== #
    o_buffer = nl.zeros((n_tile_q, par_dim(B_P_SIZE), d), dtype=acc_type,
                        buffer=nl.sbuf, lazy_initialization=True)
    l_buffer = nl.zeros((par_dim(B_P_SIZE), n_tile_q), dtype=acc_type,
                        buffer=nl.sbuf, lazy_initialization=True)
    m_buffer = nl.zeros((n_tile_q, par_dim(B_P_SIZE), 1), dtype=acc_type,
                        buffer=nl.sbuf, lazy_initialization=True)
    # =============== Global Flash Attention accumulators END ================== #


    for j in nl.sequential_range(0, num_large_k_tile):
      cur_k_tile = nl.ndarray((par_dim(B_D_SIZE), LARGE_TILE_SZ), dtype=kernel_dtype)
      cur_v_tile = nl.ndarray((LARGE_TILE_SZ // B_P_SIZE, par_dim(B_P_SIZE), B_D_SIZE), dtype=kernel_dtype)

      cur_k_tile[:, :] = nl.load(k[batch_id, head_id, :, nl.ds(j*LARGE_TILE_SZ, LARGE_TILE_SZ)])

      load_tile_size = B_P_SIZE

      v_hbm_tile = v[batch_id, head_id]
      for v_i in nl.affine_range(LARGE_TILE_SZ // load_tile_size):
        load_v_tile(v_hbm_tile=v_hbm_tile, cur_v_tile=cur_v_tile, j=j, v_i=v_i,
                    config=config)

      for i in nl.affine_range(n_tile_q):
        q_tile = nl.ndarray((B_D_SIZE, B_P_SIZE),dtype=kernel_dtype)
        q_hbm_tile = q[batch_id, head_id * q_h_per_k_h + i_q_h]
        q_sbuf_tile = nl.load(q_hbm_tile[:, nl.ds(i * B_P_SIZE, B_P_SIZE)],
                              dtype=kernel_dtype) # load (d, 128) tile in SBUF
        q_tile[:, :] = q_sbuf_tile * softmax_scale

        logit_bias_tile = None
        if logit_bias is not None:
          logit_bias_tile = nl.ndarray((par_dim(B_P_SIZE), LARGE_TILE_SZ), dtype=kernel_dtype)
          logit_bias_tile[:, :] = nl.load(
            logit_bias[0, 0, nl.ds(i * B_P_SIZE, B_P_SIZE),
                       nl.ds(j * LARGE_TILE_SZ, LARGE_TILE_SZ)])

        _flash_attention_core(q_local_tile=q_tile, k=cur_k_tile, v=cur_v_tile,
                              q_h_per_k_h=q_h_per_k_h, seqlen_q=seqlen_q, nheads=h,
                              o_buffer=o_buffer[i], l_buffer=l_buffer[:, i], m_buffer=m_buffer[i],
                              batch_id=batch_id, head_id=head_id,
                              gqa_head_idx=i_q_h, q_tile_idx=i, local_k_large_tile_idx=j,
                              kernel_dtype=kernel_dtype, acc_type=acc_type,
                              flash_config=config, use_causal_mask=use_causal_mask,
                              initialize=j == 0,
                              B_P_SIZE=B_P_SIZE, B_F_SIZE=B_F_SIZE, B_D_SIZE=B_D_SIZE,
                              dropout_p=dropout_p, dropout_p_tensor=dropout_p_tensor,
                              seed_tensor=seed_local, logit_bias_tile=logit_bias_tile)

    # -------- write output to buffer on HBM ------------ #
    for i in nl.affine_range(n_tile_q):
      out = nl.multiply(o_buffer[i, :, :],
                        nl.exp(m_buffer[i, :, :] - l_buffer[:, i]),
                        dtype=kernel_dtype)

      nl.store(o[batch_id, head_id * q_h_per_k_h + i_q_h,
                 nl.ds(i*B_P_SIZE, B_P_SIZE), :], out)

    if not inference:
      nl.store(lse[batch_id, head_id * q_h_per_k_h + i_q_h, :, :], l_buffer[:, :])

  if config.training:
    return o, lse

  return o


@nki.jit
def load_dy_q(dy_ref_hbm_tile, q_ref_hbm_tile, dy_local, q_local, d_head_n_tiles, d_head_tile_size, i_q_seq_tile,
              q_seq_tile_size, softmax_scale):
  for i_d_head_tile in nl.affine_range(d_head_n_tiles):
    i_d_head_dslice = nl.ds(i_d_head_tile * d_head_tile_size, d_head_tile_size)
    i_q_seq_dslice = nl.ds(i_q_seq_tile * q_seq_tile_size, q_seq_tile_size)

    dy_local[i_d_head_tile, :, :] = nl.load(
      dy_ref_hbm_tile[i_d_head_dslice, i_q_seq_dslice],
      dtype=dy_local.dtype)

    q_local[i_d_head_tile, :, :] = nl.load(
      q_ref_hbm_tile[i_d_head_dslice, i_q_seq_dslice],
      dtype=q_local.dtype) * softmax_scale


@nki.jit
def store_dk_dv(out_dk_ref_hbm_tile, out_dv_ref_hbm_tile, local_dk, local_dv,
                d_head_n_tiles, d_head_tile_size, i_k_seq_dslice):
  for i in nl.affine_range(d_head_n_tiles):
    i_d_head_dslice = nl.ds(i * d_head_tile_size, d_head_tile_size)

    nl.store(out_dv_ref_hbm_tile[i_d_head_dslice, i_k_seq_dslice],
             value=local_dv[i, :, :])

    nl.store(out_dk_ref_hbm_tile[i_d_head_dslice, i_k_seq_dslice],
             value=local_dk[i, :, :])


@nki.jit
def load_kv(k_ref_hbm_tile, v_ref_hbm_tile, k_local, transposed_k_local, v_local,
            d_head_n_tiles, d_head_tile_size, i_k_seq_tile, k_seq_tile_size,
            k_seq_tile_size_backward):
  k_seq_fwd_bwd_tile_multipler = k_seq_tile_size // k_seq_tile_size_backward

  for i in nl.affine_range(d_head_n_tiles):
    i_d_head_dslice = nl.ds(i * d_head_tile_size, d_head_tile_size)
    i_k_seq_dslice = nl.ds(i_k_seq_tile * k_seq_tile_size, k_seq_tile_size)
    k_local[i, :, :] = nl.load(k_ref_hbm_tile[i_d_head_dslice, i_k_seq_dslice],
                                           dtype=k_local.dtype)
    v_local[i, :, :] = nl.load(v_ref_hbm_tile[i_d_head_dslice, i_k_seq_dslice],
                                           dtype=v_local.dtype)
    ##############################################################
    # Prefetch k transpose for the backward too
    ##############################################################
    for j in nl.affine_range(k_seq_fwd_bwd_tile_multipler):
      i_k_dslice = nl.ds(j * k_seq_tile_size_backward, k_seq_tile_size_backward)
      transposed_k_local[j, i, :, :] = nisa.nc_transpose(k_local[i, :, i_k_dslice])


@nki.jit
def compute_rowsum(dy_o_sum, dy_ref_hbm_tile, o_ref_hbm_tile, d_head_n_tiles, d_head_tile_size, q_seq_n_tiles,
                   q_seq_tile_size):
  mixed_dtype = dy_o_sum.dtype
  for i in nl.affine_range(q_seq_n_tiles):
    dy_o_partial = nl.zeros((par_dim(q_seq_tile_size), d_head_n_tiles), dtype=mixed_dtype)
    for j in nl.affine_range(d_head_n_tiles):
      d_head_dslice = nl.ds(j * d_head_tile_size, d_head_tile_size)
      q_seq_dslice = nl.ds(i * q_seq_tile_size, q_seq_tile_size)

      dy_local = nl.load_transpose2d(dy_ref_hbm_tile[d_head_dslice, q_seq_dslice],
                                     dtype=mixed_dtype)
      o_local = nl.load_transpose2d(o_ref_hbm_tile[d_head_dslice, q_seq_dslice],
                                    dtype=mixed_dtype)

      dy_o = nl.multiply(dy_local, o_local, dtype=mixed_dtype)
      dy_o_partial[:, j] = nisa.tensor_reduce(np.add, data=dy_o, axis=(1,),
                                              dtype=mixed_dtype)

    dy_o_sum[i, :, 0] = nisa.tensor_reduce(
      np.add, data=dy_o_partial[:, :], axis=(1,), dtype=mixed_dtype)



@nki.jit
def flash_paged_attention(query, key, value,
                          key_cache, value_cache,
                          block_tables, mask,
                          softmax_scale=None,
                          mixed_precision=True,
                          config=None,
                          return_softmax_max_sum=False):
  """
  Flash PagedAttention Forward Kernel.
    - PagedAttention Paper: https://arxiv.org/abs/2309.06180
    - Chunked Prefill Paper: https://arxiv.org/abs/2403.02310

  IO tensor layouts:
    - query: shape   (1, n_heads, d, seq_q)
    - key:   shape   (1, n_kv_heads, d, seq_k)
    - value: shape   (1, n_kv_heads, d, seq_v) if config.should_transpose_v else (1, n_kv_heads, seq_v, d)
    - key_cache: (num_blocks, block_size, n_kv_heads, d)
    - value_cache: (num_blocks, block_size, n_kv_heads, d)
    - block_tables: (num_active_blocks, )
    - mask: (seq_q, num_active_blocks * block_size)
    - o: shape (1, n_heads, seq_q, d)
    - l_m: shape (1, n_heads, seq_q, 2)
    
    - This kernel requires seq_k == seq_v
    - We use continuous batching by default, so the batch dimension is always 1, and different
      requests are concatenated along sequence dimension.
    - We use paged cache blocks (key_cache, value_cache) to store KV cache.

  IO tensor dtypes:
    - This kernel assumes all IO tensors have the same dtype except for block_tables (int32) and mask (int32)
    - If mixed_percision is True, then all Tensor Engine operation will be performed in
      bfloat16 and accumulation will be performed in float32. Otherwise the intermediates
      will be in the same type as the inputs.

  Compile-time Constants:
    - softmax_scale: scaling for softmax, is None, default is `1.0/(d**0.5)`
    - mixed_precision: flag to set non-matmul ops in fp32 precision, defualt is set to `true`, if false, we use same precision as input types
    - config: Instance of dataclass :class:`nki.kernels.attention.FlashConfig` with Performance config parameters for flash attention with default values
        seq_tile_size: `default=2048`, size of the kv tile size for attention computation reduction
        training: bool to indicate training vs inference `default=True`

  GQA support Notes:
    the spmd kernel for launching kernel should be on kv_heads instead of nheads

  Example usage:
    MHA: q: [b, h, d, s], k: [b, h, d, s], v: [b, h, s, d]
      usage: `flash_fwd[b, h](q, k, v, ...)`
    GQA: q: [b, h, d, s], k: [b, kv_h, d, s], v: [b, kv_h, s, d]
      usage: `flash_fwd[b, kv_h](q, k, v, ...)`
  """
  config = config or FlashConfig()
  B_F_SIZE = 512
  B_P_SIZE = 128
  b, h, d, seqlen_q  = query.shape
  B_D_SIZE = d
  num_blocks, block_size, k_h, _ = key_cache.shape
  assert tuple(key_cache.shape) == (num_blocks, block_size, k_h, d), 'Input shape mismatch!'
  assert tuple(value_cache.shape) == (num_blocks, block_size, k_h, d), 'Input shape mismatch!'
  assert d <= 128, f" we do not support head_dim > 128, got head dim {d}"
  kernel_dtype = nl.bfloat16 if mixed_precision else query.dtype
  acc_type = np.dtype(np.float32) if mixed_precision else kernel_dtype

  o = nl.ndarray((b, h, seqlen_q, d), dtype=query.dtype, buffer=nl.shared_hbm)
  l_m = None
  if return_softmax_max_sum:
    l_m = nl.ndarray((b, h, seqlen_q, 2), dtype=acc_type, buffer=nl.shared_hbm)

  assert nl.program_ndim() == 2,\
    f'Expect spmd grid with 2 dimensions, got {nl.program_ndim()} instead!'
  batch_id = nl.program_id(axis=0)
  head_id = nl.program_id(axis=1)

  softmax_scale = softmax_scale or (1.0 / (d ** 0.5))

  n_tile_q = seqlen_q // B_P_SIZE # since q will be loaded on tensor engine

  LARGE_TILE_SZ = config.seq_tile_size
  num_active_blocks, = block_tables.shape
  context_kv_len = num_active_blocks * block_size
  assert config.seq_tile_size >= 512, f" seq tile_size {config.seq_tile_size} cannot be less than 512"
  assert context_kv_len % LARGE_TILE_SZ == 0, f"Need context_kv_len to be divisible by {LARGE_TILE_SZ} but got {context_kv_len}"
  assert LARGE_TILE_SZ % B_P_SIZE == 0, f"Need LARGE_TILE_SZ ({LARGE_TILE_SZ}) to be divisible by B_P_SIZE ({B_P_SIZE})"
  assert B_P_SIZE % block_size == 0, f"Need B_P_SIZE ({B_P_SIZE}) to be divisible by block_size ({block_size})"
  num_large_k_tile = context_kv_len // LARGE_TILE_SZ
  num_blocks_per_large_tile = LARGE_TILE_SZ // block_size

  # inference flag
  # assert (not config.training), "flash_paged_attention supports inference only."

  block_tables_sbuf = nl.full((par_dim(B_P_SIZE), num_large_k_tile), 0, dtype=np.int32, buffer=nl.sbuf)
  for j in nl.affine_range(num_large_k_tile):
    i_p = nl.arange(num_blocks_per_large_tile)[:, None]
    block_tables_sbuf[i_p, j] = nl.load(
      block_tables[j*num_blocks_per_large_tile + i_p], dtype=np.int32)

  q_h_per_k_h = h // k_h
  # =============== Global Flash Attention accumulators ====================== #
  o_buffer = nl.zeros((n_tile_q, q_h_per_k_h, par_dim(B_P_SIZE), d), dtype=acc_type,
                      buffer=nl.sbuf, lazy_initialization=True)
  l_buffer = nl.zeros((par_dim(B_P_SIZE), n_tile_q, q_h_per_k_h), dtype=acc_type,
                      buffer=nl.sbuf, lazy_initialization=True)
  m_buffer = nl.zeros((n_tile_q, q_h_per_k_h, par_dim(B_P_SIZE), 1), dtype=acc_type,
                      buffer=nl.sbuf, lazy_initialization=True)
  # =============== Global Flash Attention accumulators END ================== #

  for j in nl.sequential_range(0, num_large_k_tile):
    cur_k_tile = nl.ndarray((par_dim(B_D_SIZE), LARGE_TILE_SZ), dtype=kernel_dtype)
    cur_v_tile = nl.ndarray((LARGE_TILE_SZ // B_P_SIZE, par_dim(B_P_SIZE), B_D_SIZE), dtype=kernel_dtype)

    for k_i in nl.affine_range(num_blocks_per_large_tile):
      loaded = nl.load(key_cache[block_tables_sbuf[k_i, j], :, head_id, :])
      cur_k_tile[:, nl.ds(k_i*block_size, block_size)] = nl.transpose(loaded)

    load_tile_size = B_P_SIZE
    num_blocks_per_partition = load_tile_size // block_size
    for partition_idx in nl.affine_range(LARGE_TILE_SZ // load_tile_size):
      for block_in_partition in nl.affine_range(num_blocks_per_partition):
        v_i = partition_idx * num_blocks_per_partition + block_in_partition
        loaded_v = nl.load(value_cache[block_tables_sbuf[v_i, j], :, head_id, :])
        cur_v_tile[partition_idx, nl.ds(block_in_partition * block_size, block_size), :] = loaded_v

    cur_mask = nl.ndarray((par_dim(B_P_SIZE), LARGE_TILE_SZ), dtype=mask.dtype)
    for m_i in nl.affine_range(LARGE_TILE_SZ // B_F_SIZE):
      cur_mask[:, nl.ds(m_i*B_F_SIZE, B_F_SIZE)] = nl.load(mask[:, nl.ds(j*LARGE_TILE_SZ+m_i*B_F_SIZE, B_F_SIZE)])

    for i_q_h in nl.affine_range(q_h_per_k_h):
      for i in nl.affine_range(n_tile_q):
        q_tile = nl.ndarray((B_D_SIZE, B_P_SIZE),dtype=kernel_dtype)
        q_hbm_tile = query[batch_id, head_id * q_h_per_k_h + i_q_h]
        q_sbuf_tile = nl.load(q_hbm_tile[:, nl.ds(i * B_P_SIZE, B_P_SIZE)],
                              dtype=kernel_dtype) # load (d, 128) tile in SBUF
        q_tile[:, :] = q_sbuf_tile * softmax_scale

        _flash_attention_core(q_local_tile=q_tile, k=cur_k_tile, v=cur_v_tile,
                              q_h_per_k_h=q_h_per_k_h, seqlen_q=seqlen_q, nheads=h,
                              o_buffer=o_buffer[i, i_q_h], l_buffer=l_buffer[:, i, i_q_h], m_buffer=m_buffer[i, i_q_h],
                              batch_id=batch_id, head_id=head_id,
                              gqa_head_idx=i_q_h, q_tile_idx=i, local_k_large_tile_idx=j,
                              kernel_dtype=kernel_dtype, acc_type=acc_type,
                              flash_config=config, use_causal_mask=False, continuous_batching_mask=cur_mask,
                              initialize=j == 0,
                              B_P_SIZE=B_P_SIZE, B_F_SIZE=B_F_SIZE, B_D_SIZE=B_D_SIZE,
                              dropout_p=0.0, dropout_p_tensor=None,
                              seed_tensor=None, logit_bias_tile=None)

  # compute attention between input query, key and value
  if key is not None and value is not None:
    B_F_SIZE = seqlen_q
    LARGE_TILE_SZ = seqlen_q
    active_config = FlashConfig(seq_tile_size=LARGE_TILE_SZ, training=False, should_transpose_v=config.should_transpose_v)

    cur_k_tile = nl.ndarray((par_dim(B_D_SIZE), LARGE_TILE_SZ), dtype=kernel_dtype)
    cur_v_tile = nl.ndarray((LARGE_TILE_SZ // B_P_SIZE, par_dim(B_P_SIZE), B_D_SIZE), dtype=kernel_dtype)

    cur_k_tile[:, :] = nl.load(key[batch_id, head_id, :, :])

    load_tile_size = B_P_SIZE
    v_hbm_tile = value[batch_id, head_id]
    for v_i in nl.affine_range(LARGE_TILE_SZ // load_tile_size):
      load_v_tile(v_hbm_tile=v_hbm_tile, cur_v_tile=cur_v_tile, j=0, v_i=v_i,
                  config=active_config)

    cur_mask = nl.ndarray((par_dim(B_P_SIZE), B_F_SIZE), dtype=mask.dtype)
    cur_mask[:, :] = nl.load(mask[:, nl.ds(context_kv_len, B_F_SIZE)])

    for i_q_h in nl.affine_range(q_h_per_k_h):
      for i in nl.affine_range(n_tile_q):
        q_tile = nl.ndarray((B_D_SIZE, B_P_SIZE), dtype=kernel_dtype)
        q_hbm_tile = query[batch_id, head_id * q_h_per_k_h + i_q_h]
        q_sbuf_tile = nl.load(q_hbm_tile[:, nl.ds(i * B_P_SIZE, B_P_SIZE)],
                              dtype=kernel_dtype) # load (d, 128) tile in SBUF
        q_tile[:, :] = q_sbuf_tile * softmax_scale
        _flash_attention_core(q_local_tile=q_tile, k=cur_k_tile, v=cur_v_tile,
                              q_h_per_k_h=q_h_per_k_h, seqlen_q=seqlen_q, nheads=h,
                              o_buffer=o_buffer[i, i_q_h], l_buffer=l_buffer[:, i, i_q_h], m_buffer=m_buffer[i, i_q_h],
                              batch_id=batch_id, head_id=head_id,
                              gqa_head_idx=i_q_h, q_tile_idx=i, local_k_large_tile_idx=0,
                              kernel_dtype=kernel_dtype, acc_type=acc_type,
                              flash_config=active_config, use_causal_mask=False, continuous_batching_mask=cur_mask,
                              initialize=False,
                              B_P_SIZE=B_P_SIZE, B_F_SIZE=B_F_SIZE, B_D_SIZE=B_D_SIZE,
                              dropout_p=0.0, dropout_p_tensor=None,
                              seed_tensor=None, logit_bias_tile=None)


  # -------- write output to buffer on HBM ------------ #
  for i_q_h in nl.affine_range(q_h_per_k_h):
    for i in nl.affine_range(n_tile_q):
      out = nl.multiply(o_buffer[i, i_q_h, :, :],
                        nl.exp(m_buffer[i, i_q_h, :, :] - l_buffer[:, i, i_q_h]),
                        dtype=kernel_dtype)

      nl.store(o[batch_id, head_id * q_h_per_k_h + i_q_h,
                 nl.ds(i*B_P_SIZE, B_P_SIZE), :], out)
      # maximum and summation statistics
      if return_softmax_max_sum:
        nl.store(l_m[batch_id, head_id * q_h_per_k_h + i_q_h, nl.ds(i*B_P_SIZE, B_P_SIZE), 0], m_buffer[i, i_q_h, :, :])
        nl.store(l_m[batch_id, head_id * q_h_per_k_h + i_q_h, nl.ds(i*B_P_SIZE, B_P_SIZE), 1], l_buffer[:, i, i_q_h])

  if return_softmax_max_sum:
    return o, l_m
  return o


def flash_attn_with_kvcache(
    q,
    k_cache,
    v_cache,
    softmax_scale,
    causal,
    block_table,
    cache_seqlens,
    softcap,
    window_size,
):
  from torch_neuronx import nki_jit

  # Decorate the NKI kernel for PyTorch tracing
  nki_tensor_add_kernel_torch = nki_jit(flash_paged_attention)
  nki_tensor_add_kernel_torch[grid_x, grid_y](a_input, b_input, c_output)
  return c_output
  pass
