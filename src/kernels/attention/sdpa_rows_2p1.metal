// The exact verify's attention rows on MLX's two-pass vector kernels, pass 1: MLX 0.32.5's `sdpa_vector_2pass_1`
// (mlx/backend/metal/kernels/sdpa_vector.h, Copyright © 2024 Apple Inc., MIT) for several query rows in one launch
// (sdk/blocks/attention/sdpa_rows.zig). Simdgroup j of threadgroup (kv head, batch, block) runs sdpa_vector_2pass_1's
// statements, in its order (the TK-key main loop while i + (TK - 1) blocks < N, then the tail), for query row R0 + j
// over its own N = N0 + j keys: what MLX's one-row call over the first N keys runs (q_len 1, so no causal mask), with
// no mask, sinks or window (their function constants false) and MLX's `blocks` for that call (the template constant
// BLOCKS). Only the addressing of the rows is this file's: the query row r of qin [1, H, S, D] at its strides; the
// partials (T), sums and maxs of row j as MLX lays out a q_len R call's. Template constants T, D, V, BLOCKS, TK;
// p = (R0, N0).
constexpr int BD = 32;
constexpr int qk_per_thread = D / BD;
constexpr int v_per_thread = V / BD;
constexpr int blocks = BLOCKS;

const uint3 tptg = threads_per_threadgroup;
const uint3 tidtg = thread_position_in_threadgroup;
const uint3 tid = threadgroup_position_in_grid;
const uint3 tpg = threadgroups_per_grid;
const uint simd_lid = thread_index_in_simdgroup;

const int N = p[1] + int(tidtg.z);
const size_t k_head_stride = kin_shape[1] == 1 ? size_t(kin_strides[0]) : size_t(kin_strides[1]);
const size_t k_seq_stride = size_t(kin_strides[2]);
const size_t v_head_stride = vin_shape[1] == 1 ? size_t(vin_strides[0]) : size_t(vin_strides[1]);
const size_t v_seq_stride = size_t(vin_strides[2]);
const float scale = sc[0];

typedef float U;

thread U q[qk_per_thread];
thread U o[v_per_thread] = {0};

// Adjust positions
const int kv_head_idx = tid.x;
const int batch_idx = tid.y;
const int block_idx = tid.z;
const int gqa_factor = tptg.y;
const int q_seq_len = tptg.z;
const int q_seq_idx = tidtg.z;
const int q_head_idx = gqa_factor * kv_head_idx + tidtg.y;
const int num_kv_heads = tpg.x;
const int num_q_heads = num_kv_heads * gqa_factor;
const int q_batch_head_idx = (batch_idx * num_q_heads + q_head_idx);
const int o_offset = q_batch_head_idx * q_seq_len + q_seq_idx;

const device T* queries = qin + q_batch_head_idx * qin_strides[1] + (p[0] + q_seq_idx) * qin_strides[2] +
    simd_lid * qk_per_thread;

const int kv_batch_head_idx = batch_idx * num_kv_heads + kv_head_idx;
const device T* keys = kin + kv_batch_head_idx * k_head_stride + block_idx * k_seq_stride +
    simd_lid * qk_per_thread;
const device T* values = vin + kv_batch_head_idx * v_head_stride + block_idx * v_seq_stride +
    simd_lid * v_per_thread;
device T* out = partials + o_offset * blocks * V + block_idx * V + simd_lid * v_per_thread;
device float* sums_p = sums + o_offset * blocks + block_idx;
device float* maxs_p = maxs + o_offset * blocks + block_idx;

// Read the query
for (int i = 0; i < qk_per_thread; i++) {
  q[i] = static_cast<U>(scale) * queries[i];
}

U max_score = Limits<U>::finite_min;
U sum_exp_score = 0;

const int stride_k_tok = blocks * int(k_seq_stride);
const int stride_v_tok = blocks * int(v_seq_stride);

// For each key
int i = block_idx;

// For each key
for (; i + (TK - 1) * blocks < N; i += blocks * TK) {
  // Prefetch: issue all TK K/V loads first so the HW can overlap them.
  T k_reg[TK][qk_per_thread];
  T v_reg[TK][v_per_thread];
  for (int t = 0; t < TK; t++) {
    const device T* kp = keys + t * stride_k_tok;
    for (int j = 0; j < qk_per_thread; j++) {
      k_reg[t][j] = kp[j];
    }
  }
  for (int t = 0; t < TK; t++) {
    const device T* vp = values + t * stride_v_tok;
    for (int j = 0; j < v_per_thread; j++) {
      v_reg[t][j] = vp[j];
    }
  }

  for (int t = 0; t < TK; t++) {
    bool use_key = true;

    if (use_key) {
      U score = 0;
      for (int j = 0; j < qk_per_thread; j++) {
        score += q[j] * static_cast<U>(k_reg[t][j]);
      }
      score = simd_sum(score);

      U new_max = max(max_score, score);
      U factor = fast::exp(max_score - new_max);
      U exp_score = fast::exp(score - new_max);
      max_score = new_max;
      sum_exp_score = sum_exp_score * factor + exp_score;
      for (int j = 0; j < v_per_thread; j++) {
        o[j] = o[j] * factor + exp_score * static_cast<U>(v_reg[t][j]);
      }
    }
  }

  keys += TK * stride_k_tok;
  values += TK * stride_v_tok;
}

// Tail (fewer than TK remaining keys)
for (; i < N; i += blocks) {
  bool use_key = true;

  if (use_key) {
    // Compute the i-th score
    U score = 0;
    for (int j = 0; j < qk_per_thread; j++) {
      score += q[j] * keys[j];
    }
    score = simd_sum(score);

    // Update the accumulators
    U new_max = max(max_score, score);
    U factor = fast::exp(max_score - new_max);
    U exp_score = fast::exp(score - new_max);

    max_score = new_max;
    sum_exp_score = sum_exp_score * factor + exp_score;

    // Update the output accumulator
    for (int j = 0; j < v_per_thread; j++) {
      o[j] = o[j] * factor + exp_score * values[j];
    }
  }

  // Move the pointers to the next kv
  keys += stride_k_tok;
  values += stride_v_tok;
}

// Write the sum and max and outputs
if (simd_lid == 0) {
  sums_p[0] = sum_exp_score;
  maxs_p[0] = max_score;
}

for (int i = 0; i < v_per_thread; i++) {
  out[i] = static_cast<T>(o[i]);
}
