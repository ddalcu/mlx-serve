// sdpa_rows_1p.metal over a rope-once MLA cache (sdk/blocks/attention/sdpa_rows.zig `attendRope1`): the same
// statements, in the same order, of MLX 0.32.5's `sdpa_vector` (mlx/backend/metal/kernels/sdpa_vector.h, Copyright ©
// 2024 Apple Inc., MIT). Only the key and value reads differ: a token's key row kin [1, 1, T, H * NOPE + (D - NOPE)]
// holds every head's NOPE non-positional key and then the one roped key all heads share; a value row vin
// [1, 1, T, H * V] holds every head's value. Key element d of head h is kin[h * NOPE + d] below NOPE and
// kin[H * NOPE + d - NOPE] past it: the element the dense [1, H, T, D] key (nope | rope broadcast) holds at d, so every
// score and output word is the dense call's. Template constants T, D, V, NOPE; p = (R0, N0).
constexpr int BN = 32;
constexpr int BD = 32;
constexpr int qk_per_thread = D / BD;
constexpr int v_per_thread = V / BD;

const uint3 tid = threadgroup_position_in_grid;
const uint3 tpg = threadgroups_per_grid;
const uint simd_gid = simdgroup_index_in_threadgroup;
const uint simd_lid = thread_index_in_simdgroup;

const int n_heads = qin_shape[1];
const int N = p[1] + int(tid.y);
const size_t k_seq_stride = size_t(kin_strides[2]);
const size_t v_seq_stride = size_t(vin_strides[2]);
const float scale = sc[0];

int inner_k_stride = BN * int(k_seq_stride);
int inner_v_stride = BN * int(v_seq_stride);

typedef float U;

thread U q[qk_per_thread];
thread U k[qk_per_thread];
thread U o[v_per_thread];

threadgroup U outputs[BN * BD];
threadgroup U max_scores[BN];
threadgroup U sum_exp_scores[BN];

// Adjust positions
const int q_batch_head_idx = tid.x;
const int q_seq_idx = tid.y;
const int head = q_batch_head_idx;
const int o_offset = q_batch_head_idx * tpg.y + q_seq_idx;
const device T* queries = qin + q_batch_head_idx * qin_strides[1] + (p[0] + q_seq_idx) * qin_strides[2] +
    simd_lid * qk_per_thread;
// The token row's start: the lane's elements are addressed through `nope_at` and `rope_at` below.
const device T* krow = kin + simd_gid * k_seq_stride;
const int d0 = int(simd_lid) * qk_per_thread;
const int nope_at = head * NOPE;
const int rope_at = n_heads * NOPE - NOPE;
const device T* values = vin + simd_gid * v_seq_stride + head * V + simd_lid * v_per_thread;

device T* out = oout + o_offset * V + simd_gid * v_per_thread;

// Restrict the keys to the window
int lo = 0;
int hi = N;

// Read the query and 0 the output accumulator
for (int i = 0; i < qk_per_thread; i++) {
  q[i] = static_cast<U>(scale) * queries[i];
}
for (int i = 0; i < v_per_thread; i++) {
  o[i] = 0;
}

U max_score = Limits<U>::finite_min;
U sum_exp_score = 0;

// For each key in valid range
for (int i = lo + simd_gid; i < hi; i += BN) {
  bool use_key = true;
  if (use_key) {
    // Read the key
    for (int j = 0; j < qk_per_thread; j++) {
      const int d = d0 + j;
      k[j] = d < NOPE ? krow[nope_at + d] : krow[rope_at + d];
    }

    // Compute the i-th score
    U score = 0;
    for (int j = 0; j < qk_per_thread; j++) {
      score += q[j] * k[j];
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
  krow += inner_k_stride;
  values += inner_v_stride;
}

// Each thread has a partial part of the output so we need to combine them.

// First let's communicate the max and sum_exp
if (simd_lid == 0) {
  max_scores[simd_gid] = max_score;
  sum_exp_scores[simd_gid] = sum_exp_score;
}
threadgroup_barrier(mem_flags::mem_threadgroup);
max_score = max_scores[simd_lid];
U new_max = simd_max(max_score);
U factor = fast::exp(max_score - new_max);
sum_exp_score = simd_sum(sum_exp_scores[simd_lid] * factor);

// Now we need to aggregate all the outputs
for (int i = 0; i < v_per_thread; i++) {
  outputs[simd_lid * BD + simd_gid] = o[i];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  o[i] = simd_sum(outputs[simd_gid * BD + simd_lid] * factor);
  o[i] = sum_exp_score == 0 ? o[i] : (o[i] / sum_exp_score);
  threadgroup_barrier(mem_flags::mem_threadgroup);
}

// And write the output
if (simd_lid == 0) {
  for (int i = 0; i < v_per_thread; i++) {
    out[i] = static_cast<T>(o[i]);
  }
}
