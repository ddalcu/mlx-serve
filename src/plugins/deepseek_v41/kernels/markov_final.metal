// The DSpark draft's Markov chain, after its last step (spec/markov.zig): threadgroup k (512 threads) gives step k's token,
// MLX's argmax of its row from its threadgroups' pairs, and with CONF the confidence head's logit at position k.
//   parts uint32 [B NT 2]: every step's (value bits, index) pair a threadgroup (the last step's pout)
//   t1    int32 [1]: the token before step 0
//   hout  H [B, D]: the last stage's collapse; emb E [V, R]: markov_embed; cw f32 [D + R]: the confidence head
//   toks  int32 [B]; conf f32 [B] with CONF (else unwritten)
// The logit is MLX's dot_product_float32_it32_tg512_sg16 over [f32(hout[k]), f32(emb[token before k])] and cw as the op
// chain calls it (a [1, D + R] by [D + R, 1] matmul): the same thread map, float4 products and sums, simdgroup sums and
// threadgroup sum, then all_reduce's 0 + x over its one partial.
const uint lane = thread_index_in_simdgroup, sg = simdgroup_index_in_threadgroup;
const uint tid = thread_index_in_threadgroup, k = threadgroup_position_in_grid.y;
constexpr uint NTC = uint(NT), DC = uint(D), RC = uint(R);
threadgroup float t_v[16];
threadgroup uint t_i[16];
threadgroup uint t_tok;
threadgroup float smem[16];
const uint tok = dsv41_mk_tg_best<512u>(parts + 2u * k * NTC, NTC, tid, lane, sg, t_v, t_i, &t_tok);
if (tid == 0) toks[k] = int(tok);
if (CONF != 0) {
  const uint prev = k == 0u ? uint(t1[0]) : dsv41_mk_tg_best<512u>(parts + 2u * (k - 1u) * NTC, NTC, tid, lane, sg, t_v, t_i, &t_tok);
  const device H* hr = hout + ulong(k) * DC;
  const device E* er = emb + ulong(prev) * RC;
  constexpr int NN = int(DC + RC);
  const int start = int(sg) * 1024 + int(lane) * 4;
  float4 c = 0.0f;
  for (int i = 0; i < 32; i += 4) {
    const int idx = start + i * 32;
    if (idx + 4 <= NN) {
      float4 av;
      for (int j = 0; j < 4; ++j) av[j] = idx + j < int(DC) ? static_cast<float>(hr[idx + j]) : static_cast<float>(er[idx + j - int(DC)]);
      c += av * *reinterpret_cast<const device float4*>(cw + idx);
    } else {
      for (int j = 0; j < 4; ++j) {
        const int nidx = idx + j;
        if (nidx < NN) c[j & 3] += (nidx < int(DC) ? static_cast<float>(hr[nidx]) : static_cast<float>(er[nidx - int(DC)])) * cw[nidx];
      }
    }
  }
  float sum = c[0] + c[1] + c[2] + c[3];
  sum = metal::simd_sum(sum);
  if (lane == 0) smem[sg] = sum;
  threadgroup_barrier(metal::mem_flags::mem_threadgroup);
  if (tid < 16u) {
    sum = smem[tid];
    sum = metal::simd_sum(sum);
    if (tid == 0) conf[k] = sum + 0.0f;
  }
}
