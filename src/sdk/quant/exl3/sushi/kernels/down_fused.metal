threadgroup float partial[4 * 16 * uint(OTPT)];
threadgroup half prepared[uint(IDIM)];
const uint ot = uint(threadgroup_position_in_grid.x) * uint(OTPT);
uint slot = uint(threadgroup_position_in_grid.y);
uint split = uint(threadgroup_position_in_grid.z);
uint sg = uint(simdgroup_index_in_threadgroup);
uint lane = uint(thread_index_in_simdgroup);
uint lid = uint(thread_index_in_threadgroup);
constexpr uint TILE = 16u;
constexpr uint N = uint(NHW);
constexpr uint PACKED_HW = N;
constexpr uint PACKED_W = N / 2u;
constexpr uint IT = uint(IDIM) / TILE;
constexpr uint OT = uint(ODIM) / TILE;
constexpr uint SGS = 4u;
constexpr uint SPLITS = 1u;
const uint eid = uint(slots[slot]);
const float sc = 0.08838834764831845f;
const uint nblocks = uint(IDIM) / 128u;
for (uint block = sg; block < nblocks; block += SGS) {
  const uint base = block * 128u;
  const size_t sb = (size_t)eid * (size_t)(IDIM) + base;
  float4 v = float4(0.0f, 0.0f, 0.0f, 0.0f);
  for (uint sp = 0u; sp < uint(NSPLIT); sp++) {
    const size_t xb = ((size_t)slot * uint(NSPLIT) + sp) * (size_t)(IDIM) + base;
    v += float4(float(ig[xb + lane]), float(ig[xb + lane + 32u]), float(ig[xb + lane + 64u]), float(ig[xb + lane + 96u]));
  }
  for (ushort bit = 1u; bit <= 16u; bit <<= 1u) {
    const float p0 = simd_shuffle_xor(v.x, bit);
    const float p1 = simd_shuffle_xor(v.y, bit);
    const float p2 = simd_shuffle_xor(v.z, bit);
    const float p3 = simd_shuffle_xor(v.w, bit);
    const bool lower = (lane & bit) == 0u;
    v.x = lower ? v.x + p0 : p0 - v.x;
    v.y = lower ? v.y + p1 : p1 - v.y;
    v.z = lower ? v.z + p2 : p2 - v.z;
    v.w = lower ? v.w + p3 : p3 - v.w;
  }
  float s0 = v.x + v.y;
  float s1 = v.x - v.y;
  float s2 = v.z + v.w;
  float s3 = v.z - v.w;
  const float g0 = (s0 + s2) * sc * float(svhg[sb + lane]);
  const float g1 = (s1 + s3) * sc * float(svhg[sb + lane + 32u]);
  const float g2 = (s0 - s2) * sc * float(svhg[sb + lane + 64u]);
  const float g3 = (s1 - s3) * sc * float(svhg[sb + lane + 96u]);
  v = float4(0.0f, 0.0f, 0.0f, 0.0f);
  for (uint sp = 0u; sp < uint(NSPLIT); sp++) {
    const size_t xbu = ((size_t)slot * uint(NSPLIT) + sp) * (size_t)(IDIM) + base;
    v += float4(float(iu[xbu + lane]), float(iu[xbu + lane + 32u]), float(iu[xbu + lane + 64u]), float(iu[xbu + lane + 96u]));
  }
  for (ushort bit = 1u; bit <= 16u; bit <<= 1u) {
    const float p0 = simd_shuffle_xor(v.x, bit);
    const float p1 = simd_shuffle_xor(v.y, bit);
    const float p2 = simd_shuffle_xor(v.z, bit);
    const float p3 = simd_shuffle_xor(v.w, bit);
    const bool lower = (lane & bit) == 0u;
    v.x = lower ? v.x + p0 : p0 - v.x;
    v.y = lower ? v.y + p1 : p1 - v.y;
    v.z = lower ? v.z + p2 : p2 - v.z;
    v.w = lower ? v.w + p3 : p3 - v.w;
  }
  s0 = v.x + v.y;
  s1 = v.x - v.y;
  s2 = v.z + v.w;
  s3 = v.z - v.w;
  const float u0 = (s0 + s2) * sc * float(svhu[sb + lane]);
  const float u1 = (s1 + s3) * sc * float(svhu[sb + lane + 32u]);
  const float u2 = (s0 - s2) * sc * float(svhu[sb + lane + 64u]);
  const float u3 = (s1 - s3) * sc * float(svhu[sb + lane + 96u]);
  const float ysig0 = 1 / (1 + exp(abs(g0)));
  const float ysig1 = 1 / (1 + exp(abs(g1)));
  const float ysig2 = 1 / (1 + exp(abs(g2)));
  const float ysig3 = 1 / (1 + exp(abs(g3)));
  const float sig0 = (g0 < 0) ? ysig0 : 1 - ysig0;
  const float sig1 = (g1 < 0) ? ysig1 : 1 - ysig1;
  const float sig2 = (g2 < 0) ? ysig2 : 1 - ysig2;
  const float sig3 = (g3 < 0) ? ysig3 : 1 - ysig3;
  const float silu0 = g0 * sig0;
  const float silu1 = g1 * sig1;
  const float silu2 = g2 * sig2;
  const float silu3 = g3 * sig3;
  const float h0 = silu0 * u0;
  const float h1 = silu1 * u1;
  const float h2 = silu2 * u2;
  const float h3 = silu3 * u3;
  v = float4(h0 * float(suhd[sb + lane]), h1 * float(suhd[sb + lane + 32u]), h2 * float(suhd[sb + lane + 64u]), h3 * float(suhd[sb + lane + 96u]));
  for (ushort bit = 1u; bit <= 16u; bit <<= 1u) {
    const float p0 = simd_shuffle_xor(v.x, bit);
    const float p1 = simd_shuffle_xor(v.y, bit);
    const float p2 = simd_shuffle_xor(v.z, bit);
    const float p3 = simd_shuffle_xor(v.w, bit);
    const bool lower = (lane & bit) == 0u;
    v.x = lower ? v.x + p0 : p0 - v.x;
    v.y = lower ? v.y + p1 : p1 - v.y;
    v.z = lower ? v.z + p2 : p2 - v.z;
    v.w = lower ? v.w + p3 : p3 - v.w;
  }
  s0 = v.x + v.y;
  s1 = v.x - v.y;
  s2 = v.z + v.w;
  s3 = v.z - v.w;
  prepared[base + lane] = half((s0 + s2) * sc);
  prepared[base + lane + 32u] = half((s1 + s3) * sc);
  prepared[base + lane + 64u] = half((s0 - s2) * sc);
  prepared[base + lane + 96u] = half((s1 - s3) * sc);
}
threadgroup_barrier(mem_flags::mem_threadgroup);
const uint tiles_per_split = (IT + SPLITS - 1u) / SPLITS;
const uint tk0 = split * tiles_per_split;
const uint tk1 = min(tk0 + tiles_per_split, IT);
const uint prow = (lane & 3u) * 2u;
const uint pcol = lane >> 2u;
const uint row0 = prow;
const uint row1 = prow + 1u;
const uint row2 = prow + 8u;
const uint row3 = prow + 9u;
float acc[OTPT][8] = {};
const device uint* trellis_e = (const device uint*)(trellis + ((size_t)eid * (size_t)IT * (size_t)OT) * PACKED_HW);
if (FUNNEL) {
  // Both k-tiles' words load before either decodes; a k range is whole H128
  // blocks (8 tiles), so the second k-tile of an iteration is always in range.
  const device uint *wp = trellis_e + ((size_t)(tk0 + sg) * (size_t)OT + ot) * PACKED_W;
  const threadgroup half *pp = prepared + (tk0 + sg) * TILE;
  const uint sh[8] = {exl3_lane_sh(N, 0u), exl3_lane_sh(N, 1u), exl3_lane_sh(N, 2u), exl3_lane_sh(N, 3u), exl3_lane_sh(N, 4u), exl3_lane_sh(N, 5u), exl3_lane_sh(N, 6u), exl3_lane_sh(N, 7u)};
  for (uint tk = tk0 + sg; tk < tk1; tk += 2u * SGS) {
    exl3_lane_bits<N> merged[2][OTPT];
    for (uint u = 0u; u < 2u; u++) {
      for (uint o = 0u; o < uint(OTPT); o++) {
        const device uint *words = wp + u * SGS * OT * PACKED_W + o * PACKED_W;
        merged[u][o] = exl3_lane<N>(words, lane);
      }
    }
    for (uint u = 0u; u < 2u; u++) {
      const float in0 = float(pp[u * SGS * TILE + row0]);
      const float in1 = float(pp[u * SGS * TILE + row1]);
      const float in2 = float(pp[u * SGS * TILE + row2]);
      const float in3 = float(pp[u * SGS * TILE + row3]);
      const float ins[8] = {in0, in1, in2, in3, in0, in1, in2, in3};
      for (uint o = 0u; o < uint(OTPT); o++) {
        for (uint p = 0u; p < 4u; p++) {
          const uint2 cw = uint2(exl3_lane_word(merged[u][o], sh[p * 2u]), exl3_lane_word(merged[u][o], sh[p * 2u + 1u])) & uint2(0xffffu);
          const float2 w = exl3_decode2(cw);
          acc[o][p * 2u] = fma(ins[p * 2u], w.x, acc[o][p * 2u]);
          acc[o][p * 2u + 1u] = fma(ins[p * 2u + 1u], w.y, acc[o][p * 2u + 1u]);
        }
      }
    }
    wp += 2u * SGS * OT * PACKED_W;
    pp += 2u * SGS * TILE;
  }
} else {
  uint w0[4];
  uint w1[4];
  uint shv[4];
  uint frv[4];
  for (uint p = 0u; p < 4u; p++) {
    const exl3_win wv = exl3_pair_window(lane * 8u + p * 2u, N);
    w0[p] = wv.i0;
    w1[p] = wv.i1;
    shv[p] = wv.sh;
    frv[p] = wv.fresh;
  }
  for (uint tk = tk0 + sg; tk < tk1; tk += SGS) {
    const device uint* words = trellis_e + ((size_t)tk * (size_t)OT + ot) * PACKED_W;
    const float in0 = float(prepared[tk * TILE + row0]);
    const float in1 = float(prepared[tk * TILE + row1]);
    const float in2 = float(prepared[tk * TILE + row2]);
    const float in3 = float(prepared[tk * TILE + row3]);
    const float ins[8] = {in0, in1, in2, in3, in0, in1, in2, in3};
    for (uint p = 0u; p < 4u; p++) {
      const ulong merged = ((ulong)words[w0[p]] << 32) | (ulong)words[w1[p]];
      const uint funnel = uint(merged >> shv[p]);
      const uint2 cw = uint2((funnel >> frv[p]) & 0xffffu, funnel & 0xffffu);
      const float2 w = exl3_decode2(cw);
      acc[0][p * 2u] = fma(ins[p * 2u], w.x, acc[0][p * 2u]);
      acc[0][p * 2u + 1u] = fma(ins[p * 2u + 1u], w.y, acc[0][p * 2u + 1u]);
    }
  }
}
for (uint o = 0u; o < uint(OTPT); o++) {
  float clo = (acc[o][0] + acc[o][1]) + (acc[o][2] + acc[o][3]);
  float chi = (acc[o][4] + acc[o][5]) + (acc[o][6] + acc[o][7]);
  clo += simd_shuffle_xor(clo, 1u);
  chi += simd_shuffle_xor(chi, 1u);
  clo += simd_shuffle_xor(clo, 2u);
  chi += simd_shuffle_xor(chi, 2u);
  if ((lane & 3u) == 0u) {
    partial[(o * SGS + sg) * 16u + pcol] = clo;
    partial[(o * SGS + sg) * 16u + pcol + 8u] = chi;
  }
}
threadgroup_barrier(mem_flags::mem_threadgroup);
if (lid < 16u * uint(OTPT)) {
  const uint o = lid >> 4u;
  float sum = 0.0f;
  for (uint g = 0u; g < SGS; g++) {
    sum += partial[(o * SGS + g) * 16u + (lid & 15u)];
  }
  y[(size_t)slot * (size_t)(ODIM) + (ot + o) * TILE + (lid & 15u)] = half(exl3_out(sum));
}
threadgroup_barrier(mem_flags::mem_threadgroup);