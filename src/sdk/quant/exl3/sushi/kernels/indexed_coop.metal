threadgroup float partial[4 * 256];
uint ot = uint(threadgroup_position_in_grid.x);
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
const uint tiles_per_split = (IT + SPLITS - 1u) / SPLITS;
const uint tk0 = split * tiles_per_split;
const uint tk1 = min(tk0 + tiles_per_split, IT);
const uint prow = (lane & 3u) * 2u;
const uint pcol = lane >> 2u;
uint pos[8];
pos[0] = prow * 16u + pcol;
pos[1] = (prow + 1u) * 16u + pcol;
pos[2] = (prow + 8u) * 16u + pcol;
pos[3] = (prow + 9u) * 16u + pcol;
pos[4] = prow * 16u + pcol + 8u;
pos[5] = (prow + 1u) * 16u + pcol + 8u;
pos[6] = (prow + 8u) * 16u + pcol + 8u;
pos[7] = (prow + 9u) * 16u + pcol + 8u;
const uint row0 = pos[0] >> 4u;
const uint row1 = pos[1] >> 4u;
const uint row2 = pos[2] >> 4u;
const uint row3 = pos[3] >> 4u;
float acc[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
const size_t xb = (size_t)slot * (size_t)(IDIM);
const size_t expert_stride = (size_t)IT * (size_t)OT * (size_t)PACKED_HW;
const device uint* trellis_e = (const device uint*)(trellis + (size_t)eid * expert_stride);
if (N == 64u) {
for (uint tk = tk0 + sg; tk < tk1; tk += SGS) {
  const device uint* words = trellis_e + ((size_t)tk * (size_t)OT + ot) * 32u;
  const ulong merged = ((ulong)words[(lane + 31u) & 31u] << 32) | (ulong)words[lane];
  const float in0 = float(x[xb + tk * TILE + row0]);
  const float in1 = float(x[xb + tk * TILE + row1]);
  const float in2 = float(x[xb + tk * TILE + row2]);
  const float in3 = float(x[xb + tk * TILE + row3]);
  const uint sh[8] = {28u, 24u, 20u, 16u, 12u, 8u, 4u, 0u};
  const float ins[8] = {in0, in1, in2, in3, in0, in1, in2, in3};
  for (uint p = 0u; p < 4u; p++) {
    const uint2 cw = uint2(uint(merged >> sh[p * 2u]), uint(merged >> sh[p * 2u + 1u])) & uint2(0xffffu);
    const float2 w = exl3_decode2(cw);
    acc[p * 2u] = fma(ins[p * 2u], w.x, acc[p * 2u]);
    acc[p * 2u + 1u] = fma(ins[p * 2u + 1u], w.y, acc[p * 2u + 1u]);
  }
}
} else {
  for (uint tk = tk0 + sg; tk < tk1; tk += SGS) {
    const device uint* words = trellis_e + ((size_t)tk * (size_t)OT + ot) * PACKED_W;
    const auto merged = exl3_lane<N>(words, lane);
    const float in0 = float(x[xb + tk * TILE + row0]);
    const float in1 = float(x[xb + tk * TILE + row1]);
    const float in2 = float(x[xb + tk * TILE + row2]);
    const float in3 = float(x[xb + tk * TILE + row3]);
    const float ins[8] = {in0, in1, in2, in3, in0, in1, in2, in3};
    for (uint p = 0u; p < 4u; p++) {
      const uint2 cw = uint2(exl3_lane_word(merged, exl3_lane_sh(N, p * 2u)), exl3_lane_word(merged, exl3_lane_sh(N, p * 2u + 1u))) & uint2(0xffffu);
      const float2 w = exl3_decode2(cw);
      acc[p * 2u] = fma(ins[p * 2u], w.x, acc[p * 2u]);
      acc[p * 2u + 1u] = fma(ins[p * 2u + 1u], w.y, acc[p * 2u + 1u]);
    }
  }
}
for (uint si = 0u; si < 8u; si++) {
  partial[sg * 256u + pos[si]] = acc[si];
}
threadgroup_barrier(mem_flags::mem_threadgroup);
if (lid < 16u) {
  float sum = 0.0f;
  for (uint r = 0u; r < 16u; r++) {
    const uint p = r * 16u + lid;
    for (uint g = 0u; g < SGS; g++) {
      sum += partial[g * 256u + p];
    }
  }
  y[(size_t)slot * (size_t)(ODIM) + ot * TILE + lid] = half(exl3_out(sum));
}
threadgroup_barrier(mem_flags::mem_threadgroup);