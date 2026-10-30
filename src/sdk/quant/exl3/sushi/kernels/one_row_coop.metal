threadgroup float partial[2u * 4u * 256u];
const uint ot = uint(threadgroup_position_in_grid.x) * 2u;
const uint slot = uint(threadgroup_position_in_grid.y);
const uint sg = uint(simdgroup_index_in_threadgroup);
const uint lane = uint(thread_index_in_simdgroup);
const uint lid = uint(thread_index_in_threadgroup);
constexpr uint TILE = 16u;
constexpr uint N = uint(NHW);
constexpr uint PACKED_W = N / 2u;
constexpr uint IT = uint(IDIM) / TILE;
constexpr uint OT = uint(ODIM) / TILE;
constexpr uint SGS = 4u;
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
float acc[2][8] = {};
const size_t expert_stride = (size_t)IT * (size_t)OT * (size_t)N;
const device uint* wp = (const device uint*)(trellis + (size_t)uint(slots[slot]) * expert_stride) + ((size_t)sg * (size_t)OT + ot) * PACKED_W;
const device half* xp = x + (size_t)slot * (size_t)(IDIM) + sg * TILE + (lane & 3u) * 4u;
constexpr uint sh[8] = {exl3_lane_sh(N, 0u), exl3_lane_sh(N, 1u), exl3_lane_sh(N, 2u), exl3_lane_sh(N, 3u), exl3_lane_sh(N, 4u), exl3_lane_sh(N, 5u), exl3_lane_sh(N, 6u), exl3_lane_sh(N, 7u)};
for (uint tk = sg; tk < IT; tk += 2u * SGS) {
  exl3_lane_bits<N> merged[2][2];
  for (uint u = 0u; u < 2u; u++) {
    for (uint o = 0u; o < 2u; o++) {
      merged[u][o] = exl3_lane<N>(wp + u * SGS * OT * PACKED_W + o * PACKED_W, lane);
    }
  }
  for (uint u = 0u; u < 2u; u++) {
    const float4 in4 = float4(*((const device half4 *)(xp + u * SGS * TILE)));
    const float ins[8] = {in4.x, in4.y, in4.z, in4.w, in4.x, in4.y, in4.z, in4.w};
    for (uint o = 0u; o < 2u; o++) {
      for (uint p = 0u; p < 4u; p++) {
        const uint2 cw = uint2(exl3_lane_word(merged[u][o], sh[p * 2u]), exl3_lane_word(merged[u][o], sh[p * 2u + 1u])) & uint2(0xffffu);
        const float2 w = exl3_decode2(cw);
        acc[o][p * 2u] = fma(ins[p * 2u], w.x, acc[o][p * 2u]);
        acc[o][p * 2u + 1u] = fma(ins[p * 2u + 1u], w.y, acc[o][p * 2u + 1u]);
      }
    }
  }
  wp += 2u * SGS * OT * PACKED_W;
  xp += 2u * SGS * TILE;
}
for (uint o = 0u; o < 2u; o++) {
  for (uint si = 0u; si < 8u; si++) {
    partial[(o * 4u + sg) * 256u + pos[si]] = acc[o][si];
  }
}
threadgroup_barrier(mem_flags::mem_threadgroup);
if (lid < 32u) {
  const uint o = lid >> 4u;
  const uint c = lid & 15u;
  float sum = 0.0f;
  for (uint r = 0u; r < 16u; r++) {
    for (uint g = 0u; g < SGS; g++) {
      sum += partial[(o * 4u + g) * 256u + r * 16u + c];
    }
  }
  y[(size_t)slot * (size_t)(ODIM) + (ot + o) * TILE + c] = half(exl3_out(sum));
}