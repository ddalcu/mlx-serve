// dec_wo_a_plain (`MLX_SERVE_DSV41_WOA_PLAIN`): the grouped mxfp8 wo_a over heads already merged and inverse-roped
// (`x` [R, H * 512] f32, `attn_merge`'s output), with no merge and no cross-threadgroup sum. A threadgroup owns TR outputs
// of one wo_a group: simdgroup hh is head hh of the group, a lane 16 of its channels (one 16-byte load a weight row).
// It streams the tile's TR rows NB at a time with the next NB in flight, each weight decoded once for every row, and
// the 8 heads' sums meet in threadgroup memory, added in head order.
constexpr int KW = HPG * D / 4, KG = HPG * D / 32;
static_assert(R <= 6 && HPG == 8 && D == 512 && TR % NB == 0, "the release's wo_a group at up to a verify block");
const uint lane = thread_index_in_simdgroup, hh = simdgroup_index_in_threadgroup;
const int tid = int(thread_position_in_threadgroup.x);
const int tile = int(threadgroup_position_in_grid.x), g = int(threadgroup_position_in_grid.y), h = g * HPG + int(hh);
const uint row0 = uint(g) * OL + uint(tile * TR);
threadgroup float red[HPG * R * TR];

const device uint4* w4 = (const device uint4*)w + row0 * (KW / 4) + hh * (D / 16) + lane;
const device uint8_t* s0 = scales + row0 * KG + hh * (D / 32) + lane / 2;
uint4 cq[NB];
uint ce[NB];
#pragma clang loop unroll(full)
for (int r = 0; r < NB; ++r) { cq[r] = w4[r * (KW / 4)]; ce[r] = uint(s0[r * KG]); }

float4 xr[R][4];
#pragma clang loop unroll(full)
for (int m = 0; m < R; ++m)
  #pragma clang loop unroll(full)
  for (int jj = 0; jj < 4; ++jj) xr[m][jj] = ((const device float4*)(x + (size_t(m) * H + h) * D))[4 * lane + jj];

for (int b = 0; b < TR; b += NB) {
  uint4 nq[NB];
  uint ne[NB];
  const bool more = b + NB < TR;
  if (more)
    #pragma clang loop unroll(full)
    for (int r = 0; r < NB; ++r) { nq[r] = w4[(b + NB + r) * (KW / 4)]; ne[r] = uint(s0[(b + NB + r) * KG]); }
  #pragma clang loop unroll(full)
  for (int r = 0; r < NB; ++r) {
    // The e8m0 scale and the 2^8 the e4m3 reads leave out (fp8_read_header.metal).
    const float2 sc = dsv41_e8m0<FOLD>(ce[r]);
    const float4 w0 = dsv41_e4m3x4<FOLD>(cq[r].x), w1 = dsv41_e4m3x4<FOLD>(cq[r].y), w2 = dsv41_e4m3x4<FOLD>(cq[r].z), w3 = dsv41_e4m3x4<FOLD>(cq[r].w);
    #pragma clang loop unroll(full)
    for (int m = 0; m < R; ++m) {
      const float v = metal::simd_sum(sc.x * ((metal::dot(xr[m][0], w0) + metal::dot(xr[m][1], w1) + metal::dot(xr[m][2], w2) + metal::dot(xr[m][3], w3)) * sc.y));
#if DSV41_WOA_PLAIN_PLANT
      if (lane == uint(r * R + m) % 32u && hh != 3u) red[(int(hh) * R + m) * TR + b + r] = v;
      if (lane == uint(r * R + m) % 32u && hh == 3u) red[(int(hh) * R + m) * TR + b + r] = 0.0f;
#else
      if (lane == uint(r * R + m) % 32u) red[(int(hh) * R + m) * TR + b + r] = v;
#endif
    }
  }
  if (more)
    #pragma clang loop unroll(full)
    for (int r = 0; r < NB; ++r) { cq[r] = nq[r]; ce[r] = ne[r]; }
}
threadgroup_barrier(metal::mem_flags::mem_threadgroup);
for (int it = tid; it < R * TR; it += HPG * 32) {
  const int m = it / TR, n = it % TR;
  float s = 0.0f;
  for (int k = 0; k < HPG; ++k) s += red[(k * R + m) * TR + n];
  out[uint(m) * OW + row0 + uint(n)] = OT(s);
}
