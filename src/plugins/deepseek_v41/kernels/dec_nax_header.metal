#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace mpp::tensor_ops;

// The matrix units' tile op on mxfp8 rows: a simdgroup contracts NCH chunks of 64 k of 32 weight rows against up to 8
// activation rows. Lane (fm, f4) of the fragment layout holds weight rows fm + 8 j (j < 4) and, of a chunk, the 16 k
// from 16 f4 on: four tiles of four k, whose contraction slots the activation fragment follows. `acc` is the 16-float
// C fragment: element i < 4 is row fm, column 4 f4 + i; i - 8 is the same 16 columns on.
struct dsv41_nax_rows {
  const device uint4* w[4];
  const device uchar* s[4];
};

struct dsv41_nax_first {
  uint4 q[4];
  uint e[4];
};

// The first chunk's weights, loaded ahead of whatever the caller has to compute before the contraction.
METAL_FUNC dsv41_nax_first dsv41_nax_head(const thread dsv41_nax_rows& r) {
  dsv41_nax_first f;
  for (int j = 0; j < 4; ++j) { f.q[j] = r.w[j][0]; f.e[j] = uint(r.s[j][0]); }
  return f;
}

// Chunk c of the lane's weights is at `w[j][WSTEP c]` (uint4), its scale at `s[j][SSTEP c]`, its 16 activations (two uint4
// of bf16) at `x[XSTEP c]` and the next one.
template <int NCH, int WSTEP, int SSTEP, int XSTEP, typename XP>
METAL_FUNC void dsv41_nax_mxfp8(thread float (&acc)[16], const thread dsv41_nax_rows& r, const thread dsv41_nax_first& f, XP x, bool live) {
  constexpr auto desc = matmul2d_descriptor(16, 32, 16, false, true, true, matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<desc, metal::execution_simdgroup> op;
  auto ct_a = op.template get_left_input_cooperative_tensor<bfloat, bfloat, float>();
  auto ct_b = op.template get_right_input_cooperative_tensor<bfloat, bfloat, float>();
  auto ct_c = op.template get_destination_cooperative_tensor<metal::remove_addrspace_t<decltype(ct_a)>, metal::remove_addrspace_t<decltype(ct_b)>, float>();
  for (short i = 0; i < 16; ++i) ct_c[i] = acc[i];
  for (short i = 0; i < 8; ++i) ct_a[i] = bfloat(0.0f);
  uint4 q[4], xv[2];
  uint e[4];
  for (int j = 0; j < 4; ++j) { q[j] = f.q[j]; e[j] = f.e[j]; }
  xv[0] = x[0];
  xv[1] = x[1];
  for (int c = 0; c < NCH; ++c) {
    uint4 qn[4], xn[2];
    uint en[4];
    if (c + 1 < NCH) {
      for (int j = 0; j < 4; ++j) { qn[j] = r.w[j][WSTEP * (c + 1)]; en[j] = uint(r.s[j][SSTEP * (c + 1)]); }
      xn[0] = x[XSTEP * (c + 1)];
      xn[1] = x[XSTEP * (c + 1) + 1];
    }
    for (int t = 0; t < 4; ++t) {
      if (live) {
        const uint4 xx = xv[t >> 1];
        const uint2 xw = (t & 1) ? uint2(xx.z, xx.w) : uint2(xx.x, xx.y);
        ct_a[0] = as_type<bfloat>(ushort(xw.x & 0xFFFFu));
        ct_a[1] = as_type<bfloat>(ushort(xw.x >> 16));
        ct_a[2] = as_type<bfloat>(ushort(xw.y & 0xFFFFu));
        ct_a[3] = as_type<bfloat>(ushort(xw.y >> 16));
      }
      for (int j = 0; j < 4; ++j) {
        const uint w = t == 0 ? q[j].x : t == 1 ? q[j].y : t == 2 ? q[j].z : q[j].w;
        // e8m0 as 2^(e - 127), times the 2^8 the e4m3 bit placement leaves out; the e4m3 pairs by the half placement, then
        // the scale and the bf16 truncation (exact: three mantissa bits).
        const float sc = as_type<float>(e[j] == 0u ? 0x400000u : (e[j] << 23)) * 256.0f;
        const float2 lo = float2(as_type<half2>(((w << 7) & 0x3F803F80u) | ((w << 8) & 0x80008000u))) * sc;
        const float2 hi = float2(as_type<half2>(((w >> 1) & 0x3F803F80u) | (w & 0x80008000u))) * sc;
        const uint p01 = (as_type<uint>(lo.x) >> 16) | (as_type<uint>(hi.x) & 0xFFFF0000u);
        const uint p23 = (as_type<uint>(lo.y) >> 16) | (as_type<uint>(hi.y) & 0xFFFF0000u);
        ct_b[4 * j + 0] = as_type<bfloat>(ushort(p01 & 0xFFFFu));
        ct_b[4 * j + 1] = as_type<bfloat>(ushort(p01 >> 16));
        ct_b[4 * j + 2] = as_type<bfloat>(ushort(p23 & 0xFFFFu));
        ct_b[4 * j + 3] = as_type<bfloat>(ushort(p23 >> 16));
      }
      op.run(ct_a, ct_b, ct_c);
    }
    if (c + 1 < NCH) {
      for (int j = 0; j < 4; ++j) { q[j] = qn[j]; e[j] = en[j]; }
      xv[0] = xn[0];
      xv[1] = xn[1];
    }
  }
  for (short i = 0; i < 16; ++i) acc[i] = ct_c[i];
}
