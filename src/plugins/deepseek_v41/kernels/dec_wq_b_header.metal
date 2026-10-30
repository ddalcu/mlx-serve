// e8m0 as 2^(e - 127), times the 2^8 the e4m3 bit placement leaves out.
inline float dsv41_wqb_scale(uint e) {
  return as_type<float>(e == 0u ? 0x400000u : (e << 23)) * 256.0f;
}

inline float dsv41_wqb_sum16(float v) {
  v += metal::simd_shuffle_xor(v, 8);
  v += metal::simd_shuffle_xor(v, 4);
  v += metal::simd_shuffle_xor(v, 2);
  return v + metal::simd_shuffle_xor(v, 1);
}

// The four e4m3 bytes of w as floats (value x 2^-8): bytes 0 and 2, then 1 and 3, each pair's fields shifted into a half2.
inline float4 dsv41_wqb_e4m3x4(uint w) {
  const float2 lo = float2(as_type<half2>(((w << 7) & 0x3F803F80u) | ((w << 8) & 0x80008000u)));
  const float2 hi = float2(as_type<half2>(((w >> 1) & 0x3F803F80u) | (w & 0x80008000u)));
  return float4(lo.x, hi.x, lo.y, hi.y);
}

// What a lane reads of NP row pairs of an mxfp8 [N, 1280] linear: lanes 0-15 the even row of each pair, lanes 16-31 the
// odd one, 16 consecutive k a step (5 steps of 256) and the step's e8m0 scale.
template <int NP>
struct dsv41_wqb_w {
  uint4 q[NP][5];
  uint e[NP][5];
};

template <int NP>
METAL_FUNC void dsv41_wqb_load(thread dsv41_wqb_w<NP>& w, const device uint4* wp, const device uchar* sp, uint lane) {
  const uint h = lane >> 4, j = lane & 15;
  for (int p = 0; p < NP; ++p) {
    const device uint4* wr = wp + (2 * p + h) * 80 + j;
    const device uchar* sr = sp + (2 * p + h) * 40 + (j >> 1);
    for (int s = 0; s < 5; ++s) {
      w.q[p][s] = wr[16 * s];
      w.e[p][s] = uint(sr[8 * s]);
    }
  }
}

// Those rows against R rows of x, held as float4 chunks: chunk (r, s, t, j) at ((r * 5 + s) * 4 + t) * 16 + j for
// k = 256 s + 16 j + 4 t, so the 16 lanes of a half read consecutive float4s.
template <int R, int NP>
METAL_FUNC void dsv41_wqb_dots(const thread dsv41_wqb_w<NP>& w, const threadgroup float4* xs, uint lane, thread float (&acc)[NP][R]) {
  const uint j = lane & 15;
  for (int p = 0; p < NP; ++p) {
    for (int r = 0; r < R; ++r) acc[p][r] = 0.0f;
    for (int s = 0; s < 5; ++s) {
      const uint u[4] = {w.q[p][s].x, w.q[p][s].y, w.q[p][s].z, w.q[p][s].w};
      float4 wv[4];
      for (int i = 0; i < 4; ++i) wv[i] = dsv41_wqb_e4m3x4(u[i]);
      const float sf = dsv41_wqb_scale(w.e[p][s]);
      for (int r = 0; r < R; ++r) {
        const threadgroup float4* xr = xs + (r * 5 + s) * 64 + j;
        const float4 x0 = xr[0], x1 = xr[16], x2 = xr[32], x3 = xr[48];
        float d = x0.x * wv[0].x;
        d = metal::fma(x0.y, wv[0].y, d);
        d = metal::fma(x0.z, wv[0].z, d);
        d = metal::fma(x0.w, wv[0].w, d);
        d = metal::fma(x1.x, wv[1].x, d);
        d = metal::fma(x1.y, wv[1].y, d);
        d = metal::fma(x1.z, wv[1].z, d);
        d = metal::fma(x1.w, wv[1].w, d);
        d = metal::fma(x2.x, wv[2].x, d);
        d = metal::fma(x2.y, wv[2].y, d);
        d = metal::fma(x2.z, wv[2].z, d);
        d = metal::fma(x2.w, wv[2].w, d);
        d = metal::fma(x3.x, wv[3].x, d);
        d = metal::fma(x3.y, wv[3].y, d);
        d = metal::fma(x3.z, wv[3].z, d);
        d = metal::fma(x3.w, wv[3].w, d);
        acc[p][r] = metal::fma(sf, d, acc[p][r]);
      }
    }
  }
}
