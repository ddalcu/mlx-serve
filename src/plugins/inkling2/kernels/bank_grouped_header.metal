// Register-only 16-value chunk decoders for grouped multi-row expert kernels (NVFP4, MXFP4, MXFP8, affine 2-8 bits).
inline void msv_ink2_decode16_fp(int fmt, const device uint* row, const device uchar* srow, uint c, thread float4 out[4]) {
  if (fmt == 2) {
    const uint4 q = ((const device uint4*)row)[c];
    const uint e = uint(srow[c / 2u]);
    const float sc = as_type<float>(e == 0u ? 0x400000u : (e << 23)) * 256.0f;
    const uint words[4] = {q.x, q.y, q.z, q.w};
    for (uint i = 0u; i < 4u; ++i) {
      const float2 lo = mlxserve_e4m3x2(words[i]);
      const float2 hi = mlxserve_e4m3x2(words[i] >> 8);
      out[i] = float4(lo.x, hi.x, lo.y, hi.y) * sc;
    }
    return;
  }
  const uint2 q = ((const device uint2*)row)[c];
  const uint b = uint(srow[fmt == 0 ? c : c / 2u]);
  const float sc = (fmt == 0 ? mlxserve_fp4_scale(b) : as_type<float>(b == 0u ? 0x400000u : (b << 23)) * 0.00390625f) * 4194304.0f;
  const float2 p0 = mlxserve_e2m1x2(q.x, 0), p1 = mlxserve_e2m1x2(q.x, 1);
  const float2 p2 = mlxserve_e2m1x2(q.x, 2), p3 = mlxserve_e2m1x2(q.x, 3);
  out[0] = float4(p0.x, p1.x, p2.x, p3.x) * sc;
  out[1] = float4(p0.y, p1.y, p2.y, p3.y) * sc;
  const float2 r0 = mlxserve_e2m1x2(q.y, 0), r1 = mlxserve_e2m1x2(q.y, 1);
  const float2 r2 = mlxserve_e2m1x2(q.y, 2), r3 = mlxserve_e2m1x2(q.y, 3);
  out[2] = float4(r0.x, r1.x, r2.x, r3.x) * sc;
  out[3] = float4(r0.y, r1.y, r2.y, r3.y) * sc;
}

template <typename S, int BITS, int GS>
inline void msv_ink2_decode16_affine(const device uchar* row, const device S* srow, const device S* brow, uint c, thread float4 out[4]) {
  const uint g = (c * 16u) / GS;
  const float sc = float(srow[g]);
  const float bi = float(brow[g]);
  if (BITS == 8) {
    const uint4 q = ((const device uint4*)row)[c];
    out[0] = float4(as_type<uchar4>(q.x)) * sc + bi;
    out[1] = float4(as_type<uchar4>(q.y)) * sc + bi;
    out[2] = float4(as_type<uchar4>(q.z)) * sc + bi;
    out[3] = float4(as_type<uchar4>(q.w)) * sc + bi;
    return;
  }
  if (BITS == 4) {
    const uint2 q = ((const device uint2*)row)[c];
    out[0] = float4(q.x & 15u, (q.x >> 4) & 15u, (q.x >> 8) & 15u, (q.x >> 12) & 15u) * sc + bi;
    out[1] = float4((q.x >> 16) & 15u, (q.x >> 20) & 15u, (q.x >> 24) & 15u, q.x >> 28) * sc + bi;
    out[2] = float4(q.y & 15u, (q.y >> 4) & 15u, (q.y >> 8) & 15u, (q.y >> 12) & 15u) * sc + bi;
    out[3] = float4((q.y >> 16) & 15u, (q.y >> 20) & 15u, (q.y >> 24) & 15u, q.y >> 28) * sc + bi;
    return;
  }
  if (BITS == 2) {
    const uint q = ((const device uint*)row)[c];
    out[0] = float4(q & 3u, (q >> 2) & 3u, (q >> 4) & 3u, (q >> 6) & 3u) * sc + bi;
    out[1] = float4((q >> 8) & 3u, (q >> 10) & 3u, (q >> 12) & 3u, (q >> 14) & 3u) * sc + bi;
    out[2] = float4((q >> 16) & 3u, (q >> 18) & 3u, (q >> 20) & 3u, (q >> 22) & 3u) * sc + bi;
    out[3] = float4((q >> 24) & 3u, (q >> 26) & 3u, (q >> 28) & 3u, q >> 30) * sc + bi;
    return;
  }
  const device ushort* p = (const device ushort*)(row + c * 2u * BITS);
  ulong lo = 0, hi = 0;
  for (int i = 0; i < BITS; ++i) {
    const ulong v = ulong(p[i]);
    if (16 * i < 64) lo |= v << (16 * i); else hi |= v << (16 * i - 64);
  }
  constexpr ulong mask = (1ul << BITS) - 1ul;
  for (uint q4 = 0u; q4 < 4u; ++q4) {
    float4 w;
    for (uint j = 0u; j < 4u; ++j) {
      const int off = int(q4 * 4u + j) * BITS;
      const ulong v = off + BITS <= 64 ? lo >> off : (off >= 64 ? hi >> (off - 64) : ((lo >> off) | (hi << (64 - off))));
      w[j] = float(v & mask) * sc + bi;
    }
    out[q4] = w;
  }
}
