// Chunk `c` (16 values) of one packed expert row against x, as a float. FMT 0 = NVFP4 (e2m1 nibbles, an e4m3 scale
// per 16), 1 = MXFP4 (an e8m0 scale per 32), 2 = MXFP8 (e4m3 bytes, an e8m0 scale per 32). `row` is the row's
// words, `srow` its scales.
inline float msv_ink2_chunk(int fmt, const device uint* row, const device uchar* srow, uint c, thread const float* x) {
  if (fmt == 2) {
    const uint4 q = ((const device uint4*)row)[c];
    const uint words[4] = {q.x, q.y, q.z, q.w};
    float d = 0.0f;
    for (int i = 0; i < 4; ++i) {
      const float2 lo = mlxserve_e4m3x2(words[i]);
      const float2 hi = mlxserve_e4m3x2(words[i] >> 8);
      d += x[4 * i] * lo.x + x[4 * i + 1] * hi.x + x[4 * i + 2] * lo.y + x[4 * i + 3] * hi.y;
    }
    const uint e = uint(srow[c / 2]);
    // The byte decode leaves 2^8 out.
    return d * as_type<float>(e == 0u ? 0x400000u : (e << 23)) * 256.0f;
  }
  const float d = mlxserve_fp4_dot16(((const device uint2*)row)[c], x);
  const uint b = uint(srow[fmt == 0 ? c : c / 2]);
  const float s = fmt == 0 ? mlxserve_fp4_scale(b) : as_type<float>(b == 0u ? 0x400000u : (b << 23)) * 0.00390625f;
  // The nibble and scale decodes leave 2^14 and 2^8 out.
  return d * s * 4194304.0f;
}

// Fast 16-value affine chunk using pre-loaded float4 x0..x3 and pre-summed sx.
template <typename S, int BITS, int GS>
inline float msv_ink2_affine_chunk_f4(const device uchar* row, const device S* srow, const device S* brow, uint c, float4 x0, float4 x1, float4 x2, float4 x3, float sx) {
  const uint g = (c * 16u) / GS;
  float qx = 0.0f;
  if (BITS == 8) {
    const uint4 q = ((const device uint4*)row)[c];
    qx = dot(float4(as_type<uchar4>(q.x)), x0) + dot(float4(as_type<uchar4>(q.y)), x1) +
         dot(float4(as_type<uchar4>(q.z)), x2) + dot(float4(as_type<uchar4>(q.w)), x3);
  } else if (BITS == 4) {
    const uint2 q = ((const device uint2*)row)[c];
    const float4 w0 = float4(q.x & 15u, (q.x >> 4) & 15u, (q.x >> 8) & 15u, (q.x >> 12) & 15u);
    const float4 w1 = float4((q.x >> 16) & 15u, (q.x >> 20) & 15u, (q.x >> 24) & 15u, q.x >> 28);
    const float4 w2 = float4(q.y & 15u, (q.y >> 4) & 15u, (q.y >> 8) & 15u, (q.y >> 12) & 15u);
    const float4 w3 = float4((q.y >> 16) & 15u, (q.y >> 20) & 15u, (q.y >> 24) & 15u, q.y >> 28);
    qx = dot(w0, x0) + dot(w1, x1) + dot(w2, x2) + dot(w3, x3);
  } else if (BITS == 2) {
    const uint q = ((const device uint*)row)[c];
    const float4 w0 = float4(q & 3u, (q >> 2) & 3u, (q >> 4) & 3u, (q >> 6) & 3u);
    const float4 w1 = float4((q >> 8) & 3u, (q >> 10) & 3u, (q >> 12) & 3u, (q >> 14) & 3u);
    const float4 w2 = float4((q >> 16) & 3u, (q >> 18) & 3u, (q >> 20) & 3u, (q >> 22) & 3u);
    const float4 w3 = float4((q >> 24) & 3u, (q >> 26) & 3u, (q >> 28) & 3u, q >> 30);
    qx = dot(w0, x0) + dot(w1, x1) + dot(w2, x2) + dot(w3, x3);
  } else {
    const device ushort* p = (const device ushort*)(row + c * 2u * BITS);
    ulong lo = 0, hi = 0;
    for (int i = 0; i < BITS; ++i) {
      const ulong v = ulong(p[i]);
      if (16 * i < 64) lo |= v << (16 * i); else hi |= v << (16 * i - 64);
    }
    const float xs[16] = { x0.x, x0.y, x0.z, x0.w, x1.x, x1.y, x1.z, x1.w, x2.x, x2.y, x2.z, x2.w, x3.x, x3.y, x3.z, x3.w };
    constexpr ulong mask = (1ul << BITS) - 1ul;
    for (int i = 0; i < 16; ++i) {
      const int off = i * BITS;
      const ulong w = off + BITS <= 64 ? lo >> off : (off >= 64 ? hi >> (off - 64) : ((lo >> off) | (hi << (64 - off))));
      qx += float(w & mask) * xs[i];
    }
  }
  return float(srow[g]) * qx + float(brow[g]) * sx;
}

// Chunk `c` (16 values) of one affine row against x: scale * sum(q x) + bias * sum(x) over the chunk's group.
template <typename S, int BITS, int GS>
inline float msv_ink2_affine_chunk(const device uchar* row, const device S* srow, const device S* brow, uint c, thread const float* x) {
  const float4 x0 = float4(x[0], x[1], x[2], x[3]);
  const float4 x1 = float4(x[4], x[5], x[6], x[7]);
  const float4 x2 = float4(x[8], x[9], x[10], x[11]);
  const float4 x3 = float4(x[12], x[13], x[14], x[15]);
  const float4 s4 = x0 + x1 + x2 + x3;
  const float sx = (s4.x + s4.y) + (s4.z + s4.w);
  return msv_ink2_affine_chunk_f4<S, BITS, GS>(row, srow, brow, c, x0, x1, x2, x3, sx);
}
