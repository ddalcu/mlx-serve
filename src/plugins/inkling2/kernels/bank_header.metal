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

// Chunk `c` (16 values) of one affine row against x: scale * sum(q x) + bias * sum(x) over the chunk's group. The row's
// BITS-bit values are one little-endian bit stream (MLX's packing at every width), so a chunk is BITS ushorts; `srow`
// and `brow` hold its group scales and biases (S).
template <typename S, int BITS, int GS>
inline float msv_ink2_affine_chunk(const device uchar* row, const device S* srow, const device S* brow, uint c, thread const float* x) {
  const device ushort* p = (const device ushort*)(row + c * 2 * BITS);
  ulong lo = 0, hi = 0;
  for (int i = 0; i < BITS; ++i) {
    const ulong v = ulong(p[i]);
    if (16 * i < 64) lo |= v << (16 * i); else hi |= v << (16 * i - 64);
  }
  float qx = 0.0f, sx = 0.0f;
  for (int i = 0; i < 16; ++i) {
    const int off = i * BITS;
    const ulong w = off + BITS <= 64 ? lo >> off : (off >= 64 ? hi >> (off - 64) : ((lo >> off) | (hi << (64 - off))));
    qx += float(w & ((1ul << BITS) - 1ul)) * x[i];
    sx += x[i];
  }
  const uint g = (c * 16) / GS;
  return float(srow[g]) * qx + float(brow[g]) * sx;
}
