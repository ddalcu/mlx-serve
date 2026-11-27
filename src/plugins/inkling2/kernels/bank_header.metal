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
  const float s = fmt == 0 ? mlxserve_fp4_e4m3_x2m8(b) : as_type<float>(b == 0u ? 0x400000u : (b << 23)) * 0.00390625f;
  // The nibble and scale decodes leave 2^14 and 2^8 out.
  return d * s * 4194304.0f;
}

// Chunk `c` (16 values) of one affine row against x: scale * sum(q x) + bias * sum(x) over the chunk's group. The row's
// BITS-bit values are one little-endian bit stream (MLX's packing at every width), so a chunk is BITS ushorts; `srow`
// and `brow` hold its group scales and biases (S).
template <typename S, int BITS, int GS>
inline float msv_ink2_affine_chunk(const device uchar* row, const device S* srow, const device S* brow, uint c, thread const float* x) {
  float qx = 0.0f, sx = 0.0f;
  if constexpr (BITS == 8) {
    const device uchar4* p4 = (const device uchar4*)(row + c * 16);
    for (int b = 0; b < 4; ++b) {
      const uchar4 q = p4[b];
      const int base = b * 4;
      qx += float(q.x) * x[base + 0]; sx += x[base + 0];
      qx += float(q.y) * x[base + 1]; sx += x[base + 1];
      qx += float(q.z) * x[base + 2]; sx += x[base + 2];
      qx += float(q.w) * x[base + 3]; sx += x[base + 3];
    }
  } else if constexpr (BITS == 4) {
    const uint2 q = *(const device uint2*)(row + c * 8u);
    const uint words[2] = {q.x, q.y};
    for (int j = 0; j < 2; ++j) {
      uint w = words[j];
      const int b = j * 8;
      for (int i = 0; i < 8; ++i) {
        qx += float(w & 0xfu) * x[b + i];
        sx += x[b + i];
        w >>= 4;
      }
    }
  } else {
    const device ushort* p = (const device ushort*)(row + c * 2 * BITS);
    ulong lo = 0, hi = 0;
    for (int i = 0; i < BITS; ++i) {
      const ulong v = ulong(p[i]);
      if (16 * i < 64) lo |= v << (16 * i); else hi |= v << (16 * i - 64);
    }
    for (int i = 0; i < 16; ++i) {
      const int off = i * BITS;
      const ulong w = off + BITS <= 64 ? lo >> off : (off >= 64 ? hi >> (off - 64) : ((lo >> off) | (hi << (64 - off))));
      qx += float(w & ((1ul << BITS) - 1ul)) * x[i];
      sx += x[i];
    }
  }
  const uint g = (c * 16) / GS;
  return float(srow[g]) * qx + float(brow[g]) * sx;
}

// Chunk `c` (16 values) decoded into four half4 vectors and the float scale (and bias, affine): the grouped kernels'
// form of `msv_ink2_chunk`, with no thread arrays.
inline void msv_ink2_decode16_v4(int fmt, const device uint* row, const device uchar* srow, uint c,
                                 thread half4& w0, thread half4& w1, thread half4& w2, thread half4& w3,
                                 thread float& scale) {
  if (fmt == 2) {
    const uint4 q = ((const device uint4*)row)[c];
    const float2 a0 = mlxserve_e4m3x2(q.x), b0 = mlxserve_e4m3x2(q.x >> 8);
    const float2 a1 = mlxserve_e4m3x2(q.y), b1 = mlxserve_e4m3x2(q.y >> 8);
    const float2 a2 = mlxserve_e4m3x2(q.z), b2 = mlxserve_e4m3x2(q.z >> 8);
    const float2 a3 = mlxserve_e4m3x2(q.w), b3 = mlxserve_e4m3x2(q.w >> 8);
    w0 = half4(half(a0.x), half(b0.x), half(a0.y), half(b0.y));
    w1 = half4(half(a1.x), half(b1.x), half(a1.y), half(b1.y));
    w2 = half4(half(a2.x), half(b2.x), half(a2.y), half(b2.y));
    w3 = half4(half(a3.x), half(b3.x), half(a3.y), half(b3.y));
    const uint e = uint(srow[c / 2]);
    scale = as_type<float>(e == 0u ? 0x400000u : (e << 23)) * 256.0f;
    return;
  }
  const uint2 q = ((const device uint2*)row)[c];
  const uint vx0 = q.x, vx1 = q.x >> 4, vx2 = q.x >> 8, vx3 = q.x >> 12;
  const half2 px0 = as_type<half2>(((vx0 & 0x70007u) << 9) | ((vx0 & 0x80008u) << 12));
  const half2 px1 = as_type<half2>(((vx1 & 0x70007u) << 9) | ((vx1 & 0x80008u) << 12));
  const half2 px2 = as_type<half2>(((vx2 & 0x70007u) << 9) | ((vx2 & 0x80008u) << 12));
  const half2 px3 = as_type<half2>(((vx3 & 0x70007u) << 9) | ((vx3 & 0x80008u) << 12));
  w0 = half4(px0.x, px1.x, px2.x, px3.x);
  w1 = half4(px0.y, px1.y, px2.y, px3.y);
  const uint vy0 = q.y, vy1 = q.y >> 4, vy2 = q.y >> 8, vy3 = q.y >> 12;
  const half2 py0 = as_type<half2>(((vy0 & 0x70007u) << 9) | ((vy0 & 0x80008u) << 12));
  const half2 py1 = as_type<half2>(((vy1 & 0x70007u) << 9) | ((vy1 & 0x80008u) << 12));
  const half2 py2 = as_type<half2>(((vy2 & 0x70007u) << 9) | ((vy2 & 0x80008u) << 12));
  const half2 py3 = as_type<half2>(((vy3 & 0x70007u) << 9) | ((vy3 & 0x80008u) << 12));
  w2 = half4(py0.x, py1.x, py2.x, py3.x);
  w3 = half4(py0.y, py1.y, py2.y, py3.y);
  const uint b = uint(srow[fmt == 0 ? c : c / 2]);
  const float s = fmt == 0 ? mlxserve_fp4_e4m3_x2m8(b) : as_type<float>(b == 0u ? 0x400000u : (b << 23)) * 0.00390625f;
  scale = s * 4194304.0f;
}

template <typename S, int BITS, int GS>
inline void msv_ink2_decode16_affine_v4(const device uchar* row, const device S* srow, const device S* brow, uint c,
                                        thread half4& w0, thread half4& w1, thread half4& w2, thread half4& w3,
                                        thread float& scale, thread float& bias) {
  if constexpr (BITS == 8) {
    const device uchar4* p4 = (const device uchar4*)(row + c * 16);
    w0 = half4(p4[0]);
    w1 = half4(p4[1]);
    w2 = half4(p4[2]);
    w3 = half4(p4[3]);
  } else if constexpr (BITS == 4) {
    const uint2 q = *(const device uint2*)(row + c * 8u);
    w0 = half4(half(q.x & 0xfu), half((q.x >> 4) & 0xfu), half((q.x >> 8) & 0xfu), half((q.x >> 12) & 0xfu));
    w1 = half4(half((q.x >> 16) & 0xfu), half((q.x >> 20) & 0xfu), half((q.x >> 24) & 0xfu), half(q.x >> 28));
    w2 = half4(half(q.y & 0xfu), half((q.y >> 4) & 0xfu), half((q.y >> 8) & 0xfu), half((q.y >> 12) & 0xfu));
    w3 = half4(half((q.y >> 16) & 0xfu), half((q.y >> 20) & 0xfu), half((q.y >> 24) & 0xfu), half(q.y >> 28));
  } else {
    const device ushort* p = (const device ushort*)(row + c * 2 * BITS);
    ulong lo = 0, hi = 0;
    for (int i = 0; i < BITS; ++i) {
      const ulong v = ulong(p[i]);
      if (16 * i < 64) lo |= v << (16 * i); else hi |= v << (16 * i - 64);
    }
    const ulong mask = (1ul << BITS) - 1ul;
    auto grab = [&](int i) -> half {
      const int off = i * BITS;
      const ulong q = off + BITS <= 64 ? lo >> off : (off >= 64 ? hi >> (off - 64) : ((lo >> off) | (hi << (64 - off))));
      return half(uint(q & mask));
    };
    w0 = half4(grab(0), grab(1), grab(2), grab(3));
    w1 = half4(grab(4), grab(5), grab(6), grab(7));
    w2 = half4(grab(8), grab(9), grab(10), grab(11));
    w3 = half4(grab(12), grab(13), grab(14), grab(15));
  }
  const uint g = (c * 16) / GS;
  scale = float(srow[g]);
  bias = float(brow[g]);
}
