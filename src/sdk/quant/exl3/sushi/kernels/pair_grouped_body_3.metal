    if (j >= m) break;
    for (uint blk = sg; blk < nb; blk += SGS) {
      const uint pbase = c0 * TILE + blk * 128u;
      const size_t xr = (size_t)(members[j] / uint(TOPK)) * (size_t)(IDIM) + pbase;
      const size_t sr = (size_t)eid * (size_t)(IDIM) + pbase;
      float4 v = float4(
        float(x[xr + lane]) * float(suh[sr + lane]),
        float(x[xr + lane + 32u]) * float(suh[sr + lane + 32u]),
        float(x[xr + lane + 64u]) * float(suh[sr + lane + 64u]),
        float(x[xr + lane + 96u]) * float(suh[sr + lane + 96u]));
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
      const float s0 = v.x + v.y;
      const float s1 = v.x - v.y;
      const float s2 = v.z + v.w;
      const float s3 = v.z - v.w;
      const uint pw = j * CHUNK + blk * 128u;
      prepared[pw + lane] = half((s0 + s2) * psc);
      prepared[pw + lane + 32u] = half((s1 + s3) * psc);
      prepared[pw + lane + 64u] = half((s0 - s2) * psc);
      prepared[pw + lane + 96u] = half((s1 - s3) * psc);
    }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const threadgroup half *pp = prepared + sg * TILE;
    for (uint tk = c0 + sg; tk < c1; tk += 2u * SGS) {
