  // Each lane owns D / 32 adjacent columns: C4 chunks of VW (float4s, or one pair at D = 64).
  constexpr int VW = D >= 128 ? 4 : 2;
  constexpr int C4 = D >= 128 ? D / 128 : 1;
  constexpr bool KEEP = ROWS * C4 <= 32;
  const int L = x_shape[2];
  const int Hx = x_shape[1];
  const uint nb = (L + ROWS - 1) / ROWS;
  const uint id = threadgroup_position_in_grid.x * SG + simdgroup_index_in_threadgroup;
  if (id >= x_shape[0] * Hx * nb) return;
  const uint bh = id / nb, blk = id % nb;
  const uint lane = thread_index_in_simdgroup;
  const int r0 = blk * ROWS;
  const int nr = ROWS == 1 ? 1 : min(L - r0, ROWS);
  const device T* base = CONTIG ? x + (bh * L + r0) * D + lane * VW * C4
      : x + (bh / Hx) * x_strides[0] + (bh % Hx) * x_strides[1] + r0 * x_strides[2] + lane * VW * C4 * x_strides[3];
  const int64_t rs = x_strides[2], cs = x_strides[3];
  #define LD4(t, u) (CONTIG ? float4(*(const device vec<T, 4>*)(base + (t) * D + 4 * (u))) \
      : cs == 1 ? float4(base[(t) * rs + 4 * (u)], base[(t) * rs + 4 * (u) + 1], base[(t) * rs + 4 * (u) + 2], base[(t) * rs + 4 * (u) + 3]) \
      : float4(base[(t) * rs + (4 * (u)) * cs], base[(t) * rs + (4 * (u) + 1) * cs], base[(t) * rs + (4 * (u) + 2) * cs], base[(t) * rs + (4 * (u) + 3) * cs]))
  #define LD2(t) (CONTIG ? float4(float2(*(const device vec<T, 2>*)(base + (t) * D)), 0.0f, 0.0f) \
      : float4(base[(t) * rs], base[(t) * rs + cs], 0.0f, 0.0f))
  #define LD(t, u) (VW == 4 ? LD4(t, u) : LD2(t))
  float4 m[C4];
  for (int u = 0; u < C4; u++)
    for (int e = 0; e < 4; e++) m[u][e] = HAS_MEAN && e < VW ? float(mean[bh * D + lane * VW * C4 + VW * u + e]) : 0.0f;
  float4 y[KEEP ? ROWS * C4 : 1];
  float amax = 0.0f;
  for (int t = 0; t < nr; t++)
    for (int u = 0; u < C4; u++) {
      const float4 v = LD(t, u) - m[u];
      if (KEEP) y[t * C4 + u] = v;
      amax = max(amax, max(max(fabs(v.x), fabs(v.y)), max(fabs(v.z), fabs(v.w))));
    }
  amax = simd_max(amax);
  const float sc = max(amax, eps) / 127.0f;
  const float inv = 1.0f / sc;
  device int8_t* ob = q + (CONTIG ? size_t((bh * L + r0) * D) : (size_t(bh) * L + r0) * D) + lane * VW * C4;
  for (int t = 0; t < nr; t++)
    for (int u = 0; u < C4; u++) {
      const float4 v = KEEP ? y[t * C4 + u] : LD(t, u) - m[u];
      const float4 c = rint(DIVIDE ? v / sc : v * inv);
      if (VW == 4) *(device char4*)(ob + t * D + 4 * u) = char4(c);
      else *(device char2*)(ob + t * D) = char2(c.xy);
    }
  #undef LD
  #undef LD2
  #undef LD4
  if (lane == 0) scales[id] = sc;
