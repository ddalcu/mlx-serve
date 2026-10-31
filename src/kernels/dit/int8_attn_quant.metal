  constexpr int C4 = D / 128;
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
  const device T* base = CONTIG ? x + (bh * L + r0) * D + lane * 4 * C4
      : x + (bh / Hx) * x_strides[0] + (bh % Hx) * x_strides[1] + r0 * x_strides[2] + lane * 4 * C4 * x_strides[3];
  const int64_t rs = x_strides[2], cs = x_strides[3];
  #define LD(t, u) (CONTIG ? float4(*(const device vec<T, 4>*)(base + (t) * D + 4 * (u))) \
      : cs == 1 ? float4(base[(t) * rs + 4 * (u)], base[(t) * rs + 4 * (u) + 1], base[(t) * rs + 4 * (u) + 2], base[(t) * rs + 4 * (u) + 3]) \
      : float4(base[(t) * rs + (4 * (u)) * cs], base[(t) * rs + (4 * (u) + 1) * cs], base[(t) * rs + (4 * (u) + 2) * cs], base[(t) * rs + (4 * (u) + 3) * cs]))
  float4 m[C4];
  for (int u = 0; u < C4; u++)
    for (int e = 0; e < 4; e++) m[u][e] = HAS_MEAN ? float(mean[bh * D + lane * 4 * C4 + 4 * u + e]) : 0.0f;
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
  device int8_t* ob = q + (CONTIG ? size_t((bh * L + r0) * D) : (size_t(bh) * L + r0) * D) + lane * 4 * C4;
  for (int t = 0; t < nr; t++)
    for (int u = 0; u < C4; u++) {
      const float4 v = KEEP ? y[t * C4 + u] : LD(t, u) - m[u];
      *(device char4*)(ob + t * D + 4 * u) = char4(rint(DIVIDE ? v / sc : v * inv));
    }
  #undef LD
  if (lane == 0) scales[id] = sc;