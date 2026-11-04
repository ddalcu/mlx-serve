  const uint lane = thread_index_in_simdgroup;
  const int unit = int(threadgroup_position_in_grid.x) * 4 + int(simdgroup_index_in_threadgroup);
  const int R = X_shape[0];
  constexpr int G = K / 64;
  const int T8 = (R + 7) / 8;
  if (unit >= T8 * G) return;
  const int tile = unit / G, g = unit % G;
  const int qid = int(lane) / 4;
  const int fm = (qid & 4) + ((int(lane) / 2) % 4);
  const int fn = (qid & 2) * 2 + (int(lane) % 2) * 2;
  const float one = ONE[0];
  const int r0 = min(8 * tile + fn, R - 1), r1 = min(8 * tile + fn + 1, R - 1);
  const uint4 xa = LOAD8(r0, 8 * g + fm), xb = LOAD8(r1, 8 * g + fm);
  device float2* xf = (device float2*)XF + (size_t(tile) * G + g) * 256 + lane;
  PRAGMA_UNROLL
  for (int s = 0; s < 8; s++) xf[32 * s] = float2(bf8(xa, s) * wpre<BITS>(s), bf8(xb, s) * wpre<BITS>(s));
  float v = sum8(xa, one), u = sum8(xb, one);
  v = fma(simd_shuffle_xor(v, ushort(2)), one, v); u = fma(simd_shuffle_xor(u, ushort(2)), one, u);
  v = fma(simd_shuffle_xor(v, ushort(4)), one, v); u = fma(simd_shuffle_xor(u, ushort(4)), one, u);
  v = fma(simd_shuffle_xor(v, ushort(16)), one, v); u = fma(simd_shuffle_xor(u, ushort(16)), one, u);
  if (fm == 0) {
    if (8 * tile + fn < R) XS[size_t(8 * tile + fn) * G + g] = v;
    if (8 * tile + fn + 1 < R) XS[size_t(8 * tile + fn + 1) * G + g] = u;
  }
