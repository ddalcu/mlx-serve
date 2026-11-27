// Inkling 2, one row: the attention read ATT [H * HD] (T) times its output gate GATE [H * HD] f32, rounded to T as
// `gated` leaves it, then the output projection Y [1, 1, N] (T): an affine weight (FMT 3: BITS-bit rows of WQ in MLX's
// packing, group GS scales WS and biases WB of S) or a dense [N, K] weight of T (FMT 4). grid (32, N / ROWS, 1),
// threadgroup (32, SGS, 1): a SIMD group owns ROWS output rows (the gated read shared), a lane 16 inputs a sweep.
const uint lane = thread_index_in_simdgroup;
const uint n0 = thread_position_in_grid.y * ROWS;
constexpr uint WBR = K * BITS / 8;
constexpr uint SPR = K / GS;
const device uchar* arow = (const device uchar*)WQ + size_t(n0) * WBR;
const device S* sa = (const device S*)WS + size_t(n0) * SPR;
const device S* ba = (const device S*)WB + size_t(n0) * SPR;
const device T* drow = (const device T*)WQ + size_t(n0) * K;
float acc[ROWS];
for (uint r = 0; r < ROWS; r++) acc[r] = 0.0f;
for (uint c = lane; c < K / 16; c += 32) {
  float xr[16];
  const device vec<T, 4>* ap = (const device vec<T, 4>*)(ATT + c * 16);
  const device float4* gp = (const device float4*)(GATE + c * 16);
  for (uint b = 0; b < 4; b++) {
    const vec<T, 4> a = ap[b];
    const float4 g = gp[b];
    const uint base = b * 4;
    xr[base + 0] = float(T(float(a[0]) * g.x));
    xr[base + 1] = float(T(float(a[1]) * g.y));
    xr[base + 2] = float(T(float(a[2]) * g.z));
    xr[base + 3] = float(T(float(a[3]) * g.w));
  }
  for (uint r = 0; r < ROWS; r++) {
    if (FMT == 3) {
      acc[r] += msv_ink2_affine_chunk<S, BITS, GS>(arow + r * WBR, sa + r * SPR, ba + r * SPR, c, xr);
    } else {
      const device vec<T, 4>* dp = (const device vec<T, 4>*)(drow + size_t(r) * K + c * 16);
      float d = 0.0f;
      for (uint b = 0; b < 4; b++) {
        const vec<T, 4> w = dp[b];
        const uint base = b * 4;
        d += float(w[0]) * xr[base + 0];
        d += float(w[1]) * xr[base + 1];
        d += float(w[2]) * xr[base + 2];
        d += float(w[3]) * xr[base + 3];
      }
      acc[r] += d;
    }
  }
}
for (uint r = 0; r < ROWS; r++) {
  const float v = simd_sum(acc[r]);
  if (lane == 0) Y[n0 + r] = T(v);
}
