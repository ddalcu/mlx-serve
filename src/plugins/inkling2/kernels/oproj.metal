// Inkling 2, one row: the attention read ATT [H * HD] (T) times its output gate GATE [H * HD] f32, rounded to T as
// `gated` leaves it, then the output projection Y [1, 1, N] (T): an affine weight (FMT 3: BITS-bit rows of WQ in MLX's
// packing, group GS scales WS and biases WB of S) or a dense [N, K] weight of T (FMT 4). grid (32, N, 1), threadgroup
// (32, SGS, 1): a SIMD group owns one output row, a lane 16 inputs a sweep.
const uint lane = thread_index_in_simdgroup;
const uint n = thread_position_in_grid.y;
constexpr uint WBR = K * BITS / 8;
constexpr uint SPR = K / GS;
const device uchar* arow = (const device uchar*)WQ + size_t(n) * WBR;
const device S* sa = (const device S*)WS + size_t(n) * SPR;
const device S* ba = (const device S*)WB + size_t(n) * SPR;
const device T* drow = (const device T*)WQ + size_t(n) * K;
float acc = 0.0f;
for (uint c = lane; c < K / 16u; c += 32u) {
  const device vec<T, 4>* ap = (const device vec<T, 4>*)(ATT + c * 16u);
  const device float4* gp = (const device float4*)(GATE + c * 16u);
  const float4 x0 = float4(vec<T, 4>(float4(ap[0]) * gp[0]));
  const float4 x1 = float4(vec<T, 4>(float4(ap[1]) * gp[1]));
  const float4 x2 = float4(vec<T, 4>(float4(ap[2]) * gp[2]));
  const float4 x3 = float4(vec<T, 4>(float4(ap[3]) * gp[3]));
  if (FMT == 3) {
    const float4 s4 = x0 + x1 + x2 + x3;
    const float sx = (s4.x + s4.y) + (s4.z + s4.w);
    acc += msv_ink2_affine_chunk_f4<S, BITS, GS>(arow, sa, ba, c, x0, x1, x2, x3, sx);
  } else {
    const device vec<T, 4>* dp = (const device vec<T, 4>*)(drow + c * 16u);
    acc += dot(float4(dp[0]), x0) + dot(float4(dp[1]), x1) + dot(float4(dp[2]), x2) + dot(float4(dp[3]), x3);
  }
}
const float v = simd_sum(acc);
if (lane == 0) Y[n] = T(v);
