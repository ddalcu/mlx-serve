using XV = metal::conditional_t<metal::is_same_v<T, half>, half2, float2>;
const int K = x_shape[x_ndim - 1];
const int N = w_shape[0];
const int KW = K / 16, KG = K / 128;
uint lane = thread_index_in_simdgroup;
uint sgi = simdgroup_index_in_threadgroup;
int grow = (threadgroup_position_in_grid.y * G + sgi) * R;
const device uint* wq = w + grow * KW + lane;
const device T* sp = scales + grow * KG + lane / 8;
float res[R][M];
for (int r = 0; r < R; ++r) for (int m = 0; m < M; ++m) res[r][m] = 0;
for (int k = 0; k < K; k += 512) {
  uint wv[R]; float sc[R];
  for (int r = 0; r < R; ++r) { wv[r] = wq[r * KW]; sc[r] = float(sp[r * KG]); }
  half2 q[R][8];
  for (int r = 0; r < R; ++r) h2dec(wv[r], q[r], 1.0h);
  for (int m = 0; m < M; ++m) {
    const device T* xr = x + m * K + k + lane * 16;
    XV xh[8];
    for (int j = 0; j < 8; ++j) xh[j] = XV(xr[j], xr[j + 8]);
    for (int r = 0; r < R; ++r) {
      float2 acc = float2(q[r][0]) * float2(xh[0]);
      for (int j = 1; j < 8; ++j) acc = fma(float2(q[r][j]), float2(xh[j]), acc);
      res[r][m] += sc[r] * (acc.x + acc.y);
    }
  }
  wq += 32; sp += 4;
}
for (int r = 0; r < R; ++r) for (int m = 0; m < M; ++m) {
  float v = simd_sum(res[r][m]);
  if (lane == 0) y[m * N + grow + r] = static_cast<T>(v);
}