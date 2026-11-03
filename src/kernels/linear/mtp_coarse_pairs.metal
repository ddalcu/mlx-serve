uint lane = thread_index_in_simdgroup;
uint sg = simdgroup_index_in_threadgroup;
auto tg = threadgroup_position_in_grid;
int K = int(K_size), N = int(N_size);
uint row = tg.z * 2;
bool pair = row + 1 < uint(x_shape[0]);
uint out_row = tg.y * 8 + sg * 4;
const device uint8_t* w8 = (const device uint8_t*)w;
float2 result[4] = {float2(0), float2(0), float2(0), float2(0)};
for (int k = 0; k < K; k += 512) {
  float2 xt[16];
  float2 sum = float2(0);
  for (uint i = 0; i < 16; i += 8) {
    size_t at = size_t(row) * K + k + lane * 16 + i;
    const device T* a = x + at;
    sum.x += a[0] + a[1] + a[2] + a[3] + a[4] + a[5] + a[6] + a[7];
    float b[8] = {0,0,0,0,0,0,0,0};
    if (pair) {
      const device T* v = a + K;
      sum.y += v[0] + v[1] + v[2] + v[3] + v[4] + v[5] + v[6] + v[7];
      for (uint j = 0; j < 8; ++j) b[j] = float(v[j]);
    }
    xt[i] = float2(float(a[0]), b[0]);
    xt[i+1] = float2(float(a[1]), b[1]) / 8.0f;
    xt[i+2] = float2(float(a[2]), b[2]) / 64.0f;
    xt[i+3] = float2(float(a[3]), b[3]) / 2.0f;
    xt[i+4] = float2(float(a[4]), b[4]) / 16.0f;
    xt[i+5] = float2(float(a[5]), b[5]) / 128.0f;
    xt[i+6] = float2(float(a[6]), b[6]) / 4.0f;
    xt[i+7] = float2(float(a[7]), b[7]) / 32.0f;
  }
  for (uint r = 0; r < 4; ++r) {
    const device uint8_t* packed = w8 + size_t(out_row+r) * (K*3/8) + k*3/8 + lane*6;
    float scale = float(sc[size_t(out_row+r) * (K/64) + k/64 + lane/4]);
    float bias = float(bi[size_t(out_row+r) * (K/64) + k/64 + lane/4]);
    float2 accum = float2(0);
    for (uint i = 0; i < 2; ++i) {
      uint a = packed[3*i], b = packed[3*i+1], c = packed[3*i+2];
      accum += float(a & 0x07) * xt[8*i];
      accum += float(a & 0x38) * xt[8*i+1];
      accum += float(a & 0xc0) * xt[8*i+2];
      accum += float(b & 0x01) * (xt[8*i+2] * 256.0f);
      accum += float(b & 0x0e) * xt[8*i+3];
      accum += float(b & 0x70) * xt[8*i+4];
      accum += float(b & 0x80) * xt[8*i+5];
      accum += float(c & 0x03) * (xt[8*i+5] * 256.0f);
      accum += float(c & 0x1c) * xt[8*i+6];
      accum += float(c & 0xe0) * xt[8*i+7];
    }
    result[r] += scale * accum + sum * bias;
  }
}
for (uint r = 0; r < 4; ++r) {
  float a = simd_sum(result[r].x), b = simd_sum(result[r].y);
  if (lane == 0) {
    y[size_t(row)*N + out_row+r] = T(a);
    if (pair) y[size_t(row+1)*N + out_row+r] = T(b);
  }
}