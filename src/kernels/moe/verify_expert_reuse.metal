auto lane = thread_index_in_simdgroup;
auto sg = simdgroup_index_in_threadgroup;
auto tg = threadgroup_position_in_grid;
uint row = tg.z;
uint eid = ids[row];
uint first = row;
while (first > 0 && ids[first - 1] == eid) --first;
if ((row - first) % 2 != 0) return;
bool pair = row + 1 < uint(ids_shape[0]) && ids[row + 1] == eid;
uint out_row = tg.y * 8 + sg * 4;
float2 result[4] = {float2(0), float2(0), float2(0), float2(0)};
constexpr uint VPT = 8;
for (uint k = 0; k < K; k += VPT * 32) {
  uint offset = k + lane * VPT;
  if (offset >= K) continue;
  float2 xt[VPT];
  float2 sum = float2(0);
  for (uint i = 0; i < VPT; i += 4) {
    size_t p = size_t(row) * K + offset + i;
    T a0 = x[p], a1 = x[p+1], a2 = x[p+2], a3 = x[p+3];
    T b0 = T(0), b1 = T(0), b2 = T(0), b3 = T(0);
    if (pair) { b0 = x[p+K]; b1 = x[p+K+1]; b2 = x[p+K+2]; b3 = x[p+K+3]; }
    sum.x += a0 + a1 + a2 + a3;
    sum.y += b0 + b1 + b2 + b3;
    xt[i] = float2(float(a0), float(b0));
    xt[i+1] = float2(float(a1), float(b1)) / 16.0f;
    xt[i+2] = float2(float(a2), float(b2)) / 256.0f;
    xt[i+3] = float2(float(a3), float(b3)) / 4096.0f;
  }
  for (uint r = 0; r < 4; ++r) {
    size_t wr = size_t(eid) * N + out_row + r;
    uint packed[VPT/8];
    for (uint i = 0; i < VPT/8; ++i) packed[i] = w[wr * (K/8) + offset/8 + i];
    float scale = float(sc[wr * (K/64) + offset/64]);
    float bias = float(bi[wr * (K/64) + offset/64]);
    float2 accum = float2(0);
    for (uint i = 0; i < VPT/4; ++i) {
      uint ws = (packed[i/2] >> (16*(i%2))) & 65535;
      accum += (xt[4*i] * float(ws & 0x000f) + xt[4*i+1] * float(ws & 0x00f0) + xt[4*i+2] * float(ws & 0x0f00) + xt[4*i+3] * float(ws & 0xf000));
    }
    result[r] += scale * accum + sum * bias;
  }
}
for (uint r = 0; r < 4; ++r) {
  float a = simd_sum(result[r].x);
  float b = simd_sum(result[r].y);
  if (lane == 0) {
    y[size_t(row) * N + out_row + r] = T(a);
    if (pair) y[size_t(row+1) * N + out_row + r] = T(b);
  }
}