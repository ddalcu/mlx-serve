uint lid = thread_position_in_threadgroup.x;
uint sg = simdgroup_index_in_threadgroup;
uint lane = thread_index_in_simdgroup;
uint tg = threadgroup_position_in_grid.x;
threadgroup T act[PD];
for (uint i = lid; i < uint(PD); i += 64) act[i] = gtab[as_type<ushort>(g[i])] * ple[i];
threadgroup_barrier(mem_flags::mem_threadgroup);
int prow = int(tg) * 8 + int(sg) * 4;
if (prow >= HID) return;
const device uint8_t* ws2 = (const device uint8_t*)pw + prow * (PD / 2) + lane * 4;
const device T* sl2 = ps + prow * (PD / 64) + lane / 8;
const device T* bl2 = pb + prow * (PD / 64) + lane / 8;
const threadgroup T* x2 = act + lane * 8;
float xt2[8];
float sum2 = 0;
for (int i = 0; i < 8; i += 4) {
  sum2 += x2[i] + x2[i + 1] + x2[i + 2] + x2[i + 3];
  xt2[i] = x2[i];
  xt2[i + 1] = x2[i + 1] / 16.0f;
  xt2[i + 2] = x2[i + 2] / 256.0f;
  xt2[i + 3] = x2[i + 3] / 4096.0f;
}
for (int row = 0; row < 4; row++) {
  const device uint16_t* wl = (const device uint16_t*)(ws2 + row * (PD / 2));
  float s = sl2[row * (PD / 64)];
  float b = bl2[row * (PD / 64)];
  float accum = 0;
  for (int i = 0; i < 2; i++) {
    accum += (xt2[4 * i] * (wl[i] & 0x000f) + xt2[4 * i + 1] * (wl[i] & 0x00f0) +
              xt2[4 * i + 2] * (wl[i] & 0x0f00) + xt2[4 * i + 3] * (wl[i] & 0xf000));
  }
  float r = simd_sum(s * accum + sum2 * b);
  if (lane == 0) y[prow + row] = T(r);
}