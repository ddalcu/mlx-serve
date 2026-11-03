using namespace mlx::core::cu;
constexpr int BITS = FMT == 1 ? 8 : 4;
constexpr int GROUP = FMT == 3 ? 16 : 32;
constexpr int CH = D * BITS / 128;
const int lane = threadIdx.x & 31;
const int n = blockIdx.x * 8 + (threadIdx.x >> 5);
const int bk = blockIdx.y;
const int b = bk / TOPK;
const uint32_t e = inds[bk];
const T* xr = x + (long)b * D;
const long row = (long)e * I + n;
const uint4* gp = reinterpret_cast<const uint4*>(reinterpret_cast<const uint8_t*>(gw) + row * (D * BITS / 8));
const uint4* upw = reinterpret_cast<const uint4*>(reinterpret_cast<const uint8_t*>(uw) + row * (D * BITS / 8));
float ag = 0.f, au = 0.f;
row_dot_lane<FMT, 1>(gp, gs + row * (D / GROUP), xr, 0, 1, CH, lane, 32, &ag);
row_dot_lane<FMT, 1>(upw, us + row * (D / GROUP), xr, 0, 1, CH, lane, 32, &au);
for (int off = 16; off > 0; off /= 2) {
  ag += __shfl_xor_sync(0xffffffff, ag, off);
  au += __shfl_xor_sync(0xffffffff, au, off);
}
if (lane == 0) {
  T g = static_cast<T>(ag);
  const T u = static_cast<T>(au);
  if (GSCALE) g = Multiply{}(g, gscale[e]);
  const T sl = Multiply{}(g, Sigmoid{}(g));
  act[(long)bk * I + n] = Multiply{}(sl, u);
}