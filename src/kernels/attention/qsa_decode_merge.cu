const int head = blockIdx.x;
const int r = blockIdx.y;
const long hrow = (long)r * gridDim.x + head;
const int d = threadIdx.x;
const int lane = threadIdx.x & 31;
__shared__ float wts[NS];
__shared__ float zsum[1];
if ((threadIdx.x >> 5) == 0) {
  constexpr int SLOTS = (NS + 31) / 32;
  float ms[SLOTS], ls[SLOTS];
  float M = -3.0e38f;
  #pragma unroll
  for (int s = 0; s < SLOTS; ++s) {
    const int j = lane + s * 32;
    const long pr = hrow * NS + j;
    ms[s] = (j < NS) ? pml[pr * 2] : -3.0e38f;
    ls[s] = (j < NS) ? pml[pr * 2 + 1] : 0.0f;
    if (ls[s] > 0.0f) M = fmaxf(M, ms[s]);
  }
  for (int o = 16; o > 0; o >>= 1) M = fmaxf(M, __shfl_xor_sync(0xffffffffu, M, o));
  float Z = 0.0f;
  #pragma unroll
  for (int s = 0; s < SLOTS; ++s) {
    const int j = lane + s * 32;
    const float w = (ls[s] > 0.0f) ? exp2f(ms[s] - M) : 0.0f;
    if (j < NS) wts[j] = w;
    Z += w * ls[s];
  }
  for (int o = 16; o > 0; o >>= 1) Z += __shfl_xor_sync(0xffffffffu, Z, o);
  if (lane == 0) zsum[0] = Z;
}
__syncthreads();
const float* pa = pacc + hrow * NS * 256 + d;
float a[4] = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
for (int j = 0; j < NS; ++j) a[j & 3] = fmaf(wts[j], pa[(long)j * 256], a[j & 3]);
out[((long)head * gridDim.y + r) * 256 + d] = static_cast<T>(((a[0] + a[1]) + (a[2] + a[3])) / zsum[0]);