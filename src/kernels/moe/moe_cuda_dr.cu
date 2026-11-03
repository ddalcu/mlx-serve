using namespace mlx::core::cu;
constexpr int BITS = FMT == 1 ? 8 : 4;
constexpr int GROUP = FMT == 3 ? 16 : 32;
constexpr int CH = I * BITS / 128;
constexpr int DPB = 32 / TOPK;
const int grp = threadIdx.x >> 3;
const int l = threadIdx.x & 7;
const int b = blockIdx.y;
const int dl = grp / TOPK;
const int k = grp % TOPK;
const int d = blockIdx.x * DPB + dl;
float acc = 0.f;
if (dl < DPB && d < D) {
  const uint32_t e = inds[b * TOPK + k];
  const long row = (long)e * D + d;
  const uint4* wp = reinterpret_cast<const uint4*>(reinterpret_cast<const uint8_t*>(dw) + row * (I * BITS / 8));
  row_dot_lane<FMT, 1>(wp, ds + row * (I / GROUP), act + (long)(b * TOPK + k) * I, 0, 1, CH, l, 8, &acc);
}
for (int off = 4; off > 0; off /= 2) acc += __shfl_xor_sync(0xffffffff, acc, off);
__shared__ T part[32];
if (l == 0) part[grp] = static_cast<T>(acc);
__syncthreads();
const int dd = blockIdx.x * DPB + (int)threadIdx.x;
if ((int)threadIdx.x < DPB && dd < D) {
  T sum = static_cast<T>(0.0f);
  // MLX's reduce Sum: a correctly rounded add in T per expert, k in order.
  for (int j = 0; j < TOPK; ++j) sum = static_cast<T>(static_cast<float>(sum) + static_cast<float>(Multiply{}(part[threadIdx.x * TOPK + j], scores[b * TOPK + j])));
  y[(long)b * D + dd] = sum;
}