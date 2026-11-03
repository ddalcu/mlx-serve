using namespace mlx::core::cu;
namespace cg = cooperative_groups;
auto block = cg::this_thread_block();
__shared__ float temp[2];
const int row = blockIdx.x;
const int token = row / HC, h = row % HC;
const int tid = threadIdx.x;
using V = msv_vec8<T>;
V xv[NCH];
float normalizer = 0;
#pragma unroll
for (int i = 0; i < NCH; ++i) {
  const long off = (long)row * H + (64 * i + tid) * 8;
  xv[i] = *reinterpret_cast<const V*>(x + off);
  if (WR) {
    const T g = wi[token * HC + h];
    const V ov = *reinterpret_cast<const V*>(wo + (long)token * H + (64 * i + tid) * 8);
    #pragma unroll
    for (int j = 0; j < 8; ++j) {
      xv[i].v[j] = Add{}(xv[i].v[j], Multiply{}(ov.v[j], g));
    }
    *reinterpret_cast<V*>(stream + off) = xv[i];
  }
  #pragma unroll
  for (int j = 0; j < 8; ++j) {
    float t = static_cast<float>(xv[i].v[j]);
    normalizer += t * t;
  }
}
auto warp = cg::tiled_partition<32>(block);
float r = cg::reduce(warp, normalizer, cg::plus<float>{});
if (warp.thread_rank() == 0) temp[warp.meta_group_rank()] = r;
block.sync();
r = warp.thread_rank() < warp.meta_group_size() ? temp[warp.thread_rank()] : float{};
normalizer = cg::reduce(warp, r, cg::plus<float>{});
const uint32_t axis_size = H;
normalizer = rsqrt(normalizer / axis_size + eps[0]);
#pragma unroll
for (int i = 0; i < NCH; ++i) {
  const long off = (long)row * H + (64 * i + tid) * 8;
  const V wv = *reinterpret_cast<const V*>(w + h * H + (64 * i + tid) * 8);
  V nv;
  #pragma unroll
  for (int j = 0; j < 8; ++j) {
    float y = static_cast<float>(xv[i].v[j]) * normalizer;
    nv.v[j] = Multiply{}(static_cast<T>(y), wv.v[j]);
  }
  *reinterpret_cast<V*>(normed + off) = nv;
}