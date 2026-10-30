using namespace mlx::core::cu;
const long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
if (idx >= (long)up_shape[0] * up_shape[1] * H) return;
const long token = idx / H, col = idx % H;
float acc = 0.f;
#pragma unroll
for (int h = 0; h < HC; ++h) {
  const long off = (token * HC + h) * H + col;
  const T p = Multiply{}(Sigmoid{}(up[off]), normed[off]);
  acc = static_cast<float>(static_cast<T>(acc + static_cast<float>(p)));
}
out[idx] = Multiply{}(static_cast<T>(acc), static_cast<T>(1.0f / HC));