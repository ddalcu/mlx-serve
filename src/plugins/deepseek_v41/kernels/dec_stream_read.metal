// dec_stream_read: the moe_fp4 ubench's bandwidth probe. A simdgroup reads 4 KB blocks of `src` (NV uint4 words, a
// multiple of 256) at a grid stride, its lanes' eight 16-byte loads a block in flight; `dep` (the last step's output)
// makes a chain of launches run one after another.
const uint lane = thread_index_in_simdgroup;
const uint sg = thread_position_in_grid.x / 32;
const uint nsg = threads_per_grid.x / 32;
const device uint4* p = (const device uint4*)src;
uint4 a = uint4(dep[0]);
for (uint b = sg; b < uint(NV) / 256; b += nsg) {
  const device uint4* q = p + b * 256 + lane;
  uint4 v[8];
#pragma unroll
  for (int k = 0; k < 8; ++k) v[k] = q[32 * k];
#pragma unroll
  for (int k = 0; k < 8; ++k) a ^= v[k];
}
const uint v = simd_xor(a.x ^ a.y ^ a.z ^ a.w);
if (thread_position_in_grid.x == 0 || v == 0x9E3779B9u) out[0] = v;
