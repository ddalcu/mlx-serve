// glm5_next's kernels/expand.metal writing its words into `dst` (a per-prompt
// region, bound as an input) instead of a fresh output; `carrier`, one word, is the output the region's publish waits
// behind (scratch.zig); `after`, bound and never read, orders it behind the region's last reader. The same loads,
// fmas and stores as glm5's.
device T* out = (device T*)dst;
const uint d4 = thread_position_in_grid.x;
const uint row = thread_position_in_grid.y;
if (d4 == 0 && row == 0) carrier[0] = 0;
if (d4 >= uint(D / 4)) return;
const auto pr = post + row * HC;
const auto cb = comb + row * HC * HC;
const float4 xv = float4(*(const device vec<T, 4>*)(x + (size_t)row * D + 4 * d4));
float4 sv[HC];
for (int j = 0; j < HC; ++j)
    sv[j] = float4(*(const device vec<T, 4>*)(res + ((size_t)row * HC + j) * D + 4 * d4));
for (int i = 0; i < HC; ++i) {
    float4 acc = 0.0f;
    for (int j = 0; j < HC; ++j) acc = fma(float4(cb[j * HC + i]), sv[j], acc);
    *(device vec<T, 4>*)(out + ((size_t)row * HC + i) * D + 4 * d4) = vec<T, 4>(fma(float4(pr[i]), xv, acc));
}
