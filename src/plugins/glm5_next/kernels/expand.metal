const uint d4 = thread_position_in_grid.x;
const uint row = thread_position_in_grid.y;
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