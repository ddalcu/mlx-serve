uint p = thread_position_in_grid.x;
uint row = p / H;
uint col = p % H;
T partial[8];
#pragma unroll
for (uint i = 0; i < 8; ++i) partial[i] = T(0);
#pragma unroll
for (uint e = 0; e < TOPK; ++e) {
  uint at = row * TOPK + e;
  T product = down[size_t(inverse[at]) * H + col] * scores[at];
  partial[e % 8] = product + partial[e % 8];
}
T total = partial[0];
#pragma unroll
for (uint i = 1; i < 8; ++i) total = partial[i] + total;
y[p] = total;