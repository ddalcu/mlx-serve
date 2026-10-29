const uint lane = thread_index_in_simdgroup;
const int r = int(thread_position_in_threadgroup.x) / 32;
const int row0 = int(thread_position_in_grid.y) * RPS;
float acc[RPS];
rq_rowdot<K, GS, RPS, BITS>((const device uint8_t*)w + size_t(row0) * (K * BITS / 8), scales + size_t(row0) * (K / GS),
                      biases + size_t(row0) * (K / GS), x + size_t(r) * K, lane, acc);
if (lane == 0)
  for (int j = 0; j < RPS; j++) {
    if (RELU2) {
      const float h = metal::max(float(bfloat(acc[j])), 0.0f);
      y[size_t(r) * N + row0 + j] = bfloat(h * h);
    } else {
      y[size_t(r) * N + row0 + j] = bfloat(acc[j]);
    }
  }