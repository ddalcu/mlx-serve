const uint lane = thread_index_in_simdgroup;
const int p = int(threadgroup_position_in_grid.z);
const size_t e = size_t(ids[p]);
const int row0 = (int(threadgroup_position_in_grid.y) * SG + int(simdgroup_index_in_threadgroup)) * RPS;
const size_t at = e * N + size_t(row0);
float acc[RPS];
rq_rowdot<K, GS, RPS, BITS>((const device uint8_t*)w + at * (K * BITS / 8), scales + at * (K / GS), biases + at * (K / GS),
                      x + size_t(p / TOPK) * K, lane, acc);
if (lane == 0)
  for (int j = 0; j < RPS; j++) {
    const float h = metal::max(float(bfloat(acc[j])), 0.0f);
    y[size_t(p) * N + row0 + j] = bfloat(h * h);
  }