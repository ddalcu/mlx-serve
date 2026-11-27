// PF_ROPE_PAIRS (deepseek_v41.zig `ropeTailOnly`): rope_tail_only.metal's output, a thread a (row, pair) over every head
// of the row. The pair's angle, cosine and sine are the stock kernel's statements once a (row, pair), not once an
// element; each head's two outputs are the stock kernel's expressions on the same two inputs, so the same words.
// `out` [rows, HH, RD]: a simdgroup's lanes are a row's consecutive pairs, so each head's loads and stores are
// RD * 2 contiguous bytes.
const uint i = thread_position_in_grid.x;
const int D = x_shape[x_ndim - 1];
const int head = D - RD;
constexpr int NP = RD / 2;
const int row = int(i / NP);
const int p = int(i % NP);
const float theta = scal[0] * float(row + off) * (1.0f / periods[p]);
const float cs = metal::fast::cos(theta), sn = metal::fast::sin(theta);
for (int h = 0; h < HH; ++h) {
  const size_t t = size_t(row) * HH + h;
  const size_t b = t * D + head + 2 * p;
  const float x1 = float(x[b]), x2 = float(x[b + 1]);
  const float r0 = float(T(x1 * cs - x2 * sn));
  const float r1 = float(T(x1 * sn + x2 * cs));
  out[t * RD + 2 * p] = OT(T(r0));
  out[t * RD + 2 * p + 1] = OT(T(r1));
}
