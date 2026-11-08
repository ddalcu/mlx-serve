// rope_tail.metal's roped channels alone (the last RD of each D-wide vector,
// no round trip), one output element a thread, `out` [rows, HH, RD]: the same arithmetic on the same pairs, so the same
// words as rope_tail's output there.
const uint i = thread_position_in_grid.x;
const int D = x_shape[x_ndim - 1];
const int head = D - RD;
const size_t t = i / RD;
const int c = head + int(i % RD);
const int row = int(t / HH);
const size_t xi = t * D + c;
const int p = (c - head) / 2;
const size_t b = xi - (c - head) % 2;
const float x1 = float(x[b]), x2 = float(x[b + 1]);
const float theta = scal[0] * float(row + off) * (1.0f / periods[p]);
const float cs = metal::fast::cos(theta), sn = metal::fast::sin(theta);
const float r = float(T((c - head) % 2 == 0 ? x1 * cs - x2 * sn : x1 * sn + x2 * cs));
out[i] = OT(T(r));
