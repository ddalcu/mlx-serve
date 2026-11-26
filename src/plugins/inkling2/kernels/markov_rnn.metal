// Inkling 2's drafter, one Markov step (1 or B lanes): the RNN's projection P [B, 3 R] (or [3 R] at B=1) f32 =
// T(T(W . fed) + B) as the matmul and the bias add round it, fed = [ST[b] | PREV[TOK[b]] | ROWS[b, AT] | NEW[b]]:
// the state [B, 1, R], the previous token's row of PREV [V, R], the step's block row of ROWS [B, steps, H] and the
// newest committed mix NEW [B, 1, H], all T. grid (32 * 3 * R, B, 1), threadgroup (256, 1, 1); TOK [B], AT [1] int32.
const uint row = thread_position_in_grid.x / 32;
const uint b = threadgroup_position_in_grid.y;
const uint lane = thread_index_in_simdgroup;
constexpr uint IN = 2 * R + 2 * H;
const uint STEPS = ROWS_shape[1];
const size_t tok = size_t(TOK[b]);
const uint at = uint(AT[0]);
const device T* w = W + size_t(row) * IN;
const device T* st = ST + size_t(b) * R;
const device T* rrow = ROWS + (size_t(b) * STEPS + at) * H;
const device T* nrow = NEW + size_t(b) * H;
float acc = 0.0f;
for (uint c = lane; c < IN; c += 32) {
    float x;
    if (c < R) x = float(st[c]);
    else if (c < 2 * R) x = float(PREV[tok * R + c - R]);
    else if (c < 2 * R + H) x = float(rrow[c - 2 * R]);
    else x = float(nrow[c - 2 * R - H]);
    acc += float(w[c]) * x;
}
acc = simd_sum(acc);
if (lane == 0) P[size_t(b) * (3 * R) + row] = float(T(T(acc) + B[row]));
