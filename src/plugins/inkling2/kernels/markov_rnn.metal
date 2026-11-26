// Inkling 2's drafter, one Markov step: the RNN's projection P [3 R] f32 = T(T(W . fed) + B) as the matmul and the
// bias add round it, fed = [ST | PREV[TOK] | ROWS[AT] | NEW]: the state [R], the previous token's row of PREV [V, R],
// the step's block row of ROWS [1, block - 1, H] and the newest committed mix NEW [H], all T. One SIMD group a row of
// W [3 R, 2 R + 2 H] (T); TOK [1] and AT [1] int32.
const uint row = thread_position_in_grid.x / 32;
const uint lane = thread_index_in_simdgroup;
constexpr uint IN = 2 * R + 2 * H;
const size_t tok = size_t(TOK[0]);
const uint at = uint(AT[0]);
const device T* w = W + size_t(row) * IN;
float acc = 0.0f;
for (uint c = lane; c < IN; c += 32) {
    float x;
    if (c < R) x = float(ST[c]);
    else if (c < 2 * R) x = float(PREV[tok * R + c - R]);
    else if (c < 2 * R + H) x = float(ROWS[at * H + c - 2 * R]);
    else x = float(NEW[c - 2 * R - H]);
    acc += float(w[c]) * x;
}
acc = simd_sum(acc);
if (lane == 0) P[row] = float(T(T(acc) + B[row]));
