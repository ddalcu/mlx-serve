// Inkling 2's drafter, one Markov RNN step with precomputed static context KAPPA [steps, 3 R] f32:
// P [3 R] f32 = T(W_step . [ST | PREV[TOK]] + KAPPA[AT]), where W_step is [3 R, 2 R] (T), ST [R] (T), PREV [V, R] (T),
// TOK [1] and AT [1] int32. One SIMD group per row of W_step.
const uint row = thread_position_in_grid.x / 32;
const uint lane = thread_index_in_simdgroup;
constexpr uint IN = 2 * R;
const size_t tok = size_t(TOK[0]);
const uint at = uint(AT[0]);
const device T* w = W + size_t(row) * IN;
float acc = 0.0f;
for (uint c = lane; c < IN; c += 32) {
    const float x = c < R ? float(ST[c]) : float(PREV[tok * R + (c - R)]);
    acc += float(w[c]) * x;
}
acc = simd_sum(acc);
if (lane == 0) P[row] = float(T(acc + KAPPA[size_t(at) * (3 * R) + row]));
