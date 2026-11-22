// Inkling 2: J loop message gates over S rows, one thread per element. A [J, S, D] f32 are the gated streams, MS
// [J, S, D] f32 their messages, ST [J, S, RANK] each gate's state (the down projection of both norms). F and G are
// the forget and message projections [J, D, RANK] (their stored float dtype, read as f32), or with PROJ their
// products with the states [J, S, D] f32 (a prompt's rows take them from a matmul).
// Y [J, S, D] f32 = (1 - tanh(F st)) A + tanh(G st) MS.
const uint e = thread_position_in_grid.x;
const uint S = ST_shape[1];
float f = 0.0f, g = 0.0f;
if (PROJ) {
    f = F[e];
    g = G[e];
} else {
    const uint j = e / (S * D);
    const uint row = (e / D) % S;
    const uint i = e % D;
    for (uint r = 0; r < RANK; r++) {
        const float s = float(ST[(j * S + row) * RANK + r]);
        f += float(F[(j * D + i) * RANK + r]) * s;
        g += float(G[(j * D + i) * RANK + r]) * s;
    }
}
// The precise tanh: the fast one is NaN at an infinity and a percent off near zero.
Y[e] = (1.0f - precise::tanh(f)) * A[e] + precise::tanh(g) * MS[e];
