// Inkling 2: J loop message gates over S rows. A [J, S, D] f32 are the gated streams, MS [J, S, D] f32 their
// messages, ST [J, S, RANK] each gate's state (the down projection of both norms). F and G are the forget and message
// projections [J, D, RANK] (their stored float dtype, read as f32), or with PROJ their products with the states
// [J, S, D] f32. With COOP (unprojected, RANK % 8 == 0), 8 consecutive lanes cooperate along RANK loading 8 elements
// per lane (16-byte coalesced bf16 loads), reduced with 3 simd_shuffle_xor steps.
// Y [J, S, D] f32 = (1 - tanh(F st)) A + tanh(G st) MS.
const uint S = ST_shape[1];
if (PROJ) {
    const uint e = thread_position_in_grid.x;
    const float f = F[e];
    const float g = G[e];
    Y[e] = (1.0f - precise::tanh(f)) * A[e] + precise::tanh(g) * MS[e];
    return;
}
if (COOP) {
    const uint sub = thread_index_in_simdgroup & 7u;
    const uint e = thread_position_in_grid.x >> 3;
    const uint j = e / (S * D);
    const uint row = (e / D) % S;
    const uint i = e % D;
    const uint st_off = (j * S + row) * RANK;
    const uint w_off = (j * D + i) * RANK;
    float f = 0.0f, g = 0.0f;
    for (uint r = sub * 8u; r < RANK; r += 64u) {
        const float4 s0 = float4(float(ST[st_off + r + 0u]), float(ST[st_off + r + 1u]), float(ST[st_off + r + 2u]), float(ST[st_off + r + 3u]));
        const float4 s1 = float4(float(ST[st_off + r + 4u]), float(ST[st_off + r + 5u]), float(ST[st_off + r + 6u]), float(ST[st_off + r + 7u]));
        const float4 f0 = float4(float(F[w_off + r + 0u]), float(F[w_off + r + 1u]), float(F[w_off + r + 2u]), float(F[w_off + r + 3u]));
        const float4 f1 = float4(float(F[w_off + r + 4u]), float(F[w_off + r + 5u]), float(F[w_off + r + 6u]), float(F[w_off + r + 7u]));
        const float4 g0 = float4(float(G[w_off + r + 0u]), float(G[w_off + r + 1u]), float(G[w_off + r + 2u]), float(G[w_off + r + 3u]));
        const float4 g1 = float4(float(G[w_off + r + 4u]), float(G[w_off + r + 5u]), float(G[w_off + r + 6u]), float(G[w_off + r + 7u]));
        f += dot(f0, s0) + dot(f1, s1);
        g += dot(g0, s0) + dot(g1, s1);
    }
    f += simd_shuffle_xor(f, 4u);
    f += simd_shuffle_xor(f, 2u);
    f += simd_shuffle_xor(f, 1u);
    g += simd_shuffle_xor(g, 4u);
    g += simd_shuffle_xor(g, 2u);
    g += simd_shuffle_xor(g, 1u);
    if (sub == 0u) Y[e] = (1.0f - precise::tanh(f)) * A[e] + precise::tanh(g) * MS[e];
    return;
}
const uint e = thread_position_in_grid.x;
const uint j = e / (S * D);
const uint row = (e / D) % S;
const uint i = e % D;
float f = 0.0f, g = 0.0f;
for (uint r = 0; r < RANK; r++) {
    const float s = float(ST[(j * S + row) * RANK + r]);
    f += float(F[(j * D + i) * RANK + r]) * s;
    g += float(G[(j * D + i) * RANK + r]) * s;
}
// The precise tanh: the fast one is NaN at an infinity and a percent off near zero.
Y[e] = (1.0f - precise::tanh(f)) * A[e] + precise::tanh(g) * MS[e];
