// Inkling 2: a prompt chunk's J loop message gates as one tiled kernel (`MLX_SERVE_INKLING2_GATE_MM`). A, MS [J, S, D]
// f32 the gated streams and their messages, ST [J, S, RANK] each gate's state, F and G [J, D, RANK] the forget and
// message projections (their stored float dtype). A threadgroup of NT threads (BM / 16 x 2 SIMD groups) takes a BM-row
// by 32-column tile of one gate: the states' and both projections' slices staged as exact f32 in threadgroup
// memory, F st and G st on the matrix units (simdgroup_float8x8, f32 accumulate over RANK in 8-wide steps), then
// Y [J, S, D] f32 = (1 - tanh(F st)) A + tanh(G st) MS from the staged sums; no [J, S, D] product is written out.
// grid (NT * D / 32, ceil(S / BM), J), threadgroup (NT, 1, 1). D and RANK multiples of 32.
// BM: the tile's rows (64 served: eight SIMD groups, 256 threads). Each output's sum over RANK runs the same 8-wide
// steps in the same order at any BM, so the same bits; a taller tile stages each projection slice for more rows.
constexpr uint NT = (BM / 16) * 2 * 32;
const uint t = thread_position_in_threadgroup.x;
const uint S = ST_shape[1];
const uint c0 = threadgroup_position_in_grid.x * 32;
const uint r0 = threadgroup_position_in_grid.y * BM;
const uint j = threadgroup_position_in_grid.z;
const uint sg = t / 32;
const uint sr = (sg / 2) * 16;
const uint sc = (sg % 2) * 16;
threadgroup float ts[BM * 32];
threadgroup float tf[32 * 32];
threadgroup float tq[BM * 32];
simdgroup_float8x8 af[2][2];
simdgroup_float8x8 aq[2][2];
for (uint m = 0; m < 2; m++) for (uint n = 0; n < 2; n++) {
    af[m][n] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    aq[m][n] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
}
for (uint k0 = 0; k0 < RANK; k0 += 32) {
    for (uint e = t; e < BM * 32; e += NT) {
        const uint a = e / 32, k = e % 32;
        const uint row = r0 + a;
        ts[e] = row < S ? float(ST[(j * S + row) * RANK + k0 + k]) : 0.0f;
    }
    for (uint e = t; e < 1024; e += NT) {
        const uint a = e / 32, k = e % 32;
        tf[e] = float(F[(j * D + c0 + a) * RANK + k0 + k]);
        tq[e] = float(G[(j * D + c0 + a) * RANK + k0 + k]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint kk = 0; kk < 32; kk += 8) {
        simdgroup_float8x8 x[2];
        simdgroup_float8x8 bf[2];
        simdgroup_float8x8 bq[2];
        for (uint m = 0; m < 2; m++) simdgroup_load(x[m], ts + (sr + m * 8) * 32 + kk, 32);
        // A projection's rows are output columns: loaded transposed, B[k][c] = F[c][k].
        for (uint n = 0; n < 2; n++) {
            simdgroup_load(bf[n], tf + (sc + n * 8) * 32 + kk, 32, ulong2(0, 0), true);
            simdgroup_load(bq[n], tq + (sc + n * 8) * 32 + kk, 32, ulong2(0, 0), true);
        }
        for (uint m = 0; m < 2; m++) for (uint n = 0; n < 2; n++) {
            simdgroup_multiply_accumulate(af[m][n], x[m], bf[n], af[m][n]);
            simdgroup_multiply_accumulate(aq[m][n], x[m], bq[n], aq[m][n]);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}
for (uint m = 0; m < 2; m++) for (uint n = 0; n < 2; n++) {
    simdgroup_store(af[m][n], ts + (sr + m * 8) * 32 + sc + n * 8, 32);
    simdgroup_store(aq[m][n], tq + (sr + m * 8) * 32 + sc + n * 8, 32);
}
threadgroup_barrier(mem_flags::mem_threadgroup);
for (uint e = t; e < BM * 32; e += NT) {
    const uint a = e / 32, c = e % 32;
    const uint row = r0 + a;
    if (row >= S) continue;
    const uint at = (j * S + row) * D + c0 + c;
    // The precise tanh, as gate.metal.
    Y[at] = (1.0f - precise::tanh(ts[e])) * A[at] + precise::tanh(tq[e]) * MS[at];
}
