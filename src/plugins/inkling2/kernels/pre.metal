// Inkling 2: a visit's stream to its attention inputs for S rows. grid (D, S), threadgroup (TG, 1), TG a multiple of
// 32 that divides D: one thread per stream element of a row. A sum over a row's stream is redone by each of its
// threadgroups: each thread takes the elements TG apart, simd_sum folds a SIMD group, and thread r folds sum r over
// the groups once (in group order) for the rest to read, so no thread folds every sum.
// HIN [1, S, D] is the stream; BLK [1, S, NB, D] f32 the open depth blocks; QY [D, Q] f32 the layer's depth queries
// (Q 0 = the input is the stream itself); NW0..NW2 [D] the J input norms' weights. Input j of a row is the softmax
// of query min(j, Q - 1) over the row's blocks and its stream's unweighted norm, mixing them.
// MSG 0: X [J, S, D] T = each input under its weighted RMS norm (eps PR[0]).
// MSG 1 (a later loop pass): BANK [1, S, NBK, D] f32 and BPART [1, S, D] f32 are the pass before's blocks and normed
// output and MQ [D, 4] f32 the layer's message queries. A [3, S, D] f32 holds the inputs unnormed, MSGO [4, S, D] f32
// the q, k, v and mlp messages (each query's softmax mix of BANK and BPART), BOTH [3, S, 2 D] GT each gate's input:
// the input's and its message's unweighted norms, side by side.
// NORMED [1, S, D] f32 = the stream's unweighted norm.
const uint t = thread_position_in_threadgroup.x;
const uint e = thread_position_in_grid.x;
const uint row = thread_position_in_grid.y;
const uint S = HIN_shape[1];
const uint ro = row * D;
const uint lane = thread_index_in_simdgroup;
const uint sg = t / 32;
constexpr uint NSG = TG / 32;
constexpr uint NS = NB + 1;
constexpr uint NM = NBK + 1;
constexpr uint QC = Q > 0 ? Q : 1;
constexpr uint R1 = 1 + Q * NS + MSG * 4 * NM;
constexpr uint R2 = QC + MSG * 4;
constexpr uint MB = 1 + Q * NS;
threadgroup float red1[NSG * R1];
threadgroup float red2[NSG * R2];
threadgroup float fold1[R1];
threadgroup float fold2[R2];
float acc[R1];
for (uint r = 0; r < R1; r++) acc[r] = 0.0f;
for (uint i = t; i < D; i += TG) {
    const float h = float(HIN[ro + i]);
    acc[0] += h * h;
    for (uint j = 0; j < Q; j++) {
        const float q = float(QY[i * Q + j]);
        for (uint n = 0; n < NB; n++) acc[1 + j * NS + n] += float(BLK[(row * NB + n) * D + i]) * q;
        acc[1 + j * NS + NB] += h * q;
    }
    if (MSG) for (uint j = 0; j < 4; j++) {
        const float q = float(MQ[i * 4 + j]);
        for (uint n = 0; n < NBK; n++) acc[MB + j * NM + n] += float(BANK[(row * NBK + n) * D + i]) * q;
        acc[MB + j * NM + NBK] += float(BPART[ro + i]) * q;
    }
}
for (uint r = 0; r < R1; r++) {
    acc[r] = simd_sum(acc[r]);
    if (lane == 0) red1[sg * R1 + r] = acc[r];
}
threadgroup_barrier(mem_flags::mem_threadgroup);
for (uint r = t; r < R1; r += TG) {
    float s = 0.0f;
    for (uint g = 0; g < NSG; g++) s += red1[g * R1 + r];
    fold1[r] = s;
}
threadgroup_barrier(mem_flags::mem_threadgroup);
for (uint r = 0; r < R1; r++) acc[r] = fold1[r];
const float hs = metal::precise::rsqrt(acc[0] / float(D) + 1e-6f);
// Each query's softmax over its sources, the live one last.
float w[QC * NS];
for (uint j = 0; j < Q; j++) {
    acc[1 + j * NS + NB] *= hs;
    float mx = acc[1 + j * NS];
    for (uint n = 1; n < NS; n++) mx = max(mx, acc[1 + j * NS + n]);
    float tot = 0.0f;
    for (uint n = 0; n < NS; n++) { w[j * NS + n] = exp(acc[1 + j * NS + n] - mx); tot += w[j * NS + n]; }
    for (uint n = 0; n < NS; n++) w[j * NS + n] /= tot;
}
float wm[4 * NM];
if (MSG) for (uint j = 0; j < 4; j++) {
    float mx = acc[MB + j * NM];
    for (uint n = 1; n < NM; n++) mx = max(mx, acc[MB + j * NM + n]);
    float tot = 0.0f;
    for (uint n = 0; n < NM; n++) { wm[j * NM + n] = exp(acc[MB + j * NM + n] - mx); tot += wm[j * NM + n]; }
    for (uint n = 0; n < NM; n++) wm[j * NM + n] /= tot;
}
float ss[R2], mine[R2];
for (uint r = 0; r < R2; r++) ss[r] = 0.0f;
for (uint i = t; i < D; i += TG) {
    const float h = float(HIN[ro + i]);
    for (uint j = 0; j < QC; j++) {
        float a = h;
        if (Q > 0) {
            a = 0.0f;
            for (uint n = 0; n < NB; n++) a += w[j * NS + n] * float(BLK[(row * NB + n) * D + i]);
            a += w[j * NS + NB] * (h * hs);
        }
        ss[j] += a * a;
        if (i == e) mine[j] = a;
    }
    if (MSG) for (uint j = 0; j < 4; j++) {
        float m = 0.0f;
        for (uint n = 0; n < NBK; n++) m += wm[j * NM + n] * float(BANK[(row * NBK + n) * D + i]);
        m += wm[j * NM + NBK] * float(BPART[ro + i]);
        ss[QC + j] += m * m;
        if (i == e) mine[QC + j] = m;
    }
}
for (uint r = 0; r < R2; r++) {
    ss[r] = simd_sum(ss[r]);
    if (lane == 0) red2[sg * R2 + r] = ss[r];
}
threadgroup_barrier(mem_flags::mem_threadgroup);
for (uint r = t; r < R2; r += TG) {
    float s = 0.0f;
    for (uint g = 0; g < NSG; g++) s += red2[g * R2 + r];
    fold2[r] = s;
}
threadgroup_barrier(mem_flags::mem_threadgroup);
for (uint r = 0; r < R2; r++) ss[r] = fold2[r] / float(D);
NORMED[ro + e] = float(HIN[ro + e]) * hs;
if (MSG) {
    for (uint j = 0; j < 3; j++) {
        const uint c = min(j, QC - 1);
        A[(j * S + row) * D + e] = mine[c];
        BOTH[(j * S + row) * 2 * D + e] = GT(mine[c] * metal::precise::rsqrt(ss[c] + 1e-6f));
        BOTH[(j * S + row) * 2 * D + D + e] = GT(mine[QC + j] * metal::precise::rsqrt(ss[QC + j] + 1e-6f));
    }
    for (uint j = 0; j < 4; j++) MSGO[(j * S + row) * D + e] = mine[QC + j];
} else {
    for (uint j = 0; j < J; j++) {
        const uint c = min(j, QC - 1);
        const float nw = float(j == 0 ? NW0[e] : (j == 1 ? NW1[e] : NW2[e]));
        X[(j * S + row) * D + e] = T(mine[c] * metal::precise::rsqrt(ss[c] + PR[0]) * nw);
    }
}
