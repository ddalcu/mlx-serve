// Inkling 2: a visit between its output projection and its router for S rows. grid (D, S), threadgroup (TG, 1): one
// thread per stream element of a row; sums over the stream as pre.metal's. OP [1, S, D] (T) is the output
// projection; its short conv runs over the rows behind the window before them (columns OFF.. of ST [1, KT - 1, SW]
// f32; SEQ 0: each row behind its own window, ST [S, KT - 1, SW]) with AW [D, KT, 1] (WT), the sum rounded to WT as the conv op does, plus OP, times PR[0], rounded to T. RES
// [1, S, D] f32 is that plus the stream HIN [1, S, D], or that alone where the layer opens a block (BOUND), whose
// saved stream NORMED [1, S, D] f32 then joins the blocks BLK [1, S, NB, D] f32. The MLP's input is the softmax of QY
// [D] f32 over a row's blocks and its RES's unweighted norm, mixing them.
// MSG 0: X [1, S, D] XT = that under its RMS norm with PN [D] (eps PR[1]); X32 the same values in f32.
// MSG 1: M [1, S, D] f32 = it unnormed; BOTH [1, S, 2 D] GT = its and the mlp message MSG3 [1, S, D] f32's
// unweighted norms.
const uint t = thread_position_in_threadgroup.x;
const uint e = thread_position_in_grid.x;
const uint row = thread_position_in_grid.y;
const uint ro = row * D;
const uint lane = thread_index_in_simdgroup;
const uint sg = t / 32;
constexpr uint NSG = TG / 32;
constexpr uint NS = NB + BOUND + 1;
constexpr uint R1 = 1 + NS + MSG;
const uint SW = ST_shape[2];
const uint RPS = OP_shape[1] / ST_shape[0];
const uint slot = row / RPS;
const uint pos = row % RPS;
threadgroup float red1[NSG * R1];
threadgroup float red2[NSG];
// The folds of the SIMD groups' partial sums, each by one thread in the order every thread used to run it.
threadgroup float fold1[R1];
threadgroup float fold2[1];
float acc[R1];
float res[D / TG];
for (uint r = 0; r < R1; r++) acc[r] = 0.0f;
for (uint i = t, k = 0; i < D; i += TG, k++) {
    const float x = float(OP[ro + i]);
    float y = 0.0f;
    // Tap j reads row `pos + j` of [the slot's ST | its rows], or walks PARENT on a tree.
    for (uint j = 0; j < KT; j++) {
        int p;
        if constexpr (TREE) {
            p = int(pos);
            for (uint stp = 0; stp < KT - 1 - j; ++stp) p = (p >= 0) ? PARENT[slot * RPS + uint(p)] : (p - 1);
        } else {
            p = int(pos) + int(j) - int(KT - 1);
        }
        const float wv = (p < 0) ? ST[(slot * (KT - 1) + uint(int(KT - 1) + p)) * SW + OFF + i] : float(OP[(slot * RPS + uint(p)) * D + i]);
        y += wv * float(AW[i * KT + j]);
    }
    y = float(WT(y));
    float r = float(T((y + x) * PR[0]));
    if (!BOUND) r += float(HIN[ro + i]);
    res[k] = r;
    const float q = float(QY[i]);
    acc[0] += r * r;
    for (uint n = 0; n < NB; n++) acc[1 + n] += float(BLK[(row * NB + n) * D + i]) * q;
    if (BOUND) acc[1 + NB] += float(NORMED[ro + i]) * q;
    acc[NS] += r * q;
    if (MSG) acc[1 + NS] += float(MSG3[ro + i]) * float(MSG3[ro + i]);
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
const float rs = metal::precise::rsqrt(acc[0] / float(D) + 1e-6f);
acc[NS] *= rs;
float w[NS];
float mx = acc[1];
for (uint n = 1; n < NS; n++) mx = max(mx, acc[1 + n]);
float tot = 0.0f;
for (uint n = 0; n < NS; n++) { w[n] = exp(acc[1 + n] - mx); tot += w[n]; }
for (uint n = 0; n < NS; n++) w[n] /= tot;
constexpr uint ELEMS = ONE_TG ? (D / TG) : 1;
float ss = 0.0f;
float mine[ELEMS];
for (uint i = t, k = 0; i < D; i += TG, k++) {
    float m = 0.0f;
    for (uint n = 0; n < NB; n++) m += w[n] * float(BLK[(row * NB + n) * D + i]);
    if (BOUND) m += w[NB] * float(NORMED[ro + i]);
    m += w[NS - 1] * (res[k] * rs);
    ss += m * m;
    if (ONE_TG) mine[k] = m;
    else if (i == e) mine[0] = m;
}
ss = simd_sum(ss);
if (lane == 0) red2[sg] = ss;
threadgroup_barrier(mem_flags::mem_threadgroup);
if (t == 0) {
    float s = 0.0f;
    for (uint g = 0; g < NSG; g++) s += red2[g];
    fold2[0] = s;
}
threadgroup_barrier(mem_flags::mem_threadgroup);
ss = fold2[0] / float(D);
if constexpr (ONE_TG) {
    for (uint i = t, k = 0; i < D; i += TG, k++) {
        RES[ro + i] = res[k];
        if (MSG) {
            M[ro + i] = mine[k];
            BOTH[2 * ro + i] = GT(mine[k] * metal::precise::rsqrt(ss + 1e-6f));
            BOTH[2 * ro + D + i] = GT(float(MSG3[ro + i]) * metal::precise::rsqrt(acc[1 + NS] / float(D) + 1e-6f));
        } else {
            const XT x = XT(mine[k] * metal::precise::rsqrt(ss + PR[1]) * float(PN[i]));
            X[ro + i] = x;
            X32[ro + i] = float(x);
        }
    }
} else {
    RES[ro + e] = res[e / TG];
    if (MSG) {
        M[ro + e] = mine[0];
        BOTH[2 * ro + e] = GT(mine[0] * metal::precise::rsqrt(ss + 1e-6f));
        BOTH[2 * ro + D + e] = GT(float(MSG3[ro + e]) * metal::precise::rsqrt(acc[1 + NS] / float(D) + 1e-6f));
    } else {
        const XT x = XT(mine[0] * metal::precise::rsqrt(ss + PR[1]) * float(PN[e]));
        X[ro + e] = x;
        X32[ro + e] = float(x);
    }
}
