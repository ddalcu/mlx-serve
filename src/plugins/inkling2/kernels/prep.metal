// Inkling 2: what the attention read needs from a visit's projections for S rows. grid (H * HD + 2 * HKV * HD + the
// tail, S), threadgroup (HD, 1): one thread per output element of a row, a head of q, k or v one threadgroup whose
// threads each square their own element into BUF for the group to sum, so no thread loops over a head's inputs.
// QP, GP [1, S, H * HD], RP [1, S, H * DREL], KP, VP [1, S, HKV * HD] are the q, gate, relative, k and v projections
// (T); the k and v convs run over the rows behind the windows before them (ST [NSLOT, KT - 1, >= 2 * HKV * HD] f32, k
// then v): RPS rows of ROWS make a slot, each slot behind its own window (RPS == ROWS: one sequence; 1: every row a sequence of
// its own). Static convs take KW, VW [C, KT, 1] (WT) and round their sum to WT as the conv op does; dynamic ones (DYN)
// take KW, VW [KT, C] (WT) plus each head's projected taps KDY, VDY [1, S, HKV * KT] (T), tap `lag` on the input
// `lag` rows back. QS [H, HD] and RS [H, DREL] (S) scale q and the relative states before their L2 clips.
// PR = { qk clip, rel clip, eps, tau, conv scale, KV }.
// Q [1, S, H, HD] T (clipped, times tau), GATE [S * H * HD] f32 (sigmoid), K, V [1, HKV, S, HD] T (conv, then the L2
// clip or the unweighted RMS norm). The tail is, WIDE: REL [1, S, H, DREL] T, the clipped relative states; else (one
// row): BIAS [1, H, 1, KV] T, the row's additive bias over its KV keys, the newest last (the clipped states through
// RJ [DREL, R] f32, distance 0 last, times tau; zero for a key past the extent R).
const uint i = thread_position_in_grid.x;
const uint row = thread_position_in_grid.y;
const uint ROWS = QP_shape[1];
const uint RPS = ROWS / ST_shape[0];
const uint pos = row % RPS;
const uint t = thread_position_in_threadgroup.x;
const uint QD = H * HD;
const uint C = HKV * HD;
const uint R = RJ_shape[1];
const uint SW = ST_shape[2];
const float qk_thr = PR[0], rel_thr = PR[1], eps = PR[2], tau = PR[3], cs = PR[4];
threadgroup float buf[HD];
threadgroup float n2_sh;
if (i < QD) {
    const uint at = row * QD + i;
    const float mine = float(QP[at]) * float(QS[i]);
    buf[t] = mine * mine;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (t == 0) {
        float s = 0.0f;
        for (uint j = 0; j < HD; j++) s += buf[j];
        n2_sh = s;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float n2 = n2_sh;
    const float f = min(qk_thr / (sqrt(n2) + 1e-6f), 1.0f);
    T q = T(mine * f);
    if (tau != 1.0f) q = T(float(q) * tau);
    Q[at] = q;
    GATE[at] = 1.0f / (1.0f + exp(-float(GP[at])));
    return;
}
uint r = i - QD;
if (r < 2 * C) {
    const bool is_v = r >= C;
    const uint c = is_v ? r - C : r;
    const uint h = c / HD;
    const uint off = is_v ? C : 0u;
    const float x = float(is_v ? VP[row * C + c] : KP[row * C + c]);
    const uint slot = row / RPS;
    float y = 0.0f;
    for (uint j = 0; j < KT; j++) {
        // Tap j reads row `pos + j` of [the slot's ST | its rows], or walks PARENT on a tree.
        int p;
        if constexpr (TREE) {
            p = int(pos);
            for (uint stp = 0; stp < KT - 1 - j; ++stp) p = (p >= 0) ? PARENT[slot * RPS + uint(p)] : (p - 1);
        } else {
            p = int(pos) + int(j) - int(KT - 1);
        }
        float wv = x;
        if (p < 0) wv = ST[(slot * (KT - 1) + uint(int(KT - 1) + p)) * SW + off + c];
        else if (j + 1 < KT) wv = float(is_v ? VP[(slot * RPS + uint(p)) * C + c] : KP[(slot * RPS + uint(p)) * C + c]);
        float wt;
        if (DYN) {
            const uint lag = KT - 1 - j;
            const uint tap = (row * HKV + h) * KT + lag;
            wt = float(is_v ? VW[lag * C + c] : KW[lag * C + c]) + float(is_v ? VDY[tap] : KDY[tap]);
        } else {
            wt = float(is_v ? VW[c * KT + j] : KW[c * KT + j]);
        }
        y += wv * wt;
    }
    if (!DYN) y = float(WT(y));
    const float mine = float(T((y + x) * cs));
    buf[t] = mine * mine;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (t == 0) {
        float s = 0.0f;
        for (uint j = 0; j < HD; j++) s += buf[j];
        n2_sh = s;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float n2 = n2_sh;
    const uint at = (h * ROWS + row) * HD + c % HD;
    if (is_v) V[at] = T(mine * rsqrt(n2 / float(HD) + eps));
    else K[at] = T(mine * min(qk_thr / (sqrt(n2) + 1e-6f), 1.0f));
    return;
}
r -= 2 * C;
// A head's relative states, clipped.
const uint KV = WIDE ? 0u : uint(PR[5]);
const uint h = WIDE ? r / DREL : r / KV;
if (!WIDE) {
    const uint col = r % KV;
    // Key `col` sits R - KV + col columns into the logits; keys past R are zero.
    if (col + R < KV) {
        BIAS[r] = T(0.0f);
        return;
    }
}
const uint rp = row * H * DREL + h * DREL;
float n2 = 0.0f;
for (uint j = 0; j < DREL; j++) {
    const float v = float(RP[rp + j]) * float(RS[h * DREL + j]);
    n2 += v * v;
}
const float f = min(rel_thr / (sqrt(n2) + 1e-6f), 1.0f);
if (WIDE) {
    const uint j = r % DREL;
    REL[rp + j] = T(float(RP[rp + j]) * float(RS[h * DREL + j]) * f);
    return;
}
const uint col = r % KV;
const uint d = col + R - KV;
float acc = 0.0f;
for (uint j = 0; j < DREL; j++) acc += float(T(float(RP[rp + j]) * float(RS[h * DREL + j]) * f)) * RJ[j * R + d];
BIAS[r] = T(acc * tau);
