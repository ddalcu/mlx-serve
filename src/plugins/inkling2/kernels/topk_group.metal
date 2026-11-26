// Inkling 2, a few rows: fused topK router selection + expert tile grouping in one launch (bit-identical to
// topk.metal + group.metal). Ranks up to RPAR = TG / E rows in parallel, then groups all N = S * KSEL picks into
// tiles of at most 4 picks of one expert. grid (TG, 1, 1), threadgroup (TG, 1, 1) with TG = RPAR * E.
// Outputs: IDX [1, S, KSEL] u32, WTS [1, S, KSEL] f32, TEXP [N] u32, TCNT [N] u32, TPICK [N * 4] u32.
constexpr uint NSG = TG / 32;
constexpr uint RPAR = TG / E;
threadgroup float key[RPAR * E];
threadgroup float raw[RPAR * KSEL];
threadgroup uint sel_r[RPAR * KSEL];
threadgroup uint idx_s[MAXN];
threadgroup uint sg_sums[NSG];
const uint tid = thread_position_in_threadgroup.x;
const uint sg = simdgroup_index_in_threadgroup;
const uint lane = thread_index_in_simdgroup;
const uint S = SC_shape[0];
const uint N = S * KSEL;
const uint slot_r = tid / E;
const uint e = tid % E;
const float rb_e = RB[e];
for (uint r0 = 0; r0 < S; r0 += RPAR) {
    const uint r = r0 + slot_r;
    float sig = 0.0f;
    if (r < S) {
        sig = 1.0f / (1.0f + exp(-SC[r * E + e]));
        key[slot_r * E + e] = sig + rb_e;
        if (e < KSEL) { sel_r[slot_r * KSEL + e] = e; raw[slot_r * KSEL + e] = 0.0f; }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (r < S) {
        const float mine = key[slot_r * E + e];
        uint rank = 0;
        const uint kb = slot_r * E;
        for (uint f = 0; f < E; ++f) rank += (key[kb + f] > mine || (key[kb + f] == mine && f < e)) ? 1u : 0u;
        if (rank < KSEL && mine == mine) { sel_r[slot_r * KSEL + rank] = e; raw[slot_r * KSEL + rank] = sig; }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (r < S && e < KSEL) {
        float tot = 0.0f;
        for (uint k = 0; k < KSEL; ++k) tot += raw[slot_r * KSEL + k];
        const uint ex = sel_r[slot_r * KSEL + e];
        idx_s[r * KSEL + e] = ex;
        IDX[r * KSEL + e] = ex;
        WTS[r * KSEL + e] = raw[slot_r * KSEL + e] / (tot + 1e-20f) * PR[0];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}
uint cnt = 0;
if (tid < E) {
    for (uint p = 0; p < N; ++p) cnt += (idx_s[p] == tid) ? 1u : 0u;
}
const uint my_tiles = (cnt + 3u) >> 2;
const uint sg_off = simd_prefix_exclusive_sum(my_tiles);
const uint sg_tot = simd_sum(my_tiles);
if (lane == 0) sg_sums[sg] = sg_tot;
threadgroup_barrier(mem_flags::mem_threadgroup);
uint sg_base = 0, tot = 0;
for (uint s = 0; s < NSG; ++s) {
    if (s < sg) sg_base += sg_sums[s];
    tot += sg_sums[s];
}
if (tid < E && cnt > 0u) {
    const uint base = sg_base + sg_off;
    for (uint t = 0; t < my_tiles; ++t) {
        TEXP[base + t] = tid;
        TCNT[base + t] = min(4u, cnt - (t << 2));
    }
    uint k = 0;
    for (uint p = 0; p < N; ++p) {
        if (idx_s[p] == tid) {
            TPICK[((base + (k >> 2)) << 2) + (k & 3u)] = p;
            ++k;
        }
    }
}
for (uint t = tot + tid; t < N; t += TG) {
    TEXP[t] = 0xFFFFFFFFu;
    TCNT[t] = 0u;
}
