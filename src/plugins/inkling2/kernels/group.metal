// Inkling 2, a few rows: group N = S * KSEL picks IDX [N] u32 by expert into tiles of at most BM = 4 picks each.
// grid (TG, 1, 1), threadgroup (TG, 1, 1) with TG = ceil(E / 32) * 32.
// Outputs: TEXP [N] u32 (expert id or 0xFFFFFFFF), TCNT [N] u32 (1..4), TPICK [N * 4] u32.
constexpr uint NSG = TG / 32;
threadgroup uint idx_s[N];
threadgroup uint sg_sums[NSG];
const uint tid = thread_position_in_threadgroup.x;
const uint sg = simdgroup_index_in_threadgroup;
const uint lane = thread_index_in_simdgroup;
for (uint p = tid; p < N; p += TG) idx_s[p] = IDX[p];
threadgroup_barrier(mem_flags::mem_threadgroup);
uint cnt = 0;
if (tid < E) {
    for (uint p = 0; p < N; ++p) cnt += (idx_s[p] == tid) ? 1u : 0u;
}
const uint my_tiles = (cnt + 3u) >> 2;
const uint sg_off = simd_prefix_exclusive_sum(my_tiles);
const uint sg_tot = simd_sum(my_tiles);
if (lane == 0) sg_sums[sg] = sg_tot;
threadgroup_barrier(mem_flags::mem_threadgroup);
uint sg_base = 0;
uint tot = 0;
for (uint s = 0; s < NSG; ++s) {
    if (s < sg) sg_base += sg_sums[s];
    tot += sg_sums[s];
}
if (tid < E && cnt > 0u) {
    const uint base = sg_base + sg_off;
    for (uint t = 0; t < my_tiles; ++t) {
        const uint off = base + t;
        TEXP[off] = tid;
        TCNT[off] = min(4u, cnt - (t << 2));
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
