const uint tid = thread_position_in_threadgroup.x;
const uint lane = tid % 32;
const uint sg = tid / 32;
const int q_pos = dims[0];
const int P = dims[1];
threadgroup atomic_uint hist[256];
threadgroup uint st[3];
threadgroup uint part_g[32];
threadgroup uint part_e[32];
if (tid == 0) { st[0] = 0u; st[1] = 0u; st[2] = uint(SEL_K); }
for (int pass = 0; pass < 4; ++pass) {
    const uint shift = uint(24 - 8 * pass);
    if (tid < 256u) atomic_store_explicit(&hist[tid], 0u, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint prefix = st[0];
    const uint pmask = st[1];
    for (int i = int(tid); i < P; i += 1024) {
        const uint k = msv_score_key(scores[i]);
        if ((k & pmask) == prefix) atomic_fetch_add_explicit(&hist[(k >> shift) & 255u], 1u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        const uint need = st[2];
        uint cum = 0u;
        int bin = 255;
        for (; bin > 0; --bin) {
            const uint c = atomic_load_explicit(&hist[bin], memory_order_relaxed);
            if (cum + c >= need) break;
            cum += c;
        }
        st[2] = need - cum;
        st[0] = prefix | (uint(bin) << shift);
        st[1] = pmask | (255u << shift);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}
const uint thr = st[0];
const uint need = st[2];
const int seg = (P + 1023) / 1024;
const int s0 = int(tid) * seg;
const int s1 = metal::min(s0 + seg, P);
uint g = 0u, e = 0u;
for (int i = s0; i < s1; ++i) {
    const uint k = msv_score_key(scores[i]);
    g += k > thr ? 1u : 0u;
    e += k == thr ? 1u : 0u;
}
const uint pg = simd_prefix_exclusive_sum(g);
const uint pe = simd_prefix_exclusive_sum(e);
if (lane == 31) { part_g[sg] = pg + g; part_e[sg] = pe + e; }
threadgroup_barrier(mem_flags::mem_threadgroup);
if (sg == 0) {
    const uint a = part_g[lane];
    const uint c = part_e[lane];
    part_g[lane] = simd_prefix_exclusive_sum(a);
    part_e[lane] = simd_prefix_exclusive_sum(c);
}
threadgroup_barrier(mem_flags::mem_threadgroup);
uint er = part_e[sg] + pe;
uint pos = part_g[sg] + pg + metal::min(er, need);
for (int i = s0; i < s1; ++i) {
    const uint k = msv_score_key(scores[i]);
    bool take = k > thr;
    if (k == thr) { take = er < need; ++er; }
    if (take) {
        const bool valid = (i + 1) * KPOOL - 1 <= q_pos;
        for (int c = 0; c < KPOOL; ++c) out[int(pos) * KPOOL + c] = valid ? i * KPOOL + c : -1;
        ++pos;
    }
}
for (int col = SEL_K * KPOOL + int(tid); col < OUT_W; col += 1024) {
    int v = -1;
    if (col >= TOPK_W && col < TOPK_W + TAIL_W) {
        const int t = col - TOPK_W;
        const int tail_count = (q_pos + 1) % KPOOL;
        if (t < tail_count) v = q_pos + 1 - tail_count + t;
    }
    out[col] = v;
}