// GLM-5.3-Flash's DSA indexer over a prompt pass's query rows (`dsa_prompt.zig`), the model's dims fixed here: 32
// indexer heads of 128, pools of 4 tokens (a pool's key in its last row's gate half of the 256-wide cache row), the 512
// best pools (2,048 tokens), then the row's unfinished pool's tokens (2,051 slots a row).
//
// A threadgroup (8 simdgroups) takes query rows g, g + G, g + 2G, ... of the launch's G groups, one at a time:
// 1. The row's scores, each as kernels/dsa_indexer_nax.metal computes it: q_h . k_p as the same 16x32x16 MMAs on the
//    same operands (q rows on the left, pool keys transposed on the right) over k in steps of 16 from zero, the row's
//    32 heads the MMA's rows (two tiles of 16) where the NAX text has 32 query rows of one head; then relu(.) * w_h
//    added over the heads in order from zero, in f32, the NAX text's statement. Simdgroup j takes the 32-pool blocks
//    j, j + 8, ... the row can see; each block's head sums go through threadgroup memory a quarter of the heads at a
//    time, a pool a lane. The scores go to the group's scratch row; a pool the row cannot see yet (not complete by its
//    position) is never scored: it scores -1e30, the NAX text's mask.
// 2. The 512 best over the scratch row: a 4-pass 8-bit radix select of the 512th best key (`glm5_dsa_score_key`:
//    MLX's GPU sort order) and how many of its equals to take (the pools the row cannot see counted at once), then an
//    ordered compaction in pool order (ties to the lower pool) writing each pool's 4 tokens (-1 for a pool not complete
//    by the row), then the row's unfinished pool's tokens.
constexpr int H = 32;
constexpr int D = 128;
constexpr int ROW_W = 2 * D;
constexpr int KPOOL = 4;
constexpr int SEL_K = 512;
constexpr int TOPK_W = SEL_K * KPOOL;
constexpr int TAIL_W = KPOOL - 1;
constexpr int OUT_W = TOPK_W + TAIL_W;
constexpr int NSG = 8;
constexpr int NT = NSG * 32;
constexpr int BP = 32;
constexpr int SPAD = BP + 1;
constexpr float MASKED = -1e30f;

const int S = dims[0];
const int q_start = dims[1];
const int P = dims[2];
const int G = dims[3];
const uint sg = simdgroup_index_in_threadgroup;
const uint lane = thread_index_in_simdgroup;
const uint tid = sg * 32u + lane;
const int grp = int(threadgroup_position_in_grid.x);

// The NAX text's fragment coordinates.
const short qid = short(lane >> 2);
const short fm = short((qid & 4) | ((lane >> 1) & 3));
const short fn = short(((qid & 2) | (lane & 1)) * 4);

threadgroup float stg[NSG][8][SPAD];
threadgroup float wq[H];
threadgroup atomic_uint hist[256];
threadgroup uint st[3];
threadgroup uint above_sg[NSG];
threadgroup uint part_g[NSG];
threadgroup uint part_e[NSG];

device float* srow = scr + size_t(grp) * size_t(P);
const uint km = glm5_dsa_score_key(MASKED);

constexpr auto desc = matmul2d_descriptor(
    16, 32, 16, false, true, true,
    matmul2d_descriptor::mode::multiply_accumulate);
matmul2d<desc, execution_simdgroup> op;

// A quarter of a block's heads into threadgroup memory: the rows fm + 8 * I of tile C (its 8 heads), a pool a column.
#define GLM5_DSA_STAGE(C, I)                                                   \
    for (short tn = 0; tn < 2; ++tn) {                                         \
        for (short j = 0; j < 4; ++j) {                                        \
            stg[sg][fm][tn * 16 + fn + j] = C[tn * 8 + (I) * 4 + j];           \
        }                                                                      \
    }                                                                          \
    simdgroup_barrier(mem_flags::mem_threadgroup);
// The quarter's 8 heads added to the lane's pool in head order (the NAX text's statement).
#define GLM5_DSA_SUM(H0)                                                       \
    for (short h = 0; h < 8; ++h) {                                            \
        acc += max(stg[sg][h][lane], 0.0f) * wq[(H0) + h];                     \
    }                                                                          \
    simdgroup_barrier(mem_flags::mem_threadgroup);

for (int qi = grp; qi < S; qi += G) {
    const int q_pos = q_start + qi;
    // the pools complete by the row's position: (p + 1) * KPOOL - 1 <= q_pos
    const int pv = metal::min(P, (q_pos + 1) / KPOOL);
    if (tid < uint(H)) {
        wq[tid] = w[size_t(qi) * H + tid];
    }
    if (tid == 0u) {
        st[0] = 0u;
        st[1] = 0u;
        st[2] = uint(SEL_K);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ---- 1. The scores of the pools the row can see.
    const device T* qa0 = q + size_t(qi) * (H * D) + size_t(fm) * D + fn;  // head fm
    const device T* qa1 = qa0 + 8 * D;                                         // head fm + 8
    const device T* qa2 = qa0 + 16 * D;                                        // head 16 + fm
    const device T* qa3 = qa0 + 24 * D;                                        // head 24 + fm
    for (int sn = int(sg) * BP; sn < pv; sn += NSG * BP) {
        const device T* kb[2][2];
        for (short tn = 0; tn < 2; ++tn) {
            for (short i = 0; i < 2; ++i) {
                const int pr = metal::min(sn + tn * 16 + fm + i * 8, P - 1);
                kb[tn][i] = cache + (size_t(pr) * KPOOL + (KPOOL - 1)) * ROW_W + D + fn;
            }
        }
        auto ct_a0 = op.template get_left_input_cooperative_tensor<T, T, float>();
        auto ct_a1 = op.template get_left_input_cooperative_tensor<T, T, float>();
        auto ct_b = op.template get_right_input_cooperative_tensor<T, T, float>();
        auto c0 = op.template get_destination_cooperative_tensor<
            metal::remove_addrspace_t<decltype(ct_a0)>,
            metal::remove_addrspace_t<decltype(ct_b)>, float>();
        auto c1 = op.template get_destination_cooperative_tensor<
            metal::remove_addrspace_t<decltype(ct_a0)>,
            metal::remove_addrspace_t<decltype(ct_b)>, float>();
        for (short e = 0; e < 16; ++e) {
            c0[e] = 0.0f;
            c1[e] = 0.0f;
        }
        for (short kk = 0; kk < D; kk += 16) {
            for (short j = 0; j < 4; ++j) {
                ct_a0[j] = qa0[kk + j];
                ct_a0[4 + j] = qa1[kk + j];
                ct_a1[j] = qa2[kk + j];
                ct_a1[4 + j] = qa3[kk + j];
            }
            for (short tn = 0; tn < 2; ++tn) {
                for (short i = 0; i < 2; ++i) {
                    for (short j = 0; j < 4; ++j) {
                        ct_b[tn * 8 + i * 4 + j] = kb[tn][i][kk + j];
                    }
                }
            }
            op.run(ct_a0, ct_b, c0);
            op.run(ct_a1, ct_b, c1);
        }
        float acc = 0.0f;
        GLM5_DSA_STAGE(c0, 0)
        GLM5_DSA_SUM(0)
        GLM5_DSA_STAGE(c0, 1)
        GLM5_DSA_SUM(8)
        GLM5_DSA_STAGE(c1, 0)
        GLM5_DSA_SUM(16)
        GLM5_DSA_STAGE(c1, 1)
        GLM5_DSA_SUM(24)
        const int p = sn + int(lane);
        if (p < pv) {
            srow[p] = acc;
        }
    }
    threadgroup_barrier(mem_flags::mem_device);

    // ---- 2. The 512 best pools: the radix select.
    for (int pass = 0; pass < 4; ++pass) {
        const uint shift = uint(24 - 8 * pass);
        atomic_store_explicit(&hist[tid], 0u, memory_order_relaxed);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const uint prefix = st[0];
        const uint pmask = st[1];
        const uint need = st[2];
        for (int i = int(tid); i < pv; i += NT) {
            const uint k = glm5_dsa_score_key(srow[i]);
            if ((k & pmask) == prefix) {
                atomic_fetch_add_explicit(&hist[(k >> shift) & 255u], 1u, memory_order_relaxed);
            }
        }
        // the pools the row cannot see, each MASKED, at once
        if (tid == 0u && pv < P && (km & pmask) == prefix) {
            atomic_fetch_add_explicit(&hist[(km >> shift) & 255u], uint(P - pv), memory_order_relaxed);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // thread t holds bin 255 - t: the bins above it are the threads before it
        const uint c = atomic_load_explicit(&hist[255u - tid], memory_order_relaxed);
        const uint pre = simd_prefix_exclusive_sum(c);
        if (lane == 31u) {
            above_sg[sg] = pre + c;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        uint above = pre;
        for (uint j = 0u; j < sg; ++j) {
            above += above_sg[j];
        }
        // the one bin whose count from the top reaches `need` (the highest such bin)
        if (above < need && above + c >= need) {
            st[2] = need - above;
            st[0] = prefix | ((255u - tid) << shift);
            st[1] = pmask | (255u << shift);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // ---- The ordered compaction and the tokens.
    const uint thr = st[0];
    const uint need = st[2];
    const int seg = (P + NT - 1) / NT;
    const int s0 = int(tid) * seg;
    const int s1 = metal::min(s0 + seg, P);
    uint gc = 0u;
    uint ec = 0u;
    for (int i = s0; i < s1; ++i) {
        const uint k = i < pv ? glm5_dsa_score_key(srow[i]) : km;
        gc += k > thr ? 1u : 0u;
        ec += k == thr ? 1u : 0u;
    }
    const uint pg = simd_prefix_exclusive_sum(gc);
    const uint pe = simd_prefix_exclusive_sum(ec);
    if (lane == 31u) {
        part_g[sg] = pg + gc;
        part_e[sg] = pe + ec;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0u) {
        const uint a = lane < uint(NSG) ? part_g[lane] : 0u;
        const uint b = lane < uint(NSG) ? part_e[lane] : 0u;
        const uint xa = simd_prefix_exclusive_sum(a);
        const uint xb = simd_prefix_exclusive_sum(b);
        if (lane < uint(NSG)) {
            part_g[lane] = xa;
            part_e[lane] = xb;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint er = part_e[sg] + pe;
    uint pos = part_g[sg] + pg + metal::min(er, need);
    device int* o = out + size_t(qi) * size_t(OUT_W);
    for (int i = s0; i < s1; ++i) {
        const uint k = i < pv ? glm5_dsa_score_key(srow[i]) : km;
        bool take = k > thr;
        if (k == thr) {
            take = er < need;
            ++er;
        }
        if (take) {
            const bool valid = (i + 1) * KPOOL - 1 <= q_pos;
            for (int c = 0; c < KPOOL; ++c) {
                o[int(pos) * KPOOL + c] = valid ? i * KPOOL + c : -1;
            }
            ++pos;
        }
    }
    if (tid < uint(TAIL_W)) {
        const int t = int(tid);
        const int tail_count = (q_pos + 1) % KPOOL;
        o[TOPK_W + t] = t < tail_count ? q_pos + 1 - tail_count + t : -1;
    }
    // the scratch row, the select's state and the weights are the next row's
    threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
}
#undef GLM5_DSA_STAGE
#undef GLM5_DSA_SUM
