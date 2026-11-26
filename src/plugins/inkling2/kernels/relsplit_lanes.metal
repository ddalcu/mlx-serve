// Inkling 2, 2 to 16 rows on non-NAX GPUs (M3/M4): relsplit_nax's 4-lane, single-pass, tree-aware relative attention
// using simdgroup_half8x8 matrix ops. Loads each 16-key K/V tile once into threadgroup memory for all G query heads
// and NR (1 or 2) 8-row fragments (up to 16 rows in one pass), up to 4 slots per launch, with DIRECT gated output
// when splits == 1 and inline GLOBAL log-scaling. E is the matrices' element: half, or float for a float32 stream.
// grid (splits * 32 * G, HKV, here), threadgroup (32 * G, 1, 1).
typedef typename msv_ink2_el<T>::type E;
constexpr float LOG2E = 1.4426950408889634f;
constexpr int BK = 16;
constexpr int LDK = BK + 8;
constexpr int LDV = D + 8;
constexpr int G = HQ / HKV;
const int zl = int(threadgroup_position_in_grid.z);
const int L = sizes[zl * 8];
if (L == 0) return;
const int S = sizes[zl * 8 + 1];
const int first = sizes[zl * 8 + 2];
const int SC = sizes[zl * 8 + 3];
const int offset = sizes[zl * 8 + 4];
const int row_in = sizes[zl * 8 + 5];
const int row_out = sizes[zl * 8 + 6];
const int kps = sizes[32];
const int log_n_floor = sizes[33];
const int LT = sizes[34];
const device T* kp[4] = {k0, k1, k2, k3};
const device T* vp[4] = {v0, v1, v2, v3};
const int64_t k_head[4] = {k0_strides[1], k1_strides[1], k2_strides[1], k3_strides[1]};
const int64_t k_row[4] = {k0_strides[2], k1_strides[2], k2_strides[2], k3_strides[2]};
const int split = int(threadgroup_position_in_grid.x);
const int kvh = int(threadgroup_position_in_grid.y);
const uint tid = thread_position_in_threadgroup.x;
const uint sg = simdgroup_index_in_threadgroup;
const uint lane = thread_index_in_simdgroup;
const int head = kvh * G + int(sg);
const float qscale = params[0] * LOG2E;
const float log_alpha = params[1];
const short fm = ((lane / 4) & 4) + (lane / 2) % 4;
const short fn = ((lane / 4) & 2) * 2 + (lane % 2) * 2;

bool row_ok[NR];
int r_depth[NR];
uint r_mask[NR];
int diag[NR];
float tau[NR];
const device T* bias_ptr[NR];
for (int fr = 0; fr < NR; ++fr) {
    const int r = fr * 8 + int(fm);
    row_ok[fr] = r < L;
    r_depth[fr] = r;
    r_mask[fr] = 0xFFFFFFFFu;
    if constexpr (TREE) {
        if (row_ok[fr]) {
            r_depth[fr] = TMETA[(row_in + r) * 2];
            r_mask[fr] = uint(TMETA[(row_in + r) * 2 + 1]);
        }
    }
    diag[fr] = S - L + r_depth[fr];
    tau[fr] = 1.0f;
    if constexpr (GLOBAL) {
        if (row_ok[fr]) {
            const int tp = offset + 1 + r_depth[fr];
            if (tp > log_n_floor) tau[fr] = 1.0f + log_alpha * metal::log(float(tp) / float(log_n_floor));
        }
    }
    bias_ptr[fr] = rel + (size_t(row_in + (row_ok[fr] ? r : 0)) * HQ + head) * R;
}

threadgroup E Ks[D * LDK];
threadgroup E Vs[BK * LDV];
simdgroup_matrix<E, 8, 8> q_frag[NR][D / 8];
for (int fr = 0; fr < NR; ++fr) {
    const int r = fr * 8 + int(fm);
    for (int db = 0; db < D / 8; ++db) {
        vec<E, 2> qv = vec<E, 2>(E(0));
        if (row_ok[fr]) {
            const device T* qp = q + (size_t(row_in + r) * HQ + head) * D + db * 8 + fn;
            qv = vec<E, 2>(E(float(qp[0])), E(float(qp[1])));
        }
        q_frag[fr][db].thread_elements()[0] = qv.x;
        q_frag[fr][db].thread_elements()[1] = qv.y;
    }
}
simdgroup_float8x8 acc[NR][D / 8];
float row_max[NR];
float row_sum[NR];
for (int fr = 0; fr < NR; ++fr) {
    for (int db = 0; db < D / 8; ++db) acc[fr][db] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    row_max[fr] = -1e30f;
    row_sum[fr] = 0.0f;
    if (split == 0 && row_ok[fr]) {
        const int r = fr * 8 + int(fm);
        row_max[fr] = float(sinks[(row_in + r) * sinks_strides[1] + head * sinks_strides[2]]) * LOG2E;
        row_sum[fr] = 1.0f;
    }
}

const int kbeg = first + split * kps;
const int kend = min(S, kbeg + kps);
const device T* k_head_p = kp[zl] + kvh * k_head[zl];
const device T* v_head_p = vp[zl] + kvh * k_head[zl];
const device T* kb_head_p = kb + kvh * kb_strides[1] + size_t(row_in) * kb_strides[2];
const device T* vb_head_p = vb + kvh * vb_strides[1] + size_t(row_in) * vb_strides[2];

for (int k0 = kbeg; k0 < kend; k0 += BK) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int c = int(tid); c < BK * D / 8; c += 32 * G) {
        const int r = c / (D / 8);
        const int col = (c % (D / 8)) * 8;
        const int key = k0 + r;
        vec<T, 4> kv[2] = {vec<T, 4>(0), vec<T, 4>(0)};
        vec<T, 4> vv[2] = {vec<T, 4>(0), vec<T, 4>(0)};
        if (key < kend) {
            const bool own = key >= SC;
            auto kptr = (const device vec<T, 4>*)((own ? kb_head_p + (key - SC) * kb_strides[2] : k_head_p + key * k_row[zl]) + col);
            auto vptr = (const device vec<T, 4>*)((own ? vb_head_p + (key - SC) * vb_strides[2] : v_head_p + key * k_row[zl]) + col);
            kv[0] = kptr[0]; kv[1] = kptr[1]; vv[0] = vptr[0]; vv[1] = vptr[1];
        }
        for (int j = 0; j < 8; ++j) {
            Ks[(col + j) * LDK + r] = E(float(kv[j / 4][j % 4]));
            Vs[r * LDV + col + j] = E(float(vv[j / 4][j % 4]));
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const bool plain_tile = WINDOW == 0 && (k0 + BK <= min(kend, SC)) && (k0 + BK - 1 <= (S - L) - R);
    for (int fr = 0; fr < NR; ++fr) {
        simdgroup_float8x8 scores[BK / 8];
        for (int kb_i = 0; kb_i < BK / 8; ++kb_i) {
            scores[kb_i] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
            for (int db = 0; db < D / 8; ++db) {
                simdgroup_matrix<E, 8, 8> k_frag;
                simdgroup_load(k_frag, Ks + db * 8 * LDK + kb_i * 8, LDK);
                simdgroup_multiply_accumulate(scores[kb_i], q_frag[fr][db], k_frag, scores[kb_i]);
            }
        }
        float tile_max = -INFINITY;
        for (int kb_i = 0; kb_i < BK / 8; ++kb_i) {
            for (int e = 0; e < 2; ++e) {
                const int key = k0 + kb_i * 8 + fn + e;
                float s = -INFINITY;
                if (plain_tile) {
                    if (row_ok[fr]) s = scores[kb_i].thread_elements()[e] * qscale * tau[fr];
                } else if (row_ok[fr] && key < kend) {
                    int d;
                    bool vis = true;
                    if (TREE && key >= SC) {
                        const int kc = key - SC;
                        vis = (kc < L) && (((r_mask[fr] >> kc) & 1u) != 0u);
                        d = r_depth[fr] - (kc < L ? TMETA[(row_in + kc) * 2] : 0);
                    } else {
                        d = diag[fr] - key;
                    }
                    if (vis && d >= 0 && (WINDOW == 0 || d < WINDOW)) {
                        const float b = d < R ? float(bias_ptr[fr][d]) * LOG2E : 0.0f;
                        s = (scores[kb_i].thread_elements()[e] * qscale + b) * tau[fr];
                    }
                }
                scores[kb_i].thread_elements()[e] = s;
                tile_max = max(tile_max, s);
            }
        }
        tile_max = max(tile_max, simd_shuffle_xor(tile_max, 1));
        tile_max = max(tile_max, simd_shuffle_xor(tile_max, 8));
        const float new_max = max(row_max[fr], tile_max);
        const float factor = fast::exp2(row_max[fr] - new_max);
        float tile_sum = 0.0f;
        simdgroup_matrix<E, 8, 8> probs[BK / 8];
        for (int kb_i = 0; kb_i < BK / 8; ++kb_i) {
            for (int e = 0; e < 2; ++e) {
                const float p = fast::exp2(scores[kb_i].thread_elements()[e] - new_max);
                probs[kb_i].thread_elements()[e] = E(p);
                tile_sum += p;
            }
        }
        tile_sum += simd_shuffle_xor(tile_sum, 1);
        tile_sum += simd_shuffle_xor(tile_sum, 8);
        row_sum[fr] = row_sum[fr] * factor + tile_sum;
        row_max[fr] = new_max;
        for (int db = 0; db < D / 8; ++db) {
            acc[fr][db].thread_elements()[0] *= factor;
            acc[fr][db].thread_elements()[1] *= factor;
            for (int kb_i = 0; kb_i < BK / 8; ++kb_i) {
                simdgroup_matrix<E, 8, 8> v_frag;
                simdgroup_load(v_frag, Vs + kb_i * 8 * LDV + db * 8, LDV);
                simdgroup_multiply_accumulate(acc[fr][db], probs[kb_i], v_frag, acc[fr][db]);
            }
        }
    }
}

for (int fr = 0; fr < NR; ++fr) {
    if (!row_ok[fr]) continue;
    const int r = fr * 8 + int(fm);
    if constexpr (DIRECT) {
        const float inv = 1.0f / row_sum[fr];
        const size_t goff = (size_t(row_in + r) * HQ + head) * D;
        const size_t ooff = (size_t(row_out + r) * HQ + head) * D;
        for (int db = 0; db < D / 8; ++db) {
            const int c0 = db * 8 + fn;
            OUT[ooff + c0] = T(float(T(acc[fr][db].thread_elements()[0] * inv)) * GATE[goff + c0]);
            OUT[ooff + c0 + 1] = T(float(T(acc[fr][db].thread_elements()[1] * inv)) * GATE[goff + c0 + 1]);
        }
    } else {
        const size_t at = (size_t(split) * LT + row_out + r) * HQ + head;
        for (int db = 0; db < D / 8; ++db) {
            PART[at * D + db * 8 + fn] = acc[fr][db].thread_elements()[0];
            PART[at * D + db * 8 + fn + 1] = acc[fr][db].thread_elements()[1];
        }
        if (fn == 0) {
            ML[at * 2] = row_max[fr];
            ML[at * 2 + 1] = row_sum[fr];
        }
    }
}
