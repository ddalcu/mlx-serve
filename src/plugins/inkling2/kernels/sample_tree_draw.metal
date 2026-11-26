// Inkling 2: exact GPU sampling + multi-child sparse-16 tree rejection sampling per verify row.
// grid (256, B * W, 1), threadgroup (256, 1, 1).
// Uses VROW [B * W, 4] from sample_tree_search, plus CANDS [B, R, K], NEXT_Q [B, W, K], DRAWS [B, W, 4], TMETA [B * W, 2].
// Output: SELECTED [B * W] int32.
const uint row = threadgroup_position_in_grid.y;
const uint tid = thread_position_in_threadgroup.x;
const uint lane = thread_index_in_simdgroup;
const uint sg = simdgroup_index_in_threadgroup;
const uint b = row / uint(W);
const uint r = row % uint(W);

const float temp = LPARAMS[b * 4u + 0u];
const uint top_k = uint(LPARAMS[b * 4u + 1u]);
const float rmax = VROW[size_t(row) * 4u + 0u];
const float cut_l = VROW[size_t(row) * 4u + 1u];
const int cut_id = as_type<int>(VROW[size_t(row) * 4u + 2u]);
const float kept_mass = VROW[size_t(row) * 4u + 3u];

if (temp <= 0.0f || top_k == 1u) {
    if (tid == 0u) SELECTED[row] = cut_id;
    return;
}

const float inv_t = 1.0f / temp;
const device T* src = LOGITS + size_t(row) * size_t(STRIDE);

auto admitted_prob = [&](uint col) -> float {
    if (col >= uint(V)) return 0.0f;
    if constexpr (MASK != 0) if (SUPPRESS[col]) return 0.0f;
    const float raw = float(src[col]);
    if (isnan(raw) || raw == -INFINITY) return 0.0f;
    const float v = raw * inv_t;
    if (v > cut_l || (v == cut_l && int(col) <= cut_id)) return metal::exp(v - rmax) / kept_mass;
    return 0.0f;
};

auto mix64 = [](ulong v) -> ulong {
    v = (v ^ (v >> 30)) * 0xbf58476d1ce4e5b9ul;
    v = (v ^ (v >> 27)) * 0x94d049bb133111ebul;
    return v ^ (v >> 31);
};
auto uniform01 = [&](ulong seed, uint tag) -> float {
    const ulong h = mix64(seed ^ (0x9e3779b97f4a7c15ul * ulong(row * 8u + tag + 1u)));
    return (float(h >> 41) + 0.5f) * 0x1p-23f;
};

threadgroup int sh_done;
threadgroup float sh_target_mass;
threadgroup int sh_cands[K];
threadgroup uint sh_has_cands;

if (tid == 0u) {
    sh_done = 0;
    sh_has_cands = 0u;
    const ulong seed = (ulong(SEEDS[b * 2u + 1u]) << 32) | ulong(SEEDS[b * 2u + 0u]);
    const uint depth = uint(TMETA[size_t(row) * 2u + 0u]);
    const size_t draw_off = size_t(row) * 4u;
    bool has_draw = (TREE != 0) && (depth < uint(R)) && (DRAWS[draw_off + 0u] < uint(K));

    if (!has_draw) {
        // Plain exact draw from filtered target distribution (leaf / bonus row / chain with no drawn child).
        sh_target_mass = uniform01(seed, 5u) * kept_mass;
    } else {
        sh_has_cands = 1u;
        const size_t cand_off = (size_t(b) * uint(R) + depth) * uint(K);
        const size_t q_off = size_t(row) * uint(K);
        float p_rem[K];
        float q_rem[K];
        float p_cand_sum = 0.0f;
        for (uint i = 0u; i < uint(K); ++i) {
            const int cid = CANDS[cand_off + i];
            sh_cands[i] = cid;
            p_rem[i] = (cid >= 0) ? admitted_prob(uint(cid)) : 0.0f;
            p_cand_sum += p_rem[i];
            q_rem[i] = NEXT_Q[q_off + i];
        }
        float outside_mass = max(0.0f, 1.0f - p_cand_sum);
        float outside_scale = 1.0f;
        float q_mass = 1.0f;
        bool accepted = false;
        int picked_tok = cut_id;

        for (uint ch = 0u; ch < 4u; ++ch) {
            const uint di = uint(DRAWS[draw_off + ch]);
            if (di >= uint(K)) break;
            const float qc = (q_mass > 0.0f) ? (q_rem[di] / q_mass) : 0.0f;
            const float ap = (qc > 0.0f) ? min(1.0f, p_rem[di] / qc) : 0.0f;
            const float u = uniform01(seed, ch);
            if (u < ap) {
                picked_tok = sh_cands[di];
                accepted = true;
                break;
            }
            // Compute exact residual normalizer Z = sum_v max(p_rem(v) - q_cur(v), 0).
            float z = outside_mass * outside_scale;
            float next_p[K];
            for (uint i = 0u; i < uint(K); ++i) {
                const float qi = (q_mass > 0.0f) ? (q_rem[i] / q_mass) : 0.0f;
                next_p[i] = max(0.0f, p_rem[i] - qi);
                z += next_p[i];
            }
            if (z <= 1e-12f) {
                // Degenerate zero-residual case (p == q): draw directly from p_rem.
                break;
            }
            for (uint i = 0u; i < uint(K); ++i) p_rem[i] = next_p[i] / z;
            outside_scale /= z;
            q_mass = max(0.0f, q_mass - q_rem[di]);
            q_rem[di] = 0.0f;
        }

        if (accepted) {
            SELECTED[row] = picked_tok;
            sh_done = 1;
        } else {
            // Draw from final residual distribution: first check the K candidates' residual mass.
            float cand_res = 0.0f;
            for (uint i = 0u; i < uint(K); ++i) cand_res += p_rem[i];
            const float u_res = uniform01(seed, 5u);
            if (u_res < cand_res || !(outside_mass * outside_scale > 1e-12f)) {
                float acc = 0.0f;
                const float target = (u_res < cand_res) ? u_res : (u_res * cand_res);
                int chosen = sh_cands[0];
                for (uint i = 0u; i < uint(K); ++i) {
                    acc += p_rem[i];
                    if (acc >= target) { chosen = sh_cands[i]; break; }
                }
                SELECTED[row] = chosen;
                sh_done = 1;
            } else {
                // Target raw exp-mass outside the K candidates.
                sh_target_mass = ((u_res - cand_res) / outside_scale) * kept_mass;
            }
        }
    }
}
threadgroup_barrier(mem_flags::mem_threadgroup);
if (sh_done != 0) return;

// Full-vocab coalesced scan over admitted tokens (excluding sh_cands when sh_has_cands == 1).
const uint excl = sh_has_cands;
const uint per_t = (uint(V) + 255u) / 256u;
const uint t_lo = tid * per_t;
const uint t_hi = min(uint(V), t_lo + per_t);

auto is_excluded = [&](uint col) -> bool {
    if (!excl) return false;
    for (uint i = 0u; i < uint(K); ++i) if (int(col) == sh_cands[i]) return true;
    return false;
};

float my_sum = 0.0f;
int my_last = -1;
for (uint col = t_lo; col < t_hi; ++col) {
    if (is_excluded(col)) continue;
    if constexpr (MASK != 0) if (SUPPRESS[col]) continue;
    const float raw = float(src[col]);
    if (isnan(raw) || raw == -INFINITY) continue;
    const float v = raw * inv_t;
    if (v > cut_l || (v == cut_l && int(col) <= cut_id)) {
        my_sum += metal::exp(v - rmax);
        my_last = int(col);
    }
}

// Exclusive prefix sum across 256 threads.
const float sg_inc = simd_prefix_inclusive_sum(my_sum);
const float sg_exc = sg_inc - my_sum;
const float sg_tot = simd_broadcast(sg_inc, 31);
threadgroup float sg_sums[8];
threadgroup int tg_last[256];
tg_last[tid] = my_last;
if (lane == 0u) sg_sums[sg] = sg_tot;
threadgroup_barrier(mem_flags::mem_threadgroup);

float sg_base = 0.0f;
for (uint s = 0u; s < sg; ++s) sg_base += sg_sums[s];
const float before = sg_base + sg_exc;
const float after = before + my_sum;
const float target = sh_target_mass;

threadgroup atomic_int chosen;
if (tid == 0u) atomic_store_explicit(&chosen, -1, memory_order_relaxed);
threadgroup_barrier(mem_flags::mem_threadgroup);

if (my_sum > 0.0f && target >= before && target < after) {
    float acc = before;
    int hit = my_last;
    for (uint col = t_lo; col < t_hi; ++col) {
        if (is_excluded(col)) continue;
        if constexpr (MASK != 0) if (SUPPRESS[col]) continue;
        const float raw = float(src[col]);
        if (isnan(raw) || raw == -INFINITY) continue;
        const float v = raw * inv_t;
        if (v > cut_l || (v == cut_l && int(col) <= cut_id)) {
            acc += metal::exp(v - rmax);
            if (acc >= target) { hit = int(col); break; }
        }
    }
    atomic_store_explicit(&chosen, hit, memory_order_relaxed);
}
threadgroup_barrier(mem_flags::mem_threadgroup);

if (tid == 0u) {
    int tok = atomic_load_explicit(&chosen, memory_order_relaxed);
    if (tok < 0) {
        for (int t = 255; t >= 0; --t) {
            if (tg_last[t] >= 0) { tok = tg_last[t]; break; }
        }
        if (tok < 0) tok = cut_id;
    }
    SELECTED[row] = tok;
}
