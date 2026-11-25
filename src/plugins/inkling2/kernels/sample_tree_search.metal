// Inkling 2: find exact top-k / top-p / min-p filter cutoff per verify row via 512-bin histogram + exact boundary scan.
// grid (256, B * W, 1), threadgroup (256, 1, 1).
// VROW [B * W, 4] f32 = { row_max, cut_logit, as_type<float>(cut_last_id), kept_mass }.
// A token col is admitted iff (v > cut_logit) || (v == cut_logit && int(col) <= cut_last_id), matching ranksDescending.
const uint row = threadgroup_position_in_grid.y;
const uint tid = thread_position_in_threadgroup.x;
const uint b = row / uint(W);
const float temp = LPARAMS[b * 4u + 0u];
const uint top_k = uint(LPARAMS[b * 4u + 1u]);
const float top_p = LPARAMS[b * 4u + 2u];
const float min_p = LPARAMS[b * 4u + 3u];
const bool sampled = temp > 0.0f;
const float inv_t = sampled ? (1.0f / temp) : 1.0f;

// Merge shards in thread 0.
threadgroup float sh_max;
threadgroup float sh_sum;
threadgroup uint sh_adm;
threadgroup int sh_argmax;
if (tid == 0u) {
    float gmx = -INFINITY;
    int gid = 0;
    uint gadm = 0u;
    for (uint s = 0u; s < uint(SHARDS); ++s) {
        const size_t at = (size_t(row) * uint(SHARDS) + s) * 4u;
        const float sm = PMASS[at + 0u];
        const int si = as_type<int>(PMASS[at + 3u]);
        gadm += uint(PMASS[at + 2u]);
        if (sm > gmx || (sm == gmx && si < gid)) { gmx = sm; gid = si; }
    }
    float gsum = 0.0f;
    if (sampled && gmx > -INFINITY) {
        for (uint s = 0u; s < uint(SHARDS); ++s) {
            const size_t at = (size_t(row) * uint(SHARDS) + s) * 4u;
            const float sm = PMASS[at + 0u];
            if (sm > -INFINITY) gsum += PMASS[at + 1u] * metal::exp(sm - gmx);
        }
    }
    sh_max = gmx;
    sh_sum = gsum;
    sh_adm = gadm;
    sh_argmax = gid;
}
threadgroup_barrier(mem_flags::mem_threadgroup);

const float rmax = sh_max;
const float rsum = sh_sum;
const uint radm = sh_adm;
const int rargmax = sh_argmax;

// Fast exit: greedy, top_k == 1, or <= 1 admitted token.
if (!sampled || top_k == 1u || radm <= 1u || !(rsum > 0.0f)) {
    if (tid == 0u) {
        const size_t o = size_t(row) * 4u;
        VROW[o + 0u] = rmax;
        VROW[o + 1u] = rmax;
        VROW[o + 2u] = as_type<float>(rargmax);
        VROW[o + 3u] = 1.0f;
    }
    return;
}

const float min_p_cut = (min_p > 0.0f) ? (rmax + metal::log(min_p)) : -INFINITY;
const bool has_k = (top_k > 0u && top_k < radm);
const bool has_p = (top_p > 0.0f && top_p < 1.0f);
const device T* src = LOGITS + size_t(row) * size_t(STRIDE);

// Fast exit: no top-k and no top-p (temperature + optional min-p only).
if (!has_k && !has_p) {
    float km = rsum;
    if (min_p > 0.0f) {
        float loc = 0.0f;
        for (uint col = tid; col < uint(V); col += 256u) {
            if constexpr (MASK != 0) if (SUPPRESS[col]) continue;
            const float raw = float(src[col]);
            if (isnan(raw) || raw == -INFINITY) continue;
            const float v = raw * inv_t;
            if (v >= min_p_cut) loc += metal::exp(v - rmax);
        }
        loc = simd_sum(loc);
        threadgroup float ps[8];
        if (thread_index_in_simdgroup == 0u) ps[simdgroup_index_in_threadgroup] = loc;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid == 0u) {
            km = 0.0f;
            for (uint s = 0u; s < 8u; ++s) km += ps[s];
        }
    }
    if (tid == 0u) {
        const size_t o = size_t(row) * 4u;
        VROW[o + 0u] = rmax;
        VROW[o + 1u] = min_p_cut;
        VROW[o + 2u] = as_type<float>(int(V));
        VROW[o + 3u] = (km > 0.0f) ? km : 1.0f;
    }
    return;
}

// 512-bin histogram over [rmax - 20.0, rmax] to isolate the exact boundary bin in ONE pass.
constexpr uint NBINS = 512;
constexpr float SPAN = 20.0f;
constexpr float INV_STEP = float(NBINS - 2) / SPAN;
threadgroup atomic_uint bin_cnt[NBINS];
threadgroup atomic_uint bin_mass[NBINS];
for (uint i = tid; i < NBINS; i += 256u) {
    atomic_store_explicit(&bin_cnt[i], 0u, memory_order_relaxed);
    atomic_store_explicit(&bin_mass[i], 0u, memory_order_relaxed);
}
threadgroup_barrier(mem_flags::mem_threadgroup);

const float mscale = 16777216.0f / rsum;
for (uint col = tid; col < uint(V); col += 256u) {
    if constexpr (MASK != 0) if (SUPPRESS[col]) continue;
    const float raw = float(src[col]);
    if (isnan(raw) || raw == -INFINITY) continue;
    const float v = raw * inv_t;
    if (v < min_p_cut) continue;
    const float diff = rmax - v;
    uint bin;
    if (diff <= 0.0f) bin = 0u;
    else if (diff >= SPAN) bin = NBINS - 1u;
    else bin = min(NBINS - 2u, 1u + uint(diff * INV_STEP));
    atomic_fetch_add_explicit(&bin_cnt[bin], 1u, memory_order_relaxed);
    const uint mq = uint(metal::exp(-diff) * mscale);
    if (mq > 0u) atomic_fetch_add_explicit(&bin_mass[bin], mq, memory_order_relaxed);
}
threadgroup_barrier(mem_flags::mem_threadgroup);

threadgroup uint cut_bin;
threadgroup uint prev_cnt;
threadgroup float prev_mass;
threadgroup float bin_hi;
threadgroup float bin_lo;
if (tid == 0u) {
    const uint k_lim = has_k ? top_k : radm;
    const float p_lim = has_p ? top_p : 2.0f;
    uint cc = 0u;
    float cm = 0.0f;
    uint cb = NBINS - 1u;
    for (uint i = 0u; i < NBINS; ++i) {
        const uint bc = atomic_load_explicit(&bin_cnt[i], memory_order_relaxed);
        const float bm = float(atomic_load_explicit(&bin_mass[i], memory_order_relaxed)) / 16777216.0f;
        if (bc > 0u && (cc + bc >= k_lim || cm + bm >= p_lim)) {
            cb = i;
            break;
        }
        cc += bc;
        cm += bm;
    }
    cut_bin = cb;
    prev_cnt = cc;
    prev_mass = cm;
    if (cb == 0u) { bin_hi = rmax; bin_lo = rmax; }
    else if (cb == NBINS - 1u) { bin_hi = rmax - SPAN; bin_lo = -INFINITY; }
    else {
        bin_hi = rmax - float(cb - 1u) / INV_STEP;
        bin_lo = rmax - float(cb) / INV_STEP;
    }
}
threadgroup_barrier(mem_flags::mem_threadgroup);

// Pass 2: exact sum of bins < cut_bin AND collect up to 512 tokens inside cut_bin for exact cutoff.
constexpr uint BCAP = 512;
threadgroup float b_val[BCAP];
threadgroup int b_id[BCAP];
threadgroup atomic_uint b_n;
if (tid == 0u) atomic_store_explicit(&b_n, 0u, memory_order_relaxed);
threadgroup_barrier(mem_flags::mem_threadgroup);

const uint cb = cut_bin;
float loc_above = 0.0f;
for (uint col = tid; col < uint(V); col += 256u) {
    if constexpr (MASK != 0) if (SUPPRESS[col]) continue;
    const float raw = float(src[col]);
    if (isnan(raw) || raw == -INFINITY) continue;
    const float v = raw * inv_t;
    if (v < min_p_cut) continue;
    const float diff = rmax - v;
    uint bin;
    if (diff <= 0.0f) bin = 0u;
    else if (diff >= SPAN) bin = NBINS - 1u;
    else bin = min(NBINS - 2u, 1u + uint(diff * INV_STEP));
    if (bin < cb) {
        loc_above += metal::exp(v - rmax);
    } else if (bin == cb) {
        const uint pos = atomic_fetch_add_explicit(&b_n, 1u, memory_order_relaxed);
        if (pos < BCAP) {
            b_val[pos] = v;
            b_id[pos] = int(col);
        }
    }
}
loc_above = simd_sum(loc_above);
threadgroup float sg_above[8];
if (thread_index_in_simdgroup == 0u) sg_above[simdgroup_index_in_threadgroup] = loc_above;
threadgroup_barrier(mem_flags::mem_threadgroup);

if (tid == 0u) {
    float exact_above = 0.0f;
    for (uint s = 0u; s < 8u; ++s) exact_above += sg_above[s];
    const uint bn = atomic_load_explicit(&b_n, memory_order_relaxed);
    const uint k_lim = has_k ? top_k : radm;
    const float p_lim_sum = has_p ? (top_p * rsum) : (rsum * 2.0f);
    if (bn == 0u || bn > BCAP) {
        // Fallback when >512 ties/tokens fall in cut_bin: keep everything down to bin_lo (or exact_above if already enough).
        const float cl = (exact_above > 0.0f && (prev_cnt >= k_lim || exact_above >= p_lim_sum)) ? bin_hi : max(bin_lo, min_p_cut);
        const float km = (exact_above > 0.0f) ? exact_above : rsum;
        const size_t o = size_t(row) * 4u;
        VROW[o + 0u] = rmax;
        VROW[o + 1u] = cl;
        VROW[o + 2u] = as_type<float>(int(V));
        VROW[o + 3u] = km;
        return;
    }
    // Insertion sort bn <= 512 items descending by (val, -id).
    for (uint i = 1u; i < bn; ++i) {
        const float vv = b_val[i];
        const int ii = b_id[i];
        int j = int(i) - 1;
        while (j >= 0 && (vv > b_val[j] || (vv == b_val[j] && ii < b_id[j]))) {
            b_val[j + 1] = b_val[j];
            b_id[j + 1] = b_id[j];
            --j;
        }
        b_val[j + 1] = vv;
        b_id[j + 1] = ii;
    }
    uint cc = prev_cnt;
    float cm = exact_above;
    float cl = b_val[0];
    int cid = b_id[0];
    for (uint i = 0u; i < bn; ++i) {
        const float w = metal::exp(b_val[i] - rmax);
        cm += w;
        ++cc;
        cl = b_val[i];
        cid = b_id[i];
        if (cc >= k_lim || cm >= p_lim_sum) break;
    }
    const size_t o = size_t(row) * 4u;
    VROW[o + 0u] = rmax;
    VROW[o + 1u] = cl;
    VROW[o + 2u] = as_type<float>(cid);
    VROW[o + 3u] = (cm > 0.0f) ? cm : 1.0f;
}
