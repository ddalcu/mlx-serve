// Inkling 2, batched decode: one row a slot, each behind its own cache, read for up to 8 slots in one launch (a
// slot's K, V and K's strides are a buffer each, and a kernel takes 31). grid (splits * 32 NSG, HKV, NS), threadgroup
// (32 NSG, 1, 1): slot z, KV head y, keys [first + x * KB, + KB) of that slot, its G = HQ / HKV query heads sharing
// each key's K/V read; SIMD groups take the keys in turn, a lane D / 32 columns (all of D under 32 lanes), f32 math.
// q [1, NS, HQ, D] and rel [1, NS, HQ, R] (T), the slots' rows and bias rows; sinks [1, NS, HQ] f32; k0..k7, v0..v7
// [1, HKV, keys, D] (T, k's strides for both), slot i's row at its last key; KEYS [NS] i32; scale. WINDOW > 0 keeps
// the keys fewer than WINDOW behind a row. PART [splits, NS * HQ, D] and ML [splits, NS * HQ, 2] f32 as relsplit's.
constexpr float LOG2E = 1.4426950408889634f;
constexpr int G = HQ / HKV;
constexpr int DL = (D + 31) / 32;
const int slot = threadgroup_position_in_grid.z;
const int kvh = threadgroup_position_in_grid.y;
const int split = threadgroup_position_in_grid.x;
const uint sg = simdgroup_index_in_threadgroup;
const uint lane = thread_index_in_simdgroup;
const device T* kp[8] = {k0, k1, k2, k3, k4, k5, k6, k7};
const device T* vp[8] = {v0, v1, v2, v3, v4, v5, v6, v7};
const int64_t head[8] = {k0_strides[1], k1_strides[1], k2_strides[1], k3_strides[1], k4_strides[1], k5_strides[1], k6_strides[1], k7_strides[1]};
const int64_t row[8] = {k0_strides[2], k1_strides[2], k2_strides[2], k3_strides[2], k4_strides[2], k5_strides[2], k6_strides[2], k7_strides[2]};
const int S = KEYS[slot];
const int first = WINDOW > 0 ? max(0, S - WINDOW) : 0;
const int kbeg = first + split * KB;
const int kend = min(S, kbeg + KB);
const float qscale = scale * LOG2E;
float qv[G][DL], acc[G][DL], m[G], l[G];
for (int g = 0; g < G; g++) {
    const int h = kvh * G + g;
    for (int i = 0; i < DL; i++) {
        const int c = lane * DL + i;
        qv[g][i] = c < D ? float(q[(slot * HQ + h) * D + c]) * qscale : 0.0f;
        acc[g][i] = 0.0f;
    }
    // A finite floor: a split with no key a row sees joins at weight 0.
    m[g] = -1e30f;
    l[g] = 0.0f;
    if (split == 0 && sg == 0) {
        m[g] = sinks[slot * HQ + h] * LOG2E;
        l[g] = 1.0f;
    }
}
const device T* kh = kp[slot] + size_t(kvh) * head[slot];
const device T* vh = vp[slot] + size_t(kvh) * head[slot];
const size_t step = row[slot];
for (int j = kbeg + int(sg); j < kend; j += NSG) {
    float kd[DL], vd[DL];
    for (int i = 0; i < DL; i++) {
        const int c = lane * DL + i;
        kd[i] = c < D ? float(kh[size_t(j) * step + c]) : 0.0f;
        vd[i] = c < D ? float(vh[size_t(j) * step + c]) : 0.0f;
    }
    const int d = S - 1 - j;
    for (int g = 0; g < G; g++) {
        float dot = 0.0f;
        for (int i = 0; i < DL; i++) dot += qv[g][i] * kd[i];
        float s = simd_sum(dot);
        if (d < R) s += float(rel[(slot * HQ + kvh * G + g) * R + d]) * LOG2E;
        const float mn = max(m[g], s);
        const float f = fast::exp2(m[g] - mn);
        const float p = fast::exp2(s - mn);
        l[g] = l[g] * f + p;
        for (int i = 0; i < DL; i++) acc[g][i] = acc[g][i] * f + p * vd[i];
        m[g] = mn;
    }
}
threadgroup float sm[NSG][G];
threadgroup float sl[NSG][G];
threadgroup float so[NSG][G][DL * 32];
for (int g = 0; g < G; g++) {
    if (lane == 0) {
        sm[sg][g] = m[g];
        sl[sg][g] = l[g];
    }
    for (int i = 0; i < DL; i++) so[sg][g][lane * DL + i] = acc[g][i];
}
threadgroup_barrier(mem_flags::mem_threadgroup);
// SIMD group g < G joins query head g's groups.
if (int(sg) < G) {
    const int g = sg;
    float mx = -INFINITY;
    for (int t = 0; t < NSG; t++) mx = max(mx, sm[t][g]);
    float ls = 0.0f;
    float os[DL];
    for (int i = 0; i < DL; i++) os[i] = 0.0f;
    for (int t = 0; t < NSG; t++) {
        const float w = fast::exp2(sm[t][g] - mx);
        ls += w * sl[t][g];
        for (int i = 0; i < DL; i++) os[i] += w * so[t][g][lane * DL + i];
    }
    const size_t at = (size_t(split) * NS + slot) * HQ + kvh * G + g;
    for (int i = 0; i < DL; i++) {
        const int c = lane * DL + i;
        if (c < D) PART[at * D + c] = os[i];
    }
    if (lane == 0) {
        ML[at * 2] = mx;
        ML[at * 2 + 1] = ls;
    }
}
