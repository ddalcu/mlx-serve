// Inkling 2, up to 8 verify rows behind a cache without concatenating KV: identical math to relsplit.metal, reading
// cached keys [first..C) from (kc, vc) and the L new rows [0..L) from (kn, vn). grid (splits * 32 G, HKV, 1),
// threadgroup (32 G, 1, 1): a threadgroup takes logical keys [first + split * KPS, + KPS) of one KV head, SIMD group g
// its query head kv * G + g, the L rows as one 8-row fragment. q [1, HQ, L, D], kc, vc [1, HKV, C, D], kn, vn
// [1, HKV, L, D] (T, any strides over a unit last axis); rel [1, L, HQ, R] (T); sinks [1, L, HQ] f32;
// sizes = { L, C, first } with logical total S = C + L. WINDOW > 0 keeps keys fewer than WINDOW behind a row.
// PART [splits, L, HQ, D] f32, ML [splits, L, HQ, 2] f32.
constexpr float LOG2E = 1.4426950408889634f;
constexpr int BK = 16;
constexpr int LDK = BK + 8;
constexpr int LDV = D + 8;
constexpr int G = HQ / HKV;
const int L = sizes[0];
const int C = sizes[1];
const int S = C + L;
const int split = threadgroup_position_in_grid.x;
const int kvh = threadgroup_position_in_grid.y;
const uint tid = thread_position_in_threadgroup.x;
const uint sg = simdgroup_index_in_threadgroup;
const uint lane = thread_index_in_simdgroup;
const int head = kvh * G + int(sg);
const float qscale = scale * LOG2E;
const short fm = ((lane / 4) & 4) + (lane / 2) % 4;
const short fn = ((lane / 4) & 2) * 2 + (lane % 2) * 2;
const bool row_ok = fm < L;
const int diag = C + fm;
const int kbeg = sizes[2] + split * KPS;
const int kend = min(S, kbeg + KPS);

threadgroup half Ks[D * LDK];
threadgroup half Vs[BK * LDV];
simdgroup_half8x8 q_frag[D / 8];
for (int db = 0; db < D / 8; ++db) {
    half2 qv = half2(0.0h);
    if (row_ok) {
        const device T* qp = q + head * q_strides[1] + fm * q_strides[2] + db * 8 + fn;
        qv = half2(float(qp[0]), float(qp[1]));
    }
    q_frag[db].thread_elements()[0] = qv.x;
    q_frag[db].thread_elements()[1] = qv.y;
}
simdgroup_float8x8 acc[D / 8];
for (int db = 0; db < D / 8; ++db) acc[db] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
float row_max = -1e30f, row_sum = 0.0f;
if (split == 0 && row_ok) {
    row_max = sinks[fm * HQ + head] * LOG2E;
    row_sum = 1.0f;
}
const device T* bias = rel + (size_t(row_ok ? fm : 0) * HQ + head) * R;
const device T* kc_head = kc + kvh * kc_strides[1];
const device T* vc_head = vc + kvh * vc_strides[1];
const device T* kn_head = kn + kvh * kn_strides[1];
const device T* vn_head = vn + kvh * vn_strides[1];
for (int k0 = kbeg; k0 < kend; k0 += BK) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int c = tid; c < BK * D / 8; c += 32 * G) {
        const int r = c / (D / 8);
        const int col = (c % (D / 8)) * 8;
        const int key = k0 + r;
        vec<T, 4> kv[2] = {vec<T, 4>(0), vec<T, 4>(0)};
        vec<T, 4> vv[2] = {vec<T, 4>(0), vec<T, 4>(0)};
        if (key < kend) {
            const device T* kr = key < C ? (kc_head + key * kc_strides[2]) : (kn_head + (key - C) * kn_strides[2]);
            const device T* vr = key < C ? (vc_head + key * vc_strides[2]) : (vn_head + (key - C) * vn_strides[2]);
            auto kp = (const device vec<T, 4>*)(kr + col);
            auto vp = (const device vec<T, 4>*)(vr + col);
            kv[0] = kp[0]; kv[1] = kp[1]; vv[0] = vp[0]; vv[1] = vp[1];
        }
        for (int j = 0; j < 8; ++j) {
            Ks[(col + j) * LDK + r] = half(float(kv[j / 4][j % 4]));
            Vs[r * LDV + col + j] = half(float(vv[j / 4][j % 4]));
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    simdgroup_float8x8 scores[BK / 8];
    for (int kb = 0; kb < BK / 8; ++kb) {
        scores[kb] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        for (int db = 0; db < D / 8; ++db) {
            simdgroup_half8x8 k_frag;
            simdgroup_load(k_frag, Ks + db * 8 * LDK + kb * 8, LDK);
            simdgroup_multiply_accumulate(scores[kb], q_frag[db], k_frag, scores[kb]);
        }
    }
    float tile_max = -INFINITY;
    for (int kb = 0; kb < BK / 8; ++kb) {
        for (int e = 0; e < 2; ++e) {
            const int key = k0 + kb * 8 + fn + e;
            const int d = diag - key;
            const bool visible = row_ok && key < kend && d >= 0 && (WINDOW == 0 || d < WINDOW);
            float s = -INFINITY;
            if (visible) {
                s = scores[kb].thread_elements()[e] * qscale;
                if (d < R) s += float(bias[d]) * LOG2E;
            }
            scores[kb].thread_elements()[e] = s;
            tile_max = max(tile_max, s);
        }
    }
    tile_max = max(tile_max, simd_shuffle_xor(tile_max, 1));
    tile_max = max(tile_max, simd_shuffle_xor(tile_max, 8));
    const float new_max = max(row_max, tile_max);
    const float factor = fast::exp2(row_max - new_max);
    float tile_sum = 0.0f;
    simdgroup_half8x8 probs[BK / 8];
    for (int kb = 0; kb < BK / 8; ++kb) {
        for (int e = 0; e < 2; ++e) {
            const float p = fast::exp2(scores[kb].thread_elements()[e] - new_max);
            probs[kb].thread_elements()[e] = half(p);
            tile_sum += p;
        }
    }
    tile_sum += simd_shuffle_xor(tile_sum, 1);
    tile_sum += simd_shuffle_xor(tile_sum, 8);
    row_sum = row_sum * factor + tile_sum;
    row_max = new_max;
    for (int db = 0; db < D / 8; ++db) {
        acc[db].thread_elements()[0] *= factor;
        acc[db].thread_elements()[1] *= factor;
        for (int kb = 0; kb < BK / 8; ++kb) {
            simdgroup_half8x8 v_frag;
            simdgroup_load(v_frag, Vs + kb * 8 * LDV + db * 8, LDV);
            simdgroup_multiply_accumulate(acc[db], probs[kb], v_frag, acc[db]);
        }
    }
}
if (row_ok) {
    const size_t at = (size_t(split) * L + fm) * HQ + head;
    for (int db = 0; db < D / 8; ++db) {
        PART[at * D + db * 8 + fn] = acc[db].thread_elements()[0];
        PART[at * D + db * 8 + fn + 1] = acc[db].thread_elements()[1];
    }
    if (fn == 0) {
        ML[at * 2] = row_max;
        ML[at * 2 + 1] = row_sum;
    }
}
