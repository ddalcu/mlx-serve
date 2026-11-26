// Inkling 2, up to 8 rows behind a cache: relattn.metal's read split over the keys, the G = HQ / HKV query heads of a
// KV head sharing each K/V tile (MLX's attention reads the keys once per row and head). grid (splits * 32 G, HKV, 1),
// threadgroup (32 G, 1, 1): a threadgroup takes keys [first + split * KPS, + KPS) of one KV head, SIMD group g its
// query head kv * G + g, the rows as one 8-row fragment. q [1, HQ, L, D] and the keys and values (T, any strides over
// a unit last axis), the rows at the last L of the S keys: k, v [1, HKV, ., D] hold the first SC keys and kb, vb the
// ones after them (a verify's own rows, behind a cache it does not append to; SC = S reads k, v alone). rel
// [1, L, HQ, R] (T) each row's bias at the distances below R; sinks [1, L, HQ] f32, the first split starting each
// row from its sink; sizes = { L, S, first, SC }. WINDOW > 0 keeps the keys fewer than WINDOW behind a row.
// PART [splits, L, HQ, D] f32 = the unnormalised sums, ML [splits, L, HQ, 2] f32 = each row's running max (log2
// units) and sum.
constexpr float LOG2E = 1.4426950408889634f;
constexpr int BK = 16;
constexpr int LDK = BK + 8;
constexpr int LDV = D + 8;
constexpr int G = HQ / HKV;
const int L = sizes[0];
const int S = sizes[1];
const int SC = sizes[3];
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
const int diag = S - L + fm;
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
// A finite floor: a split whose keys a row cannot see leaves it at weight 0 in the combine.
float row_max = -1e30f, row_sum = 0.0f;
if (split == 0 && row_ok) {
    row_max = sinks[fm * HQ + head] * LOG2E;
    row_sum = 1.0f;
}
const device T* bias = rel + (size_t(row_ok ? fm : 0) * HQ + head) * R;
const device T* k_head = k + kvh * k_strides[1];
const device T* v_head = v + kvh * v_strides[1];
const device T* kb_head = kb + kvh * kb_strides[1];
const device T* vb_head = vb + kvh * vb_strides[1];
for (int k0 = kbeg; k0 < kend; k0 += BK) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int c = tid; c < BK * D / 8; c += 32 * G) {
        const int r = c / (D / 8);
        const int col = (c % (D / 8)) * 8;
        const int key = k0 + r;
        vec<T, 4> kv[2] = {vec<T, 4>(0), vec<T, 4>(0)};
        vec<T, 4> vv[2] = {vec<T, 4>(0), vec<T, 4>(0)};
        if (key < kend) {
            const bool own = key >= SC;
            auto kp = (const device vec<T, 4>*)((own ? kb_head + (key - SC) * kb_strides[2] : k_head + key * k_strides[2]) + col);
            auto vp = (const device vec<T, 4>*)((own ? vb_head + (key - SC) * vb_strides[2] : v_head + key * v_strides[2]) + col);
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
