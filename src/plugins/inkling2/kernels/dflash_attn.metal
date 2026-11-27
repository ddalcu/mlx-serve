// Inkling 2 DFlash2: each lane's block rows read its context ring and the block's own keys, up to 16 lanes a launch (a
// lane's ring a buffer). grid (splits * 32 G, HKV, NL), threadgroup (32 G, 1, 1): lane z, KV head y, keys [x * kps,
// + kps) of the lane's n context keys then its B block keys, SIMD group g query head y * G + g, the B <= 8 rows one
// 8-row fragment. q [NL, HQ, B, D] and kb, vb [NL, HKV, B, D] (T), r0..r15 [1, 2 HKV, C, D] (T, keys then values):
// context position p at slot p % C. LANES [NL, 4] i32 = (position, n, C, kps): the n positions before position are
// the context, which row j reads only within WINDOW of position + j. PART [splits, NL * B * HQ, D] f32 the unnormalised
// sums, ML [splits, NL * B * HQ, 2] f32 each row's max (log2 units) and sum. Matrices are half under a 16-bit T, float
// under float.
using M = metal::conditional_t<metal::is_same_v<T, float>, float, half>;
constexpr float LOG2E = 1.4426950408889634f;
constexpr int BK = 16;
constexpr int LDK = BK + 8;
constexpr int LDV = D + 8;
constexpr int G = HQ / HKV;
const int z = threadgroup_position_in_grid.z;
const int kvh = threadgroup_position_in_grid.y;
const int split = threadgroup_position_in_grid.x;
const uint tid = thread_position_in_threadgroup.x;
const uint sg = simdgroup_index_in_threadgroup;
const uint lane = thread_index_in_simdgroup;
const device T* rp[16] = {r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15};
const int NL = q_shape[0];
const int position = LANES[z * 4];
const int n = LANES[z * 4 + 1];
const int C = LANES[z * 4 + 2];
const int kps = LANES[z * 4 + 3];
const int kbeg = split * kps;
const int kend = min(n + B, kbeg + kps);
const int head = kvh * G + int(sg);
const float qscale = scale * LOG2E;
const short fm = ((lane / 4) & 4) + (lane / 2) % 4;
const short fn = ((lane / 4) & 2) * 2 + (lane % 2) * 2;
const bool row_ok = fm < B;

threadgroup M Ks[D * LDK];
threadgroup M Vs[BK * LDV];
simdgroup_matrix<M, 8, 8> q_frag[D / 8];
for (int db = 0; db < D / 8; ++db) {
    M q0 = M(0), q1 = M(0);
    if (row_ok) {
        const device T* qr = q + ((size_t(z) * HQ + head) * B + fm) * D + db * 8 + fn;
        q0 = M(float(qr[0]));
        q1 = M(float(qr[1]));
    }
    q_frag[db].thread_elements()[0] = q0;
    q_frag[db].thread_elements()[1] = q1;
}
simdgroup_float8x8 acc[D / 8];
for (int db = 0; db < D / 8; ++db) acc[db] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
// A finite floor: a split whose keys a row cannot see joins at weight 0.
float row_max = -1e30f, row_sum = 0.0f;
const device T* k_ring = rp[z] + size_t(kvh) * C * D;
const device T* v_ring = rp[z] + size_t(HKV + kvh) * C * D;
const device T* k_own = kb + (size_t(z) * HKV + kvh) * B * D;
const device T* v_own = vb + (size_t(z) * HKV + kvh) * B * D;
for (int k0i = kbeg; k0i < kend; k0i += BK) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int c = tid; c < BK * D / 8; c += 32 * G) {
        const int r = c / (D / 8);
        const int col = (c % (D / 8)) * 8;
        const int key = k0i + r;
        vec<T, 4> kv[2] = {vec<T, 4>(0), vec<T, 4>(0)};
        vec<T, 4> vv[2] = {vec<T, 4>(0), vec<T, 4>(0)};
        if (key < kend) {
            const size_t at = key < n ? size_t((position - n + key) % C) * D : size_t(key - n) * D;
            auto kq = (const device vec<T, 4>*)((key < n ? k_ring : k_own) + at + col);
            auto vq = (const device vec<T, 4>*)((key < n ? v_ring : v_own) + at + col);
            kv[0] = kq[0]; kv[1] = kq[1]; vv[0] = vq[0]; vv[1] = vq[1];
        }
        for (int j = 0; j < 8; ++j) {
            Ks[(col + j) * LDK + r] = M(float(kv[j / 4][j % 4]));
            Vs[r * LDV + col + j] = M(float(vv[j / 4][j % 4]));
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    simdgroup_float8x8 scores[BK / 8];
    for (int kb8 = 0; kb8 < BK / 8; ++kb8) {
        scores[kb8] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        for (int db = 0; db < D / 8; ++db) {
            simdgroup_matrix<M, 8, 8> k_frag;
            simdgroup_load(k_frag, Ks + db * 8 * LDK + kb8 * 8, LDK);
            simdgroup_multiply_accumulate(scores[kb8], q_frag[db], k_frag, scores[kb8]);
        }
    }
    float tile_max = -INFINITY;
    for (int kb8 = 0; kb8 < BK / 8; ++kb8) {
        for (int e = 0; e < 2; ++e) {
            const int key = k0i + kb8 * 8 + fn + e;
            const bool visible = row_ok && key < kend && (key >= n || n + fm - key < WINDOW);
            const float s = visible ? scores[kb8].thread_elements()[e] * qscale : -INFINITY;
            scores[kb8].thread_elements()[e] = s;
            tile_max = max(tile_max, s);
        }
    }
    tile_max = max(tile_max, simd_shuffle_xor(tile_max, 1));
    tile_max = max(tile_max, simd_shuffle_xor(tile_max, 8));
    const float new_max = max(row_max, tile_max);
    const float factor = fast::exp2(row_max - new_max);
    float tile_sum = 0.0f;
    simdgroup_matrix<M, 8, 8> probs[BK / 8];
    for (int kb8 = 0; kb8 < BK / 8; ++kb8) {
        for (int e = 0; e < 2; ++e) {
            const float p = fast::exp2(scores[kb8].thread_elements()[e] - new_max);
            probs[kb8].thread_elements()[e] = M(p);
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
        for (int kb8 = 0; kb8 < BK / 8; ++kb8) {
            simdgroup_matrix<M, 8, 8> v_frag;
            simdgroup_load(v_frag, Vs + kb8 * 8 * LDV + db * 8, LDV);
            simdgroup_multiply_accumulate(acc[db], probs[kb8], v_frag, acc[db]);
        }
    }
}
if (row_ok) {
    const size_t at = (size_t(split) * NL * B + size_t(z) * B + fm) * HQ + head;
    for (int db = 0; db < D / 8; ++db) {
        PART[at * D + db * 8 + fn] = acc[db].thread_elements()[0];
        PART[at * D + db * 8 + fn + 1] = acc[db].thread_elements()[1];
    }
    if (fn == 0) {
        ML[at * 2] = row_max;
        ML[at * 2 + 1] = row_sum;
    }
}
