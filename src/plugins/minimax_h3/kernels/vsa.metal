constexpr int D = 128;
constexpr int NSG = 4;
constexpr int NT = NSG * 32;
constexpr int BK = 16;
constexpr int NKB = BK / 8;
constexpr int LD = D + 8;
constexpr float NEG = -3.0e38f;
threadgroup T ksm[BK * LD];
threadgroup T vsm[BK * LD];
const uint tid = thread_position_in_threadgroup.x;
const uint sid = simdgroup_index_in_threadgroup;
const uint lane = thread_index_in_simdgroup;
const int qsub = int(threadgroup_position_in_grid.x);
const int q_tile = int(threadgroup_position_in_grid.y);
const int head = int(threadgroup_position_in_grid.z);
const int n_prefix = meta[0];
const int n_video = meta[1];
const int k_sel = meta[2];
const int n_tiles = meta[3];
const float scale_log2 = scale * 1.44269504088896341f;
const int qt = n_prefix + q_tile;
const int q_valid = vbs[qt];
const int row_base = qsub * NSG * 8;
if (row_base >= q_valid) return;
const long plane = long(head) * n_tiles * TILE;
const int srow0 = row_base + int(sid) * 8;
const int fm = int(((lane / 4) & 4) + ((lane / 2) % 4));
const int fn = int(((lane / 4) & 2) * 2 + (lane % 2) * 2);
simdgroup_matrix<T, 8, 8> qf[16];
simdgroup_matrix<float, 8, 8> acc[16];
const device T* qp = q + (plane + long(qt) * TILE + srow0) * D;
for (int kk = 0; kk < 16; kk++) {
    simdgroup_load(qf[kk], qp + kk * 8, D);
    acc[kk] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
}
float row_m = NEG;
float row_l = 0.0f;
const device int* picks = sel + (long(head) * n_video + q_tile) * k_sel;
for (int s = 0; s < k_sel; s++) {
    const int kt = picks[s];
    const int k_valid = vbs[kt];
    const long kv0 = plane + long(kt) * TILE;
    for (int j0 = 0; j0 < k_valid; j0 += BK) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int i = int(tid); i < BK * (D / 4); i += NT) {
            const int r = i / (D / 4);
            const int c4 = i - r * (D / 4);
            const long src = (kv0 + j0 + r) * D + c4 * 4;
            *(threadgroup vec<T, 4>*)(ksm + r * LD + c4 * 4) = *(const device vec<T, 4>*)(k + src);
            *(threadgroup vec<T, 4>*)(vsm + r * LD + c4 * 4) = *(const device vec<T, 4>*)(v + src);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        simdgroup_matrix<float, 8, 8> sf[NKB];
        for (int c = 0; c < NKB; c++) sf[c] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        for (int kk = 0; kk < 16; kk++) {
            for (int c = 0; c < NKB; c++) {
                simdgroup_matrix<T, 8, 8> km;
                simdgroup_load(km, (const threadgroup T*)(ksm + c * 8 * LD + kk * 8), LD, ulong2(0, 0), true);
                simdgroup_multiply_accumulate(sf[c], qf[kk], km, sf[c]);
            }
        }
        simdgroup_matrix<T, 8, 8> pf[NKB];
        float cmax = NEG;
        for (int c = 0; c < NKB; c++) {
            for (int e = 0; e < 2; e++) {
                const int col = j0 + c * 8 + fn + e;
                const float sv = col < k_valid ? sf[c].thread_elements()[e] * scale_log2 : NEG;
                sf[c].thread_elements()[e] = sv;
                cmax = metal::max(cmax, sv);
            }
        }
        cmax = metal::max(cmax, simd_shuffle_xor(cmax, ushort(1)));
        cmax = metal::max(cmax, simd_shuffle_xor(cmax, ushort(8)));
        const float m_new = metal::max(row_m, cmax);
        const float alpha = metal::fast::exp2(row_m - m_new);
        row_m = m_new;
        row_l *= alpha;
        for (int c = 0; c < NKB; c++) {
            for (int e = 0; e < 2; e++) {
                const float pv = metal::fast::exp2(sf[c].thread_elements()[e] - m_new);
                row_l += pv;
                pf[c].thread_elements()[e] = static_cast<T>(pv);
            }
        }
        for (int kk = 0; kk < 16; kk++) {
            acc[kk].thread_elements()[0] *= alpha;
            acc[kk].thread_elements()[1] *= alpha;
        }
        for (int kk = 0; kk < 16; kk++) {
            for (int c = 0; c < NKB; c++) {
                simdgroup_matrix<T, 8, 8> vm;
                simdgroup_load(vm, (const threadgroup T*)(vsm + c * 8 * LD + kk * 8), LD);
                simdgroup_multiply_accumulate(acc[kk], pf[c], vm, acc[kk]);
            }
        }
    }
}
float l = row_l;
l += simd_shuffle_xor(l, ushort(1));
l += simd_shuffle_xor(l, ushort(8));
const float inv = l > 0.0f ? 1.0f / l : 0.0f;
const int orow = srow0 + fm;
if (orow < q_valid) {
    device T* op = out + ((long(head) * n_video + q_tile) * TILE + orow) * D + fn;
    for (int kk = 0; kk < 16; kk++) {
        op[kk * 8] = static_cast<T>(acc[kk].thread_elements()[0] * inv);
        op[kk * 8 + 1] = static_cast<T>(acc[kk].thread_elements()[1] * inv);
    }
}