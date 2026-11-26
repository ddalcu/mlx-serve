// Inkling 2, 2 to 16 rows on the NAX tensor unit: relsplit.metal's split-K relative attention as one 16-row pass, for up to
// four slots' blocks a launch (a verify of several slots).
// grid (splits * 32 * G, HKV, 4), threadgroup (32 * G, 1, 1): a threadgroup's G = HQ / HKV SIMD groups walk the same
// K/V tiles of one KV head of lane z, the lane's rows L of the stacked projections.
// q [1, LI, HQ, D] (T), k0..k3 [1, HKV, SC, D] (T, a lane's cache, V laid out as K), kb, vb [1, HKV, LI, D] (T, the lanes'
// own keys and values stacked, unread when SC == S), rel [1, LI, HQ, R] (T, unscaled bias), sinks [1, LI, HQ] (T),
// GATE [LI * HQ * D] f32 (sigmoid), LI the rows of every lane of the call, and sizes = { per lane: L (0 = no lane), S,
// first, SC, offset, row_in (its first row of those), row_out (its first row of this launch's), 0; then kps,
// log_n_floor, LT (this launch's rows) }, params = { scale, log_alpha },
// DIRECT == 0: PART [splits, LT, HQ, D] f32, ML [splits, LT, HQ, 2] f32: a split past a lane's keys joins at weight 0.
// DIRECT == 1 (splits == 1): OUT [1, LT, HQ * D] (T), normalised and gated in place.
using ext = metal::dextents<int32_t, 2>;
using strides = metal::array<int32_t, 2>;
using dtensor = metal::tensor<device T, ext, metal::tensor_inline>;
constexpr int BK = 32;
constexpr float LOG2E = 1.4426950408889634f;
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
const uint sg = simdgroup_index_in_threadgroup;
const uint lane = thread_index_in_simdgroup;
const int head = kvh * G + int(sg);
const short fm = ((lane >> 2) & 4) | ((lane >> 1) & 3);
const short fn = (((lane >> 2) & 2) | (lane & 1)) * 4;
const float sc = params[0] * LOG2E;
const float log_alpha = params[1];

const int ra = fm, rb = fm + 8;
const bool oka = ra < L, okb = rb < L;
int da = ra, db = rb;
uint ma = 0xFFFFFFFFu, mb = 0xFFFFFFFFu;
if constexpr (TREE) {
    if (oka) { da = TMETA[(row_in + ra) * 2]; ma = uint(TMETA[(row_in + ra) * 2 + 1]); }
    if (okb) { db = TMETA[(row_in + rb) * 2]; mb = uint(TMETA[(row_in + rb) * 2 + 1]); }
}
const int dga = S - L + da, dgb = S - L + db;

float tau[2] = {1.0f, 1.0f};
if constexpr (GLOBAL) {
    const int ta = offset + 1 + da;
    const int tb = offset + 1 + db;
    if (oka && ta > log_n_floor) tau[0] = 1.0f + log_alpha * metal::log(float(ta) / float(log_n_floor));
    if (okb && tb > log_n_floor) tau[1] = 1.0f + log_alpha * metal::log(float(tb) / float(log_n_floor));
}

dtensor tQ((device T*)q + size_t(row_in) * HQ * D + head * D, ext(D, L), strides{1, HQ * D});
constexpr auto dqk = matmul2d_descriptor(16, BK, D, false, true, true, matmul2d_descriptor::mode::multiply_accumulate);
constexpr auto dpv = matmul2d_descriptor(16, D, BK, false, false, true, matmul2d_descriptor::mode::multiply_accumulate);
matmul2d<dqk, metal::execution_simdgroup> qk;
matmul2d<dpv, metal::execution_simdgroup> pv;

auto Qc = qk.template get_left_input_cooperative_tensor<T, T, float>();
if (L == 16) Qc.load(tQ.template slice<D, 16>(0, 0)); else Qc.load(tQ.slice(0, 0));
using qct_t = metal::remove_addrspace_t<decltype(Qc)>;
using kslice_t = decltype(tQ.template slice<D, BK>(0, 0));
using vslice_t = decltype(tQ.template slice<D, BK>(0, 0));
using pct_t = decltype(pv.template get_left_input_cooperative_tensor<T, T, float>(
    metal::declval<thread decltype(qk.template get_destination_cooperative_tensor<qct_t, kslice_t, T>())&>()));
auto Oc = pv.template get_destination_cooperative_tensor<metal::remove_addrspace_t<pct_t>, vslice_t, float>();
#pragma clang loop unroll(full)
for (uint16_t i = 0; i < Oc.get_capacity(); i++) Oc[i] = 0;

float m[2] = {-1e30f, -1e30f};
float den[2] = {0.0f, 0.0f};
if (split == 0) {
    if (oka) {
        m[0] = float(sinks[(row_in + ra) * sinks_strides[1] + head * sinks_strides[2]]) * LOG2E;
        den[0] = 1.0f;
    }
    if (okb) {
        m[1] = float(sinks[(row_in + rb) * sinks_strides[1] + head * sinks_strides[2]]) * LOG2E;
        den[1] = 1.0f;
    }
}

const device T* biasa = rel + (size_t(row_in + (oka ? ra : 0)) * HQ + head) * R;
const device T* biasb = rel + (size_t(row_in + (okb ? rb : 0)) * HQ + head) * R;
const int kbeg = first + split * kps;
const int kend = metal::min(S, kbeg + kps);

// 1. Cached keys [kbeg, min(kend, SC))
const int c_end = metal::min(kend, SC);
if (kbeg < c_end) {
    dtensor tK((device T*)kp[zl] + kvh * k_head[zl], ext(D, SC), strides{1, int(k_row[zl])});
    dtensor tV((device T*)vp[zl] + kvh * k_head[zl], ext(D, SC), strides{1, int(k_row[zl])});
    for (int k0 = kbeg; k0 < c_end; k0 += BK) {
        const bool full = (k0 + BK <= c_end);
        auto Sc = qk.template get_destination_cooperative_tensor<qct_t, kslice_t, float>();
#pragma clang loop unroll(full)
        for (uint16_t i = 0; i < Sc.get_capacity(); i++) Sc[i] = 0;
        if (full) { auto ks = tK.template slice<D, BK>(0, k0); qk.run(Qc, ks, Sc); }
        else { auto ks = tK.slice(0, k0); qk.run(Qc, ks, Sc); }

        const bool plain = WINDOW == 0 && full && (k0 + BK - 1 <= (S - L) - R);
        float mx_[2] = {m[0], m[1]}, f[2], sum[2] = {0.0f, 0.0f};
#pragma clang loop unroll(full)
        for (int i = 0; i < BK / 2; i++) {
            const int h = (i / 4) % 2;
            const bool okr = h ? okb : oka;
            float e = -INFINITY;
            if (plain) {
                if (okr) e = Sc[i] * sc * tau[h];
            } else {
                const int col = k0 + (i / 8) * 16 + fn + i % 4;
                const int dist = (h ? dgb : dga) - col;
                const bool ok = okr && col < c_end && dist >= 0 && (WINDOW == 0 || dist < WINDOW);
                if (ok) {
                    const float b = dist < R ? float((h ? biasb : biasa)[dist]) * LOG2E : 0.0f;
                    e = (Sc[i] * sc + b) * tau[h];
                }
            }
            Sc[i] = e;
            mx_[h] = metal::max(mx_[h], e);
        }
#pragma clang loop unroll(full)
        for (int r = 0; r < 2; r++) {
            mx_[r] = metal::max(mx_[r], simd_shuffle_xor(mx_[r], 1));
            mx_[r] = metal::max(mx_[r], simd_shuffle_xor(mx_[r], 8));
            f[r] = fast::exp2(m[r] - mx_[r]);
            m[r] = mx_[r];
        }
        auto Ph = qk.template get_destination_cooperative_tensor<qct_t, kslice_t, T>();
#pragma clang loop unroll(full)
        for (int i = 0; i < BK / 2; i++) {
            const float p = fast::exp2(Sc[i] - mx_[(i / 4) % 2]);
            Ph[i] = T(p);
            sum[(i / 4) % 2] += p;
        }
#pragma clang loop unroll(full)
        for (int r = 0; r < 2; r++) {
            sum[r] += simd_shuffle_xor(sum[r], 1);
            sum[r] += simd_shuffle_xor(sum[r], 8);
            den[r] = den[r] * f[r] + sum[r];
        }
#pragma clang loop unroll(full)
        for (int i = 0; i < D / 2; i++) Oc[i] *= f[(i / 4) % 2];
        auto Pc = pv.template get_left_input_cooperative_tensor<T, T, float>(Ph);
        if (full) { auto vs = tV.template slice<D, BK>(0, k0); pv.run(Pc, vs, Oc); }
        else { auto vs = tV.slice(0, k0); pv.run(Pc, vs, Oc); }
    }
}

// 2. Own verify keys [max(kbeg, SC), kend) (at most 16 keys: one NAX tile)
const int o_beg = metal::max(kbeg, SC);
if (o_beg < kend) {
    const int n_own = S - SC;
    dtensor tKb((device T*)kb + kvh * kb_strides[1] + size_t(row_in) * kb_strides[2], ext(D, n_own), strides{1, int(kb_strides[2])});
    dtensor tVb((device T*)vb + kvh * kb_strides[1] + size_t(row_in) * kb_strides[2], ext(D, n_own), strides{1, int(kb_strides[2])});
    for (int k0 = o_beg; k0 < kend; k0 += BK) {
        const int off = k0 - SC;
        auto Sc = qk.template get_destination_cooperative_tensor<qct_t, kslice_t, float>();
#pragma clang loop unroll(full)
        for (uint16_t i = 0; i < Sc.get_capacity(); i++) Sc[i] = 0;
        { auto ks = tKb.slice(0, off); qk.run(Qc, ks, Sc); }

        float mx_[2] = {m[0], m[1]}, f[2], sum[2] = {0.0f, 0.0f};
#pragma clang loop unroll(full)
        for (int i = 0; i < BK / 2; i++) {
            const int h = (i / 4) % 2;
            const int col = k0 + (i / 8) * 16 + fn + i % 4;
            int dist;
            bool anc = true;
            if constexpr (TREE) {
                const int kc = col - SC;
                const int kdepth = (kc >= 0 && kc < L) ? TMETA[(row_in + kc) * 2] : kc;
                anc = (kc >= 0 && kc < L) && ((((h ? mb : ma) >> kc) & 1u) != 0u);
                dist = (h ? db : da) - kdepth;
            } else {
                dist = (h ? dgb : dga) - col;
            }
            const bool ok = (h ? okb : oka) && col < kend && anc && dist >= 0 && (WINDOW == 0 || dist < WINDOW);
            float e = -INFINITY;
            if (ok) {
                const float b = dist < R ? float((h ? biasb : biasa)[dist]) * LOG2E : 0.0f;
                e = (Sc[i] * sc + b) * tau[h];
            }
            Sc[i] = e;
            mx_[h] = metal::max(mx_[h], e);
        }
#pragma clang loop unroll(full)
        for (int r = 0; r < 2; r++) {
            mx_[r] = metal::max(mx_[r], simd_shuffle_xor(mx_[r], 1));
            mx_[r] = metal::max(mx_[r], simd_shuffle_xor(mx_[r], 8));
            f[r] = fast::exp2(m[r] - mx_[r]);
            m[r] = mx_[r];
        }
        auto Ph = qk.template get_destination_cooperative_tensor<qct_t, kslice_t, T>();
#pragma clang loop unroll(full)
        for (int i = 0; i < BK / 2; i++) {
            const float p = fast::exp2(Sc[i] - mx_[(i / 4) % 2]);
            Ph[i] = T(p);
            sum[(i / 4) % 2] += p;
        }
#pragma clang loop unroll(full)
        for (int r = 0; r < 2; r++) {
            sum[r] += simd_shuffle_xor(sum[r], 1);
            sum[r] += simd_shuffle_xor(sum[r], 8);
            den[r] = den[r] * f[r] + sum[r];
        }
#pragma clang loop unroll(full)
        for (int i = 0; i < D / 2; i++) Oc[i] *= f[(i / 4) % 2];
        auto Pc = pv.template get_left_input_cooperative_tensor<T, T, float>(Ph);
        { auto vs = tVb.slice(0, off); pv.run(Pc, vs, Oc); }
    }
}

if constexpr (DIRECT) {
    const float inv[2] = {1.0f / den[0], 1.0f / den[1]};
    using gtensor = metal::tensor<device float, ext, metal::tensor_inline>;
    gtensor tG((device float*)GATE + size_t(row_in) * HQ * D + head * D, ext(D, L), strides{1, HQ * D});
    auto Gc = pv.template get_destination_cooperative_tensor<metal::remove_addrspace_t<pct_t>, vslice_t, float>();
    if (L == 16) Gc.load(tG.template slice<D, 16>(0, 0)); else Gc.load(tG.slice(0, 0));
    auto Oh = pv.template get_destination_cooperative_tensor<metal::remove_addrspace_t<pct_t>, vslice_t, T>();
#pragma clang loop unroll(full)
    for (int i = 0; i < D / 2; i++) Oh[i] = T(float(T(Oc[i] * inv[(i / 4) % 2])) * Gc[i]);
    dtensor tO((device T*)OUT + size_t(row_out) * HQ * D + head * D, ext(D, L), strides{1, HQ * D});
    if (L == 16) { auto os = tO.template slice<D, 16>(0, 0); Oh.store(os); }
    else { auto os = tO.slice(0, 0); Oh.store(os); }
} else {
    using ptensor = metal::tensor<device float, ext, metal::tensor_inline>;
    ptensor tP(PART + (size_t(split) * LT + row_out) * HQ * D + head * D, ext(D, L), strides{1, HQ * D});
    if (L == 16) { auto ps = tP.template slice<D, 16>(0, 0); Oc.store(ps); }
    else { auto ps = tP.slice(0, 0); Oc.store(ps); }
    if (fn == 0) {
        if (oka) {
            const size_t at = (size_t(split) * LT + row_out + ra) * HQ + head;
            ML[at * 2] = m[0];
            ML[at * 2 + 1] = den[0];
        }
        if (okb) {
            const size_t at = (size_t(split) * LT + row_out + rb) * HQ + head;
            ML[at * 2] = m[1];
            ML[at * 2 + 1] = den[1];
        }
    }
}
