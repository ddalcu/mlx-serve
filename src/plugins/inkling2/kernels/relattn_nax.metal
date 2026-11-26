// Inkling 2, a prompt's rows on the NAX tensor unit: relattn.metal's causal flash read (on-the-fly relative bias from
// rel_states . rproj, a sink logit per row, no mask or bias matrix built), a SIMD group a tile of 16 rows of one head.
// S = Q K^T is one 16 x 32 x D matmul2d and P V one 16 x D x 32, both reading K and V from device memory, P entering in
// T straight from the score tile's registers; the scores, the online softmax and the sums stay in f32.
// q [1, HQ, L, D] and k, v [1, HKV, S, D] (T, any strides over a unit last axis), the rows at the last L of the S keys;
// rel [1, L, HQ, DREL] (T, contiguous) each row's clipped relative state, rproj [R, DREL] (T, contiguous) the transposed
// projection; sinks [1, L, HQ] f32; sizes = { L, S }. WINDOW > 0 keeps the keys fewer than WINDOW behind a row.
// grid (ceil(L / 64) * 128, HQ, 1), threadgroup (128, 1, 1); out [1, L, HQ, D] (T).
// Global layers run a separate plain loop past distance R so long-context tiles carry zero bias registers or branches.
using ext = metal::dextents<int32_t, 2>;
using strides = metal::array<int32_t, 2>;
using dtensor = metal::tensor<device T, ext, metal::tensor_inline>;
constexpr int BK = 32;
constexpr float LOG2E = 1.4426950408889634f;
const int L = sizes[0];
const int S = sizes[1];
const uint lane = thread_index_in_simdgroup, sg = simdgroup_index_in_threadgroup;
const int head = threadgroup_position_in_grid.y;
const int kvh = head / (HQ / HKV);
const int row0 = int(threadgroup_position_in_grid.x) * 64 + int(sg) * 16;
if (row0 >= L) return;

// The lane holds columns fn..fn + 3 of each 16-wide piece of the tile, of rows fm and fm + 8.
const short fm = ((lane >> 2) & 4) | ((lane >> 1) & 3);
const short fn = (((lane >> 2) & 2) | (lane & 1)) * 4;
const float sc = scale * LOG2E;

dtensor tQ((device T*)q + head * q_strides[1], ext(D, L), strides{1, int(q_strides[2])});
dtensor tK((device T*)k + kvh * k_strides[1], ext(D, S), strides{1, int(k_strides[2])});
dtensor tV((device T*)v + kvh * v_strides[1], ext(D, S), strides{1, int(v_strides[2])});
dtensor tO(out + head * D, ext(D, L), strides{1, HQ * D});

constexpr auto dqk = matmul2d_descriptor(16, BK, D, false, true, true, matmul2d_descriptor::mode::multiply_accumulate);
constexpr auto dpv = matmul2d_descriptor(16, D, BK, false, false, true, matmul2d_descriptor::mode::multiply_accumulate);
matmul2d<dqk, metal::execution_simdgroup> qk;
matmul2d<dpv, metal::execution_simdgroup> pv;

const bool qfull = row0 + 16 <= L;
auto Qc = qk.template get_left_input_cooperative_tensor<T, T, float>();
if (qfull) Qc.load(tQ.template slice<D, 16>(0, row0)); else Qc.load(tQ.slice(0, row0));
using qct_t = metal::remove_addrspace_t<decltype(Qc)>;
using kslice_t = decltype(tK.template slice<D, BK>(0, 0));
using vslice_t = decltype(tV.template slice<D, BK>(0, 0));
using pct_t = decltype(pv.template get_left_input_cooperative_tensor<T, T, float>(
    metal::declval<thread decltype(qk.template get_destination_cooperative_tensor<qct_t, kslice_t, T>())&>()));
auto Oc = pv.template get_destination_cooperative_tensor<metal::remove_addrspace_t<pct_t>, vslice_t, float>();
#pragma clang loop unroll(full)
for (uint16_t i = 0; i < Oc.get_capacity(); i++) Oc[i] = 0;

// This lane's two rows: a row past L reads row 0's inputs and is never stored.
const int ra = row0 + fm, rb = row0 + fm + 8;
const bool oka = ra < L, okb = rb < L;
const int dga = S - L + ra, dgb = S - L + rb;
// Each row's softmax starts from its sink logit: sum 1, output 0, the sink's value being zero.
float m[2] = {sinks[(oka ? ra : 0) * HQ + head] * LOG2E, sinks[(okb ? rb : 0) * HQ + head] * LOG2E};
float den[2] = {1.0f, 1.0f};

const int d0 = S - L + row0;
const int d_last = S - L + metal::min(row0 + 16, L) - 1;
int k0 = WINDOW > 0 ? metal::max(0, d0 - WINDOW + 1) / BK * BK : 0;

// 1. Plain loop (global layers past distance R): always full, unmasked, zero relative bias.
if constexpr (WINDOW == 0) {
    const int k_plain_lim = d0 - R - BK + 1;
    for (; k0 <= k_plain_lim; k0 += BK) {
        auto Sc = qk.template get_destination_cooperative_tensor<qct_t, kslice_t, float>();
#pragma clang loop unroll(full)
        for (uint16_t i = 0; i < Sc.get_capacity(); i++) Sc[i] = 0;
        { auto ks = tK.template slice<D, BK>(0, k0); qk.run(Qc, ks, Sc); }

        float mx_[2] = {m[0], m[1]}, f[2], sum[2] = {0.0f, 0.0f};
#pragma clang loop unroll(full)
        for (int i = 0; i < BK / 2; i++) {
            const int h = (i / 4) % 2;
            const float e = Sc[i] * sc;
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
        { auto vs = tV.template slice<D, BK>(0, k0); pv.run(Pc, vs, Oc); }
    }
}

// 2. Near/causal loop: tiles within distance R or containing causal/window boundary.
if (k0 <= d_last) {
    const device T* rsa = rel + (size_t(oka ? ra : 0) * HQ + head) * DREL;
    const device T* rsb = rel + (size_t(okb ? rb : 0) * HQ + head) * DREL;
    vec<T, 4> ra4[(DREL + 3) / 4], rb4[(DREL + 3) / 4];
    if constexpr (DREL % 4 == 0) {
        const device vec<T, 4>* sa4 = (const device vec<T, 4>*)rsa;
        const device vec<T, 4>* sb4 = (const device vec<T, 4>*)rsb;
#pragma clang loop unroll(full)
        for (int d = 0; d < DREL / 4; d++) { ra4[d] = sa4[d]; rb4[d] = sb4[d]; }
    }
    for (; k0 <= d_last; k0 += BK) {
        const bool tail = k0 + BK > S;
        auto Sc = qk.template get_destination_cooperative_tensor<qct_t, kslice_t, float>();
#pragma clang loop unroll(full)
        for (uint16_t i = 0; i < Sc.get_capacity(); i++) Sc[i] = 0;
        if (!tail) { auto ks = tK.template slice<D, BK>(0, k0); qk.run(Qc, ks, Sc); }
        else { auto ks = tK.slice(0, k0); qk.run(Qc, ks, Sc); }

        float mx_[2] = {m[0], m[1]}, f[2], sum[2] = {0.0f, 0.0f};
#pragma clang loop unroll(full)
        for (int i = 0; i < BK / 2; i++) {
            const int h = (i / 4) % 2;
            const int col = k0 + (i / 8) * 16 + fn + i % 4;
            const int dist = (h ? dgb : dga) - col;
            const bool ok = (h ? okb : oka) && dist >= 0 && (WINDOW == 0 || dist < WINDOW);
            float e = -INFINITY;
            if (ok) {
                e = Sc[i] * sc;
                if (dist < R) {
                    const device T* rp = rproj + size_t(dist) * DREL;
                    float b = 0.0f;
                    if constexpr (DREL % 4 == 0) {
                        const device vec<T, 4>* rp4 = (const device vec<T, 4>*)rp;
#pragma clang loop unroll(full)
                        for (int d = 0; d < DREL / 4; d++) b += metal::dot(float4(h ? rb4[d] : ra4[d]), float4(rp4[d]));
                    } else {
                        const device T* rv = h ? rsb : rsa;
#pragma clang loop unroll(full)
                        for (int d = 0; d < DREL; d++) b += float(rv[d]) * float(rp[d]);
                    }
                    e += b * LOG2E;
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
        if (!tail) { auto vs = tV.template slice<D, BK>(0, k0); pv.run(Pc, vs, Oc); }
        else { auto vs = tV.slice(0, k0); pv.run(Pc, vs, Oc); }
    }
}

const float inv[2] = {1.0f / den[0], 1.0f / den[1]};
auto Oh = pv.template get_destination_cooperative_tensor<metal::remove_addrspace_t<pct_t>, vslice_t, T>();
#pragma clang loop unroll(full)
for (int i = 0; i < D / 2; i++) Oh[i] = T(Oc[i] * inv[(i / 4) % 2]);
if (qfull) { auto os = tO.template slice<D, 16>(0, row0); Oh.store(os); } else { auto os = tO.slice(0, row0); Oh.store(os); }
