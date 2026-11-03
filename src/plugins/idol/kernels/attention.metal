// Ported from idol-sdk-mlx (idol/mlx_dit.py `_NAX_ATTN_SOURCE`, Apache-2.0). Flash attention at head width 64 on the
// NAX tensor unit: q, k, v f16 heads-major [1, H, L, 64], out f16 token-major [L, H*64]. Per simdgroup 16 query rows:
// S = Q K^T as one 16x32x64 op, P V as one 16x64x32 op, both reading K and V from device memory, P in f16 straight
// from the score tile's registers; scores, softmax and accumulation in f32. HAS_BIAS adds kbias [L] (f32, base 2) to
// every query's score of each key.
using ext = metal::dextents<int32_t, 2>;
using strides = metal::array<int32_t, 2>;
using dtensor = metal::tensor<device half, ext, metal::tensor_inline>;
constexpr int D = 64, BQ = 64, BK = 32;
const uint lane = thread_index_in_simdgroup, sg = simdgroup_index_in_threadgroup;
const int qb = threadgroup_position_in_grid.x, h = threadgroup_position_in_grid.y;
const int H = q_shape[1], L = q_shape[2], HD = H * D;
const short fn = (((lane >> 2) & 2) | (lane & 1)) * 4;
const float sc = scale * M_LOG2E_F;

dtensor tQ((device half*)q + (long)h * L * D, ext(D, L), strides{1, D});
dtensor tK((device half*)k + (long)h * L * D, ext(D, L), strides{1, D});
dtensor tV((device half*)v + (long)h * L * D, ext(D, L), strides{1, D});
dtensor tO(out + h * D, ext(D, L), strides{1, HD});

constexpr auto dqk = matmul2d_descriptor(16, BK, D, false, true, true, matmul2d_descriptor::mode::multiply_accumulate);
constexpr auto dpv = matmul2d_descriptor(16, D, BK, false, false, true, matmul2d_descriptor::mode::multiply_accumulate);
matmul2d<dqk, metal::execution_simdgroup> qk;
matmul2d<dpv, metal::execution_simdgroup> pv;

const int row0 = qb * BQ + sg * 16;
if (row0 >= L) return;
const bool qfull = row0 + 16 <= L;
auto Qc = qk.get_left_input_cooperative_tensor<half, half, float>();
if (qfull) Qc.load(tQ.slice<D, 16>(0, row0)); else Qc.load(tQ.slice(0, row0));
using qct_t = metal::remove_addrspace_t<decltype(Qc)>;
using kslice_t = decltype(tK.slice<D, BK>(0, 0));
using vslice_t = decltype(tV.slice<D, BK>(0, 0));
using pct_t = decltype(pv.get_left_input_cooperative_tensor<half, half, float>(
    metal::declval<thread decltype(qk.get_destination_cooperative_tensor<qct_t, kslice_t, half>())&>()));
auto Oc = pv.get_destination_cooperative_tensor<metal::remove_addrspace_t<pct_t>, vslice_t, float>();
#pragma clang loop unroll(full)
for (uint16_t i = 0; i < Oc.get_capacity(); i++) Oc[i] = 0;
float m[2] = {-FLT_MAX, -FLT_MAX}, s[2] = {0, 0};

// Score / output tile element i: column (i / 8) * 16 + fn + i % 4 of row fm + (i / 4) % 2 * 8, fm this lane's row.
for (int kb = 0; kb * BK < L; kb++) {
    const bool tail = (kb + 1) * BK > L;
    auto Sc = qk.get_destination_cooperative_tensor<qct_t, kslice_t, float>();
#pragma clang loop unroll(full)
    for (uint16_t i = 0; i < Sc.get_capacity(); i++) Sc[i] = 0;
    if (!tail) { auto ks = tK.slice<D, BK>(0, kb * BK); qk.run(Qc, ks, Sc); }
    else { auto ks = tK.slice(0, kb * BK); qk.run(Qc, ks, Sc); }

    float mx_[2] = {m[0], m[1]}, f[2], sum[2] = {0, 0};
#pragma clang loop unroll(full)
    for (int i = 0; i < BK / 2; i++) {
        float e = Sc[i] * sc;
        const int col = kb * BK + (i / 8) * 16 + fn + i % 4;
        if (HAS_BIAS) e += kbias[metal::min(col, L - 1)];
        if (tail && col >= L) e = -FLT_MAX;
        Sc[i] = e;
        mx_[i / 4 % 2] = metal::max(mx_[i / 4 % 2], e);
    }
#pragma clang loop unroll(full)
    for (int r = 0; r < 2; r++) {
        mx_[r] = metal::max(mx_[r], simd_shuffle_xor(mx_[r], 1));
        mx_[r] = metal::max(mx_[r], simd_shuffle_xor(mx_[r], 8));
        f[r] = fast::exp2(m[r] - mx_[r]);
        m[r] = mx_[r];
    }
    auto Ph = qk.get_destination_cooperative_tensor<qct_t, kslice_t, half>();
#pragma clang loop unroll(full)
    for (int i = 0; i < BK / 2; i++) {
        const float p = fast::exp2(Sc[i] - mx_[i / 4 % 2]);
        Ph[i] = half(p);
        sum[i / 4 % 2] += p;
    }
#pragma clang loop unroll(full)
    for (int r = 0; r < 2; r++) {
        sum[r] += simd_shuffle_xor(sum[r], 1);
        sum[r] += simd_shuffle_xor(sum[r], 8);
        s[r] = s[r] * f[r] + sum[r];
    }
#pragma clang loop unroll(full)
    for (int i = 0; i < D / 2; i++) Oc[i] *= f[i / 4 % 2];
    auto Pc = pv.get_left_input_cooperative_tensor<half, half, float>(Ph);
    if (!tail) { auto vs = tV.slice<D, BK>(0, kb * BK); pv.run(Pc, vs, Oc); }
    else { auto vs = tV.slice(0, kb * BK); pv.run(Pc, vs, Oc); }
}

const float inv[2] = {1.0f / s[0], 1.0f / s[1]};
auto Oh = pv.get_destination_cooperative_tensor<metal::remove_addrspace_t<pct_t>, vslice_t, half>();
#pragma clang loop unroll(full)
for (int i = 0; i < D / 2; i++) Oh[i] = half(Oc[i] * inv[i / 4 % 2]);
if (qfull) { auto os = tO.slice<D, 16>(0, row0); Oh.store(os); } else { auto os = tO.slice(0, row0); Oh.store(os); }
