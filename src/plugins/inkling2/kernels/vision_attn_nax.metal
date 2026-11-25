// Inkling 2 vision: bidirectional flash attention on Metal 4 NAX at head width 64.
// Q, K, V [1, H, L, 64] (T: bf16 or f16), scale (0-d f32) -> O [L, H*64] (T, token-major).
// grid (ceil(L / 64) * 128, H, 1), threadgroup (128, 1, 1): 4 SIMD groups per threadgroup, 16 query rows each.
constexpr int D = 64;
constexpr int BQ = 64;
constexpr float LOG2E = 1.4426950408889634f;

const uint sg = simdgroup_index_in_threadgroup;
const int qb = int(threadgroup_position_in_grid.x);
const int head = int(threadgroup_position_in_grid.y);
const int L = Q_shape[2];
const int row0 = qb * BQ + int(sg) * 16;
if (row0 >= L) return;

using ext2 = metal::dextents<int32_t, 2>;
using str2 = metal::array<int32_t, 2>;
using dtensor = metal::tensor<device T, ext2, metal::tensor_inline>;

dtensor tQ((device T*)Q + size_t(head) * L * D, ext2(D, L), str2{1, D});
dtensor tK((device T*)K + size_t(head) * L * D, ext2(D, L), str2{1, D});
dtensor tV((device T*)V + size_t(head) * L * D, ext2(D, L), str2{1, D});
dtensor tO(O + head * D, ext2(D, L), str2{1, H * D});

constexpr auto qk_desc = matmul2d_descriptor(
    16, BK, D, false, true, true, matmul2d_descriptor::mode::multiply_accumulate);
constexpr auto pv_desc = matmul2d_descriptor(
    16, D, BK, false, false, true, matmul2d_descriptor::mode::multiply_accumulate);
matmul2d<qk_desc, metal::execution_simdgroup> qk_op;
matmul2d<pv_desc, metal::execution_simdgroup> pv_op;

const bool qfull = row0 + 16 <= L;
auto q_tile = qk_op.template get_left_input_cooperative_tensor<T, T, float>();
if (qfull) q_tile.load(tQ.template slice<D, 16>(0, row0));
else q_tile.load(tQ.slice(0, row0));

using qct_t = metal::remove_addrspace_t<decltype(q_tile)>;
using kslice_t = decltype(tK.template slice<D, BK>(0, 0));
using vslice_t = decltype(tV.template slice<D, BK>(0, 0));
using pt_dummy_t = decltype(qk_op.template get_destination_cooperative_tensor<qct_t, kslice_t, half>());
using pleft_t = decltype(pv_op.template get_left_input_cooperative_tensor<half, T, float>(metal::declval<thread pt_dummy_t&>()));
using pct_t = metal::remove_addrspace_t<pleft_t>;

auto oc = pv_op.template get_destination_cooperative_tensor<pct_t, vslice_t, float>();
#pragma clang loop unroll(full)
for (uint16_t i = 0; i < oc.get_capacity(); ++i) oc[i] = 0.0f;

float row_max[2] = {-INFINITY, -INFINITY};
float row_sum[2] = {0.0f, 0.0f};
const float qscale = scale * LOG2E;

auto max_nan = [](float a, float b) -> float {
    return isnan(a) ? a : (isnan(b) ? b : max(a, b));
};

for (int k0 = 0; k0 < L; k0 += BK) {
    const bool tail = k0 + BK > L;
    auto sc = qk_op.template get_destination_cooperative_tensor<qct_t, kslice_t, float>();
#pragma clang loop unroll(full)
    for (uint16_t i = 0; i < sc.get_capacity(); ++i) sc[i] = 0.0f;

    if (!tail) {
        auto ks = tK.template slice<D, BK>(0, k0);
        qk_op.run(q_tile, ks, sc);
    } else {
        auto ks = tK.slice(0, k0);
        qk_op.run(q_tile, ks, sc);
    }

    float tile_max[2] = {-INFINITY, -INFINITY};
#pragma clang loop unroll(full)
    for (uint16_t i = 0; i < sc.get_capacity(); ++i) {
        const auto idx = sc.get_multidimensional_index(i);
        const int r = idx[1] >> 3;
        const int col = k0 + idx[0];
        const float val = (tail && col >= L) ? -INFINITY : sc[i] * qscale;
        sc[i] = val;
        tile_max[r] = max_nan(tile_max[r], val);
    }

    float new_max[2];
    float factor[2];
#pragma clang loop unroll(full)
    for (int r = 0; r < 2; ++r) {
        float m = max_nan(tile_max[r], simd_shuffle_xor(tile_max[r], ushort(1)));
        m = max_nan(m, simd_shuffle_xor(m, ushort(8)));
        new_max[r] = max_nan(row_max[r], m);
        factor[r] = (row_max[r] == new_max[r]) ? 1.0f : fast::exp2(row_max[r] - new_max[r]);
        row_max[r] = new_max[r];
    }

    auto pt = qk_op.template get_destination_cooperative_tensor<qct_t, kslice_t, half>();
    float tile_sum[2] = {0.0f, 0.0f};
#pragma clang loop unroll(full)
    for (uint16_t i = 0; i < sc.get_capacity(); ++i) {
        const auto idx = sc.get_multidimensional_index(i);
        const int r = idx[1] >> 3;
        const int col = k0 + idx[0];
        const float p = (tail && col >= L) ? 0.0f : fast::exp2(sc[i] - new_max[r]);
        pt[i] = half(p);
        tile_sum[r] += p;
    }

#pragma clang loop unroll(full)
    for (int r = 0; r < 2; ++r) {
        tile_sum[r] += simd_shuffle_xor(tile_sum[r], ushort(1));
        tile_sum[r] += simd_shuffle_xor(tile_sum[r], ushort(8));
        row_sum[r] = row_sum[r] * factor[r] + tile_sum[r];
    }

#pragma clang loop unroll(full)
    for (uint16_t i = 0; i < oc.get_capacity(); ++i) {
        const int r = oc.get_multidimensional_index(i)[1] >> 3;
        oc[i] *= factor[r];
    }

    auto p_left = pv_op.template get_left_input_cooperative_tensor<half, T, float>(pt);
    if (!tail) {
        auto vs = tV.template slice<D, BK>(0, k0);
        pv_op.run(p_left, vs, oc);
    } else {
        auto vs = tV.slice(0, k0);
        pv_op.run(p_left, vs, oc);
    }
}

const float inv[2] = {1.0f / row_sum[0], 1.0f / row_sum[1]};
auto ot = pv_op.template get_destination_cooperative_tensor<pct_t, vslice_t, T>();
#pragma clang loop unroll(full)
for (uint16_t i = 0; i < oc.get_capacity(); ++i) {
    const int r = oc.get_multidimensional_index(i)[1] >> 3;
    ot[i] = T(oc[i] * inv[r]);
}
if (qfull) {
    auto os = tO.template slice<D, 16>(0, row0);
    ot.store(os);
} else {
    auto os = tO.slice(0, row0);
    ot.store(os);
}
