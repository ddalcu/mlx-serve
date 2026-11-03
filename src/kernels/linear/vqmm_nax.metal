using namespace metal;
using namespace mpp::tensor_ops;

constexpr int BM = BMT;
constexpr int BN = 32;
constexpr int BK = 16;
constexpr int NSG = 8;
constexpr int K = KCONST;
constexpr int K_bytes = K * BITS / 8;
constexpr int K_by_gs = K / GS;
constexpr int K_chunk = K / NSG;

uint tid = thread_position_in_threadgroup.x;
uint sg_id = simdgroup_index_in_threadgroup;
uint lane = thread_index_in_simdgroup;
uint tg_n = threadgroup_position_in_grid.y;
int N = int(N_size);
int n0 = int(tg_n) * BN;
int k_begin = int(sg_id) * K_chunk;
int k_end = k_begin + K_chunk;

threadgroup T B_tile[NSG][BK * BN];
threadgroup float partial[NSG][BM * BN];

constexpr auto desc = matmul2d_descriptor(
    BM,
    32,
    16,
    false,
    false,
    false,
    matmul2d_descriptor::mode::multiply_accumulate);
matmul2d<desc, metal::execution_simdgroup> op;

tensor<device T, dextents<int, 2>, tensor_inline> A(
    (device T*)x,
    dextents<int, 2>{K, BM},
    array<int, 2>{1, K});
tensor<threadgroup T, dextents<int, 2>, tensor_inline> B(
    B_tile[sg_id],
    dextents<int, 2>{BN, BK},
    array<int, 2>{1, BN});
tensor<threadgroup float, dextents<int, 2>, tensor_inline> C(
    partial[sg_id],
    dextents<int, 2>{BN, BM},
    array<int, 2>{1, BN});

auto ct_c = op.template get_destination_cooperative_tensor<
    tensor<device T, extents<int, 16, BM>, tensor_inline>,
    tensor<threadgroup T, extents<int, 32, 16>, tensor_inline>,
    float>();
_Pragma("unroll")
for (uint16_t i = 0; i < ct_c.get_capacity(); ++i) {
    ct_c[i] = 0.0f;
}

int n_global = n0 + int(lane);
for (int k0 = k_begin; k0 < k_end; k0 += BK) {
    const device uchar* wp =
        ((const device uchar*)w_q) + n_global * K_bytes + (k0 * BITS) / 8;
    float scale = float(scales[n_global * K_by_gs + (k0 / GS)]);
    float bias = float(biases[n_global * K_by_gs + (k0 / GS)]);

    if constexpr (BITS == 4) {
        _Pragma("unroll")
        for (int pack = 0; pack < 2; ++pack) {
            uint32_t p =
                uint32_t(wp[pack * 4 + 0]) |
                (uint32_t(wp[pack * 4 + 1]) << 8) |
                (uint32_t(wp[pack * 4 + 2]) << 16) |
                (uint32_t(wp[pack * 4 + 3]) << 24);
            _Pragma("unroll")
            for (int ki = 0; ki < 8; ++ki) {
                uint32_t q = (p >> (ki * 4)) & 0xFu;
                B_tile[sg_id][(pack * 8 + ki) * BN + int(lane)] =
                    T(float(q) * scale + bias);
            }
        }
    } else if constexpr (BITS == 5) {
        _Pragma("unroll")
        for (int pack = 0; pack < 2; ++pack) {
            ulong p =
                ulong(wp[pack * 5 + 0]) |
                (ulong(wp[pack * 5 + 1]) << 8) |
                (ulong(wp[pack * 5 + 2]) << 16) |
                (ulong(wp[pack * 5 + 3]) << 24) |
                (ulong(wp[pack * 5 + 4]) << 32);
            _Pragma("unroll")
            for (int ki = 0; ki < 8; ++ki) {
                uint32_t q = uint32_t((p >> (ki * 5)) & 0x1Ful);
                B_tile[sg_id][(pack * 8 + ki) * BN + int(lane)] =
                    T(float(q) * scale + bias);
            }
        }
    } else if constexpr (BITS == 6) {
        _Pragma("unroll")
        for (int pack = 0; pack < 4; ++pack) {
            uint32_t p =
                uint32_t(wp[pack * 3 + 0]) |
                (uint32_t(wp[pack * 3 + 1]) << 8) |
                (uint32_t(wp[pack * 3 + 2]) << 16);
            _Pragma("unroll")
            for (int ki = 0; ki < 4; ++ki) {
                uint32_t q = (p >> (ki * 6)) & 0x3Fu;
                B_tile[sg_id][(pack * 4 + ki) * BN + int(lane)] =
                    T(float(q) * scale + bias);
            }
        }
    } else {
        static_assert(BITS == 8, "unsupported NAX affine width");
        _Pragma("unroll")
        for (int ki = 0; ki < BK; ++ki) {
            uint32_t q = uint32_t(wp[ki]);
            B_tile[sg_id][ki * BN + int(lane)] =
                T(float(q) * scale + bias);
        }
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);

    auto tA = A.template slice<16, BM>(k0, 0);
    auto tB = B.template slice<32, 16>(0, 0);
    op.run(tA, tB, ct_c);
    simdgroup_barrier(mem_flags::mem_threadgroup);
}

auto tC = C.template slice<32, BM>(0, 0);
ct_c.store(tC);
threadgroup_barrier(mem_flags::mem_threadgroup);

for (int off = int(tid); off < BM * BN; off += NSG * 32) {
    float acc01 = partial[0][off] + partial[1][off];
    float acc23 = partial[2][off] + partial[3][off];
    float acc45 = partial[4][off] + partial[5][off];
    float acc67 = partial[6][off] + partial[7][off];
    float acc = (acc01 + acc23) + (acc45 + acc67);
    int row = off / BN;
    int col = off - row * BN;
    y[row * N + n0 + col] = T(acc);
}