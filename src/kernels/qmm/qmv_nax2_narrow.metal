const int K = x_shape[x_ndim - 1];
int M = 1;
for (int i = 0; i < x_ndim - 1; ++i) M *= x_shape[i];
const int N = w_shape[0];
constexpr int GS = 128;
const int K_w = K / 16;   // uint32 words per weight row
const int K_g = K / GS;
typedef metal::vec<T, 8> frag_t;

threadgroup float red_all[2 * 16 * 32];
const uint simd_lid = thread_index_in_simdgroup;
const uint cb = simdgroup_index_in_threadgroup >> 1;
const uint ks = simdgroup_index_in_threadgroup & 1;
threadgroup float* red = red_all + cb * (16 * 32);
const int col0 = int(threadgroup_position_in_grid.y) * 64 + 32 * int(cb);
// Rows come in 16-high blocks; the last one is short when M is not a
// multiple of 16. Weights are re-read per block, which is what bounds how
// far this tiling is worth taking (see narrowMaxRows).
const int m0 = int(threadgroup_position_in_grid.z) * 16;
const int rows = min(M - m0, 16);

// Fragment coordinate of this lane (BaseNAXFrag::get_coord): elements
// 0..3 sit at (fm, fn..fn+3), elements 4..7 at (fm + 8, fn..fn+3).
const short qid = short(simd_lid >> 2);
const short fm = ((qid & 4) | short((simd_lid >> 1) & 3));
const short fn = ((qid & 2) | short(simd_lid & 1)) * 4;
// This lane's four k values of a 16-code word are byte (fn / 4).
const ushort bsh = ushort(8 * (fn >> 2));

// Weight rows col0 + fm + 8 * j (j = 0, 1 -> B0; 2, 3 -> B1), clamped.
int wrow[4];
#pragma unroll
for (int j = 0; j < 4; ++j) wrow[j] = min(col0 + int(fm) + 8 * j, N - 1);
// Activation rows fm and fm + 8, clamped to the live rows (never stored).
const device T* xa0 = x + (m0 + min(int(fm), rows - 1)) * K + fn;
const device T* xa1 = x + (m0 + min(int(fm) + 8, rows - 1)) * K + fn;

// The accumulator lives in the tensor op's destination cooperative
// tensor for the whole K loop and is copied out once at the end.
constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(
    16, 32, 16, false, true, true,
    mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
mpp::tensor_ops::matmul2d<desc, metal::execution_simdgroup> gemm_op;
auto ct_a = gemm_op.template get_left_input_cooperative_tensor<T, T, float>();
auto ct_b = gemm_op.template get_right_input_cooperative_tensor<T, T, float>();
auto ct_c = gemm_op.template get_destination_cooperative_tensor<
    metal::remove_addrspace_t<decltype(ct_a)>,
    metal::remove_addrspace_t<decltype(ct_b)>, float>();
#pragma unroll
for (short i = 0; i < 16; ++i) ct_c[i] = 0.0f;

const int n_groups = K_g;
const ushort wq = ushort(fn >> 2);   // this lane's uint4 within the line
ushort qlane[4];
#pragma unroll
for (int st = 0; st < 4; ++st)
  qlane[st] = ushort((simd_lid & ~0x9u) | uint(st & 1) | (uint(st >> 1) << 3));
// Blocks are numbered per simdgroup: block i covers groups
// 2 * (ks + 2 * i) and the one after it.
const int n_blocks_total = (n_groups + 1) / 2;
const int my_blocks = max((n_blocks_total - int(ks) + 1) / 2, 0);
auto block_line = [&](int i, int j) -> uint4 {
  const int gb = 2 * (int(ks) + 2 * i);
  return *((const device uint4*)(w + wrow[j] * K_w + gb * 8) + wq);
};
uint4 ring[2][4];
#pragma unroll
for (int r = 0; r < 2; ++r) {
  if (r < my_blocks) {
#pragma unroll
    for (int j = 0; j < 4; ++j) ring[r][j] = block_line(r, j);
  }
}

for (int i = 0; i < my_blocks; ++i) {
  const int gb = 2 * (int(ks) + 2 * i);
  volatile int compiler_barrier;
#pragma unroll
  for (int gh = 0; gh < 2; ++gh) {
    const int g = gb + gh;
    if (g >= n_groups) break;
    float s0[4], s1[4], s2[4], s3[4], bz[4];
    half2 hs[4], hb[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      hs[j] = half2(half(scales[wrow[j] * K_g + g]));
      hb[j] = half2(half(biases[wrow[j] * K_g + g]));
      const float sv = float(scales[wrow[j] * K_g + g]);
      bz[j] = float(biases[wrow[j] * K_g + g]);
      s0[j] = sv;
      s1[j] = sv * 0.25f;
      s2[j] = sv * 0.0625f;
      s3[j] = sv * 0.015625f;
    }
#pragma unroll
    for (int st8 = 0; st8 < 8; ++st8) {
      const int st = gh * 8 + st8;   // 0..15 within the block
      const int k = g * GS + st8 * 16;
      frag_t B0, B1;
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        const uint word = simd_shuffle(ring[0][j][st & 3], qlane[st >> 2]);
        const uint by = (word >> bsh) & 0xffu;
        float v0, v1, v2, v3;
        if (HDQ) {
          // Codes 0..3 dropped into the mantissa of 1024.0h come out as
          // exact halves: no int->float conversion, one rounding as before.
          const uint u = by | (by << 14);
          const half2 h01 = as_type<half2>((u & 0x00030003u) | 0x64006400u) - half2(1024.0h);
          const half2 h23 = as_type<half2>(((u >> 4) & 0x00030003u) | 0x64006400u) - half2(1024.0h);
          const half2 d01 = fma(h01, hs[j], hb[j]);
          const half2 d23 = fma(h23, hs[j], hb[j]);
          v0 = d01.x; v1 = d01.y; v2 = d23.x; v3 = d23.y;
        } else {
          v0 = s0[j] * float(by & 0x03u) + bz[j];
          v1 = s1[j] * float(by & 0x0cu) + bz[j];
          v2 = s2[j] * float(by & 0x30u) + bz[j];
          v3 = s3[j] * float(by & 0xc0u) + bz[j];
        }
        if (j < 2) {
          B0[4 * j + 0] = T(v0);
          B0[4 * j + 1] = T(v1);
          B0[4 * j + 2] = T(v2);
          B0[4 * j + 3] = T(v3);
        } else {
          B1[4 * (j - 2) + 0] = T(v0);
          B1[4 * (j - 2) + 1] = T(v1);
          B1[4 * (j - 2) + 2] = T(v2);
          B1[4 * (j - 2) + 3] = T(v3);
        }
      }
#pragma unroll
      for (int q = 0; q < 4; ++q) {
        ct_a[q] = xa0[k + q];
        ct_a[4 + q] = xa1[k + q];
      }
#pragma unroll
      for (short q = 0; q < 8; ++q) {
        ct_b[q] = B0[q];
        ct_b[8 + q] = B1[q];
      }
      gemm_op.run(ct_a, ct_b, ct_c);
    }
  }
  (void)compiler_barrier;
#pragma unroll
  for (int j = 0; j < 4; ++j) ring[0][j] = ring[1][j];
  if (i + 2 < my_blocks) {
#pragma unroll
    for (int j = 0; j < 4; ++j) ring[1][j] = block_line(i + 2, j);
  }
}

// Sum the two K slices onto ks == 0; identical fragment layouts line up
// element by element.
metal::vec<float, 8> C0, C1;
#pragma unroll
for (short i = 0; i < 8; ++i) {
  C0[i] = ct_c[i];
  C1[i] = ct_c[8 + i];
}
if (ks == 1) {
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    red[i * 32 + simd_lid] = C0[i];
    red[(8 + i) * 32 + simd_lid] = C1[i];
  }
}
threadgroup_barrier(mem_flags::mem_threadgroup);
if (ks != 0) return;
#pragma unroll
for (int i = 0; i < 8; ++i) {
  C0[i] += red[i * 32 + simd_lid];
  C1[i] += red[(8 + i) * 32 + simd_lid];
}
// C0[i] holds row fm + (i / 4) * 8, column col0 + fn + i % 4; C1 the
// column 16 further.
#pragma unroll
for (int i = 0; i < 8; ++i) {
  const int v = int(fm) + (i / 4) * 8;
  const int c = col0 + int(fn) + (i % 4);
  if (v < rows) {
    if (c < N) y[(m0 + v) * N + c] = static_cast<T>(C0[i]);
    if (c + 16 < N) y[(m0 + v) * N + c + 16] = static_cast<T>(C1[i]);
  }
}