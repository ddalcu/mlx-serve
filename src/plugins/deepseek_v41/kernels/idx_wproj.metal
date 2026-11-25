// MLX_SERVE_DSV41_PF_IDX_PASS (deepseek_v41.zig `idxHeadWeights`): the indexer's head weights over a whole prompt
// pass, y[L, 32] = x[L, 5120] . W[32, 5120]^T in f32, rounded once to OT. The model's shape is compiled in (N = 32
// index heads, K = 5120 hidden). One threadgroup is 16 rows and all 32 heads; its 8 simdgroups each take an eighth of
// K (640), 20 steps of 32. A lane loads 8 adjacent K values (one 16-byte load) for each operand row it holds and feeds
// them to two 16-deep matrix steps (step A the first 4, step B the last 4: the same K values on both operands, so each
// dot is whole), each step into its own accumulator. Per output: the 8 eighths' A and B partials (16 sums of 320
// products) added as a tree, ((s0 + s1) + (s2 + s3)) + ((s4 + s5) + (s6 + s7)), s = A + B.
// Floor at 17,639 rows: x 180.6 MB read once (0.170 ms at 1.065 TB/s), W 320 KB a threadgroup from L2; 5.8 GFLOP.
constexpr int N = 32;
constexpr int KD = 5120;
constexpr int NSG = 8;
constexpr int KQ = KD / NSG;
static_assert(KQ % 32 == 0, "an eighth of K in steps of 32");

const int L = int(x_shape[0]);
const uint sg = simdgroup_index_in_threadgroup;
const uint lane = thread_index_in_simdgroup;
const int r0 = int(threadgroup_position_in_grid.y) * 16;
const short qid = short(lane >> 2);
const short fm = short((qid & 4) | ((lane >> 1) & 3));
const short fn = short(((qid & 2) | (lane & 1)) * 4);

threadgroup float part[NSG][16][N];

constexpr auto desc = matmul2d_descriptor(16, 32, 16, false, true, false, matmul2d_descriptor::mode::multiply_accumulate);
matmul2d<desc, execution_simdgroup> op;
auto a = op.template get_left_input_cooperative_tensor<T, T, float>();
auto b = op.template get_right_input_cooperative_tensor<T, T, float>();
auto ca = op.template get_destination_cooperative_tensor<
    metal::remove_addrspace_t<decltype(a)>, metal::remove_addrspace_t<decltype(b)>, float>();
auto cb = op.template get_destination_cooperative_tensor<
    metal::remove_addrspace_t<decltype(a)>, metal::remove_addrspace_t<decltype(b)>, float>();
for (short e = 0; e < 16; ++e) {
  ca[e] = 0.0f;
  cb[e] = 0.0f;
}

// This lane's x rows (half i: row r0 + fm + 8 i, clamped into the input; rows past L are never stored) and W rows
// (head 16 tn + fm + 8 i), at its 8 K values of the eighth's first step.
const int k0 = int(sg) * KQ + 2 * fn;
const device T* xr[2];
for (short i = 0; i < 2; ++i) xr[i] = x + size_t(metal::min(r0 + fm + 8 * i, L - 1)) * KD + k0;
const device T* wr[2][2];
for (short tn = 0; tn < 2; ++tn)
  for (short i = 0; i < 2; ++i) wr[tn][i] = w + size_t(16 * tn + fm + 8 * i) * KD + k0;

_Pragma("clang loop unroll_count(4)")
for (int kk = 0; kk < KQ; kk += 32) {
  metal::vec<T, 8> wv[2][2];
  for (short tn = 0; tn < 2; ++tn)
    for (short i = 0; i < 2; ++i) wv[tn][i] = *(const device metal::vec<T, 8>*)(wr[tn][i] + kk);
  metal::vec<T, 8> xv[2];
  for (short i = 0; i < 2; ++i) xv[i] = *(const device metal::vec<T, 8>*)(xr[i] + kk);
  // step A: each operand's first 4 values; step B: the last 4
  for (short tn = 0; tn < 2; ++tn)
    for (short i = 0; i < 2; ++i)
      for (short j = 0; j < 4; ++j) b[tn * 8 + i * 4 + j] = wv[tn][i][j];
  for (short j = 0; j < 4; ++j) {
    a[j] = xv[0][j];
    a[4 + j] = xv[1][j];
  }
  op.run(a, b, ca);
  for (short tn = 0; tn < 2; ++tn)
    for (short i = 0; i < 2; ++i)
      for (short j = 0; j < 4; ++j) b[tn * 8 + i * 4 + j] = wv[tn][i][4 + j];
  for (short j = 0; j < 4; ++j) {
    a[j] = xv[0][4 + j];
    a[4 + j] = xv[1][4 + j];
  }
  op.run(a, b, cb);
}

// the eighth's partial (A + B) into threadgroup memory, at the destination layout's (row, head)
for (short e = 0; e < 16; ++e) {
  const short tn = e >> 3, i = (e >> 2) & 1, j = e & 3;
  part[sg][fm + 8 * i][16 * tn + fn + j] = ca[e] + cb[e];
}
threadgroup_barrier(metal::mem_flags::mem_threadgroup);
// 256 threads, 2 outputs each: thread t the row t / 16, heads 2 (t % 16) and + 1
const uint t = sg * 32 + lane;
const int rr = int(t) >> 4;
const int row = r0 + rr;
if (row < L) {
  const int h0 = 2 * (int(t) & 15);
  metal::vec<OT, 2> o;
  for (short k = 0; k < 2; ++k) {
    const int h = h0 + k;
    o[k] = OT(((part[0][rr][h] + part[1][rr][h]) + (part[2][rr][h] + part[3][rr][h])) + ((part[4][rr][h] + part[5][rr][h]) + (part[6][rr][h] + part[7][rr][h])));
  }
  *(device metal::vec<OT, 2>*)(y + size_t(row) * N + h0) = o;
}
