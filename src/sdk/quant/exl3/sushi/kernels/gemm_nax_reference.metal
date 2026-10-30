uint win = uint(threadgroup_position_in_grid.y);
uint sg = uint(simdgroup_index_in_threadgroup);
uint lane = uint(thread_index_in_simdgroup);
constexpr uint TILE = 16u;
constexpr uint N = uint(NHW);
constexpr uint PACKED_HW = N;
constexpr uint PACKED_W = N / 2u;
constexpr uint IT = uint(IDIM) / TILE;
constexpr uint OT = uint(ODIM) / TILE;
const uint start = wstarts[win];
const uint n = wnlive[win];
if (n == 0u || n > uint(WIN)) return;
constexpr auto desc = matmul2d_descriptor(16, 32, 16, false, true, true, matmul2d_descriptor::mode::multiply_accumulate);
matmul2d<desc, execution_simdgroup> op;
auto left = op.get_left_input_cooperative_tensor<half, half, float>();
auto right = op.get_right_input_cooperative_tensor<half, half, float>();
auto destination = op.get_destination_cooperative_tensor<metal::remove_addrspace_t<decltype(left)>, metal::remove_addrspace_t<decltype(right)>, float>();
auto dest_hi = op.get_destination_cooperative_tensor<metal::remove_addrspace_t<decltype(left)>, metal::remove_addrspace_t<decltype(right)>, float>();
const short2 origin = nax_origin(lane);
const uint output_base = uint(threadgroup_position_in_grid.x) * 128u + sg * 32u;
uint row = start;
const uint end = start + n;
while (row < end) {
const uint run0 = row;
const uint eid = uint(eids[row]);
uint run_end = row + 1u;
while (run_end < end && uint(eids[run_end]) == eid) run_end++;
const uint nlive = run_end - row;
const uint n_hi = (nlive > TILE) ? (nlive - TILE) : 0u;
nax_zero(destination);
if (n_hi > 0u) nax_zero(dest_hi);
const device uint *trellis_e = (const device uint *)(trellis + ((size_t)eid * (size_t)IT * (size_t)OT) * PACKED_HW);
for (uint tk = 0u; tk < IT; tk++) {
  const uint kbase = tk * TILE;
  const device uint *words0 = trellis_e + ((size_t)tk * (size_t)OT + output_base / TILE) * PACKED_W;
  const nfrag w0 = nax_wfrag_k<N>(words0, lane);
  const nfrag w1 = nax_wfrag_k<N>(words0 + PACKED_W, lane);
  for (short s = 0; s < 8; s++) {
    right[s] = w0[s];
    right[8 + s] = w1[s];
  }
  nax_left(left, x, run0, uint(IDIM), kbase, origin, nlive);
  op.run(left, right, destination);
  if (n_hi > 0u) {
    nax_left(left, x, run0 + TILE, uint(IDIM), kbase, origin, n_hi);
    op.run(left, right, dest_hi);
  }
}
nax_store(destination, y, run0, uint(ODIM), output_base, origin, nlive);
if (n_hi > 0u) nax_store(dest_hi, y, run0 + TILE, uint(ODIM), output_base, origin, n_hi);
row = run_end;
}