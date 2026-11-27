// pf_route_scan (MLX_SERVE_DSV41_PF_ROUTE_PACK): the prefill route pack's offsets, prefill_route_scan.metal's words
// (kernels/moe) in one 1024-thread threadgroup: offsets[b * E + e] = (pairs of experts below e) + (pairs of expert e in
// blocks below b), counts [NB, E] the per-256-pair-block histograms. That kernel walks each expert's NB blocks serially
// from one thread twice (2 NB dependent steps, NB = 414 at a 17,638-row pass); here thread (s, q) owns expert quad q
// (a uint4, 128-bit loads) over slice s of the blocks, so a thread walks NB / S blocks: the slices' sums, their
// exclusive scan per expert, the experts' exclusive scan (simdgroup prefix sums), then the slice's offsets. Integer
// sums: the same words in any order.
static_assert(E % 128 == 0 && E <= 1024, "expert quads over whole simdgroups");
constexpr int Q = E / 4;
constexpr int S = 1024 / Q;
constexpr int NG = E / 32;
threadgroup uint4 part[S * Q];
threadgroup uint tot[E];
threadgroup uint sgs[NG];
const uint tid = thread_index_in_threadgroup;
const int q = int(tid) % Q;
const int s = int(tid) / Q;
const int nb = int(counts_shape[0]) / E;
const int per = (nb + S - 1) / S;
const int b0 = s < S ? min(nb, s * per) : nb;
const int b1 = s < S ? min(nb, b0 + per) : nb;
const device uint4* c4 = (const device uint4*)counts;
device uint4* o4 = (device uint4*)offsets;
uint4 acc = uint4(0);
for (int b = b0; b < b1; ++b) acc += c4[b * Q + q];
if (s < S) part[s * Q + q] = acc;
threadgroup_barrier(mem_flags::mem_threadgroup);
if (int(tid) < Q) {
  uint4 run = uint4(0);
  for (int j = 0; j < S; ++j) {
    const uint4 t = part[j * Q + int(tid)];
    part[j * Q + int(tid)] = run;
    run += t;
  }
  for (int c = 0; c < 4; ++c) tot[4 * int(tid) + c] = run[c];
}
threadgroup_barrier(mem_flags::mem_threadgroup);
uint pre = 0;
if (int(tid) < E) {
  const uint t = tot[tid];
  pre = simd_prefix_exclusive_sum(t);
  if (thread_index_in_simdgroup == 31) sgs[tid / 32] = pre + t;
}
threadgroup_barrier(mem_flags::mem_threadgroup);
if (int(tid) < E) {
  for (uint g = 0; g < tid / 32; ++g) pre += sgs[g];
  tot[tid] = pre;
}
threadgroup_barrier(mem_flags::mem_threadgroup);
if (s < S) {
  uint4 run = uint4(tot[4 * q], tot[4 * q + 1], tot[4 * q + 2], tot[4 * q + 3]) + part[s * Q + q];
  if (PLANT != 0 && s == 0 && q == 0) run.x += 1;
  for (int b = b0; b < b1; ++b) {
    o4[b * Q + q] = run;
    run += c4[b * Q + q];
  }
}
