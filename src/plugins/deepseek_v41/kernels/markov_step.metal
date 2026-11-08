// Step K of the DSpark draft's Markov chain (spec/markov.zig): row K of the head's logits plus the bigram bias of the token
// before it, and that row's argmax as the threadgroups' bests. NT threadgroups of 256 threads; T the activation dtype
// (markov_head's), E markov_embed's.
//   lg   f32 [B, V]: the draft head's logits
//   emb  E [V, R], mh T [V, R]: markov_embed and markov_head
//   t1   int32 [1]: the token before step 0
//   pin  uint32 [K NT 2]: steps 0..K-1's (value bits, index) pair a threadgroup (a placeholder at K = 0)
//   pout uint32 [(K + 1) NT 2]: pin, then this step's pairs
//   row  f32 [V] with KEEP: the biased row (else unwritten)
const uint lane = thread_index_in_simdgroup, sg = simdgroup_index_in_threadgroup;
const uint tid = thread_index_in_threadgroup, tg = threadgroup_position_in_grid.x;
constexpr uint SGN = 8u;
constexpr uint NTC = uint(NT), KC = uint(K), VC = uint(V), RC = uint(R);
constexpr uint Q = VC / 4u;
constexpr uint NS = NTC * SGN;
constexpr uint NI = RC / 128u;
threadgroup float t_v[SGN];
threadgroup uint t_i[SGN];
threadgroup uint t_tok;
// The earlier steps' pairs ride along, so the last kernel reads every step's from one array.
if (tid < 2u * KC) pout[2u * ((tid >> 1) * NTC + tg) + (tid & 1u)] = pin[2u * ((tid >> 1) * NTC + tg) + (tid & 1u)];
// The token before this step: t1, or MLX's argmax of step K - 1's row from its threadgroups' pairs (every threadgroup
// reduces them alike).
uint prev;
if (KC == 0u) {
  prev = uint(t1[0]);
} else {
  prev = dsv41_mk_tg_best<SGN * 32u>(pin + 2u * (KC - 1u) * NTC, NTC, tid, lane, sg, t_v, t_i, &t_tok);
}
// The GEMV's vector, as MLX reads it: markov_embed's row cast to the activation dtype, read as float.
const device E* er = emb + ulong(prev) * RC;
float vc[NI][4];
for (uint it = 0; it < NI; ++it)
  for (uint tn = 0; tn < 4u; ++tn) vc[it][tn] = static_cast<float>(static_cast<T>(er[it * 128u + lane * 4u + tn]));
// The rows, four a simdgroup at a time (quads gs, gs + NS, ...; with ZZ an odd step takes them from the far end, the rows
// the step before read last first). Each row is summed as MLX's gemv_<T>_bm8_bn1_sm1_sn32_tm4_tn4 sums it: lane L takes
// elements 128 it + 4 L + tn, it then tn ascending, from 0, the shuffle-down tree folds the lanes into lane 0, and the
// sum is rounded to T; then MLX's f32 add of the logits and its argmax.
const uint gs = tg * SGN + sg;
float bv = -metal::numeric_limits<float>::infinity();
uint bi = DSV41_MK_NONE;
for (uint q = gs; q < Q; q += NS) {
  const uint r0 = 4u * ((ZZ != 0 && (KC & 1u) != 0u) ? (Q - 1u - q) : q);
  const device T* mr = mh + ulong(r0) * RC;
  float result[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  for (uint it = 0; it < NI; ++it) {
    for (uint tm = 0; tm < 4u; ++tm) {
      T inter[4];
      for (uint tn = 0; tn < 4u; ++tn) inter[tn] = mr[tm * RC + it * 128u + lane * 4u + tn];
      for (uint tn = 0; tn < 4u; ++tn) result[tm] += inter[tn] * vc[it][tn];
    }
  }
  for (uint tm = 0; tm < 4u; ++tm)
    for (ushort sn = 16; sn >= 1; sn >>= 1) result[tm] += metal::simd_shuffle_down(result[tm], sn);
  if (lane == 0) {
    for (uint tm = 0; tm < 4u; ++tm) {
      const float v = lg[KC * VC + r0 + tm] + static_cast<float>(static_cast<T>(result[tm]));
      if (KEEP != 0) row[r0 + tm] = v;
      if (dsv41_mk_better(v, r0 + tm, bv, bi)) {
        bv = v;
        bi = r0 + tm;
      }
    }
  }
}
// The threadgroup's best pair.
if (lane == 0) {
  t_v[sg] = bv;
  t_i[sg] = bi;
}
threadgroup_barrier(metal::mem_flags::mem_threadgroup);
if (sg == 0) {
  bv = lane < SGN ? t_v[lane] : -metal::numeric_limits<float>::infinity();
  bi = lane < SGN ? t_i[lane] : DSV41_MK_NONE;
  dsv41_mk_simd_best(bv, bi);
  if (lane == 0) {
    pout[2u * (KC * NTC + tg)] = as_type<uint>(bv);
    pout[2u * (KC * NTC + tg) + 1u] = bi;
  }
}
