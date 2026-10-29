// The decode top-k's shared parts: order keys, the radix select over keys held in registers, the ordered compaction,
// and the candidate-block pipeline (blocks by their largest score, then the keys inside them).
constant uint DEC_KEY_NINF = 0x007FFFFFu;
constant uint DEC_KEY_PINF = 0xFF800000u;

// Bigger float, bigger key (-0 below +0, a NaN above the infinity of its sign). 0 pads: no float maps to it.
inline uint dec_key(float f) {
  const uint u = as_type<uint>(f);
  return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}

// The `krem`-th largest key of the threadgroup's registers: `thr`, and in `rem` how many of the keys equal to it the
// selection takes. Four 8-bit passes over two 256-bin histograms; `hist` is followed by two words through which one
// simdgroup hands each pass's digit to the rest.
template <int TG, int N>
inline void dec_radix(const thread uint (&kk)[N], uint krem, threadgroup atomic_uint* hist, uint tid, uint lane, thread uint& thr, thread uint& rem) {
  threadgroup atomic_uint* sel = hist + 512;
  for (uint b = tid; b < 256; b += TG) metal::atomic_store_explicit(hist + b, 0u, metal::memory_order_relaxed);
  metal::threadgroup_barrier(metal::mem_flags::mem_threadgroup);
  uint prefix = 0;
  for (uint pass = 0; pass < 4; ++pass) {
    const uint shift = 24 - 8 * pass;
    const uint hmask = pass == 0 ? 0u : (0xFFFFFFFFu << (shift + 8));
    threadgroup atomic_uint* h = hist + (pass & 1) * 256;
    threadgroup atomic_uint* next = hist + ((pass + 1) & 1) * 256;
    for (uint b = tid; b < 256; b += TG) metal::atomic_store_explicit(next + b, 0u, metal::memory_order_relaxed);
    #pragma clang loop unroll(full)
    for (int e = 0; e < N; ++e)
      if ((kk[e] & hmask) == prefix) metal::atomic_fetch_add_explicit(h + ((kk[e] >> shift) & 255u), 1u, metal::memory_order_relaxed);
    metal::threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    if (tid < 32) {
      // Bins from 255 down, 8 a lane: the digit where the count from the top reaches krem.
      uint c[8], tot = 0;
      #pragma clang loop unroll(full)
      for (uint j = 0; j < 8; ++j) { c[j] = metal::atomic_load_explicit(h + (255u - (lane * 8 + j)), metal::memory_order_relaxed); tot += c[j]; }
      const uint before = metal::simd_prefix_exclusive_sum(tot);
      uint pk = 0;
      if (before < krem && krem <= before + tot) {
        uint acc = before, j = 0;
        for (; j < 7; ++j) { if (acc + c[j] >= krem) break; acc += c[j]; }
        pk = (acc << 8) | (255u - (lane * 8 + j));
      }
      pk = metal::simd_max(pk);
      if (lane == 0) metal::atomic_store_explicit(sel + (pass & 1), pk, metal::memory_order_relaxed);
    }
    metal::threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    const uint pk = metal::atomic_load_explicit(sel + (pass & 1), metal::memory_order_relaxed);
    prefix |= (pk & 255u) << shift;
    krem -= pk >> 8;
  }
  thr = prefix;
  rem = krem;
}

// Elements sit in simdgroup order: element e of a thread is number (simdgroup * N + e) * 32 + lane of the threadgroup's, so
// a simdgroup's loads of one e are contiguous. `bef[e]` is where the selection {key > above} plus the first `rem` keys
// equal to `eq` (counted when `ties`) puts element e, in that order: elements above before it << 16, ties before it;
// `kept` is the selection's size.
template <int TG, int N>
inline void dec_scan(const thread uint (&kk)[N], uint above, uint eq, bool ties, uint rem, threadgroup uint* part, uint sg, uint lane,
                     thread uint (&bef)[N], thread uint& kept) {
  uint run = 0;
  #pragma clang loop unroll(full)
  for (int e = 0; e < N; ++e) {
    const uint v = (uint(kk[e] > above) << 16) | uint(ties && kk[e] == eq);
    bef[e] = run + metal::simd_prefix_exclusive_sum(v);
    run += metal::simd_sum(v);
  }
  if (lane == 0) part[sg] = run;
  metal::threadgroup_barrier(metal::mem_flags::mem_threadgroup);
  const uint pv = lane < TG / 32 ? part[lane] : 0u;
  const uint off = metal::simd_shuffle(metal::simd_prefix_exclusive_sum(pv), sg), all = metal::simd_sum(pv);
  #pragma clang loop unroll(full)
  for (int e = 0; e < N; ++e) bef[e] += off;
  kept = (all >> 16) + metal::min(all & 0xFFFFu, rem);
}

// The largest key among a block's `BS` scores (-inf past the row's `n`).
template <int BS>
inline uint dec_block_key(const device float* sr, int n, int b) {
  uint m = 0;
  if ((b + 1) * BS <= n && BS % 4 == 0) {
    const device packed_float4* v = (const device packed_float4*)(sr + b * BS);
    #pragma clang loop unroll(full)
    for (int j = 0; j < BS / 4; ++j) {
      const float4 x = float4(v[j]);
      m = metal::max(m, metal::max(metal::max(dec_key(x.x), dec_key(x.y)), metal::max(dec_key(x.z), dec_key(x.w))));
    }
    return m;
  }
  for (int j = 0; j < BS; ++j) {
    const int i = b * BS + j;
    m = metal::max(m, i < n ? dec_key(sr[i]) : DEC_KEY_NINF);
  }
  return m;
}

// From block keys `bk` (ids `bid`, ascending; key 0 pads) to the picks: select the KBLK largest blocks (ties to the lower
// id), keep those above -inf as `cs` / `cand` (ascending, -1 padded), then select the K largest of their keys. A pick lies
// in the first K + 1 blocks, so the KBLK blocks (K + 1 at least) hold the row's top-K. `sr` is the row's n scores.
template <int TG, int N, int K, int KBLK, int BS, typename P>
inline void dec_pick_blocks(const thread uint (&bk)[N], const thread int (&bid)[N], uint nblk, P sr, int n, int lens,
                            threadgroup atomic_uint* hist, threadgroup uint* part, threadgroup int* cs, device int* cand, device int* picks,
                            uint tid, uint lane, uint sg) {
  constexpr int E = (KBLK * BS + TG - 1) / TG;
  uint thr, rem, kept;
  dec_radix<TG, N>(bk, metal::min(uint(KBLK), nblk), hist, tid, lane, thr, rem);
  const uint above = metal::max(thr, DEC_KEY_NINF);
  const bool ties = thr > DEC_KEY_NINF;
  uint bef[N];
  dec_scan<TG, N>(bk, above, thr, ties, rem, part, sg, lane, bef, kept);
  #pragma clang loop unroll(full)
  for (int e = 0; e < N; ++e) {
    const bool gt = bk[e] > above, eq = ties && bk[e] == thr;
    const uint g = bef[e] >> 16, q = bef[e] & 0xFFFFu;
    if (gt || (eq && q < rem)) {
      const uint pos = g + (gt ? metal::min(q, rem) : q);
      cs[pos] = bid[e];
      cand[pos] = bid[e];
    }
  }
  for (uint p = kept + tid; p < KBLK; p += TG) { cs[p] = -1; cand[p] = -1; }
  metal::threadgroup_barrier(metal::mem_flags::mem_threadgroup);

  uint kk[E];
  #pragma clang loop unroll(full)
  for (int e = 0; e < E; ++e) {
    const int s = (int(sg) * E + e) * 32 + int(lane);
    uint key = 0;
    if (s < KBLK * BS) {
      const int b = cs[s / BS], i = b * BS + s % BS;
      if (b >= 0 && i < n) key = dec_key(sr[i]);
    }
    kk[e] = key;
  }
  const uint want = metal::min(uint(K), uint(n));
  dec_radix<TG, E>(kk, want, hist, tid, lane, thr, rem);
  uint bef2[E];
  dec_scan<TG, E>(kk, thr, thr, true, rem, part, sg, lane, bef2, kept);
  #pragma clang loop unroll(full)
  for (int e = 0; e < E; ++e) {
    const bool gt = kk[e] > thr, eq = kk[e] == thr;
    const uint g = bef2[e] >> 16, q = bef2[e] & 0xFFFFu;
    if (gt || (eq && q < rem)) {
      const int s = (int(sg) * E + e) * 32 + int(lane), b = cs[s / BS], i = b * BS + s % BS;
      picks[g + (gt ? metal::min(q, rem) : q)] = b >= 0 && i < lens ? i : -1;
    }
  }
  for (uint p = want + tid; p < K; p += TG) picks[p] = -1;
}
