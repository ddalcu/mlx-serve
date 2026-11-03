  constexpr uint TG = 1024;
  constexpr uint NSG = TG / 32;
  const uint t = thread_position_in_threadgroup.x;
  const uint row = threadgroup_position_in_grid.x;
  const uint lane = thread_index_in_simdgroup;
  const uint sg = simdgroup_index_in_threadgroup;
  const size_t base = size_t(row) * V;
  const float inv_t = cfg[0];
  const float top_p = cfg[1];
  const float min_p = cfg[3];
  const uint cap = (kcap[0] == 0u || kcap[0] > C) ? C : kcap[0];
  const ulong seed = ulong(seeds[2 * row]) | (ulong(seeds[2 * row + 1]) << 32);
  const uint position = positions[row];

  threadgroup float fsh[NSG];
  threadgroup float fsh2[NSG];
  threadgroup uint ush[NSG];
  threadgroup atomic_uint hist[256];
  threadgroup uint st[4];
  threadgroup uint ck[C];
  threadgroup uint ci[C];
  threadgroup atomic_uint fill_hi;
  threadgroup atomic_uint fill_tie;

  // row max (fixed order: each thread's stride, simd reduction, simdgroups in order)
  float lm = -INFINITY;
  for (uint i = t; i < V; i += TG) lm = max(lm, float(L[base + i]) * inv_t);
  lm = simd_max(lm);
  if (lane == 0) fsh[sg] = lm;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float m = -INFINITY;
  for (uint s = 0; s < NSG; s++) m = max(m, fsh[s]);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  // normalizer, and the count and mass of the tokens within NEAR, NEAR / 2 and NEAR / 4 of the max
  float ls = 0.0f, lnear[3] = {0.0f, 0.0f, 0.0f};
  uint near_count[3] = {0u, 0u, 0u};
  for (uint i = t; i < V; i += TG) {
    const float v = float(L[base + i]) * inv_t;
    const float e = metal::exp(v - m);
    ls += e;
    for (int w = 0; w < 3; w++) {
      if (v >= m - cfg[2] / float(1 << w)) { lnear[w] += e; near_count[w]++; }
    }
  }
  ls = simd_sum(ls);
  if (lane == 0) fsh[sg] = ls;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float z = 0.0f;
  for (uint s = 0; s < NSG; s++) z += fsh[s];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  // the widest window whose tokens are at most C and hold what the rule reads (top_p of the mass with a
  // margin, or the top_k tokens): those tokens are a top segment of the (value desc, id asc) order
  int window = -1;
  uint offset = 0u, n_near = 0u;
  for (int w = 0; w < 3; w++) {
    const float znear_part = simd_sum(lnear[w]);
    const uint before_in_simd = simd_prefix_exclusive_sum(near_count[w]);
    const uint simd_count = simd_sum(near_count[w]);
    if (lane == 0) { fsh2[sg] = znear_part; ush[sg] = simd_count; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float znear = 0.0f;
    uint off = before_in_simd, count = 0u;
    for (uint s = 0; s < NSG; s++) {
      znear += fsh2[s];
      if (s < sg) off += ush[s];
      count += ush[s];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const bool holds = count <= C && (kcap[0] == 0u
        ? (top_p > 0.0f && top_p < 1.0f && znear >= (top_p + 1e-4f) * z)
        : count >= min(kcap[0], uint(C)));
    if (window < 0 && holds) { window = w; offset = off; n_near = count; }
  }
  const float floor_v = window < 0 ? INFINITY : m - cfg[2] / float(1 << window);

  // When a window holds the rule's candidates they are gathered at offsets from the prefix sum above (no
  // atomics), then sorted; otherwise the radix path finds the C largest. Both give the same candidates for
  // the rule, so the same token.
  const bool fast = window >= 0;
  uint n_cand = 0u;
  uint sort_n = C;
  ck[t] = 0u;
  ci[t] = 0xFFFFFFFFu;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (fast) {
    uint at = offset;
    for (uint i = t; i < V; i += TG) {
      const float v = float(L[base + i]) * inv_t;
      if (v >= floor_v) { ck[at] = tf_key(v); ci[at] = i; at++; }
    }
    n_cand = n_near;
    sort_n = 32u;
    while (sort_n < n_near) sort_n <<= 1;
  } else {
    // key of the C-th largest value: radix select over the key's bytes, most significant first
    uint prefix = 0u, pmask = 0u, need = min(uint(C), uint(V)), above = 0u, ties = 0u;
    for (int shift = 24; shift >= 0; shift -= 8) {
      if (t < 256) atomic_store_explicit(&hist[t], 0u, memory_order_relaxed);
      threadgroup_barrier(mem_flags::mem_threadgroup);
      for (uint i = t; i < V; i += TG) {
        const uint k = tf_key(float(L[base + i]) * inv_t);
        if ((k & pmask) == prefix) atomic_fetch_add_explicit(&hist[(k >> shift) & 255u], 1u, memory_order_relaxed);
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      if (t == 0) {
        uint cum = 0u;
        int b = 255;
        for (; b > 0; b--) {
          const uint c = atomic_load_explicit(&hist[b], memory_order_relaxed);
          if (cum + c >= need) break;
          cum += c;
        }
        st[0] = uint(b); st[1] = cum; st[2] = atomic_load_explicit(&hist[b], memory_order_relaxed);
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      prefix |= st[0] << shift; pmask |= 255u << shift;
      above += st[1]; need -= st[1]; ties = st[2];
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    // more ties at that value than places left: the lowest ids (a second radix select, over ids)
    uint idcut = 0xFFFFFFFFu;
    if (ties > need) {
      uint ipre = 0u, imask = 0u, ineed = need;
      for (int shift = 16; shift >= 0; shift -= 8) {
        if (t < 256) atomic_store_explicit(&hist[t], 0u, memory_order_relaxed);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint i = t; i < V; i += TG) {
          if (tf_key(float(L[base + i]) * inv_t) == prefix && (i & imask) == ipre)
            atomic_fetch_add_explicit(&hist[(i >> shift) & 255u], 1u, memory_order_relaxed);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (t == 0) {
          uint cum = 0u;
          uint b = 0u;
          for (; b < 255u; b++) {
            const uint c = atomic_load_explicit(&hist[b], memory_order_relaxed);
            if (cum + c >= ineed) break;
            cum += c;
          }
          st[0] = b; st[1] = cum;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        ipre |= st[0] << shift; imask |= 255u << shift; ineed -= st[1];
        threadgroup_barrier(mem_flags::mem_threadgroup);
      }
      idcut = ipre;
    }
    if (t == 0) {
      atomic_store_explicit(&fill_hi, 0u, memory_order_relaxed);
      atomic_store_explicit(&fill_tie, 0u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = t; i < V; i += TG) {
      const uint k = tf_key(float(L[base + i]) * inv_t);
      if (k > prefix) {
        const uint s = atomic_fetch_add_explicit(&fill_hi, 1u, memory_order_relaxed);
        ck[s] = k; ci[s] = i;
      } else if (k == prefix && i <= idcut) {
        const uint s = above + atomic_fetch_add_explicit(&fill_tie, 1u, memory_order_relaxed);
        ck[s] = k; ci[s] = i;
      }
    }
    n_cand = above + need;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // bitonic sort by (value desc, id asc): the order does not depend on how the candidates were gathered
  for (uint k = 2; k <= sort_n; k <<= 1) {
    for (uint j = k >> 1; j > 0; j >>= 1) {
      const uint p = t ^ j;
      if (p > t && p < sort_n) {
        const uint ka = ck[t], kb = ck[p], ia = ci[t], ib = ci[p];
        const bool a_first = ka > kb || (ka == kb && ia < ib);
        if (a_first != ((t & k) == 0)) { ck[t] = kb; ck[p] = ka; ci[t] = ib; ci[p] = ia; }
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
  }

  // the nucleus: softmax over the whole row (top_k 0) or over the top_k, cut where it reaches top_p
  if (t == 0) {
    const uint n = min(n_cand, cap);
    float norm = z;
    if (kcap[0] != 0u) {
      norm = 0.0f;
      for (uint j = 0; j < n; j++) norm += metal::exp(tf_val(ck[j]) - m);
    }
    uint keep = n;
    if (top_p > 0.0f && top_p < 1.0f) {
      float cum = 0.0f;
      for (uint j = 0; j < n; j++) {
        cum += metal::exp(tf_val(ck[j]) - m) / norm;
        if (cum >= top_p) { keep = j + 1; break; }
      }
    }
    if (min_p > 0.0f) {
      const float floor_v = m + metal::log(min_p);
      for (uint j = 0; j < keep; j++) {
        if (tf_val(ck[j]) < floor_v) { keep = j; break; }
      }
    }
    st[0] = keep;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const uint keep = st[0];

  // Gumbel-max over the kept candidates; ties go to the earlier candidate
  float score = -INFINITY;
  uint best = 0xFFFFFFFFu;
  if (t < keep) {
    score = tf_val(ck[t]) - metal::log(-metal::log(tf_uniform(seed, position, ci[t])));
    best = t;
  }
  const float sm = simd_max(score);
  const uint pick = simd_min(score == sm ? best : 0xFFFFFFFFu);
  if (lane == 0) { fsh[sg] = sm; ush[sg] = pick; }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (t == 0) {
    float bs = -INFINITY;
    uint bj = 0xFFFFFFFFu;
    for (uint s = 0; s < NSG; s++) {
      if (fsh[s] > bs || (fsh[s] == bs && ush[s] < bj)) { bs = fsh[s]; bj = ush[s]; }
    }
    TOK[row] = ci[bj];
  }