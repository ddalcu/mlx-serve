// dec_wo_a: the split merge, the inverse rope and the grouped mxfp8 wo_a in one dispatch. A threadgroup owns NT
// outputs of one wo_a group and ONE head of its HPG: it merges that head's splits (the sink, the inverse rope) into
// threadgroup memory, contracts its rows' 512 weights against every row's head and publishes the partial sums; the last
// of a tile's HPG threadgroups to arrive adds them in head order. NAX: a simdgroup takes 32 rows on the matrix units
// against the head rounded to bf16 (as the chain hands it to wo_a); else NR rows on the vector units against the f32 head.
const uint lane = thread_index_in_simdgroup, sg = simdgroup_index_in_threadgroup;
const int tid = int(thread_position_in_threadgroup.x);
const int h = int(threadgroup_position_in_grid.y), g = h / HPG, hh = h % HPG;
const int tile = int(threadgroup_position_in_grid.x);
constexpr int NT = NSG * NR, NTHR = NSG * 32, TILES = OL / NT, KW = HPG * D / 4, KG = HPG * D / 32, D4 = D / 4;
// A merged row of the matrix units' tile, padded apart in the banks.
constexpr int PITCH = D + 8;
const uint row0 = uint(g) * OL + uint(tile * NT) + sg * NR;

threadgroup float4 xs[NAX ? R * PITCH / 8 : R * D4];
threadgroup float coef[R * (S + 1)];

if constexpr (NAX) {
  const short qid = short(lane >> 2);
  const short fm = (qid & 4) | short((lane >> 1) & 3);
  const short f4 = (qid & 2) | short(lane & 1);
  dsv41_nax_rows rows;
  for (int j = 0; j < 4; ++j) {
    const uint row = row0 + uint(fm) + 8u * uint(j);
    rows.w[j] = (const device uint4*)w + row * (KW / 4) + hh * (D / 16) + f4;
    rows.s[j] = scales + row * KG + hh * (D / 32) + (f4 >> 1);
  }
  const dsv41_nax_first first = dsv41_nax_head(rows);

  // Each row's split weights and 1 / total, then the head: pairs of channels, a thread R of them.
  if (tid < R) {
    const uint base = (uint(tid) * H + h) * S;
    float mx = sink[h];
    for (int j = 0; j < S; ++j) mx = metal::max(mx, pm[base + j]);
    float total = metal::exp(sink[h] - mx);
    for (int j = 0; j < S; ++j) {
      const float wj = pl[base + j] > 0.0f ? metal::exp(pm[base + j] - mx) : 0.0f;
      total += pl[base + j] * wj;
      coef[tid * (S + 1) + j] = wj;
    }
    coef[tid * (S + 1) + S] = 1.0f / total;
  }
  threadgroup_barrier(metal::mem_flags::mem_threadgroup);
  threadgroup uint* xw = (threadgroup uint*)xs;
  for (int it = tid; it < R * (D / 2); it += NTHR) {
    const int m = it / (D / 2), p = it % (D / 2);
    const uint base = (uint(m) * H + h) * S;
    float2 o = float2(0.0f);
    for (int j = 0; j < S; ++j) o += ((const device float2*)(pacc + (base + j) * D))[p] * coef[m * (S + 1) + j];
    o *= coef[m * (S + 1) + S];
    if (p >= (D - RD) / 2) {
      const float pos = float(meta[0] + m);
      const float t0 = pos * (1.0f / periods[p - (D - RD) / 2]);
      const float c0 = metal::fast::cos(t0), n0 = metal::fast::sin(t0);
      o = float2(o.x * c0 - o.y * n0, o.x * n0 + o.y * c0);
    }
    xw[m * (PITCH / 2) + p] = uint(as_type<ushort>(bfloat(o.x))) | (uint(as_type<ushort>(bfloat(o.y))) << 16);
  }
  threadgroup_barrier(metal::mem_flags::mem_threadgroup);

  const bool live = int(fm) < R;
  float acc[16];
  for (int i = 0; i < 16; ++i) acc[i] = 0.0f;
  dsv41_nax_mxfp8<D / 64, 4, 2, 8>(acc, rows, first, (const threadgroup uint4*)(xw + (live ? int(fm) : 0) * (PITCH / 2)) + f4 * 2, live);
  if (live) {
    const uint col = row0 + uint(f4) * 4u;
    for (int i = 0; i < 4; ++i) {
      metal::atomic_store_explicit((device metal::atomic_float*)parts + (hh * R + int(fm)) * OW + col + i, acc[i], metal::memory_order_relaxed);
      metal::atomic_store_explicit((device metal::atomic_float*)parts + (hh * R + int(fm)) * OW + col + 16 + i, acc[8 + i], metal::memory_order_relaxed);
    }
  }
} else {
  const device uint4* w4 = (const device uint4*)w + row0 * (KW / 4) + hh * (D / 16) + lane;
  const device uint8_t* s0 = scales + row0 * KG + hh * (D / 32) + lane / 2;

  uint4 cq[NR];
  uint ce[NR];
  for (int r = 0; r < NR; ++r) { cq[r] = w4[r * (KW / 4)]; ce[r] = uint(s0[r * KG]); }

  for (int it = tid; it < R * D4; it += NTHR) {
    const int m = it / D4, q = it % D4;
    const uint base = (uint(m) * H + h) * S;
    float mx = sink[h];
    for (int j = 0; j < S; ++j) mx = metal::max(mx, pm[base + j]);
    float total = metal::exp(sink[h] - mx);
    float4 o = float4(0.0f);
    for (int j = 0; j < S; ++j) {
      const float wj = pl[base + j] > 0.0f ? metal::exp(pm[base + j] - mx) : 0.0f;
      total += pl[base + j] * wj;
      o += ((const device float4*)(pacc + (base + j) * D))[q] * wj;
    }
    o /= total;
    if (q >= (D - RD) / 4) {
      const int p = 2 * q - (D - RD) / 2;
      const float pos = float(meta[0] + m);
      const float t0 = pos * (1.0f / periods[p]), t1 = pos * (1.0f / periods[p + 1]);
      const float c0 = metal::fast::cos(t0), n0 = metal::fast::sin(t0), c1 = metal::fast::cos(t1), n1 = metal::fast::sin(t1);
      o = float4(o.x * c0 - o.y * n0, o.x * n0 + o.y * c0, o.z * c1 - o.w * n1, o.z * n1 + o.w * c1);
    }
    xs[m * D4 + (q & 3) * 32 + (q >> 2)] = o;
  }
  threadgroup_barrier(metal::mem_flags::mem_threadgroup);

  // Past three rows the heads go through registers two rows at a time and the weights decode again per pass: four rows
  // at once leave a pipeline M1/M2 cap below this threadgroup.
  constexpr int MB = R <= 3 ? R : 2;
  for (int m0 = 0; m0 < R; m0 += MB) {
    float4 xr[MB][4];
    #pragma clang loop unroll(full)
    for (int mm = 0; mm < MB; ++mm)
      #pragma clang loop unroll(full)
      for (int j = 0; j < 4; ++j) xr[mm][j] = (m0 + mm < R) ? xs[(m0 + mm) * D4 + j * 32 + lane] : float4(0.0f);
    #pragma clang loop unroll(full)
    for (int r = 0; r < NR; ++r) {
      // e8m0 as 2^(e - 127), times the 2^8 the e4m3 bit placement leaves out.
      const float sc = as_type<float>(ce[r] == 0u ? 0x400000u : (ce[r] << 23)) * 256.0f;
      const float4 w0 = dsv41_wo_a_e4m3x4(cq[r].x), w1 = dsv41_wo_a_e4m3x4(cq[r].y), w2 = dsv41_wo_a_e4m3x4(cq[r].z), w3 = dsv41_wo_a_e4m3x4(cq[r].w);
      #pragma clang loop unroll(full)
      for (int mm = 0; mm < MB; ++mm) {
        if (m0 + mm < R) {
          const int m = m0 + mm;
          const float v = metal::simd_sum(sc * (metal::dot(xr[mm][0], w0) + metal::dot(xr[mm][1], w1) + metal::dot(xr[mm][2], w2) + metal::dot(xr[mm][3], w3)));
          if (lane == uint(r * R + m) % 32u) metal::atomic_store_explicit((device metal::atomic_float*)parts + (hh * R + m) * OW + row0 + r, v, metal::memory_order_relaxed);
        }
      }
    }
  }
}

// Partials cross threadgroups through L2: plain loads can hit another core's stale L1.
metal::atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst, metal::thread_scope_device);
threadgroup_barrier(metal::mem_flags::mem_device);
threadgroup bool last;
if (tid == 0) {
  const uint old = metal::atomic_fetch_add_explicit((device metal::atomic_uint*)counter + g * TILES + tile, 1u, metal::memory_order_relaxed);
  last = ((old + 1u) % uint(HPG)) == 0u;
}
threadgroup_barrier(metal::mem_flags::mem_threadgroup);
if (!last) return;
metal::atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst, metal::thread_scope_device);
for (int it = tid; it < R * NT; it += NTHR) {
  const int m = it / NT;
  const uint o = uint(m) * OW + uint(g) * OL + uint(tile * NT + it % NT);
  float s = 0.0f;
  for (int k = 0; k < HPG; ++k) s += metal::atomic_load_explicit((device metal::atomic_float*)parts + k * R * OW + o, metal::memory_order_relaxed);
  out[o] = OT(s);
}
