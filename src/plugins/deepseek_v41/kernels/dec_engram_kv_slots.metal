// dec_engram_kv_slots: one decode row's Engram wkv, kv[1, N] = x . W^T, slot-blocked (`MLX_SERVE_DSV41_ENGRAM_KV_SLOTS`).
// x is the row's K / HD table rows (slots) read by id out of the table's codes, as `dec_engram_kv.metal` reads them,
// staged once a threadgroup in bf16 (e2m1 and e4m3 codes under an e8m0 scale are exact there). A simdgroup owns RT
// output rows; a lane takes 16 weights of each row per 512 inputs (a uint4 of e4m3 and its e8m0 exponent), so half a
// simdgroup covers one slot. Each slot's partial is reduced across its 16 lanes, then the slots are added pairwise in
// one fixed order, so the sum does not depend on where a partial came from. The sums carry their rounding errors
// (`dsv41_two_sum`, exact in the Safe math mode MLX compiles these in): a slot's partial is its lanes' sum rounded
// once, the row the slots' sum rounded once, so it sits closer to the exact sum than a plain f32 reduction.
// With KVC (the contribution cache, decode/engram_kvc.zig) `plan[s]` names slot s's entry in `vals` (f32 [entries, N]):
// bit 31 set, the partial is read from it and the slot's weights are never loaded; clear, it is computed and stored
// there. `vals` is bound as an input and written through a cast: `plan` (the look-up's output) orders this behind it.
static_assert(HD == 256, "half a simdgroup is one table row");
static_assert(K % 512 == 0 && K <= 8192, "a lane's 16 inputs per 512 cover two slots; the stack holds 16 pairs");
constexpr int SG = 8;
constexpr int RT = 4;
constexpr int NI = K / 512;
constexpr int KW = K / 4;
constexpr int KG = K / 32;
const ulong wbase = ulong(geo[0]) | (ulong(geo[1]) << 32);
const ulong sbase = ulong(geo[3]) | (ulong(geo[4]) << 32);
const ulong wstride = ulong(geo[2]);
const ulong sstride = ulong(geo[5]);
const device uchar* cb = (const device uchar*)codes;
const device uchar* sb = (const device uchar*)tsc;
threadgroup bfloat xs[K];
const uint lane = thread_index_in_simdgroup;
const uint sg = simdgroup_index_in_threadgroup;
const uint tid = thread_index_in_threadgroup;

// The row x: 16 inputs a task, input k of slot k / HD from table row ids[k / HD], stored where the lane that reads it
// finds it (input j of lane l's 16 in block i at i * 512 + j * 32 + l: no bank conflicts on either side).
for (int q = int(tid); q < K / 16; q += SG * 32) {
  const int k = q * 16;
  const ulong id = ulong(ids[k / HD]);
  const int p = k % HD;
  const float e = mlxserve_fp4_e8m0(uint(sb[sbase + id * sstride + p / 32]));
  const device uchar* rc = cb + wbase + id * wstride;
  float v[16];
  if constexpr (TB == 4) {
    const uint2 cw = *(const device uint2*)(rc + p / 2);
    for (int i = 0; i < 8; ++i) {
      v[i] = mlxserve_fp4_e2m1_value(cw.x >> (4 * i)) * e;
      v[8 + i] = mlxserve_fp4_e2m1_value(cw.y >> (4 * i)) * e;
    }
  } else {
    const uint4 cw = *(const device uint4*)(rc + p);
    const uint words[4] = {cw.x, cw.y, cw.z, cw.w};
    for (int i = 0; i < 4; ++i) {
      const float2 lo = mlxserve_e4m3x2(words[i]);
      const float2 hi = mlxserve_e4m3x2(words[i] >> 8);
      v[4 * i + 0] = (lo.x * 256.0f) * e;
      v[4 * i + 1] = (hi.x * 256.0f) * e;
      v[4 * i + 2] = (lo.y * 256.0f) * e;
      v[4 * i + 3] = (hi.y * 256.0f) * e;
    }
  }
  threadgroup bfloat* dst = xs + (q / 32) * 512 + (q % 32);
  for (int i = 0; i < 16; ++i) dst[i * 32] = bfloat(v[i]);
}
threadgroup_barrier(mem_flags::mem_threadgroup);

const int n0 = int(threadgroup_position_in_grid.x) * (SG * RT) + int(sg) * RT;
const device uint4* wr = (const device uint4*)w + (size_t)n0 * (KW / 4);
const device uchar* sr = wsc + (size_t)n0 * KG;
// sh[b] + sl[b]: the sum of the last 2^b slot pairs not yet added into a wider level, and its carried error.
float sh[5][RT];
float sl[5][RT];
#pragma unroll
for (int i = 0; i < NI; ++i) {
  const int u = i * 32 + int(lane);
#if KVC
  const uint pl = plan[2 * i + int(lane / 16)];
  const bool hit = (pl >> 31) != 0;
  device float* vr = (device float*)vals + (size_t)(pl & 0x7fffffffu) * N + n0;
#else
  const bool hit = false;
#endif
  float part[RT];
  float perr[RT];
  for (int r = 0; r < RT; ++r) part[r] = perr[r] = 0.0f;
  if (!hit) {
    uint4 qw[RT];
    uchar se[RT];
    #pragma unroll
    for (int r = 0; r < RT; ++r) {
      qw[r] = wr[(size_t)r * (KW / 4) + u];
      se[r] = sr[(size_t)r * KG + u / 2];
    }
    float xv[16];
    #pragma unroll
    for (int j = 0; j < 16; ++j) xv[j] = float(xs[i * 512 + j * 32 + int(lane)]);
    #pragma unroll
    for (int r = 0; r < RT; ++r) {
      const uint words[4] = {qw[r].x, qw[r].y, qw[r].z, qw[r].w};
      float dj[4];
      #pragma unroll
      for (int j = 0; j < 4; ++j) {
        const float2 lo = mlxserve_e4m3x2(words[j]);
        const float2 hi = mlxserve_e4m3x2(words[j] >> 8);
        dj[j] = (xv[4 * j + 0] * lo.x + xv[4 * j + 1] * hi.x) + (xv[4 * j + 2] * lo.y + xv[4 * j + 3] * hi.y);
      }
      // The scale is a power of two: it moves the sum and its error exactly.
      const float sc = 256.0f * mlxserve_fp4_e8m0(uint(se[r]));
      const float2 a = dsv41_two_sum(dj[0], dj[1]);
      const float2 b = dsv41_two_sum(dj[2], dj[3]);
      const float2 t = dsv41_two_sum(a.x, b.x);
      part[r] = t.x * sc;
      perr[r] = ((a.y + b.y) + t.y) * sc;
    }
  }
  // Slot 2i in lanes 0..15, slot 2i + 1 in lanes 16..31: each half's sum (all lanes, uniform), the pair's sum, then
  // the pair onto the stack (i's trailing one bits: the levels it closes).
  #pragma unroll
  for (int r = 0; r < RT; ++r) {
    // A butterfly over the half's 16 lanes: every lane ends with the same sum and error (the exact error of a
    // rounded sum does not depend on the order of its two terms).
    float p = part[r];
    float q = perr[r];
    for (int m = 1; m < 16; m <<= 1) {
      const float2 t = dsv41_two_sum(p, simd_shuffle_xor(p, m));
      q = (q + simd_shuffle_xor(q, m)) + t.y;
      p = t.x;
    }
    p = p + q;
#if KVC
    if (hit) {
      p = vr[r];
    } else if ((lane & 15) == 0) {
      vr[r] = p;
    }
#endif
    float2 c = dsv41_two_sum(simd_shuffle(p, 0), simd_shuffle(p, 16));
    int b = 0;
    for (; ((i >> b) & 1) != 0; ++b) {
      const float2 t = dsv41_two_sum(sh[b][r], c.x);
      c = float2(t.x, (sl[b][r] + c.y) + t.y);
    }
    sh[b][r] = c.x;
    sl[b][r] = c.y;
  }
}
// The levels NI leaves set, widest (earliest slots) first, then the carried error once.
float acc[RT];
#pragma unroll
for (int r = 0; r < RT; ++r) {
  float h = 0.0f;
  float l = 0.0f;
  bool first = true;
  for (int b = 4; b >= 0; --b)
    if (((NI >> b) & 1) != 0) {
      if (first) {
        h = sh[b][r];
        l = sl[b][r];
      } else {
        const float2 t = dsv41_two_sum(h, sh[b][r]);
        h = t.x;
        l = (l + sl[b][r]) + t.y;
      }
      first = false;
    }
  acc[r] = h + l;
}
if (lane < RT) {
  float v = acc[0];
  for (int r = 1; r < RT; ++r) v = int(lane) == r ? acc[r] : v;
  kv[n0 + int(lane)] = T(v);
}
