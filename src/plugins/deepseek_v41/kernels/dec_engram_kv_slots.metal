// dec_engram_kv_slots: one decode row's Engram wkv, kv[1, N] = x . W^T, slot-blocked (`MLX_SERVE_DSV41_ENGRAM_KV_SLOTS`).
// x is the row's K / HD table rows (slots) read by id out of the table's codes, as `dec_engram_kv.metal` reads them,
// staged once a threadgroup in bf16 (e2m1 and e4m3 codes under an e8m0 scale are exact there). A simdgroup owns RT
// output rows; a lane takes 16 weights of each row per 512 inputs (a uint4 of e4m3 and its e8m0 exponent), so half a
// simdgroup covers one slot. Each slot's partial is reduced across its 16 lanes, then the slots are added in slot
// order, so the sum does not depend on where a partial came from.
// With KVC (the contribution cache, decode/engram_kvc.zig) `plan[s]` names slot s's entry in `vals` (f32 [entries, N]):
// bit 31 set, the partial is read from it and the slot's weights are never loaded; clear, it is computed and stored
// there. `vals` is bound as an input and written through a cast: `plan` (the look-up's output) orders this behind it.
static_assert(HD == 256, "half a simdgroup is one table row");
static_assert(K % 512 == 0, "a lane's 16 inputs per 512 cover two slots");
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
float acc[RT];
for (int r = 0; r < RT; ++r) acc[r] = 0.0f;
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
  for (int r = 0; r < RT; ++r) part[r] = 0.0f;
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
      float d = 0.0f;
      #pragma unroll
      for (int j = 0; j < 4; ++j) {
        const float2 lo = mlxserve_e4m3x2(words[j]);
        const float2 hi = mlxserve_e4m3x2(words[j] >> 8);
        d += xv[4 * j + 0] * lo.x;
        d += xv[4 * j + 1] * hi.x;
        d += xv[4 * j + 2] * lo.y;
        d += xv[4 * j + 3] * hi.y;
      }
      part[r] = (d * 256.0f) * mlxserve_fp4_e8m0(uint(se[r]));
    }
  }
  // Slot 2i in lanes 0..15, slot 2i + 1 in lanes 16..31: each half's sum (all lanes, uniform), then both slots into
  // the total in slot order.
  #pragma unroll
  for (int r = 0; r < RT; ++r) {
    float p = part[r];
    p += simd_shuffle_xor(p, 1);
    p += simd_shuffle_xor(p, 2);
    p += simd_shuffle_xor(p, 4);
    p += simd_shuffle_xor(p, 8);
#if KVC
    if (hit) {
      p = vr[r];
    } else if ((lane & 15) == 0) {
      vr[r] = p;
    }
#endif
    acc[r] += simd_shuffle(p, 0);
    acc[r] += simd_shuffle(p, 16);
  }
}
if (lane < RT) {
  float v = acc[0];
  for (int r = 1; r < RT; ++r) v = int(lane) == r ? acc[r] : v;
  kv[n0 + int(lane)] = T(v);
}
