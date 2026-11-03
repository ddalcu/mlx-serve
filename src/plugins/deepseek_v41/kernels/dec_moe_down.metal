// dec_moe_down: the routed and shared down projections of ROWS decode rows, the combine and the hc expand into the next
// stream. A thread owns one output row and one 128-value unit of the reduction axis; a threadgroup RB rows, every
// expert the rows pick (each distinct one read once) and the shared expert.
constexpr int UNITS = K / 128;
constexpr int NT = RB * UNITS;
constexpr int NV = K / 8;
constexpr int KW = K * 3 / 32;
constexpr int KG = K / 64;
constexpr int PS = 148;
constexpr int PAIRS = ROWS * TOPK;
const uint t = thread_position_in_threadgroup.x;
const uint lane = thread_index_in_simdgroup;
const uint rb = t / UNITS, u = t - rb * UNITS;
const uint d0 = threadgroup_position_in_grid.x * RB;

threadgroup float pbuf[2 * UNITS * PS];
threadgroup float xsb[2 * UNITS * 2];
threadgroup uint jx[PAIRS], je[PAIRS], jr[PAIRS], jn[PAIRS];

// The jobs: the row pairs ordered by expert (first pick first), the activation row of each, its expert, its row,
// and whether it is the first of its expert (the weights load once for the rest).
if (ROWS > 1) {
  if (t < 32) {
    const uint p0 = t, p1 = t + 32;
    const bool v0 = p0 < uint(PAIRS), v1 = p1 < uint(PAIRS);
    const uint e0 = v0 ? uint(inds[p0]) : 0xFFFFFFFFu, e1 = v1 ? uint(inds[p1]) : 0xFFFFFFFEu;
    uint m0l = 0u, m0h = 0u, m1l = 0u, m1h = 0u;
    for (uint q = 0; q < uint(PAIRS); ++q) {
      const uint eq = uint(inds[q]);
      const uint bit = 1u << (q & 31u);
      if (q < 32u) { m0l |= eq == e0 ? bit : 0u; m1l |= eq == e1 ? bit : 0u; }
      else { m0h |= eq == e0 ? bit : 0u; m1h |= eq == e1 ? bit : 0u; }
    }
    const uint lead0 = m0l != 0u ? metal::ctz(m0l) : 32u + metal::ctz(m0h);
    const uint lead1 = m1l != 0u ? metal::ctz(m1l) : 32u + metal::ctz(m1h);
    const uint sz0 = metal::popcount(m0l) + metal::popcount(m0h), sz1 = metal::popcount(m1l) + metal::popcount(m1h);
    const uint rk0 = metal::popcount(m0l & ((1u << p0) - 1u));
    const uint rk1 = metal::popcount(m1l) + metal::popcount(m1h & ((1u << (p1 - 32u)) - 1u));
    const uint own0 = v0 && lead0 == p0 ? sz0 : 0u, own1 = v1 && lead1 == p1 ? sz1 : 0u;
    const uint ex0 = metal::simd_prefix_exclusive_sum(own0);
    const uint ex1 = metal::simd_sum(own0) + metal::simd_prefix_exclusive_sum(own1);
    const uint base0 = metal::simd_shuffle(ex0, lead0 & 31u);
    const uint sa = metal::simd_shuffle(ex0, lead1 & 31u), sb = metal::simd_shuffle(ex1, (lead1 - 32u) & 31u);
    const uint base1 = lead1 < 32u ? sa : sb;
    if (v0) { const uint pos = base0 + rk0; jx[pos] = (p0 / TOPK) * (TOPK + 1) + p0 % TOPK; je[pos] = e0; jr[pos] = p0 / TOPK; jn[pos] = rk0 == 0u ? 1u : 0u; }
    if (v1) { const uint pos = base1 + rk1; jx[pos] = (p1 / TOPK) * (TOPK + 1) + p1 % TOPK; je[pos] = e1; jr[pos] = p1 / TOPK; jn[pos] = rk1 == 0u ? 1u : 0u; }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
}

// Stage the activation row XR (K bf16) for the 3-bit dots: 34 terms a 32 values, each its value times 2^-shift of its
// field in its word (the two halves of a field across words 0|1 and 1|2 as two terms), and each 64-value group's sum.
#define STAGE3(XR, BUF) { \
  for (uint c = t; c - t < uint(NV); c += NT) { \
    const bool on = c < uint(NV); \
    uint4 v = uint4(0); \
    if (on) v = ((const device uint4*)(x + size_t(XR) * K))[c]; \
    const uint n0 = c * 8; \
    const uint ub = n0 / 128, qb = (n0 % 128) / 32, i0 = n0 % 32; \
    float xf[8]; \
    { const float2 f0 = dsv41_bf2(v.x), f1 = dsv41_bf2(v.y), f2 = dsv41_bf2(v.z), f3 = dsv41_bf2(v.w); \
      xf[0] = f0.x; xf[1] = f0.y; xf[2] = f1.x; xf[3] = f1.y; xf[4] = f2.x; xf[5] = f2.y; xf[6] = f3.x; xf[7] = f3.y; } \
    threadgroup float* pb = pbuf + (BUF) * UNITS * PS + ub * PS + qb * 36; \
    float part = 0.0f; \
    for (int k = 0; k < 8; ++k) { \
      part += xf[k]; \
      const uint i = i0 + k; \
      const uint sh = (3u * i) & 31u; \
      if (on) { \
        if (sh >= 30u) { \
          const uint term = i == 10u ? 10u : 22u; \
          pb[term] = xf[k] * as_type<float>((127u - sh) << 23); \
          pb[term + 1] = xf[k] * (sh == 30u ? 4.0f : 2.0f); \
        } else { \
          pb[i + uint(i >= 11u) + uint(i >= 22u)] = xf[k] * as_type<float>((127u - sh) << 23); \
        } \
      } \
    } \
    part += metal::simd_shuffle_xor(part, 1u); \
    part += metal::simd_shuffle_xor(part, 2u); \
    part += metal::simd_shuffle_xor(part, 4u); \
    if ((lane & 7u) == 0u && on) xsb[(BUF) * UNITS * 2 + n0 / 64] = part; \
  } \
}

// Stage the activation row XR as floats, for the mxfp8 dots.
#define STAGE8(XR, BUF) { \
  for (uint c = t; c - t < uint(NV); c += NT) { \
    if (c < uint(NV)) { \
      const uint4 v = ((const device uint4*)(x + size_t(XR) * K))[c]; \
      const uint n0 = c * 8; \
      threadgroup float* pb = pbuf + (BUF) * UNITS * PS + (n0 / 128) * PS + (n0 % 128); \
      const float2 f0 = dsv41_bf2(v.x), f1 = dsv41_bf2(v.y), f2 = dsv41_bf2(v.z), f3 = dsv41_bf2(v.w); \
      ((threadgroup float4*)pb)[0] = float4(f0.x, f0.y, f1.x, f1.y); \
      ((threadgroup float4*)pb)[1] = float4(f2.x, f2.y, f3.x, f3.y); \
    } \
  } \
}

float accr[ROWS];
for (int a = 0; a < ROWS; ++a) accr[a] = 0.0f;

STAGE3(ROWS > 1 ? jx[0] : 0u, 0)
threadgroup_barrier(mem_flags::mem_threadgroup);

uint4 wr[3];
uint sc_ = 0u, bi_ = 0u;
for (int f = 0; f < PAIRS; ++f) {
  const int cur = f & 1;
  if (f + 1 < PAIRS) STAGE3(ROWS > 1 ? jx[f + 1] : uint(f + 1), cur ^ 1) else STAGE8(uint(TOPK), cur ^ 1)
  if (ROWS == 1 || jn[f] != 0u) {
    const size_t row = size_t(ROWS > 1 ? je[f] : uint(inds[f])) * N + d0 + rb;
    const device uint4* wp = (const device uint4*)(w + row * KW + 12 * u);
    wr[0] = wp[0]; wr[1] = wp[1]; wr[2] = wp[2];
    sc_ = ((const device uint*)(scales + row * KG))[u];
    bi_ = ((const device uint*)(biases + row * KG))[u];
  }
  const threadgroup float* pu = pbuf + cur * UNITS * PS + u * PS;
  float dq = 0.0f, c = 0.0f;
  for (int q = 0; q < 4; ++q) {
    float p[36];
    for (int h = 0; h < 9; ++h) {
      const float4 v = ((const threadgroup float4*)(pu + q * 36))[h];
      p[4 * h] = v.x; p[4 * h + 1] = v.y; p[4 * h + 2] = v.z; p[4 * h + 3] = v.w;
    }
    dq = dsv41_dot96(p, wr[(3 * q) / 4][(3 * q) % 4], wr[(3 * q + 1) / 4][(3 * q + 1) % 4], wr[(3 * q + 2) / 4][(3 * q + 2) % 4], dq);
    if (q & 1) {
      const float xs = xsb[cur * UNITS * 2 + u * 2 + (q >> 1)];
      const float2 s = dsv41_bf2(sc_), b = dsv41_bf2(bi_);
      c = metal::fma(q == 1 ? s.x : s.y, dq, metal::fma(q == 1 ? b.x : b.y, xs, c));
      dq = 0.0f;
    }
  }
  const uint r = ROWS > 1 ? jr[f] : 0u;
  for (int a = 0; a < ROWS; ++a) accr[a] += r == uint(a) ? c : 0.0f;
  threadgroup_barrier(mem_flags::mem_threadgroup);
}

// The shared expert: its weights load once, then each row's shared activation in turn.
{
  uint4 sr[8];
  uint ss_;
  {
    const size_t row = d0 + rb;
    const device uint4* sp = (const device uint4*)(sw + row * (K / 4) + 32 * u);
    for (int i = 0; i < 8; ++i) sr[i] = sp[i];
    ss_ = ((const device uint*)(sscales + row * (K / 32)))[u];
  }
  for (int rr = 0; rr < ROWS; ++rr) {
    const int cur = (PAIRS + rr) & 1;
    if (rr + 1 < ROWS) STAGE8(uint(rr + 1) * (TOPK + 1) + TOPK, cur ^ 1)
    const threadgroup float* pu = pbuf + cur * UNITS * PS + u * PS;
    float c = 0.0f;
    for (int q = 0; q < 4; ++q) {
      float xf[32];
      for (int h = 0; h < 8; ++h) {
        const float4 v = ((const threadgroup float4*)(pu + q * 32))[h];
        xf[4 * h] = v.x; xf[4 * h + 1] = v.y; xf[4 * h + 2] = v.z; xf[4 * h + 3] = v.w;
      }
      float d = 0.0f;
      for (int k = 0; k < 8; ++k) {
        const uint wk = sr[2 * q + k / 4][k % 4];
        const float2 lo = mlxserve_e4m3x2(wk), hi = mlxserve_e4m3x2(wk >> 8);
        d = metal::fma(xf[4 * k], lo.x, d);
        d = metal::fma(xf[4 * k + 1], hi.x, d);
        d = metal::fma(xf[4 * k + 2], lo.y, d);
        d = metal::fma(xf[4 * k + 3], hi.y, d);
      }
      // e8m0 as 2^(e - 127), times the 2^8 the e4m3 bit placement leaves out.
      const uint e8 = (ss_ >> (8 * q)) & 0xFFu;
      c = metal::fma(as_type<float>(e8 == 0u ? 0x400000u : (e8 << 23)) * 256.0f, d, c);
    }
    for (int a = 0; a < ROWS; ++a) accr[a] += rr == a ? c : 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
}

// Sum a row's units, round y where the composed path does, and expand into the new stream.
threadgroup float* red = pbuf;
for (int a = 0; a < ROWS; ++a) red[(a * RB + rb) * UNITS + u] = accr[a];
threadgroup_barrier(mem_flags::mem_threadgroup);
for (uint i = t; i < uint(ROWS * RB); i += NT) {
  const uint r = i / RB, d = d0 + i % RB;
  float s = 0.0f;
  for (int uu = 0; uu < UNITS; ++uu) s += red[i * UNITS + uu];
  const float y = float(T(s));
  for (int h = 0; h < HC; ++h) {
    float a = 0.0f;
    for (int j = 0; j < HC; ++j) a = metal::fma(comb[(r * HC + j) * HC + h], float(res[(r * HC + j) * N + d]), a);
    out[(r * HC + h) * N + d] = T(metal::fma(post[r * HC + h], y, a));
  }
}
