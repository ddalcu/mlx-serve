// PF_HC_V2 (prompt/hc_v2.zig): one pass over a wide prompt layer's hc stream rows that forms each row's stream, stores
// it where a later stage reads it, collapses it with `cpre` (rounded to T, its squares summed) and takes its 24 mixes
// against `fn`, all from registers: the stream is read once and never read back. The header's PRO picks the row:
// 0 the stream `x` as it is (the attention side), 1 x1 = post * y + comb^T x rounded to T (the FFN side, x1 not
// stored), 2 x1 as in 1, then x2 = post2 * y2 + comb2^T x1 rounded to T and stored (the layer's tail and the next
// layer's attention side). MIXON 0: the tail's x2 alone. COMB 1 (PRO 2): y2 is moe_tail.metal's combine, made here
// from the MoE's sorted down rows `dn` (K a row, unsorted by `inv`) and shared rows `sh`: f32 sums in route order plus
// the shared row, rounded to T; the combined rows never go to memory. Every fma and rounding is hc_expand_norm's, hc_mix_norm's and
// glm5's expand's; the sums run in another order.
//
// The release's shapes are compile-time: HC = 4, MIX = 24, D (5120) a template int, S d-slices a row (a template int).
// A threadgroup is 8 simdgroups x 8 rows = 64 rows of one d-slice (D / S columns), in steps of 32 columns. Lane (mr, g)
// of a simdgroup holds row mr's columns g * 8 .. g * 8 + 7 of every copy: one uint4 (8 bf16) a copy, so a row's 4 lanes
// read 64 contiguous bytes a plane and a step. Those 8 values of copy i are the lane's elements (mr, 2g), (mr, 2g + 1)
// of 4 8 x 8 A fragments (k-tiles i * 4 + t, columns 2t, 2t + 1 of its 8): the mixes take the row values straight from
// registers, no staging. A step of `fn` (f32 [MIX, HC * D], cache-resident) is gathered as it is stored into
// threadgroup memory in the B fragments' lane order (hc_v2.zig `packAt`): 16 k-tiles x 3 mix tiles x 64 floats =
// 12 KB, each thread's 12 element offsets fixed for the launch, read as one float2 a lane a fragment (consecutive
// words: no bank conflict). The next step's stream words and fn words are loaded into
// registers before this step's math (two steps in flight a lane). The f32 8x8 products run on the vector ALUs; they
// are the register blocking (a fn word serves 8 rows, a row word 8 mixes), not extra throughput.
//
// Outputs: `x2` [L, HC, D] (PRO 2), `hraw` [L, D] the collapse rounded to T before its norm (MIXON), `part`
// [S, L, MIX + 2] f32 (MIXON): the slice's 24 dots, its stream squares and its collapse squares; hc_v2_fin.metal
// sums the slices, makes the mixes, the split and the normed collapse.
constexpr int HC = 4;
constexpr int MIX = (2 + HC) * HC;
constexpr int PW = MIX + 2;
constexpr int DC = 32;
constexpr int NSG = 8;
constexpr int ROWS = NSG * 8;
constexpr int KT = HC * DC / 8;
constexpr int FSTEP = KT * 3 * 64;
constexpr int NSTEP = D / DC / S;
#if PRO == 0
constexpr int NP = HC;
#elif PRO == 1
constexpr int NP = HC + 1;
#elif COMB
constexpr int NP = HC + 1;
#else
constexpr int NP = HC + 2;
#endif
static_assert(D % (DC * S) == 0, "hc_v2_mix: D in whole steps a slice");
static_assert(FSTEP % (NSG * 32) == 0, "hc_v2_mix: fn's step in whole floats a thread");

const uint sg = simdgroup_index_in_threadgroup;
const uint lane = thread_index_in_simdgroup;
const uint tid = sg * 32 + lane;
const int L = x_shape[0];
const int slice = int(threadgroup_position_in_grid.y);
const int r0 = int(threadgroup_position_in_grid.x) * ROWS;
const short qid = short(lane / 4);
const short mr = (qid & 4) + short((lane / 2) % 4);
const short mc = (qid & 2) * 2 + short(lane % 2) * 2;
const int row = r0 + int(sg) * 8 + mr;
const bool live = row < L;
const size_t lr = size_t(live ? row : 0);
const int dlane = slice * (D / S) + (mc / 2) * 8;

// bf16 element e of 8 packed in a uint4: exact (a bf16 is an f32's high half).
auto bfv = [](uint4 w, int e) -> float {
  const uint u = w[e >> 1];
  return as_type<float>((e & 1) ? (u & 0xffff0000u) : (u << 16));
};

#if MIXON
float cp[HC];
for (int j = 0; j < HC; ++j) cp[j] = cpre[lr * HC + j];
#endif
#if PRO >= 1
float pa[HC], ca[HC * HC];
for (int i = 0; i < HC; ++i) pa[i] = post[lr * HC + i];
for (int q = 0; q < HC * HC; ++q) ca[q] = comb[lr * HC * HC + q];
#endif
#if COMB
uint iv[K];
#if TOK
// MoeParts.tok: the down rows in pair order, read in place (no unsort read).
for (int j = 0; j < K; ++j) iv[j] = live ? uint(lr * K + j) : 0u;
#else
for (int j = 0; j < K; ++j) iv[j] = live ? inv[lr * K + j] : 0u;
#endif
#endif
#if PRO == 2
float pb[HC], cb[HC * HC];
for (int i = 0; i < HC; ++i) pb[i] = post2[lr * HC + i];
for (int q = 0; q < HC * HC; ++q) cb[q] = comb2[lr * HC * HC + q];
#endif

uint4 raw[NP], nxt[NP];
auto fetch = [&](int st, thread uint4* r) {
  const int d = dlane + st * DC;
  for (int j = 0; j < HC; ++j) r[j] = live ? *(const device uint4*)(x + (lr * HC + j) * D + d) : uint4(0);
#if PRO >= 1
  r[HC] = live ? *(const device uint4*)(y + lr * D + d) : uint4(0);
#endif
#if PRO == 2 && !COMB
  r[HC + 1] = live ? *(const device uint4*)(y2 + lr * D + d) : uint4(0);
#endif
};

#if MIXON
threadgroup float fs[FSTEP];
constexpr int FQ = FSTEP / (NSG * 32);
// Packed float p = tid + q * 256 of a step: B fragment (kt, mt) lane ln's v-th value, fn[mt * 8 + mc + v] at the
// column the fragment's k = mr carries (copy kt / 4, step column (mr / 2) * 8 + 2 (kt % 4) + mr % 2).
uint fo[FQ];
for (int q = 0; q < FQ; ++q) {
  const int p = int(tid) + q * NSG * 32;
  const int v = p & 1, ln = (p >> 1) & 31, mt = (p >> 6) % 3, kt = p / 192;
  const int lq = ln >> 2;
  const int kr = (lq & 4) + ((ln >> 1) & 3), kc = (lq & 2) * 2 + (ln & 1) * 2;
  fo[q] = uint((mt * 8 + kc + v) * (HC * D) + (kt >> 2) * D + (kr >> 1) * 8 + 2 * (kt & 3) + (kr & 1));
}
float fr[FQ];
#define HC2_FFETCH(st)                                                                     \
  for (int q = 0; q < FQ; ++q) fr[q] = fn[fo[q] + uint((slice * NSTEP + (st)) * DC)];
#define HC2_FSTORE()                                                                       \
  for (int q = 0; q < FQ; ++q) fs[int(tid) + q * NSG * 32] = fr[q];
simdgroup_matrix<float, 8, 8> c[3];
for (int t = 0; t < 3; ++t) c[t] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
float ssx = 0.0f, ssh = 0.0f;
HC2_FFETCH(0)
HC2_FSTORE()
threadgroup_barrier(metal::mem_flags::mem_threadgroup);
#endif
fetch(0, raw);

for (int st = 0; st < NSTEP; ++st) {
  const bool more = st + 1 < NSTEP;
  if (more) {
    fetch(st + 1, nxt);
#if MIXON
    HC2_FFETCH(st + 1)
#endif
  }
  const int d = dlane + st * DC;
#if COMB
  uint4 dw[K], sw;
  for (int j = 0; j < K; ++j) dw[j] = live ? *(const device uint4*)(dn + size_t(iv[j]) * D + d) : uint4(0);
  sw = live ? *(const device uint4*)(sh + lr * D + d) : uint4(0);
#endif
  float s[HC][8];
  #pragma clang loop unroll(full)
  for (int e = 0; e < 8; ++e) {
    float xv[HC];
    for (int j = 0; j < HC; ++j) xv[j] = bfv(raw[j], e);
#if PRO == 0
    for (int i = 0; i < HC; ++i) s[i][e] = xv[i];
#else
    const float yv = bfv(raw[HC], e);
    float x1[HC];
    for (int i = 0; i < HC; ++i) {
      float acc = 0.0f;
      for (int j = 0; j < HC; ++j) acc = fma(ca[j * HC + i], xv[j], acc);
      x1[i] = float(T(fma(pa[i], yv, acc)));
    }
#if PRO == 1
    for (int i = 0; i < HC; ++i) s[i][e] = x1[i];
#else
#if COMB
    float ys = 0.0f;
    for (int j = 0; j < K; ++j) ys += bfv(dw[j], e);
    const float y2v = float(T(ys + bfv(sw, e)));
#else
    const float y2v = bfv(raw[HC + 1], e);
#endif
    for (int i = 0; i < HC; ++i) {
      float acc = 0.0f;
      for (int j = 0; j < HC; ++j) acc = fma(cb[j * HC + i], x1[j], acc);
      s[i][e] = float(T(fma(pb[i], y2v, acc)));
    }
#endif
#endif
  }
#if PRO == 2
  if (live) for (int i = 0; i < HC; ++i) {
    device T* o = x2 + (lr * HC + i) * D + d;
    *(device vec<T, 4>*)o = vec<T, 4>(T(s[i][0]), T(s[i][1]), T(s[i][2]), T(s[i][3]));
    *(device vec<T, 4>*)(o + 4) = vec<T, 4>(T(s[i][4]), T(s[i][5]), T(s[i][6]), T(s[i][7]));
  }
#endif
#if MIXON
  // The collapse with the previous mix (hc_norm's statements a column), rounded to T; its squares and the stream's.
  T hv[8];
  #pragma clang loop unroll(full)
  for (int e = 0; e < 8; ++e) {
    float h = 0.0f;
    for (int i = 0; i < HC; ++i) h += cp[i] * s[i][e];
    hv[e] = T(h);
    ssh += float(hv[e]) * float(hv[e]);
    for (int i = 0; i < HC; ++i) ssx += s[i][e] * s[i][e];
  }
  if (live) {
    device T* o = hraw + lr * D + d;
    *(device vec<T, 4>*)o = vec<T, 4>(hv[0], hv[1], hv[2], hv[3]);
    *(device vec<T, 4>*)(o + 4) = vec<T, 4>(hv[4], hv[5], hv[6], hv[7]);
  }
  // The 24 dots: copy i's 8 values are k-tiles i * 4 .. i * 4 + 3, two values each.
  #pragma clang loop unroll(full)
  for (int i = 0; i < HC; ++i) {
    #pragma clang loop unroll(full)
    for (int t = 0; t < 4; ++t) {
      simdgroup_matrix<float, 8, 8> a;
      reinterpret_cast<thread float2&>(a.thread_elements()) = float2(s[i][2 * t], s[i][2 * t + 1]);
      const int kt = i * 4 + t;
      #pragma clang loop unroll(full)
      for (int mt = 0; mt < 3; ++mt) {
        simdgroup_matrix<float, 8, 8> b;
        reinterpret_cast<thread float2&>(b.thread_elements()) = *(const threadgroup float2*)(fs + ((kt * 3 + mt) * 32 + int(lane)) * 2);
        simdgroup_multiply_accumulate(c[mt], a, b, c[mt]);
      }
    }
  }
  threadgroup_barrier(metal::mem_flags::mem_threadgroup);
  if (more) { HC2_FSTORE() }
  threadgroup_barrier(metal::mem_flags::mem_threadgroup);
#endif
  if (more) for (int p = 0; p < NP; ++p) raw[p] = nxt[p];
}

#if MIXON
// C's lane holds row mr's mixes mt * 8 + mc and + 1; a row's squares are summed over its 4 lanes (lane bits 0 and 3).
for (int mt = 0; mt < 3; ++mt) {
  const float2 cv = reinterpret_cast<thread float2&>(c[mt].thread_elements());
  if (live) *(device float2*)(part + (size_t(slice) * L + row) * PW + mt * 8 + mc) = cv;
}
ssx += metal::simd_shuffle_xor(ssx, ushort(1));
ssx += metal::simd_shuffle_xor(ssx, ushort(8));
ssh += metal::simd_shuffle_xor(ssh, ushort(1));
ssh += metal::simd_shuffle_xor(ssh, ushort(8));
if (live && (lane & 9u) == 0u) {
  part[(size_t(slice) * L + row) * PW + MIX] = ssx;
  part[(size_t(slice) * L + row) * PW + MIX + 1] = ssh;
}
#endif
