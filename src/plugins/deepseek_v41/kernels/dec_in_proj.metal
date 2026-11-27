// dec_in_proj: the attention input of R rows of one layer in one launch. A threadgroup of D / 8 threads, a thread 8 positions of K.
// Threadgroups in order: q_kv rows (mxfp8, RT each), weights_proj and compressor rows (bf16, RT each), one hc mix each.
// h = rms(collapse(x, cpre), attn_norm) is made in every threadgroup from x (FUSE), else `hin` holds it. MIXR
// (MLX_SERVE_DSV41_IN_PROJ_MIXR): the mixes first, a threadgroup a (mix, row), each row's sums as one mix
// threadgroup sums them.
static_assert(RT == 4, "a threadgroup's rows are the lanes of one float4 of dots");
constexpr int HC = 4, NMIX = (2 + HC) * HC, NT = D / 8, NSG = NT / 32;
constexpr int NGQ = NQ / RT, NGW = NW / RT, NGC = NC / RT;
constexpr int NMT = MIXR ? NMIX * R : 0;
const uint t = thread_position_in_threadgroup.x, lane = thread_index_in_simdgroup, sg = simdgroup_index_in_threadgroup;
const int tg0 = threadgroup_position_in_grid.x;
const int tg = tg0 - NMT;
threadgroup float part[(RT + 2) * R * NSG];
threadgroup float* const pa = part;
threadgroup float* const ps = part + RT * R * NSG;

if (MIXR ? tg0 < NMT : tg >= NGQ + NGW + NGC) {
  constexpr int NR = MIXR ? 1 : R;
  const int m = MIXR ? tg0 / R : tg - (NGQ + NGW + NGC);
  const int r0 = MIXR ? tg0 % R : 0;
  const device float* fr = fn_w + size_t(m) * (HC * D);
  float4 f[HC][2];
  #pragma unroll
  for (int j = 0; j < HC; ++j) {
    f[j][0] = *(const device float4*)(fr + j * D + 8 * t);
    f[j][1] = *(const device float4*)(fr + j * D + 8 * t + 4);
  }
  float dd[NR], sq[NR];
  #pragma unroll
  for (int rr = 0; rr < NR; ++rr) {
    const int r = r0 + rr;
    float a = 0.0f, b = 0.0f;
    #pragma unroll
    for (int j = 0; j < HC; ++j) {
      const uint4 xv = *(const device uint4*)(x + (size_t(r) * HC + j) * D + 8 * t);
      const uint xw[4] = {xv.x, xv.y, xv.z, xv.w};
      #pragma unroll
      for (int q = 0; q < 4; ++q) {
        const float2 u = dsv41_bf2(xw[q]);
        const float4 fa = f[j][q / 2];
        a += u.x * ((q & 1) ? fa.z : fa.x);
        b += u.x * u.x;
        a += u.y * ((q & 1) ? fa.w : fa.y);
        b += u.y * u.y;
      }
    }
    dd[rr] = simd_sum(a);
    sq[rr] = simd_sum(b);
  }
  if (lane == 0) {
    #pragma unroll
    for (int rr = 0; rr < NR; ++rr) {
      ps[rr * NSG + sg] = dd[rr];
      ps[(NR + rr) * NSG + sg] = sq[rr];
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sg == 0) {
    #pragma unroll
    for (int rr = 0; rr < NR; ++rr) {
      const float a = simd_sum(lane < NSG ? ps[rr * NSG + lane] : 0.0f);
      const float b = simd_sum(lane < NSG ? ps[(NR + rr) * NSG + lane] : 0.0f);
      if (lane == 0) z[(r0 + rr) * NMIX + m] = a * metal::precise::rsqrt(b / float(HC * D) + eps[0]) * scale[m] + base[m];
    }
  }
  return;
}

const bool is_q = tg < NGQ, is_w = !is_q && tg < NGQ + NGW;
const int row0 = (is_q ? tg : (is_w ? tg - NGQ : tg - NGQ - NGW)) * RT;
uint2 wq8[RT];
uint sc8[RT];
uint4 wb[RT];
if (is_q) {
  #pragma unroll
  for (int i = 0; i < RT; ++i) {
    wq8[i] = *(const device uint2*)(wq + size_t(row0 + i) * (D / 4) + 2 * t);
    sc8[i] = uint(wsc[size_t(row0 + i) * (D / 32) + t / 4]);
  }
} else {
  const device bfloat16_t* wr = is_w ? wp : cw;
  #pragma unroll
  for (int i = 0; i < RT; ++i) wb[i] = *(const device uint4*)(wr + size_t(row0 + i) * D + 8 * t);
}

float hf[R][8];
if (FUSE) {
  float hb[R][8], ss[R];
  #pragma unroll
  for (int r = 0; r < R; ++r) {
    float v[HC][8];
    #pragma unroll
    for (int j = 0; j < HC; ++j) {
      const uint4 xv = *(const device uint4*)(x + (size_t(r) * HC + j) * D + 8 * t);
      const uint xw[4] = {xv.x, xv.y, xv.z, xv.w};
      #pragma unroll
      for (int q = 0; q < 4; ++q) {
        const float2 u = dsv41_bf2(xw[q]);
        v[j][2 * q] = u.x;
        v[j][2 * q + 1] = u.y;
      }
    }
    float cp[HC], s = 0.0f;
    #pragma unroll
    for (int j = 0; j < HC; ++j) cp[j] = cpre[r * HC + j];
    #pragma unroll
    for (int k = 0; k < 8; ++k) {
      float h8 = 0.0f;
      #pragma unroll
      for (int j = 0; j < HC; ++j) h8 += cp[j] * v[j][k];
      hb[r][k] = float(bfloat16_t(h8));
      s += hb[r][k] * hb[r][k];
    }
    ss[r] = simd_sum(s);
  }
  const uint4 nv = *(const device uint4*)(attn_w + 8 * t);
  if (lane == 0) {
    #pragma unroll
    for (int r = 0; r < R; ++r) ps[r * NSG + sg] = ss[r];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const uint nw[4] = {nv.x, nv.y, nv.z, nv.w};
  #pragma unroll
  for (int r = 0; r < R; ++r) {
    float tot = 0.0f;
    for (int g = 0; g < NSG; ++g) tot += ps[r * NSG + g];
    const float inv = metal::precise::rsqrt(tot / float(D) + eps[0]);
    #pragma unroll
    for (int q = 0; q < 4; ++q) {
      const float2 w2 = dsv41_bf2(nw[q]);
      hf[r][2 * q] = float(bfloat16_t(bfloat16_t(w2.x) * bfloat16_t(hb[r][2 * q] * inv)));
      hf[r][2 * q + 1] = float(bfloat16_t(bfloat16_t(w2.y) * bfloat16_t(hb[r][2 * q + 1] * inv)));
    }
  }
  if (tg == 0) {
    #pragma unroll
    for (int r = 0; r < R; ++r) {
      uint o[4];
      #pragma unroll
      for (int q = 0; q < 4; ++q) o[q] = (as_type<uint>(hf[r][2 * q]) >> 16) | (as_type<uint>(hf[r][2 * q + 1]) & 0xFFFF0000u);
      *(device uint4*)(h + size_t(r) * D + 8 * t) = uint4(o[0], o[1], o[2], o[3]);
    }
  }
} else {
  #pragma unroll
  for (int r = 0; r < R; ++r) {
    const uint4 hv = *(const device uint4*)(hin + size_t(r) * D + 8 * t);
    const uint hw[4] = {hv.x, hv.y, hv.z, hv.w};
    #pragma unroll
    for (int q = 0; q < 4; ++q) {
      const float2 u = dsv41_bf2(hw[q]);
      hf[r][2 * q] = u.x;
      hf[r][2 * q + 1] = u.y;
    }
  }
}

// This thread's dot of each row against its 8 positions; a simdgroup's sums go to threadgroup memory a float4 at a time.
float dv[RT][R];
if (is_q) {
  #pragma unroll
  for (int i = 0; i < RT; ++i) {
    // The e8m0 scale and the 2^8 the e4m3 reads leave out (fp8_read_header.metal).
    const float2 s = dsv41_e8m0<FOLD>(sc8[i]);
    const float2 a0 = dsv41_e4m3x2<FOLD>(wq8[i].x), a1 = dsv41_e4m3x2<FOLD>(wq8[i].x >> 8);
    const float2 a2 = dsv41_e4m3x2<FOLD>(wq8[i].y), a3 = dsv41_e4m3x2<FOLD>(wq8[i].y >> 8);
    const float w8[8] = {a0.x, a1.x, a0.y, a1.y, a2.x, a3.x, a2.y, a3.y};
    #pragma unroll
    for (int r = 0; r < R; ++r) {
      float d = 0.0f;
      #pragma unroll
      for (int k = 0; k < 8; ++k) d = metal::fma(hf[r][k], w8[k], d);
      dv[i][r] = (d * s.y) * s.x;
    }
  }
} else {
  #pragma unroll
  for (int i = 0; i < RT; ++i) {
    const uint bw[4] = {wb[i].x, wb[i].y, wb[i].z, wb[i].w};
    #pragma unroll
    for (int r = 0; r < R; ++r) {
      float d = 0.0f;
      #pragma unroll
      for (int q = 0; q < 4; ++q) {
        const float2 w2 = dsv41_bf2(bw[q]);
        d = metal::fma(hf[r][2 * q], w2.x, d);
        d = metal::fma(hf[r][2 * q + 1], w2.y, d);
      }
      dv[i][r] = d;
    }
  }
}
#pragma unroll
for (int r = 0; r < R; ++r) {
  const float4 sums = simd_sum(float4(dv[0][r], dv[1][r], dv[2][r], dv[3][r]));
  if (lane == 0) {
    #pragma unroll
    for (int i = 0; i < RT; ++i) pa[(i * R + r) * NSG + sg] = sums[i];
  }
}
threadgroup_barrier(mem_flags::mem_threadgroup);
if (t < RT * R) {
  const int i = t / R, r = t % R, n = row0 + i;
  float a = 0.0f;
  for (int g = 0; g < NSG; ++g) a += pa[(i * R + r) * NSG + g];
  if (is_q) {
    if (n < QA) qa[r * QA + n] = bfloat16_t(a);
    else kv[r * (NQ - QA) + (n - QA)] = bfloat16_t(a);
  } else if (is_w) {
    iw[r * NW + n] = bfloat16_t(a);
  } else {
    comp[r * NC + n] = CT(a);
  }
}
