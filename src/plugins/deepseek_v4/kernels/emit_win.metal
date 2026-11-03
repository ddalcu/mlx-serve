int w = thread_position_in_grid.y;
int lane = thread_position_in_threadgroup.x;
if (w >= NWIN) return; // uniform per threadgroup (grid.y == NWIN)
const int CD = (OVERLAP != 0 ? 2 : 1) * D;
const int ROWS = (OVERLAP != 0 ? 2 : 1) * R;
const int HALF = RD / 2;
const int QW = (ROTATE != 0) ? D : (D - RD);
const int NG = QW / GROUP;
const float CODE_MAX = (ROTATE != 0) ? 6.0f : 448.0f;
const float AMAX_FLOOR = (ROTATE != 0) ? metal::ldexp(6.0f, -126) : 1e-4f;
const int MANT_BITS = (ROTATE != 0) ? 1 : 3;
const int MIN_EXP = (ROTATE != 0) ? 0 : -6;
const float MIN_SUB = metal::ldexp(1.0f, MIN_EXP);
threadgroup float row[D];
threadgroup float row2[(ROTATE != 0) ? D : 1];
threadgroup float red[TG];
threadgroup float gsc[(QW / GROUP) > 0 ? (QW / GROUP) : 1];
// phase 1: per-feature masked-softmax combine over the window rows.
// 3 passes (max / exp-sum / weighted acc) re-reading the sources — the
// composed chain's softmax→multiply→sum rounding order, no register
// caching so ratio-128 windows (ROWS 128) don't spill.
for (int j = lane; j < D; j += TG) {
  float mx = -INFINITY;
  float sum = 0.0f;
  float acc = 0.0f;
  float inv = 0.0f;
  for (int pass = 0; pass < 3; ++pass) {
    if (pass == 2) inv = 1.0f / sum;
    for (int i = 0; i < ROWS; ++i) {
      int ext; int col;
      if (OVERLAP != 0) {
        if (i < R) { ext = w * R + i; col = j; }
        else { ext = R + w * R + (i - R); col = D + j; }
      } else { ext = w * R + i; col = j; }
      float kv; float sc;
      if (ext < PRE_N) {
        kv = pre_kv[ext * CD + col];
        sc = pre_sc[ext * CD + col];
      } else {
        int cr = ext - PRE_N;
        kv = cin[cr * CIN_W + COL_OFF + col];
        sc = cin[cr * CIN_W + COL_OFF + CD + col] + ape[((BASE_MOD + cr) % R) * CD + col];
      }
      if (pass == 0) mx = metal::max(mx, sc);
      else if (pass == 1) sum += metal::exp(sc - mx);
      else acc += (metal::exp(sc - mx) * inv) * kv;
    }
  }
  row[j] = acc;
}
threadgroup_barrier(metal::mem_flags::mem_threadgroup);
// phase 2: RMSNorm (tree-reduced sum of squares, weight applied)
float ss = 0.0f;
for (int j = lane; j < D; j += TG) ss += row[j] * row[j];
red[lane] = ss;
threadgroup_barrier(metal::mem_flags::mem_threadgroup);
for (int off = TG / 2; off > 0; off >>= 1) {
  if (lane < off) red[lane] += red[lane + off];
  threadgroup_barrier(metal::mem_flags::mem_threadgroup);
}
float rinv = metal::rsqrt(red[0] / (float)D + consts[0]);
for (int j = lane; j < D; j += TG) row[j] = row[j] * rinv * nw[j];
threadgroup_barrier(metal::mem_flags::mem_threadgroup);
// phase 3: interleaved-pair rope on the tail RD dims at the block start
for (int p = lane; p < HALF; p += TG) {
  float a = row[D - RD + 2 * p];
  float b = row[D - RD + 2 * p + 1];
  float cv = cosr[w * HALF + p];
  float sv = sinr[w * HALF + p];
  row[D - RD + 2 * p] = a * cv - b * sv;
  row[D - RD + 2 * p + 1] = a * sv + b * cv;
}
threadgroup_barrier(metal::mem_flags::mem_threadgroup);
// phase 4 (indexer only): Hadamard rotate = row @ hada
if (ROTATE != 0) {
  for (int j = lane; j < D; j += TG) {
    float hacc = 0.0f;
    for (int kk = 0; kk < D; ++kk) hacc += row[kk] * hada[kk * D + j];
    row2[j] = hacc;
  }
  threadgroup_barrier(metal::mem_flags::mem_threadgroup);
}
// phase 5: ue8m0 per-group scales — 2^ceil(log2(amax/code_max)), exact
for (int g = lane; g < NG; g += TG) {
  float amax = AMAX_FLOOR;
  for (int e = 0; e < GROUP; ++e) {
    float v = (ROTATE != 0) ? row2[g * GROUP + e] : row[g * GROUP + e];
    amax = metal::max(amax, metal::abs(v));
  }
  float q = amax / CODE_MAX;
  int ee = 0;
  float mm = metal::frexp(q, ee);
  int ce = (mm == 0.5f) ? (ee - 1) : ee;
  gsc[g] = metal::ldexp(1.0f, ce);
}
threadgroup_barrier(metal::mem_flags::mem_threadgroup);
// phase 6: quant-dequant (rope tail stays raw on the fp8 path) + write
for (int j = lane; j < D; j += TG) {
  float v = (ROTATE != 0) ? row2[j] : row[j];
  float res = v;
  if (ROTATE != 0 || j < D - RD) {
    float scale = gsc[j / GROUP];
    float y = metal::clamp(v / scale, -CODE_MAX, CODE_MAX);
    float a = metal::abs(y);
    float af = metal::max(a, MIN_SUB);
    int ee2 = 0;
    float mm2 = metal::frexp(af, ee2);
    int e2 = ee2 - 1; // floor(log2(af)): mm2 in [0.5, 1)
    (void)mm2;
    if (e2 < MIN_EXP) e2 = MIN_EXP;
    float quantum = metal::ldexp(1.0f, e2 - MANT_BITS);
    float qv = metal::min(metal::rint(a / quantum) * quantum, CODE_MAX);
    float sq = (y < 0.0f) ? -qv : ((y > 0.0f) ? qv : 0.0f);
    res = sq * scale;
  }
  outw[w * D + j] = res;
}