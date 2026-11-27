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
// phase 5: ue8m0 per-group scales, 2^ceil(log2(amax/code_max)): MLX's e8m0 rounded up, the max keeping a NaN
for (int g = lane; g < NG; g += TG) {
  float amax = AMAX_FLOOR;
  for (int e = 0; e < GROUP; ++e) {
    float v = (ROTATE != 0) ? row2[g * GROUP + e] : row[g * GROUP + e];
    amax = mlxserve_maximum(amax, metal::abs(v));
  }
  gsc[g] = mlxserve_mx_scale_round_up(amax / CODE_MAX);
}
threadgroup_barrier(metal::mem_flags::mem_threadgroup);
// phase 6: quant-dequant (rope tail stays raw on the fp8 path) + write
for (int j = lane; j < D; j += TG) {
  float v = (ROTATE != 0) ? row2[j] : row[j];
  float res = v;
  if (ROTATE != 0 || j < D - RD) {
    float scale = gsc[j / GROUP];
    float y = mlxserve_clip(v / scale, -CODE_MAX, CODE_MAX);
    // MLX's encoder on |y|, the sign the composed chain's sign(y): a zero comes out +0.
    const float a = metal::abs(y);
    const float qv = (ROTATE != 0) ? mlxserve_fp4_e2m1_round(a) : mlxserve_fp8_e4m3_round(a);
    res = (qv * float((y > 0.0f) - (y < 0.0f))) * scale;
  }
  outw[w * D + j] = res;
}