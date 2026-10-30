int h = thread_position_in_grid.y;
int lane = thread_position_in_threadgroup.x;
if (h >= H) return; // uniform per threadgroup (grid.y == H)
const int HALF = RD / 2;
const int QW = (POST == 1) ? (D - RD) : ((POST == 2) ? D : 0);
const int GROUP = (POST == 2) ? 32 : 64;
const float CODE_MAX = (POST == 2) ? 6.0f : 448.0f;
const float AMAX_FLOOR = (POST == 2) ? metal::ldexp(6.0f, -126) : 1e-4f;
const int MANT_BITS = (POST == 2) ? 1 : 3;
const int MIN_EXP = (POST == 2) ? 0 : -6;
const float MIN_SUB = metal::ldexp(1.0f, MIN_EXP);
threadgroup float row[D];
threadgroup float row2[(POST == 2) ? D : 1];
threadgroup float red[TG];
threadgroup float gsc[(QW > 0) ? (QW / GROUP) : 1];
// load (+ per-head RMSNorm)
if (RMS != 0) {
  float ss = 0.0f;
  for (int j = lane; j < D; j += TG) { float v = x[h * D + j]; ss += v * v; }
  red[lane] = ss;
  threadgroup_barrier(metal::mem_flags::mem_threadgroup);
  for (int off = TG / 2; off > 0; off >>= 1) {
    if (lane < off) red[lane] += red[lane + off];
    threadgroup_barrier(metal::mem_flags::mem_threadgroup);
  }
  float rinv = metal::rsqrt(red[0] / (float)D + consts[0]);
  for (int j = lane; j < D; j += TG) row[j] = x[h * D + j] * rinv * w[j];
} else {
  for (int j = lane; j < D; j += TG) row[j] = x[h * D + j];
}
threadgroup_barrier(metal::mem_flags::mem_threadgroup);
// interleaved-pair rope on the tail RD dims (INV = conjugate rotation)
for (int p = lane; p < HALF; p += TG) {
  float a = row[D - RD + 2 * p];
  float b = row[D - RD + 2 * p + 1];
  float cv = cosr[p];
  float sv = (INV != 0) ? -sinr[p] : sinr[p];
  row[D - RD + 2 * p] = a * cv - b * sv;
  row[D - RD + 2 * p + 1] = a * sv + b * cv;
}
threadgroup_barrier(metal::mem_flags::mem_threadgroup);
if (POST == 2) {
  for (int j = lane; j < D; j += TG) {
    float hacc = 0.0f;
    for (int kk = 0; kk < D; ++kk) hacc += row[kk] * hada[kk * D + j];
    row2[j] = hacc;
  }
  threadgroup_barrier(metal::mem_flags::mem_threadgroup);
}
if (QW > 0) {
  const int NG = QW / GROUP;
  for (int g = lane; g < NG; g += TG) {
    float amax = AMAX_FLOOR;
    for (int e = 0; e < GROUP; ++e) {
      float v = (POST == 2) ? row2[g * GROUP + e] : row[g * GROUP + e];
      amax = metal::max(amax, metal::abs(v));
    }
    float q = amax / CODE_MAX;
    int ee = 0;
    float mm = metal::frexp(q, ee);
    int ce = (mm == 0.5f) ? (ee - 1) : ee;
    gsc[g] = metal::ldexp(1.0f, ce);
  }
  threadgroup_barrier(metal::mem_flags::mem_threadgroup);
}
for (int j = lane; j < D; j += TG) {
  float v = (POST == 2) ? row2[j] : row[j];
  float res = v;
  if (POST == 2 || (POST == 1 && j < D - RD)) {
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
  y_out[h * D + j] = res;
}