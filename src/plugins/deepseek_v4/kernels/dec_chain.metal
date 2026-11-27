int h = thread_position_in_grid.y;
int lane = thread_position_in_threadgroup.x;
if (h >= H) return; // uniform per threadgroup (grid.y == H)
const int HALF = RD / 2;
const int QW = (POST == 1) ? (D - RD) : ((POST == 2) ? D : 0);
const int GROUP = (POST == 2) ? 32 : 64;
const float CODE_MAX = (POST == 2) ? 6.0f : 448.0f;
const float AMAX_FLOOR = (POST == 2) ? metal::ldexp(6.0f, -126) : 1e-4f;
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
      amax = mlxserve_maximum(amax, metal::abs(v));
    }
    // ue8m0: MLX's e8m0 rounded up (2^ceil(log2(q)) for every finite q here), the max keeping a NaN.
    gsc[g] = mlxserve_mx_scale_round_up(amax / CODE_MAX);
  }
  threadgroup_barrier(metal::mem_flags::mem_threadgroup);
}
for (int j = lane; j < D; j += TG) {
  float v = (POST == 2) ? row2[j] : row[j];
  float res = v;
  if (POST == 2 || (POST == 1 && j < D - RD)) {
    float scale = gsc[j / GROUP];
    float y = mlxserve_clip(v / scale, -CODE_MAX, CODE_MAX);
    // MLX's encoder on |y|, the sign the composed chain's sign(y): a zero comes out +0.
    const float a = metal::abs(y);
    const float qv = (POST == 2) ? mlxserve_fp4_e2m1_round(a) : mlxserve_fp8_e4m3_round(a);
    res = (qv * float((y > 0.0f) - (y < 0.0f))) * scale;
  }
  y_out[h * D + j] = res;
}