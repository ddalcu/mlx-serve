constexpr int NR = 4;              // RMS_N_READS
uint gid = thread_position_in_grid.y;
uint lid = thread_position_in_threadgroup.x;
uint sg = simdgroup_index_in_threadgroup;
uint sl = thread_index_in_simdgroup;
threadgroup float local_inv[1];
threadgroup float local_sums[32];
if (lid < 32) local_sums[lid] = 0;
threadgroup_barrier(mem_flags::mem_threadgroup);
size_t base = (size_t)gid * (size_t)AXIS + (size_t)lid * NR;
float x1[NR];
float x2[NR];
float acc1 = 0;
float acc2 = 0;
for (int i = 0; i < NR; i++) {
  bool in = int(lid) * NR + i < AXIS;
  x1[i] = in ? float(b1[base + i]) : 0.0f;
  x2[i] = (B2 && in) ? float(b2[base + i]) : 0.0f;
  acc1 += x1[i] * x1[i];
  acc2 += x2[i] * x2[i];
}
float inv1 = PRE1 ? msv_row_inv_rms(acc1, eps1, AXIS, local_sums, local_inv, sg, sl) : 0.0f;
float inv2 = (B2 && PRE2) ? msv_row_inv_rms(acc2, eps2, AXIS, local_sums, local_inv, sg, sl) : 0.0f;
float xc[NR];
float accc = 0;
for (int i = 0; i < NR; i++) {
  int idx = int(lid) * NR + i;
  if (idx < AXIS) {
    T c = PRE1 ? T(w1[idx] * T(x1[i] * inv1)) : T(b1[base + i]);
    if (B2) {
      T n2 = PRE2 ? T(w2[idx] * T(x2[i] * inv2)) : T(b2[base + i]);
      c = c + n2;
    }
    xc[i] = float(c);
  } else {
    xc[i] = 0.0f;
  }
  accc += xc[i] * xc[i];
}
float invc = PREC ? msv_row_inv_rms(accc, epsc, AXIS, local_sums, local_inv, sg, sl) : 0.0f;
float xr[NR];
float accr = 0;
for (int i = 0; i < NR; i++) {
  int idx = int(lid) * NR + i;
  if (idx < AXIS) {
    T c = PREC ? T(wc[idx] * T(xc[i] * invc)) : T(xc[i]);
    T sv = a[base + i] + c;
    if (SCALE) sv = sv * scale[0];
    res[base + i] = sv;
    xr[i] = float(sv);
  } else {
    xr[i] = 0.0f;
  }
  accr += xr[i] * xr[i];
}
if (POST || POST2) {
  float invr = msv_row_inv_rms(accr, eps, AXIS, local_sums, local_inv, sg, sl);
  for (int i = 0; i < NR; i++) {
    int idx = int(lid) * NR + i;
    if (idx < AXIS) {
      T v = T(xr[i] * invr);
      if (POST) out[base + i] = wp[idx] * v;
      if (POST2) out2[base + i] = wp2[idx] * v;
    }
  }
}