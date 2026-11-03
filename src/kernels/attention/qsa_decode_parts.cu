const int r = blockIdx.z;
const int kL = k_shape[2] - ((int)gridDim.z - 1) + r;
const int split = blockIdx.x;
const int hk = blockIdx.y;
const int lane = threadIdx.x & 31;
const int head = hk * GQA + (threadIdx.x >> 5);
const int nb = kL / RATIO;
const int sel_len = min(nb, KB) * RATIO;
const int tail_start = nb * RATIO;
const int L = sel_len + (kL - tail_start);
const int j_lo = split * CH;
const int j_hi = min(j_lo + CH, L);
const float scale_log2e = scl[0] * 1.44269504088896340736f;
const T* Kp = k + (long)hk * k_strides[1];
const T* Vp = v + (long)hk * v_strides[1];
const int* rb = blocks + (long)r * blocks_strides[1];
// A pick that is not a complete block before this row (a later row's block) adds no key: the
// tail covers its visible tokens.
int bp[CH / RATIO];
unsigned bad = 0u;
#pragma unroll
for (int i = 0; i < CH / RATIO; ++i) {
  const int b = j_lo / RATIO + i;
  const bool sel = b * RATIO < sel_len;
  const int pick = sel ? rb[b] : 0;
  bp[i] = (sel && pick < nb) ? pick * RATIO : 0;
  if (sel && pick >= nb) bad |= ((1u << RATIO) - 1u) << (i * RATIO);
}
float qf[8];
msv_unpack8(*((const uint4*)(q + (long)head * q_strides[1] + (long)r * q_strides[2]) + lane), qf);
#pragma unroll
for (int d = 0; d < 8; ++d) qf[d] *= scale_log2e;
float m = -3.0e38f, l = 0.0f, acc[8];
#pragma unroll
for (int d = 0; d < 8; ++d) acc[d] = 0.0f;
#pragma unroll
for (int bi = 0; bi < CH / 8; ++bi) {
  const int jb = j_lo + bi * 8;
  if (jb < j_hi) {
    uint4 kr[8], vr[8];
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
      const int jj = bi * 8 + i;
      const int j = j_lo + jj;
      const int pos = (j >= j_hi) ? 0 : ((j < sel_len) ? bp[jj / RATIO] + jj % RATIO : tail_start + (j - sel_len));
      kr[i] = *((const uint4*)(Kp + (long)pos * k_strides[2]) + lane);
      vr[i] = *((const uint4*)(Vp + (long)pos * v_strides[2]) + lane);
    }
    float part[8];
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
      float kf[8];
      msv_unpack8(kr[i], kf);
      float a0 = qf[0] * kf[0], a1 = qf[1] * kf[1];
      a0 = fmaf(qf[2], kf[2], a0); a1 = fmaf(qf[3], kf[3], a1);
      a0 = fmaf(qf[4], kf[4], a0); a1 = fmaf(qf[5], kf[5], a1);
      a0 = fmaf(qf[6], kf[6], a0); a1 = fmaf(qf[7], kf[7], a1);
      part[i] = a0 + a1;
    }
    const float total = msv_reduce8(part, lane);
    const int jk = bi * 8 + (lane >> 2);
    const float y = (jb + (lane >> 2) < j_hi && !((bad >> jk) & 1u)) ? total : -INFINITY;
    float bm = fmaxf(y, __shfl_xor_sync(0xffffffffu, y, 4));
    bm = fmaxf(bm, __shfl_xor_sync(0xffffffffu, bm, 8));
    bm = fmaxf(bm, __shfl_xor_sync(0xffffffffu, bm, 16));
    const float m_new = fmaxf(m, bm);
    const float factor = exp2f(m - m_new);
    const float p = exp2f(y - m_new);
    float lb = p + __shfl_xor_sync(0xffffffffu, p, 4);
    lb += __shfl_xor_sync(0xffffffffu, lb, 8);
    lb += __shfl_xor_sync(0xffffffffu, lb, 16);
    l = l * factor + lb;
    m = m_new;
    #pragma unroll
    for (int d = 0; d < 8; ++d) acc[d] *= factor;
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
      float vf[8];
      msv_unpack8(vr[i], vf);
      const float pi = __shfl_sync(0xffffffffu, p, 4 * i);
      #pragma unroll
      for (int d = 0; d < 8; ++d) acc[d] = fmaf(pi, vf[d], acc[d]);
    }
  }
}
const long row = ((long)r * gridDim.y * GQA + head) * NS + split;
float* pa = pacc + row * 256 + lane * 8;
*((float4*)pa) = make_float4(acc[0], acc[1], acc[2], acc[3]);
*((float4*)(pa + 4)) = make_float4(acc[4], acc[5], acc[6], acc[7]);
if (lane == 0) {
  pml[row * 2] = m;
  pml[row * 2 + 1] = l;
}