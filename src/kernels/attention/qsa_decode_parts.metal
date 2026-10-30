const int kL = k_shape[2];
const int split = int(threadgroup_position_in_grid.x);
const int hk = int(threadgroup_position_in_grid.y);
const ushort lane = ushort(thread_index_in_simdgroup);
const int head = hk * GQA + int(simdgroup_index_in_threadgroup);

const int nb = kL / RATIO;
const int sel_len = metal::min(nb, KB) * RATIO;
const int tail_start = nb * RATIO;
const int L = sel_len + (kL - tail_start);
const int j_lo = split * CH;
const int j_hi = metal::min(j_lo + CH, L);
const float scale_log2e = scl[0] * 1.44269504088896340736f;

const device T* Kp = k + (long)hk * k_strides[1];
const device T* Vp = v + (long)hk * v_strides[1];

// The chunk's block bases, loaded up front so no key waits on a block id.
int bp[CH / RATIO];
#pragma unroll
for (int i = 0; i < CH / RATIO; ++i) {
  const int b = j_lo / RATIO + i;
  bp[i] = (b * RATIO < sel_len) ? blocks[b] * RATIO : 0;
}
float qf[8];
msv_unpack8(*((const device uint4*)(q + (long)head * q_strides[1]) + lane), qf);
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
      kr[i] = *((const device uint4*)(Kp + (long)pos * k_strides[2]) + lane);
      vr[i] = *((const device uint4*)(Vp + (long)pos * v_strides[2]) + lane);
    }
    float part[8];
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
      float kf[8];
      msv_unpack8(kr[i], kf);
      float a0 = qf[0] * kf[0], a1 = qf[1] * kf[1];
      a0 = fma(qf[2], kf[2], a0); a1 = fma(qf[3], kf[3], a1);
      a0 = fma(qf[4], kf[4], a0); a1 = fma(qf[5], kf[5], a1);
      a0 = fma(qf[6], kf[6], a0); a1 = fma(qf[7], kf[7], a1);
      part[i] = a0 + a1;
    }
    const float total = msv_reduce8(part, lane);
    const float y = (jb + int(lane >> 2) < j_hi) ? total : -INFINITY;
    float bm = metal::max(y, simd_shuffle_xor(y, 4));
    bm = metal::max(bm, simd_shuffle_xor(bm, 8));
    bm = metal::max(bm, simd_shuffle_xor(bm, 16));
    const float m_new = metal::max(m, bm);
    const float factor = metal::exp2(m - m_new);
    const float p = metal::exp2(y - m_new);
    float lb = p + simd_shuffle_xor(p, 4);
    lb += simd_shuffle_xor(lb, 8);
    lb += simd_shuffle_xor(lb, 16);
    l = l * factor + lb;
    m = m_new;
    #pragma unroll
    for (int d = 0; d < 8; ++d) acc[d] *= factor;
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
      float vf[8];
      msv_unpack8(vr[i], vf);
      const float pi = simd_shuffle(p, ushort(4 * i));
      #pragma unroll
      for (int d = 0; d < 8; ++d) acc[d] = fma(pi, vf[d], acc[d]);
    }
  }
}

const long row = (long)head * NS + split;
device float* pa = pacc + row * 256 + lane * 8;
*((device float4*)pa) = float4(acc[0], acc[1], acc[2], acc[3]);
*((device float4*)(pa + 4)) = float4(acc[4], acc[5], acc[6], acc[7]);
if (lane == 0) {
  pml[row * 2] = m;
  pml[row * 2 + 1] = l;
}