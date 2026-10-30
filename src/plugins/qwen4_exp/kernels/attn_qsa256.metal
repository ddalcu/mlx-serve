constexpr int BD = 256;
constexpr int LDK = BK + 8;
constexpr int LDV = BD + 8;
constexpr int NT = 32 * NSG;
constexpr int KT = BK / 8;

const int qL = q_shape[2];
const int kL = k_shape[2];
const int Hq = q_shape[1];
const int Hk = k_shape[1];
const int gqa = Hq / Hk;
const int KB = blocks_shape[2];

const int s = int(threadgroup_position_in_grid.x);
const int hk = int(threadgroup_position_in_grid.y);
const int bb = int(threadgroup_position_in_grid.z);
const ushort lane = ushort(thread_index_in_simdgroup);
const ushort warp = ushort(simdgroup_index_in_threadgroup);
const int tix = int(thread_index_in_threadgroup);

const float scale_log2e = scl[0] * 1.44269504088896340736f;
// Query s sits at absolute key position p (bottom-right aligned). Its
// visible keys: `count` complete blocks, then the tail [tail_start, p].
const int p = (kL - qL) + s;
const int complete = (p + 1) / RATIO;
const int count = metal::min(complete, KB);
const int sel_len = count * RATIO;
const int tail_start = complete * RATIO;
const int L = sel_len + (p + 1 - tail_start);

const device int* blk = blocks + (long)bb * blocks_strides[0] + (long)s * blocks_strides[1];
const device T* Kp = k + bb * k_strides[0] + hk * k_strides[1];
const device T* Vp = v + bb * v_strides[0] + hk * v_strides[1];

threadgroup T KVs[LDK * BD];
threadgroup T* Ks = KVs;
threadgroup T* Vs = KVs;

const short2 sc = msv_coord(lane);
const short sn = sc.x;
const short sm = sc.y;
const short tm = 8 * short(warp);
const int Ks_off = sm * LDK + sn;
const int Vs_off = sm * LDV + sn;
const int row = tm + sm;
const bool row_ok = row < gqa;

float2 Qfrag[BD / 8];
if (row_ok) {
  const device T* Qrow = q + bb * q_strides[0] + (long)(hk * gqa + row) * q_strides[1] + (long)s * q_strides[2];
  for (int dd = 0; dd < BD / 8; ++dd) {
    const vec<T, 2> pr = *((const device vec<T, 2>*)(Qrow + dd * 8 + sn));
    Qfrag[dd] = float2(float(pr.x), float(pr.y));
  }
} else {
  for (int dd = 0; dd < BD / 8; ++dd) Qfrag[dd] = float2(0.0f);
}
float2 Ofrag[BD / 8];
for (int i = 0; i < BD / 8; ++i) Ofrag[i] = float2(0.0f);
float max_score = -3.0e38f;
float sum_score = 0.0f;

for (int t0 = 0; t0 < L; t0 += BK) {
  const int rows_k = metal::min(BK, L - t0);

  threadgroup_barrier(metal::mem_flags::mem_threadgroup);
  for (int i = tix; i < BK * (BD / 8); i += NT) {
    const int r = i >> 5;
    const int c8 = i & 31;
    uint4 w = uint4(0);
    if (r < rows_k) {
      const int pos = msv_qsa_pos(blk, t0 + r, sel_len, tail_start, RATIO);
      w = *((const device uint4*)(Kp + (long)pos * k_strides[2]) + c8);
    }
    thread T* e = (thread T*)&w;
    const int cb = c8 * 8;
    for (int j = 0; j < 8; ++j) Ks[(cb + j) * LDK + r] = e[j];
  }
  threadgroup_barrier(metal::mem_flags::mem_threadgroup);

  float2 Sfrag[KT];
  for (int i = 0; i < KT; ++i) Sfrag[i] = float2(0.0f);
  for (int dd = 0; dd < BD / 8; ++dd) {
    const float2 qf = Qfrag[dd];
    const int kbase = Ks_off + dd * 8 * LDK;
    for (int kt = 0; kt < KT; ++kt) {
      const float2 kf = float2(float(Ks[kbase + kt * 8]), float(Ks[kbase + kt * 8 + 1]));
      msv_mma(Sfrag[kt], qf, kf);
    }
  }
  for (int kt = 0; kt < KT; ++kt) Sfrag[kt] *= scale_log2e;
  if (rows_k < BK) {
    for (int kt = 0; kt < KT; ++kt) {
      if (kt * 8 + sn >= rows_k) Sfrag[kt].x = -INFINITY;
      if (kt * 8 + sn + 1 >= rows_k) Sfrag[kt].y = -INFINITY;
    }
  }

  threadgroup_barrier(metal::mem_flags::mem_threadgroup);
  for (int i = tix; i < BK * (BD / 8); i += NT) {
    const int r = i >> 5;
    const int c8 = i & 31;
    uint4 w = uint4(0);
    if (r < rows_k) {
      const int pos = msv_qsa_pos(blk, t0 + r, sel_len, tail_start, RATIO);
      w = *((const device uint4*)(Vp + (long)pos * v_strides[2]) + c8);
    }
    *((threadgroup uint4*)(Vs + r * LDV) + c8) = w;
  }

  float new_max = max_score;
  for (int kt = 0; kt < KT; ++kt) new_max = metal::max(new_max, msv_row_max(Sfrag[kt]));
  float rowsum = 0.0f;
  for (int kt = 0; kt < KT; ++kt) {
    Sfrag[kt] = metal::exp2(Sfrag[kt] - new_max);
    rowsum += msv_row_sum(Sfrag[kt]);
  }
  const float factor = metal::exp2(max_score - new_max);
  max_score = new_max;
  sum_score = sum_score * factor + rowsum;
  for (int i = 0; i < BD / 8; ++i) Ofrag[i] *= factor;

  threadgroup_barrier(metal::mem_flags::mem_threadgroup);
  for (int id = 0; id < BD / 8; ++id) {
    const int vbase = Vs_off + id * 8;
    for (int kt = 0; kt < KT; ++kt) {
      const float2 vf = float2(float(Vs[vbase + kt * 8 * LDV]), float(Vs[vbase + kt * 8 * LDV + 1]));
      msv_mma(Ofrag[id], Sfrag[kt], vf);
    }
  }
}

if (row_ok) {
  const float inv = 1.0f / sum_score;
  device T* Optr = out + (((long)bb * Hq + (hk * gqa + row)) * (long)qL + (long)s) * BD + sn;
  for (int id = 0; id < BD / 8; ++id) {
    Optr[id * 8] = T(Ofrag[id].x * inv);
    Optr[id * 8 + 1] = T(Ofrag[id].y * inv);
  }
}