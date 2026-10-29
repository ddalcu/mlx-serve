
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

// Partials for the merge pass, UNNORMALIZED: the running max and the
// running sum belong to this split alone, and only the merge knows the
// row's global max. `msv_row_max`/`msv_row_sum` reduce across the four
// lanes holding a row, so every lane of the row carries the same m/l
// and exactly one of them (sn == 0) writes the pair.
//
// An empty split writes l = 0, which is the merge's skip flag: a
// NON-empty split always has l >= 1, because its own maximum element
// contributes exp2(max - max) = 1. So l == 0 identifies emptiness
// exactly, and the merge never evaluates (-inf) - (-inf).
if (row_ok) {
  const int hq = hk * gqa + row;
  const long pidx = ((long)hq * (long)qL + (long)s) * NSPLIT + jsp;
  device float* Aptr = pacc + pidx * BD + sn;
  for (int id = 0; id < BD / 8; ++id) {
    Aptr[id * 8] = Ofrag[id].x;
    Aptr[id * 8 + 1] = Ofrag[id].y;
  }
  if (sn == 0) {
    const bool empty = (t_lo >= t_hi);
    pml[pidx * 2] = empty ? -INFINITY : max_score;
    pml[pidx * 2 + 1] = empty ? 0.0f : sum_score;
  }
}