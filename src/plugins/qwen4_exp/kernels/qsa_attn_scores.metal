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

