threadgroup_barrier(mem_flags::mem_threadgroup);
if (tid == 0) {
  float gt = 0.0f;
  for (uint g = 1; g < TG / 32; ++g) gt += gp[g];
  glogit[row] = TOUT(gt);
}