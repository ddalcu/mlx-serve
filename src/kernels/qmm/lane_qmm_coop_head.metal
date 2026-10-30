  const ushort sg = simdgroup_index_in_threadgroup;
  const ushort lane = thread_index_in_simdgroup;
  const ushort slice = sg >> 1;
  const int M = mdims[0];
  constexpr int KG = K / GS;
  const int n0 = threadgroup_position_in_grid.x * 64;
  const int rb = threadgroup_position_in_grid.y * 16 * TMR;
  const int g_begin = (slice * KG) / SK;
  const int g_end = ((slice + 1) * KG) / SK;
  constexpr auto desc = matmul2d_descriptor(16 * TMR, 64, GS, false, true, false, matmul2d_descriptor::mode::multiply);
  matmul2d<desc, execution_simdgroups<2>> op;
  tensor<device bfloat, dextents<int32_t, 2>, tensor_inline> tA((device bfloat*)X + (int64_t)rb * K, dextents<int32_t, 2>(K, M - rb));
  tensor<device uint4b_format, dextents<int32_t, 2>, tensor_inline> tB((device uchar*)W, dextents<int32_t, 2>(K, N));
  auto a0 = tA.slice(0, 0);
  auto b0 = tB.slice(0, 0);
  auto P = op.template get_destination_cooperative_tensor<decltype(a0), decltype(b0), float>();
  constexpr int CAP = 16 * TMR;                             // 16 TMR x 64 outputs over 64 threads
  short ecol[CAP], erow[CAP];
  for (int i = 0; i < CAP; i++) { auto ids = P.get_multidimensional_index(i); ecol[i] = ids[0]; erow[i] = ids[1]; }
  float C[CAP];
  for (int i = 0; i < CAP; i++) C[i] = 0.0f;
#if !TILED
  const device uint* sbw = (const device uint*)SBt;
#endif