  const ushort lane = thread_index_in_simdgroup;
  const ushort sg = simdgroup_index_in_threadgroup;     // K slice
  const short qid = lane >> 2;
  const short fm = (qid & 4) | ((lane >> 1) & 3);       // fragment row of this lane (and fm + 8)
  const short fn = ((qid & 2) | (lane & 1)) * 4;        // first of its four fragment columns
  const int M = mdims[0];
  constexpr int KG = K / GS;
  constexpr int NF = 2;
  const int n0 = threadgroup_position_in_grid.x * 32;
  const int rb = threadgroup_position_in_grid.y * 16 * TMR;
  const int g_begin = (sg * KG) / SK;
  const int g_end = ((sg + 1) * KG) / SK;
  constexpr auto desc = matmul2d_descriptor(16 * TMR, 32, GS, false, true, false, matmul2d_descriptor::mode::multiply);
  matmul2d<desc, execution_simdgroup> op;
  tensor<device bfloat, dextents<int32_t, 2>, tensor_inline> tA((device bfloat*)X + (int64_t)rb * K, dextents<int32_t, 2>(K, M - rb));
#if BITS == 4
  tensor<device uint4b_format, dextents<int32_t, 2>, tensor_inline> tB((device uchar*)W, dextents<int32_t, 2>(K, N));
#elif BITS == 8 && !TILED
  tensor<device uint8_t, dextents<int32_t, 2>, tensor_inline> tB((device uint8_t*)W, dextents<int32_t, 2>(K, N));
#elif BITS == 8
#else
  // 5- and 6-bit: lane l widens column n0 + l's group into bytes for the bf16 x uint8 op.
  constexpr int WPG = GS * BITS / 32;
  threadgroup uint stage_all[SK * 32 * (GS / 4)];
  threadgroup uint* stage = stage_all + sg * 32 * (GS / 4);
  tensor<threadgroup uint8_t, dextents<int32_t, 2>, tensor_inline> b((threadgroup uint8_t*)stage, dextents<int32_t, 2>(GS, 32));
  const device uint* wcol = TILED ? (const device uint*)W + (int64_t)(R0 / 32 + threadgroup_position_in_grid.x) * KG * 32 * WPG + lane * WPG
                                  : (const device uint*)W + (int64_t)(n0 + lane) * (K * BITS / 32);
#endif
  float C[TMR][NF * 8];
  for (int t = 0; t < TMR; t++) for (int i = 0; i < NF * 8; i++) C[t][i] = 0.0f;
#if !TILED
  const device uint4* sbv = (const device uint4*)SBt;
#endif
  bool colok[NF];
  for (int f = 0; f < NF; f++) colok[f] = n0 + f * 16 + fn < N;