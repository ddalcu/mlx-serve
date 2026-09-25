// Body of `msv_w8a8` (see w8a8_header.metal for the contract and the includes).

  const uint2 tgp = uint2(threadgroup_position_in_grid.xy);
  const uint sgit = simdgroup_index_in_threadgroup;
  const uint lane = thread_index_in_simdgroup;

  const int M = xq_shape[0], K = xq_shape[1];
  const int N = wq_shape[1];
  const int m0 = int(tgp.y) * W8A8_BM, n0 = int(tgp.x) * W8A8_BN;
  const int sg = int(sgit);
  const int rm = m0 + (sg >> 1) * 32;
  const int cn = n0 + (sg & 1) * 64;

  constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(
      16, 32, 32, false, true, true,
      mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
  mpp::tensor_ops::matmul2d<desc, metal::execution_simdgroup> op;
  auto B0 = op.get_right_input_cooperative_tensor<int8_t, int8_t, int32_t>();
  auto B1 = op.get_right_input_cooperative_tensor<int8_t, int8_t, int32_t>();
  auto C0 = op.get_destination_cooperative_tensor<metal::remove_addrspace_t<decltype(B0)>, metal::remove_addrspace_t<decltype(B0)>, int32_t>();
  auto C1 = op.get_destination_cooperative_tensor<metal::remove_addrspace_t<decltype(B0)>, metal::remove_addrspace_t<decltype(B0)>, int32_t>();
  auto C2 = op.get_destination_cooperative_tensor<metal::remove_addrspace_t<decltype(B0)>, metal::remove_addrspace_t<decltype(B0)>, int32_t>();
  auto C3 = op.get_destination_cooperative_tensor<metal::remove_addrspace_t<decltype(B0)>, metal::remove_addrspace_t<decltype(B0)>, int32_t>();
  auto A0 = op.get_left_input_cooperative_tensor<int8_t, int8_t, int32_t>();
  auto A1 = op.get_left_input_cooperative_tensor<int8_t, int8_t, int32_t>();
  #pragma clang loop unroll(full)
  for (uint i = 0; i < 16; ++i) {
    if (C0.is_valid_element(i)) { C0[i] = 0; C1[i] = 0; C2[i] = 0; C3[i] = 0; }
  }

  const int kb = int(lane & 1) * 4 + int((lane >> 3) & 1) * 8;
  const int q  = int(((lane >> 1) & 3) + ((lane >> 4) & 1) * 4);

  for (int k0 = 0; k0 < K; k0 += 32) {
    #pragma clang loop unroll(full)
    for (int g = 0; g < 4; ++g) {
      const int row0 = cn + q + g * 8;
      const int row1 = row0 + 32;
      const long wbase0 = ((long)(k0 / 32) * (long)N + row0) * 32L + kb;
      const long wbase1 = ((long)(k0 / 32) * (long)N + row1) * 32L + kb;
      uchar4 v00 = (row0 < N) ? *(const device uchar4*)(wq + wbase0) : uchar4(0);
      uchar4 v01 = (row0 < N) ? *(const device uchar4*)(wq + wbase0 + 16) : uchar4(0);
      uchar4 v10 = (row1 < N) ? *(const device uchar4*)(wq + wbase1) : uchar4(0);
      uchar4 v11 = (row1 < N) ? *(const device uchar4*)(wq + wbase1 + 16) : uchar4(0);
      B0[g*4+0]=v00.x; B0[g*4+1]=v00.y; B0[g*4+2]=v00.z; B0[g*4+3]=v00.w;
      B0[16+g*4+0]=v01.x; B0[16+g*4+1]=v01.y; B0[16+g*4+2]=v01.z; B0[16+g*4+3]=v01.w;
      B1[g*4+0]=v10.x; B1[g*4+1]=v10.y; B1[g*4+2]=v10.z; B1[g*4+3]=v10.w;
      B1[16+g*4+0]=v11.x; B1[16+g*4+1]=v11.y; B1[16+g*4+2]=v11.z; B1[16+g*4+3]=v11.w;
    }
    #pragma clang loop unroll(full)
    for (int t = 0; t < W8A8_TM; ++t) {
      const int r0 = rm + t * 16;
      const int rl = r0 + q;
      const bool lo_ok = rl < M;
      const bool hi_ok = (rl + 8) < M;
      const long base = (long)rl * K + k0 + kb;
      uchar4 a0 = lo_ok ? *(const device uchar4*)(xq + base) : uchar4(0);
      uchar4 a1 = lo_ok ? *(const device uchar4*)(xq + base + 16) : uchar4(0);
      uchar4 a2 = hi_ok ? *(const device uchar4*)(xq + base + 8 * K) : uchar4(0);
      uchar4 a3 = hi_ok ? *(const device uchar4*)(xq + base + 8 * K + 16) : uchar4(0);
      thread uint* ap = (thread uint*)((t == 0) ? &A0[0] : &A1[0]);
      ap[0] = as_type<uint>(a0);
      ap[1] = as_type<uint>(a2);
      ap[2] = as_type<uint>(a1);
      ap[3] = as_type<uint>(a3);
    }
    op.run(A0, B0, C0);
    op.run(A1, B0, C1);
    op.run(A0, B1, C2);
    op.run(A1, B1, C3);
  }
  #pragma clang loop unroll(full)
  for (int t = 0; t < W8A8_TM; ++t) {
    const int r0 = rm + t * 16;
    const thread int32_t* cf0 = (t == 0) ? &C0[0] : &C1[0];
    const thread int32_t* cf1 = (t == 0) ? &C2[0] : &C3[0];
    #pragma clang loop unroll(full)
    for (int i = 0; i < 16; ++i) {
      const int r = r0 + q + ((i >> 2) & 1) * 8;
      const int c0 = cn + kb + (i & 3) + ((i >> 3) & 1) * 16;
      const int c1 = c0 + 32;
      if (r < M && c0 < N) y[(long)r * N + c0] = bfloat16_t(float(cf0[i]) * sx[r] * sw[c0]);
      if (r < M && c1 < N) y[(long)r * N + c1] = bfloat16_t(float(cf1[i]) * sx[r] * sw[c1]);
    }
  }
