  threadgroup float xchg[WM * 2 * 16 * 32];
  omlx_nax::attention_nax_bdv_qt<
      T, 16 * WM, 32, 256, 256, WM, 2,
      ALIGN_Q, ALIGN_K, true, 0, false,
      true, true, float, int8_t>(
      q, k, v, out,
      reinterpret_cast<const device omlx_nax::AttnParams*>(params),
      q_strides, k_strides, v_strides,
      sinks, state, state_out, xchg,
      simdgroup_index_in_threadgroup,
      thread_index_in_simdgroup,
      threadgroup_position_in_grid,
      qi, ki, qs, ks);