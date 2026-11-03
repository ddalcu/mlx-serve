  threadgroup float xchg[WN == 2 ? WM * 2 * 16 * 32 : 1];
  omlx_nax::attention_nax_bdv<
      T, 16 * WM, 32, DK, 128, WM, WN,
      ALIGN_Q, ALIGN_K, true, WINDOW, HAS_SINKS,
      FIRST, LAST, float>(
      q, k, v, out,
      reinterpret_cast<const device omlx_nax::AttnParams*>(params),
      q_strides, k_strides, v_strides,
      sinks, state, state_out, xchg,
      simdgroup_index_in_threadgroup,
      thread_index_in_simdgroup,
      threadgroup_position_in_grid);