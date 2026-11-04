  for (int r = 0; r < RPW; ++r) {
    auto t_state = state_seq + t * seq_stride_arr[0] + (n * Dv + dv0 + r) * Dk;
    for (int i = 0; i < n_per_t; ++i) t_state[n_per_t * dk_idx + i] = static_cast<StT>(state[r][i]);
  }
