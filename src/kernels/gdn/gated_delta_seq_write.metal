  if (t + 1 < T) {
    auto t_state = state_seq + t * seq_stride + seq_base;
    for (int i = 0; i < n_per_t; ++i) {
      auto s_idx = n_per_t * dk_idx + i;
      t_state[s_idx] = static_cast<StT>(state[i]);
    }
  }
