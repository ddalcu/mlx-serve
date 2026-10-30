    // mimo_fp8_dequant_bf16 (fp8res prompt route): wb[n, k] = bf16(e4m3(w[n, k]) * s[n / 64, k / 128]), one word (4 codes)
    // per thread. The f32 product is the converter's (fp8r_dq4: one f32 rounding), the BF16 rounding is the BF16 pack's:
    // wb is the BF16 pack's tensor bit for bit (the pack proof), so the stock GEMM over it is the BF16 tier's prompt GEMM.
    const size_t idx = size_t(thread_position_in_grid.x) * 4;
    if (idx >= size_t(N) * size_t(K)) {
      return;
    }
    const int n = int(idx / size_t(K));
    const int k = int(idx % size_t(K));
    const float s256 = scales[size_t(n / 64) * size_t(K / 128) + size_t(k / 128)] * 256.0f;
    float v[4];
    fp8r_dq4(*(const device uint*)(w + idx), s256, v);
    for (int j = 0; j < 4; j++) {
      wb[idx + j] = static_cast<bfloat16_t>(v[j]);
    }
