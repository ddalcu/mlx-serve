    float row_sum[TMR];
    for (int t = 0; t < TMR; t++) {
      float half_sum = 0.0f;
      if (xlive[t]) for (int i = 0; i < GS / 2; i++) half_sum += float(xp[t][g * GS + i]);
      const float other = simd_shuffle_xor(half_sum, ushort(1));
      row_sum[t] = xh ? other + half_sum : half_sum + other;
    }
