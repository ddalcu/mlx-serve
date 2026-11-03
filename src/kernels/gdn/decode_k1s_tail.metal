    // The next token reads the state serial decoding stored (StT), not the f32.
    for (int i = 0; i < 4; ++i) st[j][i] = float(static_cast<StT>(st[j][i]));
    if (t + 1 < TL) {
      uint sbase = t * (HV * DV * DK) + (hv * DV + dv) * DK + lane * 4;
      for (int i = 0; i < 4; ++i) state_seq[sbase + i] = static_cast<StT>(st[j][i]);
    }
  }
}
for (int j = 0; j < R; ++j) {
  uint base = (hv * DV + row0 + j) * DK + lane * 4;
  for (int i = 0; i < 4; ++i) state_out[base + i] = static_cast<StT>(st[j][i]);
}