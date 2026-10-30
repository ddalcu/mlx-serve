  const nfrag w0 = nax_wfrag_k<N>(words, lane);
  const nfrag w1 = nax_wfrag_k<N>(words + PACKED_W, lane);
  for (short s = 0; s < 8; s++) {
    right[s] = w0[s];
    right[8 + s] = w1[s];
  }
  nax_left_rows(left, xl0 + tk * TILE, xl1 + tk * TILE);
  op.run(left, right, destination);
