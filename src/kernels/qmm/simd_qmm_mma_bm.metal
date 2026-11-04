    simdgroup_matrix<float, 8, 8> P[RT][NT];
    PRAGMA_UNROLL
    for (int rt = 0; rt < RT; rt++)
      for (int t = 0; t < NT; t++) P[rt][t] = simdgroup_matrix<float, 8, 8>(0.0f);
    PRAGMA_UNROLL
    for (int s = 0; s < 8; s++) {
      const float ps = wpre<BITS>(s);
      simdgroup_matrix<float, 8, 8> bm[RT];
      PRAGMA_UNROLL
      for (int rt = 0; rt < RT; rt++) {