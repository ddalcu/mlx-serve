
    // q3ht_mixfin (exact vs stock for the same mm and ssq): rsqrt(ssq * (1 / (HC D)) + norm_eps) (the smallm_all
    // _hc_mixes_split mean statement), mixes = mm * rsqrt, then hc_split_sinkhorn's statements: pre = sigmoid(mixes *
    // scale[0] + base) + hc_eps, post = 2 * sigmoid(mixes * scale[1] + base), comb = mixes * scale[2] + base.
    const int m = int(threadgroup_position_in_grid.y);
    const int n = int(thread_position_in_threadgroup.x);
    if (n < N) {
        const float ms = static_cast<float>(ssq[m]) * 4.882812573e-05f;         // separate statements: no contraction
        const float rs = metal::precise::rsqrt(ms + 9.999999683e-21f);
        const float mixv = static_cast<float>(mm[m * N + n]) * rs;
        const float bn = static_cast<float>(base[n]);
        if (n < HC) {
            const float a = mixv * static_cast<float>(scale[0]);
            const float b = a + bn;
            pre[m * HC + n] = q3ht_sigmoid(b) + 9.999999975e-07f;
        } else if (n < 2 * HC) {
            const float a = mixv * static_cast<float>(scale[1]);
            const float b = a + bn;
            post[m * HC + n - HC] = 2.0f * q3ht_sigmoid(b);
        } else {
            const float a = mixv * static_cast<float>(scale[2]);
            comb[m * (HC * HC) + n - 2 * HC] = a + bn;
        }
    }
