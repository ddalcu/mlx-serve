        const float2 f = XF2[(size_t(min(rb / 8 + rt, T8 - 1)) * G + g) * 256 + 32 * s + lane];
        bm[rt].thread_elements()[0] = f.x;
        bm[rt].thread_elements()[1] = f.y;