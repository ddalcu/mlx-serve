    const uint t = thread_position_in_grid.y;
    const uint h = thread_position_in_grid.x * 4;
    if (h >= H) return;
    float4 acc = float4(0.0f);
    for (int k = 0; k < K; ++k) {
        const uint lc = loc[t * K + k];
        const device T* part;
        switch (lc >> 24) {
        case 0: part = p0; break;
        case 1: part = p1; break;
        case 2: part = p2; break;
        case 3: part = p3; break;
        case 4: part = p4; break;
        case 5: part = p5; break;
        case 6: part = p6; break;
        case 7: part = p7; break;
        case 8: part = p8; break;
        case 9: part = p9; break;
        case 10: part = p10; break;
        case 11: part = p11; break;
        case 12: part = p12; break;
        case 13: part = p13; break;
        case 14: part = p14; break;
        case 15: part = p15; break;
        case 16: part = p16; break;
        case 17: part = p17; break;
        case 18: part = p18; break;
        case 19: part = p19; break;
        case 20: part = p20; break;
        case 21: part = p21; break;
        case 22: part = p22; break;
        case 23: part = p23; break;
        case 24: part = p24; break;
        case 25: part = p25; break;
        case 26: part = p26; break;
        case 27: part = p27; break;
        default: part = p0; break;
        }
        const vec<T, 4> y = *(const device vec<T, 4>*)(part + ulong(lc & 0xFFFFFFu) * H + h);
        acc = acc + float4(y) * w[t * K + k];
    }
    *(device vec<T, 4>*)(out + ulong(t) * H + h) = vec<T, 4>(acc);
