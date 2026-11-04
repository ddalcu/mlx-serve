
    // GLM-5.3 EXL3 decode, stage 4: each row's routed pairs summed in routed order (their scores applied by stage 3),
    // f32, written as T. ct: per (row, routed j) the pair's source wave << 24 | its row in that wave's output, or
    // 0xFFFFFFFF where an earlier combine of the chain summed it (into acc, when ACC). One thread per 4 hidden values.
    const uint t = thread_position_in_grid.y;
    const uint h = thread_position_in_grid.x * 4u;
    if (h >= uint(HID)) return;
    float4 a = float4(0.0f);
    if (ACC) a = *reinterpret_cast<const device float4*>(acc + size_t(t) * uint(HID) + h);
    for (uint j = 0u; j < uint(TOPK); ++j) {
        const uint lc = ct[t * uint(TOPK) + j];
        if (lc == 0xffffffffu) continue;
        const device float* src;
        switch (lc >> 24u) {
            case 0u: src = s0; break;
            case 1u: src = s1; break;
            case 2u: src = s2; break;
            case 3u: src = s3; break;
            case 4u: src = s4; break;
            case 5u: src = s5; break;
            case 6u: src = s6; break;
            case 7u: src = s7; break;
            case 8u: src = s8; break;
            case 9u: src = s9; break;
            case 10u: src = s10; break;
            case 11u: src = s11; break;
            case 12u: src = s12; break;
            case 13u: src = s13; break;
            case 14u: src = s14; break;
            default: src = s15; break;
        }
        a += *reinterpret_cast<const device float4*>(src + size_t(lc & 0xffffffu) * uint(HID) + h);
    }
    *reinterpret_cast<device vec<T, 4>*>(out + size_t(t) * uint(HID) + h) = vec<T, 4>(a);
