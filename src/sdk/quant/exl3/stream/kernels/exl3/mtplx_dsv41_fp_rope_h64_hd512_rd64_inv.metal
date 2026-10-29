
    using namespace metal;
    constexpr uint H  = 64;
    constexpr uint HD = 512;
    constexpr uint RD = 64;
    constexpr uint HALF = RD / 2;
    constexpr uint TAIL0 = HD - RD;

    const uint gid = thread_position_in_grid.x;   // one thread per (row,head,d)
    const uint total = uint(rows) * H * HD;
    if (gid >= total) return;

    const uint d    = gid % HD;
    const uint tmp  = gid / HD;
    const uint head = tmp % H;
    const uint row  = tmp / H;
    const uint Sc = uint(S);
    const uint s_idx = (Sc > 0u) ? (row % Sc) : 0u;
    const uint head_base = (row * H + head) * HD;

    float o;
    if (d < TAIL0) {
        o = float(x[head_base + d]);
    } else {
        uint j = d - TAIL0;
        uint p = j >> 1;
        uint cs = s_idx * HALF + p;
        float c = float(cos[cs]);
        float sn = - float(sin[cs]);
        float v0 = float(x[head_base + TAIL0 + 2u * p]);
        float v1 = float(x[head_base + TAIL0 + 2u * p + 1u]);
        o = (j & 1u) ? (v0 * sn + v1 * c) : (v0 * c - v1 * sn);
    }
    out[gid] = static_cast<T>(o);
