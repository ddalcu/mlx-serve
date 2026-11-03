
    using namespace metal;
    constexpr uint TG = 128;
    constexpr uint D  = 512;
    constexpr uint RD = 64;
    constexpr uint HALF = RD / 2;

    const uint row  = threadgroup_position_in_grid.x;
    const uint lane = thread_position_in_threadgroup.x;
    const uint Sc = uint(S);
    const uint s_idx = (Sc > 0u) ? (row % Sc) : 0u;
    const float epsf = float(eps);
    const uint base = row * D;
    const uint cs_base = s_idx * HALF;

    threadgroup float red[TG];
    threadgroup float nrm[D];   // the normed row (pre-RoPE), shared for the pair reads

    float ss = 0.0f;
    for (uint i = lane; i < D; i += TG) {
        float v = float(x[base + i]);
        ss += v * v;
    }
    red[lane] = ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = TG >> 1; stride > 0; stride >>= 1) {
        if (lane < stride) { red[lane] = red[lane] + red[lane + stride]; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float inv = metal::precise::rsqrt(red[0] / float(D) + epsf);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint i = lane; i < D; i += TG) {
        nrm[i] = float(weight[i]) * (float(x[base + i]) * inv);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // head part (< D-RD): copy through; tail: interleaved-pair RoPE
    const uint tail0 = D - RD;
    for (uint i = lane; i < D; i += TG) {
        float o;
        if (i < tail0) {
            o = nrm[i];
        } else {
            uint j = i - tail0;          // 0..RD-1 within the rope tail
            uint p = j >> 1;             // pair index
            float c = float(cos[cs_base + p]);
            float sn = float(sin[cs_base + p]);
            float v0 = nrm[tail0 + 2u * p];
            float v1 = nrm[tail0 + 2u * p + 1u];
            o = (j & 1u) ? (v0 * sn + v1 * c) : (v0 * c - v1 * sn);
        }
        out[base + i] = static_cast<T>(o);
    }
