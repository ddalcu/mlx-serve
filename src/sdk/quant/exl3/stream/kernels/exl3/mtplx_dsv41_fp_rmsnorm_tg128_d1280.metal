
    using namespace metal;
    constexpr uint TG = 128;
    constexpr uint D  = 1280;

    const uint row  = threadgroup_position_in_grid.x;
    const uint lane = thread_position_in_threadgroup.x;
    const float epsf = float(eps);
    const uint base = row * D;

    threadgroup float red[TG];

    // partial sum of squares over this lane's strided slice of the row
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
    float mean = red[0] / float(D);
    float inv = metal::precise::rsqrt(mean + epsf);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint i = lane; i < D; i += TG) {
        float v = float(x[base + i]) * inv;
        out[base + i] = static_cast<T>(float(weight[i]) * v);
    }
