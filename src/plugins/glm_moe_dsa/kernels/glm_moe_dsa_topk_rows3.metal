    const uint row = threadgroup_position_in_grid.y;
    const uint n = uint(params[row]);
    const device float* srow = s0;
    switch (row) {
        case 1: srow = s1; break;
        case 2: srow = s2; break;
        default: break;
    }
    const uint tid = thread_position_in_threadgroup.x;
    const uint lane = thread_index_in_simdgroup;
    const uint sg = simdgroup_index_in_threadgroup;
    threadgroup atomic_uint hist[2048];
    threadgroup uint part[32];
    threadgroup uint st[6];
    threadgroup atomic_uint taken;
    threadgroup ulong sel[K];
    // st: the composite prefix (value key, index), its masks, the elements still to take.
    if (tid == 0) {
        st[0] = 0; st[1] = 0; st[2] = 0; st[3] = 0; st[4] = uint(K); st[5] = 0;
        atomic_store_explicit(&taken, 0u, memory_order_relaxed);
    }
    uint ibits = 0;
    while ((1u << ibits) < n) ibits++;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // Passes over the composite key from its top: the value key in 11 + 11 + 10 bits, then the index in 11-bit
    // digits (only while the threshold's ties outnumber the elements still to take).
    const uint passes = 3 + (ibits + 10) / 11;
    for (uint p = 0; p < passes; ++p) {
        const bool on_value = p < 3;
        const uint width = on_value ? (p < 2 ? 11u : 10u) : min(11u, ibits - (p - 3) * 11u);
        const uint shift = on_value ? (p == 0 ? 21u : (p == 1 ? 10u : 0u)) : ibits - (p - 3) * 11u - width;
        const uint nbins = 1u << width;
        if (p >= 3 && st[5] == 1u) break;
        for (uint b = tid; b < 2048u; b += 1024u) atomic_store_explicit(&hist[b], 0u, memory_order_relaxed);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const uint pv = st[0], mv = st[1], pi = st[2], mi = st[3];
        for (uint i = tid; i < n; i += 1024u) {
            const float f = srow[i];
            uint u = as_type<uint>(f);
            if (f == 0.0f) u = 0u;
            const uint kv = (u & 0x80000000u) ? ~u : (u | 0x80000000u);
            if ((kv & mv) != pv || (i & mi) != pi) continue;
            const uint d = ((on_value ? kv : i) >> shift) & (nbins - 1u);
            atomic_fetch_add_explicit(&hist[d], 1u, memory_order_relaxed);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // Thread t holds bins nbins - 1 - 2t and nbins - 2 - 2t: its inclusive prefix counts the elements from the
        // top bin down to its lower bin.
        const int bh = int(nbins) - 1 - 2 * int(tid);
        const int bl = bh - 1;
        const uint ch = bh >= 0 ? atomic_load_explicit(&hist[bh], memory_order_relaxed) : 0u;
        const uint cl = bl >= 0 ? atomic_load_explicit(&hist[bl], memory_order_relaxed) : 0u;
        const uint s = ch + cl;
        const uint inc = simd_prefix_inclusive_sum(s);
        if (lane == 31) part[sg] = inc;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        uint before = 0;
        for (uint k = 0; k < sg; ++k) before += part[k];
        const uint upto = before + inc;
        const uint above = upto - s;
        const uint need = st[4];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (above < need && upto >= need) {
            uint bin, left, ties;
            if (above + ch >= need) { bin = uint(bh); left = need - above; ties = ch; }
            else { bin = uint(bl); left = need - above - ch; ties = cl; }
            if (on_value) { st[0] = pv | (bin << shift); st[1] = mv | ((nbins - 1u) << shift); }
            else { st[2] = pi | (bin << shift); st[3] = mi | ((nbins - 1u) << shift); }
            st[4] = left;
            // Every element of the bin taken: no lower digit decides among them.
            st[5] = ties == left ? 1u : 0u;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    // The k elements at or above the threshold: the prefix's elements whose remaining digits are all taken (the
    // threshold's bin, kept whole) and every element whose composite key is above the prefix.
    const uint pv = st[0], mv = st[1], pi = st[2], mi = st[3];
    for (uint i = tid; i < n; i += 1024u) {
        const float f = srow[i];
        uint u = as_type<uint>(f);
        if (f == 0.0f) u = 0u;
        const uint kv = (u & 0x80000000u) ? ~u : (u | 0x80000000u);
        const bool hi_v = (kv & mv) > pv;
        const bool eq_v = (kv & mv) == pv;
        const bool keep = hi_v || (eq_v && ((i & mi) >= pi));
        if (keep) {
            const uint at = atomic_fetch_add_explicit(&taken, 1u, memory_order_relaxed);
            if (at < uint(K)) sel[at] = (ulong(kv) << 32) | ulong(i);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // Ascending (value, index): a bitonic sort of the k composite keys.
    for (uint size = 2; size <= uint(K); size <<= 1) {
        for (uint stride = size >> 1; stride > 0; stride >>= 1) {
            for (uint t = tid; t < uint(K) / 2u; t += 1024u) {
                const uint i = (t / stride) * 2u * stride + (t % stride);
                const uint j = i + stride;
                const bool up = (i & size) == 0u;
                const ulong a = sel[i];
                const ulong b = sel[j];
                if ((a > b) == up) { sel[i] = b; sel[j] = a; }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
    for (uint j = tid; j < uint(K); j += 1024u) out[ulong(row) * uint(K) + j] = int(uint(sel[j] & 0xffffffffu));