
    using namespace metal;
    constexpr uint TG   = 256;
    constexpr uint EPT  = 8;
    constexpr uint TILE = TG * EPT;
    constexpr uint REPL = 8;
    constexpr uint KD_NINF = 0xFF800000u;   // descending key of -inf

    const uint row  = threadgroup_position_in_grid.x;
    const uint lane = thread_position_in_threadgroup.x;
    const uint Nn = uint(N);
    const uint Kk = uint(K);
    const uint Ww = uint(W);
    const bool all_finite = (ALL_FINITE != 0);
    const uint clim = uint(clen[row]);
    const uint sbase = row * Nn;
    const uint obase = row * Ww;

    threadgroup atomic_uint hist[REPL * 256u];
    threadgroup uint red[TG];
    threadgroup uint sh[2];

    uint thr_key = 0u;   // the k-th best descending key (the reference's `thr`)
    uint rem     = Kk;   // k - n_gt: how many of the tied group to take

    if (!all_finite) {
        uint prefix = 0u;
        uint pmask  = 0u;
        for (uint p = 0u; p < 4u; ++p) {
            uint shift = 24u - 8u * p;
            for (uint i = lane; i < REPL * 256u; i += TG) {
                atomic_store_explicit(&hist[i], 0u, memory_order_relaxed);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            uint rep = (lane & (REPL - 1u)) * 256u;
            for (uint base = 0u; base < Nn; base += TILE) {
                for (uint e = 0u; e < EPT; ++e) {
                    uint i = base + e * TG + lane;
                    if (i < Nn) {
                        uint kd = dsv41_kd(score[sbase + i]);
                        if ((kd & pmask) == prefix) {
                            atomic_fetch_add_explicit(
                                &hist[rep + ((kd >> shift) & 0xFFu)], 1u,
                                memory_order_relaxed);
                        }
                    }
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            uint acc = 0u;                       // TG == 256 == bin count
            for (uint r = 0u; r < REPL; ++r) {
                acc += atomic_load_explicit(&hist[r * 256u + lane], memory_order_relaxed);
            }
            red[lane] = acc;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (lane == 0u) {
                uint c = 0u;
                uint b = 255u;
                for (uint q = 0u; q < 256u; ++q) {
                    uint h = red[q];
                    if (c + h >= rem) { b = q; break; }
                    c += h;
                }
                sh[0] = b;
                sh[1] = c;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            uint b = sh[0];
            uint c = sh[1];
            prefix |= (b << shift);
            pmask  |= (0xFFu << shift);
            rem    -= c;
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        thr_key = prefix;
    }

    // ---- emit: one ascending pass, two threadgroup scans per tile ----------
    uint run_eq  = 0u;   // #(kd == thr_key) already seen (the reference tie_rank)
    uint run_out = 0u;   // #outputs already written

    for (uint base = 0u; base < Nn; base += TILE) {
        bool is_lt[EPT];
        bool is_eq[EPT];
        uint local_eq = 0u;
        for (uint e = 0u; e < EPT; ++e) {
            uint i = base + lane * EPT + e;       // contiguous == ascending
            bool ok = (i < Nn);
            uint kd = ok ? dsv41_kd(score[sbase + i]) : 0xFFFFFFFFu;
            if (all_finite) {
                is_lt[e] = ok && (kd != KD_NINF);  // reference: score > -inf
                is_eq[e] = false;
            } else {
                is_lt[e] = ok && (kd < thr_key);
                is_eq[e] = ok && (kd == thr_key);
            }
            local_eq += is_eq[e] ? 1u : 0u;
        }

        red[lane] = local_eq;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint off = 1u; off < TG; off <<= 1) {
            uint t = (lane >= off) ? red[lane - off] : 0u;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            red[lane] = red[lane] + t;
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        uint eq_excl  = red[lane] - local_eq;
        uint eq_total = red[TG - 1u];
        threadgroup_barrier(mem_flags::mem_threadgroup);

        uint my_eq = run_eq + eq_excl;
        bool keep[EPT];
        uint local_keep = 0u;
        for (uint e = 0u; e < EPT; ++e) {
            // NOT named `sel`: that is the output buffer's parameter name.
            bool pick = is_lt[e];
            if (is_eq[e]) {
                pick = (my_eq < rem);
                my_eq += 1u;
            }
            uint i = base + lane * EPT + e;
            bool kp = pick && (i < Nn) && (i < clim);
            keep[e] = kp;
            local_keep += kp ? 1u : 0u;
            if (i < Nn) { mask[sbase + i] = kp; }
        }

        red[lane] = local_keep;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint off = 1u; off < TG; off <<= 1) {
            uint t = (lane >= off) ? red[lane - off] : 0u;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            red[lane] = red[lane] + t;
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        uint keep_excl  = red[lane] - local_keep;
        uint keep_total = red[TG - 1u];
        threadgroup_barrier(mem_flags::mem_threadgroup);

        uint pos = run_out + keep_excl;
        for (uint e = 0u; e < EPT; ++e) {
            if (keep[e]) {
                if (pos < Ww) { sel[obase + pos] = int(base + lane * EPT + e); }
                pos += 1u;
            }
        }

        run_eq  += eq_total;
        run_out += keep_total;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    for (uint p = run_out + lane; p < Ww; p += TG) { sel[obase + p] = -1; }
