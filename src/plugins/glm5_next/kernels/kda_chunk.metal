// MLX_SERVE_KDA_PROMPT_CHUNK (kda_chunk.zig, rounding-class): a KDA layer's prompt recurrence and its bounded gate
// as the chunkwise delta rule (fla's chunk_kda form) over 16-row chunks, the state carried once a chunk in f32 simdgroup
// matrices. For a value row s of the state (its Dk key channels) the per-token route runs, a token at a time,
//   s = g s;  u = beta (v - s.k);  s = s + u k;  y = s.q        g = exp(lb sigmoid(exp(A_log) (a + dt_bias)))
// and over a chunk's rows i, from the chunk's first state s0, with G_i a key channel's lg_0 + ... + lg_i (lg = log g):
//   z_i = v_i - (e^G_i k_i).s0      u = T z      T = (I + diag(beta) A)^-1 diag(beta)
//   A_ij = sum_c k_i k_j e^(G_i - G_j) (j < i)   B_ij = sum_c q_i k_j e^(G_i - G_j) (j <= i)
//   y_i = (e^G_i q_i).s0 + sum_j B_ij u_j        s_C = e^G_C s0 + sum_j u_j e^(G_C - G_j) k_j (C the last row)
// Each exponent is taken against the chunk's middle (m: the sum through row 7), so no factor passes e^(8 x 5) at the
// gate's bound of -5: no overflow, and no underflow of a term that counts (a flushed factor's term is below 1e-20).
//   Kg_i = k_i e^(G_i - G_m)   Qg_i = q_i e^(G_i - G_m)   Ki_j = k_j e^(G_m - G_j)   s' = e^G_m s0
//   z_i = v_i - Kg_i.s'   A_ij = Kg_i.Ki_j   B_ij = Qg_i.Ki_j   y_i = Qg_i.s' + sum_j B_ij u_j
//   s_C = e^(G_C - G_m) (s' + sum_j u_j Ki_j)
// The gate's exponent lg is kdaGateChain's statements (the f32 add, the multiply by exp(A_log), unary_ops.h Sigmoid,
// the multiply by the bound); rows past T are zero rows (k, q, v and beta 0, lg 0).
// Grid (256 x H x Dv / 64, 1, B), threadgroup (256, 1, 1): a threadgroup one head's 64 value rows, its 8 simdgroups 8
// value rows each, held as 16 8x8 f32 matrices (key channels in blocks of 8). A chunk: the gate walk (two threads a key
// channel, rows 0-7 summed down to the middle and rows 8-15 up from it), A and B (six 8x8 blocks over the key channels),
// T by forward substitution (16 lanes, a column each), then each simdgroup's z, u, y and state; 4 barriers. The next
// chunk's inputs load into registers while a chunk runs. Every loop is unrolled (MLX steel's STEEL_PRAGMA_UNROLL), so the
// state's 16 matrices, the inputs and the solve's column stay in registers.
#define KDA_CHUNK_UNROLL _Pragma("clang loop unroll(full)")
    constexpr int C = 16;
    constexpr int R = 64;
    constexpr int KS = Dk + 4;
    constexpr int SS = C + 4;
    constexpr int NB = Dk / 8;
    const int tid = int(thread_position_in_threadgroup.x);
    const int sg = int(simdgroup_index_in_threadgroup);
    const int lane = int(thread_index_in_simdgroup);
    const int blk = int(threadgroup_position_in_grid.x);
    const int bi = int(threadgroup_position_in_grid.z);
    const int hd = blk / (Dv / R);
    // an 8x8 matrix's elements a lane holds (MLX steel's BaseMMAFrag): row fm, columns fn and fn + 1
    const int qid = lane / 4;
    const int fm = (qid & 4) + ((lane / 2) % 4);
    const int fn = (qid & 2) * 2 + (lane % 2) * 2;
    // this lane's value row of the state, and its key channel and rows in the gate walk
    const int rv = (blk % (Dv / R)) * R + sg * 8 + fm;
    const int ch = tid % Dk;
    const int part = tid / Dk;
    const int nT = T;
    const int NC = (nT + C - 1) / C;
    const size_t row_qk = (size_t)H * Dk;
    const size_t row_v = (size_t)H * Dv;
    auto q_at = q + ((size_t)bi * nT * H + hd) * Dk + ch;
    auto k_at = k + ((size_t)bi * nT * H + hd) * Dk + ch;
    auto a_at = a + ((size_t)bi * nT * H + hd) * Dk + ch;
    auto v_at = v + ((size_t)bi * nT * H + hd) * Dv + rv;
    auto beta_at = beta + (size_t)bi * nT * H + hd;
    auto y_at = y + ((size_t)bi * nT * H + hd) * Dv + rv;
    const float ea = metal::precise::exp(A_log[hd]);
    const float dtb = dt_bias[hd * Dk + ch];
    const float lb = lower_bound[0];

    threadgroup float Kg[C * KS];
    threadgroup float Qg[C * KS];
    threadgroup float Ki[C * KS];
    threadgroup float Nt[C * SS];  // Nt[j][i] = beta_i A_ij (j < i), else 0
    threadgroup float Bt[C * SS];  // Bt[j][i] = B_ij (j <= i), else 0
    threadgroup float Tt[C * SS];  // Tt[j][i] = T_ij
    threadgroup float em[Dk];      // e^G_m
    threadgroup float ec[Dk];      // e^(G_C - G_m)
    threadgroup float bs[C];       // beta

    simdgroup_matrix<float, 8, 8> st[NB];
    {
        auto s_in = state_in + ((size_t)(bi * H + hd) * Dv + rv) * Dk + fn;
        KDA_CHUNK_UNROLL for (int cb = 0; cb < NB; ++cb) {
            st[cb].thread_elements()[0] = s_in[cb * 8];
            st[cb].thread_elements()[1] = s_in[cb * 8 + 1];
        }
    }

    InT pa[8], pk[8], pq[8], pv[4], cv[4];
    InT pb = InT(0);
#define KDA_CHUNK_FETCH(T0) { \
        KDA_CHUNK_UNROLL for (int rr = 0; rr < 8; ++rr) { \
            const int t = (T0) + part * 8 + rr; \
            const bool ok = t < nT; \
            const size_t off = (size_t)(ok ? t : 0) * row_qk; \
            pa[rr] = ok ? a_at[off] : InT(0); \
            pk[rr] = ok ? k_at[off] : InT(0); \
            pq[rr] = ok ? q_at[off] : InT(0); \
        } \
        KDA_CHUNK_UNROLL for (int e = 0; e < 4; ++e) { \
            const int t = (T0) + (e / 2) * 8 + fn + (e % 2); \
            pv[e] = t < nT ? v_at[(size_t)t * row_v] : InT(0); \
        } \
        if (tid < C) pb = (T0) + tid < nT ? beta_at[(size_t)((T0) + tid) * H] : InT(0); \
    }

    KDA_CHUNK_FETCH(0)
    for (int n = 0; n < NC; ++n) {
        const int t0 = n * C;
        // the gate walk: this thread's 8 rows of key channel ch
        {
            float lg[8];
            KDA_CHUNK_UNROLL for (int rr = 0; rr < 8; ++rr) {
                const float x = ea * (static_cast<float>(pa[rr]) + dtb);
                const float lv = kda_sigmoid(x) * lb;
                lg[rr] = t0 + part * 8 + rr < nT ? lv : 0.0f;
            }
            if (part == 0) {
                // rows 7 down to 0: G_i - G_m = -(lg_(i+1) + ... + lg_7) = -d
                float d = 0.0f;
                KDA_CHUNK_UNROLL for (int rr = 7; rr >= 0; --rr) {
                    const float up = metal::precise::exp(-d);
                    const float dn = metal::precise::exp(d);
                    const float kf = static_cast<float>(pk[rr]);
                    Kg[rr * KS + ch] = kf * up;
                    Qg[rr * KS + ch] = static_cast<float>(pq[rr]) * up;
                    Ki[rr * KS + ch] = kf * dn;
                    d += lg[rr];
                }
                em[ch] = metal::precise::exp(d);
            } else {
                // rows 8 up to 15: G_i - G_m = lg_8 + ... + lg_i = l
                float l = 0.0f;
                KDA_CHUNK_UNROLL for (int rr = 0; rr < 8; ++rr) {
                    l += lg[rr];
                    const float dn = metal::precise::exp(l);
                    const float up = metal::precise::exp(-l);
                    const float kf = static_cast<float>(pk[rr]);
                    Kg[(8 + rr) * KS + ch] = kf * dn;
                    Qg[(8 + rr) * KS + ch] = static_cast<float>(pq[rr]) * dn;
                    Ki[(8 + rr) * KS + ch] = kf * up;
                }
                ec[ch] = metal::precise::exp(l);
            }
            if (tid < C) bs[tid] = static_cast<float>(pb);
        }
        KDA_CHUNK_UNROLL for (int e = 0; e < 4; ++e) cv[e] = pv[e];
        if (n + 1 < NC) KDA_CHUNK_FETCH(t0 + C)
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // A and B: blocks (j block, i block) (0, 0), (0, 1) and (1, 1) of each (j > i in block (1, 0): zero, never read)
        if (sg < 6) {
            const int pr = sg % 3;
            const int jb = pr == 2 ? 1 : 0;
            const int ib = pr == 0 ? 0 : 1;
            const threadgroup float* X = sg < 3 ? Kg : Qg;
            simdgroup_matrix<float, 8, 8> acc0 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
            simdgroup_matrix<float, 8, 8> acc1 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
            KDA_CHUNK_UNROLL for (int cb = 0; cb < NB; cb += 2) {
                simdgroup_matrix<float, 8, 8> l0, r0, l1, r1;
                reinterpret_cast<thread float2&>(l0.thread_elements()) = *(const threadgroup float2*)(&Ki[(jb * 8 + fm) * KS + cb * 8 + fn]);
                r0.thread_elements()[0] = X[(ib * 8 + fn) * KS + cb * 8 + fm];
                r0.thread_elements()[1] = X[(ib * 8 + fn + 1) * KS + cb * 8 + fm];
                reinterpret_cast<thread float2&>(l1.thread_elements()) = *(const threadgroup float2*)(&Ki[(jb * 8 + fm) * KS + cb * 8 + 8 + fn]);
                r1.thread_elements()[0] = X[(ib * 8 + fn) * KS + cb * 8 + 8 + fm];
                r1.thread_elements()[1] = X[(ib * 8 + fn + 1) * KS + cb * 8 + 8 + fm];
                simdgroup_multiply_accumulate(acc0, l0, r0, acc0);
                simdgroup_multiply_accumulate(acc1, l1, r1, acc1);
            }
            KDA_CHUNK_UNROLL for (int e = 0; e < 2; ++e) {
                const int j = jb * 8 + fm;
                const int i = ib * 8 + fn + e;
                const float val = acc0.thread_elements()[e] + acc1.thread_elements()[e];
                if (sg < 3) Nt[j * SS + i] = j < i ? bs[i] * val : 0.0f;
                else Bt[j * SS + i] = j <= i ? val : 0.0f;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // T = (I + N)^-1 diag(beta), N = diag(beta) A: lane `lane` solves column `lane` of (I + N) x = e, then T's
        // column is x beta_lane (Tt[lane][i] = T[i][lane])
        if (sg == 7 && lane < C) {
            float x[C];
            KDA_CHUNK_UNROLL for (int i = 0; i < C; ++i) x[i] = i == lane ? 1.0f : 0.0f;
            KDA_CHUNK_UNROLL for (int j = 0; j < C - 1; ++j) {
                const float xj = x[j];
                KDA_CHUNK_UNROLL for (int i = 1; i < C; ++i) if (i > j) x[i] = metal::fma(-Nt[j * SS + i], xj, x[i]);
            }
            const float bc = bs[lane];
            KDA_CHUNK_UNROLL for (int i = 0; i < C; ++i) Tt[lane * SS + i] = x[i] * bc;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // this simdgroup's 8 value rows: s' = e^G_m s0, Kg.s' and Qg.s', z, u = T z, y, the state
        {
            simdgroup_matrix<float, 8, 8> pz0 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
            simdgroup_matrix<float, 8, 8> pz1 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
            simdgroup_matrix<float, 8, 8> po0 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
            simdgroup_matrix<float, 8, 8> po1 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
            KDA_CHUNK_UNROLL for (int cb = 0; cb < NB; ++cb) {
                const float2 m2 = *(const threadgroup float2*)(&em[cb * 8 + fn]);
                st[cb].thread_elements()[0] *= m2.x;
                st[cb].thread_elements()[1] *= m2.y;
                simdgroup_matrix<float, 8, 8> k0, k1, q0, q1;
                k0.thread_elements()[0] = Kg[fn * KS + cb * 8 + fm];
                k0.thread_elements()[1] = Kg[(fn + 1) * KS + cb * 8 + fm];
                k1.thread_elements()[0] = Kg[(8 + fn) * KS + cb * 8 + fm];
                k1.thread_elements()[1] = Kg[(9 + fn) * KS + cb * 8 + fm];
                q0.thread_elements()[0] = Qg[fn * KS + cb * 8 + fm];
                q0.thread_elements()[1] = Qg[(fn + 1) * KS + cb * 8 + fm];
                q1.thread_elements()[0] = Qg[(8 + fn) * KS + cb * 8 + fm];
                q1.thread_elements()[1] = Qg[(9 + fn) * KS + cb * 8 + fm];
                simdgroup_multiply_accumulate(pz0, st[cb], k0, pz0);
                simdgroup_multiply_accumulate(pz1, st[cb], k1, pz1);
                simdgroup_multiply_accumulate(po0, st[cb], q0, po0);
                simdgroup_multiply_accumulate(po1, st[cb], q1, po1);
            }
            // z (as z^T: value rows x chunk rows) = v - Kg.s'
            pz0.thread_elements()[0] = static_cast<float>(cv[0]) - pz0.thread_elements()[0];
            pz0.thread_elements()[1] = static_cast<float>(cv[1]) - pz0.thread_elements()[1];
            pz1.thread_elements()[0] = static_cast<float>(cv[2]) - pz1.thread_elements()[0];
            pz1.thread_elements()[1] = static_cast<float>(cv[3]) - pz1.thread_elements()[1];
            // u^T = z^T T^T (T lower triangular: block (1, 0) of Tt is zero)
            simdgroup_matrix<float, 8, 8> u0 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
            simdgroup_matrix<float, 8, 8> u1 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
            simdgroup_matrix<float, 8, 8> f00, f01, f11;
            reinterpret_cast<thread float2&>(f00.thread_elements()) = *(const threadgroup float2*)(&Tt[fm * SS + fn]);
            reinterpret_cast<thread float2&>(f01.thread_elements()) = *(const threadgroup float2*)(&Tt[fm * SS + 8 + fn]);
            reinterpret_cast<thread float2&>(f11.thread_elements()) = *(const threadgroup float2*)(&Tt[(8 + fm) * SS + 8 + fn]);
            simdgroup_multiply_accumulate(u0, pz0, f00, u0);
            simdgroup_multiply_accumulate(u1, pz0, f01, u1);
            simdgroup_multiply_accumulate(u1, pz1, f11, u1);
            // y^T = Qg.s' + u^T B^T (B lower triangular)
            reinterpret_cast<thread float2&>(f00.thread_elements()) = *(const threadgroup float2*)(&Bt[fm * SS + fn]);
            reinterpret_cast<thread float2&>(f01.thread_elements()) = *(const threadgroup float2*)(&Bt[fm * SS + 8 + fn]);
            reinterpret_cast<thread float2&>(f11.thread_elements()) = *(const threadgroup float2*)(&Bt[(8 + fm) * SS + 8 + fn]);
            simdgroup_multiply_accumulate(po0, u0, f00, po0);
            simdgroup_multiply_accumulate(po1, u0, f01, po1);
            simdgroup_multiply_accumulate(po1, u1, f11, po1);
            {
                const int ta = t0 + fn;
                const int tb = t0 + 8 + fn;
                if (ta < nT) y_at[(size_t)ta * row_v] = static_cast<InT>(po0.thread_elements()[0]);
                if (ta + 1 < nT) y_at[(size_t)(ta + 1) * row_v] = static_cast<InT>(po0.thread_elements()[1]);
                if (tb < nT) y_at[(size_t)tb * row_v] = static_cast<InT>(po1.thread_elements()[0]);
                if (tb + 1 < nT) y_at[(size_t)(tb + 1) * row_v] = static_cast<InT>(po1.thread_elements()[1]);
            }
            // the state: e^(G_C - G_m) (s' + u^T Ki)
            KDA_CHUNK_UNROLL for (int cb = 0; cb < NB; ++cb) {
                simdgroup_matrix<float, 8, 8> i0, i1;
                reinterpret_cast<thread float2&>(i0.thread_elements()) = *(const threadgroup float2*)(&Ki[fm * KS + cb * 8 + fn]);
                reinterpret_cast<thread float2&>(i1.thread_elements()) = *(const threadgroup float2*)(&Ki[(8 + fm) * KS + cb * 8 + fn]);
                simdgroup_multiply_accumulate(st[cb], u0, i0, st[cb]);
                simdgroup_multiply_accumulate(st[cb], u1, i1, st[cb]);
                const float2 c2 = *(const threadgroup float2*)(&ec[cb * 8 + fn]);
                st[cb].thread_elements()[0] *= c2.x;
                st[cb].thread_elements()[1] *= c2.y;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
#undef KDA_CHUNK_FETCH

    {
        auto s_out = state_out + ((size_t)(bi * H + hd) * Dv + rv) * Dk + fn;
        KDA_CHUNK_UNROLL for (int cb = 0; cb < NB; ++cb) {
            s_out[cb * 8] = st[cb].thread_elements()[0];
            s_out[cb * 8 + 1] = st[cb].thread_elements()[1];
        }
    }
#undef KDA_CHUNK_UNROLL
