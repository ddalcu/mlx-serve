#pragma METAL fp contract(off)
    const float SCALE = as_type<float>(1035273459u);
    const float KINV = as_type<float>(1004388352u);
    const float KBIAS = as_type<float>(3227320320u);
    uint p = threadgroup_position_in_grid.z;
    uint b = threadgroup_position_in_grid.x;
    uint e = threadgroup_position_in_grid.y;
    bool dn = p == 2u;
    const uint IN_ = dn ? 2304u : 5120u;
    const uint OUT_ = dn ? 5120u : 2304u;
    uint bj = dn ? (b % 40u) : (b % 18u);
    uint bi = dn ? (b / 40u) : (b / 18u);
    const device int16_t* code = p == 0u ? code_g : (p == 1u ? code_u : code_d);
    const constant int64_t* code_strides = p == 0u ? code_g_strides : (p == 1u ? code_u_strides : code_d_strides);
    const device float16_t* rout = p == 0u ? rout_g : (p == 1u ? rout_u : rout_d);
    const constant int64_t* rout_strides = p == 0u ? rout_g_strides : (p == 1u ? rout_u_strides : rout_d_strides);
    const device float16_t* rin = p == 0u ? rin_g : (p == 1u ? rin_u : rin_d);
    const constant int64_t* rin_strides = p == 0u ? rin_g_strides : (p == 1u ? rin_u_strides : rin_d_strides);
    device bfloat16_t* out = p == 0u ? og : (p == 1u ? ou : od);
    uint tid = thread_position_in_threadgroup.x;
    uint g = tid >> 5u;
    uint l = tid & 31u;
    threadgroup half LUT[1024];
    _Pragma("clang loop unroll(full)")
    for (uint j = 0u; j < 4u; ++j) {
        LUT[tid + 256u * j] = static_cast<half>(float(tid + 256u * j) * KINV + KBIAS);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    int64_t slot = (int64_t) slots[e];
    int64_t s3 = code_strides[3];
    int64_t tstep = 2 * code_strides[2];
    const device int16_t* tb = code + (slot * code_strides[0]
        + (int64_t) (bi * 8u + g) * code_strides[1]
        + (int64_t) (bj * 8u + (l >> 4u)) * code_strides[2]);
    bool hh = (l & 8u) != 0u;
    uint wb = 6u * (l & 7u) + 46u;
    int64_t wo[8];
    _Pragma("clang loop unroll(full)")
    for (uint u = 0u; u < 8u; ++u) {
        wo[u] = (int64_t) ((wb + u) % 48u) * s3;
    }
    float V[16][4];
    _Pragma("clang loop unroll(full)")
    for (uint m = 0u; m < 4u; ++m) {
        const device int16_t* tp = tb + (int64_t) m * tstep;
        uint P[4];
        _Pragma("clang loop unroll(full)")
        for (uint t = 0u; t < 4u; ++t) {
            P[t] = (uint)(ushort) tp[wo[2u * t]] | ((uint)(ushort) tp[wo[2u * t + 1u]] << 16u);
        }
        {
            uint lo = P[1];
            uint hi = P[0];
            uint sh = (hh ? 14u : 26u);
            uint x = (lo >> sh) | ((hi << 1u) << (31u - sh));
            { uint r_ = (x & 0xffffu) * 2212286765u; uint t_ = (r_ & 0x00ff00ffu) + ((r_ >> 8u) & 0x00ff00ffu);
              V[1u][m] = float(LUT[(t_ * 0x00010001u) >> 16u]); }
            { uint r_ = ((x >> 3u) & 0xffffu) * 2212286765u; uint t_ = (r_ & 0x00ff00ffu) + ((r_ >> 8u) & 0x00ff00ffu);
              V[0u][m] = float(LUT[(t_ * 0x00010001u) >> 16u]); }
        }
        {
            uint lo = (hh ? P[2] : P[1]);
            uint hi = (hh ? P[1] : P[0]);
            uint sh = (hh ? 22u : 2u);
            uint x = (lo >> sh) | ((hi << 1u) << (31u - sh));
            { uint r_ = (x & 0xffffu) * 2212286765u; uint t_ = (r_ & 0x00ff00ffu) + ((r_ >> 8u) & 0x00ff00ffu);
              V[3u][m] = float(LUT[(t_ * 0x00010001u) >> 16u]); }
            { uint r_ = ((x >> 3u) & 0xffffu) * 2212286765u; uint t_ = (r_ & 0x00ff00ffu) + ((r_ >> 8u) & 0x00ff00ffu);
              V[2u][m] = float(LUT[(t_ * 0x00010001u) >> 16u]); }
        }
        {
            uint lo = P[2];
            uint hi = P[1];
            uint sh = (hh ? 1u : 13u);
            uint x = (lo >> sh) | ((hi << 1u) << (31u - sh));
            { uint r_ = (x & 0xffffu) * 2212286765u; uint t_ = (r_ & 0x00ff00ffu) + ((r_ >> 8u) & 0x00ff00ffu);
              V[4u][m] = float(LUT[(t_ * 0x00010001u) >> 16u]); }
        }
        {
            uint lo = (hh ? P[3] : P[2]);
            uint hi = (hh ? P[2] : P[1]);
            uint sh = (hh ? 30u : 10u);
            uint x = (lo >> sh) | ((hi << 1u) << (31u - sh));
            { uint r_ = (x & 0xffffu) * 2212286765u; uint t_ = (r_ & 0x00ff00ffu) + ((r_ >> 8u) & 0x00ff00ffu);
              V[5u][m] = float(LUT[(t_ * 0x00010001u) >> 16u]); }
        }
        {
            uint lo = P[3];
            uint hi = P[2];
            uint sh = (hh ? 6u : 18u);
            uint x = (lo >> sh) | ((hi << 1u) << (31u - sh));
            { uint r_ = (x & 0xffffu) * 2212286765u; uint t_ = (r_ & 0x00ff00ffu) + ((r_ >> 8u) & 0x00ff00ffu);
              V[7u][m] = float(LUT[(t_ * 0x00010001u) >> 16u]); }
            { uint r_ = ((x >> 3u) & 0xffffu) * 2212286765u; uint t_ = (r_ & 0x00ff00ffu) + ((r_ >> 8u) & 0x00ff00ffu);
              V[6u][m] = float(LUT[(t_ * 0x00010001u) >> 16u]); }
        }
        {
            uint lo = P[1];
            uint hi = P[0];
            uint sh = (hh ? 8u : 20u);
            uint x = (lo >> sh) | ((hi << 1u) << (31u - sh));
            { uint r_ = (x & 0xffffu) * 2212286765u; uint t_ = (r_ & 0x00ff00ffu) + ((r_ >> 8u) & 0x00ff00ffu);
              V[9u][m] = float(LUT[(t_ * 0x00010001u) >> 16u]); }
            { uint r_ = ((x >> 3u) & 0xffffu) * 2212286765u; uint t_ = (r_ & 0x00ff00ffu) + ((r_ >> 8u) & 0x00ff00ffu);
              V[8u][m] = float(LUT[(t_ * 0x00010001u) >> 16u]); }
        }
        {
            uint lo = P[2];
            uint hi = P[1];
            uint sh = (hh ? 16u : 28u);
            uint x = (lo >> sh) | ((hi << 1u) << (31u - sh));
            { uint r_ = (x & 0xffffu) * 2212286765u; uint t_ = (r_ & 0x00ff00ffu) + ((r_ >> 8u) & 0x00ff00ffu);
              V[11u][m] = float(LUT[(t_ * 0x00010001u) >> 16u]); }
            { uint r_ = ((x >> 3u) & 0xffffu) * 2212286765u; uint t_ = (r_ & 0x00ff00ffu) + ((r_ >> 8u) & 0x00ff00ffu);
              V[10u][m] = float(LUT[(t_ * 0x00010001u) >> 16u]); }
        }
        {
            uint lo = (hh ? P[3] : P[2]);
            uint hi = (hh ? P[2] : P[1]);
            uint sh = (hh ? 24u : 4u);
            uint x = (lo >> sh) | ((hi << 1u) << (31u - sh));
            { uint r_ = (x & 0xffffu) * 2212286765u; uint t_ = (r_ & 0x00ff00ffu) + ((r_ >> 8u) & 0x00ff00ffu);
              V[13u][m] = float(LUT[(t_ * 0x00010001u) >> 16u]); }
            { uint r_ = ((x >> 3u) & 0xffffu) * 2212286765u; uint t_ = (r_ & 0x00ff00ffu) + ((r_ >> 8u) & 0x00ff00ffu);
              V[12u][m] = float(LUT[(t_ * 0x00010001u) >> 16u]); }
        }
        {
            uint lo = P[3];
            uint hi = P[2];
            uint sh = (hh ? 0u : 12u);
            uint x = (lo >> sh) | ((hi << 1u) << (31u - sh));
            { uint r_ = (x & 0xffffu) * 2212286765u; uint t_ = (r_ & 0x00ff00ffu) + ((r_ >> 8u) & 0x00ff00ffu);
              V[15u][m] = float(LUT[(t_ * 0x00010001u) >> 16u]); }
            { uint r_ = ((x >> 3u) & 0xffffu) * 2212286765u; uint t_ = (r_ & 0x00ff00ffu) + ((r_ >> 8u) & 0x00ff00ffu);
              V[14u][m] = float(LUT[(t_ * 0x00010001u) >> 16u]); }
        }
    }
    _Pragma("clang loop unroll(full)")
    for (uint h = 1u; h < 16u; h <<= 1u) {
        _Pragma("clang loop unroll(full)")
        for (uint i = 0u; i < 16u; ++i) {
            if ((i & h) == 0u) {
                _Pragma("clang loop unroll(full)")
                for (uint m = 0u; m < 4u; ++m) {
                    float a = V[i][m];
                    float b = V[i + h][m];
                    V[i][m] = a + b;
                    V[i + h][m] = a - b;
                }
            }
        }
    }
    threadgroup float X[4096];
    uint cr = l & 15u;
    uint hs = (l >> 4u) << 4u;
    uint ii = g + (hs >> 1u);
    float W[8][2][4];
    _Pragma("clang loop unroll(full)")
    for (uint m = 0u; m < 4u; ++m) {
        _Pragma("clang loop unroll(full)")
        for (uint i = 0u; i < 16u; ++i) {
            X[(g * 16u + i) * 32u + (l ^ (((i >> 3u) & 1u) << 4u))] = V[i][m];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        _Pragma("clang loop unroll(full)")
        for (uint u = 0u; u < 2u; ++u) {
            float x[8];
            _Pragma("clang loop unroll(full)")
            for (uint k = 0u; k < 8u; ++k) {
                x[k] = X[(k * 16u + ii) * 32u + ((cr + 16u * u) ^ hs)];
            }
            _Pragma("clang loop unroll(full)")
            for (uint h = 1u; h < 8u; h <<= 1u) {
                _Pragma("clang loop unroll(full)")
                for (uint k = 0u; k < 8u; ++k) {
                    if ((k & h) == 0u) {
                        float a = x[k];
                        float b = x[k + h];
                        if (h == 4u) {
                            x[k] = (a + b) * SCALE;
                            x[k + h] = (a - b) * SCALE;
                        } else {
                            x[k] = a + b;
                            x[k + h] = a - b;
                        }
                    }
                }
            }
            _Pragma("clang loop unroll(full)")
            for (uint k = 0u; k < 8u; ++k) {
                W[k][u][m] = x[k];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    _Pragma("clang loop unroll(full)")
    for (uint h = 1u; h < 16u; h <<= 1u) {
        bool upper = (l & h) != 0u;
        _Pragma("clang loop unroll(full)")
        for (uint k = 0u; k < 8u; ++k) {
            _Pragma("clang loop unroll(full)")
            for (uint u = 0u; u < 2u; ++u) {
                _Pragma("clang loop unroll(full)")
                for (uint m = 0u; m < 4u; ++m) {
                    float v = W[k][u][m];
                    float o = simd_shuffle_xor(v, (ushort) h);
                    W[k][u][m] = upper ? (o - v) : (v + o);
                }
            }
        }
    }
    _Pragma("clang loop unroll(full)")
    for (uint k = 0u; k < 8u; ++k) {
        _Pragma("clang loop unroll(full)")
        for (uint m = 0u; m < 4u; ++m) {
            float a = W[k][0][m];
            float b = W[k][1][m];
            W[k][0][m] = a + b;
            W[k][1][m] = a - b;
        }
    }
    _Pragma("clang loop unroll(full)")
    for (uint k = 0u; k < 8u; ++k) {
        _Pragma("clang loop unroll(full)")
        for (uint u = 0u; u < 2u; ++u) {
            _Pragma("clang loop unroll(full)")
            for (uint m = 0u; m < 4u; m += 2u) {
                float a = W[k][u][m];
                float b = W[k][u][m + 1u];
                W[k][u][m] = a + b;
                W[k][u][m + 1u] = a - b;
            }
        }
    }
    int64_t rbase = slot * rout_strides[0];
    int64_t ibase = slot * rin_strides[0];
    float RO[2][4];
    _Pragma("clang loop unroll(full)")
    for (uint u = 0u; u < 2u; ++u) {
        _Pragma("clang loop unroll(full)")
        for (uint m = 0u; m < 4u; ++m) {
            RO[u][m] = float(rout[rbase + (int64_t) (bj * 128u + cr + 16u * u + 32u * m) * rout_strides[1]]);
        }
    }
    _Pragma("clang loop unroll(full)")
    for (uint k = 0u; k < 8u; ++k) {
        uint r = bi * 128u + k * 16u + ii;
        float ri = float(rin[ibase + (int64_t) r * rin_strides[1]]);
        _Pragma("clang loop unroll(full)")
        for (uint u = 0u; u < 2u; ++u) {
            uint64_t o = ((uint64_t) e * IN_ + r) * OUT_ + bj * 128u + cr + 16u * u;
            _Pragma("clang loop unroll(full)")
            for (uint m = 0u; m < 2u; ++m) {
                float a = W[k][u][m];
                float b = W[k][u][m + 2u];
                float lo = a + b;
                float hi = a - b;
                float ylo = lo * SCALE;
                float yhi = hi * SCALE;
                float zlo = (ylo * RO[u][m]) * ri;
                float zhi = (yhi * RO[u][m + 2u]) * ri;
                out[o + 32u * m] = static_cast<bfloat16_t>(zlo);
                out[o + 32u * m + 64u] = static_cast<bfloat16_t>(zhi);
            }
        }
    }
