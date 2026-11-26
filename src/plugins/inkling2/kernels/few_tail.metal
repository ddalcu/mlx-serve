// Inkling 2, a few rows: reduce grouped DN [S * KSEL, 1, D] (T) into XM [1, S, D], run the MLP short conv + residual
// RES into HOUT [1, S, D] (T), and either write verify conv windows WIN [NSLOT, KT - 1 + RPS, SW] f32 (OPEN) or
// write the updated conv state WIN [NSLOT, KT - 1, SW] f32 directly (STATE): the S rows are NSLOT slots of RPS rows,
// each behind its own window ST [NSLOT, KT - 1, SW].
const uint e = thread_position_in_grid.x;
const uint S = RES_shape[1];
const uint NSLOT = ST_shape[0];
const uint RPS = S / NSLOT;
const uint D = RES_shape[2];
const uint SW = ST_shape[2];
const uint OFF = SW - D;
if (e < D) {
    const uint c = e;
    // Pass 1: sum KSEL picks once per row s into XM[s, c] (and WIN's XM tail when OPEN).
    for (uint s = 0; s < S; ++s) {
        float x = 0.0f;
        for (uint k = 0; k < KSEL; ++k) {
            const uint at = s * KSEL + k;
            x += float(DN[size_t(at) * D + c]) * (SCALED ? SC[IDX[at]] : 1.0f) * WTS[at];
        }
        const T xm = T(x);
        XM[s * D + c] = xm;
        if constexpr (OPEN) {
            WIN[((s / RPS) * (KT - 1 + RPS) + KT - 1 + s % RPS) * SW + OFF + c] = float(xm);
        }
    }
    // Pass 2: MLP short conv (reading XM back) + residual -> HOUT.
    float mw[KT];
    for (uint j = 0; j < KT; ++j) mw[j] = float(MW[c * KT + j]);
    for (uint s = 0; s < S; ++s) {
        const uint slot = s / RPS;
        const uint pos = s % RPS;
        float y = 0.0f;
        float last_x = 0.0f;
        for (uint j = 0; j < KT; ++j) {
            int p;
            if constexpr (TREE) {
                p = int(pos);
                for (uint stp = 0; stp < KT - 1 - j; ++stp) p = (p >= 0) ? PARENT[slot * RPS + uint(p)] : (p - 1);
            } else {
                p = int(pos) + int(j) - int(KT - 1);
            }
            float wv;
            if (p < 0) {
                wv = ST[(slot * (KT - 1) + uint(int(KT - 1) + p)) * SW + OFF + c];
            } else {
                wv = float(XM[(slot * RPS + uint(p)) * D + c]);
            }
            y += wv * mw[j];
            if (j + 1 == KT) last_x = wv;
        }
        y = float(WT(y));
        HOUT[s * D + c] = T(RES[s * D + c] + float(T((y + last_x) * PR[0])));
    }
    if constexpr (STATE) {
        for (uint slot = 0; slot < NSLOT; ++slot) {
            for (uint j = 0; j < KT - 1; ++j) {
                const uint at = RPS + j;
                float val;
                if (at < KT - 1) val = ST[(slot * (KT - 1) + at) * SW + OFF + c];
                else val = float(XM[(slot * RPS + at - (KT - 1)) * D + c]);
                WIN[(slot * (KT - 1) + j) * SW + OFF + c] = val;
            }
        }
    }
    return;
}
if constexpr (OPEN) {
    const uint we = e - D;
    const uint st_elems = NSLOT * (KT - 1) * SW;
    if (we < st_elems) {
        const uint slot = we / ((KT - 1) * SW);
        WIN[slot * (KT - 1 + RPS) * SW + we % ((KT - 1) * SW)] = ST[we];
        return;
    }
    const uint re = we - st_elems;
    const uint r = re / OFF;
    const uint col = re % OFF;
    const uint C = KP_shape[2];
    float val;
    if (col < C) val = float(KP[r * C + col]);
    else if (col < 2 * C) val = float(VP[r * C + col - C]);
    else val = float(OP[r * D + col - 2 * C]);
    WIN[((r / RPS) * (KT - 1 + RPS) + KT - 1 + r % RPS) * SW + col] = val;
} else if constexpr (STATE) {
    const uint we = e - D;
    if (we >= NSLOT * (KT - 1) * OFF) return;
    const uint col = we % OFF;
    const uint slot = we / ((KT - 1) * OFF);
    const uint j = (we / OFF) % (KT - 1);
    const uint at = RPS + j;
    float val;
    if (at < KT - 1) {
        val = ST[(slot * (KT - 1) + at) * SW + col];
    } else {
        const uint r = slot * RPS + at - (KT - 1);
        const uint C = KP_shape[2];
        if (col < C) val = float(KP[r * C + col]);
        else if (col < 2 * C) val = float(VP[r * C + col - C]);
        else val = float(OP[r * D + col - 2 * C]);
    }
    WIN[(slot * (KT - 1) + j) * SW + col] = val;
}
