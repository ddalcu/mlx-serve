// Inkling 2, a prompt's S rows: the picked experts' outputs back to their rows, the MLP's short conv and the final
// add, one thread per output element (or 4 elements with VEC4). DN [N, 1, D] (TI) are the (row, pick) pairs' down
// projections in expert order, IDX [N] u32 their experts and INV [N] the place of pair (row, pick) in that order.
// Each is times SC[IDX] with SCALED (f32 [E]) and its routing weight WTS [1, S, K] (f32), summed in pick order and
// rounded to T: XM [1, S, D], the MLP's output. Then the conv over the rows behind the window before them (the last
// D columns of ST [NSLOT, KT - 1, SW] f32: RPS rows a slot, each slot behind its own window) with MW [D, KT, 1] (WT),
// its sum rounded to WT as the conv op does, plus XM, times PR[0], rounded to T, and the residual RES [1, S, D] f32
// added: HOUT [1, S, D] (T), the visit's output stream.
const uint e = thread_position_in_grid.x;
const uint D = DN_shape[2];
const uint SW = ST_shape[2];
const uint RPS = RES_shape[1] / ST_shape[0];
const uint OFF = SW - D;
const uint s = VEC4 ? (e / (D / 4)) : (e / D);
const uint c = VEC4 ? ((e % (D / 4)) * 4) : (e % D);
const uint slot = s / RPS;
const uint pos = s % RPS;
int row_src[KT];
if constexpr (TREE) {
    int cur = int(pos);
    int need = int(KT) - 1;
    row_src[need] = cur;
    while (need > 0 && cur >= 0) {
        cur = PARENT[slot * RPS + uint(cur)];
        --need;
        if (cur >= 0) row_src[need] = cur;
        else {
            for (int z = need; z >= 0; --z) row_src[z] = -(need - z + 1);
            break;
        }
    }
} else {
    for (uint j = 0; j < KT; ++j) row_src[j] = int(pos) + int(j) - (int(KT) - 1);
}
if constexpr (VEC4) {
    float4 win[KT];
    for (uint j = 0; j < KT; j++) {
        if (row_src[j] < 0) {
            const uint st_row = uint(int(KT) - 1 + row_src[j]);
            win[j] = *((const device float4*)(ST + (slot * (KT - 1) + st_row) * SW + OFF + c));
            continue;
        }
        const uint r = slot * RPS + uint(row_src[j]);
        float4 x = float4(0.0f);
        for (uint k = 0; k < K; k++) {
            const uint at = uint(INV[r * K + k]);
            const float sc = SCALED ? SC[IDX[at]] : 1.0f;
            const float wt = WTS[r * K + k];
            const float4 dn = float4(*((const device vec<T, 4>*)(DN + size_t(at) * D + c)));
            x += (dn * sc) * wt;
        }
        win[j] = float4(vec<T, 4>(x));
    }
    const size_t base = size_t(s) * D + c;
    *((device vec<T, 4>*)(XM + base)) = vec<T, 4>(win[KT - 1]);
    float4 y = float4(0.0f);
    for (uint j = 0; j < KT; j++) {
        const float4 mw = float4(float(MW[(c + 0) * KT + j]), float(MW[(c + 1) * KT + j]),
                                 float(MW[(c + 2) * KT + j]), float(MW[(c + 3) * KT + j]));
        y += win[j] * mw;
    }
    y = float4(vec<WT, 4>(y));
    const float4 res = *((const device float4*)(RES + base));
    *((device vec<T, 4>*)(HOUT + base)) = vec<T, 4>(res + float4(vec<T, 4>((y + win[KT - 1]) * PR[0])));
} else {
    float win[KT];
    for (uint j = 0; j < KT; j++) {
        // Tap j reads row `row_src[j]` of the slot's outputs, or ST when negative.
        if (row_src[j] < 0) {
            const uint st_row = uint(int(KT) - 1 + row_src[j]);
            win[j] = ST[(slot * (KT - 1) + st_row) * SW + OFF + c];
            continue;
        }
        const uint r = slot * RPS + uint(row_src[j]);
        float x = 0.0f;
        for (uint k = 0; k < K; k++) {
            const uint at = uint(INV[r * K + k]);
            x += float(DN[size_t(at) * D + c]) * (SCALED ? SC[IDX[at]] : 1.0f) * WTS[r * K + k];
        }
        win[j] = float(T(x));
    }
    XM[e] = T(win[KT - 1]);
    float y = 0.0f;
    for (uint j = 0; j < KT; j++) y += win[j] * float(MW[c * KT + j]);
    y = float(WT(y));
    HOUT[e] = T(RES[e] + float(T((y + win[KT - 1]) * PR[0])));
}
