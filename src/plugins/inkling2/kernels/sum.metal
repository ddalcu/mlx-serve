// Inkling 2, a prompt's S rows: the picked experts' outputs back to their rows, the MLP's short conv and the final
// add, one thread per output element. DN [N, 1, D] (TI) are the (row, pick) pairs' down projections in expert order,
// IDX [N] u32 their experts and INV [N] the place of pair (row, pick) in that order. Each is times SC[IDX] with
// SCALED (f32 [E]) and its routing weight WTS [1, S, K] (f32), summed in pick order and rounded to T: XM [1, S, D],
// the MLP's output. Then the conv over the rows behind the window before them (the last D columns of ST
// [NSLOT, KT - 1, SW] f32: RPS rows a slot, each slot behind its own window) with MW [D, KT, 1] (WT), its sum rounded to WT as the conv op does, plus XM, times PR[0],
// rounded to T, and the residual RES [1, S, D] f32 added: HOUT [1, S, D] (T), the visit's output stream.
const uint e = thread_position_in_grid.x;
const uint D = DN_shape[2];
const uint SW = ST_shape[2];
const uint RPS = RES_shape[1] / ST_shape[0];
const uint OFF = SW - D;
const uint s = e / D, c = e % D;
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
