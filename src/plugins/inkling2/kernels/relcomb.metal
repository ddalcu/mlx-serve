// Inkling 2: relsplit.metal's splits combined and times the output gate. grid (D, L * HQ): one thread per output
// element. PART [NS, L * HQ, D] f32, ML [NS, L * HQ, 2] f32, GATE [L * HQ * D] f32 (sigmoid). OUT [1, L, HQ * D] (T)
// = the read rounded to T, times its gate, rounded to T (as gated.metal).
const uint d = thread_position_in_grid.x;
const uint lh = thread_position_in_grid.y;
const uint NS = PART_shape[0];
const uint LH = PART_shape[1];
float m = -INFINITY;
for (uint s = 0; s < NS; s++) m = max(m, ML[(s * LH + lh) * 2]);
float num = 0.0f, den = 0.0f;
for (uint s = 0; s < NS; s++) {
    const float w = precise::exp2(ML[(s * LH + lh) * 2] - m);
    num += w * PART[(s * LH + lh) * D + d];
    den += w * ML[(s * LH + lh) * 2 + 1];
}
OUT[lh * D + d] = T(float(T(num / den)) * GATE[lh * D + d]);
