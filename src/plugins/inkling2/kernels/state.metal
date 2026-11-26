// Inkling 2: a visit's new conv state NST [1, KT - 1, SW] f32 after S rows, one thread per element: the last KT - 1
// rows of the old state ST followed by the rows' conv inputs [KP | VP | OP | XM] (the k, v and output projections
// and the MLP's output, [1, S, .]: C, C, D and D wide). The S rows are slots of RPS rows, each slot's state
// (ST, NST [S / RPS, KT - 1, SW]) moving on by its rows (RPS == S: one sequence; 1: every row a sequence of its own).
const uint e = thread_position_in_grid.x;
const uint SW = ST_shape[2];
const uint S = KP_shape[1];
const uint RPS = S / ST_shape[0];
const uint C = KP_shape[2];
const uint D = XM_shape[2];
const uint col = e % SW;
// Row `at` of [the slot's ST | its rows].
const uint slot = e / ((KT - 1) * SW);
const uint at = RPS + (e / SW) % (KT - 1);
if (at < KT - 1) {
    NST[e] = ST[(slot * (KT - 1) + at) * SW + col];
    return;
}
const uint r = slot * RPS + at - (KT - 1);
if (col < C) NST[e] = float(KP[r * C + col]);
else if (col < 2 * C) NST[e] = float(VP[r * C + col - C]);
else if (col < SW - D) NST[e] = float(OP[r * D + col - 2 * C]);
else NST[e] = float(XM[r * D + col - (SW - D)]);
