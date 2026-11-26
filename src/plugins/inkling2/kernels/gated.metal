// Inkling 2, one row: the attention read ATT [H * HD] (T) times its output gate GATE [H * HD] f32, rounded to T.
const uint e = thread_position_in_grid.x;
OUT[e] = T(float(ATT[e]) * GATE[e]);
