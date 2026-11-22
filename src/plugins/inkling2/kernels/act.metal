// Inkling 2, one row: the routed experts' soft-capped SwiGLU, one thread per element. GU [KSEL, 1, 2 * I] (TI) holds
// each picked expert's gate (even) and up (odd) outputs; with SCALED each is times SC[IDX[k]] first (NVFP4's
// per-expert scale, f32 [E]). PR = { 1 / gate cap, gate cap, 1 / up cap, up cap }.
// ACT [KSEL, 1, I] (T) = cap_g tanh(g / cap_g) * cap_u tanh(u / cap_u) * sigmoid(g).
const uint e = thread_position_in_grid.x;
const uint I = GU_shape[2] / 2;
const uint k = e / I, i = e % I;
const float sc = SCALED ? SC[IDX[k]] : 1.0f;
const float g = float(GU[k * 2 * I + 2 * i]) * sc;
const float u = float(GU[k * 2 * I + 2 * i + 1]) * sc;
// The precise variants: the fast tanh is NaN at an infinity and a percent off near zero.
const float capped = (precise::tanh(g * PR[0]) * PR[1]) * (precise::tanh(u * PR[2]) * PR[3]);
ACT[e] = T(capped / (1.0f + precise::exp(-g)));
