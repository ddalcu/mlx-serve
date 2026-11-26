// Inkling 2: the router's picks for S rows. grid (E, S), threadgroup (E, 1): one thread per expert, a row in ONE
// threadgroup. SC [S, E] f32 are the router's logits, RB [E] f32 the selection bias. IDX [1, S, K] u32 = the K
// experts of the largest sigmoid(SC) + RB, by rank (ties to the lower index); WTS [1, S, K] f32 = their sigmoid(SC)
// over its sum, times PR[0].
threadgroup float key[E];
threadgroup float raw[K];
threadgroup uint sel[K];
const uint e = thread_position_in_grid.x;
const uint row = thread_position_in_grid.y;
const float s = 1.0f / (1.0f + exp(-SC[row * E + e]));
key[e] = s + RB[e];
// A NaN score ranks nowhere: every slot keeps an expert that exists.
if (e < K) { sel[e] = e; raw[e] = 0.0f; }
threadgroup_barrier(mem_flags::mem_threadgroup);
const float mine = key[e];
uint rank = 0;
for (uint f = 0; f < E; f++) rank += (key[f] > mine || (key[f] == mine && f < e)) ? 1u : 0u;
if (rank < K && mine == mine) { sel[rank] = e; raw[rank] = s; }
threadgroup_barrier(mem_flags::mem_threadgroup);
if (e < K) {
    float tot = 0.0f;
    for (uint k = 0; k < K; k++) tot += raw[k];
    IDX[row * K + e] = sel[e];
    WTS[row * K + e] = raw[e] / (tot + 1e-20f) * PR[0];
}
