// pf_route_scatter (MLX_SERVE_DSV41_PF_ROUTE_PACK): kernels/moe/prefill_route_scatter.metal's stable placement with
// each pair's routing weight put at its place too (w[inverse[i]] = wts[i]), the zero fill and put_along_axis the
// pack otherwise runs after it. Every place is written once: the same words.
threadgroup uint keys[256];
const uint lane = thread_index_in_threadgroup;
const uint block = threadgroup_position_in_grid.x;
const uint i = block * 256 + lane;
const bool live = i < uint(ids_shape[0]);
const uint key = live ? ids[i] : 0xffffffffu;
keys[lane] = key;
threadgroup_barrier(mem_flags::mem_threadgroup);
if (!live || key >= E) return;
uint rank = 0;
for (uint j = 0; j < lane; ++j) rank += uint(keys[j] == key);
const uint pos = offsets[block * E + key] + rank;
inverse[i] = pos;
sorted[pos] = key;
lhs[pos] = i / TOPK;
w[pos] = wts[i];
