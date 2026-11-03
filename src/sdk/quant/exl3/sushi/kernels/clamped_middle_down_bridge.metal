prepared[base + lane] = half((s0 + s2) * sc);
prepared[base + lane + 32u] = half((s1 + s3) * sc);
prepared[base + lane + 64u] = half((s0 - s2) * sc);
prepared[base + lane + 96u] = half((s1 - s3) * sc);
 }
}
threadgroup_barrier(mem_flags::mem_threadgroup);
const threadgroup half* x = prepared;
for (uint output_tile = 0; output_tile < uint(OTPT); ++output_tile) {
 uint ot = uint(threadgroup_position_in_grid.x) * uint(OTPT) + output_tile;
