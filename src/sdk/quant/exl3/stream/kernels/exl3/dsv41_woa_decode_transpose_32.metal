
threadgroup T tile[32 * 33];
uint column = thread_position_in_threadgroup.x;
uint row = thread_position_in_threadgroup.y;
uint k_base = threadgroup_position_in_grid.x * 32;
uint r_base = threadgroup_position_in_grid.y * 32;
uint group = threadgroup_position_in_grid.z;
for (uint j = 0; j != 32; j += 8) {
    uint r = group * 1024 + r_base + row + j;
    uint k = k_base + column;
    uint code = (packed[r * 1024 + k / 4] >> ((k % 4) * 8)) & 255u;
    uint exponent = uint(scales[r * 128 + k_base / 32]);
    uint scale_word = exponent ? (exponent << 23) : 0x00400000u;
    float scale = as_type<float>(scale_word);
    tile[(row + j) * 33 + column] = T(scale * unpack_e4m3(code));
}
threadgroup_barrier(mem_flags::mem_threadgroup);
for (uint j = 0; j != 32; j += 8) {
    uint k = group * 4096 + k_base + row + j;
    out[k * 1024 + r_base + column] = tile[column * 33 + row + j];
}
