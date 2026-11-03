{
    const uint tail = (uint(x_shape[0]) / 8) % (TILE_TOKENS / 8);
    if (tail == 0 || threadgroup_position_in_grid.y + 1 < threadgroups_per_grid.y) GG_TILE(4)
    else if (tail == 1) GG_TILE(1)
    else if (tail == 2) GG_TILE(2)
    else GG_TILE(3)
}
