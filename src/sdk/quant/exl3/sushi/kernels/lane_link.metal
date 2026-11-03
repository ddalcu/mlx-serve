const uint i = thread_position_in_grid.x;
if (i >= uint(LEN)) return;
out[i] = a[i];
if (i == 0u && float(dep[0]) == 1.5e30f) out[0] = a[1];