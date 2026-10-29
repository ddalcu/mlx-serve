    const uint l = thread_index_in_simdgroup;
    const int cb = int(threadgroup_position_in_grid.x * 6u + simdgroup_index_in_threadgroup);
    const int r = int(threadgroup_position_in_grid.y);
    dig2_swiglu_one(z0, z1, hd, rout0, rout1, rin2, tbl[rhs[r]], r, cb, l);
