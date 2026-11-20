const uint t = thread_position_in_threadgroup.x;
const uint lane = thread_index_in_simdgroup;
const uint sg = simdgroup_index_in_threadgroup;
const int ob = int(threadgroup_position_in_grid.x) * NO;
threadgroup float red[SG * NO * R];
threadgroup float ssr[SG * R];
