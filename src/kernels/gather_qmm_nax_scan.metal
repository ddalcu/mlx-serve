// SPDX-License-Identifier: Apache-2.0
// Ported from oMLX (jundot/omlx) omlx/patches/m5_gather_qmm_nax.py @ d6b2b92.
    threadgroup uint32_t run_start[MAXE];
    threadgroup uint32_t run_end[MAXE];
    threadgroup uint32_t simd_tot[32];
    msv_gqmm_tile_scan<BM>(
        idx, params, tiles, tile_count, run_start, run_end, simd_tot,
        thread_index_in_threadgroup, threads_per_threadgroup.x,
        simdgroup_index_in_threadgroup, thread_index_in_simdgroup);
