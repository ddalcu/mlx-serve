// The indexer tail rows of a forward's DSA layers copied into one buffer of their own (`dsa_append.zig`,
// MLX_SERVE_GLM_DSA_TAILS): input t<l> (layer l's last R cache rows, ROW_W wide) to out [N, R, ROW_W], one thread an
// element. GLM5_TAILS_N (1: the MTP head's layer, or 11: the trunk's) is defined before this text.
const uint c = thread_position_in_grid.x;
const uint r = thread_position_in_grid.y;
const uint l = thread_position_in_grid.z;
const uint R = threads_per_grid.y;
const size_t i = size_t(r) * ROW_W + c;
T v;
switch (l) {
case 0: v = t0[i]; break;
#if GLM5_TAILS_N == 11
case 1: v = t1[i]; break;
case 2: v = t2[i]; break;
case 3: v = t3[i]; break;
case 4: v = t4[i]; break;
case 5: v = t5[i]; break;
case 6: v = t6[i]; break;
case 7: v = t7[i]; break;
case 8: v = t8[i]; break;
case 9: v = t9[i]; break;
case 10: v = t10[i]; break;
#endif
default: return;
}
out[(size_t(l) * R + r) * ROW_W + c] = v;
