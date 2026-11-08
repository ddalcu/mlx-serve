const int which = ob / 1024, col0 = ob % 1024;
const device bfloat16_t* W = which == 0 ? WQ : (which == 1 ? WK : WV);
const device bfloat16_t* Bv = which == 0 ? BQ : (which == 1 ? BK : BV);
