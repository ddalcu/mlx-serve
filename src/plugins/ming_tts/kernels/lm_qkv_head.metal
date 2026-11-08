const int tg = int(threadgroup_position_in_grid.x), qtg = 896 / NO, ktg = 128 / NO;
const int which = tg < qtg ? 0 : (tg < qtg + ktg ? 1 : 2);
const int loc = tg - (which == 0 ? 0 : (which == 1 ? qtg : qtg + ktg));
const device bfloat16_t* W = which == 0 ? WQ : (which == 1 ? WK : WV);
const device bfloat16_t* Bv = which == 0 ? BQ : (which == 1 ? BK : BV);
#define COL(n) (which == 2 ? loc * NO + (n) : ((loc * (NO / 2) + (n) / 2) / 32) * 64 + (loc * (NO / 2) + (n) / 2) % 32 + 32 * ((n) % 2))
