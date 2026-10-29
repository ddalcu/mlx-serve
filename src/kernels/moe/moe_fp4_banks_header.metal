// Slot banks of a streamed bank: pair e reads bank kinds[e] (at row inds[e]); every weight read of the fp4 decode
// bodies resolves through it, so a pair's arithmetic is the one-bank kernel's.
#define wg_q (kinds[e] == 0 ? wg_q0 : kinds[e] == 1 ? wg_q1 : wg_q2)
#define g_scales (kinds[e] == 0 ? g_scales0 : kinds[e] == 1 ? g_scales1 : g_scales2)
#define wu_q (kinds[e] == 0 ? wu_q0 : kinds[e] == 1 ? wu_q1 : wu_q2)
#define u_scales (kinds[e] == 0 ? u_scales0 : kinds[e] == 1 ? u_scales1 : u_scales2)
#define wd_q (kinds[e] == 0 ? wd_q0 : kinds[e] == 1 ? wd_q1 : wd_q2)
#define d_scales (kinds[e] == 0 ? d_scales0 : kinds[e] == 1 ? d_scales1 : d_scales2)
