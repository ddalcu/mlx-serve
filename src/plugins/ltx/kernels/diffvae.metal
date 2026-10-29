const int Nt = q_shape[1];
const int Nh = q_shape[2];
const int Nw = q_shape[3];
const int NH = q_shape[4];
const int w = int(thread_position_in_grid.x);
const int h = int(thread_position_in_grid.y);
const int z = int(thread_position_in_grid.z);
const int t = z / NH;
const int nh = z - t * NH;
if (w >= Nw || h >= Nh || t >= Nt) return;

const int kt = KT < Nt ? KT : Nt;
const int kh = KH < Nh ? KH : Nh;
const int kw = KW < Nw ? KW : Nw;
const int t0 = ltx_na_start(Nt, KT, t);
const int h0 = ltx_na_start(Nh, KH, h);
const int w0 = ltx_na_start(Nw, KW, w);

const size_t w_stride = (size_t)NH * HD;
const size_t qoff = ((((size_t)t * Nh + h) * Nw + w) * NH + nh) * HD;

T qv[HD];
for (int d = 0; d < HD; ++d) qv[d] = q[qoff + d];
float acc[HD];
for (int d = 0; d < HD; ++d) acc[d] = 0.0f;
// Finite sentinel, not -INFINITY: the first key's correction is then a
// plain exp() that underflows to 0 instead of an inf-minus-inf NaN.
float m = -1e30f;
float lsum = 0.0f;

for (int a = 0; a < kt; ++a) {
    for (int b = 0; b < kh; ++b) {
        size_t kv = ((((size_t)(t0 + a) * Nh + (h0 + b)) * Nw + w0) * NH + nh) * HD;
        for (int c = 0; c < kw; ++c, kv += w_stride) {
            float sc = 0.0f;
            for (int d = 0; d < HD; ++d) sc += float(qv[d]) * float(k[kv + d]);
            float p;
            if (sc > m) {
                const float corr = metal::precise::exp(m - sc);
                m = sc;
                lsum *= corr;
                for (int d = 0; d < HD; ++d) acc[d] *= corr;
                p = 1.0f;
            } else {
                p = metal::precise::exp(sc - m);
            }
            lsum += p;
            for (int d = 0; d < HD; ++d) acc[d] += p * float(v[kv + d]);
        }
    }
}

const float inv = 1.0f / lsum;
for (int d = 0; d < HD; ++d) out[qoff + d] = static_cast<T>(acc[d] * inv);