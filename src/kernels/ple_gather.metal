// One thread per output element (token t, head h, column i). `params` layout: see ple_gpu.zig.
const ulong idx = thread_position_in_grid.x;
const long S = params[0];
const long NH = params[1];
const long DIM = params[5];
if (long(idx) >= S * NH * DIM) return;
const long i = long(idx) % DIM;
const long h = (long(idx) / DIM) % NH;
const long ctx = params[3] - 1;
const uint eos = uint(params[4]);
const long c = long(idx) / (DIM * NH) + ctx;

// NgramHash.rowIds: a shifted token reads as eos once the shift crosses an eos.
ulong mixed = ulong(ple_tok(prev, ids, ctx, c)) * ulong(params[13]);
bool cut = false;
const long order = 2 + h / params[2];
for (long pos = 1; pos < order; pos++) {
  const uint tok = ple_tok(prev, ids, ctx, c - pos);
  cut = cut || tok == eos;
  mixed ^= ulong(cut ? eos : tok) * ulong(params[13 + pos]);
}
const long vocab = params[21 + h];
long r = long(mixed) % vocab;
if (r < 0) r += vocab;
const ulong row = ulong(r + params[53 + h]);

const long bits = params[6];
ushort q16;
if (bits == 16) {
  q16 = ple_u16(table, ulong(params[10]) + (row * ulong(DIM) + ulong(i)) * 2);
} else {
  const ulong off = ulong(i * bits);
  const ulong word = ulong(params[10]) + row * ulong(params[8]) * 4 + (off / 32) * 4;
  const ulong v = ple_word(table, word, off % 32 + ulong(bits) > 32 ? 8 : 4);
  const uint q = uint(v >> (off % 32)) & ((1u << uint(bits)) - 1u);
  const ulong g = row * ulong(params[9]) + ulong(i / params[7]);
  const float scale = as_type<float>(uint(ple_u16(table, ulong(params[11]) + g * 2)) << 16);
  const float bias = as_type<float>(uint(ple_u16(table, ulong(params[12]) + g * 2)) << 16);
  // q < 2^8 times an 8-bit-mantissa scale is exact, so FMA contraction cannot move a bit.
  const uint u = as_type<uint>(float(q) * scale + bias);
  q16 = ushort((u + 0x7FFFu + ((u >> 16) & 1u)) >> 16);
}
out[idx] = as_type<bfloat16_t>(q16);
