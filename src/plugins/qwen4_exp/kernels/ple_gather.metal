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
if (params[86] != 0) {
  // NVFP4: e2m1 code (low nibble first) times its e4m3 group scale times the global scale, in
  // NgramTable.dequantNvfp4Row's order. Bit placement into half scales them by 2^-14 and 2^-8.
  const uint b = uint(table[ulong(params[10]) + row * ulong(params[8]) * 4 + ulong(i / 2)]);
  const uint c = (b >> uint((i % 2) * 4)) & 0xFu;
  const float e2m1 = float(as_type<half>(ushort(((c & 0x7u) << 9) | ((c & 0x8u) << 12)))) * 16384.0f;
  const uint sb = uint(table[ulong(params[11]) + row * ulong(params[9]) + ulong(i / 16)]);
  const float e4m3 = float(as_type<half>(ushort(((sb & 0x7Fu) << 7) | ((sb & 0x80u) << 8)))) * 256.0f;
  const float sc = e4m3 * as_type<float>(uint(params[85]));
  const uint u = as_type<uint>(e2m1 * sc);
  q16 = ushort((u + 0x7FFFu + ((u >> 16) & 1u)) >> 16);
} else if (bits == 16) {
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
  // The global weight scale applies after, as in NgramTable.dequantRow.
  const float ws = as_type<float>(uint(params[85]));
  const uint u = as_type<uint>((float(q) * scale + bias) * ws);
  q16 = ushort((u + 0x7FFFu + ((u >> 16) & 1u)) >> 16);
}
out[idx] = as_type<bfloat16_t>(q16);
