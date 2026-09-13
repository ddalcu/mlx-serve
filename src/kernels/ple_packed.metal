const uint i = thread_position_in_grid.x;
if (ulong(i) >= ulong(slots_shape[0]) * 160) return;
const uint row = slots[i / 160], c = i % 160, g = c / 32;
const device uchar* p = packed + ulong(row) * 100;
const uint q = (reinterpret_cast<const device uint*>(p)[c / 8] >> (4 * (c % 8))) & 15u;
const float sc = as_type<float>(uint(reinterpret_cast<const device ushort*>(p + 80)[g]) << 16);
const float bi = as_type<float>(uint(reinterpret_cast<const device ushort*>(p + 90)[g]) << 16);
const uint b = as_type<uint>(float(q) * sc + bi);
out[i] = as_type<bfloat>(ushort((b + 0x7fffu + ((b >> 16) & 1u)) >> 16));
