// Bank-aware cooperative GEMV prelude: ids[slot] = bank << 24 | slot row; the row reads its own bank's trellis.
const uint banked_id = uint(ids[threadgroup_position_in_grid.y]);
const uint bank_of = banked_id >> 24u;
const auto trellis = bank_of == 0u ? t0 : (bank_of == 1u ? t1 : t2);
