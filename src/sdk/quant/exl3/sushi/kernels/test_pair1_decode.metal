const uint i = thread_position_in_grid.x;
const uint2 cw = uint2(codes[i], codes[65535u - i]);
pairs[i] = as_type<uint>(exl3_pairh(cw));
const float2 w = exl3_decode2(cw);
stored[i] = as_type<uint>(half2(half(exl3_out(w.x)), half(exl3_out(w.y))));
carried[i] = as_type<uint>(w.x);