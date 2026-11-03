inline float2 dsv41_bf2(uint w) { return float2(as_type<float>(w << 16), as_type<float>(w & 0xFFFF0000u)); }
