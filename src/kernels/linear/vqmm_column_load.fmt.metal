  uint32_t p{d}; ulong r{d}; uint32_t a{d}0; uint32_t a{d}1;
  if constexpr (BITS == 4) {{ p{d} = w_q[(n0 + {d}) * K_by_p + pack]; }}
  else {{
    const device uchar* wp = ((const device uchar*)w_q) + (n0 + {d}) * K_bytes + pack * BITS;
    if constexpr (BITS == 8) {{
      const device uint* wu = (const device uint*)wp;
      r{d} = ulong(wu[0]) | (ulong(wu[1]) << 32);
    }} else if constexpr (BITS == 5) {{
      r{d} = ulong(wp[0]) | (ulong(wp[1]) << 8) | (ulong(wp[2]) << 16) |
             (ulong(wp[3]) << 24) | (ulong(wp[4]) << 32);
    }} else {{
      const device ushort* wh = (const device ushort*)wp;
      uint32_t h0 = uint32_t(wh[0]); uint32_t h1 = uint32_t(wh[1]); uint32_t h2 = uint32_t(wh[2]);
      a{d}0 = h0 | ((h1 & 0xFFu) << 16);
      a{d}1 = (h1 >> 8) | (h2 << 8);
    }}
  }}
