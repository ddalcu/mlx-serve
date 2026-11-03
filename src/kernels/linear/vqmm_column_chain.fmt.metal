  {{
    float sj = s{d}; float bj = b{d};
    if constexpr (BITS == 4) {{
      for (int ki = 0; ki < 8; ++ki) {{
        float wv = float((p{d} >> (ki * 4)) & 0xFu) * sj + bj;
{s}      }}
    }} else if constexpr (BITS == 8) {{
      for (int ki = 0; ki < 8; ++ki) {{
        float wv = float(uint32_t((r{d} >> (ki * 8)) & 0xFFul)) * sj + bj;
{s}      }}
    }} else if constexpr (BITS == 5) {{
      for (int ki = 0; ki < 8; ++ki) {{
        float wv = float(uint32_t((r{d} >> (ki * 5)) & 0x1Ful)) * sj + bj;
{s}      }}
    }} else {{
      for (int ki = 0; ki < 4; ++ki) {{
        float wv = float((a{d}0 >> (ki * 6)) & 0x3Fu) * sj + bj;
{s}      }}
      for (int ki = 0; ki < 4; ++ki) {{
        float wv = float((a{d}1 >> (ki * 6)) & 0x3Fu) * sj + bj;
{s}      }}
    }}
  }}
