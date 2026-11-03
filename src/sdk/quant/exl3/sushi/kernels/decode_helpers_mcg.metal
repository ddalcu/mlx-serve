  const uint2 r = ((cw * uint2(0xCBAC1FEDu)) & uint2(0x8FFF8FFFu)) ^ uint2(0x3B603B60u);
  const half4 h = as_type<half4>(r);
  return half2(h.x + h.y, h.z + h.w);