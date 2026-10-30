const int t = blockIdx.x;
const int d = threadIdx.x * 8;
float acc[8];
#pragma unroll
for (int i = 0; i < 8; ++i) acc[i] = 0.f;
for (int j = 0; j < TOPK; ++j) {
  const uint32_t r = inverse[t * TOPK + j];
  const __nv_bfloat16 sc = scores[t * TOPK + j];
  const uint4 v = *reinterpret_cast<const uint4*>(down + (long)r * H + d);
  const __nv_bfloat16* vb = reinterpret_cast<const __nv_bfloat16*>(&v);
  #pragma unroll
  for (int i = 0; i < 8; ++i) acc[i] = __bfloat162float(__float2bfloat16_rn(acc[i] + __bfloat162float(__hmul(vb[i], sc))));
}
uint4 o;
__nv_bfloat16* ob = reinterpret_cast<__nv_bfloat16*>(&o);
#pragma unroll
for (int i = 0; i < 8; ++i) ob[i] = __float2bfloat16_rn(acc[i]);
*reinterpret_cast<uint4*>(y + (long)t * H + d) = o;