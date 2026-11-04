__device__ __forceinline__ uint32_t msv_pack2(float a, float b, __nv_bfloat16*) {
  __nv_bfloat162 v = __floats2bfloat162_rn(a, b);
  return *reinterpret_cast<uint32_t*>(&v);
}
__device__ __forceinline__ uint32_t msv_pack2(float a, float b, __half*) {
  __half2 v = __floats2half2_rn(a, b);
  return *reinterpret_cast<uint32_t*>(&v);
}
// A view's data offset can leave x or w off a 16-byte boundary: those calls read bytewise.
__device__ __forceinline__ uint4 msv_ld16(const void* p, bool aligned) {
  if (aligned) return *reinterpret_cast<const uint4*>(p);
  uint4 v;
  memcpy(&v, p, 16);
  return v;
}
__device__ __forceinline__ void msv_mma(float* c, const uint32_t* a, const uint32_t* b, __nv_bfloat16*) {
  asm("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}
__device__ __forceinline__ void msv_mma(float* c, const uint32_t* a, const uint32_t* b, __half*) {
  asm("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}
