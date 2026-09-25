// int8xint8 NAX GEMM with per-row (per-token / per-output-channel) scales:
//   y[m,n] = (sum_k xq[m,k]*wq[k/32,n,k%32]) * sx[m] * sw[n]
// One threadgroup owns 128x128 of y; 8 simdgroups, each 32x64 (TM=2 x TN=2) over
// the 16x32x32 int8 NAX tile, K consumed in 32-wide steps.
//
// xq_shape = [M, K], xq is row-major. wq_shape = [K/32, N, 32], wq is
// K-block-major with no N or K padding. Both payloads are physically row-major
// within their declared shapes.
//
// The int8 fragment layout is NOT the bf16 one: for this descriptor each lane
// element group of four is four CONTIGUOUS k (kb = (lane&1)*4 + ((lane>>3)&1)*8)
// and the fragment row is q = ((lane>>1)&3) + ((lane>>4)&1)*4. That is what makes
// the uchar4 loads below work; a bf16-shaped fragment load would be wrong here.
#include <metal_stdlib>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using bfloat16_t = bfloat;

#define W8A8_BM 128
#define W8A8_BN 128
#define W8A8_TM 2
