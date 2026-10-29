// GLM-5.3 EXL3 helpers both families share (mlx-stream native): the mcg codebook at multiplier 0xCBAC1FED, window 16;
// a trellis tile of 16 x 16 weights in 16 K halfwords (8 K u32 words, the stream read MSB first per little-endian word,
// weight t's codeword the 16 bits ending at bit K (t + 1), circular); weight t = 8 l + j of lane l sits at tensor-core
// row (2 (l & 3) + {0, 1, 8, 9}[j & 3]) and column (l >> 2) + 8 (j >> 2) of the tile.

constant constexpr float G53_HSC = 0.08838834764831845f;

// Two codewords to two weights: the f16 sum of the two halves of (cw * mult) & 0x8FFF8FFF ^ 0x3B603B60.
static inline half2 g53_mcg2(uint2 cw) {
  const uint2 r = ((cw * uint2(0xCBAC1FEDu)) & uint2(0x8FFF8FFFu)) ^ uint2(0x3B603B60u);
  const half4 h = as_type<half4>(r);
  return half2(h.x + h.y, h.z + h.w);
}
