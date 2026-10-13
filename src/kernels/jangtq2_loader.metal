// SPDX-License-Identifier: Apache-2.0
// Ported from vMLX (jjang-ai/vmlx) vmlx_engine/jangh/kernels.py _LOADER @ 5f007e6d.

// v2 tile loader: identical byte walk to MLX QuantizedBlockLoader (bitstream packing); decode = scale[row]*level(q)
template <typename T, short BROWS, short BCOLS, short dst_ld, short tgp_size, short bits>
struct TQBlockLoader {
  MLX_MTL_CONST short pack_factor = get_pack_factor<bits, 8>();
  MLX_MTL_CONST short bytes_per_pack = get_bytes_per_pack<bits>();
  MLX_MTL_CONST short BCOLS_PACKED = BCOLS / pack_factor;
  MLX_MTL_CONST short n_reads = (BCOLS_PACKED * BROWS < tgp_size) ? 1 : (BCOLS_PACKED * BROWS) / tgp_size;
  MLX_MTL_CONST short NCB = 1 << bits;
  const int src_ld; const int tile_stride; const short thread_idx; const short bi; const short bj;
  threadgroup T* dst; const device uint8_t* src; float s_row;
  TQBlockLoader(const device uint8_t* src_, const device half* scales_, const int src_ld_,
                threadgroup T* dst_, ushort simd_group_id, ushort simd_lane_id)
      : src_ld(src_ld_), tile_stride(BCOLS_PACKED * bytes_per_pack),
        thread_idx(simd_group_id * 32 + simd_lane_id),
        bi(n_reads * thread_idx / BCOLS_PACKED), bj((n_reads * thread_idx) % BCOLS_PACKED),
        dst(dst_ + bi * dst_ld + bj * pack_factor),
        src(src_ + bi * src_ld * bytes_per_pack / pack_factor + bj * bytes_per_pack) {
    s_row = (bi < BROWS) ? float(scales_[bi]) : 0.0f;
  }
  METAL_FUNC void decode_(short i) const {
    uint v = src[i * bytes_per_pack];
    if (bytes_per_pack > 1) v |= uint(src[i * bytes_per_pack + 1]) << 8;
    if (bytes_per_pack > 2) v |= uint(src[i * bytes_per_pack + 2]) << 16;
    for (short j = 0; j < pack_factor; j++)
      dst[i * pack_factor + j] = T(s_row * tq_level<bits>((v >> (j * bits)) & (NCB - 1)));
  }
  void load_unsafe() const {
    if (BCOLS_PACKED * BROWS < tgp_size && bi >= BROWS) return;
    for (short i = 0; i < n_reads; i++) decode_(i);
  }
  void load_safe(short2 src_tile_dim) const {   // K % BK == 0 is enforced host-side; only rows can be ragged
    if (BCOLS_PACKED * BROWS < tgp_size && bi >= BROWS) return;
    if (bi >= src_tile_dim.y) { for (short i = 0; i < n_reads * pack_factor; i++) dst[i] = T(0); return; }
    for (short i = 0; i < n_reads; i++) decode_(i);
  }
  void next() { src += tile_stride; }
};
