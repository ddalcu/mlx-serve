// gemv_twin (decode/gemv_twin.zig): MLX's GEMVKernel<T, BM, BN, SM, SN, TM, TN>::run (gemv.h, MLX 64ea011), NR rows of x a threadgroup.
const int in_vec_size = int(x_shape[1]);
const int out_vec_size = int(w_shape[0]);
const int matrix_ld = in_vec_size;
const int n_rows = int(x_shape[0]);
const uint3 tid = threadgroup_position_in_grid;
const uint simd_gid = simdgroup_index_in_threadgroup;
const uint simd_lid = thread_index_in_simdgroup;

constexpr int threadsM = BM * SM;
constexpr int threadsN = BN * SN;
constexpr int blockM = threadsM * TM;
constexpr int blockN = threadsN * TN;
constexpr int tgp_mem_size = BN > 1 ? BN * (blockM + TM) : 0;
constexpr bool needs_tgp_reduction = BN > 1;
threadgroup float tgp_memory[NR * (tgp_mem_size > 0 ? tgp_mem_size : 1)];

// This threadgroup's output tile and rows: NR rows from row r0 of its chunk (grid z the chunks; ILV: the chunks
// interleaved with the tiles along x, a tile's chunks adjacent), nr of them in x
const int n_chunks = (n_rows + NR - 1) / NR;
const int chunk = ILV != 0 ? int(tid.x) % n_chunks : int(tid.z);
const int tile = ILV != 0 ? int(tid.x) / n_chunks : int(tid.x);
const int r0 = chunk * NR;
const int nr = metal::min(NR, n_rows - r0);
const device T* mat = w;
const device T* in_vec = x + size_t(r0) * size_t(in_vec_size);
device T* out_vec = y + size_t(r0) * size_t(out_vec_size);

// Thread local accumulation results, a row's own
thread float result[NR][TM];
_Pragma("clang loop unroll(full)")
for (int r = 0; r < NR; r++) {
  _Pragma("clang loop unroll(full)")
  for (int tm = 0; tm < TM; tm++) {
    result[r][tm] = 0;
  }
}
thread T inter[TN];
thread float v_coeff[NR][TN];

const int thrM = SN != 32 ? int(simd_lid) / SN : 0;
const int thrN = SN != 32 ? int(simd_lid) % SN : int(simd_lid);

const int sgN = BN != 1 ? (int(simd_gid) % BN) : 0;

const int simdM = BN != 1 ? SM * (int(simd_gid) / BN) : int(SM * simd_gid);
const int simdN = BN != 1 ? SN * (int(simd_gid) % BN) : 0;

int bm = (simdM + thrM) * TM;
int bn = (simdN + thrN) * TN;

// Block position
int out_row = tile * blockM + bm;

// Exit simdgroup if rows out of bound
if (out_row >= out_vec_size)
  return;

// Adjust tail simdgroup to ensure in bound reads
out_row = out_row + TM <= out_vec_size ? out_row : out_vec_size - TM;

// Advance matrix
mat += size_t(out_row) * size_t(matrix_ld);

const int loop_stride = blockN;
const int in_size = in_vec_size;
const int n_iter = in_size / loop_stride;
const int last_iter = loop_stride * n_iter;
const int leftover = in_size - last_iter;

// Loop over in_vec in blocks of blockN
for (int i = 0; i < n_iter; ++i) {
  // load_unsafe<AccT>(in_vec, v_coeff, bn), each row's
  _Pragma("clang loop unroll(full)")
  for (int r = 0; r < NR; r++) {
    if (r < nr) {
      const device T* vr = in_vec + size_t(r) * size_t(in_vec_size);
      _Pragma("clang loop unroll(full)")
      for (int tn = 0; tn < TN; tn++) {
        v_coeff[r][tn] = static_cast<float>(vr[bn + tn]);
      }
    }
  }

  // Per thread work loop
  int mat_offset = 0;
  _Pragma("clang loop unroll(full)")
  for (int tm = 0; tm < TM; tm++) {
    // Load for the row: load_unsafe(mat, inter, mat_offset + bn)
    _Pragma("clang loop unroll(full)")
    for (int tn = 0; tn < TN; tn++) {
      inter[tn] = static_cast<T>(mat[mat_offset + bn + tn]);
    }

    // Accumulate results, each row's into its own
    _Pragma("clang loop unroll(full)")
    for (int r = 0; r < NR; r++) {
      if (r < nr) {
        _Pragma("clang loop unroll(full)")
        for (int tn = 0; tn < TN; tn++) {
          result[r][tm] += inter[tn] * v_coeff[r][tn];
        }
      }
    }

    mat_offset += matrix_ld;
  }

  bn += blockN;
}

if (leftover > 0) {
  // load_safe<AccT>(in_vec, v_coeff, bn, in_size), each row's
  _Pragma("clang loop unroll(full)")
  for (int r = 0; r < NR; r++) {
    if (r < nr) {
      const device T* vr = in_vec + size_t(r) * size_t(in_vec_size);
      if (bn + TN <= in_size) {
        _Pragma("clang loop unroll(full)")
        for (int tn = 0; tn < TN; tn++) {
          v_coeff[r][tn] = static_cast<float>(vr[bn + tn]);
        }
      } else {
        _Pragma("clang loop unroll(full)")
        for (int tn = 0; tn < TN; tn++) {
          v_coeff[r][tn] = bn + tn < in_size ? static_cast<float>(vr[bn + tn]) : float(0);
        }
      }
    }
  }

  // Per thread work loop
  _Pragma("clang loop unroll(full)")
  for (int tm = 0; tm < TM; tm++) {
    // Load for the row: load_safe(&mat[tm * matrix_ld], inter, bn, in_size)
    const device T* src = &mat[tm * matrix_ld];
    if (bn + TN <= in_size) {
      _Pragma("clang loop unroll(full)")
      for (int tn = 0; tn < TN; tn++) {
        inter[tn] = static_cast<T>(src[bn + tn]);
      }
    } else {
      _Pragma("clang loop unroll(full)")
      for (int tn = 0; tn < TN; tn++) {
        inter[tn] = bn + tn < in_size ? static_cast<T>(src[bn + tn]) : T(0);
      }
    }

    // Accumulate results, each row's into its own
    _Pragma("clang loop unroll(full)")
    for (int r = 0; r < NR; r++) {
      if (r < nr) {
        _Pragma("clang loop unroll(full)")
        for (int tn = 0; tn < TN; tn++) {
          result[r][tm] += inter[tn] * v_coeff[r][tn];
        }
      }
    }
  }
}

// Simdgroup accumulations, each row's
_Pragma("clang loop unroll(full)")
for (int r = 0; r < NR; r++) {
  _Pragma("clang loop unroll(full)")
  for (int tm = 0; tm < TM; tm++) {
    _Pragma("clang loop unroll(full)")
    for (ushort sn = (SN / 2); sn >= 1; sn >>= 1) {
      result[r][tm] += simd_shuffle_down(result[r][tm], sn);
    }
  }
}

// Threadgroup accumulation results, each row's in its own block of the threadgroup memory
if (needs_tgp_reduction) {
  if (thrN == 0) {
    _Pragma("clang loop unroll(full)")
    for (int r = 0; r < NR; r++) {
      threadgroup float* tgp_results = tgp_memory + r * tgp_mem_size + sgN * (blockM + TM) + bm;
      _Pragma("clang loop unroll(full)")
      for (int tm = 0; tm < TM; tm++) {
        tgp_results[tm] = result[r][tm];
      }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (sgN == 0) {
      _Pragma("clang loop unroll(full)")
      for (int r = 0; r < NR; r++) {
        threadgroup float* tgp_results = tgp_memory + r * tgp_mem_size + sgN * (blockM + TM) + bm;
        _Pragma("clang loop unroll(full)")
        for (int sgn = 1; sgn < BN; sgn++) {
          _Pragma("clang loop unroll(full)")
          for (int tm = 0; tm < TM; tm++) {
            result[r][tm] += tgp_results[sgn * (blockM + TM) + tm];
          }
        }
      }
    }
  }
}

// Write outputs, each row's
if (simdN == 0 && thrN == 0) {
  _Pragma("clang loop unroll(full)")
  for (int r = 0; r < NR; r++) {
    if (r < nr) {
      _Pragma("clang loop unroll(full)")
      for (int tm = 0; tm < TM; tm++) {
        out_vec[size_t(r) * size_t(out_vec_size) + out_row + tm] = static_cast<T>(result[r][tm]);
      }
    }
  }
}
