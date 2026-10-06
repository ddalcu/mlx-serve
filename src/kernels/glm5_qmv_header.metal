// Ported from oMLX (jundot/omlx) omlx/patches/mlx_vlm_glm5_next_compat/decode_kernels.py,
// oMLX 0.7.0 (Apache-2.0, see NOTICE): MLX's qmv_fast arithmetic for one-token MoE decode.

#include <metal_simdgroup>
#include <metal_stdlib>
using namespace metal;

template <int bits>
constexpr int glm_pack_factor() {
  return (bits == 3 || bits == 5) ? 8 : (bits == 6 ? 4 : 32 / bits);
}

template <int bits>
constexpr int glm_bytes_per_pack() {
  return ((bits & (bits - 1)) == 0) ? 4 : (bits == 5 ? 5 : 3);
}

// Verbatim copy of MLX quantized.h load_vector (U = float).
template <typename T, int values_per_thread, int bits>
inline float glm_load_vector(const device T* x, thread float* x_thread) {
  float sum = 0;
  if (bits == 4) {
    for (int i = 0; i < values_per_thread; i += 4) {
      sum += x[i] + x[i + 1] + x[i + 2] + x[i + 3];
      x_thread[i] = x[i];
      x_thread[i + 1] = x[i + 1] / 16.0f;
      x_thread[i + 2] = x[i + 2] / 256.0f;
      x_thread[i + 3] = x[i + 3] / 4096.0f;
    }
  } else if (bits == 5) {
    for (int i = 0; i < values_per_thread; i += 8) {
      sum += x[i] + x[i + 1] + x[i + 2] + x[i + 3] + x[i + 4] + x[i + 5] +
          x[i + 6] + x[i + 7];
      x_thread[i] = x[i];
      x_thread[i + 1] = x[i + 1] / 32.0f;
      x_thread[i + 2] = x[i + 2] / 4.0f;
      x_thread[i + 3] = x[i + 3] / 128.0f;
      x_thread[i + 4] = x[i + 4] / 16.0f;
      x_thread[i + 5] = x[i + 5] / 2.0f;
      x_thread[i + 6] = x[i + 6] / 64.0f;
      x_thread[i + 7] = x[i + 7] / 8.0f;
    }
  } else if (bits == 6) {
    for (int i = 0; i < values_per_thread; i += 4) {
      sum += x[i] + x[i + 1] + x[i + 2] + x[i + 3];
      x_thread[i] = x[i];
      x_thread[i + 1] = x[i + 1] / 64.0f;
      x_thread[i + 2] = x[i + 2] / 16.0f;
      x_thread[i + 3] = x[i + 3] / 4.0f;
    }
  } else if (bits == 8) {
    for (int i = 0; i < values_per_thread; i++) {
      sum += x[i];
      x_thread[i] = x[i];
    }
  }
  return sum;
}

// Verbatim copy of MLX quantized.h qdot (U = float).
template <int values_per_thread, int bits>
inline float glm_qdot(
    const device uint8_t* w,
    const thread float* x_thread,
    float scale,
    float bias,
    float sum) {
  float accum = 0;
  if (bits == 4) {
    const device uint16_t* ws = (const device uint16_t*)w;
    for (int i = 0; i < (values_per_thread / 4); i++) {
      accum +=
          (x_thread[4 * i] * (ws[i] & 0x000f) +
           x_thread[4 * i + 1] * (ws[i] & 0x00f0) +
           x_thread[4 * i + 2] * (ws[i] & 0x0f00) +
           x_thread[4 * i + 3] * (ws[i] & 0xf000));
    }
  } else if (bits == 5) {
    for (int i = 0; i < (values_per_thread / 8); i++) {
      x_thread += 8 * i;
      w += 5 * i;
      accum += (w[0] & 0x1f) * x_thread[0];
      accum += (w[0] & 0xe0) * x_thread[1];
      accum += (w[1] & 0x3) * (x_thread[1] * 256.0f);
      accum += (w[1] & 0x7c) * x_thread[2];
      accum += (w[1] & 0x80) * x_thread[3];
      accum += (w[2] & 0xf) * (x_thread[3] * 256.0f);
      accum += (w[2] & 0xf0) * x_thread[4];
      accum += (w[3] & 0x1) * (x_thread[4] * 256.0f);
      accum += (w[3] & 0x3e) * x_thread[5];
      accum += (w[3] & 0xc0) * x_thread[6];
      accum += (w[4] & 0x7) * (x_thread[6] * 256.0f);
      accum += (w[4] & 0xf8) * x_thread[7];
    }
  } else if (bits == 6) {
    for (int i = 0; i < (values_per_thread / 4); i++) {
      x_thread += 4 * i;
      w += 3 * i;
      accum += (w[0] & 0x3f) * x_thread[0];
      accum += (w[0] & 0xc0) * x_thread[1];
      accum += (w[1] & 0x0f) * (x_thread[1] * 256.0f);
      accum += (w[1] & 0xf0) * x_thread[2];
      accum += (w[2] & 0x03) * (x_thread[2] * 256.0f);
      accum += (w[2] & 0xfc) * x_thread[3];
    }
  } else if (bits == 8) {
    for (int i = 0; i < values_per_thread; i++) {
      accum += x_thread[i] * w[i];
    }
  }
  return scale * accum + sum * bias;
}

// qmv_fast_impl for RPS consecutive rows of one [N, K] affine matrix
// (row pointers already offset to the first row), one simdgroup.  Leaves the
// per-lane partial sums in `result`; the caller simd_sums them.
template <typename T, int K, int group_size, int bits, int RPS>
inline void glm_qmv_rows(
    const device uint8_t* ws,
    const device T* scales,
    const device T* biases,
    const device T* x,
    uint simd_lid,
    thread float* result) {
  constexpr int packs_per_thread = bits == 2 ? 1 : 2;
  constexpr int pack_factor = glm_pack_factor<bits>();
  constexpr int bytes_per_pack = glm_bytes_per_pack<bits>();
  constexpr int values_per_thread = pack_factor * packs_per_thread;
  constexpr int block_size = values_per_thread * 32;
  constexpr int scale_step_per_thread = group_size / values_per_thread;
  constexpr int in_vec_size_w = K * bytes_per_pack / pack_factor;
  constexpr int in_vec_size_g = K / group_size;

  thread float x_thread[values_per_thread];
  ws += simd_lid * packs_per_thread * bytes_per_pack;
  scales += simd_lid / scale_step_per_thread;
  biases += simd_lid / scale_step_per_thread;
  x += simd_lid * values_per_thread;

  for (int k = 0; k < K; k += block_size) {
    float sum = glm_load_vector<T, values_per_thread, bits>(x, x_thread);
    for (int row = 0; row < RPS; row++) {
      const device uint8_t* wl = ws + row * in_vec_size_w;
      const device T* sl = scales + row * in_vec_size_g;
      const device T* bl = biases + row * in_vec_size_g;
      float s = sl[0];
      float b = bl[0];
      result[row] += glm_qdot<values_per_thread, bits>(wl, x_thread, s, b, sum);
    }
    ws += block_size * bytes_per_pack / pack_factor;
    scales += block_size / group_size;
    biases += block_size / group_size;
    x += block_size;
  }
}

// Verbatim copy of MLX quantized.h dequantize (U = float) for 4/5/6/8 bits.
template <int N, int bits>
inline void glm_dequantize(const device uint8_t* w, float scale, float bias, thread float* w_local) {
  const float s = float(scale);
  const float b = float(bias);
  if (bits == 4) {
    float sc[2] = {s, s / 16.0f};
    for (int i = 0; i < (N / 2); i++) {
      w_local[2 * i] = static_cast<float>(sc[0] * (w[i] & 0x0f) + b);
      w_local[2 * i + 1] = static_cast<float>(sc[1] * (w[i] & 0xf0) + b);
    }
  } else if (bits == 5) {
    for (int i = 0; i < (N / 8); i++) {
      w_local += 8 * i;
      w += 5 * i;
      w_local[0] = static_cast<float>((w[0] & 0x1f) * s + b);
      w_local[1] =
          static_cast<float>((((w[0] & 0xe0) >> 5) + ((w[1] & 0x3) << 3)) * s + b);
      w_local[2] = static_cast<float>(((w[1] & 0x7c) >> 2) * s + b);
      w_local[3] =
          static_cast<float>((((w[1] & 0x80) >> 7) + ((w[2] & 0xf) << 1)) * s + b);
      w_local[4] =
          static_cast<float>((((w[2] & 0xf0) >> 4) + ((w[3] & 0x1) << 4)) * s + b);
      w_local[5] = static_cast<float>(((w[3] & 0x3e) >> 1) * s + b);
      w_local[6] =
          static_cast<float>((((w[3] & 0xc0) >> 6) + ((w[4] & 0x7) << 2)) * s + b);
      w_local[7] = static_cast<float>(((w[4] & 0xf8) >> 3) * s + b);
    }
  } else if (bits == 6) {
    for (int i = 0; i < (N / 4); i++) {
      w_local += 4 * i;
      w += 3 * i;
      w_local[0] = static_cast<float>((w[0] & 0x3f) * s + b);
      w_local[1] =
          static_cast<float>((((w[0] >> 6) & 0x03) + ((w[1] & 0x0f) << 2)) * s + b);
      w_local[2] =
          static_cast<float>((((w[1] >> 4) & 0x0f) + ((w[2] & 0x03) << 4)) * s + b);
      w_local[3] = static_cast<float>(((w[2] >> 2) & 0x3f) * s + b);
    }
  } else if (bits == 8) {
    for (int i = 0; i < N; i++) {
      w_local[i] = static_cast<float>(s * w[i] + b);
    }
  }
}

// MLX qmv_wide_impl (affine, k_lanes = 8) for one weight row and NV input
// vectors: each lane reduces groups k_lane, k_lane + 8, ... in 8-value
// sub-chunks; the caller applies the 4/2/1 shuffle-down ladder.
template <typename T, int K, int GS, int BITS, int NV>
inline void glm_qmv_wide_row(
    const device uint8_t* wrow,
    const device T* srow,
    const device T* brow,
    const device T* x,
    int nv,
    int k_lane,
    thread float* result) {
  constexpr int sub = 8;
  constexpr int G = K / GS;
  for (int g = k_lane; g < G; g += 8) {
    float scale = srow[g];
    float bias = brow[g];
    for (int sc = 0; sc < GS / sub; sc++) {
      const int k0 = g * GS + sc * sub;
      const device uint8_t* wc = wrow + k0 * BITS / 8;
      float w_dq[sub];
      glm_dequantize<sub, BITS>(wc, scale, bias, w_dq);
      for (int v = 0; v < NV; v++) {
        if (v < nv) {
          const device T* xc = x + v * K + k0;
          float acc = 0;
          for (int i = 0; i < sub; i++) {
            acc += static_cast<float>(xc[i]) * w_dq[i];
          }
          result[v] += acc;
        }
      }
    }
  }
}

// Same expressions as MLX's Sigmoid / Minimum / Maximum functors.
template <typename T>
inline T glm_sigmoid(T x) {
  auto y = 1 / (1 + metal::exp(metal::abs(x)));
  return (x < 0) ? y : 1 - y;
}
// MLX's Sigmoid as its precompiled kernels evaluate it: the release metallib
// is built with -fno-fast-math, so metal::exp is the precise exp there,
// while runtime-compiled kernels (custom kernels, compiled graphs, JIT
// builds) get the default one. See eager_sigmoid_precise().
template <typename T>
inline T glm_sigmoid_precise(T x) {
  auto y = 1 / (1 + metal::precise::exp(metal::abs(x)));
  return (x < 0) ? y : 1 - y;
}
template <typename T>
inline T glm_minimum(T x, T y) {
  if (metal::isnan(x)) {
    return x;
  }
  return x < y ? x : y;
}
template <typename T>
inline T glm_maximum(T x, T y) {
  if (metal::isnan(x)) {
    return x;
  }
  return x > y ? x : y;
}

// The router's top-k selection (the select kernel's loop, one simdgroup):
// argpartition order of the biased scores, i.e. descending values with ties
// to the lower expert index and NaNs last (lowest index first). Every lane
// ends with the same picked[].
template <int E, int TOPK, typename P>
inline void glm_router_topk(P bz, uint lane, thread int* picked) {
  constexpr int PER = (E + 31) / 32;
  float vals[PER];
  bool taken[PER];
  for (int j = 0; j < PER; j++) {
    const int e = j * 32 + int(lane);
    vals[j] = e < E ? bz[e] : -INFINITY;
    taken[j] = e >= E;
  }
  for (int r = 0; r < TOPK; r++) {
    float best = -INFINITY;
    int best_e = 0x7fffffff;
    for (int j = 0; j < PER; j++) {
      const int e = j * 32 + int(lane);
      if (!taken[j] && !isnan(vals[j]) &&
          (best_e == 0x7fffffff || vals[j] > best || (vals[j] == best && e < best_e))) {
        best = vals[j];
        best_e = e;
      }
    }
    for (ushort off = 16; off >= 1; off >>= 1) {
      float ob = simd_shuffle_xor(best, off);
      int oe = simd_shuffle_xor(best_e, off);
      const bool other_better = oe != 0x7fffffff &&
          (best_e == 0x7fffffff || ob > best || (ob == best && oe < best_e));
      if (other_better) {
        best = ob;
        best_e = oe;
      }
    }
    if (best_e == 0x7fffffff) {
      for (int j = 0; j < PER; j++) {
        const int e = j * 32 + int(lane);
        if (!taken[j] && e < best_e) {
          best_e = e;
        }
      }
      for (ushort off = 16; off >= 1; off >>= 1) {
        best_e = min(best_e, simd_shuffle_xor(best_e, off));
      }
    }
    picked[r] = best_e;
    if ((best_e % 32) == int(lane)) {
      taken[best_e / 32] = true;
    }
  }
}

// Glm5NextClampedSwiGLU / Glm5NextMLP epilogue on bfloat16 projections:
//   silu(minimum(gate, limit)) * minimum(maximum(up, -limit), limit)
template <typename T>
inline T glm_clamped_swiglu(T gate, T up, T limit, T neg_limit) {
  T g = glm_minimum(gate, limit);
  T s = g * glm_sigmoid(g);
  T u = glm_minimum(glm_maximum(up, neg_limit), limit);
  return s * u;
}
