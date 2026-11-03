
    constexpr int packs_per_thread = 2;
    constexpr int num_simdgroups = 2;
    constexpr int pack_factor = 4;                     // get_pack_factor<32, 8>()
    constexpr int bytes_per_pack = 4;                  // get_bytes_per_pack<32>()
    constexpr int group_size = 32;
    constexpr int values_per_thread = pack_factor * packs_per_thread;
    constexpr int block_size = values_per_thread * 32;
    constexpr int scale_step_per_thread = group_size / values_per_thread;
    static_assert(K % block_size == 0, "qmv_fast needs K % 256 == 0");
    static_assert(N % (num_simdgroups * R) == 0, "N must tile by 2R rows");
    typedef float U;
    const int in_vec_size = K;
    const uint3 tid = threadgroup_position_in_grid;
    const uint simd_gid = simdgroup_index_in_threadgroup;
    const uint simd_lid = thread_index_in_simdgroup;
    const device uint8_t* ws = (const device uint8_t*)w;
    const device uint8_t* sc = scales;
    const int vec0 = int(tid.x) * V;
    const device bfloat16_t* xp = x + vec0 * K;

    thread U x_thread[values_per_thread];
    thread U w_dq[R][values_per_thread];
    thread U s[R];
    thread U result[V][R];
    for (int v = 0; v < V; v++) {
      for (int row = 0; row < R; row++) {
        result[v][row] = 0;
      }
    }

    const int in_vec_size_w = in_vec_size * bytes_per_pack / pack_factor;
    const int in_vec_size_g = in_vec_size / group_size;
    const int out_row = tid.y * (num_simdgroups * R) + simd_gid * R;

    ws += out_row * in_vec_size_w + simd_lid * packs_per_thread * bytes_per_pack;
    sc += out_row * in_vec_size_g + simd_lid / scale_step_per_thread;
    xp += simd_lid * values_per_thread;

    for (int k = 0; k < in_vec_size; k += block_size) {
      for (int row = 0; row < R; row++) {
        auto wl = (const device uint8_t*)(ws + row * in_vec_size_w);
        const device auto* sl = sc + row * in_vec_size_g;
        s[row] = dsv41_dequantize_scale(sl[0]);
        for (int i = 0; i < values_per_thread; i++) {
          w_dq[row][i] = dsv41_dequantize8(wl[i]);
        }
      }
      for (int v = 0; v < V; v++) {
        if (vec0 + v < M) {
          for (int i = 0; i < values_per_thread; i++) {
            x_thread[i] = xp[v * K + i];
          }
          for (int row = 0; row < R; row++) {
            result[v][row] += dsv41_qdot8<U, values_per_thread>(w_dq[row], x_thread, s[row]);
          }
        }
      }

      ws += block_size * bytes_per_pack / pack_factor;
      sc += block_size / group_size;
      xp += block_size;
    }

    for (int v = 0; v < V; v++) {
      for (int row = 0; row < R; row++) {
        result[v][row] = simd_sum(result[v][row]);
        if (simd_lid == 0 && vec0 + v < M) {
          y[(vec0 + v) * N + out_row + row] = static_cast<bfloat16_t>(result[v][row]);
        }
      }
    }
