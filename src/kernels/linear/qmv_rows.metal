using U = float;
auto simd_lid = thread_position_in_threadgroup.x;
auto simd_gid = thread_position_in_threadgroup.y;
auto tg = threadgroup_position_in_grid;
const int in_vec_size = int(K_size);
const int out_vec_size = int(N_size);
constexpr int packs_per_thread = 2;
constexpr int pack_factor = (BITS == 3 || BITS == 5) ? 8 : (BITS == 6 ? 4 : 32 / BITS);
constexpr int bytes_per_pack = (BITS == 3 || BITS == 6) ? 3 : (BITS == 5 ? 5 : 4);
constexpr int values_per_thread = pack_factor * packs_per_thread;
constexpr int block_size = values_per_thread * 32;
constexpr int scale_step_per_thread = GS / values_per_thread;
const device uint8_t* ws = (const device uint8_t*)w_q;
const int in_vec_size_w = in_vec_size * bytes_per_pack / pack_factor;
const int in_vec_size_g = in_vec_size / GS;
const int out_row = int(tg.y) * 8 + int(simd_gid) * 4;
ws += out_row * in_vec_size_w + int(simd_lid) * packs_per_thread * bytes_per_pack;
const device T* scales_p = scales + out_row * in_vec_size_g + int(simd_lid) / scale_step_per_thread;
const device T* biases_p = biases + out_row * in_vec_size_g + int(simd_lid) / scale_step_per_thread;
const device T* xp = x + int(tg.z) * in_vec_size + int(simd_lid) * values_per_thread;
device T* yp = y + int(tg.z) * out_vec_size + out_row;
thread U x_thread[values_per_thread];
thread U result[4] = {0, 0, 0, 0};
for (int k = 0; k < in_vec_size; k += block_size) {
  U sum = 0;
  if (BITS == 3) {
    for (int i = 0; i < values_per_thread; i += 8) {
      sum += xp[i] + xp[i + 1] + xp[i + 2] + xp[i + 3] + xp[i + 4] + xp[i + 5] + xp[i + 6] + xp[i + 7];
      x_thread[i] = xp[i];
      x_thread[i + 1] = xp[i + 1] / 8.0f;
      x_thread[i + 2] = xp[i + 2] / 64.0f;
      x_thread[i + 3] = xp[i + 3] / 2.0f;
      x_thread[i + 4] = xp[i + 4] / 16.0f;
      x_thread[i + 5] = xp[i + 5] / 128.0f;
      x_thread[i + 6] = xp[i + 6] / 4.0f;
      x_thread[i + 7] = xp[i + 7] / 32.0f;
    }
  } else if (BITS == 4) {
    for (int i = 0; i < values_per_thread; i += 4) {
      sum += xp[i] + xp[i + 1] + xp[i + 2] + xp[i + 3];
      x_thread[i] = xp[i];
      x_thread[i + 1] = xp[i + 1] / 16.0f;
      x_thread[i + 2] = xp[i + 2] / 256.0f;
      x_thread[i + 3] = xp[i + 3] / 4096.0f;
    }
  } else if (BITS == 5) {
    for (int i = 0; i < values_per_thread; i += 8) {
      sum += xp[i] + xp[i + 1] + xp[i + 2] + xp[i + 3] + xp[i + 4] + xp[i + 5] + xp[i + 6] + xp[i + 7];
      x_thread[i] = xp[i];
      x_thread[i + 1] = xp[i + 1] / 32.0f;
      x_thread[i + 2] = xp[i + 2] / 4.0f;
      x_thread[i + 3] = xp[i + 3] / 128.0f;
      x_thread[i + 4] = xp[i + 4] / 16.0f;
      x_thread[i + 5] = xp[i + 5] / 2.0f;
      x_thread[i + 6] = xp[i + 6] / 64.0f;
      x_thread[i + 7] = xp[i + 7] / 8.0f;
    }
  } else if (BITS == 6) {
    for (int i = 0; i < values_per_thread; i += 4) {
      sum += xp[i] + xp[i + 1] + xp[i + 2] + xp[i + 3];
      x_thread[i] = xp[i];
      x_thread[i + 1] = xp[i + 1] / 64.0f;
      x_thread[i + 2] = xp[i + 2] / 16.0f;
      x_thread[i + 3] = xp[i + 3] / 4.0f;
    }
  } else {
    for (int i = 0; i < values_per_thread; i++) {
      sum += xp[i];
      x_thread[i] = xp[i];
    }
  }
  for (int rr = 0; rr < 4; rr++) {
    const device uint8_t* wl = ws + rr * in_vec_size_w;
    U scv = scales_p[rr * in_vec_size_g];
    U biv = biases_p[rr * in_vec_size_g];
    U accum = 0;
    if (BITS == 3) {
      for (int i = 0; i < values_per_thread / 8; i++) {
        const device uint8_t* w3 = wl + 3 * i;
        thread U* x3 = x_thread + 8 * i;
        accum += (w3[0] & 0x07) * x3[0];
        accum += (w3[0] & 0x38) * x3[1];
        accum += (w3[0] & 0xc0) * x3[2];
        accum += (w3[1] & 0x01) * (x3[2] * 256.0f);
        accum += (w3[1] & 0x0e) * x3[3];
        accum += (w3[1] & 0x70) * x3[4];
        accum += (w3[1] & 0x80) * x3[5];
        accum += (w3[2] & 0x03) * (x3[5] * 256.0f);
        accum += (w3[2] & 0x1c) * x3[6];
        accum += (w3[2] & 0xe0) * x3[7];
      }
    } else if (BITS == 5) {
      for (int i = 0; i < values_per_thread / 8; i++) {
        const device uint8_t* w5 = wl + 5 * i;
        thread U* x5 = x_thread + 8 * i;
        accum += (w5[0] & 0x1f) * x5[0];
        accum += (w5[0] & 0xe0) * x5[1];
        accum += (w5[1] & 0x3) * (x5[1] * 256.0f);
        accum += (w5[1] & 0x7c) * x5[2];
        accum += (w5[1] & 0x80) * x5[3];
        accum += (w5[2] & 0xf) * (x5[3] * 256.0f);
        accum += (w5[2] & 0xf0) * x5[4];
        accum += (w5[3] & 0x1) * (x5[4] * 256.0f);
        accum += (w5[3] & 0x3e) * x5[5];
        accum += (w5[3] & 0xc0) * x5[6];
        accum += (w5[4] & 0x7) * (x5[6] * 256.0f);
        accum += (w5[4] & 0xf8) * x5[7];
      }
    } else if (BITS == 6) {
      for (int i = 0; i < values_per_thread / 4; i++) {
        const device uint8_t* w6 = wl + 3 * i;
        thread U* x6 = x_thread + 4 * i;
        accum += (w6[0] & 0x3f) * x6[0];
        accum += (w6[0] & 0xc0) * x6[1];
        accum += (w6[1] & 0x0f) * (x6[1] * 256.0f);
        accum += (w6[1] & 0xf0) * x6[2];
        accum += (w6[2] & 0x03) * (x6[2] * 256.0f);
        accum += (w6[2] & 0xfc) * x6[3];
      }
    } else if (BITS == 4) {
      const device uint16_t* w16 = (const device uint16_t*)wl;
      for (int i = 0; i < (values_per_thread / 4); i++) {
        accum += (x_thread[4 * i] * (w16[i] & 0x000f) + x_thread[4 * i + 1] * (w16[i] & 0x00f0) + x_thread[4 * i + 2] * (w16[i] & 0x0f00) + x_thread[4 * i + 3] * (w16[i] & 0xf000));
      }
    } else {
      for (int i = 0; i < values_per_thread; i++) {
        accum += x_thread[i] * wl[i];
      }
    }
    result[rr] += scv * accum + sum * biv;
  }
  ws += block_size * bytes_per_pack / pack_factor;
  scales_p += block_size / GS;
  biases_p += block_size / GS;
  xp += block_size;
}
for (int rr = 0; rr < 4; rr++) {
  result[rr] = simd_sum(result[rr]);
  if (simd_lid == 0) {
    yp[rr] = static_cast<T>(result[rr]);
  }
}