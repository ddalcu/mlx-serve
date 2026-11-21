// Port of mlx-lm mlx_lm/models/inkling2/kernels.py `_DEPTH_ATTENTION_SOURCE` (MIT, see NOTICE): each
// token's learned queries over its K block states and its partial state, one threadgroup a row.
constexpr int N = K + 1;
uint row = threadgroup_position_in_grid.x;
uint rows = threadgroups_per_grid.x;
uint tid = thread_position_in_threadgroup.x;
uint lane = thread_index_in_simdgroup;
uint sg = simdgroup_index_in_threadgroup;

float acc[N * Q];
for (int i = 0; i < N * Q; i++) {
    acc[i] = 0.0f;
}
for (int d = tid; d < D; d += TG) {
    float x[N];
    for (int i = 0; i < K; i++) {
        x[i] = blocks[(size_t(i) * rows + row) * D + d];
    }
    x[K] = partial[size_t(row) * D + d];
    for (int j = 0; j < Q; j++) {
        float qv = queries[j * D + d];
        for (int i = 0; i < N; i++) {
            acc[i * Q + j] += x[i] * qv;
        }
    }
}

threadgroup float buf[N * Q][TG / 32];
for (int i = 0; i < N * Q; i++) {
    float v = simd_sum(acc[i]);
    if (lane == 0) {
        buf[i][sg] = v;
    }
}
threadgroup_barrier(mem_flags::mem_threadgroup);

float w[N * Q];
for (int i = 0; i < N * Q; i++) {
    float t = 0.0f;
    for (int s = 0; s < TG / 32; s++) {
        t += buf[i][s];
    }
    w[i] = t * scale;
}
for (int j = 0; j < Q; j++) {
    float m = -INFINITY;
    for (int i = 0; i < N; i++) {
        m = max(m, w[i * Q + j]);
    }
    float z = 0.0f;
    for (int i = 0; i < N; i++) {
        w[i * Q + j] = metal::exp(w[i * Q + j] - m);
        z += w[i * Q + j];
    }
    for (int i = 0; i < N; i++) {
        w[i * Q + j] /= z;
    }
}

for (int d = tid; d < D; d += TG) {
    float x[N];
    for (int i = 0; i < K; i++) {
        x[i] = blocks[(size_t(i) * rows + row) * D + d];
    }
    x[K] = partial[size_t(row) * D + d];
    for (int j = 0; j < Q; j++) {
        float o = 0.0f;
        for (int i = 0; i < N; i++) {
            o += w[i * Q + j] * x[i];
        }
        out[(size_t(row) * Q + j) * D + d] = o;
    }
}
