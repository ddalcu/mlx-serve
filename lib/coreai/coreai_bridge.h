// C ABI of libcoreai_bridge.dylib (CoreAIBridge.swift, from ANEMLL Forge, MIT; see NOTICE).
// The dylib links Core AI (macOS 27), so mlx-serve dlopens it at run time (src/coreai_ffi.zig)
// and keeps running on older macOS without it. Every call that can fail takes (err, errcap)
// and returns NULL / nonzero with a message.
#pragma once
#include <stdint.h>

int32_t cai_abi_version(void);
void cai_release(void *handle);

// compute: 0 neural engine preferred, 1 GPU preferred, 2 CPU only, 3 default.
void *cai_model_load(const char *path, int32_t compute, int32_t check_version, char *err, int32_t errcap);
void *cai_function_load(void *model, const char *name, char *err, int32_t errcap);
// JSON: {"inputs"|"outputs"|"states": [{name, dtype, shape, strides (elements), nbytes, dynamic}]}.
// Returns the buffer size it needs (call with cap 0 to size).
int32_t cai_function_describe(void *fn, char *buf, int32_t cap);

// dtype codes: 0 float16, 1 float32, 2 int32, 3 int8, 4 uint8. strides NULL = packed. Zero-filled IOSurface.
void *cai_buffer_create(int32_t dtype, int32_t rank, const int64_t *shape, const int64_t *strides, char *err,
                        int32_t errcap);
void *cai_buffer_address(void *buffer);
int64_t cai_buffer_nbytes(void *buffer);

void *cai_binding_create(void *fn, int32_t n_in, const char *const *in_names, void *const *in_bufs, int32_t n_out,
                         const char *const *out_names, void *const *out_bufs, int32_t n_st,
                         const char *const *st_names, void *const *st_bufs, char *err, int32_t errcap);
// Runs n bindings in order, blocking until all are done. 0 = ok.
int32_t cai_run(int32_t n, void *const *bindings, double *times_ms, char *err, int32_t errcap);
