/* An MLX-owned array without a fill (mlx_alloc_shim.cpp): the slot banks' grown rows when the grow skips the zero fill.
 * The array is evaluated (its data pointer is valid at once); its bytes are unspecified until written. And an alias:
 * one array's pages under a buffer of its own (a pool both of a streamed bank's phases carve). */
#ifndef MLX_SERVE_MLX_ALLOC_SHIM_H
#define MLX_SERVE_MLX_ALLOC_SHIM_H

#include <stddef.h>
#include <stdint.h>

#include "mlx/c/array.h"

#ifdef __cplusplus
extern "C" {
#endif

/* *out (an mlx_array_new() handle) becomes a [shape] array of dtype over one fresh allocator buffer. 0, or -1
 * (dsv41_alloc_last_error). */
int dsv41_alloc_uninit(mlx_array *out, const int *shape, size_t ndim, mlx_dtype dtype);
const char *dsv41_alloc_last_error(void);

/* *out (an mlx_array_new() handle) becomes a [shape] array of dtype over base's bytes from offset, a no-copy MTLBuffer
 * of its own that holds base. base: evaluated, at its buffer's start; offset: page-aligned; the page-rounded range
 * within base's buffer. 0, or -1 (dsv41_alloc_last_error). */
int dsv41_alloc_alias(mlx_array *out, mlx_array base, uint64_t offset, const int *shape, size_t ndim, mlx_dtype dtype);

/* MLX's command-buffer split on the Metal device: ops operations and mb MB of bound arrays a buffer takes before it is
 * committed (below 1 keeps the current one); the previous pair in *prev_ops / *prev_mb. 0, or -1
 * (dsv41_alloc_last_error). */
int dsv41_command_buffer_limits(int ops, int mb, int *prev_ops, int *prev_mb);

#ifdef __cplusplus
}
#endif

#endif
