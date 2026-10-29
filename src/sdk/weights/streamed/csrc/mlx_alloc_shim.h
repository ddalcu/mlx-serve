/* An MLX-owned array without a fill (mlx_alloc_shim.cpp): the slot banks' grown rows when the grow skips the zero fill.
 * The array is evaluated (its data pointer is valid at once); its bytes are unspecified until written. */
#ifndef MLX_SERVE_MLX_ALLOC_SHIM_H
#define MLX_SERVE_MLX_ALLOC_SHIM_H

#include <stddef.h>

#include "mlx/c/array.h"

#ifdef __cplusplus
extern "C" {
#endif

/* *out (an mlx_array_new() handle) becomes a [shape] array of dtype over one fresh allocator buffer. 0, or -1
 * (dsv41_alloc_last_error). */
int dsv41_alloc_uninit(mlx_array *out, const int *shape, size_t ndim, mlx_dtype dtype);
const char *dsv41_alloc_last_error(void);

#ifdef __cplusplus
}
#endif

#endif
