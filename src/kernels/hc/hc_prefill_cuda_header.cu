#include "mlx/backend/cuda/device/binary_ops.cuh"
#include "mlx/backend/cuda/device/unary_ops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
template <typename T> struct alignas(16) msv_vec8 { T v[8]; };
