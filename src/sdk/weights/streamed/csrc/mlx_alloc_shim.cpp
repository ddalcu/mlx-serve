// An MLX-owned array without a fill: one buffer from MLX's allocator (MetalAllocator::malloc), wrapped as an evaluated
// array that frees it through the allocator. No kernel writes it and nothing is wrapped no-copy; its bytes are whatever
// the allocator hands out. Built against MLX 0.32.2's headers: re-verify on every MLX bump.
#include <cstdint>
#include <exception>
#include <string>

#include "mlx/allocator.h"
#include "mlx/array.h"
#include "mlx/c/private/array.h"
#include "mlx/c/private/enums.h"
#include "mlx_alloc_shim.h"

namespace mx = mlx::core;

namespace {
thread_local std::string g_error;
}  // namespace

extern "C" {

const char* dsv41_alloc_last_error(void) {
  return g_error.c_str();
}

int dsv41_alloc_uninit(mlx_array* out, const int* shape, size_t ndim, mlx_dtype dtype) {
  try {
    mx::Shape s(shape, shape + ndim);
    const mx::Dtype dt = mlx_dtype_to_cpp(dtype);
    size_t n = dt.size();
    for (int d : s) {
      if (d < 0) {
        g_error = "negative dimension";
        return -1;
      }
      n *= static_cast<size_t>(d);
    }
    mx::allocator::Buffer buf = mx::allocator::malloc(n);
    if (buf.ptr() == nullptr) {
      g_error = "allocator returned no buffer";
      return -1;
    }
    mlx_array_set_(*out, mx::array(buf, std::move(s), dt));
    return 0;
  } catch (const std::exception& e) {
    g_error = e.what();
    return -1;
  } catch (...) {
    g_error = "dsv41_alloc_uninit failed";
    return -1;
  }
}

}  // extern "C"

// MiMo P7 (`prefill_combine: "inplace"`): the wide call's sources' GPU addresses (csrc/mimo_gpu_address.inc).
#include "mimo_gpu_address.inc"

// MiMo X12 (`gpu_keep_warm`): the warm queue, kernel and tiny buffer (csrc/mimo_gpu_warm.inc). Kept in this translation
// unit because the host compiles a fixed list of the plugin's csrc files; DeepSeek never calls it.
#include "mimo_gpu_warm.inc"

// MiMo X12 measurement: the GPU's energy, DVFS-state and thermal-state residency (IOReport) and DVFS table (csrc/mimo_gpu_power.inc).
#include "mimo_gpu_power.inc"
// glm5_next: MLX's gather_qmm in nvfp4 mode with the per-expert global scale (mlx-c has no argument for it).
#include "glm5_gather_qmm.inc"
// Every arch's MLX custom kernels: the launch cache (sdk_ext/metal_launch.zig; the pinned MLX rebuilds and rehashes a
// kernel's source on every apply).
#include "mlx_launch_cache.inc"
