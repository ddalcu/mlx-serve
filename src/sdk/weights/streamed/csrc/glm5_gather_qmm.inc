// glm5_next (src/arch/glm5_next/nvfp4.zig): MLX's gather_qmm in nvfp4 mode with its per-expert global scale. MLX's C++
// op takes the scale (`global_scale`, one f32 per expert of `w`, GPU only); mlx-c's `mlx_gather_qmm` has no argument for
// it (the host's patch passes nullopt). This is a new C entry, not a change of mlx_gather_qmm: the binary keeps the
// pinned MLX stage. Included by mlx_alloc_shim.cpp (the host compiles a fixed list of csrc files: no host change).
#include <optional>

#include "mlx/ops.h"
#include "mlx/c/private/stream.h"

extern "C" {

// *res (an mlx_array_new() handle) = gather_qmm(x, w, scales, rhs_indices) with transpose, group 16, bits 4, mode
// "nvfp4" and `global_scale`, `sorted` the rhs_indices' order; 0, or -1 (dsv41_alloc_last_error names the reason).
int glm5_gather_qmm_nvfp4(mlx_array* res, mlx_array x, mlx_array w, mlx_array scales, mlx_array global_scale,
                          mlx_array rhs_indices, bool sorted, mlx_stream s) {
  try {
    mlx_array_set_(*res, mx::gather_qmm(mlx_array_get_(x), mlx_array_get_(w), mlx_array_get_(scales), std::nullopt,
                                        std::nullopt, mlx_array_get_(rhs_indices), true, 16, 4, "nvfp4",
                                        mlx_array_get_(global_scale), sorted, mlx_stream_get_(s)));
    return 0;
  } catch (const std::exception& e) {
    g_error = e.what();
    return -1;
  } catch (...) {
    g_error = "glm5_gather_qmm_nvfp4 failed";
    return -1;
  }
}

}  // extern "C"
