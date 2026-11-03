// MiMo P7 (`prefill_combine: "inplace"`, src/mimo_v2/prefill_combine.zig): the GPU address of each evaluated source of a
// wide call, for the fused combine's table form (the text reads every source in place through `tbl`, past Metal's 31
// buffer arguments). Address = the source's MTLBuffer gpuAddress + its byte offset (MLX `array::offset`). A source must
// be evaluated (made available here), its buffer an MTLBuffer (a GPU array) and its rows contiguous, else -1 with the reason (nothing written
// past the failing source). The sources are not bound to the combine's encoder: they are read through MLX's residency
// sets (every MLX buffer under the host's wired policy `max`), which the route checks once at construction.
// Included by mlx_alloc_shim.cpp (the host compiles a fixed list of csrc files: no host change).
#include "mlx/backend/metal/device.h"

extern "C" {

int mimo_gpu_addresses(const mlx_array* arrs, size_t n, uint64_t* out) {
  try {
    for (size_t i = 0; i < n; ++i) {
      const mx::array& a = mlx_array_get_(arrs[i]);
      if (a.status() == mx::array::Status::unscheduled) {
        g_error = "mimo_gpu_addresses: a source is not evaluated";
        return -1;
      }
      // The sources are not inputs of the combine's primitive, so MLX's encoder fences never order the combine after
      // their producers: each source is made available (its producer's command buffer complete) before its address is
      // taken. The caller evaluated and synchronized them (`Source.forward` .joinless), so this returns at once; it
      // marks the status `evaluated` -> `available` (MLX's eval waits the event, it does not set the status).
      mx::array(a).wait();
      if (!a.flags().row_contiguous) {
        g_error = "mimo_gpu_addresses: a source is not row-contiguous";
        return -1;
      }
      const auto* buf = static_cast<const MTL::Buffer*>(a.buffer().ptr());
      if (buf == nullptr) {
        g_error = "mimo_gpu_addresses: a source has no Metal buffer";
        return -1;
      }
      out[i] = static_cast<uint64_t>(buf->gpuAddress()) + static_cast<uint64_t>(a.offset());
    }
    return 0;
  } catch (const std::exception& e) {
    g_error = e.what();
    return -1;
  } catch (...) {
    g_error = "mimo_gpu_addresses failed";
    return -1;
  }
}

}  // extern "C"
