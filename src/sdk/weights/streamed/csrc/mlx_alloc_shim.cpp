// An MLX-owned array without a fill: one buffer from MLX's allocator (MetalAllocator::malloc), wrapped as an evaluated
// array that frees it through the allocator. No kernel writes it and nothing is wrapped no-copy; its bytes are whatever
// the allocator hands out. And an alias (`dsv41_alloc_alias`): another evaluated array's pages under an MTLBuffer of
// their own (newBufferWithBytesNoCopy), the base held by the alias's deleter; MLX allocates and counts nothing for it.
// Built against MLX 0.32.5's headers: re-verify on every MLX bump.
#include <mach/vm_page_size.h>

#include <cstdint>
#include <exception>
#include <string>

#include "mlx/allocator.h"
#include "mlx/array.h"
#include "mlx/backend/metal/device.h"
#include "mlx/c/private/array.h"
#include "mlx/c/private/enums.h"
#include "mlx/config.h"
#include "mlx/utils.h"
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
    mx::allocator::Data data = mx::allocator::malloc(n);
    if (data.buffer().ptr() == nullptr) {
      g_error = "allocator returned no buffer";
      return -1;
    }
    mlx_array_set_(*out, mx::array(std::move(data), std::move(s), dt));
    return 0;
  } catch (const std::exception& e) {
    g_error = e.what();
    return -1;
  } catch (...) {
    g_error = "dsv41_alloc_uninit failed";
    return -1;
  }
}

// An autorelease pool around the alias's Metal calls (mimo_gpu_warm.inc's form): libmlx does not export
// mlx::core::metal::new_scoped_memory_pool, though its header declares it; NS::AutoreleasePool's class and selectors
// are libmlx's metal-cpp exports this translation unit already links.
namespace dsv41_alias {
struct Pool {
  NS::AutoreleasePool* p;
  Pool() : p(NS::AutoreleasePool::alloc()->init()) {}
  ~Pool() { p->release(); }
};
}  // namespace dsv41_alias

// MLX's own buffers' options (MetalAllocator's resource_options): shared storage, hazards tracked by MLX, not Metal.
static constexpr MTL::ResourceOptions alias_options =
    MTL::ResourceStorageModeShared | MTL::ResourceHazardTrackingModeUntracked;

int dsv41_alloc_alias(mlx_array* out, mlx_array base, uint64_t offset, const int* shape, size_t ndim, mlx_dtype dtype) {
  try {
    const mx::array& b = mlx_array_get_(base);
    if (b.status() == mx::array::unscheduled || b.data_shared_ptr() == nullptr) {
      g_error = "alias: the base is not evaluated";
      return -1;
    }
    auto* pb = static_cast<MTL::Buffer*>(const_cast<void*>(b.buffer().ptr()));
    if (pb == nullptr) {
      g_error = "alias: the base has no buffer";
      return -1;
    }
    if (b.offset() != 0) {
      g_error = "alias: the base is a view off its buffer's start";
      return -1;
    }
    mx::Shape s(shape, shape + ndim);
    const mx::Dtype dt = mlx_dtype_to_cpp(dtype);
    size_t n = dt.size();
    for (int d : s) {
      if (d < 0) {
        g_error = "alias: negative dimension";
        return -1;
      }
      n *= static_cast<size_t>(d);
    }
    if (n == 0) {
      g_error = "alias: no bytes";
      return -1;
    }
    const size_t page = vm_page_size;
    const size_t len = (n + page - 1) / page * page;
    if (offset % page != 0) {
      g_error = "alias: the offset is off the VM page";
      return -1;
    }
    if (offset > pb->length() || len > pb->length() - offset) {
      g_error = "alias: the range passes the base's buffer";
      return -1;
    }
    dsv41_alias::Pool pool;
    MTL::Device* dev = mx::metal::device(mx::Device::gpu).mtl_device();
    MTL::Buffer* a = dev->newBuffer(static_cast<char*>(pb->contents()) + offset, len, alias_options, nullptr);
    if (a == nullptr) {
      g_error = "alias: Metal made no buffer over the range";
      return -1;
    }
    // The deleter holds the base: its pages outlive every alias over them. It releases only the alias's own buffer.
    mx::array keep = b;
    auto deleter = [keep](mx::allocator::Buffer x) {
      (void)keep;
      dsv41_alias::Pool p;
      static_cast<MTL::Buffer*>(x.ptr())->release();
    };
    mx::allocator::Data data(mx::allocator::Buffer(static_cast<void*>(a)), deleter);
    mlx_array_set_(*out, mx::array(std::move(data), std::move(s), dt));
    return 0;
  } catch (const std::exception& e) {
    g_error = e.what();
    return -1;
  } catch (...) {
    g_error = "dsv41_alloc_alias failed";
    return -1;
  }
}

// MLX's command-buffer split: every commit check (CommandEncoder::needs_commit) reads it through mlx::core::config,
// over the device's default for its architecture.
int dsv41_command_buffer_limits(int ops, int mb, int* prev_ops, int* prev_mb) {
  try {
    auto [default_ops, default_mb] = mx::metal::device(mx::Device::gpu).get_default_max_ops_mb_per_buffer();
    *prev_ops = mx::env::max_ops_per_buffer(default_ops);
    *prev_mb = mx::env::max_mb_per_buffer(default_mb);
    if (ops > 0) mx::config::update("MLX_MAX_OPS_PER_BUFFER", ops);
    if (mb > 0) mx::config::update("MLX_MAX_MB_PER_BUFFER", mb);
    return 0;
  } catch (const std::exception& e) {
    g_error = e.what();
    return -1;
  } catch (...) {
    g_error = "dsv41_command_buffer_limits failed";
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
