// mlx_launch_cache (mlx-stream sdk_ext/metal_launch.zig): MLX's custom Metal kernels launched without rebuilding and
// rehashing their source on every call. In the pinned MLX (64ea011c) every `mlx_fast_metal_kernel_apply` regenerates
// the kernel's whole source (backend/common/metal_kernel.cpp `write_signature` + the template instantiation, header and
// body: kilobytes), copies it into a new CustomKernel primitive, and at encode hashes all of it (`std::hash`) to name the
// library MLX already compiled (backend/metal/custom_kernel.cpp). Here a kernel object made through
// `mlxlc_kernel_new` keeps a cache of its variants. A variant is everything MLX's generated source depends on beyond
// the kernel itself: the template arguments, each input's dtype and address class (scalar / constant: fewer than 8
// elements / device) and the output dtypes. A variant's launch runs:
//   1. through MLX's own apply (the same kernel, the same graph), recording the CustomKernel's generated name, its
//      library name (name_hash_options, as custom_kernel.cpp forms it), its shape infos and contiguity;
//   2. while the variant has no pipeline, each launch first looks MLX's device cache up for that library
//      (`Device::get_library` with a builder that refuses: a miss leaves the cache untouched and the launch runs MLX's
//      apply again, as 1); once MLX has evaluated one of its launches the lookup finds the library MLX built and
//      `get_kernel` its pipeline (MLX's cached object, kept by the device for the process);
//   3. every later launch encodes the pipeline directly (`LaunchCached`, CustomKernel::eval_gpu's encode step for step:
//      the same buffers, shape / stride / ndim bytes, outputs, grid and threadgroup clamp).
// The grid, the threadgroup and the output shapes come from the call's config every launch, as MLX's. A config with an
// init value (MLX fills the outputs first) always runs MLX's apply. ensure_row_contiguous inputs not known contiguous
// at launch go through `contiguous` (MLX's copy when strided, the buffer itself otherwise: CustomKernel's own copy rule).
// Same pipeline, same arguments: bit-identical results. Built against MLX 0.32.x (64ea011c) headers: CustomKernel's
// state() layout, Device's cache lookups and the mlx-c config struct below are re-verified on every MLX bump (the CPU
// test `metal_launch` checks the config layout through mlx-c's setters).
#include <chrono>
#include <cstdio>
#include <cstring>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "mlx/backend/metal/device.h"
#include "mlx/c/fast.h"
#include "mlx/c/private/stream.h"
#include "mlx/c/private/vector.h"
#include "mlx/fast.h"
#include "mlx/fast_primitives.h"
#include "mlx/ops.h"
#include "mlx/primitives.h"

namespace mlxlc {

// mlx-c's private config (lib/mlxc-src/mlx/c/fast.cpp `mlx_fast_metal_kernel_config_cpp_`), field for field.
struct ConfigCpp {
  std::vector<mx::Shape> output_shapes;
  std::vector<mx::Dtype> output_dtypes;
  std::tuple<int, int, int> grid;
  std::tuple<int, int, int> thread_group;
  std::vector<std::pair<std::string, mx::fast::TemplateArg>> template_args;
  std::optional<float> init_value;
  bool verbose;
};

constexpr int max_constant_array_size = 8;  // metal_kernel.cpp's

inline uint8_t input_class(const mx::array& x) {
  return static_cast<uint8_t>((static_cast<int>(x.dtype().val()) << 2) |
                              (x.ndim() == 0 ? 1 : (x.size() < max_constant_array_size ? 2 : 0)));
}

struct Variant {
  std::vector<std::pair<std::string, mx::fast::TemplateArg>> template_args;
  std::vector<uint8_t> in_class;
  std::vector<mx::Dtype> out_dtypes;
  // Recorded from MLX's CustomKernel at the variant's first launch.
  std::string kernel_name;
  std::string lib_name;
  std::vector<std::tuple<bool, bool, bool>> shape_infos;
  bool ensure_row_contiguous = true;
  MTL::ComputePipelineState* pipeline = nullptr;  // MLX's cached pipeline (the device keeps it for the process)
};

struct NotBuilt {};

struct Cache {
  std::mutex mu;
  std::vector<std::unique_ptr<Variant>> variants;
};

std::mutex g_mu;
std::unordered_map<void*, std::shared_ptr<Cache>> g_caches;  // keyed by the mlx-c kernel object

std::shared_ptr<Cache> cache_of(void* ctx) {
  std::lock_guard<std::mutex> lk(g_mu);
  auto it = g_caches.find(ctx);
  return it == g_caches.end() ? nullptr : it->second;
}

class LaunchCached : public mx::Primitive {
 public:
  // `cache` keeps the variant alive while a graph holds this launch (the kernel may be freed before its evaluation).
  LaunchCached(mx::Stream s, std::shared_ptr<Cache> cache, const Variant* v, std::tuple<int, int, int> grid,
               std::tuple<int, int, int> tg)
      : mx::Primitive(s), cache_(std::move(cache)), v_(v), grid_(grid), tg_(tg) {}

  void eval_cpu(const std::vector<mx::array>&, std::vector<mx::array>&) override {
    throw std::runtime_error("Custom kernels only run on GPU.");
  }

  // CustomKernel::eval_gpu (backend/metal/custom_kernel.cpp) with the pipeline resolved: inputs arrive contiguous
  // where the kernel asks for it (the launch's `contiguous`), no init value (such configs never reach here).
  void eval_gpu(const std::vector<mx::array>& inputs, std::vector<mx::array>& outputs) override {
    for (auto& out : outputs) out.set_data(mx::allocator::malloc(out.nbytes()));
    auto& enc = mx::metal::get_command_encoder(stream());
    enc.set_compute_pipeline_state(v_->pipeline);
    int index = 0;
    for (size_t i = 0; i < inputs.size(); i++) {
      const mx::array& in = inputs[i];
      const auto& si = v_->shape_infos[i];
      enc.set_input_array(in, index);
      index++;
      if (in.ndim() > 0) {
        int ndim = in.ndim();
        if (std::get<0>(si)) {
          enc.set_vector_bytes(in.shape(), ndim, index);
          index++;
        }
        if (std::get<1>(si)) {
          enc.set_vector_bytes(in.strides(), ndim, index);
          index++;
        }
        if (std::get<2>(si)) {
          enc.set_bytes(ndim, index);
          index++;
        }
      }
    }
    for (auto& out : outputs) {
      enc.set_output_array(out, index);
      index++;
    }
    const auto [tx, ty, tz] = tg_;
    auto tg_size = tx * ty * tz;
    auto max_tg_size = v_->pipeline->maxTotalThreadsPerThreadgroup();
    if (tg_size > max_tg_size) {
      throw std::invalid_argument("Thread group size (" + std::to_string(tg_size) +
                                  ") is greater than  the maximum allowed threads per threadgroup (" +
                                  std::to_string(max_tg_size) + ").");
    }
    const auto [gx, gy, gz] = grid_;
    enc.dispatch_threads(MTL::Size(gx, gy, gz), MTL::Size(std::min(tx, gx), std::min(ty, gy), std::min(tz, gz)));
  }

  const char* name() const override {
    return "CustomKernelCached";
  }

 private:
  std::shared_ptr<Cache> cache_;
  const Variant* v_;
  std::tuple<int, int, int> grid_;
  std::tuple<int, int, int> tg_;
};

// Tests: a graph input with no data and no producer that runs (the benchmark's inputs: nothing is allocated, so the
// CPU lane creates no device).
class Placeholder : public mx::Primitive {
 public:
  using mx::Primitive::Primitive;
  void eval_cpu(const std::vector<mx::array>&, std::vector<mx::array>&) override {
    throw std::runtime_error("mlxlc placeholder evaluated");
  }
  void eval_gpu(const std::vector<mx::array>&, std::vector<mx::array>&) override {
    throw std::runtime_error("mlxlc placeholder evaluated");
  }
  const char* name() const override {
    return "MlxlcPlaceholder";
  }
};

Variant* find_variant(Cache& c, const ConfigCpp& cfg, const std::vector<mx::array>& ins) {
  for (auto& v : c.variants) {
    if (v->in_class.size() != ins.size() || v->out_dtypes != cfg.output_dtypes || v->template_args != cfg.template_args)
      continue;
    bool same = true;
    for (size_t i = 0; i < ins.size() && same; i++) same = v->in_class[i] == input_class(ins[i]);
    if (same) return v.get();
  }
  return nullptr;
}

// MLX's apply (its own error reporting on failure: its return code), then (first launch of a variant) the record.
int mlx_apply(mlx_vector_array* outputs, mlx_fast_metal_kernel cls, mlx_vector_array inputs,
               mlx_fast_metal_kernel_config config, mlx_stream stream, Cache& c, Variant* v, const ConfigCpp& cfg,
               const std::vector<mx::array>& ins) {
  const int rc = mlx_fast_metal_kernel_apply(outputs, cls, inputs, config, stream);
  if (rc != 0) return rc;
  const auto& outs = mlx_vector_array_get_(*outputs);
  if (outs.empty() || !outs[0].has_primitive()) return 0;
  auto& prim = outs[0].primitive();
  if (std::string(prim.name()) != "CustomKernel") return 0;
  if (v == nullptr) {
    auto& ck = static_cast<mx::fast::CustomKernel&>(prim);
    auto st = ck.state();
    auto nv = std::make_unique<Variant>();
    nv->template_args = cfg.template_args;
    for (auto& x : ins) nv->in_class.push_back(input_class(x));
    nv->out_dtypes = cfg.output_dtypes;
    nv->kernel_name = std::get<0>(st);
    const std::string& source = std::get<1>(st);
    char hex[32];
    std::snprintf(hex, sizeof hex, "%zx", std::hash<std::string>{}(source));
    nv->lib_name = nv->kernel_name + "_" + hex + "_" + std::to_string(std::get<10>(st));
    nv->shape_infos = std::get<4>(st);
    nv->ensure_row_contiguous = std::get<5>(st);
    c.variants.push_back(std::move(nv));
  }
  return 0;
}

}  // namespace mlxlc

extern "C" {

// mlx_fast_metal_kernel_new, with a launch cache for the kernel (freed by mlxlc_kernel_free).
mlx_fast_metal_kernel mlxlc_kernel_new(const char* name, const mlx_vector_string input_names,
                                       const mlx_vector_string output_names, const char* source, const char* header,
                                       bool ensure_row_contiguous, bool atomic_outputs) {
  auto k = mlx_fast_metal_kernel_new(name, input_names, output_names, source, header, ensure_row_contiguous,
                                     atomic_outputs);
  if (k.ctx != nullptr) {
    std::lock_guard<std::mutex> lk(mlxlc::g_mu);
    mlxlc::g_caches[k.ctx] = std::make_shared<mlxlc::Cache>();
  }
  return k;
}

void mlxlc_kernel_free(mlx_fast_metal_kernel k) {
  if (k.ctx != nullptr) {
    std::lock_guard<std::mutex> lk(mlxlc::g_mu);
    mlxlc::g_caches.erase(k.ctx);
  }
  mlx_fast_metal_kernel_free(k);
}

// mlx_fast_metal_kernel_apply's contract (0, or nonzero with mlx's error / dsv41_alloc_last_error set).
int mlxlc_apply(mlx_vector_array* outputs, mlx_fast_metal_kernel cls, const mlx_vector_array inputs,
                const mlx_fast_metal_kernel_config config, const mlx_stream stream) {
  try {
    auto cache = mlxlc::cache_of(cls.ctx);
    if (!cache) return mlx_fast_metal_kernel_apply(outputs, cls, inputs, config, stream);
    const auto& cfg = *static_cast<const mlxlc::ConfigCpp*>(config.ctx);
    const auto& ins = mlx_vector_array_get_(inputs);
    std::lock_guard<std::mutex> lk(cache->mu);
    mlxlc::Variant* v = cfg.init_value ? nullptr : mlxlc::find_variant(*cache, cfg, ins);
    if (cfg.init_value) {
      return mlx_fast_metal_kernel_apply(outputs, cls, inputs, config, stream);
    }
    if (v != nullptr && v->pipeline == nullptr) {
      // The library MLX built for this variant, once it has evaluated a launch of it (a lookup; a miss throws from
      // the builder before the cache is touched, and this launch runs MLX's apply).
      auto& d = mx::metal::device(mx::Device::gpu);
      MTL::Library* lib = nullptr;
      try {
        lib = d.get_library(v->lib_name, mx::CompileOptions{}, []() -> std::string { throw mlxlc::NotBuilt{}; });
      } catch (const mlxlc::NotBuilt&) {
      }
      if (lib != nullptr) v->pipeline = d.get_kernel(v->kernel_name, lib);
    }
    if (v == nullptr || v->pipeline == nullptr) {
      return mlxlc::mlx_apply(outputs, cls, inputs, config, stream, *cache, v, cfg, ins);
    }
    auto s = mlx_stream_get_(stream);
    std::vector<mx::array> launch_in;
    launch_in.reserve(ins.size());
    for (const auto& x : ins) {
      if (v->ensure_row_contiguous && !(x.status() == mx::array::Status::available && x.flags().row_contiguous))
        launch_in.push_back(mx::contiguous(x, false, s));
      else
        launch_in.push_back(x);
    }
    auto prim = std::make_shared<mlxlc::LaunchCached>(s, cache, v, cfg.grid, cfg.thread_group);
    mlx_vector_array_set_(*outputs, mx::array::make_arrays(cfg.output_shapes, cfg.output_dtypes, prim, launch_in));
    return 0;
  } catch (const std::exception& e) {
    g_error = e.what();
    return -1;
  } catch (...) {
    g_error = "mlxlc_apply failed";
    return -1;
  }
}

// Tests (CPU, no device): the config as the shim reads it: grid xyz, threadgroup xyz, outputs, template args, init value
// set (9 ints).
void mlxlc_test_config(mlx_fast_metal_kernel_config config, int out[9]) {
  const auto& c = *static_cast<const mlxlc::ConfigCpp*>(config.ctx);
  out[0] = std::get<0>(c.grid);
  out[1] = std::get<1>(c.grid);
  out[2] = std::get<2>(c.grid);
  out[3] = std::get<0>(c.thread_group);
  out[4] = std::get<1>(c.thread_group);
  out[5] = std::get<2>(c.thread_group);
  out[6] = static_cast<int>(c.output_shapes.size());
  out[7] = static_cast<int>(c.template_args.size());
  out[8] = c.init_value ? 1 : 0;
}

// Tests (CPU, no device: no buffer is allocated): host ns per launch over `iters` launches of `k` with `config` on
// `n_in` inputs given by shape and dtype (placeholder graph nodes on an explicit GPU stream, unevaluated as a decode
// launch's activations arrive; the graphs are never evaluated). out[0]: MLX's apply plus the encode-time work it adds per launch (the hash of
// the generated source and the library name, custom_kernel.cpp); out[1]: the cached launch (a variant marked resolved
// with a placeholder pipeline that is never encoded); out[2]: the generated source's bytes. 0, or -1 (g_error).
int mlxlc_test_bench(mlx_fast_metal_kernel k, mlx_fast_metal_kernel_config config, int n_in, const int* ndims,
                     const int* shapes, const mlx_dtype* dtypes, int iters, int64_t out[3]) {
  try {
    auto s = mx::Stream(0, mx::Device::gpu);
    mlx_stream ms = mlx_stream_new_(s);
    auto cache = mlxlc::cache_of(k.ctx);
    if (!cache) throw std::runtime_error("mlxlc_test_bench: the kernel was not made through mlxlc_kernel_new");
    const auto& cfg = *static_cast<const mlxlc::ConfigCpp*>(config.ctx);
    std::vector<mx::array> ins;
    for (int i = 0, at = 0; i < n_in; at += ndims[i], i++)
      ins.push_back(mx::array(mx::Shape(shapes + at, shapes + at + ndims[i]), mlx_dtype_to_cpp(dtypes[i]),
                              std::make_shared<mlxlc::Placeholder>(s), {}));
    mlx_vector_array vin = mlx_vector_array_new();
    mlx_vector_array_set_(vin, ins);
    size_t bytes = 0;
    volatile size_t sink = 0;
    auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < iters; i++) {
      mlx_vector_array outs = mlx_vector_array_new();
      if (mlx_fast_metal_kernel_apply(&outs, k, vin, config, ms) != 0) throw std::runtime_error("apply failed");
      const auto& o = mlx_vector_array_get_(outs);
      auto st = static_cast<mx::fast::CustomKernel&>(o[0].primitive()).state();
      const std::string& src = std::get<1>(st);
      bytes = src.size();
      char hex[32];
      std::snprintf(hex, sizeof hex, "%zx", std::hash<std::string>{}(src));
      std::string lib = std::get<0>(st) + "_" + hex + "_" + std::to_string(std::get<10>(st));
      sink = sink + lib.size();
      mlx_vector_array_free(outs);
    }
    auto t1 = std::chrono::steady_clock::now();
    {
      std::lock_guard<std::mutex> lk(cache->mu);
      if (mlxlc::find_variant(*cache, cfg, ins) == nullptr) {
        auto nv = std::make_unique<mlxlc::Variant>();
        nv->template_args = cfg.template_args;
        for (auto& x : ins) nv->in_class.push_back(mlxlc::input_class(x));
        nv->out_dtypes = cfg.output_dtypes;
        nv->shape_infos.assign(ins.size(), {false, false, false});
        nv->pipeline = reinterpret_cast<MTL::ComputePipelineState*>(uintptr_t{1});
        cache->variants.push_back(std::move(nv));
      }
    }
    auto t2 = std::chrono::steady_clock::now();
    for (int i = 0; i < iters; i++) {
      mlx_vector_array outs = mlx_vector_array_new();
      if (mlxlc_apply(&outs, k, vin, config, ms) != 0) throw std::runtime_error(g_error);
      mlx_vector_array_free(outs);
    }
    auto t3 = std::chrono::steady_clock::now();
    {
      std::lock_guard<std::mutex> lk(cache->mu);
      cache->variants.clear();
    }
    mlx_vector_array_free(vin);
    mlx_stream_free(ms);
    out[0] = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count() / iters;
    out[1] = std::chrono::duration_cast<std::chrono::nanoseconds>(t3 - t2).count() / iters;
    out[2] = static_cast<int64_t>(bytes);
    (void)sink;
    return 0;
  } catch (const std::exception& e) {
    g_error = e.what();
    return -1;
  }
}

// Tests (CPU, no device): an explicit GPU stream object for graphs that are built and never evaluated (index 0, never
// registered with MLX's scheduler: nothing is encoded on it).
mlx_stream mlxlc_test_gpu_stream(void) {
  return mlx_stream_new_(mx::Stream(0, mx::Device::gpu));
}

// Tests (CPU, no device): a graph input of `shape` and `dtype` with no data and no producer that runs (a Placeholder
// node, mlxlc_test_bench's inputs: nothing is allocated). 0, or -1 (g_error).
int mlxlc_test_placeholder(const int* shape, int ndim, mlx_dtype dtype, mlx_stream s, mlx_array* out) {
  try {
    const auto& st = mlx_stream_get_(s);
    mlx_array_set_(*out, mx::array(mx::Shape(shape, shape + ndim), mlx_dtype_to_cpp(dtype),
                                    std::make_shared<mlxlc::Placeholder>(st), {}));
    return 0;
  } catch (const std::exception& e) {
    g_error = e.what();
    return -1;
  }
}

// Tests (CPU, no device: a walk of the unevaluated graph): the custom-kernel launches reachable from `outputs` through
// their inputs (MLX's CustomKernel nodes and the cache's CustomKernelCached ones), each launch once (a launch's several
// outputs share its primitive).
int mlxlc_test_count_launches(const mlx_vector_array outputs) {
  const auto& outs = mlx_vector_array_get_(outputs);
  std::vector<mx::array> stack(outs.begin(), outs.end());
  std::unordered_set<std::uintptr_t> visited;
  std::unordered_set<const void*> launches;
  while (!stack.empty()) {
    mx::array a = stack.back();
    stack.pop_back();
    if (!visited.insert(a.id()).second || !a.has_primitive()) continue;
    const std::string name = a.primitive().name();
    if (name == "CustomKernel" || name == "CustomKernelCached") launches.insert(&a.primitive());
    for (const auto& in : a.inputs()) stack.push_back(in);
  }
  return static_cast<int>(launches.size());
}

// Tests (CPU, no device: a walk of the unevaluated graph): the primitive of every node reachable from `outputs` (each node
// once: a multi-output primitive once), names joined by ',' into `out` (at most `cap` bytes with its terminator); returns
// the nodes, or -1 when the names do not fit.
int mlxlc_test_graph_ops(const mlx_vector_array outputs, char* out, int cap) {
  const auto& outs = mlx_vector_array_get_(outputs);
  std::vector<mx::array> stack(outs.begin(), outs.end());
  std::unordered_set<std::uintptr_t> visited;
  std::unordered_set<const void*> prims;
  std::string names;
  int n = 0;
  while (!stack.empty()) {
    mx::array a = stack.back();
    stack.pop_back();
    if (!visited.insert(a.id()).second || !a.has_primitive()) continue;
    if (prims.insert(&a.primitive()).second) {
      if (n > 0) names += ",";
      names += a.primitive().name();
      n++;
    }
    for (const auto& in : a.inputs()) stack.push_back(in);
  }
  if (static_cast<int>(names.size()) + 1 > cap) return -1;
  std::memcpy(out, names.c_str(), names.size() + 1);
  return n;
}

}  // extern "C"
