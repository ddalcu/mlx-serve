// MiMo X12 (`gpu_keep_warm: "waits"`, src/mimo_v2/gpu_warm.zig): a trivial resident kernel on its OWN Metal command
// queue and its own tiny buffer, committed in short chunks by the warm thread while the decode lane's event-gated miss
// parts wait for their bytes (the stream's MTLSharedEvent below the value they wait for), so the GPU is not idle in
// those gaps. Nothing here reads or writes a model array: the kernel's only memory is `buf` (groups x threads floats).
//
// Why a separate queue and not MLX's stream: MLX's stream queue is in order, and the gated wait is already committed
// at the head of the next layer's barrier set when the GPU goes idle; work committed to that queue after it would wait
// behind it, and work placed before it could not be ended when the bytes land. A second MTLCommandQueue on MLX's
// device runs beside the blocked one; the warm thread stops committing at the event, so at most the chunks already in
// flight (each ~500 us of ONE simdgroup on one core) outlive the wait. MLX's own queue, encoders and limits are not
// touched, and no MLX call is made off the inference thread.
//
// Kept in this translation unit because the host compiles a fixed list of the plugin's csrc files; DeepSeek never
// calls it. Built against MLX 0.32.x's metal-cpp (libmlx exports every selector used here): re-verify on an MLX bump.
#include <time.h>

#include <cstdint>
#include <exception>
#include <string>

#include "mlx/backend/metal/device.h"

namespace mimo_warm {

thread_local std::string g_error;

// One dependent FMA chain per thread: no loads, no threadgroup memory, a store of the result so the chain is kept.
// `iters` sets the chunk's length (the warm thread adapts it to the target from each chunk's GPU time).
const char* const kSource = R"MSL(
#include <metal_stdlib>
using namespace metal;
kernel void mimo_gpu_warm(device float* out [[buffer(0)]], constant uint& iters [[buffer(1)]],
                          uint tid [[thread_position_in_grid]]) {
  float a = 1.0f + float(tid) * 1e-7f;
  for (uint i = 0; i < iters; i++) {
    a = fma(a, 0.9999999f, 1e-7f);
  }
  out[tid] = a;
}
)MSL";

constexpr uint32_t kMaxGroups = 256;
constexpr uint32_t kMaxThreads = 256;

struct Warm {
  MTL::CommandQueue* queue = nullptr;
  MTL::ComputePipelineState* pso = nullptr;
  MTL::Buffer* buf = nullptr;
  uint32_t groups = 1;
  uint32_t threads = 32;
};

struct Pool {
  NS::AutoreleasePool* p;
  Pool() : p(NS::AutoreleasePool::alloc()->init()) {}
  ~Pool() { p->release(); }
};

int fail(const char* what) {
  g_error = what;
  return -1;
}

int fail_ns(const char* what, NS::Error* err) {
  g_error = what;
  if (err != nullptr && err->localizedDescription() != nullptr) {
    g_error += ": ";
    g_error += err->localizedDescription()->utf8String();
  }
  return -1;
}

void destroy(Warm* w) {
  if (w == nullptr) return;
  if (w->buf != nullptr) w->buf->release();
  if (w->pso != nullptr) w->pso->release();
  if (w->queue != nullptr) w->queue->release();
  delete w;
}

}  // namespace mimo_warm

extern "C" {

const char* mimo_warm_last_error(void) {
  return mimo_warm::g_error.c_str();
}

// The warm queue, pipeline and buffer on MLX's GPU device (`groups` threadgroups of `threads` threads per chunk).
// Construction only, under the caller's lease (it compiles one tiny library; no kernel runs). 0, or -1.
int mimo_warm_create(uint32_t groups, uint32_t threads, void** out) {
  using namespace mimo_warm;
  *out = nullptr;
  if (groups == 0 || groups > kMaxGroups || threads == 0 || threads > kMaxThreads || threads % 32 != 0)
    return fail("groups 1..256 and threads a multiple of 32 up to 256");
  try {
    Pool pool;
    MTL::Device* dev = mx::metal::device(mx::Device::gpu).mtl_device();
    if (dev == nullptr) return fail("no MLX GPU device");
    auto* w = new Warm();
    w->groups = groups;
    w->threads = threads;
    w->queue = dev->newCommandQueue();
    if (w->queue == nullptr) {
      destroy(w);
      return fail("newCommandQueue failed");
    }
    NS::Error* err = nullptr;
    MTL::Library* lib = dev->newLibrary(NS::String::string(kSource, NS::UTF8StringEncoding), nullptr, &err);
    if (lib == nullptr) {
      destroy(w);
      return fail_ns("the warm kernel did not compile", err);
    }
    MTL::Function* fn = lib->newFunction(NS::String::string("mimo_gpu_warm", NS::UTF8StringEncoding));
    if (fn == nullptr) {
      lib->release();
      destroy(w);
      return fail("the warm kernel's function is missing");
    }
    err = nullptr;
    w->pso = dev->newComputePipelineState(fn, &err);
    fn->release();
    lib->release();
    if (w->pso == nullptr) {
      destroy(w);
      return fail_ns("the warm pipeline did not build", err);
    }
    if (w->pso->maxTotalThreadsPerThreadgroup() < threads) {
      destroy(w);
      return fail("the warm pipeline takes fewer threads per threadgroup than asked");
    }
    w->buf = dev->newBuffer(static_cast<size_t>(kMaxGroups) * kMaxThreads * sizeof(float), MTL::ResourceStorageModeShared);
    if (w->buf == nullptr) {
      destroy(w);
      return fail("newBuffer failed");
    }
    *out = w;
    return 0;
  } catch (const std::exception& e) {
    return fail(e.what());
  } catch (...) {
    return fail("the MLX GPU device is unavailable");
  }
}

void mimo_warm_destroy(void* h) {
  mimo_warm::destroy(static_cast<mimo_warm::Warm*>(h));
}

// One chunk: `iters` FMAs per thread, encoded and committed on the warm queue. Writes the retained command buffer to
// *cb (the caller retires it). Warm thread only. 0, or -1.
int mimo_warm_commit(void* h, uint32_t iters, void** cb) {
  using namespace mimo_warm;
  *cb = nullptr;
  auto* w = static_cast<Warm*>(h);
  if (w == nullptr) return fail("no warm device");
  Pool pool;
  MTL::CommandBuffer* buf = w->queue->commandBufferWithUnretainedReferences();
  if (buf == nullptr) return fail("commandBufferWithUnretainedReferences failed");
  buf->retain();
  MTL::ComputeCommandEncoder* enc = buf->computeCommandEncoder();
  if (enc == nullptr) {
    buf->release();
    return fail("computeCommandEncoder failed");
  }
  enc->setComputePipelineState(w->pso);
  enc->setBuffer(w->buf, 0, 0);
  enc->setBytes(&iters, sizeof(iters), 1);
  enc->dispatchThreadgroups(MTL::Size(w->groups, 1, 1), MTL::Size(w->threads, 1, 1));
  enc->endEncoding();
  buf->commit();
  *cb = buf;
  return 0;
}

// Waits for a committed chunk, writes its GPU start / end (s, the host's mach-absolute clock), releases it. 0, or -1
// when Metal completed it with an error.
int mimo_warm_retire(void* cb, double* gpu_start, double* gpu_end) {
  using namespace mimo_warm;
  auto* buf = static_cast<MTL::CommandBuffer*>(cb);
  *gpu_start = 0;
  *gpu_end = 0;
  if (buf == nullptr) return fail("no chunk");
  Pool pool;
  buf->waitUntilCompleted();
  const MTL::CommandBufferStatus st = buf->status();
  *gpu_start = buf->GPUStartTime();
  *gpu_end = buf->GPUEndTime();
  buf->release();
  if (st != MTL::CommandBufferStatusCompleted) return fail("a warm chunk completed with an error");
  return 0;
}

// The stream's event (an id<MTLSharedEvent>, `expert_event.Event.object`): its signaled value.
uint64_t mimo_warm_event_value(uint64_t event) {
  return reinterpret_cast<MTL::SharedEvent*>(event)->signaledValue();
}

// The host clock Metal's GPU timestamps use (mach-absolute, ns).
uint64_t mimo_warm_now_ns(void) {
  return clock_gettime_nsec_np(CLOCK_UPTIME_RAW);
}

}  // extern "C"
