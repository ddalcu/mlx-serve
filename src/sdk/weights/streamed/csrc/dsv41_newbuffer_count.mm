// PROFILE builds only (-Ddsv41-decode-timers=true): counts -[MTLDevice newBufferWithLength:options:], the call MLX's
// Metal allocator makes when its buffer cache misses (MetalAllocator::malloc; MLX keeps no allocation or hit counter of
// its own), and the bytes asked, for the cell's MLX_NEWBUFFER read-out (the stock-route growth's buffer-cache-churn
// candidate: a decode whose cycles miss the cache allocates a fresh buffer each cycle). The device's class's method is
// overridden once (its original kept and called), so every MTLDevice of that class counts. Never in a window's timed
// build: the measured binaries do not compile this file. Non-ARC Objective-C++.
#import <Metal/Metal.h>
#include <objc/runtime.h>
#include <atomic>
#include <stdint.h>

typedef id (*NewBufImp)(id, SEL, NSUInteger, MTLResourceOptions);

static std::atomic<NewBufImp> g_orig{nullptr};
static std::atomic<uint64_t> g_calls{0};
static std::atomic<uint64_t> g_bytes{0};

static id nb_counted(id self, SEL sel, NSUInteger len, MTLResourceOptions opt) {
  g_calls.fetch_add(1, std::memory_order_relaxed);
  g_bytes.fetch_add((uint64_t)len, std::memory_order_relaxed);
  return g_orig.load(std::memory_order_acquire)(self, sel, len, opt);
}

/* Installs the count once: 0, or -1 (no device), -2 (no such method). */
extern "C" int dsv41nb_install(void) {
  if (g_orig.load(std::memory_order_acquire) != nullptr) return 0;
  id<MTLDevice> d = MTLCreateSystemDefaultDevice();
  if (d == nil) return -1;
  Method m = class_getInstanceMethod(object_getClass(d), @selector(newBufferWithLength:options:));
  if (m == NULL) {
    [d release];
    return -2;
  }
  // The original is stored before the override is installed, so a concurrent call never sees a null original.
  g_orig.store((NewBufImp)method_getImplementation(m), std::memory_order_release);
  method_setImplementation(m, (IMP)nb_counted);
  [d release];
  return 0;
}

/* out[0] = calls, out[1] = bytes asked, since the install. */
extern "C" void dsv41nb_read(uint64_t out[2]) {
  out[0] = g_calls.load(std::memory_order_relaxed);
  out[1] = g_bytes.load(std::memory_order_relaxed);
}
