// PROFILE builds only (see dsv41_cb_timeline.h). MLX exposes no hook on its command buffers' commit (CommandEncoder
// commits buffer_ from queue_->commandBufferWithUnretainedReferences() through metal-cpp's -commit), so -commit is
// overridden on every class of a live buffer's chain (object_getClass on the instance, then its superclasses) that
// implements it itself, each class's own implementation kept as its original. A later buffer whose class chain holds
// no hooked class is hooked the same way (`dsv41tl_rehook`). One override serves every hooked class: the outermost
// call records the buffer (nested -commit calls a subclass's implementation makes into its superclass's run the next
// original along the chain), adds a completed handler for GPUStartTime / GPUEndTime, then calls the original.
// Diagnostics count every override entry, the tagged ones and the buffers of another queue. Non-ARC Objective-C++.
#import <Metal/Metal.h>
#include <objc/message.h>
#include <objc/runtime.h>
#include <pthread.h>
#include <stdlib.h>
#include <atomic>
#include <string.h>
#include <time.h>

#include "dsv41_cb_timeline.h"

typedef void (*CommitImp)(id, SEL);

#define TL_MAX_CLASSES 16
static Class g_classes[TL_MAX_CLASSES];
static CommitImp g_origs[TL_MAX_CLASSES];
static std::atomic<int> g_nclasses{0};
static id g_queue = nil;
static Dsv41tlRow *g_rows = NULL;
static uint32_t g_cap = 0;
static std::atomic<uint32_t> g_committed{0};
static std::atomic<uint32_t> g_completed{0};
static std::atomic<uint32_t> g_dropped{0};
static std::atomic<uint64_t> g_tag{0};
static std::atomic<uint64_t> g_calls{0};
static std::atomic<uint64_t> g_tagged{0};
static std::atomic<uint64_t> g_other_queue{0};
/* The thread's current outer call: the object whose -commit is running and how deep along its chain. A -commit of
 * another object made from inside it starts at depth 0 (its own outermost call, recorded). */
static thread_local int t_depth = 0;
static thread_local id t_self = nil;
/* install / rehook run on one thread (the inference thread): hook_chain is not reentrant. Plain statics: the first
 * hook_chain (the install) sets them before any other call can reach hook_chain. */
static pthread_t g_owner;
static int g_owned = 0;

uint64_t dsv41tl_now(void) {
  return clock_gettime_nsec_np(CLOCK_UPTIME_RAW);
}

static int hooked_index(Class c) {
  int n = g_nclasses.load(std::memory_order_acquire);
  for (int i = 0; i < n; i++) if (g_classes[i] == c) return i;
  return -1;
}

/* The original for this call: the depth-th hooked class along self's chain (depth 0 = the class dispatch found). */
static CommitImp original_for(id self, int depth) {
  for (Class c = object_getClass(self); c != Nil; c = class_getSuperclass(c)) {
    int i = hooked_index(c);
    if (i < 0) continue;
    if (depth == 0) return g_origs[i];
    depth--;
  }
  return NULL;
}

static void tl_commit(id self, SEL sel) {
  int depth = self == t_self ? t_depth : 0;
  CommitImp orig = original_for(self, depth);
  if (orig == NULL) abort();   /* dispatch reached the override through a hooked class: the original exists */
  if (depth == 0) {
    g_calls.fetch_add(1, std::memory_order_relaxed);
    uint64_t tag = g_tag.load(std::memory_order_relaxed);
    if (tag != 0) {
      g_tagged.fetch_add(1, std::memory_order_relaxed);
      int same = [(id<MTLCommandBuffer>)self commandQueue] == g_queue;
      if (!same) g_other_queue.fetch_add(1, std::memory_order_relaxed);
      uint32_t i = g_committed.fetch_add(1);
      if (i < g_cap) {
        Dsv41tlRow *r = &g_rows[i];
        r->tag = tag;
        r->queue = (uint64_t)same;
        r->host_commit = dsv41tl_now();
        [(id<MTLCommandBuffer>)self addCompletedHandler:^(id<MTLCommandBuffer> cb) {
          r->gpu_start = (uint64_t)(cb.GPUStartTime * 1e9);
          r->gpu_end = (uint64_t)(cb.GPUEndTime * 1e9);
          r->status = (uint64_t)cb.status;
          r->host_done = dsv41tl_now();
          g_completed.fetch_add(1, std::memory_order_release);
        }];
      } else {
        g_dropped.fetch_add(1);
      }
    }
  }
  id saved_self = t_self;
  int saved_depth = t_depth;
  t_self = self;
  t_depth = depth + 1;
  orig(self, sel);
  t_self = saved_self;
  t_depth = saved_depth;
}

/* Every class of obj's chain that implements -commit itself, not yet hooked: hooked. Returns the classes added, or
 * -2 when the chain implements no -commit, -3 when the table is full. */
#define TL_MAX_SEEN 32
static Class g_seen[TL_MAX_SEEN];
static int g_nseen = 0;

static int hook_chain(id obj) {
  if (!g_owned) {
    g_owner = pthread_self();
    g_owned = 1;
  } else if (!pthread_equal(g_owner, pthread_self())) {
    return -4;   /* not the thread that installed: hook_chain is not reentrant */
  }
  Class dyn = object_getClass(obj);
  for (int i = 0; i < g_nseen; i++) if (g_seen[i] == dyn) return 0;   /* this dynamic class's chain is done */
  int added = 0;
  int found = 0;
  for (Class c = object_getClass(obj); c != Nil; c = class_getSuperclass(c)) {
    unsigned n = 0;
    Method *ms = class_copyMethodList(c, &n);
    Method own = NULL;
    for (unsigned k = 0; k < n; k++) if (method_getName(ms[k]) == @selector(commit)) own = ms[k];
    free(ms);
    if (own == NULL) continue;
    found = 1;
    if (hooked_index(c) >= 0) continue;
    int i = g_nclasses.load();
    if (i >= TL_MAX_CLASSES) return -3;
    g_classes[i] = c;
    g_origs[i] = (CommitImp)method_getImplementation(own);
    g_nclasses.store(i + 1, std::memory_order_release);
    method_setImplementation(own, (IMP)tl_commit);
    added++;
  }
  if (!found) return -2;
  if (g_nseen < TL_MAX_SEEN) g_seen[g_nseen++] = dyn;
  return added;
}

int dsv41tl_install(void *cmdbuf, Dsv41tlRow *rows, uint32_t cap) {
  if (cmdbuf == NULL || rows == NULL || cap == 0) return -1;
  id buf = (id)cmdbuf;
  int r = hook_chain(buf);
  if (r < 0) return r;
  g_tag.store(0);
  g_queue = [(id<MTLCommandBuffer>)buf commandQueue];
  g_rows = rows;
  g_cap = cap;
  g_committed.store(0);
  g_completed.store(0);
  g_dropped.store(0);
  g_calls.store(0);
  g_tagged.store(0);
  g_other_queue.store(0);
  return 0;
}

int dsv41tl_rehook(void *cmdbuf) {
  if (cmdbuf == NULL) return -1;
  return hook_chain((id)cmdbuf);
}

void dsv41tl_tag(uint64_t tag) {
  g_tag.store(tag, std::memory_order_relaxed);
}

void dsv41tl_counts(uint32_t out[3]) {
  out[0] = g_committed.load();
  out[1] = g_completed.load(std::memory_order_acquire);
  out[2] = g_dropped.load();
}

void dsv41tl_hook_stats(uint64_t out[4]) {
  out[0] = g_calls.load();
  out[1] = g_tagged.load();
  out[2] = g_other_queue.load();
  out[3] = (uint64_t)g_nclasses.load();
}

size_t dsv41tl_hooked_names(char *out, size_t cap) {
  size_t n = 0;
  int k = g_nclasses.load();
  for (int i = 0; i < k && n + 2 < cap; i++) {
    const char *name = class_getName(g_classes[i]);
    size_t len = strlen(name);
    if (n + len + 2 >= cap) break;
    if (i > 0) out[n++] = ',';
    memcpy(out + n, name, len);
    n += len;
  }
  if (cap > 0) out[n < cap ? n : cap - 1] = 0;
  return n;
}

// ── the host self-test: a fake buffer hierarchy (no Metal object) ──
// FakeBase implements -commit (counts), -commandQueue, -addCompletedHandler: (runs the block now), GPUStartTime 1 s,
// GPUEndTime 2 s, status 4; FakeSub overrides -commit (counts, then [super commit]); FakeSib inherits FakeBase's;
// FakeOther is an unrelated root with its own -commit. Returns 0, or the number of the first failed check.
static int st_base = 0, st_sub = 0, st_other = 0;
static id st_queue = nil;
static void fb_commit(id, SEL) { st_base++; }
static id fb_queue(id, SEL) { return st_queue; }
static void fb_handler(id self, SEL, void (^block)(id)) { block(self); }
static double fb_start(id, SEL) { return 1.0; }
static double fb_end(id, SEL) { return 2.0; }
static NSUInteger fb_status(id, SEL) { return 4; }
static void fs_commit(id self, SEL sel) {
  st_sub++;
  struct objc_super sup = { self, class_getSuperclass(object_getClass(self)) };
  ((void (*)(struct objc_super *, SEL))objc_msgSendSuper)(&sup, sel);
}
static void fo_commit(id, SEL) { st_other++; }
static int st_nest = 0;
static id st_inner = nil;
static void fn_commit(id self, SEL sel) {   /* commits another buffer from inside its own commit, then its super's */
  st_nest++;
  [st_inner commit];
  struct objc_super sup = { self, class_getSuperclass(object_getClass(self)) };
  ((void (*)(struct objc_super *, SEL))objc_msgSendSuper)(&sup, sel);
}

static int selftest_body(Dsv41tlRow *rows, uint32_t cap);

static Class st_class(const char *name, Class super) {
  Class c = objc_getClass(name);
  if (c) return c;
  c = objc_allocateClassPair(super, name, 0);
  return c;
}

int dsv41tl_selftest(Dsv41tlRow *rows, uint32_t cap) {
  static int done = -1;   /* once per process: the hooks it installs stay */
  if (done >= 0) return done;
  Class base = objc_getClass("Dsv41tlFakeBase");
  if (base == Nil) {
    base = st_class("Dsv41tlFakeBase", objc_getClass("NSObject"));
    class_addMethod(base, @selector(commit), (IMP)fb_commit, "v@:");
    class_addMethod(base, @selector(commandQueue), (IMP)fb_queue, "@@:");
    class_addMethod(base, @selector(addCompletedHandler:), (IMP)fb_handler, "v@:@?");
    class_addMethod(base, @selector(GPUStartTime), (IMP)fb_start, "d@:");
    class_addMethod(base, @selector(GPUEndTime), (IMP)fb_end, "d@:");
    class_addMethod(base, @selector(status), (IMP)fb_status, "Q@:");
    objc_registerClassPair(base);
    Class sub = st_class("Dsv41tlFakeSub", base);
    class_addMethod(sub, @selector(commit), (IMP)fs_commit, "v@:");
    objc_registerClassPair(sub);
    Class sib = st_class("Dsv41tlFakeSib", base);
    objc_registerClassPair(sib);
    Class other = st_class("Dsv41tlFakeOther", objc_getClass("NSObject"));
    class_addMethod(other, @selector(commit), (IMP)fo_commit, "v@:");
    class_addMethod(other, @selector(commandQueue), (IMP)fb_queue, "@@:");
    class_addMethod(other, @selector(addCompletedHandler:), (IMP)fb_handler, "v@:@?");
    class_addMethod(other, @selector(GPUStartTime), (IMP)fb_start, "d@:");
    class_addMethod(other, @selector(GPUEndTime), (IMP)fb_end, "d@:");
    class_addMethod(other, @selector(status), (IMP)fb_status, "Q@:");
    objc_registerClassPair(other);
    Class nest = st_class("Dsv41tlFakeNest", base);
    class_addMethod(nest, @selector(commit), (IMP)fn_commit, "v@:");
    objc_registerClassPair(nest);
  }
  done = selftest_body(rows, cap);
  return done;
}

static int selftest_body(Dsv41tlRow *rows, uint32_t cap) {
  st_queue = (id)objc_getClass("Dsv41tlFakeBase");
  id sub = [[objc_getClass("Dsv41tlFakeSub") alloc] init];
  id sib = [[objc_getClass("Dsv41tlFakeSib") alloc] init];
  id other = [[objc_getClass("Dsv41tlFakeOther") alloc] init];
  st_base = st_sub = st_other = 0;
  if (dsv41tl_install((void *)sub, rows, cap) != 0) return 1;
  uint64_t st[4];
  // untagged: the originals run, nothing recorded, the entry counted
  [sub commit];
  dsv41tl_hook_stats(st);
  if (st_sub != 1 || st_base != 1 || st[0] != 1 || g_committed.load() != 0) return 2;
  // tagged: one row for one commit (the subclass's [super commit] runs FakeBase's original, not a second record)
  dsv41tl_tag(7);
  [sub commit];
  if (st_sub != 2 || st_base != 2 || g_committed.load() != 1) return 3;
  if (rows[0].tag != 7 || rows[0].queue != 1 || rows[0].gpu_start != 1000000000ull || rows[0].gpu_end != 2000000000ull || rows[0].status != 4) return 4;
  // a sibling class inheriting the hooked base's -commit is recorded through it
  [sib commit];
  if (st_base != 3 || g_committed.load() != 2) return 5;
  // an unrelated class is not, until a buffer of it is hooked
  [other commit];
  if (st_other != 1 || g_committed.load() != 2) return 6;
  if (dsv41tl_rehook((void *)other) != 1) return 7;
  [other commit];
  if (st_other != 2 || g_committed.load() != 3) return 8;
  dsv41tl_hook_stats(st);
  if (st[0] != 4 || st[1] != 3 || st[2] != 0 || st[3] < 3) return 9;
  // a commit of another buffer made from inside a commit is its own outermost call: recorded, its original run
  id nest = [[objc_getClass("Dsv41tlFakeNest") alloc] init];
  st_inner = sib;
  if (dsv41tl_rehook((void *)nest) != 1) return 10;
  int base0 = st_base;
  [nest commit];
  if (st_nest != 1 || st_base != base0 + 2 || g_committed.load() != 5) return 11;
  [nest release];
  dsv41tl_tag(0);
  [sub release];
  [sib release];
  [other release];
  return 0;
}
