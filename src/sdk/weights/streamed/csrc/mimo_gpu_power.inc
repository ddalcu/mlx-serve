// MiMo X12 measurement (mimo/gpu-warm): the GPU's energy, DVFS-state residency and thermal-limit residency from IOReport
// (libIOReport, no root, read-only kernel counters) and the GPU's DVFS frequency table from the IORegistry (pmgr's
// "voltage-states9": OFF, then one state per P-state). Callers: the `gpuwarm` bench (test builds) and the MTP profile
// (profile builds) at a request's decode begin / end; no served path calls it. Every CoreFoundation, IOKit and
// IOReport function is resolved with dlsym (no link dependency). Kept in this translation unit because the host
// compiles a fixed list of the plugin's csrc files.
#include <dlfcn.h>

#include <cstdint>
#include <cstring>
#include <mutex>
#include <string>

namespace mimo_power {

// Opaque stand-ins (the CF / IOKit headers' own names are in this translation unit already, through metal-cpp).
typedef const void* Ref;        // Ref, Ref, Ref, Ref, Ref
typedef void* MRef;             // MRef
typedef unsigned int Port;      // Port, Port
typedef int Kret;               // Kret
typedef void* Subscription;     // IOReportSubscriptionRef

constexpr uint32_t kUtf8 = 0x08000100;

struct Api {
  // CoreFoundation
  Ref (*str)(const void*, const char*, uint32_t) = nullptr;
  bool (*cstr)(Ref, char*, long, uint32_t) = nullptr;
  void (*release)(Ref) = nullptr;
  const void* (*dict_get)(Ref, const void*) = nullptr;
  long (*arr_count)(Ref) = nullptr;
  const void* (*arr_at)(Ref, long) = nullptr;
  long (*data_len)(Ref) = nullptr;
  const uint8_t* (*data_ptr)(Ref) = nullptr;
  // IOKit
  MRef (*matching)(const char*) = nullptr;
  Kret (*services)(Port, Ref, Port*) = nullptr;
  Port (*next)(Port) = nullptr;
  Kret (*entry_name)(Port, char*) = nullptr;
  Ref (*property)(Port, Ref, const void*, uint32_t) = nullptr;
  Kret (*obj_release)(Port) = nullptr;
  // IOReport
  MRef (*channels)(Ref, Ref, uint64_t, uint64_t, uint64_t) = nullptr;
  void (*merge)(MRef, MRef, Ref) = nullptr;
  Subscription (*subscribe)(void*, MRef, MRef*, uint64_t, Ref) = nullptr;
  Ref (*samples)(Subscription, MRef, Ref) = nullptr;
  Ref (*delta)(Ref, Ref, Ref) = nullptr;
  Ref (*group)(Ref) = nullptr;
  Ref (*subgroup)(Ref) = nullptr;
  Ref (*name)(Ref) = nullptr;
  Ref (*unit)(Ref) = nullptr;
  int64_t (*simple)(Ref, int32_t) = nullptr;
  int32_t (*state_count)(Ref) = nullptr;
  Ref (*state_name)(Ref, int32_t) = nullptr;
  int64_t (*residency)(Ref, int32_t) = nullptr;
};

std::mutex g_mu;
Api g_api;
bool g_open = false;
Subscription g_sub = nullptr;
MRef g_subbed = nullptr;
thread_local std::string g_error;

int fail(const char* what) {
  g_error = what;
  return -1;
}

template <typename F>
bool sym(void* lib, const char* name, F& out) {
  out = reinterpret_cast<F>(dlsym(lib, name));
  return out != nullptr;
}

bool load(Api& a) {
  void* cf = dlopen("/System/Library/Frameworks/CoreFoundation.framework/CoreFoundation", RTLD_LAZY | RTLD_LOCAL);
  void* iok = dlopen("/System/Library/Frameworks/IOKit.framework/IOKit", RTLD_LAZY | RTLD_LOCAL);
  void* ior = dlopen("/usr/lib/libIOReport.dylib", RTLD_LAZY | RTLD_LOCAL);
  if (cf == nullptr || iok == nullptr || ior == nullptr) return false;
  return sym(cf, "CFStringCreateWithCString", a.str) && sym(cf, "CFStringGetCString", a.cstr) &&
         sym(cf, "CFRelease", a.release) && sym(cf, "CFDictionaryGetValue", a.dict_get) &&
         sym(cf, "CFArrayGetCount", a.arr_count) && sym(cf, "CFArrayGetValueAtIndex", a.arr_at) &&
         sym(cf, "CFDataGetLength", a.data_len) && sym(cf, "CFDataGetBytePtr", a.data_ptr) &&
         sym(iok, "IOServiceMatching", a.matching) && sym(iok, "IOServiceGetMatchingServices", a.services) &&
         sym(iok, "IOIteratorNext", a.next) && sym(iok, "IORegistryEntryGetName", a.entry_name) &&
         sym(iok, "IORegistryEntryCreateCFProperty", a.property) && sym(iok, "IOObjectRelease", a.obj_release) &&
         sym(ior, "IOReportCopyChannelsInGroup", a.channels) && sym(ior, "IOReportMergeChannels", a.merge) &&
         sym(ior, "IOReportCreateSubscription", a.subscribe) && sym(ior, "IOReportCreateSamples", a.samples) &&
         sym(ior, "IOReportCreateSamplesDelta", a.delta) && sym(ior, "IOReportChannelGetGroup", a.group) &&
         sym(ior, "IOReportChannelGetSubGroup", a.subgroup) && sym(ior, "IOReportChannelGetChannelName", a.name) &&
         sym(ior, "IOReportChannelGetUnitLabel", a.unit) && sym(ior, "IOReportSimpleGetIntegerValue", a.simple) &&
         sym(ior, "IOReportStateGetCount", a.state_count) && sym(ior, "IOReportStateGetNameForIndex", a.state_name) &&
         sym(ior, "IOReportStateGetResidency", a.residency);
}

// A CF string's text into buf ("" when null or not convertible).
const char* text(Ref s, char* buf, long cap) {
  buf[0] = 0;
  if (s != nullptr) g_api.cstr(s, buf, cap, kUtf8);
  return buf;
}

}  // namespace mimo_power

extern "C" {

#define MIMO_POWER_STATES 32

// One interval between two samples. Residencies in seconds (IOReport's 24 MHz ticks).
typedef struct {
  double gpu_energy_j;   // "Energy Model" / "GPU Energy" (nJ), else "GPU" (mJ); -1 when neither is present
  double seconds;        // the GPUPH channel's total residency (the interval)
  uint32_t n_states;     // GPUPH states: [0] OFF, [k] Pk
  uint32_t has_cltm;     // the GPU_CLTM channel was present
  double state_s[MIMO_POWER_STATES];
  double cltm_s;         // GPU_CLTM residency in any state but NO_CLTM (closed-loop thermal limiting)
  char state_names[MIMO_POWER_STATES][8];
} MimoPowerDelta;

const char* mimo_power_last_error(void) {
  return mimo_power::g_error.c_str();
}

// Opens the process's one IOReport subscription (the GPU energy channel, the GPU P-state and thermal-state channels).
// Idempotent. 0, or -1 (IOReport unavailable on this system).
int mimo_power_open(void) {
  using namespace mimo_power;
  std::lock_guard<std::mutex> lk(g_mu);
  if (g_open) return 0;
  if (!load(g_api)) return fail("CoreFoundation / IOKit / IOReport symbols unavailable");
  Api& a = g_api;
  Ref energy = a.str(nullptr, "Energy Model", kUtf8);
  Ref gpu = a.str(nullptr, "GPU Stats", kUtf8);
  Ref perf = a.str(nullptr, "GPU Performance States", kUtf8);
  Ref cltm = a.str(nullptr, "CLTM-induced GPU Performance States", kUtf8);
  MRef want = a.channels(energy, nullptr, 0, 0, 0);
  MRef p = a.channels(gpu, perf, 0, 0, 0);
  MRef t = a.channels(gpu, cltm, 0, 0, 0);
  if (want == nullptr || p == nullptr) {
    for (Ref x : {energy, gpu, perf, cltm}) a.release(x);
    for (Ref x : {static_cast<Ref>(want), static_cast<Ref>(p), static_cast<Ref>(t)})
      if (x != nullptr) a.release(x);
    return fail("IOReport has no Energy Model or GPU Performance States channels");
  }
  a.merge(want, p, nullptr);
  a.release(p);
  if (t != nullptr) {
    a.merge(want, t, nullptr);
    a.release(t);
  }
  MRef subbed = nullptr;
  Subscription sub = a.subscribe(nullptr, want, &subbed, 0, nullptr);
  for (Ref x : {energy, gpu, perf, cltm}) a.release(x);
  if (sub == nullptr || subbed == nullptr) return fail("IOReportCreateSubscription failed");
  g_sub = sub;
  g_subbed = subbed;
  g_open = true;
  return 0;
}

// A cumulative sample (release with mimo_power_free), or null.
void* mimo_power_sample(void) {
  using namespace mimo_power;
  if (!g_open) {
    fail("not open");
    return nullptr;
  }
  return const_cast<void*>(g_api.samples(g_sub, g_subbed, nullptr));
}

void mimo_power_free(void* s) {
  if (s != nullptr && mimo_power::g_open) mimo_power::g_api.release(s);
}

// The interval from sample a to sample b. 0, or -1.
int mimo_power_delta(void* sa, void* sb, MimoPowerDelta* out) {
  using namespace mimo_power;
  std::memset(out, 0, sizeof(*out));
  out->gpu_energy_j = -1;
  if (!g_open || sa == nullptr || sb == nullptr) return fail("no samples");
  Api& a = g_api;
  Ref d = a.delta(sa, sb, nullptr);
  if (d == nullptr) return fail("IOReportCreateSamplesDelta failed");
  Ref key = a.str(nullptr, "IOReportChannels", kUtf8);
  Ref arr = a.dict_get(d, key);
  a.release(key);
  if (arr == nullptr) {
    a.release(d);
    return fail("the delta has no IOReportChannels");
  }
  double gpu_nj = -1, gpu_mj = -1;
  char g[64], sg[96], nm[96], un[32], st[32];
  const long n = a.arr_count(arr);
  for (long i = 0; i < n; i++) {
    Ref ch = a.arr_at(arr, i);
    text(a.group(ch), g, sizeof g);
    text(a.name(ch), nm, sizeof nm);
    if (std::strcmp(g, "Energy Model") == 0) {
      text(a.unit(ch), un, sizeof un);
      const double v = static_cast<double>(a.simple(ch, 0));
      if (std::strcmp(nm, "GPU Energy") == 0 && std::strcmp(un, "nJ") == 0) gpu_nj = v;
      if (std::strcmp(nm, "GPU") == 0 && std::strcmp(un, "mJ") == 0) gpu_mj = v;
      continue;
    }
    if (std::strcmp(g, "GPU Stats") != 0) continue;
    text(a.subgroup(ch), sg, sizeof sg);
    const int32_t ns = a.state_count(ch);
    if (std::strcmp(sg, "GPU Performance States") == 0 && std::strcmp(nm, "GPUPH") == 0) {
      out->n_states = static_cast<uint32_t>(ns < MIMO_POWER_STATES ? ns : MIMO_POWER_STATES);
      for (uint32_t k = 0; k < out->n_states; k++) {
        out->state_s[k] = static_cast<double>(a.residency(ch, static_cast<int32_t>(k))) / 24e6;
        out->seconds += out->state_s[k];
        text(a.state_name(ch, static_cast<int32_t>(k)), st, sizeof st);
        std::strncpy(out->state_names[k], st, sizeof(out->state_names[k]) - 1);
      }
    } else if (std::strcmp(sg, "CLTM-induced GPU Performance States") == 0) {
      out->has_cltm = 1;
      for (int32_t k = 0; k < ns; k++) {
        text(a.state_name(ch, k), st, sizeof st);
        if (std::strcmp(st, "NO_CLTM") != 0) out->cltm_s += static_cast<double>(a.residency(ch, k)) / 24e6;
      }
    }
  }
  a.release(d);
  out->gpu_energy_j = gpu_nj >= 0 ? gpu_nj * 1e-9 : (gpu_mj >= 0 ? gpu_mj * 1e-3 : -1);
  if (out->n_states == 0) return fail("no GPUPH channel in the delta");
  return 0;
}

// The GPU's DVFS table (MHz) from pmgr's "voltage-states9" (entry 0 = OFF), up to cap entries. The count, or -1.
int mimo_power_freqs(uint32_t* mhz, uint32_t cap) {
  using namespace mimo_power;
  {
    std::lock_guard<std::mutex> lk(g_mu);
    if (g_api.str == nullptr && !load(g_api)) return fail("CoreFoundation / IOKit symbols unavailable");
  }
  Api& a = g_api;
  Port it = 0;
  if (a.services(0, a.matching("AppleARMIODevice"), &it) != 0) return fail("IOServiceGetMatchingServices failed");
  int count = -1;
  Ref key = a.str(nullptr, "voltage-states9", kUtf8);
  for (Port e = a.next(it); e != 0; e = a.next(it)) {
    char name[128] = {0};
    if (count < 0 && a.entry_name(e, name) == 0 && std::strcmp(name, "pmgr") == 0) {
      Ref p = a.property(e, key, nullptr, 0);
      if (p != nullptr) {
        const long len = a.data_len(p);
        const uint8_t* b = a.data_ptr(p);
        count = 0;
        for (long off = 0; off + 8 <= len && static_cast<uint32_t>(count) < cap; off += 8) {
          uint32_t hz;
          std::memcpy(&hz, b + off, 4);
          mhz[count++] = hz / 1000000u;
        }
        a.release(p);
      }
    }
    a.obj_release(e);
  }
  a.obj_release(it);
  a.release(key);
  if (count < 0) return fail("pmgr has no voltage-states9");
  return count;
}

}  // extern "C"
