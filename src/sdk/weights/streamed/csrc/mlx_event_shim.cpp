// Event-gated pass-through primitives over MLX 0.32.2 (see mlx_event_shim.h): the DSV4.1 program's q3ev4 extension
// (lookahead4/ext/q3ev4.cpp) with its nanobind surface replaced by a C ABI over mlx-c handles.
//
// GPU wait: eval_gpu ends the stream's open compute pass and encodes MTLCommandBuffer.encodeWaitForEvent:value: into
// the stream's CURRENT command buffer (the Metal calls of MLX's own CommandEncoder::wait_event; mx::Event is not
// exported from libmlx, so the MTLSharedEvent is ours), then opens the next compute pass in the same command buffer
// and records the outputs as that pass's outputs. Every later pass of the buffer comes after the wait, and a
// consumer MLX puts into a later command buffer reads an output whose last producer is the post-wait pass, so its
// encoder waits on that pass's fence: the wait gates every consumer whatever MLX's commit points are.
// CPU wait: the evaluating thread waits for the host word before returning, so every consumer is dispatched after
// it (q3ev4 queued the wait on the stream's worker instead; this keeps std::function out of the libmlx boundary).
#include <atomic>
#include <chrono>
#include <cstdint>
#include <exception>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "mlx/array.h"
#include "mlx/backend/metal/device.h"
#include "mlx/c/private/array.h"
#include "mlx/c/private/stream.h"
#include "mlx/primitives.h"
#include "mlx_event_shim.h"

namespace mx = mlx::core;

namespace {

enum Kind : int { KIND_METAL = 1, KIND_HOST = 2 };

// Kept for the process life: primitives and the pool hold raw pointers.
struct EventRec {
  explicit EventRec(int k) : kind(k) {}
  int kind;
  MTL::SharedEvent* mtl = nullptr;
  int64_t* word = nullptr;
  int64_t timeout_ns = 0;
};

std::mutex g_mu;
std::vector<EventRec*> g_events;  // id - 1 -> record
std::atomic<int64_t> g_gpu_waits{0}, g_gpu_signals{0}, g_host_ready{0}, g_host_blocked{0}, g_host_timeouts{0},
    g_host_wait_ns{0}, g_cpu_passthrough{0};
thread_local std::string g_error;

int32_t add(EventRec* rec) {
  std::lock_guard<std::mutex> lk(g_mu);
  g_events.push_back(rec);
  return static_cast<int32_t>(g_events.size());
}

EventRec* lookup(int32_t id) {
  std::lock_guard<std::mutex> lk(g_mu);
  if (id < 1 || id > static_cast<int32_t>(g_events.size())) return nullptr;
  return g_events[id - 1];
}

void host_wait(const EventRec& ev, uint64_t value) {
  if (static_cast<uint64_t>(__atomic_load_n(ev.word, __ATOMIC_ACQUIRE)) >= value) {
    g_host_ready++;
    return;
  }
  g_host_blocked++;
  auto t0 = std::chrono::steady_clock::now();
  int spins = 0;
  while (static_cast<uint64_t>(__atomic_load_n(ev.word, __ATOMIC_ACQUIRE)) < value) {
    if (++spins < 256) {
      std::this_thread::yield();
      continue;
    }
    std::this_thread::sleep_for(std::chrono::microseconds(20));
    auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - t0).count();
    if (ns > ev.timeout_ns) {
      g_host_timeouts++;
      return;
    }
  }
  g_host_wait_ns +=
      std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - t0).count();
}

class WaitEvent : public mx::Primitive {
 public:
  WaitEvent(mx::Stream s, EventRec* ev, uint64_t value, int n_gated, bool track_inputs)
      : mx::Primitive(s), ev_(ev), value_(value), n_gated_(n_gated), track_inputs_(track_inputs) {}

  void eval_cpu(const std::vector<mx::array>& inputs, std::vector<mx::array>& outputs) override {
    for (int i = 0; i < n_gated_; i++) outputs[i].copy_shared_buffer(inputs[i]);
    if (ev_->kind != KIND_HOST) {
      g_cpu_passthrough++;
      return;
    }
    host_wait(*ev_, value_);
  }

  // A host event never reaches a GPU stream (refused when the wait is built).
  void eval_gpu(const std::vector<mx::array>& inputs, std::vector<mx::array>& outputs) override {
    for (int i = 0; i < n_gated_; i++) outputs[i].copy_shared_buffer(inputs[i]);
    if (ev_->kind != KIND_METAL) return;
    auto& enc = mx::metal::get_command_encoder(stream());
    enc.end_encoding();
    enc.get_command_buffer()->encodeWait(ev_->mtl, value_);
    enc.barrier();
    if (track_inputs_) {
      for (int i = 0; i < n_gated_; i++) enc.set_input_array(inputs[i], 0);
    }
    for (int i = 0; i < n_gated_; i++) enc.register_output_array(outputs[i]);
    g_gpu_waits++;
  }

  const char* name() const override {
    return "Dsv41WaitEvent";
  }

 private:
  EventRec* ev_;
  uint64_t value_;
  int n_gated_;
  bool track_inputs_;
};

class SignalEvent : public mx::Primitive {
 public:
  SignalEvent(mx::Stream s, EventRec* ev, uint64_t value, int n_out)
      : mx::Primitive(s), ev_(ev), value_(value), n_out_(n_out) {}

  void eval_cpu(const std::vector<mx::array>& inputs, std::vector<mx::array>& outputs) override {
    for (int i = 0; i < n_out_; i++) outputs[i].copy_shared_buffer(inputs[i]);
  }

  void eval_gpu(const std::vector<mx::array>& inputs, std::vector<mx::array>& outputs) override {
    for (int i = 0; i < n_out_; i++) outputs[i].copy_shared_buffer(inputs[i]);
    if (ev_->kind != KIND_METAL) return;
    auto& enc = mx::metal::get_command_encoder(stream());
    enc.end_encoding();
    enc.get_command_buffer()->encodeSignalEvent(ev_->mtl, value_);
    g_gpu_signals++;
  }

  const char* name() const override {
    return "Dsv41SignalEvent";
  }

 private:
  EventRec* ev_;
  uint64_t value_;
  int n_out_;
};

int fail(const char* what) {
  g_error = what;
  return -1;
}

}  // namespace

extern "C" {

int32_t dsv41ev_abi(void) {
  return DSV41EV_ABI;
}

const char* dsv41ev_last_error(void) {
  return g_error.c_str();
}

int32_t dsv41ev_create_metal(uint64_t start, uint64_t* object) {
  try {
    MTL::SharedEvent* e = mx::metal::device(mx::Device::gpu).mtl_device()->newSharedEvent();
    if (!e) {
      g_error = "newSharedEvent failed";
      return 0;
    }
    e->setSignaledValue(start);
    auto* rec = new EventRec(KIND_METAL);
    rec->mtl = e;
    *object = reinterpret_cast<uint64_t>(e);
    return add(rec);
  } catch (...) {
    g_error = "the MLX GPU device is unavailable";
    return 0;
  }
}

int32_t dsv41ev_create_host(int64_t* word, int64_t timeout_ns) {
  if (!word || (reinterpret_cast<uintptr_t>(word) % alignof(int64_t)) || timeout_ns <= 0) {
    g_error = "a host event needs an aligned int64 word and a positive timeout";
    return 0;
  }
  auto* rec = new EventRec(KIND_HOST);
  rec->word = word;
  rec->timeout_ns = timeout_ns;
  return add(rec);
}

int dsv41ev_wait(const mlx_array* xs, size_t n, int32_t event, uint64_t value, const mlx_array* deps, size_t n_deps,
                 bool track_inputs, mlx_stream s, mlx_array* outs) {
  try {
    if (n == 0) return fail("nothing to gate");
    if (value == 0) return fail("values start at 1 (a fresh event holds 0)");
    EventRec* ev = lookup(event);
    if (!ev) return fail("unknown event id");
    const mx::Stream& stream = mlx_stream_get_(s);
    if (ev->kind == KIND_HOST && stream.device.type == mx::Device::DeviceType::gpu)
      return fail("a host event cannot gate GPU work");
    std::vector<mx::Shape> shapes;
    std::vector<mx::Dtype> dtypes;
    std::vector<mx::array> inputs;
    for (size_t i = 0; i < n; i++) {
      const mx::array& x = mlx_array_get_(xs[i]);
      shapes.push_back(x.shape());
      dtypes.push_back(x.dtype());
      inputs.push_back(x);
    }
    for (size_t i = 0; i < n_deps; i++) inputs.push_back(mlx_array_get_(deps[i]));
    auto res = mx::array::make_arrays(
        std::move(shapes), dtypes,
        std::make_shared<WaitEvent>(stream, ev, value, static_cast<int>(n), track_inputs), inputs);
    for (size_t i = 0; i < n; i++) mlx_array_set_(outs[i], std::move(res[i]));
    return 0;
  } catch (const std::exception& e) {
    return fail(e.what());
  } catch (...) {
    return fail("dsv41ev_wait failed");
  }
}

int dsv41ev_signal(const mlx_array* xs, size_t n, int32_t event, uint64_t value, mlx_stream s, mlx_array* outs) {
  try {
    if (n == 0 || value == 0) return fail("arrays and a value >= 1");
    EventRec* ev = lookup(event);
    if (!ev) return fail("unknown event id");
    if (ev->kind == KIND_HOST) return fail("only the pool signals a host event");
    const mx::Stream& stream = mlx_stream_get_(s);
    std::vector<mx::Shape> shapes;
    std::vector<mx::Dtype> dtypes;
    std::vector<mx::array> inputs;
    for (size_t i = 0; i < n; i++) {
      const mx::array& x = mlx_array_get_(xs[i]);
      shapes.push_back(x.shape());
      dtypes.push_back(x.dtype());
      inputs.push_back(x);
    }
    auto res = mx::array::make_arrays(
        std::move(shapes), dtypes, std::make_shared<SignalEvent>(stream, ev, value, static_cast<int>(n)), inputs);
    for (size_t i = 0; i < n; i++) mlx_array_set_(outs[i], std::move(res[i]));
    return 0;
  } catch (const std::exception& e) {
    return fail(e.what());
  } catch (...) {
    return fail("dsv41ev_signal failed");
  }
}

uint64_t dsv41ev_value(int32_t event) {
  EventRec* ev = lookup(event);
  if (!ev) return 0;
  if (ev->kind == KIND_METAL) return ev->mtl->signaledValue();
  if (ev->kind == KIND_HOST) return static_cast<uint64_t>(__atomic_load_n(ev->word, __ATOMIC_ACQUIRE));
  return 0;
}

void dsv41ev_stats(int64_t out[8]) {
  out[0] = g_gpu_waits.load();
  out[1] = g_gpu_signals.load();
  out[2] = g_host_ready.load();
  out[3] = g_host_blocked.load();
  out[4] = g_host_timeouts.load();
  out[5] = g_host_wait_ns.load();
  out[6] = g_cpu_passthrough.load();
  std::lock_guard<std::mutex> lk(g_mu);
  out[7] = static_cast<int64_t>(g_events.size());
}

}  // extern "C"
