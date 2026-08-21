//! Lock-free, zero-when-off instrumentation core for mlx-serve.
//!
//! Design contract (the whole point — mlx-serve's pitch is best-in-class
//! serving performance, so observability must never touch that):
//!
//!   * OFF  — a single `?*Metrics` null-check per REQUEST (never per token).
//!            No allocation, no atomics, no work. Unmeasurable.
//!   * ON   — a handful of RELAXED atomic adds per request, recorded at the
//!            `finishSlot` funnel which is already off the per-token decode
//!            path. The inference thread never locks, waits, or blocks on the
//!            metrics subsystem. Gauges are sampled on a separate thread;
//!            `/metrics` renders on the scrape connection thread.
//!
//! Thread-safety: every write is `std.atomic.Value(u64)` with `.monotonic`
//! (relaxed) ordering. Writers are N connection threads + the inference
//! thread; readers (render) tolerate a slightly inconsistent cross-counter
//! snapshot — fine for metrics. No mutex anywhere on the write path.

const std = @import("std");

// ---------------------------------------------------------------------------
// Histogram bounds (comptime constants shared by Metrics.init)
// ---------------------------------------------------------------------------

/// 10 latency buckets spanning 10 ms → 10 s, stored in nanoseconds.
/// The Prometheus renderer divides by 1e9 before emitting `le="X.XXX"`.
const LATENCY_BOUNDS_NS: [10]u64 = .{
    10_000_000, // 10 ms
    25_000_000, // 25 ms
    50_000_000, // 50 ms
    100_000_000, // 100 ms
    250_000_000, // 250 ms
    500_000_000, // 500 ms
    1_000_000_000, // 1 s
    2_500_000_000, // 2.5 s
    5_000_000_000, // 5 s
    10_000_000_000, // 10 s
};

/// 8 token-count buckets: 32 → 8 192 (raw integers, no scaling).
const TOKEN_BOUNDS: [8]u64 = .{ 32, 128, 256, 512, 1024, 2048, 4096, 8192 };

// ---------------------------------------------------------------------------
// Metrics — the single global struct allocated when --metrics is on
// ---------------------------------------------------------------------------

/// All instrumented metrics for one mlx-serve instance.
/// Zero-allocation after init; every field is a lock-free primitive.
pub const Metrics = struct {
    // Latency histograms (observe in nanoseconds; rendered as seconds)
    ttft_ns: Histogram(10), // time to first token
    e2e_latency_ns: Histogram(10), // end-to-end request latency
    prefill_time_ns: Histogram(10), // prefill phase
    decode_time_ns: Histogram(10), // decode phase

    // Per-request token histograms (raw counts)
    prompt_tokens_hist: Histogram(8),
    output_tokens_hist: Histogram(8),

    // Counters
    prompt_tokens_total: Counter,
    /// Prompt tokens actually FORWARDED through the model during prefill, i.e.
    /// `prompt_tokens - cached_tokens`. Divide the prefill-time histogram by
    /// THIS, never by `prompt_tokens_total`: with the hot prefix cache warm, most
    /// billed prompt tokens are restored, not computed, and the ratio inflates
    /// prefill tok/s by `prompt/(prompt-cached)` (measured 10.6x on a warm
    /// multi-turn 35B MoE session). `vllm:prompt_tokens_total` keeps the vLLM
    /// meaning (all billed prompt tokens) for dashboard compatibility.
    prefill_tokens_total: Counter,
    /// Prompt tokens served straight from the hot prefix cache. Token-level
    /// companion to `prefix_cache_hits_total`, which counts REQUESTS that hit.
    /// Invariant: prefill_tokens_total + prefix_cache_tokens_total == prompt_tokens_total.
    prefix_cache_tokens_total: Counter,
    generation_tokens_total: Counter,
    requests_success_total: Counter,
    requests_cancelled_total: Counter,
    prefix_cache_queries_total: Counter,
    prefix_cache_hits_total: Counter,

    // Gauges (sampled by the background sampler thread; single writer)
    requests_running: Gauge,
    requests_waiting: Gauge,
    gpu_utilization_pct: Gauge, // 0–100
    memory_mb: Gauge, // megabytes (phys_footprint)
    // Real-time throughput source: completed generation tokens PLUS tokens
    // generated so far by in-flight slots. The sampler thread sets this from
    // `generation_tokens_total` plus the scheduler's `inflight_generated_tokens`
    // aggregate (published race-free once per decode tick — no per-token write
    // on the inference path), so the dashboard can derive a live tok/s without
    // waiting for requests to complete. See server.zig sampleGauges +
    // liveGenerationTokens below.
    generation_tokens_live: Gauge,
    // Tokens forwarded so far by the prefill currently running, 0 when none is.
    // The OTHER half of the live picture: `generation_tokens_live` only moves
    // during decode, and the prompt-token counter / prefill-time histogram only
    // move when a request completes — so a long prefill was invisible. Set by
    // the sampler from `Scheduler.inflight_prefill_tokens` (published once per
    // prefill CHUNK, never per token).
    prefill_tokens_live: Gauge,
    // Slots currently in prefill. Flips as soon as prefill starts, so the panel
    // can name the phase without waiting for the first chunk's token count.
    requests_prefilling: Gauge,
    // MLX allocator split. `memory_mb` above is the whole process footprint;
    // these two say where it went. `mlx_active_bytes` is memory in USE,
    // `mlx_cache_bytes` is MLX's reclaimable buffer pool — memory the process
    // HOLDS but is not using, which MLX only returns to the OS on a
    // `mlx_clear_cache()` or when its own (enormous) GC limit trips.
    // `memory_mb - active` staying flat while `cache` climbs is the #110
    // signature; without these two the gap has no name on any surface.
    mlx_active_bytes: Gauge,
    mlx_cache_bytes: Gauge,
    // ANE prefill offload (`--ane-prefill`, A8): what the Neural Engine is
    // holding right now — int8 weight-copy bytes and covered layer count
    // (mlp + gdn) summed across resident engines. Zero whenever the offload
    // is off or no covered model is resident (zero-when-off invariant).
    ane_int8_bytes: Gauge,
    ane_layers: Gauge,

    pub fn init() Metrics {
        return .{
            .ttft_ns = Histogram(10).init(LATENCY_BOUNDS_NS),
            .e2e_latency_ns = Histogram(10).init(LATENCY_BOUNDS_NS),
            .prefill_time_ns = Histogram(10).init(LATENCY_BOUNDS_NS),
            .decode_time_ns = Histogram(10).init(LATENCY_BOUNDS_NS),
            .prompt_tokens_hist = Histogram(8).init(TOKEN_BOUNDS),
            .output_tokens_hist = Histogram(8).init(TOKEN_BOUNDS),
            .prompt_tokens_total = Counter.init(),
            .prefill_tokens_total = Counter.init(),
            .prefix_cache_tokens_total = Counter.init(),
            .generation_tokens_total = Counter.init(),
            .requests_success_total = Counter.init(),
            .requests_cancelled_total = Counter.init(),
            .prefix_cache_queries_total = Counter.init(),
            .prefix_cache_hits_total = Counter.init(),
            .requests_running = Gauge.init(),
            .requests_waiting = Gauge.init(),
            .gpu_utilization_pct = Gauge.init(),
            .memory_mb = Gauge.init(),
            .generation_tokens_live = Gauge.init(),
            .prefill_tokens_live = Gauge.init(),
            .requests_prefilling = Gauge.init(),
            .mlx_active_bytes = Gauge.init(),
            .mlx_cache_bytes = Gauge.init(),
            .ane_int8_bytes = Gauge.init(),
            .ane_layers = Gauge.init(),
        };
    }

    /// Record per-request metrics at slot completion.
    /// Called exactly once per request from the `finishSlot` funnel.
    ///
    /// On success ("stop" | "length" | "tool_calls"): updates all latency
    /// histograms, token histograms, and counters.
    ///
    /// On cancel ("cancelled"): only increments `requests_cancelled_total`.
    /// Latency histograms are NOT touched — a cancelled slot's decode_ns is
    /// zero or garbage and would poison the distribution.
    ///
    /// Other reasons (Zig error names from markError): silently ignored for
    /// now; these are rare and already visible in --log-level warn output.
    ///
    /// Parameters:
    ///   - `real_ttft_ns`: time from request arrival (Slot.init, pre-queue-wait)
    ///     to first token = queue_wait + prefill. Captured directly at prefill
    ///     completion (`Slot.first_token_ns`) — exact, never derived by
    ///     subtraction. For single-user servers queue wait ≈ 0, so this equals
    ///     `prefill_ns`; for concurrent servers it correctly includes queuing.
    ///   - `prefill_ns`: model prefill phase only (no queue wait).
    ///   - `decode_ns`: decode phase duration.
    ///   - e2e latency = real_ttft_ns + decode_ns (= queue_wait + prefill + decode).
    pub fn recordRequest(
        self: *Metrics,
        finish_reason: []const u8,
        real_ttft_ns: u64,
        prefill_ns: u64,
        decode_ns: u64,
        prompt_tokens: u32,
        completion_tokens: u32,
        cached_tokens: u32,
    ) void {
        // Wire prefix-cache stats unconditionally: every request queries the cache.
        self.prefix_cache_queries_total.inc();
        if (cached_tokens > 0) self.prefix_cache_hits_total.inc();

        const is_success = std.mem.eql(u8, finish_reason, "stop") or
            std.mem.eql(u8, finish_reason, "length") or
            std.mem.eql(u8, finish_reason, "tool_calls");

        if (!is_success) {
            if (std.mem.eql(u8, finish_reason, "cancelled"))
                self.requests_cancelled_total.inc();
            // Error reasons (OOM, etc.) are rare; logged by the scheduler.
            return;
        }

        // Success path — record all instrumented metrics.
        self.requests_success_total.inc();
        self.prompt_tokens_total.add(prompt_tokens);
        // Saturating: a restored prefix can never exceed the prompt, but never
        // let a bad accounting wrap the counter.
        const restored = @min(cached_tokens, prompt_tokens);
        self.prefill_tokens_total.add(prompt_tokens - restored);
        self.prefix_cache_tokens_total.add(restored);
        self.generation_tokens_total.add(completion_tokens);

        self.ttft_ns.observe(real_ttft_ns);
        self.prefill_time_ns.observe(prefill_ns);
        self.decode_time_ns.observe(decode_ns);
        // e2e = queue_wait + prefill + decode  (real_ttft_ns already includes queue)
        self.e2e_latency_ns.observe(real_ttft_ns + decode_ns);

        self.prompt_tokens_hist.observe(@as(u64, prompt_tokens));
        self.output_tokens_hist.observe(@as(u64, completion_tokens));
    }
};

/// Pure: the live-tokens gauge value = completed generation tokens PLUS the
/// tokens generated so far by still-decoding slots. The scheduler publishes the
/// in-flight aggregate (`inflight_generated_tokens`) once per decode tick; the
/// sampler thread calls this each tick to set `generation_tokens_live`. Kept
/// here (not inline in the sampler) so the arithmetic is unit-testable and the
/// "at rest ⇒ live == total" invariant is pinned without a running server.
pub fn liveGenerationTokens(m: *const Metrics, inflight: u64) u64 {
    return m.generation_tokens_total.load() + inflight;
}

// ---------------------------------------------------------------------------
// renderPrometheus — Prometheus text exposition format (OpenMetrics-compat)
// ---------------------------------------------------------------------------

/// Write all metrics in Prometheus text format to `w`.
/// Called only on the scrape connection thread — never on the inference path.
pub fn renderPrometheus(m: *const Metrics, w: *std.Io.Writer) !void {
    const ns_to_s = 1.0 / 1_000_000_000.0;

    // --- Counters ---
    try writeCounter(w, "vllm:prompt_tokens_total", "Total prompt tokens processed", m.prompt_tokens_total.load());
    try writeCounter(w, "mlx_serve:prefill_tokens_total", "Prompt tokens actually forwarded through prefill (excludes prefix-cache restores) — the correct numerator for prefill tok/s", m.prefill_tokens_total.load());
    try writeCounter(w, "mlx_serve:prefix_cache_tokens_total", "Prompt tokens restored from the hot prefix cache instead of being computed", m.prefix_cache_tokens_total.load());
    try writeCounter(w, "vllm:generation_tokens_total", "Total generated tokens", m.generation_tokens_total.load());
    try writeCounter(w, "vllm:request_success_total", "Completed requests", m.requests_success_total.load());
    try writeCounter(w, "vllm:request_cancelled_total", "Requests cancelled by client disconnect", m.requests_cancelled_total.load());
    try writeCounter(w, "vllm:prefix_cache_queries_total", "Prefix cache lookup count", m.prefix_cache_queries_total.load());
    try writeCounter(w, "vllm:prefix_cache_hits_total", "Prefix cache hit count", m.prefix_cache_hits_total.load());

    // --- Gauges ---
    try writeGauge(w, "vllm:num_requests_running", "Number of requests currently being processed", m.requests_running.load());
    try writeGauge(w, "vllm:num_requests_waiting", "Number of requests waiting in the queue", m.requests_waiting.load());
    try writeGauge(w, "mlx_serve:gpu_utilization_pct", "GPU utilization percentage (IOKit AGXAccelerator)", m.gpu_utilization_pct.load());
    try writeGauge(w, "mlx_serve:memory_mb", "Server physical memory footprint in megabytes (phys_footprint)", m.memory_mb.load());
    try writeGauge(w, "mlx_serve:generation_tokens_live", "Generation tokens completed plus generated-so-far by in-flight slots (real-time tok/s source)", m.generation_tokens_live.load());
    try writeGauge(w, "mlx_serve:prefill_tokens_live", "Prompt tokens forwarded so far by the in-flight prefill (0 when idle; real-time prefill tok/s source)", m.prefill_tokens_live.load());
    try writeGauge(w, "mlx_serve:requests_prefilling", "Requests currently in the prefill phase", m.requests_prefilling.load());
    try writeGauge(w, "mlx_serve:mlx_active_bytes", "Bytes MLX's allocator currently has in use", m.mlx_active_bytes.load());
    try writeGauge(w, "mlx_serve:mlx_cache_bytes", "Bytes parked in MLX's reclaimable buffer pool (held by the process, not in use)", m.mlx_cache_bytes.load());
    try writeGauge(w, "mlx_serve:ane_int8_bytes", "Bytes of int8 weight copies held by the ANE prefill offload (0 when off)", m.ane_int8_bytes.load());
    try writeGauge(w, "mlx_serve:ane_layers", "Layers covered by compiled ANE prefill programs, mlp + gdn (0 when off)", m.ane_layers.load());

    // --- Latency histograms (nanoseconds → seconds) ---
    try writeHistogram(w, "vllm:time_to_first_token_seconds", "Time to first token in seconds", &m.ttft_ns, ns_to_s);
    try writeHistogram(w, "vllm:e2e_request_latency_seconds", "End-to-end request latency in seconds", &m.e2e_latency_ns, ns_to_s);
    try writeHistogram(w, "vllm:request_prefill_time_seconds", "Prefill phase latency in seconds", &m.prefill_time_ns, ns_to_s);
    try writeHistogram(w, "vllm:request_decode_time_seconds", "Decode phase latency in seconds", &m.decode_time_ns, ns_to_s);

    // --- Token-count histograms (raw counts, scale = 1.0) ---
    try writeHistogram(w, "vllm:request_prompt_tokens", "Per-request prompt token count distribution", &m.prompt_tokens_hist, 1.0);
    try writeHistogram(w, "vllm:request_generation_tokens", "Per-request output token count distribution", &m.output_tokens_hist, 1.0);
}

// ---------------------------------------------------------------------------
// renderJson — open JSON feed (drives the index-page live metrics panel)
// ---------------------------------------------------------------------------

/// Write all metrics as a JSON object to `w`.
/// Called only on the scrape connection thread.
pub fn renderJson(m: *const Metrics, w: *std.Io.Writer) !void {
    const ns_to_s = 1.0 / 1_000_000_000.0;

    try w.print(
        "{{\"counters\":{{" ++
            "\"prompt_tokens_total\":{d}," ++
            "\"prefill_tokens_total\":{d}," ++
            "\"prefix_cache_tokens_total\":{d}," ++
            "\"generation_tokens_total\":{d}," ++
            "\"requests_success_total\":{d}," ++
            "\"requests_cancelled_total\":{d}," ++
            "\"prefix_cache_queries_total\":{d}," ++
            "\"prefix_cache_hits_total\":{d}" ++
            "}},\"gauges\":{{" ++
            "\"requests_running\":{d}," ++
            "\"requests_waiting\":{d}," ++
            "\"gpu_utilization_pct\":{d}," ++
            "\"memory_mb\":{d}," ++
            "\"generation_tokens_live\":{d}," ++
            "\"prefill_tokens_live\":{d}," ++
            "\"requests_prefilling\":{d}," ++
            "\"mlx_active_bytes\":{d}," ++
            "\"mlx_cache_bytes\":{d}," ++
            "\"ane_int8_bytes\":{d}," ++
            "\"ane_layers\":{d}" ++
            "}},\"histograms\":{{",
        .{
            m.prompt_tokens_total.load(),
            m.prefill_tokens_total.load(),
            m.prefix_cache_tokens_total.load(),
            m.generation_tokens_total.load(),
            m.requests_success_total.load(),
            m.requests_cancelled_total.load(),
            m.prefix_cache_queries_total.load(),
            m.prefix_cache_hits_total.load(),
            m.requests_running.load(),
            m.requests_waiting.load(),
            m.gpu_utilization_pct.load(),
            m.memory_mb.load(),
            m.generation_tokens_live.load(),
            m.prefill_tokens_live.load(),
            m.requests_prefilling.load(),
            m.mlx_active_bytes.load(),
            m.mlx_cache_bytes.load(),
            m.ane_int8_bytes.load(),
            m.ane_layers.load(),
        },
    );

    try writeHistogramJson(w, "time_to_first_token_seconds", &m.ttft_ns, ns_to_s);
    try w.print(",", .{});
    try writeHistogramJson(w, "e2e_request_latency_seconds", &m.e2e_latency_ns, ns_to_s);
    try w.print(",", .{});
    try writeHistogramJson(w, "prefill_time_seconds", &m.prefill_time_ns, ns_to_s);
    try w.print(",", .{});
    try writeHistogramJson(w, "decode_time_seconds", &m.decode_time_ns, ns_to_s);
    try w.print(",", .{});
    try writeHistogramJson(w, "prompt_tokens", &m.prompt_tokens_hist, 1.0);
    try w.print(",", .{});
    try writeHistogramJson(w, "output_tokens", &m.output_tokens_hist, 1.0);

    try w.print("}}}}", .{});
}

// ---------------------------------------------------------------------------
// Internal render helpers
// ---------------------------------------------------------------------------

fn writeCounter(w: *std.Io.Writer, name: []const u8, help: []const u8, value: u64) !void {
    try w.print("# HELP {s} {s}\n", .{ name, help });
    try w.print("# TYPE {s} counter\n", .{name});
    try w.print("{s} {d}\n\n", .{ name, value });
}

fn writeGauge(w: *std.Io.Writer, name: []const u8, help: []const u8, value: u64) !void {
    try w.print("# HELP {s} {s}\n", .{ name, help });
    try w.print("# TYPE {s} gauge\n", .{name});
    try w.print("{s} {d}\n\n", .{ name, value });
}

/// Render a Histogram(N) in Prometheus text format.
/// `hist` is `*const Histogram(N)` for any comptime N (accepted via anytype).
/// `scale` converts raw units to display units (e.g. 1e-9 for ns→s, 1.0 for counts).
fn writeHistogram(w: *std.Io.Writer, name: []const u8, help: []const u8, hist: anytype, scale: f64) !void {
    try w.print("# HELP {s} {s}\n", .{ name, help });
    try w.print("# TYPE {s} histogram\n", .{name});

    var cumulative: u64 = 0;
    for (hist.bounds, 0..) |bound, i| {
        cumulative += hist.buckets[i].load(.monotonic);
        const bound_f = @as(f64, @floatFromInt(bound)) * scale;
        try w.print("{s}_bucket{{le=\"{d}\"}} {d}\n", .{ name, bound_f, cumulative });
    }
    // +Inf bucket (index == bounds.len)
    cumulative += hist.buckets[hist.bounds.len].load(.monotonic);
    try w.print("{s}_bucket{{le=\"+Inf\"}} {d}\n", .{ name, cumulative });

    const sum_f = @as(f64, @floatFromInt(hist.sum.load(.monotonic))) * scale;
    try w.print("{s}_sum {d}\n", .{ name, sum_f });
    try w.print("{s}_count {d}\n\n", .{ name, hist.count.load(.monotonic) });
}

/// Render a Histogram(N) as a JSON object with cumulative bucket counts.
/// `scale` converts raw units to display units (1e-9 for ns→s, 1.0 for counts).
fn writeHistogramJson(w: *std.Io.Writer, name: []const u8, hist: anytype, scale: f64) !void {
    try w.print("\"{s}\":{{\"count\":{d},\"sum\":{d},\"bounds\":[", .{
        name,
        hist.count.load(.monotonic),
        @as(f64, @floatFromInt(hist.sum.load(.monotonic))) * scale,
    });

    for (hist.bounds, 0..) |bound, i| {
        if (i > 0) try w.print(",", .{});
        try w.print("{d}", .{@as(f64, @floatFromInt(bound)) * scale});
    }

    try w.print("],\"bucket_counts\":[", .{});
    var cumulative: u64 = 0;
    for (hist.bounds, 0..) |_, i| {
        if (i > 0) try w.print(",", .{});
        cumulative += hist.buckets[i].load(.monotonic);
        try w.print("{d}", .{cumulative});
    }
    // +Inf bucket (index == bounds.len)
    cumulative += hist.buckets[hist.bounds.len].load(.monotonic);
    try w.print(",{d}]}}", .{cumulative});
}

/// Monotonically increasing lock-free counter.
/// Safe to add from any thread; use load() to snapshot.
pub const Counter = struct {
    value: std.atomic.Value(u64),

    pub fn init() Counter {
        return .{ .value = std.atomic.Value(u64).init(0) };
    }

    /// Increment by n. Lock-free; safe from any thread.
    pub fn add(self: *Counter, n: u64) void {
        _ = self.value.fetchAdd(n, .monotonic);
    }

    /// Increment by 1. Convenience wrapper over add(1).
    pub fn inc(self: *Counter) void {
        self.add(1);
    }

    pub fn load(self: *const Counter) u64 {
        return self.value.load(.monotonic);
    }
};

/// Instantaneous gauge — can be set to any u64. Intended for a single
/// owner (sampler thread); multiple concurrent writers are unsound because
/// there is no CAS retry and the last store wins non-deterministically.
pub const Gauge = struct {
    value: std.atomic.Value(u64),

    pub fn init() Gauge {
        return .{ .value = std.atomic.Value(u64).init(0) };
    }

    /// Set to v. Single-owner assumption — see struct doc.
    pub fn set(self: *Gauge, v: u64) void {
        self.value.store(v, .monotonic);
    }

    pub fn load(self: *const Gauge) u64 {
        return self.value.load(.monotonic);
    }
};

/// Fixed-capacity, lock-free cumulative histogram. `N` is the number of finite
/// upper bounds; there is an implicit `+Inf` bucket, so `buckets.len == N + 1`.
/// Values are unsigned native units (nanoseconds for latencies, raw counts for
/// token tallies); the Prometheus renderer applies a scale factor for display.
pub fn Histogram(comptime N: usize) type {
    return struct {
        const Self = @This();

        /// Ascending finite upper bounds, in native units.
        bounds: [N]u64,
        /// Per-bucket observation counts. `buckets[i]` counts values in
        /// `(bounds[i-1], bounds[i]]`; `buckets[N]` is the `+Inf` overflow.
        buckets: [N + 1]std.atomic.Value(u64),
        /// Sum of all observed values, native units.
        sum: std.atomic.Value(u64),
        /// Total observation count (== sum of all buckets).
        count: std.atomic.Value(u64),

        pub fn init(bounds: [N]u64) Self {
            return .{
                .bounds = bounds,
                .buckets = @splat(std.atomic.Value(u64).init(0)),
                .sum = std.atomic.Value(u64).init(0),
                .count = std.atomic.Value(u64).init(0),
            };
        }

        /// Record one observation. Lock-free; safe from any thread.
        pub fn observe(self: *Self, v: u64) void {
            var i: usize = 0;
            while (i < N and v > self.bounds[i]) : (i += 1) {}
            _ = self.buckets[i].fetchAdd(1, .monotonic);
            _ = self.sum.fetchAdd(v, .monotonic);
            _ = self.count.fetchAdd(1, .monotonic);
        }
    };
}

test "Counter.add accumulates monotonically" {
    const testing = std.testing;
    var c = Counter.init();
    c.add(10);
    c.add(5);
    c.inc();
    try testing.expectEqual(@as(u64, 16), c.load());
}

test "Gauge.set and load" {
    const testing = std.testing;
    var g = Gauge.init();
    try testing.expectEqual(@as(u64, 0), g.load());
    g.set(75);
    try testing.expectEqual(@as(u64, 75), g.load());
    g.set(0);
    try testing.expectEqual(@as(u64, 0), g.load());
}

test "liveGenerationTokens = total + inflight aggregate" {
    const testing = std.testing;
    var m = Metrics.init();
    m.generation_tokens_total.add(100);
    // Live gauge = completed (100) + in-flight aggregate (37) published by the
    // scheduler once per decode tick.
    try testing.expectEqual(@as(u64, 137), liveGenerationTokens(&m, 37));
    // At rest (nothing decoding ⇒ inflight aggregate 0) the live gauge must
    // equal the completion counter — the invariant test_metrics.sh checks.
    try testing.expectEqual(@as(u64, 100), liveGenerationTokens(&m, 0));
}

test "prefill throughput must exclude cache-restored tokens" {
    const testing = std.testing;
    // `prompt_tokens_total` counts the FULL prompt, including tokens the hot
    // prefix cache restored without forwarding them. Dividing that by
    // `prefill_time_seconds` (which only ticks for work actually done) inflates
    // prefill tok/s by prompt/(prompt-cached). Live 2026-07-09: a 35B MoE with
    // a 91% token-level cache hit rate reported 9.8K tok/s where the server's
    // own log line said ~220-1100. The panel needs the FORWARDED token count.
    var m = Metrics.init();
    m.recordRequest("stop", 1_000_000, 100_000_000, 500_000_000, 1000, 50, 900);

    // vLLM semantics preserved: every prompt token is billed.
    try testing.expectEqual(@as(u64, 1000), m.prompt_tokens_total.load());
    // ...but only 100 tokens were actually pushed through the model.
    try testing.expectEqual(@as(u64, 100), m.prefill_tokens_total.load());
    try testing.expectEqual(@as(u64, 900), m.prefix_cache_tokens_total.load());

    // Universal invariant: forwarded + restored == billed.
    try testing.expectEqual(
        m.prompt_tokens_total.load(),
        m.prefill_tokens_total.load() + m.prefix_cache_tokens_total.load(),
    );

    // A cold request (no cache hit) forwards everything.
    m.recordRequest("stop", 1_000_000, 100_000_000, 500_000_000, 400, 10, 0);
    try testing.expectEqual(@as(u64, 500), m.prefill_tokens_total.load());
    try testing.expectEqual(@as(u64, 900), m.prefix_cache_tokens_total.load());

    // Degenerate: cached >= prompt must saturate (restore 10, not 99) and never wrap.
    m.recordRequest("stop", 1_000_000, 100_000_000, 500_000_000, 10, 1, 99);
    try testing.expectEqual(@as(u64, 500), m.prefill_tokens_total.load());
    try testing.expectEqual(@as(u64, 910), m.prefix_cache_tokens_total.load());
    // Invariant survives the saturation.
    try testing.expectEqual(
        m.prompt_tokens_total.load(),
        m.prefill_tokens_total.load() + m.prefix_cache_tokens_total.load(),
    );

    var buf: [8192]u8 = undefined;
    var w = std.Io.Writer.fixed(&buf);
    try renderPrometheus(&m, &w);
    try testing.expect(std.mem.indexOf(u8, w.buffered(), "mlx_serve:prefill_tokens_total 500") != null);
    try testing.expect(std.mem.indexOf(u8, w.buffered(), "mlx_serve:prefix_cache_tokens_total 910") != null);

    // The index panel polls /metrics.json and divides by `prefill_tokens_total`.
    var jbuf: [8192]u8 = undefined;
    var jw = std.Io.Writer.fixed(&jbuf);
    try renderJson(&m, &jw);
    const j = jw.buffered();
    try testing.expect(std.mem.indexOf(u8, j, "\"prefill_tokens_total\":500") != null);
    try testing.expect(std.mem.indexOf(u8, j, "\"prefix_cache_tokens_total\":910") != null);
}

test "prefill progress is exposed live, not only at request completion" {
    const testing = std.testing;
    // `prompt_tokens_total` and the prefill_time histogram only advance when a
    // request FINISHES, and generated tokens only accrue during decode. So a
    // multi-minute prefill pinned the GPU while the panel showed 0 / "—".
    // `prefill_tokens_live` is the missing signal: tokens forwarded so far by
    // the in-flight prefill, 0 when none is running.
    var m = Metrics.init();
    m.prefill_tokens_live.set(16384);

    var buf: [8192]u8 = undefined;
    var w = std.Io.Writer.fixed(&buf);
    try renderPrometheus(&m, &w);
    const out = w.buffered();
    try testing.expect(std.mem.indexOf(u8, out, "# TYPE mlx_serve:prefill_tokens_live gauge") != null);
    try testing.expect(std.mem.indexOf(u8, out, "mlx_serve:prefill_tokens_live 16384") != null);

    var jbuf: [8192]u8 = undefined;
    var jw = std.Io.Writer.fixed(&jbuf);
    try renderJson(&m, &jw);
    try testing.expect(std.mem.indexOf(u8, jw.buffered(), "\"prefill_tokens_live\":16384") != null);

    // At rest it is zero — "prefilling" must mean exactly that.
    m.prefill_tokens_live.set(0);
    try testing.expectEqual(@as(u64, 0), m.prefill_tokens_live.load());

    // The phase flag is separate from the token count: it flips at prefill
    // START, so the panel isn't blind for the ~40 s a 27B takes to finish its
    // first 8192-token chunk (and it also covers ds4/llama, whose prefill never
    // reaches the MLX chunk loop and so never moves the token gauge).
    m.requests_prefilling.set(1);
    var b2: [8192]u8 = undefined;
    var w2 = std.Io.Writer.fixed(&b2);
    try renderPrometheus(&m, &w2);
    try testing.expect(std.mem.indexOf(u8, w2.buffered(), "mlx_serve:requests_prefilling 1") != null);

    var j2: [8192]u8 = undefined;
    var jw2 = std.Io.Writer.fixed(&j2);
    try renderJson(&m, &jw2);
    try testing.expect(std.mem.indexOf(u8, jw2.buffered(), "\"requests_prefilling\":1") != null);
}

test "renderPrometheus emits well-formed Prometheus text" {
    const testing = std.testing;
    var m = Metrics.init();
    m.prompt_tokens_total.add(42);
    m.generation_tokens_total.add(100);
    m.requests_running.set(3);
    m.generation_tokens_live.set(137); // 100 completed + 37 in-flight
    // Observe one 50ms TTFT (50_000_000 ns) — lands in bucket for 50ms le bound
    m.ttft_ns.observe(50_000_000);

    var buf: [32 * 1024]u8 = undefined;
    var w: std.Io.Writer = .fixed(&buf);
    try renderPrometheus(&m, &w);
    const out = buf[0..w.end];

    // Counter TYPE + value line present
    try testing.expect(std.mem.indexOf(u8, out, "# TYPE vllm:prompt_tokens_total counter") != null);
    try testing.expect(std.mem.indexOf(u8, out, "vllm:prompt_tokens_total 42") != null);
    // Gauge TYPE + value line present
    try testing.expect(std.mem.indexOf(u8, out, "# TYPE vllm:num_requests_running gauge") != null);
    try testing.expect(std.mem.indexOf(u8, out, "vllm:num_requests_running 3") != null);
    // Real-time live-token gauge present (drives the panel's low-latency tok/s)
    try testing.expect(std.mem.indexOf(u8, out, "# TYPE mlx_serve:generation_tokens_live gauge") != null);
    try testing.expect(std.mem.indexOf(u8, out, "mlx_serve:generation_tokens_live 137") != null);
    // Histogram has _bucket{le="+Inf"}, _sum, _count lines
    try testing.expect(std.mem.indexOf(u8, out, "vllm:time_to_first_token_seconds_bucket{le=\"+Inf\"}") != null);
    try testing.expect(std.mem.indexOf(u8, out, "vllm:time_to_first_token_seconds_sum ") != null);
    try testing.expect(std.mem.indexOf(u8, out, "vllm:time_to_first_token_seconds_count 1") != null);
}

test "Counter concurrent writes are not lost" {
    const testing = std.testing;
    const N_THREADS = 4;
    const N_ITERS = 10_000;
    var c = Counter.init();

    const Ctx = struct {
        counter: *Counter,
        fn run(ctx: *@This()) void {
            for (0..N_ITERS) |_| ctx.counter.inc();
        }
    };

    var ctxs: [N_THREADS]Ctx = undefined;
    var threads: [N_THREADS]std.Thread = undefined;
    for (&ctxs, &threads) |*ctx, *t| {
        ctx.* = .{ .counter = &c };
        t.* = try std.Thread.spawn(.{}, Ctx.run, .{ctx});
    }
    for (&threads) |*t| t.join();
    try testing.expectEqual(@as(u64, N_THREADS * N_ITERS), c.load());
}

test "Metrics.recordRequest increments correct fields on success" {
    const testing = std.testing;
    var m = Metrics.init();
    // real_ttft=60ms (includes 10ms queue wait + 50ms prefill), prefill=50ms, decode=200ms
    m.recordRequest("stop", 60_000_000, 50_000_000, 200_000_000, 128, 64, 20);

    try testing.expectEqual(@as(u64, 1), m.requests_success_total.load());
    try testing.expectEqual(@as(u64, 0), m.requests_cancelled_total.load());
    try testing.expectEqual(@as(u64, 128), m.prompt_tokens_total.load());
    try testing.expectEqual(@as(u64, 64), m.generation_tokens_total.load());
    // TTFT histogram got one observation (60ms)
    try testing.expectEqual(@as(u64, 1), m.ttft_ns.count.load(.monotonic));
    try testing.expectEqual(@as(u64, 60_000_000), m.ttft_ns.sum.load(.monotonic));
    // Prefill histogram separately (50ms)
    try testing.expectEqual(@as(u64, 1), m.prefill_time_ns.count.load(.monotonic));
    try testing.expectEqual(@as(u64, 50_000_000), m.prefill_time_ns.sum.load(.monotonic));
    // Decode histogram (200ms)
    try testing.expectEqual(@as(u64, 1), m.decode_time_ns.count.load(.monotonic));
    // e2e = real_ttft + decode = 60ms + 200ms = 260ms
    try testing.expectEqual(@as(u64, 260_000_000), m.e2e_latency_ns.sum.load(.monotonic));
    // Token histograms
    try testing.expectEqual(@as(u64, 1), m.prompt_tokens_hist.count.load(.monotonic));
    try testing.expectEqual(@as(u64, 1), m.output_tokens_hist.count.load(.monotonic));
    // Cache counters — called with cached_tokens=20, so query+hit both increment
    try testing.expectEqual(@as(u64, 1), m.prefix_cache_queries_total.load());
    try testing.expectEqual(@as(u64, 1), m.prefix_cache_hits_total.load());
}

test "Metrics.recordRequest skips histograms on cancelled request" {
    const testing = std.testing;
    var m = Metrics.init();
    m.recordRequest("cancelled", 50_000_000, 50_000_000, 0, 128, 0, 0);

    try testing.expectEqual(@as(u64, 0), m.requests_success_total.load());
    try testing.expectEqual(@as(u64, 1), m.requests_cancelled_total.load());
    // Latency histograms must NOT be touched
    try testing.expectEqual(@as(u64, 0), m.ttft_ns.count.load(.monotonic));
    try testing.expectEqual(@as(u64, 0), m.decode_time_ns.count.load(.monotonic));
    // Token counters also NOT touched on cancel
    try testing.expectEqual(@as(u64, 0), m.prompt_tokens_total.load());
    try testing.expectEqual(@as(u64, 0), m.generation_tokens_total.load());
    // Cache counters ARE touched even on cancel (the cache was still queried)
    try testing.expectEqual(@as(u64, 1), m.prefix_cache_queries_total.load());
    // cached_tokens=0 → no hit
    try testing.expectEqual(@as(u64, 0), m.prefix_cache_hits_total.load());
}

test "Histogram.observe places values in the correct bucket" {
    const testing = std.testing;
    var h = Histogram(3).init(.{ 10, 100, 1000 });

    h.observe(5); // → bucket 0   (<= 10)
    h.observe(50); // → bucket 1  (<= 100)
    h.observe(500); // → bucket 2 (<= 1000)
    h.observe(5000); // → bucket 3 (+Inf overflow)
    h.observe(10); // → bucket 0  (boundary is inclusive upper)

    try testing.expectEqual(@as(u64, 2), h.buckets[0].load(.monotonic));
    try testing.expectEqual(@as(u64, 1), h.buckets[1].load(.monotonic));
    try testing.expectEqual(@as(u64, 1), h.buckets[2].load(.monotonic));
    try testing.expectEqual(@as(u64, 1), h.buckets[3].load(.monotonic));
    try testing.expectEqual(@as(u64, 5), h.count.load(.monotonic));
    try testing.expectEqual(@as(u64, 5565), h.sum.load(.monotonic));
}

test "renderJson emits valid JSON with correct structure" {
    const testing = std.testing;
    var m = Metrics.init();
    m.requests_success_total.inc();
    m.prompt_tokens_total.add(100);
    m.generation_tokens_live.set(55);
    m.ttft_ns.observe(50_000_000); // 50ms

    var buf: [64 * 1024]u8 = undefined;
    var w: std.Io.Writer = .fixed(&buf);
    try renderJson(&m, &w);
    const out = buf[0..w.end];

    // Must be valid-ish JSON (starts { ends })
    try testing.expect(out[0] == '{');
    try testing.expect(out[out.len - 1] == '}');
    // Must contain key structure fields
    try testing.expect(std.mem.indexOf(u8, out, "\"counters\"") != null);
    try testing.expect(std.mem.indexOf(u8, out, "\"gauges\"") != null);
    try testing.expect(std.mem.indexOf(u8, out, "\"histograms\"") != null);
    try testing.expect(std.mem.indexOf(u8, out, "\"requests_success_total\":1") != null);
    try testing.expect(std.mem.indexOf(u8, out, "\"prompt_tokens_total\":100") != null);
    try testing.expect(std.mem.indexOf(u8, out, "\"generation_tokens_live\":55") != null);
    try testing.expect(std.mem.indexOf(u8, out, "\"time_to_first_token_seconds\"") != null);
    try testing.expect(std.mem.indexOf(u8, out, "\"bucket_counts\"") != null);
}

test "renderJson output parses as valid JSON via stdlib parser" {
    const testing = std.testing;
    // Zero-state (first load): verify sum=0 doesn't produce invalid tokens like 0e0
    var m0 = Metrics.init();
    var buf0: [64 * 1024]u8 = undefined;
    var w0: std.Io.Writer = .fixed(&buf0);
    try renderJson(&m0, &w0);
    const out0 = buf0[0..w0.end];

    const parsed0 = try std.json.parseFromSlice(std.json.Value, testing.allocator, out0, .{});
    defer parsed0.deinit();
    const obj0 = parsed0.value.object;
    try testing.expect(obj0.get("counters") != null);
    try testing.expect(obj0.get("gauges") != null);
    try testing.expect(obj0.get("histograms") != null);

    // Non-zero state: observe values and re-verify parseability
    var m1 = Metrics.init();
    m1.requests_success_total.inc();
    m1.prompt_tokens_total.add(512);
    m1.ttft_ns.observe(50_000_000); // 50ms
    m1.gpu_utilization_pct.set(42);

    var buf1: [64 * 1024]u8 = undefined;
    var w1: std.Io.Writer = .fixed(&buf1);
    try renderJson(&m1, &w1);
    const out1 = buf1[0..w1.end];

    const parsed1 = try std.json.parseFromSlice(std.json.Value, testing.allocator, out1, .{});
    defer parsed1.deinit();
    const obj1 = parsed1.value.object;
    try testing.expect(obj1.get("counters") != null);
    // Verify counters sub-object has expected key
    const counters = obj1.get("counters").?.object;
    try testing.expectEqual(@as(i64, 1), counters.get("requests_success_total").?.integer);
}
