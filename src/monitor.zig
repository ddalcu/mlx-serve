const std = @import("std");

pub const history_capacity = 1800;
pub const archive_capacity = 1024;
pub const sample_interval_ms: u64 = 2000;
pub const request_capacity = 256;
pub const event_capacity = 128;
pub const active_capacity = 128;
pub const model_capacity = 64;

pub fn forwardedDelta(accounted: u64, forwarded: u64) u64 {
    return forwarded -| accounted;
}

pub fn claimCompletion(recorded: *std.atomic.Value(bool)) bool {
    return !recorded.swap(true, .acq_rel);
}

const Name = struct {
    bytes: [128]u8 = @splat(0),
    len: u8 = 0,

    fn from(value: []const u8) Name {
        var result: Name = .{};
        const n = @min(value.len, result.bytes.len);
        @memcpy(result.bytes[0..n], value[0..n]);
        result.len = @intCast(n);
        return result;
    }

    fn slice(self: *const Name) []const u8 {
        return self.bytes[0..self.len];
    }
};

pub const Phase = enum { queued, prefill, decode };
pub const Outcome = enum { success, cancelled, failed, rejected };

pub const ActiveRequest = struct {
    id: u64 = 0,
    model: Name = .{},
    phase: Phase = .queued,
    started_at_ms: u64 = 0,
    phase_started_at_ms: u64 = 0,
    queue_ms: ?u64 = null,
    prompt_tokens: ?u32 = null,
    cached_tokens: ?u32 = null,
    output_tokens: ?u32 = null,
};

pub const Request = struct {
    id: u64 = 0,
    model: Name = .{},
    outcome: Outcome = .failed,
    error_code: Name = .{},
    started_at_ms: u64 = 0,
    finished_at_ms: u64 = 0,
    queue_ms: ?u64 = null,
    prefill_ms: ?u64 = null,
    decode_ms: ?u64 = null,
    ttft_ms: ?u64 = null,
    e2e_ms: u64 = 0,
    prompt_tokens: u32 = 0,
    output_tokens: u32 = 0,
    cached_tokens: u32 = 0,
};

pub const CompletionInput = struct {
    id: u64,
    model: []const u8,
    outcome: Outcome,
    error_code: []const u8 = "",
    started_at_ms: u64,
    finished_at_ms: u64,
    queue_ms: ?u64 = null,
    prefill_ms: ?u64 = null,
    decode_ms: ?u64 = null,
    ttft_ms: ?u64 = null,
    ttft_ns: ?u64 = null,
    prompt_tokens: u32 = 0,
    output_tokens: u32 = 0,
    cached_tokens: u32 = 0,
};

pub const Sample = struct {
    at_ms: u64 = 0,
    continuity_id: u64 = 0,
    running: u64 = 0,
    queued: u64 = 0,
    prefill_tokens_total: u64 = 0,
    prefill_tokens_forwarded_live_total: u64 = 0,
    /// Monotonic emitted output-token total, retained under the history field name.
    generation_tokens_live: u64 = 0,
    prefill_active_ns_total: ?u64 = null,
    decode_active_ns_total: ?u64 = null,
    requests_completed_total: u64 = 0,
    cache_queries_total: ?u64 = null,
    cache_hits_total: ?u64 = null,
    ttft_ns_sum: ?u64 = null,
    ttft_count: ?u64 = null,
    cpu_pct: ?f32 = null,
    gpu_pct: ?f32 = null,
    process_bytes: ?u64 = null,
    process_memory_byte_seconds_total: f64 = 0,
    process_memory_observed_seconds_total: f64 = 0,
    mlx_active_bytes: ?u64 = null,
    mlx_cache_bytes: ?u64 = null,
};

pub const EventInput = struct {
    at_ms: u64,
    kind: []const u8,
    model: []const u8 = "",
    request_id: u64 = 0,
    code: []const u8 = "",
    message: []const u8 = "",
};

pub const Event = struct {
    at_ms: u64 = 0,
    kind: Name = .{},
    model: Name = .{},
    request_id: u64 = 0,
    code: Name = .{},
    message: Name = .{},
};

pub const ModelTotal = struct {
    model: Name = .{},
    model_hash: u64 = 0,
    success: u64 = 0,
    cancelled: u64 = 0,
    failed: u64 = 0,
    prompt_tokens: u64 = 0,
    output_tokens: u64 = 0,
};

pub const Snapshot = struct {
    active: [active_capacity]ActiveRequest = undefined,
    active_len: usize = 0,
    requests: [request_capacity]Request = undefined,
    request_len: usize = 0,
    history: [history_capacity]Sample = undefined,
    history_len: usize = 0,
    history_archive: [archive_capacity]Sample = undefined,
    history_archive_len: usize = 0,
    archive_compacted: bool = false,
    archive_sample_interval_ms: u64 = sample_interval_ms,
    lifetime_totals: ?Sample = null,
    events: [event_capacity]Event = undefined,
    event_len: usize = 0,
    requests_dropped: u64 = 0,
    events_dropped: u64 = 0,
    active_dropped: u64 = 0,
    model_totals: [model_capacity]ModelTotal = undefined,
    model_len: usize = 0,
    model_untracked_requests: u64 = 0,
};

pub const TtftTotals = struct {
    ns_sum: u64,
    count: u64,
};

pub const ActiveTime = struct {
    prefill_ns: u64,
    decode_ns: u64,
};

pub const ProcessingPhase = enum { prefill, decode };

pub const ProcessingClock = struct {
    version: std.atomic.Value(u64) = .init(0),
    phase: std.atomic.Value(u8) = .init(0),
    since_ns: std.atomic.Value(u64) = .init(0),
    prefill_ns: std.atomic.Value(u64) = .init(0),
    decode_ns: std.atomic.Value(u64) = .init(0),

    fn accrue(self: *ProcessingClock, now_ns: u64) void {
        const elapsed = now_ns -| self.since_ns.load(.seq_cst);
        switch (self.phase.load(.seq_cst)) {
            1 => self.prefill_ns.store(self.prefill_ns.load(.seq_cst) +| elapsed, .seq_cst),
            2 => self.decode_ns.store(self.decode_ns.load(.seq_cst) +| elapsed, .seq_cst),
            else => {},
        }
        self.since_ns.store(now_ns, .seq_cst);
    }

    pub fn begin(self: *ProcessingClock, phase: ProcessingPhase, now_ns: u64) ?ProcessingPhase {
        _ = self.version.fetchAdd(1, .seq_cst);
        const previous: ?ProcessingPhase = switch (self.phase.load(.seq_cst)) {
            1 => .prefill,
            2 => .decode,
            else => null,
        };
        self.accrue(now_ns);
        self.phase.store(if (phase == .prefill) 1 else 2, .seq_cst);
        _ = self.version.fetchAdd(1, .seq_cst);
        return previous;
    }

    pub fn end(self: *ProcessingClock, previous: ?ProcessingPhase, now_ns: u64) void {
        _ = self.version.fetchAdd(1, .seq_cst);
        self.accrue(now_ns);
        self.phase.store(if (previous) |phase| (if (phase == .prefill) @as(u8, 1) else 2) else 0, .seq_cst);
        _ = self.version.fetchAdd(1, .seq_cst);
    }

    pub fn snapshot(self: *const ProcessingClock, now_ns: u64) ?ActiveTime {
        for (0..16) |_| {
            const before = self.version.load(.seq_cst);
            if (before & 1 != 0) continue;
            var result: ActiveTime = .{
                .prefill_ns = self.prefill_ns.load(.seq_cst),
                .decode_ns = self.decode_ns.load(.seq_cst),
            };
            const phase = self.phase.load(.seq_cst);
            const since_ns = self.since_ns.load(.seq_cst);
            if (before != self.version.load(.seq_cst)) continue;
            const elapsed = now_ns -| since_ns;
            switch (phase) {
                1 => result.prefill_ns +|= elapsed,
                2 => result.decode_ns +|= elapsed,
                else => {},
            }
            return result;
        }
        return null;
    }
};

pub const Monitor = struct {
    mutex: std.c.pthread_mutex_t = .{},
    history_mutex: std.c.pthread_mutex_t = .{},
    next_id: std.atomic.Value(u64) = .init(1),
    active: [active_capacity]ActiveRequest = undefined,
    active_len: usize = 0,
    requests: [request_capacity]Request = undefined,
    request_next: usize = 0,
    request_len: usize = 0,
    history: [history_capacity]Sample = undefined,
    history_next: usize = 0,
    history_len: usize = 0,
    history_archive: [archive_capacity]Sample = undefined,
    history_archive_len: usize = 0,
    archive_stride: u64 = 1,
    archive_samples_seen: u64 = 0,
    archive_compacted: bool = false,
    events: [event_capacity]Event = undefined,
    event_next: usize = 0,
    event_len: usize = 0,
    requests_dropped: u64 = 0,
    events_dropped: u64 = 0,
    active_dropped: u64 = 0,
    model_totals: [model_capacity]ModelTotal = undefined,
    model_len: usize = 0,
    model_untracked_requests: u64 = 0,
    ttft_ns_sum: u64 = 0,
    ttft_count: u64 = 0,
    processing_clock: ProcessingClock = .{},

    pub fn init() Monitor {
        return .{};
    }

    fn lock(self: *Monitor) void {
        _ = std.c.pthread_mutex_lock(&self.mutex);
    }

    fn unlock(self: *Monitor) void {
        _ = std.c.pthread_mutex_unlock(&self.mutex);
    }

    fn lockHistory(self: *Monitor) void {
        _ = std.c.pthread_mutex_lock(&self.history_mutex);
    }

    fn unlockHistory(self: *Monitor) void {
        _ = std.c.pthread_mutex_unlock(&self.history_mutex);
    }

    pub fn beginProcessing(self: *Monitor, phase: ProcessingPhase, now_ns: u64) ?ProcessingPhase {
        return self.processing_clock.begin(phase, now_ns);
    }

    pub fn endProcessing(self: *Monitor, previous: ?ProcessingPhase, now_ns: u64) void {
        self.processing_clock.end(previous, now_ns);
    }

    pub fn activeTime(self: *Monitor, now_ns: u64) ?ActiveTime {
        return self.processing_clock.snapshot(now_ns);
    }

    pub fn beginRequest(self: *Monitor, model: []const u8, started_at_ms: u64) u64 {
        const id = self.next_id.fetchAdd(1, .monotonic);
        self.lock();
        defer self.unlock();
        if (self.active_len < active_capacity) {
            self.active[self.active_len] = .{ .id = id, .model = Name.from(model), .started_at_ms = started_at_ms, .phase_started_at_ms = started_at_ms };
            self.active_len += 1;
        } else {
            self.active_dropped += 1;
        }
        return id;
    }

    pub fn updateRequestPhase(self: *Monitor, id: u64, phase: Phase, at_ms: u64, queue_ms: ?u64) void {
        self.lock();
        defer self.unlock();
        for (self.active[0..self.active_len]) |*request| {
            if (request.id != id) continue;
            request.phase = phase;
            request.phase_started_at_ms = at_ms;
            if (queue_ms) |duration| request.queue_ms = duration;
            break;
        }
    }

    pub fn updateRequestTokens(self: *Monitor, id: u64, prompt_tokens: u32, cached_tokens: u32) void {
        self.lock();
        defer self.unlock();
        for (self.active[0..self.active_len]) |*request| {
            if (request.id != id) continue;
            request.prompt_tokens = prompt_tokens;
            request.cached_tokens = cached_tokens;
            break;
        }
    }

    pub fn updateRequestOutput(self: *Monitor, id: u64, output_tokens: u32) void {
        self.lock();
        defer self.unlock();
        for (self.active[0..self.active_len]) |*request| {
            if (request.id != id) continue;
            request.output_tokens = output_tokens;
            break;
        }
    }

    pub fn completeRequest(self: *Monitor, input: CompletionInput) void {
        self.lock();
        defer self.unlock();
        if (input.outcome == .success) if (input.ttft_ns) |ns| {
            self.ttft_ns_sum += ns;
            self.ttft_count += 1;
        };
        for (self.active[0..self.active_len], 0..) |request, i| {
            if (request.id != input.id) continue;
            self.active_len -= 1;
            self.active[i] = self.active[self.active_len];
            break;
        }
        if (self.request_len == request_capacity) self.requests_dropped += 1 else self.request_len += 1;
        self.requests[self.request_next] = .{
            .id = input.id,
            .model = Name.from(input.model),
            .outcome = input.outcome,
            .error_code = Name.from(input.error_code),
            .started_at_ms = input.started_at_ms,
            .finished_at_ms = input.finished_at_ms,
            .queue_ms = input.queue_ms,
            .prefill_ms = input.prefill_ms,
            .decode_ms = input.decode_ms,
            .ttft_ms = input.ttft_ms,
            .e2e_ms = input.finished_at_ms -| input.started_at_ms,
            .prompt_tokens = input.prompt_tokens,
            .output_tokens = input.output_tokens,
            .cached_tokens = input.cached_tokens,
        };
        self.request_next = (self.request_next + 1) % request_capacity;
        var total: ?*ModelTotal = null;
        for (self.model_totals[0..self.model_len]) |*entry| {
            if (entry.model_hash == std.hash.Wyhash.hash(0, input.model)) {
                total = entry;
                break;
            }
        }
        if (total == null and self.model_len < model_capacity) {
            self.model_totals[self.model_len] = .{ .model = Name.from(input.model), .model_hash = std.hash.Wyhash.hash(0, input.model) };
            total = &self.model_totals[self.model_len];
            self.model_len += 1;
        }
        if (total) |entry| {
            switch (input.outcome) {
                .success => entry.success += 1,
                .cancelled => entry.cancelled += 1,
                .failed => entry.failed += 1,
                .rejected => {},
            }
            entry.prompt_tokens += input.prompt_tokens;
            entry.output_tokens += input.output_tokens;
        } else {
            self.model_untracked_requests += 1;
        }
    }

    pub fn ttftTotals(self: *Monitor) TtftTotals {
        self.lock();
        defer self.unlock();
        return .{ .ns_sum = self.ttft_ns_sum, .count = self.ttft_count };
    }

    pub fn appendSample(self: *Monitor, input: Sample) void {
        self.lockHistory();
        defer self.unlockHistory();
        var sample = input;
        if (self.history_len > 0) {
            const previous = self.history[(self.history_next + history_capacity - 1) % history_capacity];
            const dt = sample.at_ms -| previous.at_ms;
            sample.continuity_id = previous.continuity_id + @as(u64, @intFromBool(dt == 0 or dt > 3 * sample_interval_ms or countersReset(previous, sample)));
            sample.process_memory_byte_seconds_total = previous.process_memory_byte_seconds_total;
            sample.process_memory_observed_seconds_total = previous.process_memory_observed_seconds_total;
            if (dt > 0 and dt <= 3 * sample_interval_ms) {
                if (previous.process_bytes) |before| {
                    if (sample.process_bytes) |after| {
                        const seconds = @as(f64, @floatFromInt(dt)) / 1000;
                        sample.process_memory_byte_seconds_total += (@as(f64, @floatFromInt(before)) + @as(f64, @floatFromInt(after))) / 2 * seconds;
                        sample.process_memory_observed_seconds_total += seconds;
                    }
                }
            }
        }
        if (self.history_len == history_capacity) self.archiveSample(self.history[self.history_next]);
        self.history[self.history_next] = sample;
        self.history_next = (self.history_next + 1) % history_capacity;
        if (self.history_len < history_capacity) self.history_len += 1;
    }

    fn archiveSample(self: *Monitor, sample: Sample) void {
        self.archive_samples_seen += 1;
        if ((self.archive_samples_seen - 1) % self.archive_stride != 0) return;
        if (self.history_archive_len == archive_capacity) {
            for (0..archive_capacity / 2) |i| self.history_archive[i] = self.history_archive[i * 2];
            self.history_archive_len = archive_capacity / 2;
            self.archive_stride *= 2;
            self.archive_compacted = true;
            if ((self.archive_samples_seen - 1) % self.archive_stride != 0) return;
        }
        self.history_archive[self.history_archive_len] = sample;
        self.history_archive_len += 1;
    }

    pub fn recordEvent(self: *Monitor, input: EventInput) void {
        self.lock();
        defer self.unlock();
        if (self.event_len == event_capacity) self.events_dropped += 1 else self.event_len += 1;
        self.events[self.event_next] = .{
            .at_ms = input.at_ms,
            .kind = Name.from(input.kind),
            .model = Name.from(input.model),
            .request_id = input.request_id,
            .code = Name.from(input.code),
            .message = Name.from(input.message),
        };
        self.event_next = (self.event_next + 1) % event_capacity;
    }

    pub fn snapshot(self: *Monitor) Snapshot {
        var result: Snapshot = .{};
        self.lock();
        result.active_len = self.active_len;
        @memcpy(result.active[0..result.active_len], self.active[0..result.active_len]);
        result.request_len = self.request_len;
        for (0..result.request_len) |i| result.requests[i] = self.requests[(self.request_next + request_capacity - result.request_len + i) % request_capacity];
        result.event_len = self.event_len;
        for (0..result.event_len) |i| result.events[i] = self.events[(self.event_next + event_capacity - result.event_len + i) % event_capacity];
        result.requests_dropped = self.requests_dropped;
        result.events_dropped = self.events_dropped;
        result.active_dropped = self.active_dropped;
        result.model_len = self.model_len;
        @memcpy(result.model_totals[0..result.model_len], self.model_totals[0..result.model_len]);
        result.model_untracked_requests = self.model_untracked_requests;
        self.unlock();

        self.lockHistory();
        result.history_len = self.history_len;
        for (0..result.history_len) |i| result.history[i] = self.history[(self.history_next + history_capacity - result.history_len + i) % history_capacity];
        result.history_archive_len = self.history_archive_len;
        @memcpy(result.history_archive[0..result.history_archive_len], self.history_archive[0..result.history_archive_len]);
        result.archive_compacted = self.archive_compacted;
        result.archive_sample_interval_ms = self.archive_stride * sample_interval_ms;
        if (result.history_len > 0) result.lifetime_totals = result.history[result.history_len - 1];
        self.unlockHistory();
        return result;
    }

    pub fn renderJson(self: *Monitor, w: *std.Io.Writer, extra_fields: []const u8) !void {
        const snapshot_value = self.snapshot();
        try w.print("{{\"schema_version\":1,\"retention\":{{\"history_seconds\":3600,\"archive_capacity\":{d},\"archive_compacted\":{},\"archive_sample_interval_ms\":{d},\"request_capacity\":256,\"event_capacity\":128,\"model_capacity\":64,\"requests_dropped\":{d},\"events_dropped\":{d},\"active_dropped\":{d},\"model_untracked_requests\":{d}}},\"active_requests\":[", .{ archive_capacity, snapshot_value.archive_compacted, snapshot_value.archive_sample_interval_ms, snapshot_value.requests_dropped, snapshot_value.events_dropped, snapshot_value.active_dropped, snapshot_value.model_untracked_requests });
        for (snapshot_value.active[0..snapshot_value.active_len], 0..) |request, i| {
            if (i != 0) try w.writeAll(",");
            try w.print("{{\"id\":{d},\"model\":", .{request.id});
            try std.json.Stringify.encodeJsonString(request.model.slice(), .{}, w);
            try w.print(",\"phase\":\"{s}\",\"started_at_ms\":{d},\"phase_started_at_ms\":{d},\"queue_ms\":", .{ @tagName(request.phase), request.started_at_ms, request.phase_started_at_ms });
            try writeOptional(w, request.queue_ms);
            try w.writeAll(",\"prompt_tokens\":");
            try writeOptional(w, request.prompt_tokens);
            try w.writeAll(",\"cached_tokens\":");
            try writeOptional(w, request.cached_tokens);
            try w.writeAll(",\"output_tokens\":");
            try writeOptional(w, request.output_tokens);
            try w.writeAll("}");
        }
        try w.writeAll("],\"recent_requests\":[");
        for (snapshot_value.requests[0..snapshot_value.request_len], 0..) |request, i| {
            if (i != 0) try w.writeAll(",");
            try w.print("{{\"id\":{d},\"model\":", .{request.id});
            try std.json.Stringify.encodeJsonString(request.model.slice(), .{}, w);
            try w.print(",\"outcome\":\"{s}\",\"error_code\":", .{@tagName(request.outcome)});
            if (request.error_code.len == 0) try w.writeAll("null") else try std.json.Stringify.encodeJsonString(request.error_code.slice(), .{}, w);
            try w.print(",\"started_at_ms\":{d},\"finished_at_ms\":{d},\"queue_ms\":", .{ request.started_at_ms, request.finished_at_ms });
            try writeOptional(w, request.queue_ms);
            try w.writeAll(",\"prefill_ms\":");
            try writeOptional(w, request.prefill_ms);
            try w.writeAll(",\"decode_ms\":");
            try writeOptional(w, request.decode_ms);
            try w.writeAll(",\"ttft_ms\":");
            try writeOptional(w, request.ttft_ms);
            try w.print(",\"e2e_ms\":{d},\"prompt_tokens\":{d},\"output_tokens\":{d},\"cached_tokens\":{d}}}", .{ request.e2e_ms, request.prompt_tokens, request.output_tokens, request.cached_tokens });
        }
        try w.writeAll("],\"history\":[");
        for (snapshot_value.history[0..snapshot_value.history_len], 0..) |sample, i| {
            if (i != 0) try w.writeAll(",");
            try writeSample(w, sample);
        }
        try w.writeAll("],\"history_archive\":[");
        for (snapshot_value.history_archive[0..snapshot_value.history_archive_len], 0..) |sample, i| {
            if (i != 0) try w.writeAll(",");
            try writeSample(w, sample);
        }
        try w.writeAll("],\"lifetime_totals\":");
        if (snapshot_value.lifetime_totals) |sample| try writeSample(w, sample) else try w.writeAll("null");
        try w.writeAll(",\"model_totals\":[");
        for (snapshot_value.model_totals[0..snapshot_value.model_len], 0..) |total, i| {
            if (i != 0) try w.writeAll(",");
            try w.writeAll("{\"model\":");
            try std.json.Stringify.encodeJsonString(total.model.slice(), .{}, w);
            try w.print(",\"success\":{d},\"cancelled\":{d},\"failed\":{d},\"prompt_tokens\":{d},\"output_tokens\":{d}}}", .{ total.success, total.cancelled, total.failed, total.prompt_tokens, total.output_tokens });
        }
        try w.writeAll("],\"events\":[");
        for (snapshot_value.events[0..snapshot_value.event_len], 0..) |event, i| {
            if (i != 0) try w.writeAll(",");
            try w.print("{{\"at_ms\":{d},\"kind\":", .{event.at_ms});
            try std.json.Stringify.encodeJsonString(event.kind.slice(), .{}, w);
            try w.writeAll(",\"model\":");
            try std.json.Stringify.encodeJsonString(event.model.slice(), .{}, w);
            try w.print(",\"request_id\":{d},\"code\":", .{event.request_id});
            try std.json.Stringify.encodeJsonString(event.code.slice(), .{}, w);
            try w.writeAll(",\"message\":");
            try std.json.Stringify.encodeJsonString(event.message.slice(), .{}, w);
            try w.writeAll("}");
        }
        try w.writeAll("]");
        try w.writeAll(extra_fields);
        try w.writeAll("}");
    }
};

fn countersReset(before: Sample, after: Sample) bool {
    if (after.prefill_tokens_total < before.prefill_tokens_total or
        after.prefill_tokens_forwarded_live_total < before.prefill_tokens_forwarded_live_total or
        after.generation_tokens_live < before.generation_tokens_live or
        after.requests_completed_total < before.requests_completed_total) return true;
    inline for (.{ "prefill_active_ns_total", "decode_active_ns_total", "cache_queries_total", "cache_hits_total", "ttft_ns_sum", "ttft_count" }) |field| {
        if (@field(before, field)) |old| {
            if (@field(after, field)) |new| {
                if (new < old) return true;
            }
        }
    }
    return false;
}

fn writeSample(w: *std.Io.Writer, sample: Sample) !void {
    try w.print("{{\"at_ms\":{d},\"continuity_id\":{d},\"running\":{d},\"queued\":{d},\"prefill_tokens_total\":{d},\"prefill_tokens_forwarded_live_total\":{d},\"generation_tokens_live\":{d},\"prefill_active_ns_total\":", .{ sample.at_ms, sample.continuity_id, sample.running, sample.queued, sample.prefill_tokens_total, sample.prefill_tokens_forwarded_live_total, sample.generation_tokens_live });
    try writeOptional(w, sample.prefill_active_ns_total);
    try w.writeAll(",\"decode_active_ns_total\":");
    try writeOptional(w, sample.decode_active_ns_total);
    try w.print(",\"requests_completed_total\":{d},\"cache_queries_total\":", .{sample.requests_completed_total});
    try writeOptional(w, sample.cache_queries_total);
    try w.writeAll(",\"cache_hits_total\":");
    try writeOptional(w, sample.cache_hits_total);
    try w.writeAll(",\"ttft_ns_sum\":");
    try writeOptional(w, sample.ttft_ns_sum);
    try w.writeAll(",\"ttft_count\":");
    try writeOptional(w, sample.ttft_count);
    try w.writeAll(",\"cpu_pct\":");
    try writeOptional(w, sample.cpu_pct);
    try w.writeAll(",\"gpu_pct\":");
    try writeOptional(w, sample.gpu_pct);
    try w.writeAll(",\"process_bytes\":");
    try writeOptional(w, sample.process_bytes);
    try w.print(",\"process_memory_byte_seconds_total\":{d},\"process_memory_observed_seconds_total\":{d},\"mlx_active_bytes\":", .{ sample.process_memory_byte_seconds_total, sample.process_memory_observed_seconds_total });
    try writeOptional(w, sample.mlx_active_bytes);
    try w.writeAll(",\"mlx_cache_bytes\":");
    try writeOptional(w, sample.mlx_cache_bytes);
    try w.writeAll("}");
}

fn writeOptional(w: *std.Io.Writer, value: anytype) !void {
    if (value) |number| {
        if (@TypeOf(number) == f32 and !std.math.isFinite(number)) return w.writeAll("null");
        try w.print("{d}", .{number});
    } else {
        try w.writeAll("null");
    }
}

test "bounded rings retain chronological order and outcome metadata" {
    const testing = std.testing;
    var monitor = Monitor.init();
    for (0..request_capacity + 3) |i| {
        const id = monitor.beginRequest("m", @intCast(i));
        monitor.completeRequest(.{ .id = id, .model = "m", .outcome = if (i % 2 == 0) .success else .failed, .started_at_ms = @intCast(i), .finished_at_ms = @intCast(i + 1) });
    }
    for (0..history_capacity + 2) |i| monitor.appendSample(.{ .at_ms = @intCast(i) });
    for (0..event_capacity + 2) |i| monitor.recordEvent(.{ .at_ms = @intCast(i), .kind = "test" });
    const snap = monitor.snapshot();
    try testing.expectEqual(@as(usize, request_capacity), snap.request_len);
    try testing.expectEqual(@as(u64, 3), snap.requests_dropped);
    try testing.expectEqual(@as(u64, 4), snap.requests[0].id);
    try testing.expectEqual(@as(u64, 2), snap.history[0].at_ms);
    try testing.expectEqual(@as(usize, 2), snap.history_archive_len);
    try testing.expectEqual(@as(u64, 0), snap.history_archive[0].at_ms);
    try testing.expectEqual(@as(u64, 1), snap.history_archive[1].at_ms);
    try testing.expectEqual(@as(u64, 2), snap.events[0].at_ms);
    try testing.expectEqual(@as(u64, 2), snap.events_dropped);
    try testing.expectEqual(@as(usize, 0), snap.active_len);
}

test "request updates finish while sampled history is locked" {
    const testing = std.testing;
    var monitor = Monitor.init();
    monitor.appendSample(.{ .at_ms = 1000 });
    const Ctx = struct {
        monitor: *Monitor,
        done: *std.atomic.Value(bool),

        fn run(ctx: *@This()) void {
            const id = ctx.monitor.beginRequest("model", 1000);
            ctx.monitor.updateRequestPhase(id, .decode, 1010, 10);
            ctx.monitor.updateRequestTokens(id, 4, 2);
            ctx.monitor.updateRequestOutput(id, 3);
            ctx.monitor.completeRequest(.{ .id = id, .model = "model", .outcome = .success, .started_at_ms = 1000, .finished_at_ms = 1020, .ttft_ns = 5_000_000, .output_tokens = 3 });
            ctx.monitor.recordEvent(.{ .at_ms = 1020, .kind = "completed" });
            ctx.done.store(true, .release);
        }
    };
    var done = std.atomic.Value(bool).init(false);
    var ctx = Ctx{ .monitor = &monitor, .done = &done };
    monitor.lockHistory();
    const thread = std.Thread.spawn(.{}, Ctx.run, .{&ctx}) catch |err| {
        monitor.unlockHistory();
        return err;
    };
    const io = std.Io.Threaded.global_single_threaded.io();
    for (0..500) |_| {
        if (done.load(.acquire)) break;
        std.Io.sleep(io, .fromMilliseconds(1), .real) catch {};
    }
    const completed_while_history_locked = done.load(.acquire);
    monitor.unlockHistory();
    thread.join();
    try testing.expect(completed_while_history_locked);
    try testing.expectEqual(@as(u64, 1), monitor.ttftTotals().count);
    const snap = monitor.snapshot();
    try testing.expectEqual(@as(usize, 0), snap.active_len);
    try testing.expectEqual(@as(usize, 1), snap.request_len);
    try testing.expectEqual(@as(u32, 3), snap.requests[0].output_tokens);
    try testing.expectEqual(@as(usize, 1), snap.event_len);
    try testing.expectEqual(@as(usize, 1), snap.history_len);
    try testing.expectEqual(@as(u64, 1000), snap.lifetime_totals.?.at_ms);
}

test "archive compacts while retaining start and ordered cumulative samples" {
    const testing = std.testing;
    var monitor = Monitor.init();
    const count = history_capacity + archive_capacity * 5;
    for (0..count) |i| monitor.appendSample(.{
        .at_ms = @intCast(i * sample_interval_ms),
        .generation_tokens_live = @intCast(i),
        .process_bytes = @intCast(100 + i),
    });
    const snap = monitor.snapshot();
    try testing.expectEqual(@as(usize, history_capacity), snap.history_len);
    try testing.expect(snap.archive_compacted);
    try testing.expect(snap.archive_sample_interval_ms > sample_interval_ms);
    try testing.expect(snap.history_archive_len <= archive_capacity);
    try testing.expectEqual(@as(u64, 0), snap.history_archive[0].at_ms);
    for (snap.history_archive[1..snap.history_archive_len], 1..) |sample, i| {
        try testing.expect(sample.at_ms > snap.history_archive[i - 1].at_ms);
        try testing.expect(sample.at_ms < snap.history[0].at_ms);
        try testing.expectEqual(sample.at_ms / sample_interval_ms, sample.generation_tokens_live);
        try testing.expectEqual(@as(u64, 0), sample.continuity_id);
    }
    try testing.expectEqual(@as(u64, @intCast((count - 1) * sample_interval_ms)), snap.lifetime_totals.?.at_ms);
    try testing.expectEqual(@as(u64, @intCast(count - 1)), snap.lifetime_totals.?.generation_tokens_live);
    try testing.expectEqual(@as(f64, @floatFromInt((count - 1) * sample_interval_ms)) / 1000, snap.lifetime_totals.?.process_memory_observed_seconds_total);
}

test "memory integral skips missing and gapped pairs and continuity marks counter resets" {
    const testing = std.testing;
    var monitor = Monitor.init();
    monitor.appendSample(.{ .at_ms = 1000, .process_bytes = 100, .generation_tokens_live = 5 });
    monitor.appendSample(.{ .at_ms = 3000, .process_bytes = 300, .generation_tokens_live = 6 });
    monitor.appendSample(.{ .at_ms = 5000, .process_bytes = null, .generation_tokens_live = 7 });
    monitor.appendSample(.{ .at_ms = 7000, .process_bytes = 1000, .generation_tokens_live = 8 });
    monitor.appendSample(.{ .at_ms = 15000, .process_bytes = 2000, .generation_tokens_live = 9 });
    monitor.appendSample(.{ .at_ms = 17000, .process_bytes = 1000, .generation_tokens_live = 1 });
    monitor.appendSample(.{ .at_ms = 19000, .process_bytes = 0, .generation_tokens_live = 2 });
    monitor.appendSample(.{ .at_ms = 21000, .process_bytes = 1000, .generation_tokens_live = 3 });
    const snap = monitor.snapshot();
    try testing.expectEqual(@as(f64, 5_400), snap.lifetime_totals.?.process_memory_byte_seconds_total);
    try testing.expectEqual(@as(f64, 8), snap.lifetime_totals.?.process_memory_observed_seconds_total);
    try testing.expectEqual(@as(u64, 0), snap.history[3].continuity_id);
    try testing.expectEqual(@as(u64, 1), snap.history[4].continuity_id);
    try testing.expectEqual(@as(u64, 2), snap.history[5].continuity_id);
    try testing.expectEqual(@as(u64, 2), snap.history[7].continuity_id);
}

test "JSON escapes metadata and keeps unavailable measurements null" {
    const testing = std.testing;
    var monitor = Monitor.init();
    const id = monitor.beginRequest("model\"\\", 100);
    monitor.completeRequest(.{ .id = id, .model = "model\"\\", .outcome = .failed, .error_code = "bad\ncode", .started_at_ms = 100, .finished_at_ms = 105 });
    monitor.appendSample(.{ .at_ms = 105 });
    var output = std.Io.Writer.Allocating.init(testing.allocator);
    defer output.deinit();
    try monitor.renderJson(&output.writer, "");
    const json = try std.json.parseFromSlice(std.json.Value, testing.allocator, output.written(), .{});
    defer json.deinit();
    const requests = json.value.object.get("recent_requests").?.array.items;
    try testing.expectEqualStrings("model\"\\", requests[0].object.get("model").?.string);
    try testing.expectEqualStrings("bad\ncode", requests[0].object.get("error_code").?.string);
    try testing.expect(requests[0].object.get("queue_ms").? == .null);
    const history = json.value.object.get("history").?.array.items;
    try testing.expectEqual(@as(i64, 0), history[0].object.get("continuity_id").?.integer);
    try testing.expectEqual(@as(i64, 105), json.value.object.get("lifetime_totals").?.object.get("at_ms").?.integer);
    try testing.expectEqual(@as(usize, 0), json.value.object.get("history_archive").?.array.items.len);
    try testing.expectEqual(@as(i64, sample_interval_ms), json.value.object.get("retention").?.object.get("archive_sample_interval_ms").?.integer);
    try testing.expect(history[0].object.get("cpu_pct").? == .null);
    try testing.expect(history[0].object.get("cache_queries_total").? == .null);
    try testing.expect(history[0].object.get("cache_hits_total").? == .null);
    try testing.expect(history[0].object.get("ttft_ns_sum").? == .null);
    try testing.expect(history[0].object.get("ttft_count").? == .null);
}

test "JSON history preserves sampled cache and TTFT totals" {
    const testing = std.testing;
    var monitor = Monitor.init();
    monitor.appendSample(.{ .at_ms = 100, .cache_queries_total = 12, .cache_hits_total = 7, .ttft_ns_sum = 123_000_000, .ttft_count = 3, .prefill_active_ns_total = 1_200, .decode_active_ns_total = 800 });
    var output = std.Io.Writer.Allocating.init(testing.allocator);
    defer output.deinit();
    try monitor.renderJson(&output.writer, "");
    const json = try std.json.parseFromSlice(std.json.Value, testing.allocator, output.written(), .{});
    defer json.deinit();
    const sample = json.value.object.get("history").?.array.items[0].object;
    try testing.expectEqual(@as(i64, 12), sample.get("cache_queries_total").?.integer);
    try testing.expectEqual(@as(i64, 7), sample.get("cache_hits_total").?.integer);
    try testing.expectEqual(@as(i64, 123_000_000), sample.get("ttft_ns_sum").?.integer);
    try testing.expectEqual(@as(i64, 3), sample.get("ttft_count").?.integer);
    try testing.expectEqual(@as(i64, 1_200), sample.get("prefill_active_ns_total").?.integer);
    try testing.expectEqual(@as(i64, 800), sample.get("decode_active_ns_total").?.integer);
}

test "JSON archive and lifetime totals preserve compacted samples" {
    const testing = std.testing;
    var monitor = Monitor.init();
    for (0..history_capacity + archive_capacity + 2) |i| monitor.appendSample(.{
        .at_ms = @intCast(i * sample_interval_ms),
        .process_bytes = 100,
    });
    var output = std.Io.Writer.Allocating.init(testing.allocator);
    defer output.deinit();
    try monitor.renderJson(&output.writer, "");
    const json = try std.json.parseFromSlice(std.json.Value, testing.allocator, output.written(), .{});
    defer json.deinit();
    const root = json.value.object;
    const archive = root.get("history_archive").?.array.items;
    try testing.expect(archive.len <= archive_capacity);
    try testing.expectEqual(@as(i64, 0), archive[0].object.get("at_ms").?.integer);
    try testing.expectEqual(@as(i64, @intCast((history_capacity + archive_capacity + 1) * sample_interval_ms)), root.get("lifetime_totals").?.object.get("at_ms").?.integer);
    try testing.expectEqual(@as(i64, 4_000), root.get("retention").?.object.get("archive_sample_interval_ms").?.integer);
    try testing.expect(root.get("retention").?.object.get("archive_compacted").?.bool);
}

test "TTFT totals update as one successful completion pair" {
    const testing = std.testing;
    var monitor = Monitor.init();
    var recorded = std.atomic.Value(bool).init(false);
    const first = monitor.beginRequest("model", 0);
    if (claimCompletion(&recorded)) monitor.completeRequest(.{ .id = first, .model = "model", .outcome = .success, .started_at_ms = 0, .finished_at_ms = 1, .ttft_ns = 100_000_000 });
    if (claimCompletion(&recorded)) monitor.completeRequest(.{ .id = first, .model = "model", .outcome = .success, .started_at_ms = 0, .finished_at_ms = 1, .ttft_ns = 100_000_000 });
    try testing.expectEqualDeep(TtftTotals{ .ns_sum = 100_000_000, .count = 1 }, monitor.ttftTotals());

    const cancelled = monitor.beginRequest("model", 2);
    monitor.completeRequest(.{ .id = cancelled, .model = "model", .outcome = .cancelled, .started_at_ms = 2, .finished_at_ms = 3, .ttft_ns = 200_000_000 });
    const failed = monitor.beginRequest("model", 4);
    monitor.completeRequest(.{ .id = failed, .model = "model", .outcome = .failed, .started_at_ms = 4, .finished_at_ms = 5, .ttft_ns = 300_000_000 });
    const unknown = monitor.beginRequest("model", 6);
    monitor.completeRequest(.{ .id = unknown, .model = "model", .outcome = .success, .started_at_ms = 6, .finished_at_ms = 7 });
    try testing.expectEqualDeep(TtftTotals{ .ns_sum = 100_000_000, .count = 1 }, monitor.ttftTotals());

    const second = monitor.beginRequest("model", 8);
    monitor.completeRequest(.{ .id = second, .model = "model", .outcome = .success, .started_at_ms = 8, .finished_at_ms = 9, .ttft_ns = 250_000_000 });
    try testing.expectEqualDeep(TtftTotals{ .ns_sum = 350_000_000, .count = 2 }, monitor.ttftTotals());
}

test "request phases keep queue duration and model totals count each outcome" {
    const testing = std.testing;
    var monitor = Monitor.init();
    const first = monitor.beginRequest("model-a", 100);
    monitor.updateRequestPhase(first, .prefill, 130, 30);
    monitor.updateRequestTokens(first, 80, 20);
    monitor.updateRequestOutput(first, 3);
    monitor.updateRequestPhase(first, .decode, 180, null);
    var active = monitor.snapshot();
    try testing.expectEqual(Phase.decode, active.active[0].phase);
    try testing.expectEqual(@as(?u64, 30), active.active[0].queue_ms);
    try testing.expectEqual(@as(?u32, 80), active.active[0].prompt_tokens);
    try testing.expectEqual(@as(?u32, 3), active.active[0].output_tokens);
    monitor.completeRequest(.{ .id = first, .model = "model-a", .outcome = .success, .started_at_ms = 100, .finished_at_ms = 210, .queue_ms = 30, .prefill_ms = 50, .decode_ms = 30, .ttft_ms = 80, .prompt_tokens = 80, .output_tokens = 10 });
    const second = monitor.beginRequest("model-a", 220);
    monitor.completeRequest(.{ .id = second, .model = "model-a", .outcome = .cancelled, .started_at_ms = 220, .finished_at_ms = 230 });
    const third = monitor.beginRequest("model-b", 240);
    monitor.completeRequest(.{ .id = third, .model = "model-b", .outcome = .failed, .started_at_ms = 240, .finished_at_ms = 250 });
    active = monitor.snapshot();
    try testing.expectEqual(@as(usize, 0), active.active_len);
    try testing.expectEqual(@as(u64, 1), active.model_totals[0].success);
    try testing.expectEqual(@as(u64, 1), active.model_totals[0].cancelled);
    try testing.expectEqual(@as(u64, 1), active.model_totals[1].failed);
    try testing.expectEqual(@as(?u64, 30), active.requests[0].queue_ms);
    try testing.expectEqual(@as(u64, 110), active.requests[0].e2e_ms);
}

test "prefill credit and completion claim do not double count" {
    const testing = std.testing;
    var credited: u64 = 0;
    for ([_]u64{ 128, 256, 256, 400 }) |progress| {
        const delta = forwardedDelta(credited, progress);
        credited += delta;
    }
    try testing.expectEqual(@as(u64, 400), credited);
    credited += forwardedDelta(credited, 401);
    try testing.expectEqual(@as(u64, 401), credited);
    try testing.expectEqual(@as(u64, 0), forwardedDelta(credited, 0));
    var recorded = std.atomic.Value(bool).init(false);
    try testing.expect(claimCompletion(&recorded));
    try testing.expect(!claimCompletion(&recorded));
}

test "processing clock includes unfinished chunks and excludes idle time" {
    const testing = std.testing;
    var monitor = Monitor.init();
    const prefill_previous = monitor.beginProcessing(.prefill, 1_000);
    try testing.expectEqual(@as(?ProcessingPhase, null), prefill_previous);
    try testing.expectEqualDeep(ActiveTime{ .prefill_ns = 500, .decode_ns = 0 }, monitor.activeTime(1_500).?);
    const decode_previous = monitor.beginProcessing(.decode, 2_000);
    try testing.expectEqual(@as(?ProcessingPhase, .prefill), decode_previous);
    try testing.expectEqualDeep(ActiveTime{ .prefill_ns = 1_000, .decode_ns = 250 }, monitor.activeTime(2_250).?);
    const nested_previous = monitor.beginProcessing(.decode, 2_300);
    try testing.expectEqual(@as(?ProcessingPhase, .decode), nested_previous);
    monitor.endProcessing(nested_previous, 2_500);
    monitor.endProcessing(decode_previous, 2_700);
    monitor.endProcessing(prefill_previous, 3_000);
    try testing.expectEqualDeep(ActiveTime{ .prefill_ns = 1_300, .decode_ns = 700 }, monitor.activeTime(3_000).?);
    try testing.expectEqualDeep(ActiveTime{ .prefill_ns = 1_300, .decode_ns = 700 }, monitor.activeTime(10_000).?);
    const next_previous = monitor.beginProcessing(.decode, 12_000);
    monitor.endProcessing(next_previous, 12_100);
    try testing.expectEqualDeep(ActiveTime{ .prefill_ns = 1_300, .decode_ns = 800 }, monitor.activeTime(20_000).?);
}

test "processing clock reports unavailable if a write cannot finish" {
    const testing = std.testing;
    var monitor = Monitor.init();
    monitor.processing_clock.version.store(1, .seq_cst);
    try testing.expectEqual(@as(?ActiveTime, null), monitor.activeTime(1_000));
    monitor.appendSample(.{ .at_ms = 1_000, .prefill_active_ns_total = null, .decode_active_ns_total = null });
    var output = std.Io.Writer.Allocating.init(testing.allocator);
    defer output.deinit();
    try monitor.renderJson(&output.writer, "");
    const json = try std.json.parseFromSlice(std.json.Value, testing.allocator, output.written(), .{});
    defer json.deinit();
    const sample = json.value.object.get("history").?.array.items[0].object;
    try testing.expect(sample.get("prefill_active_ns_total").? == .null);
    try testing.expect(sample.get("decode_active_ns_total").? == .null);
}
