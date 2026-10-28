//! Slot-boundary history tests: real output delivery and durable storage, without a model forward.
const std = @import("std");
const testing = std.testing;
const abi = @import("../../abi/engine.zig");
const env = @import("../../env.zig");
const Slot = @import("slot.zig").Slot;
const Transformer = @import("transformer.zig").Transformer;
const KVCache = @import("transformer.zig").KVCache;
const Generator = @import("generate.zig").Generator;
const LoadedModel = @import("../../host/registry/registry.zig").LoadedModel;
const Tokenizer = @import("../../sdk/tokenizer/tokenizer.zig").Tokenizer;
const history = @import("spec/history.zig");
const storage = @import("spec/token_history.zig");
const Stopwatch = @import("../../sdk/util/io_util.zig").Stopwatch;

// The observer copies only events actually delivered by Slot. It neither generates
// tokens nor calls publication; returning stop models a disconnect during a block.
const Observer = struct {
    tokens: [32]u32 = @splat(0),
    len: usize = 0,
    finished: usize = 0,
    reason: ?@FieldType(abi.Event, "reason") = null,
    stop_after: usize = std.math.maxInt(usize),
    id: ?u64 = null,
    invalid: bool = false,

    fn sink(self: *Observer) abi.Sink {
        return .{ .ctx = self, .event = event };
    }

    fn event(ctx: *anyopaque, id: u64, ev: *const abi.Event) callconv(.c) abi.Action {
        const self: *Observer = @ptrCast(@alignCast(ctx));
        if (self.id) |prior| {
            if (prior != id) self.invalid = true;
        } else self.id = id;
        switch (ev.kind) {
            .prefilled => {},
            .tokens => {
                if (self.finished != 0) self.invalid = true;
                for (ev.tokens.slice()) |token| {
                    if (self.len == self.tokens.len) {
                        self.invalid = true;
                        return .stop;
                    }
                    self.tokens[self.len] = token;
                    self.len += 1;
                }
                if (self.len >= self.stop_after) return .stop;
            },
            .finished => {
                self.finished += 1;
                self.reason = ev.reason;
            },
        }
        return .go;
    }

    fn emitted(self: *const Observer) []const u32 {
        return self.tokens[0..self.len];
    }
};

const Fixture = struct {
    tmp: testing.TmpDir,
    path: [:0]u8,
    store_env: @TypeOf(env.override(.MLX_SERVE_LOOKUP_STORE, null)),
    corpus_env: @TypeOf(env.override(.MLX_SERVE_LOOKUP_CORPUS, null)),
    readonly_env: @TypeOf(env.override(.MLX_SERVE_LOOKUP_READ_ONLY, null)),
    tok: Tokenizer,
    xfm: Transformer,
    model: LoadedModel,

    fn init() !*Fixture {
        const a = testing.allocator;
        const self = try a.create(Fixture);
        errdefer a.destroy(self);
        var tmp = testing.tmpDir(.{});
        errdefer tmp.cleanup();
        var path_buf: [std.Io.Dir.max_path_bytes]u8 = undefined;
        const path = try a.dupeSentinel(u8, path_buf[0..try tmp.dir.realPath(testing.io, &path_buf)], 0);
        errdefer a.free(path);
        // Use the tokenizer parser's small-vocabulary fixture pattern, not an
        // identity-only stub. Its input is freed before any request uses it.
        const json = try a.dupe(u8,
            \\{"model":{"type":"BPE","vocab":{"a":0,"b":1,"c":2,"d":3,"e":4,"f":5,"g":6,"h":7,"i":8,"j":9,"k":10,"l":11,"m":12,"n":13,"o":14,"p":15},"merges":[]},"pre_tokenizer":{"type":"ByteLevel"},"added_tokens":[]}
        );
        var tok = blk: {
            defer a.free(json);
            break :blk try @import("../../sdk/tokenizer/load.zig").parseTokenizerContent(testing.io, a, json);
        };
        errdefer tok.deinit();
        // Same empty ownership shape as modules/shell.zig, but no scalar MLX
        // arrays: these tests never forward or sample. Every field read by Slot,
        // Request and Generator.deinit is initialized, with no undefined object.
        var xfm: Transformer = .{
            .config = .{ .num_hidden_layers = 0 },
            .cache = try KVCache.init(a, 0),
            .s = .{},
            .allocator = a,
            .emb_w = .{},
            .emb_s = .{},
            .emb_b = .{},
            .emb_scale = null,
            .lm_head_w = .{},
            .lm_head_s = .{},
            .lm_head_b = .{},
            .owns_lm_head = false,
            .gelu_coeff = null,
            .gelu_inner = null,
            .half = .{},
            .one = .{},
            .three = null,
            .moe_layers = null,
            .ssm_entries = null,
            .moe_seq_offset = 0,
            .prompt_cache = null,
        };
        errdefer xfm.cache.deinit();
        try storage.initializePersistent(testing.io, a, path, tok.content_sha256.?);
        self.* = .{
            .tmp = tmp,
            .path = path,
            .store_env = env.override(.MLX_SERVE_LOOKUP_STORE, path.ptr),
            .corpus_env = env.override(.MLX_SERVE_LOOKUP_CORPUS, null),
            .readonly_env = env.override(.MLX_SERVE_LOOKUP_READ_ONLY, "0"),
            .tok = tok,
            .xfm = xfm,
            .model = .{
                .allocator = a,
                .id = "history-slot-test",
                .path = "",
                .bytes_on_disk = null,
                .arch_hint = "",
                .config = &self.xfm.config,
                .weights = null,
                .transformer = &self.xfm,
                .tokenizer = &self.tok,
                .chat_config = null,
                .vision_encoder = null,
                .drafter = null,
                .drafter_path = "",
                .drafter_block_size = 0,
                .prefix_cache = null,
                .refcount = .init(0),
                .last_used_ns = 0,
                .bytes_resident = 0,
                .state = .ready,
                .error_name = null,
            },
        };
        return self;
    }

    fn deinit(self: *Fixture) void {
        if (self.xfm.token_history) |*cache| cache.deinit() catch unreachable;
        // The CPU shell owns no MLX handles; only its empty cache and history.
        self.xfm.cache.deinit();
        self.tok.deinit();
        self.readonly_env.restore();
        self.corpus_env.restore();
        self.store_env.restore();
        testing.allocator.free(self.path);
        self.tmp.cleanup();
        testing.allocator.destroy(self);
    }

    fn slot(self: *Fixture, prompt: []const u32, observer: ?*Observer) !*Slot {
        const result = try Slot.init(testing.allocator, testing.io, &self.xfm.config, .{
            .model = &self.model,
            // A restored request forwards only the tail, but publishes the full prompt.
            .prompt_ids = prompt[prompt.len - 1 ..],
            .full_prompt = prompt,
            .sampling = .{ .temperature = 0 },
            .eos_token_ids = &.{4},
            .max_tokens = 2,
            .sink = if (observer) |o| o.sink() else null,
        }, .dense);
        self.attachGenerator(result);
        return result;
    }

    fn attachGenerator(self: *Fixture, sl: *Slot) void {
        sl.legacy_gen = Generator{
            .xfm = &self.xfm,
            .ctx = sl.ctx,
            .tok = &self.tok,
            .next_token_id = 3,
            .step = 0,
            .max_tokens = sl.max_tokens,
            .sampling = sl.sampling,
            .prompt_tokens = @intCast(sl.full_prompt.len),
            .completion_tokens = 0,
            .finish_reason = "length",
            .done = false,
            .eos_token_ids = sl.eos_token_ids,
            .generated_ids = .empty,
            .timeout_ns = 0,
            .timer = Stopwatch.init(testing.io),
        };
        sl.state = .decoding;
    }

    fn expectDocuments(self: *Fixture, expected: []const u32) !void {
        // A different cache forces a fresh durable read, not the model's snapshot.
        var reader = storage.HistoryCache.init(testing.allocator);
        defer reader.deinit() catch unreachable;
        var lease = try reader.acquirePersistent(testing.io, self.path, self.tok.content_sha256.?);
        defer lease.deinit();
        try testing.expectEqualSlices(u32, expected, lease.index().tokens);
    }

    fn expectTurn(self: *Fixture, prompt: []const u32, emitted: []const u32) !void {
        var document: std.ArrayList(u32) = .empty;
        defer document.deinit(testing.allocator);
        try document.appendSlice(testing.allocator, prompt);
        try document.appendSlice(testing.allocator, emitted);
        try document.append(testing.allocator, storage.separator);
        try self.expectDocuments(document.items);
    }
};

fn commit(sl: *Slot, ids: []const u32) !void {
    try sl.legacy_gen.?.generated_ids.appendSlice(testing.allocator, ids);
}

fn delivered(sl: *Slot, observer: *Observer, streaming: bool) ![]const u32 {
    if (streaming) {
        try testing.expectEqual(@as(usize, 0), sl.out_buf.items.len);
        try testing.expect(!observer.invalid);
        return observer.emitted();
    }
    // Observe the actual connection-thread interface, not just the producer's buffer.
    while (true) switch (sl.waitNext()) {
        .token => |id| {
            observer.tokens[observer.len] = id;
            observer.len += 1;
        },
        .done => break,
        .err => return error.UnexpectedSlotError,
    };
    return observer.emitted();
}

test "history Slot early duplicate publishes delivered sink and non-sink tokens once" {
    for ([_]bool{ true, false }) |streaming| {
        const rig = try Fixture.init();
        defer rig.deinit();
        var observer: Observer = .{};
        const sl = try rig.slot(&.{ 1, 2 }, if (streaming) &observer else null);
        var live = true;
        defer if (live) sl.deinit();
        // publishFirstToken emits before the generator commits; its decoder push is swallowed.
        sl.pushTokenWithLogprob(3, null);
        sl.early_first = 3;
        try commit(sl, &.{ 3, 4, 5 });
        sl.pushTokenWithLogprob(3, null);
        sl.pushTokenWithLogprob(4, null);
        sl.markFinished("length");
        sl.markFinished("length");
        const output = try delivered(sl, &observer, streaming);
        try testing.expectEqualSlices(u32, &.{ 3, 4 }, output);
        try rig.expectTurn(&.{ 1, 2 }, output);
        sl.deinit();
        live = false;
        try rig.expectTurn(&.{ 1, 2 }, output);
        if (streaming) try testing.expectEqual(@as(usize, 1), observer.finished);
    }
}

test "history Slot cancellation before generator commit keeps early engine event" {
    for ([_]bool{ true, false }) |streaming| {
        for ([_]bool{ true, false }) |terminal_call| {
            const rig = try Fixture.init();
            defer rig.deinit();
            var observer: Observer = .{ .stop_after = 1 };
            const sl = try rig.slot(&.{ 1, 2 }, if (streaming) &observer else null);
            var live = true;
            defer if (live) sl.deinit();
            // Prefill can publish before legacy_gen has been installed on the slot.
            sl.legacy_gen.?.deinit(testing.allocator);
            sl.legacy_gen = null;
            sl.pushTokenWithLogprob(3, null);
            sl.early_first = 3;
            if (!streaming) sl.cancel();
            try testing.expect(sl.cancelled.load(.acquire));
            if (terminal_call) sl.markFinished("cancelled");
            const output = try delivered(sl, &observer, streaming);
            try testing.expectEqualSlices(u32, &.{3}, output);
            sl.deinit();
            live = false;
            try rig.expectTurn(&.{ 1, 2 }, output);
            if (streaming) {
                try testing.expectEqual(@as(usize, 1), observer.finished);
                try testing.expectEqual(.cancelled, observer.reason.?);
            }
        }
    }
}

test "history Slot cancelled EOS and length boundaries exclude committed unseen tail" {
    for ([_][]const u8{ "cancelled", "stop", "length" }) |reason| {
        for ([_]bool{ true, false }) |streaming| {
            const rig = try Fixture.init();
            defer rig.deinit();
            var observer: Observer = .{ .stop_after = if (std.mem.eql(u8, reason, "cancelled")) 2 else std.math.maxInt(usize) };
            const sl = try rig.slot(&.{ 1, 2 }, if (streaming) &observer else null);
            defer sl.deinit();
            try commit(sl, &.{ 3, 4, 5, 6 });
            sl.pushTokenWithLogprob(3, null);
            sl.pushTokenWithLogprob(4, null);
            if (std.mem.eql(u8, reason, "cancelled")) {
                if (!streaming) sl.cancel();
                try testing.expect(sl.cancelled.load(.acquire));
            }
            sl.markFinished(reason);
            const output = try delivered(sl, &observer, streaming);
            try testing.expectEqualSlices(u32, &.{ 3, 4 }, output);
            try rig.expectTurn(&.{ 1, 2 }, output);
            if (streaming) {
                try testing.expectEqual(@as(usize, 1), observer.finished);
                try testing.expectEqualStrings(reason, @tagName(observer.reason.?));
            }
        }
    }
}

test "history Slot failures after output never append even through finish and cleanup" {
    const Failure = enum { mark_error, already_errored_finish, already_errored_deinit, output_allocation };
    for ([_]Failure{ .mark_error, .already_errored_finish, .already_errored_deinit, .output_allocation }) |failure| {
        const rig = try Fixture.init();
        defer rig.deinit();
        var observer: Observer = .{};
        const sl = try rig.slot(&.{ 1, 2 }, if (failure == .output_allocation) null else &observer);
        var live = true;
        defer if (live) sl.deinit();
        try commit(sl, &.{ 3, 4, 5 });
        sl.pushTokenWithLogprob(3, null);
        switch (failure) {
            .mark_error => {
                sl.markError("ForwardFailed");
                sl.markError("ForwardFailed");
                sl.markFinished("length");
            },
            .already_errored_finish => {
                sl.state = .errored;
                sl.markFinished("length");
            },
            .already_errored_deinit => sl.state = .errored,
            .output_allocation => {
                // Preserve the first actual output, then force the next append to allocate.
                sl.out_buf.shrinkAndFree(testing.allocator, sl.out_buf.items.len);
                var failing = testing.FailingAllocator.init(testing.allocator, .{ .fail_index = 0, .resize_fail_index = 0 });
                sl.allocator = failing.allocator();
                sl.pushTokenWithLogprob(4, null);
                sl.allocator = testing.allocator;
                try testing.expectEqual(.errored, sl.state);
                try testing.expectEqualSlices(u32, &.{3}, sl.out_buf.items);
                sl.markFinished("length");
            },
        }
        try rig.expectDocuments(&.{});
        sl.deinit();
        live = false;
        try rig.expectDocuments(&.{});
        if (failure != .output_allocation) {
            try testing.expectEqualSlices(u32, &.{3}, observer.emitted());
            try testing.expectEqual(@as(usize, 1), observer.finished);
        }
    }
}

test "history Slot generator retry teardown does not publish an attempt" {
    const rig = try Fixture.init();
    defer rig.deinit();
    var observer: Observer = .{};
    const sl = try rig.slot(&.{ 1, 2 }, &observer);
    defer sl.deinit();
    _ = try history.match(&sl.legacy_gen.?, testing.allocator, &.{1}, 2, 4);
    try commit(sl, &.{ 5, 6 });
    // The retry owns internal generated state but emitted no token. This is the
    // same teardown seam used when a restored prefill is discarded and rebuilt.
    sl.legacy_gen.?.deinit(testing.allocator);
    sl.legacy_gen = null;
    try rig.expectDocuments(&.{});
    rig.attachGenerator(sl);
    try commit(sl, &.{3});
    sl.pushTokenWithLogprob(3, null);
    sl.markFinished("length");
    try rig.expectTurn(&.{ 1, 2 }, observer.emitted());
}

test "history overlapping Slots finish B then A without losing turns or replacing A lease" {
    const rig = try Fixture.init();
    defer rig.deinit();
    try storage.appendFinalTurn(testing.io, testing.allocator, rig.path, rig.tok.content_sha256.?, &.{ 12, 13 }, &.{14});
    var oa: Observer = .{};
    var ob: Observer = .{};
    const a = try rig.slot(&.{ 1, 2 }, &oa);
    defer a.deinit();
    const b = try rig.slot(&.{ 8, 9 }, &ob);
    defer b.deinit();
    const first = (try history.match(&a.legacy_gen.?, testing.allocator, &.{12}, 13, 4)).?;
    try testing.expectEqualSlices(u32, &.{14}, first.draft);
    _ = try history.match(&b.legacy_gen.?, testing.allocator, &.{12}, 13, 4);
    const lease_a = a.legacy_gen.?.external_history.?.persistent.?.index();
    try testing.expectEqual(lease_a, b.legacy_gen.?.external_history.?.persistent.?.index());
    try commit(a, &.{ 3, 4 });
    a.pushTokenWithLogprob(3, null);
    try commit(b, &.{ 10, 11 });
    b.pushTokenWithLogprob(10, null);
    b.pushTokenWithLogprob(11, null);
    b.markFinished("length");
    try rig.expectDocuments(&.{ 12, 13, 14, storage.separator, 8, 9, 10, 11, storage.separator });
    try testing.expect((try history.match(&a.legacy_gen.?, testing.allocator, &.{8}, 9, 4)) == null);
    const retained = (try history.match(&a.legacy_gen.?, testing.allocator, &.{12}, 13, 4)).?;
    try testing.expectEqualSlices(u32, &.{14}, retained.draft);
    try testing.expectEqual(lease_a, a.legacy_gen.?.external_history.?.persistent.?.index());
    a.pushTokenWithLogprob(4, null);
    a.markFinished("length");
    a.markFinished("length");
    b.markFinished("length");
    var expected: std.ArrayList(u32) = .empty;
    defer expected.deinit(testing.allocator);
    try expected.appendSlice(testing.allocator, &.{ 12, 13, 14, storage.separator, 8, 9 });
    try expected.appendSlice(testing.allocator, ob.emitted());
    try expected.appendSlice(testing.allocator, &.{ storage.separator, 1, 2 });
    try expected.appendSlice(testing.allocator, oa.emitted());
    try expected.append(testing.allocator, storage.separator);
    try rig.expectDocuments(expected.items);
    try testing.expectEqual(@as(usize, 1), oa.finished);
    try testing.expectEqual(@as(usize, 1), ob.finished);
}

test "history cleanup stays queued without allocation until the inference owner drains it" {
    const scheduler = @import("scheduler.zig");
    const loop = @import("runtime/loop.zig");
    const rig = try Fixture.init();
    defer rig.deinit();
    try storage.appendFinalTurn(testing.io, testing.allocator, rig.path, rig.tok.content_sha256.?, &.{12}, &.{13});
    var failing = testing.FailingAllocator.init(testing.allocator, .{ .fail_index = 0, .resize_fail_index = 0 });
    var sch: scheduler.Scheduler = undefined;
    sch.allocator = failing.allocator();
    sch.io = testing.io;
    sch.queue_mu = .init;
    sch.queue_cond = .init;
    sch.submit_cond = .init;
    sch.pending = .empty;
    defer sch.pending.deinit(testing.allocator);
    sch.decoding = .empty;
    sch.offloaded = .empty;
    sch.cleanup_queue = .{};
    sch.vision_queue = .empty;
    sch.embed_queue = .empty;
    sch.load_queue = .empty;
    sch.gen_queue = .empty;
    sch.unload_queue = .empty;
    sch.metrics = null;
    sch.hot_prefix_cache = null;
    sch.hot_cache_digests = &.{};
    sch.digest_mu = .init;
    sch.digest_residency = 0;
    sch.resident_hot_cache_bytes = .init(0);
    sch.reclaimable_hot_cache_bytes = .init(0);
    var observers: [18]Observer = @splat(.{});
    var slots: [18]*Slot = undefined;
    var owned: usize = 0;
    defer for (slots[0..owned]) |sl| sl.deinit();
    for (&slots, &observers, 0..) |*sl, *observer, i| {
        sl.* = try rig.slot(&.{@intCast(i)}, observer);
        owned += 1;
        _ = try history.match(&sl.*.legacy_gen.?, testing.allocator, &.{12}, 13, 4);
        try commit(sl.*, &.{ 3, 4 });
        sl.*.pushTokenWithLogprob(3, null);
        try sch.pending.append(testing.allocator, sl.*);
    }
    sch.in_flight = slots.len;
    slots[0].in_pass.store(1, .release);
    const Connection = struct {
        fn run(s: *scheduler.Scheduler, batch: []const *Slot) void {
            for (batch, 0..) |sl, i| {
                if (i == batch.len - 1) sl.markError("ForwardFailed");
                scheduler.complete(s, sl);
                scheduler.complete(s, sl);
            }
        }
    };
    const conn = try std.Thread.spawn(.{}, Connection.run, .{ &sch, &slots });
    var joined = false;
    defer if (!joined) {
        slots[0].in_pass.store(0, .release);
        conn.join();
    };
    while (true) {
        sch.queue_mu.lockUncancelable(testing.io);
        const removed = sch.pending.items.len < slots.len;
        sch.queue_mu.unlock(testing.io);
        if (removed) break;
        std.Io.sleep(testing.io, .fromMilliseconds(1), .real) catch {};
    }
    scheduler.complete(&sch, slots[0]);
    try testing.expectEqual(slots.len, sch.in_flight);
    // The pass still borrows this slot; completion must neither queue nor release it yet.
    try testing.expect(sch.cleanup_queue.first == null);
    var neighbor = try history.cache(&rig.xfm).acquirePersistent(testing.io, rig.path, rig.tok.content_sha256.?);
    defer neighbor.deinit();
    const snapshot = neighbor.index();
    try testing.expectEqual(snapshot, slots[0].legacy_gen.?.external_history.?.persistent.?.index());
    slots[0].in_pass.store(0, .release);
    conn.join();
    joined = true;
    try testing.expectEqual(@as(usize, 0), sch.in_flight);
    try testing.expectEqual(@as(usize, 0), sch.pending.items.len);
    for (slots, observers) |sl, observer| {
        try testing.expect(!sl.history_publication.finalized);
        try testing.expectEqual(@as(usize, if (sl.state == .errored) 1 else 0), observer.finished);
    }
    try rig.expectDocuments(&.{ 12, 13, storage.separator });
    try testing.expect(loop.hasWorkPendingLocked(&sch));
    owned = 0; // The real drain now owns and destroys every queued slot.
    defer while (loop.drainSideQueues(&sch)) {};
    try testing.expect(loop.drainSideQueues(&sch));
    try testing.expect(loop.hasWorkPendingLocked(&sch));
    for (observers[0..16]) |observer| try testing.expectEqual(@as(usize, 1), observer.finished);
    try testing.expectEqual(@as(usize, 0), observers[16].finished);
    try testing.expect(loop.drainSideQueues(&sch));
    try testing.expect(!loop.hasWorkPendingLocked(&sch));
    try testing.expect(!loop.drainSideQueues(&sch));
    for (observers) |observer| try testing.expectEqual(@as(usize, 1), observer.finished);
    try testing.expectEqualSlices(u32, &.{ 12, 13, storage.separator }, snapshot.tokens);
    var expected: std.ArrayList(u32) = .empty;
    defer expected.deinit(testing.allocator);
    try expected.appendSlice(testing.allocator, snapshot.tokens);
    for (0..17) |i| try expected.appendSlice(testing.allocator, &.{ @intCast(i), 3, storage.separator });
    try rig.expectDocuments(expected.items);
}
