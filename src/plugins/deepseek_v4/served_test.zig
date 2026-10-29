//! The flip proof for both DeepSeek plugins: a tiny pack served by the plugin behind the driver's shell (the engine's
//! parse of a served type, the driver's map taken through the in-tree hatch), driven through `forwardWith` and the
//! generator as the scheduler drives them: chunked prefill, host and pipelined device decodes, a resumed prompt, image
//! rows at a chunk boundary, DSpark rounds. Every value is checked against the deleted legacy forward's, recorded in
//! the tapes in `fixtures/` (`engines/mlx/plugin/tape.zig`).

const std = @import("std");
const abi = @import("../../abi/plugin.zig");
const mlx = @import("../../mlx/ffi.zig");
const ops = @import("../../mlx/ops.zig");
const model = @import("../../sdk/config/model.zig");
const driver = @import("../../engines/mlx/plugin/driver.zig");
const lazyForward = @import("../../engines/mlx/generate/step.zig").lazyForward;
const KVCache = @import("../../engines/mlx/kv/cache.zig").KVCache;
const KVQuantConfig = @import("../../engines/mlx/cache/kv_quant.zig").KVQuantConfig;
const ForwardCtx = @import("../../engines/mlx/forward_ctx.zig").ForwardCtx;
const Transformer = @import("../../engines/mlx/transformer.zig").Transformer;
const sdk = @import("../../sdk/root.zig");
const fixture = @import("fixture.zig");
const intree = @import("../../engines/mlx/plugin/intree.zig");
const prefix_cache = @import("../../engines/mlx/cache/prefix_cache.zig");
const verdict = @import("../../engines/mlx/batching/verdict.zig");
const env = @import("../../env.zig");
const Generator = @import("../../engines/mlx/generate/generator.zig").Generator;
const SamplingParams = @import("../../engines/mlx/sampling/params.zig").SamplingParams;
const Tokenizer = @import("../../sdk/tokenizer/tokenizer.zig").Tokenizer;
const Tape = @import("../../engines/mlx/plugin/tape.zig").Tape;
const prefill_request = @import("../../engines/mlx/prefill.zig");
const commit_request = @import("../../engines/mlx/commit.zig");
const finish_request = @import("../../engines/mlx/finish.zig");
const scheduler = @import("../../engines/mlx/scheduler.zig");
const RequestSlot = @import("../../engines/mlx/slot.zig").Slot;
const registry = @import("../../host/registry/registry.zig");

const testing = std.testing;
const Array = mlx.mlx_array;

/// One request's cache and context.
const Slot = struct {
    kv: KVCache,
    off: usize = 0,
    ctx: ForwardCtx = undefined,

    fn init(s: *Slot) !void {
        s.* = .{ .kv = try KVCache.init(testing.allocator, 1) };
        s.ctx = .{ .cache = &s.kv, .moe_seq_offset = &s.off, .ssm_entries = null, .capture_hidden = null, .vision_embeddings = null };
    }
};

/// The pack served by its plugin, its expected values played back from a tape.
const Served = struct {
    tmp: testing.TmpDir,
    cfg: model.ModelConfig,
    xfm: Transformer,
    slot: Slot,
    tape: Tape,
    /// The values since the last greedy run depend on its tokens: compared only on the recording's device.
    after_greedy: bool = false,

    fn open(table: *const abi.ModelTable, files: []const sdk.fixture.File, tape: []const u8) !*Served {
        const a = testing.allocator;
        const io = std.Io.Threaded.global_single_threaded.io();
        const p = try a.create(Served);
        errdefer a.destroy(p);
        p.after_greedy = false;
        p.tape = if (tape.len > 0) try Tape.player(tape) else .{};
        p.tmp = testing.tmpDir(.{});
        errdefer p.tmp.cleanup();
        for (files) |f| try p.tmp.dir.writeFile(io, .{ .sub_path = f.name, .data = f.bytes });
        var buf: [512]u8 = undefined;
        const dir = buf[0..try p.tmp.dir.realPath(io, &buf)];
        // The engine's parse of a served type: the legacy parse with the plugin's binding attached.
        p.cfg = try model.parseConfig(io, a, dir);
        errdefer p.cfg.deinit(a);
        if (p.cfg.plugin == null or p.cfg.plugin.?.table != table) return error.NotServed;
        p.xfm = try driver.load(io, a, &p.cfg, dir, mlx.gpuStream());
        try p.slot.init();
        return p;
    }

    fn close(p: *Served) void {
        p.slot.kv.deinit();
        p.xfm.deinit();
        p.cfg.deinit(testing.allocator);
        p.tmp.cleanup();
        testing.allocator.destroy(p);
    }

    /// Host integers, checked like an output.
    fn checkInts(p: *Served, g: *ops.Graph, got: []const u32, tokens: bool) !void {
        try p.tape.expectNext(try g.fromSlice(u32, got, &.{@intCast(got.len)}), tokens);
    }

    /// One chunk; its logits (owned by `g`), checked.
    fn step(p: *Served, g: *ops.Graph, ids: []const i32) !Array {
        const out = try g.adopt(try p.xfm.forwardWith(&p.slot.ctx, try g.fromSlice(i32, ids, &.{ 1, @intCast(ids.len) })));
        try p.tape.expectNext(out, p.after_greedy);
        return out;
    }

    /// Greedy decodes through the pipelined step (the token stays a lazy argmax on the GPU): tokens and logits.
    fn greedy(p: *Served, g: *ops.Graph, from: Array, n: usize) !void {
        p.after_greedy = true;
        var last = from;
        for (0..n) |_| {
            const sh = ops.shape(last);
            const next = try g.argmax(try g.reshape(try g.sliceAxis(last, 1, sh[1] - 1, sh[1]), &.{ 1, sh[2] }), -1, false);
            last = try g.adopt(try lazyForward(&p.xfm, &p.slot.ctx, next));
            const host = try g.astype(next, .uint32);
            try mlx.check(mlx.mlx_array_eval(host));
            try p.checkInts(g, mlx.mlx_array_data_uint32(host).?[0..1], true);
            try p.tape.expectNext(last, true);
        }
    }

    /// A new request: position 0, and the values no longer follow the greedy run.
    fn restart(p: *Served) void {
        p.slot.kv.step = 0;
        p.after_greedy = false;
    }
};

fn expectServed(comptime P: type, files: []const sdk.fixture.File, tape: []const u8) !void {
    const p = try Served.open(&P.table, files, tape);
    defer p.close();
    var g = ops.Graph.init(testing.allocator, mlx.gpuStream());
    defer g.deinit();
    // The checks the scheduler reads: module-owned state and the module's own drafter; V4 runs serial,
    // V4.1 batches its slots (`Caps.slot_state`).
    const sx = &p.xfm;
    try testing.expect(sx.ownsModuleDecodeState() and sx.moduleSpecWiring() and !sx.moduleStateSpecRollback() and !sx.supportsLayerCapture());
    try testing.expectEqual(p.cfg.plugin.?.parsed.caps.slot_state, prefix_cache.HotPrefixCache.shouldUseInstance(&p.cfg, true, sx));
    try testing.expectEqual(p.cfg.plugin.?.parsed.caps.slot_state, verdict.modelBatchable(&p.cfg));
    try testing.expectEqual(@as(usize, 0), sx.dsparkStages());
    // The hatch hands engine code the plugin's own module.
    const mod: ?*anyopaque = if (intree.dsv4(sx)) |x| x else if (intree.dsv41(sx)) |x| x else null;
    try testing.expectEqual(@as(?*anyopaque, P.moduleOf(intree.module(sx).?)), mod);
    _ = try p.step(&g, &.{ 3, 17, 42, 9, 7, 25, 1 });
    _ = try p.step(&g, &.{8});
    const l = try p.step(&g, &.{ 4, 33, 7, 40 });
    try p.greedy(&g, l, 5);
    _ = try p.step(&g, &.{2});
    // A new request at position 0 starts a fresh state.
    p.restart();
    _ = try p.step(&g, &.{ 9, 8, 7 });
    try p.tape.expectDone();
}

test "deepseek_v4 served: chunked prefill, host and pipelined decodes and a new request equal the legacy forward bit for bit" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    try expectServed(@import("plugin.zig"), try fixture.v4Files(arena.allocator(), 11), @embedFile("fixtures/served.tape"));
}

test "deepseek_v41 served: chunked prefill, host and pipelined decodes and a new request equal the legacy forward bit for bit" {
    if (!mlx.metalKernelsAvailable()) return error.SkipZigTest;
    if (mlx.noGpuBackend() or mlx.virtualGpu()) return error.SkipZigTest;
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    try expectServed(@import("../deepseek_v41/plugin.zig"), try fixture.v41Files(arena.allocator(), 7), @embedFile("../deepseek_v41/fixtures/served.tape"));
}

test "deepseek_v41 served: two slots decode in one batched tick through the driver, each slot's step following its state" {
    if (!mlx.metalKernelsAvailable()) return error.SkipZigTest;
    if (mlx.noGpuBackend() or mlx.virtualGpu()) return error.SkipZigTest;
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const p = try Served.open(&@import("../deepseek_v41/plugin.zig").table, try fixture.v41Files(arena.allocator(), 7), "");
    defer p.close();
    var g = ops.Graph.init(testing.allocator, mlx.gpuStream());
    defer g.deinit();
    var other: Slot = undefined;
    try other.init();
    defer other.kv.deinit();
    const ctxs = [_]*ForwardCtx{ &p.slot.ctx, &other.ctx };
    const prompts = [_][]const i32{ &.{ 3, 17, 42, 9, 7, 25, 1, 60, 11, 2, 90 }, &.{ 5, 8, 13, 21, 34, 55, 89, 4, 6, 10, 12, 14, 16, 18 } };
    for (ctxs, prompts) |c, ids| _ = try g.adopt(try p.xfm.forwardWith(c, try g.fromSlice(i32, ids, &.{ 1, @intCast(ids.len) })));
    for (0..3) |i| {
        const rows = try driver.forwardBatched(&p.xfm, p.xfm.plugin.?, &.{ 44, 9 }, &ctxs, &.{ @intCast(ctxs[0].cache.step), @intCast(ctxs[1].cache.step) }, null);
        defer {
            for (rows) |r| _ = mlx.mlx_array_free(r);
            p.xfm.allocator.free(rows);
        }
        for (ctxs, prompts) |c, ids| try testing.expectEqual(ids.len + i + 1, c.cache.step);
    }
}

test "deepseek_v41 served: disk-disabled requests retain alternating RAM conversations after slot release" {
    if (!mlx.metalKernelsAvailable()) return error.SkipZigTest;
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const p = try Served.open(&@import("../deepseek_v41/plugin.zig").table, try fixture.v41Files(arena.allocator(), 7), "");
    defer p.close();
    const h = try Requests.open(p);
    defer h.close();
    try testing.expect(h.entry.prefix_cache.?.disk == null);
    const a = [_]u32{ 3, 17, 42, 9, 7, 25, 1, 60 };
    const b = [_]u32{ 5, 8, 13, 21, 34, 55, 89, 4 };
    for ([_][]const u32{ &a, &b, &a, &b, &a, &b }, 0..) |prompt, i| {
        var g = ops.Graph.init(testing.allocator, mlx.gpuStream());
        defer g.deinit();
        const s = try h.prefill(prompt);
        defer s.deinit();
        try testing.expectEqual(@as(u32, if (i < 2) 0 else 7), s.cached_tokens);
        try testing.expectEqual(@as(usize, prompt.len), s.cache.step);
        const probe = try g.fromSlice(i32, &.{44}, &.{ 1, 1 });
        const warm_logits = try g.adopt(try p.xfm.forwardWith(&s.legacy_gen.?.ctx, probe));
        try ops.eval(warm_logits);
        finish_request.finishSlot(&h.sch, s, "length");
        finish_request.releasePluginSlot(s);
        try testing.expectEqual(@as(u32, 0), s.ctx.plugin_slot);
        try testing.expectEqual(@as(u32, 0), s.legacy_gen.?.ctx.plugin_slot);
        try testing.expectEqual(@as(u64, 0), driver.stateBytes(&p.xfm));
        try testing.expectEqual(@as(u64, 0), h.sch.resident_live_kv_bytes.load(.monotonic));
        try testing.expectEqual(@as(usize, @min(i + 1, 2)), h.entry.prefix_cache.?.entries.items.len);
        // A separately initialized cold slot must choose the same first token.
        const warm_token = s.legacy_gen.?.next_token_id;
        const cold = try h.newSlot(prompt);
        defer cold.deinit();
        cold.skip_prefix_cache = true;
        try prefill_request.runPrefill(&h.sch, cold);
        try testing.expectEqual(@as(u32, 0), cold.cached_tokens);
        try testing.expectEqual(warm_token, cold.legacy_gen.?.next_token_id);
        const cold_logits = try g.adopt(try p.xfm.forwardWith(&cold.legacy_gen.?.ctx, probe));
        try testing.expect(try sameValues(&g, warm_logits, cold_logits));
        finish_request.releasePluginSlot(cold);
    }
}

/// Real request functions over a tiny plugin and a registry; no inference worker is started.
const Requests = struct {
    sch: scheduler.Scheduler = undefined,
    entry: *registry.LoadedModel,
    tok: Tokenizer,

    fn open(p: *Served) !*Requests {
        const a = testing.allocator;
        const h = try a.create(Requests);
        errdefer a.destroy(h);
        const io = std.Io.Threaded.global_single_threaded.io();
        h.sch.registry = try registry.ModelRegistry.init(a, io, null, 4, 0, null);
        h.entry = try h.sch.registry.registerStub("fixture", "/fixture", 1);
        h.tok = Tokenizer.initEmptyForTests(a, .byte_level_bpe);
        h.entry.config = &p.cfg;
        h.entry.transformer = &p.xfm;
        h.entry.tokenizer = &h.tok;
        h.entry.state = .ready;
        h.entry.prefix_cache = try a.create(prefix_cache.HotPrefixCache);
        h.entry.prefix_cache.?.* = prefix_cache.HotPrefixCache.init(a, 4);
        h.sch.allocator = a;
        h.sch.io = io;
        h.sch.metrics = null;
        h.sch.queue_mu = .init;
        h.sch.decoding = .empty;
        h.sch.pending = .empty;
        h.sch.cleanup_queue = .empty;
        h.sch.vision_queue = .empty;
        h.sch.embed_queue = .empty;
        h.sch.max_concurrent = 4;
        h.sch.boundary_depth = 0;
        h.sch.boundary_outer = null;
        h.sch.hot_prefix_cache = h.entry.prefix_cache;
        h.sch.digest_mu = .init;
        h.sch.hot_cache_digests = &.{};
        h.sch.digest_residency = 0;
        h.sch.resident_hot_cache_bytes = .init(0);
        h.sch.reclaimable_hot_cache_bytes = .init(0);
        h.sch.resident_live_kv_bytes = .init(0);
        return h;
    }

    fn close(h: *Requests) void {
        h.entry.transformer = null;
        h.entry.config = null;
        h.entry.tokenizer = null;
        h.sch.registry.deinit();
        h.sch.cleanup_queue.deinit(testing.allocator);
        testing.allocator.free(h.sch.hot_cache_digests);
        h.tok.deinit();
        testing.allocator.destroy(h);
    }

    fn newSlot(h: *Requests, prompt: []const u32) !*RequestSlot {
        return RequestSlot.init(testing.allocator, h.sch.io, h.entry.config.?, .{
            .model = h.entry,
            .prompt_ids = prompt,
            .sampling = .{ .temperature = 0 },
            .eos_token_ids = &.{},
            .max_tokens = 2,
        }, KVQuantConfig.dense);
    }

    fn prefill(h: *Requests, prompt: []const u32) !*RequestSlot {
        const s = try h.newSlot(prompt);
        errdefer s.deinit();
        try prefill_request.runPrefill(&h.sch, s);
        return s;
    }

    fn attachDisk(h: *Requests, p: *Served) !void {
        var buf: [512]u8 = undefined;
        const path = buf[0..try p.tmp.dir.realPath(h.sch.io, &buf)];
        const hc = h.entry.prefix_cache.?;
        hc.disk = try @import("../../engines/mlx/load.zig").initPrefixDisk(testing.allocator, h.sch.io, path, "module-requests", 0, 512, prefix_cache.ModuleBinding.init(&p.xfm).?, path);
        hc.disk.?.armTestSpace(1 << 40, 2 << 40);
        hc.armDiskTier(&p.cfg);
        try testing.expect(!hc.ssd_first);
        try testing.expect(hc.disk.?.writer != null);
    }

    fn finishDisk(h: *Requests) !void {
        const hc = h.entry.prefix_cache.?;
        for (0..32) |_| {
            const result = hc.flushOwedSlice(h.entry.transformer.?.s);
            hc.disk.?.drainWriter();
            hc.disk.?.pollModuleWrites();
            if (result == .done) return;
        }
        return error.ModuleFlushDidNotFinish;
    }

    fn seedDisk(h: *Requests, prompt: []const u32) !u64 {
        const s = try h.prefill(prompt);
        defer s.deinit();
        return h.commitDisk(s);
    }

    fn commitDisk(h: *Requests, s: *RequestSlot) !u64 {
        commit_request.commitSlotIfApplicable(&h.sch, s);
        finish_request.releasePluginSlot(s);
        const hc = h.entry.prefix_cache.?;
        const id = hc.entries.items[0].module.?.disk_id.?;
        try h.finishDisk();
        try testing.expect(hc.disk.?.entryWholeOnDisk(id));
        hc.setBudget(1);
        hc.setBudget(0);
        try testing.expectEqual(@as(usize, 0), hc.entries.items.len);
        return id;
    }

    fn mediaSlot(h: *Requests, prompt: []const u32, rows: Array, media: []const prefix_cache.MediaSpan) !*RequestSlot {
        var owned = mlx.mlx_array_new();
        errdefer _ = mlx.mlx_array_free(owned);
        try mlx.check(mlx.mlx_array_set(&owned, rows));
        return RequestSlot.init(testing.allocator, h.sch.io, h.entry.config.?, .{
            .model = h.entry,
            .prompt_ids = prompt,
            .vision_embeddings = owned,
            .media = media,
            .sampling = .{ .temperature = 0 },
            .eos_token_ids = &.{},
            .max_tokens = 2,
        }, KVQuantConfig.dense);
    }
};

fn diskPrompt() [513]u32 {
    var prompt: [513]u32 = undefined;
    for (&prompt, 0..) |*id, i| id.* = @intCast(3 + i % 80);
    return prompt;
}

test "deepseek_v41 served: disk import restores exact counters and cold-reference logits" {
    if (!mlx.metalKernelsAvailable() or mlx.noGpuBackend() or mlx.virtualGpu()) return error.SkipZigTest;
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    for ([_]bool{ false, true }) |dspark| {
        const files = if (dspark) try fixture.v41DsparkFiles(arena.allocator(), 7) else try fixture.v41Files(arena.allocator(), 7);
        const p = try Served.open(&@import("../deepseek_v41/plugin.zig").table, files, "");
        defer p.close();
        const h = try Requests.open(p);
        defer h.close();
        try h.attachDisk(p);
        const prompt = diskPrompt();
        const id = try h.seedDisk(&prompt);
        const warm = try h.prefill(&prompt);
        defer warm.deinit();
        try testing.expectEqual(@as(u32, 512), warm.cached_tokens);
        try testing.expectEqual(@as(usize, 513), warm.cache.step);
        try testing.expect(h.entry.prefix_cache.?.disk.?.entryWholeOnDisk(id));
        var g = ops.Graph.init(testing.allocator, mlx.gpuStream());
        defer g.deinit();
        const probe = try g.fromSlice(i32, &.{44}, &.{ 1, 1 });
        const logits = try g.adopt(try p.xfm.forwardWith(&warm.legacy_gen.?.ctx, probe));
        try ops.eval(logits);
        const token = warm.legacy_gen.?.next_token_id;
        finish_request.releasePluginSlot(warm);
        const cold = try h.newSlot(&prompt);
        defer cold.deinit();
        cold.skip_prefix_cache = true;
        try prefill_request.runPrefill(&h.sch, cold);
        try testing.expectEqual(@as(u32, 0), cold.cached_tokens);
        try testing.expectEqual(token, cold.legacy_gen.?.next_token_id);
        const reference = try g.adopt(try p.xfm.forwardWith(&cold.legacy_gen.?.ctx, probe));
        try testing.expect(try sameValues(&g, logits, reference));
        finish_request.releasePluginSlot(cold);
    }
}

test "deepseek_v41 served: entries zero and RAM disabled bypass a valid disk history" {
    if (!mlx.metalKernelsAvailable() or mlx.noGpuBackend() or mlx.virtualGpu()) return error.SkipZigTest;
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const p = try Served.open(&@import("../deepseek_v41/plugin.zig").table, try fixture.v41Files(arena.allocator(), 7), "");
    defer p.close();
    const h = try Requests.open(p);
    defer h.close();
    try h.attachDisk(p);
    const prompt = diskPrompt();
    const id = try h.seedDisk(&prompt);
    const hc = h.entry.prefix_cache.?;
    for ([_]bool{ false, true }) |ram| {
        hc.ram_enabled = ram;
        hc.max_entries = if (ram) 0 else 4;
        const s = try h.prefill(&prompt);
        defer s.deinit();
        try testing.expectEqual(@as(u32, 0), s.cached_tokens);
        commit_request.commitSlotIfApplicable(&h.sch, s);
        finish_request.releasePluginSlot(s);
        try testing.expectEqual(@as(usize, 0), hc.entries.items.len);
        try testing.expect(hc.disk.?.entryWholeOnDisk(id));
    }
}

test "deepseek_v41 served: peak refusal precedes tensor loading and preserves the disk record" {
    if (!mlx.metalKernelsAvailable() or mlx.noGpuBackend() or mlx.virtualGpu()) return error.SkipZigTest;
    const old_fits = scheduler.prefill_admission_fits;
    defer scheduler.prefill_admission_fits = old_fits;
    scheduler.prefill_admission_fits = null;
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const p = try Served.open(&@import("../deepseek_v41/plugin.zig").table, try fixture.v41Files(arena.allocator(), 7), "");
    defer p.close();
    const h = try Requests.open(p);
    defer h.close();
    try h.attachDisk(p);
    const prompt = diskPrompt();
    const id = try h.seedDisk(&prompt);
    const part = try std.fmt.allocPrint(testing.allocator, "{s}/e{d}/m000.safetensors", .{ h.entry.prefix_cache.?.disk.?.root, id });
    defer testing.allocator.free(part);
    const original = try std.Io.Dir.cwd().readFileAlloc(h.sch.io, part, testing.allocator, .limited(64 << 20));
    defer testing.allocator.free(original);
    try std.Io.Dir.cwd().writeFile(h.sch.io, .{ .sub_path = part, .data = "temporarily unavailable tensor file" });
    const limits = @import("../../host/registry/memory_ceiling.zig").overrides();
    const old = limits.*;
    defer limits.* = old;
    limits.* = .{ .static_ceiling = 1, .wired_limit_mb = 0 };
    const s = try h.prefill(&prompt);
    defer s.deinit();
    try testing.expectEqual(@as(u32, 0), s.cached_tokens);
    finish_request.releasePluginSlot(s);
    try std.Io.Dir.cwd().writeFile(h.sch.io, .{ .sub_path = part, .data = original });
    try testing.expect(h.entry.prefix_cache.?.disk.?.entryWholeOnDisk(id));
    limits.* = old;
    const warm = try h.prefill(&prompt);
    defer warm.deinit();
    try testing.expectEqual(@as(u32, 512), warm.cached_tokens);
    finish_request.releasePluginSlot(warm);
}

test "deepseek_v41 served: idle staging pressure cancels its source before materialization" {
    if (!mlx.metalKernelsAvailable() or mlx.noGpuBackend() or mlx.virtualGpu()) return error.SkipZigTest;
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const p = try Served.open(&@import("../deepseek_v41/plugin.zig").table, try fixture.v41Files(arena.allocator(), 7), "");
    defer p.close();
    const h = try Requests.open(p);
    defer h.close();
    try h.attachDisk(p);
    const prompt = diskPrompt();
    const s = try h.prefill(&prompt);
    defer s.deinit();
    commit_request.commitSlotIfApplicable(&h.sch, s);
    finish_request.releasePluginSlot(s);
    const hc = h.entry.prefix_cache.?;
    try testing.expect(hc.nextModuleStagingBytes() > 0);
    const limits = @import("../../host/registry/memory_ceiling.zig").overrides();
    const old = limits.*;
    defer limits.* = old;
    limits.* = .{ .static_ceiling = 1, .wired_limit_mb = 0 };
    try testing.expectEqual(.done, @import("../../engines/mlx/cache/budget.zig").flushOwedDiskSlices(&h.sch));
    try testing.expectEqual(@as(u64, 0), hc.disk.?.bytes_persisted);
    try testing.expectEqual(@as(u64, 0), hc.residentBytes());
    try testing.expectEqual(@as(u64, 1), hc.module_export_losses_total);
}

test "deepseek_v41 served: unsupported disk hooks preserve RAM reuse" {
    if (!mlx.metalKernelsAvailable() or mlx.noGpuBackend() or mlx.virtualGpu()) return error.SkipZigTest;
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const p = try Served.open(&@import("../deepseek_v41/plugin.zig").table, try fixture.v41Files(arena.allocator(), 7), "");
    defer p.close();
    const h = try Requests.open(p);
    defer h.close();
    try h.attachDisk(p);
    const Refuse = struct {
        var calls: usize = 0;
        fn plan(_: *anyopaque, _: abi.Str, out: *u64) callconv(.c) abi.Verdict {
            calls += 1;
            out.* = 0;
            return abi.Verdict.refuse(.unsupported, "fixture has RAM snapshots only");
        }
    };
    const bound = p.xfm.plugin.?;
    const old = bound.table;
    defer bound.table = old;
    var hooks = old.cache.aux.?.*;
    hooks.import_plan = &Refuse.plan;
    var cache_hooks = old.cache.*;
    cache_hooks.aux = &hooks;
    var table = old.*;
    table.cache = &cache_hooks;
    bound.table = &table;
    Refuse.calls = 0;
    const prompt = diskPrompt();
    for (0..2) |i| {
        const s = try h.prefill(&prompt);
        defer s.deinit();
        try testing.expectEqual(@as(u32, if (i == 0) 0 else 512), s.cached_tokens);
        commit_request.commitSlotIfApplicable(&h.sch, s);
        finish_request.releasePluginSlot(s);
    }
    try testing.expect(Refuse.calls > 0);
    try testing.expectEqual(@as(usize, 0), h.entry.prefix_cache.?.disk.?.entries.items.len);
    try testing.expectEqual(@as(u64, 0), h.entry.prefix_cache.?.disk.?.bytes_persisted);
}

fn expectDiskMedia(appended: bool) !void {
    if (!mlx.metalKernelsAvailable() or mlx.noGpuBackend()) return error.SkipZigTest;
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const files = try fixture.v41Files(arena.allocator(), 7);
    const json = try std.mem.replaceOwned(u8, arena.allocator(), files[0].bytes, "\"eos_token_id\": 1,", "\"eos_token_id\": 1, \"image_token_id\": 5,");
    const p = try Served.open(&@import("../deepseek_v41/plugin.zig").table, &.{ .{ .name = "config.json", .bytes = json }, files[1] }, "");
    defer p.close();
    const h = try Requests.open(p);
    defer h.close();
    try h.attachDisk(p);
    var prompt = @as([515]u32, @splat(3));
    prompt[64] = 5;
    prompt[513] = 5;
    const spans = [_]prefix_cache.MediaSpan{ .{ .start = 64, .key = 42 }, .{ .start = 513, .key = 99 } };
    var g = ops.Graph.init(testing.allocator, mlx.gpuStream());
    defer g.deinit();
    const original = try g.astype(try g.fromSlice(f32, &@as([128]f32, @splat(0.25)), &.{ 1, 1, 128 }), .bfloat16);
    const seed = try h.mediaSlot(prompt[0..513], original, spans[0..1]);
    defer seed.deinit();
    try prefill_request.runPrefill(&h.sch, seed);
    const id = try h.commitDisk(seed);
    var values = @as([256]f32, @splat(0.75));
    if (appended) @memset(values[0..128], 0.25);
    const changed = try g.astype(try g.fromSlice(f32, &values, &.{ 1, 2, 128 }), .bfloat16);
    const next = try h.mediaSlot(&prompt, changed, &spans);
    defer next.deinit();
    try prefill_request.runPrefill(&h.sch, next);
    try testing.expectEqual(@as(u32, if (appended) 512 else 0), next.cached_tokens);
    finish_request.releasePluginSlot(next);
    try testing.expect(h.entry.prefix_cache.?.disk.?.entryWholeOnDisk(id));
    const again = try h.mediaSlot(prompt[0..513], original, spans[0..1]);
    defer again.deinit();
    try prefill_request.runPrefill(&h.sch, again);
    try testing.expectEqual(@as(u32, 512), again.cached_tokens);
    finish_request.releasePluginSlot(again);
}

test "deepseek_v41 served: changed covered media rows miss without poisoning disk history" {
    if (mlx.virtualGpu()) return error.SkipZigTest;
    try expectDiskMedia(false);
}

test "deepseek_v41 served: appended media leaves an exact disk checkpoint eligible" {
    if (mlx.virtualGpu()) return error.SkipZigTest;
    try expectDiskMedia(true);
}

test "deepseek_v41 served: disk imports use their owning model with a foreign cache view" {
    if (!mlx.metalKernelsAvailable() or mlx.noGpuBackend() or mlx.virtualGpu()) return error.SkipZigTest;
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const files = try fixture.v41Files(arena.allocator(), 7);
    const p = try Served.open(&@import("../deepseek_v41/plugin.zig").table, files, "");
    defer p.close();
    const q = try Served.open(&@import("../deepseek_v41/plugin.zig").table, files, "");
    defer q.close();
    const a = try Requests.open(p);
    defer a.close();
    const b = try Requests.open(q);
    defer b.close();
    try a.attachDisk(p);
    try b.attachDisk(q);
    const prompt = diskPrompt();
    _ = try a.seedDisk(&prompt);
    _ = try b.seedDisk(&prompt);
    a.sch.hot_prefix_cache = b.entry.prefix_cache;
    b.sch.hot_prefix_cache = a.entry.prefix_cache;
    for ([_]*Requests{ a, b }) |h| {
        const s = try h.prefill(&prompt);
        defer s.deinit();
        try testing.expectEqual(@as(u32, 512), s.cached_tokens);
        finish_request.releasePluginSlot(s);
        try testing.expectEqual(@as(u64, 0), driver.stateBytes(h.entry.transformer.?));
    }
}

test "deepseek_v41 served: failed imported restore resets the prompt and retires only its record" {
    if (mlx.virtualGpu()) return error.SkipZigTest;
    try expectImportFallback(.restore);
}

test "deepseek_v41 served: failed restored prompt setup retries cold with zero cached tokens" {
    if (mlx.virtualGpu()) return error.SkipZigTest;
    try expectImportFallback(.prompt);
}

test "deepseek_v41 served: failed tensor import releases its inputs and prefills cold" {
    if (mlx.virtualGpu()) return error.SkipZigTest;
    try expectImportFallback(.arrays);
}

fn expectImportFallback(comptime failure: enum { arrays, restore, prompt }) !void {
    if (!mlx.metalKernelsAvailable() or mlx.noGpuBackend() or mlx.virtualGpu()) return error.SkipZigTest;
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const p = try Served.open(&@import("../deepseek_v41/plugin.zig").table, try fixture.v41Files(arena.allocator(), 7), "");
    defer p.close();
    const h = try Requests.open(p);
    defer h.close();
    try h.attachDisk(p);
    const prompt = diskPrompt();
    const id = try h.seedDisk(&prompt);
    const Refuse = struct {
        var calls: usize = 0;
        var original: *const abi.AuxHooks = undefined;
        fn arrays(_: *anyopaque, _: abi.Span(u32), _: abi.Str, _: abi.Span(abi.AuxTensor), out: *abi.Handle) callconv(.c) abi.Verdict {
            calls += 1;
            out.* = null;
            return abi.Verdict.refuse(.failed, "injected import failure");
        }
        fn restore(_: *anyopaque, _: u32, _: abi.Handle, _: u64) callconv(.c) abi.Verdict {
            calls += 1;
            return abi.Verdict.refuse(.failed, "injected restore failure");
        }
        fn beginPrompt(mp: *anyopaque, slot: u32, ids: abi.Span(u32), rows: abi.Array, n: u64, capture: bool) callconv(.c) abi.Verdict {
            if (n == 0) return original.begin_prompt(mp, slot, ids, rows, n, capture);
            calls += 1;
            return abi.Verdict.refuse(.failed, "injected prompt setup failure");
        }
    };
    const bound = p.xfm.plugin.?;
    const old = bound.table;
    defer bound.table = old;
    var hooks = old.cache.aux.?.*;
    switch (failure) {
        .arrays => hooks.import_arrays = &Refuse.arrays,
        .restore => hooks.restore_at = &Refuse.restore,
        .prompt => hooks.begin_prompt = &Refuse.beginPrompt,
    }
    Refuse.original = old.cache.aux.?;
    var cache_hooks = old.cache.*;
    cache_hooks.aux = &hooks;
    var table = old.*;
    table.cache = &cache_hooks;
    bound.table = &table;
    Refuse.calls = 0;
    const s = try h.prefill(&prompt);
    defer s.deinit();
    try testing.expectEqual(@as(usize, 1), Refuse.calls);
    try testing.expectEqual(@as(u32, 0), s.cached_tokens);
    try testing.expectEqual(@as(usize, 513), s.cache.step);
    try testing.expect(h.entry.prefix_cache.?.disk.?.entryPoisoned(id));
    finish_request.releasePluginSlot(s);
    try testing.expectEqual(@as(u64, 0), driver.stateBytes(&p.xfm));
}

test "deepseek_v41 served: cancel, pad-only, failed forward and mismatched snapshot IDs commit nothing" {
    if (!mlx.metalKernelsAvailable() or mlx.noGpuBackend() or mlx.virtualGpu()) return error.SkipZigTest;
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const p = try Served.open(&@import("../deepseek_v41/plugin.zig").table, try fixture.v41Files(arena.allocator(), 7), "");
    defer p.close();
    const h = try Requests.open(p);
    defer h.close();
    for (0..5) |mode| {
        const s = try h.prefill(&.{ 3, 17, 42, 9, 7, 25, 1, 60 });
        defer s.deinit();
        switch (mode) {
            0 => s.cancelled.store(true, .release),
            1 => mlx.latchErrorForTest("failed module forward"),
            2 => s.error_code = try testing.allocator.dupe(u8, "PluginForwardFailed"),
            3 => try s.legacy_gen.?.generated_ids.append(testing.allocator, 0),
            4 => intree.dsv41(&p.xfm).?.dec_state.?.snap.?.ids[0] = 99,
            else => unreachable,
        }
        defer mlx.dropLatchedErrorUnless(false);
        commit_request.commitSlotIfApplicable(&h.sch, s);
        try testing.expectEqual(@as(usize, 0), h.entry.prefix_cache.?.entries.items.len);
        finish_request.releasePluginSlot(s);
        finish_request.releasePluginSlot(s);
        try testing.expectEqual(@as(u64, 0), driver.stateBytes(&p.xfm));
    }
    const abandoned = try h.newSlot(&.{ 3, 17, 42, 9, 7, 25, 1, 60 });
    const id = @import("../../engines/mlx/plugin/settle.zig").slotId(p.xfm.plugin.?, &abandoned.ctx);
    try testing.expect(driver.auxBeginPrompt(&p.xfm, id, abandoned.full_prompt, null, 0, true).isOk());
    var g = ops.Graph.init(testing.allocator, mlx.gpuStream());
    defer g.deinit();
    try ops.eval(try g.adopt(try p.xfm.forwardWith(&abandoned.ctx, try g.fromSlice(i32, &.{ 3, 17, 42, 9, 7, 25, 1 }, &.{ 1, 7 }))));
    abandoned.cancelled.store(true, .release);
    try h.sch.cleanup_queue.append(testing.allocator, abandoned);
    try testing.expect(@import("../../engines/mlx/runtime/loop.zig").drainSideQueues(&h.sch));
    try testing.expectEqual(@as(usize, 0), h.entry.prefix_cache.?.entries.items.len);
    try testing.expectEqual(@as(u64, 0), driver.stateBytes(&p.xfm));
}

test "deepseek_v41 served: restored state is discarded once when only cold prefill fits" {
    if (!mlx.metalKernelsAvailable() or mlx.noGpuBackend() or mlx.virtualGpu()) return error.SkipZigTest;
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const p = try Served.open(&@import("../deepseek_v41/plugin.zig").table, try fixture.v41Files(arena.allocator(), 7), "");
    defer p.close();
    const h = try Requests.open(p);
    defer h.close();
    const prompt = [_]u32{ 3, 17, 42, 9, 7, 25, 1, 60 };
    const first = try h.prefill(&prompt);
    defer first.deinit();
    commit_request.commitSlotIfApplicable(&h.sch, first);
    finish_request.releasePluginSlot(first);
    const idle = try h.prefill(&.{ 5, 8, 13, 21, 34, 55, 89, 4 });
    defer idle.deinit();
    commit_request.commitSlotIfApplicable(&h.sch, idle);
    finish_request.releasePluginSlot(idle);
    const Guard = struct {
        var calls: usize = 0;
        var refuse_cold = false;
        var mdl: *@import("../deepseek_v41/deepseek_v41.zig").Dsv41Model = undefined;
        var cache: *prefix_cache.HotPrefixCache = undefined;
        fn fits(_: *const model.ModelConfig, _: usize, _: u32, _: KVQuantConfig, n: u64, _: u64, _: bool, _: bool) bool {
            calls += 1;
            return n == 0 and mdl.dec_state.?.n == 0 and cache.entries.items.len == 0 and !refuse_cold;
        }
    };
    const old = scheduler.prefill_admission_fits;
    defer scheduler.prefill_admission_fits = old;
    scheduler.prefill_admission_fits = &Guard.fits;
    Guard.calls = 0;
    Guard.refuse_cold = false;
    Guard.mdl = intree.dsv41(&p.xfm).?;
    Guard.cache = h.entry.prefix_cache.?;
    const next = try h.prefill(&prompt);
    defer next.deinit();
    try testing.expectEqual(@as(u32, 0), next.cached_tokens);
    try testing.expectEqual(@as(usize, prompt.len), next.cache.step);
    try testing.expect(Guard.calls <= 6);
    try testing.expectEqual(@as(usize, 0), h.entry.prefix_cache.?.entries.items.len);
    commit_request.commitSlotIfApplicable(&h.sch, next);
    finish_request.releasePluginSlot(next);
    Guard.calls = 0;
    Guard.refuse_cold = true;
    const refused = try h.newSlot(&prompt);
    defer refused.deinit();
    try testing.expectError(error.PrefillDoesNotFit, prefill_request.runPrefill(&h.sch, refused));
    try testing.expect(Guard.calls <= 6);
    try testing.expectEqual(@as(usize, 0), refused.cache.step);
    finish_request.releasePluginSlot(refused);
}

test "deepseek_v41 served: two models retain separate histories even with a foreign current-cache view" {
    if (!mlx.metalKernelsAvailable() or mlx.noGpuBackend() or mlx.virtualGpu()) return error.SkipZigTest;
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const files = try fixture.v41Files(arena.allocator(), 7);
    const p = try Served.open(&@import("../deepseek_v41/plugin.zig").table, files, "");
    defer p.close();
    const q = try Served.open(&@import("../deepseek_v41/plugin.zig").table, files, "");
    defer q.close();
    const a = try Requests.open(p);
    defer a.close();
    const b = try Requests.open(q);
    defer b.close();
    a.sch.hot_prefix_cache = b.entry.prefix_cache;
    const prompt = [_]u32{ 3, 17, 42, 9, 7, 25, 1, 60 };
    for ([_]*Requests{ a, b, a, b }, 0..) |h, i| {
        const s = try h.prefill(&prompt);
        defer s.deinit();
        try testing.expectEqual(@as(u32, if (i < 2) 0 else 7), s.cached_tokens);
        commit_request.commitSlotIfApplicable(&h.sch, s);
        finish_request.releasePluginSlot(s);
    }
    try testing.expectEqual(@as(usize, 1), a.entry.prefix_cache.?.entries.items.len);
    try testing.expectEqual(@as(usize, 1), b.entry.prefix_cache.?.entries.items.len);
}

test "deepseek_v41 served: ordinary unload frees retained module entries before their plugin binding" {
    if (!mlx.metalKernelsAvailable() or mlx.noGpuBackend() or mlx.virtualGpu()) return error.SkipZigTest;
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const p = try Served.open(&@import("../deepseek_v41/plugin.zig").table, try fixture.v41Files(arena.allocator(), 7), "");
    defer {
        p.slot.kv.deinit();
        p.cfg.deinit(testing.allocator);
        p.tmp.cleanup();
        testing.allocator.destroy(p);
    }
    const h = try Requests.open(p);
    defer h.close();
    const xfm = try testing.allocator.create(Transformer);
    xfm.* = p.xfm;
    h.entry.transformer = xfm;
    const s = try h.prefill(&.{ 3, 17, 42, 9, 7, 25, 1, 60 });
    commit_request.commitSlotIfApplicable(&h.sch, s);
    finish_request.releasePluginSlot(s);
    s.deinit();
    try testing.expect(h.entry.prefix_cache.?.residentBytes() > 0);
    h.sch.hot_prefix_cache = null;
    @import("../../engines/mlx/runtime/model_state.zig").freeMlx(h.entry);
    try testing.expect(h.entry.prefix_cache == null);
    try testing.expect(h.entry.transformer == null);
}

test "deepseek_v41 served: an excluded instance keeps legacy resume through request prefill" {
    if (!mlx.metalKernelsAvailable() or mlx.noGpuBackend()) return error.SkipZigTest;
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const P = @import("../deepseek_v41/plugin.zig");
    const p = try Served.open(&P.table, try fixture.v41Files(arena.allocator(), 7), "");
    defer p.close();
    const m: *P.Model = @ptrCast(@alignCast(p.xfm.plugin.?.model));
    m.cfg.dsv41_streamed = true;
    try testing.expect(!prefix_cache.HotPrefixCache.shouldUseInstance(&p.cfg, true, &p.xfm));
    const h = try Requests.open(p);
    defer h.close();
    h.entry.prefix_cache.?.deinit();
    testing.allocator.destroy(h.entry.prefix_cache.?);
    h.entry.prefix_cache = null;
    h.sch.hot_prefix_cache = null;
    const prompt = [_]u32{ 3, 17, 42, 9, 7, 25, 1, 60 };
    for (0..2) |i| {
        const s = try h.prefill(&prompt);
        defer s.deinit();
        try testing.expectEqual(@as(u32, if (i == 0) 0 else 7), s.cached_tokens);
        finish_request.releasePluginSlot(s);
    }
}

test "deepseek_v41 served: image rows from Forward.media splice as the legacy seam splices them, across a chunk boundary" {
    if (!mlx.metalKernelsAvailable()) return error.SkipZigTest;
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const aa = arena.allocator();
    const files = try fixture.v41Files(aa, 7);
    const json = try std.mem.replaceOwned(u8, aa, files[0].bytes, "\"eos_token_id\": 1,", "\"eos_token_id\": 1, \"image_token_id\": 5,");
    try testing.expect(!std.mem.eql(u8, json, files[0].bytes));
    const p = try Served.open(&@import("../deepseek_v41/plugin.zig").table, &.{ .{ .name = "config.json", .bytes = json }, files[1] }, @embedFile("../deepseek_v41/fixtures/image.tape"));
    defer p.close();
    var g = ops.Graph.init(testing.allocator, mlx.gpuStream());
    defer g.deinit();
    var prng = std.Random.DefaultPrng.init(0xd541);
    var buf: [5 * 128]f32 = undefined;
    for (&buf) |*x| x.* = prng.random().floatNorm(f32) * 0.5;
    const rows = try g.astype(try g.adopt(mlx.mlx_array_new_data(&buf, &[_]c_int{ 1, 5, 128 }, 3, .float32)), .bfloat16);
    const chunks = [_]struct { ids: []const i32, offset: usize }{
        .{ .ids = &.{ 3, 17, 5, 5, 5 }, .offset = 0 },
        .{ .ids = &.{ 5, 5, 7, 8 }, .offset = 3 },
        .{ .ids = &.{9}, .offset = 5 },
    };
    // Without rows the image token is text: the same first chunk, to show the rows move the logits.
    const text = try p.step(&g, chunks[0].ids);
    p.restart();
    p.slot.ctx.vision_embeddings = rows;
    for (chunks, 0..) |ch, i| {
        p.slot.ctx.vision_splice_offset = ch.offset;
        const got = try p.step(&g, ch.ids);
        if (i == 0) try testing.expect(!try sameValues(&g, got, text));
    }
    try p.tape.expectDone();
}

fn sameValues(g: *ops.Graph, x: Array, y: Array) !bool {
    const fx = try g.contiguous(try g.astype(x, .float32));
    const fy = try g.contiguous(try g.astype(y, .float32));
    try ops.eval(fx);
    try ops.eval(fy);
    const n = mlx.mlx_array_size(fx);
    try testing.expectEqual(n, mlx.mlx_array_size(fy));
    const xs = mlx.mlx_array_data_float32(fx).?[0..n];
    const ys = mlx.mlx_array_data_float32(fy).?[0..n];
    for (xs, ys) |a, b| {
        try testing.expect(std.math.isFinite(a) and std.math.isFinite(b));
    }
    return std.mem.eql(f32, xs, ys);
}

/// DSpark rounds through the generator (the fixture's peaked head makes drafts agree: rounds accept none, some and
/// all): each round's tokens, accepted drafts, next token and position equal the legacy shell's (the tape); at least
/// `min_accepted` drafts accepted over the run.
fn expectDspark(comptime P: type, files: []const sdk.fixture.File, sampling: SamplingParams, min_accepted: u64, tape: []const u8) !void {
    if (mlx.virtualGpu()) return error.SkipZigTest;
    const a = testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();
    const ds = env.override(.MLX_SERVE_DSV4_DSPARK, "1");
    defer ds.restore();
    const p = try Served.open(&P.table, files, tape);
    defer p.close();
    var g = ops.Graph.init(testing.allocator, mlx.gpuStream());
    defer g.deinit();
    try testing.expectEqual(@as(usize, 2), p.xfm.dsparkStages());
    var tok: Tokenizer = undefined;
    const prompt = [_]u32{ 3, 17, 42, 9, 7, 25, 1, 60, 11, 2, 30, 31 };
    var gen = try Generator.initWithOptions(io, a, &p.xfm, &tok, &prompt, 48, sampling, &.{}, .{
        .pld_enabled = true,
        .mtp_enabled = true,
        .skip_lazy_preforward = true,
        .lookup_prompt = &prompt,
    });
    defer gen.deinit(a);
    try testing.expect(gen.dspark_enabled);
    // Each round as `[tokens..., accepted drafts, next token, position]`.
    var rec: [16]u32 = undefined;
    for (0..16) |_| {
        const r = (try gen.nextDspark(a)) orelse return error.RoundsEnded;
        defer a.free(r.tokens);
        @memcpy(rec[0..r.tokens.len], r.tokens);
        rec[r.tokens.len..][0..3].* = .{ r.accepted_tokens, gen.next_token_id, @intCast(gen.ctx.cache.step) };
        try p.checkInts(&g, rec[0 .. r.tokens.len + 3], true);
    }
    // How many drafts the random pack accepts is a property of the recording GPU's numerics.
    if (p.tape.same_device) try testing.expect(gen.dspark_attempted >= 4 and gen.dspark_accepted_tokens >= min_accepted);
    try p.tape.expectDone();
}

test "deepseek_v4 served: DSpark rounds (greedy, then seeded sampling) equal the legacy shell's" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const files = try fixture.v4DsparkFiles(arena.allocator(), 11);
    try expectDspark(@import("plugin.zig"), files, .{ .temperature = 0 }, 4, @embedFile("fixtures/dspark-greedy.tape"));
    try expectDspark(@import("plugin.zig"), files, .{ .temperature = 0.8, .top_p = 0.95, .seed = 0xd5 }, 4, @embedFile("fixtures/dspark-sampled.tape"));
}

test "deepseek_v41 served: DSpark rounds (greedy, then seeded sampling) equal the legacy shell's" {
    if (!mlx.metalKernelsAvailable()) return error.SkipZigTest;
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const files = try fixture.v41DsparkFiles(arena.allocator(), 7);
    try expectDspark(@import("../deepseek_v41/plugin.zig"), files, .{ .temperature = 0 }, 4, @embedFile("../deepseek_v41/fixtures/dspark-greedy.tape"));
    try expectDspark(@import("../deepseek_v41/plugin.zig"), files, .{ .temperature = 0.8, .top_p = 0.95, .seed = 0xd5 }, 4, @embedFile("../deepseek_v41/fixtures/dspark-sampled.tape"));
}

test "deepseek_v4 served: mlx-community's withdrawn preview pack is refused by the legacy parse and, at load, by name" {
    const preview = @embedFile("fixtures/real/mlx-community_DeepSeek-V4-Flash-4bit.json");
    try testing.expectError(error.UnsupportedDsv4Config, model.parseConfigFromJson(testing.allocator, preview));
    const io = std.Io.Threaded.global_single_threaded.io();
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.writeFile(io, .{ .sub_path = "config.json", .data = preview });
    var buf: [512]u8 = undefined;
    const dir = buf[0..try tmp.dir.realPath(io, &buf)];
    try testing.expectError(error.PluginDeclinedNoLegacy, model.parseConfig(io, testing.allocator, dir));
}
