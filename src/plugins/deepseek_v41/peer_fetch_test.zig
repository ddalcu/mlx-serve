//! In-process tests for peer prefix-cache export, wire import, identity/stale/memory refusals, and HTTP gates.

const std = @import("std");
const abi = @import("../../abi/plugin.zig");
const mlx = @import("../../mlx/ffi.zig");
const sdk = @import("../../sdk/root.zig");
const model = @import("../../sdk/config/model.zig");
const driver = @import("../../engines/mlx/plugin/driver.zig");
const fixture = @import("../deepseek_v4/fixture.zig");
const cache_transfer = @import("../../host/api/cache_transfer.zig");
const cache_rdma = @import("../../host/api/cache_rdma.zig");
const prefix_cache = @import("../../engines/mlx/cache/prefix_cache.zig");
const peer_wire = @import("../../engines/mlx/cache/peer_wire.zig");
const peer_transfer = @import("../../engines/mlx/cache/peer_transfer.zig");
const pack_identity = @import("../../engines/mlx/cache/pack_identity.zig");
const env = @import("../../env.zig");
const prefill_request = @import("../../engines/mlx/prefill.zig");
const decode_request = @import("../../engines/mlx/decode.zig");
const commit_request = @import("../../engines/mlx/commit.zig");
const finish_request = @import("../../engines/mlx/finish.zig");
const scheduler = @import("../../engines/mlx/scheduler.zig");
const RequestSlot = @import("../../engines/mlx/slot.zig").Slot;
const KVQuantConfig = @import("../../engines/mlx/cache/kv_quant.zig").KVQuantConfig;
const Tokenizer = @import("../../sdk/tokenizer/tokenizer.zig").Tokenizer;
const registry = @import("../../host/registry/registry.zig");
const cpu_state = @import("../../host/registry/cpu_state.zig");
const memory_ceiling = @import("../../host/registry/memory_ceiling.zig");
const http_server = @import("../../host/http/server.zig");
const Conn = @import("../../host/http/conn.zig").Conn;
const router = @import("../../host/http/router.zig");
const HostScheduler = @import("../../host/gpu_thread/scheduler.zig").Scheduler;
const mlx_engine = @import("../../host/engines/mlx.zig");
const flash_attn = @import("prompt/flash_attn.zig");

const testing = std.testing;

const Greedy = struct { cached: u32, tokens: []u32 };

const PeerHarness = struct {
    tmp: testing.TmpDir,
    path_buf: [512]u8 = undefined,
    path_len: usize = 0,
    cfg: model.ModelConfig,
    xfm: @import("../../engines/mlx/transformer.zig").Transformer,
    sch: scheduler.Scheduler = undefined,
    entry: *registry.LoadedModel,
    tok: Tokenizer,

    fn open(files: []const sdk.fixture.File) !*PeerHarness {
        const a = testing.allocator;
        const io = std.Io.Threaded.global_single_threaded.io();
        const h = try a.create(PeerHarness);
        errdefer a.destroy(h);
        h.tmp = testing.tmpDir(.{ .iterate = true });
        errdefer h.tmp.cleanup();
        for (files) |f| try h.tmp.dir.writeFile(io, .{ .sub_path = f.name, .data = f.bytes });
        h.path_len = try h.tmp.dir.realPath(io, &h.path_buf);
        const dir = h.path_buf[0..h.path_len];

        h.cfg = try model.parseConfig(io, a, dir);
        errdefer h.cfg.deinit(a);
        h.xfm = try driver.load(io, a, &h.cfg, dir, mlx.gpuStream());
        errdefer h.xfm.deinit();

        const reg = try registry.ModelRegistry.init(a, io, null, 4, 0, null);
        errdefer reg.deinit();
        h.entry = try reg.registerStub("fixture", dir, 1);
        h.tok = Tokenizer.initEmptyForTests(a, .byte_level_bpe);
        errdefer h.tok.deinit();
        h.entry.config = &h.cfg;
        h.entry.transformer = &h.xfm;
        h.entry.tokenizer = &h.tok;
        h.entry.state = .ready;
        h.entry.prefix_cache = try a.create(prefix_cache.HotPrefixCache);
        h.entry.prefix_cache.?.* = prefix_cache.HotPrefixCache.init(a, 8);

        h.sch = .{
            .allocator = a,
            .io = io,
            .registry = reg,
            .current_model = h.entry,
            .xfm = null,
            .weights = null,
            .vision_encoder = null,
            .drafter = null,
            .drafter_block_size = 4,
            .warmup_eager = false,
            .warmup_full = false,
            .kv_quant_config = KVQuantConfig.dense,
            .gguf_ctx_size = 0,
            .prefix_cache_capacity = 8,
            .prefix_cache_ram_enabled = true,
            .prefix_cache_mem_bytes = 0,
            .prefix_cache_mem_resolver = null,
            .prefix_cache_disk_bytes = 0,
            .ssm_checkpoint_stride = 0,
            .ssm_checkpoint_max = 16,
            .mtp_enabled = false,
            .mtp_head_kv_quant = false,
            .mtp_depth = 0,
            .llama = .{},
            .ds4_mtp = false,
            .ds4_dspark = false,
            .ds4_ssd_streaming = false,
            .ane_prefill = false,
            .ane = false,
            .ane_slots = 1,
            .ane_chunk_resolver = null,
            .ane_headroom_resolver = null,
            .no_drafter = true,
            .drafter_dir = "",
            .primary_model_dir = dir,
            .draft_block_size = 4,
            .draft_block_size_explicit = false,
            .drafter_path = "",
            .hot_prefix_cache = h.entry.prefix_cache,
            .max_concurrent = 4,
            .force_batched = false,
            .queue_mu = .init,
            .queue_cond = .init,
            .pending = .empty,
            .decoding = .empty,
            .vision_queue = .empty,
            .embed_queue = .empty,
            .load_queue = .empty,
            .gen_queue = .empty,
            .unload_queue = .empty,
            .cleanup_queue = .empty,
            .metrics = null,
            .inflight_generated_tokens = .init(0),
            .inflight_prefill_tokens = .init(0),
            .inflight_prefill_expected = .init(0),
            .requests_prefilling = .init(0),
            .in_flight = 0,
            .queue_cap = 36,
            .submit_cond = .init,
            .session_cond = .init,
            .inference_thread = null,
            .shutdown = .init(false),
            .started = .init(true),
            .started_mu = .init,
            .started_cond = .init,
            .load_failed = .init(false),
            .load_error_name = null,
        };
        try h.rehash();
        return h;
    }

    fn rehash(h: *PeerHarness) !void {
        try pack_identity.start(h.sch.io, testing.allocator, &h.entry.pack_identity, h.entry.path);
        _ = try pack_identity.waitReady(h.sch.io, h.entry.pack_identity);
    }

    fn close(h: *PeerHarness) void {
        pack_identity.drop(&h.entry.pack_identity);
        h.entry.transformer = null;
        h.entry.config = null;
        h.entry.tokenizer = null;
        h.sch.registry.deinit();
        h.sch.pending.deinit(testing.allocator);
        h.sch.decoding.deinit(testing.allocator);
        h.sch.offloaded.deinit(testing.allocator);
        h.sch.vision_queue.deinit(testing.allocator);
        h.sch.embed_queue.deinit(testing.allocator);
        h.sch.load_queue.deinit(testing.allocator);
        h.sch.gen_queue.deinit(testing.allocator);
        h.sch.unload_queue.deinit(testing.allocator);
        h.sch.cleanup_queue.deinit(testing.allocator);
        h.sch.cache_export_queue.deinit(testing.allocator);
        h.sch.cache_import_queue.deinit(testing.allocator);
        testing.allocator.free(h.sch.hot_cache_digests);
        h.tok.deinit();
        h.xfm.deinit();
        _ = mlx.mlx_clear_cache();
        h.cfg.deinit(testing.allocator);
        h.tmp.cleanup();
        testing.allocator.destroy(h);
    }

    fn attachDisk(h: *PeerHarness) !void {
        const dir = h.path_buf[0..h.path_len];
        const hc = h.entry.prefix_cache.?;
        hc.disk = try @import("../../engines/mlx/load.zig").initPrefixDisk(
            testing.allocator,
            h.sch.io,
            dir,
            "peer-disk",
            0,
            512,
            prefix_cache.ModuleBinding.init(&h.xfm).?,
            dir,
        );
        hc.disk.?.armTestSpace(1 << 40, 2 << 40);
        hc.armDiskTier(&h.cfg);
        hc.module_quiet_ms = 0;
    }

    fn runGreedy(h: *PeerHarness, prompt: []const u32, max_tokens: u32, commit: bool) !Greedy {
        return h.runGreedyTagged(prompt, max_tokens, commit, 0);
    }

    fn runGreedyTagged(h: *PeerHarness, prompt: []const u32, max_tokens: u32, commit: bool, session: u64) !Greedy {
        const s = try RequestSlot.init(testing.allocator, h.sch.io, h.entry.config.?, .{
            .model = h.entry,
            .prompt_ids = prompt,
            .sampling = .{ .temperature = 0 },
            .eos_token_ids = &.{},
            .max_tokens = max_tokens,
            .session = session,
        }, h.sch.kv_quant_config);
        defer s.deinit();
        try prefill_request.runPrefill(&h.sch, s);
        while (s.completion_tokens < max_tokens and !s.finished) {
            try decode_request.runSingleDecodeTick(&h.sch, s);
        }
        const gen_ids = try testing.allocator.dupe(u32, s.legacy_gen.?.generated_ids.items);
        errdefer testing.allocator.free(gen_ids);
        const cached = s.cached_tokens;
        if (!s.finished) {
            if (commit) commit_request.commitSlotIfApplicable(&h.sch, s);
            finish_request.releasePluginSlot(s);
        }
        return .{ .cached = cached, .tokens = gen_ids };
    }
};

fn buildPrompt(comptime len: usize) [len]u32 {
    var out: [len]u32 = undefined;
    for (&out, 0..) |*id, i| id.* = @intCast(3 + (i * 7) % 80);
    return out;
}

fn expectPromptEntries(hc: *const prefix_cache.HotPrefixCache, prompts: []const []const u32) !void {
    var count: usize = 0;
    for (hc.entries.items) |e| {
        const m = e.module orelse continue;
        if (m.tail_from == 0) count += 1;
    }
    try testing.expectEqual(prompts.len, count);
    for (prompts) |ids| {
        var matches: usize = 0;
        for (hc.entries.items) |e| {
            const m = e.module orelse continue;
            if (m.tail_from == 0 and std.mem.eql(u32, e.tokens, ids)) matches += 1;
        }
        try testing.expectEqual(@as(usize, 1), matches);
    }
}

fn expectResidentIds(hc: *const prefix_cache.HotPrefixCache, ids: []const u64) !void {
    try testing.expectEqual(ids.len, hc.entries.items.len);
    for (ids, hc.entries.items) |id, e| try testing.expectEqual(id, e.id);
}

test "dsv41 peer fetch: v41ServableFiles builds a servable pack with byte tokenizer and chat template" {
    const io = std.Io.Threaded.global_single_threaded.io();
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    const files = try fixture.v41ServableFiles(arena.allocator(), 7);
    for (files) |f| try tmp.dir.writeFile(io, .{ .sub_path = f.name, .data = f.bytes });
    var buf: [512]u8 = undefined;
    const dir = buf[0..try tmp.dir.realPath(io, &buf)];

    var state = try cpu_state.preloadCpuState(testing.allocator, io, dir, 0, null);
    defer cpu_state.freeCpuState(testing.allocator, &state);
    try testing.expectEqual(@as(u32, 320), state.config.vocab_size);
    try testing.expect(state.chat_config.chat_template.len > 0);
    const ids = try state.tok.encode(testing.allocator, "<｜begin▁of▁sentence｜><｜User｜>hello<｜Assistant｜></think>");
    defer testing.allocator.free(ids);
    try testing.expect(ids.len >= 8);
    try testing.expectEqual(@as(u32, 256), ids[0]);
}

test "dsv41 peer fetch: export from model A and importWireModule into model B preserves cached length and greedy output" {
    if (!mlx.metalKernelsAvailable() or mlx.noGpuBackend() or mlx.virtualGpu()) return error.SkipZigTest;
    const old_fits = scheduler.prefill_admission_fits;
    defer scheduler.prefill_admission_fits = old_fits;
    scheduler.prefill_admission_fits = null;

    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const files = try fixture.v41Files(arena.allocator(), 7);
    const model_a = try PeerHarness.open(files);
    defer model_a.close();
    const model_b = try PeerHarness.open(files);
    defer model_b.close();

    const full_prompt = buildPrompt(33);
    const turn1 = try model_a.runGreedy(full_prompt[0..25], 2, true);
    defer testing.allocator.free(turn1.tokens);
    try testing.expectEqual(@as(u32, 0), turn1.cached);
    try expectPromptEntries(model_a.entry.prefix_cache.?, &.{full_prompt[0..24]});

    var expired_export: @import("../../host/gpu_thread/requests.zig").CacheExportRequest = .{
        .model = model_a.entry,
        .prompt_ids = &full_prompt,
        .has_tools = false,
        .cache_key = 42,
        .max_bytes = router.max_cache_export_bytes,
        .deadline_ms = 0,
        .allocator = testing.allocator,
    };
    peer_transfer.runCacheExportRequest(&model_a.sch, &expired_export);
    try testing.expectEqual(@as(@TypeOf(expired_export.outcome), .deadline_passed), expired_export.outcome);

    const exported = try peer_transfer.exportWireModule(
        testing.allocator,
        model_a.sch.io,
        &model_a.sch,
        model_a.entry,
        &full_prompt,
        false,
        null,
        0,
        42,
        router.max_cache_export_bytes,
    );
    try testing.expect(exported == .ok);
    const wire = exported.ok.bytes;
    defer testing.allocator.free(wire);
    try testing.expectEqual(@as(usize, 24), exported.ok.n);
    var future_export: @import("../../host/gpu_thread/requests.zig").CacheExportRequest = .{
        .model = model_a.entry,
        .prompt_ids = &full_prompt,
        .has_tools = false,
        .cache_key = 42,
        .max_bytes = router.max_cache_export_bytes,
        .deadline_ms = std.math.maxInt(u64),
        .allocator = testing.allocator,
    };
    peer_transfer.runCacheExportRequest(&model_a.sch, &future_export);
    try testing.expect(future_export.outcome == .ok);
    testing.allocator.free(future_export.outcome.ok.bytes);
    const capped = try peer_transfer.exportWireModule(
        testing.allocator,
        model_a.sch.io,
        &model_a.sch,
        model_a.entry,
        &full_prompt,
        false,
        null,
        0,
        42,
        wire.len - 1,
    );
    try testing.expectEqual(@as(@TypeOf(capped), .{ .too_large = @intCast(wire.len) }), capped);

    var expired_import: @import("../../host/gpu_thread/requests.zig").CacheImportRequest = .{
        .model = model_b.entry,
        .body = wire,
        .deadline_ms = 0,
        .allocator = testing.allocator,
    };
    peer_transfer.runCacheImportRequest(&model_b.sch, &expired_import);
    try testing.expectEqual(@as(@TypeOf(expired_import.outcome), .{ .refused = .deadline_passed }), expired_import.outcome);
    try testing.expectEqual(@as(usize, 0), model_b.entry.prefix_cache.?.entries.items.len);

    var future_import: @import("../../host/gpu_thread/requests.zig").CacheImportRequest = .{
        .model = model_b.entry,
        .body = wire,
        .deadline_ms = std.math.maxInt(u64),
        .allocator = testing.allocator,
    };
    peer_transfer.runCacheImportRequest(&model_b.sch, &future_import);
    const imported = future_import.outcome;
    try testing.expect(imported == .ok);
    try testing.expectEqual(@as(usize, 24), imported.ok.n);
    try testing.expectEqual(@as(usize, 1), model_b.entry.prefix_cache.?.entries.items.len);

    const warm_a = try model_a.runGreedy(&full_prompt, 6, false);
    defer testing.allocator.free(warm_a.tokens);
    const warm_b = try model_b.runGreedy(&full_prompt, 6, false);
    defer testing.allocator.free(warm_b.tokens);

    try testing.expectEqual(@as(u32, 24), warm_a.cached);
    try testing.expectEqual(warm_a.cached, warm_b.cached);
    try testing.expectEqualSlices(u32, warm_a.tokens, warm_b.tokens);

    // Disk-only record on holder is not exported (RAM-only match)
    {
        const disk_only = try PeerHarness.open(files);
        defer disk_only.close();
        try disk_only.attachDisk();
        var disk_prompt: [513]u32 = undefined;
        for (&disk_prompt, 0..) |*id, i| id.* = @intCast(3 + i % 80);
        const seeded = try disk_only.runGreedy(&disk_prompt, 2, true);
        defer testing.allocator.free(seeded.tokens);
        const hc = disk_only.entry.prefix_cache.?;
        const disk_id = hc.entries.items[0].module.?.disk_id.?;
        for (0..32) |_| {
            const r = hc.flushOwedSlice(disk_only.xfm.s);
            hc.disk.?.drainWriter();
            hc.disk.?.pollModuleWrites();
            if (r == .done) break;
        }
        try testing.expect(hc.disk.?.entryWholeOnDisk(disk_id));
        hc.setBudget(1);
        hc.setBudget(0);
        try testing.expectEqual(@as(usize, 0), hc.entries.items.len);
        const disk_match = hc.findModuleMatch(prefix_cache.ModuleBinding.init(&disk_only.xfm).?, &disk_prompt, false, &.{}, null, KVQuantConfig.dense);
        try testing.expect(disk_match != null and disk_match.?.source == .disk);
        const disk_exp = try peer_transfer.exportWireModule(
            testing.allocator,
            disk_only.sch.io,
            &disk_only.sch,
            disk_only.entry,
            &disk_prompt,
            false,
            null,
            0,
            0,
            router.max_cache_export_bytes,
        );
        try testing.expectEqual(@as(@TypeOf(disk_exp), .not_found), disk_exp);
    }
}

fn exportBySession(h: *PeerHarness, session: u64) @import("../../host/gpu_thread/requests.zig").CacheExportOutcome {
    return exportBySessionCapped(h, session, router.max_cache_export_bytes);
}

fn exportBySessionCapped(h: *PeerHarness, session: u64, max_bytes: usize) @import("../../host/gpu_thread/requests.zig").CacheExportOutcome {
    var req: @import("../../host/gpu_thread/requests.zig").CacheExportRequest = .{
        .model = h.entry,
        .prompt_ids = &.{},
        .has_tools = false,
        .session = session,
        .max_bytes = max_bytes,
        .allocator = testing.allocator,
    };
    peer_transfer.runCacheExportRequest(&h.sch, &req);
    return req.outcome;
}

fn importWire(h: *PeerHarness, wire: []const u8) @import("../../host/gpu_thread/requests.zig").CacheImportOutcome {
    var req: @import("../../host/gpu_thread/requests.zig").CacheImportRequest = .{
        .model = h.entry,
        .body = wire,
        .allocator = testing.allocator,
    };
    peer_transfer.runCacheImportRequest(&h.sch, &req);
    return req.outcome;
}

test "peer transfer session: an export by X-Session-Key imports on another box, and a next prompt that drops the reply restores the boundary" {
    if (!mlx.metalKernelsAvailable() or mlx.noGpuBackend() or mlx.virtualGpu()) return error.SkipZigTest;
    const old_fits = scheduler.prefill_admission_fits;
    defer scheduler.prefill_admission_fits = old_fits;
    scheduler.prefill_admission_fits = null;

    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const files = try fixture.v41Files(arena.allocator(), 7);
    const model_a = try PeerHarness.open(files);
    defer model_a.close();
    const model_b = try PeerHarness.open(files);
    defer model_b.close();

    const full_prompt = buildPrompt(33);
    const turn1 = try model_a.runGreedyTagged(full_prompt[0..25], 2, true, 77);
    defer testing.allocator.free(turn1.tokens);

    try testing.expectEqual(@as(@TypeOf(exportBySession(model_a, 78)), .not_found), exportBySession(model_a, 78));
    const exported = exportBySession(model_a, 77);
    try testing.expect(exported == .ok);
    defer testing.allocator.free(exported.ok.bytes);

    const imported = importWire(model_b, exported.ok.bytes);
    try testing.expect(imported == .ok);
    // The receiver keeps the 24-token boundary as the session's tip, as the sender does.
    try testing.expectEqual(@as(usize, 24), model_b.entry.prefix_cache.?.sessionTip(77).?.n);

    // The next prompt continues the user text, not the reply: a short shared prefix cannot be cut, so the boundary
    // must restore whole on the receiver, exactly as it does on the sender.
    const warm_a = try model_a.runGreedy(&full_prompt, 6, false);
    defer testing.allocator.free(warm_a.tokens);
    const warm_b = try model_b.runGreedy(&full_prompt, 6, false);
    defer testing.allocator.free(warm_b.tokens);
    try testing.expectEqual(@as(u32, 24), warm_b.cached);
    try testing.expectEqual(warm_a.cached, warm_b.cached);
    try testing.expectEqualSlices(u32, warm_a.tokens, warm_b.tokens);
}

test "peer transfer session: an export carries the decode-end checkpoint, so the next turn keeps the reply on another box" {
    if (!mlx.metalKernelsAvailable() or mlx.noGpuBackend() or mlx.virtualGpu()) return error.SkipZigTest;
    const old_fits = scheduler.prefill_admission_fits;
    defer scheduler.prefill_admission_fits = old_fits;
    scheduler.prefill_admission_fits = null;

    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const files = try fixture.v41Files(arena.allocator(), 7);
    const model_a = try PeerHarness.open(files);
    defer model_a.close();
    const model_b = try PeerHarness.open(files);
    defer model_b.close();

    const base = buildPrompt(48);
    const turn1 = try model_a.runGreedyTagged(base[0..25], 4, true, 77);
    defer testing.allocator.free(turn1.tokens);
    // The next turn re-sends the prompt and the reply, then new user text, as an agent session does.
    var next: [25 + 4 + 8]u32 = undefined;
    @memcpy(next[0..25], base[0..25]);
    @memcpy(next[25..29], turn1.tokens[0..4]);
    @memcpy(next[29..], base[40..48]);

    const exported = exportBySession(model_a, 77);
    try testing.expect(exported == .ok);
    defer testing.allocator.free(exported.ok.bytes);
    try testing.expect(importWire(model_b, exported.ok.bytes) == .ok);
    try testing.expectEqual(@as(u64, 0), model_b.sch.hot_cache_checked_out_bytes.load(.monotonic));
    // A cap the decode-end entry does not fit sends the prompt boundary instead of refusing the export.
    const capped = exportBySessionCapped(model_a, 77, exported.ok.bytes.len - 1);
    try testing.expect(capped == .ok);
    defer testing.allocator.free(capped.ok.bytes);
    try testing.expectEqual(@as(usize, 24), capped.ok.n);
    // Both members and the copy must fit at once. Only the returned body stays reserved.
    for ([_]struct { room: usize, n: usize }{
        .{ .room = capped.ok.bytes.len + 16, .n = 24 },
        .{ .room = exported.ok.bytes.len, .n = 24 },
        .{ .room = 2 * exported.ok.bytes.len, .n = exported.ok.n },
    }) |budget| {
        const held = http_server.CACHE_TRANSFER_BUDGET_BYTES - budget.room;
        try testing.expect(http_server.reserveCacheExportBytes(held));
        defer http_server.releaseCacheExportBytes(held);
        var req: @import("../../host/gpu_thread/requests.zig").CacheExportRequest = .{
            .model = model_a.entry,
            .prompt_ids = &.{},
            .has_tools = false,
            .session = 77,
            .max_bytes = router.max_cache_export_bytes,
            .admit_transfer_budget = true,
            .allocator = testing.allocator,
        };
        peer_transfer.runCacheExportRequest(&model_a.sch, &req);
        try testing.expect(req.outcome == .ok);
        defer http_server.releaseCacheExportBytes(@intCast(req.outcome.ok.bytes.len));
        defer testing.allocator.free(req.outcome.ok.bytes);
        try testing.expectEqual(budget.n, req.outcome.ok.n);
        try testing.expectEqual(held + req.outcome.ok.bytes.len, http_server.cacheTransferReservedBytes());
    }

    const warm_a = try model_a.runGreedy(&next, 4, false);
    defer testing.allocator.free(warm_a.tokens);
    const warm_b = try model_b.runGreedy(&next, 4, false);
    defer testing.allocator.free(warm_b.tokens);
    // The bar is a same-box continuation, which restores past the 24-token prompt boundary.
    try testing.expect(warm_a.cached > 25);
    try testing.expectEqual(warm_a.cached, warm_b.cached);
    try testing.expectEqualSlices(u32, warm_a.tokens, warm_b.tokens);
}

test "peer transfer session: a next prompt that drops the reply restores from the exported tail by a cut" {
    if (!mlx.metalKernelsAvailable() or mlx.noGpuBackend() or mlx.virtualGpu()) return error.SkipZigTest;
    const old_fits = scheduler.prefill_admission_fits;
    defer scheduler.prefill_admission_fits = old_fits;
    scheduler.prefill_admission_fits = null;

    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const files = try fixture.v41Files(arena.allocator(), 7);
    const model_a = try PeerHarness.open(files);
    defer model_a.close();
    const model_b = try PeerHarness.open(files);
    defer model_b.close();

    // Past MODULE_CUT_COST, so a cut is worth restoring.
    const prompt = buildPrompt(720);
    const turn1 = try model_a.runGreedyTagged(prompt[0..640], 2, true, 77);
    defer testing.allocator.free(turn1.tokens);
    const exported = exportBySession(model_a, 77);
    try testing.expect(exported == .ok);
    defer testing.allocator.free(exported.ok.bytes);
    try testing.expect(exported.ok.n > 640);
    try testing.expect(importWire(model_b, exported.ok.bytes) == .ok);

    // The next prompt continues the user text instead of the reply (a template that drops it, an edited history).
    const warm_a = try model_a.runGreedy(&prompt, 4, false);
    defer testing.allocator.free(warm_a.tokens);
    const warm_b = try model_b.runGreedy(&prompt, 4, false);
    defer testing.allocator.free(warm_b.tokens);
    try testing.expectEqual(@as(u32, 639), warm_a.cached);
    try testing.expect(warm_b.cached >= warm_a.cached);
    try testing.expectEqualSlices(u32, warm_a.tokens, warm_b.tokens);
}

test "peer transfer session: a session whose RAM entry is gone exports from the SSD and resumes on another box" {
    if (!mlx.metalKernelsAvailable() or mlx.noGpuBackend() or mlx.virtualGpu()) return error.SkipZigTest;
    const old_fits = scheduler.prefill_admission_fits;
    defer scheduler.prefill_admission_fits = old_fits;
    scheduler.prefill_admission_fits = null;

    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const files = try fixture.v41Files(arena.allocator(), 7);
    const model_a = try PeerHarness.open(files);
    defer model_a.close();
    const model_b = try PeerHarness.open(files);
    defer model_b.close();
    try model_a.attachDisk();

    var prompt: [513]u32 = undefined;
    for (&prompt, 0..) |*id, i| id.* = @intCast(3 + i % 80);
    const seeded = try model_a.runGreedyTagged(&prompt, 2, true, 9);
    defer testing.allocator.free(seeded.tokens);
    const hc = model_a.entry.prefix_cache.?;
    const disk_id = hc.entries.items[0].module.?.disk_id.?;
    for (0..32) |_| {
        const r = hc.flushOwedSlice(model_a.xfm.s);
        hc.disk.?.drainWriter();
        hc.disk.?.pollModuleWrites();
        if (r == .done) break;
    }
    try testing.expect(hc.disk.?.entryWholeOnDisk(disk_id));
    const tip_tokens = hc.sessionTip(9).?.n;
    hc.setBudget(1);
    hc.setBudget(0);
    try testing.expectEqual(@as(usize, 0), hc.entries.items.len);

    try testing.expectEqual(@as(@TypeOf(exportBySession(model_a, 10)), .not_found), exportBySession(model_a, 10));
    const capped = exportBySessionCapped(model_a, 9, 1);
    try testing.expect(capped == .too_large);
    try testing.expectEqual(@as(usize, 0), hc.entries.items.len);
    const exported = exportBySession(model_a, 9);
    try testing.expect(exported == .ok);
    defer testing.allocator.free(exported.ok.bytes);
    try testing.expectEqual(tip_tokens, exported.ok.n);
    try testing.expectEqual(@as(usize, 0), hc.entries.items.len);

    const imported = importWire(model_b, exported.ok.bytes);
    try testing.expect(imported == .ok);
    try testing.expectEqual(tip_tokens, model_b.entry.prefix_cache.?.sessionTip(9).?.n);

    const warm_a = try model_a.runGreedy(&prompt, 6, false);
    defer testing.allocator.free(warm_a.tokens);
    const warm_b = try model_b.runGreedy(&prompt, 6, false);
    defer testing.allocator.free(warm_b.tokens);
    try testing.expectEqual(@as(u32, @intCast(tip_tokens)), warm_b.cached);
    try testing.expectEqual(warm_a.cached, warm_b.cached);
    try testing.expectEqualSlices(u32, warm_a.tokens, warm_b.tokens);
}

fn landOwedWrites(h: *PeerHarness) void {
    const hc = h.entry.prefix_cache.?;
    for (0..32) |_| {
        const r = hc.flushOwedSlice(h.xfm.s);
        hc.disk.?.drainWriter();
        hc.disk.?.pollModuleWrites();
        if (r == .done) break;
    }
}

var imports_seen: usize = 0;

fn failFirstImport() void {
    imports_seen += 1;
    if (imports_seen == 1) mlx.latchErrorForTest("injected session import failure");
}

test "peer transfer session: a newest SSD import that fails with an MLX error leaves the older tip exportable" {
    if (!mlx.metalKernelsAvailable() or mlx.noGpuBackend() or mlx.virtualGpu()) return error.SkipZigTest;
    const old_fits = scheduler.prefill_admission_fits;
    defer scheduler.prefill_admission_fits = old_fits;
    scheduler.prefill_admission_fits = null;
    const old_probe = peer_transfer.after_aux_import_for_test;
    defer peer_transfer.after_aux_import_for_test = old_probe;
    imports_seen = 0;

    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const files = try fixture.v41Files(arena.allocator(), 7);
    const model_a = try PeerHarness.open(files);
    defer model_a.close();
    try model_a.attachDisk();

    var prompt: [514]u32 = undefined;
    for (&prompt, 0..) |*id, i| id.* = @intCast(3 + i % 80);
    const first = try model_a.runGreedyTagged(prompt[0..513], 2, true, 9);
    defer testing.allocator.free(first.tokens);
    landOwedWrites(model_a);
    const second = try model_a.runGreedyTagged(&prompt, 2, true, 9);
    defer testing.allocator.free(second.tokens);
    landOwedWrites(model_a);
    const hc = model_a.entry.prefix_cache.?;
    try testing.expectEqual(@as(usize, 513), hc.sessionTip(9).?.n);
    hc.setBudget(1);
    hc.setBudget(0);
    try testing.expectEqual(@as(usize, 0), hc.entries.items.len);

    peer_transfer.after_aux_import_for_test = failFirstImport;
    const exported = exportBySession(model_a, 9);
    try testing.expect(exported == .ok);
    defer testing.allocator.free(exported.ok.bytes);
    try testing.expectEqual(@as(usize, 512), exported.ok.n);
    try testing.expect(imports_seen >= 2);
}

test "peer transfer session: two sessions on one conversation each export their own entry" {
    if (!mlx.metalKernelsAvailable() or mlx.noGpuBackend() or mlx.virtualGpu()) return error.SkipZigTest;
    const old_fits = scheduler.prefill_admission_fits;
    defer scheduler.prefill_admission_fits = old_fits;
    scheduler.prefill_admission_fits = null;

    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const files = try fixture.v41Files(arena.allocator(), 7);
    const model_a = try PeerHarness.open(files);
    defer model_a.close();

    const full_prompt = buildPrompt(33);
    const short = try model_a.runGreedyTagged(full_prompt[0..25], 2, true, 1);
    defer testing.allocator.free(short.tokens);
    const long = try model_a.runGreedyTagged(&full_prompt, 2, true, 2);
    defer testing.allocator.free(long.tokens);

    const first = exportBySession(model_a, 1);
    try testing.expect(first == .ok);
    defer testing.allocator.free(first.ok.bytes);
    // The longer commit reached past session 1's prompt and dropped its decode-end entry: its boundary goes.
    try testing.expectEqual(@as(usize, 24), first.ok.n);
    const second = exportBySession(model_a, 2);
    try testing.expect(second == .ok);
    defer testing.allocator.free(second.ok.bytes);
    try testing.expectEqual(@as(usize, 35), second.ok.n);
}

test "peer transfer session: an imported session stays exportable after another session's decode-end entry that matched it is dropped" {
    if (!mlx.metalKernelsAvailable() or mlx.noGpuBackend() or mlx.virtualGpu()) return error.SkipZigTest;
    const old_fits = scheduler.prefill_admission_fits;
    defer scheduler.prefill_admission_fits = old_fits;
    scheduler.prefill_admission_fits = null;

    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const files = try fixture.v41Files(arena.allocator(), 7);
    const model_a = try PeerHarness.open(files);
    defer model_a.close();
    const model_b = try PeerHarness.open(files);
    defer model_b.close();

    // B serves session 77 on the prompt; A serves session 78 on the same prompt (same greedy reply) and moves it to B.
    const full_prompt = buildPrompt(33);
    const turn_77 = try model_b.runGreedyTagged(full_prompt[0..25], 2, true, 77);
    defer testing.allocator.free(turn_77.tokens);
    const turn_78 = try model_a.runGreedyTagged(full_prompt[0..25], 2, true, 78);
    defer testing.allocator.free(turn_78.tokens);
    const wire_78 = exportBySession(model_a, 78);
    try testing.expect(wire_78 == .ok);
    defer testing.allocator.free(wire_78.ok.bytes);
    try testing.expect(importWire(model_b, wire_78.ok.bytes) == .ok);

    // Session 77's next turn on B drops the reply: B drops every decode-end entry of that prompt.
    const next_77 = try model_b.runGreedyTagged(&full_prompt, 2, true, 77);
    defer testing.allocator.free(next_77.tokens);
    const again = exportBySession(model_b, 78);
    try testing.expect(again == .ok);
    defer testing.allocator.free(again.ok.bytes);
    try testing.expectEqual(@as(usize, 24), again.ok.n);
}

test "peer transfer session: an import another session's entry already covers still names its session" {
    if (!mlx.metalKernelsAvailable() or mlx.noGpuBackend() or mlx.virtualGpu()) return error.SkipZigTest;
    const old_fits = scheduler.prefill_admission_fits;
    defer scheduler.prefill_admission_fits = old_fits;
    scheduler.prefill_admission_fits = null;

    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const files = try fixture.v41Files(arena.allocator(), 7);
    const model_a = try PeerHarness.open(files);
    defer model_a.close();
    const model_b = try PeerHarness.open(files);
    defer model_b.close();

    const full_prompt = buildPrompt(33);
    const turn_77 = try model_a.runGreedyTagged(full_prompt[0..25], 2, true, 77);
    defer testing.allocator.free(turn_77.tokens);
    const wire_77 = exportBySession(model_a, 77);
    try testing.expect(wire_77 == .ok);
    defer testing.allocator.free(wire_77.ok.bytes);
    const turn_78 = try model_a.runGreedyTagged(full_prompt[0..25], 2, true, 78);
    defer testing.allocator.free(turn_78.tokens);
    const wire_78 = exportBySession(model_a, 78);
    try testing.expect(wire_78 == .ok);
    defer testing.allocator.free(wire_78.ok.bytes);

    try testing.expect(importWire(model_b, wire_77.ok.bytes) == .ok);
    // 78's boundary is covered (stale, claimed for 78); its decode-end entry goes in as 78's own tail.
    const covered = importWire(model_b, wire_78.ok.bytes);
    try testing.expect(covered == .ok);
    try testing.expectEqual(@as(usize, 24), model_b.entry.prefix_cache.?.sessionTip(78).?.n);

    const owned = exportBySession(model_b, 78);
    try testing.expect(owned == .ok);
    defer testing.allocator.free(owned.ok.bytes);
    // Session 78 owns the boundary that covered it and its own decode-end entry (the same prompt and greedy reply).
    try testing.expectEqual(@as(usize, 27), owned.ok.n);
    try testing.expectEqualSlices(u32, turn_77.tokens[0..2], turn_78.tokens[0..2]);
}

/// The token ids of each envelope a wire carries, the boundary first: one plain envelope or a session bundle.
fn expectWireIds(wire: []const u8, wants: []const []const u32) !void {
    var members: [peer_wire.MAX_BUNDLE_MEMBERS]peer_wire.BundleMember = undefined;
    const list = if (peer_wire.isBundle(wire)) try peer_wire.readBundle(wire, &members) else blk: {
        members[0] = .{ .tail_from = 0, .body = wire };
        break :blk members[0..1];
    };
    try testing.expectEqual(wants.len, list.len);
    for (list, wants) |m, want| {
        var decoded = try peer_wire.decode(testing.allocator, m.body);
        defer decoded.deinit();
        try testing.expectEqualSlices(u32, want, decoded.ids);
    }
}

const SessionBundleFixture = struct {
    sender: *PeerHarness,
    receiver: *PeerHarness,
    wire: []u8,
    members: [peer_wire.MAX_BUNDLE_MEMBERS]peer_wire.BundleMember,

    const session = 77;
    const prompt = buildPrompt(25);
    const boundary = prompt[0..24];

    fn open() !SessionBundleFixture {
        var arena = std.heap.ArenaAllocator.init(testing.allocator);
        defer arena.deinit();
        const files = try fixture.v41Files(arena.allocator(), 7);
        const sender = try PeerHarness.open(files);
        errdefer sender.close();
        const receiver = try PeerHarness.open(files);
        errdefer receiver.close();
        const turn = try sender.runGreedyTagged(&prompt, 2, true, session);
        defer testing.allocator.free(turn.tokens);
        const exported = exportBySession(sender, session);
        try testing.expect(exported == .ok);
        errdefer testing.allocator.free(exported.ok.bytes);
        var members: [peer_wire.MAX_BUNDLE_MEMBERS]peer_wire.BundleMember = undefined;
        try testing.expectEqual(@as(usize, 2), (try peer_wire.readBundle(exported.ok.bytes, &members)).len);
        return .{ .sender = sender, .receiver = receiver, .wire = exported.ok.bytes, .members = members };
    }

    fn close(f: SessionBundleFixture) void {
        testing.allocator.free(f.wire);
        f.receiver.close();
        f.sender.close();
    }

    fn expectBoundaryOnly(f: SessionBundleFixture) !void {
        const exported = exportBySession(f.receiver, session);
        try testing.expect(exported == .ok);
        defer testing.allocator.free(exported.ok.bytes);
        try expectWireIds(exported.ok.bytes, &.{boundary});
    }
};

test "peer transfer session: a covering prompt from another session cannot claim the imported reply as a tip" {
    if (!mlx.metalKernelsAvailable() or mlx.noGpuBackend() or mlx.virtualGpu()) return error.SkipZigTest;
    const old_fits = scheduler.prefill_admission_fits;
    defer scheduler.prefill_admission_fits = old_fits;
    scheduler.prefill_admission_fits = null;
    const f = try SessionBundleFixture.open();
    defer f.close();

    var tail = try peer_wire.decode(testing.allocator, f.members[1].body);
    defer tail.deinit();
    var other_prompt: [28]u32 = undefined;
    try testing.expectEqual(other_prompt.len - 1, tail.ids.len);
    @memcpy(other_prompt[0..tail.ids.len], tail.ids);
    other_prompt[tail.ids.len] = 42;
    const other = try f.receiver.runGreedyTagged(&other_prompt, 1, true, 88);
    defer testing.allocator.free(other.tokens);
    const hc = f.receiver.entry.prefix_cache.?;
    try expectPromptEntries(hc, &.{tail.ids});

    const out = importWire(f.receiver, f.wire);
    try testing.expect(out == .ok);
    try testing.expectEqual(SessionBundleFixture.boundary.len, hc.sessionTip(SessionBundleFixture.session).?.n);
    try expectPromptEntries(hc, &.{ tail.ids, SessionBundleFixture.boundary });
    try f.expectBoundaryOnly();
}

test "peer transfer session: a retried bundle preserves the receiver's newer reply" {
    if (!mlx.metalKernelsAvailable() or mlx.noGpuBackend() or mlx.virtualGpu()) return error.SkipZigTest;
    const old_fits = scheduler.prefill_admission_fits;
    defer scheduler.prefill_admission_fits = old_fits;
    scheduler.prefill_admission_fits = null;
    const f = try SessionBundleFixture.open();
    defer f.close();

    try testing.expect(importWire(f.receiver, f.wire) == .ok);
    const newer = try f.receiver.runGreedyTagged(&SessionBundleFixture.prompt, 4, true, SessionBundleFixture.session);
    defer testing.allocator.free(newer.tokens);
    const before = exportBySession(f.receiver, SessionBundleFixture.session);
    try testing.expect(before == .ok);
    defer testing.allocator.free(before.ok.bytes);
    try testing.expectEqual(@as(usize, 29), before.ok.n);

    const retry = importWire(f.receiver, f.wire);
    const after = exportBySession(f.receiver, SessionBundleFixture.session);
    try testing.expect(after == .ok);
    defer testing.allocator.free(after.ok.bytes);
    try testing.expectEqual(before.ok.n, after.ok.n);
    try testing.expectEqualSlices(u8, before.ok.bytes, after.ok.bytes);
    try testing.expectEqual(@as(@TypeOf(retry), .{ .refused = .stale }), retry);
}

test "peer transfer session: a mismatched bundle refuses the tail and retains the boundary" {
    if (!mlx.metalKernelsAvailable() or mlx.noGpuBackend() or mlx.virtualGpu()) return error.SkipZigTest;
    const old_fits = scheduler.prefill_admission_fits;
    defer scheduler.prefill_admission_fits = old_fits;
    scheduler.prefill_admission_fits = null;
    const f = try SessionBundleFixture.open();
    defer f.close();

    const other = try f.sender.runGreedyTagged(&SessionBundleFixture.prompt, 2, true, 88);
    defer testing.allocator.free(other.tokens);
    const exported = exportBySession(f.sender, 88);
    try testing.expect(exported == .ok);
    defer testing.allocator.free(exported.ok.bytes);
    var other_members: [peer_wire.MAX_BUNDLE_MEMBERS]peer_wire.BundleMember = undefined;
    try testing.expectEqual(@as(usize, 2), (try peer_wire.readBundle(exported.ok.bytes, &other_members)).len);
    const wire = try testing.allocator.alloc(u8, peer_wire.bundleSize(&.{ f.members[0].body.len, other_members[1].body.len }));
    defer testing.allocator.free(wire);
    peer_wire.writeBundle(wire, &.{ f.members[0], other_members[1] });

    const out = importWire(f.receiver, wire);
    try expectPromptEntries(f.receiver.entry.prefix_cache.?, &.{SessionBundleFixture.boundary});
    try f.expectBoundaryOnly();
    try testing.expectEqual(@as(@TypeOf(out), .{ .refused = .bad_envelope }), out);
    try testing.expectEqual(@as(usize, 1), f.receiver.entry.prefix_cache.?.entries.items.len);
}

fn restrictHeadroomForTail(f: SessionBundleFixture) !void {
    var tail = try peer_wire.decode(testing.allocator, f.members[1].body);
    defer tail.deinit();
    var plan: abi.AuxImportPlan = .{};
    try testing.expect(driver.auxImportPlanDetails(&f.receiver.xfm, tail.plugin_header, &plan).isOk());
    const peak = try prefill_request.moduleImportPeak(plan.peak, tail.tensors) + f.members[1].body.len;
    var active: usize = 0;
    _ = mlx.mlx_get_active_memory(&active);
    memory_ceiling.overrides().* = .{ .static_ceiling = active + peak - 1, .wired_limit_mb = 0 };
}

test "peer transfer session: a bundle with room for one entry keeps its boundary" {
    if (!mlx.metalKernelsAvailable() or mlx.noGpuBackend() or mlx.virtualGpu()) return error.SkipZigTest;
    const old_fits = scheduler.prefill_admission_fits;
    defer scheduler.prefill_admission_fits = old_fits;
    scheduler.prefill_admission_fits = null;
    const f = try SessionBundleFixture.open();
    defer f.close();
    const limits = memory_ceiling.overrides();
    const prev = limits.*;
    defer limits.* = prev;

    const hc = f.receiver.entry.prefix_cache.?;
    hc.max_entries = 1;
    // Seed the exact boundary without a session so this bundle can claim it.
    const seed = try f.receiver.runGreedy(&SessionBundleFixture.prompt, 1, true);
    defer testing.allocator.free(seed.tokens);
    try expectPromptEntries(hc, &.{SessionBundleFixture.boundary});
    try restrictHeadroomForTail(f);

    const out = importWire(f.receiver, f.wire);
    try expectPromptEntries(hc, &.{SessionBundleFixture.boundary});
    try testing.expect(out == .ok);
    try testing.expectEqual(SessionBundleFixture.boundary.len, out.ok.n);
    try f.expectBoundaryOnly();
}

test "peer transfer session: an allocation failure in the tail leaves a successful boundary import" {
    if (!mlx.metalKernelsAvailable() or mlx.noGpuBackend() or mlx.virtualGpu()) return error.SkipZigTest;
    const old_fits = scheduler.prefill_admission_fits;
    defer scheduler.prefill_admission_fits = old_fits;
    scheduler.prefill_admission_fits = null;
    const f = try SessionBundleFixture.open();
    defer f.close();

    const h = f.receiver;
    var counting = testing.FailingAllocator.init(testing.allocator, .{});
    const base = try peer_transfer.importWireModule(counting.allocator(), h.sch.io, &h.sch, h.entry, f.members[0].body);
    try testing.expect(base == .ok);
    try testing.expect(counting.alloc_index > 0);
    h.entry.prefix_cache.?.invalidateAll("replay with a failing allocator");

    var failing = testing.FailingAllocator.init(testing.allocator, .{ .fail_index = counting.alloc_index });
    const out = try peer_transfer.importWireModule(failing.allocator(), h.sch.io, &h.sch, h.entry, f.wire);
    try testing.expect(failing.has_induced_failure);
    try testing.expect(out == .ok);
    try testing.expectEqual(SessionBundleFixture.boundary.len, out.ok.n);
    try f.expectBoundaryOnly();
}

test "peer transfer session: tail admission under memory pressure preserves other sessions' boundaries" {
    if (!mlx.metalKernelsAvailable() or mlx.noGpuBackend() or mlx.virtualGpu()) return error.SkipZigTest;
    const old_fits = scheduler.prefill_admission_fits;
    defer scheduler.prefill_admission_fits = old_fits;
    scheduler.prefill_admission_fits = null;
    const f = try SessionBundleFixture.open();
    defer f.close();
    const limits = memory_ceiling.overrides();
    const prev = limits.*;
    defer limits.* = prev;

    var other_prompt = buildPrompt(17);
    other_prompt[0] = 90;
    const other = try f.receiver.runGreedyTagged(&other_prompt, 1, true, 88);
    defer testing.allocator.free(other.tokens);
    const seed = try f.receiver.runGreedy(&SessionBundleFixture.prompt, 1, true);
    defer testing.allocator.free(seed.tokens);
    const hc = f.receiver.entry.prefix_cache.?;
    try expectPromptEntries(hc, &.{ other_prompt[0..16], SessionBundleFixture.boundary });
    try restrictHeadroomForTail(f);

    const out = importWire(f.receiver, f.wire);
    try expectPromptEntries(hc, &.{ other_prompt[0..16], SessionBundleFixture.boundary });
    try testing.expect(out == .ok);
    try testing.expectEqual(SessionBundleFixture.boundary.len, out.ok.n);
    try f.expectBoundaryOnly();
}

test "peer transfer session: two sessions on different prompts of one length each export their own tokens" {
    if (!mlx.metalKernelsAvailable() or mlx.noGpuBackend() or mlx.virtualGpu()) return error.SkipZigTest;
    const old_fits = scheduler.prefill_admission_fits;
    defer scheduler.prefill_admission_fits = old_fits;
    scheduler.prefill_admission_fits = null;

    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const files = try fixture.v41Files(arena.allocator(), 7);
    const model_a = try PeerHarness.open(files);
    defer model_a.close();

    const first = buildPrompt(25);
    var second: [25]u32 = undefined;
    for (&second, 0..) |*id, i| id.* = @intCast(3 + (i * 11) % 79);
    const turn_1 = try model_a.runGreedyTagged(&first, 2, true, 1);
    defer testing.allocator.free(turn_1.tokens);
    const turn_2 = try model_a.runGreedyTagged(&second, 2, true, 2);
    defer testing.allocator.free(turn_2.tokens);

    const wire_1 = exportBySession(model_a, 1);
    try testing.expect(wire_1 == .ok);
    defer testing.allocator.free(wire_1.ok.bytes);
    const wire_2 = exportBySession(model_a, 2);
    try testing.expect(wire_2 == .ok);
    defer testing.allocator.free(wire_2.ok.bytes);
    var want_1: [27]u32 = undefined;
    @memcpy(want_1[0..25], &first);
    @memcpy(want_1[25..], turn_1.tokens[0..2]);
    var want_2: [27]u32 = undefined;
    @memcpy(want_2[0..25], &second);
    @memcpy(want_2[25..], turn_2.tokens[0..2]);
    try expectWireIds(wire_1.ok.bytes, &.{ first[0..24], &want_1 });
    try expectWireIds(wire_2.ok.bytes, &.{ second[0..24], &want_2 });
}

test "peer transfer session: an import a longer boundary covers exports exactly the tokens it carried" {
    if (!mlx.metalKernelsAvailable() or mlx.noGpuBackend() or mlx.virtualGpu()) return error.SkipZigTest;
    const old_fits = scheduler.prefill_admission_fits;
    defer scheduler.prefill_admission_fits = old_fits;
    scheduler.prefill_admission_fits = null;

    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const files = try fixture.v41Files(arena.allocator(), 7);
    const holder = try PeerHarness.open(files);
    defer holder.close();
    const sender = try PeerHarness.open(files);
    defer sender.close();

    var full_prompt = buildPrompt(33);
    const short_turn = try sender.runGreedyTagged(full_prompt[0..25], 2, true, 6);
    defer testing.allocator.free(short_turn.tokens);
    // The holder's longer prompt continues the sender's reply, so its boundary covers the sender's decode-end entry.
    @memcpy(full_prompt[25..27], short_turn.tokens[0..2]);
    const long_turn = try holder.runGreedyTagged(&full_prompt, 2, true, 5);
    defer testing.allocator.free(long_turn.tokens);
    try testing.expect(holder.entry.prefix_cache.?.hasModuleCovering(prefix_cache.ModuleBinding.init(&holder.xfm).?, full_prompt[0..27], &.{}, false, KVQuantConfig.dense));
    const short_wire = exportBySession(sender, 6);
    try testing.expect(short_wire == .ok);
    defer testing.allocator.free(short_wire.ok.bytes);

    const imported = importWire(holder, short_wire.ok.bytes);
    try testing.expect(imported == .ok);

    const own = exportBySession(holder, 6);
    try testing.expect(own == .ok);
    defer testing.allocator.free(own.ok.bytes);
    var carried: [27]u32 = undefined;
    @memcpy(carried[0..25], full_prompt[0..25]);
    @memcpy(carried[25..], short_turn.tokens[0..2]);
    try testing.expectEqual(@as(usize, 27), own.ok.n);
    try expectWireIds(own.ok.bytes, &.{ full_prompt[0..24], &carried });
    const other = exportBySession(holder, 5);
    try testing.expect(other == .ok);
    defer testing.allocator.free(other.ok.bytes);
    try testing.expectEqual(@as(usize, 35), other.ok.n);
}

test "dsv41 peer fetch: importWireModule refuses layout key, pack identity, settings, quant config, stale entry and low memory headroom" {
    if (!mlx.metalKernelsAvailable() or mlx.noGpuBackend() or mlx.virtualGpu()) return error.SkipZigTest;
    const old_fits = scheduler.prefill_admission_fits;
    defer scheduler.prefill_admission_fits = old_fits;
    scheduler.prefill_admission_fits = null;

    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const files = try fixture.v41Files(arena.allocator(), 7);
    const model_a = try PeerHarness.open(files);
    defer model_a.close();
    const model_b = try PeerHarness.open(files);
    defer model_b.close();

    const full_prompt = buildPrompt(33);
    const seed_short = try model_a.runGreedy(full_prompt[0..17], 2, true);
    defer testing.allocator.free(seed_short.tokens);
    const exp_short = try peer_transfer.exportWireModule(
        testing.allocator,
        model_a.sch.io,
        &model_a.sch,
        model_a.entry,
        full_prompt[0..25],
        false,
        null,
        0,
        0,
        router.max_cache_export_bytes,
    );
    const wire_short = exp_short.ok.bytes;
    defer testing.allocator.free(wire_short);
    try testing.expectEqual(@as(usize, 16), exp_short.ok.n);

    const seed_long = try model_a.runGreedy(full_prompt[0..25], 2, true);
    defer testing.allocator.free(seed_long.tokens);
    const exp_long = try peer_transfer.exportWireModule(
        testing.allocator,
        model_a.sch.io,
        &model_a.sch,
        model_a.entry,
        &full_prompt,
        false,
        null,
        0,
        0,
        router.max_cache_export_bytes,
    );
    const wire_long = exp_long.ok.bytes;
    defer testing.allocator.free(wire_long);
    try testing.expectEqual(@as(usize, 24), exp_long.ok.n);

    // 1. layout_key refusal
    {
        const bad_layout = try testing.allocator.dupe(u8, wire_long);
        defer testing.allocator.free(bad_layout);
        bad_layout[32] ^= 1;
        peer_wire.stampChecksum(bad_layout);
        const out = try peer_transfer.importWireModule(testing.allocator, model_b.sch.io, &model_b.sch, model_b.entry, bad_layout);
        try testing.expectEqual(@as(@TypeOf(out), .{ .refused = .layout_key }), out);
    }

    // 2. pack_identity refusal (changed config.json and changed weight file size)
    {
        const orig_cfg = try model_b.tmp.dir.readFileAlloc(model_b.sch.io, "config.json", testing.allocator, .limited(1 << 20));
        defer testing.allocator.free(orig_cfg);
        const alt_cfg = try std.fmt.allocPrint(testing.allocator, "{s} ", .{orig_cfg});
        defer testing.allocator.free(alt_cfg);
        try model_b.tmp.dir.writeFile(model_b.sch.io, .{ .sub_path = "config.json", .data = alt_cfg });
        try model_b.rehash();
        const out_cfg = try peer_transfer.importWireModule(testing.allocator, model_b.sch.io, &model_b.sch, model_b.entry, wire_long);
        try testing.expectEqual(@as(@TypeOf(out_cfg), .{ .refused = .pack_identity }), out_cfg);
        try model_b.tmp.dir.writeFile(model_b.sch.io, .{ .sub_path = "config.json", .data = orig_cfg });

        try model_b.tmp.dir.writeFile(model_b.sch.io, .{ .sub_path = "extra.safetensors", .data = "extra" });
        try model_b.rehash();
        const out_size = try peer_transfer.importWireModule(testing.allocator, model_b.sch.io, &model_b.sch, model_b.entry, wire_long);
        try testing.expectEqual(@as(@TypeOf(out_size), .{ .refused = .pack_identity }), out_size);
        try model_b.tmp.dir.deleteFile(model_b.sch.io, "extra.safetensors");
        try model_b.rehash();
    }

    // 3. settings refusal (changed model flag)
    {
        model_b.cfg.lm_head_mxfp8_override = true;
        defer model_b.cfg.lm_head_mxfp8_override = null;
        const out = try peer_transfer.importWireModule(testing.allocator, model_b.sch.io, &model_b.sch, model_b.entry, wire_long);
        try testing.expectEqual(@as(@TypeOf(out), .{ .refused = .settings }), out);
    }

    // 4. quant_config refusal
    {
        model_b.sch.kv_quant_config = KVQuantConfig.affine(8);
        defer model_b.sch.kv_quant_config = KVQuantConfig.dense;
        const out = try peer_transfer.importWireModule(testing.allocator, model_b.sch.io, &model_b.sch, model_b.entry, wire_long);
        try testing.expectEqual(@as(@TypeOf(out), .{ .refused = .quant_config }), out);
    }

    // 5. no memory headroom refusal before import_arrays
    {
        const limits = memory_ceiling.overrides();
        const prev = limits.*;
        defer limits.* = prev;
        limits.* = .{ .static_ceiling = 1, .wired_limit_mb = 0 };
        const out = try peer_transfer.importWireModule(testing.allocator, model_b.sch.io, &model_b.sch, model_b.entry, wire_long);
        try testing.expectEqual(@as(@TypeOf(out), .{ .refused = .no_memory }), out);
        try testing.expectEqual(@as(usize, 0), model_b.entry.prefix_cache.?.entries.items.len);
    }

    // 6. stale check: shorter resident rung accepts a longer tip; duplicate and shorter incoming tips refuse
    {
        const ok_short = try peer_transfer.importWireModule(testing.allocator, model_b.sch.io, &model_b.sch, model_b.entry, wire_short);
        try testing.expect(ok_short == .ok);
        const dup_short = try peer_transfer.importWireModule(testing.allocator, model_b.sch.io, &model_b.sch, model_b.entry, wire_short);
        try testing.expectEqual(@as(@TypeOf(dup_short), .{ .refused = .stale }), dup_short);

        const ok_long = try peer_transfer.importWireModule(testing.allocator, model_b.sch.io, &model_b.sch, model_b.entry, wire_long);
        try testing.expect(ok_long == .ok);
        const stale_short = try peer_transfer.importWireModule(testing.allocator, model_b.sch.io, &model_b.sch, model_b.entry, wire_short);
        try testing.expectEqual(@as(@TypeOf(stale_short), .{ .refused = .stale }), stale_short);
    }
}

test "dsv41 peer fetch: a pack with the same config, names and sizes but other weights refuses the import as pack_identity" {
    if (!mlx.metalKernelsAvailable() or mlx.noGpuBackend() or mlx.virtualGpu()) return error.SkipZigTest;
    const old_fits = scheduler.prefill_admission_fits;
    defer scheduler.prefill_admission_fits = old_fits;
    scheduler.prefill_admission_fits = null;

    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const files_a = try fixture.v41Files(arena.allocator(), 7);
    const files_b = try fixture.v41Files(arena.allocator(), 8);
    try testing.expectEqual(files_a[1].bytes.len, files_b[1].bytes.len);
    const model_a = try PeerHarness.open(files_a);
    defer model_a.close();
    const model_b = try PeerHarness.open(files_b);
    defer model_b.close();

    const prompt = buildPrompt(25);
    const seeded = try model_a.runGreedy(prompt[0..17], 2, true);
    defer testing.allocator.free(seeded.tokens);
    const exp = try peer_transfer.exportWireModule(testing.allocator, model_a.sch.io, &model_a.sch, model_a.entry, &prompt, false, null, 0, 0, router.max_cache_export_bytes);
    const wire = exp.ok.bytes;
    defer testing.allocator.free(wire);
    const out = try peer_transfer.importWireModule(testing.allocator, model_b.sch.io, &model_b.sch, model_b.entry, wire);
    try testing.expectEqual(@as(@TypeOf(out), .{ .refused = .pack_identity }), out);
}

test "dsv41 peer fetch: two packs that differ only at byte 65536 of a 5 MiB weight file refuse the import as pack_identity" {
    if (!mlx.metalKernelsAvailable() or mlx.noGpuBackend() or mlx.virtualGpu()) return error.SkipZigTest;
    const old_fits = scheduler.prefill_admission_fits;
    defer scheduler.prefill_admission_fits = old_fits;
    scheduler.prefill_admission_fits = null;

    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const files = try fixture.v41Files(arena.allocator(), 7);
    const model_a = try PeerHarness.open(files);
    defer model_a.close();
    const model_b = try PeerHarness.open(files);
    defer model_b.close();
    const pad = try arena.allocator().alloc(u8, 5 * 1024 * 1024);
    @memset(pad, 0);
    try model_a.tmp.dir.writeFile(model_a.sch.io, .{ .sub_path = "pad.bin", .data = pad });
    try model_a.rehash();
    pad[65536] = 1;
    try model_b.tmp.dir.writeFile(model_b.sch.io, .{ .sub_path = "pad.bin", .data = pad });
    try model_b.rehash();

    const prompt = buildPrompt(25);
    const seeded = try model_a.runGreedy(prompt[0..17], 2, true);
    defer testing.allocator.free(seeded.tokens);
    const exp = try peer_transfer.exportWireModule(testing.allocator, model_a.sch.io, &model_a.sch, model_a.entry, &prompt, false, null, 0, 0, router.max_cache_export_bytes);
    const wire = exp.ok.bytes;
    defer testing.allocator.free(wire);
    const out = try peer_transfer.importWireModule(testing.allocator, model_b.sch.io, &model_b.sch, model_b.entry, wire);
    try testing.expectEqual(@as(@TypeOf(out), .{ .refused = .pack_identity }), out);
}

test "dsv41 peer fetch: a file-backed model gets an identity and reaches the cache" {
    if (!mlx.metalKernelsAvailable() or mlx.noGpuBackend() or mlx.virtualGpu()) return error.SkipZigTest;
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const h = try PeerHarness.open(try fixture.v41Files(arena.allocator(), 7));
    defer h.close();
    try h.tmp.dir.writeFile(h.sch.io, .{ .sub_path = "model.gguf", .data = "GGUF" });
    const dir_path = h.entry.path;
    h.entry.path = try std.fmt.allocPrint(arena.allocator(), "{s}/model.gguf", .{h.path_buf[0..h.path_len]});
    defer h.entry.path = dir_path;
    try h.rehash();

    const prompt = buildPrompt(25);
    const exp = try peer_transfer.exportWireModule(testing.allocator, h.sch.io, &h.sch, h.entry, &prompt, false, null, 0, 0, router.max_cache_export_bytes);
    try testing.expectEqual(@as(@TypeOf(exp), .not_found), exp);
    const imp = try peer_transfer.importWireModule(testing.allocator, h.sch.io, &h.sch, h.entry, "abc");
    try testing.expectEqual(@as(@TypeOf(imp), .{ .refused = .truncated_body }), imp);
}

test "dsv41 peer fetch: export and import answer identity_pending until the hash completes" {
    if (!mlx.metalKernelsAvailable() or mlx.noGpuBackend() or mlx.virtualGpu()) return error.SkipZigTest;
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const h = try PeerHarness.open(try fixture.v41Files(arena.allocator(), 7));
    defer h.close();
    pack_identity.holdForTest();
    defer pack_identity.hold_for_test.store(false, .release);
    try pack_identity.start(h.sch.io, testing.allocator, &h.entry.pack_identity, h.entry.path);

    const prompt = buildPrompt(25);
    const exp = try peer_transfer.exportWireModule(testing.allocator, h.sch.io, &h.sch, h.entry, &prompt, false, null, 0, 0, router.max_cache_export_bytes);
    try testing.expectEqual(@as(@TypeOf(exp), .identity_pending), exp);
    const imp = try peer_transfer.importWireModule(testing.allocator, h.sch.io, &h.sch, h.entry, "abc");
    try testing.expectEqual(@as(@TypeOf(imp), .{ .refused = .identity_pending }), imp);
}

test "dsv41 peer fetch: a transfer never starts the hash, and without one answers transfer_unsupported" {
    if (!mlx.metalKernelsAvailable() or mlx.noGpuBackend() or mlx.virtualGpu()) return error.SkipZigTest;
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const h = try PeerHarness.open(try fixture.v41Files(arena.allocator(), 7));
    defer h.close();
    pack_identity.drop(&h.entry.pack_identity);

    const prompt = buildPrompt(25);
    const exp = try peer_transfer.exportWireModule(testing.allocator, h.sch.io, &h.sch, h.entry, &prompt, false, null, 0, 0, router.max_cache_export_bytes);
    try testing.expectEqual(@as(@TypeOf(exp), .transfer_unsupported), exp);
    const imp = try peer_transfer.importWireModule(testing.allocator, h.sch.io, &h.sch, h.entry, "abc");
    try testing.expectEqual(@as(@TypeOf(imp), .{ .refused = .transfer_unsupported }), imp);
    try testing.expect(h.entry.pack_identity == null);
}

test "dsv41 peer fetch: a read error during the hash answers transfer_unsupported" {
    if (!mlx.metalKernelsAvailable() or mlx.noGpuBackend() or mlx.virtualGpu()) return error.SkipZigTest;
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const h = try PeerHarness.open(try fixture.v41Files(arena.allocator(), 7));
    defer h.close();
    pack_identity.holdForTest();
    defer pack_identity.hold_for_test.store(false, .release);
    try pack_identity.start(h.sch.io, testing.allocator, &h.entry.pack_identity, h.entry.path);
    pack_identity.waitParkedForTest(h.sch.io);
    try h.tmp.dir.writeFile(h.sch.io, .{ .sub_path = "config.json", .data = "" });
    pack_identity.hold_for_test.store(false, .release);
    try testing.expectError(error.PackIdentityUnsupported, pack_identity.waitReady(h.sch.io, h.entry.pack_identity));

    const prompt = buildPrompt(25);
    const exp = try peer_transfer.exportWireModule(testing.allocator, h.sch.io, &h.sch, h.entry, &prompt, false, null, 0, 0, router.max_cache_export_bytes);
    try testing.expectEqual(@as(@TypeOf(exp), .transfer_unsupported), exp);
}

test "dsv41 peer fetch: an import that cannot fit is refused before it evicts a resident" {
    if (!mlx.metalKernelsAvailable() or mlx.noGpuBackend() or mlx.virtualGpu()) return error.SkipZigTest;
    const old_fits = scheduler.prefill_admission_fits;
    defer scheduler.prefill_admission_fits = old_fits;
    scheduler.prefill_admission_fits = null;

    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const files = try fixture.v41Files(arena.allocator(), 7);
    const model_a = try PeerHarness.open(files);
    defer model_a.close();
    const model_b = try PeerHarness.open(files);
    defer model_b.close();

    const full_prompt = buildPrompt(33);
    const seed = try model_a.runGreedy(full_prompt[0..25], 2, true);
    defer testing.allocator.free(seed.tokens);
    const exp = try peer_transfer.exportWireModule(testing.allocator, model_a.sch.io, &model_a.sch, model_a.entry, &full_prompt, false, null, 0, 0, router.max_cache_export_bytes);
    const wire = exp.ok.bytes;
    defer testing.allocator.free(wire);

    var other: [25]u32 = undefined;
    for (&other, 0..) |*id, i| id.* = @intCast(5 + (i * 11) % 70);
    const r1 = try model_b.runGreedy(other[0..17], 2, true);
    defer testing.allocator.free(r1.tokens);
    var other2 = other;
    other2[0] = 90;
    const r2 = try model_b.runGreedy(other2[0..17], 2, true);
    defer testing.allocator.free(r2.tokens);
    const hc = model_b.entry.prefix_cache.?;
    try expectPromptEntries(hc, &.{ other[0..16], other2[0..16] });
    const resident_ids = try testing.allocator.alloc(u64, hc.entries.items.len);
    defer testing.allocator.free(resident_ids);
    for (hc.entries.items, resident_ids) |e, *id| id.* = e.id;

    {
        const limits = memory_ceiling.overrides();
        const prev = limits.*;
        defer limits.* = prev;
        limits.* = .{ .static_ceiling = 1, .wired_limit_mb = 0 };
        const out = try peer_transfer.importWireModule(testing.allocator, model_b.sch.io, &model_b.sch, model_b.entry, wire);
        try testing.expectEqual(@as(@TypeOf(out), .{ .refused = .no_memory }), out);
        try expectResidentIds(hc, resident_ids);
    }
    {
        const prev = hc.max_kv_bytes;
        defer hc.max_kv_bytes = prev;
        hc.max_kv_bytes = 1;
        const out = try peer_transfer.importWireModule(testing.allocator, model_b.sch.io, &model_b.sch, model_b.entry, wire);
        try testing.expectEqual(@as(@TypeOf(out), .{ .refused = .commit_declined }), out);
        try expectResidentIds(hc, resident_ids);
    }
    {
        hc.ram_enabled = false;
        defer hc.ram_enabled = true;
        const out = try peer_transfer.importWireModule(testing.allocator, model_b.sch.io, &model_b.sch, model_b.entry, wire);
        try testing.expectEqual(@as(@TypeOf(out), .{ .refused = .commit_declined }), out);
        try expectResidentIds(hc, resident_ids);
    }
}

test "dsv41 peer fetch: an import_arrays failure on a memory latch answers no_memory and inserts nothing" {
    if (!mlx.metalKernelsAvailable() or mlx.noGpuBackend() or mlx.virtualGpu()) return error.SkipZigTest;
    const old_fits = scheduler.prefill_admission_fits;
    defer scheduler.prefill_admission_fits = old_fits;
    scheduler.prefill_admission_fits = null;

    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const files = try fixture.v41Files(arena.allocator(), 7);
    const model_a = try PeerHarness.open(files);
    defer model_a.close();
    const model_b = try PeerHarness.open(files);
    defer model_b.close();

    const full_prompt = buildPrompt(25);
    const turn1 = try model_a.runGreedy(full_prompt[0..17], 2, true);
    defer testing.allocator.free(turn1.tokens);
    const exp = try peer_transfer.exportWireModule(testing.allocator, model_a.sch.io, &model_a.sch, model_a.entry, &full_prompt, false, null, 0, 0, router.max_cache_export_bytes);
    const wire = exp.ok.bytes;
    defer testing.allocator.free(wire);

    const Probe = struct {
        var latch_msg: [:0]const u8 = "";

        fn importArrays(_: *anyopaque, _: abi.Span(u32), _: abi.Str, _: abi.Span(abi.AuxTensor), out: *abi.Handle) callconv(.c) abi.Verdict {
            out.* = null;
            mlx.latchErrorForTest(latch_msg);
            return abi.Verdict.refuse(.unsupported, "forced import failure");
        }
    };

    const bound = model_b.xfm.plugin.?;
    const old_table = bound.table;
    defer bound.table = old_table;
    var hooks = old_table.cache.aux.?.*;
    hooks.import_arrays = &Probe.importArrays;
    var cache_hooks = old_table.cache.*;
    cache_hooks.aux = &hooks;
    var table = old_table.*;
    table.cache = &cache_hooks;
    bound.table = &table;

    const cases = [_]struct { msg: [:0]const u8, want: @import("../../host/gpu_thread/requests.zig").CacheImportRefusal }{
        .{ .msg = "[METAL] Command buffer execution failed: Insufficient Memory", .want = .no_memory },
        .{ .msg = "[broadcast_shapes] Shapes cannot be broadcast", .want = .import_arrays },
    };
    for (cases) |c| {
        Probe.latch_msg = c.msg;
        defer mlx.dropLatchedErrorUnless(false);
        const out = try peer_transfer.importWireModule(testing.allocator, model_b.sch.io, &model_b.sch, model_b.entry, wire);
        try testing.expectEqual(@as(@TypeOf(out), .{ .refused = c.want }), out);
        try testing.expectEqual(@as(usize, 0), model_b.entry.prefix_cache.?.entries.items.len);
    }
}

test "dsv41 peer fetch: a failed transfer keeps its error name without allocating and logs it once" {
    if (!mlx.metalKernelsAvailable() or mlx.noGpuBackend() or mlx.virtualGpu()) return error.SkipZigTest;
    const old_fits = scheduler.prefill_admission_fits;
    defer scheduler.prefill_admission_fits = old_fits;
    scheduler.prefill_admission_fits = null;
    const log = @import("../../sdk/util/log.zig");
    const requests = @import("../../host/gpu_thread/requests.zig");

    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const h = try PeerHarness.open(try fixture.v41Files(arena.allocator(), 7));
    defer h.close();
    const prompt = buildPrompt(25);
    const seed = try h.runGreedy(prompt[0..17], 2, true);
    defer testing.allocator.free(seed.tokens);
    const wire = try peer_transfer.exportWireModule(testing.allocator, h.sch.io, &h.sch, h.entry, &prompt, false, null, 0, 0, router.max_cache_export_bytes);
    defer testing.allocator.free(wire.ok.bytes);

    var log_buf: [512]u8 = undefined;
    const log_path = try std.fmt.bufPrint(&log_buf, "{s}/transfer.log", .{h.path_buf[0..h.path_len]});
    try log.openFile(log_path, 0);
    var failing = testing.FailingAllocator.init(testing.allocator, .{ .fail_index = 0 });

    var exp: requests.CacheExportRequest = .{ .model = h.entry, .prompt_ids = &prompt, .has_tools = false, .max_bytes = router.max_cache_export_bytes, .allocator = failing.allocator() };
    peer_transfer.runCacheExportRequest(&h.sch, &exp);
    var imp: requests.CacheImportRequest = .{ .model = h.entry, .body = wire.ok.bytes, .allocator = failing.allocator() };
    peer_transfer.runCacheImportRequest(&h.sch, &imp);
    log.closeFile();

    try testing.expect(exp.done and imp.done);
    try testing.expectEqualStrings("OutOfMemory", exp.error_name orelse "");
    try testing.expectEqualStrings("OutOfMemory", imp.error_name orelse "");
    const text = try h.tmp.dir.readFileAlloc(h.sch.io, "transfer.log", testing.allocator, .limited(1 << 20));
    defer testing.allocator.free(text);
    try testing.expectEqual(@as(usize, 1), std.mem.count(u8, text, "peer cache export failed: OutOfMemory\n"));
    try testing.expectEqual(@as(usize, 1), std.mem.count(u8, text, "peer cache import failed: OutOfMemory\n"));
}

fn expectLeverRefusesImport(comptime lever: @EnumLiteral(), value: [*:0]const u8) !void {
    if (!mlx.metalKernelsAvailable() or mlx.noGpuBackend() or mlx.virtualGpu()) return error.SkipZigTest;
    const old_fits = scheduler.prefill_admission_fits;
    defer scheduler.prefill_admission_fits = old_fits;
    scheduler.prefill_admission_fits = null;
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const files = try fixture.v41Files(arena.allocator(), 7);
    const model_a = try PeerHarness.open(files);
    defer model_a.close();
    const model_b = try PeerHarness.open(files);
    defer model_b.close();

    const prompt = buildPrompt(25);
    const seeded = try model_a.runGreedy(prompt[0..17], 2, true);
    defer testing.allocator.free(seeded.tokens);
    const exp = try peer_transfer.exportWireModule(testing.allocator, model_a.sch.io, &model_a.sch, model_a.entry, &prompt, false, null, 0, 0, router.max_cache_export_bytes);
    const wire = exp.ok.bytes;
    defer testing.allocator.free(wire);

    const base = peer_transfer.modelSettingsKey(model_b.entry, &model_b.cfg, &model_b.xfm);
    const saved = env.override(lever, value);
    defer saved.restore();
    try testing.expect(base != peer_transfer.modelSettingsKey(model_b.entry, &model_b.cfg, &model_b.xfm));
    const out = try peer_transfer.importWireModule(testing.allocator, model_b.sch.io, &model_b.sch, model_b.entry, wire);
    try testing.expectEqual(@as(@TypeOf(out), .{ .refused = .settings }), out);
}

test "dsv41 peer fetch: MLX_SERVE_DSV41_ATTN_F16 off refuses an import as settings" {
    try expectLeverRefusesImport(.MLX_SERVE_DSV41_ATTN_F16, "0");
}

test "dsv41 peer fetch: MLX_SERVE_DSV41_CED off refuses an import as settings" {
    try expectLeverRefusesImport(.MLX_SERVE_DSV41_CED, "0");
}

test "dsv41 peer fetch: the settings key keeps the flash mode its model loaded with, whatever a later load sets" {
    if (!mlx.metalKernelsAvailable() or mlx.noGpuBackend() or mlx.virtualGpu()) return error.SkipZigTest;
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const chosen = flash_attn.ps().mode;
    defer flash_attn.ps().mode = chosen;
    const unset = env.override(.MLX_SERVE_DSV41_FLASH_ATTN, null);
    defer unset.restore();
    const h = try PeerHarness.open(try fixture.v41Files(arena.allocator(), 7));
    defer h.close();
    try testing.expect(flash_attn.ps().mode == .off);
    const loaded_off = peer_transfer.modelSettingsKey(h.entry, &h.cfg, &h.xfm);

    flash_attn.ps().mode = .on;
    try testing.expectEqual(loaded_off, peer_transfer.modelSettingsKey(h.entry, &h.cfg, &h.xfm));
    {
        const text = env.override(.MLX_SERVE_DSV41_FLASH_ATTN, "1");
        defer text.restore();
        try testing.expectEqual(loaded_off, peer_transfer.modelSettingsKey(h.entry, &h.cfg, &h.xfm));
    }

    const flash_on = env.override(.MLX_SERVE_DSV41_FLASH_ATTN, "1");
    defer flash_on.restore();
    const h_on = try PeerHarness.open(try fixture.v41Files(arena.allocator(), 7));
    defer h_on.close();
    try testing.expect(flash_attn.ps().mode == .on);
    try testing.expect(loaded_off != peer_transfer.modelSettingsKey(h_on.entry, &h_on.cfg, &h_on.xfm));
}

test "dsv41 peer fetch: a plugin's settings_key hook is part of the settings key and no hook adds nothing" {
    if (!mlx.metalKernelsAvailable() or mlx.noGpuBackend() or mlx.virtualGpu()) return error.SkipZigTest;
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const h = try PeerHarness.open(try fixture.v41Files(arena.allocator(), 7));
    defer h.close();
    const Probe = struct {
        var value: u64 = 1;
        fn settingsKey(_: *anyopaque) callconv(.c) u64 {
            return value;
        }
    };
    const bound = h.xfm.plugin.?;
    const old_table = bound.table;
    defer bound.table = old_table;
    var hooks = old_table.cache.aux.?.*;
    var cache_hooks = old_table.cache.*;
    cache_hooks.aux = &hooks;
    var table = old_table.*;
    table.cache = &cache_hooks;
    bound.table = &table;

    hooks.settings_key = &Probe.settingsKey;
    const one = peer_transfer.modelSettingsKey(h.entry, &h.cfg, &h.xfm);
    Probe.value = 2;
    const two = peer_transfer.modelSettingsKey(h.entry, &h.cfg, &h.xfm);
    try testing.expect(one != two);
    hooks.settings_key = null;
    const none = peer_transfer.modelSettingsKey(h.entry, &h.cfg, &h.xfm);
    try testing.expect(none != one and none != two);
    hooks.size = @offsetOf(abi.AuxHooks, "settings_key");
    hooks.settings_key = &Probe.settingsKey;
    try testing.expectEqual(none, peer_transfer.modelSettingsKey(h.entry, &h.cfg, &h.xfm));
}

test "dsv41 peer fetch: LAYER_CAP refuses an import as settings" {
    try expectLeverRefusesImport(.MLX_SERVE_DIAG, "layer-cap=3");
}

test "dsv41 peer fetch: DSV4_SKIP_MOE refuses an import as settings" {
    try expectLeverRefusesImport(.MLX_SERVE_DIAG, "dsv4-skip-moe");
}

test "dsv41 peer fetch: FORCE_GPU_FAMILY_FALLBACK refuses an import as settings" {
    try expectLeverRefusesImport(.MLX_SERVE_DIAG, "force-gpu-family-fallback");
}

test "dsv41 peer fetch: MLX_SERVE_KERNEL_DIR set answers transfer_unsupported on export and import" {
    if (!mlx.metalKernelsAvailable() or mlx.noGpuBackend() or mlx.virtualGpu()) return error.SkipZigTest;
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const h = try PeerHarness.open(try fixture.v41Files(arena.allocator(), 7));
    defer h.close();
    const saved = env.override(.MLX_SERVE_KERNEL_DIR, "/nonexistent-kernel-dir");
    defer saved.restore();
    const prompt = buildPrompt(25);
    const exp = try peer_transfer.exportWireModule(testing.allocator, h.sch.io, &h.sch, h.entry, &prompt, false, null, 0, 0, router.max_cache_export_bytes);
    try testing.expectEqual(@as(@TypeOf(exp), .transfer_unsupported), exp);
    const imp = try peer_transfer.importWireModule(testing.allocator, h.sch.io, &h.sch, h.entry, "abc");
    try testing.expectEqual(@as(@TypeOf(imp), .{ .refused = .transfer_unsupported }), imp);
}

test "dsv41 peer fetch: each numerics lever unset or at its default gives one settings key" {
    if (!mlx.metalKernelsAvailable() or mlx.noGpuBackend() or mlx.virtualGpu()) return error.SkipZigTest;
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const h = try PeerHarness.open(try fixture.v41Files(arena.allocator(), 7));
    defer h.close();
    const s0 = env.override(.MLX_SERVE_DSV41_ATTN_F16, null);
    defer s0.restore();
    const s1 = env.override(.MLX_SERVE_DSV41_CED, null);
    defer s1.restore();
    const s2 = env.override(.MLX_SERVE_DSV41_FLASH_ATTN, null);
    defer s2.restore();
    const s3 = env.override(.DSV41_TP_SUM, null);
    defer s3.restore();
    const s4 = env.override(.MLX_SERVE_DIAG, null);
    defer s4.restore();
    const unset = peer_transfer.modelSettingsKey(h.entry, &h.cfg, &h.xfm);
    inline for (.{
        .{ .MLX_SERVE_DSV41_ATTN_F16, "1" },
        .{ .MLX_SERVE_DSV41_CED, "1" },
        .{ .MLX_SERVE_DSV41_FLASH_ATTN, "1" },
        .{ .MLX_SERVE_DSV41_FLASH_ATTN, "" },
        .{ .DSV41_TP_SUM, "auto" },
        .{ .MLX_SERVE_DIAG, "layer-cap=0,dsv4-skip-moe=0,force-gpu-family-fallback=0" },
    }) |d| {
        const saved = env.override(d[0], d[1]);
        defer saved.restore();
        try testing.expectEqual(unset, peer_transfer.modelSettingsKey(h.entry, &h.cfg, &h.xfm));
    }
}

test "dsv41 peer fetch: export peak refusal releases its byte reservation before materialization" {
    if (!mlx.metalKernelsAvailable() or mlx.noGpuBackend() or mlx.virtualGpu()) return error.SkipZigTest;
    try testing.expectEqual(@as(u64, 0), http_server.serverState().cache_transfer_outbound_reserved.load(.acquire));
    const old_ceiling = memory_ceiling.overrides().static_ceiling;
    defer memory_ceiling.overrides().static_ceiling = old_ceiling;

    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const h = try PeerHarness.open(try fixture.v41Files(arena.allocator(), 7));
    defer h.close();
    const prompt = buildPrompt(25);
    const seed = try h.runGreedy(prompt[0..17], 2, true);
    defer testing.allocator.free(seed.tokens);

    memory_ceiling.overrides().static_ceiling = 1;
    var exp: @import("../../host/gpu_thread/requests.zig").CacheExportRequest = .{
        .model = h.entry,
        .prompt_ids = &prompt,
        .has_tools = false,
        .max_bytes = router.max_cache_export_bytes,
        .admit_transfer_budget = true,
        .allocator = testing.allocator,
    };
    peer_transfer.runCacheExportRequest(&h.sch, &exp);

    try testing.expect(exp.outcome == .no_memory);
    try testing.expectEqual(@as(u64, 0), http_server.serverState().cache_transfer_outbound_reserved.load(.acquire));
}

test "dsv41 peer fetch: a commit declined after import_arrays frees the imported snapshot once" {
    if (!mlx.metalKernelsAvailable() or mlx.noGpuBackend() or mlx.virtualGpu()) return error.SkipZigTest;
    const old_fits = scheduler.prefill_admission_fits;
    defer scheduler.prefill_admission_fits = old_fits;
    scheduler.prefill_admission_fits = null;

    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const files = try fixture.v41Files(arena.allocator(), 7);
    const model_a = try PeerHarness.open(files);
    defer model_a.close();
    const model_b = try PeerHarness.open(files);
    defer model_b.close();

    const full_prompt = buildPrompt(25);
    const turn1 = try model_a.runGreedy(full_prompt[0..17], 2, true);
    defer testing.allocator.free(turn1.tokens);
    const exp = try peer_transfer.exportWireModule(
        testing.allocator,
        model_a.sch.io,
        &model_a.sch,
        model_a.entry,
        &full_prompt,
        false,
        null,
        0,
        0,
        router.max_cache_export_bytes,
    );
    const wire = exp.ok.bytes;
    defer testing.allocator.free(wire);

    const Probe = struct {
        var real_free: *const fn (*anyopaque, abi.Handle) callconv(.c) void = undefined;
        var real_describe: *const fn (*anyopaque, abi.Handle, *abi.AuxSnapshotInfo) callconv(.c) abi.Verdict = undefined;
        var real_bytes: *const fn (*anyopaque, abi.Handle) callconv(.c) u64 = undefined;
        var free_calls: usize = 0;
        var corrupt_describe: bool = false;
        var inflate_bytes: bool = false;

        fn free(mp: *anyopaque, h: abi.Handle) callconv(.c) void {
            free_calls += 1;
            real_free(mp, h);
        }

        fn describe(mp: *anyopaque, h: abi.Handle, out: *abi.AuxSnapshotInfo) callconv(.c) abi.Verdict {
            const v = real_describe(mp, h, out);
            if (corrupt_describe and v.isOk()) out.n += 1;
            return v;
        }

        fn bytes(mp: *anyopaque, h: abi.Handle) callconv(.c) u64 {
            const b = real_bytes(mp, h);
            return if (inflate_bytes) b << 20 else b;
        }
    };

    const bound = model_b.xfm.plugin.?;
    const old_table = bound.table;
    defer bound.table = old_table;
    var hooks = old_table.cache.aux.?.*;
    Probe.real_free = hooks.free;
    Probe.real_describe = hooks.describe;
    Probe.real_bytes = hooks.bytes;
    hooks.free = &Probe.free;
    hooks.describe = &Probe.describe;
    hooks.bytes = &Probe.bytes;
    var cache_hooks = old_table.cache.*;
    cache_hooks.aux = &hooks;
    var table = old_table.*;
    table.cache = &cache_hooks;
    bound.table = &table;

    // Declined inside commitModule (the snapshot outgrows the budget the wire fit) -> commitModule's defer frees once.
    Probe.free_calls = 0;
    Probe.corrupt_describe = false;
    Probe.inflate_bytes = true;
    model_b.entry.prefix_cache.?.max_kv_bytes = wire.len << 4;
    const declined = try peer_transfer.importWireModule(testing.allocator, model_b.sch.io, &model_b.sch, model_b.entry, wire);
    try testing.expectEqual(@as(@TypeOf(declined), .{ .refused = .commit_declined }), declined);
    try testing.expectEqual(@as(usize, 1), Probe.free_calls);
    try testing.expectEqual(@as(usize, 0), model_b.entry.prefix_cache.?.entries.items.len);

    // Mismatch in step 3(e) before commitModule -> freed directly once.
    Probe.free_calls = 0;
    Probe.corrupt_describe = true;
    Probe.inflate_bytes = false;
    model_b.entry.prefix_cache.?.max_kv_bytes = 0;
    const bad_desc = try peer_transfer.importWireModule(testing.allocator, model_b.sch.io, &model_b.sch, model_b.entry, wire);
    try testing.expectEqual(@as(@TypeOf(bad_desc), .{ .refused = .describe_mismatch }), bad_desc);
    try testing.expectEqual(@as(usize, 1), Probe.free_calls);
    try testing.expectEqual(@as(usize, 0), model_b.entry.prefix_cache.?.entries.items.len);
}

const LiveNode = struct {
    tmp: testing.TmpDir,
    path_buf: [512]u8 = undefined,
    path_len: usize = 0,
    reg: *registry.ModelRegistry,
    entry: *registry.LoadedModel,
    sch: *HostScheduler,

    fn open(io: std.Io, files: []const sdk.fixture.File) !*LiveNode {
        const n = try openHashing(io, files);
        errdefer n.close();
        _ = try pack_identity.waitReady(io, n.entry.pack_identity);
        return n;
    }

    fn openHashing(io: std.Io, files: []const sdk.fixture.File) !*LiveNode {
        const a = testing.allocator;
        const n = try a.create(LiveNode);
        errdefer a.destroy(n);
        n.tmp = testing.tmpDir(.{ .iterate = true });
        errdefer n.tmp.cleanup();
        for (files) |f| try n.tmp.dir.writeFile(io, .{ .sub_path = f.name, .data = f.bytes });
        n.path_len = try n.tmp.dir.realPath(io, &n.path_buf);
        const dir = n.path_buf[0..n.path_len];

        n.reg = try registry.ModelRegistry.init(a, io, null, 4, 0, null);
        errdefer n.reg.deinit();
        n.entry = try n.reg.registerStub("dsv41-tiny", dir, 1);
        try n.reg.setDefault("dsv41-tiny");

        var cpu = try cpu_state.preloadCpuState(a, io, dir, 0, null);
        errdefer cpu_state.freeCpuState(a, &cpu);

        n.sch = try mlx_engine.start(a, io, .{
            .registry = n.reg,
            .entry = n.entry,
            .config = cpu.config,
            .tok = cpu.tok,
            .chat_config = cpu.chat_config,
            .model_dir = dir,
            .warmup_eager = false,
            .prefix_cache_capacity = 8,
            .prefix_cache_ram_enabled = true,
            .prefix_cache_mem_bytes = 64 * 1024 * 1024,
        }, 4);
        return n;
    }

    fn close(n: *LiveNode) void {
        n.sch.deinit();
        _ = mlx.mlx_clear_cache();
        n.reg.deinit();
        n.tmp.cleanup();
        testing.allocator.destroy(n);
    }
};

const HttpReply = struct {
    status: u16,
    raw: []u8,
    body: []const u8,

    fn deinit(r: *HttpReply, a: std.mem.Allocator) void {
        a.free(r.raw);
    }
};

fn expectStatus(name: []const u8, want: u16, reply: HttpReply) !void {
    if (reply.status != want) {
        std.debug.print("check {s}: got {d} want {d} body={s}\n", .{
            name,
            reply.status,
            want,
            reply.body[0..@min(reply.body.len, 256)],
        });
    }
    try testing.expectEqual(want, reply.status);
}

const HttpClientAction = enum { read_reply, half_close, cancel_after_content };

fn sendHttpRaw(
    io: std.Io,
    node: *LiveNode,
    peer_addr: std.Io.net.IpAddress,
    head_bytes: []const u8,
    body: []const u8,
    action: HttpClientAction,
) !HttpReply {
    const a = testing.allocator;
    const prev_reg = http_server.global_registry;
    const prev_sch = http_server.launch().global_scheduler;
    http_server.global_registry = node.reg;
    http_server.launch().global_scheduler = node.sch;
    defer {
        http_server.global_registry = prev_reg;
        http_server.launch().global_scheduler = prev_sch;
    }

    var sv: [2]std.posix.fd_t = undefined;
    const sp_rc = std.c.socketpair(1, 1, 0, &sv);
    if (sp_rc != 0) std.debug.print("check socketpair: got {d} want 0\n", .{sp_rc});
    try testing.expectEqual(@as(c_int, 0), sp_rc);
    defer _ = std.c.close(sv[1]);

    const worker = try std.Thread.spawn(.{}, struct {
        fn run(fd: std.posix.fd_t, addr: std.Io.net.IpAddress, wio: std.Io) void {
            var conn: Conn = undefined;
            Conn.init(&conn, .{ .socket = .{ .handle = fd, .address = addr } }, wio);
            defer conn.close();
            router.handleConnection(testing.allocator, &conn) catch |err| {
                std.debug.print("sendHttp worker error: {s}\n", .{@errorName(err)});
            };
        }
    }.run, .{ sv[0], peer_addr, io });

    try sendAllFd(sv[1], head_bytes);
    if (body.len > 0) try sendAllFd(sv[1], body);
    if (action == .half_close) _ = std.c.shutdown(sv[1], 1);

    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(a);
    var buf: [16 * 1024]u8 = undefined;
    var cancel_pending = action == .cancel_after_content;
    while (true) {
        const got = std.c.recv(sv[1], &buf, buf.len, 0);
        if (got < 0 and std.posix.errno(got) == .INTR) continue;
        if (got <= 0) break;
        try out.appendSlice(a, buf[0..@intCast(got)]);
        if (cancel_pending and std.mem.indexOf(u8, out.items, "\"delta\":{\"content\":") != null) {
            mlx_engine.api().cancel_all(node.sch);
            cancel_pending = false;
        }
    }
    if (action != .half_close) _ = std.c.shutdown(sv[1], 1);
    worker.join();

    const raw = try out.toOwnedSlice(a);
    errdefer a.free(raw);
    const sep = std.mem.indexOf(u8, raw, "\r\n\r\n") orelse {
        std.debug.print("check http_reply_framing: got {d}B raw={s}\n", .{ raw.len, raw[0..@min(raw.len, 256)] });
        return error.MalformedHttpReply;
    };
    const line_end = std.mem.indexOf(u8, raw[0..sep], "\r\n") orelse sep;
    var sp = std.mem.splitScalar(u8, raw[0..line_end], ' ');
    _ = sp.next();
    const code_str = sp.next() orelse return error.MalformedHttpReply;
    const status = try std.fmt.parseInt(u16, code_str, 10);
    return .{ .status = status, .raw = raw, .body = raw[sep + 4 ..] };
}

fn sendHttp(
    io: std.Io,
    node: *LiveNode,
    peer_addr: std.Io.net.IpAddress,
    head: []const u8,
    body: []const u8,
) !HttpReply {
    const req_bytes = try std.fmt.allocPrint(testing.allocator, "{s}Content-Length: {d}\r\n\r\n", .{ head, body.len });
    defer testing.allocator.free(req_bytes);
    return sendHttpRaw(io, node, peer_addr, req_bytes, body, .read_reply);
}

fn sendAllFd(fd: std.posix.fd_t, data: []const u8) !void {
    var off: usize = 0;
    while (off < data.len) {
        const sent = std.c.send(fd, data[off..].ptr, data.len - off, 0);
        if (sent < 0 and std.posix.errno(sent) == .INTR) continue;
        if (sent <= 0) return error.SendFailed;
        off += @intCast(sent);
    }
}

test "dsv41 peer fetch: HTTP export and import answer 409 identity_pending while the load's hash runs, and shutdown joins it" {
    if (!mlx.metalKernelsAvailable() or mlx.noGpuBackend() or mlx.virtualGpu()) return error.SkipZigTest;
    var threaded: std.Io.Threaded = .init(testing.allocator, .{});
    defer threaded.deinit();
    const io = threaded.io();

    const prev_ct = http_server.g_cache_transfer;
    defer http_server.g_cache_transfer = prev_ct;
    var ct_digest: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash("peer-secret-60-0123456789abcdef0123", &ct_digest, .{});
    http_server.g_cache_transfer = .{ .key = ct_digest };

    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    pack_identity.holdForTest();
    defer pack_identity.hold_for_test.store(false, .release);
    const node = try LiveNode.openHashing(io, try fixture.v41ServableFiles(arena.allocator(), 7));
    defer node.close();

    const loopback: std.Io.net.IpAddress = .{ .ip4 = .{ .bytes = .{ 127, 0, 0, 1 }, .port = 50000 } };
    const body = "{\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"temperature\":0,\"max_tokens\":4}";
    var exp = try sendHttp(io, node, loopback, "POST /v1/chat/completions HTTP/1.1\r\nContent-Type: application/json\r\nAuthorization: Bearer peer-secret-60-0123456789abcdef0123\r\nX-Cache-Export: 1\r\n", body);
    defer exp.deinit(testing.allocator);
    try expectStatus("export_identity_pending_409", 409, exp);
    try testing.expect(std.mem.indexOf(u8, exp.body, "\"reason\":\"identity_pending\"") != null);

    var imp = try sendHttp(io, node, loopback, "POST /v1/cache/import HTTP/1.1\r\nContent-Type: application/octet-stream\r\nAuthorization: Bearer peer-secret-60-0123456789abcdef0123\r\n", "abc");
    defer imp.deinit(testing.allocator);
    try expectStatus("import_identity_pending_409", 409, imp);
    try testing.expect(std.mem.indexOf(u8, imp.body, "\"reason\":\"identity_pending\"") != null);
}

test "dsv41 peer fetch: unloading while the load's hash runs joins it" {
    if (!mlx.metalKernelsAvailable() or mlx.noGpuBackend() or mlx.virtualGpu()) return error.SkipZigTest;
    var threaded: std.Io.Threaded = .init(testing.allocator, .{});
    defer threaded.deinit();
    const io = threaded.io();

    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    pack_identity.holdForTest();
    defer pack_identity.hold_for_test.store(false, .release);
    const node = try LiveNode.openHashing(io, try fixture.v41ServableFiles(arena.allocator(), 7));
    defer node.close();
    try testing.expectEqual(pack_identity.Status.pending, pack_identity.status(node.entry.pack_identity));
    try node.sch.unloadModel("dsv41-tiny");
    try testing.expect(node.entry.pack_identity == null);
}

test "dsv41 peer fetch: HTTP export and import enforce peer/auth gates, 422 media/speech, 413 cap, 404 miss and warm B turn" {
    if (!mlx.metalKernelsAvailable() or mlx.noGpuBackend() or mlx.virtualGpu()) return error.SkipZigTest;
    try testing.expectEqual(@as(u64, 0), http_server.cacheTransferReservedBytes());
    var threaded: std.Io.Threaded = .init(testing.allocator, .{});
    defer threaded.deinit();
    const io = threaded.io();

    const prev_key = http_server.g_api_key;
    const prev_strict = http_server.g_api_key_strict;
    const prev_ct = http_server.g_cache_transfer;
    defer {
        http_server.g_api_key = prev_key;
        http_server.g_api_key_strict = prev_strict;
        http_server.g_cache_transfer = prev_ct;
    }
    http_server.g_api_key = "serving-secret-60";
    http_server.g_api_key_strict = false;
    var ct_digest: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash("peer-secret-60-0123456789abcdef0123", &ct_digest, .{});
    http_server.g_cache_transfer = .{ .key = ct_digest };

    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const files = try fixture.v41ServableFiles(arena.allocator(), 7);
    const loopback: std.Io.net.IpAddress = .{ .ip4 = .{ .bytes = .{ 127, 0, 0, 1 }, .port = 50000 } };
    const remote_peer: std.Io.net.IpAddress = .{ .ip4 = .{ .bytes = .{ 192, 168, 86, 42 }, .port = 50002 } };

    const turn1_body = "{\"messages\":[{\"role\":\"user\",\"content\":\"Explain cache locality in systems programming.\"}],\"temperature\":0,\"max_tokens\":4,\"stream\":true}";
    const turn2_body = "{\"messages\":[{\"role\":\"user\",\"content\":\"Explain cache locality in systems programming.\"},{\"role\":\"assistant\",\"content\":\"hi\"},{\"role\":\"user\",\"content\":\"Now add one example.\"}],\"temperature\":0,\"max_tokens\":6,\"stream\":false}";
    const turn2_stream_body = "{\"messages\":[{\"role\":\"user\",\"content\":\"Explain cache locality in systems programming.\"},{\"role\":\"assistant\",\"content\":\"hi\"},{\"role\":\"user\",\"content\":\"Now add one example.\"}],\"temperature\":0,\"max_tokens\":6,\"stream\":true}";

    var exp_reply: HttpReply = undefined;
    var exp_ready = false;
    defer if (exp_ready) exp_reply.deinit(testing.allocator);

    var a_turn2: HttpReply = undefined;
    var a_turn2_ready = false;
    defer if (a_turn2_ready) a_turn2.deinit(testing.allocator);

    {
        const node_a = try LiveNode.open(io, files);
        defer node_a.close();

        for ([_][]const u8{ "abc", "0", "-5" }) |deadline| {
            const export_head = try std.fmt.allocPrint(
                testing.allocator,
                "POST /v1/chat/completions HTTP/1.1\r\nContent-Type: application/json\r\nAuthorization: Bearer peer-secret-60-0123456789abcdef0123\r\nX-Cache-Export: 1\r\nX-Cache-Deadline-Ms: {s}\r\n",
                .{deadline},
            );
            defer testing.allocator.free(export_head);
            var export_reply = try sendHttp(io, node_a, loopback, export_head, turn1_body);
            defer export_reply.deinit(testing.allocator);
            try expectStatus("export_invalid_deadline_400", 400, export_reply);
        }

        {
            // Loopback-only mode refuses remote peers (403); key mode refuses missing/serving keys (401).
            http_server.g_cache_transfer = .loopback;
            defer http_server.g_cache_transfer = .{ .key = ct_digest };
            var r1 = try sendHttp(io, node_a, remote_peer, "POST /v1/chat/completions HTTP/1.1\r\nContent-Type: application/json\r\nX-Cache-Export: 1\r\n", turn1_body);
            defer r1.deinit(testing.allocator);
            try expectStatus("export_remote_loopback_mode_403", 403, r1);
        }
        {
            var r1 = try sendHttp(io, node_a, loopback, "POST /v1/chat/completions HTTP/1.1\r\nContent-Type: application/json\r\nX-Cache-Export: 1\r\n", turn1_body);
            defer r1.deinit(testing.allocator);
            try expectStatus("export_no_auth_401", 401, r1);
        }
        {
            var r_serving = try sendHttp(io, node_a, remote_peer, "POST /v1/chat/completions HTTP/1.1\r\nContent-Type: application/json\r\nAuthorization: Bearer serving-secret-60\r\nX-Cache-Export: 1\r\n", turn1_body);
            defer r_serving.deinit(testing.allocator);
            try expectStatus("export_serving_key_rejected_401", 401, r_serving);
        }
        {
            const media_body = "{\"messages\":[{\"role\":\"user\",\"content\":[{\"type\":\"image_url\",\"image_url\":{\"url\":\"not-a-valid-image\"}},{\"type\":\"text\",\"text\":\"hi\"}]}]}";
            var r1 = try sendHttp(io, node_a, loopback, "POST /v1/chat/completions HTTP/1.1\r\nContent-Type: application/json\r\nAuthorization: Bearer peer-secret-60-0123456789abcdef0123\r\nX-Cache-Export: 1\r\n", media_body);
            defer r1.deinit(testing.allocator);
            try expectStatus("export_media_422", 422, r1);

            const speech_body = "{\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"modalities\":[\"text\",\"audio\"]}";
            var r2 = try sendHttp(io, node_a, loopback, "POST /v1/chat/completions HTTP/1.1\r\nContent-Type: application/json\r\nAuthorization: Bearer peer-secret-60-0123456789abcdef0123\r\nX-Cache-Export: 1\r\n", speech_body);
            defer r2.deinit(testing.allocator);
            try expectStatus("export_speech_422", 422, r2);
        }
        {
            var miss = try sendHttp(io, node_a, loopback, "POST /v1/chat/completions HTTP/1.1\r\nContent-Type: application/json\r\nAuthorization: Bearer peer-secret-60-0123456789abcdef0123\r\nX-Cache-Export: 1\r\n", turn2_stream_body);
            defer miss.deinit(testing.allocator);
            try expectStatus("export_cold_miss_404", 404, miss);
        }
        {
            var t1 = try sendHttp(io, node_a, loopback, "POST /v1/chat/completions HTTP/1.1\r\nContent-Type: application/json\r\n", turn1_body);
            defer t1.deinit(testing.allocator);
            try expectStatus("seed_turn1_200", 200, t1);
            const saw_done = std.mem.indexOf(u8, t1.body, "data: [DONE]") != null;
            if (!saw_done) {
                std.debug.print("check seed_turn1_done: got body({d}B)={s} want data: [DONE]\n", .{
                    t1.body.len,
                    t1.body[0..@min(t1.body.len, 256)],
                });
            }
            try testing.expect(saw_done);
        }

        const export_head = "POST /v1/chat/completions HTTP/1.1\r\nContent-Type: application/json\r\nAuthorization: Bearer peer-secret-60-0123456789abcdef0123\r\nX-Cache-Export: 1\r\nX-Cache-Deadline-Ms: 60000\r\n";
        const inbound_before = http_server.serverState().cache_transfer_inbound_reserved.load(.acquire);
        const old_outbound = http_server.serverState().cache_transfer_outbound_reserved.load(.acquire);
        http_server.serverState().cache_transfer_outbound_reserved.store(http_server.CACHE_TRANSFER_BUDGET_BYTES - 1, .release);
        defer http_server.serverState().cache_transfer_outbound_reserved.store(old_outbound, .release);
        var aggregate_refusal = try sendHttp(io, node_a, loopback, export_head, turn2_stream_body);
        defer aggregate_refusal.deinit(testing.allocator);
        try expectStatus("export_aggregate_bytes_busy_503", 503, aggregate_refusal);
        try testing.expect(std.mem.indexOf(u8, aggregate_refusal.raw, "cache_export_no_memory") != null);
        try testing.expectEqual(http_server.CACHE_TRANSFER_BUDGET_BYTES - 1, http_server.serverState().cache_transfer_outbound_reserved.load(.acquire));
        http_server.serverState().cache_transfer_outbound_reserved.store(old_outbound, .release);
        try testing.expectEqual(inbound_before, http_server.serverState().cache_transfer_inbound_reserved.load(.acquire));

        exp_reply = try sendHttp(io, node_a, loopback, export_head, turn2_stream_body);
        exp_ready = true;
        try expectStatus("export_warm_200", 200, exp_reply);
        try testing.expect(std.mem.indexOf(u8, exp_reply.raw, "X-Cache-Import-Peak: ") != null);
        try testing.expectEqual(@as(u64, 0), http_server.cacheTransferReservedBytes());
        if (exp_reply.body.len <= peer_wire.FIXED_HEADER_BYTES) {
            std.debug.print("check export_wire_len: got {d} want > {d}\n", .{ exp_reply.body.len, peer_wire.FIXED_HEADER_BYTES });
        }
        try testing.expect(exp_reply.body.len > peer_wire.FIXED_HEADER_BYTES);

        {
            cache_transfer.max_export_bytes_override = exp_reply.body.len - 1;
            defer cache_transfer.max_export_bytes_override = null;
            var over_cap = try sendHttp(io, node_a, loopback, "POST /v1/chat/completions HTTP/1.1\r\nContent-Type: application/json\r\nAuthorization: Bearer peer-secret-60-0123456789abcdef0123\r\nX-Cache-Export: 1\r\n", turn2_stream_body);
            defer over_cap.deinit(testing.allocator);
            try expectStatus("export_over_cap_413", 413, over_cap);
        }

        a_turn2 = try sendHttp(io, node_a, loopback, "POST /v1/chat/completions HTTP/1.1\r\nContent-Type: application/json\r\n", turn2_body);
        a_turn2_ready = true;
        try expectStatus("node_a_turn2_200", 200, a_turn2);
    }

    var decoded_exp = try peer_wire.decode(testing.allocator, exp_reply.body);
    defer decoded_exp.deinit();
    const exported_n = decoded_exp.ids.len;
    if (exported_n == 0) {
        std.debug.print("check decoded_exported_n: got {d} want > 0\n", .{exported_n});
    }
    try testing.expect(exported_n > 0);

    const a_parsed = try std.json.parseFromSlice(std.json.Value, testing.allocator, a_turn2.body, .{});
    defer a_parsed.deinit();

    var b_turn2: HttpReply = undefined;
    var b_turn2_ready = false;
    defer if (b_turn2_ready) b_turn2.deinit(testing.allocator);

    {
        const node_b = try LiveNode.open(io, files);
        defer node_b.close();

        for ([_][]const u8{ "abc", "0", "-5" }) |deadline| {
            const import_head = try std.fmt.allocPrint(
                testing.allocator,
                "POST /v1/cache/import HTTP/1.1\r\nContent-Type: application/octet-stream\r\nAuthorization: Bearer peer-secret-60-0123456789abcdef0123\r\nX-Cache-Deadline-Ms: {s}\r\n",
                .{deadline},
            );
            defer testing.allocator.free(import_head);
            var import_reply = try sendHttp(io, node_b, remote_peer, import_head, "abc");
            defer import_reply.deinit(testing.allocator);
            try expectStatus("import_invalid_deadline_400", 400, import_reply);
        }

        {
            http_server.g_cache_transfer = .loopback;
            defer http_server.g_cache_transfer = .{ .key = ct_digest };
            var r2 = try sendHttp(io, node_b, remote_peer, "POST /v1/cache/import HTTP/1.1\r\nContent-Type: application/octet-stream\r\n", "abc");
            defer r2.deinit(testing.allocator);
            try expectStatus("import_remote_loopback_mode_403", 403, r2);
        }
        {
            var r2 = try sendHttp(io, node_b, loopback, "POST /v1/cache/import HTTP/1.1\r\nContent-Type: application/octet-stream\r\n", "abc");
            defer r2.deinit(testing.allocator);
            try expectStatus("import_no_auth_401", 401, r2);
        }
        {
            const oversize_head = try std.fmt.allocPrint(
                testing.allocator,
                "POST /v1/cache/import HTTP/1.1\r\nContent-Type: application/octet-stream\r\nAuthorization: Bearer peer-secret-60-0123456789abcdef0123\r\nContent-Length: {d}\r\n\r\n",
                .{router.max_cache_transfer_body_bytes + 1},
            );
            defer testing.allocator.free(oversize_head);
            var r2 = try sendHttpRaw(io, node_b, remote_peer, oversize_head, "", .half_close);
            defer r2.deinit(testing.allocator);
            try expectStatus("import_over_cap_413", 413, r2);
            try testing.expect(std.mem.indexOf(u8, r2.raw, "cache_transfer_too_large") != null);
        }

        const peak_start = (std.mem.indexOf(u8, exp_reply.raw, "X-Cache-Import-Peak: ") orelse return error.MissingPlannedPeakHeader) + "X-Cache-Import-Peak: ".len;
        const peak_end = peak_start + (std.mem.indexOfScalar(u8, exp_reply.raw[peak_start..], '\r') orelse return error.MalformedPlannedPeakHeader);
        const import_head = try std.fmt.allocPrint(testing.allocator, "POST /v1/cache/import HTTP/1.1\r\nContent-Type: application/octet-stream\r\nAuthorization: Bearer peer-secret-60-0123456789abcdef0123\r\nX-Cache-Import-Peak: {s}\r\n", .{exp_reply.raw[peak_start..peak_end]});
        defer testing.allocator.free(import_head);
        var imp_reply = try sendHttp(io, node_b, remote_peer, import_head, exp_reply.body);
        defer imp_reply.deinit(testing.allocator);
        try expectStatus("import_remote_keyed_200", 200, imp_reply);
        try testing.expectEqual(@as(u64, 0), http_server.cacheTransferReservedBytes());
        const imp_json = try std.json.parseFromSlice(std.json.Value, testing.allocator, imp_reply.body, .{});
        defer imp_json.deinit();
        const got_imp_n = imp_json.value.object.get("n").?.integer;
        if (got_imp_n != @as(i64, @intCast(exported_n))) {
            std.debug.print("check import_n: got {d} want {d} body={s}\n", .{ got_imp_n, exported_n, imp_reply.body });
        }
        try testing.expectEqual(@as(i64, @intCast(exported_n)), got_imp_n);

        b_turn2 = try sendHttp(io, node_b, loopback, "POST /v1/chat/completions HTTP/1.1\r\nContent-Type: application/json\r\n", turn2_body);
        b_turn2_ready = true;
        try expectStatus("node_b_turn2_200", 200, b_turn2);
    }

    const b_parsed = try std.json.parseFromSlice(std.json.Value, testing.allocator, b_turn2.body, .{});
    defer b_parsed.deinit();
    const a_cached = a_parsed.value.object.get("usage").?.object.get("prompt_tokens_details").?.object.get("cached_tokens").?.integer;
    const b_cached = b_parsed.value.object.get("usage").?.object.get("prompt_tokens_details").?.object.get("cached_tokens").?.integer;
    if (a_cached != @as(i64, @intCast(exported_n))) {
        std.debug.print("check node_a_cached: got {d} want {d} body={s}\n", .{ a_cached, exported_n, a_turn2.body });
    }
    try testing.expectEqual(@as(i64, @intCast(exported_n)), a_cached);
    if (b_cached != a_cached) {
        std.debug.print("check node_b_cached: got {d} want {d} body={s}\n", .{ b_cached, a_cached, b_turn2.body });
    }
    try testing.expectEqual(a_cached, b_cached);

    const a_msg = a_parsed.value.object.get("choices").?.array.items[0].object.get("message").?.object;
    const b_msg = b_parsed.value.object.get("choices").?.array.items[0].object.get("message").?.object;
    const a_content = if (a_msg.get("content")) |v| (if (v == .string) v.string else "") else "";
    const b_content = if (b_msg.get("content")) |v| (if (v == .string) v.string else "") else "";
    const a_reason = if (a_msg.get("reasoning_content")) |v| (if (v == .string) v.string else "") else "";
    const b_reason = if (b_msg.get("reasoning_content")) |v| (if (v == .string) v.string else "") else "";
    if (!std.mem.eql(u8, a_content, b_content)) {
        std.debug.print("check warm_b_content: got '{s}' want '{s}'\n", .{ b_content, a_content });
    }
    try testing.expectEqualStrings(a_content, b_content);
    if (!std.mem.eql(u8, a_reason, b_reason)) {
        std.debug.print("check warm_b_reasoning: got '{s}' want '{s}'\n", .{ b_reason, a_reason });
    }
    try testing.expectEqualStrings(a_reason, b_reason);
}


test "dsv41 peer fetch: HTTP cache import accepts a body above 512 MiB when slow gate is enabled" {
    if (!env.on(.CACHE_TRANSFER_SLOW_E2E)) return error.SkipZigTest;
    if (!mlx.metalKernelsAvailable() or mlx.noGpuBackend() or mlx.virtualGpu()) return error.SkipZigTest;
    var threaded: std.Io.Threaded = .init(testing.allocator, .{});
    defer threaded.deinit();
    const io = threaded.io();

    const prev_ct = http_server.g_cache_transfer;
    defer http_server.g_cache_transfer = prev_ct;
    http_server.g_cache_transfer = .loopback;

    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const files = try fixture.v41ServableFiles(arena.allocator(), 7);
    const node = try LiveNode.open(io, files);
    defer node.close();

    const body = try testing.allocator.alloc(u8, (512 * 1024 * 1024) + 1);
    defer testing.allocator.free(body);
    @memset(body, 0);
    var reply = try sendHttp(io, node, .{ .ip4 = .{ .bytes = .{ 127, 0, 0, 1 }, .port = 50000 } }, "POST /v1/cache/import HTTP/1.1\r\nContent-Type: application/octet-stream\r\n", body);
    defer reply.deinit(testing.allocator);
    try expectStatus("import_over_512m_reaches_transfer_decoder", 400, reply);
    try testing.expect(std.mem.indexOf(u8, reply.raw, "cache_transfer_too_large") == null);
    try testing.expectEqual(@as(u64, 0), http_server.cacheTransferReservedBytes());
}

fn sendRdmaHttp(
    io: std.Io,
    node: *LiveNode,
    mgr: *cache_rdma.Manager,
    method: []const u8,
    raw_path: []const u8,
    auth_hdr: []const u8,
    body: []const u8,
) !HttpReply {
    const prev_reg = http_server.global_registry;
    const prev_sch = http_server.launch().global_scheduler;
    http_server.global_registry = node.reg;
    http_server.launch().global_scheduler = node.sch;
    defer {
        http_server.global_registry = prev_reg;
        http_server.launch().global_scheduler = prev_sch;
    }

    var sockets: [2]std.posix.fd_t = undefined;
    try testing.expectEqual(@as(c_int, 0), std.c.socketpair(1, 1, 0, &sockets));
    defer _ = std.c.close(sockets[1]);
    var conn: Conn = undefined;
    Conn.init(&conn, .{ .socket = .{ .handle = sockets[0], .address = .{ .ip4 = .{ .bytes = .{ 127, 0, 0, 1 }, .port = 50000 } } } }, io);
    defer conn.close();

    var hdr_buf: [512]u8 = undefined;
    const hdrs = try std.fmt.bufPrint(&hdr_buf, "{s} {s} HTTP/1.1\r\nAuthorization: {s}\r\n", .{ method, raw_path, auth_hdr });
    const path = if (std.mem.indexOfScalar(u8, raw_path, '?')) |q| raw_path[0..q] else raw_path;
    try cache_rdma.handleRequestOn(mgr, testing.allocator, &conn, method, raw_path, path, hdrs, body);
    try conn.flush();
    try testing.expectEqual(@as(c_int, 0), std.c.shutdown(sockets[0], 1));

    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(testing.allocator);
    var tmp: [4096]u8 = undefined;
    while (true) {
        const n = std.c.recv(sockets[1], &tmp, tmp.len, 0);
        if (n <= 0) break;
        try out.appendSlice(testing.allocator, tmp[0..@intCast(n)]);
    }
    const raw = try out.toOwnedSlice(testing.allocator);
    errdefer testing.allocator.free(raw);
    const sp = std.mem.indexOfScalar(u8, raw, ' ') orelse return error.MalformedResponse;
    const status = try std.fmt.parseInt(u16, raw[sp + 1 .. sp + 4], 10);
    const body_part = (std.mem.cut(u8, raw, "\r\n\r\n") orelse return error.MalformedResponse)[1];
    return .{ .status = status, .raw = raw, .body = body_part };
}

test "dsv41 peer fetch: RDMA offer -> accept -> pull -> delete transfers live KV cache and hits warm B turn" {
    if (!mlx.metalKernelsAvailable() or mlx.noGpuBackend() or mlx.virtualGpu()) return error.SkipZigTest;
    try testing.expectEqual(@as(u64, 0), http_server.cacheTransferReservedBytes());
    var threaded: std.Io.Threaded = .init(testing.allocator, .{});
    defer threaded.deinit();
    const io = threaded.io();

    const prev_ct = http_server.g_cache_transfer;
    defer http_server.g_cache_transfer = prev_ct;
    const transfer_key = "rdma-transfer-key-0123456789abcdef0123";
    var digest: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(transfer_key, &digest, .{});
    http_server.g_cache_transfer = .{ .key = digest };
    const auth_hdr = "Bearer " ++ transfer_key;

    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const files = try fixture.v41ServableFiles(arena.allocator(), 7);
    const loopback: std.Io.net.IpAddress = .{ .ip4 = .{ .bytes = .{ 127, 0, 0, 1 }, .port = 50000 } };

    const turn1_body = "{\"messages\":[{\"role\":\"user\",\"content\":\"Explain cache locality in systems programming.\"}],\"temperature\":0,\"max_tokens\":4,\"stream\":false}";
    const turn2_body = "{\"messages\":[{\"role\":\"user\",\"content\":\"Explain cache locality in systems programming.\"},{\"role\":\"assistant\",\"content\":\"hi\"},{\"role\":\"user\",\"content\":\"Now add one example.\"}],\"temperature\":0,\"max_tokens\":6,\"stream\":false}";

    var fabric = cache_rdma.LoopbackFabric.init(testing.allocator);
    defer fabric.deinit();
    var src_mgr: cache_rdma.Manager = .{ .transport = fabric.transport() };
    defer src_mgr.deinitAll(io);
    var dst_mgr: cache_rdma.Manager = .{ .transport = fabric.transport() };
    defer dst_mgr.deinitAll(io);

    const node_a = try LiveNode.open(io, files);
    defer node_a.close();
    const node_b = try LiveNode.open(io, files);
    defer node_b.close();

    // Seed session on node A with X-Session-Key.
    {
        var t1 = try sendHttp(io, node_a, loopback, "POST /v1/chat/completions HTTP/1.1\r\nContent-Type: application/json\r\nX-Session-Key: rdma-live-sess\r\n", turn1_body);
        defer t1.deinit(testing.allocator);
        try expectStatus("rdma_seed_turn1_200", 200, t1);
    }

    // Step 1: Offer Turn-1 session KV on node A BEFORE running reference Turn 2 on node A.
    var offer_rep = try sendRdmaHttp(io, node_a, &src_mgr, "POST", "/v1/cache/rdma/offer", auth_hdr, "{\"session_key\":\"rdma-live-sess\",\"peer_id\":\"peer-b\",\"rdma_dev\":\"rdma_en2\",\"expires_ms\":30000}");
    defer offer_rep.deinit(testing.allocator);
    try expectStatus("rdma_offer_200", 200, offer_rep);
    const offer_parsed = try std.json.parseFromSlice(std.json.Value, testing.allocator, offer_rep.body, .{});
    defer offer_parsed.deinit();
    const o = offer_parsed.value.object;
    const tid = o.get("transfer_id").?.string;
    const byte_len: u64 = @intCast(o.get("byte_len").?.integer);
    try testing.expect(byte_len > peer_wire.FIXED_HEADER_BYTES);
    try testing.expectEqual(byte_len, http_server.cacheTransferReservedBytes());

    // Reference Turn 2 on node A (restores Turn-1 KV).
    var a_turn2 = try sendHttp(io, node_a, loopback, "POST /v1/chat/completions HTTP/1.1\r\nContent-Type: application/json\r\nX-Session-Key: rdma-live-sess\r\n", turn2_body);
    defer a_turn2.deinit(testing.allocator);
    try expectStatus("rdma_node_a_turn2_200", 200, a_turn2);

    // Step 1.5: Out-of-band QP accept (dst with reset=true, then src with transfer_id).
    const dst_acc_body = try std.fmt.allocPrint(testing.allocator, "{{\"peer_id\":\"peer-a\",\"rdma_dev\":\"rdma_en2\",\"peer_gid\":\"{s}\",\"peer_qpn\":{d},\"peer_psn\":{d},\"reset\":true}}", .{
        o.get("gid").?.string, o.get("qpn").?.integer, o.get("psn").?.integer,
    });
    defer testing.allocator.free(dst_acc_body);
    var dst_acc = try sendRdmaHttp(io, node_b, &dst_mgr, "POST", "/v1/cache/rdma/accept", auth_hdr, dst_acc_body);
    defer dst_acc.deinit(testing.allocator);
    try expectStatus("rdma_dst_accept_200", 200, dst_acc);
    const dst_acc_parsed = try std.json.parseFromSlice(std.json.Value, testing.allocator, dst_acc.body, .{});
    defer dst_acc_parsed.deinit();

    const src_acc_body = try std.fmt.allocPrint(testing.allocator, "{{\"transfer_id\":\"{s}\",\"peer_id\":\"peer-b\",\"rdma_dev\":\"rdma_en2\",\"peer_gid\":\"{s}\",\"peer_qpn\":{d},\"peer_psn\":{d}}}", .{
        tid,
        dst_acc_parsed.value.object.get("gid").?.string,
        dst_acc_parsed.value.object.get("qpn").?.integer,
        dst_acc_parsed.value.object.get("psn").?.integer,
    });
    defer testing.allocator.free(src_acc_body);
    var src_acc = try sendRdmaHttp(io, node_a, &src_mgr, "POST", "/v1/cache/rdma/accept", auth_hdr, src_acc_body);
    defer src_acc.deinit(testing.allocator);
    try expectStatus("rdma_src_accept_200", 200, src_acc);

    // Step 2: Pull on node B -> RDMA transfer + peer_wire decode + HotPrefixCache commit.
    const pull_body = try std.fmt.allocPrint(testing.allocator, "{{\"transfer_id\":\"{s}\",\"session_key\":\"rdma-live-sess\",\"peer_id\":\"peer-a\",\"rdma_dev\":\"rdma_en2\",\"deadline_ms\":30000,\"offer\":{s}}}", .{ tid, offer_rep.body });
    defer testing.allocator.free(pull_body);
    var pull_rep = try sendRdmaHttp(io, node_b, &dst_mgr, "POST", "/v1/cache/rdma/pull", auth_hdr, pull_body);
    defer pull_rep.deinit(testing.allocator);
    try expectStatus("rdma_pull_200", 200, pull_rep);
    try testing.expectEqual(@as(usize, 1), fabric.reads_completed);
    try testing.expectEqual(byte_len, fabric.bytes_transferred);

    // Step 3: Clean unpin on node A releases pinned buffer & export byte budget.
    const del_path = try std.fmt.allocPrint(testing.allocator, "/v1/cache/rdma/offer/{s}?clean=1", .{tid});
    defer testing.allocator.free(del_path);
    var del_rep = try sendRdmaHttp(io, node_a, &src_mgr, "DELETE", del_path, auth_hdr, "");
    defer del_rep.deinit(testing.allocator);
    try expectStatus("rdma_delete_200", 200, del_rep);
    try testing.expectEqual(@as(u64, 0), http_server.cacheTransferReservedBytes());

    // Turn 2 on node B hits the RDMA-imported prefix cache with bit-exact greedy output.
    var b_turn2 = try sendHttp(io, node_b, loopback, "POST /v1/chat/completions HTTP/1.1\r\nContent-Type: application/json\r\nX-Session-Key: rdma-live-sess\r\n", turn2_body);
    defer b_turn2.deinit(testing.allocator);
    try expectStatus("rdma_node_b_turn2_200", 200, b_turn2);

    const a_parsed = try std.json.parseFromSlice(std.json.Value, testing.allocator, a_turn2.body, .{});
    defer a_parsed.deinit();
    const b_parsed = try std.json.parseFromSlice(std.json.Value, testing.allocator, b_turn2.body, .{});
    defer b_parsed.deinit();
    const a_cached = a_parsed.value.object.get("usage").?.object.get("prompt_tokens_details").?.object.get("cached_tokens").?.integer;
    const b_cached = b_parsed.value.object.get("usage").?.object.get("prompt_tokens_details").?.object.get("cached_tokens").?.integer;
    try testing.expect(a_cached > 0);
    try testing.expectEqual(a_cached, b_cached);

    const a_msg = a_parsed.value.object.get("choices").?.array.items[0].object.get("message").?.object;
    const b_msg = b_parsed.value.object.get("choices").?.array.items[0].object.get("message").?.object;
    const a_content = if (a_msg.get("content")) |v| (if (v == .string) v.string else "") else "";
    const b_content = if (b_msg.get("content")) |v| (if (v == .string) v.string else "") else "";
    try testing.expectEqualStrings(a_content, b_content);

    // Quiesce tears down all QPs cleanly on both nodes.
    var qa = try sendRdmaHttp(io, node_a, &src_mgr, "POST", "/v1/cache/rdma/quiesce", auth_hdr, "{}");
    defer qa.deinit(testing.allocator);
    try expectStatus("rdma_quiesce_a_200", 200, qa);
    var qb = try sendRdmaHttp(io, node_b, &dst_mgr, "POST", "/v1/cache/rdma/quiesce", auth_hdr, "{}");
    defer qb.deinit(testing.allocator);
    try expectStatus("rdma_quiesce_b_200", 200, qb);
    try testing.expectEqual(@as(usize, 0), src_mgr.activeQpCount(io));
    try testing.expectEqual(@as(usize, 0), dst_mgr.activeQpCount(io));
}
