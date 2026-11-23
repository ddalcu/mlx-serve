//! Grammar rows in a batched decode tick: a constrained slot draws under its own token mask, built on the CPU while
//! the forward that produced the rows runs on the GPU; every other row samples exactly as it did before.

const std = @import("std");
const mlx = @import("../../../mlx/ffi.zig");
const generate_mod = @import("../generate.zig");
const log = @import("../../../sdk/util/log.zig");
const Generator = @import("../generate/generator.zig").Generator;
const applyGrammarMask = @import("../sampling/sampler.zig").applyGrammarMask;
const MAX_BATCH_GROUP = @import("verdict.zig").MAX_BATCH_GROUP;

var engaged_logged = false;

/// One token per row into `out`: row i is `rows[i]`, drawn with `gens[i]`'s sampling (its `draw` advances). A row
/// whose generator carries a grammar draws under that grammar's mask for this position.
pub fn sampleBatchRows(allocator: std.mem.Allocator, out: []i32, gens: []const *Generator, rows: []const mlx.mlx_array, s: mlx.mlx_stream) !void {
    const n = rows.len;
    std.debug.assert(out.len == n and gens.len == n and n <= MAX_BATCH_GROUP);
    var params: [MAX_BATCH_GROUP]generate_mod.SamplingParams = undefined;
    var drawn: [MAX_BATCH_GROUP]mlx.mlx_array = undefined;
    var masked: [MAX_BATCH_GROUP]mlx.mlx_array = undefined;
    var masked_n: usize = 0;
    defer for (masked[0..masked_n]) |m| {
        _ = mlx.mlx_array_free(m);
    };
    var grammar_n: usize = 0;
    for (gens) |g| grammar_n += @intFromBool(g.sampling.constraint != null);
    if (grammar_n > 0) {
        // Start the forward now: the masks below are built on the CPU while it runs.
        const vec = mlx.mlx_vector_array_new_data(rows.ptr, n);
        defer _ = mlx.mlx_vector_array_free(vec);
        try mlx.check(mlx.mlx_async_eval(vec));
        // Output equality alone cannot tell a batched grammar row from a serial one: the guards grep this marker.
        if (!engaged_logged) {
            engaged_logged = true;
            log.info("[batched] grammar rows engaged (slots={d}, grammar={d})\n", .{ n, grammar_n });
        }
    }
    for (gens, rows, 0..) |g, row, i| {
        params[i] = g.sampling;
        drawn[i] = row;
        const c = g.sampling.constraint orelse continue;
        try c.fillPayloadMask(g.eos_token_ids);
        masked[masked_n] = mlx.mlx_array_new();
        masked_n += 1;
        try applyGrammarMask(allocator, &masked[masked_n - 1], row, c.mask_buf, s);
        drawn[i] = masked[masked_n - 1];
    }
    try generate_mod.sampleRows(out, drawn[0..n], params[0..n], s);
    for (gens, params[0..n]) |g, p| g.sampling.draw = p.draw;
}

// ── tests ──

const testing = std.testing;
const model_mod = @import("../../../sdk/config/model.zig");
const plugin_driver = @import("../plugin/driver.zig");
const loadTokenizer = @import("../../../sdk/tokenizer/load.zig").loadTokenizer;
const Tokenizer = @import("../../../sdk/tokenizer/tokenizer.zig").Tokenizer;
const token_mask = @import("../../../sdk/grammar/token_mask.zig");
const json_grammar = @import("../../../sdk/grammar/json_grammar.zig");
const json_schema = @import("../../../sdk/grammar/json_schema.zig");
const SchemaConstraint = @import("../../../sdk/grammar/constraint.zig").SchemaConstraint;
const tiny_qwen35 = @import("../../../sdk/testing/tiny_qwen35.zig");
const tensors_mod = @import("../../../sdk/testing/tensors.zig");
const byte_tokenizer = @import("../../../sdk/testing/byte_tokenizer.zig");
const proof_fixture = @import("../../../sdk/bases/proof/fixture.zig");
const Transformer = @import("../transformer.zig").Transformer;
const ForwardCtx = @import("../forward_ctx.zig").ForwardCtx;
const KVCache = @import("../../../sdk/kv/cache.zig").KVCache;
const SSMCacheEntry = @import("../ssm/entry.zig").SSMCacheEntry;
const Slot = @import("../slot.zig").Slot;
const Scheduler = @import("../scheduler.zig").Scheduler;
const runDecodeTick = @import("../decode.zig").runDecodeTick;
const LoadedModel = @import("../../../host/registry/registry.zig").LoadedModel;

const EOS = [_]u32{ 2, 3 };
const OBJECT_SCHEMA =
    \\{"type":"object","properties":{"a":{"type":"string","enum":["xy","zw"]},"b":{"type":"boolean"}},"required":["a","b"],"additionalProperties":false}
;
const ENUM_SCHEMA =
    \\{"type":"string","enum":["alpha","beta","gamma"]}
;

/// A tiny random-weight pack on disk, served by its plugin: an attention-only qwen3 trunk (the plain batched kernel)
/// or tiny_qwen35's GatedDeltaNet trunk (the GDN one), over a byte tokenizer.
const Model = struct {
    tmp: testing.TmpDir,
    dir: []u8,
    cfg: model_mod.ModelConfig,
    xfm: Transformer,
    tok: Tokenizer,
    token_bytes: token_mask.TokenBytes,
    gdn: bool,

    const QWEN3 =
        \\{"model_type":"qwen3","hidden_size":128,"intermediate_size":256,"num_hidden_layers":2,"num_attention_heads":4,
        \\ "num_key_value_heads":2,"head_dim":32,"vocab_size":264,"rms_norm_eps":1e-6,"rope_theta":1000000.0,
        \\ "max_position_embeddings":4096,"hidden_act":"silu","eos_token_id":[2,3],"tie_word_embeddings":false}
    ;

    fn qwen3Weights(t: *tensors_mod.Tensors) !void {
        try t.rand("model.embed_tokens.weight", .{}, &.{ 264, 128 }, .BF16, 0.5);
        try t.rand("model.norm.weight", .{}, &.{128}, .BF16, 0.1);
        try t.rand("lm_head.weight", .{}, &.{ 264, 128 }, .BF16, 0.1);
        for (0..2) |i| {
            const p = "model.layers.{d}.";
            try t.rand(p ++ "input_layernorm.weight", .{i}, &.{128}, .BF16, 0.1);
            try t.rand(p ++ "post_attention_layernorm.weight", .{i}, &.{128}, .BF16, 0.1);
            try t.rand(p ++ "self_attn.q_proj.weight", .{i}, &.{ 128, 128 }, .BF16, 0.06);
            try t.rand(p ++ "self_attn.k_proj.weight", .{i}, &.{ 64, 128 }, .BF16, 0.06);
            try t.rand(p ++ "self_attn.v_proj.weight", .{i}, &.{ 64, 128 }, .BF16, 0.06);
            try t.rand(p ++ "self_attn.o_proj.weight", .{i}, &.{ 128, 128 }, .BF16, 0.03);
            try t.rand(p ++ "self_attn.q_norm.weight", .{i}, &.{32}, .BF16, 0.1);
            try t.rand(p ++ "self_attn.k_norm.weight", .{i}, &.{32}, .BF16, 0.1);
            try t.rand(p ++ "mlp.gate_proj.weight", .{i}, &.{ 256, 128 }, .BF16, 0.06);
            try t.rand(p ++ "mlp.up_proj.weight", .{i}, &.{ 256, 128 }, .BF16, 0.06);
            try t.rand(p ++ "mlp.down_proj.weight", .{i}, &.{ 128, 256 }, .BF16, 0.04);
        }
    }

    fn open(gdn: bool) !*Model {
        const a = testing.allocator;
        const io = std.Io.Threaded.global_single_threaded.io();
        var arena = std.heap.ArenaAllocator.init(a);
        defer arena.deinit();
        var t = tensors_mod.Tensors.init(arena.allocator(), 0x6a73);
        if (gdn) try tiny_qwen35.weights(&t) else try qwen3Weights(&t);
        const m = try a.create(Model);
        errdefer a.destroy(m);
        m.tmp = testing.tmpDir(.{});
        errdefer m.tmp.cleanup();
        try m.tmp.dir.writeFile(io, .{ .sub_path = "config.json", .data = if (gdn) tiny_qwen35.config else QWEN3 });
        try m.tmp.dir.writeFile(io, .{ .sub_path = "tokenizer.json", .data = try byte_tokenizer.json(arena.allocator(), &.{}) });
        try m.tmp.dir.writeFile(io, .{ .sub_path = "tokenizer_config.json", .data = "{}" });
        try m.tmp.dir.writeFile(io, .{ .sub_path = "model.safetensors", .data = try t.bytes() });
        var path_buf: [512]u8 = undefined;
        m.dir = try a.dupe(u8, path_buf[0..try m.tmp.dir.realPath(io, &path_buf)]);
        errdefer a.free(m.dir);
        m.cfg = try model_mod.parseConfig(io, a, m.dir);
        errdefer m.cfg.deinit(a);
        m.xfm = try plugin_driver.load(io, a, &m.cfg, m.dir, mlx.gpuStream());
        errdefer m.xfm.deinit();
        m.tok = try loadTokenizer(io, a, m.dir);
        errdefer m.tok.deinit();
        m.token_bytes = try token_mask.build(a, &m.tok);
        m.gdn = gdn;
        try testing.expectEqual(gdn, m.xfm.supportsBatchedGdnDecode());
        return m;
    }

    fn close(m: *Model) void {
        const a = testing.allocator;
        m.token_bytes.deinit();
        m.tok.deinit();
        m.xfm.deinit();
        m.cfg.deinit(a);
        a.free(m.dir);
        m.tmp.cleanup();
        a.destroy(m);
    }
};

/// One request on `Model`: its own caches and generator, and what it emitted.
const Row = struct {
    m: *Model,
    cache: KVCache,
    off: usize = 0,
    entries: ?[]SSMCacheEntry,
    sc: SchemaConstraint = undefined,
    schema: ?[]const u8,
    gen: Generator = undefined,
    out: std.ArrayList(u32) = .empty,
    prompt_len: usize,
    ended: bool = false,

    fn create(m: *Model, schema: ?[]const u8, sampling: generate_mod.SamplingParams) !*Row {
        const a = testing.allocator;
        const io = std.Io.Threaded.global_single_threaded.io();
        const prompt = [_]u32{ 'j', 's', 'o', 'n', '?', ' ' };
        const r = try a.create(Row);
        errdefer a.destroy(r);
        r.* = .{ .m = m, .cache = try KVCache.init(a, m.cfg.num_hidden_layers), .entries = if (m.gdn) try proof_fixture.newEntries(m.cfg.num_hidden_layers) else null, .schema = schema, .prompt_len = prompt.len };
        var sp = sampling;
        if (schema) |src| {
            var parsed = try std.json.parseFromSlice(std.json.Value, a, src, .{});
            defer parsed.deinit();
            try r.sc.initFromValue(a, parsed.value, &m.token_bytes);
            sp.constraint = &r.sc.constraint;
        }
        const ctx: ForwardCtx = .{ .cache = &r.cache, .moe_seq_offset = &r.off, .ssm_entries = r.entries, .capture_hidden = null, .vision_embeddings = null };
        r.gen = try Generator.initWithOptions(io, a, &m.xfm, &m.tok, &prompt, 64, sp, &EOS, .{ .ctx = ctx });
        return r;
    }

    fn destroy(r: *Row) void {
        const a = testing.allocator;
        r.m.xfm.ssmGroupRelease(&r.gen.ctx) catch {};
        r.gen.deinit(a);
        if (r.schema != null) r.sc.deinit();
        r.out.deinit(a);
        if (r.entries) |es| proof_fixture.freeEntries(es);
        r.cache.deinit();
        a.destroy(r);
    }

    /// Positions in the cache: the prompt, every emitted token, and a pipelined lookahead.
    fn expectPositions(r: *const Row) !void {
        const pos = if (r.m.gdn) r.off else r.cache.step;
        try testing.expectEqual(r.prompt_len + r.out.items.len + @intFromBool(r.gen.has_pending_token), pos);
    }

    /// One serial decode step, as `runSingleDecodeTick` takes it: out of the batch's SSM group first.
    fn serialStep(r: *Row) !void {
        if (r.ended) return;
        try r.m.xfm.ssmGroupRelease(&r.gen.ctx);
        if (try r.gen.next(testing.allocator)) |t| try r.out.append(testing.allocator, t) else r.ended = true;
    }

    /// The emitted bytes are a prefix of a document the row's schema accepts.
    fn expectValid(r: *const Row) !void {
        const src = r.schema orelse return;
        const a = testing.allocator;
        var parsed = try std.json.parseFromSlice(std.json.Value, a, src, .{});
        defer parsed.deinit();
        var schema = try json_schema.parse(a, parsed.value);
        defer schema.deinit();
        var g = try json_grammar.Grammar.init(a, &schema);
        defer g.deinit();
        for (r.out.items) |t| {
            const bytes = r.m.token_bytes.bytes[t] orelse return error.NonByteToken;
            for (bytes) |b| if (!try g.acceptByte(b)) return error.SchemaViolation;
        }
        try testing.expect(!r.sc.grammar.dead);
    }
};

fn hostRow(row: mlx.mlx_array, out: []f32) !void {
    var f = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(f);
    try mlx.check(mlx.mlx_astype(&f, row, .float32, mlx.gpuStream()));
    try mlx.check(mlx.mlx_array_eval(f));
    const data = mlx.mlx_array_data_float32(f) orelse return error.MlxError;
    @memcpy(out, data[0..out.len]);
}

fn argmaxWhere(logits: []const f32, mask: ?[]const bool) ?i32 {
    var best: ?usize = null;
    for (logits, 0..) |v, i| {
        if (mask) |m| if (i >= m.len or !m[i]) continue;
        if (best == null or v > logits[best.?]) best = i;
    }
    return if (best) |b| @intCast(b) else null;
}

/// One batched decode tick over `rows`, as `runBatchedDecodeTick` runs it: join, one forward, one draw per row, emit
/// the previous token, stop or advance. A greedy row's draw must be the argmax of its OWN logits under its OWN mask.
/// A mirror of `runBatchedDecodeTick`, not tick.zig itself: tick.zig's own order (the payload span published before
/// the emit, `noteBatchedSample` after every stop check) is guarded only by tests/test_batched_grammar.sh on a live model.
fn batchedTick(m: *Model, rows: []const *Row) !void {
    const a = testing.allocator;
    var live: [8]*Row = undefined;
    var n: usize = 0;
    for (rows) |r| {
        if (r.ended) continue;
        switch (try r.gen.joinBatch(a)) {
            .stopped => {
                r.ended = true;
                continue;
            },
            .joined => |emitted| if (emitted) |t| {
                try r.out.append(a, t);
                if (generate_mod.isEosId(r.gen.next_token_id, &EOS)) {
                    r.ended = true;
                    continue;
                }
            },
        }
        live[n] = r;
        n += 1;
    }
    if (n == 0) return;
    var next: [8]u32 = undefined;
    var ctxs: [8]*ForwardCtx = undefined;
    var offs: [8]u32 = undefined;
    var gens: [8]*Generator = undefined;
    for (live[0..n], 0..) |r, i| {
        next[i] = r.gen.next_token_id;
        ctxs[i] = &r.gen.ctx;
        offs[i] = @intCast(if (m.gdn) r.off else r.cache.step);
        gens[i] = &r.gen;
    }
    const rows_out = if (m.gdn)
        try m.xfm.forwardMoeBatchedDecode(next[0..n], ctxs[0..n], offs[0..n], null)
    else
        try m.xfm.forwardBatchedDecode(next[0..n], ctxs[0..n], offs[0..n]);
    defer {
        for (rows_out) |r| _ = mlx.mlx_array_free(r);
        a.free(rows_out);
    }
    if (m.gdn) for (live[0..n]) |r| {
        r.off += 1;
    };
    // The oracle: each row's own logits, and the mask its own grammar state allows here.
    var host: [8][264]f32 = undefined;
    var want_mask: [8][264]bool = undefined;
    for (live[0..n], rows_out, 0..) |r, row, i| {
        try hostRow(row, &host[i]);
        const c = r.gen.sampling.constraint orelse continue;
        const buf = want_mask[i][0..c.mask_buf.len];
        _ = try token_mask.buildMask(c.grammar, c.token_bytes, buf);
        if (c.grammar.isComplete()) for (EOS) |e| {
            buf[e] = true;
        };
    }
    var ids: [8]i32 = undefined;
    try sampleBatchRows(a, ids[0..n], gens[0..n], rows_out, m.xfm.s);
    for (live[0..n], 0..) |r, i| {
        if (r.gen.sampling.temperature >= 0.01) continue;
        const mask: ?[]const bool = if (r.gen.sampling.constraint) |c| want_mask[i][0..c.mask_buf.len] else null;
        try testing.expectEqual(argmaxWhere(&host[i], mask), ids[i]);
    }
    for (live[0..n], 0..) |r, i| {
        const emit = r.gen.next_token_id;
        try r.gen.generated_ids.append(a, emit);
        r.gen.advanceStep(1);
        try r.out.append(a, emit);
        r.gen.next_token_id = @intCast(ids[i]);
        if (generate_mod.isEosId(r.gen.next_token_id, &EOS) or r.out.items.len >= 64) {
            r.ended = true;
            continue;
        }
        try r.gen.noteBatchedSample(r.gen.next_token_id);
    }
}

fn eachPack(comptime body: fn (*Model) anyerror!void) !void {
    if (mlx.noGpuBackend() or mlx.virtualGpu()) return error.SkipZigTest;
    for ([_]bool{ false, true }) |gdn| {
        const m = try Model.open(gdn);
        defer m.close();
        try body(m);
    }
}

test "batched grammar rows: each draws under its own schema's mask, the plain row unmasked (attention and GDN)" {
    try eachPack(struct {
        fn run(m: *Model) !void {
            const rows = [_]*Row{ try Row.create(m, OBJECT_SCHEMA, .{ .temperature = 0 }), try Row.create(m, null, .{ .temperature = 0 }), try Row.create(m, ENUM_SCHEMA, .{ .temperature = 0 }) };
            defer for (rows) |r| r.destroy();
            for (0..16) |_| try batchedTick(m, &rows);
            for (rows) |r| try r.expectValid();
            try testing.expect(rows[0].out.items.len > 4 and rows[2].out.items.len > 4);
        }
    }.run);
}

test "a grammar row keeps its stream across joins and leaves: serial, batched, serial, batched again" {
    try eachPack(struct {
        fn run(m: *Model) !void {
            const a_ = try Row.create(m, OBJECT_SCHEMA, .{ .temperature = 0 });
            const b_ = try Row.create(m, null, .{ .temperature = 0 });
            const c_ = try Row.create(m, ENUM_SCHEMA, .{ .temperature = 0.9, .seed = 7 });
            defer for ([_]*Row{ a_, b_, c_ }) |r| r.destroy();
            for (0..3) |_| try a_.serialStep();
            try a_.expectPositions();
            for (0..3) |_| try batchedTick(m, &.{ a_, b_ });
            for (0..2) |_| try batchedTick(m, &.{ a_, b_, c_ });
            for ([_]*Row{ a_, b_, c_ }) |r| try r.expectPositions();
            for (0..3) |_| for ([_]*Row{ a_, c_ }) |r| try r.serialStep();
            try a_.expectPositions();
            try c_.expectPositions();
            for (0..3) |_| try batchedTick(m, &.{ a_, c_ });
            for ([_]*Row{ a_, c_ }) |r| {
                try r.expectPositions();
                try r.expectValid();
                try testing.expect(!r.ended or (r.gen.finish_reason.len > 0 and r.sc.grammar.isComplete()));
            }
            try testing.expect(a_.out.items.len >= 10);
        }
    }.run);
}

test "a grammar request batched alone draws the same tokens as the serial path (greedy and seeded)" {
    try eachPack(struct {
        fn run(m: *Model) !void {
            for ([_]generate_mod.SamplingParams{ .{ .temperature = 0 }, .{ .temperature = 0.9, .seed = 11 } }) |sp| {
                const serial = try Row.create(m, OBJECT_SCHEMA, sp);
                defer serial.destroy();
                const batched = try Row.create(m, OBJECT_SCHEMA, sp);
                defer batched.destroy();
                for (0..24) |_| try serial.serialStep();
                for (0..24) |_| try batchedTick(m, &.{batched});
                const n = @min(serial.out.items.len, batched.out.items.len);
                try testing.expect(n > 4);
                try testing.expectEqualSlices(u32, serial.out.items[0..n], batched.out.items[0..n]);
            }
        }
    }.run);
}

test "the serial constrained step: the mask of the grammar before the token, one seeded draw, the next logits pending, a stop that never reaches the stream" {
    try eachPack(struct {
        fn run(m: *Model) !void {
            const r = try Row.create(m, ENUM_SCHEMA, .{ .temperature = 0 });
            defer r.destroy();
            const c = r.gen.sampling.constraint.?;
            var host: [264]f32 = undefined;
            var want: [264]bool = undefined;
            for (0..40) |_| {
                if (r.ended) break;
                try testing.expect(r.gen.has_pending_logits);
                try hostRow(r.gen.pending_logits, &host);
                const mask = want[0..c.mask_buf.len];
                _ = try token_mask.buildMask(c.grammar, c.token_bytes, mask);
                if (c.grammar.isComplete()) for (EOS) |e| {
                    mask[e] = true;
                };
                const draw0 = r.gen.sampling.draw;
                const len0 = r.gen.generated_ids.items.len;
                try r.serialStep();
                try testing.expectEqual(draw0 +% 1, r.gen.sampling.draw);
                const want_id = argmaxWhere(&host, mask);
                if (r.ended) {
                    try testing.expectEqualStrings("stop", r.gen.finish_reason);
                    try testing.expect(generate_mod.isEosId(r.gen.next_token_id, &EOS));
                    try testing.expectEqual(want_id, @as(i32, @intCast(r.gen.next_token_id)));
                    try testing.expectEqual(len0, r.gen.generated_ids.items.len);
                    try testing.expect(!r.sc.grammar.dead);
                } else {
                    try testing.expectEqual(want_id, @as(i32, @intCast(r.out.items[r.out.items.len - 1])));
                    try testing.expect(r.gen.has_pending_logits);
                    try testing.expectEqual(len0 + 1, r.gen.generated_ids.items.len);
                    try r.expectPositions();
                }
            }
            try testing.expect(r.ended);
            try testing.expectEqualStrings("stop", r.gen.finish_reason);
            try r.expectValid();
        }
    }.run);
}

/// Fails its first allocation, then serves from `testing.allocator`: the failure is one the slot can still name.
const FailsOnce = struct {
    failed: bool = false,

    fn allocator(self: *FailsOnce) std.mem.Allocator {
        return .{ .ptr = self, .vtable = &.{ .alloc = alloc, .resize = resize, .remap = remap, .free = free } };
    }
    fn alloc(ctx: *anyopaque, len: usize, a: std.mem.Alignment, ra: usize) ?[*]u8 {
        const self: *FailsOnce = @ptrCast(@alignCast(ctx));
        if (!self.failed) {
            self.failed = true;
            return null;
        }
        return testing.allocator.rawAlloc(len, a, ra);
    }
    fn resize(_: *anyopaque, mem: []u8, a: std.mem.Alignment, n: usize, ra: usize) bool {
        return testing.allocator.rawResize(mem, a, n, ra);
    }
    fn remap(_: *anyopaque, mem: []u8, a: std.mem.Alignment, n: usize, ra: usize) ?[*]u8 {
        return testing.allocator.rawRemap(mem, a, n, ra);
    }
    fn free(_: *anyopaque, mem: []u8, a: std.mem.Alignment, ra: usize) void {
        testing.allocator.rawFree(mem, a, ra);
    }
};

/// A scheduler slot over `r`: the row's generator moved in, its cache position mirrored where the batched tick reads it.
fn slotOver(r: *Row, lm: *LoadedModel, allocator: ?std.mem.Allocator) !*Slot {
    const slot = try Slot.init(testing.allocator, std.Io.Threaded.global_single_threaded.io(), &r.m.cfg, .{
        .model = lm,
        .prompt_ids = &.{ 'j', 's', 'o', 'n', '?', ' ' },
        .sampling = r.gen.sampling,
        .eos_token_ids = &EOS,
        .max_tokens = 64,
    }, .dense);
    if (allocator) |a| slot.allocator = a;
    slot.legacy_gen = r.gen;
    slot.cache.step = r.cache.step;
    slot.moe_seq_offset = r.off;
    return slot;
}

/// The generator goes back to its row, which frees it.
fn releaseSlot(slot: *Slot, r: *Row) void {
    r.gen = slot.legacy_gen.?;
    slot.legacy_gen = null;
    slot.deinit();
}

test "runDecodeTick: a grammar row whose guard fails is failed alone; its neighbour still decodes (attention and GDN)" {
    try eachPack(struct {
        fn run(m: *Model) !void {
            var lm: LoadedModel = undefined;
            lm.engine = null;
            lm.transformer = &m.xfm;
            lm.config = &m.cfg;
            lm.tokenizer = &m.tok;
            lm.prefix_cache = null;
            var sch: Scheduler = undefined;
            sch.allocator = testing.allocator;
            sch.io = std.Io.Threaded.global_single_threaded.io();
            sch.force_batched = true;
            sch.metrics = null;
            sch.inflight_generated_tokens = std.atomic.Value(u64).init(0);

            // The reference: B alone through the same entry.
            const ref_row = try Row.create(m, ENUM_SCHEMA, .{ .temperature = 0 });
            defer ref_row.destroy();
            const ref_slot = try slotOver(ref_row, &lm, null);
            defer releaseSlot(ref_slot, ref_row);
            var ref_active = [_]*Slot{ref_slot};
            try runDecodeTick(&sch, &ref_active);
            try testing.expect(ref_slot.state != .errored);
            try testing.expectEqual(@as(usize, 1), ref_slot.out_buf.items.len);

            // A's thinking budget is due at once; committing the closer through a failing allocator errors its guard.
            var bound: generate_mod.ThinkBound = .{ .budget = 0, .opener_id = null, .closer_id = 1, .forced = &.{'z'}, .in_think = true };
            const a_row = try Row.create(m, OBJECT_SCHEMA, .{ .temperature = 0, .think_bound = &bound });
            defer a_row.destroy();
            const b_row = try Row.create(m, ENUM_SCHEMA, .{ .temperature = 0 });
            defer b_row.destroy();
            var fails = FailsOnce{};
            const a_slot = try slotOver(a_row, &lm, fails.allocator());
            defer releaseSlot(a_slot, a_row);
            const b_slot = try slotOver(b_row, &lm, null);
            defer releaseSlot(b_slot, b_row);
            var active = [_]*Slot{ a_slot, b_slot };
            try runDecodeTick(&sch, &active);

            // The failure is the guard's: A's budget fired before the batch, not an allocation in the tick.
            try testing.expect(fails.failed and bound.fired);
            try testing.expectEqual(.errored, a_slot.state);
            try testing.expectEqualStrings("OutOfMemory", a_slot.error_code.?);
            try testing.expectEqual(@as(usize, 0), a_slot.out_buf.items.len);
            try testing.expect(b_slot.state != .errored);
            try testing.expectEqualSlices(u32, ref_slot.out_buf.items, b_slot.out_buf.items);
        }
    }.run);
}
