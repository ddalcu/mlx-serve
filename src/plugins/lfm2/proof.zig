//! Test harness for the hybrid plugins' parity proofs: one tiny random pack on disk served twice, by the legacy
//! `Transformer` and by a plugin's table (parse, the engine's loader into `LoadCtx.weights`, `load`, `forward` over
//! the engine's `cache_ops` with recurrent entries), each step's logits bit-equal. Shared by lfm2 and nemotron_h.

const std = @import("std");
const abi = @import("../../abi/plugin.zig");
const mlx = @import("../../mlx/ffi.zig");
const ops = @import("../../mlx/ops.zig");
const model = @import("../../sdk/config/model.zig");
const cache_ops = @import("../../engines/mlx/kv/cache_ops.zig");
const KVCache = @import("../../engines/mlx/kv/cache.zig").KVCache;
const fixture = @import("../../sdk/bases/proof/fixture.zig");
const Resident = @import("../../sdk/weights/resident.zig").Resident;
const FsPackDir = @import("../../host/registry/pack_dir.zig").FsPackDir;
const both = @import("../../engines/mlx/plugin/both.zig");
const driver = @import("../../engines/mlx/plugin/driver.zig");
const tape_mod = @import("../../engines/mlx/plugin/tape.zig");
const mlx_gguf = @import("../../sdk/quant/gguf/claim.zig");

const testing = std.testing;
const Array = mlx.mlx_array;

pub const Step = fixture.Step;
pub const steps = fixture.steps;

fn tableStep(comptime P: type, m: *anyopaque, g: *ops.Graph, slot: *cache_ops.Slot, mode: abi.Mode, ids: []const u32, pos: u32, row_exact: bool) !mlx.mlx_array {
    const handles = [_]abi.Handle{slot.handle()};
    var out: abi.ForwardOut = .{};
    const v = P.table.forward(m, &.{
        .mode = mode,
        .logits = .all_rows,
        .rows = @intCast(ids.len),
        .tokens = .of(ids),
        .positions = .of(&.{pos}),
        .caches = .of(&handles),
        .cache_ops = &cache_ops.ops,
        .row_exact = row_exact,
    }, &out);
    if (!v.isOk()) {
        std.debug.print("forward: {s}\n", .{v.why.slice()});
        return error.PluginForward;
    }
    if (out.hidden.ctx != null) _ = mlx.mlx_array_free(.{ .ctx = out.hidden.ctx });
    return g.adopt(.{ .ctx = out.logits.ctx });
}

/// As `expectVerifyEqualsSerial`, on a loaded model `m` of `n_layers` layers, on stream `s`.
pub fn expectVerifyEqualsSerialOn(comptime P: type, m: *anyopaque, n_layers: u32, s: mlx.mlx_stream, prompt: []const u32, drafts: []const u32) !void {
    const a = testing.allocator;
    var caches: [2]KVCache = undefined;
    var entries: [2][]@import("../../engines/mlx/ssm/entry.zig").SSMCacheEntry = undefined;
    var slots: [2]cache_ops.Slot = undefined;
    for (&caches, &entries, &slots) |*kc, *e, *sl| {
        kc.* = try KVCache.init(a, n_layers);
        e.* = try fixture.newEntries(n_layers);
        sl.* = .{ .kv = kc, .ssm = e.* };
    }
    defer for (&caches, entries) |*kc, e| {
        kc.deinit();
        fixture.freeEntries(e);
    };
    var g = ops.Graph.init(a, s);
    defer g.deinit();
    const p: u32 = @intCast(prompt.len);
    for (&slots) |*sl| _ = try tableStep(P, m, &g, sl, .prefill, prompt, 0, true);
    const verify = try tableStep(P, m, &g, &slots[1], .verify, drafts, p, true);
    for (drafts, 0..) |d, i| {
        const want = try tableStep(P, m, &g, &slots[0], .decode, &.{d}, p + @as(u32, @intCast(i)), true);
        const row = try g.slice(verify, &.{ @intCast(i), 0 }, &.{ @intCast(i + 1), ops.shape(verify)[1] });
        try ops.expectBitEqual(row, want);
    }
}

/// Plugin `P`'s last-row logits on a fresh slot as f32 on the host: one prefill of `prompt`, then one decode step per
/// id of `decode`, the rows in step order. Caller frees (testing allocator).
pub fn hostLogits(comptime P: type, m: *anyopaque, n_layers: u32, s: mlx.mlx_stream, prompt: []const u32, decode: []const u32) ![]f32 {
    const a = testing.allocator;
    var kc = try KVCache.init(a, n_layers);
    defer kc.deinit();
    const e = try fixture.newEntries(n_layers);
    defer fixture.freeEntries(e);
    var slot: cache_ops.Slot = .{ .kv = &kc, .ssm = e };
    var g = ops.Graph.init(a, s);
    defer g.deinit();
    var rows: std.ArrayList(Array) = .empty;
    defer rows.deinit(a);
    const pre = try tableStep(P, m, &g, &slot, .prefill, prompt, 0, false);
    const sh = ops.shape(pre);
    try rows.append(a, try g.slice(pre, &.{ sh[0] - 1, 0 }, &.{ sh[0], sh[1] }));
    for (decode, 0..) |id, i| try rows.append(a, try tableStep(P, m, &g, &slot, .decode, &.{id}, @intCast(prompt.len + i), false));
    const all = try g.astype(try g.concat(rows.items, 0), .float32);
    try mlx.check(mlx.mlx_array_eval(all));
    const d = mlx.mlx_array_data_float32(all) orelse return error.NoData;
    return a.dupe(f32, d[0..mlx.mlx_array_size(all)]);
}

pub const fixture_file = struct { name: []const u8, bytes: []const u8 };

/// The legacy reading of `json` (`parseConfigFromJson`), freed by the caller.
pub fn legacyConfig(json: []const u8) !model.ModelConfig {
    return model.parseConfigFromJson(testing.allocator, json);
}

/// `parse`'s shell against the legacy `ModelConfig`'s generic fields.
pub fn expectShell(s: *const abi.Shell, c: *const model.ModelConfig) !void {
    try testing.expectEqual(c.vocab_size, s.vocab_size);
    try testing.expectEqual(c.hidden_size, s.hidden_size);
    try testing.expectEqual(c.intermediate_size, s.intermediate_size);
    try testing.expectEqual(c.num_hidden_layers, s.num_hidden_layers);
    try testing.expectEqual(c.num_attention_heads, s.num_attention_heads);
    try testing.expectEqual(c.num_key_value_heads, s.num_key_value_heads);
    try testing.expectEqual(c.head_dim, s.head_dim);
    try testing.expectEqual(@as(u64, c.max_position_embeddings), s.max_position_embeddings);
    try testing.expectEqual(c.quant_bits, s.quant.bits);
    if (c.quant_bits > 0) try testing.expectEqual(c.quant_group_size, s.quant.group_size);
    try testing.expectEqual(c.bos_token_id, if (s.bos >= 0) @as(?u32, @intCast(s.bos)) else null);
    try testing.expectEqualSlices(u32, c.eosTokenSlice(), s.eos[0..s.eos_count]);
}

/// A GGUF file built in memory as llama.cpp writes one: metadata, the tensor directory (ggml dims innermost first),
/// then the tensors' bytes, each aligned to 32.
pub const Gguf = struct {
    kv: std.ArrayList(u8) = .empty,
    info: std.ArrayList(u8) = .empty,
    data: std.ArrayList(u8) = .empty,
    n_kv: u64 = 0,
    n_tensors: u64 = 0,

    const a = testing.allocator;
    const Type = enum(u32) { u32 = 4, f32 = 6, bool = 7, string = 8, array = 9 };

    pub fn deinit(g: *Gguf) void {
        g.kv.deinit(a);
        g.info.deinit(a);
        g.data.deinit(a);
    }

    fn num(list: *std.ArrayList(u8), v: anytype) !void {
        try list.appendSlice(a, std.mem.asBytes(&v));
    }

    fn str(list: *std.ArrayList(u8), s: []const u8) !void {
        try num(list, @as(u64, s.len));
        try list.appendSlice(a, s);
    }

    fn key(g: *Gguf, k: []const u8, t: Type) !void {
        try str(&g.kv, k);
        try num(&g.kv, @backingInt(t));
        g.n_kv += 1;
    }

    pub fn int(g: *Gguf, k: []const u8, v: u32) !void {
        try g.key(k, .u32);
        try num(&g.kv, v);
    }

    pub fn float(g: *Gguf, k: []const u8, v: f32) !void {
        try g.key(k, .f32);
        try num(&g.kv, v);
    }

    pub fn boolean(g: *Gguf, k: []const u8, v: bool) !void {
        try g.key(k, .bool);
        try num(&g.kv, @as(u8, @intFromBool(v)));
    }

    pub fn string(g: *Gguf, k: []const u8, v: []const u8) !void {
        try g.key(k, .string);
        try str(&g.kv, v);
    }

    pub fn ints(g: *Gguf, k: []const u8, vs: []const u32) !void {
        try g.key(k, .array);
        try num(&g.kv, @backingInt(Type.u32));
        try num(&g.kv, @as(u64, vs.len));
        for (vs) |v| try num(&g.kv, v);
    }

    /// A gpt2 vocabulary of `n` tokens, `eos` its end of sequence.
    pub fn vocab(g: *Gguf, n: u32, eos: u32) !void {
        try g.string("tokenizer.ggml.model", "gpt2");
        try g.string("tokenizer.ggml.pre", "qwen2");
        try g.key("tokenizer.ggml.tokens", .array);
        try num(&g.kv, @backingInt(Type.string));
        try num(&g.kv, @as(u64, n));
        var buf: [16]u8 = undefined;
        for (0..n) |i| try str(&g.kv, try std.fmt.bufPrint(&buf, "t{d}", .{i}));
        try g.int("tokenizer.ggml.eos_token_id", eos);
    }

    /// The tensor `name` of ggml type `ty` (ggml dims, innermost first) holding `bytes`.
    pub fn tensor(g: *Gguf, name: []const u8, dims: []const u64, ty: u32, bytes: []const u8) !void {
        try str(&g.info, name);
        try num(&g.info, @as(u32, @intCast(dims.len)));
        for (dims) |d| try num(&g.info, d);
        try num(&g.info, ty);
        try num(&g.info, @as(u64, g.data.items.len));
        try g.data.appendSlice(a, bytes);
        while (g.data.items.len % 32 != 0) try g.data.append(a, 0);
        g.n_tensors += 1;
    }

    /// `[rows, in]` (or `[experts, rows, in]`) blocks of `ty` (`fixture.ggufBlocks`).
    pub fn quant(g: *Gguf, r: std.Random, name: []const u8, shape: []const u64, comptime ty: @import("../../sdk/quant/gguf/quants.zig").GgmlType) !void {
        var n: u64 = 1;
        for (shape) |d| n *= d;
        const raw = try fixture.ggufBlocks(r, ty, @intCast(n / ty.blockElems()));
        defer a.free(raw);
        var dims: [3]u64 = undefined;
        for (shape, 0..) |d, i| dims[shape.len - 1 - i] = d;
        try g.tensor(name, dims[0..shape.len], @backingInt(ty), raw);
    }

    /// An f32 tensor of ggml `dims` (innermost first), uniform in `offset ± scale`.
    pub fn f32s(g: *Gguf, r: std.Random, name: []const u8, dims: []const u64, scale: f32, offset: f32) !void {
        var n: u64 = 1;
        for (dims) |d| n *= d;
        const vals = try a.alloc(f32, @intCast(n));
        defer a.free(vals);
        for (vals) |*v| v.* = offset + scale * (r.float(f32) * 2 - 1);
        try g.tensor(name, dims, 0, std.mem.sliceAsBytes(vals));
    }

    /// Writes the file to `dir/name`.
    pub fn write(g: *const Gguf, io: std.Io, dir: std.Io.Dir, name: []const u8) !void {
        var out: std.ArrayList(u8) = .empty;
        defer out.deinit(a);
        try out.appendSlice(a, "GGUF");
        try num(&out, @as(u32, 3));
        try num(&out, g.n_tensors);
        try num(&out, g.n_kv);
        try out.appendSlice(a, g.kv.items);
        try out.appendSlice(a, g.info.items);
        while (out.items.len % 32 != 0) try out.append(a, 0);
        try out.appendSlice(a, g.data.items);
        try dir.writeFile(io, .{ .sub_path = name, .data = out.items });
    }
};

/// A GGUF file (`write` fills `model.gguf`) served through the engine's `--mlx-gguf` path by plugin `table`, addressed
/// by its directory or by the file itself, against its tape.
pub fn openGguf(table: *const abi.ModelTable, g: *const Gguf, by_file: bool, oracle: both.Oracle) !both.Both {
    const a = testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();
    const prev = mlx_gguf.claimState().enabled;
    mlx_gguf.claimState().enabled = true;
    defer mlx_gguf.claimState().enabled = prev;
    var tmp = testing.tmpDir(.{});
    errdefer tmp.cleanup();
    try g.write(io, tmp.dir, "model.gguf");
    var path_buf: [512]u8 = undefined;
    const root = path_buf[0..try tmp.dir.realPath(io, &path_buf)];
    const dir = if (by_file) try std.fmt.allocPrint(a, "{s}/model.gguf", .{root}) else try a.dupe(u8, root);
    errdefer a.free(dir);
    var cfg = try model.parseConfig(io, a, dir);
    errdefer cfg.deinit(a);
    const b = cfg.plugin orelse return error.TestGgufNotServed;
    if (b.table != table) return error.TestGgufWrongPlugin;
    const xfm = try driver.load(io, a, &cfg, dir, fixture.stream());
    const t: tape_mod.Tape = switch (oracle) {
        .tape => |bytes| try tape_mod.Tape.player(bytes),
        .none => .{},
    };
    return .{ .tmp = tmp, .dir = dir, .cfg = cfg, .xfm = xfm, .tape = t };
}
