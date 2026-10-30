//! Stable Audio 3 as a media plugin: the table over the existing engine (`stable_audio.zig`), a job = one WAV.
//! Minimal by design (the media driver's first adopter); the backend's own port replaces the host-`sse` bridge.

const std = @import("std");
const abi = @import("../../abi/media.zig");
const sdk = @import("../../sdk/root.zig");
const mlx = @import("../../mlx/ffi.zig");
const job = @import("../../sdk/media/job.zig");
const sa = @import("stable_audio.zig");
pub const config = @import("config.zig");

pub const plugin = config.plugin;

pub const table = abi.MediaTable{
    .name = abi.Str.of(plugin.name),
    .claims = config.claims,
    .required_file = abi.Str.of("t5gemma-b-b-ul2/model.safetensors"),
    .modality = .audio,
    .caps = &config.caps,
    .bill_load = &config.billLoad,
    .bill_request = &config.billRequest,
    .load = &load,
    .unload = &unload,
    .step = &step,
    .encode = &encode,
};

const gpa = std.heap.c_allocator;
const Verdict = abi.Verdict;

const Engine = struct {
    sa: *sa.Engine,
    wav: []u8 = &.{},
};

fn load(ctx: *const abi.LoadCtx, out: *Verdict) callconv(.c) ?*anyopaque {
    const io = std.Io.Threaded.global_single_threaded.io();
    const s: mlx.mlx_stream = if (ctx.stream.ctx != null) .{ .ctx = ctx.stream.ctx } else mlx.mlx_default_gpu_stream_new();
    const e = gpa.create(Engine) catch {
        out.* = Verdict.named(.busy, @errorName(error.OutOfMemory));
        return null;
    };
    e.* = .{ .sa = sa.Engine.loadOn(io, gpa, ctx.model_dir.slice(), s) catch |err| {
        gpa.destroy(e);
        out.* = Verdict.named(if (err == error.OutOfMemory) .busy else .failed, @errorName(err));
        return null;
    } };
    return e;
}

fn unload(ep: *anyopaque) callconv(.c) void {
    const e: *Engine = @ptrCast(@alignCast(ep));
    e.sa.deinit();
    gpa.free(e.wav);
    gpa.destroy(e);
}

fn step(ep: *anyopaque, req: *const abi.Request, progress: *const abi.Progress) callconv(.c) Verdict {
    const e: *Engine = @ptrCast(@alignCast(ep));
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const b = switch (config.Body.parse(arena.allocator(), req.body.slice())) {
        .ok => |v| v,
        .bad => |v| return v,
    };
    const wav = e.sa.generateWav(gpa, .{ .prompt = b.prompt, .seconds = b.seconds, .steps = b.steps, .seed = b.seed }, job.progressOf(progress)) catch |err| return switch (err) {
        error.Cancelled => job.client_left,
        else => job.failure(err, "sound generation failed"),
    };
    gpa.free(e.wav);
    e.wav = wav;
    return .ok;
}

fn encode(ep: *anyopaque, out: *abi.Str) callconv(.c) Verdict {
    const e: *Engine = @ptrCast(@alignCast(ep));
    if (e.wav.len == 0) return Verdict.refuse(.failed, "stable_audio3: no finished job to encode");
    out.* = .of(e.wav);
    return .ok;
}

// ── the tiny pack ──

/// A tensor of the tiny pack: its bytes are the first `numel` values of blob `blob` (aliased tensors share a blob,
/// which keeps the pack a few MB at the published widths).
const Spec = struct { name: []const u8, shape: []const u64, bf16: bool = false, blob: usize };

/// The safetensors bytes of `specs` over `blobs` (f32 values; a bf16 tensor stores their top halves).
fn aliasedSafetensors(a: std.mem.Allocator, specs: []const Spec, blobs: []const []const f32) ![]u8 {
    var hdr: std.ArrayList(u8) = .empty;
    // Each (blob, dtype) a spec reads is written once; `offs[2 * blob + bf16]` is where it starts.
    const offs = try a.alloc(?u64, blobs.len * 2);
    @memset(offs, null);
    var data: std.ArrayList(u8) = .empty;
    for (specs) |sp| {
        const slot = &offs[2 * sp.blob + @intFromBool(sp.bf16)];
        if (slot.* != null) continue;
        slot.* = data.items.len;
        const bl = blobs[sp.blob];
        if (!sp.bf16) try data.appendSlice(a, std.mem.sliceAsBytes(bl)) else for (bl) |x|
            try data.appendSlice(a, std.mem.asBytes(&@as(u16, @truncate(@as(u32, @bitCast(x)) >> 16))));
    }
    try hdr.append(a, '{');
    for (specs, 0..) |sp, k| {
        var numel: u64 = 1;
        for (sp.shape) |d| numel *= d;
        std.debug.assert(numel <= blobs[sp.blob].len);
        const start = offs[2 * sp.blob + @intFromBool(sp.bf16)].?;
        const end = start + numel * @as(u64, if (sp.bf16) 2 else 4);
        try hdr.print(a, "{s}\"{s}\":{{\"dtype\":\"{s}\",\"shape\":[", .{ if (k == 0) "" else ",", sp.name, if (sp.bf16) "BF16" else "F32" });
        for (sp.shape, 0..) |d, j| try hdr.print(a, "{s}{d}", .{ if (j == 0) "" else ",", d });
        try hdr.print(a, "],\"data_offsets\":[{d},{d}]}}", .{ start, end });
    }
    try hdr.append(a, '}');
    const out = try a.alloc(u8, 8 + hdr.items.len + data.items.len);
    std.mem.writeInt(u64, out[0..8], hdr.items.len, .little);
    @memcpy(out[8..][0..hdr.items.len], hdr.items);
    @memcpy(out[8 + hdr.items.len ..], data.items);
    return out;
}

/// The tiny pack the conformance suite and the driver's proof run: Stability's layout at the published T5Gemma and
/// decoder widths (the engine fixes them), a 2-layer 64-wide DiT, one random tensor per role shared across layers.
pub fn conformanceFixture(a: std.mem.Allocator) !sdk.fixture.Fixture {
    const E = 64;
    const F = 16;
    const COND = 768;
    const DEC = 768;
    const rnd = sdk.fixture.randomF32;
    // Blobs, by role: values drawn once at the scale the role needs.
    const blobs = [_][]const f32{
        try rnd(a, 1, 32 * COND, 1), // 0 embeddings, rows
        try rnd(a, 2, COND, 0.1), // 1 norm weights
        try rnd(a, 3, 5 * DEC * DEC, 0.04), // 2 wide projections
        try rnd(a, 4, 2 * F * DEC, 0.04), // 3 narrow in-projections
        try rnd(a, 5, DEC * F, 0.2), // 4 narrow out-projections
        try rnd(a, 6, 6 * E * COND, 0.06), // 5 conditioner / DiT matrices
        try rnd(a, 7, 6 * E, 0.1), // 6 biases
        try rnd(a, 8, 2 * 256 * 256, 0.06), // 7 DiT convs
        try rnd(a, 9, COND, 1), // 8 unit-scale vectors (gammas, tokens, padding)
        try rnd(a, 10, 512, 1), // 9 weight-norm gains
    };
    var specs: std.ArrayList(Spec) = .empty;
    const put = struct {
        fn f(al: std.mem.Allocator, l: *std.ArrayList(Spec), name: []const u8, shape: []const u64, bf16: bool, blob: usize) !void {
            try l.append(al, .{ .name = try al.dupe(u8, name), .shape = try al.dupe(u64, shape), .bf16 = bf16, .blob = blob });
        }
    }.f;
    const putN = struct {
        fn f(al: std.mem.Allocator, l: *std.ArrayList(Spec), comptime fmt: []const u8, n: usize, shape: []const u64, bf16: bool, blob: usize) !void {
            for (0..n) |i| try put(al, l, try std.fmt.allocPrint(al, fmt, .{i}), shape, bf16, blob);
        }
    }.f;

    // T5Gemma (bf16, the published dtype), in its own file.
    var t5: std.ArrayList(Spec) = .empty;
    try put(a, &t5, "model.encoder.embed_tokens.weight", &.{ 32, COND }, true, 0);
    try put(a, &t5, "model.encoder.norm.weight", &.{COND}, true, 1);
    const tl = "model.encoder.layers.{d}.";
    inline for (.{ "pre_self_attn_layernorm", "post_self_attn_layernorm", "pre_feedforward_layernorm", "post_feedforward_layernorm" }) |nm|
        try putN(a, &t5, tl ++ nm ++ ".weight", 12, &.{COND}, true, 1);
    inline for (.{ "q", "k", "v", "o" }) |nm| try putN(a, &t5, tl ++ "self_attn." ++ nm ++ "_proj.weight", 12, &.{ COND, COND }, true, 2);
    try putN(a, &t5, tl ++ "mlp.gate_proj.weight", 12, &.{ F, COND }, true, 3);
    try putN(a, &t5, tl ++ "mlp.up_proj.weight", 12, &.{ F, COND }, true, 3);
    try putN(a, &t5, tl ++ "mlp.down_proj.weight", 12, &.{ COND, F }, true, 4);

    // Conditioner + DiT (f32), SAME-S decoder (bf16).
    const cp = "conditioner.conditioners.";
    try put(a, &specs, cp ++ "seconds_total.embedder.embedding.1.weight", &.{ COND, 256 }, false, 5);
    try put(a, &specs, cp ++ "seconds_total.embedder.embedding.1.bias", &.{COND}, false, 1);
    try put(a, &specs, cp ++ "prompt.padding_embedding", &.{COND}, false, 8);
    const mm = "model.model.";
    inline for (.{ "to_cond_embed", "to_global_embed" }) |nm| {
        try put(a, &specs, mm ++ nm ++ ".0.weight", &.{ E, COND }, false, 5);
        try put(a, &specs, mm ++ nm ++ ".2.weight", &.{ E, E }, false, 5);
    }
    try put(a, &specs, mm ++ "to_timestep_embed.0.weight", &.{ E, 256 }, false, 5);
    try put(a, &specs, mm ++ "to_timestep_embed.0.bias", &.{E}, false, 6);
    try put(a, &specs, mm ++ "to_timestep_embed.2.weight", &.{ E, E }, false, 5);
    try put(a, &specs, mm ++ "to_timestep_embed.2.bias", &.{E}, false, 6);
    try put(a, &specs, mm ++ "preprocess_conv.weight", &.{ 256, 256, 1 }, false, 7);
    try put(a, &specs, mm ++ "postprocess_conv.weight", &.{ 256, 256, 1 }, false, 7);
    const tp = mm ++ "transformer.";
    try put(a, &specs, tp ++ "global_cond_embedder.0.weight", &.{ E, E }, false, 5);
    try put(a, &specs, tp ++ "global_cond_embedder.0.bias", &.{E}, false, 6);
    try put(a, &specs, tp ++ "global_cond_embedder.2.weight", &.{ 6 * E, E }, false, 5);
    try put(a, &specs, tp ++ "global_cond_embedder.2.bias", &.{6 * E}, false, 6);
    try put(a, &specs, tp ++ "memory_tokens", &.{ 4, E }, false, 8);
    try put(a, &specs, tp ++ "project_in.weight", &.{ E, 256 }, false, 5);
    try put(a, &specs, tp ++ "project_out.weight", &.{ 256, E }, false, 5);
    const dl = tp ++ "layers.{d}.";
    inline for (.{ "pre_norm.gamma", "cross_attend_norm.gamma", "ff_norm.gamma" }) |nm| try putN(a, &specs, dl ++ nm, 2, &.{E}, false, 8);
    inline for (.{ "self_attn.q_norm.gamma", "self_attn.k_norm.gamma", "cross_attn.q_norm.gamma", "cross_attn.k_norm.gamma" }) |nm| try putN(a, &specs, dl ++ nm, 2, &.{32}, false, 8);
    try putN(a, &specs, dl ++ "self_attn.to_qkv.weight", 2, &.{ 3 * E, E }, false, 5);
    try putN(a, &specs, dl ++ "self_attn.to_out.weight", 2, &.{ E, E }, false, 5);
    try putN(a, &specs, dl ++ "cross_attn.to_q.weight", 2, &.{ E, E }, false, 5);
    try putN(a, &specs, dl ++ "cross_attn.to_kv.weight", 2, &.{ 2 * E, E }, false, 5);
    try putN(a, &specs, dl ++ "cross_attn.to_out.weight", 2, &.{ E, E }, false, 5);
    try putN(a, &specs, dl ++ "ff.ff.0.proj.weight", 2, &.{ 2 * F, E }, false, 5);
    try putN(a, &specs, dl ++ "ff.ff.0.proj.bias", 2, &.{2 * F}, false, 6);
    try putN(a, &specs, dl ++ "ff.ff.2.weight", 2, &.{ E, F }, false, 4);
    try putN(a, &specs, dl ++ "ff.ff.2.bias", 2, &.{E}, false, 6);
    try putN(a, &specs, dl ++ "to_scale_shift_gate", 2, &.{6 * E}, false, 6);
    try putN(a, &specs, dl ++ "to_local_embed.0.bias", 2, &.{8}, false, 8);
    try putN(a, &specs, dl ++ "to_local_embed.2.weight", 2, &.{ E, 8 }, false, 5);
    try putN(a, &specs, dl ++ "to_local_embed.2.bias", 2, &.{E}, false, 6);
    try put(a, &specs, "pretransform.model.bottleneck.running_std", &.{1}, false, 8);
    const dp = "pretransform.model.decoder.layers.";
    try put(a, &specs, dp ++ "1.weight", &.{ DEC, 256 }, true, 2);
    try put(a, &specs, dp ++ "1.bias", &.{DEC}, false, 1);
    try put(a, &specs, dp ++ "3.new_tokens", &.{DEC}, false, 8);
    try put(a, &specs, dp ++ "3.mapping.weight_g", &.{ 512, 1, 1 }, false, 9);
    try put(a, &specs, dp ++ "3.mapping.weight_v", &.{ 512, DEC, 3 }, true, 2);
    try put(a, &specs, dp ++ "3.mapping.bias", &.{512}, false, 1);
    const bl = dp ++ "3.transformers.{d}.";
    inline for (.{ "pre_norm", "ff_norm", "self_attn.q_norm", "self_attn.k_norm" }) |nm| {
        const w: u64 = if (std.mem.endsWith(u8, nm, "_norm") and std.mem.startsWith(u8, nm, "self_attn")) 64 else DEC;
        try putN(a, &specs, bl ++ nm ++ ".alpha", 6, &.{1}, false, 8);
        try putN(a, &specs, bl ++ nm ++ ".gamma", 6, &.{w}, false, 8);
        try putN(a, &specs, bl ++ nm ++ ".beta", 6, &.{w}, false, 1);
    }
    try putN(a, &specs, bl ++ "self_attn.to_qkv.weight", 6, &.{ 5 * DEC, DEC }, true, 2);
    try putN(a, &specs, bl ++ "self_attn.to_out.weight", 6, &.{ DEC, DEC }, true, 2);
    try putN(a, &specs, bl ++ "ff.ff.0.proj.weight", 6, &.{ 2 * F, DEC }, true, 3);
    try putN(a, &specs, bl ++ "ff.ff.0.proj.bias", 6, &.{2 * F}, false, 6);
    try putN(a, &specs, bl ++ "ff.ff.2.weight", 6, &.{ DEC, F }, true, 4);
    try putN(a, &specs, bl ++ "ff.ff.2.bias", 6, &.{DEC}, false, 1);

    const model_config =
        \\{"model_type":"diffusion_cond_inpaint","model":{"diffusion":{"config":{"embed_dim":64,"depth":2,"num_heads":2,"num_memory_tokens":4}},"conditioning":{"configs":[{"id":"prompt","type":"t5gemma"}]}}}
    ;
    // Byte-level BPE over the prompt's letters only: ids stay inside the 32-row embedding.
    const tokenizer =
        \\{"pre_tokenizer":{"type":"ByteLevel"},"model":{"type":"BPE","vocab":{"r":0,"a":1,"i":2,"n":3,"\u0120":4,"o":5,"f":6},"merges":[]}}
    ;
    return .{
        .files = try a.dupe(sdk.fixture.File, &.{
            .{ .name = "model_config.json", .bytes = model_config },
            .{ .name = "model.safetensors", .bytes = try aliasedSafetensors(a, specs.items, &blobs) },
            .{ .name = "t5gemma-b-b-ul2/model.safetensors", .bytes = try aliasedSafetensors(a, t5.items, &blobs) },
            .{ .name = "t5gemma-b-b-ul2/tokenizer.json", .bytes = tokenizer },
        }),
        .body = "{\"prompt\":\"rain\",\"duration_seconds\":1,\"steps\":2,\"seed\":3}",
    };
}

// ── tests ──

const testing = std.testing;

test "stable_audio3 plugin: the tiny pack claims by its Stability marker and bills its safetensors" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const fx = try conformanceFixture(arena.allocator());
    const mem: sdk.fixture.MemDir = .{ .files = fx.files };
    const dir = mem.packDir();
    try testing.expectEqual(abi.Claim.by_marker, config.claims(.{}, &dir));
    try testing.expectEqual(abi.Claim.by_type, config.claims(.of("stable_audio3"), &dir));
    try testing.expectEqual(abi.Claim.no, config.claims(.of("flux2"), &dir));
    const want = fx.file("model.safetensors").?.len + fx.file("t5gemma-b-b-ul2/model.safetensors").?.len;
    try testing.expectEqual(@as(u64, want), config.billLoad(&dir, &.{}));
}
