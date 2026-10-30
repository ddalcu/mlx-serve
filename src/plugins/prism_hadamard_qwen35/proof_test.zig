//! The Prism table served through the engine's plugin driver, proven against the legacy shell over one tiny Hadamard
//! pack: every projection 2-bit group 128 with biases at -scales, rotated by block-1024 sign vectors (one shared by
//! every reader of the residual, so the seams fuse), the unpacked tensors f32 (narrowed to f16 at load). Steps,
//! rollbacks, the lookup loop, batched ticks, an image prompt and the MTP head equal the legacy shell bit for bit.

const std = @import("std");
const mlx = @import("../../mlx/ffi.zig");
const fixture = @import("../../sdk/bases/proof/fixture.zig");
const driver = @import("../../engines/mlx/plugin/driver.zig");
const served = @import("../qwen3_5/served_test.zig");
const mtp_test = @import("../qwen3_5/mtp_test.zig");
const q35 = @import("../qwen3_5/plugin.zig");
const prism = @import("plugin.zig");

const testing = std.testing;
const Array = mlx.mlx_array;

const H: c_int = 1024;
const VOCAB: c_int = 256;
const INTER: c_int = 2048;
const HEADS: c_int = 8;
const KV_HEADS: c_int = 2;
const HD: c_int = 256;
const HK: c_int = 4;
const HV: c_int = 16;
const D: c_int = 128;

fn config(comptime tied: bool, comptime vision: bool) []const u8 {
    const t = if (tied) "true" else "false";
    const vl = if (vision) "\"image_token_id\":250,\"vision_config\":{\"depth\":1}," else "";
    const sections = if (vision) ",\"mrope_section\":[11,11,10],\"mrope_interleaved\":true" else "";
    return "{\"model_type\":\"prism_hadamard_qwen35\"," ++ vl ++ "\"modules\":[{\"path\":\"lm_head\",\"block\":1024}]," ++
        "\"text_config\":{\"model_type\":\"qwen3_5_text\",\"hidden_size\":1024,\"intermediate_size\":2048," ++
        "\"num_hidden_layers\":4,\"num_attention_heads\":8,\"num_key_value_heads\":2,\"head_dim\":256,\"vocab_size\":256," ++
        "\"rms_norm_eps\":1e-6,\"full_attention_interval\":2,\"linear_num_key_heads\":4,\"linear_num_value_heads\":16," ++
        "\"linear_key_head_dim\":128,\"linear_value_head_dim\":128,\"linear_conv_kernel_dim\":4,\"max_position_embeddings\":4096," ++
        "\"rope_parameters\":{\"rope_theta\":100000.0,\"partial_rotary_factor\":0.25" ++ sections ++ "},\"tie_word_embeddings\":" ++ t ++ "}," ++
        "\"tie_word_embeddings\":" ++ t ++ ",\"quantization\":{\"group_size\":128,\"bits\":2,\"mode\":\"affine\"}}";
}

/// Three sign vectors: the residual's (width H), the value/attention width's and the MLP's (both 2048: a width two
/// vectors share, which the seams never read).
const Signs = struct { h: Array, v: Array, m: Array };

fn signs(n: c_int, seed: u64) Array {
    var prng = std.Random.DefaultPrng.init(seed);
    var buf: [4096]f32 = undefined;
    for (buf[0..@intCast(n)]) |*x| x.* = if (prng.random().boolean()) 1 else -1;
    return mlx.mlx_array_new_data(&buf, &[_]c_int{n}, 1, .float32);
}

/// `<base>` 2-bit group 128 over f16 rows (Prism's f16 scales), every bias exactly -scale, rotated by `sg`.
fn rotated(p: *fixture.Pack, base: []const u8, rows: c_int, cols: c_int, sg: Array) !void {
    const n: usize = @intCast(rows * cols);
    const host = try testing.allocator.alloc(f32, n);
    defer testing.allocator.free(host);
    const scale = 1.0 / @sqrt(@as(f32, @floatFromInt(cols)));
    for (host) |*v| v.* = scale * (p.rnd.random().float(f32) * 2 - 1);
    const f = mlx.mlx_array_new_data(host.ptr, &[_]c_int{ rows, cols }, 2, .float32);
    defer _ = mlx.mlx_array_free(f);
    var h = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(h);
    try mlx.check(mlx.mlx_astype(&h, f, .float16, p.s));
    var triple = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(triple);
    try mlx.check(mlx.mlx_quantize(&triple, h, mlx.mlx_optional_int.some(128), mlx.mlx_optional_int.some(2), "affine", .{}, p.s));
    var w = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(w);
    var sc = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(sc);
    var neg = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(neg);
    try mlx.check(mlx.mlx_vector_array_get(&w, triple, 0));
    try mlx.check(mlx.mlx_vector_array_get(&sc, triple, 1));
    try mlx.check(mlx.mlx_negative(&neg, sc, p.s));
    var nb: [192]u8 = undefined;
    for ([_][]const u8{ "weight", "scales", "biases", "signs" }, [_]Array{ w, sc, neg, sg }) |leaf, a| {
        try mlx.check(mlx.mlx_array_eval(a));
        try p.put(try std.fmt.bufPrint(&nb, "{s}.{s}", .{ base, leaf }), a);
    }
}

fn f32Norm(p: *fixture.Pack, name: []const u8, n: c_int) !void {
    try p.denseAs(name, &.{n}, 0.2, 1.0, .float32);
}

fn at(buf: []u8, layer: usize, name: []const u8) ![]const u8 {
    return std.fmt.bufPrint(buf, "language_model.model.layers.{d}.{s}", .{ layer, name });
}

/// Linear-attention layers 0 and 2 (dense f32 a/b rows, as Prism ships them), gated attention 1 and 3.
fn writePack(p: *fixture.Pack, tied: bool) !void {
    const sg: Signs = .{ .h = signs(H, 11), .v = signs(2048, 12), .m = signs(INTER, 13) };
    defer for ([_]Array{ sg.h, sg.v, sg.m }) |a| {
        _ = mlx.mlx_array_free(a);
    };
    var buf: [160]u8 = undefined;
    try rotated(p, "language_model.model.embed_tokens", VOCAB, H, sg.h);
    try f32Norm(p, "language_model.model.norm.weight", H);
    if (!tied) try rotated(p, "language_model.lm_head", VOCAB, H, sg.h);
    for (0..4) |i| {
        try f32Norm(p, try at(&buf, i, "input_layernorm.weight"), H);
        try f32Norm(p, try at(&buf, i, "post_attention_layernorm.weight"), H);
        try rotated(p, try at(&buf, i, "mlp.gate_proj"), INTER, H, sg.h);
        try rotated(p, try at(&buf, i, "mlp.up_proj"), INTER, H, sg.h);
        try rotated(p, try at(&buf, i, "mlp.down_proj"), H, INTER, sg.m);
        if (i % 2 == 0) {
            const conv = 2 * HK * D + HV * D;
            try rotated(p, try at(&buf, i, "linear_attn.in_proj_qkv"), conv, H, sg.h);
            try rotated(p, try at(&buf, i, "linear_attn.in_proj_z"), HV * D, H, sg.h);
            try p.denseAs(try at(&buf, i, "linear_attn.in_proj_a.weight"), &.{ HV, H }, 1.0 / 32.0, 0, .float32);
            try p.denseAs(try at(&buf, i, "linear_attn.in_proj_b.weight"), &.{ HV, H }, 1.0 / 32.0, 0, .float32);
            try p.denseAs(try at(&buf, i, "linear_attn.conv1d.weight"), &.{ conv, 4, 1 }, 0.5, 0, .float32);
            try p.denseAs(try at(&buf, i, "linear_attn.A_log"), &.{HV}, 0.5, 0.2, .float32);
            try p.denseAs(try at(&buf, i, "linear_attn.dt_bias"), &.{HV}, 0.5, 0, .float32);
            try f32Norm(p, try at(&buf, i, "linear_attn.norm.weight"), D);
            try rotated(p, try at(&buf, i, "linear_attn.out_proj"), H, HV * D, sg.v);
        } else {
            try rotated(p, try at(&buf, i, "self_attn.q_proj"), 2 * HEADS * HD, H, sg.h);
            try rotated(p, try at(&buf, i, "self_attn.k_proj"), KV_HEADS * HD, H, sg.h);
            try rotated(p, try at(&buf, i, "self_attn.v_proj"), KV_HEADS * HD, H, sg.h);
            try rotated(p, try at(&buf, i, "self_attn.o_proj"), H, HEADS * HD, sg.v);
            try f32Norm(p, try at(&buf, i, "self_attn.q_norm.weight"), HD);
            try f32Norm(p, try at(&buf, i, "self_attn.k_norm.weight"), HD);
        }
    }
}

fn openPack(p: *served.Pair, comptime tied: bool, comptime vision: bool, comptime seed: u64) !void {
    var pk = fixture.Pack.init(seed);
    defer pk.deinit();
    try writePack(&pk, tied);
    try p.open(&prism.table, &pk, config(tied, vision), try served.oracle("prism_hadamard_qwen35", std.fmt.comptimePrint("served-{x}", .{seed})));
    const m: *q35.Loaded = @ptrCast(@alignCast(p.xfm.plugin.?.model));
    try testing.expect(m.body == .prism and m.rht != null);
    // The shell's registry binds the rotated head the engine's readouts read.
    try testing.expect(p.xfm.rht.?.get(p.xfm.lm_head_w) != null);
}

test "prism served: steps, rollbacks and the lookup loop equal the legacy shell" {
    var p: served.Pair = undefined;
    try openPack(&p, false, false, 0xb001);
    defer p.close();
    try testing.expectEqual(driver.Positions.gdn, driver.positionsOf(&p.xfm));
    try served.expectSolo(&p);
}

fn int8Prefill(c: *@import("../../sdk/config/model.zig").ModelConfig) void {
    c.int8_prefill_override = true;
}

test "prism served: int8_prefill (LOSSY) takes the int8 route on a 70-row chunk as the legacy shell does, steps equal" {
    if (!mlx.metalKernelsAvailable() or !@import("../../mlx/device.zig").naxAvailable()) return error.SkipZigTest;
    var pk = fixture.Pack.init(0xb006);
    defer pk.deinit();
    try writePack(&pk, false);
    var p: served.Pair = undefined;
    try p.openSettings(&prism.table, &pk, config(false, false), &int8Prefill, try served.oracle("prism_hadamard_qwen35", "served-b006"));
    defer p.close();
    const m: *q35.Loaded = @ptrCast(@alignCast(p.xfm.plugin.?.model));
    try testing.expect(m.rht.?.int8 != null);
    try served.expectSolo(&p);
}

test "prism served: a tied pack (the head reads the rotated table) equals the legacy shell" {
    var p: served.Pair = undefined;
    try openPack(&p, true, false, 0xb002);
    defer p.close();
    try served.expectSolo(&p);
}

test "prism served: batched decode ticks and a batched verify equal the legacy GDN batched path" {
    var p: served.Pair = undefined;
    try openPack(&p, false, false, 0xb003);
    defer p.close();
    try served.expectBatched(&p);
}

test "prism served: an image prompt on the M-RoPE table equals the legacy shell" {
    var p: served.Pair = undefined;
    try openPack(&p, false, true, 0xb004);
    defer p.close();
    try served.expectImage(&p);
}

test "prism served: a DFlash drafter on the rotated trunk reads its embedding and head and emits the legacy stream" {
    const io = std.Io.Threaded.global_single_threaded.io();
    const dflash_mod = @import("../../engines/mlx/spec/dflash/dflash.zig");
    var pk = fixture.Pack.init(0xb007);
    defer pk.deinit();
    try writePack(&pk, false);
    var p: served.Pair = undefined;
    try p.open(&prism.table, &pk, config(false, false), try served.oracle("prism_hadamard_qwen35", "dflash-b007"));
    defer p.close();
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const asst = buf[0..try tmp.dir.realPath(io, &buf)];
    try dflash_mod.TinyFix.writeAssistantAs(io, tmp.dir, asst, fixture.stream(), false, @intCast(H));
    try served.expectDflash(&p, asst, H, "lmHeadForeign");
}

test "prism served: the MTP head over the rotated trunk equals the legacy arm (steps, draft picks, rounds)" {
    var pk = fixture.Pack.init(0xb005);
    defer pk.deinit();
    try writePack(&pk, false);
    try mtp_test.expectHeadAt(&prism.table, &pk, config(false, false), false, .{ .hidden = @intCast(H), .heads = @intCast(HEADS), .kv_heads = @intCast(KV_HEADS), .inter = @intCast(INTER) }, "prism_hadamard_qwen35", "mtp-b005");
}
