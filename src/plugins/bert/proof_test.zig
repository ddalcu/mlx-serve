//! Parity proof for the bert plugin: BERTs of tiny random weights (4-bit, and 2-bit group 128) served by the plugin
//! through the driver's shell, the hidden rows bit-equal for one sequence and for a padded batch under its key mask to
//! the deleted legacy `forwardBertWith` (its GELU compiled), replayed from `fixtures/*.tape`; and `parse` equal to the
//! engine's parse.

const std = @import("std");
const abi = @import("../../abi/plugin.zig");
const ops = @import("../../mlx/ops.zig");
const mlx = @import("../../mlx/ffi.zig");
const fixture = @import("../../sdk/bases/proof/fixture.zig");
const buildKeyPadMask = @import("../../engines/mlx/embed/pooling.zig").buildKeyPadMask;
const proof = @import("../lfm2/proof.zig");
const bert = @import("plugin.zig");
const config = @import("config.zig");
const both = @import("../../engines/mlx/plugin/both.zig");

const testing = std.testing;

const H: c_int = 128;
const INTER: c_int = 256;
const LAYERS = 2;

const config_json =
    \\{"model_type":"bert","vocab_size":256,"hidden_size":128,"intermediate_size":256,"num_hidden_layers":2,
    \\ "num_attention_heads":4,"max_position_embeddings":64,"type_vocab_size":2,"layer_norm_eps":1e-12,
    \\ "pooling_mode":"cls","quantization":{"group_size":64,"bits":4}}
;

/// The same pack at 2 bits, group 128.
const config_2bit = config_json[0 .. config_json.len - "{\"group_size\":64,\"bits\":4}}".len] ++ "{\"group_size\":128,\"bits\":2}}";

fn ln(p: *fixture.Pack, buf: []u8, base: []const u8) !void {
    try p.norm(try std.fmt.bufPrint(buf, "{s}.weight", .{base}), H);
    try p.dense(try std.fmt.bufPrint(buf, "{s}.bias", .{base}), &.{H}, 0.1, 0);
}

fn quantized(p: *fixture.Pack, base: []const u8, rows: c_int, cols: c_int, two_bit: bool) !void {
    return if (two_bit) p.affine(base, &.{ rows, cols }, 2, 128) else p.weight(base, rows, cols, true);
}

fn lin(p: *fixture.Pack, buf: []u8, base: []const u8, rows: c_int, cols: c_int, two_bit: bool) !void {
    try quantized(p, base, rows, cols, two_bit);
    try p.dense(try std.fmt.bufPrint(buf, "{s}.bias", .{base}), &.{rows}, 0.1, 0);
}

fn writePack(p: *fixture.Pack, two_bit: bool) !void {
    var buf: [160]u8 = undefined;
    var nb: [160]u8 = undefined;
    try quantized(p, "embeddings.word_embeddings", 256, H, two_bit);
    try quantized(p, "embeddings.position_embeddings", 64, H, two_bit);
    try quantized(p, "embeddings.token_type_embeddings", 2, H, two_bit);
    try ln(p, &buf, "embeddings.LayerNorm");
    for (0..LAYERS) |i| {
        const at = struct {
            fn f(bb: []u8, layer: usize, name: []const u8) ![]const u8 {
                return std.fmt.bufPrint(bb, "encoder.layer.{d}.{s}", .{ layer, name });
            }
        }.f;
        try lin(p, &buf, try at(&nb, i, "attention.self.query"), H, H, two_bit);
        try lin(p, &buf, try at(&nb, i, "attention.self.key"), H, H, two_bit);
        try lin(p, &buf, try at(&nb, i, "attention.self.value"), H, H, two_bit);
        try lin(p, &buf, try at(&nb, i, "attention.output.dense"), H, H, two_bit);
        try ln(p, &buf, try at(&nb, i, "attention.output.LayerNorm"));
        try lin(p, &buf, try at(&nb, i, "intermediate.dense"), INTER, H, two_bit);
        try lin(p, &buf, try at(&nb, i, "output.dense"), H, INTER, two_bit);
        try ln(p, &buf, try at(&nb, i, "output.LayerNorm"));
    }
}


/// Hidden rows through the driver's shell on the legacy config (one unmasked sequence, a padded batch under its key
/// mask), against the recorded `forwardBertWith` (its GELU compiled, as the engine's load compiled it).
fn servedCase(json: []const u8, two_bit: bool, comptime seed: u64) !void {
    const a = testing.allocator;
    var p = fixture.Pack.init(seed);
    defer p.deinit();
    try writePack(&p, two_bit);
    const name = std.fmt.comptimePrint("fixtures/{x}.tape", .{seed});
    var b = try both.Both.replay(&bert.table, &p, json, true, @embedFile(name));
    defer b.close();
    const m: *bert.Loaded = @ptrCast(@alignCast(b.xfm.plugin.?.model));
    try testing.expect(m.gelu.ctx != null);
    var g = ops.Graph.init(a, b.xfm.s);
    defer g.deinit();
    const one = try g.fromSlice(i32, &.{ 101, 7, 42, 250, 3, 102 }, &.{ 1, 6 });
    const ids = try g.fromSlice(i32, &.{ 101, 7, 42, 250, 102, 101, 9, 102, 0, 0 }, &.{ 2, 5 });
    const mask = try g.adopt(try buildKeyPadMask(a, &.{ 5, 3 }, 5, b.xfm.s));
    for ([_]mlx.mlx_array{ one, ids }, [_]?mlx.mlx_array{ null, mask }) |x, km| {
        const got = try g.adopt(try b.xfm.forwardEmbeddingMasked(x, km, null));
        try b.check(got, false);
    }
    // An encoder refuses decode.
    const toks = [_]u32{ 101, 7 };
    var refused: abi.ForwardOut = .{};
    try testing.expectEqual(abi.Verdict.Code.unsupported, bert.table.forward(m, &.{ .mode = .decode, .rows = 1, .tokens = .of(toks[0..1]), .positions = .of(&.{6}) }, &refused).code);
    try b.expectDone();
}

// ── tests ──

test "bert served: hidden rows (one sequence, a padded batch under its key mask) are bit-equal to forwardBertWith" {
    try servedCase(config_json, false, 0xbe28);
}

test "bert served: a 2-bit group-128 pack (stock kernels, as the engine's BERT) is bit-equal" {
    try servedCase(config_2bit, true, 0xbe29);
}

test "bert plugin: parse equals the engine's parse on the shell, the norm eps and the pooling contract" {
    var want = try proof.legacyConfig(config_json);
    defer want.deinit(testing.allocator);
    var p: abi.Parsed = .{};
    try testing.expect(config.parseJson(config_json, &p).isOk());
    defer config.freeParsed(&p);
    try proof.expectShell(&p.shell, &want);
    try testing.expectEqual(want.layer_norm_eps, config.of(&p).layer_norm_eps);
    try testing.expectEqualStrings(@tagName(want.pooling_mode.?), @tagName(p.embed.pooling));
    try testing.expect(want.is_encoder_only and p.caps.embed);
}

test "bert served: a pack whose folder name picks the pooling serves under the legacy config's pooling" {
    const io = std.Io.Threaded.global_single_threaded.io();
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.createDirPath(io, "bge-small-en");
    const json =
        \\{"model_type":"bert","vocab_size":256,"hidden_size":128,"intermediate_size":256,"num_hidden_layers":2,
        \\ "num_attention_heads":4,"max_position_embeddings":64,"type_vocab_size":2,"layer_norm_eps":1e-12,
        \\ "quantization":{"group_size":64,"bits":4}}
    ;
    try tmp.dir.writeFile(io, .{ .sub_path = "bge-small-en/config.json", .data = json });
    var buf: [512]u8 = undefined;
    const root = buf[0..try tmp.dir.realPath(io, &buf)];
    var pbuf: [600]u8 = undefined;
    const dir = try std.fmt.bufPrint(&pbuf, "{s}/bge-small-en", .{root});
    var cfg = try @import("../../sdk/config/model.zig").parseConfig(io, testing.allocator, dir);
    defer cfg.deinit(testing.allocator);
    try testing.expect(cfg.plugin != null);
    try testing.expectEqual(.cls, cfg.pooling_mode.?);
}
