//! Parity proof and contract tests for the hy_v4_flash plugin.

const std = @import("std");
const testing = std.testing;
const abi = @import("../../abi/plugin.zig");
const mlx = @import("../../mlx/ffi.zig");
const plugin = @import("plugin.zig");
const config = @import("config.zig");
const hy4 = @import("hy_v4_flash.zig");
const fixture = @import("../../sdk/testing/fixture.zig");
const engram = @import("../../sdk/weights/engram.zig");
const weights_load = @import("../../sdk/weights/load.zig");

test "hy_v4_flash: table declares hy_v4_flash and refuses foreign types" {
    const claims = plugin.table.claims;
    const mem: fixture.MemDir = .{ .files = &.{} };
    const pd = mem.packDir();
    const claim = claims(abi.Str.of("hy_v4_flash"), abi.Str.of("{}"), &pd);
    try testing.expect(claim != .no);

    const not_claim = claims(abi.Str.of("hy_v3"), abi.Str.of("{}"), &pd);
    try testing.expect(not_claim == .no);
}

test "hy_v4_flash: config parse sets module-owned decode caps and merges eos 120025" {
    const cfg_json =
        \\{
        \\  "model_type": "hy_v4_flash",
        \\  "vocab_size": 120832,
        \\  "hidden_size": 3072,
        \\  "num_hidden_layers": 42,
        \\  "num_attention_heads": 32,
        \\  "num_key_value_heads": 32,
        \\  "head_dim": 96,
        \\  "num_experts_per_tok": 16,
        \\  "moe_intermediate_size": 1536,
        \\  "eos_token_id": [2, 120020]
        \\}
    ;
    var parsed: abi.Parsed = .{};
    const mem: fixture.MemDir = .{ .files = &.{} };
    const pd = mem.packDir();
    const res = config.parse(&pd, abi.Str.of(cfg_json), abi.Str.of("{}"), null, true, &parsed);
    try testing.expectEqual(abi.Verdict.ok, res);
    defer config.freeParsed(&parsed);

    try testing.expect(parsed.caps.owns_decode_state);
    try testing.expect(!parsed.caps.prefix_cache);
    try testing.expect(parsed.caps.moe);
    try testing.expectEqualSlices(u32, &.{ 2, 120020, config.hy_eos }, parsed.shell.eos[0..parsed.shell.eos_count]);

    const c = config.of(&parsed);
    try testing.expectEqual(@as(usize, 3072), c.hidden);
    try testing.expectEqual(@as(usize, 42), c.layers);
    try testing.expectEqual(@as(usize, 120832), c.vocab);
}

test "hy_v4_flash: layout reflects indexed vs sliding attention layers" {
    const cfg_json =
        \\{
        \\  "model_type": "hy_v4_flash",
        \\  "vocab_size": 120832,
        \\  "hidden_size": 3072,
        \\  "num_hidden_layers": 3,
        \\  "num_attention_heads": 32,
        \\  "num_key_value_heads": 32,
        \\  "head_dim": 96,
        \\  "num_experts_per_tok": 16,
        \\  "moe_intermediate_size": 1536,
        \\  "sliding_window": 512,
        \\  "layer_types": ["indexed_attention", "sliding_attention", "indexed_attention"]
        \\}
    ;
    var parsed: abi.Parsed = .{};
    const mem: fixture.MemDir = .{ .files = &.{} };
    const pd = mem.packDir();
    const res = config.parse(&pd, abi.Str.of(cfg_json), abi.Str.of("{}"), null, true, &parsed);
    try testing.expectEqual(abi.Verdict.ok, res);
    defer config.freeParsed(&parsed);

    var layers: [3]abi.LayerCache = undefined;
    const n = plugin.layout(&parsed, null, &layers, 3);
    try testing.expectEqual(@as(u32, 3), n);
    try testing.expectEqual(abi.LayerKind.full, layers[0].kind);
    try testing.expectEqual(abi.LayerKind.sliding, layers[1].kind);
    try testing.expectEqual(@as(u32, 512), layers[1].window);
    try testing.expectEqual(abi.LayerKind.full, layers[2].kind);
}

test "hy_v4_flash: external engram_table.bin opens through openHy4 and Table.gather" {
    const io = std.Io.Threaded.global_single_threaded.io();
    var td = testing.tmpDir(.{});
    defer td.cleanup();
    // 4 rows, head_dim 32: weight U32 [4,4], scales/biases BF16 [4,1].
    const hdr = "{\"weight\":{\"dtype\":\"U32\",\"shape\":[4,4],\"data_offsets\":[0,64]}," ++
        "\"scales\":{\"dtype\":\"BF16\",\"shape\":[4,1],\"data_offsets\":[64,72]}," ++
        "\"biases\":{\"dtype\":\"BF16\",\"shape\":[4,1],\"data_offsets\":[72,80]}}";
    const pad = (8 - hdr.len % 8) % 8;
    var bin: [8 + hdr.len + pad + 80]u8 = @splat(0);
    std.mem.writeInt(u64, bin[0..8], hdr.len + pad, .little);
    @memcpy(bin[8..][0..hdr.len], hdr);
    @memset(bin[8 + hdr.len ..][0..pad], ' ');
    const data = bin[8 + hdr.len + pad ..];
    for (data[0..64]) |*x| x.* = 0x11; // all nibbles = 1
    for (0..4) |r| {
        std.mem.writeInt(u16, data[64 + 2 * r ..][0..2], 0x4000, .little); // scale 2
        std.mem.writeInt(u16, data[72 + 2 * r ..][0..2], 0x3f00, .little); // bias 0.5
    }
    try td.dir.writeFile(io, .{ .sub_path = "engram_table.bin", .data = &bin });
    const spec_json = "{\"model_type\":\"hy_v4_flash\",\"engram_table\":{\"file\":\"engram_table.bin\",\"bits\":4,\"group_size\":32,\"head_dim\":32,\"rows\":4,\"layers\":[1,11,21,31],\"table_rows\":[1,1,1,1]}}";
    var buf: [512]u8 = undefined;
    const dir = buf[0..try td.dir.realPath(io, &buf)];
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const spec = (try config.parseHy4Spec(a, spec_json)).?;
    defer a.free(spec.file);
    defer a.free(spec.layers);
    defer a.free(spec.table_rows);
    try testing.expectEqual(@as(u64, 2), spec.rowOffset(2));
    var t = try hy4.openHy4(a, dir, &spec);
    defer t.close();
    const s = mlx.mlx_default_cpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    const got = try t.gather(testing.allocator, &.{ 3, 0 }, s);
    defer _ = mlx.mlx_array_free(got);
    try mlx.check(mlx.mlx_array_eval(got));
    const v = mlx.mlx_array_data_float32(got).?;
    for (0..64) |i| try testing.expectEqual(@as(f32, 2.5), v[i]); // 1*2 + 0.5
}

test "hy_v4_flash: tiny fixture forward logits match reference python oracle" {
    var path_buf: [std.fs.max_path_bytes]u8 = undefined;
    const rel_path = "src/plugins/hy_v4_flash/fixtures/tiny_fixture\x00";
    const fixture_dir_c = std.c.realpath(rel_path, &path_buf) orelse return; // Skip if run from outside repository root
    const fixture_dir = std.mem.span(fixture_dir_c);

    const cfg_path = try std.fmt.allocPrint(testing.allocator, "{s}/config.json", .{fixture_dir});
    defer testing.allocator.free(cfg_path);
    const cfg_raw = std.Io.Dir.cwd().readFileAlloc(testing.io, cfg_path, testing.allocator, .limited(1 << 20)) catch return;
    defer testing.allocator.free(cfg_raw);

    var parsed: abi.Parsed = .{};
    const mem: fixture.MemDir = .{ .files = &.{} };
    const pd = mem.packDir();
    const res = config.parse(&pd, abi.Str.of(cfg_raw), abi.Str.of("{}"), null, true, &parsed);
    try testing.expectEqual(abi.Verdict.ok, res);
    defer config.freeParsed(&parsed);

    const s = mlx.mlx_default_cpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    var why: abi.Verdict = .{};
    const m_opaque = plugin.table.load(&.{
        .model_dir = abi.Str.of(fixture_dir),
        .dir = &pd,
        .parsed = &parsed,
        .stream = .{ .ctx = s.ctx },
    }, &why) orelse {
        std.debug.print("load failed: {s}\n", .{why.why.slice()});
        return error.PluginLoadFailed;
    };
    defer plugin.table.unload(m_opaque);

    const ids = [_]u32{ 59, 40, 44, 56, 37, 49, 53, 16, 6, 21, 20, 55, 57, 3, 32, 52 };
    var out: abi.ForwardOut = .{};
    const v = plugin.table.forward(m_opaque, &.{
        .mode = .prefill,
        .logits = .all_rows,
        .rows = @intCast(ids.len),
        .tokens = abi.Span(u32).of(&ids),
        .positions = abi.Span(u32).of(&.{0}),
    }, &out);
    try testing.expect(v.isOk());
    defer _ = mlx.mlx_array_free(.{ .ctx = out.logits.ctx });

    const logits_arr: mlx.mlx_array = .{ .ctx = out.logits.ctx };
    try mlx.check(mlx.mlx_array_eval(logits_arr));
    const logits_data = mlx.mlx_array_data_float32(logits_arr).?;

    const io = std.Io.Threaded.global_single_threaded.io();
    var fx_weights = try weights_load.loadWeights(io, testing.allocator, fixture_dir);
    defer fx_weights.deinit();
    const ref_logits = fx_weights.get("logits") orelse return error.MissingRefLogits;
    try mlx.check(mlx.mlx_array_eval(ref_logits));
    const ref_data = mlx.mlx_array_data_float32(ref_logits).?;

    try testing.expectEqual(@as(usize, 16 * 64), @as(usize, @intCast(mlx.mlx_array_size(logits_arr))));
    for (0..16 * 64) |i| {
        const diff = @abs(logits_data[i] - ref_data[i]);
        try testing.expect(diff < 0.05);
    }
}
