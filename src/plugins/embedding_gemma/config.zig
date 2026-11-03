//! EmbeddingGemma (`gemma3_text` with `use_bidirectional_attention`, mlx-community's layout): the claim and the parse.
//! The Gemma 3 trunk read as the engine reads it (`gemma3/config.zig readGemma3`), an encoder: no chat, no KV cache,
//! prefill only (`caps.embed`). MLX-free.

const std = @import("std");
const abi = @import("../../abi/plugin.zig");
const gemma = @import("../gemma3/config.zig");

pub const plugin = @import("../../sdk/plugin.zig").Plugin{
    .name = "embedding_gemma",
    .kind = .model,
    .claims = &model_types,
    .gate_models = &.{"mlx-community/embeddinggemma-300m-8bit"},
};

pub const model_types = [_][]const u8{ "gemma3", "gemma3_text" };

pub const Config = gemma.Config;
pub const freeParsed = gemma.freeParsed;
pub const of = gemma.of;
pub const layout = gemma.encoderLayout;
const field = gemma.field;
const unsupported = gemma.unsupported;

/// A Gemma 3 pack trained bidirectionally, by its marker: outranks the gemma3 plugin's claim by type.
pub fn claims(model_type: abi.Str, config_json: abi.Str, _: *const abi.PackDir) callconv(.c) abi.Claim {
    const known = for (model_types) |t| {
        if (std.mem.eql(u8, t, model_type.slice())) break true;
    } else false;
    if (!known) return .no;
    var d: gemma.Doc = undefined;
    if (!gemma.Doc.open(config_json, &d).isOk()) return .no;
    defer d.close();
    return if (bidirectional(&d)) .by_marker else .no;
}

fn bidirectional(d: *const gemma.Doc) bool {
    const v = field(d.cfg, "use_bidirectional_attention") orelse return false;
    return v == .bool and v.bool;
}

pub fn parse(dir: *const abi.PackDir, config_json: abi.Str, settings_json: abi.Str, span: ?*const abi.LayerSpan, stub: bool, out: *abi.Parsed) callconv(.c) abi.Verdict {
    _ = settings_json;
    var d: gemma.Doc = undefined;
    const opened = gemma.Doc.open(config_json, &d);
    if (!opened.isOk()) return opened;
    defer d.close();
    if (span != null) return unsupported("a layer span (the cluster split)");
    if (claims(abi.Str.of(d.model_type), config_json, dir) == .no) return unsupported("not a bidirectional Gemma 3 pack");
    if (field(d.root, "vision_config") != null) return unsupported("a vision tower");
    if (gemma.foreignQuant(d.root)) return unsupported("a quantization_config naming a quant_method");
    var c: Config = .{};
    const v = gemma.readGemma3(&d, &c);
    if (!v.isOk()) return v;
    const pooling = gemma.readPooling(dir, d.root, stub) catch return abi.Verdict.refuse(.failed, "gemma: the pooling contract names a mode this plugin does not implement");
    const fin = gemma.finish(dir, d.root, c, stub, .{ .embed = true, .prefix_cache = false }, out);
    if (!fin.isOk()) return fin;
    out.embed.pooling = pooling;
    return .ok;
}

// ── tests ──

const testing = std.testing;
const fixture = @import("../../sdk/root.zig").fixture;

pub fn parseJson(json: []const u8, out: *abi.Parsed) abi.Verdict {
    const mem: fixture.MemDir = .{ .files = &.{.{ .name = "config.json", .bytes = json }} };
    const dir = mem.packDir();
    return parse(&dir, abi.Str.of(json), .{}, null, true, out);
}

test "embedding_gemma config: the bidirectional marker claims and parses an encoder; a chat gemma3_text is not claimed" {
    const json =
        \\{"model_type":"gemma3_text","use_bidirectional_attention":true,"hidden_size":768,"num_hidden_layers":24,
        \\"num_attention_heads":3,"num_key_value_heads":1,"head_dim":256,"intermediate_size":1152,"sliding_window":512,
        \\"bos_token_id":2,"eos_token_id":1,"vocab_size":262144,"pooling_mode":"mean","quantization":{"group_size":64,"bits":8}}
    ;
    const dir: abi.PackDir = undefined;
    try testing.expectEqual(abi.Claim.by_marker, claims(.of("gemma3_text"), .of(json), &dir));
    try testing.expectEqual(abi.Claim.no, claims(.of("gemma3_text"), .of("{\"model_type\":\"gemma3_text\"}"), &dir));
    var p: abi.Parsed = .{};
    try testing.expect(parseJson(json, &p).isOk());
    defer freeParsed(&p);
    try testing.expect(p.caps.embed and !p.caps.prefix_cache and !p.caps.batches_decode);
    try testing.expectEqual(abi.Pooling.mean, p.embed.pooling);
    try testing.expectEqual(@as(u32, 2), p.shell.bos);
    try testing.expect(of(&p).norm_has_offset);
}
