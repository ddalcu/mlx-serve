//! DeepSeek-V4.1-Flash's claims and parse: V4's family config plus the KV/index sources, the candidate source, Engram,
//! the DSpark head's own MoE size and the release's vision tower. MLX-free (discovery runs it).

const std = @import("std");
const abi = @import("../../abi/plugin.zig");
const sdk = @import("../../sdk/root.zig");
const v4 = @import("../deepseek_v4/config.zig");

pub const plugin = @import("../../sdk/plugin.zig").Plugin{
    .name = "deepseek_v41",
    .kind = .model,
    .claims = &model_types,
    .gate_models = &.{"ddalcu/DeepSeek-V4.1-Flash-1017-iQ-MLX-3.0bpw"},
};

pub const model_types = [_][]const u8{"deepseek_v41"};

pub const claims = sdk.claimsByType(&model_types);

pub const Config = v4.Config;
pub const of = v4.of;
pub const freeParsed = v4.freeParsed;

const field = v4.field;
const value = v4.value;
const int = v4.int;
const float = v4.float;
const ParseError = v4.ParseError;

pub fn parse(dir: *const abi.PackDir, config_json: abi.Str, settings_json: abi.Str, span: ?*const abi.LayerSpan, stub: bool, out: *abi.Parsed) callconv(.c) abi.Verdict {
    _ = span;
    const result = v4.parseInto(dir, config_json, stub, true, streamsPack(settings_json.slice()), out, &extend);
    if (result.isOk()) out.caps.prefix_cache = !of(out).dsv41_streamed;
    return result;
}

/// The per-model `stream_experts` setting (`true` or a row count) asks for an MLX pack's routed experts to stream.
fn streamsPack(settings_json: []const u8) bool {
    if (settings_json.len == 0) return false;
    const parsed = std.json.parseFromSlice(std.json.Value, std.heap.page_allocator, settings_json, .{}) catch return false;
    defer parsed.deinit();
    if (parsed.value != .object) return false;
    const v = parsed.value.object.get("stream_experts") orelse return false;
    return switch (v) {
        .bool => |b| b,
        .integer => |n| n > 0,
        else => false,
    };
}

/// V4.1's keys past the shared arm. Every compressed layer must find a source of its own ratio at or above it: the
/// reference reads whichever source ran last, so a mismatch would attend over the wrong cache.
fn extend(c: *Config, j: v4.Json) ParseError!void {
    try parseVision(c, j.root);
    const o = j.cfg;
    const refuse = error.UnsupportedDsv41Config;
    const n_layers = c.num_hidden_layers;
    if (n_layers == 0 or n_layers > 64 or c.dsv4_n_compress_ratios < n_layers) return refuse;
    if (c.dsv4_hc_mult != 4) return refuse;
    c.dsv41_kv_sources = try layerMask(o, "kv_source_layer_ids", n_layers);
    c.dsv41_index_sources = try layerMask(o, "index_source_layer_ids", n_layers);
    if (field(o, "candidate_source_layer_id")) |v| c.dsv41_candidate_source = std.math.cast(i32, try value(.integer, v)) orelse return error.InvalidConfigField;
    if (field(o, "candidate_topk_blocks")) |v| c.dsv41_candidate_topk_blocks = try int(u32, v);
    if (field(o, "candidate_block_size")) |v| c.dsv41_candidate_block_size = try int(u32, v);

    const ratios = c.dsv4_compress_ratios[0..n_layers];
    var last_kv: ?usize = null;
    var last_owner: ?usize = null;
    var last_index: ?usize = null;
    for (ratios, 0..) |r, l| {
        if (r > 2) return refuse;
        const kv = has(c.dsv41_kv_sources, l);
        const ix = has(c.dsv41_index_sources, l);
        if (r == 0) {
            if (kv or ix) return refuse;
            continue;
        }
        if (kv) last_kv = l;
        if (kv and ix) last_owner = l;
        if (ix) last_index = l;
        const src = last_kv orelse return refuse;
        if (ratios[src] != r) return refuse;
        if (ix) {
            const owner = last_owner orelse return refuse;
            if (ratios[owner] != r) return refuse;
        } else if (last_index == null) return refuse;
    }
    if (c.dsv41_candidate_source >= 0) {
        const cs: usize = @intCast(c.dsv41_candidate_source);
        if (cs >= n_layers or !has(c.dsv41_index_sources, cs)) return refuse;
        if (c.dsv41_candidate_topk_blocks == 0 or c.dsv41_candidate_block_size == 0) return refuse;
    }
    if (field(o, "engram_layer_ids")) |v| {
        const items = (try value(.array, v)).items;
        if (items.len > c.dsv41_engram_layers.len) return refuse;
        for (items, 0..) |item, i| {
            const l = try int(u32, item);
            if (l >= n_layers) return refuse;
            c.dsv41_engram_layers[i] = @intCast(l);
        }
        c.dsv41_n_engram_layers = @intCast(items.len);
    }
    if (c.dsv41_n_engram_layers > 0) {
        const rows = (try value(.array, field(o, "engram_num_embeddings") orelse return refuse)).items;
        if (rows.len != c.dsv41_n_engram_layers) return refuse;
        for (rows, 0..) |item, i| c.dsv41_engram_rows[i] = std.math.cast(u64, try value(.integer, item)) orelse return error.InvalidConfigField;
        if (field(o, "engram_max_ngram_size")) |x| c.dsv41_engram_max_ngram = try int(u32, x);
        if (field(o, "engram_vocab_size")) |x| c.dsv41_engram_vocab_size = std.math.cast(u64, try value(.integer, x)) orelse return error.InvalidConfigField;
        if (field(o, "engram_n_heads")) |x| c.dsv41_engram_n_heads = try int(u32, x);
        if (field(o, "engram_head_dim")) |x| c.dsv41_engram_head_dim = try int(u32, x);
        if (field(o, "engram_pad_token_id")) |x| c.dsv41_engram_pad_id = try int(u32, x);
        if (field(o, "engram_compressed_vocab_size")) |x| c.dsv41_engram_compressed_vocab = try int(u32, x);
        if (c.dsv41_engram_max_ngram < 2 or c.dsv41_engram_max_ngram > 8 or
            c.dsv41_engram_n_heads == 0 or c.dsv41_engram_n_heads > 16 or
            c.dsv41_engram_head_dim == 0 or c.dsv41_engram_head_dim % 32 != 0 or
            c.dsv41_engram_compressed_vocab == 0 or c.dsv41_engram_vocab_size == 0) return refuse;
    }
    if (field(o, "dspark_n_routed_experts")) |v| c.dsv41_dspark_experts = try int(u32, v);
    if (field(o, "dspark_num_experts_per_tok")) |v| c.dsv41_dspark_top_k = try int(u32, v);
    for (c.dsv4_dspark_target_layers[0..c.dsv4_n_dspark_target_layers]) |l| if (l >= n_layers) return refuse;
}

/// The release's tower splices at `<｜deepseek_image｜>`: a pack without `vision_config` or the image token serves text.
fn parseVision(c: *Config, root: std.json.ObjectMap) ParseError!void {
    c.dsv41_vision = c.has_vision and c.image_token_id != 0;
    c.has_vision = c.dsv41_vision;
    if (!c.dsv41_vision) return;
    const vc = field(root, "vision_config").?.object;
    c.vision_rope_theta = 10000.0;
    if (field(vc, "rope_theta")) |v| c.vision_rope_theta = try float(v);
    if (field(vc, "downsample_ratio")) |v| c.dv_downsample = try int(u32, v);
    if (field(vc, "max_image_tokens")) |v| c.dv_max_image_tokens = try int(u32, v);
    if (field(vc, "min_pixels")) |v| c.dv_min_pixels = try int(u32, v);
    if (field(vc, "max_wh_ratio")) |v| if (v == .integer) {
        c.dv_max_wh_ratio = try int(u32, v);
    };
    // The 2-D rotary splits a head into h and w halves of rotate_half pairs.
    if (c.vision_num_heads == 0 or c.vision_hidden_size % c.vision_num_heads != 0 or
        (c.vision_hidden_size / c.vision_num_heads) % 4 != 0 or c.vision_patch_size == 0 or
        c.dv_downsample == 0 or c.dv_max_image_tokens < 4) return error.InvalidConfigField;
}

fn has(mask: u64, l: usize) bool {
    return (mask >> @intCast(l)) & 1 != 0;
}

fn layerMask(o: std.json.ObjectMap, key: []const u8, n: u32) ParseError!u64 {
    var mask: u64 = 0;
    const v = field(o, key) orelse return 0;
    for ((try value(.array, v)).items) |item| {
        const l = try int(u32, item);
        if (l >= n) return error.UnsupportedDsv41Config;
        mask |= @as(u64, 1) << @intCast(l);
    }
    return mask;
}

// ── tests ──

const testing = std.testing;
const fixture = sdk.fixture;

fn parseFiles(files: []const fixture.File, stub: bool, out: *abi.Parsed) abi.Verdict {
    const mem: fixture.MemDir = .{ .files = files };
    const dir = mem.packDir();
    return parse(&dir, abi.Str.of(files[0].bytes), .{}, null, stub, out);
}

fn parseWithSettings(settings: []const u8, out: *abi.Parsed) abi.Verdict {
    const mem: fixture.MemDir = .{ .files = &.{.{ .name = "config.json", .bytes = iq3 }} };
    const dir = mem.packDir();
    return parse(&dir, abi.Str.of(iq3), abi.Str.of(settings), null, true, out);
}

const iq3 = @embedFile("../../fixtures/model-configs/deepseek_v41-iq-mlx-3.0bpw.json");

test "deepseek_v41 config: the release pack parses with its sources, Engram, DSpark head and tower" {
    var p: abi.Parsed = .{};
    try testing.expect(parseFiles(&.{.{ .name = "config.json", .bytes = iq3 }}, true, &p).isOk());
    defer freeParsed(&p);
    const c = of(&p);
    try testing.expectEqual(@as(u64, (1 << 2) | (1 << 8) | (1 << 14) | (1 << 20)), c.dsv41_kv_sources);
    try testing.expectEqual(@as(u32, 2), c.dsv41_n_engram_layers);
    try testing.expectEqual(@as(u32, 128), c.dsv41_dspark_experts);
    try testing.expect(c.dsv41_vision and p.caps.images and !p.caps.spec_rollback);
    try testing.expect(p.caps.slot_state and p.caps.batches_decode and p.caps.owns_decode_state);
    try testing.expect(p.caps.prefix_cache);
    try testing.expectEqual(@as(u32, 129264), p.placeholder_ids.image);
    try testing.expectEqual(abi.ThinkingDefault.on, p.chat.thinking_default);
    try testing.expectEqual(abi.EffortVocab.dsv41, p.chat.effort_vocab);
    try testing.expectEqual(@as(u32, 3), p.spec.native_stages);
}

test "deepseek_v41 config: an experts.bin bank streams, its prompt passed whole" {
    var p: abi.Parsed = .{};
    const v = parseFiles(&.{ .{ .name = "config.json", .bytes = iq3 }, .{ .name = "experts.bin", .bytes = "" } }, true, &p);
    try testing.expect(v.isOk());
    defer freeParsed(&p);
    try testing.expect(of(&p).dsv41_streamed and of(&p).dsv41_exl3_bank and p.caps.whole_prompt_prefill and p.caps.streamed_weights);
    try testing.expect(!p.caps.slot_state and !p.caps.batches_decode);
    try testing.expect(!p.caps.prefix_cache);
    var q: abi.Parsed = .{};
    try testing.expect(parseFiles(&.{.{ .name = "config.json", .bytes = iq3 }}, true, &q).isOk());
    defer freeParsed(&q);
    try testing.expect(!of(&q).dsv41_streamed and !q.caps.whole_prompt_prefill);
}

test "deepseek_v41 config: stream_experts streams an MLX pack that has no experts.bin" {
    for ([_]struct { settings: []const u8, streams: bool }{
        .{ .settings = "{\"stream_experts\":true}", .streams = true },
        .{ .settings = "{\"stream_experts\":160}", .streams = true },
        .{ .settings = "{\"stream_experts\":0}", .streams = false },
        .{ .settings = "{\"mtp\":false}", .streams = false },
        .{ .settings = "", .streams = false },
    }) |c| {
        var p: abi.Parsed = .{};
        try testing.expect(parseWithSettings(c.settings, &p).isOk());
        defer freeParsed(&p);
        try testing.expectEqual(c.streams, of(&p).dsv41_streamed);
        try testing.expectEqual(c.streams, p.caps.whole_prompt_prefill);
        try testing.expect(!of(&p).dsv41_exl3_bank);
    }
}

test "deepseek_v41 config: a layer that would read another ratio's cache is refused by name" {
    const bad =
        \\{"model_type":"deepseek_v41","text_config":{"num_hidden_layers":3,"num_key_value_heads":1,"hc_mult":4,
        \\ "compress_ratios":[2,1,1],"kv_source_layer_ids":[0],"index_source_layer_ids":[0]}}
    ;
    var p: abi.Parsed = .{};
    const v = parseFiles(&.{.{ .name = "config.json", .bytes = bad }}, true, &p);
    try testing.expect(!v.isOk() and v.code == .unsupported and p.private == null);
}
