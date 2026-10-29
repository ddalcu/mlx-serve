//! The engine's `ModelConfig` from the family's `Config`, for the layered loader and the forward helpers that still
//! read it: the same names one to one, the pack's sampling defaults, the Sushi EXL3 spec. Pinned field for field
//! against the engine's own parse (both plugins).

const std = @import("std");
const model = @import("../../sdk/config/model.zig");
const exl3 = @import("../../sdk/quant/exl3/format.zig");
const Config = @import("config.zig").Config;

fn opt(x: f32) ?f32 {
    return if (std.math.isNan(x)) null else x;
}

pub fn toModelConfig(c: *const Config) !model.ModelConfig {
    var m: model.ModelConfig = .{};
    inline for (@typeInfo(Config).@"struct".field_names) |name| {
        if (comptime std.mem.eql(u8, name, "gen") or std.mem.eql(u8, name, "expert_quant_json") or std.mem.eql(u8, name, "streamed") or std.mem.startsWith(u8, name, "advertises_")) continue;
        const src = @field(c, name);
        const Dst = @TypeOf(@field(m, name));
        @field(m, name) = if (@TypeOf(src) == Dst) src else std.meta.stringToEnum(Dst, @tagName(src)).?;
    }
    m.hidden_act = .silu;
    m.norm_has_offset = false;
    m.legacy.thinking = .on;
    if (c.isGlm5()) {
        m.legacy.caps.batch_pads_kv = false;
        m.legacy.caps.dflash_policy = true;
    }
    m.expert_bank = c.streamed;
    m.scale_embeddings = false;
    m.has_pre_ff_norm = false;
    m.has_qk_norm = false;
    m.gen_temperature = opt(c.gen.temperature);
    m.gen_top_p = opt(c.gen.top_p);
    m.gen_min_p = opt(c.gen.min_p);
    m.gen_top_k = if (c.gen.top_k > 0) @intCast(c.gen.top_k) else null;
    m.gen_enable_thinking = switch (c.gen.enable_thinking) {
        .unset => null,
        .on => true,
        .off => false,
    };
    if (c.expert_quant_json.len > 0) {
        var arena = std.heap.ArenaAllocator.init(std.heap.c_allocator);
        defer arena.deinit();
        const text = try std.fmt.allocPrint(arena.allocator(), "{{\"expert_quant\":{s}}}", .{c.expert_quant_json});
        const root = try std.json.parseFromSliceLeaky(std.json.Value, arena.allocator(), text, .{});
        m.exl3 = try exl3.parse(root.object, m.num_experts_per_tok);
    }
    return m;
}

// ── tests ──

const testing = std.testing;
const sdk = @import("../../sdk/root.zig");
const abi = @import("../../abi/plugin.zig");
const glm_config = @import("config.zig");
const ling_config = @import("../bailing_hybrid/config.zig");

const Parse = *const fn (*const abi.PackDir, abi.Str, abi.Str, ?*const abi.LayerSpan, bool, *abi.Parsed) callconv(.c) abi.Verdict;

/// The plugin's stub parse, through `toModelConfig`, equals the engine's parse of the same bytes in every field.
pub fn expectEngineParse(json: []const u8, parse: Parse) !void {
    var want = try model.parseConfigFromJson(testing.allocator, json);
    defer want.deinit(testing.allocator);
    const mem: sdk.fixture.MemDir = .{ .files = &.{.{ .name = "config.json", .bytes = json }} };
    const dir = mem.packDir();
    var p: abi.Parsed = .{};
    const v = parse(&dir, .of(json), .{}, null, true, &p);
    if (!v.isOk()) std.debug.print("parse: {s}\n", .{v.why.slice()});
    try testing.expect(v.isOk());
    defer glm_config.freeParsed(&p);
    var got = try toModelConfig(glm_config.of(&p));
    defer got.deinit(testing.allocator);
    try testing.expectEqualDeep(want, got);
}

/// What discovery reads of a served pack (`discovery.applyServedStub`) equals its legacy reading of the same bytes,
/// but a ModelOpt pack's bits, which now name what its loaded row names.
fn expectDiscoveryStub(json: []const u8, parse: Parse) !void {
    const want = @import("../../host/registry/discovery.zig").parseStubMeta(testing.allocator, json, true);
    const mem: sdk.fixture.MemDir = .{ .files = &.{.{ .name = "config.json", .bytes = json }} };
    const dir = mem.packDir();
    var p: abi.Parsed = .{};
    try testing.expect(parse(&dir, .of(json), .{}, null, true, &p).isOk());
    defer glm_config.freeParsed(&p);
    try testing.expectEqual(want.vocab_size, p.shell.vocab_size);
    try testing.expectEqual(want.hidden_size, p.shell.hidden_size);
    try testing.expectEqual(want.num_hidden_layers, p.shell.num_hidden_layers);
    try testing.expectEqual(want.max_position_embeddings, std.math.cast(u32, p.shell.max_position_embeddings) orelse std.math.maxInt(u32));
    // A ModelOpt pack names no `quantization.bits`: legacy discovery read 0, its loaded row (`config.quant_bits`) 4.
    try testing.expectEqual(if (glm_config.of(&p).modelopt) 4 else want.quant_bits, p.shell.quant.bits);
    try testing.expectEqual(want.is_moe, p.caps.moe);
    try testing.expectEqual(want.has_vision, p.caps.images);
    try testing.expectEqual(want.has_video, p.caps.video);
}

test "glm plugins: discovery reads every fixture pack as the legacy stub reading does" {
    for ([_][]const u8{
        @embedFile("../../fixtures/model-configs/glm5_next-mlx-community-4bit.json"),
        @embedFile("../../fixtures/model-configs/glm5_next-nvidia-fp4-local.json"),
        @embedFile("../../fixtures/model-configs/glm5_next-sushi-2.4bpw.json"),
    }) |json| try expectDiscoveryStub(json, &glm_config.parse);
    // A `_text` pack keeping its vestigial vision block reads as text only.
    const text = try std.mem.replaceOwned(u8, testing.allocator, @embedFile("../../fixtures/model-configs/glm5_next-mlx-community-4bit.json"), "\"model_type\": \"glm5_next\"", "\"model_type\": \"glm5_next_text\"");
    defer testing.allocator.free(text);
    try testing.expect(std.mem.indexOf(u8, text, "glm5_next_text") != null);
    try expectEngineParse(text, &glm_config.parse);
    try expectDiscoveryStub(text, &glm_config.parse);
    for ([_][]const u8{
        @embedFile("../../fixtures/model-configs/bailing_hybrid-ling-3.0-tiny-4bit.json"),
        @embedFile("../../fixtures/model-configs/bailing_hybrid-ling-3.0-flash-oq4e-mtp.json"),
    }) |json| try expectDiscoveryStub(json, &ling_config.parse);
}

test "glm5_next parse: equals the engine's ModelConfig on every GLM-5.3-Flash fixture (MLX, ModelOpt FP4, Sushi)" {
    try expectEngineParse(@embedFile("../../fixtures/model-configs/glm5_next-mlx-community-4bit.json"), &glm_config.parse);
    try expectEngineParse(@embedFile("../../fixtures/model-configs/glm5_next-nvidia-fp4-local.json"), &glm_config.parse);
    try expectEngineParse(@embedFile("../../fixtures/model-configs/glm5_next-sushi-2.4bpw.json"), &glm_config.parse);
    // Flat text-only, explicit kda bound and conv width, nulls read as omitted keys.
    try expectEngineParse(
        \\{"model_type":"glm5_next_text","hidden_size":64,"num_hidden_layers":8,"num_attention_heads":4,
        \\ "num_key_value_heads":2,"vocab_size":128,"q_lora_rank":32,"kv_lora_rank":64,"qk_nope_head_dim":16,
        \\ "v_head_dim":16,"index_topk":8,"index_kpool":4,"index_n_heads":2,"index_head_dim":16,"n_routed_experts":4,
        \\ "num_experts_per_tok":2,"moe_intermediate_size":32,"n_shared_experts":1,"first_k_dense_replace":2,
        \\ "linear_attn_config":{"num_heads":2,"head_dim":32,"short_conv_kernel_size":4,"gate_lower_bound":-4.0},
        \\ "swiglu_limit":null,"eos_token_id":[1,2],"tie_word_embeddings":true,"rope_theta":null}
    , &glm_config.parse);
}

test "bailing_hybrid parse: equals the engine's ModelConfig on every Ling 3.0 fixture" {
    try expectEngineParse(@embedFile("../../fixtures/model-configs/bailing_hybrid-ling-3.0-tiny-4bit.json"), &ling_config.parse);
    try expectEngineParse(@embedFile("../../fixtures/model-configs/bailing_hybrid-ling-3.0-flash-oq4e-mtp.json"), &ling_config.parse);
}

test "glm parse: what the engine refuses the plugins refuse" {
    inline for (.{
        .{ "{\"model_type\":\"glm5_next_text\",\"num_hidden_layers\":2,\"kv_lora_rank\":64,\"index_kpool\":4,\"index_topk\":8,\"qk_rope_head_dim\":64}", false },
        .{ "{\"model_type\":\"glm5_next_text\",\"num_hidden_layers\":2,\"index_kpool\":4,\"index_topk\":6}", false },
        .{ "{\"model_type\":\"glm5_next_text\",\"num_hidden_layers\":2,\"index_kpool\":4,\"index_topk\":8,\"hc_mult\":2}", false },
        .{ "{\"model_type\":\"glm5_next_text\",\"num_hidden_layers\":4,\"index_kpool\":4,\"index_topk\":8,\"layer_types\":[\"linear_attention\",\"deepseek_sparse_attention\",\"linear_attention\",\"linear_attention\"]}", false },
        .{ "{\"model_type\":\"bailing_hybrid\",\"kv_lora_rank\":64,\"kda_lower_bound\":0.5}", true },
        .{ "{\"model_type\":\"bailing_hybrid\",\"kv_lora_rank\":64,\"qk_nope_head_dim\":8,\"qk_rope_head_dim\":8,\"qk_head_dim\":20}", true },
        .{ "{\"model_type\":\"bailing_hybrid\",\"kv_lora_rank\":64,\"gated_attention_proj_granularity_type\":\"element_wise\"}", true },
    }) |case| {
        try testing.expect(std.meta.isError(model.parseConfigFromJson(testing.allocator, case[0])));
        const mem: sdk.fixture.MemDir = .{ .files = &.{.{ .name = "config.json", .bytes = case[0] }} };
        const dir = mem.packDir();
        var p: abi.Parsed = .{};
        const v = if (case[1]) ling_config.parse(&dir, .of(case[0]), .{}, null, true, &p) else glm_config.parse(&dir, .of(case[0]), .{}, null, true, &p);
        try testing.expect(!v.isOk() and p.private == null);
    }
}

test "glm5_next parse: the load's parse reads the processor and generation sidecars as the engine's parseConfig does" {
    const io = testing.io;
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    const cfg = @embedFile("../../fixtures/model-configs/glm5_next-mlx-community-4bit.json");
    const files = [_]sdk.fixture.File{
        .{ .name = "config.json", .bytes = cfg },
        .{ .name = "processor_config.json", .bytes = "{\"image_processor\":{\"min_image_tokens\":32,\"max_image_tokens\":2048,\"size\":{\"shortest_edge\":3136,\"longest_edge\":1003520}}}" },
        .{ .name = "generation_config.json", .bytes = "{\"temperature\":0.7,\"top_p\":0.95,\"eos_token_id\":[154820,154827,154829]}" },
    };
    for (files) |f| try tmp.dir.writeFile(io, .{ .sub_path = f.name, .data = f.bytes });
    var buf: [512]u8 = undefined;
    const path = buf[0..try tmp.dir.realPath(io, &buf)];
    var want = try model.parseConfigLegacy(io, testing.allocator, path);
    defer want.deinit(testing.allocator);
    const mem: sdk.fixture.MemDir = .{ .files = &files };
    const dir = mem.packDir();
    var p: abi.Parsed = .{};
    try testing.expect(glm_config.parse(&dir, .of(cfg), .{}, null, false, &p).isOk());
    defer glm_config.freeParsed(&p);
    var got = try toModelConfig(glm_config.of(&p));
    defer got.deinit(testing.allocator);
    try testing.expectEqual(@as(u32, 2048), got.glmv_max_image_tokens);
    try testing.expectEqualDeep(want, got);
}

/// The plugin's bill terms equal the engine's arithmetic over its own parse of `json`.
fn expectBill(json: []const u8, comptime P: type) !void {
    var want = try model.parseConfigFromJson(testing.allocator, json);
    defer want.deinit(testing.allocator);
    const mem: sdk.fixture.MemDir = .{ .files = &.{.{ .name = "config.json", .bytes = json }} };
    const dir = mem.packDir();
    var p: abi.Parsed = .{};
    try testing.expect(P.table.parse(&dir, .of(json), .{}, null, true, &p).isOk());
    defer P.table.free_parsed(&p);
    var b: abi.Bill = .{};
    P.table.bill(&p, &.{}, &b);
    try testing.expectEqual(want.kvBytesPerToken(), b.kv_per_token);
    try testing.expectEqual(want.ssmCheckpointBytes(), b.ssm_checkpoint);
    try testing.expectEqual(want.prefillScoreHeadDim(), b.prefill_score_width);
}

test "glm plugins bill: per-token KV, the SSM checkpoint and the score width equal the engine's on every fixture" {
    const glm = @import("plugin.zig");
    const ling = @import("../bailing_hybrid/plugin.zig");
    try expectBill(@embedFile("../../fixtures/model-configs/glm5_next-mlx-community-4bit.json"), glm);
    try expectBill(@embedFile("../../fixtures/model-configs/glm5_next-nvidia-fp4-local.json"), glm);
    try expectBill(@embedFile("../../fixtures/model-configs/glm5_next-sushi-2.4bpw.json"), glm);
    try expectBill(@embedFile("../../fixtures/model-configs/bailing_hybrid-ling-3.0-tiny-4bit.json"), ling);
    try expectBill(@embedFile("../../fixtures/model-configs/bailing_hybrid-ling-3.0-flash-oq4e-mtp.json"), ling);
}
