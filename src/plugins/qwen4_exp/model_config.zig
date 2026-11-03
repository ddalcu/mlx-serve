//! The engine's `ModelConfig` from the family's `Config`, for the trunk module and the weight loader that still read
//! it: the same names one to one plus the arch's constants. Pinned field for field against the engine's own parse.

const std = @import("std");
const model = @import("../../sdk/config/model.zig");
const exl3 = @import("../../sdk/quant/exl3/format.zig");
const qwen4_exp = @import("qwen4_exp.zig");
const Config = @import("config.zig").Config;

fn convert(comptime Dst: type, src: anytype) Dst {
    const Src = @TypeOf(src);
    if (Src == Dst) return src;
    return switch (@typeInfo(Src)) {
        .optional => if (src) |v| convert(@typeInfo(Dst).optional.child, v) else null,
        else => std.meta.stringToEnum(Dst, @tagName(src)).?,
    };
}

/// `dir` (the pack's folder, absent in a stub) places the n-gram table the way the engine's `parseConfig` does:
/// `ngram_table.bin` beside the weights, or the shards the index names, measured. `ModelConfig.deinit` frees it.
pub fn toModelConfig(a: std.mem.Allocator, c: *const Config, dir: ?[]const u8) !model.ModelConfig {
    var m: model.ModelConfig = .{};
    inline for (@typeInfo(Config).@"struct".field_names) |name| {
        if (comptime std.mem.eql(u8, name, "embedded_ngram") or std.mem.eql(u8, name, "mrope_inv_freq") or std.mem.eql(u8, name, "expert_quant")) continue;
        @field(m, name) = convert(@TypeOf(@field(m, name)), @field(c, name));
    }
    m.model_type = "qwen4_exp";
    m.legacy.caps.long_ctx = true;
    m.weight_prefix = "language_model.model";
    m.norm_has_offset = false;
    m.has_final_norm = false; // the hyper-connection mixer replaces model.norm
    m.scale_embeddings = false;
    m.has_pre_ff_norm = false;
    m.has_qk_norm = true;
    m.hidden_act = .silu;
    m.has_sliding_window = false;
    m.attn_output_gate = true;
    m.kda_sigmoid_out_gate = true; // output_gate_type "sigmoid"
    m.rope_scaling_factor = 1.0;
    m.rope_local_base_freq = m.rope_theta;
    if (dir) |d| {
        m.ngram_table_path = try std.fmt.allocPrint(a, "{s}/ngram_table.bin", .{d});
        errdefer m.deinit(a);
        if (try qwen4_exp.inspectEmbedded(d, try model.qwen4EmbeddedSpec(&m))) |info| m.embedded_ple_payload_bytes = info.payload_bytes;
        if (c.expert_quant) m.exl3 = try sushiSpec(a, d, m.num_experts_per_tok);
    }
    return m;
}


/// A Sushi pack's `expert_quant` (config.json in `dir`): its EXL3 banks' codebook, window and rate.
fn sushiSpec(a: std.mem.Allocator, dir: []const u8, top_k: u32) !exl3.Spec {
    const io = std.Io.Threaded.global_single_threaded.io();
    var d = try std.Io.Dir.cwd().openDir(io, dir, .{});
    defer d.close(io);
    const bytes = try d.readFileAlloc(io, "config.json", a, .limited(16 * 1024 * 1024));
    defer a.free(bytes);
    const doc = try std.json.parseFromSlice(std.json.Value, a, bytes, .{});
    defer doc.deinit();
    if (doc.value != .object) return error.InvalidConfigJson;
    return exl3.parse(doc.value.object, top_k);
}

// ── tests ──

const testing = std.testing;
const abi = @import("../../abi/plugin.zig");
const fixture = @import("../../sdk/testing/fixture.zig");
const config = @import("config.zig");

test "qwen4_exp model config: the parse bounds are the n-gram hash's arrays" {
    try testing.expectEqual(@as(usize, config.max_ngram_size), qwen4_exp.MAX_NGRAM_SIZE);
    try testing.expectEqual(@as(usize, config.max_heads), qwen4_exp.MAX_HEADS);
}

/// The plugin's parse of `files` (staged on disk for the engine), through `toModelConfig`, equals the engine's
/// `parseConfig` of the same folder in every field.
fn expectEngineParse(files: []const fixture.File) !void {
    const io = testing.io;
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    for (files) |f| try tmp.dir.writeFile(io, .{ .sub_path = f.name, .data = f.bytes });
    var buf: [std.fs.max_path_bytes]u8 = undefined;
    const path = buf[0..try tmp.dir.realPath(io, &buf)];
    var want = try model.parseConfigLegacy(io, testing.allocator, path);
    defer want.deinit(testing.allocator);
    const mem: fixture.MemDir = .{ .files = files };
    const pd = mem.packDir();
    var p: abi.Parsed = .{};
    const v = config.parse(&pd, .of(files[0].bytes), .{}, null, false, &p);
    if (!v.isOk()) {
        std.debug.print("parse: {s}\n", .{v.why.slice()});
        return error.Refused;
    }
    defer config.freeParsed(&p);
    var got = try toModelConfig(testing.allocator, config.of(&p), path);
    defer got.deinit(testing.allocator);
    try testing.expectEqualDeep(want, got);
    try testing.expectEqual(want.contextCap(), @as(u32, @intCast(p.shell.context_cap)));
    try testing.expectEqualSlices(u32, want.eosTokenSlice(), p.shell.eos[0..p.shell.eos_count]);
    try testing.expectEqual(want.ssmCheckpointBytes(), p.cache.ssm_state.checkpoint_bytes);
    try testing.expectEqual(want.attnCacheLayerCount(), config.of(&p).attnLayers());
    try testing.expectEqualStrings(want.cacheLayoutNamespace() orelse "", p.cache.layout_namespace.slice());
}

const iq_pack = @embedFile("../../fixtures/model-configs/qwen4_exp-iq-mlx-4.7bpw.json");
const iq_generation =
    \\{"bos_token_id": 248044, "do_sample": true, "eos_token_id": [248046, 248044], "pad_token_id": 248044,
    \\ "temperature": 1.0, "top_k": 20, "top_p": 0.95}
;
const nvidia_pack = @embedFile("../../fixtures/model-configs/qwen4_exp-nvidia-fp4.json");
const nvidia_processor =
    \\{"size": {"longest_edge": 16777216, "shortest_edge": 65536}, "patch_size": 16, "temporal_patch_size": 2,
    \\ "merge_size": 2, "processor_class": "Qwen3VLProcessor"}
;

test "qwen4_exp model config: the iQ-MLX 4.7bpw pack equals the engine's parse, with its generation config" {
    try expectEngineParse(&.{ .{ .name = "config.json", .bytes = iq_pack }, .{ .name = "generation_config.json", .bytes = iq_generation } });
}

test "qwen4_exp model config: NVIDIA's ModelOpt FP4 pack equals the engine's parse, with its processor bounds" {
    try expectEngineParse(&.{ .{ .name = "config.json", .bytes = nvidia_pack }, .{ .name = "preprocessor_config.json", .bytes = nvidia_processor } });
}

test "qwen4_exp model config: the flat text pack and a YaRN window equal the engine's parse" {
    try expectEngineParse(&.{.{ .name = "config.json", .bytes = config.tiny_text }});
    const yarn =
        \\{"model_type":"qwen4_exp","text_config":{"vocab_size":128,"hidden_size":64,"num_hidden_layers":4,
        \\ "num_attention_heads":4,"num_key_value_heads":2,"head_dim":32,"full_attention_interval":2,"layer_types":["linear_attention",
        \\ "full_attention","linear_attention","full_attention"],"linear_num_key_heads":2,"linear_num_value_heads":4,
        \\ "linear_key_head_dim":16,"linear_value_head_dim":16,"num_experts":4,"num_experts_per_tok":2,
        \\ "moe_intermediate_size":32,"ple_layer_ids":[1],"ple_embed_dim":64,"heads_per_ngram":2,"hc_count":2,
        \\ "max_position_embeddings":262144,"eos_token_id":[7,9],"rope_parameters":{"rope_type":"yarn","factor":4.0,
        \\ "original_max_position_embeddings":262144,"rope_theta":10000000,"partial_rotary_factor":0.25,
        \\ "mrope_section":[2,1,1],"mrope_interleaved":true}},"qwen4_norm_convention":"delta",
        \\ "quantization":{"bits":8,"group_size":32,"mode":"affine"},"vision_config":{"depth":2,"hidden_size":32,
        \\ "num_heads":2,"intermediate_size":64,"patch_size":16,"out_hidden_size":64}}
    ;
    try expectEngineParse(&.{
        .{ .name = "config.json", .bytes = yarn },
        .{ .name = "processor_config.json", .bytes = "{\"image_processor\":{\"min_pixels\":4096,\"max_pixels\":65536}}" },
        .{ .name = "generation_config.json", .bytes = "{\"eos_token_id\":[9,11],\"default_chat_template_kwargs\":{\"enable_thinking\":false}}" },
    });
}

test "qwen4_exp model config: an index naming n-gram shards measures the embedded table as the engine does" {
    const io = testing.io;
    var td = testing.tmpDir(.{});
    defer td.cleanup();
    try @import("ple_table.zig").writeFixture(&td, .valid);
    const cfg =
        \\{"model_type":"qwen4_exp_text","vocab_size":16,"hidden_size":64,"num_hidden_layers":2,"head_dim":32,
        \\ "full_attention_interval":2,"linear_num_key_heads":2,"linear_num_value_heads":2,"ple_layer_ids":[2],
        \\ "ple_embed_dim":32,"ngram_size":2,"heads_per_ngram":1,"ngram_vocab_size_base":2,
        \\ "make_ngram_vocab_size_divisible_by":6,"split_ngram_parts":3}
    ;
    try td.dir.writeFile(io, .{ .sub_path = "config.json", .data = cfg });
    var buf: [std.fs.max_path_bytes]u8 = undefined;
    const path = buf[0..try td.dir.realPath(io, &buf)];
    var want = try model.parseConfigLegacy(io, testing.allocator, path);
    defer want.deinit(testing.allocator);
    try testing.expect(want.embedded_ple_payload_bytes != null);
    const index = try td.dir.readFileAlloc(io, "model.safetensors.index.json", testing.allocator, .limited(1 << 20));
    defer testing.allocator.free(index);
    const mem: fixture.MemDir = .{ .files = &.{ .{ .name = "config.json", .bytes = cfg }, .{ .name = "model.safetensors.index.json", .bytes = index } } };
    const pd = mem.packDir();
    var p: abi.Parsed = .{};
    try testing.expect(config.parse(&pd, .of(cfg), .{}, null, false, &p).isOk());
    defer config.freeParsed(&p);
    try testing.expect(config.of(&p).embedded_ngram);
    var got = try toModelConfig(testing.allocator, config.of(&p), path);
    defer got.deinit(testing.allocator);
    try testing.expectEqualDeep(want, got);
}

test "qwen4_exp model config: a Sushi pack's EXL3 experts equal the engine's parse" {
    const sushi = "{\"expert_quant\":{\"format\":\"exl3\",\"k\":2,\"codebook\":\"mcg\",\"window\":15}," ++ config.tiny_text[1..];
    try expectEngineParse(&.{.{ .name = "config.json", .bytes = sushi }});
}
