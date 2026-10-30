//! Who serves a pack, and the engine's `ModelConfig` for one a plugin serves: the registry's claimant parses the pack
//! into `Parsed` and one adapter builds the generic config the engine reads from it. A served type's pack its plugin
//! declines is refused by name; a pack of another type (an mlx-stream pack) keeps the legacy parse alone.

const std = @import("std");
const abi = @import("../../../abi/plugin.zig");
const common = @import("../../../abi/common.zig");
const sdk_plugin = @import("../../../sdk/plugin.zig");
const registry = @import("../../../plugins/registry.zig");
const served = @import("served.zig");
const model = @import("../../../sdk/config/model.zig");
const log = @import("../../../sdk/util/log.zig");
const FsPackDir = @import("../../../host/registry/pack_dir.zig").FsPackDir;
const intree = @import("intree.zig");
const outside = @import("../../../host/plugins/loader.zig");
const model_settings = @import("../../../host/registry/model_settings.zig");

const ModelConfig = model.ModelConfig;

/// A served pack's plugin and what its parse produced; owned by the `ModelConfig` it rides (`ModelConfig.deinit`).
pub const Binding = struct {
    table: *const abi.ModelTable,
    parsed: abi.Parsed = .{},
    /// An outside plugin's (`host/plugins`): the legacy parse knows nothing of its packs.
    outside: bool = false,

    pub fn destroy(b: *Binding, a: std.mem.Allocator) void {
        b.table.free_parsed(&b.parsed);
        a.destroy(b);
    }
};

/// The pack's claimant: a loaded outside plugin that wins it over the built-ins, else the registry's when its type is
/// served; null otherwise (two claimants at one rank too).
pub fn claimant(model_type: []const u8, config_json: []const u8, dir: *const abi.PackDir) ?*const abi.ModelTable {
    if (outside.winner(model_type, config_json, dir)) |p| if (p.entry().model) |t| return t;
    return claimantIn(&registry.entries, model_type, config_json, dir);
}

fn claimantIn(list: []const sdk_plugin.Entry, model_type: []const u8, config_json: []const u8, dir: *const abi.PackDir) ?*const abi.ModelTable {
    if (!served.serves(model_type)) return null;
    var ranks: [64]abi.Claim = undefined;
    if (list.len > ranks.len) return null;
    const got = sdk_plugin.claimsOf(list, model_type, config_json, dir, &ranks);
    const i = (common.pick(got) catch return null) orelse return null;
    return list[i].model;
}

/// The plugin's parse of the pack at `model_dir` (`config_json` its config.json), as the generic `ModelConfig` of its
/// facts alone carrying its binding (`adapted`); null when no plugin serves it.
pub fn parseServed(io: std.Io, a: std.mem.Allocator, model_dir: []const u8, config_json: []const u8) !?ModelConfig {
    const b = (try bindServed(io, a, model_dir, config_json)) orelse return null;
    return adapted(b);
}

/// The generic `ModelConfig` of a binding's facts, owning it: for a type the legacy parse cannot read.
pub fn adapted(b: *Binding) ModelConfig {
    var cfg = modelConfig(&b.parsed);
    applyLayout(&cfg, b.table, &b.parsed);
    // An outside plugin's images ride the host's patch-grid path (`host/media/plugin_media.zig`).
    if (!b.outside) applyMedia(&cfg, b.table, &b.parsed);
    cfg.plugin = b;
    return cfg;
}

/// The plugin's claim and parse of the pack at `model_dir` (caller owns the binding); null for a pack of a type no
/// plugin serves (an mlx-stream pack is its stream module's), `error.PluginDeclinedNoLegacy` for a served type's pack
/// its plugin declines. The engine's config of a served pack is the legacy parse with this binding attached.
pub fn bindServed(io: std.Io, a: std.mem.Allocator, model_dir: []const u8, config_json: []const u8) !?*Binding {
    const why = switch (try resolve(io, a, model_dir, config_json)) {
        .bound => |b| return b,
        .declined => |why| why,
    };
    var arena = std.heap.ArenaAllocator.init(a);
    defer arena.deinit();
    const model_type = sdk_plugin.modelTypeOf(arena.allocator(), config_json) catch return null;
    if (!served.serves(model_type)) return null;
    if (streamPack(io, a, model_dir, config_json)) {
        log.info("[plugin] {s}: {s}; its stream module serves it\n", .{ model_type, why });
        return null;
    }
    log.err("[plugin] {s}: {s}; no other path serves it\n", .{ model_type, why });
    return error.PluginDeclinedNoLegacy;
}

/// An mlx-stream pack: a declared expert bank, or an `experts.bin` beside the trunk.
fn streamPack(io: std.Io, a: std.mem.Allocator, model_dir: []const u8, config_json: []const u8) bool {
    if (model.declaresExpertStreaming(a, config_json)) return true;
    var buf: [std.fs.max_path_bytes]u8 = undefined;
    const bank = std.fmt.bufPrint(&buf, "{s}/experts.bin", .{model_dir}) catch return false;
    std.Io.Dir.accessAbsolute(io, bank, .{}) catch return false;
    return true;
}

/// `table`'s parse of the pack at `dir` (caller owns the binding), for an engine that serves a pack through one named
/// plugin whatever else claims it (a decision pack's trunk); null when the parse declines it.
pub fn bindTable(io: std.Io, a: std.mem.Allocator, dir: []const u8, table: *const abi.ModelTable) !?*Binding {
    return bindTableWith(io, a, dir, table, "{}");
}

/// `bindTable` over the pack's config.json with `overrides` deep-merged in (`model.mergeConfigJson`).
pub fn bindTableWith(io: std.Io, a: std.mem.Allocator, dir: []const u8, table: *const abi.ModelTable, overrides: []const u8) !?*Binding {
    var fs = try FsPackDir.open(io, dir);
    defer fs.close();
    const pd = fs.packDir();
    const buf = try a.alloc(u8, 1 << 20);
    defer a.free(buf);
    var why: abi.Verdict = .{};
    const raw = pd.readFile("config.json", buf, &why) orelse return error.ConfigUnreadable;
    const cfg = try model.mergeConfigJson(a, raw, overrides);
    defer a.free(cfg);
    const b = try a.create(Binding);
    b.* = .{ .table = table };
    var settings: [48]u8 = undefined;
    if (!table.parse(&pd, .of(cfg), .of(parseSettings(a, io, &settings, dir)), null, false, &b.parsed).isOk()) {
        a.destroy(b);
        return null;
    }
    return b;
}

/// The settings a plugin's parse reads: the pack's per-model `stream_experts`, which decides the caps of a pack that
/// can stream (`buf` holds the text).
fn parseSettings(a: std.mem.Allocator, io: std.Io, buf: []u8, model_dir: []const u8) []const u8 {
    var path: [std.fs.max_path_bytes]u8 = undefined;
    var s = model_settings.load(a, io, model_settings.defaultPath(&path));
    defer s.deinit();
    var o = s.lookup(a, model_dir);
    defer o.deinit(a);
    return switch (o.stream_experts orelse return "") {
        .on => "{\"stream_experts\":true}",
        .rows => |n| std.fmt.bufPrint(buf, "{{\"stream_experts\":{d}}}", .{n}) catch "",
    };
}

/// `config_json` with the launch's `--config-overrides` merged in, as the legacy parse reads it; in `a`.
pub fn withOverrides(a: std.mem.Allocator, config_json: []const u8) ![]const u8 {
    const ov = model.getConfigOverrides() orelse return config_json;
    return model.mergeConfigJson(a, config_json, ov);
}

/// Who serves a pack: its plugin's binding (caller owns it), or why none does (static text).
pub const Outcome = union(enum) { bound: *Binding, declined: []const u8 };

pub fn resolve(io: std.Io, a: std.mem.Allocator, model_dir: []const u8, raw_json: []const u8) !Outcome {
    var arena = std.heap.ArenaAllocator.init(a);
    defer arena.deinit();
    // The plugin parses what the legacy parse reads: the overrides land in both.
    const config_json = try withOverrides(arena.allocator(), raw_json);
    const model_type = try sdk_plugin.modelTypeOf(arena.allocator(), config_json);
    if (!served.serves(model_type) and outside.loaded().len == 0) return .{ .declined = "type not served" };
    var fs = FsPackDir.open(io, model_dir) catch return .{ .declined = "pack dir unreadable" };
    defer fs.close();
    const dir = fs.packDir();
    const table = claimant(model_type, config_json, &dir) orelse
        return .{ .declined = if (served.serves(model_type)) "no single claimant" else "type not served" };
    const b = try a.create(Binding);
    b.* = .{ .table = table, .outside = outside.ownerOf(table) != null };
    var settings: [48]u8 = undefined;
    const v = table.parse(&dir, abi.Str.of(config_json), .of(parseSettings(a, io, &settings, model_dir)), null, false, &b.parsed);
    if (!v.isOk()) {
        a.destroy(b);
        return .{ .declined = v.why.slice() };
    }
    // The engine reads an embedding pack's pooling from its folder name, which `parse` never sees: an encoder serves
    // under that config's pooling, a decoder only where its plugin serves the embedding batch.
    if (!b.parsed.caps.embed and !served.servesDecoderEmbedding(model_type) and model.poolingFromDirName(std.fs.path.basename(model_dir), model_type) != null) {
        b.destroy(a);
        return .{ .declined = "an embedding pack by its folder name" };
    }
    return .{ .bound = b };
}

/// What discovery reads of a served pack: the stub parse's shell and caps.
pub const Stub = @import("../../../host/engines/mlx.zig").PluginStub;

/// The plugin's stub parse of the pack at `model_dir`; null when no plugin serves it.
pub fn stubOf(io: std.Io, a: std.mem.Allocator, model_dir: []const u8, raw_json: []const u8) ?Stub {
    var arena = std.heap.ArenaAllocator.init(a);
    defer arena.deinit();
    const config_json = withOverrides(arena.allocator(), raw_json) catch return null;
    const model_type = sdk_plugin.modelTypeOf(arena.allocator(), config_json) catch return null;
    if (!served.serves(model_type) and outside.loaded().len == 0) return null;
    var fs = FsPackDir.open(io, model_dir) catch return null;
    defer fs.close();
    const dir = fs.packDir();
    const table = claimant(model_type, config_json, &dir) orelse return null;
    var p: abi.Parsed = .{};
    if (!table.parse(&dir, abi.Str.of(config_json), .{}, null, true, &p).isOk()) return null;
    defer table.free_parsed(&p);
    return .{ .shell = p.shell, .caps = p.caps, .speaks = p.speech.codebooks != 0 };
}

fn optF32(v: f32) ?f32 {
    return if (std.math.isNan(v)) null else v;
}

/// The engine's generic config of a parsed pack. Fields the forward alone reads (norm epsilon, rope schedule, tied
/// head, attention scale) stay the plugin's and keep their defaults here; `contextCap` and the SSD namespace read
/// the binding.
pub fn modelConfig(p: *const abi.Parsed) ModelConfig {
    const s = &p.shell;
    var c: ModelConfig = .{
        .model_type = s.model_type.slice(),
        .weight_prefix = if (p.weights.prefix.len > 0) p.weights.prefix.slice() else "model",
        .vocab_size = s.vocab_size,
        .hidden_size = s.hidden_size,
        .intermediate_size = s.intermediate_size,
        .intermediate_size_declared = s.intermediate_size > 0,
        .num_hidden_layers = s.num_hidden_layers,
        .num_attention_heads = s.num_attention_heads,
        .num_key_value_heads = s.num_key_value_heads,
        .head_dim = s.head_dim,
        .max_position_embeddings = std.math.cast(u32, s.max_position_embeddings) orelse std.math.maxInt(u32),
        .num_experts_per_tok = s.num_experts_per_tok,
        .moe_intermediate_size = s.moe_intermediate_size,
        .bos_token_id = if (s.bos >= 0) std.math.cast(u32, s.bos) else null,
        .gen_temperature = optF32(s.gen.temperature),
        .gen_top_p = optF32(s.gen.top_p),
        .gen_min_p = optF32(s.gen.min_p),
        .gen_top_k = if (s.gen.top_k > 0) @intCast(s.gen.top_k) else null,
        .gen_enable_thinking = switch (s.gen.enable_thinking) {
            .unset => null,
            .on => true,
            .off => false,
        },
        // The generic model: every layer full attention, plain norms, no Gemma traits.
        .has_sliding_window = false,
        .norm_has_offset = false,
        .scale_embeddings = p.spec.embed_scale > 0,
        .has_pre_ff_norm = false,
        .has_qk_norm = false,
        .hidden_act = .silu,
        .has_vision = p.caps.images,
        .image_token_id = p.placeholder_ids.image,
        .video_token_id = p.placeholder_ids.video,
        .audio_token_id = p.placeholder_ids.audio,
        // Recurrent layers: the slots get SSM entries, the prefix cache keeps checkpoints.
        .has_hybrid_layers = p.cache.ssm_state.entries > 0,
        .ssm_state_f32 = p.cache.ssm_state.dtype == .f32,
        .pooling_mode = switch (p.embed.pooling) {
            .unset => null,
            .mean => .mean,
            .cls => .cls,
            .last_token => .last_token,
        },
    };
    if (s.quant.bits > 0) {
        c.quant_bits = s.quant.bits;
        if (s.quant.group_size > 0) c.quant_group_size = s.quant.group_size;
        c.quant_mode = switch (s.quant.mode) {
            .none, .affine => .affine,
            .nvfp4 => .nvfp4,
            .mxfp4 => .mxfp4,
            .mxfp8 => .mxfp8,
            .gguf => .gguf,
        };
    }
    for (s.eos[0..@min(s.eos_count, abi.max_eos)]) |e| c.addEosToken(e);
    applySamplingDefaults(&c, p.chat.sampling_defaults);
    return c;
}

/// The family's sampling defaults where the pack's generation_config left a knob null; a wild 1.0/1.0 signature the
/// reference ignores resolves to its temperature (`applyFamilySamplingDefaults`, `applyDsv4ReferenceSampling`).
fn applySamplingDefaults(c: *ModelConfig, d: abi.SamplingDefaults) void {
    if (c.gen_top_k == null and d.top_k >= 0) c.gen_top_k = @intCast(d.top_k);
    if (c.gen_top_p == null) c.gen_top_p = optF32(d.top_p);
    const wild = (c.gen_temperature orelse 0) == 1.0 and (c.gen_top_p orelse 0) == 1.0;
    if (wild) if (optF32(d.wild_temp_override)) |t| {
        c.gen_temperature = t;
    };
}

/// A plugin with its own towers: an encoder loads for any media its caps declare.
pub fn applyMedia(c: *ModelConfig, table: *const abi.ModelTable, p: *const abi.Parsed) void {
    if (table.media == null) return;
    c.plugin_media = true;
    c.has_vision = p.caps.images or p.caps.video or p.caps.audio;
}

/// The per-layer facts the engine reads off the config (`isGlobalLayer`, the sliding window, KV sharing), from the
/// plugin's cache layout: a model without the hook keeps every layer full attention.
pub fn applyLayout(c: *ModelConfig, table: *const abi.ModelTable, p: *const abi.Parsed) void {
    const layout = table.cache.layout orelse return;
    var buf: [128]abi.LayerCache = undefined;
    const n = layout(p, null, &buf, buf.len);
    var window: u32 = 0;
    for (buf[0..n], 0..) |l, i| {
        if (l.kind == .sliding and (window == 0 or l.window < window)) window = l.window;
        c.layer_is_global[i] = l.kind == .full;
        if (l.kv_source >= 0) c.num_kv_shared_layers += 1;
    }
    if (window == 0) return;
    c.has_sliding_window = true;
    c.has_explicit_layer_types = true;
    c.sliding_window = window;
}

// ── tests ──

const testing = std.testing;
const fixture = @import("../../../sdk/testing/fixture.zig");

test "plugin claimant: a served type goes to its one claimant; an unserved type or two claimants to legacy" {
    const T = struct {
        fn mk(comptime name: []const u8, comptime types: []const []const u8) abi.ModelTable {
            return .{ .name = .of(name), .claims = sdk_plugin.claimsByType(types), .parse = undefined, .free_parsed = undefined, .bill = undefined, .load = undefined, .unload = undefined, .forward = undefined, .cache = undefined, .weights = undefined };
        }
    };
    const a = comptime T.mk("a", &.{"llama"});
    const twin = comptime T.mk("twin", &.{"llama"});
    const other = comptime T.mk("other", &.{"not_served"});
    const mem: fixture.MemDir = .{ .files = &.{} };
    const dir = mem.packDir();
    const one = [_]sdk_plugin.Entry{ .{ .plugin = .{ .name = "a", .kind = .model }, .model = &a }, .{ .plugin = .{ .name = "other", .kind = .model }, .model = &other } };
    try testing.expectEqual(@as(?*const abi.ModelTable, &a), claimantIn(&one, "llama", "{}", &dir));
    try testing.expectEqual(@as(?*const abi.ModelTable, null), claimantIn(&one, "not_served", "{}", &dir));
    const two = [_]sdk_plugin.Entry{ .{ .plugin = .{ .name = "a", .kind = .model }, .model = &a }, .{ .plugin = .{ .name = "twin", .kind = .model }, .model = &twin } };
    try testing.expectEqual(@as(?*const abi.ModelTable, null), claimantIn(&two, "llama", "{}", &dir));
}

test "plugin config: the cache layout sets the sliding window, the global layers and KV sharing; embed_scale the scaling" {
    const T = struct {
        fn layout(_: *const abi.Parsed, _: ?*const abi.LayerSpan, out: [*]abi.LayerCache, cap: u32) callconv(.c) u32 {
            const ls = [_]abi.LayerCache{ .{ .kind = .sliding, .window = 512 }, .{ .kind = .full }, .{ .kind = .sliding, .window = 512, .kv_source = 0 }, .{ .kind = .full, .kv_source = 1 } };
            const n = @min(cap, ls.len);
            @memcpy(out[0..n], ls[0..n]);
            return n;
        }
    };
    const t: abi.ModelTable = .{ .name = .of("t"), .claims = undefined, .parse = undefined, .free_parsed = undefined, .bill = undefined, .load = undefined, .unload = undefined, .forward = undefined, .cache = &.{ .layout = &T.layout }, .weights = undefined };
    const p: abi.Parsed = .{ .shell = .{ .num_hidden_layers = 4, .hidden_size = 64 }, .spec = .{ .embed_scale = 8 } };
    var c = modelConfig(&p);
    applyLayout(&c, &t, &p);
    try testing.expect(c.has_sliding_window and c.scale_embeddings);
    try testing.expectEqual(@as(u32, 512), c.sliding_window);
    try testing.expectEqual(@as(u32, 2), c.num_kv_shared_layers);
    for ([_]bool{ false, true, false, true }, 0..) |g, i| try testing.expectEqual(g, c.isGlobalLayer(@intCast(i)));
    var plain = modelConfig(&.{});
    applyLayout(&plain, &t, &.{ .shell = .{ .num_hidden_layers = 0 } });
    try testing.expect(!plain.scale_embeddings);
}

test "plugin config: a served model's facts layer on the legacy config; an unset fact keeps the legacy arm" {
    const t: abi.ModelTable = .{ .name = .of("t"), .claims = undefined, .parse = undefined, .free_parsed = undefined, .bill = undefined, .load = undefined, .unload = undefined, .forward = undefined, .cache = &.{}, .weights = undefined };
    var b: Binding = .{ .table = &t };
    // A legacy dense qwen3_5 config: its arm answers while the plugin states nothing.
    var c: ModelConfig = .{ .model_type = "qwen3_5", .legacy = .{ .caps = .{ .long_ctx = true } } };
    c.plugin = &b;
    try testing.expect(c.rowExactArch() and !c.isMoe() and !c.longCtxGated());
    // A generic legacy config: the facts answer.
    c.model_type = "llama";
    try testing.expect(!c.rowExactArch());
    try testing.expectEqual(@as(u32, 0), c.mtpDepth(0));
    b.parsed.caps = .{ .moe = true, .long_ctx = true, .exact_drafted = true };
    b.parsed.spec = .{ .mtp_default_depth = 3 };
    b.parsed.chat = .{ .thinking_default = .on_with_tools };
    try testing.expect(c.isMoe() and c.longCtxGated() and c.rowExactArch());
    try testing.expectEqual(@as(u32, 3), c.mtpDepth(0));
    try testing.expectEqual(@as(u32, 5), c.mtpDepth(5));
    try testing.expect(c.defaultEnableThinking(true) and !c.defaultEnableThinking(false));
    c.gen_enable_thinking = false;
    try testing.expect(!c.defaultEnableThinking(true));
    // Unbound, the legacy parse's caps answer.
    c.plugin = null;
    try testing.expect(c.longCtxGated() and !c.isMoe());
}

test "plugin config: sampling defaults fill the knobs the pack left null; a wild signature takes the reference temperature" {
    var p: abi.Parsed = .{ .shell = .{ .model_type = .of("qwen3") }, .chat = .{ .sampling_defaults = .{ .top_k = 64, .top_p = 0.95 } } };
    const c = modelConfig(&p);
    try testing.expectEqual(@as(?u32, 64), c.gen_top_k);
    try testing.expectEqual(@as(?f32, 0.95), c.gen_top_p);
    p.shell.gen = .{ .top_p = 0.8 };
    try testing.expectEqual(@as(?f32, 0.8), modelConfig(&p).gen_top_p);
    // No family facts: nothing is filled, whatever the display type says.
    try testing.expectEqual(@as(?u32, null), modelConfig(&.{ .shell = .{ .model_type = .of("qwen3") } }).gen_top_k);
    var wild: abi.Parsed = .{ .shell = .{ .gen = .{ .temperature = 1.0, .top_p = 1.0 } }, .chat = .{ .sampling_defaults = .{ .wild_temp_override = 0.6 } } };
    try testing.expectEqual(@as(?f32, 0.6), modelConfig(&wild).gen_temperature);
    wild.shell.gen.temperature = 0.7;
    try testing.expectEqual(@as(?f32, 0.7), modelConfig(&wild).gen_temperature);
}
