//! The served path on tiny random llama packs written to disk: `parseConfig` hands the pack to the llama plugin,
//! `driver.load` builds the shell, and every forward the engine makes (prefill, decode, verify, the lazy pipelined
//! decode step, captures, batched rows) is bit-equal to the legacy `Transformer`'s on the same pack (recorded to
//! `fixtures/`). Discovery's facts and the generic `ModelConfig` fields equal the legacy reading.

const std = @import("std");
const mlx = @import("../../../mlx/ffi.zig");
const ops = @import("../../../mlx/ops.zig");
const model = @import("../../../sdk/config/model.zig");
const driver = @import("driver.zig");
const plugin_config = @import("config.zig");
const fixture = @import("../../../sdk/bases/proof/fixture.zig");
const discovery = @import("../../../host/registry/discovery.zig");
const KVCache = @import("../kv/cache.zig").KVCache;
const ForwardCtx = @import("../forward_ctx.zig").ForwardCtx;
const Transformer = @import("../transformer.zig").Transformer;
const abi = @import("../../../abi/plugin.zig");
const quant_abi = @import("../../../abi/quant.zig");
const llama = @import("../../../plugins/llama/plugin.zig");
const HotPrefixCache = @import("../cache/prefix_cache.zig").HotPrefixCache;
const cache_ops = @import("../kv/cache_ops.zig");
const both = @import("both.zig");

const testing = std.testing;
const Array = mlx.mlx_array;

/// A tiny llama pack's geometry and options.
pub const Geo = struct {
    hidden: u32 = 128,
    heads: u32 = 4,
    kv_heads: u32 = 2,
    head_dim: u32 = 32,
    inter: u32 = 256,
    vocab: u32 = 256,
    layers: u32 = 2,
    quantized: bool = false,
    llama3: bool = false,
    tied: bool = false,
};

pub fn configJson(a: std.mem.Allocator, g: Geo) ![]u8 {
    return std.fmt.allocPrint(a, "{{\"model_type\":\"llama\",\"hidden_size\":{d},\"intermediate_size\":{d},\"num_hidden_layers\":{d}," ++
        "\"num_attention_heads\":{d},\"num_key_value_heads\":{d},\"head_dim\":{d},\"vocab_size\":{d},\"rms_norm_eps\":1e-5," ++
        "\"rope_theta\":500000.0,\"max_position_embeddings\":4096,\"hidden_act\":\"silu\",\"bos_token_id\":1,\"eos_token_id\":[2,3]," ++
        "\"tie_word_embeddings\":{}{s}{s}}}", .{
        g.hidden,                                                                                                                                                                     g.inter,                                                                     g.layers, g.heads, g.kv_heads, g.head_dim, g.vocab, g.tied,
        if (g.llama3) ",\"rope_scaling\":{\"factor\":8.0,\"low_freq_factor\":1.0,\"high_freq_factor\":4.0,\"original_max_position_embeddings\":64,\"rope_type\":\"llama3\"}" else "", if (g.quantized) ",\"quantization\":{\"group_size\":64,\"bits\":4}" else "",
    });
}

fn at(buf: []u8, layer: usize, name: []const u8) ![]const u8 {
    return std.fmt.bufPrint(buf, "model.layers.{d}.{s}", .{ layer, name });
}

fn writePack(p: *fixture.Pack, g: Geo) !void {
    var buf: [128]u8 = undefined;
    const h: c_int = @intCast(g.hidden);
    const q: c_int = @intCast(g.heads * g.head_dim);
    const kv: c_int = @intCast(g.kv_heads * g.head_dim);
    const inter: c_int = @intCast(g.inter);
    try p.weight("model.embed_tokens", @intCast(g.vocab), h, g.quantized);
    try p.norm("model.norm.weight", h);
    if (!g.tied) try p.weight("lm_head", @intCast(g.vocab), h, g.quantized);
    for (0..g.layers) |i| {
        try p.norm(try at(&buf, i, "input_layernorm.weight"), h);
        try p.norm(try at(&buf, i, "post_attention_layernorm.weight"), h);
        try p.weight(try at(&buf, i, "self_attn.q_proj"), q, h, g.quantized);
        try p.weight(try at(&buf, i, "self_attn.k_proj"), kv, h, g.quantized);
        try p.weight(try at(&buf, i, "self_attn.v_proj"), kv, h, g.quantized);
        try p.weight(try at(&buf, i, "self_attn.o_proj"), h, q, g.quantized);
        try p.weight(try at(&buf, i, "mlp.gate_proj"), inter, h, g.quantized);
        try p.weight(try at(&buf, i, "mlp.up_proj"), inter, h, g.quantized);
        try p.weight(try at(&buf, i, "mlp.down_proj"), h, inter, g.quantized);
    }
}

/// The pack served through `parseConfig` and `driver.load`, as the engine loads it.
pub const Served = struct {
    tmp: testing.TmpDir,
    dir: []u8,
    cfg: model.ModelConfig,
    xfm: Transformer,
    layers: u32,

    pub fn open(sv: *Served, g: Geo, seed: u64) !void {
        const a = testing.allocator;
        const io = std.Io.Threaded.global_single_threaded.io();
        var p = fixture.Pack.init(seed);
        defer p.deinit();
        try writePack(&p, g);
        const json = try configJson(a, g);
        defer a.free(json);
        sv.* = .{ .tmp = testing.tmpDir(.{}), .dir = &.{}, .cfg = undefined, .xfm = undefined, .layers = g.layers };
        errdefer sv.tmp.cleanup();
        try sv.tmp.dir.writeFile(io, .{ .sub_path = "config.json", .data = json });
        var path_buf: [512]u8 = undefined;
        sv.dir = try a.dupe(u8, path_buf[0..try sv.tmp.dir.realPath(io, &path_buf)]);
        errdefer a.free(sv.dir);
        try p.save(sv.dir);
        sv.cfg = try model.parseConfig(io, a, sv.dir);
        errdefer sv.cfg.deinit(a);
        if (sv.cfg.plugin == null) return error.TestExpectedServedPack;
        sv.xfm = try driver.load(io, a, &sv.cfg, sv.dir, fixture.stream());
    }

    pub fn close(sv: *Served) void {
        sv.xfm.deinit();
        sv.cfg.deinit(testing.allocator);
        testing.allocator.free(sv.dir);
        sv.tmp.cleanup();
    }
};

/// One slot's cache and context.
pub const Slot = struct {
    kv: KVCache,
    off: usize = 0,
    ctx: ForwardCtx = undefined,

    pub fn init(s: *Slot, layers: u32) !void {
        s.* = .{ .kv = try KVCache.init(testing.allocator, layers) };
        s.ctx = .{ .cache = &s.kv, .moe_seq_offset = &s.off, .ssm_entries = null, .capture_hidden = null, .vision_embeddings = null };
    }
    pub fn deinit(s: *Slot) void {
        s.kv.deinit();
    }
};

/// The pack of `g` through the llama table on the legacy parse's config (as `parseConfig` builds it), against the
/// legacy forward's values recorded to `fixtures/<seed>.tape`.
fn openProof(p: *fixture.Pack, g: Geo, comptime seed: u64) !both.Both {
    const a = testing.allocator;
    try writePack(p, g);
    const json = try configJson(a, g);
    defer a.free(json);
    const name = std.fmt.comptimePrint("fixtures/{x}.tape", .{seed});
    return both.Both.replay(&llama.table, p, json, true, @embedFile(name));
}

fn proofCase(g: Geo, comptime seed: u64) !void {
    var p = fixture.Pack.init(seed);
    defer p.deinit();
    var b = try openProof(&p, g, seed);
    defer b.close();
    try testing.expect(b.xfm.plugin != null);
    if (g.llama3) try testing.expectEqualStrings("llama3-rope-v1", b.cfg.cacheLayoutNamespace().?);
    try b.expectSteps(&both.steps);
    try b.expectGreedy();
    try b.expectBatched();
    try b.expectDone();
}

// ── tests ──

test "plugin driver: a llama 4-bit pack served by its plugin is bit-equal to the legacy forward (steps, greedy, batched)" {
    try proofCase(.{ .quantized = true }, 0x11a3a);
}

test "plugin driver: a llama bf16 pack with the llama3 rope and a tied head is bit-equal to the legacy forward" {
    try proofCase(.{ .llama3 = true, .tied = true }, 0x5eed3);
}

test "plugin driver: a DFlash drafter on a served llama trunk reads its embedding and head and emits the legacy stream" {
    const a = testing.allocator;
    // The tiny assistant's geometry (`dflash.TinyFix`): hidden 64, targets layers 0 and 2, mask id 127.
    var p = fixture.Pack.init(0xdf1a5);
    defer p.deinit();
    var b = try openProof(&p, .{ .hidden = 64, .head_dim = 16, .inter = 128, .vocab = 128, .layers = 4, .tied = true }, 0xdf1a5);
    defer b.close();
    try testing.expect(b.xfm.supportsLayerCapture());
    var g = ops.Graph.init(a, fixture.stream());
    defer g.deinit();
    const ids = try fixture.tokens(&g, &.{ 4, 90, 127, 0, 33 });
    const hidden = try g.reshape(try g.adopt(try b.xfm.rawEmbedding(ids)), &.{ 1, 5, 64 });
    try b.check(try g.adopt(try b.xfm.rawEmbedding(ids)), false);
    try b.check(try g.adopt(try b.xfm.lmHeadForDraft(hidden)), false);
    var tokens: [16]u32 = undefined;
    var accepted: std.ArrayList(u32) = .empty;
    defer accepted.deinit(a);
    try both.dflashRun(&b.xfm, &tokens, &accepted);
    for ([_][]const u32{ &tokens, accepted.items }) |got| try b.check(try g.fromSlice(u32, got, &.{@intCast(got.len)}), true);
    try b.expectDone();
}

/// Fields the forward alone reads: the plugin keeps them, the adapter leaves their defaults.
const plugin_owned = [_][]const u8{ "rms_norm_eps", "rope_theta", "rope_local_base_freq", "rope_llama3", "tie_word_embeddings", "query_pre_attn_scalar", "plugin" };

fn expectGenericFieldsEqual(served: *const model.ModelConfig, legacy: *const model.ModelConfig) !void {
    @setEvalBranchQuota(100_000);
    inline for (@typeInfo(model.ModelConfig).@"struct".field_names) |name| {
        const owned = comptime for (plugin_owned) |n| {
            if (std.mem.eql(u8, n, name)) break true;
        } else false;
        // The window is read only on a sliding model.
        const idle = comptime std.mem.startsWith(u8, name, "sliding_window");
        if (!owned and !(idle and !legacy.has_sliding_window)) {
            const x = @field(served, name);
            const y = @field(legacy, name);
            const same = switch (@typeInfo(@FieldType(model.ModelConfig, name))) {
                .pointer => std.mem.eql(u8, x, y),
                .optional => |o| if (@typeInfo(o.child) == .pointer) (x == null) == (y == null) and (x == null or std.mem.eql(u8, x.?, y.?)) else std.meta.eql(x, y),
                else => std.meta.eql(x, y),
            };
            if (!same) {
                std.debug.print("plugin config: field {s} differs from the legacy parse\n", .{name});
                return error.ConfigFieldDiffers;
            }
        }
    }
    try testing.expectEqual(legacy.contextCap(), served.contextCap());
    try testing.expectEqualSlices(u32, legacy.eosTokenSlice(), served.eosTokenSlice());
}

test "plugin config: the served ModelConfig equals the legacy parse on every generic field, and discovery's facts too" {
    const a = testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();
    const mistral = @embedFile("../../../fixtures/model-configs/mistral.json");
    // The fixture's bitsandbytes block keeps it on the legacy path; without it the pack is the llama plugin's.
    const mistral_plain = try std.mem.replaceOwned(u8, a, mistral, "\"quantization_config\"", "\"quantization_config_off\"");
    defer a.free(mistral_plain);
    const mistral_as_llama = try std.mem.replaceOwned(u8, a, mistral_plain, "\"mistral\"", "\"llama\"");
    defer a.free(mistral_as_llama);
    const q4 = try configJson(a, .{ .quantized = true });
    defer a.free(q4);
    const bf16 = try configJson(a, .{ .llama3 = true, .tied = true });
    defer a.free(bf16);
    for ([_][]const u8{ q4, bf16, mistral_as_llama }) |json| {
        var tmp = testing.tmpDir(.{});
        defer tmp.cleanup();
        try tmp.dir.writeFile(io, .{ .sub_path = "config.json", .data = json });
        try tmp.dir.writeFile(io, .{ .sub_path = "generation_config.json", .data = "{\"temperature\":0.6,\"top_p\":0.9,\"eos_token_id\":[2,9]}" });
        var path_buf: [512]u8 = undefined;
        const dir = path_buf[0..try tmp.dir.realPath(io, &path_buf)];
        var served = try model.parseConfig(io, a, dir);
        defer served.deinit(a);
        try testing.expect(served.plugin != null);
        var legacy = try model.parseConfigFromJson(a, json);
        defer legacy.deinit(a);
        // The sidecar half of the legacy load (`parseConfig` past `parseConfigFromJson`).
        legacy.gen_temperature = 0.6;
        legacy.gen_top_p = 0.9;
        legacy.mergeEosTokens(&.{ 2, 9 });
        try expectGenericFieldsEqual(&served, &legacy);

        const stub = discovery.readStubMeta(io, a, dir);
        const want = discovery.parseStubMeta(a, json, false);
        inline for (.{ "vocab_size", "hidden_size", "num_hidden_layers", "max_position_embeddings", "quant_bits", "is_moe", "has_vision", "has_video" }) |f|
            try testing.expectEqual(@field(want, f), @field(stub, f));
    }
}

test "plugin config: a shape its plugin declines is refused by name; a drafter sidecar or launch overrides serve" {
    const a = testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();
    const q4 = try configJson(a, .{ .quantized = true });
    defer a.free(q4);
    const sliding = try std.mem.concat(a, u8, &.{ q4[0 .. q4.len - 1], ",\"layer_types\":[\"full_attention\",\"sliding_attention\"]}" });
    defer a.free(sliding);
    const cases = [_]struct { json: []const u8, drafter: bool = false, refused: bool = true }{
        .{ .json = "{\"model_type\":\"qwen3\",\"hidden_size\":128,\"num_attention_heads\":4}" },
        .{ .json = sliding },
        .{ .json = q4, .drafter = true, .refused = false },
    };
    for (cases) |c| {
        var tmp = testing.tmpDir(.{});
        defer tmp.cleanup();
        try tmp.dir.writeFile(io, .{ .sub_path = "config.json", .data = c.json });
        if (c.drafter) {
            try tmp.dir.createDirPath(io, "drafter");
            try tmp.dir.writeFile(io, .{ .sub_path = "drafter/config.json", .data = "{\"block_size\":16,\"mask_token_id\":5,\"target_layer_ids\":[0,1]}" });
        }
        var path_buf: [512]u8 = undefined;
        const dir = path_buf[0..try tmp.dir.realPath(io, &path_buf)];
        if (c.refused) {
            try testing.expectError(error.PluginDeclinedNoLegacy, plugin_config.parseServed(io, a, dir, c.json));
        } else {
            var got = (try plugin_config.parseServed(io, a, dir, c.json)) orelse return error.TestExpectedPluginParse;
            got.deinit(a);
        }
    }
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.writeFile(io, .{ .sub_path = "config.json", .data = q4 });
    var path_buf: [512]u8 = undefined;
    const dir = path_buf[0..try tmp.dir.realPath(io, &path_buf)];
    // The overrides reach the plugin's parse as they reach the legacy one.
    model.setConfigOverrides("{\"max_position_embeddings\":8192}");
    defer model.setConfigOverrides(null);
    var got = (try plugin_config.parseServed(io, a, dir, q4)) orelse return error.TestExpectedPluginParse;
    defer got.deinit(a);
    try testing.expectEqual(@as(u32, 8192), got.max_position_embeddings);
}

test "plugin config: a served pack's engine config is the legacy parse with the binding attached" {
    const a = testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();
    var b: Served = undefined;
    try b.open(.{ .quantized = true }, 0xc0f6);
    defer b.close();
    try testing.expect(b.cfg.plugin != null);
    var legacy = try model.parseConfigLegacy(io, a, b.dir);
    defer legacy.deinit(a);
    try testing.expect(legacy.plugin == null);
    try both.expectGenericFieldsEqual(&b.cfg, &legacy, &.{});
}

test "plugin load: a trunk_weight that refuses serves the model without drafters" {
    const T = struct {
        fn refuse(m: *anyopaque, which: abi.TrunkWeight, out: *quant_abi.Weight) callconv(.c) abi.Verdict {
            if (which == .head) return abi.Verdict.refuse(.unsupported, "a head in no shape a drafter reads");
            return llama.table.spec.?.trunk_weight.?(m, which, out);
        }
        const spec: abi.SpecHooks = .{ .trunk_weight = &refuse };
        const table: abi.ModelTable = blk: {
            var t = llama.table;
            t.spec = &spec;
            break :blk t;
        };
    };
    var b: Served = undefined;
    try b.open(.{ .quantized = true }, 0x7e5f);
    defer b.close();
    try testing.expect(b.xfm.plugin.?.drafters);
    const io = std.Io.Threaded.global_single_threaded.io();
    const llama_table = b.cfg.plugin.?.table;
    b.cfg.plugin.?.table = &T.table;
    defer b.cfg.plugin.?.table = llama_table;
    var x = try driver.load(io, testing.allocator, &b.cfg, b.dir, fixture.stream());
    defer x.deinit();
    try testing.expect(!x.plugin.?.drafters and !x.supportsLayerCapture() and x.emb_w.ctx == null);
}

test "plugin load: a tower map takes the pack's vision tower tensors, a text load reads none" {
    const a = testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();
    const g: Geo = .{ .quantized = true };
    var p = fixture.Pack.init(0x70e7);
    defer p.deinit();
    try writePack(&p, g);
    try p.norm("vision_tower.blocks.0.norm1.weight", 32);
    try p.norm("multi_modal_projector.norm.weight", 32);
    const json = try configJson(a, g);
    defer a.free(json);
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.writeFile(io, .{ .sub_path = "config.json", .data = json });
    var path_buf: [512]u8 = undefined;
    const dir = path_buf[0..try tmp.dir.realPath(io, &path_buf)];
    try p.save(dir);
    var cfg = try model.parseConfig(io, a, dir);
    defer cfg.deinit(a);
    var tower = model.Weights.init(a);
    defer tower.deinit();
    var x = try driver.loadWith(io, a, &cfg, dir, fixture.stream(), &tower);
    defer x.deinit();
    try testing.expectEqual(@as(u32, 2), tower.map.count());
    try testing.expect(tower.map.contains("vision_tower.blocks.0.norm1.weight") and tower.map.contains("multi_modal_projector.norm.weight"));
    var y = try driver.load(io, a, &cfg, dir, fixture.stream());
    y.deinit();
}

test "plugin cache: a module-owned state's resume point and resident bytes come from the table" {
    const T = struct {
        var media_seen = false;
        fn resumeAt(_: *anyopaque, _: u32, prompt: abi.Span(u32), media: abi.Array) callconv(.c) u64 {
            media_seen = media.ctx != null;
            return prompt.len; // all of it: the engine keeps the last token to forward
        }
        fn bytes(_: *anyopaque) callconv(.c) u64 {
            return 4096;
        }
        const cache: abi.CacheHooks = .{ .resume_prompt = &resumeAt, .state_bytes = &bytes };
        const table: abi.ModelTable = blk: {
            var t = llama.table;
            t.cache = &cache;
            break :blk t;
        };
    };
    var b: Served = undefined;
    try b.open(.{}, 0x2e5a);
    defer b.close();
    const prompt = [_]u32{ 4, 8, 15, 16 };
    var ctx: ForwardCtx = undefined;
    ctx.plugin_slot = 0;
    try testing.expectEqual(@as(?u64, null), driver.resumePrompt(&b.xfm, &ctx, &prompt, null));
    try testing.expectEqual(@as(u64, 0), driver.stateBytes(&b.xfm));
    b.xfm.plugin.?.table = &T.table;
    try testing.expectEqual(@as(?u64, 3), driver.resumePrompt(&b.xfm, &ctx, &prompt, null));
    try testing.expect(!T.media_seen);
    try testing.expectEqual(@as(u64, 4096), driver.stateBytes(&b.xfm));
}

test "plugin forward: a module-owned model's prompt reads only the last row's logits and the step moves by the rows" {
    const T = struct {
        /// The module keeps its own state: the double forwards on a scratch cache and never touches the slot's.
        fn forward(m: *anyopaque, f: *const abi.Forward, out: *abi.ForwardOut) callconv(.c) abi.Verdict {
            var scratch = KVCache.init(testing.allocator, 2) catch return abi.Verdict.refuse(.failed, "scratch");
            defer scratch.deinit();
            var slot: cache_ops.Slot = .{ .kv = &scratch };
            const handles = [_]abi.Handle{slot.handle()};
            var g = f.*;
            g.caches = .of(&handles);
            g.positions = .of(&[_]u32{0});
            return llama.table.forward(m, &g, out);
        }
        const table: abi.ModelTable = blk: {
            var t = llama.table;
            t.forward = &forward;
            break :blk t;
        };
    };
    var b: Served = undefined;
    try b.open(.{ .quantized = true }, 0x30d5);
    defer b.close();
    var s: [2]Slot = undefined;
    for (&s) |*x| try x.init(b.layers);
    defer for (&s) |*x| x.deinit();
    var g = ops.Graph.init(testing.allocator, fixture.stream());
    defer g.deinit();
    const ids = try fixture.tokens(&g, &.{ 7, 8, 9, 10, 11 });
    const want = try g.adopt(try b.xfm.forwardWith(&s[0].ctx, ids));
    b.xfm.plugin.?.table = &T.table;
    b.xfm.config.plugin.?.parsed.caps.owns_decode_state = true;
    const got = try g.adopt(try b.xfm.forwardWith(&s[1].ctx, ids));
    try testing.expectEqualSlices(c_int, &.{ 1, 1, 256 }, ops.shape(got));
    try ops.expectBitEqual(got, try g.sliceAxis(want, 1, 4, 5));
    try testing.expectEqual(@as(usize, 5), s[1].kv.step);
}

test "plugin caps: module decode state, readonly sharing, spec rollback, whole-prompt prefill and the prefix cache read the served caps" {
    var b: Served = undefined;
    try b.open(.{}, 0xca95);
    defer b.close();
    const caps = &b.xfm.config.plugin.?.parsed.caps;
    try testing.expect(!b.xfm.ownsModuleDecodeState() and !b.xfm.moduleSpecWiring() and !b.xfm.prefillsWholePrompt());
    try testing.expect(HotPrefixCache.shouldUse(&b.xfm.config, true));
    caps.owns_decode_state = true;
    caps.shared_readonly_state = true;
    caps.spec_rollback = true;
    caps.whole_prompt_prefill = true;
    caps.prefix_cache = false;
    try testing.expect(b.xfm.ownsModuleDecodeState() and b.xfm.sharesModuleReadonlyState() and b.xfm.moduleStateSpecRollback());
    try testing.expect(b.xfm.prefillsWholePrompt() and !HotPrefixCache.shouldUse(&b.xfm.config, true));
}

fn expectLegacy(got: ?model.ModelConfig) !void {
    var c = got orelse return;
    c.deinit(testing.allocator);
    return error.TestExpectedLegacyParse;
}

test {
    _ = @import("served.zig");
    _ = @import("proof_mtp.zig");
    _ = @import("proof_settle.zig");
    _ = @import("mlx_lm_configs_test.zig");
    _ = @import("media.zig");
    _ = @import("intree.zig");
    _ = plugin_config;
}

