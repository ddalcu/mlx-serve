//! Parity proof for the nemotron_h plugin: tiny random packs (4-bit `M*E-E`; the grouped-norm Mamba2 width; the
//! Puzzle latent MoE; a bf16 pack with a `layers_block_type` list and a tied head) served by the table bit-equal to
//! the legacy `forwardHybridWith` at every step, the Mamba2 op chain too, and `parse` equal to the engine's parse.

const mlx = @import("../../mlx/ffi.zig");
const std = @import("std");
const abi = @import("../../abi/plugin.zig");
const fixture = @import("../../sdk/bases/proof/fixture.zig");
const proof = @import("../lfm2/proof.zig");
const both = @import("../../engines/mlx/plugin/both.zig");
const driver = @import("../../engines/mlx/plugin/driver.zig");
const model = @import("../../sdk/config/model.zig");
const nemotron = @import("plugin.zig");
const config = @import("config.zig");

const testing = std.testing;

const H: c_int = 128;
const HEADS: c_int = 4;
const KV_HEADS: c_int = 2;
const HD: c_int = 32;
const INTER: c_int = 256;
const VOCAB: c_int = 256;

const Shape = struct {
    /// The Mamba2 head width: 64 makes the grouped norm's group 128 wide (the kernel's floor).
    mamba_hd: c_int = 32,
    /// 8 heads put the in-projection's width on a multiple of 8 (inside the row-exact kernels).
    mamba_heads: c_int = 4,
    /// The experts' latent width; 0 = experts read the full hidden.
    latent: c_int = 0,
    quantized: bool = true,
    /// The pattern as a `layers_block_type` list, not the string.
    list: bool = false,
    tied: bool = false,
    /// The Puzzle trunk's own `model_type`.
    puzzle: bool = false,
    /// The omni wrapper: the trunk under `llm_config` / `language_model.`, a RADIO `vision_config`, `<image>` = 250.
    omni: bool = false,
    /// Quantized at 2 bits, group 128 (the experts, 64 wide, at group 64); ternary = every bias -scale.
    two_bit: enum { no, q2, ternary } = .no,
};

fn tensorAs(p: *fixture.Pack, base: []const u8, shape: []const c_int, sh: Shape) !void {
    const group: c_int = if (@rem(shape[shape.len - 1], 128) == 0) 128 else 64;
    return switch (sh.two_bit) {
        .no => p.tensor(base, shape, sh.quantized),
        .q2 => p.affine(base, shape, 2, group),
        .ternary => p.ternary(base, shape, group),
    };
}

const pattern = "M*E-E";

fn configJson(buf: []u8, sh: Shape) ![]const u8 {
    if (sh.omni) {
        var inner_buf: [2048]u8 = undefined;
        var inner_sh = sh;
        inner_sh.omni = false;
        const inner = try configJson(&inner_buf, inner_sh);
        return std.fmt.bufPrint(buf, "{{\"model_type\":\"nemotron_h_omni\",\"llm_config\":{s},\"vision_config\":{{\"model_type\":\"radio\",\"hidden_size\":32,\"num_attention_heads\":2}},\"image_token_id\":250}}", .{inner});
    }
    const layers = if (sh.list) "\"layers_block_type\":[\"mamba\",\"attention\",\"moe\",\"mlp\",\"moe\"]" else "\"hybrid_override_pattern\":\"M*E-E\"";
    return std.fmt.bufPrint(buf,
        \\{{"model_type":"{s}","hidden_size":128,"intermediate_size":256,"num_hidden_layers":5,"num_attention_heads":4,
        \\"num_key_value_heads":2,"head_dim":32,"vocab_size":256,"layer_norm_epsilon":1e-5,"rope_theta":10000.0,
        \\"max_position_embeddings":4096,{s},"mamba_num_heads":{d},"mamba_head_dim":{d},
        \\"n_groups":2,"ssm_state_size":32,"conv_kernel":4,"time_step_limit":[0.001,0.1],"n_routed_experts":8,
        \\"num_experts_per_tok":3,"moe_intermediate_size":64,"moe_shared_expert_intermediate_size":128,
        \\"routed_scaling_factor":2.5,"n_group":1,"topk_group":1,"norm_topk_prob":true,"tie_word_embeddings":{s},
        \\"moe_latent_size":{d},"eos_token_id":[2,11],"bos_token_id":1{s}}}
    , .{ if (sh.puzzle) "nemotron_h_puzzle" else "nemotron_h", layers, sh.mamba_heads, sh.mamba_hd, if (sh.tied) "true" else "false", sh.latent, if (sh.two_bit != .no) ",\"quantization\":{\"group_size\":128,\"bits\":2}" else if (sh.quantized) ",\"quantization\":{\"group_size\":64,\"bits\":4}" else "" });
}

fn at(buf: []u8, sh: Shape, layer: usize, name: []const u8) ![]const u8 {
    return std.fmt.bufPrint(buf, "{s}backbone.layers.{d}.{s}", .{ root(sh), layer, name });
}

fn root(sh: Shape) []const u8 {
    return if (sh.omni) "language_model." else "";
}

fn writePack(p: *fixture.Pack, sh: Shape) !void {
    var buf: [160]u8 = undefined;
    const mh = sh.mamba_heads;
    const d_inner = mh * sh.mamba_hd;
    const conv_dim = d_inner + 2 * 2 * 32;
    const ein: c_int = if (sh.latent > 0) sh.latent else H;
    var rb: [64]u8 = undefined;
    try tensorAs(p, try std.fmt.bufPrint(&rb, "{s}backbone.embeddings", .{root(sh)}), &.{ VOCAB, H }, sh);
    try p.norm(try std.fmt.bufPrint(&rb, "{s}backbone.norm_f.weight", .{root(sh)}), H);
    if (!sh.tied) try tensorAs(p, try std.fmt.bufPrint(&rb, "{s}lm_head", .{root(sh)}), &.{ VOCAB, H }, sh);
    for (pattern, 0..) |kind, i| {
        try p.norm(try at(&buf, sh, i, "norm.weight"), H);
        switch (kind) {
            'M' => {
                try tensorAs(p, try at(&buf, sh, i, "mixer.in_proj"), &.{ d_inner + conv_dim + mh, H }, sh);
                try p.dense(try at(&buf, sh, i, "mixer.conv1d.weight"), &.{ conv_dim, 4, 1 }, 0.6, 0.1);
                try p.dense(try at(&buf, sh, i, "mixer.conv1d.bias"), &.{conv_dim}, 0.2, 0);
                try p.dense(try at(&buf, sh, i, "mixer.A_log"), &.{mh}, 1.0, 0.3);
                try p.dense(try at(&buf, sh, i, "mixer.D"), &.{mh}, 1.0, 1.0);
                try p.dense(try at(&buf, sh, i, "mixer.dt_bias"), &.{mh}, 1.0, 0);
                try p.norm(try at(&buf, sh, i, "mixer.norm.weight"), d_inner);
                try tensorAs(p, try at(&buf, sh, i, "mixer.out_proj"), &.{ H, d_inner }, sh);
            },
            '*' => {
                try tensorAs(p, try at(&buf, sh, i, "mixer.q_proj"), &.{ HEADS * HD, H }, sh);
                try tensorAs(p, try at(&buf, sh, i, "mixer.k_proj"), &.{ KV_HEADS * HD, H }, sh);
                try tensorAs(p, try at(&buf, sh, i, "mixer.v_proj"), &.{ KV_HEADS * HD, H }, sh);
                try tensorAs(p, try at(&buf, sh, i, "mixer.o_proj"), &.{ H, HEADS * HD }, sh);
            },
            'E' => {
                try p.dense(try at(&buf, sh, i, "mixer.gate.weight"), &.{ 8, H }, 0.5, 0);
                try p.denseAs(try at(&buf, sh, i, "mixer.gate.e_score_correction_bias"), &.{8}, 0.2, 0, .float32);
                try tensorAs(p, try at(&buf, sh, i, "mixer.switch_mlp.fc1"), &.{ 8, 64, ein }, sh);
                try tensorAs(p, try at(&buf, sh, i, "mixer.switch_mlp.fc2"), &.{ 8, ein, 64 }, sh);
                try tensorAs(p, try at(&buf, sh, i, "mixer.shared_experts.up_proj"), &.{ 128, H }, sh);
                try tensorAs(p, try at(&buf, sh, i, "mixer.shared_experts.down_proj"), &.{ H, 128 }, sh);
                if (sh.latent > 0) {
                    try tensorAs(p, try at(&buf, sh, i, "mixer.fc1_latent_proj"), &.{ sh.latent, H }, sh);
                    try tensorAs(p, try at(&buf, sh, i, "mixer.fc2_latent_proj"), &.{ H, sh.latent }, sh);
                }
            },
            else => {
                try tensorAs(p, try at(&buf, sh, i, "mixer.up_proj"), &.{ INTER, H }, sh);
                try tensorAs(p, try at(&buf, sh, i, "mixer.down_proj"), &.{ H, INTER }, sh);
            },
        }
    }
}

/// After `proof.steps`: a chunk wider than the in-place expert window (the sorted MoE gather), then more decode.
const wide_steps = [_]proof.Step{
    .{ .mode = .prefill, .ids = &.{ 3, 17, 42, 99, 7, 250, 1, 8, 66, 31, 120, 5, 77, 200 } },
    .{ .mode = .decode, .ids = &.{88} },
    .{ .mode = .verify, .ids = &.{ 5, 61, 200 } },
    .{ .mode = .decode, .ids = &.{64} },
};

/// The default 4-bit trunk pack and its config, for the MTP head proof.
pub fn writeTrunk(p: *fixture.Pack) !void {
    return writePack(p, .{});
}

pub fn trunkConfig(buf: []u8) ![]const u8 {
    return configJson(buf, .{});
}

/// The trunk inside the row-exact kernels (8 Mamba2 heads), and its config.
pub fn writeRowExactTrunk(p: *fixture.Pack) !void {
    return writePack(p, .{ .mamba_heads = 8 });
}

pub fn rowExactTrunkConfig(buf: []u8) ![]const u8 {
    return configJson(buf, .{ .mamba_heads = 8 });
}


/// `sh`'s pack served by the driver on the legacy parse's config (recurrent entries per slot), against its oracle.
fn openCase(p: *fixture.Pack, sh: Shape, comptime seed: u64) !both.Both {
    try writePack(p, sh);
    var cbuf: [4096]u8 = undefined;
    const json = try configJson(&cbuf, sh);
    const name = std.fmt.comptimePrint("fixtures/{x}.tape", .{seed});
    return both.Both.replay(&nemotron.table, p, json, true, @embedFile(name));
}

/// Steps past a chunk wider than the in-place expert window, then the pipelined greedy decode; `drafter` binds a
/// DFlash drafter (the row-exact lanes: prompt chunks of up to 16 rows attend row by row), whose verify rows then
/// equal serial decode.
fn servedCase(sh: Shape, drafter: bool, comptime seed: u64) !void {
    var p = fixture.Pack.init(seed);
    defer p.deinit();
    var b = try openCase(&p, sh, seed);
    defer b.close();
    if (drafter) {
        try testing.expect(b.xfm.config.row_exact_covered);
        b.xfm.config.dflash_bound = true;
    }
    try b.expectSteps(&(proof.steps ++ wide_steps));
    try b.expectSteps(&both.long_steps);
    if (sh.omni) try b.expectMedia(250, H);
    try b.expectGreedy();
    try b.expectDone();
    if (drafter) {
        const m: *nemotron.Loaded = @ptrCast(@alignCast(b.xfm.plugin.?.model));
        try testing.expectEqual(.rows, m.base.row_exact);
        try proof.expectVerifyEqualsSerialOn(nemotron, m, b.xfm.config.num_hidden_layers, b.xfm.s, &.{ 3, 17, 42, 99, 7, 250, 1, 8, 66, 31, 120, 5, 77, 200, 9, 14, 28, 2, 6, 40 }, &.{ 88, 5, 61, 200, 12 });
    }
}

fn expectParse(json: []const u8) !void {
    var want = try proof.legacyConfig(json);
    defer want.deinit(testing.allocator);
    var p: abi.Parsed = .{};
    try testing.expect(config.parseJson(json, &p).isOk());
    defer config.freeParsed(&p);
    try proof.expectShell(&p.shell, &want);
    const c = config.of(&p);
    try testing.expectEqual(want.rms_norm_eps, c.rms_norm_eps);
    try testing.expectEqual(want.tie_word_embeddings, c.tie_word_embeddings);
    try testing.expectEqual(want.mamba_num_heads, c.mamba.heads);
    try testing.expectEqual(want.mamba_head_dim, c.mamba.head_dim);
    try testing.expectEqual(want.mamba_n_groups, c.mamba.groups);
    try testing.expectEqual(want.ssm_state_size, c.mamba.state);
    try testing.expectEqual(want.mamba_conv_kernel, c.mamba.kernel);
    try testing.expectEqual(want.time_step_min, c.mamba.dt_min);
    try testing.expectEqual(want.time_step_max, c.mamba.dt_max);
    try testing.expectEqual(want.num_experts, c.moe.experts);
    try testing.expectEqual(want.num_experts_per_tok, c.moe.top_k);
    try testing.expectEqual(want.moe_intermediate_size, c.moe.intermediate);
    try testing.expectEqual(want.shared_expert_intermediate_size, c.moe.shared_intermediate);
    try testing.expectEqual(want.moe_n_group, c.moe.n_group);
    try testing.expectEqual(want.moe_topk_group, c.moe.topk_group);
    try testing.expectEqual(want.moe_route_norm, c.moe.norm);
    try testing.expectEqual(want.router_scaling_factor, c.moe.scale);
    try testing.expectEqual(want.moe_latent_size, c.moe.latent);
    try testing.expectEqual(want.isMoe(), p.caps.moe);
    try testing.expectEqualStrings(want.cacheLayoutNamespace().?, p.cache.layout_namespace.slice());
    for (0..c.num_hidden_layers) |i| {
        const k: config.LayerKind = switch (want.layer_block_types[i]) {
            .mamba2 => .mamba2,
            .moe => .moe,
            .mlp => .mlp,
            else => .attention,
        };
        try testing.expectEqual(k, c.layers[i]);
    }
}

// ── tests ──

test "nemotron_h served: Mamba2 + NoPE attention + folded MoE (4-bit) is bit-equal to the engine's forward" {
    try servedCase(.{}, false, 0x7e11);
}

test "nemotron_h served: the grouped-norm Mamba2 width is bit-equal" {
    try servedCase(.{ .mamba_hd = 64 }, false, 0x6e6e);
}

test "nemotron_h served: the Puzzle latent MoE (routes on the full hidden) is bit-equal" {
    try servedCase(.{ .latent = 64, .puzzle = true }, false, 0x7e12);
}

test "nemotron_h served: a bf16 pack with a layers_block_type list and a tied head is bit-equal" {
    try servedCase(.{ .quantized = false, .list = true, .tied = true }, false, 0x7e13);
}

test "nemotron_h plugin: parse equals the engine's parse on the shell and the plugin's own fields" {
    var buf: [2048]u8 = undefined;
    try expectParse(try configJson(&buf, .{}));
    try expectParse(try configJson(&buf, .{ .latent = 64, .list = true, .tied = true }));
}

test "nemotron_h served: with a drafter bound, the row-exact lanes are bit-equal, verify rows equal serial decode" {
    try servedCase(.{ .mamba_heads = 8 }, true, 0x7e14);
}

test "nemotron_h served: an omni pack's RADIO media rows splice bit-equal" {
    try servedCase(.{ .omni = true }, false, 0x7e15);
}

test "nemotron_h served: 2-bit packs, plain (stock qmm) and ternary (qmv2's kernels on both sides), are bit-equal" {
    inline for (.{ .q2, .ternary }, .{ 0x7e41, 0x7e42 }) |two, seed| {
        var p = fixture.Pack.init(seed);
        defer p.deinit();
        var b = try openCase(&p, .{ .two_bit = two }, seed);
        defer b.close();
        const m: *const nemotron.Loaded = @ptrCast(@alignCast(b.xfm.plugin.?.model));
        try testing.expectEqual(two == .ternary, m.base.weights.head.ternary);
        try b.expectSteps(&both.long_steps);
        try b.expectGreedy();
        try b.expectDone();
    }
}

test "nemotron_h served: an omni pack is served, its RADIO tower and projector load into the engine's map" {
    const a = testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();
    const sh: Shape = .{ .omni = true };
    var p = fixture.Pack.init(0x7e16);
    defer p.deinit();
    try writePack(&p, sh);
    const keys = [_][]const u8{ "vision_model.radio_model.model.blocks.0.norm1.weight", "mlp1.0.weight" };
    for (keys) |k| try p.norm(k, 32);
    var cbuf: [4096]u8 = undefined;
    const json = try configJson(&cbuf, sh);
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.writeFile(io, .{ .sub_path = "config.json", .data = json });
    var path_buf: [512]u8 = undefined;
    const dir = path_buf[0..try tmp.dir.realPath(io, &path_buf)];
    try p.save(dir);
    var cfg = try model.parseConfig(io, a, dir);
    defer cfg.deinit(a);
    try testing.expect(cfg.plugin != null and cfg.radio_vision);
    var tower = model.Weights.init(a);
    defer tower.deinit();
    var x = try driver.loadWith(io, a, &cfg, dir, fixture.stream(), &tower);
    defer x.deinit();
    try testing.expectEqual(@as(u32, 2), tower.map.count());
    for (keys) |k| try testing.expect(tower.map.contains(k));
    try testing.expectEqual(@as(u32, 250), cfg.plugin.?.parsed.placeholder_ids.image);
}

// ── a `--mlx-gguf` file ──

const gguf_arch = @import("../../sdk/quant/gguf/arch.zig");

const GH: c_int = 256;
const G_MH: c_int = 8;
const G_INNER: c_int = G_MH * 32;
const G_CONV: c_int = G_INNER + 2 * 2 * 32;
const G_EXPERTS: c_int = 4;

/// The config `sdk/quant/gguf/arch/nemotron_h.zig` rebuilds from a GGUF's metadata, at this pack's geometry: an
/// `M*EME` MoE trunk (Nemotron 3 Nano's shape) or an `M*-M-` dense one.
fn ggufConfig(moe: bool) ![]u8 {
    var hp = std.mem.zeroes(gguf_arch.nemotron_h.Hparams);
    hp.n_layers = 5;
    hp.hidden = GH;
    hp.ffn = if (moe) 0 else 512;
    hp.kinds[1] = .attention;
    hp.kinds[2] = if (moe) .moe else .mlp;
    hp.kinds[4] = if (moe) .moe else .mlp;
    hp.n_heads = 4;
    hp.n_kv_heads = 2;
    hp.head_dim = 64;
    hp.mamba_heads = G_MH;
    hp.mamba_head_dim = 32;
    hp.n_groups = 2;
    hp.state_size = 32;
    hp.conv_kernel = 4;
    hp.expand = 2;
    hp.rope_theta = 10000;
    hp.rms_eps = 1e-5;
    hp.ctx_len = 4096;
    hp.vocab = VOCAB;
    hp.eos = 11;
    if (moe) {
        hp.n_experts = G_EXPERTS;
        hp.n_experts_used = 2;
        hp.expert_ffn = GH;
        hp.shared_ffn = GH;
        hp.routed_scale = 2.5;
        hp.norm_topk = true;
    }
    return gguf_arch.nemotron_h.configJson(testing.allocator, hp);
}

/// The weights a Nemotron-H GGUF loads into: k-quant, legacy and i-quant blocks across the projections (q6_k table
/// and head), one float projection, `A_log` f32 (the loader's `-exp` undone), the router and every other float
/// tensor bf16.
fn writeGgufPack(p: *fixture.Pack, moe: bool) !void {
    var buf: [160]u8 = undefined;
    const sh: Shape = .{};
    try p.gguf("backbone.embeddings", &.{ VOCAB, GH }, .q6_k);
    try p.norm("backbone.norm_f.weight", GH);
    try p.gguf("lm_head", &.{ VOCAB, GH }, .q6_k);
    for (0..5) |i| {
        try p.norm(try at(&buf, sh, i, "norm.weight"), GH);
        switch (i) {
            0, 3 => {
                try p.gguf(try at(&buf, sh, i, "mixer.in_proj"), &.{ G_INNER + G_CONV + G_MH, GH }, .q4_k);
                try p.dense(try at(&buf, sh, i, "mixer.conv1d.weight"), &.{ G_CONV, 4, 1 }, 0.6, 0.1);
                try p.dense(try at(&buf, sh, i, "mixer.conv1d.bias"), &.{G_CONV}, 0.2, 0);
                try p.denseAs(try at(&buf, sh, i, "mixer.A_log"), &.{G_MH}, 1.0, 0.3, .float32);
                try p.dense(try at(&buf, sh, i, "mixer.D"), &.{G_MH}, 1.0, 1.0);
                try p.dense(try at(&buf, sh, i, "mixer.dt_bias"), &.{G_MH}, 1.0, 0);
                try p.norm(try at(&buf, sh, i, "mixer.norm.weight"), G_INNER);
                if (i == 3) try p.weight(try at(&buf, sh, i, "mixer.out_proj"), GH, G_INNER, false) else try p.gguf(try at(&buf, sh, i, "mixer.out_proj"), &.{ GH, G_INNER }, .q5_k);
            },
            1 => {
                try p.gguf(try at(&buf, sh, i, "mixer.q_proj"), &.{ 4 * 64, GH }, .q4_k);
                try p.gguf(try at(&buf, sh, i, "mixer.k_proj"), &.{ 2 * 64, GH }, .q4_k);
                try p.gguf(try at(&buf, sh, i, "mixer.v_proj"), &.{ 2 * 64, GH }, .q8_0);
                try p.gguf(try at(&buf, sh, i, "mixer.o_proj"), &.{ GH, 4 * 64 }, .iq4_xs);
            },
            else => if (moe) {
                try p.dense(try at(&buf, sh, i, "mixer.gate.weight"), &.{ G_EXPERTS, GH }, 0.5, 0);
                try p.dense(try at(&buf, sh, i, "mixer.gate.e_score_correction_bias"), &.{G_EXPERTS}, 0.2, 0);
                try p.gguf(try at(&buf, sh, i, "mixer.switch_mlp.fc1"), &.{ G_EXPERTS, GH, GH }, .q4_k);
                try p.gguf(try at(&buf, sh, i, "mixer.switch_mlp.fc2"), &.{ G_EXPERTS, GH, GH }, .q6_k);
                try p.gguf(try at(&buf, sh, i, "mixer.shared_experts.up_proj"), &.{ GH, GH }, .q4_k);
                try p.gguf(try at(&buf, sh, i, "mixer.shared_experts.down_proj"), &.{ GH, GH }, .q6_k);
            } else {
                try p.gguf(try at(&buf, sh, i, "mixer.up_proj"), &.{ 512, GH }, .q4_k);
                try p.gguf(try at(&buf, sh, i, "mixer.down_proj"), &.{ GH, 512 }, .q6_k);
            },
        }
    }
}


fn ggufCase(moe: bool, comptime seed: u64) !void {
    if (!mlx.metalKernelsAvailable()) return error.SkipZigTest;
    var p = fixture.Pack.init(seed);
    defer p.deinit();
    try writeGgufPack(&p, moe);
    const json = try ggufConfig(moe);
    defer testing.allocator.free(json);
    const name = std.fmt.comptimePrint("fixtures/gguf-{x}.tape", .{seed});
    var b = try both.Both.replay(&nemotron.table, &p, json, true, @embedFile(name));
    defer b.close();
    const m: *const nemotron.Loaded = @ptrCast(@alignCast(b.xfm.plugin.?.model));
    try testing.expect(m.base.weights.head.mode == .gguf and m.base.row_exact == .off);
    try b.expectSteps(&(proof.steps ++ wide_steps));
    try b.expectSteps(&both.long_steps);
    try b.expectGreedy();
    try b.expectDone();
}

test "nemotron_h served: a --mlx-gguf MoE file (gguf expert banks, a float projection, A_log f32) is bit-equal" {
    try ggufCase(true, 0x7e51);
}

test "nemotron_h served: a --mlx-gguf dense file (relu2 MLP layers) is bit-equal" {
    try ggufCase(false, 0x7e52);
}

test "nemotron_h plugin: a --mlx-gguf file's rebuilt config parses as the engine's" {
    for ([_]bool{ false, true }) |moe| {
        const json = try ggufConfig(moe);
        defer testing.allocator.free(json);
        try expectParse(json);
    }
}

/// A real Nemotron-H GGUF file (`nemotron_h_moe` `M*EME` or `nemotron_h` `M*-M-`) at the `ggufConfig` geometry, as
/// llama.cpp names and lays out its tensors (`ssm_a` = -exp(A_log), the per-head vectors under a leading axis of 1):
/// the engine's own GGUF loader reads it on both sides.
fn writeGgufFile(g: *proof.Gguf, moe: bool, seed: u64) !void {
    var prng = std.Random.DefaultPrng.init(seed);
    const r = prng.random();
    const arch = if (moe) "nemotron_h_moe" else "nemotron_h";
    var kb: [96]u8 = undefined;
    const k = struct {
        fn f(b: []u8, a: []const u8, s: []const u8) ![]const u8 {
            return std.fmt.bufPrint(b, "{s}.{s}", .{ a, s });
        }
    }.f;
    try g.string("general.architecture", arch);
    try g.int(try k(&kb, arch, "block_count"), 5);
    try g.ints(try k(&kb, arch, "attention.head_count_kv"), &.{ 0, 2, 0, 0, 0 });
    const ff: u32 = if (moe) GH else 512;
    try g.ints(try k(&kb, arch, "feed_forward_length"), &.{ 0, 0, ff, 0, ff });
    try g.int(try k(&kb, arch, "embedding_length"), GH);
    try g.int(try k(&kb, arch, "attention.head_count"), 4);
    try g.int(try k(&kb, arch, "attention.key_length"), 64);
    try g.int(try k(&kb, arch, "ssm.inner_size"), G_INNER);
    try g.int(try k(&kb, arch, "ssm.time_step_rank"), G_MH);
    try g.int(try k(&kb, arch, "ssm.group_count"), 2);
    try g.int(try k(&kb, arch, "ssm.state_size"), 32);
    try g.int(try k(&kb, arch, "ssm.conv_kernel"), 4);
    try g.float(try k(&kb, arch, "rope.freq_base"), 10000);
    try g.float(try k(&kb, arch, "attention.layer_norm_rms_epsilon"), 1e-5);
    try g.int(try k(&kb, arch, "context_length"), 4096);
    if (moe) {
        try g.int(try k(&kb, arch, "expert_count"), G_EXPERTS);
        try g.int(try k(&kb, arch, "expert_used_count"), 2);
        try g.int(try k(&kb, arch, "expert_feed_forward_length"), GH);
        try g.int(try k(&kb, arch, "expert_shared_feed_forward_length"), GH);
        try g.float(try k(&kb, arch, "expert_weights_scale"), 2.5);
        try g.boolean(try k(&kb, arch, "expert_weights_norm"), true);
    }
    try g.vocab(VOCAB, 11);
    const W: u64 = GH;
    try g.quant(r, "token_embd.weight", &.{ VOCAB, W }, .q6_k);
    try g.f32s(r, "output_norm.weight", &.{W}, 0.2, 1);
    try g.quant(r, "output.weight", &.{ VOCAB, W }, .q6_k);
    var nb: [96]u8 = undefined;
    const blk = struct {
        fn f(b: []u8, l: usize, leaf: []const u8) ![]const u8 {
            return std.fmt.bufPrint(b, "blk.{d}.{s}", .{ l, leaf });
        }
    }.f;
    for (0..5) |i| {
        try g.f32s(r, try blk(&nb, i, "attn_norm.weight"), &.{W}, 0.2, 1);
        switch (i) {
            0, 3 => {
                try g.quant(r, try blk(&nb, i, "ssm_in.weight"), &.{ G_INNER + G_CONV + G_MH, W }, .q4_k);
                try g.f32s(r, try blk(&nb, i, "ssm_conv1d.weight"), &.{ 4, G_CONV }, 0.6, 0.1);
                try g.f32s(r, try blk(&nb, i, "ssm_conv1d.bias"), &.{G_CONV}, 0.2, 0);
                try g.f32s(r, try blk(&nb, i, "ssm_a"), &.{ 1, G_MH }, 0.5, -1.5);
                try g.f32s(r, try blk(&nb, i, "ssm_d"), &.{ 1, G_MH }, 1.0, 1.0);
                try g.f32s(r, try blk(&nb, i, "ssm_dt.bias"), &.{G_MH}, 1.0, 0);
                try g.f32s(r, try blk(&nb, i, "ssm_norm.weight"), &.{ G_INNER / 2, 2 }, 0.2, 1);
                if (i == 3) try g.f32s(r, try blk(&nb, i, "ssm_out.weight"), &.{ G_INNER, W }, 0.06, 0) else try g.quant(r, try blk(&nb, i, "ssm_out.weight"), &.{ W, G_INNER }, .q5_k);
            },
            1 => {
                try g.quant(r, try blk(&nb, i, "attn_q.weight"), &.{ 4 * 64, W }, .q4_k);
                try g.quant(r, try blk(&nb, i, "attn_k.weight"), &.{ 2 * 64, W }, .q4_k);
                try g.quant(r, try blk(&nb, i, "attn_v.weight"), &.{ 2 * 64, W }, .q8_0);
                try g.quant(r, try blk(&nb, i, "attn_output.weight"), &.{ W, 4 * 64 }, .iq4_xs);
            },
            else => if (moe) {
                try g.f32s(r, try blk(&nb, i, "ffn_gate_inp.weight"), &.{ W, G_EXPERTS }, 0.5, 0);
                try g.f32s(r, try blk(&nb, i, "exp_probs_b.bias"), &.{G_EXPERTS}, 0.2, 0);
                try g.quant(r, try blk(&nb, i, "ffn_up_exps.weight"), &.{ G_EXPERTS, W, W }, .q4_k);
                try g.quant(r, try blk(&nb, i, "ffn_down_exps.weight"), &.{ G_EXPERTS, W, W }, .q6_k);
                try g.quant(r, try blk(&nb, i, "ffn_up_shexp.weight"), &.{ W, W }, .q4_k);
                try g.quant(r, try blk(&nb, i, "ffn_down_shexp.weight"), &.{ W, W }, .q6_k);
            } else {
                try g.quant(r, try blk(&nb, i, "ffn_up.weight"), &.{ 512, W }, .q4_k);
                try g.quant(r, try blk(&nb, i, "ffn_down.weight"), &.{ W, 512 }, .q6_k);
            },
        }
    }
}

fn ggufFileCase(moe: bool, by_file: bool, comptime seed: u64) !void {
    if (!mlx.metalKernelsAvailable()) return error.SkipZigTest;
    var g: proof.Gguf = .{};
    defer g.deinit();
    try writeGgufFile(&g, moe, seed);
    const name = std.fmt.comptimePrint("fixtures/gguf-file-{x}.tape", .{seed});
    var b = try proof.openGguf(&nemotron.table, &g, by_file, .{ .tape = @embedFile(name) });
    defer b.close();
    try b.expectSteps(&(proof.steps ++ wide_steps));
    try b.expectSteps(&both.long_steps);
    try b.expectGreedy();
    try b.expectDone();
}

test "nemotron_h served: a MoE GGUF file the engine loads (--mlx-gguf), addressed by the file, is bit-equal" {
    try ggufFileCase(true, true, 0x7e53);
}

test "nemotron_h served: a dense GGUF file the engine loads (--mlx-gguf), addressed by its folder, is bit-equal" {
    try ggufFileCase(false, false, 0x7e54);
}

// ── kernel arms against their composed chains, on the plugin alone ──

const residual_norm = @import("../../sdk/blocks/norms/residual_norm.zig");
const mamba2_block = @import("../../sdk/bases/blocks/mamba2.zig");

/// The trunk pack (`M*E-E`, 4-bit) served by the plugin alone: last-row logits of one prefill and three decode steps.
fn trunkLogits(seed: u64) ![]f32 {
    var p = fixture.Pack.init(seed);
    defer p.deinit();
    try writePack(&p, .{});
    var cbuf: [4096]u8 = undefined;
    var b = try both.Both.openAs(&nemotron.table, &p, try configJson(&cbuf, .{}), true, .none);
    defer b.close();
    return proof.hostLogits(nemotron, b.xfm.plugin.?.model, b.xfm.config.num_hidden_layers, b.xfm.s, &.{ 3, 17, 42, 99, 7, 250 }, &.{ 88, 5, 61 });
}

/// Largest |a - b| and |a| over two logit rows.
fn maxErr(a: []const f32, b: []const f32) !struct { err: f32, ref: f32 } {
    var e: f32 = 0;
    var r: f32 = 0;
    for (a, b) |x, y| {
        try testing.expect(std.math.isFinite(x) and std.math.isFinite(y));
        e = @max(e, @abs(x - y));
        r = @max(r, @abs(x));
    }
    try testing.expect(r > 0.01);
    return .{ .err = e, .ref = r };
}

// Bar: the decode seams' fused residual add + next norm (the MoE combine folding the shared expert and the scores
// included) is bit-equal to the same steps with the fusion off.
test "nemotron_h served: the fused add+norm decode path equals the unfused blocks" {
    const fused = try trunkLogits(0x7e61);
    defer testing.allocator.free(fused);
    const saved = residual_norm.add_norm_max_rows;
    residual_norm.add_norm_max_rows = 0;
    defer residual_norm.add_norm_max_rows = saved;
    const plain = try trunkLogits(0x7e61);
    defer testing.allocator.free(plain);
    try testing.expectEqual(@as(f32, 0), (try maxErr(fused, plain)).err);
}

/// An `M` pack (one Mamba2 layer, as the bar's reference measured it), 4-bit, served by the plugin alone:
/// last-row logits of one prefill and three decode steps.
fn mambaLogits() ![]f32 {
    var p = fixture.Pack.init(0x7e62);
    defer p.deinit();
    var buf: [160]u8 = undefined;
    const sh: Shape = .{};
    const mh = sh.mamba_heads;
    const d_inner = mh * sh.mamba_hd;
    const conv_dim = d_inner + 2 * 2 * 32;
    try p.tensor("backbone.embeddings", &.{ VOCAB, H }, true);
    try p.norm("backbone.norm_f.weight", H);
    try p.tensor("lm_head", &.{ VOCAB, H }, true);
    {
        const i = 0;
        try p.norm(try at(&buf, sh, i, "norm.weight"), H);
        try p.tensor(try at(&buf, sh, i, "mixer.in_proj"), &.{ d_inner + conv_dim + mh, H }, true);
        try p.dense(try at(&buf, sh, i, "mixer.conv1d.weight"), &.{ conv_dim, 4, 1 }, 0.6, 0.1);
        try p.dense(try at(&buf, sh, i, "mixer.conv1d.bias"), &.{conv_dim}, 0.2, 0);
        try p.dense(try at(&buf, sh, i, "mixer.A_log"), &.{mh}, 1.0, 0.3);
        try p.dense(try at(&buf, sh, i, "mixer.D"), &.{mh}, 1.0, 1.0);
        try p.dense(try at(&buf, sh, i, "mixer.dt_bias"), &.{mh}, 1.0, 0);
        try p.norm(try at(&buf, sh, i, "mixer.norm.weight"), d_inner);
        try p.tensor(try at(&buf, sh, i, "mixer.out_proj"), &.{ H, d_inner }, true);
    }
    const json =
        \\{"model_type":"nemotron_h","hidden_size":128,"intermediate_size":256,"num_hidden_layers":1,"num_attention_heads":4,
        \\"num_key_value_heads":2,"head_dim":32,"vocab_size":256,"layer_norm_epsilon":1e-5,"rope_theta":10000.0,
        \\"max_position_embeddings":4096,"hybrid_override_pattern":"M","mamba_num_heads":4,"mamba_head_dim":32,
        \\"n_groups":2,"ssm_state_size":32,"conv_kernel":4,"time_step_limit":[0.001,0.1],"tie_word_embeddings":false,
        \\"eos_token_id":2,"quantization":{"group_size":64,"bits":4}}
    ;
    var b = try both.Both.openAs(&nemotron.table, &p, json, true, .none);
    defer b.close();
    return proof.hostLogits(nemotron, b.xfm.plugin.?.model, 1, b.xfm.s, &.{ 3, 17, 42, 99, 7, 250, 1, 8, 66, 31, 120, 5 }, &.{ 88, 5, 61 });
}

// Bar: the fused Mamba2 step agrees with the op chain over a prefill window and single decode steps, to bf16 rounding.
test "nemotron_h served: the fused Mamba2 step matches the op chain" {
    const fused = try mambaLogits();
    defer testing.allocator.free(fused);
    const saved = mamba2_block.fused_max_rows;
    mamba2_block.fused_max_rows = 0;
    defer mamba2_block.fused_max_rows = saved;
    const chain = try mambaLogits();
    defer testing.allocator.free(chain);
    const m = try maxErr(fused, chain);
    if (m.err > 1.5e-2 * m.ref) {
        std.debug.print("mamba2 fused vs chain: max_err={d:.5} max_ref={d:.5}\n", .{ m.err, m.ref });
        return error.FusedChainMismatch;
    }
}

// Bar: a NoPE attention layer's last-token output does not move when the earlier tokens are reordered; a rotated key
// would.
test "nemotron_h served: a NoPE attention layer ignores the order of earlier tokens" {
    var p = fixture.Pack.init(0x7e63);
    defer p.deinit();
    try p.tensor("backbone.embeddings", &.{ VOCAB, H }, false);
    try p.norm("backbone.norm_f.weight", H);
    try p.tensor("lm_head", &.{ VOCAB, H }, false);
    try p.norm("backbone.layers.0.norm.weight", H);
    try p.tensor("backbone.layers.0.mixer.q_proj", &.{ HEADS * HD, H }, false);
    try p.tensor("backbone.layers.0.mixer.k_proj", &.{ KV_HEADS * HD, H }, false);
    try p.tensor("backbone.layers.0.mixer.v_proj", &.{ KV_HEADS * HD, H }, false);
    try p.tensor("backbone.layers.0.mixer.o_proj", &.{ H, HEADS * HD }, false);
    const json =
        \\{"model_type":"nemotron_h","hidden_size":128,"intermediate_size":256,"num_hidden_layers":1,"num_attention_heads":4,
        \\"num_key_value_heads":2,"head_dim":32,"vocab_size":256,"layer_norm_epsilon":1e-5,"rope_theta":10000.0,
        \\"max_position_embeddings":4096,"hybrid_override_pattern":"*","tie_word_embeddings":false,"eos_token_id":2}
    ;
    var b = try both.Both.openAs(&nemotron.table, &p, json, true, .none);
    defer b.close();
    const m = b.xfm.plugin.?.model;
    const a = try proof.hostLogits(nemotron, m, 1, b.xfm.s, &.{ 1, 5, 2, 6, 3 }, &.{});
    defer testing.allocator.free(a);
    const r = try proof.hostLogits(nemotron, m, 1, b.xfm.s, &.{ 6, 2, 5, 1, 3 }, &.{});
    defer testing.allocator.free(r);
    const e = try maxErr(a, r);
    if (e.err > 1e-2 * e.ref) {
        std.debug.print("NoPE attention moved with token order: max_err={d:.6} max_ref={d:.6}\n", .{ e.err, e.ref });
        return error.AttentionIsPositional;
    }
}

test "nemotron_h plugin: a pack the plugin declines is refused by name (no legacy forward to fall back to)" {
    var buf: [4096]u8 = undefined;
    try both.expectNoLegacyFallback(try configJson(&buf, .{}), "expert_quant");
}
