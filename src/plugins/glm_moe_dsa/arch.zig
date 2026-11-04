//! GLM-5.3 behind the `arch` kind (sdk.Arch.of): every declaration wraps the arch's own entry. The host reaches the
//! module through these, once per prompt, serial step, handover or draft round; nothing per layer. The draft lane is
//! the release's MTP layer (`glm_moe_dsa_mtp`), off unless the model's `mtp_depth` sets it.

const std = @import("std");
const sdk = @import("sdk");
const sdk_ext = @import("../../sdk/weights/streamed/sdk_ext.zig");
const glm = @import("glm_moe_dsa.zig");
const settings = @import("settings.zig");
const module = @import("module.zig");
const bill_mod = @import("bill.zig");
const bank_mod = @import("bank.zig");
const host_caps = @import("../../sdk/weights/streamed/host_caps.zig");

pub const name = "glm_moe_dsa";

/// GLM-5.3 on a host without its surface (`host_caps.glm`: sushi's EXL3 MoE and the DSA kernels, `mlx_host`'s pub
/// `exl3` / `dsa_nax`): the parse and the construction refuse by name, so no config or module of this arch exists
/// there and the entries that take one are unreachable (each compiles to its body on a host with the surface).
const host_surface_missing = "glm_moe_dsa: this host exports no GLM-5.3 surface (mlx_host.exl3: " ++
    (if (host_caps.exl3) "yes" else "no") ++ ", mlx_host.dsa_nax: " ++ (if (host_caps.dsa_nax) "yes" else "no") ++
    "): GLM-5.3 needs a host built with both (GlmHostSurfaceMissing)";

pub const caps: sdk.Caps = .{
    .owns_decode_state = true,
    .prefill_whole_prompt = true,
    .prefill_yields_last_logits = true,
    .residents_past_page_cache = true,
};

pub const Config = settings.Config;
/// The pack's bank picks the module at load (`module.Served`). On a host without the GLM-5.3 surface no module exists
/// (the construction refuses by name); the SDK's tables still name a type, one that reaches no host surface.
pub const Module = if (host_caps.glm) module.Served else struct {};

/// G6: the process's one expert reader, claimed at the load claim (a second load that needs it is refused by name,
/// `error.ExpertReaderInUse`), released when the loaded model goes.
pub const claimProcess = if (bank_mod.Stream.uses_reader) sdk_ext.expert.takeReader else {};
pub const releaseProcess = if (bank_mod.Stream.uses_reader) sdk_ext.expert.giveReader else {};

pub fn claims(p: *const sdk.ConfigPeek) ?sdk.Priority {
    const t = p.modelType() orelse return null;
    return if (std.mem.eql(u8, t, glm.model_type)) .native else null;
}

/// The arch's own parse of config.json against the release this build serves (refusals by name), and the pack's
/// directory beside it.
pub fn parse(gpa: std.mem.Allocator, p: *const sdk.ConfigPeek, diag: *sdk.Diag) !*Config {
    if (comptime !host_caps.glm) {
        diag.set("{s}", .{host_surface_missing});
        return error.GlmHostSurfaceMissing;
    }
    var model = try glm.Config.parse(gpa, p.text, &glm.glm53, diag);
    const c = gpa.create(Config) catch |e| {
        model.deinit(gpa);
        return e;
    };
    c.* = .{ .model = model };
    errdefer freeConfig(gpa, c);
    if (p.model_dir.len > 0) c.model_dir = try gpa.dupe(u8, p.model_dir);
    return c;
}

pub fn freeConfig(gpa: std.mem.Allocator, c: *Config) void {
    if (c.model_dir) |d| gpa.free(d);
    c.deinit(gpa);
    gpa.destroy(c);
}

pub fn shell(c: *const Config) sdk.Shell {
    const m = c.model orelse return .{};
    return .{ .num_experts = m.n_routed_experts, .num_layers = m.n_layers };
}

pub fn applySettings(c: *Config, raw: std.json.Value) void {
    c.applySettings(raw);
}

/// The load preflight's requirement: the bill's process bound at the fill's floor rows.
pub fn loadBytes(gpa: std.mem.Allocator, io: std.Io, c: *const Config, facts: *const sdk.LoadFacts, ceiling: u64) !u64 {
    if (comptime !host_caps.glm) unreachable;
    _ = ceiling;
    try c.checkRetired();
    try c.checkCtxSize();
    try c.checkMaxOutput();
    try c.checkUnitsCap();
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const built = c.withFacts(facts);
    var diag: glm.Diag = .{};
    return bill_mod.loadRequirementBytes(arena.allocator(), io, &built, &diag) catch |e| {
        sdk.log.err("glm_moe_dsa: load preflight refused: {s} ({s})\n", .{ diag.message(), @errorName(e) });
        return e;
    };
}

/// G4: the bill's terms for prompts up to the request's length (the fill's floor rows in the process bound). Pure
/// host: the pack's shard headers and the bank's manifest through `io`.
pub fn bill(gpa: std.mem.Allocator, io: std.Io, req: *const sdk.BillRequest) !sdk.MemoryBill {
    if (comptime !host_caps.glm) unreachable;
    const c: *const Config = @ptrCast(@alignCast(req.cfg));
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const mb = try bill_mod.sdkBillOf(arena.allocator(), io, c, req.prompt_tokens, null);
    var out = mb;
    out.terms = try gpa.dupe(sdk.MemoryBill.Term, mb.terms);
    errdefer gpa.free(out.terms);
    out.slot_table = try gpa.dupe(u64, mb.slot_table);
    return out;
}

/// The generation a request may take past its prompt (`max_output`): the host serves the billed context plus it.
pub fn maxOutput(c: *const Config) u64 {
    return c.maxOutput();
}

/// The construction admitted every prompt up to the billed context (both phases under the target); a longer prompt
/// is the module's refusal by name before its pass (`ContextOverBill`), never a memory guess here.
pub fn promptBytes(c: *const Config, seq: u64, max_tokens: u32) u64 {
    _ = .{ c, seq, max_tokens };
    return 0;
}

pub fn init(load: *const sdk.LoadCtx, c: *const Config) !*Module {
    if (comptime !host_caps.glm) {
        sdk.log.err("{s}\n", .{host_surface_missing});
        return error.GlmHostSurfaceMissing;
    }
    const built = c.withFacts(&load.facts);
    return Module.init(load.gpa, load.io, &built, load.weights, load.stream, .{ .ceiling = load.ceiling, .wired_margin = load.facts.wired_margin_bytes });
}

pub fn deinit(m: *Module) void {
    if (comptime !host_caps.glm) unreachable;
    m.deinit();
}

/// The prompt pass: from position 0, or after the positions `restorePrefix` kept (`req.prompt_tokens` counts them).
/// The prompt pass: the request's sampler bound (`draft_lane.arm`), its last row the plugin's first token when bound.
pub fn prefill(m: *Module, ids: []const u32, req: sdk.RequestShape) !sdk.mlx.mlx_array {
    if (comptime !host_caps.glm) unreachable;
    const lane = draft_lane.laneName(m);
    return switch (m.*) {
        inline else => |x| blk: {
            x.sampler.begin(lane);
            break :blk x.sampler.chosen(try x.prefillAt(req.prompt_tokens - ids.len, ids), x.position());
        },
    };
}

pub fn restorePrefix(m: *Module, prefix: []const u32) u64 {
    if (comptime !host_caps.glm) unreachable;
    return switch (m.*) {
        inline else => |x| x.restorePrefix(prefix),
    };
}

/// A serial step: its row the plugin's token at the next position when the request is bound.
pub fn step(m: *Module, ids: []const u32) !sdk.mlx.mlx_array {
    if (comptime !host_caps.glm) unreachable;
    return switch (m.*) {
        inline else => |x| x.sampler.chosen(try x.extend(ids), x.position()),
    };
}

pub fn position(m: *const Module) u64 {
    if (comptime !host_caps.glm) unreachable;
    return switch (m.*) {
        inline else => |x| x.position(),
    };
}

pub fn handover(m: *Module, h: sdk.DecodeHandover) !void {
    if (comptime !host_caps.glm) unreachable;
    return switch (m.*) {
        inline else => |x| x.decodeHandover(h),
    };
}

/// The request's end: the decode's stats line.
pub fn requestEnd(m: *Module) void {
    if (comptime !host_caps.glm) unreachable;
    switch (m.*) {
        inline else => |x| x.requestEnd(),
    }
}

/// The MTP layer's lane (`mtp_depth` drafts a round; 0 = serial). Acceptance is the model's setting: exact (greedy
/// requests: the argmax; sampled ones: the drafts accepted with the tempered, filtered target's probability) unless it
/// names typical (with its delta). Logprobs, grammar and penalties consume logits the lane never shapes and stay serial.
pub const draft_lane = struct {
    /// The MTP lane's depth, or 1 without one: the host still arms every clean request, so the plugin binds its
    /// sampler and draws every serial token.
    pub fn blockSize(m: *const Module) u32 {
        if (comptime !host_caps.glm) unreachable;
        return @max(depth(m), 1);
    }

    fn depth(m: *const Module) u32 {
        if (comptime !host_caps.glm) unreachable;
        return switch (m.*) {
            inline else => |x| x.mtpDepth(),
        };
    }

    pub fn laneName(m: *const Module) []const u8 {
        return if (depth(m) > 0) "glm-mtp" else "serial";
    }

    pub fn arm(m: *Module, req: sdk.ArmRequest) sdk.DraftArm {
        if (comptime !host_caps.glm) unreachable;
        const mode = switch (m.*) {
            inline else => |x| blk: {
                x.sampler.arm(req);
                break :blk (x.mtp orelse return .off).mode;
            },
        };
        if (!req.clean) return .off;
        return switch (mode) {
            .typical => .typical,
            else => if (req.greedy) .greedy else .stochastic,
        };
    }

    pub fn round(m: *Module, a: std.mem.Allocator, t1: u32, accepted_cap: u32, sampling: sdk.SamplingParams) !sdk.DraftRound {
        if (comptime !host_caps.glm) unreachable;
        return switch (m.*) {
            inline else => |x| x.mtpRound(a, t1, accepted_cap, sampling),
        };
    }

    pub fn stats(m: *const Module) sdk.DraftStats {
        if (comptime !host_caps.glm) unreachable;
        return switch (m.*) {
            inline else => |x| x.draftStats(),
        };
    }
};

const testing = std.testing;

test "glm plugin: claims its own model_type at native priority and declines the rest" {
    if (comptime !host_caps.glm) return error.SkipZigTest;
    const real = try glm.testConfigJson(testing.allocator, "");
    defer testing.allocator.free(real);
    try sdk.testing.expectClaims(claims, &.{
        .{ .config = real, .want = .native },
        .{ .config = "{\"model_type\": \"deepseek_v41\"}", .want = null },
        .{ .config = "{\"model_type\": \"glm4_moe\"}", .want = null },
        .{ .config = "{\"architectures\": [\"GlmMoeDsaForCausalLM\"]}", .want = null },
    });
}

test "glm plugin: parse builds the arch's config and its pack dir, and refuses by the arch's own names" {
    if (comptime !host_caps.glm) return error.SkipZigTest;
    const real = try glm.testConfigJson(testing.allocator, "");
    defer testing.allocator.free(real);
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    var diag: sdk.Diag = .{};
    const c = try parse(testing.allocator, &try sdk.ConfigPeek.parse(arena.allocator(), "/m", real), &diag);
    defer freeConfig(testing.allocator, c);
    try testing.expectEqualStrings("/m", c.model_dir.?);
    try testing.expectEqual(sdk.Shell{ .num_experts = 256, .num_layers = 78 }, shell(c));
    const bad = try std.mem.replaceOwned(u8, arena.allocator(), real, "\"sigmoid\"", "\"softmax\"");
    try testing.expectError(error.NotImplemented, parse(testing.allocator, &try sdk.ConfigPeek.parse(arena.allocator(), "/m", bad), &diag));
    try testing.expect(std.mem.indexOf(u8, diag.message(), "scoring_func") != null);
    const tiny = try glm.tinyConfigJson(arena.allocator(), glm.tiny_quant);
    try testing.expectError(error.DimsNotImplemented, parse(testing.allocator, &try sdk.ConfigPeek.parse(arena.allocator(), "/m", tiny), &diag));
}

test "glm plugin: the table the registry builds (owns its decode state, a handover, a request end, a prefix restore, its bills, its max output, a draft lane)" {
    if (comptime !host_caps.glm) return error.SkipZigTest;
    const vt = comptime sdk.Arch.of(@This());
    try testing.expect(vt.caps.owns_decode_state and !vt.caps.batches_decode and vt.caps.prefill_whole_prompt and vt.caps.prefill_yields_last_logits);
    try testing.expect(vt.handover != null and vt.request_end != null and vt.restore_prefix != null and vt.bill != null and vt.prompt_bytes != null and vt.spec == .draft_lane);
    try testing.expect(vt.claim_process != null and vt.release_process != null);
    const cfg: Config = .{ .max_output_tokens = 4096 };
    try testing.expectEqual(@as(u64, 4096), vt.max_output.?(&cfg));
    try testing.expectEqual(@as(u64, 131072), vt.max_output.?(&Config{}));
    try vt.claim_process.?();
    try testing.expectError(error.ExpertReaderInUse, vt.claim_process.?());
    vt.release_process.?();
}

test "glm plugin: the term-wise bill bounds the process exactly as the load preflight bills it (a tiny pack)" {
    if (comptime !host_caps.glm) return error.SkipZigTest;
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    var rbuf: [512]u8 = undefined;
    const dir = try bank_mod.tmpRoot(&tmp, &rbuf);
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const text = try glm.tinyConfigJson(arena.allocator(), glm.tiny_quant);
    var model = try glm.Config.parse(testing.allocator, text, null, null);
    const bin = try bank_mod.writeSynth(testing.allocator, testing.io, tmp.dir, &model, .{});
    defer testing.allocator.free(bin);
    try glm.writeResidents(testing.allocator, testing.io, tmp.dir, &model);
    var cfg: Config = .{ .model = model, .model_dir = dir };
    defer cfg.deinit(testing.allocator);
    const facts: sdk.LoadFacts = .{ .wired_margin_bytes = 2 << 30 };
    const p = try sdk.ConfigPeek.parse(arena.allocator(), dir, text);
    const req: sdk.BillRequest = .{ .peek = &p, .cfg = &cfg, .routes = &{}, .prompt_tokens = bill_mod.fill_prompt_tokens, .max_tokens = bill_mod.fill_max_tokens, .ceiling = 64 << 30, .stop = 2 << 30 };
    const floor: sdk.Rows = .{ .prompt = bill_mod.min_fill_rows, .decode = bill_mod.min_fill_rows };
    try sdk.testing.expectBillBoundsLoad(@This(), testing.allocator, testing.io, &cfg, &facts, &req, floor);
}
