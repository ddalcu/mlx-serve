//! GLM-5.3's config on the host side: the model directory, the parsed model config, the load's facts and the model
//! settings it takes (`model-settings.json`) with the defaults they fall back to. The MTP draft lane is off unless
//! `mtp_depth` sets it; its acceptance is exact unless `mtp_acceptance` names typical.

const std = @import("std");
const host_caps = @import("../../sdk/weights/streamed/host_caps.zig");
const sdk = @import("sdk");
const log = @import("sdk").log;
const glm = @import("glm_moe_dsa.zig");
const acceptance = sdk.acceptance;
const expert_tally = @import("../../sdk/weights/streamed/sdk_ext.zig").expert.tally;

pub const Config = struct {
    /// The pack's directory (config.json, the resident shards, the expert bank); owned by the host's config.
    model_dir: ?[]const u8 = null,
    /// The parsed config.json (owned: `deinit`).
    model: ?glm.Config = null,
    /// The load's facts: the memory in use before the load, the decode and prompt slot rows per layer (null = the
    /// fill's), the residents past the page cache (null = on).
    memory_baseline_bytes: ?u64 = null,
    expert_rows: ?u32 = null,
    expert_prefill_rows: ?u32 = null,
    nocache_weights: ?bool = null,
    /// The longest prompt the module admits (`ctx_size`; null: the standard request's `default_context`). The
    /// construction bills every prompt up to it; a longer one is refused by name before its pass.
    max_context_tokens: ?u32 = null,
    /// A `ctx_size` above the model's limit: refused by name at load (`error.CtxSizeOverModelLimit`).
    ctx_size_over_limit: ?i64 = null,
    /// The most tokens a request may generate past its prompt (`max_output`; null: `default_max_output`). The
    /// construction bills a request of the billed context plus this many tokens; the decode handover grows the slot
    /// rows for the request's own prompt and `max_tokens`.
    max_output_tokens: ?u32 = null,
    /// A `max_output` above the model's limit: refused by name at load (`error.MaxOutputOverModelLimit`).
    max_output_over_limit: ?i64 = null,
    /// A retired key the settings name: refused by name at load (`error.RetiredSettingsKey`).
    retired_key: ?RetiredKey = null,
    /// The routed waves wait on the reads' events instead of the host (null = on).
    expert_event_gates: ?bool = null,
    /// The prompt pass layer by layer over the whole prompt (null = on); off, chunk by chunk.
    layer_major_prefill: ?bool = null,
    /// Prompt routes live at once in one layer (null = 5).
    expert_wide_depth: ?u8 = null,
    /// The decode lookahead's records read ahead per routed call (null = 2; 1 .. `max_lookahead_budget`).
    expert_lookahead_budget: ?u32 = null,
    /// The most fill units the module takes (null = off; `min_units_cap` or more): the construction's fill and the
    /// handover's grow stop at it, so two arms of a measurement hold the same rows when the box beside them moves. It
    /// only lowers what the bill and the live reading admit (never raises it), so it is always safe. The harness's forced
    /// rows (`expert_rows`) are not capped.
    expert_units_cap: ?u32 = null,
    /// An `expert_units_cap` under the fill's floor: refused by name at load (`error.UnitsCapUnderFillFloor`).
    expert_units_cap_under_floor: ?i64 = null,
    /// The least the rest of the box (`glm_moe_dsa_module.Box.others`) reads, in bytes (null = off): the load's baseline
    /// and the handover's reading take at least this, so two arms of a measurement admit at the same target when the
    /// box beside them moves. It only lowers the admission (never raises it), so it is always safe; the twin of
    /// `expert_units_cap` for an arm whose gain is the rows themselves.
    rest_of_box_floor_bytes: ?u64 = null,
    /// The expert stream's rate tally (`expert_resident_ratio`, `expert_tally_window`): every model's keys and defaults.
    tally: expert_tally.Settings = .{},
    /// The MTP draft lane's drafts per round (null or 0 = off; 1 .. `max_mtp_depth`).
    mtp_depth: ?u32 = null,
    /// A `mtp_depth` past the route limit: refused by name at load (`error.MtpDepthOverRouteLimit`).
    mtp_depth_over_limit: ?i64 = null,
    /// The lane's acceptance (`sdk.acceptance`'s names: "exact", "typical"; null = exact). Another name is kept for
    /// the refusal at load (`error.MtpAcceptanceNotImplemented`).
    mtp_acceptance: ?MtpAcceptance = null,
    /// The typical acceptance's delta (null = `sdk.acceptance.DEFAULT_TYPICAL_DELTA`); read only under typical.
    mtp_typical_delta: ?f32 = null,

    /// The longest `ctx_size` the bill takes (GLM-5.3's max_position_embeddings).
    pub const max_ctx_size: i64 = 1 << 20;
    /// The generation a request may take past its prompt when the model sets no `max_output` (128K).
    pub const default_max_output: u32 = 131072;
    /// The longest `max_output` the bill takes (the positions past the prompt within max_position_embeddings).
    pub const max_max_output: i64 = 1 << 20;
    /// The prompt routes one layer may hold live at once (`sdk_ext.expert.stream.max_wide_depth`).
    pub const max_wide_depth: i64 = 5;
    /// The lookahead's widest budget (`sdk_ext.expert.lookahead.max_budget`).
    pub const max_lookahead_budget: i64 = @import("../../sdk/weights/streamed/sdk_ext.zig").expert.lookahead.max_budget;
    /// The least units cap: the fill's floor (`phase_memory.min_fill_rows`), under which a load is refused.
    pub const min_units_cap: i64 = @import("../../sdk/weights/streamed/phase_memory.zig").min_fill_rows;
    /// The deepest draft a round verifies: its depth + 1 rows of top-8 routes in the stream's decode lane
    /// (`sdk_ext.expert.policy.max_route_ids` = 48).
    pub const max_mtp_depth: u32 = @import("../../sdk/weights/streamed/sdk_ext.zig").expert.policy.max_route_ids / glm.routed_top_k - 1;

    pub const MtpAcceptance = enum { exact, typical, other };

    /// Keys this arch no longer takes. A pack that still sets one is refused by name at load, never ignored.
    pub const RetiredKey = enum { numeric_tier };

    pub fn deinit(c: *Config, gpa: std.mem.Allocator) void {
        if (c.model) |*m| m.deinit(gpa);
        c.model = null;
    }

    /// The host's load facts onto this config (the module and the bill read them under these names).
    pub fn withFacts(c: Config, facts: *const sdk.LoadFacts) Config {
        var out = c;
        out.memory_baseline_bytes = facts.memory_baseline_bytes;
        out.expert_rows = facts.expert_rows;
        out.expert_prefill_rows = facts.expert_prefill_rows;
        out.nocache_weights = facts.nocache_weights;
        return out;
    }

    /// This model's model-settings.json object: each key this arch takes, set when present and valid (anything
    /// else is unset), and one `[model-settings]` line when any was.
    pub fn applySettings(c: *Config, raw: std.json.Value) void {
        const obj = switch (raw) {
            .object => |o| o,
            else => return,
        };
        var any = false;
        inline for (.{ "expert_event_gates", "layer_major_prefill" }) |key| {
            if (obj.get(key)) |v| if (v == .bool) {
                @field(c, key) = v.bool;
                any = true;
            };
        }
        inline for (comptime std.meta.fieldNames(RetiredKey)) |key| {
            if (obj.get(key) != null) c.retired_key = @field(RetiredKey, key);
        }
        if (obj.get("expert_wide_depth")) |v| if (v == .integer and v.integer >= 1 and v.integer <= max_wide_depth) {
            c.expert_wide_depth = @intCast(v.integer);
            any = true;
        };
        if (obj.get("expert_lookahead_budget")) |v| if (v == .integer and v.integer >= 1 and v.integer <= max_lookahead_budget) {
            c.expert_lookahead_budget = @intCast(v.integer);
            any = true;
        };
        if (obj.get("expert_units_cap")) |v| if (v == .integer and v.integer >= 1 and v.integer <= std.math.maxInt(u32)) {
            if (v.integer >= min_units_cap) c.expert_units_cap = @intCast(v.integer) else c.expert_units_cap_under_floor = v.integer;
            any = true;
        };
        if (obj.get("rest_of_box_floor_bytes")) |v| if (v == .integer and v.integer >= 1) {
            c.rest_of_box_floor_bytes = @intCast(v.integer);
            any = true;
        };
        if (c.tally.apply(obj)) any = true;
        if (obj.get("ctx_size")) |v| if (v == .integer and v.integer >= 1) {
            if (v.integer <= max_ctx_size) c.max_context_tokens = @intCast(v.integer) else c.ctx_size_over_limit = v.integer;
            any = true;
        };
        if (obj.get("max_output")) |v| if (v == .integer and v.integer >= 1) {
            if (v.integer <= max_max_output) c.max_output_tokens = @intCast(v.integer) else c.max_output_over_limit = v.integer;
            any = true;
        };
        if (obj.get("mtp_depth")) |v| if (v == .integer and v.integer >= 0) {
            if (v.integer <= max_mtp_depth) c.mtp_depth = @intCast(v.integer) else c.mtp_depth_over_limit = v.integer;
            any = true;
        };
        if (obj.get("mtp_acceptance")) |v| if (v == .string) {
            const m = acceptance.fromName(v.string);
            c.mtp_acceptance = if (m) |x| switch (x) {
                .exact => .exact,
                .typical => .typical,
                else => .other,
            } else .other;
            any = true;
        };
        if (obj.get("mtp_typical_delta")) |v| {
            const d: ?f64 = switch (v) {
                .float => |f| f,
                .integer => |i| @floatFromInt(i),
                else => null,
            };
            if (d) |x| if (std.math.isFinite(x) and x > 0) {
                c.mtp_typical_delta = @floatCast(x);
                any = true;
            };
        }
        if (any) log.info("[model-settings] glm_moe_dsa: event_gates={s} layer_major_prefill={s} wide_depth={d} lookahead_budget={d} units_cap={d} rest_of_box_floor={d} resident_ratio={d} tally_window={d} billed_context={d} max_output={d} mtp_depth={d} mtp_acceptance={s} mtp_typical_delta={d}\n", .{
            onOff(c.expert_event_gates), onOff(c.layer_major_prefill), c.expert_wide_depth orelse 0, c.lookaheadBudget(), c.expert_units_cap orelse 0, c.rest_of_box_floor_bytes orelse 0, c.tally.ratio, c.tally.window, c.max_context_tokens orelse 0, c.maxOutput(), c.mtpDepth(), c.acceptanceName(), c.typical().delta,
        });
    }

    /// The generation a request may take past its prompt (`max_output`, else `default_max_output`).
    pub fn maxOutput(c: *const Config) u32 {
        return c.max_output_tokens orelse default_max_output;
    }

    /// A `max_output` over the model's limit, refused by name (never a silent fall back to the default).
    pub fn checkMaxOutput(c: *const Config) error{MaxOutputOverModelLimit}!void {
        const v = c.max_output_over_limit orelse return;
        log.warn("glm_moe_dsa: load refused: max_output {d} is over the model's limit of {d} tokens (MaxOutputOverModelLimit)\n", .{ v, max_max_output });
        return error.MaxOutputOverModelLimit;
    }

    /// The lane's drafts per round (0 = off).
    pub fn mtpDepth(c: *const Config) u32 {
        return c.mtp_depth orelse 0;
    }

    /// The lane's acceptance as the SDK names it: exact unless the setting names typical.
    pub fn mtpAcceptance(c: *const Config) acceptance.Mode {
        return if (c.mtp_acceptance == .typical) .{ .typical = c.typical() } else .exact;
    }

    fn typical(c: *const Config) @FieldType(acceptance.Mode, "typical") {
        return .{ .delta = c.mtp_typical_delta orelse acceptance.DEFAULT_TYPICAL_DELTA };
    }

    fn acceptanceName(c: *const Config) []const u8 {
        return if (c.mtp_acceptance) |m| @tagName(m) else "default";
    }

    /// The draft lane's settings, refused by name at load: a depth past the route limit, an acceptance the lane does
    /// not implement.
    pub fn checkMtp(c: *const Config) error{ MtpDepthOverRouteLimit, MtpAcceptanceNotImplemented }!void {
        if (c.mtp_depth_over_limit) |v| {
            log.warn("glm_moe_dsa: load refused: mtp_depth {d} is over the route limit: a round verifies depth + 1 rows of {d} routes in one {d}-route call, so the depth is at most {d} (MtpDepthOverRouteLimit)\n", .{ v, glm.routed_top_k, @import("../../sdk/weights/streamed/sdk_ext.zig").expert.policy.max_route_ids, max_mtp_depth });
            return error.MtpDepthOverRouteLimit;
        }
        if (c.mtp_acceptance == .other and c.mtpDepth() > 0) {
            log.warn("glm_moe_dsa: load refused: mtp_acceptance names a mode the draft lane does not implement (exact and typical are; MtpAcceptanceNotImplemented)\n", .{});
            return error.MtpAcceptanceNotImplemented;
        }
    }

    fn onOff(v: ?bool) []const u8 {
        return if (v) |b| (if (b) "on" else "off") else "default";
    }

    pub fn eventGates(c: *const Config) bool {
        return c.expert_event_gates orelse true;
    }

    pub fn layerMajor(c: *const Config) bool {
        return c.layer_major_prefill orelse true;
    }

    pub fn wideDepth(c: *const Config) u8 {
        return c.expert_wide_depth orelse 5;
    }

    pub fn lookaheadBudget(c: *const Config) u32 {
        return c.expert_lookahead_budget orelse 2;
    }

    /// A `ctx_size` over the model's limit, refused by name (never a silent fall back to the standard context).
    pub fn checkCtxSize(c: *const Config) error{CtxSizeOverModelLimit}!void {
        const v = c.ctx_size_over_limit orelse return;
        log.warn("glm_moe_dsa: load refused: ctx_size {d} is over the model's limit of {d} tokens (CtxSizeOverModelLimit)\n", .{ v, max_ctx_size });
        return error.CtxSizeOverModelLimit;
    }

    /// A retired key, refused by name (never a silent ignore).
    pub fn checkRetired(c: *const Config) error{RetiredSettingsKey}!void {
        const k = c.retired_key orelse return;
        log.warn("glm_moe_dsa: load refused: model-settings.json sets {s}, a key this arch no longer takes; remove it (RetiredSettingsKey)\n", .{@tagName(k)});
        return error.RetiredSettingsKey;
    }

    /// A units cap under the fill's floor, refused by name (a capped fill would take fewer rows than a load may).
    pub fn checkUnitsCap(c: *const Config) error{UnitsCapUnderFillFloor}!void {
        const v = c.expert_units_cap_under_floor orelse return;
        log.warn("glm_moe_dsa: load refused: expert_units_cap {d} is under the fill's floor of {d} units (UnitsCapUnderFillFloor)\n", .{ v, min_units_cap });
        return error.UnitsCapUnderFillFloor;
    }
};

const testing = std.testing;

fn settingsOf(json: []const u8) !Config {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    var c: Config = .{};
    c.applySettings(try std.json.parseFromSliceLeaky(std.json.Value, arena.allocator(), json, .{}));
    return c;
}

test "glm settings: the gates and the prompt order are bools; anything else is unset" {
    if (comptime !host_caps.glm) return error.SkipZigTest;
    try testing.expectEqual(@as(?bool, false), (try settingsOf("{\"expert_event_gates\": false}")).expert_event_gates);
    try testing.expectEqual(@as(?bool, null), (try settingsOf("{\"expert_event_gates\": 0}")).expert_event_gates);
    try testing.expectEqual(@as(?bool, false), (try settingsOf("{\"layer_major_prefill\": false}")).layer_major_prefill);
    const d: Config = .{};
    try testing.expect(d.eventGates() and d.layerMajor() and d.wideDepth() == 5);
    try testing.expectEqual(Config{}, try settingsOf("[1]"));
}

test "glm settings: expert_wide_depth is 1 to 5; ctx_size bills every prompt up to it, over the limit it is kept for the refusal" {
    if (comptime !host_caps.glm) return error.SkipZigTest;
    try testing.expectEqual(@as(?u8, 5), (try settingsOf("{\"expert_wide_depth\": 5}")).expert_wide_depth);
    try testing.expectEqual(@as(?u8, null), (try settingsOf("{\"expert_wide_depth\": 6}")).expert_wide_depth);
    try testing.expectEqual(@as(?u8, null), (try settingsOf("{\"expert_wide_depth\": 0}")).expert_wide_depth);
    try testing.expectEqual(@as(u32, 2), (Config{}).lookaheadBudget());
    try testing.expectEqual(@as(u32, 1), (try settingsOf("{\"expert_lookahead_budget\": 1}")).lookaheadBudget());
    try testing.expectEqual(@as(u32, 4), (try settingsOf("{\"expert_lookahead_budget\": 4}")).lookaheadBudget());
    try testing.expectEqual(@as(u32, 2), (try settingsOf("{\"expert_lookahead_budget\": 5}")).lookaheadBudget());
    try testing.expectEqual(@as(u32, 2), (try settingsOf("{\"expert_lookahead_budget\": 0}")).lookaheadBudget());
    try testing.expectEqual(@as(?u32, null), (Config{}).expert_units_cap);
    try testing.expectEqual(@as(?u32, 85), (try settingsOf("{\"expert_units_cap\": 85}")).expert_units_cap);
    try testing.expectEqual(@as(?u32, null), (try settingsOf("{\"expert_units_cap\": 0}")).expert_units_cap);
    try testing.expectEqual(@as(?u32, 16), (try settingsOf("{\"expert_units_cap\": 16}")).expert_units_cap);
    const under = try settingsOf("{\"expert_units_cap\": 15}");
    try testing.expectEqual(@as(?u32, null), under.expert_units_cap);
    try testing.expectError(error.UnitsCapUnderFillFloor, under.checkUnitsCap());
    try (Config{}).checkUnitsCap();
    try testing.expectEqual(@as(?u32, null), (try settingsOf("{\"expert_units_cap\": \"85\"}")).expert_units_cap);
    try testing.expectEqual(@as(?u64, null), (Config{}).rest_of_box_floor_bytes);
    try testing.expectEqual(@as(?u64, 21_000_000_000), (try settingsOf("{\"rest_of_box_floor_bytes\": 21000000000}")).rest_of_box_floor_bytes);
    try testing.expectEqual(@as(?u64, null), (try settingsOf("{\"rest_of_box_floor_bytes\": 0}")).rest_of_box_floor_bytes);
    try testing.expectEqual(@as(?u32, 131072), (try settingsOf("{\"ctx_size\": 131072}")).max_context_tokens);
    try testing.expectEqual(@as(?u32, null), (try settingsOf("{\"ctx_size\": \"131072\"}")).max_context_tokens);
    const over = try settingsOf("{\"ctx_size\": 2097152}");
    try testing.expectEqual(@as(?u32, null), over.max_context_tokens);
    try testing.expectError(error.CtxSizeOverModelLimit, over.checkCtxSize());
    try (Config{}).checkCtxSize();
}

test "glm settings: the retired numeric_tier is refused by name at load" {
    if (comptime !host_caps.glm) return error.SkipZigTest;
    try testing.expectError(error.RetiredSettingsKey, (try settingsOf("{\"numeric_tier\": \"stock\"}")).checkRetired());
    try (Config{}).checkRetired();
}

test "glm settings: max_output defaults to 128K, takes 1 to 1,048,576, and over the limit is kept for the refusal" {
    if (comptime !host_caps.glm) return error.SkipZigTest;
    try testing.expectEqual(@as(u32, 131072), (Config{}).maxOutput());
    try testing.expectEqual(@as(u32, 4096), (try settingsOf("{\"max_output\": 4096}")).maxOutput());
    try testing.expectEqual(@as(u32, 131072), (try settingsOf("{\"max_output\": 0}")).maxOutput());
    try testing.expectEqual(@as(u32, 131072), (try settingsOf("{\"max_output\": \"4096\"}")).maxOutput());
    const over = try settingsOf("{\"max_output\": 2097152}");
    try testing.expectEqual(@as(u32, 131072), over.maxOutput());
    try testing.expectError(error.MaxOutputOverModelLimit, over.checkMaxOutput());
    try (Config{}).checkMaxOutput();
}

test "glm settings: mtp_depth is 0 to the route limit's 5 (past it kept for the refusal); acceptance is exact unless typical is named" {
    if (comptime !host_caps.glm) return error.SkipZigTest;
    try testing.expectEqual(@as(u32, 5), Config.max_mtp_depth);
    try testing.expectEqual(@as(u32, 0), (Config{}).mtpDepth());
    try testing.expectEqual(@as(u32, 3), (try settingsOf("{\"mtp_depth\": 3}")).mtpDepth());
    try testing.expectEqual(@as(u32, 5), (try settingsOf("{\"mtp_depth\": 5}")).mtpDepth());
    try testing.expectEqual(@as(u32, 0), (try settingsOf("{\"mtp_depth\": 0}")).mtpDepth());
    try testing.expectEqual(@as(u32, 0), (try settingsOf("{\"mtp_depth\": \"3\"}")).mtpDepth());
    const over = try settingsOf("{\"mtp_depth\": 6}");
    try testing.expectEqual(@as(u32, 0), over.mtpDepth());
    try testing.expectError(error.MtpDepthOverRouteLimit, over.checkMtp());
    try (Config{}).checkMtp();
    // Never typical by default: no setting, exact, an unknown name, or a delta alone all stay exact.
    for ([_][]const u8{ "{}", "{\"mtp_acceptance\": \"exact\"}", "{\"mtp_typical_delta\": 0.5}", "{\"mtp_acceptance\": 1}" }) |j|
        try testing.expect((try settingsOf(j)).mtpAcceptance() == .exact);
    const t = try settingsOf("{\"mtp_acceptance\": \"typical\"}");
    try testing.expectEqual(acceptance.DEFAULT_TYPICAL_DELTA, t.mtpAcceptance().typical.delta);
    const t3 = try settingsOf("{\"mtp_acceptance\": \"typical\", \"mtp_typical_delta\": 0.3, \"mtp_depth\": 2}");
    try testing.expectEqual(@as(f32, 0.3), t3.mtpAcceptance().typical.delta);
    try testing.expectEqual(@as(f32, 1.0), t3.mtpAcceptance().typical.eps);
    try testing.expectEqual(@as(?f32, null), (try settingsOf("{\"mtp_typical_delta\": -1}")).mtp_typical_delta);
    // A mode the lane does not implement (tokenv3) or a name nobody knows is refused once the lane is on.
    try (try settingsOf("{\"mtp_acceptance\": \"tokenv3\"}")).checkMtp();
    try testing.expectError(error.MtpAcceptanceNotImplemented, (try settingsOf("{\"mtp_acceptance\": \"tokenv3\", \"mtp_depth\": 1}")).checkMtp());
    try testing.expectError(error.MtpAcceptanceNotImplemented, (try settingsOf("{\"mtp_acceptance\": \"fast\", \"mtp_depth\": 1}")).checkMtp());
}

test "glm settings: the load facts replace the fill's inputs, the settings stay" {
    if (comptime !host_caps.glm) return error.SkipZigTest;
    var s = try settingsOf("{\"expert_wide_depth\": 3}");
    s.expert_rows = 99;
    const c = s.withFacts(&.{ .memory_baseline_bytes = 9_000_000_000, .expert_prefill_rows = 120, .nocache_weights = false, .wired_margin_bytes = 2 << 30 });
    try testing.expectEqual(@as(?u64, 9_000_000_000), c.memory_baseline_bytes);
    try testing.expectEqual(@as(?u32, null), c.expert_rows);
    try testing.expectEqual(@as(?u32, 120), c.expert_prefill_rows);
    try testing.expectEqual(@as(?u8, 3), c.expert_wide_depth);
}
