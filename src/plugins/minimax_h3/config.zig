//! MiniMax-H3's claims, caps and bills, MLX-free: a `minimax_h3` config.json; the partition (FL2VA, REF2VA, FastH3
//! Omni) from its declared `tasks`; the staged residency plan and each request's own activations as bills.

const std = @import("std");
const abi = @import("../../abi/media.zig");
const env = @import("../../env.zig");
const sdk = @import("../../sdk/root.zig");
const pack_bytes = @import("../../sdk/media/pack_bytes.zig");
const config_fields = @import("../../sdk/media/config_fields.zig");
const h3 = @import("geometry.zig");

pub const plugin = @import("../../sdk/plugin.zig").Plugin{
    .name = "minimax_h3",
    .kind = .media,
    .claims = &model_types,
    .gate_models = &.{"ddalcu/MiniMax-H3-MLX-Serve-4bit"},
};

pub const model_types = [_][]const u8{"minimax_h3"};

pub const claims = sdk.mediaClaimsByType(&model_types);

pub const required_file = "transformer.safetensors";

pub fn caps(dir: *const abi.PackDir, out: *abi.MediaCaps) callconv(.c) void {
    _ = dir;
    out.* = .{};
}

/// True when the pack's `config.json` declares `task`. The partitions share every file and geometry number: only the
/// converter's declared task list tells them apart, and an FL2VA pack handed references would silently ignore them.
pub fn declaresTask(a: std.mem.Allocator, config_json: []const u8, task: []const u8) bool {
    var parsed = std.json.parseFromSlice(std.json.Value, a, config_json, .{}) catch return false;
    defer parsed.deinit();
    if (parsed.value != .object) return false;
    const tasks = parsed.value.object.get("tasks") orelse return false;
    if (tasks != .array) return false;
    for (tasks.array.items) |t| {
        if (t == .string and std.mem.eql(u8, t.string, task)) return true;
    }
    return false;
}

/// The pack's geometry from its `config.json` (published values where a key is absent); a key that does not fit
/// the forward is refused by name.
pub fn geometry(a: std.mem.Allocator, config_json: []const u8) !h3.Geometry {
    var parsed = std.json.parseFromSlice(std.json.Value, a, config_json, .{}) catch return error.BadPackConfig;
    defer parsed.deinit();
    if (parsed.value != .object) return error.BadPackConfig;
    const root = parsed.value.object;
    var g: h3.Geometry = .{};
    try config_fields.fill(h3.Config, .{
        .{ "hidden_size", "hidden_size" },             .{ "num_layers", "num_layers" },
        .{ "token_refiner_num_layers", "token_refiner_num_layers" }, .{ "num_attention_heads", "num_attention_heads" },
        .{ "attention_head_dim", "attention_head_dim" }, .{ "ffn_hidden_size", "ffn_hidden_size" },
        .{ "latents_dim", "latents_dim" },             .{ "audio_latents_dim", "audio_latents_dim" },
        .{ "text_dim", "text_dim" },                   .{ "timestep_input_dim", "timestep_input_dim" },
        .{ "time_embed_hidden_size", "time_embed_hidden_size" }, .{ "time_embed_dim", "time_embed_dim" },
        .{ "rope_inv_freq_len", "rope_inv_freq_len" },
    }, try config_fields.section(root, "transformer"), &g.dit);
    try config_fields.fill(h3.TeConfig, .{
        .{ "hidden", "hidden" },     .{ "layers", "layers" },             .{ "heads", "heads" },
        .{ "kv_heads", "kv_heads" }, .{ "head_dim", "head_dim" },         .{ "intermediate", "intermediate" },
        .{ "theta", "theta" },
    }, try config_fields.section(root, "text_encoder"), &g.te);
    try config_fields.fill(@TypeOf(g.vae), .{ .{ "layers", "layers" }, .{ "heads", "heads" }, .{ "head_dim", "head_dim" } }, try config_fields.section(root, "video_vae"), &g.vae);
    if (root.get("two_time")) |v| {
        if (v != .object or root.get("pdd_steps") != null) return error.BadPackConfig;
        g.two_time = try h3.TwoTime.parse(v.object);
        try config_fields.fill(h3.Config, .{.{ "gate", "two_time_gate" }}, v.object, &g.dit);
        if (!(g.dit.two_time_gate > 0)) return error.BadPackConfig;
    }
    const d = g.dit;
    if (d.rotDim() > d.attention_head_dim or d.timestep_input_dim % 2 != 0 or d.text_dim != @as(u32, @intCast(g.te.hidden)) or
        @rem(g.te.heads, g.te.kv_heads) != 0 or g.vae.in_channels != d.latents_dim) return error.BadPackConfig;
    return g;
}

// ── the bills ──

/// The DiT's resident share once AdaLN precompute frees the modulation weights (~35% of the file on the shipped
/// packs), billed at 0.65 so a pack whose AdaLN share is smaller still fails safe.
pub const DIT_RESIDENT_PCT: u64 = 65;

/// What one denoising run holds above its weights, per sequence row (process footprint on an M5 Pro, 8.9k-37.7k
/// rows), and the floor every request carries.
pub const ROW_BYTES: u64 = 384 * 1024;
pub const REQUEST_BASE_BYTES: u64 = 1 << 30;

/// The DiT term. `precompute` is false for a resident DiT, whose modulation weights are never freed
/// and the whole file stays resident, so the shed size would under-bill by ~12 GiB into an uncatchable Metal OOM.
pub fn ditResidentBytes(dit_file: u64, precompute: bool) u64 {
    if (!precompute) return dit_file;
    return dit_file * DIT_RESIDENT_PCT / 100;
}

/// The staged residency plan as a bill: the text encoder is loaded, run and FREED before the DiT loads, and the DiT
/// is released before the VAEs load, so the peak is the BIGGEST stage, never a sum (the two VAEs are one stage).
/// The generating stages carry the smallest request's transients; each request prices its own (`requestRows`).
pub fn peakBytes(te: u64, dit_resident: u64, video_vae: u64, audio_vae: u64) u64 {
    const generating = @max(dit_resident, video_vae + audio_vae);
    if (te == 0 and generating == 0) return 0; // unknown dir: never block
    return @max(te, generating + REQUEST_BASE_BYTES);
}

/// Sequence rows one window puts through the DiT: video latents on the 32-pixel grid, stereo audio latents, and a
/// frame of rows per keyframe. Prompt and reference rows ride the margin.
pub fn requestRows(width: u32, height: u32, frames: u32, keyframes: u32) u64 {
    const shape = h3.temporalShape(frames);
    const frame_rows: u64 = (height / (h3.VAE_SPATIAL * h3.PATCH_H)) * (width / (h3.VAE_SPATIAL * h3.PATCH_W));
    return (shape.latent_t + keyframes) * frame_rows + 2 * @as(u64, shape.audio_t);
}

/// The request's activations: the per-step transients, plus the fast recipe's per-block attention cache
/// (`bcast_row_bytes` = layers x hidden x bf16) while that recipe is on.
pub fn activationBytes(rows: u64, bcast_row_bytes: u64) u64 {
    return REQUEST_BASE_BYTES + rows * (ROW_BYTES + bcast_row_bytes);
}

/// What H3 holds when it keeps every piece loaded between requests: the text encoder, the WHOLE DiT (the AdaLN
/// weights stay, since they serve any schedule), both VAEs, the LoRA and the request's activations. Zero when a size
/// is unknown, which never keeps anything.
pub fn residentBytes(te: u64, dit_file: u64, vaes: u64, lora: u64, activations: u64) u64 {
    if (te == 0 or dit_file == 0) return 0;
    return te + dit_file + vaes + lora + activations;
}

/// Memory the next request can still use: the tighter of host RAM that is free and the GPU working-set room. Both
/// exclude what the cache already holds. Zero when host RAM is unknown.
pub fn availBytes(host_avail: u64, gpu_limit: u64, gpu_active: u64) u64 {
    if (host_avail == 0) return 0;
    if (gpu_limit == 0) return host_avail;
    return @min(host_avail, gpu_limit -| gpu_active);
}

/// What a staged DiT run is checked against: the bytes it needs and the bytes that hold them. CUDA keeps file-loaded
/// weights in managed memory that pages to host RAM, so there the DiT is billed to RAM and only the activations to
/// the GPU.
pub fn canvasBill(dit: u64, activations: u64, host_avail: u64, gpu_limit: u64, gpu_active: u64, weights_page: bool) struct { need: u64, avail: u64 } {
    if (!weights_page) return .{ .need = dit + activations, .avail = availBytes(host_avail, gpu_limit, gpu_active) };
    if (dit > host_avail) return .{ .need = dit, .avail = host_avail };
    return .{ .need = activations, .avail = if (gpu_limit == 0) host_avail else gpu_limit -| gpu_active };
}

/// Headroom kept free beyond the resident set: an eighth of the RAM it does not need, floored.
pub fn residentMargin(total_ram: u64) u64 {
    return @max(10 * 1024 * 1024 * 1024, total_ram / 16);
}

/// `MLX_SERVE_H3_RESIDENT=0` turns residency off (the A/B arm and the kill switch).
pub fn residentEnabled() bool {
    const raw = env.get(.MLX_SERVE_H3_RESIDENT) orelse return true;
    return !std.mem.eql(u8, std.mem.span(raw), "0");
}

/// Keep H3 resident only while it fits: what is free right now plus what the cache already holds must cover the
/// whole set and a margin. All or nothing, because a half-resident set breaks the staged plan's disjoint-stage peak.
/// `avail` excludes the cache's own bytes.
pub fn keepResident(avail: u64, cache_now: u64, need: u64, margin: u64) bool {
    return need != 0 and avail +| cache_now >= need +| margin;
}

/// The DiT's shards as loaded: int8 copies under `--w8a8`, packed while `--ane-video` reads them.
pub fn ditFileBytes(dir: *const abi.PackDir, m: *const abi.Machine) u64 {
    return pack_bytes.ditBytes(dir, m.w8a8 and !m.ane_video);
}

/// The staged plan's peak; the Turbo LoRA (when shipped) rides the DiT term at full size.
pub fn billLoad(dir: *const abi.PackDir, m: *const abi.Machine) callconv(.c) u64 {
    const dit = ditResidentBytes(ditFileBytes(dir, m), true) + dir.fileBytes("turbo_lora.safetensors");
    return peakBytes(dir.fileBytes("text_encoder.safetensors"), dit, dir.fileBytes("video_vae.safetensors"), dir.fileBytes("audio_vae.safetensors"));
}

/// What a video VAE encode holds beside the VAE and its lifted pool: one tile pass, and the clip's own copies for an
/// init_video clip (keyframes and references are small beside a pass).
pub fn encodeHeldBytes(width: u32, height: u32, frames: u32) u64 {
    return h3.encodePassBytes() + h3.encodeClipBytes(width, height, frames);
}

/// The pool an encode may lift to inside a plan that has `room` and holds `held` beside the lift, never past `billed`.
pub fn encodeLift(billed: u64, total_ram: u64, room: u64, held: u64) u64 {
    return @min(billed, h3.encodeLiftBytes(total_ram, room -| held));
}

/// The lift a request is billed for: what the staged encode stage (the video VAE and `held`) leaves of the memory
/// the engine saw free at load, less the margin kept for everything else. The step lifts at most this, so the bill
/// covers the pool the encode runs under.
pub fn billedEncodeLift(video_vae: u64, held: u64, m: *const abi.Machine) u64 {
    const room = availBytes(m.ram_available, m.gpu_working_set, 0) -| residentMargin(m.ram_total);
    return encodeLift(std.math.maxInt(u64), m.ram_total, room, video_vae + held);
}

/// A request with `init_video` reserves its encode stage too: the video VAE (its file bills the f32 encoder, which
/// is smaller), one pass, the clip and the billed lift. Any other request reserves the load's peak (0 here); the step
/// refuses a request that does not fit against live memory, by name (`plan`).
pub fn billRequest(dir: *const abi.PackDir, m: *const abi.Machine, body: abi.Str) callconv(.c) u64 {
    const c = initCanvas(body.slice()) orelse return 0;
    const held = encodeHeldBytes(c.width, c.height, h3.temporalShape(c.frames).frame_count);
    const vae = dir.fileBytes("video_vae.safetensors");
    return @max(billLoad(dir, m), vae + held + billedEncodeLift(vae, held, m));
}

/// The pack's file sizes a plan prices.
pub const PackSizes = struct { te: u64, dit_file: u64, video_vae: u64, audio_vae: u64, turbo_lora: u64 };

pub const PlanIn = struct {
    sizes: PackSizes,
    /// The DiT term of the staged plan: AdaLN precompute on (not a two-time pack), the Turbo LoRA in use.
    precompute: bool,
    turbo: bool,
    /// Residency is allowed (`residentEnabled`, no `--ane-video`), and what the cache holds now.
    may_keep: bool,
    cache_now: u64,
    host_avail: u64,
    total_ram: u64,
    gpu_limit: u64,
    gpu_active: u64,
    weights_page: bool,
    activations: u64,
    /// What the request's encodes hold beside the lift (`encodeHeldBytes`); `init` when one is the init_video clip.
    encode_held: u64,
    init: bool,
    billed_lift: u64,
};

pub const Plan = struct {
    /// The resident set's bytes, or null for the staged plan.
    resident: ?u64 = null,
    /// The pool the encodes run under.
    lift: u64 = 0,
    refused: ?struct { what: enum { encode, canvas }, need: u64, avail: u64, dit: u64 = 0 } = null,
};

/// A request's memory plan before any stage loads (a Metal OOM mid-step is uncatchable): resident while the whole set,
/// the request and a margin fit; else staged, refused by name when its encode or DiT stage does not fit. The encode's
/// lift then takes what the plan leaves of live free memory less that margin, never past the billed lift.
pub fn plan(in: PlanIn) Plan {
    const sz = in.sizes;
    const avail = availBytes(in.host_avail, in.gpu_limit, in.gpu_active);
    const init_held = if (in.init) in.encode_held else 0;
    const need = residentBytes(sz.te, sz.dit_file, sz.video_vae + sz.audio_vae, sz.turbo_lora, @max(in.activations, init_held));
    const margin = residentMargin(in.total_ram);
    if (in.may_keep and keepResident(avail, in.cache_now, need, margin)) {
        const set = need - @max(in.activations, init_held);
        const lift = encodeLift(in.billed_lift, in.total_ram, (avail +| in.cache_now) -| margin, set + in.encode_held);
        // What the cache holds afterwards: the weights and the request's activations. The clip's f32 copies and the
        // encode's lift are freed when the request ends, so counting them would let the next request keep the set
        // on room that is not there.
        return .{ .resident = set + in.activations, .lift = lift };
    }
    if (in.init and avail > 0 and sz.video_vae + in.encode_held > avail)
        return .{ .refused = .{ .what = .encode, .need = sz.video_vae + in.encode_held, .avail = avail } };
    const dit = ditResidentBytes(sz.dit_file, in.precompute) + (if (in.turbo) sz.turbo_lora else 0);
    const bill = canvasBill(dit, in.activations, in.host_avail, in.gpu_limit, in.gpu_active, in.weights_page);
    if (dit > 0 and bill.avail > 0 and bill.need > bill.avail)
        return .{ .refused = .{ .what = .canvas, .need = bill.need, .avail = bill.avail, .dit = dit } };
    return .{ .lift = encodeLift(in.billed_lift, in.total_ram, avail -| margin, sz.video_vae + in.encode_held) };
}

const InitCanvas = struct { width: u32 = h3.DEFAULT_SIDE, height: u32 = h3.DEFAULT_SIDE, frames: u32 = h3.DEFAULT_FRAMES };

/// The canvas of a body that carries `init_video`, read without copying its frames; null without one.
fn initCanvas(body: []const u8) ?InitCanvas {
    var stack: [256]u8 = undefined;
    var fba = std.heap.FixedBufferAllocator.init(&stack);
    var sc = std.json.Scanner.initCompleteInput(fba.allocator(), body);
    if ((sc.next() catch return null) != .object_begin) return null;
    var c: InitCanvas = .{};
    var has_init = false;
    while (true) {
        const key = switch (sc.next() catch return null) {
            .string => |k| k,
            else => break,
        };
        const field: ?*u32 = if (std.mem.eql(u8, key, "width")) &c.width else if (std.mem.eql(u8, key, "height")) &c.height else if (std.mem.eql(u8, key, "num_frames")) &c.frames else null;
        if (field) |f| {
            switch (sc.next() catch return null) {
                .number => |n| f.* = std.fmt.parseInt(u32, n, 10) catch f.*,
                .array_begin, .object_begin => return null,
                else => {},
            }
            continue;
        }
        if (std.mem.eql(u8, key, "init_video")) has_init = (sc.peekNextTokenType() catch return null) != .null;
        sc.skipValue() catch return null;
    }
    return if (has_init) c else null;
}

test "h3 staged-residency peak bills the BIGGEST stage, never a sum of disjoint ones" {
    const GB: u64 = 1024 * 1024 * 1024;
    const act = REQUEST_BASE_BYTES;
    // Three disjoint stages: the TE is freed before the DiT loads, the DiT is
    // freed before the VAEs load. Billing any two together refuses a Mac that
    // would work — the VAEs used to be added to the DiT term.
    try std.testing.expectEqual(35 * GB + act, peakBytes(28 * GB, 35 * GB, 5 * GB, 1 * GB));
    // The two VAEs DO coexist (the video decoder is still resident when the
    // audio one loads), so they are one stage.
    try std.testing.expectEqual(9 * GB + act, peakBytes(4 * GB, 3 * GB, 5 * GB, 4 * GB));
    // The TE stage carries no generation transients — one forward over a few
    // hundred prompt rows. Adding them to it is what refused the 8-bit pack.
    try std.testing.expectEqual(48 * GB, peakBytes(48 * GB, 35 * GB, 5 * GB, 1 * GB));
    // All-unknown must stay 0: the preflight treats 0 as "unknown, never block".
    try std.testing.expectEqual(@as(u64, 0), peakBytes(0, 0, 0, 0));

    // The real 8-bit pack on a 48 GB Mac (auto cap 29.95 GiB): 26.28 TE,
    // 32.83 DiT + 0.73 turbo, 4.85 + 0.56 VAEs. Measured process peak 26 GB.
    const MB: u64 = 1024 * 1024;
    const real = peakBytes(
        26_910 * MB,
        ditResidentBytes(33_618 * MB, true) + 747 * MB,
        4_966 * MB,
        573 * MB,
    );
    try std.testing.expect(real < 29 * GB); // fits the 48 GB Mac's auto cap
    try std.testing.expect(real > 24 * GB); // and stays above the measured peak

    // The 4-bit pack's files: the text encoder is the biggest stage (measured 14.23 GB resident
    // vs the DiT's 11.44 at 832x480 x 22 frames), not the DiT plus a worst-case canvas.
    const te4: u64 = 15_804_791_921;
    try std.testing.expectEqual(te4, peakBytes(te4, ditResidentBytes(18_698_813_290, true) + 779_849_816, 5_207_808_496, 605_254_808));
}

test "h3 request rows: latent frames on the 32-pixel grid, stereo audio, one frame of rows per keyframe" {
    try std.testing.expectEqual(@as(u64, 12846), requestRows(768, 448, 124, 0)); // live: 12854 rows with 8 prompt tokens
    try std.testing.expectEqual(@as(u64, 37710), requestRows(1344, 768, 124, 0)); // live: 37719 with 9
    try std.testing.expectEqual(@as(u64, 12846 + 2 * 336), requestRows(768, 448, 124, 2));
}

test "h3 request bill stays above the measured peaks and refuses the canvas that died (#764)" {
    const GB: u64 = 1 << 30;
    const dit = ditResidentBytes(18_698_813_290, true); // the 4-bit FL2VA pack
    const lora: u64 = 779_849_816;
    const bcast: u64 = 50 * 5376 * 2;
    // Process peaks measured on an M5 Pro 48 GB at 124 frames.
    try std.testing.expect(dit + lora + activationBytes(37718, 0) >= 26 * GB); // 1344x768 turbo: 26 GB
    try std.testing.expect(dit + lora + activationBytes(37718, 0) < 28 * GB);
    try std.testing.expect(dit + activationBytes(19292, bcast) >= 28 * GB); // 960x544 fast recipe: 28 GB
    try std.testing.expect(dit + activationBytes(19292, bcast) < 31 * GB); // and a 48 GB Mac with 31.5 GB free still serves it
    // 1536x672 with two keyframes under the fast recipe died with 32.58 GB available.
    try std.testing.expect(dit + activationBytes(requestRows(1536, 672, 124, 2), bcast) > 33 * GB);
}

test "h3 residency is priced on the request's activations, so a warm set yields to a big canvas (#764)" {
    const GB: u64 = 1 << 30;
    const te: u64 = 15_804_791_921; // the 4-bit FL2VA pack's files
    const dit_file: u64 = 18_698_813_290;
    const vaes: u64 = 5_207_808_496 + 605_254_808;
    const lora: u64 = 779_849_816;
    const bcast: u64 = 50 * 5376 * 2;
    const margin = residentMargin(64 * GB);
    // A 64 GB Mac with 60 GB free keeps the set warm for a small turbo run...
    const small = residentBytes(te, dit_file, vaes, lora, activationBytes(requestRows(768, 448, 124, 0), 0));
    try std.testing.expect(keepResident(60 * GB, 0, small, margin));
    // ...and releases it for a 1344x768 fast-recipe request; the flat term kept it and the run OOMed.
    const big = residentBytes(te, dit_file, vaes, lora, activationBytes(requestRows(1344, 768, 124, 0), bcast));
    try std.testing.expect(!keepResident(60 * GB, 0, big, margin));
    try std.testing.expect(keepResident(60 * GB, 0, residentBytes(te, dit_file, vaes, lora, activationBytes(0, 0)), margin));
}

test "h3 DiT term sheds the AdaLN weights precompute frees — unless it is off" {
    const GB: u64 = 1024 * 1024 * 1024;
    // Measured: the 8-bit pack's 32.83 GiB transformer.safetensors settles at
    // 20.19 GiB resident once precomputeAdaln frees the 13B modulation
    // weights, so the file size over-bills the DiT by ~12 GiB.
    const eight_bit: u64 = 32 * GB;
    try std.testing.expect(ditResidentBytes(eight_bit, true) < eight_bit);
    try std.testing.expect(ditResidentBytes(eight_bit, true) >= 20 * GB);
    // A resident DiT never frees them, so the whole file stays
    // resident and billing the shed size would be an uncatchable OOM.
    try std.testing.expectEqual(eight_bit, ditResidentBytes(eight_bit, false));
}

test "h3 residency: keeps the whole set only while it fits, counting what the cache already holds" {
    const gb: u64 = 1024 * 1024 * 1024;
    const need = residentBytes(28 * gb, 35 * gb, 6 * gb, gb, 6 * gb);
    try std.testing.expectEqual(76 * gb, need); // te + dit + vaes + lora + the 6 GiB activation term
    const margin = 12 * gb;
    try std.testing.expect(keepResident(150 * gb, 0, need, margin)); // a big Mac
    try std.testing.expect(keepResident(88 * gb, 0, need, margin)); // exactly enough
    try std.testing.expect(!keepResident(80 * gb, 0, need, margin)); // 128 GB class with other models loaded
    // The cache's own bytes are available to the next request, or a resident set would evict itself.
    try std.testing.expect(keepResident(20 * gb, 70 * gb, need, margin));
    // An unknown size never claims residency.
    try std.testing.expect(!keepResident(500 * gb, 0, residentBytes(0, 35 * gb, 6 * gb, 0, 6 * gb), margin));
    try std.testing.expect(!keepResident(500 * gb, 0, residentBytes(28 * gb, 0, 6 * gb, 0, 6 * gb), margin));
}

test "h3 residency: free memory is the tighter of host RAM and the GPU working-set room" {
    const gb: u64 = 1024 * 1024 * 1024;
    try std.testing.expectEqual(100 * gb, availBytes(100 * gb, 192 * gb, 40 * gb)); // host binds
    try std.testing.expectEqual(30 * gb, availBytes(100 * gb, 192 * gb, 162 * gb)); // GPU room binds
    try std.testing.expectEqual(100 * gb, availBytes(100 * gb, 0, 40 * gb)); // no GPU limit known
    try std.testing.expectEqual(@as(u64, 0), availBytes(0, 192 * gb, 0)); // host unknown never keeps
    try std.testing.expectEqual(@as(u64, 0), availBytes(100 * gb, 192 * gb, 200 * gb)); // over the limit
}

test "canvasBill: unified memory bills DiT + activations; CUDA bills the DiT to RAM, activations to the GPU" {
    const gb: u64 = 1 << 30;
    const mac = canvasBill(12 * gb, 6 * gb, 20 * gb, 16 * gb, 1 * gb, false);
    try std.testing.expectEqual(18 * gb, mac.need);
    try std.testing.expectEqual(15 * gb, mac.avail);
    // 16 GB card, 21 GB RAM free: 832x480 x 124 frames (12 GB DiT, 6.4 GB rows) fits.
    const cuda = canvasBill(12 * gb, 6 * gb, 21 * gb, 16 * gb, 1 * gb, true);
    try std.testing.expectEqual(6 * gb, cuda.need);
    try std.testing.expectEqual(15 * gb, cuda.avail);
    // A DiT bigger than free RAM is what binds there.
    const ram = canvasBill(25 * gb, 6 * gb, 21 * gb, 16 * gb, 1 * gb, true);
    try std.testing.expect(ram.need > ram.avail);
}

test "h3 residency: the margin scales with RAM and never drops below 10 GiB" {
    const gb: u64 = 1024 * 1024 * 1024;
    try std.testing.expectEqual(10 * gb, residentMargin(64 * gb));
    try std.testing.expectEqual(16 * gb, residentMargin(256 * gb));
}

test "declaresTask reads the pack's own task list" {
    const a = std.testing.allocator;
    // The converter writes the partition's task list; ref2va and fl2va share
    // every geometry number, so the DiT file is the ONLY thing that differs and
    // this is the only way to tell an FL2VA pack from a REF2VA one.
    try std.testing.expect(declaresTask(a, "{\"model_type\":\"minimax_h3\",\"tasks\":[\"t2va\",\"ref2va\"]}", "ref2va"));
    try std.testing.expect(!declaresTask(a, "{\"model_type\":\"minimax_h3\",\"tasks\":[\"t2va\",\"fl2va\"]}", "ref2va"));
    // FastH3 Omni generates from references only: no t2va.
    try std.testing.expect(!declaresTask(a, "{\"model_type\":\"minimax_h3\",\"tasks\":[\"ref2va\"]}", "t2va"));
    try std.testing.expect(declaresTask(a, "{\"model_type\":\"minimax_h3\",\"tasks\":[\"t2va\",\"fl2va\"]}", "t2va"));
    // Absent / malformed / wrong-typed → NOT ref2va. A pack that cannot say it
    // supports references must not be handed them: it would generate happily
    // and ignore every one of them.
    try std.testing.expect(!declaresTask(a, "{\"model_type\":\"minimax_h3\"}", "ref2va"));
    try std.testing.expect(!declaresTask(a, "{\"tasks\":\"ref2va\"}", "ref2va"));
    try std.testing.expect(!declaresTask(a, "not json", "ref2va"));
    // A substring match on the whole file would pass on the README-ish text a
    // config can legally carry; only the task LIST counts.
    try std.testing.expect(!declaresTask(a, "{\"note\":\"ref2va\",\"tasks\":[\"t2va\",\"fl2va\"]}", "ref2va"));
}

test "h3 two-time packs: the schedule and gate come from config.json; a grid that cannot be sampled, a default past the grids or a PDD contract beside it is refused" {
    const a = std.testing.allocator;
    const good = "{\"two_time\":{\"gate\":0.25,\"default_steps\":2,\"sigma_grids\":{\"2\":[1.0,0.4,0.0],\"4\":[1,0.7,0.3,0.1,0]}}}";
    const g = try geometry(a, good);
    try std.testing.expectEqual(@as(f32, 0.25), g.dit.two_time_gate);
    try std.testing.expectEqualSlices(f64, &.{ 1.0, 0.4, 0.0 }, g.two_time.?.grid(2).?);
    try std.testing.expect(g.two_time.?.grid(3) == null);
    try std.testing.expect((try geometry(a, "{}")).two_time == null);
    for ([_][]const u8{
        "{\"two_time\":{\"gate\":0.25,\"default_steps\":6,\"sigma_grids\":{\"2\":[1.0,0.4,0.0]}}}",
        "{\"two_time\":{\"gate\":0.25,\"default_steps\":2,\"sigma_grids\":{\"2\":[1.0,0.4,0.1]}}}",
        "{\"two_time\":{\"gate\":0.25,\"default_steps\":2,\"sigma_grids\":{\"2\":[1.0,0.0,0.4]}}}",
        "{\"two_time\":{\"default_steps\":2,\"sigma_grids\":{\"2\":[1.0,0.4,0.0]}}}",
        "{\"two_time\":{\"gate\":0.25,\"default_steps\":4294967298,\"sigma_grids\":{\"2\":[1.0,0.4,0.0]}}}",
        "{\"pdd_steps\":32,\"pdd_step_indices\":[0,32],\"grid_max_t\":0.999,\"two_time\":{\"gate\":0.25,\"default_steps\":2,\"sigma_grids\":{\"2\":[1.0,0.4,0.0]}}}",
    }) |bad| try std.testing.expectError(error.BadPackConfig, geometry(a, bad));
}

test "h3 init_video bill: a request with a clip reserves its encode stage and the lift its headroom holds" {
    const GB: u64 = 1 << 30;
    const mem: @import("../../sdk/testing/fixture.zig").MemDir = .{ .files = &.{.{ .name = "video_vae.safetensors", .bytes = "12345678" }} };
    const d = mem.packDir();
    const body = "{\"prompt\":\"a \\\"b\\\"\",\"width\":1344,\"height\":768,\"num_frames\":124,\"init_video\":{\"frames\":[\"iVBO\\/R\",\"x\"],\"audio\":\"UklG\"},\"init_step\":3}";
    const held = encodeHeldBytes(1344, 768, 124);
    // Room for the whole lift, for part of it (64 GB, 56 GB free, less the 10 GB margin), and for none of it.
    try std.testing.expectEqual(8 + held + 2 * h3.encodePassBytes(), billRequest(&d, &.{ .ram_total = 256 * GB, .ram_available = 240 * GB }, .of(body)));
    try std.testing.expectEqual(8 + held + (46 * GB - 8 - held), billRequest(&d, &.{ .ram_total = 64 * GB, .ram_available = 56 * GB }, .of(body)));
    try std.testing.expectEqual(8 + held, billRequest(&d, &.{ .ram_total = 36 * GB, .ram_available = 30 * GB }, .of(body)));
    // The GPU working set bounds the room like live RAM does.
    try std.testing.expectEqual(8 + held + (46 * GB - 8 - held), billRequest(&d, &.{ .ram_total = 64 * GB, .ram_available = 60 * GB, .gpu_working_set = 56 * GB }, .of(body)));
    // A small-RAM Mac, or RAM the host could not read, bills no pool lift: it never gets one.
    try std.testing.expectEqual(8 + held, billRequest(&d, &.{ .ram_total = 16 * GB, .ram_available = 15 * GB }, .of(body)));
    try std.testing.expectEqual(8 + held, billRequest(&d, &.{ .ram_total = 256 * GB }, .of(body)));
    const default_held = encodeHeldBytes(h3.DEFAULT_SIDE, h3.DEFAULT_SIDE, h3.DEFAULT_FRAMES);
    try std.testing.expectEqual(8 + default_held + 2 * h3.encodePassBytes(), billRequest(&d, &.{ .ram_total = 256 * GB, .ram_available = 240 * GB }, .of("{\"init_video\":{\"frames\":[]}}")));
    try std.testing.expectEqual(@as(u64, 0), billRequest(&d, &.{ .ram_total = 256 * GB, .ram_available = 240 * GB }, .of("{\"width\":1344,\"init_video\":null}")));
    try std.testing.expectEqual(@as(u64, 0), billRequest(&d, &.{ .ram_total = 256 * GB, .ram_available = 240 * GB }, .of("{\"width\":1344,\"height\":768}")));
}

test "h3 init_video encode stage outweighs a small canvas's DiT stage and stays above the measured peak" {
    const GB: u64 = 1 << 30;
    // The HyperFlow Flash REF2VA 8-bit pack: a two-time DiT keeps its whole file and runs no attention cache.
    const vae: u64 = 5_207_808_496;
    const dit = ditResidentBytes(14_971_410_087, false);
    try std.testing.expect(vae + encodeHeldBytes(256, 256, 22) > dit + activationBytes(requestRows(256, 256, 22, 0), 0));
    // A real 1344x768 x 124 encode on a 256 GB Mac: 1.0 GB of f32 encoder, then 36.3 GB more process footprint.
    const held = encodeHeldBytes(1344, 768, 124);
    try std.testing.expect(vae + held + billedEncodeLift(vae, held, &.{ .ram_total = 256 * GB, .ram_available = 240 * GB }) > 37 * GB + 300 * (GB >> 10));
}

/// The HyperFlow Flash REF2VA 8-bit pack's files.
const flash8: PackSizes = .{ .te = 16_460_999_117, .dit_file = 14_971_410_087, .video_vae = 5_207_808_496, .audio_vae = 605_254_808, .turbo_lora = 0 };

fn flashPlan(total: u64, free: u64, cache_now: u64, w: u32, h: u32, frames: u32, init: bool) struct { p: Plan, in: PlanIn, billed: u64, held: u64 } {
    const held = if (init) encodeHeldBytes(w, h, frames) else h3.encodePassBytes();
    const billed = billedEncodeLift(flash8.video_vae, held, &.{ .ram_total = total, .ram_available = free });
    const in: PlanIn = .{
        .sizes = flash8,
        .precompute = false,
        .turbo = false,
        .may_keep = true,
        .cache_now = cache_now,
        .host_avail = free,
        .total_ram = total,
        .gpu_limit = 0,
        .gpu_active = 0,
        .weights_page = false,
        .activations = activationBytes(requestRows(w, h, frames, 0), 0),
        .encode_held = held,
        .init = init,
        .billed_lift = billed,
    };
    return .{ .held = held, .billed = billed, .in = in, .p = plan(in) };
}

test "h3 plan: the encode lifts into what the plan leaves of free memory less the margin, never past its bill" {
    const GB: u64 = 1 << 30;
    for ([_]u64{ 36, 48, 64, 96, 128, 256 }) |t| for ([_]u64{ 4, 8, 16, 32 }) |used| for ([_]u64{ 0, 1 }) |warm| {
        if (used >= t) continue;
        const total = t * GB;
        const free = total - used * GB;
        const r = flashPlan(total, free, 0, 1344, 768, 124, true);
        const warm_cache = if (warm == 1) (r.p.resident orelse continue) else 0;
        const q = flashPlan(total, free - warm_cache, warm_cache, 1344, 768, 124, true);
        try std.testing.expect(q.p.lift <= q.billed);
        try std.testing.expect(q.p.lift <= 2 * h3.encodePassBytes());
        const margin = residentMargin(total);
        if (q.p.refused) |f| {
            try std.testing.expect(f.what != .encode or flash8.video_vae + q.held > free - warm_cache);
        } else if (q.p.resident) |bytes| {
            try std.testing.expect(bytes <= free -| margin);
        } else if (q.p.lift > 0) {
            try std.testing.expect(flash8.video_vae + q.held + q.p.lift + margin <= free);
        }
    };
}

test "h3 plan: the resident set's bytes leave out the init clip and the encode lift" {
    const GB: u64 = 1 << 30;
    // 256 GB Mac, nothing else running: resident either way, and the init request really gets a lift.
    const plain = flashPlan(256 * GB, 240 * GB, 0, 1344, 768, 124, false);
    const clip = flashPlan(256 * GB, 240 * GB, 0, 1344, 768, 124, true);
    try std.testing.expect(plain.p.resident != null and clip.p.resident != null);
    try std.testing.expect(clip.p.lift > 0);
    // The cache keeps weights + activations after either request; the clip and the lift are gone by then.
    try std.testing.expectEqual(plain.p.resident.?, clip.p.resident.?);
    try std.testing.expectEqual(residentBytes(flash8.te, flash8.dit_file, flash8.video_vae + flash8.audio_vae, 0, clip.in.activations), clip.p.resident.?);
}

test "h3 plan: other apps' memory shrinks the lift, and an encode that cannot fit even unlifted is refused" {
    const GB: u64 = 1 << 30;
    // 64 GB Mac, 768x768 x 124, staged (the resident set does not fit): room past the stage and the margin.
    const idle = flashPlan(64 * GB, 60 * GB, 0, 768, 768, 124, true);
    const busy = flashPlan(64 * GB, 44 * GB, 0, 768, 768, 124, true);
    try std.testing.expect(idle.p.resident == null and busy.p.resident == null);
    try std.testing.expectEqual(60 * GB - 10 * GB - flash8.video_vae - idle.held, idle.p.lift);
    try std.testing.expectEqual(44 * GB - 10 * GB - flash8.video_vae - busy.held, busy.p.lift);
    // With the margin gone, no lift; the encode still runs while its stage fits.
    const tight = flashPlan(64 * GB, flash8.video_vae + idle.held + 5 * GB, 0, 768, 768, 124, true);
    try std.testing.expect(tight.p.refused == null);
    try std.testing.expectEqual(@as(u64, 0), tight.p.lift);
    const none = flashPlan(64 * GB, flash8.video_vae + idle.held - 1, 0, 768, 768, 124, true);
    try std.testing.expect(none.p.refused.?.what == .encode);
    try std.testing.expectEqual(flash8.video_vae + idle.held, none.p.refused.?.need);
    // Memory freed since load never lifts past what the request was billed for.
    const late = flashPlan(64 * GB, 60 * GB, 0, 768, 768, 124, true);
    try std.testing.expect(late.p.lift > busy.billed);
    var in = late.in;
    in.billed_lift = busy.billed;
    try std.testing.expectEqual(busy.billed, plan(in).lift);
    // Without init_video the same memory refuses nothing on the encode's account.
    try std.testing.expect(flashPlan(64 * GB, flash8.video_vae + idle.held - 1, 0, 768, 768, 124, false).p.refused == null);
}
