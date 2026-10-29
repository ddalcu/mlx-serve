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

/// No reserve beyond the model's own estimate (legacy set none): the step itself refuses a canvas that does not fit,
/// by name, against live memory (`canvasBill`).
pub fn billRequest(dir: *const abi.PackDir, m: *const abi.Machine, body: abi.Str) callconv(.c) u64 {
    _ = dir;
    _ = m;
    _ = body;
    return 0;
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
