//! The MLX engine's side of `host/engines/mlx.zig`: the table of calls the host makes, exported as
//! `mlx_serve_mlx_api`. Every call that can fail returns its error's name (null = success).

const std = @import("std");
const local = @import("../../sdk/util/io_util.zig").local;
const mlx = @import("../../mlx/ffi.zig");
const host = @import("../../host/engines/mlx.zig");
const params = @import("../../host/gpu_thread/params.zig");
const requests = @import("../../host/gpu_thread/requests.zig");
const ModelConfig = @import("../../sdk/config/model.zig").ModelConfig;
const LoadedModel = @import("../../host/registry/registry.zig").LoadedModel;
const scheduler = @import("scheduler.zig");
const verdict = @import("batching/verdict.zig");
const model_state = @import("runtime/model_state.zig");
const distributed = @import("cluster/distributed.zig");
const request_policy = @import("request_policy.zig");
const configure_mod = @import("configure.zig");
const mlx_stream = @import("modules/stream.zig");
const Scheduler = scheduler.Scheduler;
const Slot = scheduler.Slot;

pub const table: host.Api = .{
    .admission = &@import("engine_admission.zig").table,
    .start = start,
    .stop = scheduler.deinit,
    .cancel_all = scheduler.cancelAllInFlight,
    .has_company = scheduler.hasCompany,
    .submit = submit,
    .complete = complete,
    .cancel = cancel,
    .ensure_loaded = ensureLoaded,
    .release = scheduler.release,
    .release_status = scheduler.releaseStatus,
    .load_error_name = scheduler.loadErrorName,
    .unload = unload,
    .free_model_state = model_state.freeMlx,
    .unload_resident = model_state.unloadResident,
    .holds_pair = distributed.holds,
    .set_foreign_bytes = mlx_stream.setForeignBytes,
    .hot_cache_bytes = hotCacheBytes,
    .run_generation = runGeneration,
    .compute_embeddings = computeEmbeddings,
    .encode_vision = encodeVision,
    .timestamp_classes = timestampClasses,
    .batches_decode = verdict.configBatchesDecode,
    .facts = facts,
    .draft_stages = request_policy.dsv4DraftStages,
    .supports_video = supportsVideo,
    .pair_fields = pairFields,
    .settings = configure_mod.settings,
    .model_props = configure_mod.modelProps,
    .model_props_json = configure_mod.modelPropsJson,
    .cluster_json = configure_mod.clusterJson,
    .plugin_stub = pluginStub,
    .gauges = configure_mod.gauges,
    .run_offline = @import("offline.zig").run,
    .cluster_worker = clusterWorker,
    .cluster_probe = clusterProbe,
    .cluster_connect = clusterConnect,
    .configure = configure_mod.apply,
    .install_admission_hooks = @import("admission/admit.zig").installSchedulerHooks,
    .set_gen_yield = setGenYield,
};

fn addr() callconv(.c) *const anyopaque {
    return &table;
}

comptime {
    @export(&addr, .{ .name = "mlx_serve_mlx_api" });
}

fn pluginStub(io: std.Io, a: std.mem.Allocator, model_dir: []const u8, raw_json: []const u8) ?@import("plugin/config.zig").Stub {
    return @import("plugin/config.zig").stubOf(local(io), a, model_dir, raw_json);
}

fn clusterWorker(io: std.Io, gpa: std.mem.Allocator, args: []const []const u8) ?[]const u8 {
    @import("modules/call.zig").call("cluster_worker", .{ local(io), gpa, args }) catch |e| return @errorName(e);
    return null;
}

fn clusterProbe(io: std.Io, args: []const []const u8) ?[]const u8 {
    distributed.cmdProbe(local(io), args) catch |e| return @errorName(e);
    return null;
}

fn clusterConnect(io: std.Io, gpa: std.mem.Allocator, coordinator: []const u8, device: []const u8, mode: u8, model: []const u8, split: ?u32) ?[]const u8 {
    distributed.connect(local(io), gpa, .{ .rank = 0, .coordinator = coordinator, .device = device }, @enumFromInt(mode), model, split) catch |e| return @errorName(e);
    return null;
}

fn setGenYield(on: bool) void {
    scheduler.gen_yield_enabled = on;
}

fn start(gpa: std.mem.Allocator, io: std.Io, p: *const params.LoadParams, max_concurrent: u32, out: *?*Scheduler) ?[]const u8 {
    out.* = scheduler.init(gpa, local(io), p.*, max_concurrent) catch |e| return @errorName(e);
    return null;
}

fn submit(sch: *Scheduler, p: *const params.SubmitParams, out: *?*Slot) ?[]const u8 {
    out.* = scheduler.submit(sch, p.*) catch |e| return @errorName(e);
    return null;
}

fn complete(sch: *Scheduler, slot: *Slot) void {
    scheduler.complete(sch, slot);
}

fn cancel(slot: *Slot) void {
    slot.cancel();
}

fn ensureLoaded(sch: *Scheduler, id: []const u8, out: *?*LoadedModel) ?[]const u8 {
    out.* = scheduler.ensureLoaded(sch, id) catch |e| return @errorName(e);
    return null;
}

fn unload(sch: *Scheduler, id: []const u8, window_ms: ?i64) ?[]const u8 {
    scheduler.unloadModelIfIdle(sch, id, window_ms) catch |e| return @errorName(e);
    return null;
}

fn hotCacheBytes(lm: *const LoadedModel) u64 {
    return if (lm.prefix_cache) |hc| hc.residentBytes() else 0;
}

fn runGeneration(sch: *Scheduler, req: *requests.GenRequest) ?[]const u8 {
    scheduler.runGeneration(sch, req) catch |e| return @errorName(e);
    return null;
}

fn computeEmbeddings(sch: *Scheduler, req: *requests.EmbedRequest, out: *[][]f32) ?[]const u8 {
    out.* = scheduler.computeEmbeddings(sch, req) catch |e| return @errorName(e);
    return null;
}

fn encodeVision(sch: *Scheduler, req: *requests.VisionEncodeRequest, out: *mlx.mlx_array) ?[]const u8 {
    out.* = scheduler.encodeVision(sch, req) catch |e| return @errorName(e);
    return null;
}

fn timestampClasses(gpa: std.mem.Allocator, lm: *LoadedModel, ids: []const u32, rows: mlx.mlx_array, timestamp_token: u32, out: *[]u32) ?[]const u8 {
    const xfm = lm.transformer orelse return "NotLoaded";
    out.* = @import("../../plugins/qwen3_asr/forced_align.zig").timestampClasses(gpa, xfm, ids, rows, timestamp_token) catch |e| return @errorName(e);
    return null;
}

fn pairFields(gpa: std.mem.Allocator) ?[]u8 {
    return distributed.pairFields(gpa) catch null;
}

fn supportsVideo(lm: *const LoadedModel) bool {
    const ve = lm.vision_encoder orelse return false;
    if (ve.plugin_media) |p| return p.parsed.caps.video;
    if (ve.radio) |radio| return radio.video_w.ctx != null;
    const c = lm.config orelse return false;
    return ve.supportsImages() and c.video_token_id != 0;
}

fn facts(lm: *const LoadedModel, out: *host.Facts) void {
    out.* = .{
        .has_mtp = lm.mtp != null,
        .has_transformer = lm.transformer != null,
        .module_owned = if (lm.transformer) |t| t.ownsModuleDecodeState() else false,
        .images = if (lm.vision_encoder) |ve| ve.supportsImages() else false,
        .audio = if (lm.vision_encoder) |ve| ve.supportsAudio() else false,
        // Transcription: Qwen3-ASR's tower, or a speech model's own (its card's ASR recipe, `handleTranscriptions`).
        .hears = if (lm.vision_encoder) |ve| ve.qwen_audio != null or (ve.supportsAudio() and lm.config.?.speech() != null) else false,
        .video = supportsVideo(lm),
        .batches_decode = if (lm.config) |c| verdict.configBatchesDecode(c) else false,
        .hot_cache_bytes = hotCacheBytes(lm),
    };
}

// ── tests ──

const testing = std.testing;

test "RADIO video availability follows loaded embedder weights" {
    var radio: @import("../../plugins/nemotron_h/radio_vision.zig").RadioVision = undefined;
    radio.video_w = .{ .ctx = null };
    var encoder: @import("vision_encoder.zig").VisionEncoder = undefined;
    encoder.radio = radio;
    encoder.qwen_audio = null;
    encoder.plugin_media = null;
    var entry: LoadedModel = undefined;
    entry.vision_encoder = &encoder;
    var config = ModelConfig{ .radio_vision = true };
    entry.config = &config;
    try testing.expect(encoder.supportsImages());
    try testing.expect(!supportsVideo(&entry));
    encoder.radio.?.video_w.ctx = @ptrCast(&radio);
    try testing.expect(supportsVideo(&entry));
    encoder.radio = null;
    config.radio_vision = false;
    config.video_token_id = 7;
    try testing.expect(supportsVideo(&entry));
    config.video_token_id = 0;
    try testing.expect(!supportsVideo(&entry));
    entry.vision_encoder = null;
    try testing.expect(!supportsVideo(&entry));
}
