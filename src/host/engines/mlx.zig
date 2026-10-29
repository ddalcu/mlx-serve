//! The host's way into the MLX engine: one table of calls (`Api`) the engine exports as `mlx_serve_mlx_api`
//! (`engines/mlx/api.zig`). Both sides compile from this tree, so Zig types cross; errors do not (each compile unit
//! numbers its own), so a call returns its error's name and these wrappers turn it back into an error here.

const std = @import("std");
const mlx = @import("../../mlx/ffi.zig");
const handles = @import("../../abi/mlx_handles.zig");
const err_name = @import("../../sdk/util/err_name.zig");
const mtp_acceptance = @import("../../sdk/spec/mtp_acceptance.zig");
const abi_plugin = @import("../../abi/plugin.zig");
const abi_engine = @import("../../abi/engine.zig");
const ModelConfig = @import("../../sdk/config/model.zig").ModelConfig;
const Scheduler = @import("../gpu_thread/scheduler.zig").Scheduler;
const BatchVerdict = @import("../gpu_thread/scheduler.zig").BatchVerdict;
const params = @import("../gpu_thread/params.zig");
const requests = @import("../gpu_thread/requests.zig");
const LoadedModel = @import("../registry/registry.zig").LoadedModel;
const Stream = @import("stream.zig").Stream;

pub const Slot = handles.Slot;
pub const Err = ?[]const u8;

pub const Api = struct {
    /// The context a model advertises and the admission guard (`abi.Admission`).
    admission: *const abi_engine.Admission,
    // ── the GPU thread
    start: *const fn (std.mem.Allocator, std.Io, *const params.LoadParams, u32, *?*Scheduler) Err,
    stop: *const fn (*Scheduler) void,
    cancel_all: *const fn (*Scheduler) void,
    has_company: *const fn (*Scheduler) bool,
    // ── requests
    submit: *const fn (*Scheduler, *const params.SubmitParams, *?*Slot) Err,
    complete: *const fn (*Scheduler, *Slot) void,
    cancel: *const fn (*Slot) void,
    // ── models
    ensure_loaded: *const fn (*Scheduler, []const u8, *?*LoadedModel) Err,
    release: *const fn (*Scheduler, *LoadedModel) void,
    release_status: *const fn (*Scheduler, *LoadedModel) void,
    load_error_name: *const fn (*Scheduler, std.mem.Allocator, []const u8) ?[]u8,
    unload: *const fn (*Scheduler, []const u8, ?i64) Err,
    free_model_state: *const fn (*LoadedModel) void,
    unload_resident: *const fn (*LoadedModel) void,
    holds_pair: *const fn ([]const u8) bool,
    set_foreign_bytes: *const fn (u64) void,
    hot_cache_bytes: *const fn (*const LoadedModel) u64,
    // ── GPU-thread jobs
    run_generation: *const fn (*Scheduler, *requests.GenRequest) Err,
    compute_embeddings: *const fn (*Scheduler, *requests.EmbedRequest, *[][]f32) Err,
    encode_vision: *const fn (*Scheduler, *requests.VisionEncodeRequest, *mlx.mlx_array) Err,
    /// Qwen3-ForcedAligner: the 80 ms step each `<timestamp>` in `ids` lands on, run on the inference thread.
    timestamp_classes: *const fn (std.mem.Allocator, *LoadedModel, []const u32, mlx.mlx_array, u32, *[]u32) Err,
    // ── what the host reports
    batches_decode: *const fn (*const ModelConfig) bool,
    facts: *const fn (*const LoadedModel, *Facts) void,
    /// DeepSeek-V4/V4.1 with DSpark draft stages loaded: MTP-capable without a head.
    draft_stages: *const fn (*const LoadedModel) bool,
    supports_video: *const fn (*const LoadedModel) bool,
    pair_fields: *const fn (std.mem.Allocator) ?[]u8,
    settings: *const fn (*Settings) void,
    model_props: *const fn (*const LoadedModel, *ModelProps) void,
    /// The model's `/props` fragments the engine owns (Neural Engine prefill, n-gram warm), then the cluster's.
    model_props_json: *const fn (*const LoadedModel, std.mem.Allocator) ?[]u8,
    cluster_json: *const fn (std.mem.Allocator) ?[]u8,
    /// The serving plugin's stub parse of the pack at a dir (discovery); null when no plugin serves it.
    plugin_stub: *const fn (std.Io, std.mem.Allocator, []const u8, []const u8) ?PluginStub,
    gauges: *const fn (*Gauges) void,
    /// The Thunderbolt cluster: `mlx-serve dist-worker`, `dist-probe`, and the host side's `--dist` connect
    /// (`mode` is the cluster's `Mode` tag).
    cluster_worker: *const fn (std.Io, std.mem.Allocator, []const []const u8) Err,
    cluster_probe: *const fn (std.Io, []const []const u8) Err,
    cluster_connect: *const fn (std.Io, std.mem.Allocator, []const u8, []const u8, u8, []const u8, ?u32) Err,
    /// `--prompt` without `--serve`: one generation on the calling thread.
    run_offline: *const fn (*const Offline) Err,
    // ── process-wide engine settings, set once from the launch flags
    configure: *const fn (*const Flags) void,
    install_admission_hooks: *const fn (bool) void,
    set_gen_yield: *const fn (bool) void,
};

/// The offline single-prompt run's inputs.
pub const Offline = struct {
    io: std.Io,
    allocator: std.mem.Allocator,
    config: *ModelConfig,
    tok: *const @import("../../sdk/tokenizer/tokenizer.zig").Tokenizer,
    chat_config: *const @import("../chat/chat.zig").ChatConfig,
    model_dir: []const u8,
    load_vision: bool,
    kv_quant: @import("../../sdk/blocks/attention/kv_quant.zig").KVQuantConfig,
    enable_mtp: bool,
    prompt: []const u8,
    temperature: f32,
    max_tokens: u32,
    stream: bool,
};

/// What discovery reads of a served pack: the stub parse's shell and caps.
pub const PluginStub = struct { shell: abi_plugin.Shell, caps: abi_plugin.Caps, speaks: bool = false };

/// What the host reads about a loaded MLX model.
pub const Facts = struct {
    has_mtp: bool = false,
    has_transformer: bool = false,
    /// The model keeps per-request decode state outside the KV cache (no prompt logprobs, no prefix reuse).
    module_owned: bool = false,
    /// The vision tower reads images, audio (any tower), speech (Qwen3-ASR's), video frames.
    images: bool = false,
    audio: bool = false,
    hears: bool = false,
    video: bool = false,
    /// The model batches decode at all (`configBatchesDecode`).
    batches_decode: bool = false,
    hot_cache_bytes: u64 = 0,
};

/// Engine-wide settings `/props` reports.
pub const Settings = struct {
    prefill_chunk: usize = 0,
    max_mtp_ctx: u32 = 0,
    prefill_decode_share: f32 = 0,
    mtp_head_kv_quant: bool = false,
    /// `--prefill-chunk` when given, else 0.
    explicit_prefill_chunk: u32 = 0,
};

/// Live engine numbers the metrics sampler publishes.
pub const Gauges = struct {
    ane_int8_bytes: u64 = 0,
    ane_layers: u64 = 0,
    ngram_warm_bytes: u64 = 0,
};

/// A loaded model's serving settings as `/props` reports them.
pub const ModelProps = struct {
    decode_attn_quant: bool = false,
    mtp_loaded: bool = false,
    mtp_acceptance: mtp_acceptance.Mode = .exact,
    mtp_greedy_tail: bool = false,
    /// The pack's own speculation lane (a stream pack), empty when none.
    native_lane: []const u8 = "",
    native_block: u32 = 0,
};

/// The launch flags that set the engine's own process-wide settings, applied once (`configure`) before a model loads.
pub const Flags = struct {
    no_vision: bool = false,
    prevent_sleep: bool = true,
    skip_mem_preflight: bool = false,
    ple_gpu: bool = false,
    /// `--mtp-min-depth` / `--mtp-max-depth`; a max of 0 leaves the planner's own cap.
    mtp_depth_min: u32 = 1,
    mtp_depth_max: u32 = 0,
    mtp_greedy_tail: bool = false,
    max_mtp_ctx: u32 = 0,
    /// 0 = the full history.
    mtp_history_window: usize = 0,
    /// Set by `--prefill-chunk`: the explicit width, which turns machine sizing off.
    prefill_chunk: ?usize = null,
    prefill_trace: bool = false,
    prefill_decode_share: f32 = 0,
    mtp_head_kv_quant: bool = false,
    mtp_acceptance: mtp_acceptance.Mode = .exact,
};

extern fn mlx_serve_mlx_api() callconv(.c) *const anyopaque;

pub fn api() *const Api {
    return @ptrCast(@alignCast(mlx_serve_mlx_api()));
}

pub fn start(gpa: std.mem.Allocator, io: std.Io, p: params.LoadParams, max_concurrent: u32) !*Scheduler {
    var out: ?*Scheduler = null;
    try err_name.check(api().start(gpa, io, &p, max_concurrent, &out));
    return out.?;
}

pub fn submit(sch: *Scheduler, p: params.SubmitParams) !*Slot {
    var out: ?*Slot = null;
    try err_name.check(api().submit(sch, &p, &out));
    return out.?;
}

pub fn ensureLoaded(sch: *Scheduler, id: []const u8) !*LoadedModel {
    var out: ?*LoadedModel = null;
    try err_name.check(api().ensure_loaded(sch, id, &out));
    return out.?;
}

pub fn unloadModel(sch: *Scheduler, id: []const u8) !void {
    try err_name.check(api().unload(sch, id, null));
}

pub fn unloadModelIfIdle(sch: *Scheduler, id: []const u8, window_ms: ?i64) !void {
    try err_name.check(api().unload(sch, id, window_ms));
}

pub fn runGeneration(sch: *Scheduler, req: *requests.GenRequest) !void {
    try err_name.check(api().run_generation(sch, req));
}

pub fn computeEmbeddings(sch: *Scheduler, req: *requests.EmbedRequest) ![][]f32 {
    var out: [][]f32 = &.{};
    try err_name.check(api().compute_embeddings(sch, req, &out));
    return out;
}

pub fn encodeVision(sch: *Scheduler, req: *requests.VisionEncodeRequest) !mlx.mlx_array {
    var out: mlx.mlx_array = .{ .ctx = null };
    try err_name.check(api().encode_vision(sch, req, &out));
    return out;
}

pub fn timestampClasses(gpa: std.mem.Allocator, lm: *LoadedModel, ids: []const u32, rows: mlx.mlx_array, timestamp_token: u32) ![]u32 {
    var out: []u32 = &.{};
    try err_name.check(api().timestamp_classes(gpa, lm, ids, rows, timestamp_token, &out));
    return out;
}

pub fn facts(lm: *const LoadedModel) Facts {
    var f: Facts = .{};
    api().facts(lm, &f);
    return f;
}

pub fn modelProps(lm: *const LoadedModel) ModelProps {
    var p: ModelProps = .{};
    api().model_props(lm, &p);
    return p;
}

pub fn gauges() Gauges {
    var g: Gauges = .{};
    api().gauges(&g);
    return g;
}

pub fn settings() Settings {
    var s: Settings = .{};
    api().settings(&s);
    return s;
}

/// A request the MLX engine runs: the host's stream and the engine's handle for it.
pub const Running = struct {
    sch: *Scheduler,
    slot: *Slot,
    stream: *Stream,

    /// Submits `p` with a fresh stream as its sink (the scheduler's allocator, as the slot's own buffers were). The
    /// engine holds the stream until its `finished`.
    pub fn submit(sch: *Scheduler, p: params.SubmitParams) !Running {
        const stream = try Stream.create(sch.allocator, sch.io);
        errdefer stream.release();
        var q = p;
        q.sink = stream.sink();
        stream.prompt_len = (p.full_prompt orelse p.prompt_ids).len;
        stream.engineHolds();
        var out: ?*Slot = null;
        const failed = api().submit(sch, &q, &out);
        if (failed) |name| {
            // A slot built and freed inside the failed submit already let go through `finished`.
            if (out == null and stream.refs.load(.acquire) == 2) stream.release();
            return err_name.toError(name);
        }
        stream.canceller = .{ .ctx = out.?, .cancel = cancelSlot };
        return .{ .sch = sch, .slot = out.?, .stream = stream };
    }

    /// Hands the slot back to the engine and lets go of the stream.
    pub fn done(self: Running) void {
        api().complete(self.sch, self.slot);
        self.stream.release();
    }
};

fn cancelSlot(ctx: *anyopaque) void {
    api().cancel(@ptrCast(@alignCast(ctx)));
}
