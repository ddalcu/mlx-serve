//! A pack's safetensors read into a `Weights` map: every shard the index names (or one file), with the tensors a load
//! leaves out dropped and f16 side tensors and 1-D tables narrowed to the bf16 activations they meet.

const std = @import("std");
const builtin = @import("builtin");
const mlx = @import("../../mlx/ffi.zig");
const log = @import("../util/log.zig");
const resident = @import("resident.zig");
const modelopt_pack = @import("../quant/nvfp4/modelopt_pack.zig");
const affineParamsFromGeometry = @import("../quant/params.zig").affineParamsFromGeometry;
pub const Weights = resident.Weights;
pub const loadWeightsFile = resident.loadWeightsFile;

/// Load all safetensors files from model_dir.
/// When `load_vision` is true, vision_tower and multi_modal_projector weights are included.
pub fn loadWeights(io: std.Io, allocator: std.mem.Allocator, model_dir: []const u8) !Weights {
    return loadWeightsOpt(io, allocator, model_dir, .{});
}

/// How a load treats stored dtypes. `keep_f16`: a pack whose activation dtype
/// is f16 (Prism Hadamard packs) keeps its f16 side tensors and tables as
/// stored; narrowing them to bf16 drops 3 mantissa bits of every group scale.
/// `dsv41` also reads past the page cache (`nocache_reader`): its weights all load at init, and a pack near the
/// RAM size would have them compressed under the file cache.
/// `index_owners`: a tensor two shards carry loads from the shard the index names.
/// `skip`: a tensor it names never enters the map (qwen4's embedded n-gram table, read from disk per lookup).
pub const LoadOpts = struct { vision: bool = false, modelopt: bool = false, audio_tower: bool = false, keep_f16: bool = false, skip: ?*const fn (key: []const u8) bool = null, defer_qwen4_norms: bool = false, dsv41: bool = false, index_owners: bool = false, shard: []const u8 = "", owners: ?*const Owners = null };

pub const Owners = std.StringHashMapUnmanaged([]const u8);

/// tensor -> shard from the index's `weight_map`, strings owned by `arena`.
pub fn indexOwners(io: std.Io, arena: std.mem.Allocator, dir: std.Io.Dir) ?Owners {
    const raw = dir.readFileAlloc(io, "model.safetensors.index.json", arena, .limited(64 * 1024 * 1024)) catch return null;
    const parsed = std.json.parseFromSliceLeaky(std.json.Value, arena, raw, .{}) catch return null;
    if (parsed != .object) return null;
    const wm = parsed.object.get("weight_map") orelse return null;
    if (wm != .object) return null;
    var out: Owners = .empty;
    var it = wm.object.iterator();
    while (it.next()) |e| {
        if (e.value_ptr.* != .string) continue;
        // An owner shard that is not on disk claims nothing.
        _ = dir.statFile(io, e.value_ptr.string, .{}) catch continue;
        out.put(arena, e.key_ptr.*, e.value_ptr.string) catch return null;
    }
    return out;
}

/// deepseek_v41 reads its Engram tables from disk per lookup (~100 GB a layer), so they never enter
/// the weight map; the vision tower only when the load asks for it.
pub fn dsv41DropsKey(key: []const u8, vision: bool) bool {
    return (!vision and dsv41VisionKey(key)) or std.mem.indexOf(u8, key, ".engram.embed.") != null;
}

pub fn dsv41VisionKey(key: []const u8) bool {
    return std.mem.startsWith(u8, key, "vision.") or std.mem.startsWith(u8, key, "aligner.") or std.mem.startsWith(u8, key, "image_");
}

/// Load ONE safetensors file (absolute path) into a Weights map — for
/// sidecar files that live beside the trunk shards (e.g. a root-level
/// `mtp.safetensors`), where a directory scan would sweep in the trunk.
pub fn loadWeightsSingleFile(allocator: std.mem.Allocator, abs_path: []const u8) !Weights {
    var weights = Weights.init(allocator);
    errdefer weights.deinit();

    const s = mlx.weightLoadStream();
    defer _ = mlx.mlx_stream_free(s);

    const pathz = try allocator.dupeSentinel(u8, abs_path, 0);
    defer allocator.free(pathz);
    try loadSafetensorsFile(allocator, &weights, pathz, s, .{});

    if (weights.count() == 0) {
        log.err("no usable weights loaded from {s} — corrupt or empty safetensors file?\n", .{abs_path});
        return error.NoWeightFiles;
    }
    return weights;
}

pub fn loadWeightsWithVision(io: std.Io, allocator: std.mem.Allocator, model_dir: []const u8) !Weights {
    return loadWeightsOpt(io, allocator, model_dir, .{ .vision = true });
}

pub fn loadWeightsOpt(io: std.Io, allocator: std.mem.Allocator, model_dir: []const u8, opts: LoadOpts) !Weights {
    var dir = try std.Io.Dir.openDirAbsolute(io, model_dir, .{ .iterate = true });
    defer dir.close(io);
    return loadWeightsFromOpenDir(io, allocator, dir, model_dir, opts);
}

/// Load every `*.safetensors` in an already-open `dir` into a Weights map.
/// `model_dir` is the on-disk path string, used both to build the per-file
/// absolute path for `mlx_load_safetensors` and to phrase the error message.
/// Kept apart from `loadWeightsOpt` so the incomplete-checkpoint guard below is
/// unit-testable against a `tmpDir` (mirrors `model_discovery.discoverModelsInDir`).
pub fn loadWeightsFromOpenDir(io: std.Io, allocator: std.mem.Allocator, dir: std.Io.Dir, model_dir: []const u8, opts: LoadOpts) !Weights {
    var weights = Weights.init(allocator);
    errdefer weights.deinit();

    const s = mlx.weightLoadStream();
    defer _ = mlx.mlx_stream_free(s);

    // The index names the shards; anything else is dead weight (issue #274:
    // a pack shipped two shards no weight_map entry names — RAM for nothing)
    // or a foreign file whose parse failure would be an uncatchable MLX abort.
    var referenced = resident.indexShardSet(io, dir);
    defer if (referenced) |*r| resident.freeShardSet(r);

    var owners_arena = std.heap.ArenaAllocator.init(allocator);
    defer owners_arena.deinit();
    var owners: ?Owners = if (opts.index_owners) indexOwners(io, owners_arena.allocator(), dir) else null;

    var file_count: u32 = 0;
    var it = dir.iterate();
    while (try it.next(io)) |entry| {
        // Accept regular files AND symlinks: HuggingFace cache snapshots store
        // every weight file as a symlink into ../../blobs/<hash>. mlx_load_safetensors
        // resolves the link at the OS level, so a symlinked *.safetensors loads fine.
        if (entry.kind != .file and entry.kind != .sym_link) continue;
        if (!std.mem.endsWith(u8, entry.name, ".safetensors")) continue;
        if (referenced) |r| if (!r.contains(entry.name)) {
            log.warn("skipping {s}: not named by model.safetensors.index.json\n", .{entry.name});
            continue;
        };

        const path_slice = try std.fmt.allocPrint(allocator, "{s}/{s}", .{ model_dir, entry.name });
        defer allocator.free(path_slice);
        const path = try allocator.dupeSentinel(u8, path_slice, 0);
        defer allocator.free(path);

        log.info("Loading {s}...\n", .{entry.name});
        var shard_opts = opts;
        shard_opts.shard = entry.name;
        if (owners) |*o| shard_opts.owners = o;
        try loadSafetensorsFile(allocator, &weights, path, s, shard_opts);
        file_count += 1;
    }
    // A ModelOpt index may name shards in a subdirectory (NVIDIA's `tensors/`); the scan above sees the root only.
    if (opts.modelopt) if (referenced) |r| {
        var kit = r.keyIterator();
        while (kit.next()) |name| {
            if (std.mem.indexOfScalar(u8, name.*, '/') == null or std.mem.indexOf(u8, name.*, "..") != null) continue;
            const path = try std.fmt.allocPrintSentinel(allocator, "{s}/{s}", .{ model_dir, name.* }, 0);
            defer allocator.free(path);
            log.info("Loading {s}...\n", .{name.*});
            var shard_opts = opts;
            shard_opts.shard = name.*;
            if (owners) |*o| shard_opts.owners = o;
            try loadSafetensorsFile(allocator, &weights, path, s, shard_opts);
            file_count += 1;
        }
    };

    // Incomplete-checkpoint guard. A dir with config/tokenizer but no (or no
    // usable) *.safetensors is the classic interrupted-download shape: the
    // small files land first, the multi-GB weight shards never finalize. Before
    // this guard the loader returned an empty map and the caller crashed with a
    // misleading `MISSING WEIGHT: <prefix>.embed_tokens.weight` (the first
    // weight looked up) + `unreachable`, pointing at the model arch instead of
    // the download. Fail here with an actionable message, mirroring the
    // tokenizer path's "incomplete download?" hint (see main.zig).
    if (weights.count() == 0) {
        log.err("no usable weights loaded from {s} ({d} *.safetensors file(s) found) — the checkpoint looks like an incomplete download (config/tokenizer present, weight shards missing). Re-download the model (e.g. `mlx-serve pull <model>`) or delete the dir and re-fetch.\n", .{ model_dir, file_count });
        return error.NoWeightFiles;
    }

    try viewByteAffineWeights(&weights, s);
    log.info("Loaded {d} weights from {d} file(s)\n", .{ weights.count(), file_count });
    if (!opts.defer_qwen4_norms) reportF16Narrowing();
    return weights;
}

/// An affine weight (`.scales` AND `.biases` beside it) stored as uint8 is MLX's packed
/// uint32 words written byte-wise (Recursive's Flint export); MLX's quantized kernels read
/// uint32 only. NVFP4/MXFP8/FP8 bytes carry no `.biases` and keep their dtype.
pub fn viewByteAffineWeights(weights: *Weights, s: mlx.mlx_stream) !void {
    var buf: [512]u8 = undefined;
    var it = weights.map.iterator();
    while (it.next()) |e| {
        const key = e.key_ptr.*;
        if (!std.mem.endsWith(u8, key, ".weight") or mlx.mlx_array_dtype(e.value_ptr.*) != .uint8) continue;
        const shape = mlx.getShape(e.value_ptr.*);
        if (shape.len == 0 or @rem(shape[shape.len - 1], 4) != 0) continue;
        const base = key[0 .. key.len - ".weight".len];
        const scales = std.fmt.bufPrint(&buf, "{s}.scales", .{base}) catch continue;
        if (weights.map.get(scales) == null) continue;
        const biases = std.fmt.bufPrint(&buf, "{s}.biases", .{base}) catch continue;
        if (weights.map.get(biases) == null) continue;
        var words = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_view(&words, e.value_ptr.*, .uint32, s));
        _ = mlx.mlx_array_free(e.value_ptr.*);
        e.value_ptr.* = words;
    }
}

/// Whether a just-loaded f16 tensor must be narrowed to the engine's bf16
/// activation dtype.
///
/// Two shapes qualify, for the same underlying reason — an f16 value that
/// meets a bf16 activation promotes the RESULT to f32:
///
///   - Quant SIDE tensors (scales/biases), which can be 2-D so they are keyed
///     on the suffix. f16 side tensors force gather_qmm/qmatmul onto a ~4x
///     slower mixed-dtype path (hy_v3 2-bit live, 0.70 vs 0.18 ms
///     per 8-expert gather — 1.2 tok/s on the 295B instead of ~15+).
///   - ANY 1-D f16 tensor: a per-channel table (norm weight, bias, A_log,
///     dt_bias) that is multiplied or added straight into the residual. Leave
///     one f16 and the residual turns f32 at the first layer and STAYS f32,
///     so every later weight read is upcast — the Laguna YaRN-mscale class,
///     one level up. Measured on prism-ml/Ternary-Bonsai-27B-mlx-2bit (the
///     only f16 checkpoint on hand, qwen3_5 GDN hybrid): 27.99 -> 23.88
///     ms/forward, 14.7%, three paired boots with cooldown.
///
/// Plain multi-dimensional WEIGHTS keep their dtype. They are matmul
/// OPERANDS, and MLX selects its kernel off that dtype, so narrowing one is a
/// kernel-selection change rather than a promotion fix — measured as a wash
/// here (23.15 vs 23.47 ms, inside boot-to-boot drift), so the minimal rule
/// is the one that ships.
///
/// The cast node stays lazy, so the load-time batch eval materializes bf16
/// directly. Delta from the 3 dropped mantissa bits: cos 0.99999994 — far
/// below any quant noise floor.
pub fn narrowsLoadedF16(key: []const u8, ndim: usize, dtype: mlx.mlx_dtype) bool {
    if (dtype != .float16) return false;
    if (std.mem.endsWith(u8, key, ".scales") or std.mem.endsWith(u8, key, ".biases")) return true;
    return ndim == 1;
}

/// An f32 layer-norm weight beside bf16 activations (NVIDIA's ModelOpt GLM pack stores
/// `post_attention_layernorm` f32; the loader narrows it for ModelOpt packs only, the one
/// reference measured): `rms_norm` promotes to the weight's dtype, so the MoE reads f32
/// and the fused hyper-connection read declines. Narrowed like the f16 norms; the f32 per-channel
/// state tables (`A_log`, `dt_bias`, `hc_*`, `e_score_correction_bias`) are not norms and stay f32, and a
/// vision tower keeps the f32 its reference runs in.
pub fn narrowsLoadedF32Norm(key: []const u8, ndim: usize, dtype: mlx.mlx_dtype) bool {
    return dtype == .float32 and ndim == 1 and std.mem.indexOf(u8, key, "language_model.") != null and
        (std.mem.endsWith(u8, key, "layernorm.weight") or std.mem.endsWith(u8, key, "language_model.norm.weight"));
}

/// ON for Metal and CUDA: an f16 norm left beside bf16 activations widens the
/// whole residual to f32. OFF for the Omarchy Vulkan backend, which has no
/// bf16 GPU kernel.
pub fn narrow1dDefault(darwin: bool, cuda: bool) bool {
    return darwin or cuda;
}

pub fn narrow1dEnabled() bool {
    if (narrow_1d_cached) |v| return v;
    const on = narrow1dDefault(builtin.os.tag.isDarwin(), mlx.cudaAvailable());
    narrow_1d_cached = on;
    return on;
}

pub fn reportF16Narrowing() void {
    if (narrowed_1d == 0) return;
    log.info("[dtype] narrowed {d} 1-D f16 tables to bf16\n", .{narrowed_1d});
    narrowed_1d = 0;
}

pub fn loadSafetensorsFile(
    allocator: std.mem.Allocator,
    weights: *Weights,
    path: [*:0]const u8,
    s: mlx.mlx_stream,
    opts: LoadOpts,
) !void {
    var tensor_map = mlx.mlx_map_string_to_array_new();
    defer _ = mlx.mlx_map_string_to_array_free(tensor_map);

    var meta_map = mlx.mlx_map_string_to_string_new();
    defer _ = mlx.mlx_map_string_to_string_free(meta_map);

    if (opts.dsv41 and comptime builtin.os.tag.isDarwin())
        try @import("nocache_reader.zig").loadSafetensors(&tensor_map, &meta_map, path, s)
    else
        try mlx.check(mlx.mlx_load_safetensors(&tensor_map, &meta_map, path, s));

    const iter = mlx.mlx_map_string_to_array_iterator_new(tensor_map);
    defer _ = mlx.mlx_map_string_to_array_iterator_free(iter);

    while (true) {
        var key: ?[*:0]const u8 = null;
        var value = mlx.mlx_array_new();

        const ret = mlx.mlx_map_string_to_array_iterator_next(&key, &value, iter);
        if (ret != 0 or key == null) {
            _ = mlx.mlx_array_free(value);
            break;
        }

        const key_str = std.mem.span(key.?);

        const foreign = if (opts.owners) |o| (if (o.get(key_str)) |owner| !std.mem.eql(u8, owner, opts.shard) else false) else false;
        if (foreign or !keepsTensor(key_str, opts) or (if (opts.skip) |skip| skip(key_str) else false) or
            (opts.dsv41 and dsv41DropsKey(key_str, opts.vision)))
        {
            _ = mlx.mlx_array_free(value);
            continue;
        }

        // Read the shape BEFORE the cast frees `value` — a freed handle's
        // ndim is a use-after-free, not a zero.
        const ndim = mlx.mlx_array_ndim(value);
        var final_value = value;
        if (!(opts.defer_qwen4_norms and qwen4NormNeedsFold(key_str)) and
            !opts.keep_f16 and (narrowsLoadedF16(key_str, ndim, mlx.mlx_array_dtype(value)) or (opts.modelopt and narrowsLoadedF32Norm(key_str, ndim, mlx.mlx_array_dtype(value)))) and
            (ndim != 1 or narrow1dEnabled()))
        {
            var cast = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_astype(&cast, value, .bfloat16, s));
            _ = mlx.mlx_array_free(value);
            final_value = cast;
            if (ndim == 1) narrowed_1d += 1;
        }

        // oMLX's V4.1 packs nest the text model under `language_model.`: kept under the release's names.
        const name = if (opts.dsv41 and std.mem.startsWith(u8, key_str, dsv41_text_prefix)) key_str[dsv41_text_prefix.len..] else key_str;
        const owned_key = try allocator.dupe(u8, name);
        try weights.map.put(owned_key, final_value);
    }
}

pub const dsv41_text_prefix = "language_model.";

pub fn qwen4NormNeedsFold(key: []const u8) bool {
    const suffixes = [_][]const u8{
        "hc_norm.weight",      "q_norm.weight",         "k_norm.weight",        "q_layernorm.weight",           "k_layernorm.weight",
        "ple.norm_key.weight", "ple.norm_query.weight", "ple.norm_conv.weight", "pre_fc_norm_embedding.weight", "pre_fc_norm_hidden.weight",
    };
    for (suffixes) |suffix| if (std.mem.endsWith(u8, key, suffix)) return true;
    return false;
}

/// `shouldKeepWeightKey`, plus Qwen3-ASR's `audio_tower.*` under the same media gate.
pub fn keepsTensor(key: []const u8, opts: LoadOpts) bool {
    if (opts.modelopt and modelopt_pack.skipsAtLoad(key)) return false;
    if (opts.audio_tower and opts.vision and std.mem.startsWith(u8, key, "audio_tower.")) return true;
    return shouldKeepWeightKey(key, opts.vision);
}

/// True if the safetensors weight `key` should be retained for the text
/// forward pass. Audio is always dropped; vision is dropped unless
/// `load_vision` is set. MTP-style head tensors (`*.mtp.*`) on Qwen3.5/3.6
/// checkpoints are kept (the binder ignores them, but the loader doesn't
/// need to know that).
pub fn shouldKeepWeightKey(key: []const u8, load_vision: bool) bool {
    // Gemma 4 12B `gemma4_unified` is encoder-free: it ships a tiny vision
    // patch embedder (`vision_embedder.*` + `embed_vision.*`) and a raw-waveform
    // audio projection (`embed_audio.*`) instead of the SigLIP vision tower and
    // conformer audio tower of earlier Gemma 4 variants. Those embedders are
    // wired in src/engines/mlx/vision_encoder.zig (UnifiedEmbedder), so keep them under the same
    // `load_vision` gate as the SigLIP weights (`--no-vision` → text only).
    const is_vision = std.mem.startsWith(u8, key, "vision_tower.") or
        std.mem.startsWith(u8, key, "vision_model.") or
        std.mem.startsWith(u8, key, "vision_projector.") or
        std.mem.startsWith(u8, key, "mlp1.") or
        std.mem.startsWith(u8, key, "embed_vision.") or
        std.mem.startsWith(u8, key, "merger.") or
        std.mem.startsWith(u8, key, "vit_merger.") or
        std.mem.startsWith(u8, key, "vision_embedder.") or
        std.mem.startsWith(u8, key, "embed_audio.") or
        std.mem.startsWith(u8, key, "multi_modal_projector.") or
        std.mem.startsWith(u8, key, "language_model.multi_modal_projector.");
    // The heavy SigLIP-era conformer audio tower is still not wired — drop it.
    const is_audio_tower = std.mem.startsWith(u8, key, "audio_tower.") or
        std.mem.startsWith(u8, key, "language_model.audio_multi_modal_projector.");
    if (is_audio_tower) return false;
    // Muse-Glimmer nests its tower/adapter/projection under "model."; the
    // mlx-community re-nest drops that prefix (its bare "vision_tower." already
    // rides the is_vision gate above). Both follow --no-vision.
    if (!load_vision and (std.mem.startsWith(u8, key, "model.vision_tower.") or
        // LFM2-VL's HF nesting puts the projector one level above its tower.
        std.mem.startsWith(u8, key, "model.multi_modal_projector.") or
        std.mem.startsWith(u8, key, "model.vision_adapter.") or
        std.mem.startsWith(u8, key, "model.vision_projection.") or
        std.mem.startsWith(u8, key, "vision_adapter.") or
        std.mem.startsWith(u8, key, "vision_projection.") or
        // avlp12's Qwen3.8 "Alis" packs spell the Qwen3-VL tower
        // `model.visual.` (pure rename of `vision_tower.`).
        std.mem.startsWith(u8, key, "model.visual."))) return false;
    if (is_vision and !load_vision) return false;
    return true;
}

/// The 1-D arm. A load-time
/// dtype normalization is invisible once the model is up, so a one-boot A/B
/// switch is the only way to attribute a future f16-checkpoint regression to
/// it. The side-tensor arm predates this and is not switchable.
pub var narrow_1d_cached: ?bool = null;

/// Count of 1-D f16 tables narrowed this load — reported once per model so a
/// declined normalization is nameable from the log instead of silently
/// reading as "this checkpoint just isn't f16".
pub var narrowed_1d: usize = 0;

/// A tiny affine-quantized tensor rewritten as the dense `[out, in]` a dense
/// checkpoint ships (untransposed), so the paths keyed on the dense layout
/// (hc prefill, the fused HC read, the softmax_gate fold) serve it too.
pub fn dequantizeSmall(w: *mlx.mlx_array, sc: *mlx.mlx_array, bi: *mlx.mlx_array, in_dim: u32, owned: *std.ArrayList(mlx.mlx_array), allocator: std.mem.Allocator, s: mlx.mlx_stream) !void {
    if (sc.ctx == null or bi.ctx == null) return;
    const qp = affineParamsFromGeometry(w.*, sc.*, in_dim) orelse return;
    var d = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(d);
    try mlx.check(mlx.mlx_dequantize(&d, w.*, sc.*, bi.*, mlx.mlx_optional_int.some(@intCast(qp.group_size)), mlx.mlx_optional_int.some(@intCast(qp.bits)), "affine", .{ .ctx = null }, .{ .value = mlx.mlx_array_dtype(sc.*), .has_value = true }, s));
    try mlx.check(mlx.mlx_array_eval(d));
    try owned.append(allocator, d);
    w.* = d;
    sc.* = mlx.mlx_array_new();
    bi.* = mlx.mlx_array_new();
}
