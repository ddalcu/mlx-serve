//! Resident loader for the original MiMo-V2.6-Flash MOPD checkpoint.
//! Native per-expert MXFP4 bytes are packed into resident MLX banks, unchanged.

const std = @import("std");
const mlx = @import("mlx.zig");
const model = @import("model.zig");
const fp8_block = @import("fp8_block.zig");

const Allocator = std.mem.Allocator;

const MAX_HEADER_BYTES: u64 = 512 * 1024 * 1024;
const MAX_JSON_BYTES: usize = 512 * 1024 * 1024;
const FP8_BLOCK: u64 = 128;

const DType = enum {
    bf16,
    f16,
    f32,
    u8,
    u16,
    u32,
    fp8_e4m3,
    unknown,
};

const TensorMeta = struct {
    dtype: DType,
    shape: []const u64,
    data_start: u64,
    data_end: u64,
    data_base: u64,
    file: []const u8,
};

const HeaderBytes = struct { bytes: []u8, data_base: u64 };

const SourceIndex = struct {
    weight_map: std.StringHashMap([]const u8),
    files: std.StringHashMap(void),
    tensors: std.StringHashMap(TensorMeta),
};

const TensorKind = enum {
    resident,
    routed_expert,
    fp8_weight,
    fp8_scale,
    skipped,
};

const QkvGeometry = struct {
    q_rows: u64,
    k_rows: u64,
    v_rows: u64,

    fn total(self: QkvGeometry) u64 {
        return self.q_rows + self.k_rows + self.v_rows;
    }
};

pub fn loadMtpWeights(io: std.Io, allocator: std.mem.Allocator, model_dir: []const u8) !model.Weights {
    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    var source = try loadSourceIndex(io, arena.allocator(), model_dir);
    var weights = model.Weights.init(allocator);
    errdefer weights.deinit();
    var it = source.tensors.iterator();
    while (it.next()) |entry| {
        const key = entry.key_ptr.*;
        const meta = entry.value_ptr.*;
        if (!isMtpKey(key) or std.mem.endsWith(u8, key, ".weight_scale_inv")) continue;
        if (meta.dtype == .fp8_e4m3) {
            try loadFp8Weight(&weights, allocator, model_dir, key, meta, &source);
            continue;
        }
        const raw = try readTensor(allocator, model_dir, meta);
        defer allocator.free(raw);
        const arr = try uploadDense(raw, meta, .{ .ctx = null });
        errdefer _ = mlx.mlx_array_free(arr);
        try putWeight(&weights, allocator, key, arr);
    }
    return weights;
}

pub fn mtpResidentBytes(io: std.Io, allocator: std.mem.Allocator, model_dir: []const u8) !u64 {
    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    const source = try loadSourceIndex(io, arena.allocator(), model_dir);
    var total: u64 = 0;
    var it = source.tensors.iterator();
    while (it.next()) |entry| {
        if (isMtpKey(entry.key_ptr.*)) total += try payloadBytes(entry.value_ptr.*, null);
    }
    return total;
}

pub fn loadVisionWeightsInto(weights: *model.Weights, io: std.Io, allocator: std.mem.Allocator, model_dir: []const u8) !void {
    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    var source = try loadSourceIndex(io, arena.allocator(), model_dir);
    var it = source.tensors.iterator();
    while (it.next()) |entry| {
        const key = entry.key_ptr.*;
        if (!isVisionKey(key)) continue;
        var meta = entry.value_ptr.*;
        var flat: [2]u64 = undefined;
        if (meta.shape.len > 4) {
            flat = .{ meta.shape[0], try shapeProduct(meta.shape[1..]) };
            meta.shape = &flat;
        }
        const raw = try readTensor(allocator, model_dir, meta);
        defer allocator.free(raw);
        const arr = try uploadDense(raw, meta, .{ .ctx = null });
        errdefer _ = mlx.mlx_array_free(arr);
        try putWeight(weights, allocator, key, arr);
    }
}

pub fn visionResidentBytes(io: std.Io, allocator: std.mem.Allocator, model_dir: []const u8) !u64 {
    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    const source = try loadSourceIndex(io, arena.allocator(), model_dir);
    var total: u64 = 0;
    var it = source.tensors.iterator();
    while (it.next()) |entry| {
        if (isVisionKey(entry.key_ptr.*)) total += try payloadBytes(entry.value_ptr.*, null);
    }
    return total;
}

fn isVisionKey(key: []const u8) bool {
    return std.mem.startsWith(u8, key, "visual.");
}

fn isMtpKey(key: []const u8) bool {
    return std.mem.startsWith(u8, key, "model.mtp.layers.");
}

fn validateConfig(config: *const model.ModelConfig) !void {
    if (config.num_hidden_layers == 0 or config.num_hidden_layers > 128 or
        config.hidden_size == 0 or config.vocab_size == 0 or
        config.num_attention_heads == 0 or config.num_key_value_heads == 0 or
        config.head_dim == 0 or config.v_head_dim == 0 or
        config.intermediate_size == 0)
        return error.InvalidMimoGeometry;
    if (config.num_hidden_layers > 1 and config.num_experts == 0)
        return error.InvalidMimoGeometry;

    for (0..config.num_hidden_layers) |i| {
        const layer: u32 = @intCast(i);
        const heads = config.layerNumHeads(layer);
        const kv = config.layerKVHeads(layer);
        const hd = config.layerHeadDim(layer);
        const vd = config.layerVHeadDim(layer);
        if (heads == 0 or kv == 0 or hd == 0 or vd == 0 or heads % kv != 0)
            return error.InvalidMimoGeometry;
        if (@as(u64, heads) * hd > std.math.maxInt(u32) or
            @as(u64, kv) * hd > std.math.maxInt(u32) or
            @as(u64, kv) * vd > std.math.maxInt(u32))
            return error.InvalidMimoGeometry;
    }
}

fn readFileAlloc(
    io: std.Io,
    allocator: Allocator,
    path: []const u8,
    limit: usize,
) ![]u8 {
    const file = std.Io.Dir.openFileAbsolute(io, path, .{}) catch
        return error.MimoSourceFileMissing;
    defer file.close(io);
    var read_buf: [8192]u8 = undefined;
    var reader = file.reader(io, &read_buf);
    return reader.interface.allocRemaining(allocator, .limited(limit)) catch
        return error.MimoSourceRead;
}

fn preadExact(fd: std.c.fd_t, dst: []u8, offset: u64) !void {
    var done: usize = 0;
    while (done < dst.len) {
        const got = std.c.pread(
            fd,
            dst[done..].ptr,
            dst.len - done,
            @intCast(offset + done),
        );
        if (got < 0) {
            if (std.c._errno().* == @backingInt(std.c.E.INTR)) continue;
            return error.MimoSourceRead;
        }
        if (got == 0) return error.SafetensorsUnexpectedEof;
        done += @intCast(got);
    }
}

fn shardPath(allocator: Allocator, model_dir: []const u8, file: []const u8) ![:0]u8 {
    if (!std.mem.eql(u8, file, std.fs.path.basename(file)) or !std.mem.endsWith(u8, file, ".safetensors"))
        return error.InvalidMimoShardPath;
    return std.fmt.allocPrintSentinel(allocator, "{s}/{s}", .{ model_dir, file }, 0);
}

fn readShardHeader(
    allocator: Allocator,
    model_dir: []const u8,
    file: []const u8,
) !HeaderBytes {
    const path = try shardPath(allocator, model_dir, file);
    defer allocator.free(path);
    const fd = std.c.open(path.ptr, .{ .ACCMODE = .RDONLY }, @as(std.c.mode_t, 0));
    if (fd < 0) return error.MissingMimoShard;
    defer _ = std.c.close(fd);

    var header_len_bytes: [8]u8 = undefined;
    try preadExact(fd, &header_len_bytes, 0);
    const header_len = std.mem.readInt(u64, &header_len_bytes, .little);
    if (header_len > MAX_HEADER_BYTES) return error.SafetensorsHeaderTooLarge;
    const header_len_usize = std.math.cast(usize, header_len) orelse
        return error.SafetensorsHeaderTooLarge;
    const header = try allocator.alloc(u8, header_len_usize);
    errdefer allocator.free(header);
    try preadExact(fd, header, 8);
    const data_base = std.math.add(u64, 8, header_len) catch
        return error.InvalidSafetensorsHeader;
    return .{ .bytes = header, .data_base = data_base };
}

fn parseDType(name: []const u8) DType {
    if (std.mem.eql(u8, name, "BF16")) return .bf16;
    if (std.mem.eql(u8, name, "F16")) return .f16;
    if (std.mem.eql(u8, name, "F32")) return .f32;
    if (std.mem.eql(u8, name, "U16")) return .u16;
    if (std.mem.eql(u8, name, "U8")) return .u8;
    if (std.mem.eql(u8, name, "U32")) return .u32;
    if (std.mem.eql(u8, name, "F8_E4M3") or std.mem.eql(u8, name, "F8_E4M3FN"))
        return .fp8_e4m3;
    return .unknown;
}

fn parseShape(allocator: Allocator, value: std.json.Value) ![]u64 {
    if (value != .array) return error.InvalidSafetensorsHeader;
    const shape = try allocator.alloc(u64, value.array.items.len);
    errdefer allocator.free(shape);
    for (value.array.items, 0..) |dim, i| {
        if (dim != .integer or dim.integer < 0) return error.InvalidSafetensorsHeader;
        shape[i] = std.math.cast(u64, dim.integer) orelse
            return error.InvalidSafetensorsHeader;
    }
    return shape;
}

fn parseOffsets(value: std.json.Value) ![2]u64 {
    if (value != .array or value.array.items.len != 2) return error.InvalidSafetensorsHeader;
    var offsets: [2]u64 = undefined;
    for (value.array.items, 0..) |v, i| {
        if (v != .integer or v.integer < 0) return error.InvalidSafetensorsHeader;
        offsets[i] = std.math.cast(u64, v.integer) orelse
            return error.InvalidSafetensorsHeader;
    }
    if (offsets[1] < offsets[0]) return error.InvalidSafetensorsHeader;
    return offsets;
}

fn parseTensorMeta(
    allocator: Allocator,
    value: std.json.Value,
    file: []const u8,
    data_base: u64,
) !TensorMeta {
    if (value != .object) return error.InvalidSafetensorsHeader;
    const dtype_value = value.object.get("dtype") orelse return error.InvalidSafetensorsHeader;
    if (dtype_value != .string) return error.InvalidSafetensorsHeader;
    const shape_value = value.object.get("shape") orelse return error.InvalidSafetensorsHeader;
    const offsets_value = value.object.get("data_offsets") orelse
        return error.InvalidSafetensorsHeader;
    const shape = try parseShape(allocator, shape_value);
    errdefer allocator.free(shape);
    const offsets = try parseOffsets(offsets_value);
    _ = std.math.add(u64, data_base, offsets[1]) catch
        return error.InvalidSafetensorsHeader;
    return .{
        .dtype = parseDType(dtype_value.string),
        .shape = shape,
        .data_start = offsets[0],
        .data_end = offsets[1],
        .data_base = data_base,
        .file = file,
    };
}

fn loadSourceIndex(io: std.Io, allocator: Allocator, model_dir: []const u8) !SourceIndex {
    if (model_dir.len == 0 or !std.fs.path.isAbsolute(model_dir))
        return error.InvalidMimoModelPath;

    const index_path = try std.fmt.allocPrint(
        allocator,
        "{s}/model.safetensors.index.json",
        .{model_dir},
    );
    const index_bytes = try readFileAlloc(io, allocator, index_path, MAX_JSON_BYTES);
    const parsed = std.json.parseFromSliceLeaky(
        std.json.Value,
        allocator,
        index_bytes,
        .{},
    ) catch return error.InvalidSafetensorsIndex;
    if (parsed != .object) return error.InvalidSafetensorsIndex;
    const weight_map_value = parsed.object.get("weight_map") orelse
        return error.InvalidSafetensorsIndex;
    if (weight_map_value != .object or weight_map_value.object.count() == 0)
        return error.InvalidSafetensorsIndex;

    var source = SourceIndex{
        .weight_map = std.StringHashMap([]const u8).init(allocator),
        .files = std.StringHashMap(void).init(allocator),
        .tensors = std.StringHashMap(TensorMeta).init(allocator),
    };
    var it = weight_map_value.object.iterator();
    while (it.next()) |entry| {
        if (entry.value_ptr.* != .string or entry.value_ptr.string.len == 0)
            return error.InvalidSafetensorsIndex;
        if (source.weight_map.contains(entry.key_ptr.*))
            return error.AmbiguousSafetensorsIndex;
        try source.weight_map.put(entry.key_ptr.*, entry.value_ptr.string);
        try source.files.put(entry.value_ptr.string, {});
    }

    var files = source.files.iterator();
    while (files.next()) |file_entry| {
        const filename = file_entry.key_ptr.*;
        var header_arena = std.heap.ArenaAllocator.init(allocator);
        defer header_arena.deinit();
        const header_scratch = header_arena.allocator();
        const header = try readShardHeader(header_scratch, model_dir, filename);
        const header_root = std.json.parseFromSliceLeaky(
            std.json.Value,
            header_scratch,
            header.bytes,
            .{},
        ) catch return error.InvalidSafetensorsHeader;
        if (header_root != .object) return error.InvalidSafetensorsHeader;

        // Index iteration is deliberate. It leaves all header-name slices in
        // the short-lived arena and copies only shapes for indexed tensors.
        var indexed = source.weight_map.iterator();
        while (indexed.next()) |indexed_entry| {
            if (!std.mem.eql(u8, indexed_entry.value_ptr.*, filename)) continue;
            const tensor_value = header_root.object.get(indexed_entry.key_ptr.*) orelse
                continue;
            if (source.tensors.contains(indexed_entry.key_ptr.*))
                return error.AmbiguousSafetensorsTensor;
            const meta = try parseTensorMeta(
                header_scratch,
                tensor_value,
                filename,
                header.data_base,
            );
            const shape = try allocator.dupe(u64, meta.shape);
            try source.tensors.put(indexed_entry.key_ptr.*, .{
                .dtype = meta.dtype,
                .shape = shape,
                .data_start = meta.data_start,
                .data_end = meta.data_end,
                .data_base = meta.data_base,
                .file = filename,
            });
        }
    }

    var indexed = source.weight_map.iterator();
    while (indexed.next()) |entry| {
        if (!source.tensors.contains(entry.key_ptr.*))
            return error.MissingIndexedSafetensorsTensor;
    }
    return source;
}

fn layerKey(key: []const u8) ?struct { layer: u32, rest: []const u8 } {
    const prefix = "model.layers.";
    if (!std.mem.startsWith(u8, key, prefix)) return null;
    const after = key[prefix.len..];
    const dot = std.mem.indexOfScalar(u8, after, '.') orelse return null;
    if (dot == 0) return null;
    const layer = std.fmt.parseInt(u32, after[0..dot], 10) catch return null;
    return .{ .layer = layer, .rest = after[dot + 1 ..] };
}

fn classifyKey(key: []const u8, config: *const model.ModelConfig) !TensorKind {
    if (std.mem.startsWith(u8, key, "visual.") or
        std.mem.startsWith(u8, key, "audio_encoder.") or
        std.mem.startsWith(u8, key, "speech_embeddings.") or
        std.mem.startsWith(u8, key, "model.mtp.") or
        std.mem.startsWith(u8, key, "mtp."))
        return .skipped;
    if (std.mem.startsWith(u8, key, "model.layers.") and
        std.mem.indexOf(u8, key, ".mlp.experts.") != null)
        return .routed_expert;

    if (std.mem.eql(u8, key, "model.embed_tokens.weight") or
        std.mem.eql(u8, key, "lm_head.weight") or
        std.mem.eql(u8, key, "model.norm.weight"))
        return .resident;

    const ref = layerKey(key) orelse return error.UnclassifiedMimoTensor;
    if (ref.layer >= config.num_hidden_layers) return error.MimoLayerOutOfRange;
    // The dense prefix is `first_k_dense_replace` layers, as the parser and the
    // transformer read it.
    if (std.mem.eql(u8, ref.rest, "self_attn.qkv_proj.weight") or
        (ref.layer < config.first_k_dense_replace and
            (std.mem.eql(u8, ref.rest, "mlp.gate_proj.weight") or
                std.mem.eql(u8, ref.rest, "mlp.up_proj.weight") or
                std.mem.eql(u8, ref.rest, "mlp.down_proj.weight"))))
        return .fp8_weight;
    if (std.mem.eql(u8, ref.rest, "self_attn.qkv_proj.weight_scale_inv") or
        (ref.layer < config.first_k_dense_replace and
            (std.mem.eql(u8, ref.rest, "mlp.gate_proj.weight_scale_inv") or
                std.mem.eql(u8, ref.rest, "mlp.up_proj.weight_scale_inv") or
                std.mem.eql(u8, ref.rest, "mlp.down_proj.weight_scale_inv"))))
        return .fp8_scale;

    if (std.mem.eql(u8, ref.rest, "input_layernorm.weight") or
        std.mem.eql(u8, ref.rest, "post_attention_layernorm.weight") or
        std.mem.eql(u8, ref.rest, "self_attn.o_proj.weight") or
        std.mem.eql(u8, ref.rest, "self_attn.attention_sink_bias") or
        std.mem.eql(u8, ref.rest, "mlp.gate.weight") or
        std.mem.eql(u8, ref.rest, "mlp.gate.e_score_correction_bias"))
        return .resident;

    return error.UnclassifiedMimoTensor;
}

fn qkvGeometry(config: *const model.ModelConfig, layer: u32) QkvGeometry {
    const heads = config.layerNumHeads(layer);
    const kv = config.layerKVHeads(layer);
    const hd = config.layerHeadDim(layer);
    const vd = config.layerVHeadDim(layer);
    return .{
        .q_rows = @as(u64, heads) * hd,
        .k_rows = @as(u64, kv) * hd,
        .v_rows = @as(u64, kv) * vd,
    };
}

fn shapeProduct(shape: []const u64) !u64 {
    var product: u64 = 1;
    for (shape) |dim| {
        product = std.math.mul(u64, product, dim) catch
            return error.SafetensorsShapeOverflow;
    }
    return product;
}

fn payloadBytes(meta: TensorMeta, expected_dtype: ?DType) !u64 {
    if (expected_dtype) |want| if (meta.dtype != want) return error.MimoTensorDtypeMismatch;
    const elem_size: u64 = switch (meta.dtype) {
        .bf16, .f16, .u16 => 2,
        .f32, .u32 => 4,
        .fp8_e4m3, .u8 => 1,
        .unknown => return error.UnsupportedMimoTensorDtype,
    };
    const want = std.math.mul(u64, try shapeProduct(meta.shape), elem_size) catch
        return error.SafetensorsShapeOverflow;
    if (meta.data_end - meta.data_start != want) return error.SafetensorsPayloadMismatch;
    return want;
}

fn expectShape(meta: TensorMeta, expected: []const u64) !void {
    if (meta.shape.len != expected.len) return error.MimoTensorShapeMismatch;
    for (meta.shape, expected) |got, want| {
        if (got != want) return error.MimoTensorShapeMismatch;
    }
}

fn qkvSplit(geometry: QkvGeometry, scale_rows: u64) !fp8_block.RowSplit {
    return fp8_block.RowSplit.qkv(geometry.q_rows, geometry.k_rows, geometry.v_rows, scale_rows);
}

fn isQkvWeightKey(key: []const u8) bool {
    return std.mem.endsWith(u8, key, ".self_attn.qkv_proj.weight");
}

fn fp8Base(key: []const u8) []const u8 {
    return key[0 .. key.len - ".weight".len];
}

fn scaleKey(allocator: Allocator, key: []const u8) ![]u8 {
    return std.fmt.allocPrint(allocator, "{s}.weight_scale_inv", .{fp8Base(key)});
}

fn validateFp8Pair(
    source: *const SourceIndex,
    allocator: Allocator,
    config: *const model.ModelConfig,
    key: []const u8,
    meta: TensorMeta,
) !void {
    if (meta.dtype != .fp8_e4m3) return error.MimoTensorDtypeMismatch;
    if (meta.shape.len != 2) return error.MimoTensorShapeMismatch;
    const ref = layerKey(key) orelse return error.UnclassifiedMimoTensor;
    const s_key = try scaleKey(allocator, key);
    const scale_meta = source.tensors.get(s_key) orelse return error.MissingFp8Scale;
    if (scale_meta.dtype != .f32 or scale_meta.shape.len != 2)
        return error.InvalidFp8ScaleShape;

    const rows = meta.shape[0];
    const cols = meta.shape[1];
    if (cols == 0 or cols % FP8_BLOCK != 0)
        return error.InvalidFp8Shape;
    if (isQkvWeightKey(key)) {
        if (cols != config.hidden_size) return error.InvalidFp8Shape;
        const geometry = qkvGeometry(config, ref.layer);
        if (rows != geometry.total()) return error.InvalidQkvGeometry;
        if (scale_meta.shape[1] != cols / FP8_BLOCK)
            return error.InvalidFp8ScaleShape;
        _ = try qkvSplit(geometry, scale_meta.shape[0]);
    } else {
        var expected_rows = rows;
        var expected_cols = cols;
        if (ref.layer >= config.first_k_dense_replace) return error.UnclassifiedMimoTensor;
        if (std.mem.eql(u8, ref.rest, "mlp.gate_proj.weight") or
            std.mem.eql(u8, ref.rest, "mlp.up_proj.weight"))
        {
            expected_rows = config.intermediate_size;
            expected_cols = config.hidden_size;
        } else if (std.mem.eql(u8, ref.rest, "mlp.down_proj.weight")) {
            expected_rows = config.hidden_size;
            expected_cols = config.intermediate_size;
        } else {
            return error.UnclassifiedMimoTensor;
        }
        if (rows != expected_rows or cols != expected_cols)
            return error.InvalidFp8Shape;
        const required_scale_rows = (rows + FP8_BLOCK - 1) / FP8_BLOCK;
        if (scale_meta.shape[0] < required_scale_rows or
            scale_meta.shape[1] != cols / FP8_BLOCK)
            return error.InvalidFp8ScaleShape;
    }
    _ = try payloadBytes(meta, .fp8_e4m3);
    _ = try payloadBytes(scale_meta, .f32);
}

fn denseExpectedShape(
    key: []const u8,
    config: *const model.ModelConfig,
) !struct { shape: [2]u64, len: usize, dtype: DType } {
    if (std.mem.eql(u8, key, "model.embed_tokens.weight") or
        std.mem.eql(u8, key, "lm_head.weight"))
        return .{ .shape = .{ config.vocab_size, config.hidden_size }, .len = 2, .dtype = .bf16 };
    if (std.mem.eql(u8, key, "model.norm.weight"))
        return .{ .shape = .{ config.hidden_size, 0 }, .len = 1, .dtype = .bf16 };

    const ref = layerKey(key) orelse return error.UnclassifiedMimoTensor;
    const layer = ref.layer;
    if (std.mem.eql(u8, ref.rest, "input_layernorm.weight") or
        std.mem.eql(u8, ref.rest, "post_attention_layernorm.weight"))
        return .{ .shape = .{ config.hidden_size, 0 }, .len = 1, .dtype = .bf16 };
    if (std.mem.eql(u8, ref.rest, "self_attn.o_proj.weight")) {
        return .{
            .shape = .{
                config.hidden_size,
                @as(u64, config.layerNumHeads(layer)) * config.layerVHeadDim(layer),
            },
            .len = 2,
            .dtype = .bf16,
        };
    }
    if (std.mem.eql(u8, ref.rest, "self_attn.attention_sink_bias"))
        return .{ .shape = .{ config.layerNumHeads(layer), 0 }, .len = 1, .dtype = .bf16 };
    if (std.mem.eql(u8, ref.rest, "mlp.gate.weight"))
        return .{ .shape = .{ config.num_experts, config.hidden_size }, .len = 2, .dtype = .bf16 };
    if (std.mem.eql(u8, ref.rest, "mlp.gate.e_score_correction_bias"))
        return .{ .shape = .{ config.num_experts, 0 }, .len = 1, .dtype = .f32 };
    return error.UnclassifiedMimoTensor;
}

fn validateDense(key: []const u8, meta: TensorMeta, config: *const model.ModelConfig) !void {
    const expected = try denseExpectedShape(key, config);
    if (meta.dtype != expected.dtype) return error.MimoTensorDtypeMismatch;
    const expected_slice = expected.shape[0..expected.len];
    try expectShape(meta, expected_slice);
    _ = try payloadBytes(meta, expected.dtype);
}

fn requireKind(
    source: *const SourceIndex,
    allocator: Allocator,
    config: *const model.ModelConfig,
    key: []const u8,
    expected: TensorKind,
) !void {
    const meta = source.tensors.get(key) orelse return error.MissingMimoRequiredTensor;
    if (try classifyKey(key, config) != expected)
        return error.MimoRequiredTensorKindMismatch;
    switch (expected) {
        .resident => try validateDense(key, meta, config),
        .routed_expert => {},
        .fp8_weight => try validateFp8Pair(source, allocator, config, key, meta),
        .fp8_scale, .skipped => {},
    }
}

fn validateRequired(
    source: *const SourceIndex,
    allocator: Allocator,
    config: *const model.ModelConfig,
) !void {
    try requireKind(source, allocator, config, "model.embed_tokens.weight", .resident);
    try requireKind(source, allocator, config, "lm_head.weight", .resident);
    try requireKind(source, allocator, config, "model.norm.weight", .resident);

    for (0..config.num_hidden_layers) |i| {
        const layer: u32 = @intCast(i);
        const layer_prefix = try std.fmt.allocPrint(allocator, "model.layers.{d}", .{layer});
        try requireKind(source, allocator, config, try std.fmt.allocPrint(allocator, "{s}.input_layernorm.weight", .{layer_prefix}), .resident);
        try requireKind(source, allocator, config, try std.fmt.allocPrint(allocator, "{s}.post_attention_layernorm.weight", .{layer_prefix}), .resident);
        try requireKind(source, allocator, config, try std.fmt.allocPrint(allocator, "{s}.self_attn.qkv_proj.weight", .{layer_prefix}), .fp8_weight);
        try requireKind(source, allocator, config, try std.fmt.allocPrint(allocator, "{s}.self_attn.o_proj.weight", .{layer_prefix}), .resident);
        if (config.layerHasAttnSinks(layer)) {
            try requireKind(source, allocator, config, try std.fmt.allocPrint(allocator, "{s}.self_attn.attention_sink_bias", .{layer_prefix}), .resident);
        }
        if (layer < config.first_k_dense_replace) {
            for ([_][]const u8{ "gate", "up", "down" }) |projection| {
                const k = try std.fmt.allocPrint(
                    allocator,
                    "{s}.mlp.{s}_proj.weight",
                    .{ layer_prefix, projection },
                );
                try requireKind(source, allocator, config, k, .fp8_weight);
            }
        } else {
            try requireKind(source, allocator, config, try std.fmt.allocPrint(allocator, "{s}.mlp.gate.weight", .{layer_prefix}), .resident);
            try requireKind(source, allocator, config, try std.fmt.allocPrint(allocator, "{s}.mlp.gate.e_score_correction_bias", .{layer_prefix}), .resident);
            try validateExpertLayer(source, allocator, config, layer);
        }
    }
}

fn validatePlan(source: *const SourceIndex, allocator: Allocator, config: *const model.ModelConfig) !void {
    try validateRequired(source, allocator, config);
    var it = source.tensors.iterator();
    while (it.next()) |entry| {
        const key = entry.key_ptr.*;
        const meta = entry.value_ptr.*;
        switch (try classifyKey(key, config)) {
            .skipped => {},
            .resident => try validateDense(key, meta, config),
            .routed_expert => {},
            .fp8_weight => try validateFp8Pair(source, allocator, config, key, meta),
            .fp8_scale => {
                const suffix = ".weight_scale_inv";
                if (!std.mem.endsWith(u8, key, suffix))
                    return error.UnclassifiedMimoTensor;
                const base = key[0 .. key.len - suffix.len];
                const weight_key = try std.fmt.allocPrint(allocator, "{s}.weight", .{base});
                if (!source.tensors.contains(weight_key))
                    return error.MissingFp8Weight;
            },
        }
    }
}

fn countResidentBytes(
    source: *const SourceIndex,
    allocator: Allocator,
    config: *const model.ModelConfig,
) !u64 {
    var total: u64 = 0;
    var it = source.tensors.iterator();
    while (it.next()) |entry| {
        const key = entry.key_ptr.*;
        const meta = entry.value_ptr.*;
        switch (try classifyKey(key, config)) {
            .skipped, .fp8_scale => {},
            .resident, .routed_expert => |kind| {
                _ = kind;
                var bytes = try payloadBytes(meta, null);
                // The transformer loader keeps an f32 copy of each router for f32 routing.
                if (layerKey(key)) |ref| if (std.mem.eql(u8, ref.rest, "mlp.gate.weight")) {
                    bytes += try shapeProduct(meta.shape) * 4;
                };
                total = std.math.add(u64, total, bytes) catch
                    return error.ResidentBytesOverflow;
            },
            .fp8_weight => {
                const scale_key = try scaleKey(allocator, key);
                const scale_meta = source.tensors.get(scale_key) orelse
                    return error.MissingFp8Scale;
                const bytes = try payloadBytes(meta, .fp8_e4m3) + try payloadBytes(scale_meta, .f32);
                total = std.math.add(u64, total, bytes) catch
                    return error.ResidentBytesOverflow;
            },
        }
    }
    return total;
}

fn readTensor(allocator: Allocator, model_dir: []const u8, meta: TensorMeta) ![]u8 {
    const len_u64 = meta.data_end - meta.data_start;
    const len = std.math.cast(usize, len_u64) orelse return error.SafetensorsShapeOverflow;
    const out = try allocator.alloc(u8, len);
    errdefer allocator.free(out);
    const path = try shardPath(allocator, model_dir, meta.file);
    defer allocator.free(path);
    const fd = std.c.open(path.ptr, .{ .ACCMODE = .RDONLY }, @as(std.c.mode_t, 0));
    if (fd < 0) return error.MissingMimoShard;
    defer _ = std.c.close(fd);
    const absolute = std.math.add(u64, meta.data_base, meta.data_start) catch
        return error.InvalidSafetensorsHeader;
    try preadExact(fd, out, absolute);
    return out;
}

fn shapeForUpload(meta: TensorMeta, shape: *[4]c_int) ![]const c_int {
    if (meta.shape.len == 0 or meta.shape.len > shape.len)
        return error.MimoTensorShapeMismatch;
    for (meta.shape, 0..) |dim, i| {
        shape[i] = std.math.cast(c_int, dim) orelse return error.MimoTensorShapeMismatch;
    }
    return shape[0..meta.shape.len];
}

fn uploadDense(raw: []const u8, meta: TensorMeta, stream: mlx.mlx_stream) !mlx.mlx_array {
    var shape: [4]c_int = undefined;
    const shape_slice = try shapeForUpload(meta, &shape);
    const dtype: mlx.mlx_dtype = switch (meta.dtype) {
        .bf16 => .bfloat16,
        .f16 => .float16,
        .f32 => .float32,
        .u16 => .uint16,
        .u32 => .uint32,
        .u8 => .uint8,
        else => return error.MimoTensorDtypeMismatch,
    };
    _ = stream;
    return mlx.mlx_array_new_data(
        @ptrCast(raw.ptr),
        shape_slice.ptr,
        @intCast(shape_slice.len),
        dtype,
    );
}

fn validateFp8Payload(codes: []const u8, scales: []const u8) !void {
    for (codes) |code| {
        if (code & 0x7f == 0x7f) return error.InvalidFp8Value;
    }
    if (scales.len % 4 != 0) return error.SafetensorsPayloadMismatch;
    const bf16_max: f32 = @bitCast(@as(u32, 0x7f7f0000));
    for (0..scales.len / 4) |i| {
        const scale: f32 = @bitCast(std.mem.readInt(u32, scales[i * 4 ..][0..4], .little));
        if (!std.math.isFinite(scale) or @abs(scale) * 448.0 > bf16_max) return error.InvalidFp8Scale;
    }
}

fn putWeight(weights: *model.Weights, allocator: Allocator, key: []const u8, arr: mlx.mlx_array) !void {
    if (weights.map.contains(key)) return error.DuplicateOutputWeight;
    const owned_key = try allocator.dupe(u8, key);
    errdefer allocator.free(owned_key);
    try weights.map.put(owned_key, arr);
}

fn loadFp8Weight(
    weights: *model.Weights,
    allocator: Allocator,
    model_dir: []const u8,
    key: []const u8,
    meta: TensorMeta,
    source: *const SourceIndex,
) !void {
    const scale_name = try scaleKey(allocator, key);
    defer allocator.free(scale_name);
    const scale_meta = source.tensors.get(scale_name) orelse return error.MissingFp8Scale;
    const raw = try readTensor(allocator, model_dir, meta);
    defer allocator.free(raw);
    const scale_raw = try readTensor(allocator, model_dir, scale_meta);
    defer allocator.free(scale_raw);
    try validateFp8Payload(raw, scale_raw);

    var shape: [4]c_int = undefined;
    const w_shape = try shapeForUpload(meta, &shape);
    var w = mlx.mlx_array_new_data(@ptrCast(raw.ptr), w_shape.ptr, @intCast(w_shape.len), .uint8);
    errdefer _ = mlx.mlx_array_free(w);
    try putWeight(weights, allocator, key, w);
    w = .{};
    var sc = try uploadDense(scale_raw, scale_meta, .{ .ctx = null });
    errdefer _ = mlx.mlx_array_free(sc);
    const sc_key = try std.fmt.allocPrint(allocator, "{s}.scales", .{fp8Base(key)});
    defer allocator.free(sc_key);
    try putWeight(weights, allocator, sc_key, sc);
    sc = .{};
}

pub fn loadWeights(io: std.Io, allocator: Allocator, model_dir: []const u8, config: *const model.ModelConfig) !model.Weights {
    try validateConfig(config);
    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    var source = try loadSourceIndex(io, arena.allocator(), model_dir);
    try validatePlan(&source, arena.allocator(), config);
    var weights = model.Weights.init(allocator);
    errdefer weights.deinit();
    const stream = mlx.mlx_default_cpu_stream_new();
    defer _ = mlx.mlx_stream_free(stream);
    var it = source.tensors.iterator();
    while (it.next()) |entry| {
        const key = entry.key_ptr.*;
        const meta = entry.value_ptr.*;
        switch (try classifyKey(key, config)) {
            .skipped, .fp8_scale, .routed_expert => {},
            .resident => {
                const raw = try readTensor(allocator, model_dir, meta);
                defer allocator.free(raw);
                const arr = try uploadDense(raw, meta, stream);
                errdefer _ = mlx.mlx_array_free(arr);
                try putWeight(&weights, allocator, key, arr);
            },
            .fp8_weight => try loadFp8Weight(&weights, allocator, model_dir, key, meta, &source),
        }
    }
    for (config.first_k_dense_replace..config.num_hidden_layers) |li| {
        for ([_][]const u8{ "gate", "up", "down" }) |proj|
            try loadExpertBank(&weights, allocator, model_dir, &source, config, @intCast(li), proj);
    }
    return weights;
}

pub fn residentBytes(io: std.Io, allocator: Allocator, model_dir: []const u8, config: *const model.ModelConfig) !u64 {
    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    var source = try loadSourceIndex(io, arena.allocator(), model_dir);
    try validateConfig(config);
    try validatePlan(&source, arena.allocator(), config);
    return countResidentBytes(&source, arena.allocator(), config);
}

fn expertTensor(source: *const SourceIndex, config: *const model.ModelConfig, li: u32, expert: u32, proj: []const u8, scales: bool) !TensorMeta {
    var buf: [256]u8 = undefined;
    const key = try std.fmt.bufPrint(&buf, "model.layers.{d}.mlp.experts.{d}.{s}_proj.{s}", .{ li, expert, proj, if (scales) "weight_scale" else "weight" });
    const meta = source.tensors.get(key) orelse return error.MissingMimoExpertTensor;
    const down = std.mem.eql(u8, proj, "down");
    const rows = if (down) config.hidden_size else config.moe_intermediate_size;
    const cols = if (down) config.moe_intermediate_size else config.hidden_size;
    if (cols == 0 or cols % 32 != 0 or rows == 0) return error.InvalidMimoExpertGeometry;
    try expectShape(meta, &.{ rows, cols / @as(u32, if (scales) 32 else 2) });
    _ = try payloadBytes(meta, .u8);
    return meta;
}

fn validateExpertLayer(source: *const SourceIndex, allocator: Allocator, config: *const model.ModelConfig, layer: u32) !void {
    _ = allocator;
    for (0..config.num_experts) |expert| {
        for ([_][]const u8{ "gate", "up", "down" }) |proj| {
            _ = try expertTensor(source, config, layer, @intCast(expert), proj, false);
            _ = try expertTensor(source, config, layer, @intCast(expert), proj, true);
        }
    }
}

fn readTensorInto(allocator: Allocator, model_dir: []const u8, meta: TensorMeta, dst: []u8) !void {
    if (dst.len != try payloadBytes(meta, .u8)) return error.SafetensorsPayloadMismatch;
    const path = try shardPath(allocator, model_dir, meta.file);
    defer allocator.free(path);
    const fd = std.c.open(path.ptr, .{ .ACCMODE = .RDONLY }, @as(std.c.mode_t, 0));
    if (fd < 0) return error.MimoSourceFileMissing;
    defer _ = std.c.close(fd);
    try preadExact(fd, dst, meta.data_base + meta.data_start);
}

fn loadExpertBank(weights: *model.Weights, allocator: Allocator, model_dir: []const u8, source: *const SourceIndex, config: *const model.ModelConfig, li: u32, proj: []const u8) !void {
    const down = std.mem.eql(u8, proj, "down");
    const rows: usize = if (down) config.hidden_size else config.moe_intermediate_size;
    const cols: usize = if (down) config.moe_intermediate_size else config.hidden_size;
    const expert_bytes = rows * cols / 2;
    // One projection at a time bounds host staging to one packed bank.
    const words = try allocator.alloc(u32, config.num_experts * expert_bytes / 4);
    defer allocator.free(words);
    const raw = std.mem.sliceAsBytes(words);
    for (0..config.num_experts) |expert| {
        const meta = try expertTensor(source, config, li, @intCast(expert), proj, false);
        try readTensorInto(allocator, model_dir, meta, raw[expert * expert_bytes ..][0..expert_bytes]);
    }
    var shape = [_]c_int{ @intCast(config.num_experts), @intCast(rows), @intCast(cols / 8) };
    var w = mlx.mlx_array_new_data(@ptrCast(words.ptr), &shape, 3, .uint32);
    errdefer _ = mlx.mlx_array_free(w);
    var name: [256]u8 = undefined;
    try putWeight(weights, allocator, try std.fmt.bufPrint(&name, "model.layers.{d}.mlp.switch_mlp.{s}_proj.weight", .{ li, proj }), w);
    w = .{};
    try loadExpertScales(weights, allocator, model_dir, source, config, li, proj, rows, cols);
}

fn loadExpertScales(weights: *model.Weights, allocator: Allocator, model_dir: []const u8, source: *const SourceIndex, config: *const model.ModelConfig, li: u32, proj: []const u8, rows: usize, cols: usize) !void {
    const per = rows * cols / 32;
    const raw = try allocator.alloc(u8, config.num_experts * per);
    defer allocator.free(raw);
    for (0..config.num_experts) |expert| {
        const meta = try expertTensor(source, config, li, @intCast(expert), proj, true);
        try readTensorInto(allocator, model_dir, meta, raw[expert * per ..][0..per]);
    }
    const shape = [_]c_int{ @intCast(config.num_experts), @intCast(rows), @intCast(cols / 32) };
    const sc = mlx.mlx_array_new_data(@ptrCast(raw.ptr), &shape, 3, .uint8);
    errdefer _ = mlx.mlx_array_free(sc);
    var name: [256]u8 = undefined;
    try putWeight(weights, allocator, try std.fmt.bufPrint(&name, "model.layers.{d}.mlp.switch_mlp.{s}_proj.scales", .{ li, proj }), sc);
}

const TestTensor = struct { key: []const u8, dtype: []const u8, shape: []const u64, bytes: []const u8 };
fn appendTestFormat(
    allocator: Allocator,
    list: *std.ArrayList(u8),
    comptime format: []const u8,
    args: anytype,
) !void {
    const text = try std.fmt.allocPrint(allocator, format, args);
    defer allocator.free(text);
    try list.appendSlice(allocator, text);
}

fn writeTestShard(
    io: std.Io,
    allocator: Allocator,
    dir: std.Io.Dir,
    filename: []const u8,
    tensors: []const TestTensor,
) !void {
    var header: std.ArrayList(u8) = .empty;
    defer header.deinit(allocator);
    try header.append(allocator, '{');
    var data_size: usize = 0;
    for (tensors, 0..) |tensor, i| {
        if (i != 0) try header.append(allocator, ',');
        try appendTestFormat(
            allocator,
            &header,
            "\"{s}\":{{\"dtype\":\"{s}\",\"shape\":[",
            .{ tensor.key, tensor.dtype },
        );
        for (tensor.shape, 0..) |dim, j| {
            if (j != 0) try header.append(allocator, ',');
            try appendTestFormat(allocator, &header, "{d}", .{dim});
        }
        const end = std.math.add(usize, data_size, tensor.bytes.len) catch
            return error.TestFixtureTooLarge;
        try appendTestFormat(
            allocator,
            &header,
            "],\"data_offsets\":[{d},{d}]}}",
            .{ data_size, end },
        );
        data_size = end;
    }
    try header.append(allocator, '}');

    const total = std.math.add(usize, 8 + header.items.len, data_size) catch
        return error.TestFixtureTooLarge;
    const file_bytes = try allocator.alloc(u8, total);
    defer allocator.free(file_bytes);
    std.mem.writeInt(u64, file_bytes[0..8], header.items.len, .little);
    @memcpy(file_bytes[8 .. 8 + header.items.len], header.items);
    var at = 8 + header.items.len;
    for (tensors) |tensor| {
        @memcpy(file_bytes[at..][0..tensor.bytes.len], tensor.bytes);
        at += tensor.bytes.len;
    }
    try dir.writeFile(io, .{ .sub_path = filename, .data = file_bytes });
}

fn nativeTestConfig() model.ModelConfig {
    var c = model.ModelConfig{
        .model_type = "mimo_v2",
        .weight_prefix = "model",
        .mimo_source_checkpoint = true,
        .hidden_size = 128,
        .intermediate_size = 128,
        .moe_intermediate_size = 128,
        .vocab_size = 16,
        .num_hidden_layers = 2,
        .num_attention_heads = 64,
        .num_key_value_heads = 8,
        .num_global_key_value_heads = 4,
        .head_dim = 192,
        .global_head_dim = 192,
        .v_head_dim = 128,
        .global_v_head_dim = 128,
        .num_experts = 2,
        .num_experts_per_tok = 1,
        .first_k_dense_replace = 1,
        .quant_bits = 4,
        .quant_group_size = 32,
        .quant_mode = .mxfp4,
        .has_explicit_layer_types = true,
        .has_attn_sinks = true,
        .attn_sinks_global = false,
        .attn_sinks_sliding = true,
        .has_qk_norm = false,
        .norm_has_offset = false,
        .has_pre_ff_norm = false,
        .hidden_act = .silu,
        .scale_embeddings = false,
        .sliding_window = 4,
        .partial_rotary_factor = 0.334,
        .attention_value_scale = 0.707,
        .moe_sigmoid_router = true,
    };
    c.layer_is_global[0] = true;
    return c;
}

fn addNativeTestTensor(a: Allocator, list: *std.ArrayList(TestTensor), key: []const u8, dtype: []const u8, shape: []const u64, fill: u8) !void {
    const size = try shapeProduct(shape) * @as(u64, if (std.mem.eql(u8, dtype, "BF16")) 2 else if (std.mem.eql(u8, dtype, "F32")) 4 else 1);
    const raw = try a.alloc(u8, @intCast(size));
    @memset(raw, fill);
    if (std.mem.eql(u8, dtype, "BF16")) {
        var i: usize = 0;
        while (i < raw.len) : (i += 2) std.mem.writeInt(u16, raw[i..][0..2], 0x3c00, .little);
    } else if (std.mem.eql(u8, dtype, "F32")) {
        var i: usize = 0;
        while (i < raw.len) : (i += 4) std.mem.writeInt(u32, raw[i..][0..4], @bitCast(@as(f32, 0.01)), .little);
    }
    try list.append(a, .{ .key = try a.dupe(u8, key), .dtype = dtype, .shape = try a.dupe(u64, shape), .bytes = raw });
}

fn writeNativeTestSource(io: std.Io, a: Allocator, dir: std.Io.Dir) !void {
    const c = nativeTestConfig();
    var ts: std.ArrayList(TestTensor) = .empty;
    for ([_][]const u8{ "model.embed_tokens.weight", "lm_head.weight" }) |key|
        try addNativeTestTensor(a, &ts, key, "BF16", &.{ c.vocab_size, c.hidden_size }, 0);
    try addNativeTestTensor(a, &ts, "model.norm.weight", "BF16", &.{c.hidden_size}, 0);
    for (0..c.num_hidden_layers) |i| {
        const li: u32 = @intCast(i);
        const pre = try std.fmt.allocPrint(a, "model.layers.{d}", .{li});
        const geometry = qkvGeometry(&c, li);
        try addNativeTestTensor(a, &ts, try std.fmt.allocPrint(a, "{s}.self_attn.qkv_proj.weight", .{pre}), "F8_E4M3", &.{ geometry.total(), c.hidden_size }, 0x20);
        try addNativeTestTensor(a, &ts, try std.fmt.allocPrint(a, "{s}.self_attn.qkv_proj.weight_scale_inv", .{pre}), "F32", &.{ if (li == 0) 108 else 116, 1 }, 0);
        try addNativeTestTensor(a, &ts, try std.fmt.allocPrint(a, "{s}.self_attn.o_proj.weight", .{pre}), "BF16", &.{ c.hidden_size, 64 * 128 }, 0);
        for ([_][]const u8{ "input_layernorm.weight", "post_attention_layernorm.weight" }) |name|
            try addNativeTestTensor(a, &ts, try std.fmt.allocPrint(a, "{s}.{s}", .{ pre, name }), "BF16", &.{c.hidden_size}, 0);
        if (li == 1) {
            try addNativeTestTensor(a, &ts, try std.fmt.allocPrint(a, "{s}.self_attn.attention_sink_bias", .{pre}), "BF16", &.{64}, 0);
            try addNativeTestTensor(a, &ts, try std.fmt.allocPrint(a, "{s}.mlp.gate.weight", .{pre}), "BF16", &.{ 2, c.hidden_size }, 0);
            try addNativeTestTensor(a, &ts, try std.fmt.allocPrint(a, "{s}.mlp.gate.e_score_correction_bias", .{pre}), "F32", &.{2}, 0);
        }
        for ([_][]const u8{ "gate", "up", "down" }) |proj| {
            if (li == 0) {
                try addNativeTestTensor(a, &ts, try std.fmt.allocPrint(a, "{s}.mlp.{s}_proj.weight", .{ pre, proj }), "F8_E4M3", &.{ 128, 128 }, 0x20);
                try addNativeTestTensor(a, &ts, try std.fmt.allocPrint(a, "{s}.mlp.{s}_proj.weight_scale_inv", .{ pre, proj }), "F32", &.{ 1, 1 }, 0);
            } else for (0..2) |e| {
                try addNativeTestTensor(a, &ts, try std.fmt.allocPrint(a, "{s}.mlp.experts.{d}.{s}_proj.weight", .{ pre, e, proj }), "U8", &.{ 128, 64 }, @intCast(0x12 + e));
                try addNativeTestTensor(a, &ts, try std.fmt.allocPrint(a, "{s}.mlp.experts.{d}.{s}_proj.weight_scale", .{ pre, e, proj }), "U8", &.{ 128, 4 }, 120);
            }
        }
    }
    try writeTestShard(io, a, dir, "source.safetensors", ts.items);
    var index: std.ArrayList(u8) = .empty;
    try index.appendSlice(a, "{\"weight_map\":{");
    for (ts.items, 0..) |t, i| try appendTestFormat(a, &index, "{s}\"{s}\":\"source.safetensors\"", .{ if (i == 0) "" else ",", t.key });
    try index.appendSlice(a, "}}");
    try dir.writeFile(io, .{ .sub_path = "model.safetensors.index.json", .data = index.items });
}

test "mimo source packs resident MXFP4 banks without changing bytes and runs mixed-precision forwards" {
    const t = std.testing;
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    var tmp = t.tmpDir(.{});
    defer tmp.cleanup();
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    try writeNativeTestSource(t.io, arena.allocator(), tmp.dir);
    var path_buf: [std.fs.max_path_bytes]u8 = undefined;
    const path_len = try tmp.dir.realPath(t.io, &path_buf);
    const path = path_buf[0..path_len];
    const c = nativeTestConfig();
    const billed = try residentBytes(t.io, t.allocator, path, &c);
    var weights = try loadWeights(t.io, t.allocator, path, &c);
    defer weights.deinit();
    const bank = weights.get("model.layers.1.mlp.switch_mlp.gate_proj.weight").?;
    try t.expectEqualSlices(c_int, &.{ 2, 128, 16 }, mlx.getShape(bank));
    try mlx.check(mlx.mlx_array_eval(bank));
    const words = mlx.mlx_array_data_uint32(bank).?;
    try t.expectEqual(@as(u32, 0x12121212), words[0]);
    try t.expectEqual(@as(u32, 0x13131313), words[128 * 16]);
    var actual: u64 = 0;
    var it = weights.map.valueIterator();
    while (it.next()) |w| actual += mlx.mlx_array_size(w.*) * mlx.mlx_array_itemsize(w.*);
    try t.expectEqual(actual + 2 * 128 * 4, billed);
    const xfm_mod = @import("transformer.zig");
    var xfm = try xfm_mod.Transformer.init(t.io, t.allocator, c, &weights);
    defer xfm.deinit();
    const ids = [_]i32{ 1, 2, 3, 4, 5, 6 };
    const tokens = mlx.mlx_array_new_data(@ptrCast(&ids), &[_]c_int{ 1, 6 }, 2, .int32);
    defer _ = mlx.mlx_array_free(tokens);
    const logits = try xfm.forward(tokens);
    defer _ = mlx.mlx_array_free(logits);
    try mlx.check(mlx.mlx_array_eval(logits));
    try t.expectEqualSlices(c_int, &.{ 1, 6, 16 }, mlx.getShape(logits));
}

test "mimo source validates the original checkpoint without loading weights (MIMO_V2_SOURCE)" {
    const raw = std.c.getenv("MIMO_V2_SOURCE") orelse return error.SkipZigTest;
    const c = try model.parseConfig(std.testing.io, std.testing.allocator, std.mem.span(raw));
    const bytes = try residentBytes(std.testing.io, std.testing.allocator, std.mem.span(raw), &c);
    try std.testing.expect(bytes > 128 * 1024 * 1024 * 1024 and bytes < 192 * 1024 * 1024 * 1024);
}
