//! Minimal GGUF header reader — just enough to decide which embedded engine
//! should serve a given `.gguf` file. We don't materialize tensors or full
//! metadata, only walk the KV pairs at the top of the file looking for two
//! keys:
//!
//!   - `general.architecture` (string) — what arch the file claims to be
//!   - `deepseek4.attention.output_lora_rank` (any numeric) — present only on
//!     the antirez/ds4-style DSV4-Flash GGUFs that the embedded ds4 engine
//!     can actually load (vanilla llama.cpp deepseek4 quants omit it)
//!
//! Routing rule (`preferredEngine`):
//!   a split GGUF (`split.count` > 1) → llama.cpp, the only engine here that
//!     reads the sibling shards (ds4 reads one file)
//!   `general.architecture == "deepseek4"` AND lora-rank key present → ds4
//!   a ds4-only arch (`deepseek41`, `qwen4exp`, `glm-dsa`, `glm5-next`) → ds4,
//!     except a qwen4exp file with tensor types ds4 rejects → llama.cpp
//!   otherwise → llama.cpp
//!
//! This replaces the basename-only heuristic in `model_discovery.zig`, which
//! mis-routed both real-world cases reported in issue #15 (a vanilla
//! deepseek4 GGUF whose name started with `deepseek-v4-flash` went to ds4
//! and crashed on missing metadata; a true ds4 GGUF whose name started with
//! `Huihui-` went to llama.cpp and crashed on unknown arch).
//!
//! Why we don't reuse libllama's metadata API: `llama_model_load_from_file`
//! rejects unknown architectures BEFORE metadata becomes readable, which is
//! the exact path we're trying to avoid taking by mistake. ds4's C API also
//! offers no metadata-peek. A purpose-built reader keeps the routing
//! decision cheap and side-effect free.

const std = @import("std");
const GgmlType = @import("quants.zig").GgmlType;

pub const Engine = enum { ds4, llama };

/// Routing decision for a GGUF entry resolved at preload time: the actual `.gguf` file (owned by the connection
/// thread, freed after the load completes; the engines copy what they keep) and which embedded engine serves it.
pub const Route = struct {
    path: []const u8,
    engine: Engine,
    disk_only_bytes: u64 = 0,
};

pub fn ds4DiskOnlyBytes(io: std.Io, allocator: std.mem.Allocator, path: []const u8) u64 {
    var info = readFromFile(io, allocator, path) catch return 0;
    defer info.deinit(allocator);
    return info.ds4_disk_only_bytes;
}

pub const Info = struct {
    /// Owned dupe of `general.architecture` value; null if the key was
    /// absent or not a string.
    architecture: ?[]u8 = null,
    has_ds4_lora_rank: bool = false,
    /// `<arch>.nextn_predict_layers` > 0: the checkpoint carries its own MTP
    /// head (ds4 arms it via `glm_mtp`; asking on a model without one refuses the open).
    embedded_mtp: bool = false,
    /// qwen4exp only: a tensor type ds4 rejects, or a non-BF16 n-gram table
    /// (llama.cpp quants such as GSQ-RCO's Q2_0 experts). ds4 would exit the process.
    ds4_unloadable: bool = false,
    /// `split.count`: shards of a split GGUF (gguf-split / convert --split-max-size).
    /// Writers put it after every other key, so the whole KV section is walked.
    split_count: u32 = 1,
    /// `<arch>.context_length`: the window the checkpoint was trained for; 0 when absent.
    context_length: u32 = 0,
    /// Validated tensor bytes in the disk-only tail that ds4 unmaps.
    ds4_disk_only_bytes: u64 = 0,

    pub fn deinit(self: *Info, allocator: std.mem.Allocator) void {
        if (self.architecture) |a| allocator.free(a);
        self.* = .{};
    }
};

// Arch names the ds4 converters write. llama.cpp loads qwen4exp too, so a
// qwen4exp file ds4 cannot load goes there instead.
const ds4_only_archs = [_][]const u8{ "deepseek41", "qwen4exp", "glm-dsa", "glm5-next" };

// Archs whose NextN layers ds4 drafts from inside the checkpoint; DeepSeek-V4 declares
// `nextn_predict_layers` but takes its head from a separate GGUF, and ds4 refuses to open it with one armed.
const ds4_embedded_mtp_archs = [_][]const u8{ "glm-dsa", "glm5-next", "qwen4exp" };

// GGML types ds4 accepts for Qwen tensors: F32 F16 Q4_0 Q8_0 Q2_K Q4_K Q5_K Q6_K IQ2_XXS BF16 MXFP4.
const ds4_qwen_tensor_types = [_]u32{ 0, 1, 2, 8, 10, 12, 13, 14, 16, 30, 39 };
const GGML_TYPE_BF16: u32 = 30;
const GGML_MAX_DIMS: u32 = 4;

pub fn preferredEngine(info: Info) Engine {
    if (info.split_count > 1) return .llama;
    if (info.architecture) |a| {
        if (std.mem.eql(u8, a, "deepseek4") and info.has_ds4_lora_rank) return .ds4;
        for (ds4_only_archs) |d| if (std.mem.eql(u8, a, d)) return if (info.ds4_unloadable) .llama else .ds4;
    }
    return .llama;
}

/// What ds4 drafts from for a file: the checkpoint's own head and/or a support GGUF beside it.
pub const Spec = struct { embedded_mtp: bool = false, support_gguf: bool = true };

/// ds4 fails the open of a DeepSeek-V4.1 handed any support GGUF (MTP or DSpark), so one beside it is skipped.
pub fn ds4Spec(info: Info) Spec {
    const a = info.architecture orelse "";
    return .{ .embedded_mtp = info.embedded_mtp, .support_gguf = !std.mem.eql(u8, a, "deepseek41") };
}

pub const Error = error{
    BadMagic,
    UnsupportedVersion,
    UnsupportedType,
    Truncated,
    KeyTooLong,
} || std.mem.Allocator.Error;

// GGUF value types — kept private to this file.
const TY_U8: u32 = 0;
const TY_I8: u32 = 1;
const TY_U16: u32 = 2;
const TY_I16: u32 = 3;
const TY_U32: u32 = 4;
const TY_I32: u32 = 5;
const TY_F32: u32 = 6;
const TY_BOOL: u32 = 7;
const TY_STRING: u32 = 8;
const TY_ARRAY: u32 = 9;
const TY_U64: u32 = 10;
const TY_I64: u32 = 11;
const TY_F64: u32 = 12;

// Defensive caps. A key over 1 KiB or a value-string over 16 MiB indicates
// a malformed file (or a malicious one); GGUFs in the wild have keys under
// 100 bytes and string values under a few KiB. Vocab arrays go through a
// separate path that doesn't materialize anything.
const MAX_KEY_LEN: u64 = 1024;
const MAX_STR_VALUE_LEN: u64 = 16 * 1024 * 1024;

/// Stream-parse metadata; disk-only accounting also requires a known file size.
pub fn parseInfo(allocator: std.mem.Allocator, reader: *std.Io.Reader) Error!Info {
    return parseInfoSized(allocator, reader, null);
}

fn parseInfoSized(allocator: std.mem.Allocator, reader: *std.Io.Reader, file_size: ?u64) Error!Info {
    var buffer: [16 * 1024]u8 = undefined;
    const limit = @min(file_size orelse std.math.maxInt(usize) - 1, std.math.maxInt(usize) - 1);
    var counted = reader.limited(.limited64(limit), &buffer);
    const r = &counted.interface;
    var alignment: u64 = 32;
    var engram_encoding: ?bool = null;
    var unique_arch = true;
    var seen_arch = false;
    var info: Info = .{};
    errdefer info.deinit(allocator);

    // Magic + version + counts.
    const magic = takeBytes(r, 4) catch return error.Truncated;
    if (!std.mem.eql(u8, magic, "GGUF")) return error.BadMagic;
    const version = takeIntT(r, u32) catch return error.Truncated;
    if (version < 2 or version > 3) return error.UnsupportedVersion;
    const tensor_count = takeIntT(r, u64) catch return error.Truncated;
    const kv_count = takeIntT(r, u64) catch return error.Truncated;

    var i: u64 = 0;
    while (i < kv_count) : (i += 1) {
        // Key.
        const key_len = takeIntT(r, u64) catch return error.Truncated;
        if (key_len > MAX_KEY_LEN) return error.KeyTooLong;
        const key_buf = takeBytes(r, @intCast(key_len)) catch return error.Truncated;

        // We need an owned copy if we're matching against it AND continuing
        // to read, because the next take call invalidates the buffer slice.
        // Match first, then advance.
        const is_arch = std.mem.eql(u8, key_buf, "general.architecture");
        if (is_arch) {
            if (seen_arch) unique_arch = false;
            seen_arch = true;
        }
        const is_ds4_lora = std.mem.eql(u8, key_buf, "deepseek4.attention.output_lora_rank");
        const is_nextn = std.mem.endsWith(u8, key_buf, ".nextn_predict_layers");
        const is_split_count = std.mem.eql(u8, key_buf, "split.count");
        const is_alignment = std.mem.eql(u8, key_buf, "general.alignment");
        const is_encoding = std.mem.eql(u8, key_buf, "deepseek41.engram.encoding") and engram_encoding == null;
        if (is_encoding) engram_encoding = false;
        // One dot: `<arch>.context_length`, never a sub-namespace's (`<arch>.vision.context_length`).
        const is_ctx_len = std.mem.endsWith(u8, key_buf, ".context_length") and
            std.mem.indexOfScalar(u8, key_buf, '.').? == key_buf.len - ".context_length".len;

        const value_type = takeIntT(r, u32) catch return error.Truncated;

        if ((is_arch or is_encoding) and value_type == TY_STRING) {
            const v_len = takeIntT(r, u64) catch return error.Truncated;
            if (v_len > MAX_STR_VALUE_LEN) return error.Truncated;
            const v_buf = takeBytes(r, @intCast(v_len)) catch return error.Truncated;
            if (is_arch) {
                const owned = try allocator.dupe(u8, v_buf);
                if (info.architecture) |old| allocator.free(old);
                info.architecture = owned;
            } else engram_encoding = std.mem.eql(u8, v_buf, "e4m3_e8m0_32_row264");
        } else if (is_alignment and value_type == TY_U32) {
            const n = takeIntT(r, u32) catch return error.Truncated;
            if (n != 0) alignment = n;
        } else if (is_ds4_lora and isNumeric(value_type)) {
            // Presence is what matters; the actual value (rank) isn't used
            // for routing. Skip past the value cleanly.
            try skipValue(r, value_type);
            info.has_ds4_lora_rank = true;
        } else if (is_nextn and isNumeric(value_type)) {
            info.embedded_mtp = (try takeNumeric(r, value_type)) > 0;
        } else if (is_split_count and isNumeric(value_type)) {
            info.split_count = @intFromFloat(std.math.clamp(try takeNumeric(r, value_type), 1, std.math.maxInt(u32)));
        } else if (is_ctx_len and isNumeric(value_type)) {
            info.context_length = @intFromFloat(std.math.clamp(try takeNumeric(r, value_type), 0, std.math.maxInt(u32)));
        } else {
            try skipValue(r, value_type);
        }
    }

    if (info.architecture) |a| {
        const engram = std.mem.eql(u8, a, "deepseek41");
        if (engram or std.mem.eql(u8, a, "qwen4exp")) {
            const tensors = try scanDs4Tensors(r, tensor_count, engram);
            info.ds4_unloadable = tensors.unloadable;
            info.ds4_disk_only_bytes = bill: {
                if (version != 3 or !unique_arch or info.split_count != 1 or (engram and !(engram_encoding orelse false))) break :bill 0;
                const size = file_size orelse break :bill 0;
                const relative = tensors.start orelse break :bill 0;
                const header_end = limit - counted.remaining.toInt64().? - (r.end - r.seek);
                const data_start = std.math.add(u64, header_end, (alignment - header_end % alignment) % alignment) catch break :bill 0;
                const start = std.math.add(u64, data_start, relative) catch break :bill 0;
                if (start % std.heap.pageSize() != 0 or data_start > size or tensors.end > size - data_start) break :bill 0;
                break :bill tensors.disk_only_bytes;
            };
        }
    }
    if (info.embedded_mtp) info.embedded_mtp = for (ds4_embedded_mtp_archs) |d| {
        if (info.architecture != null and std.mem.eql(u8, info.architecture.?, d)) break true;
    } else false;
    return info;
}

const Ds4Tensors = struct { unloadable: bool = false, disk_only_bytes: u64 = 0, start: ?u64 = null, end: u64 = 0 };

// Match the pinned ds4 tensor_type table where its geometry differs from standard ggml.
fn ds4TensorBytes(ty: u32, elements: u64) ?u64 {
    const geometry: struct { u64, u64 } = switch (ty) {
        9 => .{ 32, 40 },
        15 => .{ 256, 292 },
        19 => .{ 256, 110 },
        20 => .{ 256, 50 },
        24 => .{ 1, 1 },
        25 => .{ 1, 2 },
        26 => .{ 1, 4 },
        27, 28 => .{ 1, 8 },
        42 => return null,
        else => blk: {
            const quant: GgmlType = @fromBackingInt(@intCast(ty));
            if (!quant.supported()) return null;
            break :blk .{ quant.blockElems(), quant.blockBytes() };
        },
    };
    const blocks = std.math.divCeil(u64, elements, geometry[0]) catch return null;
    return std.math.mul(u64, blocks, geometry[1]) catch null;
}

/// The same shape and tail ordering required by ds4's model_unmap_engram/model_unmap_qwen_ngrams.
fn scanDs4Tensors(r: *std.Io.Reader, tensor_count: u64, engram: bool) Error!Ds4Tensors {
    const Table = struct { offset: u64, bytes: u64 };
    var tables: [2]?Table = .{ null, null };
    var resident_end: u64 = 0;
    var valid = true;
    var out: Ds4Tensors = .{};
    var i: u64 = 0;
    while (i < tensor_count) : (i += 1) {
        const name_len = takeIntT(r, u64) catch return error.Truncated;
        if (name_len > MAX_KEY_LEN) return error.KeyTooLong;
        const name = takeBytes(r, @intCast(name_len)) catch return error.Truncated;
        const index: ?usize = if (engram)
            (if (std.mem.eql(u8, name, "blk.1.engram_embd.weight")) 0 else if (std.mem.eql(u8, name, "blk.14.engram_embd.weight")) 1 else null)
        else if (std.mem.eql(u8, name, "per_layer_token_embd.weight")) 0 else null;
        const n_dims = takeIntT(r, u32) catch return error.Truncated;
        if (n_dims > GGML_MAX_DIMS) return error.Truncated;
        if (n_dims == 0) valid = false;
        var dims: [GGML_MAX_DIMS]u64 = @splat(0);
        var elements: u64 = 1;
        for (dims[0..n_dims]) |*dim| {
            dim.* = takeIntT(r, u64) catch return error.Truncated;
            elements = std.math.mul(u64, elements, dim.*) catch return error.Truncated;
        }
        const ty = takeIntT(r, u32) catch return error.Truncated;
        const offset = takeIntT(r, u64) catch return error.Truncated;
        if (!engram and (std.mem.indexOfScalar(u32, &ds4_qwen_tensor_types, ty) == null or (index != null and ty != GGML_TYPE_BF16))) {
            out.unloadable = true;
            return out;
        }
        const bytes = ds4TensorBytes(ty, elements) orelse {
            valid = false;
            continue;
        };
        const end = std.math.add(u64, offset, bytes) catch {
            valid = false;
            continue;
        };
        out.end = @max(out.end, end);
        if (index) |k| {
            if (tables[k] != null or n_dims != 2 or ty != @as(u32, if (engram) 24 else GGML_TYPE_BF16) or
                dims[0] == 0 or (if (engram) dims[0] != 264 else dims[0] > 160) or dims[1] == 0 or dims[1] > std.math.maxInt(u32)) valid = false;
            tables[k] = .{ .offset = offset, .bytes = bytes };
        } else {
            if (bytes == 0) valid = false;
            resident_end = @max(resident_end, end);
        }
    }
    const first = tables[0] orelse return out;
    if (!valid or resident_end > first.offset) return out;
    var bytes = first.bytes;
    if (engram) {
        const second = tables[1] orelse return out;
        if (second.offset < first.offset + first.bytes) return out;
        bytes = std.math.add(u64, bytes, second.bytes) catch return out;
    }
    out.start = first.offset;
    out.disk_only_bytes = bytes;
    return out;
}

/// True when the header names a pooling type (`<arch>.pooling_type` > 0), llama.cpp's
/// mark of an embedding or reranker model: no engine here generates text from one.
/// Stops at the first `tokenizer.` key; converters write the arch keys before it.
pub fn declaresPooling(r: *std.Io.Reader) Error!bool {
    const magic = takeBytes(r, 4) catch return error.Truncated;
    if (!std.mem.eql(u8, magic, "GGUF")) return error.BadMagic;
    const version = takeIntT(r, u32) catch return error.Truncated;
    if (version < 2 or version > 3) return error.UnsupportedVersion;
    _ = takeIntT(r, u64) catch return error.Truncated; // tensor_count
    const kv_count = takeIntT(r, u64) catch return error.Truncated;
    var i: u64 = 0;
    while (i < kv_count) : (i += 1) {
        const key_len = takeIntT(r, u64) catch return error.Truncated;
        if (key_len > MAX_KEY_LEN) return error.KeyTooLong;
        const key = takeBytes(r, @intCast(key_len)) catch return error.Truncated;
        if (std.mem.startsWith(u8, key, "tokenizer.")) return false;
        const is_pooling = std.mem.endsWith(u8, key, ".pooling_type");
        const value_type = takeIntT(r, u32) catch return error.Truncated;
        if (is_pooling and isNumeric(value_type)) return (try takeNumeric(r, value_type)) > 0;
        try skipValue(r, value_type);
    }
    return false;
}

/// `declaresPooling` for `sub_path` under `dir`; false when it cannot be read.
pub fn fileDeclaresPooling(io: std.Io, dir: std.Io.Dir, sub_path: []const u8) bool {
    var file = dir.openFile(io, sub_path, .{}) catch return false;
    defer file.close(io);
    var rbuf: [16 * 1024]u8 = undefined;
    var rs = file.reader(io, &rbuf);
    return declaresPooling(&rs.interface) catch false;
}

/// Open the GGUF file and parse Info via the file's buffered reader.
/// Caller owns the returned Info (call `.deinit(allocator)`).
pub fn readFromFile(io: std.Io, allocator: std.mem.Allocator, path: []const u8) !Info {
    var file = try std.Io.Dir.openFileAbsolute(io, path, .{});
    defer file.close(io);
    var rbuf: [16 * 1024]u8 = undefined;
    var rs = file.reader(io, &rbuf);
    return parseInfoSized(allocator, &rs.interface, (try file.stat(io)).size);
}

// ── private ──

fn isNumeric(ty: u32) bool {
    return switch (ty) {
        TY_U8, TY_I8, TY_U16, TY_I16, TY_U32, TY_I32, TY_F32, TY_U64, TY_I64, TY_F64 => true,
        else => false,
    };
}

fn fixedTypeSize(ty: u32) ?u64 {
    return switch (ty) {
        TY_U8, TY_I8, TY_BOOL => 1,
        TY_U16, TY_I16 => 2,
        TY_U32, TY_I32, TY_F32 => 4,
        TY_U64, TY_I64, TY_F64 => 8,
        else => null,
    };
}

/// Integer value of a numeric KV as f64 (floats pass through; the callers
/// only compare against zero).
fn takeNumeric(r: *std.Io.Reader, ty: u32) Error!f64 {
    return switch (ty) {
        TY_U8 => @floatFromInt(takeIntT(r, u8) catch return error.Truncated),
        TY_I8 => @floatFromInt(takeIntT(r, i8) catch return error.Truncated),
        TY_U16 => @floatFromInt(takeIntT(r, u16) catch return error.Truncated),
        TY_I16 => @floatFromInt(takeIntT(r, i16) catch return error.Truncated),
        TY_U32 => @floatFromInt(takeIntT(r, u32) catch return error.Truncated),
        TY_I32 => @floatFromInt(takeIntT(r, i32) catch return error.Truncated),
        TY_U64 => @floatFromInt(takeIntT(r, u64) catch return error.Truncated),
        TY_I64 => @floatFromInt(takeIntT(r, i64) catch return error.Truncated),
        TY_F32 => @as(f32, @bitCast(takeIntT(r, u32) catch return error.Truncated)),
        TY_F64 => @as(f64, @bitCast(takeIntT(r, u64) catch return error.Truncated)),
        else => error.UnsupportedType,
    };
}

fn skipValue(r: *std.Io.Reader, ty: u32) Error!void {
    if (fixedTypeSize(ty)) |n| {
        r.discardAll64(n) catch return error.Truncated;
        return;
    }
    switch (ty) {
        TY_STRING => {
            const n = takeIntT(r, u64) catch return error.Truncated;
            r.discardAll64(n) catch return error.Truncated;
        },
        TY_ARRAY => {
            const inner_ty = takeIntT(r, u32) catch return error.Truncated;
            const inner_count = takeIntT(r, u64) catch return error.Truncated;
            if (fixedTypeSize(inner_ty)) |sz| {
                // count * sz could overflow u64 in theory; fail-safe.
                const total = std.math.mul(u64, inner_count, sz) catch return error.Truncated;
                r.discardAll64(total) catch return error.Truncated;
            } else if (inner_ty == TY_STRING) {
                var k: u64 = 0;
                while (k < inner_count) : (k += 1) {
                    const sl = takeIntT(r, u64) catch return error.Truncated;
                    r.discardAll64(sl) catch return error.Truncated;
                }
            } else if (inner_ty == TY_ARRAY) {
                var k: u64 = 0;
                while (k < inner_count) : (k += 1) try skipValue(r, TY_ARRAY);
            } else {
                return error.UnsupportedType;
            }
        },
        else => return error.UnsupportedType,
    }
}

// Reader-method wrappers that translate the std error set into our Error.
// `takeArray(n)` would want a comptime n; we use `take(n)` for runtime n
// then convert to a slice.
fn takeBytes(r: *std.Io.Reader, n: usize) ![]const u8 {
    return r.take(n);
}

fn takeIntT(r: *std.Io.Reader, comptime T: type) !T {
    return r.takeInt(T, .little);
}

// ── tests ──

const testing = std.testing;

pub const Value = union(enum) {
    str: []const u8,
    u32_v: u32,
    u64_v: u64,
    str_array: []const []const u8,
};

pub const KV = struct { key: []const u8, value: Value };

fn appendU32(buf: *std.ArrayList(u8), allocator: std.mem.Allocator, v: u32) !void {
    var tmp: [4]u8 = undefined;
    std.mem.writeInt(u32, &tmp, v, .little);
    try buf.appendSlice(allocator, &tmp);
}

fn appendU64(buf: *std.ArrayList(u8), allocator: std.mem.Allocator, v: u64) !void {
    var tmp: [8]u8 = undefined;
    std.mem.writeInt(u64, &tmp, v, .little);
    try buf.appendSlice(allocator, &tmp);
}

fn appendStr(buf: *std.ArrayList(u8), allocator: std.mem.Allocator, s: []const u8) !void {
    try appendU64(buf, allocator, s.len);
    try buf.appendSlice(allocator, s);
}

pub const TensorInfo = struct { name: []const u8, ggml_type: u32, dims: []const u64 = &.{ 640, 2560 }, offset: u64 = 0 };

/// Build a minimal but valid GGUF v3 header from a KV list, with no tensors.
pub fn buildHeader(allocator: std.mem.Allocator, kvs: []const KV) ![]u8 {
    return buildHeaderWithTensors(allocator, kvs, &.{});
}

/// GGUF v3 header followed by tensor infos; no tensor data.
pub fn buildHeaderWithTensors(allocator: std.mem.Allocator, kvs: []const KV, tensors: []const TensorInfo) ![]u8 {
    var buf: std.ArrayList(u8) = .empty;
    errdefer buf.deinit(allocator);
    try buf.appendSlice(allocator, "GGUF");
    try appendU32(&buf, allocator, 3); // version
    try appendU64(&buf, allocator, @intCast(tensors.len)); // tensor_count
    try appendU64(&buf, allocator, @intCast(kvs.len)); // kv_count
    for (kvs) |kv| {
        try appendStr(&buf, allocator, kv.key);
        switch (kv.value) {
            .str => |s| {
                try appendU32(&buf, allocator, TY_STRING);
                try appendStr(&buf, allocator, s);
            },
            .u32_v => |v| {
                try appendU32(&buf, allocator, TY_U32);
                try appendU32(&buf, allocator, v);
            },
            .u64_v => |v| {
                try appendU32(&buf, allocator, TY_U64);
                try appendU64(&buf, allocator, v);
            },
            .str_array => |arr| {
                try appendU32(&buf, allocator, TY_ARRAY);
                try appendU32(&buf, allocator, TY_STRING);
                try appendU64(&buf, allocator, arr.len);
                for (arr) |s| try appendStr(&buf, allocator, s);
            },
        }
    }
    for (tensors) |t| {
        try appendStr(&buf, allocator, t.name);
        try appendU32(&buf, allocator, @intCast(t.dims.len));
        for (t.dims) |dim| try appendU64(&buf, allocator, dim);
        try appendU32(&buf, allocator, t.ggml_type);
        try appendU64(&buf, allocator, t.offset);
    }
    return buf.toOwnedSlice(allocator);
}

fn parseBytes(allocator: std.mem.Allocator, bytes: []const u8) Error!Info {
    var r = std.Io.Reader.fixed(bytes);
    return parseInfoSized(allocator, &r, bytes.len);
}

test "preferredEngine: llama arch → llama" {
    const bytes = try buildHeader(testing.allocator, &.{
        .{ .key = "general.architecture", .value = .{ .str = "llama" } },
    });
    defer testing.allocator.free(bytes);
    var info = try parseBytes(testing.allocator, bytes);
    defer info.deinit(testing.allocator);
    try testing.expectEqualStrings("llama", info.architecture.?);
    try testing.expect(!info.has_ds4_lora_rank);
    try testing.expectEqual(Engine.llama, preferredEngine(info));
}

test "embedded MTP is read from <arch>.nextn_predict_layers" {
    const with = try buildHeader(testing.allocator, &.{
        .{ .key = "general.architecture", .value = .{ .str = "qwen4exp" } },
        .{ .key = "qwen4exp.nextn_predict_layers", .value = .{ .u32_v = 1 } },
    });
    defer testing.allocator.free(with);
    var a = try parseBytes(testing.allocator, with);
    defer a.deinit(testing.allocator);
    try testing.expect(a.embedded_mtp);

    const zero = try buildHeader(testing.allocator, &.{
        .{ .key = "general.architecture", .value = .{ .str = "qwen4exp" } },
        .{ .key = "qwen4exp.nextn_predict_layers", .value = .{ .u32_v = 0 } },
    });
    defer testing.allocator.free(zero);
    var b = try parseBytes(testing.allocator, zero);
    defer b.deinit(testing.allocator);
    try testing.expect(!b.embedded_mtp);

    // DeepSeek-V4 declares its NextN layer but ships the head as a separate GGUF.
    const dsv4 = try buildHeader(testing.allocator, &.{
        .{ .key = "general.architecture", .value = .{ .str = "deepseek4" } },
        .{ .key = "deepseek4.nextn_predict_layers", .value = .{ .u32_v = 1 } },
    });
    defer testing.allocator.free(dsv4);
    var c = try parseBytes(testing.allocator, dsv4);
    defer c.deinit(testing.allocator);
    try testing.expect(!c.embedded_mtp);
}

test "the trained window is read from <arch>.context_length, not from a sub-namespace's key" {
    const bytes = try buildHeader(testing.allocator, &.{
        .{ .key = "general.architecture", .value = .{ .str = "glm-dsa" } },
        .{ .key = "glm-dsa.vision.context_length", .value = .{ .u32_v = 4096 } },
        .{ .key = "glm-dsa.context_length", .value = .{ .u64_v = 1048576 } },
    });
    defer testing.allocator.free(bytes);
    var info = try parseBytes(testing.allocator, bytes);
    defer info.deinit(testing.allocator);
    try testing.expectEqual(@as(u32, 1048576), info.context_length);

    const bare = try buildHeader(testing.allocator, &.{.{ .key = "general.architecture", .value = .{ .str = "glm-dsa" } }});
    defer testing.allocator.free(bare);
    var none = try parseBytes(testing.allocator, bare);
    defer none.deinit(testing.allocator);
    try testing.expectEqual(@as(u32, 0), none.context_length);
}

test "ds4Spec: DeepSeek-V4.1 takes no support GGUF in ds4; DeepSeek-V4 and the other archs do" {
    for ([_][]const u8{ "deepseek41", "deepseek4", "qwen4exp" }) |arch| {
        const bytes = try buildHeader(testing.allocator, &.{.{ .key = "general.architecture", .value = .{ .str = arch } }});
        defer testing.allocator.free(bytes);
        var info = try parseBytes(testing.allocator, bytes);
        defer info.deinit(testing.allocator);
        try testing.expectEqual(!std.mem.eql(u8, arch, "deepseek41"), ds4Spec(info).support_gguf);
    }
}

test "preferredEngine: ds4-only archs → ds4 without the lora key" {
    // The ds4 converters write these arch names; llama.cpp cannot load them.
    for ([_][]const u8{ "qwen4exp", "deepseek41", "glm-dsa", "glm5-next" }) |arch| {
        const bytes = try buildHeader(testing.allocator, &.{
            .{ .key = "general.architecture", .value = .{ .str = arch } },
        });
        defer testing.allocator.free(bytes);
        var info = try parseBytes(testing.allocator, bytes);
        defer info.deinit(testing.allocator);
        try testing.expectEqual(Engine.ds4, preferredEngine(info));
    }
}

test "preferredEngine: qwen4exp routes by tensor types ds4 can load" {
    const arch: []const KV = &.{.{ .key = "general.architecture", .value = .{ .str = "qwen4exp" } }};
    // ds4-converted: BF16 n-gram table, experts in a ds4 type.
    const ds4_file = try buildHeaderWithTensors(testing.allocator, arch, &.{
        .{ .name = "blk.0.ffn_down_exps.weight", .ggml_type = 12 }, // Q4_K
        .{ .name = "per_layer_token_embd.weight", .ggml_type = 30 }, // BF16
    });
    defer testing.allocator.free(ds4_file);
    // llama.cpp quant (e.g. GSQ-RCO IQ3_XXS): Q2_0 experts.
    const q2_0_file = try buildHeaderWithTensors(testing.allocator, arch, &.{
        .{ .name = "blk.0.ffn_down_exps.weight", .ggml_type = 42 }, // Q2_0
    });
    defer testing.allocator.free(q2_0_file);
    // llama.cpp quant with a quantized n-gram table.
    const iq4nl_ngrams = try buildHeaderWithTensors(testing.allocator, arch, &.{
        .{ .name = "blk.0.ffn_down_exps.weight", .ggml_type = 8 }, // Q8_0
        .{ .name = "per_layer_token_embd.weight", .ggml_type = 20 }, // IQ4_NL
    });
    defer testing.allocator.free(iq4nl_ngrams);

    const cases = [_]struct { bytes: []const u8, want: Engine }{
        .{ .bytes = ds4_file, .want = .ds4 },
        .{ .bytes = q2_0_file, .want = .llama },
        .{ .bytes = iq4nl_ngrams, .want = .llama },
    };
    for (cases) |c| {
        var info = try parseBytes(testing.allocator, c.bytes);
        defer info.deinit(testing.allocator);
        try testing.expectEqual(c.want, preferredEngine(info));
    }
}

test "GGUF accounting: disk-only tails require the native layout guards" {
    const Change = enum { valid, wrong_arch, wrong_encoding, missing_encoding, wrong_type, wrong_width, wrong_rank, zero_rows, row_limit, duplicate, missing, overlap, unaligned, header_alignment, resident_after, truncated, header_only, no_size, offset_overflow, unsupported_resident, native_iq4, native_iq1_crosses };
    const page: u64 = std.heap.pageSize();
    for ([_]bool{ true, false }) |engram| {
        for (std.enums.values(Change)) |change| {
            if (!engram and (change == .wrong_encoding or change == .missing_encoding or change == .missing or change == .overlap or change == .native_iq4 or change == .native_iq1_crosses)) continue;
            errdefer std.debug.print("Engram={}: {s}\n", .{ engram, @tagName(change) });
            var tensors = [_]TensorInfo{
                .{ .name = "output.weight", .ggml_type = 1, .dims = &.{256} },
                .{ .name = if (engram) "blk.1.engram_embd.weight" else "per_layer_token_embd.weight", .ggml_type = if (engram) 24 else 30, .dims = if (engram) &.{ 264, 16 } else &.{ 160, 64 }, .offset = page },
                .{ .name = "blk.14.engram_embd.weight", .ggml_type = 24, .dims = &.{ 264, 16 }, .offset = 2 * page },
            };
            var count: usize = if (engram) 3 else 2;
            switch (change) {
                .wrong_type => tensors[1].ggml_type = 1,
                .wrong_width => tensors[1].dims = &.{ 265, 16 },
                .wrong_rank => tensors[1].dims = if (engram) &.{ 264, 16, 1 } else &.{ 160, 16, 1 },
                .zero_rows => tensors[1].dims = if (engram) &.{ 264, 0 } else &.{ 160, 0 },
                .row_limit => tensors[1].dims = if (engram) &.{ 264, @as(u64, std.math.maxInt(u32)) + 1 } else &.{ 160, @as(u64, std.math.maxInt(u32)) + 1 },
                .duplicate => {
                    tensors[2] = tensors[1];
                    count = 3;
                },
                .missing => count = 2,
                .overlap => tensors[2].offset = page,
                .unaligned => tensors[1].offset += 1,
                .offset_overflow => tensors[1].offset = std.math.maxInt(u64),
                .resident_after => tensors[0].offset = 3 * page,
                .unsupported_resident => tensors[0].ggml_type = 42,
                .native_iq4 => {
                    tensors[0].ggml_type = 20;
                    tensors[0].offset = page - 50;
                },
                .native_iq1_crosses => {
                    tensors[0].ggml_type = 19;
                    tensors[0].offset = page - 50;
                },
                else => {},
            }
            const kvs = [_]KV{
                .{ .key = "general.architecture", .value = .{ .str = if (change == .wrong_arch) "glm-dsa" else if (engram) "deepseek41" else "qwen4exp" } },
                .{ .key = "general.alignment", .value = .{ .u32_v = @intCast(if (change == .header_alignment) 32 else page) } },
                .{ .key = "deepseek41.engram.encoding", .value = .{ .str = if (change == .wrong_encoding) "other" else "e4m3_e8m0_32_row264" } },
            };
            const header = try buildHeaderWithTensors(testing.allocator, kvs[0..@as(usize, if (change == .missing_encoding) 2 else 3)], tensors[0..count]);
            var bytes = std.ArrayList(u8).fromOwnedSlice(header);
            defer bytes.deinit(testing.allocator);
            if (change != .header_only) {
                try bytes.resize(testing.allocator, @intCast(if (change == .truncated) 2 * page else 8 * page));
                @memset(bytes.items[header.len..], 0);
            }
            var reader: std.Io.Reader = .fixed(bytes.items);
            var info = if (change == .no_size) try parseInfo(testing.allocator, &reader) else try parseBytes(testing.allocator, bytes.items);
            defer info.deinit(testing.allocator);
            const want: u64 = if (change == .valid or change == .native_iq4) (if (engram) 2 * 264 * 16 else 160 * 64 * 2) else 0;
            try testing.expectEqual(want, info.ds4_disk_only_bytes);
        }
    }
}

test "preferredEngine: deepseek4 + lora_rank → ds4" {
    const bytes = try buildHeader(testing.allocator, &.{
        .{ .key = "general.architecture", .value = .{ .str = "deepseek4" } },
        .{ .key = "deepseek4.attention.output_lora_rank", .value = .{ .u32_v = 1024 } },
    });
    defer testing.allocator.free(bytes);
    var info = try parseBytes(testing.allocator, bytes);
    defer info.deinit(testing.allocator);
    try testing.expectEqualStrings("deepseek4", info.architecture.?);
    try testing.expect(info.has_ds4_lora_rank);
    try testing.expectEqual(Engine.ds4, preferredEngine(info));
}

test "preferredEngine: deepseek4 without lora_rank → llama" {
    // Models like Preyazz/DeepSeek-V4-Flash-GGUF (vanilla llama.cpp quant)
    // declare arch=deepseek4 but lack the antirez/ds4 MLA metadata — these
    // must NOT route to ds4 (would crash on missing key at engine open).
    const bytes = try buildHeader(testing.allocator, &.{
        .{ .key = "general.architecture", .value = .{ .str = "deepseek4" } },
        .{ .key = "general.name", .value = .{ .str = "DeepSeek V4 Flash" } },
    });
    defer testing.allocator.free(bytes);
    var info = try parseBytes(testing.allocator, bytes);
    defer info.deinit(testing.allocator);
    try testing.expectEqualStrings("deepseek4", info.architecture.?);
    try testing.expect(!info.has_ds4_lora_rank);
    try testing.expectEqual(Engine.llama, preferredEngine(info));
}

test "preferredEngine: lora_rank present but wrong arch → llama" {
    const bytes = try buildHeader(testing.allocator, &.{
        .{ .key = "general.architecture", .value = .{ .str = "qwen2" } },
        .{ .key = "deepseek4.attention.output_lora_rank", .value = .{ .u32_v = 1024 } },
    });
    defer testing.allocator.free(bytes);
    var info = try parseBytes(testing.allocator, bytes);
    defer info.deinit(testing.allocator);
    try testing.expectEqual(Engine.llama, preferredEngine(info));
}

test "parseInfo: skips string arrays between probe keys" {
    // Real GGUFs carry `tokenizer.ggml.tokens` as a giant string array. The
    // parser must walk past it cleanly to find the lora_rank key that comes
    // after.
    const tokens = [_][]const u8{ "<s>", "</s>", "<unk>", "hello", "world" };
    const bytes = try buildHeader(testing.allocator, &.{
        .{ .key = "general.architecture", .value = .{ .str = "deepseek4" } },
        .{ .key = "tokenizer.ggml.tokens", .value = .{ .str_array = &tokens } },
        .{ .key = "deepseek4.attention.output_lora_rank", .value = .{ .u32_v = 1024 } },
    });
    defer testing.allocator.free(bytes);
    var info = try parseBytes(testing.allocator, bytes);
    defer info.deinit(testing.allocator);
    try testing.expectEqual(Engine.ds4, preferredEngine(info));
}

test "preferredEngine: a split GGUF goes to llama.cpp, whatever its arch (#586)" {
    // Split writers put `split.count` after every other key, the vocab included.
    const tokens = [_][]const u8{ "<s>", "</s>" };
    for ([_][]const u8{ "qwen4exp", "deepseek4", "llama" }) |arch| {
        const bytes = try buildHeader(testing.allocator, &.{
            .{ .key = "general.architecture", .value = .{ .str = arch } },
            .{ .key = "deepseek4.attention.output_lora_rank", .value = .{ .u32_v = 1024 } },
            .{ .key = "tokenizer.ggml.tokens", .value = .{ .str_array = &tokens } },
            .{ .key = "split.no", .value = .{ .u32_v = 0 } },
            .{ .key = "split.count", .value = .{ .u32_v = 4 } },
        });
        defer testing.allocator.free(bytes);
        var info = try parseBytes(testing.allocator, bytes);
        defer info.deinit(testing.allocator);
        try testing.expectEqual(@as(u32, 4), info.split_count);
        try testing.expectEqual(Engine.llama, preferredEngine(info));
    }
}

test "parseInfo: bad magic → BadMagic" {
    const bytes = "NOPE\x03\x00\x00\x00" ++ @as([16]u8, @splat(0));
    try testing.expectError(error.BadMagic, parseBytes(testing.allocator, bytes));
}

test "parseInfo: unsupported version → UnsupportedVersion" {
    var bytes: [16]u8 = undefined;
    @memcpy(bytes[0..4], "GGUF");
    std.mem.writeInt(u32, bytes[4..8], 1, .little); // v1 — unsupported
    std.mem.writeInt(u64, bytes[8..16], 0, .little);
    // Truncated; we never get past version check.
    try testing.expectError(error.UnsupportedVersion, parseBytes(testing.allocator, &bytes));
}

test "parseInfo: truncated mid-KV → Truncated" {
    // Header announces 5 KVs but body is empty.
    var bytes: [24]u8 = undefined;
    @memcpy(bytes[0..4], "GGUF");
    std.mem.writeInt(u32, bytes[4..8], 3, .little);
    std.mem.writeInt(u64, bytes[8..16], 0, .little); // tensor_count
    std.mem.writeInt(u64, bytes[16..24], 5, .little); // kv_count
    try testing.expectError(error.Truncated, parseBytes(testing.allocator, &bytes));
}

test "declaresPooling: an embedding GGUF names a pooling type, a chat GGUF does not" {
    const cases = [_]struct { kvs: []const KV, want: bool }{
        .{ .kvs = &.{
            .{ .key = "general.architecture", .value = .{ .str = "gemma-embedding" } },
            .{ .key = "gemma-embedding.pooling_type", .value = .{ .u32_v = 1 } },
        }, .want = true },
        .{ .kvs = &.{
            .{ .key = "general.architecture", .value = .{ .str = "qwen35" } },
            .{ .key = "qwen35.block_count", .value = .{ .u32_v = 32 } },
            .{ .key = "tokenizer.ggml.model", .value = .{ .str = "gpt2" } },
        }, .want = false },
        .{ .kvs = &.{
            .{ .key = "general.architecture", .value = .{ .str = "llama" } },
            .{ .key = "llama.pooling_type", .value = .{ .u32_v = 0 } },
        }, .want = false },
    };
    for (cases) |c| {
        const bytes = try buildHeader(testing.allocator, c.kvs);
        defer testing.allocator.free(bytes);
        var r: std.Io.Reader = .fixed(bytes);
        try testing.expectEqual(c.want, try declaresPooling(&r));
    }
}
