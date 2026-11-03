//! The loaded weights a plugin binds (the host's weight map, one concrete type: no view, no indirection per lookup) and
//! the host's loaders it reaches through `LoadCtx.loader` (called once per file at load).

const std = @import("std");
const mlx = @import("mlx_host").mlx;
const log = @import("mlx_host").log;
const Diag = @import("peek.zig").Diag;

/// Holds all loaded weights as mlx arrays, keyed by name.
pub const Weights = struct {
    map: std.StringHashMap(mlx.mlx_array),
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator) Weights {
        return .{
            .map = std.StringHashMap(mlx.mlx_array).init(allocator),
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *Weights) void {
        var it = self.map.iterator();
        while (it.next()) |entry| {
            _ = mlx.mlx_array_free(entry.value_ptr.*);
            self.allocator.free(entry.key_ptr.*);
        }
        self.map.deinit();
    }

    pub fn get(self: *const Weights, name: []const u8) ?mlx.mlx_array {
        return self.map.get(name);
    }

    /// The tensor `name`, else `err` with `diag` (when given) set to "<name>: <why>": a load's refusal by name.
    pub fn need(self: *const Weights, name: []const u8, diag: ?*Diag, err: anytype, comptime why: []const u8) @TypeOf(err)!mlx.mlx_array {
        return self.map.get(name) orelse {
            if (diag) |d| d.set("{s}: " ++ why, .{name});
            return err;
        };
    }

    /// Hand the map a new array under `name`, freeing the one it held (load-time
    /// weight fusion parks its row views here so the originals go away).
    pub fn replace(self: *Weights, name: []const u8, arr: mlx.mlx_array) void {
        if (self.map.getPtr(name)) |p| {
            _ = mlx.mlx_array_free(p.*);
            p.* = arr;
        }
    }

    /// Every array read into its own MLX buffer now, in one evaluation (the loader's arrays are lazy: each would be
    /// read at its first use, inside the forward that uses it; one evaluation lets MLX's IO threads read them in
    /// parallel). The buffers are the arrays' own (`Load` allocates each at its size): no copy, no transient past MLX's
    /// read staging.
    pub fn materialize(self: *const Weights) !void {
        const vec = mlx.mlx_vector_array_new();
        defer _ = mlx.mlx_vector_array_free(vec);
        var it = self.map.valueIterator();
        while (it.next()) |v| try mlx.check(mlx.mlx_vector_array_append_value(vec, v.*));
        try mlx.check(mlx.mlx_eval(vec));
    }

    pub fn count(self: *const Weights) u32 {
        return @intCast(self.map.count());
    }

    /// Forget `name`, freeing the map's handle (arrays built from it keep what they read).
    pub fn drop(self: *Weights, name: []const u8) void {
        if (self.map.fetchRemove(name)) |kv| {
            _ = mlx.mlx_array_free(kv.value);
            self.allocator.free(kv.key);
        }
    }
};

/// How a load treats stored dtypes. `keep_f16`: a pack whose activation dtype
/// is f16 (Prism Hadamard packs) keeps its f16 side tensors and tables as
/// stored; narrowing them to bf16 drops 3 mantissa bits of every group scale.
/// `nocache`: read the shards past the page cache (`nocache_reader`): the
/// load keeps no file pages next to the array buffers. `embedded_ple` / `defer_qwen4_norms`: the host's qwen4 load
/// options (an n-gram table embedded in the shards, norms folded after the load); a plugin leaves them false.
pub const LoadOpts = struct { vision: bool = false, keep_f16: bool = false, embedded_ple: bool = false, defer_qwen4_norms: bool = false, nocache: bool = false };

/// The host's safetensors loaders (`model.zig`'s): a model directory's shards, and one file (a sidecar the index does
/// not name) into an existing map.
pub const WeightLoader = struct {
    dir: *const fn (io: std.Io, gpa: std.mem.Allocator, model_dir: []const u8, opts: LoadOpts) anyerror!Weights,
    file: *const fn (gpa: std.mem.Allocator, weights: *Weights, path: [*:0]const u8, s: mlx.mlx_stream, opts: LoadOpts) anyerror!void,
};

/// The plugin's own loaders, as #749's host loaded these packs: every shard the index names (or every
/// `*.safetensors` when it names none here), on the CPU stream, f16 scales/biases and 1-D f16 tables narrowed to
/// bf16 unless `keep_f16`.
pub const loader: WeightLoader = .{ .dir = loadDir, .file = loadFile };

fn loadDir(io: std.Io, gpa: std.mem.Allocator, model_dir: []const u8, opts: LoadOpts) anyerror!Weights {
    var dir = try std.Io.Dir.openDirAbsolute(io, model_dir, .{ .iterate = true });
    defer dir.close(io);
    var shards = try indexShards(io, gpa, dir);
    defer {
        var it = shards.keyIterator();
        while (it.next()) |k| gpa.free(k.*);
        shards.deinit(gpa);
    }
    var w = Weights.init(gpa);
    errdefer w.deinit();
    const s = mlx.mlx_default_cpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    var files: u32 = 0;
    var it = dir.iterate();
    while (try it.next(io)) |e| {
        if (e.kind != .file and e.kind != .sym_link) continue;
        if (!std.mem.endsWith(u8, e.name, ".safetensors")) continue;
        if (shards.count() > 0 and !shards.contains(e.name)) continue;
        const path = try std.fmt.allocPrintSentinel(gpa, "{s}/{s}", .{ model_dir, e.name }, 0);
        defer gpa.free(path);
        try loadFile(gpa, &w, path.ptr, s, opts);
        files += 1;
    }
    if (w.count() == 0) return error.NoWeightFiles;
    log.info("[mlx-stream] loaded {d} tensors from {d} file(s)\n", .{ w.count(), files });
    return w;
}

/// The shards `model.safetensors.index.json` names that exist here (empty: load the directory).
fn indexShards(io: std.Io, gpa: std.mem.Allocator, dir: std.Io.Dir) !std.StringHashMapUnmanaged(void) {
    var set: std.StringHashMapUnmanaged(void) = .empty;
    const raw = dir.readFileAlloc(io, "model.safetensors.index.json", gpa, .limited(64 << 20)) catch return set;
    defer gpa.free(raw);
    var parsed = std.json.parseFromSlice(std.json.Value, gpa, raw, .{}) catch return set;
    defer parsed.deinit();
    const wm = if (parsed.value == .object) parsed.value.object.get("weight_map") orelse return set else return set;
    if (wm != .object) return set;
    for (wm.object.values()) |v| {
        if (v != .string or set.contains(v.string)) continue;
        _ = dir.statFile(io, v.string, .{}) catch continue;
        try set.put(gpa, try gpa.dupe(u8, v.string), {});
    }
    return set;
}

fn loadFile(gpa: std.mem.Allocator, w: *Weights, path: [*:0]const u8, s: mlx.mlx_stream, opts: LoadOpts) anyerror!void {
    var tensors = mlx.mlx_map_string_to_array_new();
    defer _ = mlx.mlx_map_string_to_array_free(tensors);
    var meta = mlx.mlx_map_string_to_string_new();
    defer _ = mlx.mlx_map_string_to_string_free(meta);
    if (opts.nocache) {
        const reader = try @import("nocache_reader.zig").reader(std.mem.span(path));
        // Drops our reference only: MLX keeps the reader while an array still reads through it.
        defer _ = mlx.mlx_io_reader_free(reader);
        try mlx.check(mlx.mlx_load_safetensors_reader(&tensors, &meta, reader, s));
    } else try mlx.check(mlx.mlx_load_safetensors(&tensors, &meta, path, s));
    const iter = mlx.mlx_map_string_to_array_iterator_new(tensors);
    defer _ = mlx.mlx_map_string_to_array_iterator_free(iter);
    while (true) {
        var key: ?[*:0]const u8 = null;
        var v = mlx.mlx_array_new();
        if (mlx.mlx_map_string_to_array_iterator_next(&key, &v, iter) != 0 or key == null) {
            _ = mlx.mlx_array_free(v);
            break;
        }
        const name = std.mem.span(key.?);
        if (!opts.keep_f16 and narrowsF16(name, mlx.mlx_array_ndim(v), mlx.mlx_array_dtype(v))) {
            var cast = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_astype(&cast, v, .bfloat16, s));
            _ = mlx.mlx_array_free(v);
            v = cast;
        }
        try w.map.put(try gpa.dupe(u8, name), v);
    }
}

fn narrowsF16(key: []const u8, ndim: usize, dtype: mlx.mlx_dtype) bool {
    if (dtype != .float16) return false;
    return std.mem.endsWith(u8, key, ".scales") or std.mem.endsWith(u8, key, ".biases") or ndim == 1;
}
