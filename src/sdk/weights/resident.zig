//! The resident weights: a pack's tensors held whole in one map by name (`Weights`, filled by the engine's loader with
//! the adaptations ModelOpt and Sushi packs need), the source the quant formats read it through (`abi.Source`), and
//! the shard set a safetensors index names.

const std = @import("std");
const abi = @import("../../abi/quant.zig");
const mlx = @import("../../mlx/ffi.zig");
const log = @import("../util/log.zig");

pub const Resident = struct {
    weights: Weights,

    /// A map already loaded (a sidecar file, a test); the source owns it from here.
    pub fn adopt(weights: Weights) Resident {
        return .{ .weights = weights };
    }

    pub fn deinit(self: *Resident) void {
        self.weights.deinit();
    }

    /// The formats' view; borrows `self`, which must not move while it is in use.
    pub fn source(self: *const Resident) abi.Source {
        return .{ .ctx = @constCast(self), .get = get, .resident = true };
    }

    /// The whole map behind `src` when it is a resident source, moved out (the source is left empty): an in-tree
    /// model that keeps the loader's map adopts it instead of reading the pack twice. Null for any other source.
    pub fn take(src: *const abi.Source) ?Weights {
        if (src.size < @sizeOf(abi.Source) or !src.resident) return null;
        const self: *Resident = @ptrCast(@alignCast(src.ctx.?));
        const w = self.weights;
        self.weights = Weights.init(w.allocator);
        return w;
    }

    fn get(ctx: ?*anyopaque, name: abi.Str) callconv(.c) abi.Array {
        const self: *const Resident = @ptrCast(@alignCast(ctx.?));
        const a = self.weights.get(name.slice()) orelse return .{};
        return .{ .ctx = a.ctx };
    }
};

/// Holds all loaded weights as mlx arrays, keyed by name.
pub const Weights = struct {
    map: std.StringHashMap(mlx.mlx_array),
    allocator: std.mem.Allocator,
    /// Int8 copies of block linears that replace their packed arrays (`--w8a8`); only the media
    /// DiTs that read through a name-keyed map fill it.
    w8: @import("../blocks/dit/w8a8.zig").Overlay = .{},

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
        self.w8.deinit(self.allocator);
    }

    pub fn get(self: *const Weights, name: []const u8) ?mlx.mlx_array {
        return self.map.get(name);
    }

    /// Hand the map a new array under `name`, freeing the one it held (load-time
    /// weight fusion parks its row views here so the originals go away).
    pub fn replace(self: *Weights, name: []const u8, arr: mlx.mlx_array) void {
        if (self.map.getPtr(name)) |p| {
            _ = mlx.mlx_array_free(p.*);
            p.* = arr;
        }
    }

    /// Drop every entry named `prefix...` and free its array: a loader that built a block from them (its
    /// structs hold their own refs) lets the originals go before the next block loads, which is when an
    /// int8 copy and the packed weight it replaced would otherwise both be resident.
    pub fn dropPrefix(self: *Weights, prefix: []const u8) void {
        var found = true;
        while (found) {
            found = false;
            var it = self.map.iterator();
            while (it.next()) |e| if (std.mem.startsWith(u8, e.key_ptr.*, prefix)) {
                self.remove(e.key_ptr.*);
                found = true;
                break;
            };
        }
    }

    /// `dropPrefix` for the `idx`-th block of a stack, named by `fmt` (`"blocks.{d}."`).
    pub fn dropBlock(self: *Weights, comptime fmt: []const u8, idx: usize) void {
        var buf: [128]u8 = undefined;
        self.dropPrefix(std.fmt.bufPrint(&buf, fmt, .{idx}) catch return);
    }

    /// Hand the map `arr` under `name` (the key is copied), freeing an array it held there.
    pub fn put(self: *Weights, name: []const u8, arr: mlx.mlx_array) !void {
        if (self.map.contains(name)) return self.replace(name, arr);
        const key = try self.allocator.dupe(u8, name);
        errdefer self.allocator.free(key);
        try self.map.put(key, arr);
    }

    /// Drop `name` and free its array (a loader that copied it elsewhere lets the original go).
    pub fn remove(self: *Weights, name: []const u8) void {
        const kv = self.map.fetchRemove(name) orelse return;
        _ = mlx.mlx_array_free(kv.value);
        self.allocator.free(kv.key);
    }

    pub fn count(self: *const Weights) u32 {
        return @intCast(self.map.count());
    }
};

/// The set of shard basenames in `model.safetensors.index.json`'s `weight_map`,
/// or null when there is no usable index (single-file packs, media packs).
pub fn indexShardSet(io: std.Io, dir: std.Io.Dir) ?std.StringHashMapUnmanaged(void) {
    const a = std.heap.page_allocator;
    const raw = dir.readFileAlloc(io, "model.safetensors.index.json", a, .limited(64 * 1024 * 1024)) catch return null;
    defer a.free(raw);
    var parsed = std.json.parseFromSlice(std.json.Value, a, raw, .{}) catch return null;
    defer parsed.deinit();
    if (parsed.value != .object) return null;
    const wm = parsed.value.object.get("weight_map") orelse return null;
    if (wm != .object) return null;
    var set: std.StringHashMapUnmanaged(void) = .empty;
    for (wm.object.values()) |v| {
        if (v != .string) continue;
        if (set.contains(v.string)) continue;
        const key = a.dupe(u8, v.string) catch continue;
        set.put(a, key, {}) catch {
            a.free(key);
            continue;
        };
    }
    // An index none of whose shards exist is stale (the repo was re-sharded
    // after this index was written); the directory is then the set.
    var any_present = false;
    var keys = set.keyIterator();
    while (keys.next()) |k| {
        _ = dir.statFile(io, k.*, .{}) catch continue;
        any_present = true;
        break;
    }
    if (!any_present) {
        log.warn("model.safetensors.index.json names no shard in this directory; loading every *.safetensors instead\n", .{});
        freeShardSet(&set);
        return null;
    }
    return set;
}

/// Free a set from `indexShardSet`.
pub fn freeShardSet(set: *std.StringHashMapUnmanaged(void)) void {
    var keys = set.keyIterator();
    while (keys.next()) |k| std.heap.page_allocator.free(k.*);
    set.deinit(std.heap.page_allocator);
}

/// Load ONE safetensors file into a Weights map, tensors as stored (lazy), on
/// `mlx.weightLoadStream()` (Metal has no GPU `Load`). The iterator hands a +1
/// reference in `value`; it transfers straight into the map.
pub fn loadWeightsFile(allocator: std.mem.Allocator, model_dir: []const u8, file: []const u8) !Weights {
    var w = Weights.init(allocator);
    errdefer w.deinit();
    const path = try std.fmt.allocPrintSentinel(allocator, "{s}/{s}", .{ model_dir, file }, 0);
    defer allocator.free(path);

    var tensor_map = mlx.mlx_map_string_to_array_new();
    defer _ = mlx.mlx_map_string_to_array_free(tensor_map);
    var meta_map = mlx.mlx_map_string_to_string_new();
    defer _ = mlx.mlx_map_string_to_string_free(meta_map);
    const s = mlx.weightLoadStream();
    defer _ = mlx.mlx_stream_free(s);
    try mlx.check(mlx.mlx_load_safetensors(&tensor_map, &meta_map, path, s));

    const iter = mlx.mlx_map_string_to_array_iterator_new(tensor_map);
    defer _ = mlx.mlx_map_string_to_array_iterator_free(iter);
    while (true) {
        var key: ?[*:0]const u8 = null;
        var value = mlx.mlx_array_new();
        const rc = mlx.mlx_map_string_to_array_iterator_next(&key, &value, iter);
        if (rc != 0 or key == null) {
            _ = mlx.mlx_array_free(value);
            break;
        }
        const owned_key = allocator.dupe(u8, std.mem.span(key.?)) catch |e| {
            _ = mlx.mlx_array_free(value);
            return e;
        };
        w.map.put(owned_key, value) catch |e| {
            allocator.free(owned_key);
            _ = mlx.mlx_array_free(value);
            return e;
        };
    }
    log.info("[weights] loaded {d} tensors from {s}\n", .{ w.count(), file });
    return w;
}

// ── tests ──

const testing = std.testing;

test "resident: take moves the map out of a resident source and refuses any other" {
    var w = Weights.init(testing.allocator);
    try w.put("x.weight", mlx.mlx_array_new_float(2.0));
    var r = Resident.adopt(w);
    defer r.deinit();
    const src = r.source();
    var got = Resident.take(&src) orelse return error.NotTaken;
    defer got.deinit();
    try testing.expectEqual(@as(u32, 1), got.count());
    try testing.expectEqual(@as(u32, 0), r.weights.count());
    const other: abi.Source = .{ .ctx = null, .get = struct {
        fn g(_: ?*anyopaque, _: abi.Str) callconv(.c) abi.Array {
            return .{};
        }
    }.g };
    try testing.expect(Resident.take(&other) == null);
}

test "resident: the source hands out the map's own arrays and an empty one for a missing name" {
    var w = Weights.init(testing.allocator);
    const a = mlx.mlx_array_new_float(2.0);
    try w.put("x.weight", a);
    var r = Resident.adopt(w);
    defer r.deinit();
    const src = r.source();
    try testing.expectEqual(a.ctx, src.tensor("x.weight").ctx);
    try testing.expectEqual(@as(?*anyopaque, null), src.tensor("x.scales").ctx);
}
