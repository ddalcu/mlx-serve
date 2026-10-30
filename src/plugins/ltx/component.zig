//! One LTX component file (`*.safetensors`) as a name-keyed map, quantized linears told apart by a sibling
//! `.scales`. The checkpoint stores each sub-model as its own file in one directory, so a whole-dir load would read
//! all of them at once.

const std = @import("std");
const mlx = @import("../../mlx/ffi.zig");
const log = @import("../../sdk/util/log.zig");
const lora_mod = @import("../../sdk/weights/lora.zig");
const w8a8 = @import("../../sdk/blocks/dit/w8a8.zig");

const S = mlx.mlx_stream;

pub const Component = struct {
    map: std.StringHashMap(mlx.mlx_array),
    allocator: std.mem.Allocator,
    // Runtime LoRA (non-owning; the engine owns the Stack and
    // re-installs the pointer after every transformer swap). Checked by
    // dQLin only — the DiT projections are the modules video LoRAs target.
    // A Stack may hold several simultaneously-attached adapters; their
    // deltas are summed (lora.deltaSum), never merged.
    lora: ?*const lora_mod.Stack = null,
    /// Int8 copies of the DiT block linears that REPLACE their packed `<base>.weight/.scales/.biases`
    /// (`--w8a8`); `dQLin` reads them.
    w8: w8a8.Overlay = .{},

    pub fn deinit(self: *Component) void {
        var it = self.map.iterator();
        while (it.next()) |e| {
            _ = mlx.mlx_array_free(e.value_ptr.*);
            self.allocator.free(e.key_ptr.*);
        }
        self.map.deinit();
        self.w8.deinit(self.allocator);
    }

    /// Whether the linear behind `weight_key` (`<base>.weight`) is present, packed or int8.
    pub fn hasWeight(self: *const Component, weight_key: []const u8) bool {
        return self.map.contains(weight_key) or self.w8.has(weight_key);
    }

    pub fn get(self: *const Component, key: []const u8) ?mlx.mlx_array {
        return self.map.get(key);
    }

    /// A weight is quantized (q4) iff a sibling `<base>.scales` exists. `weight`
    /// keys end in `.weight`; the quant siblings are `.scales`/`.biases`.
    pub fn isQuantized(self: *const Component, weight_key: []const u8) bool {
        if (!std.mem.endsWith(u8, weight_key, ".weight")) return false;
        const base = weight_key[0 .. weight_key.len - ".weight".len];
        var buf: [512]u8 = undefined;
        const scales_key = std.fmt.bufPrint(&buf, "{s}.scales", .{base}) catch return false;
        return self.map.contains(scales_key);
    }

    pub fn count(self: *const Component) u32 {
        return @intCast(self.map.count());
    }
};

/// Load one `*.safetensors` file (absolute path) into a Component map.
pub fn loadComponent(allocator: std.mem.Allocator, path: [:0]const u8, s: S) !Component {
    var comp = Component{ .map = std.StringHashMap(mlx.mlx_array).init(allocator), .allocator = allocator };
    errdefer comp.deinit();

    var tensor_map = mlx.mlx_map_string_to_array_new();
    defer _ = mlx.mlx_map_string_to_array_free(tensor_map);
    var meta_map = mlx.mlx_map_string_to_string_new();
    defer _ = mlx.mlx_map_string_to_string_free(meta_map);
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
        const key_slice = std.mem.span(key.?);
        const owned_key = try allocator.dupe(u8, key_slice);
        errdefer allocator.free(owned_key);
        // Transfer the iterator's reference straight into the map (the
        // model.zig loadSafetensorsFile pattern). The old copy-via-
        // mlx_array_set left `value`'s reference dangling on the floor —
        // a phantom +1 on EVERY tensor, so Component.deinit could never
        // release the buffers and each engine unload leaked the whole
        // materialized component set (~18 GB per video load→gen→unload).
        try comp.map.put(owned_key, value);
    }
    log.info("[ltx] loaded {d} tensors from {s}\n", .{ comp.count(), path });
    return comp;
}
