const std = @import("std");
const io_mod = @import("expert_io.zig");

pub const Geometry = struct {
    layers: u16,
    experts: u16,
    hidden: u32,
    intermediate: u32,
};

pub const SourceSpan = struct {
    file: u16,
    offset: u64,
    len: u64,
};

pub const QuantGeom = struct {
    bits: u32,
    group_size: u32,
};

/// Affine (bits, group_size) from PACKED shapes alone: a row holds `w_cols * 32`
/// packed bits over `in_dim` values and `s_cols` scale groups cover them.
pub fn affineGeomFromShapes(w_cols: u64, s_cols: u64, in_dim: u64) ?QuantGeom {
    if (w_cols == 0 or s_cols == 0 or in_dim == 0) return null;
    const packed_bits = std.math.mul(u64, w_cols, 32) catch return null;
    if (packed_bits % in_dim != 0 or in_dim % s_cols != 0) return null;
    const bits = packed_bits / in_dim;
    const gs = in_dim / s_cols;
    switch (bits) {
        2, 3, 4, 5, 6, 8 => {},
        else => return null,
    }
    switch (gs) {
        32, 64, 128 => {},
        else => return null,
    }
    return .{ .bits = @intCast(bits), .group_size = @intCast(gs) };
}

pub fn isExpertStreamingArch(model_type: []const u8) bool {
    return std.mem.eql(u8, model_type, "qwen4_exp");
}

/// How a qwen4_exp checkpoint stores its ROUTED experts. Both are leading-index
/// banks: `bf16_fused` is the HF checkpoint's two dense tensors per layer,
/// `quantized_split` the MLX pack's nine (three projections x weight/scales/biases).
pub const Layout = enum { bf16_fused, quantized_split };

pub const Component = enum(u4) {
    gate_w,
    gate_s,
    gate_b,
    up_w,
    up_s,
    up_b,
    down_w,
    down_s,
    down_b,
};

pub const component_count: usize = 9;

pub const Projection = enum { gate, up, down };
pub const Part = enum { weight, scales, biases };

pub fn projectionOf(c: Component) Projection {
    return switch (c) {
        .gate_w, .gate_s, .gate_b => .gate,
        .up_w, .up_s, .up_b => .up,
        .down_w, .down_s, .down_b => .down,
    };
}

pub fn partOf(c: Component) Part {
    return switch (c) {
        .gate_w, .up_w, .down_w => .weight,
        .gate_s, .up_s, .down_s => .scales,
        .gate_b, .up_b, .down_b => .biases,
    };
}

pub fn weightOf(p: Projection) Component {
    return switch (p) {
        .gate => .gate_w,
        .up => .up_w,
        .down => .down_w,
    };
}

pub fn scalesOf(p: Projection) Component {
    return switch (p) {
        .gate => .gate_s,
        .up => .up_s,
        .down => .down_s,
    };
}

pub fn tensorKey(buf: []u8, layer: u16, c: Component) ![]const u8 {
    return std.fmt.bufPrint(buf, "language_model.model.layers.{d}.mlp.switch_mlp.{s}_proj.{s}", .{
        layer,
        @tagName(projectionOf(c)),
        @tagName(partOf(c)),
    });
}

pub fn fusedTensorKey(buf: []u8, layer: u16, down: bool) ![]const u8 {
    return std.fmt.bufPrint(buf, "model.language_model.layers.{d}.mlp.experts.{s}", .{
        layer,
        if (down) "down_proj" else "gate_up_proj",
    });
}

/// True when `key` names a routed-expert bank of `layout` — the tensors the
/// streamed loader must NOT fault into RAM.
pub fn isRoutedExpertKey(layout: Layout, key: []const u8) bool {
    return switch (layout) {
        .bf16_fused => std.mem.startsWith(u8, key, "model.language_model.layers.") and
            (std.mem.endsWith(u8, key, ".mlp.experts.gate_up_proj") or
                std.mem.endsWith(u8, key, ".mlp.experts.down_proj")),
        .quantized_split => std.mem.startsWith(u8, key, "language_model.model.layers.") and
            std.mem.indexOf(u8, key, ".mlp.switch_mlp.") != null,
    };
}

fn stringAt(map: std.json.ObjectMap, key: []const u8) bool {
    const v = map.get(key) orelse return false;
    return v == .string;
}

pub fn layoutFromWeightMap(map: std.json.ObjectMap, layers: u16) ?Layout {
    if (layers == 0) return null;
    var buf: [192]u8 = undefined;
    var fused = true;
    for (0..layers) |layer| {
        const gate = fusedTensorKey(&buf, @intCast(layer), false) catch return null;
        if (!stringAt(map, gate)) {
            fused = false;
            break;
        }
        const down = fusedTensorKey(&buf, @intCast(layer), true) catch return null;
        if (!stringAt(map, down)) {
            fused = false;
            break;
        }
    }
    if (fused) return .bf16_fused;
    for (0..layers) |layer| {
        for (0..component_count) |ci| {
            const key = tensorKey(&buf, @intCast(layer), @enumFromInt(ci)) catch return null;
            if (!stringAt(map, key)) return null;
        }
    }
    return .quantized_split;
}

pub fn layoutFromIndexJson(allocator: std.mem.Allocator, model_type: []const u8, raw: []const u8, layers: u16) ?Layout {
    if (!isExpertStreamingArch(model_type)) return null;
    const parsed = std.json.parseFromSlice(std.json.Value, allocator, raw, .{}) catch return null;
    defer parsed.deinit();
    if (parsed.value != .object) return null;
    const map = parsed.value.object.get("weight_map") orelse return null;
    if (map != .object) return null;
    return layoutFromWeightMap(map.object, layers);
}

pub fn layoutOfDir(allocator: std.mem.Allocator, io: std.Io, model_type: []const u8, model_dir: []const u8, layers: u16) ?Layout {
    if (!isExpertStreamingArch(model_type)) return null;
    var dir = std.Io.Dir.openDirAbsolute(io, model_dir, .{}) catch return null;
    defer dir.close(io);
    const raw = dir.readFileAlloc(io, "model.safetensors.index.json", allocator, .limited(64 * 1024 * 1024)) catch return null;
    defer allocator.free(raw);
    return layoutFromIndexJson(allocator, model_type, raw, layers);
}

const SourceFile = struct {
    name: []u8,
    fd: std.c.fd_t,
};

/// The nine leading-index banks of an MLX-quantized qwen4_exp pack, resolved to
/// per-expert byte spans. Per-tensor (bits, group_size) come from the PACKED
/// shapes against the activation dim the projection consumes — never from the
/// model-wide `quantization` block, which a mixed pack contradicts.
pub const QuantStore = struct {
    allocator: std.mem.Allocator,
    geometry: Geometry,
    files: []SourceFile,
    spans: []SourceSpan,
    slot_bytes: [component_count]u64,
    rows: [component_count]u32,
    cols: [component_count]u32,
    dtypes: [component_count]io_mod.Dtype,
    geoms: [component_count]QuantGeom,

    fn sourceIndex(self: *const QuantStore, layer: u16, expert: u16, c: Component) usize {
        return ((@as(usize, layer) * self.geometry.experts) + expert) * component_count + @backingInt(c);
    }

    pub fn span(self: *const QuantStore, layer: u16, expert: u16, c: Component) SourceSpan {
        return self.spans[self.sourceIndex(layer, expert, c)];
    }

    pub fn slotBytes(self: *const QuantStore, c: Component) u64 {
        return self.slot_bytes[@backingInt(c)];
    }

    pub fn geomOf(self: *const QuantStore, c: Component) QuantGeom {
        return self.geoms[@backingInt(c)];
    }

    pub fn rowsOf(self: *const QuantStore, c: Component) u32 {
        return self.rows[@backingInt(c)];
    }

    pub fn colsOf(self: *const QuantStore, c: Component) u32 {
        return self.cols[@backingInt(c)];
    }

    pub fn dtypeOf(self: *const QuantStore, c: Component) io_mod.Dtype {
        return self.dtypes[@backingInt(c)];
    }

    pub fn expertBytes(self: *const QuantStore) u64 {
        var total: u64 = 0;
        for (self.slot_bytes) |b| total +|= b;
        return total;
    }

    pub fn open(allocator: std.mem.Allocator, model_dir: []const u8, geometry: Geometry) !QuantStore {
        if (geometry.layers == 0 or geometry.experts == 0 or geometry.hidden == 0 or geometry.intermediate == 0)
            return error.InvalidExpertGeometry;
        const io = std.Io.Threaded.global_single_threaded.io();
        var dir = try std.Io.Dir.openDirAbsolute(io, model_dir, .{});
        defer dir.close(io);
        const index_raw = try dir.readFileAlloc(io, "model.safetensors.index.json", allocator, .limited(64 * 1024 * 1024));
        defer allocator.free(index_raw);
        const index_parsed = std.json.parseFromSlice(std.json.Value, allocator, index_raw, .{}) catch return error.InvalidSafetensorsIndex;
        defer index_parsed.deinit();
        if (index_parsed.value != .object) return error.InvalidSafetensorsIndex;
        const weight_map_value = index_parsed.value.object.get("weight_map") orelse return error.InvalidSafetensorsIndex;
        if (weight_map_value != .object) return error.InvalidSafetensorsIndex;
        const weight_map = weight_map_value.object;

        var files_list: std.ArrayList(SourceFile) = .empty;
        errdefer {
            for (files_list.items) |file| {
                _ = std.c.close(file.fd);
                allocator.free(file.name);
            }
            files_list.deinit(allocator);
        }
        const per_layer = std.math.mul(usize, geometry.experts, component_count) catch return error.InvalidExpertGeometry;
        const span_count = std.math.mul(usize, geometry.layers, per_layer) catch return error.InvalidExpertGeometry;
        const spans = try allocator.alloc(SourceSpan, span_count);
        errdefer allocator.free(spans);

        var store = QuantStore{
            .allocator = allocator,
            .geometry = geometry,
            .files = &.{},
            .spans = spans,
            .slot_bytes = @splat(0),
            .rows = @splat(0),
            .cols = @splat(0),
            .dtypes = @splat(.other),
            .geoms = @splat(.{ .bits = 0, .group_size = 0 }),
        };

        var key_buf: [192]u8 = undefined;
        for (0..geometry.layers) |layer_usize| {
            const layer: u16 = @intCast(layer_usize);
            for (0..component_count) |ci| {
                const c: Component = @enumFromInt(ci);
                const key = tensorKey(&key_buf, layer, c) catch return error.InvalidExpertGeometry;
                const mapped = weight_map.get(key) orelse return error.MissingExpertTensor;
                if (mapped != .string) return error.InvalidSafetensorsIndex;
                const file = try openSource(allocator, &files_list, model_dir, mapped.string);
                const region = io_mod.tensorRegion(allocator, files_list.items[file].fd, key) catch |err| return switch (err) {
                    error.MissingSafetensorsTensor => error.MissingExpertTensor,
                    error.SafetensorsTensorOutOfBounds => error.ExpertTensorOutOfBounds,
                    else => error.InvalidExpertTensor,
                };
                if (region.rank != 3 or region.shape[0] != geometry.experts) return error.InvalidExpertTensor;
                if (region.shape[1] == 0 or region.shape[1] > std.math.maxInt(u32)) return error.InvalidExpertTensor;
                if (region.shape[2] == 0 or region.shape[2] > std.math.maxInt(u32)) return error.InvalidExpertTensor;
                const elem: u64 = switch (region.dtype) {
                    .bf16 => 2,
                    .u32 => 4,
                    .other => return error.InvalidExpertTensor,
                };
                const expected_bytes = elem * region.shape[0] * region.shape[1] * region.shape[2];
                if (region.tensor_bytes != expected_bytes) return error.InvalidExpertTensor;
                const per_expert = region.tensor_bytes / geometry.experts;
                if (layer == 0) {
                    store.rows[ci] = @intCast(region.shape[1]);
                    store.cols[ci] = @intCast(region.shape[2]);
                    store.dtypes[ci] = region.dtype;
                    store.slot_bytes[ci] = per_expert;
                } else if (store.rows[ci] != region.shape[1] or store.cols[ci] != region.shape[2] or
                    store.dtypes[ci] != region.dtype or store.slot_bytes[ci] != per_expert)
                {
                    return error.MixedExpertBankGeometry;
                }
                const base = std.math.add(u64, region.data_offset, region.tensor_offset) catch return error.InvalidExpertTensor;
                for (0..geometry.experts) |expert| {
                    spans[(layer_usize * geometry.experts + expert) * component_count + ci] = .{
                        .file = file,
                        .offset = base + @as(u64, expert) * per_expert,
                        .len = per_expert,
                    };
                }
            }
        }

        for ([_]Projection{ .gate, .up, .down }) |p| {
            const in_dim: u64 = if (p == .down) geometry.intermediate else geometry.hidden;
            const out_rows: u64 = if (p == .down) geometry.hidden else geometry.intermediate;
            const w = weightOf(p);
            const sc = scalesOf(p);
            if (store.dtypes[@backingInt(w)] != .u32) return error.InvalidExpertTensor;
            if (store.dtypes[@backingInt(sc)] != .bf16) return error.InvalidExpertTensor;
            if (store.rows[@backingInt(w)] != out_rows or store.rows[@backingInt(sc)] != out_rows) return error.InvalidExpertTensor;
            const geom = affineGeomFromShapes(store.cols[@backingInt(w)], store.cols[@backingInt(sc)], in_dim) orelse
                return error.UnsupportedExpertQuant;
            for ([_]Component{ w, sc, biasesOf(p) }) |c| store.geoms[@backingInt(c)] = geom;
            const b = biasesOf(p);
            if (store.dtypes[@backingInt(b)] != .bf16 or store.cols[@backingInt(b)] != store.cols[@backingInt(sc)] or
                store.rows[@backingInt(b)] != out_rows) return error.InvalidExpertTensor;
        }

        store.files = try files_list.toOwnedSlice(allocator);
        return store;
    }

    pub fn deinit(self: *QuantStore) void {
        for (self.files) |file| {
            _ = std.c.close(file.fd);
            self.allocator.free(file.name);
        }
        self.allocator.free(self.files);
        self.allocator.free(self.spans);
        self.* = undefined;
    }

    pub fn readSpan(self: *const QuantStore, span_value: SourceSpan, dst: []u8) !void {
        if (span_value.file >= self.files.len or span_value.len != dst.len) return error.InvalidExpertRead;
        try io_mod.readExact(self.files[span_value.file].fd, dst, span_value.offset);
    }

    /// The nine slices of one expert, concatenated in `Component` order.
    pub fn readExpert(self: *const QuantStore, layer: u16, expert: u16, dst: []u8) !void {
        if (layer >= self.geometry.layers or expert >= self.geometry.experts) return error.ExpertOutOfRange;
        if (dst.len != self.expertBytes()) return error.InvalidExpertRead;
        var at: usize = 0;
        for (0..component_count) |ci| {
            const c: Component = @enumFromInt(ci);
            const s = self.span(layer, expert, c);
            const len: usize = @intCast(s.len);
            try self.readSpan(s, dst[at..][0..len]);
            at += len;
        }
    }
};

pub fn biasesOf(p: Projection) Component {
    return switch (p) {
        .gate => .gate_b,
        .up => .up_b,
        .down => .down_b,
    };
}

fn openSource(allocator: std.mem.Allocator, list: *std.ArrayList(SourceFile), dir_path: []const u8, name: []const u8) !u16 {
    for (list.items, 0..) |file, i| {
        if (std.mem.eql(u8, file.name, name)) return @intCast(i);
    }
    if (list.items.len >= std.math.maxInt(u16)) return error.TooManyExpertShards;
    const path = try std.fmt.allocPrintSentinel(allocator, "{s}/{s}", .{ dir_path, name }, 0);
    defer allocator.free(path);
    const fd = std.c.open(path.ptr, .{ .ACCMODE = .RDONLY }, @as(std.c.mode_t, 0));
    if (fd < 0) return error.MissingExpertShard;
    errdefer _ = std.c.close(fd);
    _ = std.c.fcntl(fd, std.c.F.NOCACHE, @as(c_int, 1));
    const owned_name = try allocator.dupe(u8, name);
    errdefer allocator.free(owned_name);
    try list.append(allocator, .{ .name = owned_name, .fd = fd });
    return @intCast(list.items.len - 1);
}

test "quantized expert geometry solves bits and group size from packed shapes" {
    const t = std.testing;
    try t.expectEqual(QuantGeom{ .bits = 4, .group_size = 64 }, affineGeomFromShapes(320, 40, 2560).?);
    try t.expectEqual(QuantGeom{ .bits = 4, .group_size = 64 }, affineGeomFromShapes(80, 10, 640).?);
    try t.expectEqual(QuantGeom{ .bits = 8, .group_size = 64 }, affineGeomFromShapes(640, 40, 2560).?);
    try t.expect(affineGeomFromShapes(0, 40, 2560) == null);
    try t.expect(affineGeomFromShapes(320, 40, 0) == null);
    try t.expect(affineGeomFromShapes(321, 40, 2560) == null);
}

test "routed expert layout is read off the weight map" {
    const t = std.testing;
    const fused =
        \\{"weight_map":{"model.language_model.layers.0.mlp.experts.gate_up_proj":"a","model.language_model.layers.0.mlp.experts.down_proj":"a"}}
    ;
    const split =
        \\{"weight_map":{"language_model.model.layers.0.mlp.switch_mlp.gate_proj.weight":"a","language_model.model.layers.0.mlp.switch_mlp.gate_proj.scales":"a","language_model.model.layers.0.mlp.switch_mlp.gate_proj.biases":"a","language_model.model.layers.0.mlp.switch_mlp.up_proj.weight":"a","language_model.model.layers.0.mlp.switch_mlp.up_proj.scales":"a","language_model.model.layers.0.mlp.switch_mlp.up_proj.biases":"a","language_model.model.layers.0.mlp.switch_mlp.down_proj.weight":"a","language_model.model.layers.0.mlp.switch_mlp.down_proj.scales":"a","language_model.model.layers.0.mlp.switch_mlp.down_proj.biases":"a"}}
    ;
    const dense =
        \\{"weight_map":{"language_model.model.layers.0.mlp.gate_proj.weight":"a"}}
    ;
    try t.expectEqual(Layout.bf16_fused, layoutFromIndexJson(t.allocator, "qwen4_exp", fused, 1).?);
    try t.expectEqual(Layout.quantized_split, layoutFromIndexJson(t.allocator, "qwen4_exp", split, 1).?);
    try t.expect(layoutFromIndexJson(t.allocator, "qwen4_exp", dense, 1) == null);
    try t.expect(layoutFromIndexJson(t.allocator, "qwen4_exp", split, 2) == null);
    try t.expect(isRoutedExpertKey(.quantized_split, "language_model.model.layers.3.mlp.switch_mlp.down_proj.scales"));
    try t.expect(!isRoutedExpertKey(.quantized_split, "language_model.model.layers.3.mlp.shared_expert.down_proj.scales"));
    try t.expect(isRoutedExpertKey(.bf16_fused, "model.language_model.layers.3.mlp.experts.down_proj"));
}

test "layout resolution is qwen4_exp only: the same index declares nothing for another arch" {
    const t = std.testing;
    const fused =
        \\{"weight_map":{"model.language_model.layers.0.mlp.experts.gate_up_proj":"a","model.language_model.layers.0.mlp.experts.down_proj":"a"}}
    ;
    const split =
        \\{"weight_map":{"language_model.model.layers.0.mlp.switch_mlp.gate_proj.weight":"a","language_model.model.layers.0.mlp.switch_mlp.gate_proj.scales":"a","language_model.model.layers.0.mlp.switch_mlp.gate_proj.biases":"a","language_model.model.layers.0.mlp.switch_mlp.up_proj.weight":"a","language_model.model.layers.0.mlp.switch_mlp.up_proj.scales":"a","language_model.model.layers.0.mlp.switch_mlp.up_proj.biases":"a","language_model.model.layers.0.mlp.switch_mlp.down_proj.weight":"a","language_model.model.layers.0.mlp.switch_mlp.down_proj.scales":"a","language_model.model.layers.0.mlp.switch_mlp.down_proj.biases":"a"}}
    ;
    try t.expect(isExpertStreamingArch("qwen4_exp"));
    for ([_][]const u8{ "qwen4_exp_text", "qwen3_5_moe", "qwen3_5_moe_text", "qwen3_next", "llama", "deepseek_v4", "" }) |mt| {
        try t.expect(!isExpertStreamingArch(mt));
        try t.expect(layoutFromIndexJson(t.allocator, mt, split, 1) == null);
        try t.expect(layoutFromIndexJson(t.allocator, mt, fused, 1) == null);
    }
}

test "real quantized pack resolves nine regions per layer and the per expert bill" {
    const t = std.testing;
    const path = "/Users/beam/llm/models/Qwen3.8-Flash-Next-MLX-Serve-mixed-4-8bit";
    var dir = std.Io.Dir.openDirAbsolute(t.io, path, .{}) catch return error.SkipZigTest;
    defer dir.close(t.io);
    var store = try QuantStore.open(t.allocator, path, .{ .layers = 48, .experts = 512, .hidden = 2560, .intermediate = 640 });
    defer store.deinit();

    try t.expectEqual(@as(u64, 819_200), store.slotBytes(.gate_w));
    try t.expectEqual(@as(u64, 51_200), store.slotBytes(.gate_s));
    try t.expectEqual(@as(u64, 51_200), store.slotBytes(.gate_b));
    try t.expectEqual(@as(u64, 819_200), store.slotBytes(.down_w));
    try t.expectEqual(@as(u64, 51_200), store.slotBytes(.down_s));
    try t.expectEqual(@as(u64, 2_764_800), store.expertBytes());
    try t.expectEqual(QuantGeom{ .bits = 4, .group_size = 64 }, store.geomOf(.gate_w));
    try t.expectEqual(QuantGeom{ .bits = 4, .group_size = 64 }, store.geomOf(.down_w));
    try t.expectEqual(@as(u32, 640), store.rowsOf(.gate_w));
    try t.expectEqual(@as(u32, 320), store.colsOf(.gate_w));
    try t.expectEqual(@as(u32, 2560), store.rowsOf(.down_w));
    try t.expectEqual(@as(u32, 80), store.colsOf(.down_w));

    // The direct arm re-derives every offset from the shard header, so a wrong
    // stride or a swapped component fails here instead of agreeing with itself.
    const layer: u16 = 3;
    const expert: u16 = 129;
    const total: usize = @intCast(store.expertBytes());
    const through = try t.allocator.alloc(u8, total);
    defer t.allocator.free(through);
    const direct = try t.allocator.alloc(u8, total);
    defer t.allocator.free(direct);
    try store.readExpert(layer, expert, through);
    const index_raw = try dir.readFileAlloc(t.io, "model.safetensors.index.json", t.allocator, .limited(64 * 1024 * 1024));
    defer t.allocator.free(index_raw);
    const index_parsed = try std.json.parseFromSlice(std.json.Value, t.allocator, index_raw, .{});
    defer index_parsed.deinit();
    const weight_map = index_parsed.value.object.get("weight_map").?.object;
    var at: usize = 0;
    for (0..component_count) |ci| {
        const c: Component = @enumFromInt(ci);
        var key_buf: [192]u8 = undefined;
        const key = try tensorKey(&key_buf, layer, c);
        const shard = try std.fmt.allocPrintSentinel(t.allocator, "{s}/{s}", .{ path, weight_map.get(key).?.string }, 0);
        defer t.allocator.free(shard);
        const fd = try io_mod.openHinted(shard, .{});
        defer _ = std.c.close(fd);
        const region = try io_mod.tensorRegion(t.allocator, fd, key);
        const per = region.tensor_bytes / region.shape[0];
        const offset = region.data_offset + region.tensor_offset + @as(u64, expert) * per;
        try t.expectEqual(per, store.slotBytes(c));
        try t.expectEqual(offset, store.span(layer, expert, c).offset);
        const len: usize = @intCast(per);
        try io_mod.readExact(fd, direct[at..][0..len], offset);
        at += len;
    }
    try t.expectEqual(total, at);
    try t.expectEqualSlices(u8, direct, through);

    // A collapsed stride or a duplicated component would still agree above.
    const neighbour = try t.allocator.alloc(u8, total);
    defer t.allocator.free(neighbour);
    try store.readExpert(layer, expert + 1, neighbour);
    try t.expect(!std.mem.eql(u8, through, neighbour));
    const gate_len: usize = @intCast(store.slotBytes(.gate_w));
    const up_at: usize = @intCast(store.slotBytes(.gate_w) + store.slotBytes(.gate_s) + store.slotBytes(.gate_b));
    try t.expect(!std.mem.eql(u8, through[0..gate_len], through[up_at..][0..gate_len]));
}

test "an eight bit tensor beside four bit ones solves to its own width" {
    const t = std.testing;
    const path = "/Users/beam/llm/models/Qwen3.8-Flash-Next-MLX-Serve-mixed-4-8bit";
    var dir = std.Io.Dir.openDirAbsolute(t.io, path, .{}) catch return error.SkipZigTest;
    dir.close(t.io);
    const fd = try io_mod.openHinted(path ++ "/model-00051.safetensors", .{});
    defer _ = std.c.close(fd);
    const w = try io_mod.tensorRegion(t.allocator, fd, "language_model.model.layers.3.mlp.shared_expert.gate_proj.weight");
    const sc = try io_mod.tensorRegion(t.allocator, fd, "language_model.model.layers.3.mlp.shared_expert.gate_proj.scales");
    try t.expectEqual(io_mod.Dtype.u32, w.dtype);
    try t.expectEqual(io_mod.Dtype.bf16, sc.dtype);
    try t.expectEqual(QuantGeom{ .bits = 8, .group_size = 64 }, affineGeomFromShapes(w.shape[1], sc.shape[1], 2560).?);
}
