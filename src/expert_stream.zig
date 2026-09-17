const std = @import("std");
const mlx = @import("mlx.zig");
const log = @import("log.zig");
const io_mod = @import("expert_io.zig");
const kernels = @import("expert_bf16_kernels.zig");
pub const quant = @import("expert_quant.zig");

pub const Span = struct {
    offset: u64,
    len: u64,
};

pub const TensorLayout = struct {
    data_offset: u64,
    tensor_offset: u64,
    tensor_bytes: u64,
    experts: u16,

    pub fn expertSpan(self: TensorLayout, expert: u16) !Span {
        if (expert >= self.experts) return error.ExpertOutOfRange;
        if (self.experts == 0 or self.tensor_bytes % self.experts != 0) return error.InvalidExpertTensor;
        const len = self.tensor_bytes / self.experts;
        const base = std.math.add(u64, self.data_offset, self.tensor_offset) catch return error.InvalidExpertTensor;
        const delta = std.math.mul(u64, expert, len) catch return error.InvalidExpertTensor;
        return .{ .offset = std.math.add(u64, base, delta) catch return error.InvalidExpertTensor, .len = len };
    }
};

pub const CachePlan = struct {
    slots_per_layer: u16,
    expert_bytes: u64,
    cache_bytes: u64,
    workspace_bytes: u64,
    bounce_bytes: u64,
    layer_bytes: u64,
    prefill_peak_bytes: u64,
};

pub fn fillPeakBytes(expert_bytes: u64, union_size: u16, slots_per_layer: u16, batch_bytes: u64) u64 {
    const workspace = @as(u64, union_size) *| expert_bytes;
    const layer = @as(u64, slots_per_layer) *| expert_bytes;
    return workspace +| layer +| workspace +| @min(workspace, batch_bytes);
}

pub const BOUNCE_BYTES: u64 = 8 * 64 * 1024 * 1024;

pub fn expertBytes(gate_up_rows: u32, hidden: u32, intermediate: u32) !u64 {
    if (gate_up_rows == 0 or hidden == 0 or intermediate == 0) return error.InvalidExpertGeometry;
    const gate_elems = std.math.mul(u64, gate_up_rows, hidden) catch return error.InvalidExpertGeometry;
    const down_elems = std.math.mul(u64, hidden, intermediate) catch return error.InvalidExpertGeometry;
    return std.math.mul(u64, std.math.add(u64, gate_elems, down_elems) catch return error.InvalidExpertGeometry, 2) catch return error.InvalidExpertGeometry;
}

/// Per-expert bytes of whichever routed-expert layout this checkpoint ships:
/// the fused bf16 pair, or the sum of the nine quantized slices the pack stores.
pub fn expertBytesFor(allocator: std.mem.Allocator, model_dir: []const u8, geometry: Geometry, layout: quant.Layout) !u64 {
    switch (layout) {
        .bf16_fused => return expertBytes(2 * geometry.intermediate, geometry.hidden, geometry.intermediate),
        .quantized_split => {
            var store = try quant.QuantStore.open(allocator, model_dir, .{
                .layers = geometry.layers,
                .experts = geometry.experts,
                .hidden = geometry.hidden,
                .intermediate = geometry.intermediate,
            });
            defer store.deinit();
            return store.expertBytes();
        },
    }
}

pub fn cachePlanBytes(requested_bytes: u64, layers: u16, experts: u16, expert_bytes: u64) !CachePlan {
    if (layers == 0 or experts == 0 or expert_bytes == 0) return error.InvalidExpertGeometry;
    const per_slot_all_layers = std.math.mul(u64, layers, expert_bytes) catch return error.InvalidExpertGeometry;
    const slots_u64 = @min(requested_bytes / per_slot_all_layers, experts);
    if (slots_u64 == 0) return error.ExpertCacheTooSmall;
    const slots: u16 = @intCast(slots_u64);
    const workspace_bytes = std.math.mul(u64, experts, expert_bytes) catch return error.InvalidExpertGeometry;
    const bounce_bytes: u64 = BOUNCE_BYTES;
    const layer_bytes = std.math.mul(u64, slots, expert_bytes) catch return error.InvalidExpertGeometry;
    return .{
        .slots_per_layer = slots,
        .expert_bytes = expert_bytes,
        .cache_bytes = std.math.mul(u64, std.math.mul(u64, layers, slots) catch return error.InvalidExpertGeometry, expert_bytes) catch return error.InvalidExpertGeometry,
        .workspace_bytes = workspace_bytes,
        .bounce_bytes = bounce_bytes,
        .layer_bytes = layer_bytes,
        .prefill_peak_bytes = fillPeakBytes(expert_bytes, experts, slots, bounce_bytes),
    };
}

pub const MTP_UNSUPPORTED: []const u8 = "MTP speculative decode is not supported under bf16 expert streaming: spec verify captures per-position SSM state the streamed MoE forward declines";

/// PURE: why MTP cannot serve a streamed model, or null. Refused at the door — at load for
/// `--mtp` and at request parse for an explicit `enable_mtp` — because an armed head reaches
/// `error.StreamingSpecCaptureUnsupported` inside the forward instead.
pub const MtpUnderStreaming = enum { off, refuse, drop_settings };

/// An explicit `--mtp` launch flag is refused; a per-model `mtp: true` setting is
/// dropped with a warning (the setting was written for the resident load of the same
/// pack, and a dead server is the wrong answer to it).
pub fn mtpUnderStreaming(flag_mtp: bool, settings_mtp: ?bool) MtpUnderStreaming {
    if (flag_mtp) return .refuse;
    if (settings_mtp == true) return .drop_settings;
    return .off;
}

pub fn mtpRefusal(expert_streaming: bool, mtp_requested: bool) ?[]const u8 {
    if (!expert_streaming or !mtp_requested) return null;
    return MTP_UNSUPPORTED;
}

/// PURE: does this load stream its routed experts? A checkpoint with no resident
/// arm always does; a capable one only when a cache or an SSD budget was asked for,
/// so a quantized pack with neither loads exactly as it did before streaming existed.
pub fn expertStreamingEngaged(capable: bool, required: bool, explicit_cache_bytes: u64, budget_bytes: u64) bool {
    if (!capable) return false;
    return required or explicit_cache_bytes > 0 or budget_bytes > 0;
}

pub const BudgetLedger = struct {
    budget_bytes: u64,
    trunk_bytes: u64,
    mtp_bytes: u64,
    workspace_bytes: u64,
    selected_bytes: u64,
    bounce_bytes: u64,
    cache_bytes: u64,
    slots_per_layer: u16,
};

pub fn budgetOverriddenByExplicitCache(explicit_cache_bytes: u64, budget_bytes: u64) bool {
    return explicit_cache_bytes > 0 and budget_bytes > 0;
}

/// The budget is the TOTAL resident target (GiB, the `--ssd-budget-gb` unit); the expert
/// cache is what remains after trunk, MTP, the 512-expert union workspace, the selected
/// slab and the bounce buffers. The PLE table is a disk gather and is never billed.
pub fn budgetLedger(
    budget_bytes: u64,
    trunk_bytes: u64,
    mtp_bytes: u64,
    layers: u16,
    experts: u16,
    top_k: u16,
    expert_bytes: u64,
    bounce_bytes: u64,
) !BudgetLedger {
    if (layers == 0 or experts == 0 or top_k == 0 or expert_bytes == 0) return error.InvalidExpertGeometry;
    const workspace_bytes = @as(u64, experts) *| expert_bytes;
    const selected_bytes = @as(u64, @min(top_k, experts)) *| expert_bytes;
    const fixed = trunk_bytes +| mtp_bytes +| workspace_bytes +| selected_bytes +| bounce_bytes;
    if (fixed >= budget_bytes) return error.SsdBudgetBelowResident;
    const per_slot = @as(u64, layers) *| expert_bytes;
    const slots_u64 = @min((budget_bytes - fixed) / per_slot, experts);
    if (slots_u64 <= 1) return error.SsdBudgetBelowResident;
    const slots: u16 = @intCast(slots_u64);
    return .{
        .budget_bytes = budget_bytes,
        .trunk_bytes = trunk_bytes,
        .mtp_bytes = mtp_bytes,
        .workspace_bytes = workspace_bytes,
        .selected_bytes = selected_bytes,
        .bounce_bytes = bounce_bytes,
        .cache_bytes = @as(u64, slots) *| per_slot,
        .slots_per_layer = slots,
    };
}

pub const FILL_WORKERS: usize = 4;
pub const COALESCE_MAX: u64 = 64 * 1024 * 1024;

pub const BREAKDOWN_EVERY: u64 = 64;

const PENDING_READERS_MAX: usize = 2 * quant.component_count;

pub const RouteDetail = struct {
    build_ns: u64 = 0,
    x_wait_ns: u64 = 0,
    ids_wait_ns: u64 = 0,
    read_ns: u64 = 0,

    pub fn totalNs(self: RouteDetail) u64 {
        return self.build_ns +| self.x_wait_ns +| self.ids_wait_ns +| self.read_ns;
    }
};

pub const RouteClass = struct {
    layers: u32 = 0,
    build_ns: u64 = 0,
    x_wait_ns: u64 = 0,
    ids_wait_ns: u64 = 0,
    read_ns: u64 = 0,

    fn add(self: *RouteClass, detail: RouteDetail) void {
        self.layers += 1;
        self.build_ns +|= detail.build_ns;
        self.x_wait_ns +|= detail.x_wait_ns;
        self.ids_wait_ns +|= detail.ids_wait_ns;
        self.read_ns +|= detail.read_ns;
    }
};

pub const Breakdown = struct {
    forward: u64 = 0,
    rows: usize = 0,
    wall_ns: u64 = 0,
    route_ns: u64 = 0,
    fill_ns: u64 = 0,
    compute_ns: u64 = 0,
    fill_bytes: u64 = 0,
    hits: u64 = 0,
    union_members: u64 = 0,
    route_linear: RouteClass = .{},
    route_full: RouteClass = .{},
    probe: bool = false,

    pub fn otherNs(self: Breakdown) u64 {
        const accounted = self.route_ns +| self.fill_ns +| self.compute_ns;
        return if (self.wall_ns > accounted) self.wall_ns - accounted else 0;
    }
};

pub const Options = struct {
    io_workers: usize = FILL_WORKERS,
    bounce_size: usize = 64 * 1024 * 1024,
    layout: quant.Layout = .bf16_fused,
};

pub const Binding = struct {
    expert: u16,
    slot: ?u16,
    hit: bool,
};

pub const GroupResolution = struct {
    union_ids: []u16,
    remapped: []u16,
    bindings: []Binding,
    hits: usize,
    misses: usize,
    declined: usize,
    workspace: bool,

    pub fn deinit(self: *GroupResolution, allocator: std.mem.Allocator) void {
        allocator.free(self.union_ids);
        allocator.free(self.remapped);
        allocator.free(self.bindings);
        self.* = undefined;
    }
};

pub const GroupCache = struct {
    allocator: std.mem.Allocator,
    expert_to_slot: []i32,
    slot_to_expert: []u16,
    ages: []u64,
    ready: []bool,
    tick: u64 = 0,

    pub fn init(allocator: std.mem.Allocator, capacity: u16, expert_count: u16) !GroupCache {
        if (capacity == 0 or capacity > expert_count) return error.InvalidCacheCapacity;
        const expert_to_slot = try allocator.alloc(i32, expert_count);
        errdefer allocator.free(expert_to_slot);
        const slot_to_expert = try allocator.alloc(u16, capacity);
        errdefer allocator.free(slot_to_expert);
        const ages = try allocator.alloc(u64, capacity);
        errdefer allocator.free(ages);
        const ready = try allocator.alloc(bool, capacity);
        @memset(expert_to_slot, -1);
        @memset(slot_to_expert, std.math.maxInt(u16));
        @memset(ages, 0);
        @memset(ready, false);
        return .{ .allocator = allocator, .expert_to_slot = expert_to_slot, .slot_to_expert = slot_to_expert, .ages = ages, .ready = ready };
    }

    pub fn deinit(self: *GroupCache) void {
        self.allocator.free(self.expert_to_slot);
        self.allocator.free(self.slot_to_expert);
        self.allocator.free(self.ages);
        self.allocator.free(self.ready);
        self.* = undefined;
    }

    pub fn markReady(self: *GroupCache, slot: u16) void {
        self.ready[slot] = true;
    }

    fn invalidateUnready(self: *GroupCache) void {
        for (self.ready, 0..) |flag, i| {
            if (!flag) self.evict(@intCast(i));
        }
    }

    fn evict(self: *GroupCache, slot: u16) void {
        const old = self.slot_to_expert[slot];
        if (old != std.math.maxInt(u16)) self.expert_to_slot[old] = -1;
        self.slot_to_expert[slot] = std.math.maxInt(u16);
        self.ready[slot] = false;
        self.ages[slot] = 0;
    }

    fn touch(self: *GroupCache, slot: u16) void {
        self.tick +%= 1;
        if (self.tick == 0) self.tick = 1;
        self.ages[slot] = self.tick;
    }

    fn victim(self: *const GroupCache) u16 {
        var best: u16 = 0;
        for (self.slot_to_expert, 0..) |expert, i| {
            if (expert == std.math.maxInt(u16)) return @intCast(i);
            if (self.ages[i] < self.ages[best]) best = @intCast(i);
        }
        return best;
    }

    fn admit(self: *GroupCache, expert: u16) u16 {
        const slot = self.victim();
        const old = self.slot_to_expert[slot];
        if (old != std.math.maxInt(u16)) self.expert_to_slot[old] = -1;
        self.slot_to_expert[slot] = expert;
        self.expert_to_slot[expert] = slot;
        self.ready[slot] = false;
        self.touch(slot);
        return slot;
    }

    pub fn resolve(self: *GroupCache, allocator: std.mem.Allocator, occurrences: []const u16) !GroupResolution {
        const last = try allocator.alloc(usize, self.expert_to_slot.len);
        defer allocator.free(last);
        @memset(last, std.math.maxInt(usize));
        const counts = try allocator.alloc(u32, self.expert_to_slot.len);
        defer allocator.free(counts);
        @memset(counts, 0);
        for (occurrences, 0..) |expert, i| {
            if (expert >= self.expert_to_slot.len) return error.ExpertOutOfRange;
            last[expert] = i;
            counts[expert] += 1;
        }

        var union_list: std.ArrayList(u16) = .empty;
        defer union_list.deinit(allocator);
        for (occurrences, 0..) |expert, i| {
            if (last[expert] == i) try union_list.append(allocator, expert);
        }
        const union_ids = try union_list.toOwnedSlice(allocator);
        errdefer allocator.free(union_ids);
        const bindings = try allocator.alloc(Binding, union_ids.len);
        errdefer allocator.free(bindings);

        var hits: usize = 0;
        var misses: usize = 0;
        for (union_ids, 0..) |expert, i| {
            const raw_slot = self.expert_to_slot[expert];
            if (raw_slot >= 0 and self.ready[@intCast(raw_slot)]) {
                const slot: u16 = @intCast(raw_slot);
                bindings[i] = .{ .expert = expert, .slot = slot, .hit = true };
                self.touch(slot);
                hits += 1;
            } else {
                if (raw_slot >= 0) self.evict(@intCast(raw_slot));
                bindings[i] = .{ .expert = expert, .slot = null, .hit = false };
                misses += 1;
            }
        }

        const workspace = union_ids.len > self.slot_to_expert.len;
        const admission_cap = self.slot_to_expert.len - hits;
        const admit_count = @min(admission_cap, misses);
        const order = try allocator.alloc(usize, misses);
        defer allocator.free(order);
        var pending: usize = 0;
        for (bindings, 0..) |binding, i| {
            if (binding.hit) continue;
            order[pending] = i;
            pending += 1;
        }
        if (workspace) {
            const Sorter = struct {
                counts: []const u32,
                bindings: []const Binding,
                fn less(ctx: @This(), a: usize, b: usize) bool {
                    const ca = ctx.counts[ctx.bindings[a].expert];
                    const cb = ctx.counts[ctx.bindings[b].expert];
                    if (ca != cb) return ca > cb;
                    return a < b;
                }
            };
            std.sort.pdq(usize, order, Sorter{ .counts = counts, .bindings = bindings }, Sorter.less);
        }
        if (workspace) {
            var taken = admit_count;
            while (taken > 0) {
                taken -= 1;
                const i = order[taken];
                bindings[i].slot = self.admit(bindings[i].expert);
            }
        } else {
            for (order[0..admit_count]) |i| bindings[i].slot = self.admit(bindings[i].expert);
        }
        const remapped = try allocator.alloc(u16, occurrences.len);
        errdefer allocator.free(remapped);
        const union_slots = try allocator.alloc(u16, self.expert_to_slot.len);
        defer allocator.free(union_slots);
        @memset(union_slots, std.math.maxInt(u16));
        for (union_ids, 0..) |expert, i| union_slots[expert] = @intCast(i);
        for (occurrences, 0..) |expert, i| remapped[i] = union_slots[expert];

        return .{
            .union_ids = union_ids,
            .remapped = remapped,
            .bindings = bindings,
            .hits = hits,
            .misses = misses,
            .declined = misses - admit_count,
            .workspace = workspace,
        };
    }
};

pub const Geometry = struct {
    layers: u16,
    experts: u16,
    hidden: u32,
    intermediate: u32,
};

pub const Component = enum(u1) {
    gate_up,
    down,
};

pub const SourceSpan = quant.SourceSpan;

const SourceFile = struct {
    name: []u8,
    fd: std.c.fd_t,
};

pub const readExact = io_mod.readExact;

fn tensorLayout(allocator: std.mem.Allocator, fd: std.c.fd_t, key: []const u8, shape_expected: [3]u64) !TensorLayout {
    const region = io_mod.tensorRegion(allocator, fd, key) catch |err| return switch (err) {
        error.MissingSafetensorsTensor => error.MissingExpertTensor,
        error.SafetensorsTensorOutOfBounds => error.ExpertTensorOutOfBounds,
        else => error.InvalidExpertTensor,
    };
    if (region.dtype != .bf16 or region.rank != 3 or !std.mem.eql(u64, &region.shape, &shape_expected)) return error.InvalidExpertTensor;
    if (region.tensor_bytes != 2 * shape_expected[0] * shape_expected[1] * shape_expected[2]) return error.InvalidExpertTensor;
    return .{ .data_offset = region.data_offset, .tensor_offset = region.tensor_offset, .tensor_bytes = region.tensor_bytes, .experts = @intCast(shape_expected[0]) };
}

pub const SlabSpec = struct {
    rows: u32,
    cols: u32,
    dtype: mlx.mlx_dtype,
    elem_bytes: u8,
    slot_bytes: u64,
};

pub const ExpertStore = struct {
    allocator: std.mem.Allocator,
    geometry: Geometry,
    files: []SourceFile,
    spans: []SourceSpan,
    quantized: ?quant.QuantStore = null,

    fn sourceIndex(self: *const ExpertStore, layer: u16, expert: u16, component: Component) usize {
        return (@as(usize, layer) * self.geometry.experts + expert) * 2 + @backingInt(component);
    }

    pub fn span(self: *const ExpertStore, layer: u16, expert: u16, component: Component) SourceSpan {
        return self.spans[self.sourceIndex(layer, expert, component)];
    }

    pub fn layout(self: *const ExpertStore) quant.Layout {
        return if (self.quantized == null) .bf16_fused else .quantized_split;
    }

    pub fn componentCount(self: *const ExpertStore) usize {
        return if (self.quantized == null) 2 else quant.component_count;
    }

    pub fn spanAt(self: *const ExpertStore, layer: u16, expert: u16, ci: usize) SourceSpan {
        if (self.quantized) |*q| return q.span(layer, expert, @enumFromInt(ci));
        return self.spans[(@as(usize, layer) * self.geometry.experts + expert) * 2 + ci];
    }

    pub fn slabSpec(self: *const ExpertStore, ci: usize) SlabSpec {
        if (self.quantized) |*q| {
            const c: quant.Component = @enumFromInt(ci);
            const dtype: mlx.mlx_dtype = switch (q.dtypeOf(c)) {
                .bf16 => .bfloat16,
                .u32 => .uint32,
                .other => .bfloat16,
            };
            const elem: u8 = if (q.dtypeOf(c) == .u32) 4 else 2;
            return .{ .rows = q.rowsOf(c), .cols = q.colsOf(c), .dtype = dtype, .elem_bytes = elem, .slot_bytes = q.slotBytes(c) };
        }
        const rows: u32 = if (ci == 0) 2 * self.geometry.intermediate else self.geometry.hidden;
        const cols: u32 = if (ci == 0) self.geometry.hidden else self.geometry.intermediate;
        return .{ .rows = rows, .cols = cols, .dtype = .bfloat16, .elem_bytes = 2, .slot_bytes = @as(u64, rows) * cols * 2 };
    }

    pub fn perExpertBytes(self: *const ExpertStore) u64 {
        if (self.quantized) |*q| return q.expertBytes();
        var total: u64 = 0;
        for (0..2) |ci| total += self.slabSpec(ci).slot_bytes;
        return total;
    }

    pub fn fileCount(self: *const ExpertStore) usize {
        if (self.quantized) |*q| return q.files.len;
        return self.files.len;
    }

    pub fn fdAt(self: *const ExpertStore, i: usize) std.c.fd_t {
        if (self.quantized) |*q| return q.files[i].fd;
        return self.files[i].fd;
    }

    pub fn openLayout(allocator: std.mem.Allocator, model_dir: []const u8, geometry: Geometry, chosen: quant.Layout) !ExpertStore {
        if (chosen == .bf16_fused) return open(allocator, model_dir, geometry);
        const q = try quant.QuantStore.open(allocator, model_dir, .{
            .layers = geometry.layers,
            .experts = geometry.experts,
            .hidden = geometry.hidden,
            .intermediate = geometry.intermediate,
        });
        return .{ .allocator = allocator, .geometry = geometry, .files = &.{}, .spans = &.{}, .quantized = q };
    }

    pub fn open(allocator: std.mem.Allocator, model_dir: []const u8, geometry: Geometry) !ExpertStore {
        if (geometry.layers == 0 or geometry.experts == 0 or geometry.hidden == 0 or geometry.intermediate == 0) return error.InvalidExpertGeometry;
        const io = std.Io.Threaded.global_single_threaded.io();
        var dir = try std.Io.Dir.openDirAbsolute(io, model_dir, .{});
        defer dir.close(io);
        const index_raw = try dir.readFileAlloc(io, "model.safetensors.index.json", allocator, .limited(32 * 1024 * 1024));
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
        const span_count = std.math.mul(usize, std.math.mul(usize, geometry.layers, geometry.experts) catch return error.InvalidExpertGeometry, 2) catch return error.InvalidExpertGeometry;
        const spans = try allocator.alloc(SourceSpan, span_count);
        errdefer allocator.free(spans);

        const openSource = struct {
            fn call(a: std.mem.Allocator, list: *std.ArrayList(SourceFile), dir_path: []const u8, name: []const u8) !u16 {
                for (list.items, 0..) |file, i| {
                    if (std.mem.eql(u8, file.name, name)) return @intCast(i);
                }
                if (list.items.len >= std.math.maxInt(u16)) return error.TooManyExpertShards;
                const path = try std.fmt.allocPrintSentinel(a, "{s}/{s}", .{ dir_path, name }, 0);
                defer a.free(path);
                const fd = std.c.open(path.ptr, .{ .ACCMODE = .RDONLY }, @as(std.c.mode_t, 0));
                if (fd < 0) return error.MissingExpertShard;
                errdefer _ = std.c.close(fd);
                _ = std.c.fcntl(fd, std.c.F.NOCACHE, @as(c_int, 1));
                const owned_name = try a.dupe(u8, name);
                errdefer a.free(owned_name);
                try list.append(a, .{ .name = owned_name, .fd = fd });
                return @intCast(list.items.len - 1);
            }
        }.call;

        var key_buf: [192]u8 = undefined;
        for (0..geometry.layers) |layer_usize| {
            const layer: u16 = @intCast(layer_usize);
            const gate_key = std.fmt.bufPrint(&key_buf, "model.language_model.layers.{d}.mlp.experts.gate_up_proj", .{layer}) catch return error.InvalidExpertGeometry;
            const gate_map = weight_map.get(gate_key) orelse return error.MissingExpertTensor;
            if (gate_map != .string) return error.InvalidSafetensorsIndex;
            const gate_file = try openSource(allocator, &files_list, model_dir, gate_map.string);
            const gate_layout = try tensorLayout(allocator, files_list.items[gate_file].fd, gate_key, .{ geometry.experts, 2 * @as(u64, geometry.intermediate), geometry.hidden });

            const down_key = std.fmt.bufPrint(&key_buf, "model.language_model.layers.{d}.mlp.experts.down_proj", .{layer}) catch return error.InvalidExpertGeometry;
            const down_map = weight_map.get(down_key) orelse return error.MissingExpertTensor;
            if (down_map != .string) return error.InvalidSafetensorsIndex;
            const down_file = try openSource(allocator, &files_list, model_dir, down_map.string);
            const down_layout = try tensorLayout(allocator, files_list.items[down_file].fd, down_key, .{ geometry.experts, geometry.hidden, geometry.intermediate });

            for (0..geometry.experts) |expert_usize| {
                const expert: u16 = @intCast(expert_usize);
                const gate_span = try gate_layout.expertSpan(expert);
                const down_span = try down_layout.expertSpan(expert);
                const base = (@as(usize, layer) * geometry.experts + expert) * 2;
                spans[base] = .{ .file = gate_file, .offset = gate_span.offset, .len = gate_span.len };
                spans[base + 1] = .{ .file = down_file, .offset = down_span.offset, .len = down_span.len };
            }
        }

        return .{ .allocator = allocator, .geometry = geometry, .files = try files_list.toOwnedSlice(allocator), .spans = spans };
    }

    pub fn deinit(self: *ExpertStore) void {
        if (self.quantized) |*q| q.deinit();
        for (self.files) |file| {
            _ = std.c.close(file.fd);
            self.allocator.free(file.name);
        }
        if (self.files.len != 0) self.allocator.free(self.files);
        if (self.spans.len != 0) self.allocator.free(self.spans);
        self.* = undefined;
    }

    pub fn readSpan(self: *const ExpertStore, span_value: SourceSpan, dst: []u8) !void {
        if (span_value.file >= self.files.len or span_value.len != dst.len) return error.InvalidExpertRead;
        try readExact(self.files[span_value.file].fd, dst, span_value.offset);
    }

    pub fn readExpert(self: *const ExpertStore, layer: u16, expert: u16, dst: []u8) !void {
        if (layer >= self.geometry.layers or expert >= self.geometry.experts) return error.ExpertOutOfRange;
        const gate = self.span(layer, expert, .gate_up);
        const down = self.span(layer, expert, .down);
        if (dst.len != gate.len + down.len) return error.InvalidExpertRead;
        try self.readSpan(gate, dst[0..@intCast(gate.len)]);
        try self.readSpan(down, dst[@intCast(gate.len)..]);
    }
};

const TableRegion = struct {
    offset: u64,
    rows: u64,
    dim: u32,
};

fn tensor2dRegion(allocator: std.mem.Allocator, fd: std.c.fd_t, key: []const u8) !TableRegion {
    const region = io_mod.tensorRegion(allocator, fd, key) catch |err| return switch (err) {
        error.MissingSafetensorsTensor => error.MissingNgramTensor,
        error.SafetensorsTensorOutOfBounds => error.NgramTensorOutOfBounds,
        else => error.InvalidNgramTensor,
    };
    if (region.dtype != .bf16 or region.rank != 2 or region.shape[1] > std.math.maxInt(u32)) return error.InvalidNgramTensor;
    if (region.tensor_bytes != 2 * region.shape[0] * region.shape[1]) return error.InvalidNgramTensor;
    return .{ .offset = region.data_offset + region.tensor_offset, .rows = region.shape[0], .dim = @intCast(region.shape[1]) };
}

const TableShard = struct {
    file: u16,
    offset: u64,
};

const NgramKey = struct {
    layer: u16,
    shard: u16,
};

fn ngramKey(key: []const u8) !?NgramKey {
    const prefix = "model.language_model.layers.";
    const middle = ".ple.ple_embedding.ngram_embedding.shard_";
    const suffix = ".weight";
    if (!std.mem.startsWith(u8, key, prefix) or !std.mem.endsWith(u8, key, suffix)) return null;
    const body = key[prefix.len .. key.len - suffix.len];
    const split = std.mem.indexOf(u8, body, middle) orelse return null;
    const layer = std.fmt.parseInt(u16, body[0..split], 10) catch return error.InvalidNgramTensorKey;
    const shard = std.fmt.parseInt(u16, body[split + middle.len ..], 10) catch return error.InvalidNgramTensorKey;
    return .{ .layer = layer, .shard = shard };
}

pub const Bf16NgramStore = struct {
    allocator: std.mem.Allocator,
    files: []SourceFile,
    shards: []TableShard,
    rows_per_shard: u64,
    rows: u64,
    dim: u32,

    pub fn open(allocator: std.mem.Allocator, model_dir: []const u8) !Bf16NgramStore {
        const io = std.Io.Threaded.global_single_threaded.io();
        var dir = try std.Io.Dir.openDirAbsolute(io, model_dir, .{});
        defer dir.close(io);
        const index_raw = try dir.readFileAlloc(io, "model.safetensors.index.json", allocator, .limited(32 * 1024 * 1024));
        defer allocator.free(index_raw);
        const index_parsed = std.json.parseFromSlice(std.json.Value, allocator, index_raw, .{}) catch return error.InvalidSafetensorsIndex;
        defer index_parsed.deinit();
        if (index_parsed.value != .object) return error.InvalidSafetensorsIndex;
        const weight_map_value = index_parsed.value.object.get("weight_map") orelse return error.InvalidSafetensorsIndex;
        if (weight_map_value != .object) return error.InvalidSafetensorsIndex;
        const weight_map = weight_map_value.object;

        var table_layer: ?u16 = null;
        var table_shards: usize = 0;
        var max_shard: u16 = 0;
        var weight_it = weight_map.iterator();
        while (weight_it.next()) |entry| {
            const parsed_key = try ngramKey(entry.key_ptr.*) orelse continue;
            if (table_layer) |layer| {
                if (layer != parsed_key.layer) return error.MultipleNgramLayers;
            } else {
                table_layer = parsed_key.layer;
            }
            max_shard = @max(max_shard, parsed_key.shard);
            table_shards += 1;
        }
        const layer = table_layer orelse return error.MissingNgramTensor;
        const shard_count_usize = @as(usize, max_shard) + 1;
        if (table_shards != shard_count_usize or shard_count_usize > std.math.maxInt(u16)) return error.InvalidNgramShardCount;
        const shard_count: u16 = @intCast(shard_count_usize);

        var files_list: std.ArrayList(SourceFile) = .empty;
        errdefer {
            for (files_list.items) |file| {
                _ = std.c.close(file.fd);
                allocator.free(file.name);
            }
            files_list.deinit(allocator);
        }
        const shards = try allocator.alloc(TableShard, shard_count);
        errdefer allocator.free(shards);
        var rows_per_shard: u64 = 0;
        var dim: u32 = 0;
        var key_buf: [192]u8 = undefined;
        for (0..shard_count) |shard_usize| {
            const key = std.fmt.bufPrint(&key_buf, "model.language_model.layers.{d}.ple.ple_embedding.ngram_embedding.shard_{d}.weight", .{ layer, shard_usize }) catch return error.NameTooLong;
            const mapped = weight_map.get(key) orelse return error.MissingNgramTensor;
            if (mapped != .string) return error.InvalidSafetensorsIndex;
            var file_index: ?u16 = null;
            for (files_list.items, 0..) |file, i| {
                if (std.mem.eql(u8, file.name, mapped.string)) {
                    file_index = @intCast(i);
                    break;
                }
            }
            if (file_index == null) {
                const path = try std.fmt.allocPrintSentinel(allocator, "{s}/{s}", .{ model_dir, mapped.string }, 0);
                defer allocator.free(path);
                const fd = std.c.open(path.ptr, .{ .ACCMODE = .RDONLY }, @as(std.c.mode_t, 0));
                if (fd < 0) return error.MissingNgramShard;
                errdefer _ = std.c.close(fd);
                _ = std.c.fcntl(fd, std.c.F.NOCACHE, @as(c_int, 1));
                const name = try allocator.dupe(u8, mapped.string);
                errdefer allocator.free(name);
                try files_list.append(allocator, .{ .name = name, .fd = fd });
                file_index = @intCast(files_list.items.len - 1);
            }
            const region = try tensor2dRegion(allocator, files_list.items[file_index.?].fd, key);
            if (shard_usize == 0) {
                rows_per_shard = region.rows;
                dim = region.dim;
            } else if (region.rows != rows_per_shard or region.dim != dim) {
                return error.NgramShardGeometryMismatch;
            }
            shards[shard_usize] = .{ .file = file_index.?, .offset = region.offset };
        }
        const total_rows = std.math.mul(u64, rows_per_shard, shard_count) catch return error.InvalidNgramTensor;
        return .{ .allocator = allocator, .files = try files_list.toOwnedSlice(allocator), .shards = shards, .rows_per_shard = rows_per_shard, .rows = total_rows, .dim = dim };
    }

    pub fn deinit(self: *Bf16NgramStore) void {
        for (self.files) |file| {
            _ = std.c.close(file.fd);
            self.allocator.free(file.name);
        }
        self.allocator.free(self.files);
        self.allocator.free(self.shards);
        self.* = undefined;
    }

    pub fn gather(self: *const Bf16NgramStore, row_ids: []const i64, out: []f32) !void {
        if (out.len != row_ids.len * self.dim) return error.InvalidNgramOutput;
        const row_bytes: usize = @as(usize, self.dim) * 2;
        const raw = try self.allocator.alloc(u8, row_bytes);
        defer self.allocator.free(raw);
        for (row_ids, 0..) |row_signed, i| {
            try self.readRowBytes(row_signed, raw);
            try self.decodeRowBytes(raw, out[i * self.dim ..][0..self.dim]);
        }
    }

    pub fn readRowBytes(self: *const Bf16NgramStore, row_signed: i64, raw: []u8) !void {
        const row_bytes: usize = @as(usize, self.dim) * 2;
        if (raw.len != row_bytes or row_signed < 0) return error.NgramRowOutOfRange;
        const row: u64 = @intCast(row_signed);
        if (row >= self.rows) return error.NgramRowOutOfRange;
        const shard_index: usize = @intCast(row / self.rows_per_shard);
        const local = row % self.rows_per_shard;
        const shard = self.shards[shard_index];
        const offset = shard.offset + local * row_bytes;
        try readExact(self.files[shard.file].fd, raw, offset);
    }

    pub fn decodeRowBytes(self: *const Bf16NgramStore, raw: []const u8, out: []f32) !void {
        if (raw.len != @as(usize, self.dim) * 2 or out.len != self.dim) return error.InvalidNgramOutput;
        for (0..self.dim) |j| {
            const bits = std.mem.readInt(u16, raw[j * 2 ..][0..2], .little);
            out[j] = @bitCast(@as(u32, bits) << 16);
        }
    }
};

pub var slab_release_timeouts: std.atomic.Value(u64) = .init(0);

const SlabOperand = struct {
    slab: *io_mod.PageSlab,
    payload: *io_mod.ImportPayload,
    array: mlx.mlx_array,
    stride: usize,
    count: u32,

    fn create(allocator: std.mem.Allocator, count: u32, d0: u32, d1: u32) !SlabOperand {
        return createTyped(allocator, count, d0, d1, .bfloat16, 2);
    }

    fn createTyped(allocator: std.mem.Allocator, count: u32, d0: u32, d1: u32, dtype: mlx.mlx_dtype, elem_bytes: u8) !SlabOperand {
        if (count == 0 or d0 == 0 or d1 == 0 or elem_bytes == 0) return error.InvalidExpertGeometry;
        const stride = @as(usize, d0) * @as(usize, d1) * @as(usize, elem_bytes);
        const slab = try io_mod.PageSlab.create(allocator, stride * count);
        errdefer slab.destroy();
        const payload = try allocator.create(io_mod.ImportPayload);
        errdefer allocator.destroy(payload);
        payload.* = .{};
        const shape = [_]c_int{ @intCast(count), @intCast(d0), @intCast(d1) };
        const operand = try io_mod.importSlab(slab, &shape, dtype, payload, .{});
        if (!operand.aliased) {
            _ = mlx.mlx_array_free(operand.array);
            return error.ExpertSlabImportCopied;
        }
        return .{ .slab = slab, .payload = payload, .array = operand.array, .stride = stride, .count = count };
    }

    fn destroy(self: *SlabOperand, allocator: std.mem.Allocator, s: mlx.mlx_stream) void {
        _ = mlx.mlx_array_free(self.array);
        var attempt: usize = 0;
        while (self.payload.released.load(.acquire) == 0 and attempt < 32) : (attempt += 1) {
            _ = mlx.mlx_synchronize(s);
            _ = mlx.mlx_clear_cache();
            std.Thread.yield() catch {};
        }
        if (self.payload.released.load(.acquire) == 0) {
            const timeouts = slab_release_timeouts.fetchAdd(1, .monotonic) + 1;
            log.warn("[expert-stream] slab release timed out: {d} bytes stay mapped for MLX and are never freed (timeouts so far {d})\n", .{ self.slab.bytes.len, timeouts });
            self.* = undefined;
            return;
        }
        self.slab.destroy();
        allocator.destroy(self.payload);
        self.* = undefined;
    }

    fn slotBytes(self: *const SlabOperand, slot: usize) []u8 {
        return self.slab.bytes[slot * self.stride ..][0..self.stride];
    }

    fn borrow(self: *const SlabOperand) !mlx.mlx_array {
        var handle = mlx.mlx_array_new();
        errdefer _ = mlx.mlx_array_free(handle);
        try mlx.check(mlx.mlx_array_set(&handle, self.array));
        return handle;
    }
};

pub const Clock = struct {
    io: std.Io,
    start: std.Io.Timestamp,
    mark_ns: u64 = 0,

    pub fn init() Clock {
        const io = std.Io.Threaded.global_single_threaded.io();
        return .{ .io = io, .start = std.Io.Timestamp.now(io, .boot) };
    }

    pub fn lap(self: *Clock) u64 {
        const cumulative: u64 = @intCast(self.start.untilNow(self.io, .boot).nanoseconds);
        const delta = cumulative - self.mark_ns;
        self.mark_ns = cumulative;
        return delta;
    }
};

const HeldLease = struct {
    slab: *io_mod.PageSlab,
    lease: io_mod.Lease,
};

const CacheSnapshot = struct {
    expert_to_slot: []i32,
    slot_to_expert: []u16,
    ages: []u64,
    ready: []bool,
    tick: u64,

    fn restore(self: *const CacheSnapshot, cache: *GroupCache) void {
        @memcpy(cache.expert_to_slot, self.expert_to_slot);
        @memcpy(cache.slot_to_expert, self.slot_to_expert);
        @memcpy(cache.ages, self.ages);
        @memcpy(cache.ready, self.ready);
        cache.tick = self.tick;
    }

    fn deinit(self: *CacheSnapshot, allocator: std.mem.Allocator) void {
        allocator.free(self.expert_to_slot);
        allocator.free(self.slot_to_expert);
        allocator.free(self.ages);
        allocator.free(self.ready);
        self.* = undefined;
    }
};

fn abortFill(operand: *SlabOperand) void {
    _ = operand.slab.publish() catch {};
    operand.slab.retire() catch {};
}

const LayerStats = struct {
    groups: u64 = 0,
    union_members: u64 = 0,
    hits: u64 = 0,
    misses: u64 = 0,
    fill_bytes: u64 = 0,
    host_sync_ns: u64 = 0,
    fill_ns: u64 = 0,
    compute_ns: u64 = 0,
};

const LayerState = struct {
    cache: GroupCache,
    slabs: []SlabOperand = &.{},
    stats: LayerStats = .{},
};

fn createSlabSet(allocator: std.mem.Allocator, store: *const ExpertStore, count: u32, s: mlx.mlx_stream) ![]SlabOperand {
    const n = store.componentCount();
    const slabs = try allocator.alloc(SlabOperand, n);
    errdefer allocator.free(slabs);
    var made: usize = 0;
    errdefer for (slabs[0..made]) |*operand| operand.destroy(allocator, s);
    while (made < n) : (made += 1) {
        const spec = store.slabSpec(made);
        slabs[made] = try SlabOperand.createTyped(allocator, count, spec.rows, spec.cols, spec.dtype, spec.elem_bytes);
    }
    return slabs;
}

pub const Prepared = struct {
    allocator: std.mem.Allocator,
    gate: mlx.mlx_array = .{ .ctx = null },
    up: mlx.mlx_array = .{ .ctx = null },
    down: mlx.mlx_array = .{ .ctx = null },
    remapped: []u16,
    raw_gate_up: mlx.mlx_array = .{ .ctx = null },
    raw_down: mlx.mlx_array = .{ .ctx = null },
    quant_raw: [quant.component_count]mlx.mlx_array = @splat(.{ .ctx = null }),
    quantized: bool = false,
    workspace: bool = false,
    held: [quant.component_count]?HeldLease = @splat(null),
    defer_to: ?*Engine = null,

    pub fn quantOperand(self: *const Prepared, c: quant.Component) mlx.mlx_array {
        return self.quant_raw[@backingInt(c)];
    }

    pub fn deinit(self: *Prepared) void {
        if (self.gate.ctx != null) _ = mlx.mlx_array_free(self.gate);
        if (self.up.ctx != null) _ = mlx.mlx_array_free(self.up);
        if (self.down.ctx != null) _ = mlx.mlx_array_free(self.down);
        if (self.raw_gate_up.ctx != null) _ = mlx.mlx_array_free(self.raw_gate_up);
        if (self.raw_down.ctx != null) _ = mlx.mlx_array_free(self.raw_down);
        for (self.held) |entry| {
            const held = entry orelse continue;
            if (self.defer_to) |engine| {
                if (engine.deferReader(held)) continue;
            }
            held.slab.release(held.lease) catch {};
            held.slab.retire() catch {};
        }
        self.allocator.free(self.remapped);
        self.* = undefined;
    }
};

const FusedViews = struct {
    gate: mlx.mlx_array,
    up: mlx.mlx_array,
    down: mlx.mlx_array,
};

pub const Engine = struct {
    allocator: std.mem.Allocator,
    geometry: Geometry,
    plan: CachePlan,
    store: ExpertStore,
    layers: []LayerState,
    s: mlx.mlx_stream,
    forward_count: u64 = 0,
    io_pool: ?*io_mod.FillPool = null,
    fds: []std.c.fd_t = &.{},
    union_slabs: []SlabOperand = &.{},
    slab_imports: u64 = 0,
    fallback_imports: u64 = 0,
    fill_experts_total: u64 = 0,
    fill_bytes_total: u64 = 0,
    fill_ns_total: u64 = 0,
    forward_clock: Clock,
    last: Breakdown = .{},
    route_linear: RouteClass = .{},
    route_full: RouteClass = .{},
    pending: [PENDING_READERS_MAX]?HeldLease = @splat(null),

    pub fn pendingReaders(self: *const Engine) usize {
        var n: usize = 0;
        for (self.pending) |entry| {
            if (entry != null) n += 1;
        }
        return n;
    }

    pub fn drainPendingReaders(self: *Engine) void {
        for (&self.pending) |*entry| {
            const held = entry.* orelse continue;
            held.slab.release(held.lease) catch {};
            held.slab.retire() catch {};
            entry.* = null;
        }
    }

    fn deferReader(self: *Engine, held: HeldLease) bool {
        for (&self.pending) |*entry| {
            if (entry.* != null) continue;
            entry.* = held;
            return true;
        }
        return false;
    }

    pub fn init(allocator: std.mem.Allocator, model_dir: []const u8, geometry: Geometry, requested_bytes: u64, s: mlx.mlx_stream) !Engine {
        return initWithOptions(allocator, model_dir, geometry, requested_bytes, s, .{});
    }

    pub fn initWithOptions(allocator: std.mem.Allocator, model_dir: []const u8, geometry: Geometry, requested_bytes: u64, s: mlx.mlx_stream, opts: Options) !Engine {
        var store = try ExpertStore.openLayout(allocator, model_dir, geometry, opts.layout);
        errdefer store.deinit();
        const plan = try cachePlanBytes(requested_bytes, geometry.layers, geometry.experts, store.perExpertBytes());
        var engine = Engine{
            .allocator = allocator,
            .geometry = geometry,
            .plan = plan,
            .store = store,
            .layers = &.{},
            .s = s,
            .forward_clock = Clock.init(),
        };
        errdefer engine.releaseSlabs();
        const fds = try allocator.alloc(std.c.fd_t, engine.store.fileCount());
        for (fds, 0..) |*fd, i| fd.* = engine.store.fdAt(i);
        engine.fds = fds;
        engine.io_pool = try io_mod.FillPool.create(allocator, .{
            .workers = opts.io_workers,
            .bounce_cap = opts.bounce_size,
            .coalesce_max = COALESCE_MAX,
        });
        const before_fallback = io_mod.fallback_imports.load(.monotonic);
        const layers = try allocator.alloc(LayerState, geometry.layers);
        for (layers) |*layer| layer.* = .{ .cache = undefined };
        engine.layers = layers;
        var initialized: usize = 0;
        errdefer {
            for (layers[0..initialized]) |*layer| layer.cache.deinit();
            engine.layers = &.{};
            allocator.free(layers);
        }
        for (layers) |*layer| {
            var cache = try GroupCache.init(allocator, plan.slots_per_layer, geometry.experts);
            errdefer cache.deinit();
            layer.* = .{ .cache = cache };
            initialized += 1;
            layer.slabs = try createSlabSet(allocator, &engine.store, plan.slots_per_layer, s);
            engine.slab_imports += layer.slabs.len;
        }
        engine.union_slabs = try createSlabSet(allocator, &engine.store, geometry.experts, s);
        engine.slab_imports += engine.union_slabs.len;
        engine.fallback_imports = io_mod.fallback_imports.load(.monotonic) - before_fallback;
        log.info("[expert-stream] cache {d:.3} GB, {d} slots/layer, workspace {d:.3} GB, bounce {d:.3} GB, fallback_imports={d}\n", .{
            @as(f64, @floatFromInt(plan.cache_bytes)) / 1e9,
            plan.slots_per_layer,
            @as(f64, @floatFromInt(plan.workspace_bytes)) / 1e9,
            @as(f64, @floatFromInt(plan.bounce_bytes)) / 1e9,
            engine.fallback_imports,
        });
        return engine;
    }

    fn releaseSlabs(self: *Engine) void {
        for (self.layers) |*layer| {
            for (layer.slabs) |*operand| operand.destroy(self.allocator, self.s);
            if (layer.slabs.len != 0) self.allocator.free(layer.slabs);
            layer.slabs = &.{};
        }
        for (self.union_slabs) |*operand| operand.destroy(self.allocator, self.s);
        if (self.union_slabs.len != 0) self.allocator.free(self.union_slabs);
        self.union_slabs = &.{};
        if (self.io_pool) |pool| pool.destroy();
        self.io_pool = null;
        if (self.fds.len != 0) self.allocator.free(self.fds);
        self.fds = &.{};
    }

    pub fn deinit(self: *Engine) void {
        self.drainPendingReaders();
        self.releaseSlabs();
        for (self.layers) |*layer| layer.cache.deinit();
        self.allocator.free(self.layers);
        self.store.deinit();
        self.* = undefined;
    }

    pub fn cacheSlotBytes(self: *const Engine, layer: u16, slot: u16, component: Component) []u8 {
        return self.cacheSlotBytesAt(layer, slot, @backingInt(component));
    }

    pub fn cacheSlotBytesAt(self: *const Engine, layer: u16, slot: u16, ci: usize) []u8 {
        return self.layers[layer].slabs[ci].slotBytes(slot);
    }

    pub fn unionSlotBytes(self: *const Engine, position: u16, component: Component) []u8 {
        return self.unionSlotBytesAt(position, @backingInt(component));
    }

    pub fn unionSlotBytesAt(self: *const Engine, position: u16, ci: usize) []u8 {
        return self.union_slabs[ci].slotBytes(position);
    }

    pub fn slotReady(self: *const Engine, layer: u16, slot: u16) bool {
        return self.layers[layer].cache.ready[slot];
    }

    pub fn noteRoute(self: *Engine, layer: u16, full_attn: bool, detail: RouteDetail) void {
        if (layer < self.layers.len) self.layers[layer].stats.host_sync_ns +|= detail.totalNs();
        if (full_attn) self.route_full.add(detail) else self.route_linear.add(detail);
    }

    pub fn routeProbeActive(self: *const Engine) bool {
        const next = self.forward_count + 1;
        return next <= 3 or next % BREAKDOWN_EVERY == BREAKDOWN_EVERY / 2;
    }

    pub fn noteExpertCompute(self: *Engine, layer: u16, nanoseconds: u64) void {
        if (layer < self.layers.len) self.layers[layer].stats.compute_ns += nanoseconds;
    }

    pub fn layerHits(self: *const Engine, layer: u16) u64 {
        return if (layer < self.layers.len) self.layers[layer].stats.hits else 0;
    }

    pub fn layerUnionMembers(self: *const Engine, layer: u16) u64 {
        return if (layer < self.layers.len) self.layers[layer].stats.union_members else 0;
    }

    pub fn beginForward(self: *Engine) void {
        for (self.layers) |*layer| layer.stats = .{};
        self.route_linear = .{};
        self.route_full = .{};
        self.forward_clock = Clock.init();
    }

    pub fn breakdown(self: *const Engine) Breakdown {
        return self.last;
    }

    pub fn finishForward(self: *Engine, rows: usize) void {
        self.forward_count += 1;
        const wall_ns = self.forward_clock.lap();
        var acc = Breakdown{
            .forward = self.forward_count,
            .rows = rows,
            .wall_ns = wall_ns,
            .route_linear = self.route_linear,
            .route_full = self.route_full,
            .probe = self.route_full.x_wait_ns + self.route_linear.x_wait_ns > 0,
        };
        for (self.layers) |layer| {
            acc.fill_bytes += layer.stats.fill_bytes;
            acc.route_ns += layer.stats.host_sync_ns;
            acc.fill_ns += layer.stats.fill_ns;
            acc.compute_ns += layer.stats.compute_ns;
            acc.hits += layer.stats.hits;
            acc.union_members += layer.stats.union_members;
        }
        self.last = acc;
        const scheduled = self.forward_count <= 2 or self.forward_count % BREAKDOWN_EVERY == 0;
        if (!scheduled and !acc.probe) return;
        const fill_seconds = @as(f64, @floatFromInt(acc.fill_ns)) / 1e9;
        log.info("[expert-stream] forward={d} rows={d} wall_ms={d:.1} route_ms={d:.1} fill_ms={d:.1} compute_ms={d:.1} other_ms={d:.1} fill_gb={d:.3} fill_gbps={d:.2} fill_bytes_per_row={d} hits={d}/{d} slab_leaks={d}\n", .{
            acc.forward,
            acc.rows,
            @as(f64, @floatFromInt(acc.wall_ns)) / 1e6,
            @as(f64, @floatFromInt(acc.route_ns)) / 1e6,
            @as(f64, @floatFromInt(acc.fill_ns)) / 1e6,
            @as(f64, @floatFromInt(acc.compute_ns)) / 1e6,
            @as(f64, @floatFromInt(acc.otherNs())) / 1e6,
            @as(f64, @floatFromInt(acc.fill_bytes)) / 1e9,
            if (fill_seconds > 0) @as(f64, @floatFromInt(acc.fill_bytes)) / 1e9 / fill_seconds else 0,
            acc.fill_bytes / @max(rows, 1),
            acc.hits,
            acc.union_members,
            slab_release_timeouts.load(.monotonic),
        });
        log.info("[expert-stream] route forward={d} rows={d} probe={} linear n={d} build_ms={d:.2} xwait_ms={d:.2} idswait_ms={d:.2} read_ms={d:.2} full n={d} build_ms={d:.2} xwait_ms={d:.2} idswait_ms={d:.2} read_ms={d:.2}\n", .{
            acc.forward,
            acc.rows,
            acc.probe,
            acc.route_linear.layers,
            @as(f64, @floatFromInt(acc.route_linear.build_ns)) / 1e6,
            @as(f64, @floatFromInt(acc.route_linear.x_wait_ns)) / 1e6,
            @as(f64, @floatFromInt(acc.route_linear.ids_wait_ns)) / 1e6,
            @as(f64, @floatFromInt(acc.route_linear.read_ns)) / 1e6,
            acc.route_full.layers,
            @as(f64, @floatFromInt(acc.route_full.build_ns)) / 1e6,
            @as(f64, @floatFromInt(acc.route_full.x_wait_ns)) / 1e6,
            @as(f64, @floatFromInt(acc.route_full.ids_wait_ns)) / 1e6,
            @as(f64, @floatFromInt(acc.route_full.read_ns)) / 1e6,
        });
    }

    fn splitViews(self: *Engine, gate_up: mlx.mlx_array, down_raw: mlx.mlx_array, slots: c_int) !FusedViews {
        const inter: c_int = @intCast(self.geometry.intermediate);
        const hidden: c_int = @intCast(self.geometry.hidden);
        const strides = [_]c_int{ 1, 1, 1 };
        const axes = [_]c_int{ 0, 2, 1 };
        var gate_raw = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(gate_raw);
        try mlx.check(mlx.mlx_slice(&gate_raw, gate_up, &[_]c_int{ 0, 0, 0 }, 3, &[_]c_int{ slots, inter, hidden }, 3, &strides, 3, self.s));
        var up_raw = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(up_raw);
        try mlx.check(mlx.mlx_slice(&up_raw, gate_up, &[_]c_int{ 0, inter, 0 }, 3, &[_]c_int{ slots, 2 * inter, hidden }, 3, &strides, 3, self.s));
        var gate = mlx.mlx_array_new();
        errdefer _ = mlx.mlx_array_free(gate);
        try mlx.check(mlx.mlx_transpose_axes(&gate, gate_raw, &axes, 3, self.s));
        var up = mlx.mlx_array_new();
        errdefer _ = mlx.mlx_array_free(up);
        try mlx.check(mlx.mlx_transpose_axes(&up, up_raw, &axes, 3, self.s));
        var down = mlx.mlx_array_new();
        errdefer _ = mlx.mlx_array_free(down);
        try mlx.check(mlx.mlx_transpose_axes(&down, down_raw, &axes, 3, self.s));
        return .{ .gate = gate, .up = up, .down = down };
    }

    pub fn prepareHost(self: *Engine, layer_index: u16, occurrences: []const u16) !Prepared {
        if (layer_index >= self.layers.len) return error.ExpertLayerOutOfRange;
        self.drainPendingReaders();
        return self.prepareSlab(layer_index, occurrences);
    }

    fn snapshotCache(self: *Engine, cache: *const GroupCache) !CacheSnapshot {
        return .{
            .expert_to_slot = try self.allocator.dupe(i32, cache.expert_to_slot),
            .slot_to_expert = try self.allocator.dupe(u16, cache.slot_to_expert),
            .ages = try self.allocator.dupe(u64, cache.ages),
            .ready = try self.allocator.dupe(bool, cache.ready),
            .tick = cache.tick,
        };
    }

    fn runFill(self: *Engine, layer_index: u16, spans: []const io_mod.FillSpan) !void {
        if (spans.len == 0) return;
        const pool = self.io_pool orelse return error.MissingFillPool;
        var plan = try io_mod.FillPlan.init(self.allocator, spans, COALESCE_MAX, pool.opts.workers, false);
        defer plan.deinit();
        var bytes: u64 = 0;
        for (spans) |span_value| bytes += span_value.len;
        var clock = Clock.init();
        try pool.submit(self.fds, &plan);
        const outcome = pool.join();
        const elapsed = clock.lap();
        self.fill_ns_total += elapsed;
        self.layers[layer_index].stats.fill_ns += elapsed;
        try outcome;
        self.fill_bytes_total += bytes;
    }

    pub fn cancelFills(self: *Engine) void {
        if (self.io_pool) |pool| pool.requestCancel();
    }

    fn prepareSlab(self: *Engine, layer_index: u16, occurrences: []const u16) !Prepared {
        const layer = &self.layers[layer_index];
        const n = self.store.componentCount();
        if (layer.slabs.len != n) return error.MissingExpertSlab;
        var snapshot = try self.snapshotCache(&layer.cache);
        defer snapshot.deinit(self.allocator);
        var committed = false;
        var wrote_cache = false;
        errdefer if (!committed) {
            if (wrote_cache) layer.cache.invalidateUnready() else snapshot.restore(&layer.cache);
        };

        var resolution = try layer.cache.resolve(self.allocator, occurrences);
        defer resolution.deinit(self.allocator);
        const use_union = resolution.workspace;
        if (use_union and self.union_slabs.len != n) return error.MissingExpertSlab;
        const read_set = if (use_union) self.union_slabs else layer.slabs;

        var read_begun: usize = 0;
        errdefer if (!committed) for (read_set[0..read_begun]) |*operand| abortFill(operand);
        while (read_begun < n) : (read_begun += 1) _ = try read_set[read_begun].slab.tryBeginFill();
        var cache_begun: usize = 0;
        errdefer if (!committed) for (layer.slabs[0..cache_begun]) |*operand| abortFill(operand);
        if (use_union) {
            while (cache_begun < n) : (cache_begun += 1) _ = try layer.slabs[cache_begun].slab.tryBeginFill();
        }

        var spans: std.ArrayList(io_mod.FillSpan) = .empty;
        defer spans.deinit(self.allocator);
        var misses: u64 = 0;
        for (resolution.bindings, 0..) |binding, position| {
            if (binding.hit and use_union) {
                const slot = binding.slot orelse return error.ExpertBindingMissing;
                for (0..n) |ci| @memcpy(read_set[ci].slotBytes(position), layer.slabs[ci].slotBytes(slot));
                continue;
            }
            if (binding.hit) continue;
            misses += 1;
            for (0..n) |ci| {
                const source = self.store.spanAt(layer_index, binding.expert, ci);
                const dst = if (use_union)
                    read_set[ci].slotBytes(position)
                else
                    layer.slabs[ci].slotBytes(binding.slot orelse return error.ExpertBindingMissing);
                try spans.append(self.allocator, .{ .file = source.file, .offset = source.offset, .len = source.len, .dst = dst.ptr });
            }
        }
        if (!use_union and spans.items.len > 0) wrote_cache = true;
        try self.runFill(layer_index, spans.items);
        self.fill_experts_total += misses;

        if (use_union) {
            for (resolution.bindings, 0..) |binding, position| {
                if (binding.hit) continue;
                const slot = binding.slot orelse continue;
                wrote_cache = true;
                for (0..n) |ci| @memcpy(layer.slabs[ci].slotBytes(slot), read_set[ci].slotBytes(position));
                layer.cache.markReady(slot);
            }
            for (layer.slabs) |*operand| {
                _ = try operand.slab.publish();
                try operand.slab.retire();
            }
        } else {
            for (resolution.bindings) |binding| {
                if (binding.hit) continue;
                layer.cache.markReady(binding.slot.?);
            }
        }

        var held: [quant.component_count]?HeldLease = @splat(null);
        var leased: usize = 0;
        errdefer for (held[0..leased]) |entry| {
            const taken = entry orelse continue;
            taken.slab.release(taken.lease) catch {};
            taken.slab.retire() catch {};
        };
        while (leased < n) : (leased += 1) {
            _ = try read_set[leased].slab.publish();
            held[leased] = .{ .slab = read_set[leased].slab, .lease = try read_set[leased].slab.lease() };
        }

        const remapped = try self.allocator.alloc(u16, occurrences.len);
        errdefer self.allocator.free(remapped);
        if (use_union) {
            @memcpy(remapped, resolution.remapped);
        } else {
            for (occurrences, 0..) |expert, i| {
                const raw = layer.cache.expert_to_slot[expert];
                if (raw < 0) return error.ExpertBindingMissing;
                remapped[i] = @intCast(raw);
            }
        }

        const quantized = self.store.quantized != null;
        var raws: [quant.component_count]mlx.mlx_array = @splat(.{ .ctx = null });
        var views: FusedViews = .{ .gate = .{ .ctx = null }, .up = .{ .ctx = null }, .down = .{ .ctx = null } };
        if (quantized) {
            for (0..n) |ci| raws[ci] = read_set[ci].array;
        } else {
            raws[0] = try read_set[0].borrow();
            errdefer _ = mlx.mlx_array_free(raws[0]);
            raws[1] = try read_set[1].borrow();
            errdefer _ = mlx.mlx_array_free(raws[1]);
            views = try self.splitViews(raws[0], raws[1], @intCast(read_set[0].count));
        }

        layer.stats.groups += 1;
        layer.stats.union_members += resolution.union_ids.len;
        layer.stats.hits += resolution.hits;
        layer.stats.misses += resolution.misses;
        layer.stats.fill_bytes += misses * self.plan.expert_bytes;
        committed = true;
        return .{
            .allocator = self.allocator,
            .gate = views.gate,
            .up = views.up,
            .down = views.down,
            .remapped = remapped,
            .raw_gate_up = if (quantized) .{ .ctx = null } else raws[0],
            .raw_down = if (quantized) .{ .ctx = null } else raws[1],
            .quant_raw = if (quantized) raws else @splat(.{ .ctx = null }),
            .quantized = quantized,
            .workspace = resolution.workspace,
            .held = held,
            .defer_to = self,
        };
    }
};

test "expert stream span adapter derives fused first middle and last slices" {
    const t = std.testing;
    const layout = TensorLayout{ .data_offset = 4096, .tensor_offset = 8192, .tensor_bytes = 300, .experts = 3 };
    try t.expectEqual(Span{ .offset = 12288, .len = 100 }, try layout.expertSpan(0));
    try t.expectEqual(Span{ .offset = 12388, .len = 100 }, try layout.expertSpan(1));
    try t.expectEqual(Span{ .offset = 12488, .len = 100 }, try layout.expertSpan(2));
    try t.expectError(error.ExpertOutOfRange, layout.expertSpan(3));
}

fn markResolvedReady(cache: *GroupCache, resolution: *const GroupResolution) void {
    for (resolution.bindings) |binding| {
        if (binding.hit) continue;
        if (binding.slot) |slot| cache.markReady(slot);
    }
}

test "expert stream group exact LRU matches replay counts" {
    const t = std.testing;
    var cache = try GroupCache.init(t.allocator, 2, 4);
    defer cache.deinit();

    var r1 = try cache.resolve(t.allocator, &.{ 0, 1 });
    defer r1.deinit(t.allocator);
    markResolvedReady(&cache, &r1);
    try t.expectEqual(@as(usize, 0), r1.hits);
    try t.expectEqual(@as(usize, 2), r1.misses);
    try t.expectEqual(@as(usize, 0), r1.declined);

    var r2 = try cache.resolve(t.allocator, &.{ 0, 2 });
    defer r2.deinit(t.allocator);
    markResolvedReady(&cache, &r2);
    try t.expectEqual(@as(usize, 1), r2.hits);
    try t.expectEqual(@as(usize, 1), r2.misses);
    try t.expectEqual(@as(usize, 0), r2.declined);

    var r3 = try cache.resolve(t.allocator, &.{ 1, 0, 2 });
    defer r3.deinit(t.allocator);
    try t.expectEqual(@as(usize, 2), r3.hits);
    try t.expectEqual(@as(usize, 1), r3.misses);
    try t.expectEqual(@as(usize, 1), r3.declined);
    try t.expect(r3.workspace);
    try t.expectEqualSlices(u16, &.{ 1, 0, 2 }, r3.union_ids);
    try t.expectEqualSlices(u16, &.{ 0, 1, 2 }, r3.remapped);
}

test "expert stream union orders experts by last occurrence and remaps rows" {
    const t = std.testing;
    var cache = try GroupCache.init(t.allocator, 4, 8);
    defer cache.deinit();
    var result = try cache.resolve(t.allocator, &.{ 1, 2, 1, 3, 2 });
    defer result.deinit(t.allocator);
    markResolvedReady(&cache, &result);
    try t.expectEqualSlices(u16, &.{ 1, 3, 2 }, result.union_ids);
    try t.expectEqualSlices(u16, &.{ 0, 2, 0, 1, 2 }, result.remapped);
    try t.expect(!result.workspace);

    var reordered = try cache.resolve(t.allocator, &.{ 1, 2, 3 });
    defer reordered.deinit(t.allocator);
    try t.expectEqualSlices(u16, &.{ 1, 2, 3 }, reordered.union_ids);
    try t.expectEqualSlices(u16, &.{ 0, 1, 2 }, reordered.remapped);
    try t.expectEqual(@as(usize, 3), reordered.hits);
}

test "expert stream cache plan uses decimal gigabytes and reserves full workspace" {
    const t = std.testing;
    const p = try cachePlanBytes(60_000_000_000, 48, 512, try expertBytes(1280, 2560, 640));
    try t.expectEqual(@as(u16, 127), p.slots_per_layer);
    try t.expectEqual(@as(u64, 9_830_400), p.expert_bytes);
    try t.expectEqual(@as(u64, 48 * 127 * 9_830_400), p.cache_bytes);
    try t.expectEqual(@as(u64, 512 * 9_830_400), p.workspace_bytes);
    try t.expectEqual(@as(u64, 8 * 64 * 1024 * 1024), p.bounce_bytes);
}

test "expert stream refuses MTP by name, at load and at request parse" {
    const t = std.testing;
    try t.expect(mtpRefusal(false, true) == null);
    try t.expect(mtpRefusal(true, false) == null);
    try t.expect(mtpRefusal(false, false) == null);
    const why = mtpRefusal(true, true) orelse return error.TestExpectedRefusal;
    try t.expect(std.mem.indexOf(u8, why, "expert streaming") != null);
}

test "a capable checkpoint streams only when it must or when a budget was asked for" {
    const t = std.testing;
    try t.expect(!expertStreamingEngaged(false, false, 0, 0));
    try t.expect(!expertStreamingEngaged(false, true, 0, 60 << 30));
    // The dense HF checkpoint cannot load resident: it streams with no budget.
    try t.expect(expertStreamingEngaged(true, true, 0, 0));
    // A quantized pack with neither flag loads resident, exactly as before.
    try t.expect(!expertStreamingEngaged(true, false, 0, 0));
    try t.expect(expertStreamingEngaged(true, false, 0, 50 << 30));
    try t.expect(expertStreamingEngaged(true, false, 60_000_000_000, 0));
}

test "expert stream ssd budget ledger derives the cache and refuses by name" {
    const t = std.testing;
    const GiB: u64 = 1 << 30;
    const expert_bytes: u64 = 9_830_400;
    const bounce: u64 = 8 * 64 * 1024 * 1024;
    const trunk: u64 = 9_900_000_000;

    const led = try budgetLedger(100 * GiB, trunk, 0, 48, 512, 10, expert_bytes, bounce);
    try t.expectEqual(@as(u64, 512 * expert_bytes), led.workspace_bytes);
    try t.expectEqual(@as(u64, 10 * expert_bytes), led.selected_bytes);
    try t.expectEqual(@as(u64, 0), led.mtp_bytes);
    try t.expectEqual(trunk, led.trunk_bytes);
    try t.expectEqual(bounce, led.bounce_bytes);
    try t.expectEqual(@as(u16, 194), led.slots_per_layer);
    try t.expectEqual(@as(u64, 194) * 48 * expert_bytes, led.cache_bytes);

    const with_mtp = try budgetLedger(100 * GiB, trunk, 5_200_000_000, 48, 512, 10, expert_bytes, bounce);
    try t.expectEqual(@as(u16, 183), with_mtp.slots_per_layer);

    try t.expectError(error.SsdBudgetBelowResident, budgetLedger(14 * GiB, trunk, 0, 48, 512, 10, expert_bytes, bounce));
    const fixed = trunk + 512 * expert_bytes + 10 * expert_bytes + bounce;
    const per_slot = 48 * expert_bytes;
    try t.expectError(error.SsdBudgetBelowResident, budgetLedger(fixed + 2 * per_slot - 1, trunk, 0, 48, 512, 10, expert_bytes, bounce));
    const two = try budgetLedger(fixed + 2 * per_slot, trunk, 0, 48, 512, 10, expert_bytes, bounce);
    try t.expectEqual(@as(u16, 2), two.slots_per_layer);

    try t.expect(!budgetOverriddenByExplicitCache(0, 100 * GiB));
    try t.expect(!budgetOverriddenByExplicitCache(60_000_000_000, 0));
    try t.expect(budgetOverriddenByExplicitCache(60_000_000_000, 100 * GiB));
}

test "expert stream prefill peak bill covers a 501 expert union and k127 cache layer" {
    const t = std.testing;
    const p = try cachePlanBytes(60_000_000_000, 48, 512, try expertBytes(1280, 2560, 640));
    const peak = fillPeakBytes(p.expert_bytes, 501, p.slots_per_layer, p.bounce_bytes);
    try t.expectEqual(@as(u64, 127 * 9_830_400), p.layer_bytes);
    try t.expect(p.prefill_peak_bytes >= peak);
}

test "expert store maps fused safetensors spans and reads identical bytes" {
    const t = std.testing;
    const io = t.io;
    var tmp = t.tmpDir(.{});
    defer tmp.cleanup();
    const header = "{\"model.language_model.layers.0.mlp.experts.gate_up_proj\":{\"dtype\":\"BF16\",\"shape\":[3,2,2],\"data_offsets\":[0,24]},\"model.language_model.layers.0.mlp.experts.down_proj\":{\"dtype\":\"BF16\",\"shape\":[3,2,1],\"data_offsets\":[24,36]}}";
    const file_bytes = try t.allocator.alloc(u8, 8 + header.len + 36);
    defer t.allocator.free(file_bytes);
    std.mem.writeInt(u64, file_bytes[0..8], header.len, .little);
    @memcpy(file_bytes[8 .. 8 + header.len], header);
    for (file_bytes[8 + header.len ..], 0..) |*b, i| b.* = @intCast(i);
    try tmp.dir.writeFile(io, .{ .sub_path = "model-00001-of-00001.safetensors", .data = file_bytes });
    try tmp.dir.writeFile(io, .{
        .sub_path = "model.safetensors.index.json",
        .data = "{\"weight_map\":{\"model.language_model.layers.0.mlp.experts.gate_up_proj\":\"model-00001-of-00001.safetensors\",\"model.language_model.layers.0.mlp.experts.down_proj\":\"model-00001-of-00001.safetensors\"}}",
    });
    var path_buf: [std.fs.max_path_bytes]u8 = undefined;
    const path_len = try tmp.dir.realPath(io, &path_buf);
    var store = try ExpertStore.open(t.allocator, path_buf[0..path_len], .{ .layers = 1, .experts = 3, .hidden = 2, .intermediate = 1 });
    defer store.deinit();
    const data_offset: u64 = 8 + header.len;
    try t.expectEqual(SourceSpan{ .file = 0, .offset = data_offset, .len = 8 }, store.span(0, 0, .gate_up));
    try t.expectEqual(SourceSpan{ .file = 0, .offset = data_offset + 16, .len = 8 }, store.span(0, 2, .gate_up));
    try t.expectEqual(SourceSpan{ .file = 0, .offset = data_offset + 24 + 8, .len = 4 }, store.span(0, 2, .down));
    var got: [12]u8 = undefined;
    try store.readExpert(0, 1, &got);
    try t.expectEqualSlices(u8, file_bytes[8 + header.len + 8 .. 8 + header.len + 16], got[0..8]);
    try t.expectEqualSlices(u8, file_bytes[8 + header.len + 28 .. 8 + header.len + 32], got[8..12]);
}

test "bf16 ngram store gathers rows across safetensors shards" {
    const t = std.testing;
    const io = t.io;
    var tmp = t.tmpDir(.{});
    defer tmp.cleanup();
    const key0 = "model.language_model.layers.3.ple.ple_embedding.ngram_embedding.shard_0.weight";
    const key1 = "model.language_model.layers.3.ple.ple_embedding.ngram_embedding.shard_1.weight";
    const header = try std.fmt.allocPrint(t.allocator, "{{\"{s}\":{{\"dtype\":\"BF16\",\"shape\":[2,2],\"data_offsets\":[0,8]}},\"{s}\":{{\"dtype\":\"BF16\",\"shape\":[2,2],\"data_offsets\":[8,16]}}}}", .{ key0, key1 });
    defer t.allocator.free(header);
    const file_bytes = try t.allocator.alloc(u8, 8 + header.len + 16);
    defer t.allocator.free(file_bytes);
    std.mem.writeInt(u64, file_bytes[0..8], header.len, .little);
    @memcpy(file_bytes[8 .. 8 + header.len], header);
    const values = [_]u16{ 0x3f80, 0x4000, 0x4040, 0x4080, 0x40a0, 0x40c0, 0x40e0, 0x4100 };
    @memcpy(file_bytes[8 + header.len ..], std.mem.sliceAsBytes(&values));
    try tmp.dir.writeFile(io, .{ .sub_path = "table.safetensors", .data = file_bytes });
    const index = try std.fmt.allocPrint(t.allocator, "{{\"weight_map\":{{\"{s}\":\"table.safetensors\",\"{s}\":\"table.safetensors\"}}}}", .{ key0, key1 });
    defer t.allocator.free(index);
    try tmp.dir.writeFile(io, .{ .sub_path = "model.safetensors.index.json", .data = index });
    var path_buf: [std.fs.max_path_bytes]u8 = undefined;
    const path_len = try tmp.dir.realPath(io, &path_buf);
    var store = try Bf16NgramStore.open(t.allocator, path_buf[0..path_len]);
    defer store.deinit();
    try t.expectEqual(@as(u64, 4), store.rows);
    try t.expectEqual(@as(u32, 2), store.dim);
    var out: [4]f32 = undefined;
    try store.gather(&.{ 0, 3 }, &out);
    try t.expectEqualSlices(f32, &.{ 1.0, 2.0, 7.0, 8.0 }, &out);
}

test "expert stream slab operand preserves direct gather output" {
    const t = std.testing;
    const sandboxed = std.c.getenv("CODEX_SANDBOX") != null;
    const io = t.io;
    var tmp = t.tmpDir(.{});
    defer tmp.cleanup();
    const gate_key = "model.language_model.layers.0.mlp.experts.gate_up_proj";
    const down_key = "model.language_model.layers.0.mlp.experts.down_proj";
    const header = try std.fmt.allocPrint(t.allocator, "{{\"{s}\":{{\"dtype\":\"BF16\",\"shape\":[2,2,2],\"data_offsets\":[0,16]}},\"{s}\":{{\"dtype\":\"BF16\",\"shape\":[2,2,1],\"data_offsets\":[16,24]}}}}", .{ gate_key, down_key });
    defer t.allocator.free(header);
    const file_bytes = try t.allocator.alloc(u8, 8 + header.len + 24);
    defer t.allocator.free(file_bytes);
    std.mem.writeInt(u64, file_bytes[0..8], header.len, .little);
    @memcpy(file_bytes[8 .. 8 + header.len], header);
    const weights = [_]u16{ 0x3f80, 0x4000, 0x4040, 0x4080, 0x4100, 0x4110, 0x4120, 0x4130, 0x40a0, 0x40c0, 0x4140, 0x4150 };
    @memcpy(file_bytes[8 + header.len ..], std.mem.sliceAsBytes(&weights));
    try tmp.dir.writeFile(io, .{ .sub_path = "experts.safetensors", .data = file_bytes });
    const index = try std.fmt.allocPrint(t.allocator, "{{\"weight_map\":{{\"{s}\":\"experts.safetensors\",\"{s}\":\"experts.safetensors\"}}}}", .{ gate_key, down_key });
    defer t.allocator.free(index);
    try tmp.dir.writeFile(io, .{ .sub_path = "model.safetensors.index.json", .data = index });
    var path_buf: [std.fs.max_path_bytes]u8 = undefined;
    const path_len = try tmp.dir.realPath(io, &path_buf);
    const s = if (sandboxed) mlx.mlx_default_cpu_stream_new() else mlx.gpuStream();
    defer if (sandboxed) {
        _ = mlx.mlx_stream_free(s);
    };
    var engine = try Engine.initWithOptions(t.allocator, path_buf[0..path_len], .{ .layers = 1, .experts = 2, .hidden = 2, .intermediate = 1 }, 12, s, .{ .bounce_size = 1 << 20 });
    defer engine.deinit();
    {
        var prepared = try engine.prepareHost(0, &.{0});
        defer prepared.deinit();
        const x_data = [_]u16{ 0x3f80, 0x3f80 };
        const x = mlx.mlx_array_new_data(&x_data, &[_]c_int{ 1, 1, 1, 1, 2 }, 5, .bfloat16);
        defer _ = mlx.mlx_array_free(x);
        const ids_data = [_]u32{0};
        const ids = mlx.mlx_array_new_data(&ids_data, &[_]c_int{ 1, 1, 1 }, 3, .uint32);
        defer _ = mlx.mlx_array_free(ids);
        const no_idx = mlx.mlx_array{ .ctx = null };
        var streamed = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(streamed);
        try mlx.check(mlx.mlx_gather_mm(&streamed, x, prepared.gate, no_idx, ids, false, s));
        const direct_raw = mlx.mlx_array_new_data(std.mem.sliceAsBytes(weights[0..2]).ptr, &[_]c_int{ 1, 1, 2 }, 3, .bfloat16);
        defer _ = mlx.mlx_array_free(direct_raw);
        var direct_w = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(direct_w);
        try mlx.check(mlx.mlx_transpose_axes(&direct_w, direct_raw, &[_]c_int{ 0, 2, 1 }, 3, s));
        var direct = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(direct);
        try mlx.check(mlx.mlx_gather_mm(&direct, x, direct_w, no_idx, ids, false, s));
        try mlx.check(mlx.mlx_array_eval(streamed));
        try mlx.check(mlx.mlx_array_eval(direct));
        const direct_bytes = mlx.mlx_array_size(direct) * @sizeOf(u16);
        const streamed_bytes = mlx.mlx_array_size(streamed) * @sizeOf(u16);
        try t.expectEqual(direct_bytes, streamed_bytes);
        try t.expectEqualSlices(u8, @as([*]const u8, @ptrCast(mlx.mlx_array_data_bfloat16(direct).?))[0..direct_bytes], @as([*]const u8, @ptrCast(mlx.mlx_array_data_bfloat16(streamed).?))[0..streamed_bytes]);
    }
    var workspace = try engine.prepareHost(0, &.{ 0, 1 });
    defer workspace.deinit();
    try t.expect(workspace.workspace);
    try t.expectEqualSlices(u16, &.{ 0, 1 }, workspace.remapped);
    var workspace_gate = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(workspace_gate);
    try mlx.check(mlx.mlx_contiguous(&workspace_gate, workspace.gate, false, s));
    try mlx.check(mlx.mlx_array_eval(workspace_gate));
    try t.expectEqualSlices(u16, &.{ 0x3f80, 0x4000, 0x4100, 0x4110 }, mlx.mlx_array_data_bfloat16(workspace_gate).?[0..4]);
}

test "real qwen expert store spans and source bytes are exact" {
    const t = std.testing;
    const path = "/Users/beam/llm/models/Qwen/Qwen3.8-Flash-Next";
    var dir = std.Io.Dir.openDirAbsolute(t.io, path, .{}) catch return error.SkipZigTest;
    dir.close(t.io);
    var store = try ExpertStore.open(t.allocator, path, .{ .layers = 48, .experts = 512, .hidden = 2560, .intermediate = 640 });
    defer store.deinit();
    const gate_first = store.span(2, 0, .gate_up);
    const gate_last = store.span(2, 511, .gate_up);
    const down_first = store.span(2, 0, .down);
    const down_last = store.span(2, 511, .down);
    try t.expectEqual(@as(u64, 6_553_600), gate_first.len);
    try t.expectEqual(@as(u64, 3_276_800), down_first.len);
    try t.expectEqual(@as(u64, 511 * 6_553_600), gate_last.offset - gate_first.offset);
    try t.expectEqual(@as(u64, 511 * 3_276_800), down_last.offset - down_first.offset);
    const combined = try t.allocator.alloc(u8, 9_830_400);
    defer t.allocator.free(combined);
    const direct = try t.allocator.alloc(u8, 9_830_400);
    defer t.allocator.free(direct);
    try store.readExpert(2, 17, combined);
    const gate = store.span(2, 17, .gate_up);
    const down = store.span(2, 17, .down);
    try readExact(store.files[gate.file].fd, direct[0..@intCast(gate.len)], gate.offset);
    try readExact(store.files[down.file].fd, direct[@intCast(gate.len)..], down.offset);
    try t.expectEqualSlices(u8, direct, combined);
}

fn writeTinyExpertCheckpoint(allocator: std.mem.Allocator, dir: std.Io.Dir, experts: u16, hidden: u32, inter: u32) ![]u8 {
    const io = std.testing.io;
    const gate_key = "model.language_model.layers.0.mlp.experts.gate_up_proj";
    const down_key = "model.language_model.layers.0.mlp.experts.down_proj";
    const gate_bytes: usize = @as(usize, experts) * 2 * inter * hidden * 2;
    const down_bytes: usize = @as(usize, experts) * hidden * inter * 2;
    const header = try std.fmt.allocPrint(
        allocator,
        "{{\"{s}\":{{\"dtype\":\"BF16\",\"shape\":[{d},{d},{d}],\"data_offsets\":[0,{d}]}},\"{s}\":{{\"dtype\":\"BF16\",\"shape\":[{d},{d},{d}],\"data_offsets\":[{d},{d}]}}}}",
        .{ gate_key, experts, 2 * inter, hidden, gate_bytes, down_key, experts, hidden, inter, gate_bytes, gate_bytes + down_bytes },
    );
    defer allocator.free(header);
    const file_bytes = try allocator.alloc(u8, 8 + header.len + gate_bytes + down_bytes);
    errdefer allocator.free(file_bytes);
    std.mem.writeInt(u64, file_bytes[0..8], header.len, .little);
    @memcpy(file_bytes[8..][0..header.len], header);
    const payload = file_bytes[8 + header.len ..];
    var i: usize = 0;
    while (i + 1 < payload.len) : (i += 2) {
        const mixed = (@as(u64, i / 2) +% 1) *% 0x9E3779B97F4A7C15;
        const sign: u16 = @intCast((mixed >> 40) & 1);
        const bits: u16 = (sign << 15) | (0x3e00 +% @as(u16, @intCast((mixed >> 53) % 0x0200)));
        std.mem.writeInt(u16, payload[i..][0..2], bits, .little);
    }
    try dir.writeFile(io, .{ .sub_path = "experts.safetensors", .data = file_bytes });
    const index = try std.fmt.allocPrint(
        allocator,
        "{{\"weight_map\":{{\"{s}\":\"experts.safetensors\",\"{s}\":\"experts.safetensors\"}}}}",
        .{ gate_key, down_key },
    );
    defer allocator.free(index);
    try dir.writeFile(io, .{ .sub_path = "model.safetensors.index.json", .data = index });
    return file_bytes;
}

fn tinyStream() mlx.mlx_stream {
    return if (std.c.getenv("CODEX_SANDBOX") != null) mlx.mlx_default_cpu_stream_new() else mlx.gpuStream();
}

test "expert stream zero copy hit refills nothing and keeps the slab operand" {
    const t = std.testing;
    var tmp = t.tmpDir(.{});
    defer tmp.cleanup();
    const raw = try writeTinyExpertCheckpoint(t.allocator, tmp.dir, 4, 64, 64);
    defer t.allocator.free(raw);
    var path_buf: [std.fs.max_path_bytes]u8 = undefined;
    const path_len = try tmp.dir.realPath(t.io, &path_buf);
    const path = path_buf[0..path_len];
    const geom = Geometry{ .layers = 1, .experts = 4, .hidden = 64, .intermediate = 64 };
    const s = tinyStream();
    var zero = try Engine.initWithOptions(t.allocator, path, geom, 2 * 24576, s, .{ .bounce_size = 1 << 20 });
    defer zero.deinit();
    try t.expectEqual(@as(u64, 0), zero.fallback_imports);
    try t.expectEqual(@as(u64, 2 * @as(u64, geom.layers) + 2), zero.slab_imports);

    var first = try zero.prepareHost(0, &.{1});
    const slot = first.remapped[0];
    try t.expect(slot < 2);
    const slab_ptr = zero.cacheSlotBytes(0, slot, .gate_up).ptr;
    first.deinit();
    try t.expectEqual(@as(u64, 1), zero.fill_experts_total);
    try t.expect(zero.slotReady(0, slot));

    var second = try zero.prepareHost(0, &.{1});
    defer second.deinit();
    try t.expectEqual(@as(u64, 1), zero.fill_experts_total);
    try t.expectEqual(slot, second.remapped[0]);
    try t.expectEqual(slab_ptr, zero.cacheSlotBytes(0, slot, .gate_up).ptr);

    const gate_span = zero.store.span(0, 1, .gate_up);
    try t.expectEqualSlices(u8, raw[@intCast(gate_span.offset)..][0..@intCast(gate_span.len)], zero.cacheSlotBytes(0, slot, .gate_up));
    const down_span = zero.store.span(0, 1, .down);
    try t.expectEqualSlices(u8, raw[@intCast(down_span.offset)..][0..@intCast(down_span.len)], zero.cacheSlotBytes(0, slot, .down));
    try t.expectEqual(@as(u64, 0), slab_release_timeouts.load(.monotonic));
}

const TinyEngine = struct {
    tmp: std.testing.TmpDir,
    raw: []u8,
    engine: Engine,

    fn open(slots: u16, experts: u16) !TinyEngine {
        const t = std.testing;
        var tmp = t.tmpDir(.{});
        errdefer tmp.cleanup();
        const raw = try writeTinyExpertCheckpoint(t.allocator, tmp.dir, experts, 64, 64);
        errdefer t.allocator.free(raw);
        var path_buf: [std.fs.max_path_bytes]u8 = undefined;
        const path_len = try tmp.dir.realPath(t.io, &path_buf);
        const geom = Geometry{ .layers = 1, .experts = experts, .hidden = 64, .intermediate = 64 };
        const engine = try Engine.initWithOptions(t.allocator, path_buf[0..path_len], geom, @as(u64, slots) * 24576, tinyStream(), .{ .io_workers = 2, .bounce_size = 1 << 20 });
        return .{ .tmp = tmp, .raw = raw, .engine = engine };
    }

    fn close(self: *TinyEngine) void {
        self.engine.deinit();
        std.testing.allocator.free(self.raw);
        self.tmp.cleanup();
    }

    fn anyReady(self: *const TinyEngine) bool {
        for (self.engine.layers[0].cache.ready) |flag| {
            if (flag) return true;
        }
        return false;
    }
};

test "expert stream a failed component fill leaves no ready slot" {
    const t = std.testing;
    var fixture = try TinyEngine.open(2, 4);
    defer fixture.close();
    const engine = &fixture.engine;

    const down_index = (0 * @as(usize, 4) + 1) * 2 + 1;
    const good = engine.store.spans[down_index];
    engine.store.spans[down_index] = .{ .file = good.file, .offset = 1 << 40, .len = good.len };
    try t.expectError(error.FillSpanPastEof, engine.prepareHost(0, &.{1}));
    try t.expect(!fixture.anyReady());
    try t.expectEqual(@as(i32, -1), engine.layers[0].cache.expert_to_slot[1]);

    engine.store.spans[down_index] = good;
    var recovered = try engine.prepareHost(0, &.{1});
    defer recovered.deinit();
    const slot = recovered.remapped[0];
    try t.expect(engine.slotReady(0, slot));
    try t.expectEqualSlices(u8, fixture.raw[@intCast(good.offset)..][0..@intCast(good.len)], engine.cacheSlotBytes(0, slot, .down));
}

test "expert stream a failed fill never serves the expert whose slot it took" {
    const t = std.testing;
    var fixture = try TinyEngine.open(2, 4);
    defer fixture.close();
    const engine = &fixture.engine;

    var warm = try engine.prepareHost(0, &.{ 0, 1 });
    warm.deinit();
    try t.expectEqual(@as(u64, 2), engine.fill_experts_total);
    const victim_slot: u16 = @intCast(engine.layers[0].cache.expert_to_slot[0]);
    try t.expect(engine.slotReady(0, victim_slot));

    const down_index = (0 * @as(usize, 4) + 2) * 2 + 1;
    const good = engine.store.spans[down_index];
    engine.store.spans[down_index] = .{ .file = good.file, .offset = 1 << 40, .len = good.len };
    try t.expectError(error.FillSpanPastEof, engine.prepareHost(0, &.{2}));
    engine.store.spans[down_index] = good;

    try t.expect(!engine.slotReady(0, victim_slot));
    try t.expectEqual(@as(i32, -1), engine.layers[0].cache.expert_to_slot[0]);

    var again = try engine.prepareHost(0, &.{0});
    defer again.deinit();
    try t.expectEqual(@as(u64, 3), engine.fill_experts_total);
    const gate_span = engine.store.span(0, 0, .gate_up);
    try t.expectEqualSlices(u8, fixture.raw[@intCast(gate_span.offset)..][0..@intCast(gate_span.len)], engine.cacheSlotBytes(0, again.remapped[0], .gate_up));
}

test "expert stream never serves a hit from a slab that is still filling" {
    const t = std.testing;
    var fixture = try TinyEngine.open(2, 4);
    defer fixture.close();
    const engine = &fixture.engine;

    var warm = try engine.prepareHost(0, &.{1});
    const slot = warm.remapped[0];
    warm.deinit();
    try t.expectEqual(@as(u64, 1), engine.fill_experts_total);

    engine.layers[0].cache.ready[slot] = false;
    var refilled = try engine.prepareHost(0, &.{1});
    try t.expectEqual(@as(u64, 2), engine.fill_experts_total);
    refilled.deinit();
    engine.drainPendingReaders();

    _ = try engine.layers[0].slabs[0].slab.tryBeginFill();
    try t.expectError(error.SlabFilling, engine.prepareHost(0, &.{2}));
    abortFill(&engine.layers[0].slabs[0]);
}

test "expert stream a failure after the first lease leaves every slab reusable" {
    const t = std.testing;
    var fixture = try TinyEngine.open(2, 4);
    defer fixture.close();
    const engine = &fixture.engine;

    var failing = std.testing.FailingAllocator.init(t.allocator, .{ .fail_index = 0 });
    const real = engine.layers[0].slabs[1].slab.allocator;
    engine.layers[0].slabs[1].slab.allocator = failing.allocator();
    try t.expectError(error.OutOfMemory, engine.prepareHost(0, &.{1}));
    engine.layers[0].slabs[1].slab.allocator = real;

    var after = try engine.prepareHost(0, &.{2});
    defer after.deinit();
    const slot = after.remapped[0];
    const gate_span = engine.store.span(0, 2, .gate_up);
    try t.expectEqualSlices(u8, fixture.raw[@intCast(gate_span.offset)..][0..@intCast(gate_span.len)], engine.cacheSlotBytes(0, slot, .gate_up));
}

test "expert stream union workspace refuses reuse while the previous reader holds it" {
    const t = std.testing;
    var fixture = try TinyEngine.open(2, 4);
    defer fixture.close();
    const engine = &fixture.engine;

    var held = try engine.prepareHost(0, &.{ 0, 1, 2 });
    defer held.deinit();
    try t.expect(held.workspace);
    try t.expectError(error.SlabLeased, engine.prepareHost(0, &.{ 0, 1, 3 }));
}

test "expert stream a cancelled fill drains before the slab is reused" {
    const t = std.testing;
    var fixture = try TinyEngine.open(2, 4);
    defer fixture.close();
    const engine = &fixture.engine;

    const canceller = try std.Thread.spawn(.{}, struct {
        fn call(e: *Engine) void {
            e.cancelFills();
        }
    }.call, .{engine});
    const outcome = engine.prepareHost(0, &.{ 0, 1, 2 });
    canceller.join();
    if (outcome) |ok| {
        var prepared = ok;
        defer prepared.deinit();
    } else |err| {
        try t.expect(err == error.FillCancelled);
        try t.expect(!fixture.anyReady());
    }

    var after = try engine.prepareHost(0, &.{3});
    defer after.deinit();
    const slot = after.remapped[0];
    const gate_span = engine.store.span(0, 3, .gate_up);
    try t.expectEqualSlices(u8, fixture.raw[@intCast(gate_span.offset)..][0..@intCast(gate_span.len)], engine.cacheSlotBytes(0, slot, .gate_up));
}

test "expert stream prefill union copies hits and fills only the misses" {
    const t = std.testing;
    var fixture = try TinyEngine.open(4, 8);
    defer fixture.close();
    const engine = &fixture.engine;

    var warm = try engine.prepareHost(0, &.{ 0, 1, 2 });
    try t.expect(!warm.workspace);
    warm.deinit();
    try t.expectEqual(@as(u64, 3), engine.fill_experts_total);

    var chunk = try engine.prepareHost(0, &.{ 0, 1, 2, 3, 4, 4 });
    defer chunk.deinit();
    try t.expect(chunk.workspace);
    try t.expectEqual(@as(u64, 5), engine.fill_experts_total);
    try t.expectEqualSlices(u16, &.{ 0, 1, 2, 3, 4, 4 }, chunk.remapped);

    for ([_]u16{ 0, 1, 2 }) |expert| {
        const slot: u16 = @intCast(engine.layers[0].cache.expert_to_slot[expert]);
        try t.expectEqualSlices(u8, engine.cacheSlotBytes(0, slot, .gate_up), engine.unionSlotBytes(expert, .gate_up));
        try t.expectEqualSlices(u8, engine.cacheSlotBytes(0, slot, .down), engine.unionSlotBytes(expert, .down));
    }
    for ([_]u16{ 3, 4 }) |expert| {
        const source = engine.store.span(0, expert, .gate_up);
        try t.expectEqualSlices(u8, fixture.raw[@intCast(source.offset)..][0..@intCast(source.len)], engine.unionSlotBytes(expert, .gate_up));
    }

    try t.expect(engine.layers[0].cache.expert_to_slot[4] >= 0);
    try t.expectEqual(@as(i32, -1), engine.layers[0].cache.expert_to_slot[3]);
    const admitted: u16 = @intCast(engine.layers[0].cache.expert_to_slot[4]);
    try t.expect(engine.slotReady(0, admitted));
    try t.expectEqualSlices(u8, engine.unionSlotBytes(4, .gate_up), engine.cacheSlotBytes(0, admitted, .gate_up));
}

fn bf16Of(bits: u16) f32 {
    return @bitCast(@as(u32, bits) << 16);
}

test "expert stream bf16 kernels read the imported slab operand" {
    const t = std.testing;
    if (std.c.getenv("CODEX_SANDBOX") != null) return error.SkipZigTest;
    var fixture = try TinyEngine.open(2, 4);
    defer fixture.close();
    const engine = &fixture.engine;
    const s = engine.s;

    var prepared = try engine.prepareHost(0, &.{ 1, 2, 2, 1 });
    defer prepared.deinit();
    try t.expect(!prepared.workspace);

    var x_bits: [128]u16 = undefined;
    for (&x_bits, 0..) |*v, i| v.* = 0x3f00 +% @as(u16, @intCast(i % 97));
    const x2 = mlx.mlx_array_new_data(&x_bits, &[_]c_int{ 2, 64 }, 2, .bfloat16);
    defer _ = mlx.mlx_array_free(x2);
    const x5 = mlx.mlx_array_new_data(&x_bits, &[_]c_int{ 1, 2, 1, 1, 64 }, 5, .bfloat16);
    defer _ = mlx.mlx_array_free(x5);
    var ids: [4]i32 = undefined;
    for (prepared.remapped, 0..) |slot, i| ids[i] = slot;
    const slots2 = mlx.mlx_array_new_data(&ids, &[_]c_int{ 2, 2 }, 2, .int32);
    defer _ = mlx.mlx_array_free(slots2);
    const ids3 = mlx.mlx_array_new_data(&ids, &[_]c_int{ 1, 2, 2 }, 3, .int32);
    defer _ = mlx.mlx_array_free(ids3);
    const weights: [4]f32 = .{ 0.75, 0.25, 0.4, 0.6 };
    const w2 = mlx.mlx_array_new_data(&weights, &[_]c_int{ 2, 2 }, 2, .float32);
    defer _ = mlx.mlx_array_free(w2);

    const h = try kernels.gateUpSwiglu(s, x2, prepared.raw_gate_up, slots2);
    defer _ = mlx.mlx_array_free(h);
    const y = try kernels.downReduce(s, h, prepared.raw_down, slots2, w2);
    defer _ = mlx.mlx_array_free(y);
    try mlx.check(mlx.mlx_array_eval(y));

    const no_idx = mlx.mlx_array{ .ctx = null };
    var gate = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(gate);
    try mlx.check(mlx.mlx_gather_mm(&gate, x5, prepared.gate, no_idx, ids3, false, s));
    var up = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(up);
    try mlx.check(mlx.mlx_gather_mm(&up, x5, prepared.up, no_idx, ids3, false, s));
    var act = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(act);
    try mlx.check(mlx.mlx_sigmoid(&act, gate, s));
    var silu = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(silu);
    try mlx.check(mlx.mlx_multiply(&silu, act, gate, s));
    var hidden_ref = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(hidden_ref);
    try mlx.check(mlx.mlx_multiply(&hidden_ref, silu, up, s));
    var down_ref = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(down_ref);
    try mlx.check(mlx.mlx_gather_mm(&down_ref, hidden_ref, prepared.down, no_idx, ids3, false, s));
    var squeezed = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(squeezed);
    try mlx.check(mlx.mlx_squeeze(&squeezed, down_ref, s));
    const wexp = mlx.mlx_array_new_data(&weights, &[_]c_int{ 2, 2, 1 }, 3, .float32);
    defer _ = mlx.mlx_array_free(wexp);
    var weighted = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(weighted);
    try mlx.check(mlx.mlx_multiply(&weighted, squeezed, wexp, s));
    var reference = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(reference);
    try mlx.check(mlx.mlx_sum_axis(&reference, weighted, -2, false, s));
    try mlx.check(mlx.mlx_array_eval(reference));

    const got = mlx.mlx_array_data_bfloat16(y) orelse return error.KernelOutputUnreadable;
    var ref_f32 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(ref_f32);
    try mlx.check(mlx.mlx_astype(&ref_f32, reference, .float32, s));
    try mlx.check(mlx.mlx_array_eval(ref_f32));
    const want = mlx.mlx_array_data_float32(ref_f32) orelse return error.KernelOutputUnreadable;
    var num: f64 = 0;
    var den: f64 = 0;
    for (0..128) |i| {
        const diff = @as(f64, bf16Of(got[i])) - @as(f64, want[i]);
        num += diff * diff;
        den += @as(f64, want[i]) * @as(f64, want[i]);
    }
    try t.expect(den > 0);
    const rel = @sqrt(num / den);
    try t.expect(rel < 0.01);
}

test "expert stream route detail buckets by attention class and probes off the print cadence" {
    const t = std.testing;
    var fixture = try TinyEngine.open(2, 4);
    defer fixture.close();
    const engine = &fixture.engine;

    engine.beginForward();
    try t.expect(engine.routeProbeActive());
    engine.noteRoute(0, false, .{ .build_ns = 1, .x_wait_ns = 2, .ids_wait_ns = 3, .read_ns = 4 });
    engine.noteRoute(0, true, .{ .build_ns = 10, .x_wait_ns = 20, .ids_wait_ns = 30, .read_ns = 40 });
    engine.finishForward(1);
    const first = engine.breakdown();
    try t.expectEqual(@as(u32, 1), first.route_linear.layers);
    try t.expectEqual(@as(u64, 3), first.route_linear.ids_wait_ns);
    try t.expectEqual(@as(u64, 2), first.route_linear.x_wait_ns);
    try t.expectEqual(@as(u32, 1), first.route_full.layers);
    try t.expectEqual(@as(u64, 30), first.route_full.ids_wait_ns);
    try t.expectEqual(@as(u64, 110), first.route_ns);
    try t.expect(first.probe);

    engine.beginForward();
    engine.noteRoute(0, false, .{ .ids_wait_ns = 7 });
    engine.finishForward(1);
    const second = engine.breakdown();
    try t.expectEqual(@as(u64, 7), second.route_linear.ids_wait_ns);
    try t.expectEqual(@as(u32, 0), second.route_full.layers);

    engine.forward_count = BREAKDOWN_EVERY - 1;
    try t.expect(!engine.routeProbeActive());
    engine.forward_count = BREAKDOWN_EVERY + BREAKDOWN_EVERY / 2 - 1;
    try t.expect(engine.routeProbeActive());
}

test "expert stream a streamed reader holds its slab lease until the next route readback" {
    const t = std.testing;
    var fixture = try TinyEngine.open(2, 4);
    defer fixture.close();
    const engine = &fixture.engine;
    const slab = engine.layers[0].slabs[0].slab;

    var first = try engine.prepareHost(0, &.{1});
    first.deinit();
    try t.expectEqual(@as(usize, 2), engine.pendingReaders());
    try t.expectEqual(@as(usize, 1), slab.readerCount());
    try t.expectError(error.SlabLeased, slab.tryBeginFill());

    var second = try engine.prepareHost(0, &.{2});
    try t.expectEqual(@as(usize, 0), engine.pendingReaders());
    try t.expectEqual(@as(usize, 1), slab.readerCount());
    second.deinit();
    try t.expectEqual(@as(usize, 2), engine.pendingReaders());
    engine.drainPendingReaders();
    try t.expectEqual(@as(usize, 0), engine.pendingReaders());
    try t.expectEqual(@as(usize, 0), slab.readerCount());
}

pub const TinyQuantPack = struct {
    allocator: std.mem.Allocator,
    file: []u8,
    data: [quant.component_count][]align(4) u8,
    rows: [quant.component_count]u32,
    cols: [quant.component_count]u32,

    pub fn dtypeOf(ci: usize) mlx.mlx_dtype {
        return if (quant.partOf(@enumFromInt(ci)) == .weight) .uint32 else .bfloat16;
    }

    pub fn residentBank(self: *const TinyQuantPack, experts: u16, ci: usize) mlx.mlx_array {
        const shape = [_]c_int{ @intCast(experts), @intCast(self.rows[ci]), @intCast(self.cols[ci]) };
        return mlx.mlx_array_new_data(self.data[ci].ptr, &shape, 3, dtypeOf(ci));
    }

    pub fn deinit(self: *TinyQuantPack) void {
        for (self.data) |slice| self.allocator.free(slice);
        self.allocator.free(self.file);
        self.* = undefined;
    }
};

pub fn writeTinyQuantCheckpoint(
    allocator: std.mem.Allocator,
    dir: std.Io.Dir,
    seed: u64,
    experts: u16,
    hidden: u32,
    inter: u32,
    group_size: u32,
    bits: [3]u32,
) !TinyQuantPack {
    const io = std.testing.io;
    var prng = std.Random.DefaultPrng.init(seed);
    const rnd = prng.random();
    var pack = TinyQuantPack{
        .allocator = allocator,
        .file = &.{},
        .data = @splat(&.{}),
        .rows = @splat(0),
        .cols = @splat(0),
    };
    errdefer {
        for (pack.data) |slice| if (slice.len != 0) allocator.free(slice);
    }
    var header: std.ArrayList(u8) = .empty;
    defer header.deinit(allocator);
    var index: std.ArrayList(u8) = .empty;
    defer index.deinit(allocator);
    try header.append(allocator, '{');
    try index.appendSlice(allocator, "{\"weight_map\":{");
    var offset: u64 = 0;
    var key_buf: [192]u8 = undefined;
    for (0..quant.component_count) |ci| {
        const c: quant.Component = @enumFromInt(ci);
        const projection = quant.projectionOf(c);
        const in_dim: u32 = if (projection == .down) inter else hidden;
        const is_weight = quant.partOf(c) == .weight;
        pack.rows[ci] = if (projection == .down) hidden else inter;
        pack.cols[ci] = if (is_weight) in_dim * bits[@intFromEnum(projection)] / 32 else in_dim / group_size;
        const elem: u64 = if (is_weight) 4 else 2;
        const span_bytes: usize = @intCast(@as(u64, experts) * pack.rows[ci] * pack.cols[ci] * elem);
        const buffer = try allocator.alignedAlloc(u8, .@"4", span_bytes);
        pack.data[ci] = buffer;
        if (is_weight) {
            const words: []u32 = @alignCast(std.mem.bytesAsSlice(u32, buffer));
            for (words) |*word| word.* = rnd.int(u32);
        } else {
            const halves: []u16 = @alignCast(std.mem.bytesAsSlice(u16, buffer));
            const wanted = quant.partOf(c) == .scales;
            for (halves) |*half| {
                const value: f32 = if (wanted) 0.002 + rnd.float(f32) * 0.01 else -0.05 + rnd.float(f32) * 0.1;
                half.* = @truncate(@as(u32, @bitCast(value)) >> 16);
            }
        }
        const key = try quant.tensorKey(&key_buf, 0, c);
        const sep: []const u8 = if (ci == 0) "" else ",";
        const entry = try std.fmt.allocPrint(allocator, "{s}\"{s}\":{{\"dtype\":\"{s}\",\"shape\":[{d},{d},{d}],\"data_offsets\":[{d},{d}]}}", .{
            sep, key, if (is_weight) "U32" else "BF16", experts, pack.rows[ci], pack.cols[ci], offset, offset + span_bytes,
        });
        defer allocator.free(entry);
        try header.appendSlice(allocator, entry);
        const mapping = try std.fmt.allocPrint(allocator, "{s}\"{s}\":\"experts.safetensors\"", .{ sep, key });
        defer allocator.free(mapping);
        try index.appendSlice(allocator, mapping);
        offset += span_bytes;
    }
    try header.append(allocator, '}');
    try index.appendSlice(allocator, "}}");
    const file_bytes = try allocator.alloc(u8, 8 + header.items.len + @as(usize, @intCast(offset)));
    errdefer allocator.free(file_bytes);
    std.mem.writeInt(u64, file_bytes[0..8], header.items.len, .little);
    @memcpy(file_bytes[8..][0..header.items.len], header.items);
    var at: usize = 8 + header.items.len;
    for (pack.data) |slice| {
        @memcpy(file_bytes[at..][0..slice.len], slice);
        at += slice.len;
    }
    try dir.writeFile(io, .{ .sub_path = "experts.safetensors", .data = file_bytes });
    try dir.writeFile(io, .{ .sub_path = "model.safetensors.index.json", .data = index.items });
    pack.file = file_bytes;
    return pack;
}

test "expert stream quantized slabs alias the nine pack tensors and remap ids" {
    const t = std.testing;
    var tmp = t.tmpDir(.{});
    defer tmp.cleanup();
    var pack = try writeTinyQuantCheckpoint(t.allocator, tmp.dir, 0x51ab, 4, 64, 64, 32, .{ 4, 8, 4 });
    defer pack.deinit();
    const raw = pack.file;
    var path_buf: [std.fs.max_path_bytes]u8 = undefined;
    const path_len = try tmp.dir.realPath(t.io, &path_buf);
    const path = path_buf[0..path_len];
    const geom = Geometry{ .layers = 1, .experts = 4, .hidden = 64, .intermediate = 64 };
    const s = tinyStream();

    const per_expert = try expertBytesFor(t.allocator, path, geom, .quantized_split);
    try t.expectEqual(@as(u64, 2 * (64 * 8 * 4 + 2 * 64 * 2 * 2) + (64 * 16 * 4 + 2 * 64 * 2 * 2)), per_expert);

    var engine = try Engine.initWithOptions(t.allocator, path, geom, 2 * per_expert, s, .{
        .bounce_size = 1 << 20,
        .layout = .quantized_split,
    });
    defer engine.deinit();
    try t.expectEqual(@as(u64, 0), engine.fallback_imports);
    try t.expectEqual(@as(u64, quant.component_count * (@as(usize, geom.layers) + 1)), engine.slab_imports);

    var prepared = try engine.prepareHost(0, &.{ 0, 2 });
    try t.expect(prepared.quantized);
    try t.expect(prepared.gate.ctx == null);
    try t.expectEqual(@as(u64, 2), engine.fill_experts_total);
    for (0..quant.component_count) |ci| {
        const operand = prepared.quantOperand(@enumFromInt(ci));
        try t.expect(operand.ctx != null);
        const shape = mlx.getShape(operand);
        try t.expectEqual(@as(usize, 3), shape.len);
        try t.expectEqual(@as(c_int, 2), shape[0]);
        const spec = engine.store.slabSpec(ci);
        try t.expectEqual(@as(c_int, @intCast(spec.rows)), shape[1]);
        try t.expectEqual(@as(c_int, @intCast(spec.cols)), shape[2]);
        try t.expectEqual(spec.dtype, mlx.mlx_array_dtype(operand));
    }
    for ([_]u16{ 0, 2 }, 0..) |expert, i| {
        const slot = prepared.remapped[i];
        for (0..quant.component_count) |ci| {
            const source = engine.store.spanAt(0, expert, ci);
            try t.expectEqualSlices(
                u8,
                raw[@intCast(source.offset)..][0..@intCast(source.len)],
                engine.cacheSlotBytesAt(0, slot, ci),
            );
        }
    }

    var handles: [quant.component_count]?*anyopaque = undefined;
    for (&handles, 0..) |*handle, ci| handle.* = prepared.quantOperand(@enumFromInt(ci)).ctx;
    prepared.deinit();
    try t.expectEqual(quant.component_count, engine.pendingReaders());
    engine.beginForward();
    var again = try engine.prepareHost(0, &.{ 0, 2 });
    defer again.deinit();
    try t.expectEqual(@as(u64, 2), engine.fill_experts_total);
    for (handles, 0..) |handle, ci| try t.expectEqual(handle, again.quantOperand(@enumFromInt(ci)).ctx);
}

test "expert stream: under streaming an explicit --mtp refuses, a settings mtp is dropped, else off" {
    const t = std.testing;
    try t.expectEqual(MtpUnderStreaming.refuse, mtpUnderStreaming(true, null));
    try t.expectEqual(MtpUnderStreaming.refuse, mtpUnderStreaming(true, true));
    try t.expectEqual(MtpUnderStreaming.drop_settings, mtpUnderStreaming(false, true));
    try t.expectEqual(MtpUnderStreaming.off, mtpUnderStreaming(false, false));
    try t.expectEqual(MtpUnderStreaming.off, mtpUnderStreaming(false, null));
}
