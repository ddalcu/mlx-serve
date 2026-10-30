//! sdk.expert.stream: the expert stream over any bank module `B` (the bank contract, `sdk.expert.assertBank`): per-layer slot pools, routes,
//! deferred release, growth, read-ahead, gates, the phase change. `B` fixes the record topology, the segments and the slot
//! arrays at compile time, so an instance is the same code as a stream written for that bank.

const std = @import("std");
const builtin = @import("builtin");
const mlx = @import("sdk").mlx;
const expert = @import("../expert.zig");
const expert_io = @import("io.zig");
const expert_policy = @import("policy.zig");
const expert_lookahead = @import("lookahead.zig");
const expert_routes = @import("routes.zig");
const log = @import("sdk").log;
/// The route recorder (`expert.routes`) is a diagnostic: its hooks compile into profile builds (`sdk.plugin_profile`) and
/// the tests only; a served build has no recorder branch on any route.
pub const recording: bool = @import("sdk").plugin_profile or builtin.is_test;

/// `probed`: the plugin's prefill-timers build (its read-ahead probes compiled in; every other build has no field, call or
/// branch for them).
pub fn StreamOf(comptime B: type, comptime probed: bool) type {
    expert.assertBank(B);
    return struct {
        pub const n_components = B.n_components;
        pub const gu_components = B.gu_components;
        pub const Component = B.Component;
        pub const Layer = B.Layer;
        pub const LayerPolicy = expert_policy.LayerPolicy;
        pub const Phase = expert_policy.Phase;
        pub const Plan = expert_policy.Plan;
        pub const max_route_ids = expert_policy.max_route_ids;

        /// A routed id's records: `Layer.minis` when the bank declares it (GLM-5.3's EXL3 bank: an expert's tensor-parallel
        /// minis, adjacent in the file at `Layer.record_bytes` and in the slot rows), else 1.
        pub fn minisOf(l: *const Layer) u32 {
            return if (comptime @hasField(Layer, "minis")) l.minis else 1;
        }

        /// The file bytes from one of a routed id's records to the next (0 with one record).
        pub fn miniStride(l: *const Layer) u64 {
            return if (comptime @hasField(Layer, "minis")) l.record_bytes else 0;
        }

        /// A routed id's span in the file: its records back to back (the speculative class reads it whole).
        pub fn spanOf(l: *const Layer) u64 {
            return (minisOf(l) - 1) * miniStride(l) + l.logical_bytes;
        }

        /// The most records a routed id may have (a read job holds `expert_io.max_items` records).
        pub const max_minis = expert_io.max_items;

        /// The most records any layer's routed id has.
        pub fn maxMinis(bank: *const B.Bank) u32 {
            var m: u32 = 1;
            for (bank.layers) |*l| m = @max(m, minisOf(l));
            return m;
        }

        /// Layers of different segments keep separate transient scratches when the bank declares
        /// `per_geometry_transients` (GLM-5.3's EXL3 bank: its K3 and K4 bank layers), at most two; otherwise every layer
        /// shares one at the widest layer's geometry.
        pub const per_geometry = @hasDecl(B, "per_geometry_transients") and B.per_geometry_transients;
        /// A bank of several bank layers per routed layer (`B.banks_per_layer`, its `Bank.streamLayer`): the
        /// lookahead reads ahead over all the next routed layer's bank layers.
        pub const multi_bank = @hasDecl(B, "banks_per_layer") and B.banks_per_layer > 1;
        /// The bank layers per routed layer (the lookahead's precision is counted per one).
        pub const sides: usize = if (multi_bank) B.banks_per_layer else 1;
        /// Per bank layer of a routed layer (`sides`), per top-k position: the records read ahead, and those the target
        /// call routed.
        pub const ByRank = [sides][expert_lookahead.max_k][2]u64;

        /// A bank whose records' parts live apart (`B.scatter`: an MLX pack's stacked tensors): every read goes part by
        /// part (`Bank.parts(layer, expert)`, a `B.Records.Part` per component); one record per id, and no read is
        /// speculative, pre-read or direct (those are keyed by one file's record spans). Its `sidecar` is unread.
        pub const scatter = @hasDecl(B, "scatter") and B.scatter;
        comptime {
            if (scatter and @hasField(Layer, "minis")) @compileError("expert stream: a scatter bank has one record per id");
        }

        /// Two layers whose slot rows are interchangeable: the same records per id and the same segments.
        pub fn sameGeometry(x: *const Layer, y: *const Layer) bool {
            if (minisOf(x) != minisOf(y)) return false;
            for (x.segments, y.segments) |p, q| {
                if (p.length != q.length or p.rank != q.rank or p.dtype != q.dtype or !std.mem.eql(u64, p.shape[0..p.rank], q.shape[0..q.rank])) return false;
            }
            return true;
        }

        /// Scale-line isolation (a bank's `slot_line_bytes`, glm5_next's NVFP4 bank): every slot row of every component
        /// starts on its own line of this many bytes, so a row the read pool writes after a command buffer commits never
        /// shares a GPU cache line with another slot's row. A component whose row is shorter than a line (a 4-byte global
        /// scale) is padded to one line per slot: its array is [rows, line / itemsize] and the pool writes the row's first
        /// `length` bytes; the kernels read word 0 of the slot's line. 0 (every other bank): rows back to back, as before.
        pub const slot_line_bytes: u64 = if (@hasDecl(B, "slot_line_bytes")) B.slot_line_bytes else 0;
        /// A padded component's array is never smaller than this many bytes (a bank's `slot_min_bytes`): its own Metal
        /// buffer, never a sub-allocation of MLX's small-buffer heap (< 256 B) beside foreign buffers, and never an input
        /// MLX passes in the `constant` address space (< 8 elements).
        pub const slot_min_bytes: u64 = if (@hasDecl(B, "slot_min_bytes")) B.slot_min_bytes else 0;

        comptime {
            if (slot_line_bytes != 0 and (!std.math.isPowerOfTwo(slot_line_bytes) or slot_min_bytes < 256 or slot_min_bytes % slot_line_bytes != 0))
                @compileError("expert stream: a bank's slot_line_bytes must be a power of two and its slot_min_bytes a multiple of it, at least MLX's small-buffer threshold");
        }

        /// One slot row of component `c` in `layer`'s slots: the segment of each of a routed id's records, back to back,
        /// padded to `slot_line_bytes` when the bank isolates lines. Host rows and MLX rows alike.
        pub fn rowBytes(layer: *const Layer, c: usize) u64 {
            const n = layer.segments[c].length * minisOf(layer);
            return if (slot_line_bytes == 0) n else std.mem.alignForward(u64, n, slot_line_bytes);
        }

        /// Whether component `c`'s rows are padded past their segment (line isolation of a short row).
        pub fn padded(layer: *const Layer, c: usize) bool {
            return rowBytes(layer, c) != layer.segments[c].length * minisOf(layer);
        }

        /// A bank's slot-array floor (a bank's `slot_array_min_bytes`, glm5_next's NVFP4 and EXL3 banks: one VM page):
        /// every slot array of the bank holds at least this many bytes. Metal gives no file-block-aligned `contents()` to a
        /// buffer under the page (MLX allocates it unrounded), and glm5_next's direct reads need every slot row on the 4 KiB
        /// block (`checkDirectRows`): a one-row ext of a 4 KiB component was refused. 0 (every other bank: DeepSeek-V4.1,
        /// MiMo, glm_moe_dsa): no floor, the arrays hold their rows exactly (a padded component's `slot_min_bytes` only).
        pub const slot_array_min_bytes: u64 = if (@hasDecl(B, "slot_array_min_bytes")) B.slot_array_min_bytes else 0;

        /// The floor of component `c`'s array: the bank's `slot_array_min_bytes`, and a padded component's `slot_min_bytes`.
        pub fn arrayMinBytes(layer: *const Layer, c: usize) u64 {
            return if (padded(layer, c)) @max(slot_min_bytes, slot_array_min_bytes) else slot_array_min_bytes;
        }

        /// The slot rows component `c`'s array holds for `rows` slot rows: `rows`, and at least `arrayMinBytes` of rows
        /// (the rows past `rows` are never addressed).
        pub fn allocSlots(layer: *const Layer, c: usize, rows: u32) u32 {
            const n = rowBytes(layer, c);
            if (n == 0) return rows;
            return @max(rows, @as(u32, @intCast(std.math.divCeil(u64, arrayMinBytes(layer, c), n) catch unreachable)));
        }

        /// The rows of component `c`'s array (its shape's first dimension) for `rows` slot rows: `allocSlots` x minis.
        pub fn allocRows(layer: *const Layer, c: usize, rows: u32) u32 {
            return allocSlots(layer, c, rows) * minisOf(layer);
        }

        /// The MLX shape of component `c`'s array for `rows` slot rows into `shape`; returns its rank. A padded component
        /// (a scalar segment, one record per id: refused otherwise, `SlotLineGeometry`) is [allocRows, line / itemsize].
        pub fn arrayShape(layer: *const Layer, c: usize, rows: u32, shape: *[4]c_int) !usize {
            const seg = layer.segments[c];
            if (padded(layer, c)) {
                // A scalar segment's length is its element's size.
                const item = seg.length;
                if (seg.rank != 0 or minisOf(layer) != 1 or item == 0 or rowBytes(layer, c) % item != 0) return error.SlotLineGeometry;
                shape[0] = @intCast(allocRows(layer, c, rows));
                shape[1] = @intCast(rowBytes(layer, c) / item);
                return 2;
            }
            shape[0] = @intCast(allocRows(layer, c, rows));
            for (seg.shape[0..seg.rank], 1..) |d, k| shape[k] = @intCast(d);
            return seg.rank + 1;
        }

        /// The bytes of component `c`'s array for `rows` slot rows (`arrayShape`'s): the stream's frees and the bill's
        /// slot terms.
        pub fn arrayBytes(layer: *const Layer, c: usize, rows: u32) u64 {
            return @as(u64, allocSlots(layer, c, rows)) * rowBytes(layer, c);
        }

        pub const HostSlotRows = struct {
            rows: u32,
            row_bytes: [n_components]u64,
            banks: [n_components][]u8,

            /// Rows sized for `layer`'s segments; one page-aligned bank per component (`arrayBytes`: the bank's floor at least).
            pub fn init(layer: *const Layer, rows: u32) !HostSlotRows {
                var s: HostSlotRows = .{ .rows = rows, .row_bytes = undefined, .banks = undefined };
                var n: usize = 0;
                errdefer for (s.banks[0..n]) |b| std.heap.page_allocator.free(b);
                for (layer.segments, 0..) |_, c| {
                    s.row_bytes[c] = rowBytes(layer, c);
                    s.banks[c] = try std.heap.page_allocator.alloc(u8, std.math.cast(usize, arrayBytes(layer, c, rows)) orelse return error.OutOfMemory);
                    n += 1;
                }
                return s;
            }

            /// After the pool that wrote into the rows has stopped.
            pub fn deinit(self: *HostSlotRows) void {
                for (self.banks) |b| std.heap.page_allocator.free(b);
                self.* = undefined;
            }

            pub fn row(self: *const HostSlotRows, c: Component, r: u32) []u8 {
                const n = self.row_bytes[@intFromEnum(c)];
                return self.banks[@intFromEnum(c)][r * n ..][0..n];
            }

            pub fn rowDest(self: *const HostSlotRows, r: u32) [n_components]u64 {
                var d: [n_components]u64 = undefined;
                for (&d, 0..) |*a, c| a.* = @intFromPtr(self.row(@enumFromInt(c), r).ptr);
                return d;
            }
        };

        pub const LayerSlotBank = struct {
            arrays: [n_components]mlx.mlx_array,
            base: [n_components]u64,
            row_bytes: [n_components]u64,
            rows: u32,

            /// One array per segment in the bank's dtypes (`B.mlxDtype`; EXL3: code int16 [rows, in/16, out/16,
            /// 16K], rout / rin float16 [rows, out] / [rows, in]), zero-filled and
            /// evaluated on `stream` in one eval. The data pointers are taken here; the
            /// arrays stay held (never donated or recycled) until `deinit`, after the pool
            /// stops. MLX allocates through Metal even on the CPU stream: callers hold the GPU lock.
            pub fn init(layer: *const Layer, rows: u32, stream: mlx.mlx_stream) !LayerSlotBank {
                var b = try initLazy(layer, rows, stream);
                errdefer b.deinit();
                try evalArrays(&b.arrays);
                try b.bind();
                return b;
            }

            /// `init`'s zero arrays, not yet evaluated: a grow builds every layer's, evaluates them all in one eval
            /// (`Stream.grow`: one GPU round trip, not nine per layer), then `bind`s each.
            fn initLazy(layer: *const Layer, rows: u32, stream: mlx.mlx_stream) !LayerSlotBank {
                var b: LayerSlotBank = .{ .arrays = @splat(.{}), .base = @splat(0), .row_bytes = undefined, .rows = rows };
                errdefer b.deinit();
                for (layer.segments, 0..) |seg, c| {
                    var shape: [4]c_int = undefined;
                    const ndim = try arrayShape(layer, c, rows, &shape);
                    const dtype: mlx.mlx_dtype = B.mlxDtype(seg.dtype);
                    b.arrays[c] = mlx.mlx_array_new();
                    try mlx.check(mlx.mlx_zeros(&b.arrays[c], &shape, ndim, dtype, stream));
                    b.row_bytes[c] = rowBytes(layer, c);
                }
                return b;
            }

            /// `init`'s arrays without the zero fill: MLX-owned buffers (`dsv41_alloc_uninit`), evaluated as built, bound at once.
            fn initUnfilled(layer: *const Layer, rows: u32) !LayerSlotBank {
                var b: LayerSlotBank = .{ .arrays = @splat(.{}), .base = @splat(0), .row_bytes = undefined, .rows = rows };
                errdefer b.deinit();
                for (layer.segments, 0..) |seg, c| {
                    var shape: [4]c_int = undefined;
                    const ndim = try arrayShape(layer, c, rows, &shape);
                    const dtype: mlx.mlx_dtype = B.mlxDtype(seg.dtype);
                    b.arrays[c] = mlx.mlx_array_new();
                    if (dsv41_alloc_uninit(&b.arrays[c], &shape, ndim, dtype) != 0) return error.MlxUnfilledAlloc;
                    b.row_bytes[c] = rowBytes(layer, c);
                }
                try b.bind();
                return b;
            }

            /// The evaluated arrays' data pointers (after `initLazy` and an eval that covered them).
            fn bind(b: *LayerSlotBank) !void {
                for (b.arrays, &b.base) |arr, *base| base.* = @intFromPtr(mlx.mlx_array_data_uint8(arr) orelse return error.MlxNoData);
            }

            pub fn deinit(self: *LayerSlotBank) void {
                for (self.arrays) |a| {
                    if (a.ctx != null) _ = mlx.mlx_array_free(a);
                }
                self.* = undefined;
            }

            pub fn row(self: *const LayerSlotBank, c: Component, r: u32) []u8 {
                const n = self.row_bytes[@intFromEnum(c)];
                const p: [*]u8 = @ptrFromInt(self.base[@intFromEnum(c)] + r * n);
                return p[0..n];
            }

            pub fn rowDest(self: *const LayerSlotBank, r: u32) [n_components]u64 {
                var d: [n_components]u64 = undefined;
                for (&d, self.base, self.row_bytes) |*a, base, n| a.* = base + r * n;
                return d;
            }
        };

        extern fn dsv41_alloc_uninit(out: *mlx.mlx_array, shape: [*]const c_int, ndim: usize, dtype: mlx.mlx_dtype) c_int;

        /// MLX's allocated bytes (MLX slot memory only: the first read creates the Metal device).
        pub fn mlxActive() u64 {
            var n: usize = 0;
            _ = mlx.mlx_get_active_memory(&n);
            return n;
        }

        /// One eval over `arrays` (a single GPU round trip).
        pub fn evalArrays(arrays: []const mlx.mlx_array) !void {
            const vec = mlx.mlx_vector_array_new_data(arrays.ptr, arrays.len);
            defer _ = mlx.mlx_vector_array_free(vec);
            try mlx.check(mlx.mlx_eval(vec));
        }

        /// Where the stream keeps its slot rows: host pages (the default; hermetic
        /// tests and CPU checks) or MLX arrays the kernels bind, created and evaluated
        /// on `mlx` (creating any MLX array creates the Metal device: callers hold the
        /// GPU lock). Chosen once, at Stream.init.
        pub const SlotMemory = union(enum) { host, mlx: mlx.mlx_stream };

        /// One bank of slot rows. Both memories are addressed by the same row
        /// arithmetic, so the read path never asks which one it has.
        /// One eval over every MLX bank among `rows` (none for host rows).
        pub fn evalRows(a: std.mem.Allocator, rows: []const ?Rows) !void {
            var arrays: std.ArrayList(mlx.mlx_array) = .empty;
            defer arrays.deinit(a);
            for (rows) |r| if (r) |x| switch (x.backing) {
                .mlx => |m| try arrays.appendSlice(a, &m.arrays),
                .none, .host => {},
            };
            if (arrays.items.len > 0) try evalArrays(arrays.items);
        }

        pub const Rows = struct {
            rows: u32 = 0,
            base: [n_components]u64 = @splat(0),
            row_bytes: [n_components]u64 = @splat(0),
            backing: union(enum) { none, host: HostSlotRows, mlx: LayerSlotBank } = .none,

            /// The bytes these rows allocated on `layer`'s geometry (`arrayBytes` per component, the bank's floor
            /// included): what their free returns.
            fn allocBytes(self: *const Rows, layer: *const Layer) u64 {
                if (self.rows == 0) return 0;
                var n: u64 = 0;
                for (0..n_components) |c| n += arrayBytes(layer, c, self.rows);
                return n;
            }

            fn init(layer: *const Layer, rows: u32, memory: SlotMemory) !Rows {
                if (rows == 0) return .{};
                switch (memory) {
                    .host => {
                        const h = try HostSlotRows.init(layer, rows);
                        var r: Rows = .{ .rows = rows, .row_bytes = h.row_bytes, .backing = .{ .host = h } };
                        for (&r.base, h.banks) |*b, bank| b.* = @intFromPtr(bank.ptr);
                        return r;
                    },
                    .mlx => |stream| {
                        const m = try LayerSlotBank.init(layer, rows, stream);
                        return .{ .rows = rows, .base = m.base, .row_bytes = m.row_bytes, .backing = .{ .mlx = m } };
                    },
                }
            }

            /// `init` for a grow: an MLX bank's arrays built without their eval (`bind` after one eval of every layer's,
            /// `evalRows`); host rows are complete at once.
            fn initLazy(layer: *const Layer, rows: u32, memory: SlotMemory) !Rows {
                if (rows == 0) return .{};
                switch (memory) {
                    .host => return init(layer, rows, memory),
                    .mlx => |stream| {
                        const m = try LayerSlotBank.initLazy(layer, rows, stream);
                        return .{ .rows = rows, .row_bytes = m.row_bytes, .backing = .{ .mlx = m } };
                    },
                }
            }

            /// `init` without the fill: an MLX bank's buffers taken from the allocator as they are (`LayerSlotBank.initUnfilled`);
            /// host rows as `init`'s.
            fn initUnfilled(layer: *const Layer, rows: u32, memory: SlotMemory) !Rows {
                if (rows == 0) return .{};
                switch (memory) {
                    .host => return init(layer, rows, memory),
                    .mlx => {
                        const m = try LayerSlotBank.initUnfilled(layer, rows);
                        return .{ .rows = rows, .base = m.base, .row_bytes = m.row_bytes, .backing = .{ .mlx = m } };
                    },
                }
            }

            /// After the eval that covered an MLX bank's arrays: its data pointers (host rows have theirs).
            fn bind(self: *Rows) !void {
                switch (self.backing) {
                    .mlx => |*m| {
                        try m.bind();
                        self.base = m.base;
                    },
                    .none, .host => {},
                }
            }

            /// After the pool that wrote into the rows has stopped.
            fn deinit(self: *Rows) void {
                switch (self.backing) {
                    .none => {},
                    .host => |*h| h.deinit(),
                    .mlx => |*m| m.deinit(),
                }
                self.* = .{};
            }

            pub fn row(self: *const Rows, c: Component, r: u32) []u8 {
                const n = self.row_bytes[@intFromEnum(c)];
                const p: [*]u8 = @ptrFromInt(self.base[@intFromEnum(c)] + r * n);
                return p[0..n];
            }

            fn rowDest(self: *const Rows, r: u32) [n_components]u64 {
                var d: [n_components]u64 = undefined;
                for (&d, self.base, self.row_bytes) |*a, base, n| a.* = base + r * n;
                return d;
            }
        };

        /// What the stream supports as an expert source (`sdk.expert.Caps`); an arm installs a subset at construction
        /// (`Options`).
        pub const source_caps: expert.Caps = .{ .two_phase = true, .transient_release = true, .prompt_seed = true, .read_ahead = true, .wide = true, .lookahead = true, .preread = true, .event_gates = true };

        /// It reads through the process's one reader: the host takes it at its arch's load claim (`sdk.expert.takeReader`).
        pub const uses_reader = true;

        /// The source contract's slot types (`sdk.expert`).
        pub const BankKind = expert.BankKind;
        pub const SlotRef = expert.SlotRef;

        // ── Stream: per-layer slot pools, routes, deferred release, growth ──

        /// P7 (mimo/p7-prefill-reads): the prompt pass's read schedule, a construction-time route (`Options.prompt_reads`).
        /// `stock` (the control): the read-ahead posts every admitted record, in file order, while the layer's attention
        /// runs, and the call lands all of it before its seed; the executor routes group g -- plans it and posts its reads --
        /// when its depth schedule reaches it (g < depth at the call's start, else once group g - depth drained).
        /// `overlap`: the same admissions and the same plans, with the reads moved. Every group of the call is planned at
        /// its first route (`planCall`) by the same `planWith` over the same ids, with the held set (the persistent slots of
        /// the `depth - 1` plans before it) and the transient window the stock schedule gives it, so every slot, bank, tile
        /// and kernel is the stock schedule's. The read-ahead posts its first `depth x max_route_ids` records (the groups
        /// routed at the call's start) in the prediction's rank order; the rest wait for the plans, which read those a
        /// group hits before any evicts them and drop those evicted first (the stock schedule reads them twice). Each
        /// group's reads are posted, in group order, as soon as their rows are free (every earlier group reading a row has
        /// drained, the read-ahead record it overwrites has landed); a group's route returns once the read-ahead records it
        /// hits have landed. The same records land in the same rows before any kernel reads them: exact by construction.
        pub const PromptReads = enum { stock, overlap };

        /// P7 `overlap` only (`Options.prompt_ahead`): how much of the read-ahead posts while the layer's attention runs.
        /// `head` (try 2's form): its first `depth x max_route_ids` records in rank order, the rest resolved by the call's
        /// plans after the attention (read when a group hits them first, dropped when evicted first). `whole`: every
        /// admitted record in rank order, as the stock schedule posts them all (no tail, nothing dropped: a record the plans
        /// evict before any hit is read and then overwritten, the stock schedule's double read), so the drive reads the
        /// tail during the attention instead of after it, ahead of the call's misses. Neither moves an admission, a plan,
        /// a slot or a kernel: the same records land in the same rows before any kernel reads them.
        /// `whole_trim` (X21, mimo/ahead-trim): `whole`, then at the call's plans every read-ahead job no reader worker
        /// has taken yet is withdrawn from the pool's queue (`Pool.withdraw`; it wrote no row) and its records are
        /// resolved by the plans as `head`'s tail is (`p7Withdraw`, `p7Tail`): read once if a group hits them first (or
        /// none touches them), dropped if a plan evicts them first. The head/tail boundary is where the drive stands when
        /// the plans exist: a long attention leaves nothing queued (`whole`'s schedule), a short one leaves the tail
        /// (`head`'s schedule without `head`'s idle drive). Same admissions, plans, slots and kernels.
        pub const AheadPost = enum { head, whole, whole_trim };

        pub const Options = struct {
            /// Persistent rows per routed layer for the prefill phase: the caller's
            /// admission result (nothing here sizes memory).
            rows: []const u32,
            /// Widest route the caller passes (verify rows x top-k).
            max_route_ids: u32 = max_route_ids,
            /// Widest DECODE route the caller passes (its decode lane's rows x top-k): decode's window 0 holds this many
            /// rows (plus `decode_staging_rows`). 0 = `max_route_ids` (the control: every caller before this field).
            /// A decode plan's transient loads number at most its distinct ids (`LayerPolicy.planWith`), so a caller whose
            /// decode routes never exceed it never writes past the window.
            decode_route_ids: u32 = 0,
            /// Transient rows shared by every layer; at least `max_route_ids`, so no
            /// route's misses can overflow them.
            transient_rows: u32 = max_route_ids,
            /// Decode misses per completion part (a down wave and its gate).
            records_per_part: u32 = 3,
            pool: expert_io.Options = .{ .tickets = 1024 },
            /// The pool's staging buffers sized from the bank (`stagingBytes`: its widest gate/up or down span,
            /// page-rounded, plus a page of alignment) in place of `pool.staging_bytes`. A pre-read range must fit one
            /// buffer, so a bank whose spans exceed the pool's default needs it.
            staging_from_bank: bool = false,
            /// Decode routes read the next layer's predicted records ahead and
            /// pre-read their own certain misses (the lookahead4 lane).
            lookahead: ?Lookahead = null,
            /// Gate each call's reads on an event the GPU waits for (needs `lookahead`).
            event: ?Event = null,
            /// Host pages, or MLX arrays on the given stream (the serving form).
            slot_memory: SlotMemory = .host,
            /// Prefill routes live at once in one layer (the wide lane's read-ahead:
            /// 2 = the next group's reads issued before this group's waves). Each
            /// live route owns a window of `max_route_ids` transient rows, so the
            /// transient rows must hold `wide_depth` windows.
            wide_depth: u8 = 1,
            /// The phase change's transient release installed (`releaseTransient`, then the grow's window 0); off: the whole
            /// scratch stays through decode.
            transient_release: bool = false,
            /// How the grow's new rows (decode's window 0 and every layer's ext) are allocated (`GrowFill`).
            grow_fill: GrowFill = .zeros,
            /// G7: the arch's read-ahead records (`ReadAheadProbe`), in the prefill-timers build only.
            read_ahead_probe: ProbeSlot = no_probe,
            /// P7: the prompt pass's read schedule (`PromptReads`; `stock` is the control).
            prompt_reads: PromptReads = .stock,
            /// P7 `overlap`: the read-ahead posted with the attention (`AheadPost`); `whole` and `whole_trim` need `overlap`.
            prompt_ahead: AheadPost = .head,
            /// The read-ahead's admission bound in records; null: the stock rule, from the pool's tickets. A route that
            /// widens the ticket ring pins it to the narrower ring's bound so the admissions stay the control's.
            ahead_cap: ?u32 = null,
            /// P7 `overlap` relocation (glm5next/prefill-levers lever 1): at a call's plans, every read-ahead record the plans
            /// evict before any group hits it and load again later in the call into a transient row no earlier group of the
            /// call reads is copied from its read-ahead row into that transient row (`p7Relocate`), so that load holds its
            /// record and reads nothing: the record is read once, not twice. Needs `overlap` and one slot geometry (every
            /// layer's rows interchangeable with the transient rows); off (the control): every such record is read again.
            prompt_relocate: bool = false,
            /// The rate tally (`expert.tally`, `LayerPolicy.setTally`) on every layer: null, off.
            tally: ?Tally = null,
        };

        /// The rate tally's settings (`expert.tally.Settings`) and the stream's geometry: each stream layer pins at most
        /// floor(`ratio` x its experts) experts, the rate's half-life `window` decode routes of the layer.
        pub const Tally = struct {
            ratio: f64,
            window: u32,
            /// Per stream layer the experts it serves (a bank of several stream layers per routed layer splits them);
            /// null: the bank's `n_experts` each.
            layer_experts: ?[]const u32 = null,
        };

        /// The grow's new rows: `zeros` evaluates MLX zeros (a GPU fill of every row); `unfilled` takes MLX-owned buffers without
        /// a fill. Exact either way: a row is read only through a slot whose meta says its record landed (`SlotMeta.state`), and
        /// every record lands by its read into the row before that state is set.
        pub const GrowFill = enum { zeros, unfilled };

        /// What the reverse phase change (`Stream.shrink`) leaves in the prompt rows for the next prompt: `forget` (no resident,
        /// as at construction) or `keep` (the residents decode left there, their bytes and meta as their reads landed). The
        /// rows themselves are the same either way.
        pub const Residents = enum { forget, keep };

        /// G7: the prompt pass's read-ahead records, in the package's prefill-timers build only: the arch that builds the
        /// stream hands it these probes; every other build has no field, call or branch for them.
        pub const read_ahead_probed: bool = probed;
        pub const ReadAheadProbe = struct {
            /// At a layer's barrier: the records read ahead that its call routed, and its seed (the demand records).
            barrier: *const fn (layer: u32, hits: u32, seed: *const std.DynamicBitSetUnmanaged) void,
            /// At the read-ahead's admission: the predicted seed's top (hottest first, as many as the unprotected rows), the
            /// records admitted, and those neither resident nor admitted.
            admission: *const fn (layer: u32, top: []const u16, admitted: u32, blocked: u32) void,
            /// Records posted to the reader.
            posted: *const fn (layer: u32, n: u32) void,
        };
        pub const ProbeSlot = if (read_ahead_probed) ?ReadAheadProbe else void;
        pub const no_probe: ProbeSlot = if (read_ahead_probed) null else {};

        /// DSV41_LOOKAHEAD4=<k>:<tau>:<budget>:<chunks> at horizon 1.
        pub const Lookahead = struct {
            k: u32 = 8,
            /// inf keeps each row's plain top-K.
            tau: f32 = std.math.inf(f32),
            /// Records read ahead per layer call; 2 x budget staging slots.
            budget: u32 = 2,
            /// Chunks per speculative record: 1, 2, 4 or 8.
            chunks: u32 = 4,
            /// Speculative threads >= 1 start a chunk only while at most this many
            /// demand jobs run (0 = demand idle).
            idle_busy: u32 = 0,
            preread: bool = true,
        };

        pub const Event = struct {
            /// .host: an int64 word the stream owns (CPU backend, checks); .metal: an
            /// id<MTLSharedEvent> on MLX's device whose signaled value is 0.
            backend: union(enum) { host, metal: u64 } = .host,
            /// A gate whose bytes have not landed by then is forced (the stream fails).
            watchdog_ms: u32 = 2000,
        };

        /// One staging buffer that holds any read range of `bank` (`Options.staging_from_bank`): the widest gate/up or
        /// down span over its layers, rounded up to the page, plus one page (a range's aligned read starts up to a page
        /// before it).
        pub fn stagingBytes(bank: *const B.Bank) u64 {
            var widest: u64 = 0;
            for (bank.layers) |l| {
                var gu: u64 = 0;
                for (l.segments[0..gu_components]) |sg| gu += sg.length;
                widest = @max(widest, @max(gu, l.logical_bytes - gu));
            }
            const page = std.heap.pageSize();
            return std.mem.alignForward(u64, widest, page) + page;
        }

        /// Direct reads (`expert_io.Options.direct`, a construction-time route): every range a submit of `bank` reads is
        /// aligned to `block` (the file's block size, `expert_io.fileBlockSize`), so an F_NOCACHE preadv straight into the
        /// slot rows takes the drive's uncached path. Checked once, at construction, over every record of every layer:
        /// its gate/up and down offsets (`DirectReadsOffset`), every segment's offset and length and the minis' stride
        /// (`DirectReadsLength`). The rows' side is `checkDirectRows`. Refused by name; nothing per read.
        pub fn checkDirectReads(bank: *const B.Bank, block: u64) expert_io.DirectError!void {
            if (block == 0) return error.DirectReadsBlock;
            for (bank.layers, 0..) |*l, li| {
                for (l.segments) |sg| if (sg.offset % block != 0 or sg.length % block != 0) return error.DirectReadsLength;
                if (miniStride(l) % block != 0) return error.DirectReadsLength;
                for (0..bank.n_experts) |e| {
                    const sp = bank.spans(@intCast(li), @intCast(e));
                    if (sp.gu_offset % block != 0 or sp.down_offset % block != 0) return error.DirectReadsOffset;
                }
            }
        }

        /// The rows' side of `checkDirectReads`: every component's base address and row stride a multiple of `block`
        /// (`DirectReadsRow`), so every part a direct read lands (row + mini x segment length) starts on a block.
        pub fn checkDirectRows(r: *const Rows, block: u64) expert_io.DirectError!void {
            if (r.rows == 0) return;
            for (r.base, r.row_bytes) |b, n| if (b % block != 0 or n % block != 0) return error.DirectReadsRow;
        }

        /// The contract's gates, counters and refusals (`sdk.expert`).
        pub const Gates = expert.Gates;
        pub const Stats = expert.Stats;
        pub const Error = expert.Error;

        pub const SlotState = enum(u8) { empty, loading, ready, failed };

        /// What a physical row holds and how many live routes serve from it.
        pub const SlotMeta = struct { pins: u16 = 0, state: SlotState = .empty, layer: u16 = 0, expert: u16 = 0 };

        pub const Part = struct {
            /// Its loads: `plan.loads[order[first + i]]`, i < n.
            first: u32,
            n: u32,
            /// Pool tickets of the loads that read: gate/up of read i = ticket + i,
            /// down = ticket + n_reads + i.
            ticket: u32 = 0,
            n_reads: u32 = 0,
            settled: bool = false,
        };

        pub const Route = struct {
            layer: u32 = 0,
            plan: Plan = .{},
            /// The slot of each hit (plan.hits order).
            hit_slots: [max_route_ids]u32 = undefined,
            /// Load indices in placement (file offset) order; parts are runs of it.
            order: [max_route_ids]u8 = undefined,
            /// Per load (plan order): false when its slot still held the record.
            reads: [max_route_ids]bool = undefined,
            n_parts: u32 = 0,
            parts: [max_route_ids]Part = undefined,
            state: enum { free, live, released } = .free,
            /// Its transient window (prefill routes beside a live one: `Options.wide_depth`).
            window: u8 = 0,
            /// Decode layer calls with the lookahead class: this call's settle value.
            tag: i64 = 0,
            gates: ?Gates = null,
            /// Its record in the route recorder (`Stream.recorder`).
            rec: u32 = expert_routes.none,

            pub fn partsOf(r: *const Route) []const Part {
                return r.parts[0..r.n_parts];
            }

            /// The loads (expert, slot) of part `part`, in file order: what its
            /// gate/up and down kernels read once `waitGu` / `waitDown` return.
            pub fn partLoads(r: *const Route, part: u32, out: *[max_route_ids]expert_policy.Load) []expert_policy.Load {
                const p = r.parts[part];
                for (r.order[p.first..][0..p.n], out[0..p.n]) |li, *l| l.* = r.plan.loads[li];
                return out[0..p.n];
            }
        };

        /// The route being served plus released ones awaiting the next flush: every wide window live, and one more.
        pub const route_capacity = max_wide_depth + 1;
        /// Prefill routes one layer may hold live at once (`Options.wide_depth`; P1c: 5, the served default).
        pub const max_wide_depth = 5;
        comptime {
            // Every route the stream holds is gated at most once, the gate/up gate and a down gate per part (a part at
            // least one id): their gates fit the pool's ring.
            std.debug.assert(route_capacity * (max_route_ids + 1) <= expert_io.max_gates);
        }
        /// Served run 16: the phase change frees the transient scratch (`Stream.releaseTransient`) and the grow allocates decode's
        /// window 0 plus `decode_staging_rows`; the bill (deepseek_v41_bill.zig) reads this declaration.
        pub const phase_change_releases_wide_windows = true;
        /// The release as a construction-time route (on by default since served run 19E; off: the control arm).
        pub const transient_release_default = true;
        /// Slot rows decode reserves beside window 0 for staged reads: none (the lookahead and A1 stage in the read pool).
        pub const decode_staging_rows: u32 = 0;
        pub const wait_timeout_ns: i64 = 60 * std.time.ns_per_s;

        /// Expert residency for one model: per-layer persistent slot pools at the
        /// caller's row bound, a transient scratch shared by all layers, and the read
        /// pool. One inference thread calls it; pool workers only write slot bytes.
        pub const Stream = struct {
            allocator: std.mem.Allocator,
            bank: *const B.Bank,
            pool: *expert_io.Pool,
            layers: []LayerSlots,
            transient: Rows,
            transient_meta: []SlotMeta,
            /// The widest layer: the transient rows' geometry (decode's window 0 is allocated in it at the grow).
            transient_layer: u32,
            /// `per_geometry`: the other geometry's transient scratch (the layers `geom_of` marks 1), beside `transient`.
            alt: ?AltTransient = null,
            geom_of: []u1 = &.{},
            /// `releaseTransient` ran: the scratch is freed until the grow allocates window 0.
            transient_released: bool = false,
            /// The release route, installed at construction (`Options.transient_release`).
            release_installed: bool = false,
            /// The grow's allocation, installed at construction (`Options.grow_fill`).
            grow_fill: GrowFill = .zeros,
            /// The prompt phase's scratch rows and windows (`Options`): `regrowTransient` re-creates them for a later prompt.
            prompt_transient_rows: u32 = 0,
            prompt_wide_depth: u8 = 1,
            /// G7: the arch's read-ahead records (`Options.read_ahead_probe`).
            probe: ProbeSlot = no_probe,
            memory: SlotMemory = .host,
            max_route_ids: u32,
            /// Decode's window 0 rows (`Options.decode_route_ids`, resolved): the widest decode route.
            decode_route_ids: u32,
            records_per_part: u32,
            wide_depth: u8 = 1,
            phase: Phase = .prefill,
            failed: bool = false,
            routes: [route_capacity]Route = @splat(.{}),
            /// Free routes (a stack) and released ones awaiting the next flush, in
            /// release order: `route` and `flush` touch only these, never the ring.
            free: [route_capacity]u8 = blk: {
                var f: [route_capacity]u8 = undefined;
                for (&f, 0..) |*x, i| x.* = route_capacity - 1 - i;
                break :blk f;
            },
            n_free: u8 = route_capacity,
            released: [route_capacity]u8 = undefined,
            n_released: u8 = 0,
            /// The phase's route: the lookahead class (decode with a selector) and
            /// its pre-reads, set at construction and at the phase change.
            route_lookahead: bool = false,
            route_preread: bool = false,
            counters: Stats = .{},
            /// Persistent slots of released prefill routes still pinned for a deferred call's waves
            /// (`holdBase`), held in every later route of `held_layer` until `releaseHeld`.
            held_base: std.ArrayList(u32) = .empty,
            held_layer: u32 = 0,
            /// A route's held-slot scratch (live routes' slots and `held_base`).
            held_scratch: std.ArrayList(u32) = .empty,
            /// `promote`'s copies, reused across calls.
            promote_copies: std.ArrayList(Copy) = .empty,
            /// `promote`'s copies run here, behind the host: a route that loads into a window first waits for the
            /// batches that read from it (`waitWindow`), a persistent load of a layer for that layer's batches.
            copier: Copier = .{},
            read_ns: u64 = 0,
            /// `read_ns` of results with no preadv (copy-only adoptions).
            copy_only_ns: u64 = 0,
            selector: ?expert_lookahead.SelectorOf(B.routed_top_k) = null,
            /// The lookahead's precision: the records read ahead for the next routed layer (their experts, the top-k
            /// position each was chosen at, the bank layer it is in) until that layer's call routes, then per bank layer
            /// and position the records read ahead and those the call routed (`by_rank`).
            ahead_pending: [expert_lookahead.max_budget]struct { expert: u16, rank: u8, side: u8, routed: bool } = undefined,
            n_ahead_pending: u8 = 0,
            ahead_pending_layer: u32 = 0,
            by_rank: ByRank = @splat(@splat(.{ 0, 0 })),
            preread: bool = false,
            /// Direct reads (`Options.pool.direct`): the file's block every range and slot row was checked against at
            /// construction (and every row the phase changes allocate); 0 = the staging route.
            direct_block: u64 = 0,
            /// Decode layer calls so far; a call's pre-reads and speculative records
            /// carry its tag, settled by its own step.
            clock: i64 = 0,
            event_word: ?*i64 = null,
            gated: bool = false,
            /// The last event value handed out.
            gate_value: u64 = 0,
            forced_seen: i64 = 0,
            /// The thread that built the stream (mlx-serve: the inference thread, the
            /// only MLX caller); `grow` allocates slot memory and refuses any other.
            owner: std.Thread.Id,
            /// P1: the prompt pass's read-ahead in flight (one layer's predicted seed).
            ahead: Ahead,
            /// The route recorder, a diagnostic its arch arms (`expert.routes`): every decode route, the handover's
            /// residents and prompt counts. Null: off.
            recorder: ?*expert_routes.Recorder = null,
            /// The request's prompt pass started: a prompt route, seed, call plan or read-ahead ran since the request's start
            /// (`resetPromptCounts` clears it). `setPromptAhead` refuses once it is set. Residents do not set it: a kept or
            /// seeded resident is not a started prompt.
            prompt_started: bool = false,
            /// P7 (`Options.prompt_reads`, `prompt_ahead`, `ahead_cap`).
            prompt_reads: PromptReads = .stock,
            prompt_ahead: AheadPost = .head,
            ahead_cap: ?u32 = null,
            /// P7 overlap relocation (`Options.prompt_relocate`), fixed at construction.
            prompt_relocate: bool = false,
            /// P7 overlap: the prompt call being scheduled; per row (a layer's persistent slots [0, n_experts), then the
            /// transient rows) the record its plans leave and its last reader; per persistent slot the read-ahead job
            /// writing it (`no_job`: none). Empty under `stock`.
            p7: P7 = .{},
            p7_rows: []P7Row = &.{},
            ahead_job: []u8 = &.{},

            /// P1's read-ahead of one layer: `loads[0..n]` (expert, slot) in file order, `reads[i]` false when the
            /// slot still held the record; its pool jobs `parts[0..n_parts]` over them (a Route's tickets).
            const Ahead = struct {
                live: bool = false,
                /// The layer's barrier has counted it (`seedPrefill`: hits and demand, once).
                tallied: bool = false,
                layer: u32 = 0,
                n: u32 = 0,
                n_parts: u32 = 0,
                /// Records [0, posted) have jobs; P7 overlap leaves [posted, n) to the call's plans (`p7Tail`); `whole_trim`
                /// moves `posted` back over the jobs it withdraws at the plans (`p7Withdraw`).
                posted: u32 = 0,
                loads: []LayerPolicy.ReadAhead,
                reads: []bool,
                parts: []Part,
                /// Per job: landed (its reads waited, its rows' states set and unpinned).
                landed: []bool,
            };

            /// The second geometry's transient scratch: its rows, their meta and the layer whose geometry it has.
            const AltTransient = struct { rows: Rows, meta: []SlotMeta, layer: u32 };

            /// P7 overlap: one prompt call's schedule, from `planCall` to its last group's `releaseP7`.
            pub const P7 = struct {
                live: bool = false,
                layer: u32 = 0,
                distinct: []const u16 = &.{},
                width: u32 = 0,
                n_groups: u32 = 0,
                activated: u32 = 0,
                released: u32 = 0,
                /// The next part to post, in group order.
                post_g: u32 = 0,
                post_p: u32 = 0,
                /// Per group: its route (an index into `routes`) and its transient window.
                route: [route_capacity]u8 = undefined,
                window: [route_capacity]u8 = undefined,
                /// Per group and part: the last earlier group reading one of its rows (-1: none) and the read-ahead jobs
                /// writing one of them (a bit per job).
                key: [route_capacity][max_route_ids]i8 = undefined,
                ra: [route_capacity][max_route_ids]u64 = undefined,
            };

            /// P7 overlap: a row as the call's plans leave it: the record it will hold (the stock route's "the slot still
            /// holds the record" test) and its last planned reader this call (-1: none).
            pub const P7Row = struct { layer: u16 = 0, expert: u16 = 0, ready: bool = false, reader: i8 = -1 };
            const no_job: u8 = std.math.maxInt(u8);

            const LayerSlots = struct {
                policy: LayerPolicy,
                /// The prefill rows, then the rows `grow` added.
                base: Rows,
                ext: ?Rows = null,
                /// [n_experts]: one entry per persistent slot.
                meta: []SlotMeta,
                lens: [n_components]u64,
                counts: expert.LayerCounts = .{},
            };

            pub fn layerCounts(self: *const Stream, layer: u32) expert.LayerCounts {
                return self.layers[layer].counts;
            }

            const Location = struct { rows: *const Rows, row: u32, meta: *SlotMeta };

            pub fn init(a: std.mem.Allocator, bank: *const B.Bank, opt: Options) !*Stream {
                const n_layers = bank.layers.len;
                if (opt.rows.len != n_layers) return error.InvalidRows;
                for (opt.rows) |r| if (r > bank.n_experts) return error.InvalidRows;
                if (opt.wide_depth < 1 or opt.wide_depth > max_wide_depth or opt.transient_rows < @as(u32, opt.wide_depth) * opt.max_route_ids) return error.InvalidOptions;
                const minis = maxMinis(bank);
                if (opt.decode_route_ids > opt.max_route_ids) return error.InvalidOptions;
                if (opt.max_route_ids == 0 or opt.max_route_ids > max_route_ids or opt.transient_rows < opt.max_route_ids or minis > max_minis or
                    opt.records_per_part == 0 or opt.records_per_part * minis > expert_io.max_items) return error.InvalidOptions;
                // One transient row must hold any layer's record; per geometry, each of the two geometries has its scratch.
                var widest: usize = 0;
                for (bank.layers, 0..) |*l, i| if (spanOf(l) > spanOf(&bank.layers[widest])) {
                    widest = i;
                };
                var alt_layer: ?usize = null;
                for (bank.layers, 0..) |*l, i| {
                    if (sameGeometry(l, &bank.layers[widest])) continue;
                    if (per_geometry) {
                        if (alt_layer) |al| {
                            if (!sameGeometry(l, &bank.layers[al])) return error.MixedGeometry;
                        } else alt_layer = i;
                    } else {
                        if (minisOf(l) != minisOf(&bank.layers[widest])) return error.MixedGeometry;
                        for (l.segments, bank.layers[widest].segments) |sg, w| if (sg.length > w.length) return error.MixedGeometry;
                    }
                }
                if (opt.event != null and opt.lookahead == null) return error.InvalidOptions;
                if (scatter and (opt.lookahead != null or opt.pool.direct)) return error.InvalidOptions;
                if (opt.event) |ev| if (ev.watchdog_ms < 50 or ev.watchdog_ms > 60_000) return error.InvalidOptions;
                var pool_opt = opt.pool;
                if (opt.staging_from_bank) pool_opt.staging_bytes = stagingBytes(bank);
                if (opt.lookahead) |la| {
                    if (la.chunks == 0 or la.chunks > 8 or !std.math.isPowerOfTwo(la.chunks) or la.idle_busy > 1 or
                        la.budget == 0 or la.budget > expert_lookahead.max_budget) return error.InvalidOptions;
                    const page = std.heap.pageSize();
                    const record = spanOf(&bank.layers[widest]);
                    pool_opt.spec = .{ .threads = @min(la.budget, 2), .slots = 2 * la.budget, .record_bytes = record, .chunk_bytes = expert_io.chunkBytes(la.chunks, record, page), .idle_busy = la.idle_busy };
                    // A pre-read range is the record's gate/up span or its down span,
                    // back to back, each no larger than a staging buffer.
                    if (la.preread) for (bank.layers) |l| {
                        var gu: u64 = 0;
                        for (l.segments[0..gu_components]) |sg| gu += sg.length;
                        if (l.segments[0].offset != 0 or l.segments[gu_components].offset != gu) return error.MixedGeometry;
                        if (gu > pool_opt.staging_bytes or l.logical_bytes - gu > pool_opt.staging_bytes) return error.InvalidOptions;
                    };
                }
                // Direct reads: the file's block, every record's ranges and (below) every slot row aligned to it, once. A
                // pre-range starts before its plan names rows, so it stages: the route needs the pre-read off.
                var direct_block: u64 = 0;
                if (pool_opt.direct) {
                    if (opt.lookahead) |la| if (la.preread) return error.InvalidOptions;
                    direct_block = expert_io.fileBlockSize(bank.sidecar) catch return error.DirectReadsBlock;
                    try expert_io.checkDirectBlock(direct_block, std.heap.pageSize());
                    try checkDirectReads(bank, direct_block);
                }
                var selector: ?expert_lookahead.SelectorOf(B.routed_top_k) = null;
                if (opt.lookahead) |la| selector = try expert_lookahead.SelectorOf(B.routed_top_k).init(a, bank.n_experts, la.k, la.tau, la.budget);
                errdefer if (selector) |*sel| sel.deinit(a);
                // The pool writes the event word until it stops.
                var word: ?*i64 = null;
                if (opt.event) |ev| if (ev.backend == .host) {
                    word = try a.create(i64);
                    word.?.* = 0;
                };
                errdefer if (word) |w| a.destroy(w);

                const self = try a.create(Stream);
                errdefer a.destroy(self);
                const layers = try a.alloc(LayerSlots, n_layers);
                errdefer a.free(layers);
                var n_init: usize = 0;
                errdefer for (layers[0..n_init]) |*ls| {
                    ls.policy.deinit(a);
                    ls.base.deinit();
                    a.free(ls.meta);
                };
                if (opt.tally) |t| if (t.layer_experts) |le| if (le.len != n_layers) return error.InvalidOptions;
                for (layers, opt.rows, bank.layers, 0..) |*ls, rows, *geom, li| {
                    var policy = try LayerPolicy.init(a, bank.n_experts, rows);
                    errdefer policy.deinit(a);
                    if (opt.tally) |t| {
                        const experts = if (t.layer_experts) |le| le[li] else bank.n_experts;
                        policy.setTally(expert.tally.pinLimit(t.ratio, experts), expert.tally.alphaOf(t.window));
                    }
                    var base = try Rows.init(geom, rows, opt.slot_memory);
                    errdefer base.deinit();
                    const meta = try a.alloc(SlotMeta, bank.n_experts);
                    @memset(meta, .{});
                    var lens: [n_components]u64 = undefined;
                    for (&lens, geom.segments) |*l, s| l.* = s.length;
                    ls.* = .{ .policy = policy, .base = base, .meta = meta, .lens = lens };
                    n_init += 1;
                }
                var transient = try Rows.init(&bank.layers[widest], opt.transient_rows, opt.slot_memory);
                errdefer transient.deinit();
                const transient_meta = try a.alloc(SlotMeta, opt.transient_rows);
                errdefer a.free(transient_meta);
                @memset(transient_meta, .{});
                var alt_rows: ?Rows = if (alt_layer) |al| try Rows.init(&bank.layers[al], opt.transient_rows, opt.slot_memory) else null;
                errdefer if (alt_rows) |*x| x.deinit();
                const alt_meta: []SlotMeta = if (alt_layer != null) try a.alloc(SlotMeta, opt.transient_rows) else &.{};
                errdefer a.free(alt_meta);
                @memset(alt_meta, .{});
                const geom_of = try a.alloc(u1, n_layers);
                errdefer a.free(geom_of);
                for (bank.layers, geom_of) |*l, *gm| gm.* = @intFromBool(alt_layer != null and !sameGeometry(l, &bank.layers[widest]));
                const ahead_loads = try a.alloc(LayerPolicy.ReadAhead, bank.n_experts);
                errdefer a.free(ahead_loads);
                const ahead_reads = try a.alloc(bool, bank.n_experts);
                errdefer a.free(ahead_reads);
                // A job holds at most max_items / minis records (each mini one item); one more job than that division:
                // P7 overlap posts the read-ahead in two runs (head, then tail).
                const ahead_parts = try a.alloc(Part, (std.math.divCeil(u32, bank.n_experts, expert_io.max_items / minis) catch unreachable) + 1);
                errdefer a.free(ahead_parts);
                const ahead_landed = try a.alloc(bool, ahead_parts.len);
                errdefer a.free(ahead_landed);
                @memset(ahead_landed, false);
                // P7 overlap: a job is a bit of a part's blockers (u64), groups index the route ring.
                if (opt.prompt_reads == .overlap and ahead_parts.len > 64) return error.InvalidOptions;
                if (opt.prompt_ahead != .head and opt.prompt_reads != .overlap) return error.InvalidOptions;
                // Relocation copies a layer's persistent row into a transient row: one geometry, rows interchangeable.
                if (opt.prompt_relocate) {
                    if (opt.prompt_reads != .overlap or alt_layer != null) return error.InvalidOptions;
                    for (bank.layers) |*l| if (!sameGeometry(l, &bank.layers[widest])) return error.InvalidOptions;
                }
                const p7_rows: []P7Row = if (opt.prompt_reads == .overlap) try a.alloc(P7Row, bank.n_experts + opt.transient_rows) else &.{};
                errdefer if (p7_rows.len > 0) a.free(p7_rows);
                const ahead_job: []u8 = if (opt.prompt_reads == .overlap) try a.alloc(u8, bank.n_experts) else &.{};
                errdefer if (ahead_job.len > 0) a.free(ahead_job);
                @memset(ahead_job, no_job);
                const pool = try expert_io.Pool.start(a, pool_opt);
                errdefer pool.stop();
                if (opt.lookahead) |la| if (la.preread) try B.Records.armPreRead(pool, &layers[widest].lens);
                if (opt.event) |ev| {
                    const object: u64 = switch (ev.backend) {
                        .host => @intFromPtr(word.?),
                        .metal => |ptr| ptr,
                    };
                    try pool.armEvent(if (ev.backend == .host) .host else .metal, object, @as(i64, ev.watchdog_ms) * std.time.ns_per_ms, 0);
                }
                self.* = .{
                    .allocator = a,
                    .bank = bank,
                    .pool = pool,
                    .layers = layers,
                    .transient = transient,
                    .transient_meta = transient_meta,
                    .transient_layer = @intCast(widest),
                    .alt = if (alt_layer) |al| .{ .rows = alt_rows.?, .meta = alt_meta, .layer = @intCast(al) } else null,
                    .geom_of = geom_of,
                    .release_installed = opt.transient_release,
                    .grow_fill = opt.grow_fill,
                    .prompt_transient_rows = opt.transient_rows,
                    .prompt_wide_depth = opt.wide_depth,
                    .probe = opt.read_ahead_probe,
                    .memory = opt.slot_memory,
                    .max_route_ids = opt.max_route_ids,
                    .decode_route_ids = if (opt.decode_route_ids == 0) opt.max_route_ids else opt.decode_route_ids,
                    .records_per_part = opt.records_per_part,
                    .wide_depth = opt.wide_depth,
                    .selector = selector,
                    .preread = if (opt.lookahead) |la| la.preread else false,
                    .event_word = word,
                    .gated = opt.event != null,
                    .owner = std.Thread.getCurrentId(),
                    .ahead = .{ .loads = ahead_loads, .reads = ahead_reads, .parts = ahead_parts, .landed = ahead_landed },
                    .prompt_reads = opt.prompt_reads,
                    .prompt_ahead = opt.prompt_ahead,
                    .ahead_cap = opt.ahead_cap,
                    .prompt_relocate = opt.prompt_relocate,
                    .p7_rows = p7_rows,
                    .ahead_job = ahead_job,
                };
                // Relocation's copies (`p7Relocate`): at most every persistent row's components per call, reserved once.
                if (opt.prompt_relocate) try self.promote_copies.ensureTotalCapacity(a, @as(usize, n_components) * bank.n_experts);
                self.direct_block = direct_block;
                if (direct_block != 0) try self.checkRowsDirect(.construction);
                return self;
            }

            /// Stops the pool (draining its reads) before freeing the rows it writes.
            pub fn deinit(self: *Stream) void {
                const a = self.allocator;
                self.stopCopier();
                self.pool.stop();
                for (self.layers) |*ls| {
                    ls.policy.deinit(a);
                    ls.base.deinit();
                    if (ls.ext) |*e| e.deinit();
                    a.free(ls.meta);
                }
                a.free(self.layers);
                self.held_base.deinit(a);
                self.held_scratch.deinit(a);
                self.promote_copies.deinit(a);
                self.transient.deinit();
                a.free(self.transient_meta);
                if (self.alt) |*x| {
                    x.rows.deinit();
                    a.free(x.meta);
                }
                a.free(self.geom_of);
                if (self.selector) |*sel| sel.deinit(a);
                if (self.event_word) |w| a.destroy(w);
                a.free(self.ahead.loads);
                a.free(self.ahead.reads);
                a.free(self.ahead.parts);
                a.free(self.ahead.landed);
                if (self.p7_rows.len > 0) a.free(self.p7_rows);
                if (self.ahead_job.len > 0) a.free(self.ahead_job);
                a.destroy(self);
            }

            fn fail(self: *Stream, err: Error) Error {
                self.failed = true;
                return err;
            }

            pub fn locate(self: *Stream, layer: u32, slot: u32) Location {
                const ls = &self.layers[layer];
                if (slot < ls.base.rows) return .{ .rows = &ls.base, .row = slot, .meta = &ls.meta[slot] };
                if (slot < ls.policy.capacity) return .{ .rows = &ls.ext.?, .row = slot - ls.base.rows, .meta = &ls.meta[slot] };
                const t = slot - ls.policy.capacity;
                if (self.altOf(layer)) |x| return .{ .rows = &x.rows, .row = t, .meta = &x.meta[t] };
                return .{ .rows = &self.transient, .row = t, .meta = &self.transient_meta[t] };
            }

            /// The second geometry's scratch when `layer` has that geometry (null: the layer reads `transient`).
            fn altOf(self: *Stream, layer: u32) ?*AltTransient {
                if (self.alt) |*x| if (self.geom_of[layer] == 1) return x;
                return null;
            }

            /// One component row of a layer's slot (for the kernels' binding and tests).
            pub fn slotRow(self: *Stream, layer: u32, slot: u32, c: Component) []u8 {
                const loc = self.locate(layer, slot);
                return loc.rows.row(c, loc.row);
            }

            /// Tests: routes `ids` on `layer` with no scores and waits for every part (gate/up first, as the kernels do).
            pub fn serve(self: *Stream, layer: u32, ids: []const u16) Error!*Route {
                const r = try self.route(layer, ids, &.{});
                for (0..r.n_parts) |p| {
                    try self.waitGu(r, @intCast(p));
                    try self.waitDown(r, @intCast(p));
                }
                return r;
            }

            /// Tests: each routed id of `r` (`ids`, in plan order) is served from a slot whose rows hold its records'
            /// segments as `image` (the bank's experts.bin) holds them, each of the id's records at its stride.
            pub fn expectServed(self: *Stream, r: *const Route, ids: []const u16, image: []const u8) !void {
                const l = &self.bank.layers[r.layer];
                for (ids, r.plan.slotsOf()) |e, slot| {
                    const off = self.bank.recordOffset(r.layer, e);
                    for (l.segments, 0..) |seg, c| {
                        const row = self.slotRow(r.layer, slot, @enumFromInt(c));
                        try std.testing.expectEqual(rowBytes(l, c), row.len);
                        for (0..minisOf(l)) |mi| try std.testing.expectEqualSlices(u8, image[off + mi * miniStride(l) + seg.offset ..][0..seg.length], row[mi * seg.length ..][0..seg.length]);
                    }
                }
            }

            /// The bank holding a layer's slot and the slot's row in it.
            pub fn slotRef(self: *const Stream, layer: u32, slot: u32) SlotRef {
                const ls = &self.layers[layer];
                if (slot < ls.base.rows) return .{ .bank = .base, .row = slot };
                if (slot < ls.policy.capacity) return .{ .bank = .ext, .row = slot - ls.base.rows };
                return .{ .bank = .transient, .row = slot - ls.policy.capacity };
            }

            /// Per routed id of `r` (plan order), its slot's bank and row.
            pub fn refsOf(self: *const Stream, r: *const Route, out: *[max_route_ids]SlotRef) []SlotRef {
                for (r.plan.slotsOf(), out[0..r.plan.n_ids]) |slot, *ref| ref.* = self.slotRef(r.layer, slot);
                return out[0..r.plan.n_ids];
            }

            /// A layer's bank as the kernels bind it (MLX slot memory; null for host
            /// rows or a bank without rows). The stream owns the arrays: never free them.
            pub fn bankArrays(self: *const Stream, layer: u32, kind: BankKind) ?B.BankArrays {
                const rows: *const Rows = switch (kind) {
                    .base => &self.layers[layer].base,
                    .ext => if (self.layers[layer].ext) |*e| e else return null,
                    .transient => if (self.alt != null and self.geom_of[layer] == 1) &self.alt.?.rows else &self.transient,
                };
                const m = switch (rows.backing) {
                    .mlx => |*m| m,
                    else => return null,
                };
                return B.bankArraysOf(m.arrays);
            }

            /// Keep a live prefill route's persistent slots (its hits and persistent loads) pinned and held past
            /// its release, for a later call over them (the wide lane's deferred base-bank waves), until
            /// `releaseHeld`. One layer at a time.
            pub fn holdBase(self: *Stream, r: *const Route) !void {
                std.debug.assert(r.state == .live and self.phase == .prefill);
                if (self.held_base.items.len > 0 and self.held_layer != r.layer) return error.HeldOtherLayer;
                self.held_layer = r.layer;
                const cap = self.layers[r.layer].policy.capacity;
                for (r.hit_slots[0..r.plan.n_hits]) |s| if (s < cap) {
                    try self.held_base.append(self.allocator, s);
                    self.locate(r.layer, s).meta.pins += 1;
                };
                for (r.plan.loadsOf()) |l| if (l.persistent) {
                    try self.held_base.append(self.allocator, l.slot);
                    self.locate(r.layer, l.slot).meta.pins += 1;
                };
            }

            /// A seed expert of a prompt layer whose record landed in transient slot `from` (`promote`).
            pub const Promotion = struct { e: u16, from: u32 };
            /// The threads one `promote` splits its copies over.
            const promote_threads = 8;

            /// The prompt phase, each of `items` (in order) an expert of `layer` that is a seed expert
            /// (`LayerPolicy.seed`), whose record landed in its transient slot (a route planned before the layer's seed)
            /// and that no wave reads any more: the record copied into the row `LayerPolicy.admitSeed` gives it (an empty
            /// row, else the coldest probationary resident's that no live route pins), the expert resident there. The
            /// copies go to the copier (`promote_threads` threads, behind the host; a route into a source's window, a
            /// persistent load of the layer and `settleCopies` wait for them). Stops at the first item that finds no
            /// row; returns the experts made resident.
            pub fn promote(self: *Stream, layer: u32, items: []const Promotion) u32 {
                std.debug.assert(self.phase == .prefill);
                const ls = &self.layers[layer];
                const p = &ls.policy;
                const pinned = &self.held_scratch;
                pinned.clearRetainingCapacity();
                for (ls.meta[0..p.capacity], 0..) |m, s| if (m.pins != 0 or m.state == .loading) {
                    pinned.append(self.allocator, @intCast(s)) catch return 0;
                };
                const copies = &self.promote_copies;
                copies.clearRetainingCapacity();
                var done: u32 = 0;
                var bytes: u64 = 0;
                for (items) |it| {
                    const src = self.locate(layer, it.from).meta;
                    if (it.from < p.capacity or src.state != .ready or src.layer != layer or src.expert != it.e) break;
                    // A transient row of another geometry than the layer's rows (one scratch at the widest layer) is not
                    // copied.
                    const same = for (0..n_components) |c| {
                        if (self.slotRow(layer, it.from, @enumFromInt(c)).len != ls.base.row_bytes[c]) break false;
                    } else true;
                    if (!same) break;
                    copies.ensureUnusedCapacity(self.allocator, n_components) catch break;
                    const to = p.admitSeed(it.e, pinned.items) orelse break;
                    for (0..n_components) |c| {
                        const dst = self.slotRow(layer, to, @enumFromInt(c));
                        copies.appendAssumeCapacity(.{ .dst = dst, .src = self.slotRow(layer, it.from, @enumFromInt(c)) });
                        bytes += dst.len;
                    }
                    // Not read before its copy lands: no later route of this call routes `it.e` again, and the call's end
                    // settles the copies.
                    ls.meta[to] = .{ .state = .ready, .layer = @intCast(layer), .expert = it.e };
                    done += 1;
                }
                self.submitCopies(layer, copies.items, bytes, items[0..done]);
                self.counters.promoted += done;
                self.counters.promoted_bytes += bytes;
                return done;
            }

            const Copy = struct { dst: []u8, src: []const u8 };

            /// One promotion's copies, in submission order (`seq`).
            const Batch = struct { copies: []Copy, bytes: u64, seq: u64 };

            /// The thread that runs the batches in order, its queue and the fences: `done` is the last batch copied,
            /// `window_seq[w]` the last batch reading from transient window `w`, `layer_seq` the last writing a layer's
            /// rows. Its memory is libc's (the host and the copier both touch the queue).
            const Copier = struct {
                mutex: std.Io.Mutex = .init,
                cond: std.Io.Condition = .init,
                queue: std.ArrayList(Batch) = .empty,
                thread: ?std.Thread = null,
                submitted: u64 = 0,
                done: u64 = 0,
                stopping: bool = false,
                window_seq: [max_wide_depth]u64 = @splat(0),
                layer_seq: std.ArrayList(u64) = .empty,

                const io = std.Io.Threaded.global_single_threaded.io();

                fn loop(c: *Copier) void {
                    c.mutex.lockUncancelable(io);
                    defer c.mutex.unlock(io);
                    while (true) {
                        while (c.queue.items.len == 0 and !c.stopping) c.cond.waitUncancelable(io, &c.mutex);
                        if (c.queue.items.len == 0) return;
                        const b = c.queue.orderedRemove(0);
                        c.mutex.unlock(io);
                        copyParallel(b.copies, b.bytes);
                        std.heap.c_allocator.free(b.copies);
                        c.mutex.lockUncancelable(io);
                        c.done = b.seq;
                        c.cond.broadcast(io);
                    }
                }

                fn waitFor(c: *Copier, seq: u64) void {
                    if (c.thread == null) return;
                    c.mutex.lockUncancelable(io);
                    defer c.mutex.unlock(io);
                    while (c.done < seq) c.cond.waitUncancelable(io, &c.mutex);
                }
            };

            /// `copies` (one promotion of `layer`, from the transient slots of `items`) handed to the copier; run here
            /// when it cannot take them.
            fn submitCopies(self: *Stream, layer: u32, copies: []const Copy, bytes: u64, items: []const Promotion) void {
                if (bytes == 0) return;
                const c = &self.copier;
                const ca = std.heap.c_allocator;
                const owned = ca.dupe(Copy, copies) catch return copyParallel(copies, bytes);
                if (c.thread == null) {
                    c.layer_seq.resize(ca, self.layers.len) catch {
                        ca.free(owned);
                        return copyParallel(copies, bytes);
                    };
                    @memset(c.layer_seq.items, 0);
                    c.thread = std.Thread.spawn(.{}, Copier.loop, .{c}) catch {
                        ca.free(owned);
                        return copyParallel(copies, bytes);
                    };
                }
                c.mutex.lockUncancelable(Copier.io);
                defer c.mutex.unlock(Copier.io);
                c.queue.append(ca, .{ .copies = owned, .bytes = bytes, .seq = c.submitted + 1 }) catch {
                    c.mutex.unlock(Copier.io);
                    copyParallel(owned, bytes);
                    ca.free(owned);
                    c.mutex.lockUncancelable(Copier.io);
                    return;
                };
                c.submitted += 1;
                const cap = self.layers[layer].policy.capacity;
                for (items) |it| c.window_seq[(it.from - cap) / self.max_route_ids] = c.submitted;
                c.layer_seq.items[layer] = c.submitted;
                c.cond.signal(Copier.io);
            }

            /// Every promotion's copies landed (the prompt's end, the phase change, a forget, the stream's end).
            pub fn settleCopies(self: *Stream) void {
                self.copier.waitFor(self.copier.submitted);
            }

            fn stopCopier(self: *Stream) void {
                const c = &self.copier;
                const t = c.thread orelse return;
                c.mutex.lockUncancelable(Copier.io);
                c.stopping = true;
                c.cond.broadcast(Copier.io);
                c.mutex.unlock(Copier.io);
                t.join();
                c.thread = null;
                c.queue.deinit(std.heap.c_allocator);
                c.layer_seq.deinit(std.heap.c_allocator);
            }

            /// The bytes of `copies` (`total` of them) split in `promote_threads` equal runs, one thread each (the
            /// calling thread runs the last; a thread that fails to start leaves its run to the calling thread).
            fn copyParallel(copies: []const Copy, total: u64) void {
                if (total == 0) return;
                const per = (total + promote_threads - 1) / promote_threads;
                const run = struct {
                    fn f(cs: []const Copy, lo: u64, hi: u64) void {
                        var at: u64 = 0;
                        for (cs) |c| {
                            const a = at;
                            at += c.dst.len;
                            if (at <= lo or a >= hi) continue;
                            const s0: usize = @intCast(@max(lo, a) - a);
                            const s1: usize = @intCast(@min(hi, at) - a);
                            @memcpy(c.dst[s0..s1], c.src[s0..s1]);
                        }
                    }
                }.f;
                var threads: [promote_threads - 1]?std.Thread = @splat(null);
                for (&threads, 0..) |*t, i| t.* = std.Thread.spawn(.{}, run, .{ copies, i * per, @min(total, (i + 1) * per) }) catch null;
                run(copies, (promote_threads - 1) * per, total);
                for (threads, 0..) |t, i| if (t) |th| th.join() else run(copies, i * per, @min(total, (i + 1) * per));
            }

            /// Every resident's prompt row of `layer` held (pinned, never a victim) until `releaseHeld`: routes planned
            /// meanwhile load their misses into transient rows (a prompt layer's misses read before its call).
            pub fn holdResidents(self: *Stream, layer: u32) !void {
                std.debug.assert(self.phase == .prefill);
                if (self.held_base.items.len > 0 and self.held_layer != layer) return error.HeldOtherLayer;
                self.held_layer = layer;
                const p = &self.layers[layer].policy;
                for (p.slot_to_expert[0..p.capacity], 0..) |e, sl| if (e != expert_policy.no_expert) {
                    try self.held_base.append(self.allocator, @intCast(sl));
                    self.locate(layer, @intCast(sl)).meta.pins += 1;
                };
            }

            /// Unpin and stop holding what `holdBase` kept (after the deferred call's waves are evaluated).
            pub fn releaseHeld(self: *Stream) void {
                for (self.held_base.items) |s| self.locate(self.held_layer, s).meta.pins -= 1;
                self.held_base.clearRetainingCapacity();
            }

            /// Construction's end: every layer's residents and prompt state forgotten (`LayerPolicy.forgetAll`: the warm-up's),
            /// so a first prompt's seed and read-ahead start from empty rows (a later prompt's read-ahead replaces the residents
            /// it does not predict). Slot bytes stay: a later load reuses them only on
            /// an exact (layer, expert, slot) match. Refused while anything is live; returns the residents forgotten.
            pub fn forgetResidents(self: *Stream) !u32 {
                if (self.phase != .prefill) return error.NotPrefill;
                self.settleCopies();
                try self.flush();
                for (&self.routes) |*r| if (r.state != .free) return error.RoutesLive;
                if (self.ahead.live or self.held_base.items.len > 0) return error.RoutesLive;
                var n: u32 = 0;
                for (self.layers) |*ls| n += ls.policy.forgetAll();
                return n;
            }

            /// A kept request start without a shrink (MiMo's uniform rows or deferred growth): every layer's prompt state
            /// forgotten (`LayerPolicy.forgetPrompt`), its residents kept, as `shrink(.keep)` leaves them. Nothing live.
            pub fn forgetPrompt(self: *Stream) !void {
                if (self.phase != .prefill) return error.NotPrefill;
                try self.flush();
                for (&self.routes) |*r| if (r.state != .free) return error.RoutesLive;
                if (self.ahead.live or self.held_base.items.len > 0) return error.RoutesLive;
                for (self.layers) |*ls| _ = ls.policy.forgetPrompt();
            }

            /// A request's start: every layer's prompt counts zeroed (they would otherwise add up across requests). The
            /// policy's rank tie-break reads them too, so request 1 (zeroed by the construction's forget) is unchanged.
            pub fn resetPromptCounts(self: *Stream) void {
                self.prompt_started = false;
                for (self.layers) |*ls| {
                    @memset(ls.policy.prefill_freq, 0);
                    // The prompt's pins at the rows it has (the tally's rates as the last request left them).
                    ls.policy.refreshPins();
                }
            }

            /// Every layer's tally seeded from `rates` (`n_experts` per stream layer, in layer order: a tally file's).
            pub fn seedTally(self: *Stream, rates: []const f32) void {
                const n = self.bank.n_experts;
                std.debug.assert(rates.len == self.layers.len * n);
                for (self.layers, 0..) |*ls, l| ls.policy.seedTally(rates[l * n ..][0..n]);
            }

            /// The request's end (`LayerPolicy.endRequest`): each layer that routed in its decode writes its decode rates into
            /// its `n_experts` of `out`; every tally is seeded from now on. Returns the layers that routed.
            pub fn endTally(self: *Stream, out: []f32) u32 {
                const n = self.bank.n_experts;
                std.debug.assert(out.len == self.layers.len * n);
                var routed: u32 = 0;
                for (self.layers, 0..) |*ls, l| routed += @intFromBool(ls.policy.endRequest(out[l * n ..][0..n]));
                return routed;
            }

            /// P7 `overlap`: a request's read-ahead post (`AheadPost`), set once at the request's start by an arch that
            /// chooses it per request (MiMo's `prefill_read_schedule: auto`, by the prompt's length). Only at a request's
            /// start: refused on a `stock` stream, outside the prompt phase, on a failed stream, while anything of a prompt
            /// pass is live (a read-ahead, a route, a planned call, a held slot) and once the request's prompt pass started
            /// (`prompt_started`: its first route, seed, call plan or read-ahead; `resetPromptCounts` starts the next request),
            /// so a request's prompt pass never changes it. Residents kept from the last request (`prompt_residents: keep`)
            /// or seeded at the load (`resident_seed`) do not refuse it. `readAheadSeed` reads it at each layer's post as it reads the
            /// construction value, and the call's plans read it for `whole_trim`'s withdraw (X21; `auto` never sets
            /// `whole_trim`); nothing else does.
            pub fn setPromptAhead(self: *Stream, post: AheadPost) !void {
                if (self.prompt_reads != .overlap) return error.InvalidOptions;
                if (self.phase != .prefill) return error.NotPrefill;
                if (self.failed) return error.StreamFailed;
                if (self.ahead.live or self.p7.live or self.held_base.items.len > 0) return error.RoutesLive;
                for (&self.routes) |*r| if (r.state != .free) return error.RoutesLive;
                if (self.prompt_started) return error.PromptStarted;
                self.prompt_ahead = post;
            }

            /// The last `seedPrefill` of `layer`: its seed's ranks (the call's hottest experts, hottest first).
            pub fn seedRanks(self: *const Stream, layer: u32) u32 {
                return self.layers[layer].policy.seed_ranks;
            }

            /// prepare_prefill_seed: the prompt's routed ids of `layer`, before its
            /// prefill routes.
            pub fn seedPrefill(self: *Stream, layer: u32, ids: []const u16) !void {
                if (self.phase != .prefill) return error.NotPrefill;
                self.prompt_started = true;
                const ah = &self.ahead;
                // P7 overlap: the seed reads residency only; the read-ahead lands where a plan needs it (`planCall`).
                if (self.prompt_reads == .stock and ah.live and ah.layer == layer) try self.awaitReadAhead(layer);
                const policy = &self.layers[layer].policy;
                policy.prepareSeed(ids);
                // P1's engagement, once at the layer's barrier: the records read ahead that its call routes, and its seed's
                // records not read ahead (the seed's misses, loaded on demand).
                if (ah.layer != layer or ah.n == 0 or ah.tallied) return;
                ah.tallied = true;
                for (ah.loads[0..ah.n]) |l| self.counters.ahead_hits += @intFromBool(policy.call_counts[l.expert] > 0);
                self.counters.ahead_demand += policy.seed.count();
                if (comptime read_ahead_probed) if (self.probe) |pr| {
                    // The layer's barrier record: hits, and the seed's demand records.
                    var hits: u32 = 0;
                    for (ah.loads[0..ah.n]) |l| hits += @intFromBool(policy.call_counts[l.expert] > 0);
                    pr.barrier(layer, hits, &policy.seed);
                };
            }

            /// Resolves `ids` (the router's top-k of one layer call, host values read
            /// from an evaluated array) to slots: pins every slot it serves and submits
            /// the misses' reads in parts. The eval that produced `ids` also finished
            /// every released route's consumers, so their slots are recycled first.
            /// With the lookahead class a decode call first pre-reads its certain
            /// misses, and after its submits (which claim the records read ahead for
            /// it) settles what it did not claim and reads ahead the next layer's
            /// predicted records: `scores` = that layer's gate on this call's rows
            /// (rows x n_experts f32, evaluated with `ids`); empty = settle only. The
            /// read-ahead is the decode phase's: before the phase change (and on a
            /// stream without the lookahead class) the scores are not read.
            pub fn route(self: *Stream, layer: u32, ids: []const u16, scores: []const f32) Error!*Route {
                return self.routeWith(layer, ids, scores, true);
            }

            /// `route` of a call that routes again before its waves (a bank of several bank layers per routed layer:
            /// every bank layer's route but the last): its tag taken, no speculative step (the call's last route
            /// settles and reads ahead, so a record read ahead for any of its bank layers stays claimable until then).
            pub fn routeHeld(self: *Stream, layer: u32, ids: []const u16) Error!*Route {
                return self.routeWith(layer, ids, &.{}, false);
            }

            fn routeWith(self: *Stream, layer: u32, ids: []const u16, scores: []const f32, step: bool) Error!*Route {
                if (self.failed) return error.StreamFailed;
                self.prompt_started = self.prompt_started or self.phase == .prefill;
                self.notePrecision(layer, ids, step);
                std.debug.assert(ids.len > 0 and ids.len <= self.max_route_ids);
                // Decode's window 0 holds `decode_route_ids` rows: a decode route is never wider (the caller's contract).
                std.debug.assert(self.phase != .decode or ids.len <= self.decode_route_ids);
                // A layer's read-ahead lands before a route plans over its rows (a hit must never read a loading row).
                if (self.ahead.live and self.ahead.layer == layer) try self.awaitReadAhead(layer);
                const lookahead = self.route_lookahead;
                const tag = self.clock + 1;
                if (self.route_preread) try self.preRead(layer, ids, tag);
                try self.flush();
                if (self.n_free == 0) return self.fail(error.RoutesExhausted);
                self.n_free -= 1;
                const r = &self.routes[self.free[self.n_free]];
                // A prefill route beside live ones of its layer: their slots are held,
                // its transient loads take the first window no route holds; so are the
                // slots a deferred call still reads (`holdBase`).
                const held_set = &self.held_scratch;
                held_set.clearRetainingCapacity();
                var used_windows: u8 = 0;
                if (self.wide_depth > 1) for (&self.routes) |*o| {
                    if (o.state == .free) continue;
                    used_windows |= @as(u8, 1) << @intCast(o.window);
                    if (o.layer != layer) continue;
                    for (o.hit_slots[0..o.plan.n_hits]) |hs| if (hs < self.layers[layer].policy.capacity) {
                        held_set.append(self.allocator, hs) catch return self.fail(error.RoutesExhausted);
                    };
                    for (o.plan.loadsOf()) |l| if (l.persistent) {
                        held_set.append(self.allocator, l.slot) catch return self.fail(error.RoutesExhausted);
                    };
                };
                if (self.held_base.items.len > 0 and self.held_layer == layer)
                    held_set.appendSlice(self.allocator, self.held_base.items) catch return self.fail(error.RoutesExhausted);
                const window: u8 = @intCast(@ctz(~used_windows));
                if (window >= self.wide_depth) return self.fail(error.RoutesExhausted);
                // The window's rows may still be a promotion's sources.
                self.copier.waitFor(self.copier.window_seq[window]);
                r.* = .{ .layer = layer, .window = window };
                const ls = &self.layers[layer];
                ls.policy.planWith(ids, self.phase, &r.plan, .{ .transient_base = @as(u32, window) * self.max_route_ids, .held = held_set.items });
                // A persistent load may take a row a promotion of this layer still writes.
                if (r.plan.n_persistent > 0 and self.copier.thread != null) self.copier.waitFor(self.copier.layer_seq.items[layer]);
                const plan = &r.plan;
                for (plan.hitsOf(), r.hit_slots[0..plan.n_hits]) |e, *s| {
                    s.* = ls.policy.slotOf(e).?;
                    self.locate(layer, s.*).meta.pins += 1;
                }
                var skipped: u64 = 0;
                for (plan.loadsOf(), 0..) |l, i| {
                    const m = self.locate(layer, l.slot).meta;
                    // A row a live route still serves from is never refilled.
                    if (m.pins != 0 or m.state == .loading) return self.fail(error.SlotStillPinned);
                    const held = m.state == .ready and m.layer == layer and m.expert == l.expert;
                    r.reads[i] = !held;
                    skipped += @intFromBool(held);
                    if (!held) m.* = .{ .state = .loading, .layer = @intCast(layer), .expert = l.expert };
                    m.pins = 1;
                }
                if (comptime recording) if (self.recorder) |rec| if (self.phase == .decode) {
                    r.rec = rec.route(layer, ids, plan, r.reads[0..plan.n_loads]);
                };
                try self.submitParts(r);
                if (lookahead) {
                    r.tag = tag;
                    self.clock = tag;
                    if (step) try self.speculate(r, scores);
                }
                const c = &self.counters;
                c.route_calls += 1;
                c.expert_cache_hits += plan.n_hits;
                c.expert_cache_misses += plan.n_misses;
                c.expert_cache_evictions += plan.n_evictions;
                c.persistent_loads += plan.n_persistent;
                c.transient_loads += plan.n_loads - plan.n_persistent;
                c.loads_skipped += skipped;
                ls.counts.routes += 1;
                ls.counts.hits += plan.n_hits;
                ls.counts.misses += plan.n_misses;
                r.state = .live;
                return r;
            }

            /// The call's certain misses (its plan's misses) as pre-read ranges,
            /// before the plan; the submit binds them.
            fn preRead(self: *Stream, layer: u32, ids: []const u16, tag: i64) Error!void {
                var experts: [expert_lookahead.max_candidates]u16 = undefined;
                const misses = self.selector.?.certainMisses(ids, &self.layers[layer].policy, &experts);
                if (misses.len == 0) return;
                // Each record of each miss (an id of several records pre-reads every one of them).
                const geom = &self.bank.layers[layer];
                var bases: [expert_lookahead.max_candidates * max_minis]i64 = undefined;
                var n: usize = 0;
                for (misses) |e| for (0..minisOf(geom)) |m| {
                    bases[n] = @intCast(self.bank.recordOffset(layer, e) + @as(u64, @intCast(m)) * miniStride(geom));
                    n += 1;
                };
                _ = B.Records.preRead(self.pool, self.bank.sidecar, tag, bases[0..n], &self.layers[layer].lens) catch
                    return self.fail(error.PreReadRefused);
            }

            /// The routed layer a stream layer belongs to.
            fn routedOf(layer: u32) u32 {
                return if (comptime multi_bank) layer / B.banks_per_layer else layer;
            }

            /// The records read ahead for routed layer `layer`'s call: the ones its ids route are marked; at the call's
            /// last route (`last`) every one is counted at its top-k position (`by_rank`) and forgotten.
            fn notePrecision(self: *Stream, layer: u32, ids: []const u16, last: bool) void {
                if (self.n_ahead_pending == 0 or routedOf(layer) != self.ahead_pending_layer) return;
                for (self.ahead_pending[0..self.n_ahead_pending]) |*p| {
                    if (std.mem.indexOfScalar(u16, ids, p.expert) != null) p.routed = true;
                }
                if (!last) return;
                for (self.ahead_pending[0..self.n_ahead_pending]) |p| {
                    self.by_rank[p.side][p.rank][0] += 1;
                    self.by_rank[p.side][p.rank][1] += @intFromBool(p.routed);
                }
                self.n_ahead_pending = 0;
            }

            /// The lookahead's records read ahead and of them those the target call routed, per bank layer of a routed
            /// layer and top-k position.
            pub fn lookaheadByRank(self: *const Stream) ByRank {
                return self.by_rank;
            }

            /// The records just read ahead for routed layer `routed` (`side`: each one's bank layer of it, null = 0).
            fn notePending(self: *Stream, routed: u32, experts: []const u16, ranks: []const u8, side: ?[]const u8) void {
                self.ahead_pending_layer = routed;
                for (experts, ranks, self.ahead_pending[0..experts.len], 0..) |e, r, *p, i| p.* = .{ .expert = e, .rank = r, .side = if (side) |sd| sd[i] else 0, .routed = false };
                self.n_ahead_pending = @intCast(experts.len);
            }

            /// Settles every record read ahead for this call that it did not claim,
            /// then reads ahead the next layer's predicted records (keyed by offset).
            fn speculate(self: *Stream, r: *const Route, scores: []const f32) Error!void {
                if (comptime multi_bank) return self.speculateBanks(r, scores);
                const next = r.layer + 1;
                var bases: [expert_lookahead.max_budget]i64 = undefined;
                var n: usize = 0;
                var len: u64 = 0;
                if (scores.len > 0 and next < self.layers.len) {
                    const sel = &self.selector.?;
                    var chosen: [expert_lookahead.max_budget]u16 = undefined;
                    var ranks: [expert_lookahead.max_budget]u8 = undefined;
                    const picked = sel.selectWith(scores, &self.layers[next].policy, expert_lookahead.residentIn, chosen[0..sel.budget], ranks[0..sel.budget]);
                    if (comptime recording) if (self.recorder) |rec| rec.spec(r.rec, picked);
                    for (picked) |e| {
                        bases[n] = @intCast(self.bank.recordOffset(next, e));
                        n += 1;
                    }
                    self.notePending(next, picked, ranks[0..picked.len], null);
                    len = spanOf(&self.bank.layers[next]);
                }
                _ = self.pool.specStep(self.bank.sidecar, r.tag, bases[0..n], len) catch
                    return self.fail(error.SpecRefused);
            }

            /// `speculate` for a bank of several bank layers per routed layer (`B.banks_per_layer`): the next routed
            /// layer's experts over all its bank layers, each checked against its own bank layer's residency, each record
            /// its bank layer's span (one step of records of different lengths).
            fn speculateBanks(self: *Stream, r: *const Route, scores: []const f32) Error!void {
                const bpl = B.banks_per_layer;
                const next: u32 = r.layer / bpl + 1;
                var bases: [expert_lookahead.max_budget]i64 = undefined;
                var lens: [expert_lookahead.max_budget]i64 = undefined;
                var n: usize = 0;
                if (scores.len > 0 and next < self.layers.len / bpl) {
                    const sel = &self.selector.?;
                    const Ctx = struct { s: *const Stream, routed: u32 };
                    const resident = struct {
                        fn f(c: Ctx, e: u16) bool {
                            const sl = c.s.bank.streamLayer(c.routed, e);
                            return c.s.layers[sl].policy.slotOf(e) != null;
                        }
                    }.f;
                    var chosen: [expert_lookahead.max_budget]u16 = undefined;
                    var ranks: [expert_lookahead.max_budget]u8 = undefined;
                    const picked = sel.selectWith(scores, Ctx{ .s = self, .routed = next }, resident, chosen[0..sel.budget], ranks[0..sel.budget]);
                    if (comptime recording) if (self.recorder) |rec| rec.spec(r.rec, picked);
                    var side: [expert_lookahead.max_budget]u8 = undefined;
                    for (picked) |e| {
                        const sl = self.bank.streamLayer(next, e);
                        bases[n] = @intCast(self.bank.recordOffset(sl, e));
                        lens[n] = @intCast(spanOf(&self.bank.layers[sl]));
                        side[n] = @intCast(sl - bpl * next);
                        n += 1;
                    }
                    self.notePending(next, picked, ranks[0..picked.len], side[0..n]);
                }
                _ = self.pool.specStepLens(self.bank.sidecar, r.tag, bases[0..n], lens[0..n]) catch
                    return self.fail(error.SpecRefused);
            }

            /// Event gates for a live route's reads, registered before the GPU commits
            /// its waves: the gate/up wave waits for `gu` (every gate/up ticket of the
            /// call), part p's down wave for `down_first + p` (that part's down
            /// tickets). Null when nothing is read.
            pub fn gate(self: *Stream, r: *Route) Error!?Gates {
                std.debug.assert(self.gated and r.state == .live and r.gates == null);
                const n = r.n_parts;
                if (n == 0) return null;
                const lo = self.gate_value;
                const hi = lo + 1 + n;
                self.gate_value = hi;
                var values: [max_route_ids + 1]u64 = undefined;
                var counts: [max_route_ids + 1]i32 = undefined;
                // A gate/up and a down ticket per RECORD (`max_minis` per id), the pool's bound for one registration.
                var tickets: [expert_io.max_gate_tickets]i64 = undefined;
                var k: usize = 0;
                for (r.partsOf()) |p| for (0..p.n_reads) |i| {
                    tickets[k] = @intCast(p.ticket + i);
                    k += 1;
                };
                values[0] = lo + 1;
                counts[0] = @intCast(k);
                for (r.partsOf(), 0..) |p, pi| {
                    values[1 + pi] = lo + 2 + pi;
                    counts[1 + pi] = @intCast(p.n_reads);
                    for (0..p.n_reads) |i| {
                        tickets[k] = @intCast(p.ticket + p.n_reads + i);
                        k += 1;
                    }
                }
                self.pool.registerGates(values[0 .. n + 1], counts[0 .. n + 1], tickets[0..k]) catch |e| {
                    self.pool.releaseGates(hi);
                    return self.fail(switch (e) {
                        error.GateInvalid => error.GateInvalid,
                        error.GatesFull => error.GatesFull,
                        else => error.GateRefused,
                    });
                };
                r.gates = .{ .gu = lo + 1, .down_first = lo + 2, .n_parts = n };
                return r.gates;
            }

            /// The host event word (Event.backend = .host), for a CPU-stream wait.
            pub fn eventWord(self: *const Stream) ?*const i64 {
                return self.event_word;
            }

            /// Loads in file order, cut into parts (decode: the bounded parts of
            /// `records_per_part`; prefill: `max_items` chunks), one pool submit per part.
            fn submitParts(self: *Stream, r: *Route) Error!void {
                const plan = &r.plan;
                const n = plan.n_loads;
                if (n == 0) return;
                const bank = self.bank;
                for (r.order[0..n], 0..) |*o, i| o.* = @intCast(i);
                std.sort.insertion(u8, r.order[0..n], @as(*const Route, r), struct {
                    fn less(rr: *const Route, a: u8, b: u8) bool {
                        return rr.plan.loads[a].expert < rr.plan.loads[b].expert;
                    }
                }.less);
                var offsets: [max_route_ids]u64 = undefined;
                var lengths: [max_route_ids]u64 = undefined;
                const geom = &bank.layers[r.layer];
                const mn = minisOf(geom);
                // A record's file footprint (an id of several records: all of them, the last one's pad included).
                const footprint = if (mn > 1) mn * miniStride(geom) else geom.logical_bytes;
                for (r.order[0..n], 0..) |li, k| {
                    offsets[k] = bank.recordOffset(r.layer, plan.loads[li].expert);
                    lengths[k] = footprint;
                }
                var ends_buf: [max_route_ids]u32 = undefined;
                const ends = if (plan.phase == .decode)
                    expert_policy.boundedParts(offsets[0..n], lengths[0..n], self.records_per_part, &ends_buf)
                else blk: {
                    var k: u32 = 0;
                    var e: u32 = 0;
                    while (e < n) : (k += 1) {
                        e = @min(e + expert_io.max_items / mn, n);
                        ends_buf[k] = e;
                    }
                    break :blk ends_buf[0..k];
                };
                const ls = &self.layers[r.layer];
                var start: u32 = 0;
                for (ends) |end| {
                    var part: Part = .{ .first = start, .n = end - start };
                    var rows: [expert_io.max_items][n_components]u64 = undefined;
                    var gu: [expert_io.max_items]u64 = undefined;
                    var down: [expert_io.max_items]u64 = undefined;
                    var ids: [expert_io.max_items]u16 = undefined;
                    var nr: u32 = 0;
                    for (r.order[start..end]) |li| {
                        if (!r.reads[li]) continue;
                        const l = plan.loads[li];
                        nr = self.entries(r.layer, l.slot, l.expert, &rows, &gu, &down, &ids, nr);
                    }
                    if (nr > 0) {
                        part.ticket = try self.submit(r.layer, ids[0..nr], gu[0..nr], down[0..nr], rows[0..nr], &ls.lens);
                        part.n_reads = nr;
                    } else part.settled = true;
                    r.parts[r.n_parts] = part;
                    r.n_parts += 1;
                    start = end;
                }
            }

            /// One pool job over records of `layer` (experts `ids`): their gate/up and down spans in the bank's file, or
            /// each part's own place in a scatter bank.
            fn submit(self: *Stream, layer: u32, ids: []const u16, gu: []const u64, down: []const u64, rows: []const [n_components]u64, lens: *const [n_components]u64) Error!u32 {
                if (comptime scatter) {
                    var parts: [expert_io.max_items][n_components]B.Records.Part = undefined;
                    for (ids, parts[0..ids.len]) |e, *p| p.* = self.bank.parts(layer, e);
                    return B.Records.submitScatter(self.pool, parts[0..ids.len], rows, lens) catch |err| self.fail(err);
                }
                return B.Records.submit(self.pool, self.bank.sidecar, gu, down, rows, lens) catch |err| self.fail(err);
            }

            /// Load (`slot`, `e`)'s read entries from index `at`, one per record of the id (its records adjacent in the file
            /// at `miniStride` and in the slot rows at each segment's length): each record's gate/up and down spans, its
            /// expert and its rows' addresses. Returns the next index.
            fn entries(self: *Stream, layer: u32, slot: u32, e: u16, rows: *[expert_io.max_items][n_components]u64, gu: *[expert_io.max_items]u64, down: *[expert_io.max_items]u64, ids: *[expert_io.max_items]u16, at: u32) u32 {
                const geom = &self.bank.layers[layer];
                const loc = self.locate(layer, slot);
                const d0 = loc.rows.rowDest(loc.row);
                const sp = self.bank.spans(layer, e);
                const stride = miniStride(geom);
                var k = at;
                for (0..minisOf(geom)) |mi| {
                    const m: u64 = @intCast(mi);
                    for (&rows[k], d0, geom.segments) |*d, base, sg| d.* = base + m * sg.length;
                    gu[k] = sp.gu_offset + m * stride;
                    down[k] = sp.down_offset + m * stride;
                    ids[k] = e;
                    k += 1;
                }
                return k;
            }

            /// Blocks until the part's gate/up segments landed (its down segments may
            /// still be reading): the gate/up kernels of its records can run.
            pub fn waitGu(self: *Stream, r: *Route, part: u32) Error!void {
                const p = &r.parts[part];
                if (p.settled) return;
                self.waitReads(p.ticket, p.n_reads) catch |e| return self.fail(e);
                for (0..p.n_reads) |i| {
                    if (self.pool.result(p.ticket + @as(u32, @intCast(i))).status != .ok) return self.fail(error.ReadFailed);
                }
            }

            /// The host blocked until tickets [first, first + count) landed, its time in `read_wait_ns`.
            fn waitReads(self: *Stream, first: u32, count: u32) !void {
                const t0 = expert_io.Pool.nowNs();
                defer self.counters.read_wait_ns += @intCast(@max(expert_io.Pool.nowNs() - t0, 0));
                return self.pool.wait(first, count, wait_timeout_ns);
            }

            /// Blocks until every segment of the part landed; its rows are then ready.
            pub fn waitDown(self: *Stream, r: *Route, part: u32) Error!void {
                return self.settle(r, &r.parts[part]);
            }

            fn settle(self: *Stream, r: *Route, p: *Part) Error!void {
                if (p.settled) return;
                const count = 2 * p.n_reads;
                const waited = self.waitReads(p.ticket, count);
                var ok = if (waited) |_| true else |_| false;
                if (ok) {
                    for (0..count) |k| {
                        const res = self.pool.result(p.ticket + @as(u32, @intCast(k)));
                        if (res.status != .ok) ok = false;
                        self.counters.expert_bytes_read += @intCast(@max(res.payload, 0));
                        self.counters.preadv_calls += @intCast(@max(res.preadv_calls, 0));
                        self.noteRead(res);
                    }
                }
                p.settled = true;
                const ls = &self.layers[r.layer];
                // A read load's records are entries `i ..` of the part (`entries`): its first one's gate/up ticket is `i`.
                const mn = minisOf(&self.bank.layers[r.layer]);
                var i: u32 = 0;
                for (r.order[p.first..][0..p.n]) |li| {
                    if (!r.reads[li]) continue;
                    const l = r.plan.loads[li];
                    self.locate(r.layer, l.slot).meta.state = if (ok) .ready else .failed;
                    if (!ok and l.persistent) ls.policy.invalidate(l.expert);
                    // The recorder: a gate/up range that ran no preadv was copied out of a lookahead record.
                    if (comptime recording) if (self.recorder) |rec| if (ok) {
                        const gu = self.pool.result(p.ticket + i);
                        rec.settled(r.rec, l.expert, gu.preadv_calls == 0 and gu.payload > 0);
                    };
                    i += mn;
                }
                if (!ok) return self.fail(if (waited) |_| error.ReadFailed else |e| e);
            }

            /// P1: layer `layer`'s predicted seed (`experts`, hottest first) read into its persistent rows, no route, while its
            /// attention runs: `LayerPolicy.admitReadAheadHeld` (an empty row, else the coldest resident the prediction left out
            /// that no live route of the layer holds; unprotected; the seed re-protects its choices), pool jobs of
            /// `max_items` within the ticket ring. `awaitReadAhead` or a route of the layer lands it; a live one lands first.
            /// `stock`: every record now, in file order. P7 `overlap`: the prediction's rank order (the admission order), its
            /// first `wide_depth x max_route_ids` records now (the groups the call routes at its start); the rest wait for
            /// the call's plans (`planCall`: `p7Tail`), or for the layer's await (every record then). `whole` and
            /// `whole_trim`: every record now (`whole_trim` withdraws the jobs still queued at the plans: `p7Withdraw`).
            pub fn readAheadSeed(self: *Stream, layer: u32, experts: []const u16) !void {
                if (self.failed) return error.StreamFailed;
                if (self.phase != .prefill) return error.NotPrefill;
                if (self.ahead.live) try self.awaitReadAhead(self.ahead.layer);
                self.prompt_started = true;
                const ah = &self.ahead;
                const mn = minisOf(&self.bank.layers[layer]);
                // The ticket ring's room in records (each record's minis take a gate/up and a down ticket apiece).
                const fit = @min(ah.loads.len, self.ahead_cap orelse (self.pool.published.len - 2 * expert_io.max_items) / (2 * mn));
                // The rows live routes of the layer serve from and the deferred base call's are never a read-ahead's victims
                // (as a plan beside them holds them): their records are being read by a wave.
                const held_set = &self.held_scratch;
                held_set.clearRetainingCapacity();
                for (&self.routes) |*o| {
                    if (o.state == .free or o.layer != layer) continue;
                    for (o.hit_slots[0..o.plan.n_hits]) |hs| if (hs < self.layers[layer].policy.capacity) {
                        held_set.append(self.allocator, hs) catch return self.fail(error.RoutesExhausted);
                    };
                    for (o.plan.loadsOf()) |l| if (l.persistent) {
                        held_set.append(self.allocator, l.slot) catch return self.fail(error.RoutesExhausted);
                    };
                }
                if (self.held_base.items.len > 0 and self.held_layer == layer)
                    held_set.appendSlice(self.allocator, self.held_base.items) catch return self.fail(error.RoutesExhausted);
                // Nor a row a wave still pins or a load is filling (every such row is one of the above today; held by its
                // own mark so the admission can never hand out a row the check below would fail on).
                for (self.layers[layer].meta[0..self.layers[layer].policy.capacity], 0..) |m, s| if (m.pins != 0 or m.state == .loading) {
                    held_set.append(self.allocator, @intCast(s)) catch return self.fail(error.RoutesExhausted);
                };
                const admitted = self.layers[layer].policy.admitReadAheadHeld(experts, held_set.items, ah.loads[0..fit]);
                if (comptime read_ahead_probed) if (self.probe) |pr| {
                    // The predicted seed: the ranking's top as many as the layer's unprotected rows (the seed's rule); blocked:
                    // those neither resident nor admitted (no row free to take: every row protected, pinned or held).
                    const pol = &self.layers[layer].policy;
                    const room: usize = pol.capacity -| @as(u32, @intCast(pol.protected.count()));
                    const top = experts[0..@min(experts.len, room)];
                    var blocked: u32 = 0;
                    for (top) |e| blocked += @intFromBool(pol.slotOf(e) == null);
                    pr.admission(layer, top, @intCast(admitted.len), blocked);
                };
                const n: u32 = @intCast(admitted.len);
                ah.* = .{ .layer = layer, .n = n, .loads = ah.loads, .reads = ah.reads, .parts = ah.parts, .landed = ah.landed };
                if (n == 0) return;
                const now: u32 = switch (self.prompt_reads) {
                    .stock => blk: {
                        std.sort.pdq(LayerPolicy.ReadAhead, admitted, {}, struct {
                            fn less(_: void, x: LayerPolicy.ReadAhead, y: LayerPolicy.ReadAhead) bool {
                                return x.expert < y.expert;
                            }
                        }.less);
                        break :blk n;
                    },
                    .overlap => switch (self.prompt_ahead) {
                        .head => @min(n, @as(u32, self.wide_depth) * self.max_route_ids),
                        .whole, .whole_trim => n,
                    },
                };
                for (admitted, ah.reads[0..n]) |l, *rd| {
                    const m = self.locate(layer, l.slot).meta;
                    if (m.pins != 0 or m.state == .loading) return self.fail(error.SlotStillPinned);
                    rd.* = !(m.state == .ready and m.layer == layer and m.expert == l.expert);
                    if (rd.*) m.* = .{ .state = .loading, .layer = @intCast(layer), .expert = l.expert };
                    m.pins = 1;
                }
                ah.live = true;
                if (self.ahead_job.len > 0) @memset(self.ahead_job, no_job);
                try self.postAheadJobs(layer, now);
            }

            /// The read-ahead's jobs over its admitted records [`posted`, `to`): pool jobs of `max_items` records each.
            fn postAheadJobs(self: *Stream, layer: u32, to: u32) Error!void {
                const ah = &self.ahead;
                const lens = &self.layers[layer].lens;
                const mn = minisOf(&self.bank.layers[layer]);
                var start: u32 = ah.posted;
                while (start < to) {
                    const end = @min(start + expert_io.max_items / mn, to);
                    var part: Part = .{ .first = start, .n = end - start };
                    var rows: [expert_io.max_items][n_components]u64 = undefined;
                    var gu: [expert_io.max_items]u64 = undefined;
                    var down: [expert_io.max_items]u64 = undefined;
                    var ids: [expert_io.max_items]u16 = undefined;
                    var nr: u32 = 0;
                    var posted: u32 = 0;
                    for (ah.loads[start..end], ah.reads[start..end]) |l, rd| {
                        if (self.ahead_job.len > 0) self.ahead_job[l.slot] = @intCast(ah.n_parts);
                        if (!rd) continue;
                        nr = self.entries(layer, l.slot, l.expert, &rows, &gu, &down, &ids, nr);
                        posted += 1;
                    }
                    if (nr > 0) {
                        part.ticket = try self.submit(layer, ids[0..nr], gu[0..nr], down[0..nr], rows[0..nr], lens);
                        part.n_reads = nr;
                        self.counters.ahead_posted += posted;
                        if (comptime read_ahead_probed) if (self.probe) |pr| pr.posted(layer, posted);
                    } else part.settled = true;
                    ah.parts[ah.n_parts] = part;
                    ah.landed[ah.n_parts] = false;
                    ah.n_parts += 1;
                    start = end;
                }
                ah.posted = to;
            }

            /// P1: lands layer `layer`'s read-ahead (none live for it: nothing to do). Every job waited and its
            /// rows ready, the rows' pins dropped; a record that failed is forgotten by the policy and fails the
            /// stream, as a route's failed load does. Records still waiting for a plan (P7 overlap) are read first.
            pub fn awaitReadAhead(self: *Stream, layer: u32) Error!void {
                const ah = &self.ahead;
                if (!ah.live or ah.layer != layer) return;
                if (ah.posted < ah.n) try self.postAheadJobs(layer, ah.n);
                ah.live = false;
                var first_error: ?Error = null;
                for (0..ah.n_parts) |j| self.landAheadJob(@intCast(j)) catch |e| {
                    if (first_error == null) first_error = e;
                };
                if (first_error) |e| return e;
            }

            /// Lands read-ahead job `j` once: its reads waited, its rows' states set (a failed record forgotten by the
            /// policy, the stream failed), its rows unpinned.
            fn landAheadJob(self: *Stream, j: u32) Error!void {
                const ah = &self.ahead;
                if (ah.landed[j]) return;
                ah.landed[j] = true;
                const p = &ah.parts[j];
                const policy = &self.layers[ah.layer].policy;
                var failure: ?Error = null;
                if (!p.settled) {
                    p.settled = true;
                    const count = 2 * p.n_reads;
                    const waited = self.waitReads(p.ticket, count);
                    var ok = if (waited) |_| true else |_| false;
                    if (ok) for (0..count) |k| {
                        const res = self.pool.result(p.ticket + @as(u32, @intCast(k)));
                        if (res.status != .ok) ok = false;
                        self.counters.expert_bytes_read += @intCast(@max(res.payload, 0));
                        self.counters.ahead_bytes += @intCast(@max(res.payload, 0));
                        self.counters.preadv_calls += @intCast(@max(res.preadv_calls, 0));
                        self.noteRead(res);
                    };
                    for (ah.loads[p.first..][0..p.n], ah.reads[p.first..][0..p.n]) |l, rd| if (rd) {
                        self.locate(ah.layer, l.slot).meta.state = if (ok) .ready else .failed;
                        if (!ok) policy.invalidate(l.expert);
                    };
                    if (!ok) failure = if (waited) |_| error.ReadFailed else |e| e;
                }
                for (ah.loads[p.first..][0..p.n]) |l| self.locate(ah.layer, l.slot).meta.pins -= 1;
                if (failure) |e| return self.fail(e);
            }

            /// Whether read-ahead job `j` has landed or its reads have all completed (never blocks).
            fn aheadJobDone(self: *Stream, j: u32) bool {
                const ah = &self.ahead;
                if (ah.landed[j] or ah.parts[j].settled) return true;
                const p = ah.parts[j];
                self.pool.wait(p.ticket, 2 * p.n_reads, 0) catch return false;
                return true;
            }

            /// P7 overlap: the bit of the read-ahead job writing persistent slot `slot` of `layer` while it has not landed
            /// (0 otherwise).
            fn aheadJobBit(self: *const Stream, layer: u32, slot: u32) u64 {
                const ah = &self.ahead;
                if (!ah.live or ah.layer != layer or slot >= self.layers[layer].policy.capacity) return 0;
                const j = self.ahead_job[slot];
                if (j == no_job or ah.landed[j]) return 0;
                return @as(u64, 1) << @intCast(j);
            }

            // ── P7 overlap: the prompt call's plans ahead, its reads posted when their rows are free ──

            /// P7 overlap: a prompt call's groups -- `distinct` in groups of `width`, in the order the executor routes them,
            /// at its depth `wide_depth` -- all planned now as the stock schedule plans them (the same `planWith` per group
            /// over the same ids, with the held set and transient window that schedule gives it); the read-ahead's waiting
            /// records (`whole_trim`: with its jobs still queued, withdrawn: `p7Withdraw`) resolved by the plans (`p7Tail`);
            /// every read whose row is free posted, in group order. The executor then routes the groups in order
            /// (`activateP7`) and releases them in order (`releaseP7`).
            pub fn planCall(self: *Stream, layer: u32, distinct: []const u16, width: u32) Error!void {
                if (self.failed) return error.StreamFailed;
                std.debug.assert(self.prompt_reads == .overlap and self.phase == .prefill);
                self.prompt_started = true;
                if (self.p7.live) return self.fail(error.RoutesLive);
                try self.flush();
                if (self.n_free != route_capacity or self.held_base.items.len > 0) return self.fail(error.RoutesLive);
                if (width == 0 or width > self.max_route_ids) return self.fail(error.RoutesExhausted);
                const n_groups: u32 = @intCast((distinct.len + width - 1) / width);
                if (n_groups == 0 or n_groups > route_capacity) return self.fail(error.RoutesExhausted);
                const p = &self.p7;
                p.* = .{ .live = true, .layer = layer, .distinct = distinct, .width = width, .n_groups = n_groups };
                const ls = &self.layers[layer];
                const cap = ls.policy.capacity;
                const depth: u32 = self.wide_depth;
                for (0..n_groups) |gi| {
                    const g: u32 = @intCast(gi);
                    // The stock schedule routes group g beside groups g - depth + 1 .. g - 1 (`route`): their windows are
                    // taken and their persistent slots held.
                    const held = &self.held_scratch;
                    held.clearRetainingCapacity();
                    var used: u8 = 0;
                    var q: u32 = g -| (depth - 1);
                    while (q < g) : (q += 1) {
                        used |= @as(u8, 1) << @intCast(p.window[q]);
                        const o = &self.routes[p.route[q]];
                        for (o.hit_slots[0..o.plan.n_hits]) |hs| if (hs < cap) {
                            held.append(self.allocator, hs) catch return self.fail(error.RoutesExhausted);
                        };
                        for (o.plan.loadsOf()) |l| if (l.persistent) {
                            held.append(self.allocator, l.slot) catch return self.fail(error.RoutesExhausted);
                        };
                    }
                    const window: u8 = @intCast(@ctz(~used));
                    if (window >= depth) return self.fail(error.RoutesExhausted);
                    self.n_free -= 1;
                    const ri = self.free[self.n_free];
                    const r = &self.routes[ri];
                    r.* = .{ .layer = layer, .window = window, .state = .live };
                    p.route[g] = ri;
                    p.window[g] = window;
                    ls.policy.planWith(self.p7Group(g), .prefill, &r.plan, .{ .transient_base = @as(u32, window) * self.max_route_ids, .held = held.items });
                    for (r.plan.hitsOf(), r.hit_slots[0..r.plan.n_hits]) |e, *slot| slot.* = ls.policy.slotOf(e).?;
                }
                if (self.prompt_ahead == .whole_trim) try self.p7Withdraw(layer);
                try self.p7Tail(layer);
                if (self.prompt_relocate) try self.p7Relocate(layer);
                try self.p7Rows(layer);
                for (0..n_groups) |g| try self.p7Parts(@intCast(g));
                try self.p7Post(0);
            }

            /// Group `g` of the scheduled call: its ids, as the executor routes them.
            fn p7Group(self: *const Stream, g: u32) []const u16 {
                const p = &self.p7;
                return p.distinct[g * p.width .. @min((g + 1) * p.width, p.distinct.len)];
            }

            /// A row's index in `p7_rows`: persistent slots first, the transient rows after `n_experts`.
            fn p7Index(self: *const Stream, layer: u32, slot: u32) usize {
                const cap = self.layers[layer].policy.capacity;
                return if (slot < cap) slot else self.bank.n_experts + (slot - cap);
            }

            /// P7 `whole_trim`: at the call's plans, the read-ahead's jobs no reader worker has taken yet are withdrawn from
            /// the pool's queue (`Pool.withdraw`, last job first: the pool's one FIFO starts jobs in post order, so the
            /// withdrawn jobs are always the post's last ones) and their records wait again, from the first withdrawn job's
            /// record on, for `p7Tail` to resolve them as it resolves `head`'s tail. A withdrawn job read nothing and wrote
            /// no row; a job a worker has taken runs as posted. Plans, slots and kernels are untouched.
            fn p7Withdraw(self: *Stream, layer: u32) Error!void {
                const ah = &self.ahead;
                if (!ah.live or ah.layer != layer) return;
                var first = ah.n_parts;
                var unread: u64 = 0;
                // A job's reads are its records' minis, one gate/up and one down range each.
                const mn = minisOf(&self.bank.layers[layer]);
                while (first > 0) {
                    const j = first - 1;
                    if (ah.landed[j]) break;
                    const p = &ah.parts[j];
                    // A job of records whose rows already held them has no reads (no ticket): nothing to withdraw.
                    if (!p.settled) {
                        const withdrawn = self.pool.withdraw(p.ticket) catch return self.fail(error.SubmitRefused);
                        if (!withdrawn) break;
                        unread += p.n_reads / mn;
                    }
                    first = j;
                }
                if (first == ah.n_parts) return;
                const from = ah.parts[first].first;
                // Their rows wait for the plans again: no job writes them (`p7Tail` posts the kept ones anew).
                for (ah.loads[from..ah.n]) |l| self.ahead_job[l.slot] = no_job;
                ah.n_parts = first;
                ah.posted = from;
                // `ahead_posted` counts records read ahead: the withdrawn reads never ran (`p7Tail` counts the kept again).
                self.counters.ahead_posted -= unread;
            }

            /// P7 overlap: the read-ahead's records the head left (`readAheadSeed`), resolved by the call's plans: read
            /// (posted now, ordered by the first group that hits them) when a group hits one before any evicts it, or when
            /// no group touches it; dropped when evicted first -- its row is then the evicting load's, and the record is read
            /// once (where a plan loads it again) instead of twice.
            fn p7Tail(self: *Stream, layer: u32) Error!void {
                const ah = &self.ahead;
                if (!ah.live or ah.layer != layer or ah.posted == ah.n) return;
                const p = &self.p7;
                // Each waiting record's fate: the first group hitting it (kept), or evicting it first (dropped: the sentinel).
                const dropped: u8 = std.math.maxInt(u8);
                var first: [256]u8 = undefined;
                const tail = ah.n - ah.posted;
                if (tail > first.len) return self.fail(error.RoutesExhausted);
                for (ah.loads[ah.posted..ah.n], ah.reads[ah.posted..ah.n], first[0..tail]) |l, rd, *f| {
                    f.* = @intCast(p.n_groups);
                    groups: for (0..p.n_groups) |g| {
                        const r = &self.routes[p.route[g]];
                        for (r.hit_slots[0..r.plan.n_hits]) |s| if (s == l.slot) {
                            f.* = @intCast(g);
                            break :groups;
                        };
                        for (r.plan.evictionsOf()) |ev| if (ev.slot == l.slot and ev.previous == l.expert) {
                            // Only a record still to be read is dropped (one whose row already held it costs nothing).
                            f.* = if (rd) dropped else @intCast(g);
                            break :groups;
                        };
                    }
                }
                // Kept records first, by their first group; dropped records after them (never posted).
                var i: u32 = 1;
                while (i < tail) : (i += 1) {
                    var k = i;
                    while (k > 0 and first[k - 1] > first[k]) : (k -= 1) {
                        std.mem.swap(u8, &first[k - 1], &first[k]);
                        std.mem.swap(LayerPolicy.ReadAhead, &ah.loads[ah.posted + k - 1], &ah.loads[ah.posted + k]);
                        std.mem.swap(bool, &ah.reads[ah.posted + k - 1], &ah.reads[ah.posted + k]);
                    }
                }
                var kept: u32 = 0;
                while (kept < tail and first[kept] != dropped) kept += 1;
                for (ah.loads[ah.posted + kept .. ah.n]) |l| {
                    const m = self.locate(layer, l.slot).meta;
                    m.state = .empty;
                    m.pins -= 1;
                    self.counters.ahead_dropped += 1;
                }
                ah.n = ah.posted + kept;
                try self.postAheadJobs(layer, ah.n);
            }

            /// P7 overlap relocation (`Options.prompt_relocate`): each read-ahead record whose first event in the call's plans
            /// is its row's eviction (an earlier group's seed or miss load takes the row before any group hits the record)
            /// and that a later group loads again into a transient row no earlier group of the call reads (loads only: a
            /// transient row is never a hit) is copied, now, from its read-ahead row into that transient row, every
            /// component's row whole, and the row's meta made the record's: `p7Rows` then counts the row as holding it and
            /// `p7Parts` makes that load a held one (no read). The record's read-ahead job is landed first (its rows hold
            /// their records); the copies run before the call posts any read (`p7Post`), so no read writes a source row
            /// before it is copied and no group reads a target row before the copy (every earlier group of the call reads
            /// other rows; the previous call's routes are flushed). Plans, slots and kernels are the stock schedule's, and a
            /// target row holds the bytes a read of the record would write (a row-for-row copy of the record's read):
            /// exact by construction.
            fn p7Relocate(self: *Stream, layer: u32) Error!void {
                const ah = &self.ahead;
                if (!ah.live or ah.layer != layer) return;
                const p = &self.p7;
                const cap = self.layers[layer].policy.capacity;
                const copies = &self.promote_copies;
                copies.clearRetainingCapacity();
                var bytes: u64 = 0;
                records: for (ah.loads[0..ah.n]) |l| {
                    // The record's first event, in group order: a hit keeps it in place; its row's eviction marks it.
                    const evicted: usize = first: for (0..p.n_groups) |gi| {
                        const r = &self.routes[p.route[gi]];
                        for (r.hit_slots[0..r.plan.n_hits]) |s| if (s == l.slot) continue :records;
                        for (r.plan.evictionsOf()) |ev| if (ev.slot == l.slot and ev.previous == l.expert) break :first gi;
                    } else continue :records;
                    // Its load in a later group: a transient row that no group before that one loads.
                    const to: u32 = later: for (evicted + 1..p.n_groups) |gi| {
                        const r = &self.routes[p.route[gi]];
                        for (r.plan.loadsOf()) |ld| if (ld.expert == l.expert) {
                            if (ld.slot < cap) continue :records;
                            for (0..gi) |gj| for (self.routes[p.route[gj]].plan.loadsOf()) |x| if (x.slot == ld.slot) continue :records;
                            break :later ld.slot;
                        };
                    } else continue :records;
                    // The source row holds the record once its job has landed (a record its row already held has a job too).
                    const j = self.ahead_job[l.slot];
                    if (j != no_job) try self.landAheadJob(j);
                    const src = self.locate(layer, l.slot).meta;
                    std.debug.assert(src.state == .ready and src.layer == layer and src.expert == l.expert);
                    // Capacity reserved at construction: every persistent row's components.
                    for (0..n_components) |c| {
                        const dst = self.slotRow(layer, to, @enumFromInt(c));
                        copies.appendAssumeCapacity(.{ .dst = dst, .src = self.slotRow(layer, l.slot, @enumFromInt(c)) });
                        bytes += dst.len;
                    }
                    self.locate(layer, to).meta.* = .{ .state = .ready, .layer = @intCast(layer), .expert = l.expert };
                }
                if (bytes == 0) return;
                // No promotion still reads a transient row (none on an overlap stream; settled here regardless).
                self.settleCopies();
                copyParallel(copies.items, bytes);
            }

            /// P7 overlap: every row's record and reader as the call starts (the slot meta, a read-ahead record counted as
            /// landed: the stock schedule plans after the whole read-ahead landed).
            fn p7Rows(self: *Stream, layer: u32) Error!void {
                const ls = &self.layers[layer];
                const cap = ls.policy.capacity;
                const n_e = self.bank.n_experts;
                const ah = &self.ahead;
                const ra_live = ah.live and ah.layer == layer;
                if (self.transient.rows > self.p7_rows.len - n_e) return self.fail(error.RoutesExhausted);
                for (self.p7_rows[0..cap], ls.meta[0..cap], 0..) |*v, m, s| {
                    const ra = ra_live and self.ahead_job[s] != no_job;
                    if (m.state == .loading and !ra) return self.fail(error.SlotStillPinned);
                    v.* = .{ .layer = m.layer, .expert = m.expert, .ready = m.state == .ready or (m.state == .loading and ra), .reader = -1 };
                }
                for (self.p7_rows[n_e..][0..self.transient.rows], self.transient_meta[0..self.transient.rows]) |*v, m| {
                    if (m.state == .loading) return self.fail(error.SlotStillPinned);
                    v.* = .{ .layer = m.layer, .expert = m.expert, .ready = m.state == .ready, .reader = -1 };
                }
            }

            /// P7 overlap: group `g`'s loads -- whether each reads (the stock test on the rows the earlier plans leave), the
            /// last earlier group reading its row, the read-ahead job writing it -- cut into parts by (that reader, read-ahead
            /// blocked, file order) at `max_items`; the stock route's counters.
            fn p7Parts(self: *Stream, g: u32) Error!void {
                const p = &self.p7;
                const layer = p.layer;
                const r = &self.routes[p.route[g]];
                const plan = &r.plan;
                const depth: u32 = self.wide_depth;
                var key: [max_route_ids]i8 = undefined;
                var ra: [max_route_ids]u64 = undefined;
                var skipped: u64 = 0;
                for (plan.loadsOf(), 0..) |l, i| {
                    const v = &self.p7_rows[self.p7Index(layer, l.slot)];
                    const held = v.ready and v.layer == layer and v.expert == l.expert;
                    r.reads[i] = !held;
                    skipped += @intFromBool(held);
                    // The stock schedule never plans a row a live route reads: its last reader is `depth` or more groups back.
                    if (v.reader >= 0 and @as(u32, @intCast(v.reader)) + depth > g) return self.fail(error.SlotStillPinned);
                    key[i] = v.reader;
                    ra[i] = self.aheadJobBit(layer, l.slot);
                    v.layer = @intCast(layer);
                    v.expert = l.expert;
                    v.ready = true;
                }
                for (r.hit_slots[0..plan.n_hits]) |s| self.p7_rows[self.p7Index(layer, s)].reader = @intCast(g);
                for (plan.loadsOf()) |l| self.p7_rows[self.p7Index(layer, l.slot)].reader = @intCast(g);
                const n = plan.n_loads;
                for (r.order[0..n], 0..) |*o, i| o.* = @intCast(i);
                const Ctx = struct { plan: *const Plan, key: *const [max_route_ids]i8, ra: *const [max_route_ids]u64 };
                std.sort.insertion(u8, r.order[0..n], Ctx{ .plan = plan, .key = &key, .ra = &ra }, struct {
                    fn less(c: Ctx, a: u8, b: u8) bool {
                        if (c.key[a] != c.key[b]) return c.key[a] < c.key[b];
                        if ((c.ra[a] != 0) != (c.ra[b] != 0)) return c.ra[a] == 0;
                        return c.plan.loads[a].expert < c.plan.loads[b].expert;
                    }
                }.less);
                r.n_parts = 0;
                var start: u32 = 0;
                while (start < n) {
                    const k0 = key[r.order[start]];
                    const b0 = ra[r.order[start]] != 0;
                    var end = start + 1;
                    while (end < n and end - start < expert_io.max_items and key[r.order[end]] == k0 and (ra[r.order[end]] != 0) == b0) end += 1;
                    var m: u64 = 0;
                    for (r.order[start..end]) |li| m |= ra[li];
                    r.parts[r.n_parts] = .{ .first = start, .n = end - start };
                    p.key[g][r.n_parts] = k0;
                    p.ra[g][r.n_parts] = m;
                    r.n_parts += 1;
                    start = end;
                }
                const c = &self.counters;
                c.route_calls += 1;
                c.expert_cache_hits += plan.n_hits;
                c.expert_cache_misses += plan.n_misses;
                c.expert_cache_evictions += plan.n_evictions;
                c.persistent_loads += plan.n_persistent;
                c.transient_loads += plan.n_loads - plan.n_persistent;
                c.loads_skipped += skipped;
            }

            /// P7 overlap: posts the call's parts in group order from the next unposted one. A part whose reading group has
            /// not been released stops the walk; so does (groups from `force_upto` on) one whose read-ahead jobs have not
            /// completed. Groups below `force_upto` are being routed: their read-ahead jobs land here (blocking).
            fn p7Post(self: *Stream, force_upto: u32) Error!void {
                const p = &self.p7;
                const ah = &self.ahead;
                const ra_live = ah.live and ah.layer == p.layer;
                while (p.post_g < p.n_groups) : ({
                    p.post_g += 1;
                    p.post_p = 0;
                }) {
                    const g = p.post_g;
                    const r = &self.routes[p.route[g]];
                    const force = g < force_upto;
                    while (p.post_p < r.n_parts) : (p.post_p += 1) {
                        const pi = p.post_p;
                        const key = p.key[g][pi];
                        if (key >= 0 and @as(u32, @intCast(key)) >= p.released) {
                            if (force) return self.fail(error.SlotStillPinned);
                            return;
                        }
                        if (ra_live) {
                            var m = p.ra[g][pi];
                            while (m != 0) : (m &= m - 1) {
                                const j: u32 = @ctz(m);
                                if (!force and !self.aheadJobDone(j)) return;
                                try self.landAheadJob(j);
                            }
                        }
                        try self.p7Submit(r, pi);
                    }
                }
            }

            /// P7 overlap: one part's loads as the stock route takes them (a row a live route still serves from is never
            /// refilled; a read marks its row loading; every load pins its row) and its pool job.
            fn p7Submit(self: *Stream, r: *Route, pi: u32) Error!void {
                const plan = &r.plan;
                const part = &r.parts[pi];
                const ls = &self.layers[r.layer];
                var rows: [expert_io.max_items][n_components]u64 = undefined;
                var gu: [expert_io.max_items]u64 = undefined;
                var down: [expert_io.max_items]u64 = undefined;
                var ids: [expert_io.max_items]u16 = undefined;
                var nr: u32 = 0;
                for (r.order[part.first..][0..part.n]) |li| {
                    const l = plan.loads[li];
                    const loc = self.locate(r.layer, l.slot);
                    const m = loc.meta;
                    if (m.pins != 0 or m.state == .loading) return self.fail(error.SlotStillPinned);
                    if (r.reads[li]) {
                        m.* = .{ .state = .loading, .layer = @intCast(r.layer), .expert = l.expert };
                        rows[nr] = loc.rows.rowDest(loc.row);
                        const sp = self.bank.spans(r.layer, l.expert);
                        gu[nr] = sp.gu_offset;
                        down[nr] = sp.down_offset;
                        ids[nr] = l.expert;
                        nr += 1;
                    }
                    m.pins = 1;
                }
                if (nr > 0) {
                    part.ticket = try self.submit(r.layer, ids[0..nr], gu[0..nr], down[0..nr], rows[0..nr], &ls.lens);
                    part.n_reads = nr;
                } else part.settled = true;
            }

            /// P7 overlap: the executor routes group `activated` (its depth schedule reached it): the planned route, every
            /// part of it and of the groups before it posted (their blockers are gone by the stock schedule's own order; the
            /// read-ahead jobs they wait on land here), the read-ahead records it hits landed, its hits pinned.
            pub fn activateP7(self: *Stream, ids: []const u16) Error!*Route {
                if (self.failed) return error.StreamFailed;
                const p = &self.p7;
                std.debug.assert(p.live);
                const g = p.activated;
                if (g >= p.n_groups) return self.fail(error.RoutesExhausted);
                const want = self.p7Group(g);
                if (ids.ptr != want.ptr or ids.len != want.len) return self.fail(error.RoutesLive);
                try self.flush();
                try self.p7Post(g + 1);
                const r = &self.routes[p.route[g]];
                for (r.hit_slots[0..r.plan.n_hits]) |s| {
                    const bit = self.aheadJobBit(p.layer, s);
                    if (bit != 0) try self.landAheadJob(@ctz(bit));
                    self.locate(p.layer, s).meta.pins += 1;
                }
                p.activated += 1;
                return r;
            }

            /// P7 overlap: the executor releases the call's groups in order; the call's schedule ends with the last. Out
            /// of order (an error path's cleanup) the schedule no longer holds: the stream fails.
            pub fn releaseP7(self: *Stream, r: *Route) void {
                const p = &self.p7;
                if (!p.live or p.released >= p.activated or &self.routes[p.route[p.released]] != r) {
                    self.failed = true;
                    if (r.state == .live) self.release(r);
                    return;
                }
                self.release(r);
                p.released += 1;
                if (p.released == p.n_groups) p.live = false;
            }

            /// Construction (the prompt phase, nothing live): each layer's empty prompt rows filled with the experts of
            /// `order(layer)` in turn, read and landed as a read-ahead leaves them (unprotected residents a prompt's seed
            /// re-protects). Returns the records read.
            pub fn preload(self: *Stream, ctx: anytype, comptime order: fn (@TypeOf(ctx), u32) []const u16) !u64 {
                if (self.phase != .prefill) return error.NotPrefill;
                var n: u64 = 0;
                const by_rate = try self.allocator.alloc(u16, self.bank.n_experts);
                defer self.allocator.free(by_rate);
                for (0..self.layers.len) |l| {
                    const layer: u32 = @intCast(l);
                    const arch_order = order(ctx, layer);
                    // A seeded tally fills the rows hottest first (rate descending, the arch's order among equal rates).
                    const p = &self.layers[l].policy;
                    const ids = if (p.seeded and p.pin_limit > 0) blk: {
                        const out = by_rate[0..arch_order.len];
                        @memcpy(out, arch_order);
                        std.sort.insertion(u16, out, @as([]const f32, p.rate), struct {
                            fn hotter(rs: []const f32, x: u16, y: u16) bool {
                                return rs[x] > rs[y];
                            }
                        }.hotter);
                        break :blk out;
                    } else arch_order;
                    try self.readAheadSeed(layer, ids);
                    for (self.ahead.reads[0..self.ahead.n]) |rd| n += @intFromBool(rd);
                    try self.awaitReadAhead(layer);
                }
                // No prompt routed what the preload read.
                self.ahead.tallied = true;
                return n;
            }

            /// The usage seed (`expert.usage`, at construction: the prompt phase, nothing live, every layer's rows empty):
            /// each layer's empty prompt rows filled with `order(ctx, layer)` (hottest first, at most the layer's rows) by
            /// the read-ahead's load path (`readAheadSeed`: the reads a prompt's read-ahead issues, batched per layer, then
            /// landed), as many batches as the ticket ring needs. Then every layer's prompt state goes (`forgetPrompt`), so
            /// each seeded expert is a resident as `shrink(.keep)` leaves one: unprotected, recency 0, never pinned by the
            /// seed; the dynamic policy evicts it as any other. Exact by construction: a seeded row holds its record's bytes
            /// as a read-ahead's does. Returns the records read.
            pub fn seedResidents(self: *Stream, ctx: anytype, comptime order: fn (@TypeOf(ctx), u32) []const u16) !u64 {
                if (self.phase != .prefill) return error.NotPrefill;
                if (self.failed) return error.StreamFailed;
                try self.flush();
                for (&self.routes) |*r| if (r.state != .free) return error.RoutesLive;
                if (self.ahead.live or self.held_base.items.len > 0) return error.RoutesLive;
                for (self.layers) |*ls| if (ls.policy.occupancy != 0) return error.SeedRowsNotEmpty;
                var n: u64 = 0;
                for (0..self.layers.len) |l| {
                    const layer: u32 = @intCast(l);
                    const ids = order(ctx, layer);
                    var done: usize = 0;
                    while (done < ids.len) {
                        try self.readAheadSeed(layer, ids[done..]);
                        const admitted = self.ahead.n;
                        for (self.ahead.reads[0..admitted]) |rd| n += @intFromBool(rd);
                        try self.awaitReadAhead(layer);
                        if (admitted == 0) break;
                        // The admission takes the ids in order and skips none it can place (none is resident: the rows
                        // started empty); it stops at the ring's room or the layer's rows.
                        done += admitted;
                    }
                }
                // No prompt routed what the seed read.
                self.ahead.tallied = true;
                for (self.layers) |*ls| _ = ls.policy.forgetPrompt();
                return n;
            }

            /// The usage profile's route counting on (`LayerPolicy.count_routes`), once at construction: each decode route
            /// counts its experts into the request's counts even without the rate tally (`addUsage` reads them).
            pub fn countRoutes(self: *Stream) void {
                for (self.layers) |*ls| ls.policy.count_routes = true;
            }

            /// The request's usage (`expert.usage.Recorder`), at its end and before the tally's `endTally` (which zeroes the
            /// decode counts): each layer's prompt counts (`prefill_freq`: every routed id of the prompt pass) and decode
            /// counts (`request_counts`: the decode routes that requested each expert) added into `counts` (`n_experts`
            /// per stream layer, layer order). Without the tally the decode counts are zeroed here.
            pub fn addUsage(self: *Stream, counts: []u64) void {
                const n = self.bank.n_experts;
                std.debug.assert(counts.len == self.layers.len * n);
                for (self.layers, 0..) |*ls, l| {
                    const p = &ls.policy;
                    for (counts[l * n ..][0..n], p.prefill_freq, p.request_counts) |*c, f, r| c.* += @as(u64, f) + r;
                    if (p.alpha == 0) {
                        @memset(p.request_counts, 0);
                        p.request_routes = 0;
                    }
                }
            }

            /// P1's construction self-check: `experts` of `layer` (none resident, one route wide) read ahead, hashed, zeroed and
            /// forgotten, then read again by a demand route; each record's bytes must equal both ways, bit for bit. The layer
            /// is left as found (the experts not resident, their rows empty).
            pub fn checkReadAhead(self: *Stream, layer: u32, experts: []const u16) !void {
                const ls = &self.layers[layer];
                if (experts.len == 0 or experts.len > self.max_route_ids or experts.len > ls.policy.capacity - ls.policy.occupancy) return error.ReadAheadCheckShape;
                for (experts) |e| if (ls.policy.slotOf(e) != null) return error.ReadAheadCheckShape;
                var ahead_sums: [max_route_ids][32]u8 = undefined;
                try self.readAheadSeed(layer, experts);
                if (self.ahead.n != experts.len) return error.ReadAheadCheckNotAdmitted;
                try self.awaitReadAhead(layer);
                for (experts, ahead_sums[0..experts.len]) |e, *sum| {
                    const slot = ls.policy.slotOf(e).?;
                    sum.* = self.recordDigest(layer, slot);
                    const mn = minisOf(&self.bank.layers[layer]);
                    for (ls.lens, 0..) |len, c| @memset(self.slotRow(layer, slot, @enumFromInt(c))[0 .. len * mn], 0);
                    self.locate(layer, slot).meta.state = .empty;
                    ls.policy.invalidate(e);
                }
                const r = try self.route(layer, experts, &.{});
                for (r.parts[0..r.n_parts]) |*p| self.settle(r, p) catch |e| {
                    self.release(r);
                    return e;
                };
                // Every expert a load that read (the zeroed rows refilled), its bytes the read-ahead's.
                var mismatch = r.plan.n_loads != experts.len;
                for (r.plan.loadsOf(), r.reads[0..r.plan.n_loads]) |l, rd| {
                    const i = std.mem.indexOfScalar(u16, experts, l.expert) orelse {
                        mismatch = true;
                        continue;
                    };
                    const d = self.recordDigest(layer, l.slot);
                    if (!rd or !std.mem.eql(u8, &ahead_sums[i], &d)) mismatch = true;
                }
                self.release(r);
                try self.flush();
                for (experts) |e| if (ls.policy.slotOf(e)) |slot| {
                    self.locate(layer, slot).meta.state = .empty;
                    ls.policy.invalidate(e);
                };
                self.ahead.n = 0;
                if (mismatch) return error.ReadAheadCheckMismatch;
            }

            /// sha256 of the record a layer's slot holds (its component rows at their logical lengths).
            fn recordDigest(self: *Stream, layer: u32, slot: u32) [32]u8 {
                var h = std.crypto.hash.sha2.Sha256.init(.{});
                const mn = minisOf(&self.bank.layers[layer]);
                for (self.layers[layer].lens, 0..) |len, c| h.update(self.slotRow(layer, slot, @enumFromInt(c))[0 .. len * mn]);
                var d: [32]u8 = undefined;
                h.final(&d);
                return d;
            }

            /// Hands a route back. Its slots stay pinned until the next flush: the
            /// kernels that read them finish only with a later eval.
            pub fn release(self: *Stream, r: *Route) void {
                std.debug.assert(r.state == .live);
                r.state = .released;
                self.released[self.n_released] = @intCast((@intFromPtr(r) - @intFromPtr(&self.routes[0])) / @sizeOf(Route));
                self.n_released += 1;
            }

            /// Unpins every released route; call only after an eval that consumed
            /// them (`route` does, since its ids come from such an eval). A release
            /// whose reads are still landing waits for them first. A gate the
            /// watchdog forced since the last flush fails the stream here.
            pub fn flush(self: *Stream) Error!void {
                var first_error: ?Error = null;
                for (self.released[0..self.n_released]) |ri| {
                    const r = &self.routes[ri];
                    for (r.parts[0..r.n_parts]) |*p| self.settle(r, p) catch |e| {
                        if (first_error == null) first_error = e;
                    };
                    for (r.hit_slots[0..r.plan.n_hits]) |s| self.locate(r.layer, s).meta.pins -= 1;
                    for (r.plan.loadsOf()) |l| self.locate(r.layer, l.slot).meta.pins -= 1;
                    r.state = .free;
                    self.free[self.n_free] = ri;
                    self.n_free += 1;
                }
                self.n_released = 0;
                if (first_error) |e| return e;
                // The watchdog's forced gates (a count that moves only on a gated stream).
                const forced = self.pool.counter(.ev_wd_forced);
                if (forced != self.forced_seen) {
                    self.forced_seen = forced;
                    return self.fail(error.GateForced);
                }
            }

            /// The phase change's first free (the Module's frees stage, before its cache clear and boundary check): the
            /// whole transient scratch, with grow's preconditions; `grow` allocates decode's window 0. MLX rows: the
            /// allocator's active bytes must drop by the scratch's, else a holder survived. Returns the bytes freed.
            pub fn releaseTransient(self: *Stream) !u64 {
                self.settleCopies();
                if (!self.release_installed) return error.TransientReleaseNotInstalled;
                if (std.Thread.getCurrentId() != self.owner) return error.NotInferenceThread;
                if (self.phase != .prefill) return error.AlreadyGrown;
                if (self.failed) return error.StreamFailed;
                if (self.transient_released) return error.TransientAlreadyReleased;
                if (self.ahead.live) try self.awaitReadAhead(self.ahead.layer);
                try self.flush();
                for (&self.routes) |*r| if (r.state != .free) return error.RoutesLive;
                if (self.held_base.items.len > 0) return error.RoutesLive;
                var bytes: u64 = self.transient.allocBytes(&self.bank.layers[self.transient_layer]);
                if (self.alt) |*x| bytes += x.rows.allocBytes(&self.bank.layers[x.layer]);
                const before = if (self.memory == .mlx) mlxActive() else 0;
                self.transient.deinit();
                self.allocator.free(self.transient_meta);
                self.transient_meta = self.transient_meta[0..0];
                if (self.alt) |*x| {
                    x.rows.deinit();
                    self.allocator.free(x.meta);
                    x.meta = x.meta[0..0];
                }
                self.transient_released = true;
                self.wide_depth = 1;
                if (self.memory == .mlx and before -| mlxActive() < bytes) {
                    self.failed = true;
                    return error.TransientStillReferenced;
                }
                return bytes;
            }

            /// The one phase change: each layer's persistent rows become
            /// `decode_rows` (the added rows empty, residents unmoved); routes are
            /// decode routes from here on. Needs every route released and the scratch released first: decode's window 0
            /// (plus `decode_staging_rows`) is allocated here, before the added rows.
            pub fn grow(self: *Stream, decode_rows: []const u32) !void {
                self.settleCopies();
                if (std.Thread.getCurrentId() != self.owner) return error.NotInferenceThread;
                if (self.phase != .prefill) return error.AlreadyGrown;
                if (self.failed) return error.StreamFailed;
                if (self.release_installed and !self.transient_released) return error.TransientNotReleased;
                if (decode_rows.len != self.layers.len) return error.InvalidRows;
                if (self.ahead.live) try self.awaitReadAhead(self.ahead.layer);
                try self.flush();
                for (&self.routes) |*r| if (r.state != .free) return error.RoutesLive;
                if (self.held_base.items.len > 0) return error.RoutesLive;
                for (self.layers, decode_rows) |*ls, rows| {
                    if (rows < ls.policy.capacity or rows > ls.policy.n_experts) return error.InvalidRows;
                }
                const a = self.allocator;
                var window0: ?Rows = null;
                var meta0: []SlotMeta = &.{};
                errdefer if (window0) |*w| {
                    w.deinit();
                    a.free(meta0);
                };
                var alt0: ?Rows = null;
                var alt_meta0: []SlotMeta = &.{};
                errdefer if (alt0) |*w| {
                    w.deinit();
                    a.free(alt_meta0);
                };
                if (self.transient_released) {
                    // Decode's window 0 holds the widest decode route (`decode_route_ids`), per geometry (a bank of two
                    // stream geometries keeps a window of each).
                    const n0 = self.decode_route_ids + decode_staging_rows;
                    const geoms = [2]?u32{ self.transient_layer, if (self.alt) |x| x.layer else null };
                    for (geoms, [2]*?Rows{ &window0, &alt0 }, [2]*[]SlotMeta{ &meta0, &alt_meta0 }) |gl, w, m| {
                        const geom = &self.bank.layers[gl orelse continue];
                        w.* = switch (self.grow_fill) {
                            .zeros => try Rows.init(geom, n0, self.memory),
                            .unfilled => try Rows.initUnfilled(geom, n0, self.memory),
                        };
                        m.* = a.alloc(SlotMeta, w.*.?.rows) catch |e| {
                            w.*.?.deinit();
                            w.* = null;
                            return e;
                        };
                        @memset(m.*, .{});
                    }
                }
                const exts = try a.alloc(?Rows, self.layers.len);
                defer a.free(exts);
                @memset(exts, null);
                errdefer for (exts) |*e| if (e.*) |*rows| rows.deinit();
                for (self.layers, decode_rows, exts, self.bank.layers) |*ls, rows, *e, *geom| {
                    if (rows > ls.policy.capacity) e.* = switch (self.grow_fill) {
                        .zeros => try Rows.initLazy(geom, rows - ls.policy.capacity, self.memory),
                        .unfilled => try Rows.initUnfilled(geom, rows - ls.policy.capacity, self.memory),
                    };
                }
                // Every layer's new MLX arrays in one eval (growth-overlap step 1: not nine evals per layer, 360 at 40); unfilled
                // arrays are already evaluated.
                if (self.grow_fill == .zeros) try evalRows(a, exts);
                for (exts) |*e| if (e.*) |*r| try r.bind();
                // From here the stream owns the new rows (the errdefers above free none of them).
                if (window0) |w| {
                    self.transient = w;
                    self.transient_meta = meta0;
                    self.transient_released = false;
                    window0 = null;
                }
                if (alt0) |w| {
                    self.alt.?.rows = w;
                    self.alt.?.meta = alt_meta0;
                    alt0 = null;
                }
                for (self.layers, decode_rows, exts) |*ls, rows, *e| {
                    ls.ext = e.*;
                    e.* = null;
                    ls.policy.grow(rows) catch unreachable;
                    // The tally's pins at decode's capacity, once per request (a route never re-ranks them).
                    ls.policy.refreshPins();
                }
                self.phase = .decode;
                // Decode routes take one window: no decode plan ever runs beside a live route's held slots (the decode
                // policy has no held set; `shrink` restores the prompt's depth).
                self.wide_depth = 1;
                self.route_lookahead = self.selector != null;
                self.route_preread = self.route_lookahead and self.preread;
                if (self.direct_block != 0) self.checkRowsDirect(.grow) catch |e| return self.failDirect(e);
                if (comptime recording) if (self.recorder) |rec| for (self.layers, 0..) |*ls, l| {
                    rec.handover(@intCast(l), ls.policy.prefill_freq, ls.policy.slot_to_expert[0..ls.policy.capacity], ls.base.rows);
                };
            }

            /// The reverse phase change's free (the return to the prompt phase before a later prompt; the caller synchronized
            /// first): every route settled and unpinned (a cancelled request's included), each layer's grown rows freed and the residents in them
            /// forgotten (`LayerPolicy.shrink`), the prompt rows' residents forgotten or kept (`residents`), decode's window 0
            /// freed under the release route; the stream is in its prompt phase with the scratch absent until `regrowTransient`,
            /// which the caller runs only after these frees landed. The bytes freed do not depend on `residents`.
            /// Returns the bytes freed. MLX rows: the allocator's active bytes must drop by them, else a holder survived.
            pub fn shrink(self: *Stream, prompt_rows: []const u32, residents: Residents) !u64 {
                if (std.Thread.getCurrentId() != self.owner) return error.NotInferenceThread;
                if (self.phase != .decode) return error.NotGrown;
                if (self.failed) return error.StreamFailed;
                if (prompt_rows.len != self.layers.len) return error.InvalidRows;
                for (self.layers, prompt_rows) |*ls, rows| if (rows != ls.base.rows) return error.InvalidRows;
                try self.settleRoutes();
                var bytes: u64 = 0;
                for (self.layers, self.bank.layers) |*ls, *geom| if (ls.ext) |*e| {
                    bytes += e.allocBytes(geom);
                };
                if (self.release_installed) bytes += self.transient.allocBytes(&self.bank.layers[self.transient_layer]);
                if (self.release_installed) if (self.alt) |*x| {
                    bytes += x.rows.allocBytes(&self.bank.layers[x.layer]);
                };
                const before = if (self.memory == .mlx) mlxActive() else 0;
                for (self.layers, prompt_rows) |*ls, rows| {
                    for (ls.meta[rows..ls.policy.capacity]) |*m| {
                        if (m.pins != 0 or m.state == .loading) return self.fail(error.SlotStillPinned);
                        m.* = .{};
                    }
                    ls.policy.shrink(rows) catch unreachable;
                    // The one place that decides which residents a later prompt finds. `forget`: none, as at construction (its
                    // schedule then equals the first prompt's; slot bytes stay, a load of the same record skips its read).
                    // `keep`: those decode left in the prompt rows, whose bytes and meta nothing touched since their reads
                    // landed (every route settled above); only the prompt state goes.
                    _ = switch (residents) {
                        .forget => ls.policy.forgetAll(),
                        .keep => ls.policy.forgetPrompt(),
                    };
                    if (ls.ext) |*e| e.deinit();
                    ls.ext = null;
                }
                if (self.release_installed) {
                    self.transient.deinit();
                    self.allocator.free(self.transient_meta);
                    self.transient_meta = self.transient_meta[0..0];
                    if (self.alt) |*x| {
                        x.rows.deinit();
                        self.allocator.free(x.meta);
                        x.meta = x.meta[0..0];
                    }
                    self.transient_released = true;
                }
                self.phase = .prefill;
                self.route_lookahead = false;
                self.route_preread = false;
                if (self.memory == .mlx and before -| mlxActive() < bytes) {
                    self.failed = true;
                    return error.GrownRowsStillReferenced;
                }
                return bytes;
            }

            /// A request's end (the caller synchronized: no command still reads a slot): a cancelled request's live and held
            /// routes released, the prompt's read-ahead awaited, and everything settled and unpinned by the flush (reads still
            /// landing are waited for, never cancelled). Nothing live after it.
            pub fn settleRoutes(self: *Stream) !void {
                if (std.Thread.getCurrentId() != self.owner) return error.NotInferenceThread;
                if (self.failed) return error.StreamFailed;
                if (self.ahead.live) try self.awaitReadAhead(self.ahead.layer);
                for (&self.routes) |*r| if (r.state == .live) self.release(r);
                self.releaseHeld();
                try self.flush();
                for (&self.routes) |*r| if (r.state != .free) return error.RoutesLive;
            }

            /// Residents over every layer's persistent rows (the reverse change's record: what the next prompt finds).
            pub fn residentCount(self: *const Stream) u64 {
                var n: u64 = 0;
                for (self.layers) |*ls| n += ls.policy.occupancy;
                return n;
            }

            /// The prompt scratch's bytes (`regrowTransient` allocates them: `arrayBytes`, the bank's floor included).
            pub fn promptTransientBytes(self: *const Stream) u64 {
                var n: u64 = 0;
                for (0..n_components) |c| n += arrayBytes(&self.bank.layers[self.transient_layer], c, self.prompt_transient_rows);
                if (self.alt) |x| for (0..n_components) |c| {
                    n += arrayBytes(&self.bank.layers[x.layer], c, self.prompt_transient_rows);
                };
                return n;
            }

            /// The reverse phase change's allocation, after its frees landed: the prompt's scratch (`Options.transient_rows`
            /// rows, `wide_depth` windows) re-created, so the next prompt routes as the first did. Returns its bytes.
            pub fn regrowTransient(self: *Stream) !u64 {
                self.settleCopies();
                if (std.Thread.getCurrentId() != self.owner) return error.NotInferenceThread;
                if (self.phase != .prefill) return error.AlreadyGrown;
                if (self.failed) return error.StreamFailed;
                if (!self.transient_released) return error.TransientNotReleased;
                var t = try Rows.init(&self.bank.layers[self.transient_layer], self.prompt_transient_rows, self.memory);
                // Until the install below; then the stream owns `t` and `meta`.
                var installed = false;
                errdefer if (!installed) t.deinit();
                const meta = try self.allocator.alloc(SlotMeta, t.rows);
                errdefer if (!installed) self.allocator.free(meta);
                @memset(meta, .{});
                var bytes: u64 = 0;
                if (self.alt) |*x| {
                    var t2 = try Rows.init(&self.bank.layers[x.layer], self.prompt_transient_rows, self.memory);
                    errdefer t2.deinit();
                    const meta2 = try self.allocator.alloc(SlotMeta, t2.rows);
                    @memset(meta2, .{});
                    bytes += t2.allocBytes(&self.bank.layers[x.layer]);
                    x.rows = t2;
                    x.meta = meta2;
                }
                self.transient = t;
                self.transient_meta = meta;
                installed = true;
                self.transient_released = false;
                self.wide_depth = self.prompt_wide_depth;
                bytes += t.allocBytes(&self.bank.layers[self.transient_layer]);
                if (self.direct_block != 0) self.checkRowsDirect(.regrow_transient) catch |e| return self.failDirect(e);
                return bytes;
            }

            /// Every live slot row against the direct reads' block (`checkDirectRows`): at construction and after each phase
            /// change's allocation (`grow`, `regrowTransient`), never per read.
            fn checkRowsDirect(self: *const Stream, site: DirectSite) expert_io.DirectError!void {
                for (self.layers, 0..) |*ls, l| {
                    try self.directRowsAt(site, &ls.base, "base", l);
                    if (ls.ext) |*e| try self.directRowsAt(site, e, "ext", l);
                }
                if (!self.transient_released) try self.directRowsAt(site, &self.transient, "transient", self.transient_layer);
                if (self.alt) |*x| try self.directRowsAt(site, &x.rows, "alt", x.layer);
            }

            /// Where `checkRowsDirect` runs: the refusal line names it.
            const DirectSite = enum { construction, grow, regrow_transient };

            /// `checkDirectRows` on one bank of rows; a refusal logs one line (the refusal path only, never per read).
            fn directRowsAt(self: *const Stream, site: DirectSite, r: *const Rows, which: []const u8, layer: usize) expert_io.DirectError!void {
                checkDirectRows(r, self.direct_block) catch |e| {
                    const blk = self.direct_block;
                    for (r.base, r.row_bytes, 0..) |b, n, c| if (b % blk != 0 or n % blk != 0) {
                        log.err("expert stream: direct rows refused at {t}: rows={s} layer={d} component={d} n_rows={d} base=0x{x} row_bytes={d} block={d} backing={t} phase={t} transient_released={}\n", .{ site, which, layer, c, r.rows, b, n, blk, r.backing, self.phase, self.transient_released });
                        break;
                    };
                    return e;
                };
            }

            /// A phase change's direct-rows refusal after its rows were installed: the stream fails closed (every later call
            /// refused by `StreamFailed`).
            fn failDirect(self: *Stream, err: expert_io.DirectError) expert_io.DirectError {
                self.failed = true;
                return err;
            }

            /// A settled result's read time; a result with payload and no preadv was copied out of a speculative record.
            fn noteRead(self: *Stream, res: expert_io.Result) void {
                const ns: u64 = @intCast(@max(res.t_end_ns - res.t_start_ns, 0));
                self.read_ns += ns;
                if (res.preadv_calls == 0 and res.payload > 0) self.copy_only_ns += ns;
            }

            pub fn stats(self: *Stream) Stats {
                var s = self.counters;
                s.expert_read_seconds = @as(f64, @floatFromInt(self.read_ns)) / 1e9;
                s.adopt_copy_seconds = @as(f64, @floatFromInt(self.copy_only_ns)) / 1e9;
                s.read_wall_ns = @intCast(@max(self.pool.readGauge()[4], 0));
                const p = self.pool;
                const pairs = .{
                    .{ "claimed", .claimed },           .{ "spec_bytes", .spec_bytes },   .{ "spec_issued", .submitted },
                    .{ "spec_landed", .landed },        .{ "adopt_ranges", .adopt_ranges }, .{ "adopt_bytes", .adopt_bytes },
                    .{ "pre_issued", .pre_issued },     .{ "pre_served", .pre_served },   .{ "pre_expired", .pre_expired },
                    .{ "gates", .ev_gates },            .{ "gates_forced", .ev_wd_forced },
                    .{ "spec_expired", .expired },      .{ "spec_abandoned", .abandoned }, .{ "spec_discarded", .discarded },
                    .{ "spec_cancelled", .cancelled_by_demand }, .{ "direct_ranges", .direct_ranges },
                    .{ "spec_claimed_inflight", .claimed_inflight }, .{ "spec_abandoned_bytes", .abandoned_bytes },
                };
                inline for (pairs) |pr| @field(s, pr[0]) = @intCast(@max(p.counter(pr[1]), 0));
                return s;
            }

            /// Live pins on a layer's slot (tests).
            pub fn pinsOf(self: *Stream, layer: u32, slot: u32) u16 {
                return self.locate(layer, slot).meta.pins;
            }
        };


    };
}
