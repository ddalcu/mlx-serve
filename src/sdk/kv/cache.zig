//! The engine's per-slot KV cache (`KVCache`): dense or affine-quantized K/V per layer, grown in place, with
//! snapshots for rollback. In the SDK so a plugin that drives a trunk holds one without the engine.

const std = @import("std");
const env = @import("../../env.zig");
const mlx = @import("../../mlx/ffi.zig");
const kv_quant = @import("../blocks/attention/kv_quant.zig");
const KVQuantConfig = kv_quant.KVQuantConfig;
const log = @import("../util/log.zig");
const materializedOwnedCopy = @import("../../mlx/array.zig").materializedOwnedCopy;
const DenseKVView = @import("../blocks/attention/kv_view.zig").DenseKVView;

pub fn arrayBytes(arr: mlx.mlx_array) u64 {
    if (arr.ctx == null) return 0;
    return @as(u64, mlx.mlx_array_size(arr)) * @as(u64, mlx.mlx_array_itemsize(arr));
}

// ── KV Cache (standard attention) ──

pub const KVCacheEntry = struct {
    // Storage. In `off` (dense bf16) mode `keys`/`values` are the full
    // [B,H,T,D] buffers and the `*_scales`/`*_biases` fields stay null. In
    // `affine` mode `keys`/`values` hold packed uint32 codes
    // ([B,H,T, D*bits/32]) and the matching scales/biases hold
    // [B,H,T, D/group_size] bf16. Switched on `KVCache.config.scheme`.
    keys: mlx.mlx_array,
    values: mlx.mlx_array,
    keys_scales: mlx.mlx_array,
    keys_biases: mlx.mlx_array,
    values_scales: mlx.mlx_array,
    values_biases: mlx.mlx_array,

    // Views: same layout as the storage fields above but trimmed to
    // [..., offset, ...] (or last `sw` entries during sliding-window decode).
    // SDPA reads dense arrays via `KVCache.denseView`; in `off` mode the
    // dense pair aliases `key_view`/`value_view`, in `affine` mode the
    // dense pair is freshly dequantized from these triples on read.
    key_view: mlx.mlx_array,
    value_view: mlx.mlx_array,
    key_scales_view: mlx.mlx_array,
    key_biases_view: mlx.mlx_array,
    value_scales_view: mlx.mlx_array,
    value_biases_view: mlx.mlx_array,

    offset: usize, // logical token count (may be < buffer capacity)
    initialized: bool,
    /// Absolute position of buffer row 0: a sliding ring (`KVCache.updateSliding`) drops the
    /// rows behind its window, so buffer row i holds token `base + i`.
    base: usize = 0,
    /// Rows a sliding ring keeps behind its write head, and the window every restore must still
    /// cover. 0 = a full buffer that never drops a row.
    keep: usize = 0,
    window: usize = 0,
    shared_view: bool = false,
    /// Rows the restored snapshot had written; only read while `shared_view`.
    shared_rows: usize = 0,
};

pub fn newEmptyKVEntry() KVCacheEntry {
    return .{
        .keys = mlx.mlx_array_new(),
        .values = mlx.mlx_array_new(),
        .keys_scales = mlx.mlx_array_new(),
        .keys_biases = mlx.mlx_array_new(),
        .values_scales = mlx.mlx_array_new(),
        .values_biases = mlx.mlx_array_new(),
        .key_view = mlx.mlx_array_new(),
        .value_view = mlx.mlx_array_new(),
        .key_scales_view = mlx.mlx_array_new(),
        .key_biases_view = mlx.mlx_array_new(),
        .value_scales_view = mlx.mlx_array_new(),
        .value_biases_view = mlx.mlx_array_new(),
        .offset = 0,
        .initialized = false,
    };
}

/// Reset a cache entry to the empty state, freeing all storage + view
/// handles. Mirror of the per-entry reset in `KVCache.restore`; used by the
/// disk tier (kv_disk_cache.zig) when installing restored buffers.
pub fn resetKVEntry(e: *KVCacheEntry) void {
    freeKVEntry(e);
    e.* = newEmptyKVEntry();
}

pub fn freeKVEntry(e: *KVCacheEntry) void {
    _ = mlx.mlx_array_free(e.keys);
    _ = mlx.mlx_array_free(e.values);
    _ = mlx.mlx_array_free(e.keys_scales);
    _ = mlx.mlx_array_free(e.keys_biases);
    _ = mlx.mlx_array_free(e.values_scales);
    _ = mlx.mlx_array_free(e.values_biases);
    _ = mlx.mlx_array_free(e.key_view);
    _ = mlx.mlx_array_free(e.value_view);
    _ = mlx.mlx_array_free(e.key_scales_view);
    _ = mlx.mlx_array_free(e.key_biases_view);
    _ = mlx.mlx_array_free(e.value_scales_view);
    _ = mlx.mlx_array_free(e.value_biases_view);
}

pub const KVCache = struct {
    entries: []KVCacheEntry,
    step: usize, // absolute sequence position (not affected by sliding window trimming)
    allocator: std.mem.Allocator,
    config: KVQuantConfig,
    /// Capacity, in tokens, this cache was asked to hold up front (the admission bill's
    /// `reservedCacheTokens`). A grow is not in place, so a long prefill's peak carried a
    /// second copy of ~6 GB of KV at 458k; reserving on the first grow removes the transient.
    reserve_tokens: usize = 0,

    pub fn init(allocator: std.mem.Allocator, num_layers: u32) !KVCache {
        return initWithConfig(allocator, num_layers, KVQuantConfig.dense);
    }

    pub fn initWithConfig(allocator: std.mem.Allocator, num_layers: u32, config: KVQuantConfig) !KVCache {
        const entries = try allocator.alloc(KVCacheEntry, num_layers);
        errdefer allocator.free(entries);
        for (entries) |*e| {
            e.* = newEmptyKVEntry();
        }
        return .{ .entries = entries, .step = 0, .allocator = allocator, .config = config };
    }

    pub fn deinit(self: *KVCache) void {
        for (self.entries) |*e| {
            freeKVEntry(e);
        }
        self.allocator.free(self.entries);
    }

    /// Swap this cache for a freshly-built one under `config`. BUILD FIRST,
    /// then free: `initWithConfig` is fallible, and a caller that reached it
    /// through `cache.deinit(); cache = try init(...)` left a FREED cache
    /// installed on the Transformer on the error path, whose own `deinit`
    /// then freed every handle a second time. The class is "a fallible
    /// re-init behind a deinit"; keeping the order inside ONE helper is what
    /// stops it recurring, so the call sites are scan-pinned to use it.
    pub fn reinit(self: *KVCache, num_layers: u32, config: KVQuantConfig) !void {
        const fresh = try initWithConfig(self.allocator, num_layers, config);
        self.deinit();
        self.* = fresh;
    }

    /// Capture cache state for speculative-decoding rollback (PLD/drafter).
    /// GPU bytes of the buffers this cache owns. A restored share is billed to
    /// the hot-cache entry that owns its buffer.
    pub fn residentBytes(self: *const KVCache) u64 {
        var total: u64 = 0;
        for (self.entries) |e| {
            if (!e.initialized or e.shared_view) continue;
            inline for (.{ e.keys, e.values, e.keys_scales, e.keys_biases, e.values_scales, e.values_biases }) |arr| {
                total += arrayBytes(arr);
            }
        }
        return total;
    }

    /// Snapshots own array handles that share the underlying buffer with the
    /// source via refcount — cheap (no data copy) and immune to subsequent
    /// `update()` calls (which create new buffer handles when growing).
    /// `*_view` fields are excluded because `update()` recreates them every
    /// call.
    pub fn snapshot(self: *const KVCache) !KVCacheSnapshot {
        const out = try self.allocator.alloc(KVCacheEntry, self.entries.len);
        var built: usize = 0;
        errdefer {
            for (out[0..built]) |*e| freeKVEntry(e);
            self.allocator.free(out);
        }
        for (self.entries, 0..) |src, i| {
            out[i] = newEmptyKVEntry();
            built = i + 1;
            out[i].offset = src.offset;
            out[i].initialized = src.initialized;
            out[i].base = src.base;
            out[i].keep = src.keep;
            out[i].window = src.window;
            if (src.initialized) {
                try mlx.check(mlx.mlx_array_set(&out[i].keys, src.keys));
                try mlx.check(mlx.mlx_array_set(&out[i].values, src.values));
                if (self.config.scheme != .off) {
                    try mlx.check(mlx.mlx_array_set(&out[i].keys_scales, src.keys_scales));
                    try mlx.check(mlx.mlx_array_set(&out[i].keys_biases, src.keys_biases));
                    try mlx.check(mlx.mlx_array_set(&out[i].values_scales, src.values_scales));
                    try mlx.check(mlx.mlx_array_set(&out[i].values_biases, src.values_biases));
                }
            }
        }
        return .{ .entries = out, .step = self.step, .allocator = self.allocator, .config = self.config };
    }

    /// Replace cache state with `snap`. Frees current entries' arrays first;
    /// re-binds via refcount-share from snapshot. After restore, the next
    /// `update()` will recreate `*_view` fields from the restored buffers.
    pub fn restore(self: *KVCache, snap: *const KVCacheSnapshot) !void {
        std.debug.assert(self.entries.len == snap.entries.len);
        for (self.entries, snap.entries) |*dst, src| {
            freeKVEntry(dst);
            dst.* = newEmptyKVEntry();
            dst.offset = src.offset;
            dst.initialized = src.initialized;
            dst.base = src.base;
            dst.keep = src.keep;
            dst.window = src.window;
            if (src.initialized) {
                try mlx.check(mlx.mlx_array_set(&dst.keys, src.keys));
                try mlx.check(mlx.mlx_array_set(&dst.values, src.values));
                if (self.config.scheme != .off) {
                    try mlx.check(mlx.mlx_array_set(&dst.keys_scales, src.keys_scales));
                    try mlx.check(mlx.mlx_array_set(&dst.keys_biases, src.keys_biases));
                    try mlx.check(mlx.mlx_array_set(&dst.values_scales, src.values_scales));
                    try mlx.check(mlx.mlx_array_set(&dst.values_biases, src.values_biases));
                }
                dst.shared_view = true;
                dst.shared_rows = src.offset - src.base;
            }
        }
        self.step = snap.step;
    }

    const chunk_step = 256;

    /// Cap on the slack a single growth event adds, in tokens. Keeps the
    /// proportional term from over-allocating at very long context (25% of
    /// 200 K would be 50 K tokens of headroom).
    const max_grow_tokens = 8192;

    /// PURE: the capacity the K/V buffers must be grown to so `needed` tokens
    /// fit.
    ///
    /// Growing by a fixed `chunk_step` would mean one grow-copy-free every 256 tokens, and
    /// every superseded buffer is orphaned: MLX only reuses a cached buffer
    /// when `cached_size < requested + 2*page_size` (`buffer_cache.h`), and the
    /// next KV buffer is a full chunk past that ~32 KB tolerance. On a 64
    /// KiB/token model at 89 K context that strands ~5.6 GB per 256 generated
    /// tokens in MLX's allocator pool — invisible to `active_bytes`, which is
    /// how the reporter's process reached 81 GB while the panel read 19.6.
    ///
    /// Proportional growth (+25%, floored at one chunk, capped at
    /// `max_grow_tokens`) turns that into ~27 growth events over a 100 K-token
    /// walk instead of 391.
    pub fn nextCapacity(current_cap: usize, needed: usize) usize {
        if (needed <= current_cap) return current_cap;
        const grow = @min(@max(current_cap / 4, chunk_step), max_grow_tokens);
        const target = @max(needed, current_cap + grow);
        const n_chunks = (target + chunk_step - 1) / chunk_step;
        return n_chunks * chunk_step;
    }

    /// Tokens of capacity a request reserves: prompt + allowed generation + one chunk of slack.
    /// The one definition; `server.reservedCacheTokens` delegates here.
    pub fn reservedTokens(seq: u64, max_tokens: u64, chunk: u64, ctx: u64) u64 {
        // Short prompts keep the proportional policy: reserving costs them `max_tokens` of KV they may never use.
        if (seq < RESERVE_MIN_TOKENS) return 0;
        // An omitted `max_tokens` is maxInt(u32)/4 (~26 TB of KV), so bound by the context AND
        // by the generation headroom: reserving `ctx - prompt` billed 13 GB at 383k on a
        // 1M-context qwen4_exp. `ctx == 0` = unknown, no clamp.
        const want: u64 = @min(max_tokens, RESERVE_GEN_HEADROOM);
        const head_room: u64 = if (ctx > seq) @min(want, ctx - seq) else if (ctx > 0) 0 else want;
        return seq +| head_room +| @max(chunk, 1);
    }

    /// Tokens of generation the reservation pre-buys. Past it a decode grow takes the ordinary
    /// +25% policy, which happens between decode steps, never inside a chunk's lazy graph.
    /// 8192 covers essentially every real request at ~100 MB of KV on qwen4_exp at 8-bit.
    pub const RESERVE_GEN_HEADROOM: u64 = 8192;

    /// Prompt length at or above which a request reserves its capacity; below it the flat
    /// runtime floor already covers the transient.
    pub const RESERVE_MIN_TOKENS: u64 = 32768;

    /// How many times a KV capacity buffer actually grew (the moments a second copy of the whole cache exists).
    pub var kv_cap_buf_grows: usize = 0;

    /// Ask this cache to hold `tokens` up front. Idempotent and monotone.
    pub fn reserve(self: *KVCache, tokens: usize) void {
        if (tokens > self.reserve_tokens) self.reserve_tokens = tokens;
    }

    /// The capacity a grow takes: the proportional policy, raised to the reservation when there is one.
    pub fn nextCapacityReserved(self: *const KVCache, current_cap: usize, needed: usize) usize {
        const policy = nextCapacity(current_cap, needed);
        if (self.reserve_tokens <= policy) return policy;
        // Round the reservation the same way the policy rounds.
        const n_chunks = (self.reserve_tokens + chunk_step - 1) / chunk_step;
        return n_chunks * chunk_step;
    }

    pub fn update(self: *KVCache, layer: u32, new_k: mlx.mlx_array, new_v: mlx.mlx_array, s: mlx.mlx_stream, max_seq: u32) !DenseKVView {
        switch (self.config.scheme) {
            .off => return self.updateDense(layer, new_k, new_v, s, max_seq),
            .affine => return self.updateAffine(layer, new_k, new_v, s, max_seq),
        }
    }

    pub fn appendBatched(
        stacked_k: mlx.mlx_array,
        stacked_v: mlx.mlx_array,
        caches: []const *KVCache,
        layer: u32,
        s: mlx.mlx_stream,
        max_seq: u32,
        scheme: KVQuantConfig,
    ) !void {
        if (scheme.scheme == .affine and scheme.bits == 8) {
            const Once = struct {
                var logged: bool = false;
            };
            if (!Once.logged) {
                Once.logged = true;
                log.info("[batched] affine-8 kv append engaged (slots={d}) (batched_kv_append_override=false restores per-slot appends)\n", .{caches.len});
            }
            try appendAffineBatched(stacked_k, stacked_v, caches, layer, s, max_seq, scheme);
            return;
        }
        try appendBatchedPerSlot(stacked_k, stacked_v, caches, layer, s, max_seq);
    }

    pub fn appendFromStacked(
        stacked_k: mlx.mlx_array,
        stacked_v: mlx.mlx_array,
        caches: []const *KVCache,
        layer: u32,
        s: mlx.mlx_stream,
        max_seq: u32,
        use_batched: bool,
    ) !void {
        if (use_batched and caches.len > 0) {
            try appendBatched(stacked_k, stacked_v, caches, layer, s, max_seq, caches[0].config);
            return;
        }
        try appendBatchedPerSlot(stacked_k, stacked_v, caches, layer, s, max_seq);
    }

    fn appendBatchedPerSlot(
        stacked_k: mlx.mlx_array,
        stacked_v: mlx.mlx_array,
        caches: []const *KVCache,
        layer: u32,
        s: mlx.mlx_stream,
        max_seq: u32,
    ) !void {
        for (caches, 0..) |cache, i| {
            const k_slot = try sliceStackedRow(s, stacked_k, i);
            defer _ = mlx.mlx_array_free(k_slot);
            const v_slot = try sliceStackedRow(s, stacked_v, i);
            defer _ = mlx.mlx_array_free(v_slot);
            var view = try cache.update(layer, k_slot, v_slot, s, max_seq);
            view.deinit();
        }
    }

    pub fn sliceStackedRow(s: mlx.mlx_stream, stacked: mlx.mlx_array, i: usize) !mlx.mlx_array {
        const sh = mlx.getShape(stacked);
        const i_c: c_int = @intCast(i);
        const start = [_]c_int{ i_c, 0, 0, 0 };
        const stop = [_]c_int{ i_c + 1, sh[1], sh[2], sh[3] };
        const strides = [_]c_int{ 1, 1, 1, 1 };
        var out = mlx.mlx_array_new();
        errdefer _ = mlx.mlx_array_free(out);
        try mlx.check(mlx.mlx_slice(&out, stacked, &start, 4, &stop, 4, &strides, 4, s));
        return out;
    }

    fn appendAffineBatched(
        stacked_k: mlx.mlx_array,
        stacked_v: mlx.mlx_array,
        caches: []const *KVCache,
        layer: u32,
        s: mlx.mlx_stream,
        max_seq: u32,
        scheme: KVQuantConfig,
    ) !void {
        var new_kq = try kv_quant.quantizeAffine(s, stacked_k, scheme.group_size, scheme.bits);
        defer new_kq.deinit();
        var new_vq = try kv_quant.quantizeAffine(s, stacked_v, scheme.group_size, scheme.bits);
        defer new_vq.deinit();
        for (caches, 0..) |cache, i| {
            var row_kq: kv_quant.QuantizedKV = .{
                .q = mlx.mlx_array_new(),
                .scales = mlx.mlx_array_new(),
                .biases = mlx.mlx_array_new(),
            };
            defer row_kq.deinit();
            _ = mlx.mlx_array_free(row_kq.q);
            row_kq.q = try sliceStackedRow(s, new_kq.q, i);
            _ = mlx.mlx_array_free(row_kq.scales);
            row_kq.scales = try sliceStackedRow(s, new_kq.scales, i);
            _ = mlx.mlx_array_free(row_kq.biases);
            row_kq.biases = try sliceStackedRow(s, new_kq.biases, i);
            var row_vq: kv_quant.QuantizedKV = .{
                .q = mlx.mlx_array_new(),
                .scales = mlx.mlx_array_new(),
                .biases = mlx.mlx_array_new(),
            };
            defer row_vq.deinit();
            _ = mlx.mlx_array_free(row_vq.q);
            row_vq.q = try sliceStackedRow(s, new_vq.q, i);
            _ = mlx.mlx_array_free(row_vq.scales);
            row_vq.scales = try sliceStackedRow(s, new_vq.scales, i);
            _ = mlx.mlx_array_free(row_vq.biases);
            row_vq.biases = try sliceStackedRow(s, new_vq.biases, i);
            try cache.writeAffineChunk(layer, row_kq, row_vq, s, max_seq);
        }
    }

    fn updateAffine(self: *KVCache, layer: u32, new_k: mlx.mlx_array, new_v: mlx.mlx_array, s: mlx.mlx_stream, max_seq: u32) !DenseKVView {
        const entry = &self.entries[layer];
        const cfg = self.config;
        const group_size: u32 = cfg.group_size;
        const bits: u8 = cfg.bits;

        // 1. Free stale views (6 of them — dense + 4 quant scale/bias views).
        // 2. Quantize incoming K/V.
        var new_kq = try kv_quant.quantizeAffine(s, new_k, group_size, bits);
        defer new_kq.deinit();
        var new_vq = try kv_quant.quantizeAffine(s, new_v, group_size, bits);
        defer new_vq.deinit();

        // 3. Shape info: new_k is [B, heads, new_len, head_dim]. The V widths
        //    are solved from new_v's OWN head dim, not K's: an MLA arch scores
        //    over nope+rope (192) and stores values at v_head_dim (128), so a
        //    K-derived packed width writes a 16-u32 chunk into a 24-u32 window
        //    (an mlx slice_update mismatch we cannot catch). Identical on every
        //    symmetric arch — the same generalization `updateDense` carries.
        try self.writeAffineChunk(layer, new_kq, new_vq, s, max_seq);

        // 8. Dequantize K/V for SDPA. Owner of these dense arrays is the
        //    DenseKVView returned to the caller. The quant triples ride along
        //    as borrows so fused call sites can skip the dense read — mlx is
        //    lazy, so the dequant graph below costs nothing unless SDPA
        //    actually reads `.k`/`.v`. (Engagement bug class: the decode
        //    forward calls update(), not denseView() — omitting the triples
        //    here made `--kv-attn-mode fused` a silent no-op.)
        const dense_k = try kv_quant.dequantizeAffine(s, entry.key_view, entry.key_scales_view, entry.key_biases_view, group_size, bits);
        errdefer _ = mlx.mlx_array_free(dense_k);
        const dense_v = try kv_quant.dequantizeAffine(s, entry.value_view, entry.value_scales_view, entry.value_biases_view, group_size, bits);
        return .{
            .k = dense_k,
            .v = dense_v,
            .owned = true,
            .k_triple_q = entry.key_view,
            .k_triple_scales = entry.key_scales_view,
            .k_triple_biases = entry.key_biases_view,
            .v_triple_q = entry.value_view,
            .v_triple_scales = entry.value_scales_view,
            .v_triple_biases = entry.value_biases_view,
            .has_quant_triple = true,
            .bits = bits,
            .group_size = group_size,
            .kv_cap = @intCast(bufferCapacity(entry.keys)),
        };
    }

    fn writeAffineChunk(
        self: *KVCache,
        layer: u32,
        new_kq: kv_quant.QuantizedKV,
        new_vq: kv_quant.QuantizedKV,
        s: mlx.mlx_stream,
        max_seq: u32,
    ) !void {
        const entry = &self.entries[layer];
        _ = mlx.mlx_array_free(entry.key_view);
        _ = mlx.mlx_array_free(entry.value_view);
        _ = mlx.mlx_array_free(entry.key_scales_view);
        _ = mlx.mlx_array_free(entry.key_biases_view);
        _ = mlx.mlx_array_free(entry.value_scales_view);
        _ = mlx.mlx_array_free(entry.value_biases_view);
        entry.key_view = mlx.mlx_array_new();
        entry.value_view = mlx.mlx_array_new();
        entry.key_scales_view = mlx.mlx_array_new();
        entry.key_biases_view = mlx.mlx_array_new();
        entry.value_scales_view = mlx.mlx_array_new();
        entry.value_biases_view = mlx.mlx_array_new();

        const q_shape = mlx.getShape(new_kq.q);
        const vq_shape = mlx.getShape(new_vq.q);
        const sc_shape = mlx.getShape(new_kq.scales);
        const vsc_shape = mlx.getShape(new_vq.scales);
        const new_len: usize = @intCast(q_shape[2]);
        const B = q_shape[0];
        const heads = q_shape[1];
        const q_last = q_shape[3];
        const vq_last = vq_shape[3];
        const sc_last = sc_shape[3];
        const vsc_last = vsc_shape[3];

        const grow = self.planGrow(entry, new_len);
        if (entry.shared_view) {
            if (entry.initialized and grow == null) try cowAffineBuffers(s, entry);
            entry.shared_view = false;
        }

        // 4. Grow buffers if needed (6 of them, in lockstep on the seq axis).
        if (grow) |g| {
            kv_cap_buf_grows += 1;
            try growQuantBuf(s, &entry.keys, g, B, heads, q_last, .uint32);
            try growQuantBuf(s, &entry.values, g, B, heads, vq_last, .uint32);
            // Scales keep the dtype the quantizer emitted (the activation dtype): a
            // bf16 buffer under an f16 model widens every later layer to f32.
            const ksc_dt = mlx.mlx_array_dtype(new_kq.scales);
            const vsc_dt = mlx.mlx_array_dtype(new_vq.scales);
            try growQuantBuf(s, &entry.keys_scales, g, B, heads, sc_last, ksc_dt);
            try growQuantBuf(s, &entry.keys_biases, g, B, heads, sc_last, ksc_dt);
            try growQuantBuf(s, &entry.values_scales, g, B, heads, vsc_last, vsc_dt);
            try growQuantBuf(s, &entry.values_biases, g, B, heads, vsc_last, vsc_dt);
            entry.base += g.drop;
            entry.initialized = true;
        }

        // 5. slice_update each buffer at offset.
        const at = entry.offset - entry.base;
        try writeAtOffset(s, &entry.keys, at, new_kq.q);
        try writeAtOffset(s, &entry.values, at, new_vq.q);
        try writeAtOffset(s, &entry.keys_scales, at, new_kq.scales);
        try writeAtOffset(s, &entry.keys_biases, at, new_kq.biases);
        try writeAtOffset(s, &entry.values_scales, at, new_vq.scales);
        try writeAtOffset(s, &entry.values_biases, at, new_vq.biases);

        // 6. Update offset / step.
        entry.offset += new_len;
        if (layer == 0) self.step += new_len;

        // 7. Build views for all 6 buffers.
        const total: c_int = @intCast(entry.offset - entry.base);
        // `max_seq` is the TAIL SPAN this forward needs, not the raw window:
        // the caller sizes it with `slidingTailSpan` so a block-wide verify
        // gets `window + q_len - 1` and decode still gets `window`. 0 = keep
        // the whole cache. Honored at EVERY width — the old `new_len == 1`
        // gate is what silently made every spec verify read the full cache on
        // sliding layers.
        const view_start: c_int = if (max_seq > 0 and total > max_seq)
            total - @as(c_int, @intCast(max_seq))
        else
            0;
        try buildSliceView(s, &entry.key_view, entry.keys, total, view_start);
        try buildSliceView(s, &entry.value_view, entry.values, total, view_start);
        try buildSliceView(s, &entry.key_scales_view, entry.keys_scales, total, view_start);
        try buildSliceView(s, &entry.key_biases_view, entry.keys_biases, total, view_start);
        try buildSliceView(s, &entry.value_scales_view, entry.values_scales, total, view_start);
        try buildSliceView(s, &entry.value_biases_view, entry.values_biases, total, view_start);
    }

    /// `update` for a layer that attends only its last `window` keys: the buffer keeps the newest
    /// `keep` rows (>= window) and drops the rest on its next grow, so it stays near keep + one
    /// chunk however long the sequence runs. Positions stay absolute (`offset`, `step`).
    pub fn updateSliding(self: *KVCache, layer: u32, new_k: mlx.mlx_array, new_v: mlx.mlx_array, s: mlx.mlx_stream, max_seq: u32, window: u32, keep: u32) !DenseKVView {
        std.debug.assert(keep >= window and max_seq <= keep + @as(u32, @intCast(mlx.getShape(new_k)[2])));
        const e = &self.entries[layer];
        e.keep = keep;
        e.window = window;
        return self.update(layer, new_k, new_v, s, max_seq);
    }

    /// A grow: the capacity to allocate, and which of the current rows move into it.
    const Grow = struct { cap: c_int, drop: usize, kept: usize };

    /// The grow a write of `new_len` rows needs, or null. A sliding ring carries only its newest
    /// `keep` rows across and ignores the request's reservation, which sizes a full buffer.
    fn planGrow(self: *const KVCache, entry: *const KVCacheEntry, new_len: usize) ?Grow {
        const rows = entry.offset - entry.base;
        const oversized = self.restoredOversized(entry, new_len);
        if (entry.initialized and !oversized and rows + new_len <= bufferCapacity(entry.keys)) return null;
        const drop: usize = if (entry.initialized and entry.keep > 0 and rows > entry.keep) rows - entry.keep else 0;
        const kept: usize = if (entry.initialized) rows - drop else 0;
        const cur_cap = if (oversized) rows else if (entry.initialized) bufferCapacity(entry.keys) else 0;
        const cap = if (entry.keep > 0) nextCapacity(cur_cap, kept + new_len) else self.nextCapacityReserved(cur_cap, kept + new_len);
        return .{ .cap = @intCast(cap), .drop = drop, .kept = kept };
    }

    /// A restore that kept only a short prefix of a longer entry regrows from that prefix instead
    /// of copying the whole buffer: a "hi" session restored from an 80k one kept the 80k capacity.
    fn restoredOversized(self: *const KVCache, entry: *const KVCacheEntry, new_len: usize) bool {
        if (!entry.initialized or !entry.shared_view) return false;
        const rows = entry.offset - entry.base;
        return entry.shared_rows > self.nextCapacityReserved(rows, rows + new_len);
    }

    fn cowAffineBuffers(s: mlx.mlx_stream, entry: *KVCacheEntry) !void {
        const bufs = [_]*mlx.mlx_array{
            &entry.keys,
            &entry.values,
            &entry.keys_scales,
            &entry.keys_biases,
            &entry.values_scales,
            &entry.values_biases,
        };
        for (bufs) |buf| {
            const owned = try materializedOwnedCopy(s, buf.*);
            _ = mlx.mlx_array_free(buf.*);
            buf.* = owned;
        }
    }

    fn updateDense(self: *KVCache, layer: u32, new_k: mlx.mlx_array, new_v: mlx.mlx_array, s: mlx.mlx_stream, max_seq: u32) !DenseKVView {
        const entry = &self.entries[layer];

        // 1. Free stale views — drops refcount on buffer → enables buffer donation.
        //    Reset the handles at once: a failing op below must not leave freed
        //    handles for `freeKVEntry` to free again.
        _ = mlx.mlx_array_free(entry.key_view);
        _ = mlx.mlx_array_free(entry.value_view);
        entry.key_view = mlx.mlx_array_new();
        entry.value_view = mlx.mlx_array_new();

        // 2. Get shape info from new_k: [B, heads, new_len, head_dim]. The V
        //    head dim is read off new_v, NOT new_k: an MLA arch scores over
        //    nope+rope (192) but stores values at v_head_dim (128), and MLX's
        //    SDPA serves exactly that pair with a fused vector kernel. Every
        //    buffer below is sized from its own operand; the two are identical
        //    on every symmetric arch, so this is a strict generalization.
        const new_shape = mlx.getShape(new_k);
        const new_len: usize = @intCast(new_shape[2]);
        const v_head_dim = mlx.getShape(new_v)[3];

        const grow = self.planGrow(entry, new_len);
        if (entry.shared_view) {
            if (entry.initialized and grow == null) {
                const k_owned = try materializedOwnedCopy(s, entry.keys);
                _ = mlx.mlx_array_free(entry.keys);
                entry.keys = k_owned;
                const v_owned = try materializedOwnedCopy(s, entry.values);
                _ = mlx.mlx_array_free(entry.values);
                entry.values = v_owned;
            }
            entry.shared_view = false;
        }

        // 3. Grow buffer if needed
        if (grow) |g| {
            const dtype = mlx.mlx_array_dtype(new_k);
            kv_cap_buf_grows += 1;
            try growQuantBuf(s, &entry.keys, g, new_shape[0], new_shape[1], new_shape[3], dtype);
            try growQuantBuf(s, &entry.values, g, new_shape[0], new_shape[1], v_head_dim, dtype);
            entry.base += g.drop;
            entry.initialized = true;
        }

        // 4. slice_update — write new_k/new_v into buffer at offset
        try writeAtOffset(s, &entry.keys, entry.offset - entry.base, new_k);
        try writeAtOffset(s, &entry.values, entry.offset - entry.base, new_v);

        // 5. Update offset and absolute step
        entry.offset += new_len;
        if (layer == 0) self.step += new_len;

        // 6. Create views for attention.
        //    Skip slicing when the view covers the entire buffer — just reference it directly.
        //    This saves 2 C API calls per layer per token (84 calls/token for 42-layer models).
        const total: c_int = @intCast(entry.offset - entry.base);
        // `max_seq` is the TAIL SPAN this forward needs, not the raw window:
        // the caller sizes it with `slidingTailSpan` so a block-wide verify
        // gets `window + q_len - 1` and decode still gets `window`. 0 = keep
        // the whole cache. Honored at EVERY width — the old `new_len == 1`
        // gate is what silently made every spec verify read the full cache on
        // sliding layers.
        const view_start: c_int = if (max_seq > 0 and total > max_seq)
            total - @as(c_int, @intCast(max_seq))
        else
            0;

        // buildSliceView takes the whole-buffer shortcut itself (mlx-lm's
        // optimization: no slice when the view covers the buffer) and reads
        // each buffer's own last dim, so K and V may differ in width.
        try buildSliceView(s, &entry.key_view, entry.keys, total, view_start);
        try buildSliceView(s, &entry.value_view, entry.values, total, view_start);

        return .{ .k = entry.key_view, .v = entry.value_view, .owned = false };
    }

    /// Read-side accessor: return a dense `[B,H,T,D]` K/V pair for the layer.
    /// In dense mode this aliases `key_view`/`value_view` (no-op deinit).
    /// In quant mode this dequantizes on the fly from the cache's stored
    /// triples (the returned arrays are owned and freed by `deinit`).
    /// SDPA call sites use this so they don't have to know the scheme.
    pub fn denseView(self: *KVCache, layer: u32, s: mlx.mlx_stream) !DenseKVView {
        const entry = &self.entries[layer];
        if (entry.initialized and entry.key_view.ctx == null) try self.rebuildViews(entry, s);
        switch (self.config.scheme) {
            .off => return .{ .k = entry.key_view, .v = entry.value_view, .owned = false },
            .affine => {
                if (!entry.initialized) {
                    return .{ .k = entry.key_view, .v = entry.value_view, .owned = false };
                }
                const dense_k = try kv_quant.dequantizeAffine(s, entry.key_view, entry.key_scales_view, entry.key_biases_view, self.config.group_size, self.config.bits);
                errdefer _ = mlx.mlx_array_free(dense_k);
                const dense_v = try kv_quant.dequantizeAffine(s, entry.value_view, entry.value_scales_view, entry.value_biases_view, self.config.group_size, self.config.bits);
                return .{
                    .k = dense_k,
                    .v = dense_v,
                    .owned = true,
                    // Borrow the cache's quant triples so fused-attn call
                    // sites can skip the dense materialization above (the
                    // dequant arrays still get computed — mlx is lazy, so
                    // the cost is only paid if SDPA actually reads them).
                    .k_triple_q = entry.key_view,
                    .k_triple_scales = entry.key_scales_view,
                    .k_triple_biases = entry.key_biases_view,
                    .v_triple_q = entry.value_view,
                    .v_triple_scales = entry.value_scales_view,
                    .v_triple_biases = entry.value_biases_view,
                    .has_quant_triple = true,
                    .bits = self.config.bits,
                    .group_size = self.config.group_size,
                    .kv_cap = @intCast(bufferCapacity(entry.keys)),
                };
            },
        }
    }

    pub fn bufferCapacity(arr: mlx.mlx_array) usize {
        const shape = mlx.getShape(arr);
        if (shape.len < 3) return 0;
        return @intCast(shape[2]);
    }

    /// Pure half of `residentCapacityTokens`: the minimum capacity across the entries that
    /// hold one, 0 when none does. An uninitialized entry is skipped, never a veto: on a
    /// hybrid trunk only the attention layers ever call `update` (12 of 48 on qwen4_exp).
    pub const CapacityFold = struct {
        cap: usize = std.math.maxInt(usize),
        seen: usize = 0,

        /// `null` = this layer holds no buffer (a linear layer, or a cold cache).
        pub fn add(self: *CapacityFold, entry_cap: ?usize) void {
            const v = entry_cap orelse return;
            self.seen += 1;
            self.cap = @min(self.cap, v);
        }

        pub fn result(self: CapacityFold) usize {
            return if (self.seen == 0) 0 else self.cap;
        }
    };

    /// Tokens of capacity every caching layer's buffer already holds: the warm bill's one
    /// gate. Restored rows are free only while the buffer is reused in place; past this
    /// capacity a grow allocates a whole new buffer beside the old one. The MINIMUM across
    /// layers, since the credit must be provable for every buffer the prefill writes.
    pub fn residentCapacityTokens(self: *const KVCache) usize {
        var fold = CapacityFold{};
        // A sliding ring is billed as a fixed per-slot term, never by its capacity.
        for (self.entries) |*e| fold.add(if (e.initialized and e.keep == 0) bufferCapacity(e.keys) else null);
        return fold.result();
    }

    /// Buffer-grow / slice-update / view-build, parameterized over the
    /// buffer's last dim and dtype: six calls per `updateAffine` (3 buffers ×
    /// K and V) and two per `updateDense`. Every caller passes the width of
    /// the operand it is writing, which is what lets K and V differ.
    pub fn growQuantBuf(s: mlx.mlx_stream, buf: *mlx.mlx_array, g: Grow, B: c_int, heads: c_int, last_dim: c_int, dtype: mlx.mlx_dtype) !void {
        const buf_shape = [_]c_int{ B, heads, g.cap, last_dim };
        if (g.kept > 0) {
            var new_buf = mlx.mlx_array_new();
            errdefer _ = mlx.mlx_array_free(new_buf);
            try mlx.check(mlx.mlx_zeros(&new_buf, &buf_shape, 4, dtype, s));
            const off_c: c_int = @intCast(g.kept);
            const drop_c: c_int = @intCast(g.drop);
            const su_start = [_]c_int{ 0, 0, 0, 0 };
            const su_stop = [_]c_int{ B, heads, off_c, last_dim };
            const su_strides = [_]c_int{ 1, 1, 1, 1 };
            var old_data = mlx.mlx_array_new();
            errdefer _ = mlx.mlx_array_free(old_data);
            try mlx.check(mlx.mlx_slice(&old_data, buf.*, &[_]c_int{ 0, 0, drop_c, 0 }, 4, &[_]c_int{ B, heads, drop_c + off_c, last_dim }, 4, &su_strides, 4, s));
            var updated = mlx.mlx_array_new();
            errdefer _ = mlx.mlx_array_free(updated);
            try mlx.check(mlx.mlx_slice_update(&updated, new_buf, old_data, &su_start, 4, &su_stop, 4, &su_strides, 4, s));
            _ = mlx.mlx_array_free(old_data);
            old_data = .{ .ctx = null };
            _ = mlx.mlx_array_free(new_buf);
            new_buf = .{ .ctx = null };
            _ = mlx.mlx_array_free(buf.*);
            buf.* = updated;
            updated = .{ .ctx = null };
        } else {
            var new_buf = mlx.mlx_array_new();
            errdefer _ = mlx.mlx_array_free(new_buf);
            try mlx.check(mlx.mlx_zeros(&new_buf, &buf_shape, 4, dtype, s));
            _ = mlx.mlx_array_free(buf.*);
            buf.* = new_buf;
            new_buf = .{ .ctx = null };
        }
    }

    fn writeAtOffset(s: mlx.mlx_stream, buf: *mlx.mlx_array, offset: usize, new_chunk: mlx.mlx_array) !void {
        const new_shape = mlx.getShape(new_chunk);
        const new_len: c_int = new_shape[2];
        const buf_shape = mlx.getShape(buf.*);
        const off: c_int = @intCast(offset);
        const off_end: c_int = off + new_len;
        const su_start = [_]c_int{ 0, 0, off, 0 };
        const su_stop = [_]c_int{ buf_shape[0], buf_shape[1], off_end, buf_shape[3] };
        const su_strides = [_]c_int{ 1, 1, 1, 1 };
        var updated = mlx.mlx_array_new();
        errdefer _ = mlx.mlx_array_free(updated);
        try mlx.check(mlx.mlx_slice_update(&updated, buf.*, new_chunk, &su_start, 4, &su_stop, 4, &su_strides, 4, s));
        _ = mlx.mlx_array_free(buf.*);
        buf.* = updated;
    }

    fn buildSliceView(s: mlx.mlx_stream, view: *mlx.mlx_array, buf: mlx.mlx_array, total: c_int, view_start: c_int) !void {
        const buf_cap = bufferCapacity(buf);
        if (view_start == 0 and @as(usize, @intCast(total)) == buf_cap) {
            try mlx.check(mlx.mlx_array_set(view, buf));
        } else {
            const cur_shape = mlx.getShape(buf);
            const v_start = [_]c_int{ 0, 0, view_start, 0 };
            const v_stop = [_]c_int{ cur_shape[0], cur_shape[1], total, cur_shape[3] };
            const v_strides = [_]c_int{ 1, 1, 1, 1 };
            try mlx.check(mlx.mlx_slice(view, buf, &v_start, 4, &v_stop, 4, &v_strides, 4, s));
        }
    }

    pub fn seqLen(self: *const KVCache, layer: u32) usize {
        const entry = &self.entries[layer];
        if (!entry.initialized) return 0;
        return entry.offset;
    }

    pub fn appendEvalArrays(self: *const KVCache, vec: mlx.mlx_vector_array) void {
        const affine = self.config.scheme != .off;
        for (self.entries) |*entry| {
            if (!entry.initialized) continue;
            _ = mlx.mlx_vector_array_append_value(vec, entry.keys);
            _ = mlx.mlx_vector_array_append_value(vec, entry.values);
            if (affine) {
                _ = mlx.mlx_vector_array_append_value(vec, entry.keys_scales);
                _ = mlx.mlx_vector_array_append_value(vec, entry.keys_biases);
                _ = mlx.mlx_vector_array_append_value(vec, entry.values_scales);
                _ = mlx.mlx_vector_array_append_value(vec, entry.values_biases);
            }
        }
    }

    /// Evaluate all KV cache entries to materialize them on GPU.
    /// Called after prefill to ensure the cache is in optimal memory layout for decode.
    pub fn evalState(self: *KVCache) void {
        // Collect all initialized entries into a vector and batch-eval them.
        // This matches mlx_lm's `mx.eval([c.state for c in cache])` pattern.
        const vec = mlx.mlx_vector_array_new();
        defer _ = mlx.mlx_vector_array_free(vec);
        self.appendEvalArrays(vec);
        if (mlx.mlx_vector_array_size(vec) > 0) {
            _ = mlx.mlx_eval(vec);
        }
    }

    /// The sequence length of the attention state this cache holds. `step` is maintained by
    /// `update` on layer 0 only: the absolute position on an attention-first trunk, a
    /// permanent zero on a linear-layer-0 trunk (GDN, Mamba2, gated-conv, KDA), where the
    /// first initialized entry's `offset` is the KV length. Attention-first trunks keep `step`.
    pub fn kvLenForBatching(self: *const KVCache) usize {
        if (self.entries.len > 0 and self.entries[0].initialized) return self.step;
        for (self.entries) |*e| {
            if (e.initialized) return e.offset;
        }
        return self.step;
    }

    /// Truncate the KV cache to keep only the first `len` tokens on the sequence axis.
    /// A draft tree's accepted path: window rows `rows` (written at base + row)
    /// move to base + 1, base + 2, ... in order; the caller truncates after.
    pub fn compactRows(self: *KVCache, base: usize, rows: []const u32, s: mlx.mlx_stream) !void {
        if (rows.len == 0) return;
        var idx_buf: [64]i32 = undefined;
        if (rows.len > idx_buf.len) return error.SpecTreeUnsupported;
        for (self.entries) |*entry| {
            if (!entry.initialized) continue;
            // Row indices are absolute; a sliding ring's buffer starts at `base`.
            for (rows, 0..) |r, i| idx_buf[i] = @intCast(base + r - entry.base);
            const idx = mlx.mlx_array_new_data(&idx_buf, &[_]c_int{@intCast(rows.len)}, 1, .int32);
            defer _ = mlx.mlx_array_free(idx);
            // A quantized row's scales and biases move with it (groups run along head_dim).
            const bufs = [_]*mlx.mlx_array{ &entry.keys, &entry.values, &entry.keys_scales, &entry.keys_biases, &entry.values_scales, &entry.values_biases };
            for (bufs[0..if (self.config.scheme == .off) 2 else bufs.len]) |buf| {
                const sh = mlx.getShape(buf.*);
                var picked = mlx.mlx_array_new();
                defer _ = mlx.mlx_array_free(picked);
                try mlx.check(mlx.mlx_take_axis(&picked, buf.*, idx, 2, s));
                const lo: c_int = @intCast(base + 1 - entry.base);
                const hi: c_int = lo + @as(c_int, @intCast(rows.len));
                var moved = mlx.mlx_array_new();
                try mlx.check(mlx.mlx_slice_update(&moved, buf.*, picked, &[_]c_int{ 0, 0, lo, 0 }, 4, &[_]c_int{ sh[0], sh[1], hi, sh[3] }, 4, &[_]c_int{ 1, 1, 1, 1 }, 4, s));
                _ = mlx.mlx_array_free(buf.*);
                buf.* = moved;
            }
            // The rows now live in this cache's own buffer, not a restored share.
            entry.shared_view = false;
        }
    }

    pub fn truncate(self: *KVCache, len: usize, s: mlx.mlx_stream) !void {
        // A sliding ring cannot rewind past the rows it dropped: refuse before touching any entry.
        for (self.entries) |*entry| {
            if (entry.initialized and entry.base > 0 and len > 0 and len < entry.offset and len < entry.base + entry.window) return error.SlidingRowsDropped;
        }
        self.step = len;
        for (self.entries) |*entry| {
            if (!entry.initialized) continue;
            if (len >= entry.offset) continue;

            // Free stale views (all 6: dense + 4 quant scale/bias views)
            _ = mlx.mlx_array_free(entry.key_view);
            _ = mlx.mlx_array_free(entry.value_view);
            entry.key_view = mlx.mlx_array_new();
            entry.value_view = mlx.mlx_array_new();
            if (self.config.scheme != .off) {
                _ = mlx.mlx_array_free(entry.key_scales_view);
                _ = mlx.mlx_array_free(entry.key_biases_view);
                _ = mlx.mlx_array_free(entry.value_scales_view);
                _ = mlx.mlx_array_free(entry.value_biases_view);
                entry.key_scales_view = mlx.mlx_array_new();
                entry.key_biases_view = mlx.mlx_array_new();
                entry.value_scales_view = mlx.mlx_array_new();
                entry.value_biases_view = mlx.mlx_array_new();
            }

            if (len == 0) {
                _ = mlx.mlx_array_free(entry.keys);
                _ = mlx.mlx_array_free(entry.values);
                entry.keys = mlx.mlx_array_new();
                entry.values = mlx.mlx_array_new();
                if (self.config.scheme != .off) {
                    _ = mlx.mlx_array_free(entry.keys_scales);
                    _ = mlx.mlx_array_free(entry.keys_biases);
                    _ = mlx.mlx_array_free(entry.values_scales);
                    _ = mlx.mlx_array_free(entry.values_biases);
                    entry.keys_scales = mlx.mlx_array_new();
                    entry.keys_biases = mlx.mlx_array_new();
                    entry.values_scales = mlx.mlx_array_new();
                    entry.values_biases = mlx.mlx_array_new();
                }
                entry.initialized = false;
                entry.offset = 0;
                entry.base = 0;
                continue;
            }

            // Just update offset — the buffer still holds data but views will
            // only expose [0:len]. No need to shrink the pre-allocated buffer.
            entry.offset = len;
            try self.rebuildViews(entry, s);
        }
    }

    /// Slice the entry's buffers to its live rows `[0, offset - base)` as the read views. `truncate` needs
    /// them after a rewind, and so does a RESTORED entry, whose views start empty: a layer that reads the
    /// cache before it first appends (GLM's DSA pools the previous indexer rows) would otherwise read nothing.
    fn rebuildViews(self: *const KVCache, entry: *KVCacheEntry, s: mlx.mlx_stream) !void {
        // Each buffer is sliced against its OWN shape — an MLA cache's V is narrower than its K.
        const shape = mlx.getShape(entry.keys);
        if (shape.len < 4) return;
        const seq_end: c_int = @intCast(entry.offset - entry.base);
        const v_start = [_]c_int{ 0, 0, 0, 0 };
        const v_stop = [_]c_int{ shape[0], shape[1], seq_end, shape[3] };
        const v_strides = [_]c_int{ 1, 1, 1, 1 };
        try mlx.check(mlx.mlx_slice(&entry.key_view, entry.keys, &v_start, 4, &v_stop, 4, &v_strides, 4, s));
        const val_shape = mlx.getShape(entry.values);
        const val_stop = [_]c_int{ val_shape[0], val_shape[1], seq_end, val_shape[3] };
        try mlx.check(mlx.mlx_slice(&entry.value_view, entry.values, &v_start, 4, &val_stop, 4, &v_strides, 4, s));
        if (self.config.scheme != .off) {
            // K and V scale/bias groups are as wide as their OWN payload —
            // an MLA cache's V is narrower than its K, so the value views
            // cannot be sliced against the key scales' shape.
            const sc_shape = mlx.getShape(entry.keys_scales);
            const sv_stop = [_]c_int{ sc_shape[0], sc_shape[1], seq_end, sc_shape[3] };
            try mlx.check(mlx.mlx_slice(&entry.key_scales_view, entry.keys_scales, &v_start, 4, &sv_stop, 4, &v_strides, 4, s));
            try mlx.check(mlx.mlx_slice(&entry.key_biases_view, entry.keys_biases, &v_start, 4, &sv_stop, 4, &v_strides, 4, s));
            const vsc_shape = mlx.getShape(entry.values_scales);
            const vsv_stop = [_]c_int{ vsc_shape[0], vsc_shape[1], seq_end, vsc_shape[3] };
            try mlx.check(mlx.mlx_slice(&entry.value_scales_view, entry.values_scales, &v_start, 4, &vsv_stop, 4, &v_strides, 4, s));
            try mlx.check(mlx.mlx_slice(&entry.value_biases_view, entry.values_biases, &v_start, 4, &vsv_stop, 4, &v_strides, 4, s));
        }
    }
};

/// Snapshot of a `KVCache` at a point in time. Owns its array handles (which
/// share buffers with the source via refcount) and frees them in `deinit`.
/// Created by `KVCache.snapshot()` and consumed by `KVCache.restore()`.
pub const KVCacheSnapshot = struct {
    entries: []KVCacheEntry,
    step: usize,
    allocator: std.mem.Allocator,
    config: KVQuantConfig,

    pub fn deinit(self: *KVCacheSnapshot) void {
        for (self.entries) |*e| {
            freeKVEntry(e);
        }
        self.allocator.free(self.entries);
    }

    /// Shortest length this snapshot restores to: a sliding ring must still hold the window
    /// behind it. 0 when no ring has dropped a row.
    pub fn ringFloor(self: *const KVCacheSnapshot) usize {
        var floor: usize = 0;
        for (self.entries) |e| {
            if (e.initialized and e.base > 0) floor = @max(floor, e.base + e.window);
        }
        return floor;
    }

    /// Give up every array handle this snapshot holds, leaving it empty but deinit-able. The
    /// second half of restore by move: `KVCache.restore` binds through `mlx_array_set`, so the
    /// snapshot's second reference made `is_donatable()` fail and the first `writeAtOffset`
    /// privatised the whole prefix (5.13 GB / ~110 ms at 393k, 45% of the warm TTFT). Called
    /// only from `donateCheckout`, right before the slot's first write, so a request refused
    /// before then still hands back an intact entry.
    pub fn releaseHandles(self: *KVCacheSnapshot) void {
        for (self.entries) |*e| {
            freeKVEntry(e);
            e.* = newEmptyKVEntry();
        }
        self.step = 0;
    }

    /// Issue #330: a materialized copy of the first `len` tokens (sequence
    /// axis 2). `KVCache.truncate` is offset-only — the views shrink but a
    /// refcount-shared snapshot keeps the full buffer alive and
    /// `snapshotBytes` keeps billing its capacity — so a byte-budget trim
    /// must be a real slice + copy into fresh buffers. Each array is sliced
    /// against its OWN shape (affine scales/biases are [B,H,T,D/gs]; the
    /// packed codes [B,H,T,D*bits/32] — all carry T at axis 2).
    pub fn trimmedCopy(self: *const KVCacheSnapshot, len: usize, s: mlx.mlx_stream) !KVCacheSnapshot {
        const out = try self.allocator.alloc(KVCacheEntry, self.entries.len);
        var built: usize = 0;
        errdefer {
            for (out[0..built]) |*e| freeKVEntry(e);
            self.allocator.free(out);
        }
        for (self.entries, 0..) |src, i| {
            out[i] = newEmptyKVEntry();
            built = i + 1;
            out[i].initialized = src.initialized;
            out[i].offset = @min(src.offset, len);
            out[i].base = src.base;
            out[i].keep = src.keep;
            out[i].window = src.window;
            if (src.initialized and src.base > 0 and out[i].offset < src.offset and out[i].offset < src.base + src.window) return error.SlidingRowsDropped;
            if (src.initialized) {
                const keep = out[i].offset - src.base;
                out[i].keys = try trimRowsOwned(src.keys, keep, s);
                out[i].values = try trimRowsOwned(src.values, keep, s);
                if (self.config.scheme != .off) {
                    out[i].keys_scales = try trimRowsOwned(src.keys_scales, keep, s);
                    out[i].keys_biases = try trimRowsOwned(src.keys_biases, keep, s);
                    out[i].values_scales = try trimRowsOwned(src.values_scales, keep, s);
                    out[i].values_biases = try trimRowsOwned(src.values_biases, keep, s);
                }
            }
        }
        // One batched eval, like `captureSsmCheckpoint`: without it the lazy
        // copy nodes pin the parent buffers — the capacity the trim exists
        // to release.
        {
            const vec = mlx.mlx_vector_array_new();
            defer _ = mlx.mlx_vector_array_free(vec);
            var count: usize = 0;
            for (out) |*e| {
                inline for (.{ e.keys, e.values, e.keys_scales, e.keys_biases, e.values_scales, e.values_biases }) |arr| {
                    if (arr.ctx != null) {
                        _ = mlx.mlx_vector_array_append_value(vec, arr);
                        count += 1;
                    }
                }
            }
            if (count > 0) _ = mlx.mlx_eval(vec);
        }
        return .{ .entries = out, .step = @min(self.step, len), .allocator = self.allocator, .config = self.config };
    }
};

/// Slice `[0:rows]` on axis 2 and force a real copy (see
/// `materializedOwnedCopy` for why a plain slice keeps the parent alive).
fn trimRowsOwned(x: mlx.mlx_array, rows: usize, s: mlx.mlx_stream) !mlx.mlx_array {
    const sh = mlx.mlx_array_shape(x);
    var sliced = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(sliced);
    const start = [_]c_int{ 0, 0, 0, 0 };
    const stop = [_]c_int{ sh[0], sh[1], @intCast(rows), sh[3] };
    const strides = [_]c_int{ 1, 1, 1, 1 };
    try mlx.check(mlx.mlx_slice(&sliced, x, &start, 4, &stop, 4, &strides, 4, s));
    return materializedOwnedCopy(s, sliced);
}
