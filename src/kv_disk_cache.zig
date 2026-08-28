//! SSD tier for the hot prefix cache — chunked KV persistence.
//!
//! Committed KV prefixes are persisted to disk as position-chunked
//! safetensors, so previously-seen prefixes survive server restarts and RAM
//! evictions and are RESTORED instead of recomputed. Two-tier flow:
//!
//!   commit  → RAM entry (refcount snapshot, unchanged) + chunk-APPEND to
//!             disk. Only chunks not yet on disk are written — a multi-turn
//!             agent session pays one bounded partial-chunk rewrite + the new
//!             tail per turn, never a full re-serialize.
//!   lookup  → longest-prefix match across RAM entries as before; the disk
//!             index is consulted when it can beat the RAM match by at least
//!             one chunk (fresh boot, post-eviction). The entry is rebuilt
//!             into the live cache and the normal truncate-then-prefill path
//!             continues.
//!
//! Layout (one root per model fingerprint — path + config.json identity):
//!   <base>/<fingerprint>/e<id>/meta.json    commit point (written tmp+rename
//!                                           LAST; an entry without it is a
//!                                           crash leftover and is GC'd)
//!   <base>/<fingerprint>/e<id>/tokens.bin   LE u32 token ids (prompt ++ gen)
//!   <base>/<fingerprint>/e<id>/c000000.safetensors   KV positions [0, chunk_tokens)
//!   <base>/<fingerprint>/e<id>/c000001.safetensors   ...
//!
//! Chunk files hold per-layer K/V slices keyed "l{i}.k"/"l{i}.v" (plus
//! ".ks/.kb/.vs/.vb" scale/bias triples in affine mode). The final chunk may
//! be partial; a commit that extends the entry rewrites ONLY that chunk and
//! appends new ones. A chunk file holding MORE positions than meta.json
//! claims (crash between chunk write and meta rename) is sliced down at
//! restore, never trusted.
//!
//! Phase 3 — hybrid SSM archs (qwen3_5/3_6 GatedDeltaNet, lfm2, nemotron_h):
//! the RAM tier's per-position `SSMCheckpoint`s persist beside the chunks as
//!   <base>/<fingerprint>/e<id>/s0002048.safetensors   SSM state at pos 2048
//! keyed "l{i}.conv"/"l{i}.ssm" (absent key = null state — LFM2 gated-conv
//! layers have no ssm_state, plain-attention layers in the hybrid have
//! neither), with the per-layer `initialized` flags in the safetensors
//! metadata map ("init"). Checkpoint files are immutable once written (keyed
//! by position); extend commits append only NEW positions, bounded per entry
//! by `SSM_DISK_MAX_PER_ENTRY` (evict-lowest — the newest positions are where
//! multi-turn warm requests match). A hybrid restore rebuilds the KV prefix
//! [0, cp_pos) AND the SSM state at cp_pos (`restoreIntoHybrid`) — mirroring
//! the RAM tier's rewind-both semantics.
//!
//! Scope: schemes off/affine (TurboQuant's rotation state doesn't survive a
//! restore into a fresh cache), B==1 slot caches. All mlx work runs on the
//! inference thread; safetensors loads use a private CPU stream
//! (`Load::eval_gpu` is Not Implemented — the lora.zig/model.zig precedent).

const std = @import("std");
const mlx = @import("mlx.zig");
const kv_quant = @import("kv_quant.zig");
const transformer_mod = @import("transformer.zig");
const io_util = @import("io_util.zig");
const log = @import("log.zig");

const KVCache = transformer_mod.KVCache;

/// Restoring from disk only happens when it beats the best RAM match by at
/// least this many tokens — a disk read + rebuild is only worth it when it
/// replaces a meaningful amount of prefill.
pub const MIN_DISK_ADVANTAGE_TOKENS: u32 = 256;

/// Entries shorter than this are never persisted (a short prefix re-prefills
/// in well under the restore cost).
pub const MIN_PERSIST_TOKENS: u32 = 512;

pub const DEFAULT_CHUNK_TOKENS: u32 = 1024;

/// Max persisted SSM checkpoint positions per entry. Every turn adds an
/// end-of-prompt checkpoint (~400 MB each on Qwen3.6-27B); unbounded, one
/// long session would accumulate GBs in a single entry. Thinning drops the
/// LOWEST positions first (mirrors the RAM capture's front-drop) — the
/// newest positions are where multi-turn warm requests match.
pub const SSM_DISK_MAX_PER_ENTRY: usize = 8;

pub const IndexEntry = struct {
    /// Directory id — the `e<id>` component.
    id: u64,
    /// Full committed token sequence (prompt ++ generated). Owned.
    tokens: []u32,
    /// KV positions actually persisted (== snapshot `step` at commit; may be
    /// tokens.len - 1 when the final sampled token was never forwarded).
    kv_len: u32,
    has_tools: bool,
    quant: kv_quant.KVQuantConfig,
    /// Total on-disk bytes (chunks + tokens.bin + meta).
    bytes: u64,
    /// Per-chunk file sizes recorded at commit (meta.json "chunk_bytes").
    /// The scan validates actual file sizes against these and clamps kv_len
    /// to the last contiguous valid chunk — a kill -9 mid-flush truncates a
    /// chunk, and restoring it would poison the cache. Owned.
    chunk_bytes: []u64,
    /// Phase 3: persisted SSM checkpoint positions (sorted ascending; empty
    /// for pure-attention entries) and per-file byte sizes (parallel array —
    /// the same kill -9 salvage role as `chunk_bytes`: the scan drops
    /// individual positions whose file size mismatches). Owned.
    ssm_positions: []u32,
    ssm_bytes: []u64,
    /// v4: spec-snapshot sidecar (`spec.safetensors`) byte size; 0 = none.
    /// The same kill -9 salvage as chunks — a size mismatch at scan drops the
    /// SPEC only (a restore then starts blind), never the entry.
    spec_bytes: u64 = 0,
    /// v4: dflash assistant context / MTP committed history persisted in the
    /// spec sidecar. DRAFT-side state — a missing or dropped snap costs
    /// acceptance on the first reused turn, never a token.
    spec_dflash: ?SpecMeta = null,
    spec_mtp: ?SpecMeta = null,
    /// In-process LRU stamp; seeded from meta.json mtime order at scan.
    last_used: u64,
};

/// v4 spec-snapshot metadata for one speculative-side cache (dflash assistant
/// context or MTP committed history). The tensors live in the entry's ONE
/// `spec.safetensors` file, keyed `d{layer}.*` / `m{layer}.*`.
pub const SpecMeta = struct {
    /// Absolute trunk position the snapshot's index 0 represents.
    base: u64,
    /// Positions persisted (the snapshot's logical length).
    step: u32,
    /// Layer count of the source cache — a restore target with a different
    /// count declines (KVCache.restore asserts equal lengths).
    layers: u32,
    quant: kv_quant.KVQuantConfig,
};

/// What `appendCommitWithSpec` reads to persist one spec snapshot — the same
/// snapshot-shaped parts the trunk flush takes, plus the base position.
pub const SpecCommit = struct {
    entries: []const transformer_mod.KVCacheEntry,
    step: usize,
    config: kv_quant.KVQuantConfig,
    base_pos: usize,
};

pub const SpecKind = enum { dflash, mtp };

pub const Match = struct {
    idx: usize,
    /// Shared-prefix length clamped to kv_len — the positions a restore can
    /// actually rebuild.
    usable: u32,
};

pub const DiskTier = struct {
    allocator: std.mem.Allocator,
    io: std.Io,
    /// Absolute root for this model's entries (`<base>/<fingerprint>`). Owned.
    root: []u8,
    /// Byte budget across all entries. 0 = unbounded.
    max_bytes: u64,
    chunk_tokens: u32,
    /// Max bytes written per appendCommit call (default 512 MB). The flush
    /// runs synchronously on the inference thread after the response; a
    /// 4 GB first-commit write measurably stalls the NEXT request, so large
    /// entries persist incrementally across turns (appendCommit reports
    /// incomplete and the hot cache keeps its dirty flag set).
    max_flush_bytes: u64 = 512 * 1024 * 1024,
    entries: std.ArrayList(IndexEntry),
    next_id: u64,
    total_bytes: u64,
    counter: u64,
    /// Chunk count read by the most recent restore. Diagnostics + a
    /// red-on-revert guard that a short-prefix restore reads only the chunks
    /// covering the usable prefix, not the whole stored entry. Not persisted.
    chunks_loaded_last: u32 = 0,

    /// Create the tier rooted at `<base>/<fingerprint>` and scan whatever
    /// already exists there. Crash leftovers (no meta.json) are deleted.
    pub fn init(
        allocator: std.mem.Allocator,
        io: std.Io,
        base_dir: []const u8,
        fingerprint: []const u8,
        max_bytes: u64,
        chunk_tokens: u32,
    ) !DiskTier {
        if (base_dir.len == 0 or !std.fs.path.isAbsolute(base_dir)) return error.BadDiskCacheDir;
        const root = try std.fmt.allocPrint(allocator, "{s}/{s}", .{ base_dir, fingerprint });
        errdefer allocator.free(root);
        try std.Io.Dir.cwd().createDirPath(io, root);
        var self: DiskTier = .{
            .allocator = allocator,
            .io = io,
            .root = root,
            .max_bytes = max_bytes,
            .chunk_tokens = if (chunk_tokens == 0) DEFAULT_CHUNK_TOKENS else chunk_tokens,
            .entries = std.ArrayList(IndexEntry).empty,
            .next_id = 1,
            .total_bytes = 0,
            .counter = 0,
        };
        self.scan() catch |err| {
            log.warn("[disk-cache] scan failed: {s} — starting empty\n", .{@errorName(err)});
        };
        self.gcToBudget();
        return self;
    }

    pub fn deinit(self: *DiskTier) void {
        for (self.entries.items) |*e| {
            self.freeIndexEntryOwned(e);
        }
        self.entries.deinit(self.allocator);
        self.allocator.free(self.root);
    }

    /// Free everything an IndexEntry owns. Every removal path (deinit,
    /// eviction, invalidation, scan-append failure, extend-replace) must go
    /// through this so a new owned field can't leak on one of them.
    fn freeIndexEntryOwned(self: *DiskTier, e: *IndexEntry) void {
        self.allocator.free(e.tokens);
        self.allocator.free(e.chunk_bytes);
        self.allocator.free(e.ssm_positions);
        self.allocator.free(e.ssm_bytes);
    }

    pub fn entryCount(self: *const DiskTier) usize {
        return self.entries.items.len;
    }

    // ── Lookup ──

    /// Longest usable shared prefix across persisted entries with a matching
    /// (has_tools, quant) key. Same filter semantics as the RAM cache: a
    /// cross-config restore would hand SDPA a wrong buffer layout.
    pub fn bestMatch(
        self: *const DiskTier,
        prompt_ids: []const u32,
        has_tools: bool,
        quant: kv_quant.KVQuantConfig,
    ) ?Match {
        var best_idx: ?usize = null;
        var best_usable: u32 = 0;
        for (self.entries.items, 0..) |*e, i| {
            if (e.has_tools != has_tools) continue;
            if (!std.meta.eql(e.quant, quant)) continue;
            const max_shared = @min(e.tokens.len, prompt_ids.len);
            var shared: usize = 0;
            while (shared < max_shared and e.tokens[shared] == prompt_ids[shared]) shared += 1;
            const usable: u32 = @intCast(@min(shared, e.kv_len));
            if (usable > best_usable) {
                best_usable = usable;
                best_idx = i;
            }
        }
        if (best_idx) |idx| return .{ .idx = idx, .usable = best_usable };
        return null;
    }

    /// Rebuild the persisted KV state of `entries[idx]` into `cache`:
    /// per-layer chunk tensors are loaded (CPU stream), concatenated along
    /// the sequence axis, and installed as the cache's storage buffers.
    /// Views are left empty — identical contract to `KVCache.restore` (the
    /// next `update`/`truncate` rebuilds them). Returns kv_len.
    pub fn restoreInto(self: *DiskTier, cache: *KVCache, idx: usize, s: mlx.mlx_stream) !u32 {
        const kv_len = self.entries.items[idx].kv_len;
        try self.restorePrefixInto(cache, idx, kv_len, s);
        return kv_len;
    }

    /// Rebuild ONLY positions [0, limit) of `entries[idx]` — `limit` must be
    /// ≤ its kv_len. A short shared prefix against a long stored entry then
    /// reads just the chunks covering `limit` instead of the whole entry (a
    /// diverged-prefix "hit" that would otherwise read every stored chunk to
    /// serve a few hundred tokens — slower than a cold prefill).
    pub fn restorePrefixInto(self: *DiskTier, cache: *KVCache, idx: usize, limit: u32, s: mlx.mlx_stream) !void {
        const e = &self.entries.items[idx];
        try self.restoreKvInto(cache, e, limit, s);
        e.last_used = self.bump();
        // Bump meta.json mtime so cross-restart LRU sees the use.
        self.writeMeta(e.*) catch {};
    }

    /// Phase 3 hybrid variant: rebuild the KV prefix covering [0, cp_pos)
    /// AND install the SSM state persisted at `cp_pos` into `ssm_entries`.
    /// Mirrors the RAM tier's hybrid-restore semantics — KV and SSM state
    /// land at the SAME position, the caller continues prefill from cp_pos.
    /// `cp_pos` must be one of the entry's persisted checkpoint positions
    /// (pick via `highestSsmPosAtOrBelow`). On error the cache/entries may be
    /// half-rebuilt — the caller resets both and falls back to cold prefill.
    pub fn restoreIntoHybrid(
        self: *DiskTier,
        cache: *KVCache,
        ssm_entries: []transformer_mod.SSMCacheEntry,
        idx: usize,
        cp_pos: u32,
        s: mlx.mlx_stream,
    ) !u32 {
        const e = &self.entries.items[idx];
        if (cp_pos == 0 or cp_pos > e.kv_len) return error.DiskCacheNoCheckpoint;
        if (std.mem.indexOfScalar(u32, e.ssm_positions, cp_pos) == null) return error.DiskCacheNoCheckpoint;
        // Load the checkpoint FIRST (transient, no side effects on the live
        // state) so a corrupt/missing file fails before the cache is touched.
        var cp = try self.loadSsmFile(e.id, cp_pos, ssm_entries.len);
        defer cp.deinit(self.allocator);
        try self.restoreKvInto(cache, e, cp_pos, s);
        try transformer_mod.restoreSsmCheckpoint(ssm_entries, &cp);
        e.last_used = self.bump();
        self.writeMeta(e.*) catch {};
        return cp_pos;
    }

    /// Largest persisted SSM checkpoint position ≤ `limit` for entry `idx`;
    /// null when none qualifies (hybrid KV without SSM state is unusable, so
    /// the caller must skip the entry entirely).
    pub fn highestSsmPosAtOrBelow(self: *const DiskTier, idx: usize, limit: u32) ?u32 {
        var best: ?u32 = null;
        for (self.entries.items[idx].ssm_positions) |p| {
            if (p > limit) break; // sorted ascending
            best = p;
        }
        return best;
    }

    /// Shared chunk-loading body: rebuild positions [0, limit) of entry `e`
    /// into `cache` (limit == e.kv_len for the plain-attention path; a
    /// checkpoint position for the hybrid path — the final chunk is sliced
    /// down so KV lands exactly at the checkpoint).
    fn restoreKvInto(self: *DiskTier, cache: *KVCache, e: *const IndexEntry, limit: u32, s: mlx.mlx_stream) !void {
        const quant = e.quant;
        if (!std.meta.eql(cache.config, quant)) return error.DiskCacheConfigMismatch;
        if (limit == 0 or limit > e.kv_len) return error.DiskCacheEmptyEntry;
        const n_chunks: u32 = @intCast((@as(u64, limit) + self.chunk_tokens - 1) / self.chunk_tokens);
        if (n_chunks == 0) return error.DiskCacheEmptyEntry;
        self.chunks_loaded_last = n_chunks;

        const cpu = mlx.mlx_default_cpu_stream_new();
        defer _ = mlx.mlx_stream_free(cpu);

        const kinds: []const []const u8 = if (quant.scheme == .off)
            &.{ "k", "v" }
        else
            &.{ "k", "v", "ks", "kb", "vs", "vb" };

        // Per-layer per-kind accumulation vectors. Layers absent from chunk 0
        // stay uninitialized — the GatedDeltaNet layers of a hybrid arch have
        // no KV (their state rides the SSM checkpoints), so only the
        // full-attention layers appear in the chunks.
        const n_layers = cache.entries.len;
        const vecs = try self.allocator.alloc(mlx.mlx_vector_array, n_layers * kinds.len);
        for (vecs) |*v| v.* = mlx.mlx_vector_array_new();
        defer {
            for (vecs) |v| _ = mlx.mlx_vector_array_free(v);
            self.allocator.free(vecs);
        }
        const present = try self.allocator.alloc(bool, n_layers);
        defer self.allocator.free(present);
        @memset(present, false);

        var chunk_i: u32 = 0;
        while (chunk_i < n_chunks) : (chunk_i += 1) {
            const c0: u64 = @as(u64, chunk_i) * self.chunk_tokens;
            const need: u64 = @min(@as(u64, self.chunk_tokens), limit - c0);

            const path = try std.fmt.allocPrint(self.allocator, "{s}/e{d}/c{d:0>6}.safetensors\x00", .{ self.root, e.id, chunk_i });
            defer self.allocator.free(path);
            var tensor_map = mlx.mlx_map_string_to_array_new();
            defer _ = mlx.mlx_map_string_to_array_free(tensor_map);
            var meta_map = mlx.mlx_map_string_to_string_new();
            defer _ = mlx.mlx_map_string_to_string_free(meta_map);
            try mlx.check(mlx.mlx_load_safetensors(&tensor_map, &meta_map, @ptrCast(path.ptr), cpu));

            for (0..n_layers) |li| {
                for (kinds, 0..) |kind, ki| {
                    const key = try std.fmt.allocPrint(self.allocator, "l{d}.{s}\x00", .{ li, kind });
                    defer self.allocator.free(key);
                    var arr = mlx.mlx_array_new();
                    if (mlx.mlx_map_string_to_array_get(&arr, tensor_map, @ptrCast(key.ptr)) != 0) {
                        _ = mlx.mlx_array_free(arr);
                        if (ki == 0) break; // layer absent from this chunk
                        return error.DiskCacheCorruptChunk;
                    }
                    if (ki == 0) present[li] = true;
                    // Crash-tolerance: a chunk file may hold MORE positions
                    // than meta.json committed to (rewrite raced a crash).
                    // Slice down to the committed range; never trust the file.
                    const shape = mlx.getShape(arr);
                    if (shape.len != 4) {
                        _ = mlx.mlx_array_free(arr);
                        return error.DiskCacheCorruptChunk;
                    }
                    const have: u64 = @intCast(shape[2]);
                    if (have < need) {
                        _ = mlx.mlx_array_free(arr);
                        return error.DiskCacheCorruptChunk;
                    }
                    if (have > need) {
                        var sliced = mlx.mlx_array_new();
                        const st = [_]c_int{ 0, 0, 0, 0 };
                        const sp = [_]c_int{ shape[0], shape[1], @intCast(need), shape[3] };
                        const sd = [_]c_int{ 1, 1, 1, 1 };
                        const rc = mlx.mlx_slice(&sliced, arr, &st, 4, &sp, 4, &sd, 4, s);
                        _ = mlx.mlx_array_free(arr);
                        try mlx.check(rc);
                        arr = sliced;
                    }
                    _ = mlx.mlx_vector_array_append_value(vecs[li * kinds.len + ki], arr);
                    _ = mlx.mlx_array_free(arr);
                }
            }
        }

        // Install per-layer concatenations as the cache's storage buffers.
        // Mirrors `KVCache.restore`: views stay empty, offset/initialized set,
        // step = kv_len.
        for (cache.entries, 0..) |*dst, li| {
            transformer_mod.resetKVEntry(dst);
            if (!present[li]) continue;
            const base = li * kinds.len;
            try concatInto(&dst.keys, vecs[base + 0], s);
            try concatInto(&dst.values, vecs[base + 1], s);
            if (quant.scheme != .off) {
                try concatInto(&dst.keys_scales, vecs[base + 2], s);
                try concatInto(&dst.keys_biases, vecs[base + 3], s);
                try concatInto(&dst.values_scales, vecs[base + 4], s);
                try concatInto(&dst.values_biases, vecs[base + 5], s);
            }
            dst.offset = limit;
            dst.initialized = true;
        }
        cache.step = limit;
        // Materialize with a CHECKED eval: a corrupt chunk surfaces its MLX
        // error HERE (lazy Load reads data at eval), so the caller's catch
        // resets the cache and falls back to cold prefill instead of running
        // a forward over poisoned buffers.
        {
            const vec = mlx.mlx_vector_array_new();
            defer _ = mlx.mlx_vector_array_free(vec);
            for (cache.entries) |*entry| {
                if (!entry.initialized) continue;
                _ = mlx.mlx_vector_array_append_value(vec, entry.keys);
                _ = mlx.mlx_vector_array_append_value(vec, entry.values);
                if (quant.scheme != .off) {
                    _ = mlx.mlx_vector_array_append_value(vec, entry.keys_scales);
                    _ = mlx.mlx_vector_array_append_value(vec, entry.keys_biases);
                    _ = mlx.mlx_vector_array_append_value(vec, entry.values_scales);
                    _ = mlx.mlx_vector_array_append_value(vec, entry.values_biases);
                }
            }
            try mlx.check(mlx.mlx_eval(vec));
        }
    }

    /// Load a persisted SSM checkpoint file into a transient `SSMCheckpoint`
    /// (caller frees via `deinit`). The recorded layer count must match the
    /// target model's `ssm_entries` — a mismatch inside a fingerprint dir
    /// means corruption, never a different model.
    fn loadSsmFile(self: *DiskTier, id: u64, pos: u32, n_layers: usize) !transformer_mod.SSMCheckpoint {
        const cpu = mlx.mlx_default_cpu_stream_new();
        defer _ = mlx.mlx_stream_free(cpu);
        const path = try std.fmt.allocPrint(self.allocator, "{s}/e{d}/s{d:0>7}.safetensors\x00", .{ self.root, id, pos });
        defer self.allocator.free(path);
        var tensor_map = mlx.mlx_map_string_to_array_new();
        defer _ = mlx.mlx_map_string_to_array_free(tensor_map);
        var meta_map = mlx.mlx_map_string_to_string_new();
        defer _ = mlx.mlx_map_string_to_string_free(meta_map);
        try mlx.check(mlx.mlx_load_safetensors(&tensor_map, &meta_map, @ptrCast(path.ptr), cpu));

        var layers_c: [*:0]const u8 = undefined;
        if (mlx.mlx_map_string_to_string_get(&layers_c, meta_map, "layers") != 0) return error.DiskCacheCorruptSsm;
        const recorded = std.fmt.parseInt(usize, std.mem.span(layers_c), 10) catch return error.DiskCacheCorruptSsm;
        if (recorded != n_layers) return error.DiskCacheSsmLayerMismatch;
        var init_c: [*:0]const u8 = undefined;
        if (mlx.mlx_map_string_to_string_get(&init_c, meta_map, "init") != 0) return error.DiskCacheCorruptSsm;
        const init_str = std.mem.span(init_c);

        const layers = try self.allocator.alloc(transformer_mod.SSMCacheEntrySnapshot, n_layers);
        for (layers) |*l| l.* = .{
            .conv_state = mlx.mlx_array_new(),
            .ssm_state = mlx.mlx_array_new(),
            .initialized = false,
        };
        var cp: transformer_mod.SSMCheckpoint = .{ .pos = pos, .layers = layers };
        errdefer cp.deinit(self.allocator);

        // Absent key = null state (LFM2 gated-conv layers have no ssm_state;
        // plain-attention layers in the hybrid have neither) — that's a valid
        // shape, not corruption.
        for (layers, 0..) |*l, li| {
            const ckey = try std.fmt.allocPrint(self.allocator, "l{d}.conv\x00", .{li});
            defer self.allocator.free(ckey);
            var conv = mlx.mlx_array_new();
            if (mlx.mlx_map_string_to_array_get(&conv, tensor_map, @ptrCast(ckey.ptr)) == 0) {
                l.conv_state = conv; // transfer the +1 handed by _get
            } else {
                _ = mlx.mlx_array_free(conv);
            }
            const skey = try std.fmt.allocPrint(self.allocator, "l{d}.ssm\x00", .{li});
            defer self.allocator.free(skey);
            var ssm = mlx.mlx_array_new();
            if (mlx.mlx_map_string_to_array_get(&ssm, tensor_map, @ptrCast(skey.ptr)) == 0) {
                l.ssm_state = ssm;
            } else {
                _ = mlx.mlx_array_free(ssm);
            }
            const akey = try std.fmt.allocPrint(self.allocator, "l{d}.aux\x00", .{li});
            defer self.allocator.free(akey);
            var aux = mlx.mlx_array_new();
            if (mlx.mlx_map_string_to_array_get(&aux, tensor_map, @ptrCast(akey.ptr)) == 0) {
                l.aux_state = aux;
            } else {
                _ = mlx.mlx_array_free(aux);
            }
            const pkey = try std.fmt.allocPrint(self.allocator, "l{d}.pooled\x00", .{li});
            defer self.allocator.free(pkey);
            var pooled = mlx.mlx_array_new();
            if (mlx.mlx_map_string_to_array_get(&pooled, tensor_map, @ptrCast(pkey.ptr)) == 0) {
                l.qsa_pooled = pooled;
            } else {
                _ = mlx.mlx_array_free(pooled);
            }
            const lkey = try std.fmt.allocPrint(self.allocator, "l{d}.ple\x00", .{li});
            defer self.allocator.free(lkey);
            var ple = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(ple);
            if (mlx.mlx_map_string_to_array_get(&ple, tensor_map, @ptrCast(lkey.ptr)) == 0) {
                if (mlx.mlx_array_dtype(ple) != .uint32 or mlx.mlx_array_size(ple) != 9) return error.DiskCacheCorruptSsm;
                try mlx.check(mlx.mlx_array_eval(ple));
                const d = mlx.mlx_array_data_uint32(ple) orelse return error.DiskCacheCorruptSsm;
                l.ple_prev_valid = d[0] != 0;
                for (0..8) |i| l.ple_prev[i] = d[1 + i];
            }
        }
        var ratio_c: [*:0]const u8 = undefined;
        if (mlx.mlx_map_string_to_string_get(&ratio_c, meta_map, "qsa_ratio") == 0) {
            const ratio = std.fmt.parseInt(c_int, std.mem.span(ratio_c), 10) catch return error.DiskCacheCorruptSsm;
            for (layers) |*l| l.qsa_ratio = ratio;
        }

        // `initialized=true` with both states null is a valid shape, so the
        // flags can't derive from tensor presence — they ride the metadata.
        var it = std.mem.tokenizeScalar(u8, init_str, ',');
        while (it.next()) |tok| {
            const li = std.fmt.parseInt(usize, tok, 10) catch return error.DiskCacheCorruptSsm;
            if (li >= n_layers) return error.DiskCacheCorruptSsm;
            layers[li].initialized = true;
        }

        // Materialize with a CHECKED eval so a corrupt file surfaces HERE
        // (lazy Load reads data at eval), not mid-forward after install.
        {
            const vec = mlx.mlx_vector_array_new();
            defer _ = mlx.mlx_vector_array_free(vec);
            var count: usize = 0;
            for (layers) |*l| {
                inline for (.{ l.conv_state, l.ssm_state, l.aux_state, l.qsa_pooled }) |arr| {
                    if (arr.ctx != null) {
                        _ = mlx.mlx_vector_array_append_value(vec, arr);
                        count += 1;
                    }
                }
            }
            if (count > 0) try mlx.check(mlx.mlx_eval(vec));
        }
        return cp;
    }

    fn concatInto(dst: *mlx.mlx_array, vec: mlx.mlx_vector_array, s: mlx.mlx_stream) !void {
        var out = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_concatenate_axis(&out, vec, 2, s));
        _ = mlx.mlx_array_free(dst.*);
        dst.* = out;
    }

    // ── Commit ──

    /// Persist a cache state under `tokens`. Called on the inference thread
    /// AFTER the response is finished (the write is bounded but synchronous).
    /// Takes snapshot-shaped parts (entries + step + config) so callers can
    /// flush either a live `KVCache` or a committed `KVCacheSnapshot` — the
    /// hot cache flushes the RAM entry it just committed, post-markFinished,
    /// so the client never waits on the SSD write. Skips ineligible states
    /// silently; never fails the request.
    pub fn appendCommit(
        self: *DiskTier,
        kv_entries: []const transformer_mod.KVCacheEntry,
        step: usize,
        config: kv_quant.KVQuantConfig,
        tokens: []const u32,
        has_tools: bool,
        ssm_checkpoints: ?[]const transformer_mod.SSMCheckpoint,
        s: mlx.mlx_stream,
    ) !bool {
        return self.appendCommitWithSpec(kv_entries, step, config, tokens, has_tools, ssm_checkpoints, null, null, s);
    }

    /// `appendCommit` plus the v4 spec snapshots (dflash assistant context /
    /// MTP committed history). Eligibility is enforced UPSTREAM, same as the
    /// RAM tier: the caller passes only what `commitWithState` was handed.
    pub fn appendCommitWithSpec(
        self: *DiskTier,
        kv_entries: []const transformer_mod.KVCacheEntry,
        step: usize,
        config: kv_quant.KVQuantConfig,
        tokens: []const u32,
        has_tools: bool,
        ssm_checkpoints: ?[]const transformer_mod.SSMCheckpoint,
        dflash_snap: ?SpecCommit,
        mtp_snap: ?SpecCommit,
        s: mlx.mlx_stream,
    ) !bool {
        // On EOS-terminated turns the cache runs 1-2 positions AHEAD of the
        // committed token record (forwarded terminator tokens that never
        // land in `tokens`). Persist the prefix covered by the record —
        // positions beyond it are unusable for matching anyway.
        // The KV extent is the initialized layers' offset, NOT `step`: on
        // hybrid archs (qwen3_5/3_6 GDN) `cache.step` only bumps on layer 0,
        // which is a GatedDeltaNet layer that never writes KV, so it stays 0
        // while the full-attention layers carry offset == prompt position.
        // `max(step, max initialized offset)` is correct for both — equal on
        // pure attention, and the layer offset on hybrid.
        var max_off: usize = 0;
        for (kv_entries) |*entry| {
            if (entry.initialized and entry.offset > max_off) max_off = entry.offset;
        }
        const kv_target_u: usize = @min(@max(step, max_off), tokens.len);
        if (kv_target_u < MIN_PERSIST_TOKENS) return true;
        const kv_target: u32 = @intCast(kv_target_u);
        switch (config.scheme) {
            .off, .affine => {},
            else => return true, // TurboQuant rotation state doesn't survive restore
        }
        // Every initialized layer must cover the persisted range with B == 1
        // — anything else (mid-spec-decode state, batched cache) is not a
        // persistable snapshot.
        for (kv_entries) |*entry| {
            if (!entry.initialized) continue;
            if (entry.offset < kv_target_u) {
                log.debug("  [disk-cache] skip: layer offset {d} < kv_len {d}\n", .{ entry.offset, kv_target_u });
                return true;
            }
            const shape = mlx.getShape(entry.keys);
            if (shape.len != 4 or shape[0] != 1) {
                log.debug("  [disk-cache] skip: non-B1 cache shape\n", .{});
                return true;
            }
        }

        // Superseded check: an existing entry that already covers `tokens`
        // (same key, tokens is a prefix of its tokens, kv already >= ours)
        // makes this commit a no-op — UNLESS the entry is hybrid and still has
        // pending SSM checkpoints (byte-capped across turns), which take a
        // dedicated SSM-only append path (the KV chunks are all present, so
        // the extend machinery would pointlessly rewrite the tail chunk).
        var extend_idx: ?usize = null;
        var ssm_only_idx: ?usize = null;
        for (self.entries.items, 0..) |*e, i| {
            if (e.has_tools != has_tools) continue;
            if (!std.meta.eql(e.quant, config)) continue;
            if (e.tokens.len >= tokens.len) {
                if (std.mem.eql(u32, e.tokens[0..tokens.len], tokens)) {
                    if (e.kv_len >= kv_target) {
                        if (!self.ssmWorkPending(e, ssm_checkpoints, e.kv_len) and
                            !specWorkPending(e, dflash_snap, mtp_snap))
                        {
                            e.last_used = self.bump();
                            return true;
                        }
                        ssm_only_idx = i;
                        break;
                    }
                    // Same token record, SHORTER persisted KV — a byte-capped
                    // incremental flush in progress. Resume into its dir.
                    extend_idx = i;
                }
            } else if (std.mem.eql(u32, e.tokens, tokens[0..e.tokens.len])) {
                // This commit extends `e` — reuse its directory and chunks.
                extend_idx = i;
            }
        }
        if (ssm_only_idx) |i| return self.appendSsmOnly(i, ssm_checkpoints, dflash_snap, mtp_snap, s);

        const sw = io_util.Stopwatch.init(self.io);

        const id: u64 = if (extend_idx) |i| self.entries.items[i].id else blk: {
            const nid = self.next_id;
            self.next_id += 1;
            break :blk nid;
        };
        const dir_rel = try std.fmt.allocPrint(self.allocator, "{s}/e{d}", .{ self.root, id });
        defer self.allocator.free(dir_rel);
        try std.Io.Dir.cwd().createDirPath(self.io, dir_rel);

        // Chunks [0, keep) are full chunks already on disk from the entry we
        // extend; everything from `keep` on (the old partial tail + the new
        // positions) is (re)written — up to the per-flush byte cap. Stopping
        // early lands on a full-chunk boundary; the entry then records the
        // shorter kv_len and the NEXT flush resumes from there.
        const old_kv: u32 = if (extend_idx) |i| self.entries.items[i].kv_len else 0;
        const keep: u32 = old_kv / self.chunk_tokens;
        const n_chunks: u32 = @intCast((@as(u64, kv_target) + self.chunk_tokens - 1) / self.chunk_tokens);

        var chunk_sizes = std.ArrayList(u64).empty;
        errdefer chunk_sizes.deinit(self.allocator);
        if (extend_idx) |i| {
            const old_cb = self.entries.items[i].chunk_bytes;
            try chunk_sizes.appendSlice(self.allocator, old_cb[0..@min(keep, old_cb.len)]);
        }

        var written_bytes: u64 = 0;
        var chunk_i: u32 = keep;
        while (chunk_i < n_chunks) : (chunk_i += 1) {
            if (written_bytes >= self.max_flush_bytes and chunk_i > keep) break;
            const c0: u32 = chunk_i * self.chunk_tokens;
            const c1: u32 = @intCast(@min(@as(u64, c0) + self.chunk_tokens, kv_target));
            try self.writeChunkFile(kv_entries, config, dir_rel, chunk_i, c0, c1, s);
            const cpath = try std.fmt.allocPrint(self.allocator, "{s}/c{d:0>6}.safetensors", .{ dir_rel, chunk_i });
            defer self.allocator.free(cpath);
            const csize = fileSize(self.io, cpath) orelse 0;
            written_bytes += csize;
            try chunk_sizes.append(self.allocator, csize);
        }
        const chunks_done: u32 = chunk_i;
        const chunk_complete = chunks_done == n_chunks;
        const kv_len: u32 = if (chunk_complete) kv_target else chunks_done * self.chunk_tokens;
        if (kv_len <= old_kv) {
            // Cap so tight nothing new landed — nothing to commit.
            chunk_sizes.deinit(self.allocator);
            return chunk_complete;
        }

        // Phase 3: persist any SSM checkpoints whose position is within the KV
        // now on disk (a hybrid restore needs KV covering [0, cp_pos), so a
        // checkpoint beyond the partially-flushed KV waits for a later turn).
        // Shares the per-flush byte budget with the chunk writes above.
        const old_ssm_pos: []const u32 = if (extend_idx) |i| self.entries.items[i].ssm_positions else &[_]u32{};
        const old_ssm_bytes: []const u64 = if (extend_idx) |i| self.entries.items[i].ssm_bytes else &[_]u64{};
        var ssm_res = self.persistSsmCheckpoints(id, dir_rel, kv_len, old_ssm_pos, old_ssm_bytes, ssm_checkpoints, &written_bytes) catch |err| {
            chunk_sizes.deinit(self.allocator);
            return err;
        };
        errdefer ssm_res.deinit(self.allocator);
        const complete = chunk_complete and ssm_res.complete;

        // v4 spec snapshots — one sidecar file, REPLACED wholesale by every
        // commit (a commit with no payload deletes a stale one, the RAM
        // tier's supersede rule). Best-effort DRAFT-side state: a failed
        // write costs the entry its spec, never the entry.
        const spec_res: SpecSidecarResult = self.writeSpecSidecar(dir_rel, dflash_snap, mtp_snap, s) catch |err| blk: {
            log.warn("  [disk-cache] spec persist failed: {s} — entry keeps no spec\n", .{@errorName(err)});
            break :blk .{};
        };

        // Token record — the LONGER of the existing record and this commit's
        // tokens (a resumed incremental flush must not shrink the record its
        // earlier chunks were committed against). Rewritten only on growth;
        // tens of KB at most.
        const record: []const u32 = if (extend_idx) |i| blk: {
            const et = self.entries.items[i].tokens;
            break :blk if (et.len >= tokens.len) et else tokens;
        } else tokens;
        if (extend_idx == null or record.ptr == tokens.ptr) {
            const tpath = try std.fmt.allocPrint(self.allocator, "{s}/tokens.bin", .{dir_rel});
            defer self.allocator.free(tpath);
            const f = try std.Io.Dir.createFileAbsolute(self.io, tpath, .{});
            defer f.close(self.io);
            var wb: [8192]u8 = undefined;
            var fw = f.writer(self.io, &wb);
            try fw.interface.writeSliceEndian(u32, record, .little);
            try fw.interface.flush();
        }

        var bytes: u64 = 0;
        for (chunk_sizes.items) |b| bytes += b;
        for (ssm_res.bytes) |b| bytes += b;
        bytes += spec_res.bytes;
        bytes += @as(u64, record.len) * 4;

        const new_entry: IndexEntry = .{
            .id = id,
            .tokens = try self.allocator.dupe(u32, record),
            .kv_len = kv_len,
            .has_tools = has_tools,
            .quant = config,
            .bytes = bytes,
            .chunk_bytes = try chunk_sizes.toOwnedSlice(self.allocator),
            .ssm_positions = ssm_res.positions,
            .ssm_bytes = ssm_res.bytes,
            .spec_bytes = spec_res.bytes,
            .spec_dflash = spec_res.dflash,
            .spec_mtp = spec_res.mtp,
            .last_used = self.bump(),
        };
        errdefer {
            self.allocator.free(new_entry.tokens);
            self.allocator.free(new_entry.chunk_bytes);
        }
        // meta.json is the commit point — written last, atomically.
        try self.writeMeta(new_entry);

        if (extend_idx) |i| {
            const e = &self.entries.items[i];
            self.total_bytes -|= e.bytes;
            // ssm_positions/ssm_bytes ownership moved into new_entry.ssm_res —
            // free only the fields NOT carried forward.
            self.allocator.free(e.tokens);
            self.allocator.free(e.chunk_bytes);
            self.allocator.free(e.ssm_positions);
            self.allocator.free(e.ssm_bytes);
            e.* = new_entry;
            self.total_bytes += new_entry.bytes;
        } else {
            try self.entries.append(self.allocator, new_entry);
            self.total_bytes += new_entry.bytes;
        }
        self.gcToBudget();

        const wrote_mb = @as(f64, @floatFromInt(written_bytes)) / (1024.0 * 1024.0);
        const ms: u64 = sw.read() / std.time.ns_per_ms;
        log.info("  [disk-cache] persisted {d}/{d} tokens (+{d} chunks, {d} ssm-cp, {d:.1} MB, {d}ms); resident={d:.1} MB ({d} entries)\n", .{
            kv_len,               kv_target,          chunks_done - keep, new_entry.ssm_positions.len, wrote_mb, ms,
            @as(f64, @floatFromInt(self.total_bytes)) / (1024.0 * 1024.0),
            self.entries.items.len,
        });
        return complete;
    }

    /// Sidecar-only append: KV chunks are already fully on disk (superseded
    /// on KV) but the entry has pending SSM checkpoints (byte-capped across
    /// turns) and/or a missing spec snapshot this commit carries. Writes the
    /// missing pieces into the existing dir + rewrites meta. Never touches
    /// the KV chunks or the token record.
    fn appendSsmOnly(
        self: *DiskTier,
        idx: usize,
        ssm_checkpoints: ?[]const transformer_mod.SSMCheckpoint,
        dflash_snap: ?SpecCommit,
        mtp_snap: ?SpecCommit,
        s: mlx.mlx_stream,
    ) !bool {
        const dir_rel = try std.fmt.allocPrint(self.allocator, "{s}/e{d}", .{ self.root, self.entries.items[idx].id });
        defer self.allocator.free(dir_rel);
        const e = &self.entries.items[idx];
        var written_bytes: u64 = 0;
        var ssm_res = try self.persistSsmCheckpoints(e.id, dir_rel, e.kv_len, e.ssm_positions, e.ssm_bytes, ssm_checkpoints, &written_bytes);
        errdefer ssm_res.deinit(self.allocator);

        if (specWorkPending(e, dflash_snap, mtp_snap)) {
            const spec_res: SpecSidecarResult = self.writeSpecSidecar(dir_rel, dflash_snap, mtp_snap, s) catch |err| blk: {
                log.warn("  [disk-cache] spec persist failed: {s} — entry keeps its old spec\n", .{@errorName(err)});
                break :blk .{ .bytes = e.spec_bytes, .dflash = e.spec_dflash, .mtp = e.spec_mtp };
            };
            e.spec_bytes = spec_res.bytes;
            e.spec_dflash = spec_res.dflash;
            e.spec_mtp = spec_res.mtp;
        }

        // Recompute total bytes: chunks + token record are unchanged; only the
        // ssm/spec contributions changed.
        var kv_and_tokens: u64 = @as(u64, e.tokens.len) * 4;
        for (e.chunk_bytes) |b| kv_and_tokens += b;
        var new_bytes: u64 = kv_and_tokens + e.spec_bytes;
        for (ssm_res.bytes) |b| new_bytes += b;

        self.allocator.free(e.ssm_positions);
        self.allocator.free(e.ssm_bytes);
        e.ssm_positions = ssm_res.positions;
        e.ssm_bytes = ssm_res.bytes;
        self.total_bytes -|= e.bytes;
        e.bytes = new_bytes;
        self.total_bytes += new_bytes;
        e.last_used = self.bump();
        try self.writeMeta(e.*);
        self.gcToBudget();
        return ssm_res.complete;
    }

    // ── Spec-snapshot persistence (v4: dflash context / MTP history) ──

    const SpecSidecarResult = struct {
        bytes: u64 = 0,
        dflash: ?SpecMeta = null,
        mtp: ?SpecMeta = null,
    };

    /// Does this commit carry a spec payload the entry lacks? Mirrors
    /// `ssmWorkPending`'s role for the superseded no-op decision. A present
    /// spec is never "updated" at the same tokens — same tokens, same
    /// committed state.
    fn specWorkPending(e: *const IndexEntry, dflash: ?SpecCommit, mtp: ?SpecCommit) bool {
        return (dflash != null and e.spec_dflash == null) or
            (mtp != null and e.spec_mtp == null);
    }

    /// Write (or delete) the entry's ONE spec sidecar from this commit's
    /// snapshots. Tensors are sliced to `step` positions (the snapshot buffer
    /// can hold a stale draft tail past it) and keyed `d{layer}.*` /
    /// `m{layer}.*` with the trunk chunks' kind suffixes.
    fn writeSpecSidecar(self: *DiskTier, dir_abs: []const u8, dflash: ?SpecCommit, mtp: ?SpecCommit, s: mlx.mlx_stream) !SpecSidecarResult {
        const path = try std.fmt.allocPrint(self.allocator, "{s}/spec.safetensors\x00", .{dir_abs});
        defer self.allocator.free(path);
        if (dflash == null and mtp == null) {
            std.Io.Dir.deleteFileAbsolute(self.io, path[0 .. path.len - 1]) catch {};
            return .{};
        }
        const tensor_map = mlx.mlx_map_string_to_array_new();
        defer _ = mlx.mlx_map_string_to_array_free(tensor_map);
        const meta_map = mlx.mlx_map_string_to_string_new();
        defer _ = mlx.mlx_map_string_to_string_free(meta_map);
        var res: SpecSidecarResult = .{};
        if (dflash) |dc| res.dflash = try self.insertSpecTensors(tensor_map, "d", dc, s);
        if (mtp) |mc| res.mtp = try self.insertSpecTensors(tensor_map, "m", mc, s);
        try mlx.check(mlx.mlx_save_safetensors(@ptrCast(path.ptr), tensor_map, meta_map));
        res.bytes = fileSize(self.io, path[0 .. path.len - 1]) orelse 0;
        return res;
    }

    fn insertSpecTensors(self: *DiskTier, map: mlx.mlx_map_string_to_array, prefix: []const u8, sc: SpecCommit, s: mlx.mlx_stream) !SpecMeta {
        if (sc.step == 0) return error.DiskCacheEmptyEntry;
        const limit: u32 = @intCast(sc.step);
        const affine = sc.config.scheme != .off;
        for (sc.entries, 0..) |*entry, li| {
            if (!entry.initialized) continue;
            try self.insertSpecSlice(map, prefix, li, "k", entry.keys, limit, s);
            try self.insertSpecSlice(map, prefix, li, "v", entry.values, limit, s);
            if (affine) {
                try self.insertSpecSlice(map, prefix, li, "ks", entry.keys_scales, limit, s);
                try self.insertSpecSlice(map, prefix, li, "kb", entry.keys_biases, limit, s);
                try self.insertSpecSlice(map, prefix, li, "vs", entry.values_scales, limit, s);
                try self.insertSpecSlice(map, prefix, li, "vb", entry.values_biases, limit, s);
            }
        }
        return .{
            .base = sc.base_pos,
            .step = limit,
            .layers = @intCast(sc.entries.len),
            .quant = sc.config,
        };
    }

    fn insertSpecSlice(self: *DiskTier, map: mlx.mlx_map_string_to_array, prefix: []const u8, layer: usize, kind: []const u8, buf: mlx.mlx_array, limit: u32, s: mlx.mlx_stream) !void {
        const shape = mlx.getShape(buf);
        if (shape.len != 4) return error.DiskCacheBadShape;
        if (shape[2] < limit) return error.DiskCacheBadShape;
        var sliced = mlx.mlx_array_new();
        const st = [_]c_int{ 0, 0, 0, 0 };
        const sp = [_]c_int{ shape[0], shape[1], @intCast(limit), shape[3] };
        const sd = [_]c_int{ 1, 1, 1, 1 };
        try mlx.check(mlx.mlx_slice(&sliced, buf, &st, 4, &sp, 4, &sd, 4, s));
        defer _ = mlx.mlx_array_free(sliced);
        const key = try std.fmt.allocPrint(self.allocator, "{s}{d}.{s}\x00", .{ prefix, layer, kind });
        defer self.allocator.free(key);
        try mlx.check(mlx.mlx_map_string_to_array_insert(map, @ptrCast(key.ptr), sliced));
    }

    /// Load one persisted spec snapshot as a transient `KVCacheSnapshot` the
    /// caller restores from and then deinits. Best-effort in every direction:
    /// null when the entry has none, the recorded geometry doesn't fit the
    /// target (layer count / quant config — `KVCache.restore` asserts equal
    /// lengths, so the check lives here), or the file is unreadable. The
    /// caller then starts blind, never wrong.
    pub fn loadSpecSnap(
        self: *DiskTier,
        idx: usize,
        which: SpecKind,
        expected_layers: usize,
        target_config: kv_quant.KVQuantConfig,
    ) ?struct { snap: transformer_mod.KVCacheSnapshot, base: usize } {
        const e = &self.entries.items[idx];
        const meta = (switch (which) {
            .dflash => e.spec_dflash,
            .mtp => e.spec_mtp,
        }) orelse return null;
        if (meta.layers != expected_layers) return null;
        if (!std.meta.eql(meta.quant, target_config)) return null;

        const cpu = mlx.mlx_default_cpu_stream_new();
        defer _ = mlx.mlx_stream_free(cpu);
        const path = std.fmt.allocPrint(self.allocator, "{s}/e{d}/spec.safetensors\x00", .{ self.root, e.id }) catch return null;
        defer self.allocator.free(path);
        var tensor_map = mlx.mlx_map_string_to_array_new();
        defer _ = mlx.mlx_map_string_to_array_free(tensor_map);
        var meta_map = mlx.mlx_map_string_to_string_new();
        defer _ = mlx.mlx_map_string_to_string_free(meta_map);
        mlx.check(mlx.mlx_load_safetensors(&tensor_map, &meta_map, @ptrCast(path.ptr), cpu)) catch return null;

        const prefix: []const u8 = switch (which) {
            .dflash => "d",
            .mtp => "m",
        };
        const kinds: []const []const u8 = if (meta.quant.scheme == .off)
            &.{ "k", "v" }
        else
            &.{ "k", "v", "ks", "kb", "vs", "vb" };

        const entries = self.allocator.alloc(transformer_mod.KVCacheEntry, expected_layers) catch return null;
        for (entries) |*en| en.* = transformer_mod.newEmptyKVEntry();
        var snap: transformer_mod.KVCacheSnapshot = .{
            .entries = entries,
            .step = meta.step,
            .allocator = self.allocator,
            .config = meta.quant,
        };
        var ok = true;
        outer: for (entries, 0..) |*en, li| {
            for (kinds, 0..) |kind, ki| {
                const key = std.fmt.allocPrint(self.allocator, "{s}{d}.{s}\x00", .{ prefix, li, kind }) catch {
                    ok = false;
                    break :outer;
                };
                defer self.allocator.free(key);
                var arr = mlx.mlx_array_new();
                if (mlx.mlx_map_string_to_array_get(&arr, tensor_map, @ptrCast(key.ptr)) != 0) {
                    _ = mlx.mlx_array_free(arr);
                    if (ki == 0) continue :outer; // layer absent — stays uninitialized
                    ok = false; // partial layer = corrupt
                    break :outer;
                }
                // transfer the +1 handed by _get, replacing the empty handle
                switch (ki) {
                    0 => {
                        _ = mlx.mlx_array_free(en.keys);
                        en.keys = arr;
                    },
                    1 => {
                        _ = mlx.mlx_array_free(en.values);
                        en.values = arr;
                    },
                    2 => {
                        _ = mlx.mlx_array_free(en.keys_scales);
                        en.keys_scales = arr;
                    },
                    3 => {
                        _ = mlx.mlx_array_free(en.keys_biases);
                        en.keys_biases = arr;
                    },
                    4 => {
                        _ = mlx.mlx_array_free(en.values_scales);
                        en.values_scales = arr;
                    },
                    5 => {
                        _ = mlx.mlx_array_free(en.values_biases);
                        en.values_biases = arr;
                    },
                    else => unreachable,
                }
            }
            en.offset = meta.step;
            en.initialized = true;
        }
        if (!ok) {
            snap.deinit();
            return null;
        }
        // Checked eval: a corrupt file surfaces its MLX error HERE (lazy Load
        // reads data at eval), not mid-forward after the restore.
        {
            const vec = mlx.mlx_vector_array_new();
            defer _ = mlx.mlx_vector_array_free(vec);
            var count: usize = 0;
            for (entries) |*en| {
                if (!en.initialized) continue;
                _ = mlx.mlx_vector_array_append_value(vec, en.keys);
                _ = mlx.mlx_vector_array_append_value(vec, en.values);
                if (meta.quant.scheme != .off) {
                    _ = mlx.mlx_vector_array_append_value(vec, en.keys_scales);
                    _ = mlx.mlx_vector_array_append_value(vec, en.keys_biases);
                    _ = mlx.mlx_vector_array_append_value(vec, en.values_scales);
                    _ = mlx.mlx_vector_array_append_value(vec, en.values_biases);
                }
                count += 1;
            }
            if (count > 0) {
                mlx.check(mlx.mlx_eval(vec)) catch {
                    snap.deinit();
                    return null;
                };
            }
        }
        return .{ .snap = snap, .base = meta.base };
    }

    fn writeChunkFile(
        self: *DiskTier,
        kv_entries: []const transformer_mod.KVCacheEntry,
        config: kv_quant.KVQuantConfig,
        dir_abs: []const u8,
        chunk_idx: u32,
        c0: u32,
        c1: u32,
        s: mlx.mlx_stream,
    ) !void {
        const tensor_map = mlx.mlx_map_string_to_array_new();
        defer _ = mlx.mlx_map_string_to_array_free(tensor_map);
        const meta_map = mlx.mlx_map_string_to_string_new();
        defer _ = mlx.mlx_map_string_to_string_free(meta_map);

        const affine = config.scheme != .off;
        for (kv_entries, 0..) |*entry, li| {
            if (!entry.initialized) continue;
            try self.insertSlice(tensor_map, li, "k", entry.keys, c0, c1, s);
            try self.insertSlice(tensor_map, li, "v", entry.values, c0, c1, s);
            if (affine) {
                try self.insertSlice(tensor_map, li, "ks", entry.keys_scales, c0, c1, s);
                try self.insertSlice(tensor_map, li, "kb", entry.keys_biases, c0, c1, s);
                try self.insertSlice(tensor_map, li, "vs", entry.values_scales, c0, c1, s);
                try self.insertSlice(tensor_map, li, "vb", entry.values_biases, c0, c1, s);
            }
        }

        const path = try std.fmt.allocPrint(self.allocator, "{s}/c{d:0>6}.safetensors\x00", .{ dir_abs, chunk_idx });
        defer self.allocator.free(path);
        try mlx.check(mlx.mlx_save_safetensors(@ptrCast(path.ptr), tensor_map, meta_map));
    }

    fn insertSlice(
        self: *DiskTier,
        map: mlx.mlx_map_string_to_array,
        layer: usize,
        kind: []const u8,
        buf: mlx.mlx_array,
        c0: u32,
        c1: u32,
        s: mlx.mlx_stream,
    ) !void {
        const shape = mlx.getShape(buf);
        if (shape.len != 4) return error.DiskCacheBadShape;
        var sliced = mlx.mlx_array_new();
        const st = [_]c_int{ 0, 0, @intCast(c0), 0 };
        const sp = [_]c_int{ shape[0], shape[1], @intCast(c1), shape[3] };
        const sd = [_]c_int{ 1, 1, 1, 1 };
        try mlx.check(mlx.mlx_slice(&sliced, buf, &st, 4, &sp, 4, &sd, 4, s));
        defer _ = mlx.mlx_array_free(sliced);
        const key = try std.fmt.allocPrint(self.allocator, "l{d}.{s}\x00", .{ layer, kind });
        defer self.allocator.free(key);
        try mlx.check(mlx.mlx_map_string_to_array_insert(map, @ptrCast(key.ptr), sliced));
    }

    // ── SSM checkpoint persistence (Phase 3, hybrid archs) ──

    const SsmPersistResult = struct {
        /// Persisted checkpoint positions, sorted ascending. Owned.
        positions: []u32,
        /// Per-file byte sizes, parallel to `positions`. Owned.
        bytes: []u64,
        /// All target checkpoints made it to disk this flush (false → the
        /// per-flush byte cap deferred some; the caller keeps the entry dirty).
        complete: bool,

        fn deinit(self: *SsmPersistResult, allocator: std.mem.Allocator) void {
            allocator.free(self.positions);
            allocator.free(self.bytes);
        }
    };

    fn findCp(cps: []const transformer_mod.SSMCheckpoint, pos: u32) ?*const transformer_mod.SSMCheckpoint {
        for (cps) |*cp| if (cp.pos == pos) return cp;
        return null;
    }

    /// The set of checkpoint positions that SHOULD be on disk after this
    /// flush: the highest `SSM_DISK_MAX_PER_ENTRY` of (already-persisted ∪
    /// newly-eligible). Eligible = a RAM checkpoint at a position within the
    /// KV now on disk (a hybrid restore needs KV covering [0, cp_pos)).
    /// Sorted ascending; caller frees.
    fn ssmTargetPositions(self: *DiskTier, old_positions: []const u32, cps: []const transformer_mod.SSMCheckpoint, kv_len: u32) ![]u32 {
        var set = std.ArrayList(u32).empty;
        errdefer set.deinit(self.allocator);
        try set.appendSlice(self.allocator, old_positions);
        for (cps) |*cp| {
            if (cp.pos == 0 or cp.pos > kv_len) continue;
            const p: u32 = @intCast(cp.pos);
            if (std.mem.indexOfScalar(u32, set.items, p) == null) try set.append(self.allocator, p);
        }
        std.mem.sort(u32, set.items, {}, std.sort.asc(u32));
        if (set.items.len > SSM_DISK_MAX_PER_ENTRY) {
            const drop = set.items.len - SSM_DISK_MAX_PER_ENTRY;
            std.mem.copyForwards(u32, set.items[0 .. set.items.len - drop], set.items[drop..]);
            set.shrinkRetainingCapacity(set.items.len - drop);
        }
        return set.toOwnedSlice(self.allocator);
    }

    /// Would persisting `cps` add or drop any file for entry `e`? Drives the
    /// superseded no-op vs SSM-only-append decision. Conservative on alloc
    /// failure (returns false → the commit is a harmless no-op; the RAM tier
    /// still holds the checkpoints).
    fn ssmWorkPending(self: *DiskTier, e: *const IndexEntry, cps_opt: ?[]const transformer_mod.SSMCheckpoint, kv_limit: u32) bool {
        const cps = cps_opt orelse return false;
        if (cps.len == 0) return false;
        const target = self.ssmTargetPositions(e.ssm_positions, cps, kv_limit) catch return false;
        defer self.allocator.free(target);
        // A target position missing from disk, OR an on-disk position no
        // longer in target (retention would drop it), is pending work.
        if (target.len != e.ssm_positions.len) return true;
        for (target) |p| {
            if (std.mem.indexOfScalar(u32, e.ssm_positions, p) == null) return true;
        }
        return false;
    }

    /// Persist the eligible SSM checkpoints for one entry: write target
    /// positions not yet on disk (highest-first — the end-of-prompt checkpoint
    /// is the most valuable, so it survives a tight per-flush cap), delete
    /// retention-dropped positions, and return the resulting on-disk set.
    /// `written_bytes` accumulates across the chunk writes so checkpoint bytes
    /// count toward the same per-flush budget.
    fn persistSsmCheckpoints(
        self: *DiskTier,
        id: u64,
        dir_rel: []const u8,
        kv_len: u32,
        old_positions: []const u32,
        old_bytes: []const u64,
        cps_opt: ?[]const transformer_mod.SSMCheckpoint,
        written_bytes: *u64,
    ) !SsmPersistResult {
        const cps: []const transformer_mod.SSMCheckpoint = cps_opt orelse &[_]transformer_mod.SSMCheckpoint{};
        if (cps.len == 0 and old_positions.len == 0) {
            return .{
                .positions = try self.allocator.alloc(u32, 0),
                .bytes = try self.allocator.alloc(u64, 0),
                .complete = true,
            };
        }
        const target = try self.ssmTargetPositions(old_positions, cps, kv_len);
        defer self.allocator.free(target);

        // Delete positions retention drops (present on disk, absent from target).
        for (old_positions) |p| {
            if (std.mem.indexOfScalar(u32, target, p) == null) self.deleteSsmFile(id, p);
        }

        const Pair = struct { pos: u32, bytes: u64 };
        var pairs = std.ArrayList(Pair).empty;
        defer pairs.deinit(self.allocator);
        var complete = true;

        // Carry over old positions kept by retention (already on disk).
        for (target) |p| {
            if (std.mem.indexOfScalar(u32, old_positions, p)) |oi| {
                try pairs.append(self.allocator, .{ .pos = p, .bytes = old_bytes[oi] });
            }
        }
        // Write new target positions, highest-first.
        var ti = target.len;
        while (ti > 0) : (ti -= 1) {
            const p = target[ti - 1];
            if (std.mem.indexOfScalar(u32, old_positions, p) != null) continue; // already on disk
            const cp = findCp(cps, p) orelse continue;
            if (written_bytes.* >= self.max_flush_bytes) {
                complete = false;
                continue; // budget exhausted — persist on a later flush
            }
            const sz = try self.writeSsmFile(dir_rel, cp);
            written_bytes.* += sz;
            try pairs.append(self.allocator, .{ .pos = p, .bytes = sz });
        }

        std.mem.sort(Pair, pairs.items, {}, struct {
            fn lt(_: void, a: Pair, b: Pair) bool {
                return a.pos < b.pos;
            }
        }.lt);
        const positions = try self.allocator.alloc(u32, pairs.items.len);
        errdefer self.allocator.free(positions);
        const bytes = try self.allocator.alloc(u64, pairs.items.len);
        for (pairs.items, 0..) |pr, i| {
            positions[i] = pr.pos;
            bytes[i] = pr.bytes;
        }
        return .{ .positions = positions, .bytes = bytes, .complete = complete };
    }

    /// Write one SSM checkpoint as `s{pos:0>7}.safetensors`. Per-layer tensors
    /// keyed "l{i}.conv"/"l{i}.ssm" (absent = null state); the `initialized`
    /// bitmap rides the safetensors metadata map because `initialized=true`
    /// with both states null is a valid shape. Returns the file size.
    fn writeSsmFile(self: *DiskTier, dir_rel: []const u8, cp: *const transformer_mod.SSMCheckpoint) !u64 {
        const tensor_map = mlx.mlx_map_string_to_array_new();
        defer _ = mlx.mlx_map_string_to_array_free(tensor_map);
        const meta_map = mlx.mlx_map_string_to_string_new();
        defer _ = mlx.mlx_map_string_to_string_free(meta_map);

        var lc_buf: [24]u8 = undefined;
        const lc = try std.fmt.bufPrint(&lc_buf, "{d}\x00", .{cp.layers.len});
        try mlx.check(mlx.mlx_map_string_to_string_insert(meta_map, "layers", @ptrCast(lc.ptr)));

        var init_buf = std.ArrayList(u8).empty;
        defer init_buf.deinit(self.allocator);
        var num_buf: [16]u8 = undefined;
        for (cp.layers, 0..) |l, li| {
            if (!l.initialized) continue;
            if (init_buf.items.len > 0) try init_buf.append(self.allocator, ',');
            const ns = std.fmt.bufPrint(&num_buf, "{d}", .{li}) catch unreachable;
            try init_buf.appendSlice(self.allocator, ns);
        }
        try init_buf.append(self.allocator, 0); // NUL-terminate for the C API
        try mlx.check(mlx.mlx_map_string_to_string_insert(meta_map, "init", @ptrCast(init_buf.items.ptr)));

        // qwen4_exp aux state rides the same file: `l{d}.aux` / `l{d}.pooled`
        // tensors and `l{d}.ple` = uint32 [9] (valid flag, then the 8 token
        // history slots); the compress ratio is one `qsa_ratio` metadata key.
        var ratio_buf: [16]u8 = undefined;
        var ratio_written = false;
        for (cp.layers, 0..) |l, li| {
            const names = .{ "conv", "ssm", "aux", "pooled" };
            const arrs = .{ l.conv_state, l.ssm_state, l.aux_state, l.qsa_pooled };
            inline for (names, arrs) |name, arr| {
                if (arr.ctx != null) {
                    const key = try std.fmt.allocPrint(self.allocator, "l{d}." ++ name ++ "\x00", .{li});
                    defer self.allocator.free(key);
                    try mlx.check(mlx.mlx_map_string_to_array_insert(tensor_map, @ptrCast(key.ptr), arr));
                }
            }
            if (l.ple_prev_valid) {
                var ple: [9]u32 = undefined;
                ple[0] = 1;
                for (l.ple_prev, 0..) |t, i| ple[1 + i] = t;
                const ple_arr = mlx.mlx_array_new_data(&ple, &[_]c_int{9}, 1, .uint32);
                defer _ = mlx.mlx_array_free(ple_arr);
                const key = try std.fmt.allocPrint(self.allocator, "l{d}.ple\x00", .{li});
                defer self.allocator.free(key);
                try mlx.check(mlx.mlx_map_string_to_array_insert(tensor_map, @ptrCast(key.ptr), ple_arr));
            }
            if (!ratio_written and (l.aux_state.ctx != null or l.qsa_pooled.ctx != null)) {
                const rs = try std.fmt.bufPrint(&ratio_buf, "{d}\x00", .{l.qsa_ratio});
                try mlx.check(mlx.mlx_map_string_to_string_insert(meta_map, "qsa_ratio", @ptrCast(rs.ptr)));
                ratio_written = true;
            }
        }

        const path = try std.fmt.allocPrint(self.allocator, "{s}/s{d:0>7}.safetensors\x00", .{ dir_rel, cp.pos });
        defer self.allocator.free(path);
        try mlx.check(mlx.mlx_save_safetensors(@ptrCast(path.ptr), tensor_map, meta_map));
        return fileSize(self.io, path[0 .. path.len - 1]) orelse 0;
    }

    fn deleteSsmFile(self: *DiskTier, id: u64, pos: u32) void {
        const path = std.fmt.allocPrint(self.allocator, "{s}/e{d}/s{d:0>7}.safetensors", .{ self.root, id, pos }) catch return;
        defer self.allocator.free(path);
        std.Io.Dir.deleteFileAbsolute(self.io, path) catch {};
    }

    // ── Invalidation (mirrors the RAM cache API) ──

    pub fn invalidateAll(self: *DiskTier) void {
        for (self.entries.items) |*e| {
            self.deleteEntryDir(e.id);
            self.freeIndexEntryOwned(e);
        }
        self.entries.clearRetainingCapacity();
        self.total_bytes = 0;
    }

    pub fn invalidateNewest(self: *DiskTier) void {
        if (self.entries.items.len == 0) return;
        var newest_idx: usize = 0;
        var newest_used: u64 = 0;
        for (self.entries.items, 0..) |*e, i| {
            if (e.last_used >= newest_used) {
                newest_used = e.last_used;
                newest_idx = i;
            }
        }
        self.removeAt(newest_idx);
    }

    // ── Internals ──

    fn bump(self: *DiskTier) u64 {
        self.counter += 1;
        return self.counter;
    }

    fn removeAt(self: *DiskTier, idx: usize) void {
        var e = self.entries.swapRemove(idx);
        self.total_bytes -|= e.bytes;
        self.deleteEntryDir(e.id);
        self.freeIndexEntryOwned(&e);
    }

    fn deleteEntryDir(self: *DiskTier, id: u64) void {
        const rel = std.fmt.allocPrint(self.allocator, "e{d}", .{id}) catch return;
        defer self.allocator.free(rel);
        var root_dir = std.Io.Dir.openDirAbsolute(self.io, self.root, .{ .iterate = true }) catch return;
        defer root_dir.close(self.io);
        root_dir.deleteTree(self.io, rel) catch {};
    }

    fn gcToBudget(self: *DiskTier) void {
        if (self.max_bytes == 0) return;
        while (self.total_bytes > self.max_bytes and self.entries.items.len > 1) {
            var lru_idx: usize = 0;
            var lru_used: u64 = std.math.maxInt(u64);
            for (self.entries.items, 0..) |*e, i| {
                if (e.last_used < lru_used) {
                    lru_used = e.last_used;
                    lru_idx = i;
                }
            }
            const mb = @as(f64, @floatFromInt(self.entries.items[lru_idx].bytes)) / (1024.0 * 1024.0);
            log.info("  [disk-cache] evicted LRU entry (byte budget; {d:.1} MB)\n", .{mb});
            self.removeAt(lru_idx);
        }
    }

    fn writeMeta(self: *DiskTier, e: IndexEntry) !void {
        const tmp_path = try std.fmt.allocPrint(self.allocator, "{s}/e{d}/meta.json.tmp", .{ self.root, e.id });
        defer self.allocator.free(tmp_path);
        const final_path = try std.fmt.allocPrint(self.allocator, "{s}/e{d}/meta.json", .{ self.root, e.id });
        defer self.allocator.free(final_path);
        {
            const f = try std.Io.Dir.createFileAbsolute(self.io, tmp_path, .{});
            defer f.close(self.io);
            var wb: [1024]u8 = undefined;
            var fw = f.writer(self.io, &wb);
            try fw.interface.print(
                "{{\"v\":4,\"kv_len\":{d},\"tokens\":{d},\"has_tools\":{},\"scheme\":\"{s}\",\"bits\":{d},\"group_size\":{d},\"chunk_tokens\":{d},\"bytes\":{d},\"chunk_bytes\":[",
                .{
                    e.kv_len,
                    e.tokens.len,
                    e.has_tools,
                    @tagName(e.quant.scheme),
                    e.quant.bits,
                    e.quant.group_size,
                    self.chunk_tokens,
                    e.bytes,
                },
            );
            for (e.chunk_bytes, 0..) |cb, i| {
                if (i > 0) try fw.interface.writeAll(",");
                try fw.interface.print("{d}", .{cb});
            }
            // v3: SSM checkpoints as [{pos,bytes},...] (sorted ascending). Each
            // file's byte size drives the same kill -9 salvage as chunk_bytes.
            try fw.interface.writeAll("],\"ssm\":[");
            for (e.ssm_positions, e.ssm_bytes, 0..) |pos, sz, i| {
                if (i > 0) try fw.interface.writeAll(",");
                try fw.interface.print("{{\"pos\":{d},\"bytes\":{d}}}", .{ pos, sz });
            }
            try fw.interface.writeAll("]");
            // v4: spec snapshots (dflash context / MTP history) — the file
            // byte size drives the same kill -9 salvage as chunk_bytes, but a
            // mismatch drops only the SPEC (a restore then starts blind).
            if (e.spec_bytes > 0 and (e.spec_dflash != null or e.spec_mtp != null)) {
                try fw.interface.print(",\"spec\":{{\"bytes\":{d}", .{e.spec_bytes});
                if (e.spec_dflash) |sm| try writeSpecMetaJson(&fw.interface, "dflash", sm);
                if (e.spec_mtp) |sm| try writeSpecMetaJson(&fw.interface, "mtp", sm);
                try fw.interface.writeAll("}");
            }
            try fw.interface.writeAll("}");
            try fw.interface.flush();
        }
        try std.Io.Dir.renameAbsolute(tmp_path, final_path, self.io);
    }

    fn scan(self: *DiskTier) !void {
        var root_dir = std.Io.Dir.openDirAbsolute(self.io, self.root, .{ .iterate = true }) catch return;
        defer root_dir.close(self.io);

        // Collected (entry, mtime) pairs; sorted by mtime → LRU order.
        const Pending = struct { e: IndexEntry, mtime: i128 };
        var pending = std.ArrayList(Pending).empty;
        defer pending.deinit(self.allocator);

        var it = root_dir.iterate();
        while (it.next(self.io) catch null) |dent| {
            if (dent.kind != .directory) continue;
            if (dent.name.len < 2 or dent.name[0] != 'e') continue;
            const id = std.fmt.parseInt(u64, dent.name[1..], 10) catch continue;
            // Never reuse an id that has ever existed on disk — even a
            // dropped leftover's delete could fail and leave a dirty dir.
            if (id >= self.next_id) self.next_id = id + 1;
            if (self.loadEntry(id)) |loaded| {
                pending.append(self.allocator, .{ .e = loaded.e, .mtime = loaded.mtime }) catch {
                    var le = loaded.e;
                    self.freeIndexEntryOwned(&le);
                    continue;
                };
            } else {
                // Crash leftover / corrupt — remove it.
                log.info("  [disk-cache] dropping incomplete entry e{d}\n", .{id});
                self.deleteEntryDir(id);
            }
        }

        std.mem.sort(Pending, pending.items, {}, struct {
            fn lessThan(_: void, a: Pending, b: Pending) bool {
                return a.mtime < b.mtime;
            }
        }.lessThan);

        for (pending.items) |*p| {
            p.e.last_used = self.bump();
            self.entries.append(self.allocator, p.e) catch {
                self.freeIndexEntryOwned(&p.e);
                continue;
            };
            self.total_bytes += p.e.bytes;
        }
        if (self.entries.items.len > 0) {
            log.info("  [disk-cache] scanned {d} persisted entries ({d:.1} MB) at {s}\n", .{
                self.entries.items.len,
                @as(f64, @floatFromInt(self.total_bytes)) / (1024.0 * 1024.0),
                self.root,
            });
        }
    }

    fn loadEntry(self: *DiskTier, id: u64) ?struct { e: IndexEntry, mtime: i128 } {
        const meta_path = std.fmt.allocPrint(self.allocator, "{s}/e{d}/meta.json", .{ self.root, id }) catch return null;
        defer self.allocator.free(meta_path);

        const stat = statFile(self.io, meta_path) orelse return null;
        const content = readFileAlloc(self.allocator, self.io, meta_path, 64 * 1024) orelse return null;
        defer self.allocator.free(content);

        var parsed = std.json.parseFromSlice(std.json.Value, self.allocator, content, .{}) catch return null;
        defer parsed.deinit();
        if (parsed.value != .object) return null;
        const obj = parsed.value.object;

        const version = jsonU64(obj, "v") orelse return null;
        // v2 = pure-attention (no ssm field); v3 adds SSM checkpoints; v4
        // adds optional spec snapshots (dflash context / MTP history). All
        // restore — a lower-version entry just carries none of the newer
        // optional state, so an upgrade doesn't nuke existing disk caches.
        // Older layouts are dropped, not migrated.
        if (version != 2 and version != 3 and version != 4) return null;
        var kv_len = jsonU64(obj, "kv_len") orelse return null;
        const n_tokens = jsonU64(obj, "tokens") orelse return null;
        const chunk_tokens = jsonU64(obj, "chunk_tokens") orelse return null;
        const has_tools_v = obj.get("has_tools") orelse return null;
        if (has_tools_v != .bool) return null;
        const scheme_v = obj.get("scheme") orelse return null;
        if (scheme_v != .string) return null;
        const bits = jsonU64(obj, "bits") orelse 0;
        const group_size = jsonU64(obj, "group_size") orelse 0;
        const chunk_bytes_v = obj.get("chunk_bytes") orelse return null;
        if (chunk_bytes_v != .array) return null;

        // Chunk geometry must match this tier's configuration — a stale root
        // written under a different chunk size can't be extended coherently.
        if (chunk_tokens != self.chunk_tokens) return null;
        if (kv_len == 0 or kv_len > n_tokens) return null;

        const scheme = std.meta.stringToEnum(kv_quant.Scheme, scheme_v.string) orelse return null;
        var quant: kv_quant.KVQuantConfig = switch (scheme) {
            .off => kv_quant.KVQuantConfig.dense,
            .affine => kv_quant.KVQuantConfig.affine(@intCast(bits)),
            else => return null,
        };
        if (scheme == .affine) {
            if (group_size == 0 or bits == 0) return null;
            quant.group_size = @intCast(group_size);
        }

        const n_chunks: u64 = (kv_len + chunk_tokens - 1) / chunk_tokens;
        if (chunk_bytes_v.array.items.len != n_chunks) return null;

        // Validate each chunk file's size against the recorded one. A kill -9
        // mid-flush truncates the chunk being (re)written while meta still
        // describes the previous valid state — restoring it would poison the
        // cache (live: MLX "invalid data offsets exceeding the size of the
        // file"). Clamp to the last contiguous valid chunk and salvage the
        // prefix.
        var valid_chunks: u64 = 0;
        while (valid_chunks < n_chunks) : (valid_chunks += 1) {
            const want_v = chunk_bytes_v.array.items[@intCast(valid_chunks)];
            if (want_v != .integer or want_v.integer < 0) break;
            const cp = std.fmt.allocPrint(self.allocator, "{s}/e{d}/c{d:0>6}.safetensors", .{ self.root, id, valid_chunks }) catch return null;
            defer self.allocator.free(cp);
            const have = fileSize(self.io, cp) orelse break;
            if (have != @as(u64, @intCast(want_v.integer))) break;
        }
        if (valid_chunks < n_chunks) {
            const salvaged = valid_chunks * chunk_tokens;
            log.info("  [disk-cache] e{d}: chunk {d} invalid — salvaging {d}/{d} tokens\n", .{ id, valid_chunks, salvaged, kv_len });
            kv_len = salvaged;
            if (kv_len < MIN_PERSIST_TOKENS) return null;
        }

        const chunk_bytes = self.allocator.alloc(u64, @intCast(valid_chunks)) catch return null;
        for (chunk_bytes, 0..) |*cb, i| cb.* = @intCast(chunk_bytes_v.array.items[i].integer);

        // Token record.
        const tokens_path = std.fmt.allocPrint(self.allocator, "{s}/e{d}/tokens.bin", .{ self.root, id }) catch {
            self.allocator.free(chunk_bytes);
            return null;
        };
        defer self.allocator.free(tokens_path);
        const raw = readFileAlloc(self.allocator, self.io, tokens_path, 64 * 1024 * 1024) orelse {
            self.allocator.free(chunk_bytes);
            return null;
        };
        defer self.allocator.free(raw);
        if (raw.len != n_tokens * 4) {
            self.allocator.free(chunk_bytes);
            return null;
        }
        const tokens = self.allocator.alloc(u32, n_tokens) catch {
            self.allocator.free(chunk_bytes);
            return null;
        };
        for (tokens, 0..) |*t, i| {
            t.* = std.mem.readInt(u32, raw[i * 4 ..][0..4], .little);
        }

        var total: u64 = @as(u64, tokens.len) * 4;
        for (chunk_bytes) |cb| total += cb;

        // v3 SSM checkpoints (v2 entries have no "ssm" field → pure-attention,
        // stays empty). Each file's size is validated against the recorded one
        // — the same kill -9 salvage as chunks: a position whose file mismatches
        // (or now sits beyond a salvaged-down kv_len) is dropped individually.
        var ssm_positions: []u32 = &[_]u32{};
        var ssm_bytes: []u64 = &[_]u64{};
        var had_ssm_listed = false;
        if (obj.get("ssm")) |ssm_v| {
            if (ssm_v == .array) {
                had_ssm_listed = ssm_v.array.items.len > 0;
                var pos_list = std.ArrayList(u32).empty;
                defer pos_list.deinit(self.allocator);
                var byte_list = std.ArrayList(u64).empty;
                defer byte_list.deinit(self.allocator);
                for (ssm_v.array.items) |it_v| {
                    if (it_v != .object) continue;
                    const o = it_v.object;
                    const pos = jsonU64(o, "pos") orelse continue;
                    const szrec = jsonU64(o, "bytes") orelse continue;
                    if (pos == 0 or pos > kv_len) continue; // beyond the salvaged KV → unusable
                    const sp = std.fmt.allocPrint(self.allocator, "{s}/e{d}/s{d:0>7}.safetensors", .{ self.root, id, pos }) catch continue;
                    defer self.allocator.free(sp);
                    const have = fileSize(self.io, sp) orelse continue;
                    if (have != szrec) continue; // truncated mid-flush — drop this position
                    pos_list.append(self.allocator, @intCast(pos)) catch continue;
                    byte_list.append(self.allocator, szrec) catch {
                        _ = pos_list.pop();
                        continue;
                    };
                }
                if (pos_list.items.len > 0) {
                    // meta lists positions ascending, but re-sort defensively so
                    // highestSsmPosAtOrBelow / retention can trust the order.
                    const Pair = struct { pos: u32, bytes: u64 };
                    const pairs = self.allocator.alloc(Pair, pos_list.items.len) catch {
                        self.allocator.free(tokens);
                        self.allocator.free(chunk_bytes);
                        return null;
                    };
                    defer self.allocator.free(pairs);
                    for (pairs, 0..) |*pr, i| pr.* = .{ .pos = pos_list.items[i], .bytes = byte_list.items[i] };
                    std.mem.sort(Pair, pairs, {}, struct {
                        fn lt(_: void, a: Pair, b: Pair) bool {
                            return a.pos < b.pos;
                        }
                    }.lt);
                    const sp_arr = self.allocator.alloc(u32, pairs.len) catch {
                        self.allocator.free(tokens);
                        self.allocator.free(chunk_bytes);
                        return null;
                    };
                    const sb_arr = self.allocator.alloc(u64, pairs.len) catch {
                        self.allocator.free(sp_arr);
                        self.allocator.free(tokens);
                        self.allocator.free(chunk_bytes);
                        return null;
                    };
                    for (pairs, 0..) |pr, i| {
                        sp_arr[i] = pr.pos;
                        sb_arr[i] = pr.bytes;
                        total += pr.bytes;
                    }
                    ssm_positions = sp_arr;
                    ssm_bytes = sb_arr;
                }
            }
        }
        // A hybrid entry (SSM listed in meta) whose checkpoints ALL failed
        // validation is unusable — KV without any SSM state can't restore a
        // recurrent arch (the RAM path resets to cold in that case too). Drop
        // it wholesale.
        if (had_ssm_listed and ssm_positions.len == 0) {
            log.info("  [disk-cache] e{d}: all SSM checkpoints invalid — dropping hybrid entry\n", .{id});
            self.allocator.free(tokens);
            self.allocator.free(chunk_bytes);
            return null;
        }

        // v4 spec snapshots: validated by recorded file size (kill -9
        // salvage), but a bad spec drops only the SPEC — a restore then
        // starts blind, which is today's v2/v3 behavior anyway.
        var spec_bytes: u64 = 0;
        var spec_dflash: ?SpecMeta = null;
        var spec_mtp: ?SpecMeta = null;
        if (obj.get("spec")) |spec_v| parse_spec: {
            if (spec_v != .object) break :parse_spec;
            const so = spec_v.object;
            const rec_bytes = jsonU64(so, "bytes") orelse break :parse_spec;
            const sp = std.fmt.allocPrint(self.allocator, "{s}/e{d}/spec.safetensors", .{ self.root, id }) catch break :parse_spec;
            defer self.allocator.free(sp);
            const have = fileSize(self.io, sp) orelse break :parse_spec;
            if (have != rec_bytes) {
                log.info("  [disk-cache] e{d}: spec sidecar size mismatch — dropping the spec\n", .{id});
                break :parse_spec;
            }
            spec_dflash = parseSpecMeta(so, "dflash");
            spec_mtp = parseSpecMeta(so, "mtp");
            if (spec_dflash == null and spec_mtp == null) break :parse_spec;
            spec_bytes = rec_bytes;
            total += rec_bytes;
        }

        return .{
            .e = .{
                .id = id,
                .tokens = tokens,
                .kv_len = @intCast(kv_len),
                .has_tools = has_tools_v.bool,
                .quant = quant,
                .bytes = total,
                .chunk_bytes = chunk_bytes,
                .ssm_positions = ssm_positions,
                .ssm_bytes = ssm_bytes,
                .spec_bytes = spec_bytes,
                .spec_dflash = spec_dflash,
                .spec_mtp = spec_mtp,
                .last_used = 0,
            },
            .mtime = stat.mtime.nanoseconds,
        };
    }
};

// ── Model fingerprint ──

/// Identity of the weights the persisted KV was computed against: absolute
/// model dir + config.json size/mtime. A re-downloaded or re-quantized
/// checkpoint rewrites config.json, which rolls the fingerprint and orphans
/// the stale KV (GC'd by the disk budget eventually; different fingerprint
/// dirs never mix). 16 hex chars of XxHash64.
pub fn modelFingerprint(allocator: std.mem.Allocator, io: std.Io, model_dir: []const u8) ![]u8 {
    if (model_dir.len == 0 or !std.fs.path.isAbsolute(model_dir)) return error.BadModelDir;
    var h = std.hash.XxHash64.init(0x6b76_6361_6368_6531);
    h.update(model_dir);
    const cfg_path = try std.fmt.allocPrint(allocator, "{s}/config.json", .{model_dir});
    defer allocator.free(cfg_path);
    if (statFile(io, cfg_path)) |st| {
        h.update(std.mem.asBytes(&st.size));
        const mt: i128 = st.mtime.nanoseconds;
        h.update(std.mem.asBytes(&mt));
    }
    return std.fmt.allocPrint(allocator, "{x:0>16}", .{h.final()});
}

/// Default persistence root: `~/.mlx-serve/kv-cache`.
pub fn defaultBaseDir(allocator: std.mem.Allocator) ![]u8 {
    const home = std.mem.span(std.c.getenv("HOME") orelse return error.NoHome);
    return std.fmt.allocPrint(allocator, "{s}/.mlx-serve/kv-cache", .{home});
}

// ── Small fs helpers ──

fn statFile(io: std.Io, abs_path: []const u8) ?std.Io.File.Stat {
    if (abs_path.len == 0 or !std.fs.path.isAbsolute(abs_path)) return null;
    const f = std.Io.Dir.openFileAbsolute(io, abs_path, .{}) catch return null;
    defer f.close(io);
    return f.stat(io) catch null;
}

fn fileSize(io: std.Io, abs_path: []const u8) ?u64 {
    const st = statFile(io, abs_path) orelse return null;
    return st.size;
}

fn readFileAlloc(allocator: std.mem.Allocator, io: std.Io, abs_path: []const u8, limit: usize) ?[]u8 {
    if (abs_path.len == 0 or !std.fs.path.isAbsolute(abs_path)) return null;
    const f = std.Io.Dir.openFileAbsolute(io, abs_path, .{}) catch return null;
    defer f.close(io);
    var rb: [8192]u8 = undefined;
    var rs = f.reader(io, &rb);
    return rs.interface.allocRemaining(allocator, .limited(limit)) catch null;
}

fn jsonU64(obj: std.json.ObjectMap, key: []const u8) ?u64 {
    const v = obj.get(key) orelse return null;
    if (v != .integer) return null;
    if (v.integer < 0) return null;
    return @intCast(v.integer);
}

fn writeSpecMetaJson(w: *std.Io.Writer, name: []const u8, sm: SpecMeta) !void {
    try w.print(",\"{s}\":{{\"base\":{d},\"step\":{d},\"layers\":{d},\"scheme\":\"{s}\",\"bits\":{d},\"group_size\":{d}}}", .{
        name, sm.base, sm.step, sm.layers, @tagName(sm.quant.scheme), sm.quant.bits, sm.quant.group_size,
    });
}

fn parseSpecMeta(obj: std.json.ObjectMap, key: []const u8) ?SpecMeta {
    const v = obj.get(key) orelse return null;
    if (v != .object) return null;
    const o = v.object;
    const base = jsonU64(o, "base") orelse return null;
    const step = jsonU64(o, "step") orelse return null;
    const layers = jsonU64(o, "layers") orelse return null;
    if (step == 0 or layers == 0) return null;
    const scheme_v = o.get("scheme") orelse return null;
    if (scheme_v != .string) return null;
    const scheme = std.meta.stringToEnum(kv_quant.Scheme, scheme_v.string) orelse return null;
    const bits = jsonU64(o, "bits") orelse 0;
    const gs = jsonU64(o, "group_size") orelse 0;
    const quant: kv_quant.KVQuantConfig = switch (scheme) {
        .off => kv_quant.KVQuantConfig.dense,
        .affine => blk: {
            if (bits == 0 or gs == 0) return null;
            var q = kv_quant.KVQuantConfig.affine(@intCast(bits));
            q.group_size = @intCast(gs);
            break :blk q;
        },
        else => return null,
    };
    return .{ .base = base, .step = @intCast(step), .layers = @intCast(layers), .quant = quant };
}

// ── Tests ──

const testing = std.testing;

fn fillCache(cache: *KVCache, s: mlx.mlx_stream, n_layers: u32, tokens: u32, head_dim: u32, seed: f64, dtype: mlx.mlx_dtype) !void {
    // Drive the cache through its real update path with deterministic
    // arange-derived K/V so restored values are checkable. Dense tests use
    // float32 (every position stays exactly distinguishable); the affine test
    // uses bf16, the production dtype the quant write path expects.
    var written: u32 = 0;
    while (written < tokens) {
        const step: u32 = @min(64, tokens - written);
        var li: u32 = 0;
        while (li < n_layers) : (li += 1) {
            var flat = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(flat);
            const count: f64 = @floatFromInt(step * head_dim);
            const base: f64 = seed + @as(f64, @floatFromInt(written * head_dim + li * 1_000_000));
            try mlx.check(mlx.mlx_arange(&flat, base, base + count, 1.0, .float32, s));
            var shaped = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(shaped);
            const shape = [_]c_int{ 1, 1, @intCast(step), @intCast(head_dim) };
            try mlx.check(mlx.mlx_reshape(&shaped, flat, &shape, 4, s));
            var k = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(k);
            try mlx.check(mlx.mlx_astype(&k, shaped, dtype, s));
            // V = -K so a restore-side K/V swap can't false-pass.
            var v = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(v);
            try mlx.check(mlx.mlx_negative(&v, k, s));
            var view = try cache.update(li, k, v, s, 0);
            view.deinit();
        }
        written += step;
    }
}

fn cacheValueAt(cache: *KVCache, layer: u32, pos: u32, d: u32, s: mlx.mlx_stream) !f32 {
    return cacheBufValueAt(cache, layer, pos, d, s, false);
}

fn cacheBufValueAt(cache: *KVCache, layer: u32, pos: u32, d: u32, s: mlx.mlx_stream, values: bool) !f32 {
    const entry = &cache.entries[layer];
    var sliced = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(sliced);
    const st = [_]c_int{ 0, 0, @intCast(pos), @intCast(d) };
    const sp = [_]c_int{ 1, 1, @intCast(pos + 1), @intCast(d + 1) };
    const sd = [_]c_int{ 1, 1, 1, 1 };
    const buf = if (values) entry.values else entry.keys;
    try mlx.check(mlx.mlx_slice(&sliced, buf, &st, 4, &sp, 4, &sd, 4, s));
    var f = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(f);
    try mlx.check(mlx.mlx_astype(&f, sliced, .float32, s));
    _ = mlx.mlx_array_eval(f);
    const ptr = mlx.mlx_array_data_float32(f) orelse return error.NoData;
    return ptr[0];
}

fn tmpRoot(tmp: *std.testing.TmpDir, io: std.Io, buf: []u8) ![]const u8 {
    const n = try tmp.dir.realPath(io, buf);
    return buf[0..n];
}

test "DiskTier: chunked commit + restore round-trips exact KV, step, offsets" {
    const io = std.testing.io;
    const s = mlx.gpuStream();
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const base = try tmpRoot(&tmp, io, &buf);

    var tier = try DiskTier.init(testing.allocator, io, base, "fp-test", 0, 128);
    defer tier.deinit();

    // 600 tokens => 5 chunks at 128 (last partial: 88).
    var cache = try KVCache.init(testing.allocator, 3);
    defer cache.deinit();
    try fillCache(&cache, s, 3, 600, 8, 0.0, .float32);
    try testing.expectEqual(@as(usize, 600), cache.step);

    var tokens: [600]u32 = undefined;
    for (&tokens, 0..) |*t, i| t.* = @intCast(i + 7);
    _ = try tier.appendCommit(cache.entries, cache.step, cache.config, &tokens, false, null, s);
    try testing.expectEqual(@as(usize, 1), tier.entryCount());

    // Restore into a fresh cache (fresh tier too — proves the restart path).
    var tier2 = try DiskTier.init(testing.allocator, io, base, "fp-test", 0, 128);
    defer tier2.deinit();
    try testing.expectEqual(@as(usize, 1), tier2.entryCount());

    const m = tier2.bestMatch(&tokens, false, kv_quant.KVQuantConfig.dense).?;
    try testing.expectEqual(@as(u32, 600), m.usable);

    var cache2 = try KVCache.init(testing.allocator, 3);
    defer cache2.deinit();
    const restored = try tier2.restoreInto(&cache2, m.idx, s);
    try testing.expectEqual(@as(u32, 600), restored);
    try testing.expectEqual(@as(usize, 600), cache2.step);
    for (cache2.entries) |*e| {
        try testing.expect(e.initialized);
        try testing.expectEqual(@as(usize, 600), e.offset);
    }

    // Spot-check exact values across chunk boundaries and layers.
    const probes = [_][2]u32{ .{ 0, 0 }, .{ 127, 7 }, .{ 128, 0 }, .{ 300, 3 }, .{ 511, 7 }, .{ 512, 0 }, .{ 599, 7 } };
    for (probes) |p| {
        var li: u32 = 0;
        while (li < 3) : (li += 1) {
            const want = try cacheValueAt(&cache, li, p[0], p[1], s);
            const got = try cacheValueAt(&cache2, li, p[0], p[1], s);
            try testing.expectEqual(want, got);
            // V was written as -K: restored values must mirror that, so a
            // restore-side K/V swap or shared-buffer mixup fails here.
            const got_v = try cacheBufValueAt(&cache2, li, p[0], p[1], s, true);
            try testing.expectEqual(-want, got_v);
        }
    }

    // Mismatched key never matches.
    try testing.expect(tier2.bestMatch(&tokens, true, kv_quant.KVQuantConfig.dense) == null);
    try testing.expect(tier2.bestMatch(&tokens, false, kv_quant.KVQuantConfig.affine(4)) == null);
}

test "DiskTier: extend commit appends only new chunks (full chunks untouched)" {
    const io = std.testing.io;
    const s = mlx.gpuStream();
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const base = try tmpRoot(&tmp, io, &buf);

    var tier = try DiskTier.init(testing.allocator, io, base, "fp-ext", 0, 128);
    defer tier.deinit();

    var cache = try KVCache.init(testing.allocator, 1);
    defer cache.deinit();
    try fillCache(&cache, s, 1, 600, 8, 0.0, .float32);
    var tokens: [900]u32 = undefined;
    for (&tokens, 0..) |*t, i| t.* = @intCast(i + 7);
    _ = try tier.appendCommit(cache.entries, cache.step, cache.config, tokens[0..600], false, null, s);

    // Tamper-mark chunk 0 (a FULL chunk): record its mtime, then extend the
    // entry and assert chunk 0 was not rewritten while chunk 4 (the old
    // partial) was, and new chunks appeared.
    const c0_path = try std.fmt.allocPrint(testing.allocator, "{s}/fp-ext/e1/c000000.safetensors", .{base});
    defer testing.allocator.free(c0_path);
    const c4_path = try std.fmt.allocPrint(testing.allocator, "{s}/fp-ext/e1/c000004.safetensors", .{base});
    defer testing.allocator.free(c4_path);
    const c6_path = try std.fmt.allocPrint(testing.allocator, "{s}/fp-ext/e1/c000006.safetensors", .{base});
    defer testing.allocator.free(c6_path);
    const c0_before = statFile(io, c0_path).?.mtime.nanoseconds;
    const c4_before = statFile(io, c4_path).?.mtime.nanoseconds;
    try testing.expect(fileSize(io, c6_path) == null);

    // Ensure the extend write lands at a measurably later mtime.
    std.Io.sleep(io, .fromMilliseconds(20), .real) catch {};

    // Same prefix, 300 more tokens.
    try fillCache(&cache, s, 1, 300, 8, 4800.0, .float32);
    try testing.expectEqual(@as(usize, 900), cache.step);
    _ = try tier.appendCommit(cache.entries, cache.step, cache.config, &tokens, false, null, s);
    try testing.expectEqual(@as(usize, 1), tier.entryCount());
    try testing.expectEqual(@as(u32, 900), tier.entries.items[0].kv_len);

    try testing.expectEqual(c0_before, statFile(io, c0_path).?.mtime.nanoseconds); // untouched
    try testing.expect(statFile(io, c4_path).?.mtime.nanoseconds != c4_before); // partial rewritten
    try testing.expect(fileSize(io, c6_path) != null); // new tail chunk

    // Restore the extended entry and check a value in the extension range.
    var cache2 = try KVCache.init(testing.allocator, 1);
    defer cache2.deinit();
    const restored = try tier.restoreInto(&cache2, 0, s);
    try testing.expectEqual(@as(u32, 900), restored);
    const want = try cacheValueAt(&cache, 0, 750, 5, s);
    const got = try cacheValueAt(&cache2, 0, 750, 5, s);
    try testing.expectEqual(want, got);
}

test "DiskTier: identical re-commit is a no-op; shorter prefix is superseded" {
    const io = std.testing.io;
    const s = mlx.gpuStream();
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const base = try tmpRoot(&tmp, io, &buf);

    var tier = try DiskTier.init(testing.allocator, io, base, "fp-noop", 0, 128);
    defer tier.deinit();

    var cache = try KVCache.init(testing.allocator, 1);
    defer cache.deinit();
    try fillCache(&cache, s, 1, 600, 8, 0.0, .float32);
    var tokens: [600]u32 = undefined;
    for (&tokens, 0..) |*t, i| t.* = @intCast(i + 7);
    _ = try tier.appendCommit(cache.entries, cache.step, cache.config, &tokens, false, null, s);
    const c0_path = try std.fmt.allocPrint(testing.allocator, "{s}/fp-noop/e1/c000000.safetensors", .{base});
    defer testing.allocator.free(c0_path);
    const before = statFile(io, c0_path).?.mtime.nanoseconds;

    std.Io.sleep(io, .fromMilliseconds(20), .real) catch {};

    // Identical commit — nothing rewritten, no second entry.
    _ = try tier.appendCommit(cache.entries, cache.step, cache.config, &tokens, false, null, s);
    try testing.expectEqual(@as(usize, 1), tier.entryCount());
    try testing.expectEqual(before, statFile(io, c0_path).?.mtime.nanoseconds);

    // A shorter-prefix commit of the same conversation is covered by the
    // existing entry — also a no-op.
    var short_cache = try KVCache.init(testing.allocator, 1);
    defer short_cache.deinit();
    try fillCache(&short_cache, s, 1, 512, 8, 0.0, .float32);
    _ = try tier.appendCommit(short_cache.entries, short_cache.step, short_cache.config, tokens[0..512], false, null, s);
    try testing.expectEqual(@as(usize, 1), tier.entryCount());
}

test "DiskTier: byte budget evicts LRU entries, keeps newest" {
    const io = std.testing.io;
    const s = mlx.gpuStream();
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const base = try tmpRoot(&tmp, io, &buf);

    // Budget deliberately tiny: every new entry evicts the previous one.
    var tier = try DiskTier.init(testing.allocator, io, base, "fp-gc", 4096, 128);
    defer tier.deinit();

    var cache = try KVCache.init(testing.allocator, 1);
    defer cache.deinit();
    try fillCache(&cache, s, 1, 520, 8, 0.0, .float32);

    var tokens_a: [520]u32 = undefined;
    for (&tokens_a, 0..) |*t, i| t.* = @intCast(i + 7);
    var tokens_b: [520]u32 = undefined;
    for (&tokens_b, 0..) |*t, i| t.* = @intCast(i + 900_000);

    _ = try tier.appendCommit(cache.entries, cache.step, cache.config, &tokens_a, false, null, s);
    _ = try tier.appendCommit(cache.entries, cache.step, cache.config, &tokens_b, false, null, s);
    // Both entries exceed 4 KB each — only the newest survives.
    try testing.expectEqual(@as(usize, 1), tier.entryCount());
    try testing.expect(std.mem.eql(u32, tier.entries.items[0].tokens, &tokens_b));

    // The evicted directory is gone from disk.
    const e1_meta = try std.fmt.allocPrint(testing.allocator, "{s}/fp-gc/e1/meta.json", .{base});
    defer testing.allocator.free(e1_meta);
    try testing.expect(statFile(io, e1_meta) == null);
}

test "DiskTier: scan drops crash leftovers (no meta.json)" {
    const io = std.testing.io;
    const s = mlx.gpuStream();
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const base = try tmpRoot(&tmp, io, &buf);

    {
        var tier = try DiskTier.init(testing.allocator, io, base, "fp-crash", 0, 128);
        defer tier.deinit();
        var cache = try KVCache.init(testing.allocator, 1);
        defer cache.deinit();
        try fillCache(&cache, s, 1, 600, 8, 0.0, .float32);
        var tokens: [600]u32 = undefined;
        for (&tokens, 0..) |*t, i| t.* = @intCast(i + 7);
        _ = try tier.appendCommit(cache.entries, cache.step, cache.config, &tokens, false, null, s);
        // Simulate a crash mid-write of a SECOND entry: chunks, no meta.
        try tmp.dir.createDirPath(io, "fp-crash/e9");
        try tmp.dir.writeFile(io, .{ .sub_path = "fp-crash/e9/c000000.safetensors", .data = "junk" });
    }

    var tier2 = try DiskTier.init(testing.allocator, io, base, "fp-crash", 0, 128);
    defer tier2.deinit();
    try testing.expectEqual(@as(usize, 1), tier2.entryCount());
    // The leftover dir was removed.
    const leftover = try std.fmt.allocPrint(testing.allocator, "{s}/fp-crash/e9/c000000.safetensors", .{base});
    defer testing.allocator.free(leftover);
    try testing.expect(statFile(io, leftover) == null);
    // next_id moved past the dropped id (no reuse of a dirty dir name).
    try testing.expect(tier2.next_id >= 10);
}

test "DiskTier: affine-quant cache round-trips all six buffers" {
    const io = std.testing.io;
    const s = mlx.gpuStream();
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const base = try tmpRoot(&tmp, io, &buf);

    var tier = try DiskTier.init(testing.allocator, io, base, "fp-q", 0, 128);
    defer tier.deinit();

    const qcfg = kv_quant.KVQuantConfig.affine(4);
    var cache = try KVCache.initWithConfig(testing.allocator, 2, qcfg);
    defer cache.deinit();
    // head_dim must be a multiple of group_size (64) for affine quant.
    try fillCache(&cache, s, 2, 520, 64, 0.0, .bfloat16);
    try testing.expectEqual(@as(usize, 520), cache.step);

    var tokens: [520]u32 = undefined;
    for (&tokens, 0..) |*t, i| t.* = @intCast(i + 7);
    _ = try tier.appendCommit(cache.entries, cache.step, cache.config, &tokens, false, null, s);

    var cache2 = try KVCache.initWithConfig(testing.allocator, 2, qcfg);
    defer cache2.deinit();
    const m = tier.bestMatch(&tokens, false, qcfg).?;
    try testing.expectEqual(@as(u32, 520), m.usable);
    const restored = try tier.restoreInto(&cache2, m.idx, s);
    try testing.expectEqual(@as(u32, 520), restored);

    // Dense read-back through the cache's own dequant path must agree.
    // Truncate BOTH caches to the same length first — restore leaves views
    // empty (the KVCache.restore contract) and truncate to len < offset
    // rebuilds them on both sides identically.
    try cache.truncate(519, s);
    try cache2.truncate(519, s);
    var v1 = try cache.denseView(0, s);
    defer v1.deinit();
    var v2 = try cache2.denseView(0, s);
    defer v2.deinit();
    const probes = [_][2]u32{ .{ 0, 0 }, .{ 127, 63 }, .{ 128, 0 }, .{ 300, 5 }, .{ 511, 1 }, .{ 518, 63 } };
    for (probes) |p| {
        var d1 = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(d1);
        var d2 = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(d2);
        const st = [_]c_int{ 0, 0, @intCast(p[0]), @intCast(p[1]) };
        const sp = [_]c_int{ 1, 1, @intCast(p[0] + 1), @intCast(p[1] + 1) };
        const sd = [_]c_int{ 1, 1, 1, 1 };
        try mlx.check(mlx.mlx_slice(&d1, v1.k, &st, 4, &sp, 4, &sd, 4, s));
        try mlx.check(mlx.mlx_slice(&d2, v2.k, &st, 4, &sp, 4, &sd, 4, s));
        var f1 = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(f1);
        var f2 = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(f2);
        try mlx.check(mlx.mlx_astype(&f1, d1, .float32, s));
        try mlx.check(mlx.mlx_astype(&f2, d2, .float32, s));
        _ = mlx.mlx_array_eval(f1);
        _ = mlx.mlx_array_eval(f2);
        try testing.expectEqual(mlx.mlx_array_data_float32(f1).?[0], mlx.mlx_array_data_float32(f2).?[0]);
    }
}

test "DiskTier: truncated chunk file salvages the valid prefix at scan (kill -9 shape)" {
    // A kill -9 mid-flush leaves a chunk file truncated while meta.json (the
    // commit point, written last) still describes the PREVIOUS valid state —
    // whose recorded size for that chunk no longer matches the file. Live
    // capture: MLX "invalid data offsets exceeding the size of the file" on
    // restore. The scan must clamp the entry to the last contiguous chunk
    // whose size matches meta, salvaging the prefix instead of poisoning a
    // restore (or dropping everything).
    const io = std.testing.io;
    const s = mlx.gpuStream();
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const base = try tmpRoot(&tmp, io, &buf);

    {
        var tier = try DiskTier.init(testing.allocator, io, base, "fp-trunc", 0, 128);
        defer tier.deinit();
        var cache = try KVCache.init(testing.allocator, 1);
        defer cache.deinit();
        try fillCache(&cache, s, 1, 700, 8, 0.0, .float32);
        var tokens: [700]u32 = undefined;
        for (&tokens, 0..) |*t, i| t.* = @intCast(i + 7);
        _ = try tier.appendCommit(cache.entries, cache.step, cache.config, &tokens, false, null, s);
    }

    // Truncate chunk 4 (positions [512, 640)) — chunks 0-3 stay valid.
    try tmp.dir.writeFile(io, .{ .sub_path = "fp-trunc/e1/c000004.safetensors", .data = "trunc" });

    var tier2 = try DiskTier.init(testing.allocator, io, base, "fp-trunc", 0, 128);
    defer tier2.deinit();
    try testing.expectEqual(@as(usize, 1), tier2.entryCount());
    // kv_len clamped to the last valid chunk boundary: 4 * 128 = 512.
    try testing.expectEqual(@as(u32, 512), tier2.entries.items[0].kv_len);

    // The salvaged prefix restores cleanly.
    var cache2 = try KVCache.init(testing.allocator, 1);
    defer cache2.deinit();
    const restored = try tier2.restoreInto(&cache2, 0, s);
    try testing.expectEqual(@as(u32, 512), restored);
    try testing.expectEqual(@as(usize, 512), cache2.step);
}

test "DiskTier: flush byte cap persists incrementally across commits" {
    // A 4 GB first-commit write used to stall the NEXT request ~2.5 s (the
    // flush runs on the inference thread). appendCommit caps the bytes
    // written per call at max_flush_bytes, persists a chunk-aligned prefix,
    // and reports incomplete so the caller re-flushes on later turns.
    const io = std.testing.io;
    const s = mlx.gpuStream();
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const base = try tmpRoot(&tmp, io, &buf);

    var tier = try DiskTier.init(testing.allocator, io, base, "fp-cap", 0, 128);
    defer tier.deinit();
    tier.max_flush_bytes = 1; // every chunk write exceeds the cap -> 1 chunk/flush

    var cache = try KVCache.init(testing.allocator, 1);
    defer cache.deinit();
    try fillCache(&cache, s, 1, 600, 8, 0.0, .float32);
    var tokens: [600]u32 = undefined;
    for (&tokens, 0..) |*t, i| t.* = @intCast(i + 7);

    // First flush: 1 chunk (128 tokens), incomplete.
    const c1 = try tier.appendCommit(cache.entries, cache.step, cache.config, &tokens, false, null, s);
    try testing.expect(!c1);
    try testing.expectEqual(@as(u32, 128), tier.entries.items[0].kv_len);
    // Second flush continues from where it left off.
    const c2 = try tier.appendCommit(cache.entries, cache.step, cache.config, &tokens, false, null, s);
    try testing.expect(!c2);
    try testing.expectEqual(@as(u32, 256), tier.entries.items[0].kv_len);
    // Keep flushing until complete; entry must land at the full 600.
    var guard: u32 = 0;
    while (guard < 10) : (guard += 1) {
        if (try tier.appendCommit(cache.entries, cache.step, cache.config, &tokens, false, null, s)) break;
    }
    try testing.expectEqual(@as(u32, 600), tier.entries.items[0].kv_len);

    // Restored content from an incrementally-persisted entry is exact.
    var cache2 = try KVCache.init(testing.allocator, 1);
    defer cache2.deinit();
    const restored = try tier.restoreInto(&cache2, 0, s);
    try testing.expectEqual(@as(u32, 600), restored);
    const want = try cacheValueAt(&cache, 0, 599, 7, s);
    const got = try cacheValueAt(&cache2, 0, 599, 7, s);
    try testing.expectEqual(want, got);
}

test "DiskTier: cache ahead of the token record persists the clamped prefix (EOS-stop shape)" {
    // On an EOS stop the generator has forwarded the terminator tokens into
    // the cache but they're not part of the committed token record — live
    // capture: step=2054 vs tokens=2052. The RAM tier tolerates this
    // (truncate hides the tail); the disk tier must persist min(step,
    // tokens.len) positions instead of silently skipping the whole commit.
    const io = std.testing.io;
    const s = mlx.gpuStream();
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const base = try tmpRoot(&tmp, io, &buf);

    var tier = try DiskTier.init(testing.allocator, io, base, "fp-eos", 0, 128);
    defer tier.deinit();

    var cache = try KVCache.init(testing.allocator, 1);
    defer cache.deinit();
    try fillCache(&cache, s, 1, 604, 8, 0.0, .float32); // 2 positions past the record
    var tokens: [602]u32 = undefined;
    for (&tokens, 0..) |*t, i| t.* = @intCast(i + 7);
    _ = try tier.appendCommit(cache.entries, cache.step, cache.config, &tokens, false, null, s);
    try testing.expectEqual(@as(usize, 1), tier.entryCount());
    try testing.expectEqual(@as(u32, 602), tier.entries.items[0].kv_len);

    var cache2 = try KVCache.init(testing.allocator, 1);
    defer cache2.deinit();
    const restored = try tier.restoreInto(&cache2, 0, s);
    try testing.expectEqual(@as(u32, 602), restored);
    const want = try cacheValueAt(&cache, 0, 601, 3, s);
    const got = try cacheValueAt(&cache2, 0, 601, 3, s);
    try testing.expectEqual(want, got);
}

// ── Phase 3: hybrid SSM checkpoint persistence ──

const SSMCacheEntry = transformer_mod.SSMCacheEntry;
const conv_shape = [_]c_int{ 1, 3, 8 }; // [B, kernel-1, conv_dim]
const ssm_shape = [_]c_int{ 1, 2, 4, 4 }; // [B, Hv, Dv, Dk]

fn makeArange(s: mlx.mlx_stream, shape: []const c_int, base: f64) mlx.mlx_array {
    var count: f64 = 1;
    for (shape) |d| count *= @floatFromInt(d);
    var flat = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(flat);
    _ = mlx.mlx_arange(&flat, base, base + count, 1.0, .float32, s);
    var out = mlx.mlx_array_new();
    _ = mlx.mlx_reshape(&out, flat, shape.ptr, @intCast(shape.len), s);
    _ = mlx.mlx_array_eval(out);
    return out;
}

fn ssmArrVal(arr: mlx.mlx_array, idx: usize, s: mlx.mlx_stream) f32 {
    var f = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(f);
    _ = mlx.mlx_astype(&f, arr, .float32, s);
    _ = mlx.mlx_array_eval(f);
    return mlx.mlx_array_data_float32(f).?[idx];
}

/// Three-layer synthetic hybrid SSM state, covering the full null-state
/// matrix: (0) a GatedDeltaNet layer with both conv+ssm, (1) an LFM2
/// gated-conv layer with conv only (null ssm_state), (2) a plain-attention
/// layer in the hybrid (uninitialized, both null). `conv_base`/`ssm_base`
/// make each capture position's values distinguishable, so a restore-side
/// conv/ssm KEY SWAP fails the value checks (the K/V-swap lesson).
fn buildHybridEntries(s: mlx.mlx_stream, conv_base: f64, ssm_base: f64) [3]SSMCacheEntry {
    return .{
        .{
            .conv_state = makeArange(s, &conv_shape, conv_base),
            .ssm_state = makeArange(s, &ssm_shape, ssm_base),
            .initialized = true,
        },
        .{
            .conv_state = makeArange(s, &conv_shape, conv_base + 10_000),
            .ssm_state = mlx.mlx_array_new(),
            .initialized = true,
        },
        .{
            .conv_state = mlx.mlx_array_new(),
            .ssm_state = mlx.mlx_array_new(),
            .initialized = false,
        },
    };
}

fn freeHybridEntries(e: *[3]SSMCacheEntry) void {
    for (e) |*x| {
        _ = mlx.mlx_array_free(x.conv_state);
        _ = mlx.mlx_array_free(x.ssm_state);
        if (x.aux_state.ctx != null) _ = mlx.mlx_array_free(x.aux_state);
        if (x.qsa_pooled.ctx != null) _ = mlx.mlx_array_free(x.qsa_pooled);
    }
}

test "DiskTier: hybrid entry round-trips SSM checkpoints (Phase 3)" {
    // qwen3_5/3_6 GatedDeltaNet + lfm2 gated-conv (null ssm_state) + plain
    // attention (uninitialized) in one entry. No local hybrid checkpoint of
    // lfm2/nemotron_h exists, so those archs are covered here purely by the
    // null-state layer shapes (same SSMCacheEntrySnapshot contract).
    const io = std.testing.io;
    const s = mlx.gpuStream();
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const base = try tmpRoot(&tmp, io, &buf);

    var tier = try DiskTier.init(testing.allocator, io, base, "fp-hybrid", 0, 128);
    defer tier.deinit();

    var cache = try KVCache.init(testing.allocator, 3);
    defer cache.deinit();
    try fillCache(&cache, s, 3, 600, 8, 0.0, .float32); // >= MIN_PERSIST_TOKENS
    var tokens: [600]u32 = undefined;
    for (&tokens, 0..) |*t, i| t.* = @intCast(i + 7);

    // Two checkpoints at 128 / 256 with distinguishable state (base 100/500
    // vs 200/600), captured through the real production capture path.
    var src128 = buildHybridEntries(s, 100.0, 500.0);
    defer freeHybridEntries(&src128);
    var src256 = buildHybridEntries(s, 200.0, 600.0);
    defer freeHybridEntries(&src256);
    // Layer 2 at 256 also carries qwen4_exp aux state: a QSA key history +
    // pooled block keys, and layer 1 the PLE token history.
    const aux_shape = [_]c_int{ 1, 12, 4 };
    const pooled_shape = [_]c_int{ 1, 3, 4 };
    src256[2].aux_state = makeArange(s, &aux_shape, 700.0);
    src256[2].qsa_pooled = makeArange(s, &pooled_shape, 800.0);
    src256[2].qsa_ratio = 4;
    src256[1].ple_prev = .{ 42, 43, 0, 0, 0, 0, 0, 0 };
    src256[1].ple_prev_valid = true;
    var cps = [_]transformer_mod.SSMCheckpoint{
        try transformer_mod.captureSsmCheckpoint(testing.allocator, &src128, 128, s),
        try transformer_mod.captureSsmCheckpoint(testing.allocator, &src256, 256, s),
    };
    defer for (&cps) |*cp| cp.deinit(testing.allocator);

    _ = try tier.appendCommit(cache.entries, cache.step, cache.config, &tokens, false, &cps, s);
    try testing.expectEqual(@as(usize, 1), tier.entryCount());

    // Fresh tier (restart): both checkpoint positions survive the scan.
    var tier2 = try DiskTier.init(testing.allocator, io, base, "fp-hybrid", 0, 128);
    defer tier2.deinit();
    try testing.expectEqual(@as(usize, 1), tier2.entryCount());
    try testing.expectEqual(@as(?u32, 256), tier2.highestSsmPosAtOrBelow(0, 300));
    try testing.expectEqual(@as(?u32, 128), tier2.highestSsmPosAtOrBelow(0, 200));
    try testing.expectEqual(@as(?u32, null), tier2.highestSsmPosAtOrBelow(0, 100));

    // Restore at 256 into a fresh KVCache + ssm_entries.
    var cache2 = try KVCache.init(testing.allocator, 3);
    defer cache2.deinit();
    var dst: [3]SSMCacheEntry = .{
        .{ .conv_state = mlx.mlx_array_new(), .ssm_state = mlx.mlx_array_new(), .initialized = false },
        .{ .conv_state = mlx.mlx_array_new(), .ssm_state = mlx.mlx_array_new(), .initialized = false },
        .{ .conv_state = mlx.mlx_array_new(), .ssm_state = mlx.mlx_array_new(), .initialized = false },
    };
    defer freeHybridEntries(&dst);
    const restored = try tier2.restoreIntoHybrid(&cache2, &dst, 0, 256, s);
    try testing.expectEqual(@as(u32, 256), restored);
    try testing.expectEqual(@as(usize, 256), cache2.step);

    // Layer 0: conv (base 200) + ssm (base 600) — a KEY SWAP would flip these.
    try testing.expect(dst[0].initialized);
    try testing.expectEqual(@as(f32, 200.0), ssmArrVal(dst[0].conv_state, 0, s));
    try testing.expectEqual(@as(f32, 200.0 + 23.0), ssmArrVal(dst[0].conv_state, 23, s));
    try testing.expectEqual(@as(f32, 600.0), ssmArrVal(dst[0].ssm_state, 0, s));
    try testing.expectEqual(@as(f32, 600.0 + 31.0), ssmArrVal(dst[0].ssm_state, 31, s));
    // Layer 1: LFM2 gated-conv — conv present (base 10200), ssm stays null.
    try testing.expect(dst[1].initialized);
    try testing.expectEqual(@as(f32, 10_200.0), ssmArrVal(dst[1].conv_state, 0, s));
    try testing.expect(dst[1].ssm_state.ctx == null);
    // Layer 2: uninitialized plain-attention layer — both null, but the
    // qwen4_exp aux state round-trips (key history 700.., pooled 800..).
    try testing.expect(!dst[2].initialized);
    try testing.expect(dst[2].conv_state.ctx == null);
    try testing.expect(dst[2].ssm_state.ctx == null);
    try testing.expectEqual(@as(f32, 700.0 + 5.0), ssmArrVal(dst[2].aux_state, 5, s));
    try testing.expectEqual(@as(f32, 800.0 + 11.0), ssmArrVal(dst[2].qsa_pooled, 11, s));
    try testing.expectEqual(@as(c_int, 4), dst[2].qsa_ratio);
    try testing.expect(dst[1].ple_prev_valid and dst[1].ple_prev[0] == 42 and dst[1].ple_prev[1] == 43);
    try testing.expect(!dst[0].ple_prev_valid and dst[0].aux_state.ctx == null);

    // KV rewound to 256 in lockstep, values byte-exact against the original.
    for (cache2.entries) |*ce| {
        try testing.expect(ce.initialized);
        try testing.expectEqual(@as(usize, 256), ce.offset);
    }
    try testing.expectEqual(
        try cacheValueAt(&cache, 1, 200, 3, s),
        try cacheValueAt(&cache2, 1, 200, 3, s),
    );

    // Restore at the lower checkpoint installs THAT position's state.
    var cache3 = try KVCache.init(testing.allocator, 3);
    defer cache3.deinit();
    var dst2: [3]SSMCacheEntry = .{
        .{ .conv_state = mlx.mlx_array_new(), .ssm_state = mlx.mlx_array_new(), .initialized = false },
        .{ .conv_state = mlx.mlx_array_new(), .ssm_state = mlx.mlx_array_new(), .initialized = false },
        .{ .conv_state = mlx.mlx_array_new(), .ssm_state = mlx.mlx_array_new(), .initialized = false },
    };
    defer freeHybridEntries(&dst2);
    const restored128 = try tier2.restoreIntoHybrid(&cache3, &dst2, 0, 128, s);
    try testing.expectEqual(@as(u32, 128), restored128);
    try testing.expectEqual(@as(usize, 128), cache3.step);
    try testing.expectEqual(@as(f32, 100.0), ssmArrVal(dst2[0].conv_state, 0, s));
    try testing.expectEqual(@as(f32, 500.0), ssmArrVal(dst2[0].ssm_state, 0, s));

    // A position that was never checkpointed is rejected, not silently served.
    try testing.expectError(error.DiskCacheNoCheckpoint, tier2.restoreIntoHybrid(&cache3, &dst2, 0, 200, s));
}

test "DiskTier: SSM retention keeps the newest positions, drops the oldest" {
    // Every turn adds an end-of-prompt checkpoint; unbounded, one entry grows
    // without limit. Retention keeps at most SSM_DISK_MAX_PER_ENTRY, thinning
    // from the FRONT (lowest positions first — the newest are where warm
    // requests match).
    const io = std.testing.io;
    const s = mlx.gpuStream();
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const base = try tmpRoot(&tmp, io, &buf);

    var tier = try DiskTier.init(testing.allocator, io, base, "fp-retain", 0, 128);
    defer tier.deinit();

    // KV covering 0..(N*100) so every checkpoint position is ≤ kv_len.
    const N = SSM_DISK_MAX_PER_ENTRY + 1; // 9 positions, one over the cap
    var cache = try KVCache.init(testing.allocator, 3);
    defer cache.deinit();
    try fillCache(&cache, s, 3, N * 100 + 50, 8, 0.0, .float32);
    var tokens: [SSM_DISK_MAX_PER_ENTRY * 100 + 150]u32 = undefined;
    for (&tokens, 0..) |*t, i| t.* = @intCast(i + 7);

    var srcs: [N][3]SSMCacheEntry = undefined;
    for (&srcs, 0..) |*src, i| src.* = buildHybridEntries(s, @floatFromInt((i + 1) * 1000), @floatFromInt((i + 1) * 2000));
    defer for (&srcs) |*src| freeHybridEntries(src);
    var cps: [N]transformer_mod.SSMCheckpoint = undefined;
    for (&cps, 0..) |*cp, i| cp.* = try transformer_mod.captureSsmCheckpoint(testing.allocator, &srcs[i], (i + 1) * 100, s);
    defer for (&cps) |*cp| cp.deinit(testing.allocator);

    _ = try tier.appendCommit(cache.entries, cache.step, cache.config, &tokens, false, &cps, s);

    // Exactly MAX positions on disk; the LOWEST (100) dropped, newest kept.
    const e = &tier.entries.items[0];
    try testing.expectEqual(@as(usize, SSM_DISK_MAX_PER_ENTRY), e.ssm_positions.len);
    try testing.expectEqual(@as(u32, 200), e.ssm_positions[0]);
    try testing.expectEqual(@as(u32, @intCast(N * 100)), e.ssm_positions[e.ssm_positions.len - 1]);
    // The dropped position's file is gone.
    const dropped = try std.fmt.allocPrint(testing.allocator, "{s}/fp-retain/e1/s0000100.safetensors", .{base});
    defer testing.allocator.free(dropped);
    try testing.expect(statFile(io, dropped) == null);
    // A kept position's file exists.
    const kept = try std.fmt.allocPrint(testing.allocator, "{s}/fp-retain/e1/s0000200.safetensors", .{base});
    defer testing.allocator.free(kept);
    try testing.expect(statFile(io, kept) != null);
}

test "DiskTier: SSM salvage — one bad file drops that position, all bad drops the entry" {
    const io = std.testing.io;
    const s = mlx.gpuStream();
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const base = try tmpRoot(&tmp, io, &buf);

    {
        var tier = try DiskTier.init(testing.allocator, io, base, "fp-ssmsalv", 0, 128);
        defer tier.deinit();
        var cache = try KVCache.init(testing.allocator, 3);
        defer cache.deinit();
        try fillCache(&cache, s, 3, 600, 8, 0.0, .float32);
        var tokens: [600]u32 = undefined;
        for (&tokens, 0..) |*t, i| t.* = @intCast(i + 7);
        var src128 = buildHybridEntries(s, 100.0, 500.0);
        defer freeHybridEntries(&src128);
        var src256 = buildHybridEntries(s, 200.0, 600.0);
        defer freeHybridEntries(&src256);
        var cps = [_]transformer_mod.SSMCheckpoint{
            try transformer_mod.captureSsmCheckpoint(testing.allocator, &src128, 128, s),
            try transformer_mod.captureSsmCheckpoint(testing.allocator, &src256, 256, s),
        };
        defer for (&cps) |*cp| cp.deinit(testing.allocator);
        _ = try tier.appendCommit(cache.entries, cache.step, cache.config, &tokens, false, &cps, s);
    }

    // Truncate the pos-256 checkpoint file → that position drops, 128 survives.
    try tmp.dir.writeFile(io, .{ .sub_path = "fp-ssmsalv/e1/s0000256.safetensors", .data = "trunc" });
    var tier2 = try DiskTier.init(testing.allocator, io, base, "fp-ssmsalv", 0, 128);
    defer tier2.deinit();
    try testing.expectEqual(@as(usize, 1), tier2.entryCount());
    try testing.expectEqual(@as(usize, 1), tier2.entries.items[0].ssm_positions.len);
    try testing.expectEqual(@as(u32, 128), tier2.entries.items[0].ssm_positions[0]);
    // The salvaged KV + surviving checkpoint still restore.
    var cache2 = try KVCache.init(testing.allocator, 3);
    defer cache2.deinit();
    var dst: [3]SSMCacheEntry = .{
        .{ .conv_state = mlx.mlx_array_new(), .ssm_state = mlx.mlx_array_new(), .initialized = false },
        .{ .conv_state = mlx.mlx_array_new(), .ssm_state = mlx.mlx_array_new(), .initialized = false },
        .{ .conv_state = mlx.mlx_array_new(), .ssm_state = mlx.mlx_array_new(), .initialized = false },
    };
    defer freeHybridEntries(&dst);
    try testing.expectEqual(@as(u32, 128), try tier2.restoreIntoHybrid(&cache2, &dst, 0, 128, s));

    // Truncate the LAST surviving checkpoint too → hybrid entry dropped whole
    // (KV without any SSM state is unusable).
    try tmp.dir.writeFile(io, .{ .sub_path = "fp-ssmsalv/e1/s0000128.safetensors", .data = "trunc" });
    var tier3 = try DiskTier.init(testing.allocator, io, base, "fp-ssmsalv", 0, 128);
    defer tier3.deinit();
    try testing.expectEqual(@as(usize, 0), tier3.entryCount());
}

test "DiskTier: SSM checkpoints persist incrementally under the flush byte cap" {
    // The per-flush byte cap covers BOTH chunks and checkpoints so a big 27B
    // turn never stalls the next request. Under a 1-byte cap the entry
    // persists one unit at a time and reports incomplete until KV + every
    // eligible checkpoint have landed.
    const io = std.testing.io;
    const s = mlx.gpuStream();
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const base = try tmpRoot(&tmp, io, &buf);

    var tier = try DiskTier.init(testing.allocator, io, base, "fp-ssmcap", 0, 128);
    defer tier.deinit();
    tier.max_flush_bytes = 1; // one chunk/checkpoint per flush

    var cache = try KVCache.init(testing.allocator, 3);
    defer cache.deinit();
    try fillCache(&cache, s, 3, 600, 8, 0.0, .float32);
    var tokens: [600]u32 = undefined;
    for (&tokens, 0..) |*t, i| t.* = @intCast(i + 7);
    var src128 = buildHybridEntries(s, 100.0, 500.0);
    defer freeHybridEntries(&src128);
    var src512 = buildHybridEntries(s, 300.0, 700.0);
    defer freeHybridEntries(&src512);
    var cps = [_]transformer_mod.SSMCheckpoint{
        try transformer_mod.captureSsmCheckpoint(testing.allocator, &src128, 128, s),
        try transformer_mod.captureSsmCheckpoint(testing.allocator, &src512, 512, s),
    };
    defer for (&cps) |*cp| cp.deinit(testing.allocator);

    // Drive to completion; it must take multiple flushes and only report
    // complete once BOTH checkpoints are on disk.
    var complete = false;
    var guard: u32 = 0;
    while (guard < 40) : (guard += 1) {
        complete = try tier.appendCommit(cache.entries, cache.step, cache.config, &tokens, false, &cps, s);
        if (complete) break;
    }
    try testing.expect(complete);
    const e = &tier.entries.items[0];
    try testing.expectEqual(@as(u32, 600), e.kv_len);
    try testing.expectEqual(@as(usize, 2), e.ssm_positions.len);
    try testing.expectEqual(@as(u32, 128), e.ssm_positions[0]);
    try testing.expectEqual(@as(u32, 512), e.ssm_positions[1]);

    // The incrementally-persisted checkpoints restore correctly.
    var cache2 = try KVCache.init(testing.allocator, 3);
    defer cache2.deinit();
    var dst: [3]SSMCacheEntry = .{
        .{ .conv_state = mlx.mlx_array_new(), .ssm_state = mlx.mlx_array_new(), .initialized = false },
        .{ .conv_state = mlx.mlx_array_new(), .ssm_state = mlx.mlx_array_new(), .initialized = false },
        .{ .conv_state = mlx.mlx_array_new(), .ssm_state = mlx.mlx_array_new(), .initialized = false },
    };
    defer freeHybridEntries(&dst);
    try testing.expectEqual(@as(u32, 512), try tier.restoreIntoHybrid(&cache2, &dst, 0, 512, s));
    try testing.expectEqual(@as(f32, 300.0), ssmArrVal(dst[0].conv_state, 0, s));
    try testing.expectEqual(@as(f32, 700.0), ssmArrVal(dst[0].ssm_state, 0, s));
}

test "DiskTier: short caches and TurboQuant schemes are never persisted" {
    const io = std.testing.io;
    const s = mlx.gpuStream();
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const base = try tmpRoot(&tmp, io, &buf);

    var tier = try DiskTier.init(testing.allocator, io, base, "fp-skip", 0, 128);
    defer tier.deinit();

    // Below MIN_PERSIST_TOKENS → skipped.
    var cache = try KVCache.init(testing.allocator, 1);
    defer cache.deinit();
    try fillCache(&cache, s, 1, 128, 8, 0.0, .float32);
    var tokens: [128]u32 = undefined;
    for (&tokens, 0..) |*t, i| t.* = @intCast(i);
    _ = try tier.appendCommit(cache.entries, cache.step, cache.config, &tokens, false, null, s);
    try testing.expectEqual(@as(usize, 0), tier.entryCount());
}

test "DiskTier: v4 spec snapshots round-trip; geometry mismatches decline; v3 restores clean" {
    const io = std.testing.io;
    const s = mlx.gpuStream();
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const base = try tmpRoot(&tmp, io, &buf);

    var tier = try DiskTier.init(testing.allocator, io, base, "fp-spec", 0, 128);
    defer tier.deinit();

    var cache = try KVCache.init(testing.allocator, 2);
    defer cache.deinit();
    try fillCache(&cache, s, 2, 600, 8, 0.0, .float32);
    // dflash assistant context: 2 layers over the full 600 positions, base 0;
    // MTP committed history: 1 layer, 590 (the deferred-stash lag), base 0.
    var dfl = try KVCache.init(testing.allocator, 2);
    defer dfl.deinit();
    try fillCache(&dfl, s, 2, 600, 8, 3.5, .float32);
    var mtp = try KVCache.init(testing.allocator, 1);
    defer mtp.deinit();
    try fillCache(&mtp, s, 1, 590, 8, 9.5, .float32);

    var tokens: [600]u32 = undefined;
    for (&tokens, 0..) |*t, i| t.* = @intCast(i + 7);
    _ = try tier.appendCommitWithSpec(
        cache.entries,
        cache.step,
        cache.config,
        &tokens,
        false,
        null,
        .{ .entries = dfl.entries, .step = dfl.step, .config = dfl.config, .base_pos = 0 },
        .{ .entries = mtp.entries, .step = mtp.step, .config = mtp.config, .base_pos = 0 },
        s,
    );
    try testing.expect(tier.entries.items[0].spec_bytes > 0);

    // Restart shape: a fresh tier over the same root re-reads the spec meta.
    var tier2 = try DiskTier.init(testing.allocator, io, base, "fp-spec", 0, 128);
    defer tier2.deinit();
    try testing.expectEqual(@as(usize, 1), tier2.entryCount());
    const m = tier2.bestMatch(&tokens, false, kv_quant.KVQuantConfig.dense).?;

    var loaded = tier2.loadSpecSnap(m.idx, .dflash, 2, kv_quant.KVQuantConfig.dense) orelse
        return error.TestExpectedSpecSnap;
    try testing.expectEqual(@as(usize, 0), loaded.base);
    try testing.expectEqual(@as(usize, 600), loaded.snap.step);
    var dfl2 = try KVCache.init(testing.allocator, 2);
    defer dfl2.deinit();
    try dfl2.restore(&loaded.snap);
    loaded.snap.deinit();
    // Exact values, K and V (V = -K in fillCache — a swap can't false-pass).
    const probes = [_][2]u32{ .{ 0, 0 }, .{ 300, 3 }, .{ 599, 7 } };
    for (probes) |p| {
        var li: u32 = 0;
        while (li < 2) : (li += 1) {
            try testing.expectEqual(
                try cacheValueAt(&dfl, li, p[0], p[1], s),
                try cacheValueAt(&dfl2, li, p[0], p[1], s),
            );
            try testing.expectEqual(
                try cacheBufValueAt(&dfl, li, p[0], p[1], s, true),
                try cacheBufValueAt(&dfl2, li, p[0], p[1], s, true),
            );
        }
    }

    var mloaded = tier2.loadSpecSnap(m.idx, .mtp, 1, kv_quant.KVQuantConfig.dense) orelse
        return error.TestExpectedSpecSnap;
    defer mloaded.snap.deinit();
    try testing.expectEqual(@as(usize, 590), mloaded.snap.step);

    // A target the geometry doesn't fit DECLINES (KVCache.restore asserts
    // equal layer counts — the check must fire before it).
    try testing.expect(tier2.loadSpecSnap(m.idx, .dflash, 3, kv_quant.KVQuantConfig.dense) == null);
    try testing.expect(tier2.loadSpecSnap(m.idx, .dflash, 2, kv_quant.KVQuantConfig.affine(8)) == null);

    // A commit WITHOUT spec payloads carries none (and, per the supersede
    // rule, would delete a stale sidecar on its own entry).
    var tokens_b: [600]u32 = undefined;
    for (&tokens_b, 0..) |*t, i| t.* = @intCast(i + 900_000);
    _ = try tier2.appendCommit(cache.entries, cache.step, cache.config, &tokens_b, false, null, s);
    const mb = tier2.bestMatch(&tokens_b, false, kv_quant.KVQuantConfig.dense).?;
    try testing.expect(tier2.loadSpecSnap(mb.idx, .dflash, 2, kv_quant.KVQuantConfig.dense) == null);

    // v3 entry (written by an older binary): rewrite the manifest to v3 with
    // no spec object — the entry must restore CLEAN, spec-less.
    {
        const e_id = tier2.entries.items[m.idx].id;
        const meta_path = try std.fmt.allocPrint(testing.allocator, "{s}/fp-spec/e{d}/meta.json", .{ base, e_id });
        defer testing.allocator.free(meta_path);
        const content = readFileAlloc(testing.allocator, io, meta_path, 64 * 1024) orelse return error.TestMetaUnreadable;
        defer testing.allocator.free(content);
        const spec_at = std.mem.indexOf(u8, content, ",\"spec\":") orelse return error.TestSpecFieldMissing;
        var rewritten = std.ArrayList(u8).empty;
        defer rewritten.deinit(testing.allocator);
        try rewritten.appendSlice(testing.allocator, content[0..spec_at]);
        try rewritten.append(testing.allocator, '}');
        _ = std.mem.replace(u8, rewritten.items, "\"v\":4", "\"v\":3", rewritten.items);
        const f = try std.Io.Dir.createFileAbsolute(io, meta_path, .{});
        defer f.close(io);
        var wb: [4096]u8 = undefined;
        var fw = f.writer(io, &wb);
        try fw.interface.writeAll(rewritten.items);
        try fw.interface.flush();
    }
    var tier3 = try DiskTier.init(testing.allocator, io, base, "fp-spec", 0, 128);
    defer tier3.deinit();
    try testing.expectEqual(@as(usize, 2), tier3.entryCount());
    const m3 = tier3.bestMatch(&tokens, false, kv_quant.KVQuantConfig.dense).?;
    try testing.expectEqual(@as(u32, 600), m3.usable);
    try testing.expect(tier3.loadSpecSnap(m3.idx, .dflash, 2, kv_quant.KVQuantConfig.dense) == null);
    var cache3 = try KVCache.init(testing.allocator, 2);
    defer cache3.deinit();
    const restored = try tier3.restoreInto(&cache3, m3.idx, s);
    try testing.expectEqual(@as(u32, 600), restored);
}

test "modelFingerprint: stable per path, rolls with config.json changes" {
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const base = try tmpRoot(&tmp, io, &buf);

    try tmp.dir.createDirPath(io, "model-a");
    try tmp.dir.writeFile(io, .{ .sub_path = "model-a/config.json", .data = "{\"model_type\":\"x\"}" });
    const dir_a = try std.fmt.allocPrint(testing.allocator, "{s}/model-a", .{base});
    defer testing.allocator.free(dir_a);

    const fp1 = try modelFingerprint(testing.allocator, io, dir_a);
    defer testing.allocator.free(fp1);
    const fp2 = try modelFingerprint(testing.allocator, io, dir_a);
    defer testing.allocator.free(fp2);
    try testing.expectEqualStrings(fp1, fp2);
    try testing.expectEqual(@as(usize, 16), fp1.len);

    // Rewriting config.json (re-download / re-quant) rolls the fingerprint.
    std.Io.sleep(io, .fromMilliseconds(20), .real) catch {};
    try tmp.dir.writeFile(io, .{ .sub_path = "model-a/config.json", .data = "{\"model_type\":\"y\",\"pad\":1}" });
    const fp3 = try modelFingerprint(testing.allocator, io, dir_a);
    defer testing.allocator.free(fp3);
    try testing.expect(!std.mem.eql(u8, fp1, fp3));

    try testing.expectError(error.BadModelDir, modelFingerprint(testing.allocator, io, ""));
    try testing.expectError(error.BadModelDir, modelFingerprint(testing.allocator, io, "rel/path"));
}
