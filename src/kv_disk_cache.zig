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
const model = @import("model.zig");
const io_util = @import("io_util.zig");
const disk_writer = @import("kv_disk_writer.zig");
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

/// Budget floor for a decline-spill's flush: the client is already gone, so
/// the only cost is the synchronous write itself (~2-4 s at SSD speeds); the
/// tier's byte budget and LRU eviction are the real bounds. Sized to bank a
/// 122k-token hybrid candidate's KV in one spill.
pub const DECLINE_SPILL_FLUSH_FLOOR: u64 = 4 * 1024 * 1024 * 1024;

/// Max persisted SSM checkpoint positions per entry. Every turn adds an
/// end-of-prompt checkpoint; unbounded, one long session would accumulate GBs in a single
/// entry. Thinning is span-preserving (`transformer.positionDropIndex`): the lowest and the
/// newest position always survive. The count is a spacing decision priced against the tier
/// (qwen4_exp 383k entry: K=16 = 10.6 GB/entry, ~25k-token gaps; K=32 does not fit a 100 GB
/// tier). Raise it only alongside the tier's byte budget.
pub const SSM_DISK_MAX_PER_ENTRY: usize = 16;

/// The cap every arch outside the long-context gate keeps.
pub const SSM_DISK_MAX_PER_ENTRY_LEGACY: usize = 8;

/// SSD-first per-flush readback bound: only the device->host copy on the inference thread; the file write is off thread.
pub const SSD_FIRST_READBACK_BYTES: u64 = 2 * 1024 * 1024 * 1024;

/// The disk budget is derived from the volume, not only the operator's cap.
pub const DISK_RESERVE_CAP: u64 = 64 * 1024 * 1024 * 1024;
/// Below this there is no point storing anything.
pub const DISK_STORE_FLOOR: u64 = 1024 * 1024 * 1024;

/// Bytes this tier may occupy given the operator cap (0 = none) and the volume. Reserve =
/// min(64 GiB, 10% of the volume); under `DISK_STORE_FLOOR` = null ("do not store"), never 0.
pub fn diskBudgetFromFreeSpace(operator_cap: u64, free_bytes: u64, volume_bytes: u64) ?u64 {
    const reserve = @min(DISK_RESERVE_CAP, volume_bytes / 10);
    const avail = free_bytes -| reserve;
    const budget = if (operator_cap == 0) avail else @min(operator_cap, avail);
    if (budget < DISK_STORE_FLOOR) return null;
    return budget;
}

/// macOS `struct statfs`, leading fields only; the rest is slack. std has no binding for it.
const DarwinStatfs = extern struct {
    f_bsize: u32,
    f_iosize: i32,
    f_blocks: u64,
    f_bfree: u64,
    f_bavail: u64,
    f_files: u64,
    f_ffree: u64,
    tail: [4096]u8,
};
extern "c" fn statfs(path: [*:0]const u8, buf: *DarwinStatfs) c_int;

pub const VolumeSpace = struct { free: u64, total: u64 };

/// Free and total bytes of the volume holding `path`, or null when the query fails or returns
/// implausible numbers (the plausibility check is the ABI guard).
pub fn volumeSpace(path: []const u8) ?VolumeSpace {
    var buf: [std.fs.max_path_bytes + 1]u8 = undefined;
    if (path.len >= buf.len) return null;
    @memcpy(buf[0..path.len], path);
    buf[path.len] = 0;
    var st: DarwinStatfs = undefined;
    if (statfs(buf[0..path.len :0].ptr, &st) != 0) return null;
    const bsize: u64 = st.f_bsize;
    if (bsize < 512 or bsize > (1 << 20) or !std.math.isPowerOfTwo(bsize)) return null;
    if (st.f_blocks == 0 or st.f_bavail > st.f_blocks) return null;
    return .{
        .free = bsize *| st.f_bavail,
        .total = bsize *| st.f_blocks,
    };
}

/// How a `DiskTier` asks what the volume has left. Injectable: with the live probe hard-wired
/// every SSD-first test asserted a property of the tester's free disk space.
pub const SpaceProbeFn = *const fn (path: []const u8) ?VolumeSpace;

/// Test hook, armed through `DiskTier.armTestSpace`.
var test_space: ?VolumeSpace = null;
var test_qsa_overlay_mismatch = false;
var test_ssm_write_qsa_aux = false;

fn testSpaceProbe(path: []const u8) ?VolumeSpace {
    _ = path;
    return test_space;
}

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
    /// The first `inherited_chunks` chunk files are hard links into a donor entry's chunks;
    /// this entry never billed them. meta.json v6 `inherited_chunks`; 0 on older manifests.
    inherited_chunks: u32 = 0,
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
    /// A background write for this entry failed: never matched, never a donor, never reported
    /// complete; the next commit reclaims its directory. Not persisted (`scan` re-validates).
    poisoned: bool = false,
    qsa_history_bytes: u64 = 0,
    qsa_history_rows: u32 = 0,
    inherited_qsa: bool = false,
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
    /// v5, qwen4_exp MTP head only: the head's QSA aux half in the same sidecar. Null = a
    /// head-only miss at restore; the trunk entry is unaffected.
    head: ?SpecHeadMeta = null,
};

/// v5 head half of a `SpecMeta`: the scalars the head's position bookkeeping needs.
pub const SpecHeadMeta = struct {
    /// Absolute position of the head's key row 0 (`Qwen4Mtp.pos_base`).
    pos_base: i32,
    ratio: i32,
    pooled: bool,
    rows: i32 = 0,
    /// v8: head rows the sidecar carries a QSA leftover for, ascending — the trunk's
    /// checkpoint positions. Tensor `<prefix>h.lv<i>` holds the i-th. Inline: an IndexEntry
    /// is scanned and copied by value.
    marks: [transformer_mod.QSA_HEAD_MARKS_MAX]i32 = @splat(0),
    mark_count: u8 = 0,
};

/// What `appendCommitWithSpec` reads to persist one spec snapshot — the same
/// snapshot-shaped parts the trunk flush takes, plus the base position.
pub const SpecCommit = struct {
    entries: []const transformer_mod.KVCacheEntry,
    step: usize,
    config: kv_quant.KVQuantConfig,
    base_pos: usize,
    /// qwen4_exp MTP head: the QSA aux half, persisted alongside the KV.
    head_aux: ?*const transformer_mod.SSMCacheEntrySnapshot = null,
    head_pos_base: c_int = 0,
    head_marks: []const transformer_mod.QsaHeadMark = &.{},
};

/// What a commit actually achieved on disk. The old bool meant "nothing more to write", and
/// every silent skip returned it too, so `spillIdleEntries` dropped RAM copies that had no
/// disk copy at all. Only one of the three outcomes is a promise.
pub const PersistOutcome = enum {
    /// The tier holds the full prefix. The only value that may license discarding the RAM copy.
    persisted,
    /// Real bytes landed but the entry is not whole yet; the next commit resumes.
    partial,
    /// Nothing was written and nothing is promised.
    skipped,

    /// "Nothing more for the caller to write", the old bool's meaning.
    pub fn nothingPending(self: PersistOutcome) bool {
        return self != .partial;
    }
};

/// The KV extent one commit would persist, in tokens: clamped to the token record (the cache
/// runs 1-2 positions ahead on EOS turns), from the initialized layers' offset, not `step`
/// (0 on GDN hybrids). Shared by the commit and `holdsFullPrefix`.
pub fn persistTargetLen(
    kv_entries: []const transformer_mod.KVCacheEntry,
    step: usize,
    tokens_len: usize,
) usize {
    var max_off: usize = 0;
    for (kv_entries) |*entry| {
        if (entry.initialized and entry.offset > max_off) max_off = entry.offset;
    }
    return @min(@max(step, max_off), tokens_len);
}

pub const SpecKind = enum { dflash, mtp };

pub const Match = struct {
    idx: usize,
    /// Shared-prefix length clamped to kv_len — the positions a restore can
    /// actually rebuild.
    usable: u32,
};

/// `bestHybridMatch`'s result: the winning entry, its usable prefix, and the
/// restorable checkpoint position (≤ usable) that won it the race.
pub const HybridMatch = struct { idx: usize, usable: u32, cp: u32 };

fn nbytesOf(a: mlx.mlx_array) u64 {
    return @as(u64, mlx.mlx_array_size(a)) * @as(u64, mlx.mlx_array_itemsize(a));
}

/// Batched-eval meter for the staged serializer: one increment per chunk file, never per tensor.
pub var serialize_eval_count = std.atomic.Value(u64).init(0);

/// safetensors dtype spelling (`mlx::core::dtype_to_safetensor_str`); unknown dtypes refuse the staged write.
fn safetensorsDtypeName(d: mlx.mlx_dtype) ?[]const u8 {
    return switch (d) {
        .bool_ => "BOOL",
        .uint8 => "U8",
        .uint16 => "U16",
        .uint32 => "U32",
        .uint64 => "U64",
        .int8 => "I8",
        .int16 => "I16",
        .int32 => "I32",
        .int64 => "I64",
        .float16 => "F16",
        .float32 => "F32",
        .bfloat16 => "BF16",
        else => null,
    };
}

/// Raw contiguous bytes of an evaluated, contiguous array; null for a dtype with no accessor.
fn rawBytes(a: mlx.mlx_array) ?[*]const u8 {
    return switch (mlx.mlx_array_dtype(a)) {
        .bool_ => @ptrCast(mlx.mlx_array_data_bool(a) orelse return null),
        .uint8, .int8 => @ptrCast(mlx.mlx_array_data_uint8(a) orelse return null),
        .uint32, .int32 => @ptrCast(mlx.mlx_array_data_uint32(a) orelse return null),
        .float32 => @ptrCast(mlx.mlx_array_data_float32(a) orelse return null),
        .float16 => @ptrCast(mlx.mlx_array_data_float16(a) orelse return null),
        .bfloat16 => @ptrCast(mlx.mlx_array_data_bfloat16(a) orelse return null),
        else => null,
    };
}

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
    /// SSD-first mode (mirrored from `HotPrefixCache.ssd_first`): SSM checkpoints ride outside
    /// the per-flush byte budget, beside the chunk that closes their position.
    ssd_first: bool = false,
    /// Checkpoint-retention policy for the persisted position set, mirrored from
    /// `HotPrefixCache.cp_thin`. The default is the previous behaviour (keep the highest N).
    cp_thin: transformer_mod.ThinPolicy = .oldest,
    /// How many checkpoint positions one entry may keep on disk; the default is the previous cap.
    ssm_max_per_entry: usize = SSM_DISK_MAX_PER_ENTRY_LEGACY,
    /// SSD-first background writer (heap-allocated so the mutex survives `init`'s by-value
    /// return). Null = the synchronous `mlx_save_safetensors` path.
    writer: ?*disk_writer.Writer = null,
    /// The operator's `--prefix-cache-disk` value; `max_bytes` is re-derived from it before every store.
    operator_cap: u64 = 0,
    /// The free-space probe; tests arm a fixed answer with `armTestSpace`.
    space_probe: SpaceProbeFn = volumeSpace,
    /// The volume is under `DISK_STORE_FLOOR`: no new entry persists, existing ones stay restorable.
    store_declined: bool = false,
    /// `<base>` (the parent of `root`), for the root-wide sweep. Null when the dupe failed.
    base_dir: ?[]u8 = null,
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
            .operator_cap = max_bytes,
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
        self.base_dir = allocator.dupe(u8, base_dir) catch null;
        return self;
    }

    /// Test hook: answer every free-space probe with these numbers. Every SSD-first test must call this.
    pub fn armTestSpace(self: *DiskTier, free: u64, total: u64) void {
        test_space = .{ .free = free, .total = total };
        self.space_probe = testSpaceProbe;
    }

    /// Arm the background writer (SSD-first only). A spawn failure keeps the synchronous path.
    pub fn enableBackgroundWriter(self: *DiskTier) void {
        if (self.writer != null) return;
        const w = self.allocator.create(disk_writer.Writer) catch return;
        w.* = disk_writer.Writer.init(self.allocator, self.io);
        w.start() catch {
            self.allocator.destroy(w);
            log.warn("[disk-cache] background writer unavailable — writing synchronously\n", .{});
            return;
        };
        self.writer = w;
        // With the write off-thread the only inference-thread cost is the readback, and the
        // bound is no longer a correctness cliff (checkpoints ride outside it).
        self.max_flush_bytes = SSD_FIRST_READBACK_BYTES;
        log.info("[disk-cache] background writer armed (permit {d} MB, readback bound {d} MB/flush)\n", .{
            w.permit_bytes / (1024 * 1024),
            self.max_flush_bytes / (1024 * 1024),
        });
    }

    /// Wait for every staged file to land.
    pub fn drainWriter(self: *DiskTier) void {
        if (self.writer) |w| w.drain();
    }

    /// Host bytes staged for the writer and not yet written. Zero when the writer is not armed.
    pub fn stagedHostBytes(self: *DiskTier) u64 {
        const w = self.writer orelse return 0;
        return w.pendingBytes();
    }

    /// Background write failures so far.
    pub fn writeErrors(self: *DiskTier) u64 {
        const w = self.writer orelse return 0;
        return w.writeErrorCount();
    }

    /// Non-blocking: does entry `id` still have files staged or in flight? An unarmed tier
    /// answers false (its writes were synchronous).
    pub fn entryWritesPending(self: *DiskTier, id: u64) bool {
        const w = self.writer orelse return false;
        const pre = std.fmt.allocPrint(self.allocator, "{s}/e{d}/", .{ self.root, id }) catch return true;
        defer self.allocator.free(pre);
        return w.pendingPrefix(pre);
    }

    /// Attribute the writer's failed blobs to the entries that staged them and poison those
    /// entries. A counter alone misses the failure that lands between two spill passes.
    pub fn harvestWriteFailures(self: *DiskTier) usize {
        const w = self.writer orelse return 0;
        const fails = w.takeFailures();
        defer {
            for (fails) |f| self.allocator.free(f.path);
            self.allocator.free(fails);
        }
        var poisoned: usize = 0;
        for (fails) |f| {
            const id = self.entryIdFromPath(f.path) orelse {
                poisoned += self.poisonAll(f.err_name);
                continue;
            };
            for (self.entries.items) |*e| {
                if (e.id != id or e.poisoned) continue;
                self.poisonEntry(e, f.err_name);
                poisoned += 1;
            }
        }
        if (w.takeUnattributed()) poisoned += self.poisonAll("unrecorded");
        return poisoned;
    }

    /// The `e<id>` this absolute path belongs to, or null.
    fn entryIdFromPath(self: *const DiskTier, path: []const u8) ?u64 {
        if (!std.mem.startsWith(u8, path, self.root)) return null;
        var rest = path[self.root.len..];
        if (rest.len == 0 or rest[0] != '/') return null;
        rest = rest[1..];
        if (rest.len < 2 or rest[0] != 'e') return null;
        rest = rest[1..];
        const slash = std.mem.indexOfScalar(u8, rest, '/') orelse rest.len;
        return std.fmt.parseInt(u64, rest[0..slash], 10) catch null;
    }

    pub fn poisonId(self: *DiskTier, id: u64, err_name: []const u8) bool {
        for (self.entries.items) |*e| {
            if (e.id != id) continue;
            if (!e.poisoned) self.poisonEntry(e, err_name);
            return true;
        }
        return false;
    }

    /// Mark one entry dead; `chunk_bytes` is zeroed so every "is this whole?" reader answers no.
    fn poisonEntry(self: *DiskTier, e: *IndexEntry, err_name: []const u8) void {
        _ = self;
        e.poisoned = true;
        @memset(e.chunk_bytes, 0);
        log.warn("  [disk-cache] e{d} write failed ({s}) — entry invalidated\n", .{ e.id, err_name });
    }

    fn poisonAll(self: *DiskTier, err_name: []const u8) usize {
        var n: usize = 0;
        for (self.entries.items) |*e| {
            if (e.poisoned) continue;
            self.poisonEntry(e, err_name);
            n += 1;
        }
        return n;
    }

    /// Reclaim the directories of poisoned entries. Runs on the commit path, never on the spill's check.
    fn dropPoisonedEntries(self: *DiskTier) void {
        var i: usize = self.entries.items.len;
        while (i > 0) {
            i -= 1;
            if (!self.entries.items[i].poisoned) continue;
            log.info("  [disk-cache] dropping invalidated entry e{d}\n", .{self.entries.items[i].id});
            // `removeAt` swap-removes; the element moved into `i` was already checked.
            self.removeAt(i);
        }
    }

    /// Does the filesystem agree with the index about entry `id`? One stat per chunk; catches
    /// a byte that went missing with no failed write at all.
    pub fn entryWholeOnDisk(self: *DiskTier, id: u64) bool {
        for (self.entries.items) |*e| {
            if (e.id != id) continue;
            if (e.poisoned) return false;
            for (e.chunk_bytes, 0..) |want, i| {
                if (want == 0) return false;
                const cp = std.fmt.allocPrint(self.allocator, "{s}/e{d}/c{d:0>6}.safetensors", .{ self.root, id, i }) catch return false;
                defer self.allocator.free(cp);
                const st = statFile(self.io, cp) orelse return false;
                if (st.size != want) return false;
            }
            return true;
        }
        return false;
    }

    /// Is entry `id` poisoned, or gone?
    pub fn entryPoisoned(self: *const DiskTier, id: u64) bool {
        for (self.entries.items) |*e| {
            if (e.id == id) return e.poisoned;
        }
        return true;
    }

    /// Wait only for the files of entry `id`.
    fn drainEntry(self: *DiskTier, id: u64) void {
        const w = self.writer orelse return;
        const pre = std.fmt.allocPrint(self.allocator, "{s}/e{d}/", .{ self.root, id }) catch {
            w.drain();
            return;
        };
        defer self.allocator.free(pre);
        w.drainPrefix(pre);
    }

    pub fn deinit(self: *DiskTier) void {
        if (self.base_dir) |b| self.allocator.free(b);
        self.base_dir = null;
        if (self.writer) |w| {
            // Teardown must never block on a paused writer: lift the pause so the queue drains for real.
            w.setPaused(false);
            w.drain();
            w.deinit();
            self.allocator.destroy(w);
            self.writer = null;
        }
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

    /// Re-derive `max_bytes` from the volume. A failed probe keeps the operator cap; a budget
    /// under the store floor declines new stores without touching what is already persisted.
    fn refreshDiskBudget(self: *DiskTier) void {
        const vs = self.space_probe(self.root) orelse return;
        // Our own entries are already counted in `used`; add what the tier holds back.
        const budget = diskBudgetFromFreeSpace(self.operator_cap, vs.free +| self.total_bytes, vs.total);
        if (budget) |b| {
            self.store_declined = false;
            if (b != self.max_bytes) {
                self.max_bytes = b;
                self.gcToBudget();
            }
        } else if (!self.store_declined) {
            self.store_declined = true;
            // The number compared is free less the reserve, not free.
            log.warn("[disk-cache] {s}: {d} MB free less the {d} MB reserve (min 64 GiB, 10% of the volume) is below the {d} MB store floor — no NEW entries persist (already-persisted entries stay restorable)\n", .{
                self.root,
                vs.free >> 20,
                @min(DISK_RESERVE_CAP, vs.total / 10) >> 20,
                DISK_STORE_FLOOR >> 20,
            });
        }
    }

    /// Sweep other models' fingerprints under the same base: strays always, LRU once they
    /// collectively exceed one budget's worth. SSD-first only.
    pub fn sweepSiblings(self: *DiskTier) void {
        if (!self.ssd_first) return;
        const base = self.base_dir orelse return;
        self.refreshDiskBudget();
        sweepBase(self.allocator, self.io, base, self.root, self.max_bytes);
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
            if (e.poisoned) continue; // a failed write killed it: this is a MISS
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

    /// Hybrid targets rank disk entries by their RESTORABLE position — the
    /// highest SSM checkpoint at or below the usable prefix — not by the raw
    /// usable length (the RAM tier's #312 lesson: a longer raw match whose
    /// checkpoints sit past the divergence restores nothing, and must not
    /// shadow a shorter entry with a higher restorable position). Entries
    /// with no checkpoint at or below their usable prefix are skipped.
    pub fn bestHybridMatch(
        self: *const DiskTier,
        prompt_ids: []const u32,
        has_tools: bool,
        quant: kv_quant.KVQuantConfig,
        limit: u32,
    ) ?HybridMatch {
        var best: ?HybridMatch = null;
        for (self.entries.items, 0..) |*e, i| {
            if (e.poisoned) continue; // a failed write killed it: this is a MISS
            if (e.has_tools != has_tools) continue;
            if (!std.meta.eql(e.quant, quant)) continue;
            const max_shared = @min(e.tokens.len, prompt_ids.len);
            var shared: usize = 0;
            while (shared < max_shared and e.tokens[shared] == prompt_ids[shared]) shared += 1;
            const usable: u32 = @intCast(@min(@min(shared, e.kv_len), @as(usize, limit)));
            const cp = self.highestSsmPosAtOrBelow(i, usable) orelse continue;
            if (best == null or cp > best.?.cp) best = .{ .idx = i, .usable = usable, .cp = cp };
        }
        return best;
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
        // This entry's staged chunks must be on disk before the readback.
        self.drainEntry(self.entries.items[idx].id);
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
        self.drainEntry(self.entries.items[idx].id);
        const e = &self.entries.items[idx];
        if (cp_pos == 0 or cp_pos > e.kv_len) return error.DiskCacheNoCheckpoint;
        if (std.mem.indexOfScalar(u32, e.ssm_positions, cp_pos) == null) return error.DiskCacheNoCheckpoint;
        // Load the checkpoint FIRST (transient, no side effects on the live
        // state) so a corrupt/missing file fails before the cache is touched.
        var cp = try self.loadSsmFile(e.id, cp_pos, ssm_entries.len);
        defer cp.deinit(self.allocator);
        const want_rows = DiskTier.qsaHistoryRowsOf(&cp);
        // The overlay is owed unless the pooled bank is on THIS checkpoint: every checkpoint
        // carries its own raw leftover, and a mid-block one (backoff 30) is not the bank's home.
        const snap_has_pooled = transformer_mod.checkpointHasQsaPooled(&cp);
        var qsa_overlay: ?transformer_mod.SSMCheckpoint = null;
        defer if (qsa_overlay) |*q| q.deinit(self.allocator);
        if ((want_rows > 0 or e.qsa_history_rows > 0) and !snap_has_pooled) {
            var hist = self.loadQsaHistoryFile(e.id, ssm_entries.len) catch return error.DiskCacheQsaHistoryGap;
            if (!transformer_mod.checkpointQsaCoversPos(&cp, &hist, cp_pos)) {
                hist.deinit(self.allocator);
                return error.DiskCacheQsaHistoryGap;
            }
            qsa_overlay = hist;
        } else if (!snap_has_pooled and e.ssm_positions.len > 0) {
            const latest = e.ssm_positions[e.ssm_positions.len - 1];
            if (latest != cp_pos) {
                if (self.loadSsmFile(e.id, latest, ssm_entries.len)) |latest_cp_val| {
                    var latest_cp = latest_cp_val;
                    if (transformer_mod.checkpointHasQsaPooled(&latest_cp)) {
                        if (!transformer_mod.checkpointQsaCoversPos(&cp, &latest_cp, cp_pos)) {
                            latest_cp.deinit(self.allocator);
                            return error.DiskCacheQsaHistoryGap;
                        }
                        qsa_overlay = latest_cp;
                    } else {
                        latest_cp.deinit(self.allocator);
                    }
                } else |_| return error.DiskCacheQsaHistoryGap;
            }
        }
        if (test_qsa_overlay_mismatch) {
            if (qsa_overlay) |*q| q.deinit(self.allocator);
            qsa_overlay = .{ .pos = 0, .layers = try self.allocator.alloc(transformer_mod.SSMCacheEntrySnapshot, 0) };
        }
        try self.restoreKvInto(cache, e, cp_pos, s);
        errdefer cache.truncate(0, s) catch {};
        try transformer_mod.restoreSsmCheckpoint(ssm_entries, &cp);
        errdefer {
            for (ssm_entries) |*ent| {
                _ = mlx.mlx_array_free(ent.conv_state);
                _ = mlx.mlx_array_free(ent.ssm_state);
                ent.conv_state = mlx.mlx_array_new();
                ent.ssm_state = mlx.mlx_array_new();
                ent.initialized = false;
                transformer_mod.ssmFreeQsaState(ent);
                ent.ple_prev_valid = false;
            }
        }
        if (qsa_overlay) |*qsa_cp| {
            try transformer_mod.applyQsaHistoryAt(ssm_entries, qsa_cp, cp_pos, s);
        }
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
        if (fileSize(self.io, path[0 .. path.len - 1]) == null) return error.DiskCacheNoCheckpoint;
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
        var rows_c: [*:0]const u8 = undefined;
        if (mlx.mlx_map_string_to_string_get(&rows_c, meta_map, "qsa_rows") == 0) {
            const rows = std.fmt.parseInt(c_int, std.mem.span(rows_c), 10) catch return error.DiskCacheCorruptSsm;
            for (layers) |*l| l.qsa_rows = rows;
        } else {
            for (layers) |*l| {
                if (!transformer_mod.snapshotHasQsaHistory(l)) continue;
                if (l.aux_state.ctx != null) {
                    const sh = mlx.getShape(l.aux_state);
                    if (sh.len >= 2) l.qsa_rows = sh[1];
                }
            }
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
    ) !PersistOutcome {
        return self.appendCommitWithSpec(kv_entries, step, config, tokens, has_tools, ssm_checkpoints, null, null, s);
    }

    /// `appendCommit` with an explicit per-call flush bound (bytes): the loop stops after the
    /// first chunk that crosses it. The write-through hook passes one byte (one chunk per boundary).
    pub fn appendCommitBounded(
        self: *DiskTier,
        kv_entries: []const transformer_mod.KVCacheEntry,
        step: usize,
        config: kv_quant.KVQuantConfig,
        tokens: []const u32,
        has_tools: bool,
        ssm_checkpoints: ?[]const transformer_mod.SSMCheckpoint,
        s: mlx.mlx_stream,
        flush_bound: u64,
    ) !PersistOutcome {
        return self.appendCommitWithSpecBounded(kv_entries, step, config, tokens, has_tools, ssm_checkpoints, null, null, s, flush_bound);
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
    ) !PersistOutcome {
        return self.appendCommitWithSpecBounded(kv_entries, step, config, tokens, has_tools, ssm_checkpoints, dflash_snap, mtp_snap, s, self.max_flush_bytes);
    }

    fn appendCommitWithSpecBounded(
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
        flush_bound: u64,
    ) !PersistOutcome {
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
        // Anything the writer lost since the last commit is attributed before the index is read.
        _ = self.harvestWriteFailures();
        self.dropPoisonedEntries();

        const kv_target_u: usize = persistTargetLen(kv_entries, step, tokens.len);
        if (kv_target_u < MIN_PERSIST_TOKENS) return .skipped;
        const kv_target: u32 = @intCast(kv_target_u);
        switch (config.scheme) {
            .off, .affine => {},
            else => return .skipped, // TurboQuant rotation state doesn't survive restore
        }
        // Every initialized layer must cover the persisted range with B == 1
        // — anything else (mid-spec-decode state, batched cache) is not a
        // persistable snapshot.
        for (kv_entries) |*entry| {
            if (!entry.initialized) continue;
            if (entry.offset < kv_target_u) {
                log.debug("  [disk-cache] skip: layer offset {d} < kv_len {d}\n", .{ entry.offset, kv_target_u });
                return .skipped;
            }
            const shape = mlx.getShape(entry.keys);
            if (shape.len != 4 or shape[0] != 1) {
                log.debug("  [disk-cache] skip: non-B1 cache shape\n", .{});
                return .skipped;
            }
        }

        // Re-derive the budget from free space before every store.
        if (self.ssd_first) self.refreshDiskBudget();
        // The refresh gates THIS store, not merely the next one.
        if (self.store_declined) return .skipped;

        // Superseded check: an existing entry that already covers `tokens`
        // (same key, tokens is a prefix of its tokens, kv already >= ours)
        // makes this commit a no-op — UNLESS the entry is hybrid and still has
        // pending SSM checkpoints (byte-capped across turns), which take a
        // dedicated SSM-only append path (the KV chunks are all present, so
        // the extend machinery would pointlessly rewrite the tail chunk).
        var extend_idx: ?usize = null;
        var ssm_only_idx: ?usize = null;
        for (self.entries.items, 0..) |*e, i| {
            if (e.poisoned) continue; // dead: never superseded, never extended
            if (e.has_tools != has_tools) continue;
            if (!std.meta.eql(e.quant, config)) continue;
            if (e.tokens.len >= tokens.len) {
                if (std.mem.eql(u32, e.tokens[0..tokens.len], tokens)) {
                    if (e.kv_len >= kv_target) {
                        if (!self.ssmWorkPending(e, ssm_checkpoints, @intCast(e.tokens.len)) and
                            !specWorkPending(e, dflash_snap, mtp_snap))
                        {
                            // Superseded: the tier already holds this prefix in full.
                            e.last_used = self.bump();
                            return .persisted;
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
        //
        // A fresh entry may inherit its leading whole chunks from a resident entry that shares a
        // prefix, by hard link: a persisted entry's tokens are `prompt ++ generated`, so the next
        // turn diverges inside the generated span and used to rewrite every chunk. SSD-first only.
        const old_kv: u32 = if (extend_idx) |i| self.entries.items[i].kv_len else 0;
        const donor = if (extend_idx == null) self.chunkShareDonor(tokens, kv_target, has_tools, config) else null;
        var inherited: u32 = if (extend_idx) |i| self.entries.items[i].inherited_chunks else if (donor) |d| d.chunks else 0;
        var keep: u32 = if (extend_idx != null) old_kv / self.chunk_tokens else inherited;
        const n_chunks: u32 = @intCast((@as(u64, kv_target) + self.chunk_tokens - 1) / self.chunk_tokens);

        var chunk_sizes = std.ArrayList(u64).empty;
        errdefer chunk_sizes.deinit(self.allocator);
        if (extend_idx) |i| {
            const old_cb = self.entries.items[i].chunk_bytes;
            try chunk_sizes.appendSlice(self.allocator, old_cb[0..@min(keep, old_cb.len)]);
        } else if (donor) |d| {
            // Only the donor's landed chunks are linked (a contiguous prefix); may be zero.
            const linked = self.linkInheritedChunks(d, id, &chunk_sizes) catch 0;
            if (linked == 0) chunk_sizes.clearRetainingCapacity();
            inherited = linked;
            keep = linked;
        }
        var inherited_qsa = if (extend_idx) |i| self.entries.items[i].inherited_qsa else false;
        var inherited_qsa_rows: u32 = if (extend_idx) |i| self.entries.items[i].qsa_history_rows else 0;
        var inherited_qsa_bytes: u64 = if (extend_idx) |i| self.entries.items[i].qsa_history_bytes else 0;
        // A real check: a rewrite must never land on a link (the sync arm truncates in place).
        if (keep < inherited) {
            var root_dir = std.Io.Dir.openDirAbsolute(self.io, self.root, .{}) catch null;
            defer if (root_dir) |*rd| rd.close(self.io);
            if (root_dir) |rd| self.unlinkChunkRange(rd, id, keep, inherited);
            log.warn("  [disk-cache] chunk share: e{d} would rewrite an inherited chunk (keep {d} < inherited {d}) — writing from {d} instead\n", .{ id, keep, inherited, keep });
            inherited = keep;
            if (chunk_sizes.items.len > keep) chunk_sizes.shrinkRetainingCapacity(keep);
        }
        if (donor) |d| {
            if (inherited > 0 and self.linkInheritedQsa(d, id)) {
                inherited_qsa = true;
                inherited_qsa_rows = self.entries.items[d.idx].qsa_history_rows;
                inherited_qsa_bytes = self.entries.items[d.idx].qsa_history_bytes;
            }
        }

        // Phase 3 FIRST: checkpoints come off the TOP of the flush budget —
        // chunk-first budgeting starved them to ZERO on turns appending ≥
        // the cap in chunks (live 2026-09-07: cancel-salvage retries at
        // +16 chunks ≈ 544 MB vs a 512 MB cap left no checkpoint budget,
        // ever; every entry of the Sep-4 disk wave landed KV-only and
        // unrestorable). Their share is capped at half the budget so chunk
        // progress never stalls entirely. The eligibility bound is the
        // TARGET length, not the chunk progress: a checkpoint beyond the
        // chunks this flush reaches is still written (position-keyed,
        // immutable) and becomes restorable when a later flush extends
        // kv_len past it.
        const old_ssm_pos: []const u32 = if (extend_idx) |i| self.entries.items[i].ssm_positions else &[_]u32{};
        const old_ssm_bytes: []const u64 = if (extend_idx) |i| self.entries.items[i].ssm_bytes else &[_]u64{};
        var written_bytes: u64 = 0;
        var ssm_res = self.persistSsmCheckpoints(id, dir_rel, kv_target, old_ssm_pos, old_ssm_bytes, ssm_checkpoints, &written_bytes, s, self.max_flush_bytes / 2) catch |err| {
            chunk_sizes.deinit(self.allocator);
            return err;
        };
        errdefer ssm_res.deinit(self.allocator);

        // Chunks [0, keep) are full chunks already on disk (see above); the
        // rewrite runs from `keep` up to the REMAINDER of the flush bound —
        // the checkpoints above already came off the top of it.
        var chunk_i: u32 = keep;
        while (chunk_i < n_chunks) : (chunk_i += 1) {
            if (written_bytes >= flush_bound and chunk_i > keep) break;
            const c0: u32 = chunk_i * self.chunk_tokens;
            const c1: u32 = @intCast(@min(@as(u64, c0) + self.chunk_tokens, kv_target));
            const csize = try self.writeChunkFile(kv_entries, config, dir_rel, chunk_i, c0, c1, s);
            written_bytes += csize;
            try chunk_sizes.append(self.allocator, csize);
        }
        const chunks_done: u32 = chunk_i;
        const chunk_complete = chunks_done == n_chunks;
        const kv_len: u32 = if (chunk_complete) kv_target else chunks_done * self.chunk_tokens;
        if (kv_len <= old_kv and ssm_res.positions.len == old_ssm_pos.len) {
            // Cap so tight nothing new landed — nothing to commit (a
            // checkpoint-only write still counts as progress).
            chunk_sizes.deinit(self.allocator);
            return if (chunk_complete) .persisted else .partial;
        }

        const prefix_rows: u32 = if (donor) |d|
            @intCast(@min(commonPrefixLen(self.entries.items[d.idx].tokens, tokens), @as(usize, kv_len)))
        else
            kv_len;
        if (inherited_qsa) inherited_qsa_rows = @min(inherited_qsa_rows, prefix_rows);
        const qsa_res = self.persistQsaHistory(dir_rel, ssm_checkpoints, inherited_qsa, inherited_qsa_rows, prefix_rows, s) catch |err| {
            ssm_res.deinit(self.allocator);
            chunk_sizes.deinit(self.allocator);
            return err;
        };
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

        // `bytes` is what this entry created on disk: inherited (linked) chunks bill 0.
        var non_chunk: u64 = @as(u64, record.len) * 4 + spec_res.bytes;
        for (ssm_res.bytes) |b| non_chunk += b;
        if (!qsa_res.inherited) non_chunk += qsa_res.bytes;
        var bytes: u64 = non_chunk;
        for (chunk_sizes.items[@min(inherited, chunk_sizes.items.len)..]) |b| bytes += b;

        var new_entry: IndexEntry = .{
            .id = id,
            .tokens = try self.allocator.dupe(u32, record),
            .kv_len = kv_len,
            .has_tools = has_tools,
            .quant = config,
            .bytes = bytes,
            .chunk_bytes = try chunk_sizes.toOwnedSlice(self.allocator),
            .inherited_chunks = inherited,
            .ssm_positions = ssm_res.positions,
            .ssm_bytes = ssm_res.bytes,
            .spec_bytes = spec_res.bytes,
            .spec_dflash = spec_res.dflash,
            .spec_mtp = spec_res.mtp,
            .qsa_history_bytes = if (qsa_res.inherited) inherited_qsa_bytes else qsa_res.bytes,
            .qsa_history_rows = if (qsa_res.inherited) inherited_qsa_rows else qsa_res.rows,
            .inherited_qsa = qsa_res.inherited,
            .last_used = self.bump(),
        };
        errdefer {
            self.allocator.free(new_entry.tokens);
            self.allocator.free(new_entry.chunk_bytes);
        }

        if (extend_idx) |i| {
            const e = &self.entries.items[i];
            // An extension's delta is file-based; `e.bytes` after a `scan` may include chunks
            // the manifest lists as inherited (their donor died first), so it is carried forward.
            var delta: i64 = @as(i64, @intCast(non_chunk)) - @as(i64, @intCast(nonChunkBytes(e)));
            for (new_entry.chunk_bytes[@min(keep, new_entry.chunk_bytes.len)..]) |b| delta += @as(i64, @intCast(b));
            if (e.chunk_bytes.len > keep) delta -= @as(i64, @intCast(e.chunk_bytes[keep]));
            new_entry.bytes = clampAdd(e.bytes, delta);
            // meta.json is the commit point: written last, atomically.
            try self.writeMeta(new_entry);
            self.total_bytes = clampAdd(self.total_bytes, delta);
            // ssm_positions/ssm_bytes ownership moved into new_entry.ssm_res —
            // free only the fields NOT carried forward.
            self.allocator.free(e.tokens);
            self.allocator.free(e.chunk_bytes);
            self.allocator.free(e.ssm_positions);
            self.allocator.free(e.ssm_bytes);
            e.* = new_entry;
        } else {
            // meta.json is the commit point: written last, atomically.
            try self.writeMeta(new_entry);
            try self.entries.append(self.allocator, new_entry);
            self.total_bytes += new_entry.bytes;
        }
        self.gcToBudget();

        const wrote_mb = @as(f64, @floatFromInt(written_bytes)) / (1024.0 * 1024.0);
        const ms: u64 = sw.read() / std.time.ns_per_ms;
        log.info("  [disk-cache] persisted {d}/{d} tokens (+{d} chunks, {d} ssm-cp, {d:.1} MB, {d}ms); resident={d:.1} MB ({d} entries)\n", .{
            kv_len,                                                        kv_target,              chunks_done - keep, new_entry.ssm_positions.len, wrote_mb, ms,
            @as(f64, @floatFromInt(self.total_bytes)) / (1024.0 * 1024.0), self.entries.items.len,
        });
        // The one completion marker: every chunk and every wanted checkpoint is staged. A
        // bounded flush leaves it out and `disk_dirty` set. A write that already failed for this
        // entry makes `.persisted` a promise nothing can keep.
        var whole = complete;
        if (whole) {
            _ = self.harvestWriteFailures();
            if (self.entryPoisoned(id)) whole = false;
        }
        if (whole) {
            log.info("  [disk-cache] e{d} complete on disk: {d} tokens, {d} chunks, {d} ssm-cp\n", .{ id, kv_len, new_entry.chunk_bytes.len, new_entry.ssm_positions.len });
        }
        return if (whole) .persisted else .partial;
    }

    /// Does the index agree that this tier holds a complete, restorable copy of `tokens`? Two
    /// bars before a RAM entry may be discarded: the token record covers `tokens` at the same
    /// key with `kv_len` reaching the persist target, and `chunk_bytes` has one non-zero entry
    /// per implied chunk.
    pub fn holdsFullPrefix(
        self: *const DiskTier,
        kv_entries: []const transformer_mod.KVCacheEntry,
        step: usize,
        tokens: []const u32,
        has_tools: bool,
        config: kv_quant.KVQuantConfig,
    ) bool {
        return self.fullPrefixEntryId(kv_entries, step, tokens, has_tools, config) != null;
    }

    /// `holdsFullPrefix`, returning the entry's id so the caller can ask `entryWritesPending`.
    pub fn fullPrefixEntryId(
        self: *const DiskTier,
        kv_entries: []const transformer_mod.KVCacheEntry,
        step: usize,
        tokens: []const u32,
        has_tools: bool,
        config: kv_quant.KVQuantConfig,
    ) ?u64 {
        const target = persistTargetLen(kv_entries, step, tokens.len);
        if (target == 0) return null;
        for (self.entries.items) |*e| {
            if (e.poisoned) continue;
            if (e.has_tools != has_tools) continue;
            if (!std.meta.eql(e.quant, config)) continue;
            if (e.tokens.len < tokens.len) continue;
            if (!std.mem.eql(u32, e.tokens[0..tokens.len], tokens)) continue;
            if (e.kv_len < target) continue;
            const want: usize = (@as(usize, e.kv_len) + self.chunk_tokens - 1) / self.chunk_tokens;
            if (e.chunk_bytes.len < want) continue;
            var whole = true;
            for (e.chunk_bytes[0..want]) |b| {
                if (b == 0) whole = false;
            }
            if (whole) return e.id;
        }
        return null;
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
    ) !PersistOutcome {
        const dir_rel = try std.fmt.allocPrint(self.allocator, "{s}/e{d}", .{ self.root, self.entries.items[idx].id });
        defer self.allocator.free(dir_rel);
        const e = &self.entries.items[idx];
        var written_bytes: u64 = 0;
        // Write-ahead bound: the token record, not the flushed kv_len — a
        // checkpoint beyond the current chunks is position-keyed and becomes
        // restorable when a later extend raises kv_len past it.
        var ssm_res = try self.persistSsmCheckpoints(e.id, dir_rel, @intCast(e.tokens.len), e.ssm_positions, e.ssm_bytes, ssm_checkpoints, &written_bytes, s, self.max_flush_bytes);
        errdefer ssm_res.deinit(self.allocator);
        const qsa_res = try self.persistQsaHistory(dir_rel, ssm_checkpoints, e.inherited_qsa, e.qsa_history_rows, e.kv_len, s);

        // Captured before the sidecar write overwrites it: this path bills a delta.
        const old_spec_bytes: u64 = e.spec_bytes;
        const old_qsa_billed: u64 = if (e.inherited_qsa) 0 else e.qsa_history_bytes;
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
        // Delta-based like the extend path; both non-chunk terms are in the delta.
        var delta: i64 = @as(i64, @intCast(e.spec_bytes)) - @as(i64, @intCast(old_spec_bytes));
        for (ssm_res.bytes) |b| delta += @as(i64, @intCast(b));
        for (e.ssm_bytes) |b| delta -= @as(i64, @intCast(b));
        const new_qsa_billed: u64 = if (qsa_res.inherited) 0 else qsa_res.bytes;
        delta += @as(i64, @intCast(new_qsa_billed)) - @as(i64, @intCast(old_qsa_billed));

        self.allocator.free(e.ssm_positions);
        self.allocator.free(e.ssm_bytes);
        e.ssm_positions = ssm_res.positions;
        e.ssm_bytes = ssm_res.bytes;
        if (qsa_res.inherited) {
            e.inherited_qsa = true;
        } else if (qsa_res.bytes > 0) {
            e.qsa_history_bytes = qsa_res.bytes;
            e.qsa_history_rows = qsa_res.rows;
            e.inherited_qsa = false;
        }
        e.bytes = clampAdd(e.bytes, delta);
        self.total_bytes = clampAdd(self.total_bytes, delta);
        e.last_used = self.bump();
        try self.writeMeta(e.*);
        self.gcToBudget();
        return if (ssm_res.complete) .persisted else .partial;
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
            (mtp != null and e.spec_mtp == null) or
            // v5 upgrade: an entry persisted with a KV-only MTP snap gains the head's QSA half.
            (mtp != null and mtp.?.head_aux != null and
                (e.spec_mtp == null or e.spec_mtp.?.head == null));
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
        // v5 head half: the QSA raw-key history and pooled bank go in whole.
        var head: ?SpecHeadMeta = null;
        if (sc.head_aux) |a| {
            // A history that is not exactly `limit` rows is not persistable: drop the head half, keep the KV.
            const hist: c_int = if (a.qsa_rows > 0) a.qsa_rows else blk: {
                if (a.aux_state.ctx == null) break :blk 0;
                const sh = mlx.getShape(a.aux_state);
                break :blk if (sh.len >= 2) sh[1] else 0;
            };
            const rows_ok = hist == @as(c_int, @intCast(limit)) and
                (a.aux_state.ctx != null or a.qsa_pooled.ctx != null);
            if (rows_ok) {
                try self.insertSpecArray(map, prefix, "h.aux", a.aux_state);
                if (a.qsa_pooled.ctx != null) try self.insertSpecArray(map, prefix, "h.pooled", a.qsa_pooled);
                var hm: SpecHeadMeta = .{
                    .pos_base = sc.head_pos_base,
                    .ratio = a.qsa_ratio,
                    .pooled = a.qsa_pooled.ctx != null,
                    .rows = hist,
                };
                // v8: the leftovers a warm clamp reads, one tensor each (at most ratio-1 rows).
                for (sc.head_marks) |mk| {
                    if (mk.rows.ctx == null or mk.pos > hist) continue;
                    if (hm.mark_count == transformer_mod.QSA_HEAD_MARKS_MAX) break;
                    var kind_buf: [16]u8 = undefined;
                    const kind = try std.fmt.bufPrint(&kind_buf, "h.lv{d}", .{hm.mark_count});
                    try self.insertSpecArray(map, prefix, kind, mk.rows);
                    hm.marks[hm.mark_count] = mk.pos;
                    hm.mark_count += 1;
                }
                head = hm;
            }
        }
        return .{
            .base = sc.base_pos,
            .step = limit,
            .layers = @intCast(sc.entries.len),
            .quant = sc.config,
            .head = head,
        };
    }

    fn insertSpecArray(self: *DiskTier, map: mlx.mlx_map_string_to_array, prefix: []const u8, kind: []const u8, arr: mlx.mlx_array) !void {
        const key = try std.fmt.allocPrint(self.allocator, "{s}{s}\x00", .{ prefix, kind });
        defer self.allocator.free(key);
        try mlx.check(mlx.mlx_map_string_to_array_insert(map, @ptrCast(key.ptr), arr));
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
    ) ?struct {
        snap: transformer_mod.KVCacheSnapshot,
        base: usize,
        /// v5 qwen4_exp head half; null on every other entry.
        head_aux: ?transformer_mod.SSMCacheEntrySnapshot = null,
        head_pos_base: c_int = 0,
        /// v8: the head's checkpoint-position leftovers. Empty on a v5..v7 sidecar, which
        /// clamps as it does today (a mid-block warm clamp declines, head starts blind).
        head_marks: transformer_mod.QsaHeadMarkSet = .{},
    } {
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
        // v5 head half, best-effort: a pre-v5 sidecar returns none and the head declines the adoption.
        var head_aux: ?transformer_mod.SSMCacheEntrySnapshot = null;
        var head_pos_base: c_int = 0;
        var head_marks: transformer_mod.QsaHeadMarkSet = .{};
        if (meta.head) |hm| head: {
            const aux = getSpecArray(tensor_map, self.allocator, prefix, "h.aux") orelse break :head;
            var snap_aux: transformer_mod.SSMCacheEntrySnapshot = .{
                .conv_state = mlx.mlx_array_new(),
                .ssm_state = mlx.mlx_array_new(),
                .initialized = true,
                .aux_state = aux,
                .qsa_ratio = hm.ratio,
            };
            if (hm.pooled) {
                snap_aux.qsa_pooled = getSpecArray(tensor_map, self.allocator, prefix, "h.pooled") orelse {
                    transformer_mod.ssmSnapshotDeinit(&snap_aux);
                    break :head;
                };
            }
            const vec = mlx.mlx_vector_array_new();
            defer _ = mlx.mlx_vector_array_free(vec);
            _ = mlx.mlx_vector_array_append_value(vec, snap_aux.aux_state);
            if (snap_aux.qsa_pooled.ctx != null) _ = mlx.mlx_vector_array_append_value(vec, snap_aux.qsa_pooled);
            mlx.check(mlx.mlx_eval(vec)) catch {
                transformer_mod.ssmSnapshotDeinit(&snap_aux);
                break :head;
            };
            snap_aux.qsa_rows = hm.rows;
            head_aux = snap_aux;
            head_pos_base = hm.pos_base;
            for (hm.marks[0..hm.mark_count], 0..) |pos, mi| {
                var kind_buf: [16]u8 = undefined;
                const kind = std.fmt.bufPrint(&kind_buf, "h.lv{d}", .{mi}) catch break;
                const arr = getSpecArray(tensor_map, self.allocator, prefix, kind) orelse break;
                const one = mlx.mlx_vector_array_new();
                defer _ = mlx.mlx_vector_array_free(one);
                _ = mlx.mlx_vector_array_append_value(one, arr);
                mlx.check(mlx.mlx_eval(one)) catch {
                    _ = mlx.mlx_array_free(arr);
                    break;
                };
                head_marks.put(pos, arr);
            }
        }
        return .{ .snap = snap, .base = meta.base, .head_aux = head_aux, .head_pos_base = head_pos_base, .head_marks = head_marks };
    }

    /// One optional non-layer tensor out of the loaded sidecar map (+1 handle), or null.
    fn getSpecArray(map: mlx.mlx_map_string_to_array, allocator: std.mem.Allocator, prefix: []const u8, kind: []const u8) ?mlx.mlx_array {
        const key = std.fmt.allocPrint(allocator, "{s}{s}\x00", .{ prefix, kind }) catch return null;
        defer allocator.free(key);
        var arr = mlx.mlx_array_new();
        if (mlx.mlx_map_string_to_array_get(&arr, map, @ptrCast(key.ptr)) != 0) {
            _ = mlx.mlx_array_free(arr);
            return null;
        }
        return arr;
    }

    /// One staged tensor on the way to a safetensors file.
    const NamedTensor = struct { key: []u8, arr: mlx.mlx_array };

    fn freeNamed(self: *DiskTier, list: *std.ArrayList(NamedTensor)) void {
        for (list.items) |*t| {
            self.allocator.free(t.key);
            _ = mlx.mlx_array_free(t.arr);
        }
        list.deinit(self.allocator);
    }

    /// Write (or, under SSD-first, stage) one KV chunk. Returns the file's byte size.
    fn writeChunkFile(
        self: *DiskTier,
        kv_entries: []const transformer_mod.KVCacheEntry,
        config: kv_quant.KVQuantConfig,
        dir_abs: []const u8,
        chunk_idx: u32,
        c0: u32,
        c1: u32,
        s: mlx.mlx_stream,
    ) !u64 {
        var list = std.ArrayList(NamedTensor).empty;
        defer self.freeNamed(&list);

        const affine = config.scheme != .off;
        for (kv_entries, 0..) |*entry, li| {
            if (!entry.initialized) continue;
            try self.appendSlice(&list, li, "k", entry.keys, c0, c1, s);
            try self.appendSlice(&list, li, "v", entry.values, c0, c1, s);
            if (affine) {
                try self.appendSlice(&list, li, "ks", entry.keys_scales, c0, c1, s);
                try self.appendSlice(&list, li, "kb", entry.keys_biases, c0, c1, s);
                try self.appendSlice(&list, li, "vs", entry.values_scales, c0, c1, s);
                try self.appendSlice(&list, li, "vb", entry.values_biases, c0, c1, s);
            }
        }

        const path = try std.fmt.allocPrint(self.allocator, "{s}/c{d:0>6}.safetensors", .{ dir_abs, chunk_idx });
        if (self.writer) |w| {
            // The readback stays here (mlx arrays are inference-thread-owned); only bytes cross.
            const bytes = self.serializeSafetensors(list.items, no_meta, s) catch |err| {
                self.allocator.free(path);
                return err;
            };
            const n = bytes.len;
            w.submit(path, bytes); // takes both buffers
            return n;
        }
        defer self.allocator.free(path);
        const tensor_map = mlx.mlx_map_string_to_array_new();
        defer _ = mlx.mlx_map_string_to_array_free(tensor_map);
        const meta_map = mlx.mlx_map_string_to_string_new();
        defer _ = mlx.mlx_map_string_to_string_free(meta_map);
        for (list.items) |*t| {
            const key_z = try std.fmt.allocPrint(self.allocator, "{s}\x00", .{t.key});
            defer self.allocator.free(key_z);
            try mlx.check(mlx.mlx_map_string_to_array_insert(tensor_map, @ptrCast(key_z.ptr), t.arr));
        }
        const path_z = try std.fmt.allocPrint(self.allocator, "{s}\x00", .{path});
        defer self.allocator.free(path_z);
        try mlx.check(mlx.mlx_save_safetensors(@ptrCast(path_z.ptr), tensor_map, meta_map));
        return fileSize(self.io, path) orelse 0;
    }

    fn appendSlice(
        self: *DiskTier,
        list: *std.ArrayList(NamedTensor),
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
        errdefer _ = mlx.mlx_array_free(sliced);
        const st = [_]c_int{ 0, 0, @intCast(c0), 0 };
        const sp = [_]c_int{ shape[0], shape[1], @intCast(c1), shape[3] };
        const sd = [_]c_int{ 1, 1, 1, 1 };
        try mlx.check(mlx.mlx_slice(&sliced, buf, &st, 4, &sp, 4, &sd, 4, s));
        const key = try std.fmt.allocPrint(self.allocator, "l{d}.{s}", .{ layer, kind });
        errdefer self.allocator.free(key);
        try list.append(self.allocator, .{ .key = key, .arr = sliced });
    }

    /// Make every tensor contiguous in place and materialize the list with exactly one batched
    /// `mlx_eval`, as `mlx::core::save_safetensors` does. A per-tensor eval is a full GPU sync each.
    fn materializeContiguous(tensors: []NamedTensor, s: mlx.mlx_stream) !void {
        const vec = mlx.mlx_vector_array_new();
        defer _ = mlx.mlx_vector_array_free(vec);
        for (tensors) |*t| {
            var cont = mlx.mlx_array_new();
            {
                // `cont` is owned locally only until the transfer below; nothing fallible may sit
                // between this scope's close and `t.arr = cont`, or the list's `freeNamed` double-frees.
                errdefer _ = mlx.mlx_array_free(cont);
                try mlx.check(mlx.mlx_contiguous(&cont, t.arr, false, s));
            }
            _ = mlx.mlx_array_free(t.arr);
            t.arr = cont;
            try mlx.check(mlx.mlx_vector_array_append_value(vec, t.arr));
        }
        _ = serialize_eval_count.fetchAdd(1, .monotonic);
        try mlx.check(mlx.mlx_eval(vec));
    }

    /// Serialize a tensor list into one safetensors byte image, exactly as `mlx::core::save_safetensors` does.
    fn serializeSafetensors(self: *DiskTier, tensors: []NamedTensor, meta: []const MetaPair, s: mlx.mlx_stream) ![]u8 {
        try materializeContiguous(tensors, s);
        return self.encodeSafetensors(tensors, meta);
    }

    /// One `__metadata__` entry. The encoder refuses anything needing JSON escaping.
    pub const MetaPair = struct { key: []const u8, value: []const u8 };

    const no_meta: []const MetaPair = &[_]MetaPair{};

    /// Is `v` safe inside a JSON string with no escaping?
    fn plainJsonAscii(v: []const u8) bool {
        for (v) |c| {
            if (c < 0x20 or c > 0x7e or c == '"' or c == '\\') return false;
        }
        return true;
    }

    /// Header + payload encode over an already-materialized tensor list. Evaluates nothing.
    fn encodeSafetensors(self: *DiskTier, tensors: []const NamedTensor, meta: []const MetaPair) ![]u8 {
        var data_len: u64 = 0;
        for (tensors) |*t| data_len += nbytesOf(t.arr);

        var header = std.ArrayList(u8).empty;
        defer header.deinit(self.allocator);
        const hw = &header;
        try hw.appendSlice(self.allocator, "{\"__metadata__\":{");
        for (meta, 0..) |m, mi| {
            if (!plainJsonAscii(m.key) or !plainJsonAscii(m.value)) return error.DiskCacheBadMetadata;
            if (mi > 0) try hw.appendSlice(self.allocator, ",");
            try hw.print(self.allocator, "\"{s}\":\"{s}\"", .{ m.key, m.value });
        }
        try hw.appendSlice(self.allocator, "}");
        var off: u64 = 0;
        for (tensors) |*t| {
            const nb = nbytesOf(t.arr);
            const dname = safetensorsDtypeName(mlx.mlx_array_dtype(t.arr)) orelse return error.DiskCacheBadDtype;
            try hw.print(self.allocator, ",\"{s}\":{{\"dtype\":\"{s}\",\"shape\":[", .{ t.key, dname });
            for (mlx.getShape(t.arr), 0..) |d, i| {
                if (i > 0) try hw.appendSlice(self.allocator, ",");
                try hw.print(self.allocator, "{d}", .{d});
            }
            try hw.print(self.allocator, "],\"data_offsets\":[{d},{d}]}}", .{ off, off + nb });
            off += nb;
        }
        try hw.appendSlice(self.allocator, "}");

        const total = 8 + header.items.len + data_len;
        const out = try self.allocator.alloc(u8, total);
        errdefer self.allocator.free(out);
        std.mem.writeInt(u64, out[0..8], @intCast(header.items.len), .little);
        @memcpy(out[8 .. 8 + header.items.len], header.items);
        var cursor: usize = 8 + header.items.len;
        for (tensors) |*t| {
            const nb: usize = @intCast(nbytesOf(t.arr));
            if (nb == 0) continue;
            const src = rawBytes(t.arr) orelse return error.DiskCacheUnreadable;
            @memcpy(out[cursor .. cursor + nb], src[0..nb]);
            cursor += nb;
        }
        return out;
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
    /// flush: `SSM_DISK_MAX_PER_ENTRY` of (already-persisted ∪ newly-eligible), thinned
    /// span-preservingly. Eligible = a RAM checkpoint at a position within the
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
        while (set.items.len > self.ssm_max_per_entry) {
            _ = set.orderedRemove(transformer_mod.positionDropIndex(set.items, self.cp_thin));
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
    /// positions not yet on disk (highest-first — the end-of-prompt
    /// checkpoint is the most valuable), delete retention-dropped
    /// positions, and return the resulting on-disk set. `written_bytes`
    /// accumulates the bytes this call wrote; `max_spend` is this call's
    /// own byte share of the flush budget (the caller decides how much of
    /// `max_flush_bytes` the checkpoints get — chunk-first budgeting
    /// starved them to zero on chunk-heavy turns, live 2026-09-07).
    /// `kv_limit` is the WRITE-AHEAD bound, not the flushed kv_len: a
    /// checkpoint beyond the chunks this flush reaches is still written
    /// (position-keyed, immutable) and becomes restorable when a later
    /// flush extends the entry's kv_len past it.
    fn persistSsmCheckpoints(
        self: *DiskTier,
        id: u64,
        dir_rel: []const u8,
        kv_limit: u32,
        old_positions: []const u32,
        old_bytes: []const u64,
        cps_opt: ?[]const transformer_mod.SSMCheckpoint,
        written_bytes: *u64,
        s: mlx.mlx_stream,
        max_spend: u64,
    ) !SsmPersistResult {
        const cps: []const transformer_mod.SSMCheckpoint = cps_opt orelse &[_]transformer_mod.SSMCheckpoint{};
        if (cps.len == 0 and old_positions.len == 0) {
            return .{
                .positions = try self.allocator.alloc(u32, 0),
                .bytes = try self.allocator.alloc(u64, 0),
                .complete = true,
            };
        }
        const target = try self.ssmTargetPositions(old_positions, cps, kv_limit);
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
            // Under SSD-first a checkpoint is written beside its chunk, outside the byte budget;
            // elsewhere its share of the flush budget is bounded by `max_spend` (half the cap —
            // chunk-first budgeting starved them to zero, live 2026-09-07).
            if (!self.ssd_first and written_bytes.* >= max_spend) {
                complete = false;
                continue; // share exhausted — persist on a later flush
            }
            const sz = try self.writeSsmFile(dir_rel, cp, s);
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

    /// Write (or, under SSD-first, stage) one SSM checkpoint as `s{pos:0>7}.safetensors`.
    /// Per-layer tensors keyed "l{i}.conv"/"l{i}.ssm"; the `initialized` bitmap rides the
    /// metadata map. Staged because this runs inside the prefill chunk loop, where a
    /// synchronous ~56 MB write was the largest term of the mid-prefill write-through.
    fn writeSsmFile(
        self: *DiskTier,
        dir_rel: []const u8,
        cp: *const transformer_mod.SSMCheckpoint,
        s: mlx.mlx_stream,
    ) !u64 {
        var list = std.ArrayList(NamedTensor).empty;
        defer self.freeNamed(&list);
        var meta = std.ArrayList(MetaPair).empty;
        defer meta.deinit(self.allocator);

        var lc_buf: [24]u8 = undefined;
        const lc = try std.fmt.bufPrint(&lc_buf, "{d}\x00", .{cp.layers.len});
        try meta.append(self.allocator, .{ .key = "layers", .value = lc[0 .. lc.len - 1] });

        var init_buf = std.ArrayList(u8).empty;
        defer init_buf.deinit(self.allocator);
        var num_buf: [16]u8 = undefined;
        for (cp.layers, 0..) |l, li| {
            if (!l.initialized) continue;
            if (init_buf.items.len > 0) try init_buf.append(self.allocator, ',');
            const ns = std.fmt.bufPrint(&num_buf, "{d}", .{li}) catch unreachable;
            try init_buf.appendSlice(self.allocator, ns);
        }
        const init_len = init_buf.items.len;
        try init_buf.append(self.allocator, 0); // NUL-terminate for the C API
        // Taken after the last append: an earlier slice would dangle across list growth.
        try meta.append(self.allocator, .{ .key = "init", .value = init_buf.items[0..init_len] });

        // qwen4_exp aux state rides the same file: `l{d}.aux` / `l{d}.pooled`
        // tensors and `l{d}.ple` = uint32 [9] (valid flag, then the 8 token
        // history slots); the compress ratio is one `qsa_ratio` metadata key.
        var ratio_buf: [16]u8 = undefined;
        var rows_buf: [16]u8 = undefined;
        var ratio_written = false;
        var rows_written = false;
        for (cp.layers, 0..) |l, li| {
            const qsa_hist = transformer_mod.snapshotHasQsaHistory(&l);
            const names = .{ "conv", "ssm", "aux", "pooled" };
            const arrs = .{ l.conv_state, l.ssm_state, l.aux_state, l.qsa_pooled };
            inline for (names, arrs) |name, arr| {
                const skip_qsa = qsa_hist and !test_ssm_write_qsa_aux and (comptime std.mem.eql(u8, name, "pooled"));
                if (arr.ctx != null and !skip_qsa) {
                    const key = try std.fmt.allocPrint(self.allocator, "l{d}." ++ name, .{li});
                    errdefer self.allocator.free(key);
                    // The list owns every handle it holds, so a borrowed checkpoint handle is retained first.
                    var owned = mlx.mlx_array_new();
                    errdefer _ = mlx.mlx_array_free(owned);
                    try mlx.check(mlx.mlx_array_set(&owned, arr));
                    try list.append(self.allocator, .{ .key = key, .arr = owned });
                }
            }
            if (l.ple_prev_valid) {
                var ple: [9]u32 = undefined;
                ple[0] = 1;
                for (l.ple_prev, 0..) |t, i| ple[1 + i] = t;
                // `mlx_array_new_data` copies, so `ple` may die at the end of this block.
                const ple_arr = mlx.mlx_array_new_data(&ple, &[_]c_int{9}, 1, .uint32);
                errdefer _ = mlx.mlx_array_free(ple_arr);
                const key = try std.fmt.allocPrint(self.allocator, "l{d}.ple", .{li});
                errdefer self.allocator.free(key);
                try list.append(self.allocator, .{ .key = key, .arr = ple_arr });
            }
            if (!ratio_written and (l.qsa_rows > 0 or l.aux_state.ctx != null or l.qsa_pooled.ctx != null)) {
                const rs = try std.fmt.bufPrint(&ratio_buf, "{d}\x00", .{l.qsa_ratio});
                try meta.append(self.allocator, .{ .key = "qsa_ratio", .value = rs[0 .. rs.len - 1] });
                ratio_written = true;
            }
            if (!rows_written and l.qsa_rows > 0) {
                const rs = try std.fmt.bufPrint(&rows_buf, "{d}\x00", .{l.qsa_rows});
                try meta.append(self.allocator, .{ .key = "qsa_rows", .value = rs[0 .. rs.len - 1] });
                rows_written = true;
            }
        }

        if (self.writer) |w| {
            const path = try std.fmt.allocPrint(self.allocator, "{s}/s{d:0>7}.safetensors", .{ dir_rel, cp.pos });
            const bytes = self.serializeSafetensors(list.items, meta.items, s) catch |err| {
                self.allocator.free(path);
                return err;
            };
            const n = bytes.len;
            w.submit(path, bytes); // takes both buffers
            return n;
        }

        const tensor_map = mlx.mlx_map_string_to_array_new();
        defer _ = mlx.mlx_map_string_to_array_free(tensor_map);
        const meta_map = mlx.mlx_map_string_to_string_new();
        defer _ = mlx.mlx_map_string_to_string_free(meta_map);
        for (meta.items) |m| {
            const kz = try std.fmt.allocPrint(self.allocator, "{s}\x00", .{m.key});
            defer self.allocator.free(kz);
            const vz = try std.fmt.allocPrint(self.allocator, "{s}\x00", .{m.value});
            defer self.allocator.free(vz);
            try mlx.check(mlx.mlx_map_string_to_string_insert(meta_map, @ptrCast(kz.ptr), @ptrCast(vz.ptr)));
        }
        for (list.items) |*t| {
            const kz = try std.fmt.allocPrint(self.allocator, "{s}\x00", .{t.key});
            defer self.allocator.free(kz);
            try mlx.check(mlx.mlx_map_string_to_array_insert(tensor_map, @ptrCast(kz.ptr), t.arr));
        }
        const path = try std.fmt.allocPrint(self.allocator, "{s}/s{d:0>7}.safetensors\x00", .{ dir_rel, cp.pos });
        defer self.allocator.free(path);
        try mlx.check(mlx.mlx_save_safetensors(@ptrCast(path.ptr), tensor_map, meta_map));
        return fileSize(self.io, path[0 .. path.len - 1]) orelse 0;
    }

    fn qsaHistoryRowsOf(cp: *const transformer_mod.SSMCheckpoint) c_int {
        var r: c_int = 0;
        for (cp.layers) |*l| {
            if (l.qsa_rows > r) r = l.qsa_rows;
            if (!transformer_mod.snapshotHasQsaHistory(l)) continue;
            if (l.aux_state.ctx != null) {
                const sh = mlx.getShape(l.aux_state);
                if (sh.len >= 2 and sh[1] > r) r = sh[1];
            }
        }
        return r;
    }

    fn newestQsaCheckpoint(cps: []const transformer_mod.SSMCheckpoint) ?*const transformer_mod.SSMCheckpoint {
        var i = cps.len;
        while (i > 0) {
            i -= 1;
            if (transformer_mod.checkpointHasQsaPooled(&cps[i])) return &cps[i];
        }
        return null;
    }

    const QsaHistoryResult = struct { bytes: u64 = 0, rows: u32 = 0, inherited: bool = false };

    fn writeQsaHistoryFile(
        self: *DiskTier,
        dir_rel: []const u8,
        cp: *const transformer_mod.SSMCheckpoint,
        s: mlx.mlx_stream,
    ) !u64 {
        var list = std.ArrayList(NamedTensor).empty;
        defer self.freeNamed(&list);
        var meta = std.ArrayList(MetaPair).empty;
        defer meta.deinit(self.allocator);
        var ratio_buf: [16]u8 = undefined;
        var rows_buf: [16]u8 = undefined;
        var ratio_written = false;
        const rows = DiskTier.qsaHistoryRowsOf(cp);
        if (rows > 0) {
            const rs = try std.fmt.bufPrint(&rows_buf, "{d}\x00", .{rows});
            try meta.append(self.allocator, .{ .key = "qsa_rows", .value = rs[0 .. rs.len - 1] });
        }
        for (cp.layers, 0..) |l, li| {
            const names = .{ "aux", "pooled" };
            const arrs = .{ l.aux_state, l.qsa_pooled };
            inline for (names, arrs) |name, arr| {
                if (arr.ctx != null) {
                    const key = try std.fmt.allocPrint(self.allocator, "l{d}." ++ name, .{li});
                    errdefer self.allocator.free(key);
                    var owned = mlx.mlx_array_new();
                    errdefer _ = mlx.mlx_array_free(owned);
                    try mlx.check(mlx.mlx_array_set(&owned, arr));
                    try list.append(self.allocator, .{ .key = key, .arr = owned });
                }
            }
            if (!ratio_written and (l.aux_state.ctx != null or l.qsa_pooled.ctx != null)) {
                const rs = try std.fmt.bufPrint(&ratio_buf, "{d}\x00", .{l.qsa_ratio});
                try meta.append(self.allocator, .{ .key = "qsa_ratio", .value = rs[0 .. rs.len - 1] });
                ratio_written = true;
            }
        }
        if (list.items.len == 0) return 0;
        if (self.writer) |w| {
            const path = try std.fmt.allocPrint(self.allocator, "{s}/qsa.safetensors", .{dir_rel});
            const bytes = self.serializeSafetensors(list.items, meta.items, s) catch |err| {
                self.allocator.free(path);
                return err;
            };
            const n = bytes.len;
            w.submit(path, bytes);
            return n;
        }
        const tensor_map = mlx.mlx_map_string_to_array_new();
        defer _ = mlx.mlx_map_string_to_array_free(tensor_map);
        const meta_map = mlx.mlx_map_string_to_string_new();
        defer _ = mlx.mlx_map_string_to_string_free(meta_map);
        for (meta.items) |m| {
            const kz = try std.fmt.allocPrint(self.allocator, "{s}\x00", .{m.key});
            defer self.allocator.free(kz);
            const vz = try std.fmt.allocPrint(self.allocator, "{s}\x00", .{m.value});
            defer self.allocator.free(vz);
            try mlx.check(mlx.mlx_map_string_to_string_insert(meta_map, @ptrCast(kz.ptr), @ptrCast(vz.ptr)));
        }
        for (list.items) |*t| {
            const kz = try std.fmt.allocPrint(self.allocator, "{s}\x00", .{t.key});
            defer self.allocator.free(kz);
            try mlx.check(mlx.mlx_map_string_to_array_insert(tensor_map, @ptrCast(kz.ptr), t.arr));
        }
        const path = try std.fmt.allocPrint(self.allocator, "{s}/qsa.safetensors\x00", .{dir_rel});
        defer self.allocator.free(path);
        try mlx.check(mlx.mlx_save_safetensors(@ptrCast(path.ptr), tensor_map, meta_map));
        return fileSize(self.io, path[0 .. path.len - 1]) orelse 0;
    }

    fn loadQsaHistoryFile(self: *DiskTier, id: u64, n_layers: usize) !transformer_mod.SSMCheckpoint {
        const cpu = mlx.mlx_default_cpu_stream_new();
        defer _ = mlx.mlx_stream_free(cpu);
        const path = try std.fmt.allocPrint(self.allocator, "{s}/e{d}/qsa.safetensors\x00", .{ self.root, id });
        defer self.allocator.free(path);
        if (fileSize(self.io, path[0 .. path.len - 1]) == null) return error.DiskCacheNoCheckpoint;
        var tensor_map = mlx.mlx_map_string_to_array_new();
        defer _ = mlx.mlx_map_string_to_array_free(tensor_map);
        var meta_map = mlx.mlx_map_string_to_string_new();
        defer _ = mlx.mlx_map_string_to_string_free(meta_map);
        try mlx.check(mlx.mlx_load_safetensors(&tensor_map, &meta_map, @ptrCast(path.ptr), cpu));

        const layers = try self.allocator.alloc(transformer_mod.SSMCacheEntrySnapshot, n_layers);
        for (layers) |*l| l.* = .{
            .conv_state = mlx.mlx_array_new(),
            .ssm_state = mlx.mlx_array_new(),
            .initialized = false,
        };
        var cp: transformer_mod.SSMCheckpoint = .{ .pos = 0, .layers = layers };
        errdefer cp.deinit(self.allocator);
        for (layers, 0..) |*l, li| {
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
        }
        var ratio_c: [*:0]const u8 = undefined;
        if (mlx.mlx_map_string_to_string_get(&ratio_c, meta_map, "qsa_ratio") == 0) {
            const ratio = std.fmt.parseInt(c_int, std.mem.span(ratio_c), 10) catch return error.DiskCacheCorruptSsm;
            for (layers) |*l| l.qsa_ratio = ratio;
        }
        var rows_c: [*:0]const u8 = undefined;
        if (mlx.mlx_map_string_to_string_get(&rows_c, meta_map, "qsa_rows") == 0) {
            const rows = std.fmt.parseInt(c_int, std.mem.span(rows_c), 10) catch return error.DiskCacheCorruptSsm;
            for (layers) |*l| l.qsa_rows = rows;
            cp.pos = @intCast(rows);
        } else {
            cp.pos = @intCast(DiskTier.qsaHistoryRowsOf(&cp));
        }
        {
            const vec = mlx.mlx_vector_array_new();
            defer _ = mlx.mlx_vector_array_free(vec);
            var count: usize = 0;
            for (layers) |*l| {
                inline for (.{ l.aux_state, l.qsa_pooled }) |arr| {
                    if (arr.ctx != null) {
                        _ = mlx.mlx_vector_array_append_value(vec, arr);
                        count += 1;
                    }
                }
            }
            if (count > 0) try mlx.check(mlx.mlx_eval(vec));
        }
        if (cp.pos == 0) cp.pos = @intCast(DiskTier.qsaHistoryRowsOf(&cp));
        // A v7 file holds the WHOLE raw history, and a mid-block checkpoint's leftover is
        // interior to it: the ring truncation belongs to the live entry the restore seeds,
        // not to the file it slices that ring out of.
        for (layers) |*l| {
            if (l.qsa_rows == 0) l.qsa_rows = @intCast(cp.pos);
        }
        if (!transformer_mod.checkpointHasQsaHistory(&cp)) return error.DiskCacheNoCheckpoint;
        return cp;
    }

    fn persistQsaHistory(
        self: *DiskTier,
        dir_rel: []const u8,
        cps_opt: ?[]const transformer_mod.SSMCheckpoint,
        inherited: bool,
        inherited_rows: u32,
        prefix_rows: u32,
        s: mlx.mlx_stream,
    ) !QsaHistoryResult {
        const cps = cps_opt orelse return .{ .inherited = inherited, .rows = inherited_rows, .bytes = 0 };
        const src = DiskTier.newestQsaCheckpoint(cps) orelse {
            if (inherited) return .{ .inherited = true, .rows = inherited_rows, .bytes = 0 };
            return .{};
        };
        const rows: u32 = @intCast(DiskTier.qsaHistoryRowsOf(src));
        if (inherited and rows > 0 and rows <= inherited_rows and rows <= prefix_rows) {
            return .{ .inherited = true, .rows = @min(inherited_rows, prefix_rows), .bytes = 0 };
        }
        if (inherited) {
            const qpath = try std.fmt.allocPrint(self.allocator, "{s}/qsa.safetensors", .{dir_rel});
            defer self.allocator.free(qpath);
            if (self.writer) |w| w.fence(qpath);
            std.Io.Dir.deleteFileAbsolute(self.io, qpath) catch {};
        }
        const bytes = try self.writeQsaHistoryFile(dir_rel, src, s);
        return .{ .bytes = bytes, .rows = rows, .inherited = false };
    }

    fn deleteSsmFile(self: *DiskTier, id: u64, pos: u32) void {
        const path = std.fmt.allocPrint(self.allocator, "{s}/e{d}/s{d:0>7}.safetensors", .{ self.root, id, pos }) catch return;
        defer self.allocator.free(path);
        // A staged checkpoint may still be in the writer's queue; fence it before deleting, or
        // the write lands a file no index names.
        if (self.writer) |w| w.fence(path);
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
        self.total_bytes -|= self.bytesFreedByRemoving(&e);
        self.deleteEntryDir(e.id);
        self.freeIndexEntryOwned(&e);
    }

    /// Bytes deleting `e`'s directory returns to the volume: its non-chunk files plus every
    /// chunk file nobody else links (`nlink == 1`). The filesystem is the refcount.
    fn bytesFreedByRemoving(self: *DiskTier, e: *const IndexEntry) u64 {
        const dir_abs = std.fmt.allocPrint(self.allocator, "{s}/e{d}/", .{ self.root, e.id }) catch return e.bytes;
        defer self.allocator.free(dir_abs);
        if (self.writer) |w| w.fence(dir_abs);
        var freed: u64 = @as(u64, e.tokens.len) * 4 + e.spec_bytes;
        for (e.ssm_bytes) |b| freed += b;
        if (e.qsa_history_bytes > 0) {
            if (std.fmt.allocPrint(self.allocator, "{s}qsa.safetensors", .{dir_abs})) |qp| {
                defer self.allocator.free(qp);
                if (statFile(self.io, qp)) |st| {
                    if (st.nlink <= 1) freed += st.size;
                } else if (!e.inherited_qsa) {
                    freed += e.qsa_history_bytes;
                }
            } else |_| {
                if (!e.inherited_qsa) freed += e.qsa_history_bytes;
            }
        }
        for (e.chunk_bytes, 0..) |cb, i| {
            const cp = std.fmt.allocPrint(self.allocator, "{s}c{d:0>6}.safetensors", .{ dir_abs, i }) catch {
                freed += cb;
                continue;
            };
            defer self.allocator.free(cp);
            if (statFile(self.io, cp)) |st| {
                if (st.nlink <= 1) freed += st.size;
            } else if (i >= e.inherited_chunks) {
                freed += cb;
            }
        }
        return freed;
    }

    /// The resident entry whose leading chunk files a new entry for `tokens` may hard-link:
    /// same tool flag and kv-quant config, most whole chunks below the common prefix. Null on
    /// the legacy arm or under `MLX_SERVE_SSD_CHUNK_SHARE=0`.
    fn chunkShareDonor(self: *DiskTier, tokens: []const u32, kv_target: u32, has_tools: bool, config: kv_quant.KVQuantConfig) ?ChunkDonor {
        if (!self.ssd_first or !chunkShareEnabled()) return null;
        var best: ?ChunkDonor = null;
        for (self.entries.items, 0..) |*e, i| {
            if (e.poisoned) continue; // never inherit from a dead entry
            if (e.has_tools != has_tools) continue;
            if (!std.meta.eql(e.quant, config)) continue;
            const shared: u64 = @min(@min(@as(u64, commonPrefixLen(e.tokens, tokens)), @as(u64, kv_target)), @as(u64, e.kv_len));
            const whole: u32 = @intCast(shared / self.chunk_tokens);
            const usable: u32 = @min(whole, @as(u32, @intCast(e.chunk_bytes.len)));
            if (usable == 0) continue;
            if (best == null or usable > best.?.chunks) best = .{ .idx = i, .id = e.id, .chunks = usable };
        }
        return best;
    }

    const ChunkDonor = struct { idx: usize, id: u64, chunks: u32 };

    /// Hard-link the donor's leading landed chunk files into `e<id>/`; returns how many. Stops
    /// at the first chunk not landed and never touches the donor's queue (`Writer.fence` would
    /// discard it). A link failure unwinds and the caller writes every chunk itself.
    fn linkInheritedChunks(self: *DiskTier, d: ChunkDonor, id: u64, chunk_sizes: *std.ArrayList(u64)) !u32 {
        var root_dir = try std.Io.Dir.openDirAbsolute(self.io, self.root, .{});
        defer root_dir.close(self.io);
        const donor_cb = self.entries.items[d.idx].chunk_bytes;
        var linked: u32 = 0;
        errdefer self.unlinkChunkRange(root_dir, id, 0, linked);
        var i: u32 = 0;
        while (i < d.chunks) : (i += 1) {
            const old_abs = try std.fmt.allocPrint(self.allocator, "{s}/e{d}/c{d:0>6}.safetensors", .{ self.root, d.id, i });
            defer self.allocator.free(old_abs);
            if (!self.chunkLanded(old_abs, donor_cb[i])) break;
            const old_sub = try std.fmt.allocPrint(self.allocator, "e{d}/c{d:0>6}.safetensors", .{ d.id, i });
            defer self.allocator.free(old_sub);
            const new_sub = try std.fmt.allocPrint(self.allocator, "e{d}/c{d:0>6}.safetensors", .{ id, i });
            defer self.allocator.free(new_sub);
            std.Io.Dir.hardLink(root_dir, old_sub, root_dir, new_sub, self.io, .{}) catch |err| {
                log.warn("  [disk-cache] chunk share: link e{d}/c{d} -> e{d} failed: {s} — writing the chunks instead\n", .{ d.id, i, id, @errorName(err) });
                return error.ChunkShareLinkFailed;
            };
            linked += 1;
            try chunk_sizes.append(self.allocator, donor_cb[i]);
        }
        if (linked == 0) return 0;
        var mb: f64 = 0;
        for (donor_cb[0..linked]) |b| mb += @as(f64, @floatFromInt(b));
        mb /= 1024.0 * 1024.0;
        log.info("  [disk-cache] chunk share: e{d} inherits {d} chunks ({d:.1} MB) from e{d} by hard link\n", .{ id, linked, mb, d.id });
        return linked;
    }

    fn linkInheritedQsa(self: *DiskTier, d: ChunkDonor, id: u64) bool {
        const donor = &self.entries.items[d.idx];
        if (donor.qsa_history_bytes == 0 or donor.qsa_history_rows == 0) return false;
        const old_abs = std.fmt.allocPrint(self.allocator, "{s}/e{d}/qsa.safetensors", .{ self.root, d.id }) catch return false;
        defer self.allocator.free(old_abs);
        if (!self.chunkLanded(old_abs, donor.qsa_history_bytes)) return false;
        var root_dir = std.Io.Dir.openDirAbsolute(self.io, self.root, .{}) catch return false;
        defer root_dir.close(self.io);
        const old_sub = std.fmt.allocPrint(self.allocator, "e{d}/qsa.safetensors", .{d.id}) catch return false;
        defer self.allocator.free(old_sub);
        const new_sub = std.fmt.allocPrint(self.allocator, "e{d}/qsa.safetensors", .{id}) catch return false;
        defer self.allocator.free(new_sub);
        std.Io.Dir.hardLink(root_dir, old_sub, root_dir, new_sub, self.io, .{}) catch |err| {
            log.warn("  [disk-cache] chunk share: link e{d}/qsa -> e{d} failed: {s} — writing the history instead\n", .{ d.id, id, @errorName(err) });
            return false;
        };
        log.info("  [disk-cache] chunk share: e{d} inherits qsa history from e{d} by hard link\n", .{ id, d.id });
        return true;
    }

    /// Has a chunk file landed: final name, recorded size, and no write to it queued or in flight?
    fn chunkLanded(self: *DiskTier, abs_path: []const u8, want_size: u64) bool {
        const st = statFile(self.io, abs_path) orelse return false;
        if (st.size != want_size) return false;
        if (self.writer) |w| {
            if (w.isPending(abs_path)) return false;
        }
        return true;
    }

    fn unlinkChunkRange(self: *DiskTier, root_dir: std.Io.Dir, id: u64, from: u32, to: u32) void {
        var i: u32 = from;
        while (i < to) : (i += 1) {
            const sub = std.fmt.allocPrint(self.allocator, "e{d}/c{d:0>6}.safetensors", .{ id, i }) catch continue;
            defer self.allocator.free(sub);
            root_dir.deleteFile(self.io, sub) catch {};
        }
    }

    fn deleteEntryDir(self: *DiskTier, id: u64) void {
        // Epoch fence, the one removal site: staged bytes for this directory are discarded.
        const dir_abs = std.fmt.allocPrint(self.allocator, "{s}/e{d}/", .{ self.root, id }) catch null;
        defer if (dir_abs) |da| self.allocator.free(da);
        if (self.writer) |w| w.fence(dir_abs);
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
        var buf = std.ArrayList(u8).empty;
        defer buf.deinit(self.allocator);
        try self.renderMeta(&buf, e);

        const final_path = try std.fmt.allocPrint(self.allocator, "{s}/e{d}/meta.json", .{ self.root, e.id });
        // One owner per branch: the staged branch's cleanup is an `errdefer` scoped to it (a
        // `defer` does not cancel an enclosing `errdefer`, and that was a double free on ENOSPC).
        if (self.writer) |w| {
            // The index rides the same FIFO queue as the chunks, so it is the last file to land.
            errdefer self.allocator.free(final_path);
            const bytes = try self.allocator.dupe(u8, buf.items);
            w.submit(final_path, bytes);
            return;
        }
        defer self.allocator.free(final_path);
        const tmp_path = try std.fmt.allocPrint(self.allocator, "{s}/e{d}/meta.json.tmp", .{ self.root, e.id });
        defer self.allocator.free(tmp_path);
        {
            const f = try std.Io.Dir.createFileAbsolute(self.io, tmp_path, .{});
            defer f.close(self.io);
            var wb: [1024]u8 = undefined;
            var fw = f.writer(self.io, &wb);
            try fw.interface.writeAll(buf.items);
            try fw.interface.flush();
        }
        try std.Io.Dir.renameAbsolute(tmp_path, final_path, self.io);
    }

    /// The lowest manifest version that describes this entry. The version is a compatibility
    /// claim: an older reader accepts only 2..4, so stamping v6 unconditionally made a binary
    /// downgrade discard the whole tier. v6 = inherited chunks, v5 = the MTP head's QSA half.
    fn metaVersionFor(e: IndexEntry) u8 {
        if (e.qsa_history_rows > 0 or e.qsa_history_bytes > 0) return 8;
        if (e.inherited_chunks > 0) return 6;
        if (e.spec_mtp) |m| if (m.head) |h| return if (h.mark_count > 0) 8 else 5;
        return 4;
    }

    /// The meta.json body; one renderer for the synchronous and the staged path.
    fn renderMeta(self: *DiskTier, out: *std.ArrayList(u8), e: IndexEntry) !void {
        const a = self.allocator;
        try out.print(
            a,
            "{{\"v\":{d},\"kv_len\":{d},\"tokens\":{d},\"has_tools\":{},\"scheme\":\"{s}\",\"bits\":{d},\"group_size\":{d},\"chunk_tokens\":{d},\"inherited_chunks\":{d},\"bytes\":{d},\"chunk_bytes\":[",
            .{
                metaVersionFor(e),
                e.kv_len,
                e.tokens.len,
                e.has_tools,
                @tagName(e.quant.scheme),
                e.quant.bits,
                e.quant.group_size,
                self.chunk_tokens,
                e.inherited_chunks,
                e.bytes,
            },
        );
        for (e.chunk_bytes, 0..) |cb, i| {
            if (i > 0) try out.appendSlice(a, ",");
            try out.print(a, "{d}", .{cb});
        }
        // v3: SSM checkpoints as [{pos,bytes},...] (sorted ascending).
        try out.appendSlice(a, "],\"ssm\":[");
        for (e.ssm_positions, e.ssm_bytes, 0..) |pos, sz, i| {
            if (i > 0) try out.appendSlice(a, ",");
            try out.print(a, "{{\"pos\":{d},\"bytes\":{d}}}", .{ pos, sz });
        }
        try out.appendSlice(a, "]");
        if (e.qsa_history_rows > 0 or e.qsa_history_bytes > 0) {
            try out.print(a, ",\"qsa_history\":{{\"bytes\":{d},\"rows\":{d},\"inherited\":{}}}", .{
                e.qsa_history_bytes,
                e.qsa_history_rows,
                e.inherited_qsa,
            });
        }
        // v4: spec snapshots (dflash context / MTP history); a size mismatch drops only the spec.
        if (e.spec_bytes > 0 and (e.spec_dflash != null or e.spec_mtp != null)) {
            try out.print(a, ",\"spec\":{{\"bytes\":{d}", .{e.spec_bytes});
            if (e.spec_dflash) |sm| try writeSpecMetaJson(a, out, "dflash", sm);
            if (e.spec_mtp) |sm| try writeSpecMetaJson(a, out, "mtp", sm);
            try out.appendSlice(a, "}");
        }
        try out.appendSlice(a, "}");
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

        // Shared chunk files are counted once.
        var seen = std.AutoHashMap(u64, void).init(self.allocator);
        defer seen.deinit();
        for (pending.items) |*p| {
            p.e.last_used = self.bump();
            self.billChunksOnce(&p.e, &seen);
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

    /// Re-bill a scanned entry's chunk files against the inodes already counted this scan: the
    /// first entry to see an inode pays for it.
    fn billChunksOnce(self: *DiskTier, e: *IndexEntry, seen: *std.AutoHashMap(u64, void)) void {
        var billed: u64 = @as(u64, e.tokens.len) * 4 + e.spec_bytes;
        for (e.ssm_bytes) |b| billed += b;
        if (e.qsa_history_bytes > 0) {
            if (std.fmt.allocPrint(self.allocator, "{s}/e{d}/qsa.safetensors", .{ self.root, e.id })) |qp| {
                defer self.allocator.free(qp);
                if (statFile(self.io, qp)) |st| {
                    if (st.nlink <= 1) {
                        billed += st.size;
                    } else {
                        const ino: u64 = @intCast(st.inode);
                        if (seen.getOrPut(ino)) |gop| {
                            if (!gop.found_existing) billed += st.size;
                        } else |_| {
                            billed += st.size;
                        }
                    }
                } else if (!e.inherited_qsa) {
                    billed += e.qsa_history_bytes;
                }
            } else |_| {
                if (!e.inherited_qsa) billed += e.qsa_history_bytes;
            }
        }
        for (e.chunk_bytes, 0..) |cb, i| {
            const cp = std.fmt.allocPrint(self.allocator, "{s}/e{d}/c{d:0>6}.safetensors", .{ self.root, e.id, i }) catch {
                billed += cb;
                continue;
            };
            defer self.allocator.free(cp);
            const st = statFile(self.io, cp) orelse {
                billed += cb;
                continue;
            };
            if (st.nlink <= 1) {
                billed += st.size;
                continue;
            }
            const ino: u64 = @intCast(st.inode);
            const gop = seen.getOrPut(ino) catch {
                billed += st.size;
                continue;
            };
            if (!gop.found_existing) billed += st.size;
        }
        e.bytes = billed;
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
        // adds optional spec snapshots; v5 the qwen4_exp MTP head's QSA half; v6 inherited
        // chunks. All restore; a lower-version entry just carries none of the newer state.
        if (version < 2 or version > 8) return null;
        // v6: the leading `inherited_chunks` chunk files are hard links into a donor's.
        const inherited_rec: u64 = jsonU64(obj, "inherited_chunks") orelse 0;
        var kv_len = jsonU64(obj, "kv_len") orelse return null;
        const n_tokens = jsonU64(obj, "tokens") orelse return null;
        const chunk_tokens = jsonU64(obj, "chunk_tokens") orelse return null;
        const has_tools_v = obj.get("has_tools") orelse return null;
        if (has_tools_v != .bool) return null;
        const chunk_bytes_v = obj.get("chunk_bytes") orelse return null;
        if (chunk_bytes_v != .array) return null;

        // Chunk geometry must match this tier's configuration — a stale root
        // written under a different chunk size can't be extended coherently.
        if (chunk_tokens != self.chunk_tokens) return null;
        if (kv_len == 0 or kv_len > n_tokens) return null;

        const quant = manifestQuant(obj) orelse return null;

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

        const inherited: u32 = @intCast(@min(inherited_rec, chunk_bytes.len));
        var total: u64 = @as(u64, tokens.len) * 4;
        for (chunk_bytes[inherited..]) |cb| total += cb;

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

        var qsa_history_bytes: u64 = 0;
        var qsa_history_rows: u32 = 0;
        var inherited_qsa = false;
        if (obj.get("qsa_history")) |qh_v| qh_blk: {
            if (qh_v != .object) break :qh_blk;
            const qo = qh_v.object;
            const rec_bytes = jsonU64(qo, "bytes") orelse break :qh_blk;
            const rec_rows = jsonInt(u32, qo, "rows") orelse break :qh_blk;
            const inh_v = qo.get("inherited");
            inherited_qsa = inh_v != null and inh_v.? == .bool and inh_v.?.bool;
            const qp = std.fmt.allocPrint(self.allocator, "{s}/e{d}/qsa.safetensors", .{ self.root, id }) catch break :qh_blk;
            defer self.allocator.free(qp);
            const have = fileSize(self.io, qp) orelse break :qh_blk;
            if (have != rec_bytes) {
                log.info("  [disk-cache] e{d}: qsa history size mismatch — dropping the history\n", .{id});
                break :qh_blk;
            }
            qsa_history_bytes = rec_bytes;
            qsa_history_rows = rec_rows;
            if (!inherited_qsa) total += rec_bytes;
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
                .inherited_chunks = inherited,
                .chunk_bytes = chunk_bytes,
                .ssm_positions = ssm_positions,
                .ssm_bytes = ssm_bytes,
                .spec_bytes = spec_bytes,
                .spec_dflash = spec_dflash,
                .spec_mtp = spec_mtp,
                .qsa_history_bytes = qsa_history_bytes,
                .qsa_history_rows = qsa_history_rows,
                .inherited_qsa = inherited_qsa,
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
/// Root-wide LRU sweep across sibling model fingerprints (never `keep_root`, which the tier's
/// own `scan` owns), oldest first until they fit `sibling_budget`, plus index-less strays that
/// are old enough to be crash leftovers. Best effort.
pub fn sweepBase(
    allocator: std.mem.Allocator,
    io: std.Io,
    base_dir: []const u8,
    keep_root: []const u8,
    sibling_budget: u64,
) void {
    var base = std.Io.Dir.openDirAbsolute(io, base_dir, .{ .iterate = true }) catch return;
    defer base.close(io);

    const Victim = struct { path: []u8, bytes: u64, mtime: i128 };
    var victims = std.ArrayList(Victim).empty;
    defer {
        for (victims.items) |v| allocator.free(v.path);
        victims.deinit(allocator);
    }
    var total: u64 = 0;
    var strays: usize = 0;

    var fps = base.iterate();
    while (fps.next(io) catch null) |fp| {
        if (fp.kind != .directory) continue;
        const fp_abs = std.fmt.allocPrint(allocator, "{s}/{s}", .{ base_dir, fp.name }) catch continue;
        defer allocator.free(fp_abs);
        if (std.mem.eql(u8, fp_abs, keep_root)) continue; // the live tier owns its own

        var fpd = std.Io.Dir.openDirAbsolute(io, fp_abs, .{ .iterate = true }) catch continue;
        defer fpd.close(io);
        var es = fpd.iterate();
        while (es.next(io) catch null) |dent| {
            if (dent.kind != .directory) continue;
            if (dent.name.len < 2 or dent.name[0] != 'e') continue;
            const e_abs = std.fmt.allocPrint(allocator, "{s}/{s}", .{ fp_abs, dent.name }) catch continue;
            const meta = std.fmt.allocPrint(allocator, "{s}/meta.json", .{e_abs}) catch {
                allocator.free(e_abs);
                continue;
            };
            defer allocator.free(meta);
            const st = std.Io.Dir.cwd().statFile(io, meta, .{}) catch {
                // No index: a crash leftover, OR another server's flush in progress (meta lands
                // last). Age is the only signal.
                if (dirYoungerThan(io, e_abs, STRAY_MIN_AGE_NS)) {
                    allocator.free(e_abs);
                    continue;
                }
                deleteTreeAbsolute(io, e_abs);
                allocator.free(e_abs);
                strays += 1;
                continue;
            };
            // A `.tmp` older than the same bar is a crash leftover of the writer's tmp+rename.
            reapStaleTmp(io, e_abs);
            const bytes = dirBytes(io, e_abs);
            total += bytes;
            victims.append(allocator, .{ .path = e_abs, .bytes = bytes, .mtime = st.mtime.nanoseconds }) catch {
                allocator.free(e_abs);
            };
        }
    }

    if (strays > 0) log.info("  [disk-cache] swept {d} stray entry directories under {s}\n", .{ strays, base_dir });
    if (total <= sibling_budget) return;

    std.mem.sort(Victim, victims.items, {}, struct {
        fn lt(_: void, a: Victim, b: Victim) bool {
            return a.mtime < b.mtime;
        }
    }.lt);
    var freed: u64 = 0;
    for (victims.items) |v| {
        if (total -| freed <= sibling_budget) break;
        deleteTreeAbsolute(io, v.path);
        freed += v.bytes;
    }
    if (freed > 0) {
        log.info("  [disk-cache] root-wide LRU freed {d} MB across other models ({d} MB held, {d} MB budget)\n", .{
            freed >> 20,
            total >> 20,
            sibling_budget >> 20,
        });
    }
}

/// How old an index-less entry directory must be before a sweep may treat it as a crash leftover.
const STRAY_MIN_AGE_NS: i128 = 10 * 60 * @as(i128, std.time.ns_per_s);

/// True when any regular file directly inside `dir_abs` was modified within `age_ns`. Unreadable = young.
fn dirYoungerThan(io: std.Io, dir_abs: []const u8, age_ns: i128) bool {
    var d = std.Io.Dir.openDirAbsolute(io, dir_abs, .{ .iterate = true }) catch return true;
    defer d.close(io);
    const now = std.Io.Timestamp.now(io, .real).nanoseconds;
    var it = d.iterate();
    while (it.next(io) catch null) |dent| {
        if (dent.kind != .file) continue;
        const st = d.statFile(io, dent.name, .{}) catch return true;
        if (now -| st.mtime.nanoseconds < age_ns) return true;
    }
    return false;
}

/// Delete `.tmp` files in `dir_abs` older than `STRAY_MIN_AGE_NS`.
fn reapStaleTmp(io: std.Io, dir_abs: []const u8) void {
    var d = std.Io.Dir.openDirAbsolute(io, dir_abs, .{ .iterate = true }) catch return;
    defer d.close(io);
    const now = std.Io.Timestamp.now(io, .real).nanoseconds;
    var it = d.iterate();
    while (it.next(io) catch null) |dent| {
        if (dent.kind != .file) continue;
        if (!std.mem.endsWith(u8, dent.name, ".tmp")) continue;
        const st = d.statFile(io, dent.name, .{}) catch continue;
        if (now -| st.mtime.nanoseconds < STRAY_MIN_AGE_NS) continue;
        d.deleteFile(io, dent.name) catch {};
    }
}

/// Total bytes of the regular files directly inside `dir_abs`.
fn dirBytes(io: std.Io, dir_abs: []const u8) u64 {
    var d = std.Io.Dir.openDirAbsolute(io, dir_abs, .{ .iterate = true }) catch return 0;
    defer d.close(io);
    var total: u64 = 0;
    var it = d.iterate();
    while (it.next(io) catch null) |dent| {
        if (dent.kind != .file) continue;
        const st = d.statFile(io, dent.name, .{}) catch continue;
        total += st.size;
    }
    return total;
}

fn deleteTreeAbsolute(io: std.Io, dir_abs: []const u8) void {
    const parent = std.fs.path.dirname(dir_abs) orelse return;
    const name = std.fs.path.basename(dir_abs);
    var pd = std.Io.Dir.openDirAbsolute(io, parent, .{ .iterate = true }) catch return;
    defer pd.close(io);
    pd.deleteTree(io, name) catch {};
}

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
    if (model.getConfigOverrides()) |raw| h.update(raw);
    return std.fmt.allocPrint(allocator, "{x:0>16}", .{h.final()});
}

/// Default persistence root: `~/.mlx-serve/kv-cache`.
pub fn defaultBaseDir(allocator: std.mem.Allocator) ![]u8 {
    const home = std.mem.span(std.c.getenv("HOME") orelse return error.NoHome);
    return std.fmt.allocPrint(allocator, "{s}/.mlx-serve/kv-cache", .{home});
}

// ── Small fs helpers ──

/// Bytes of an entry's files that are never shared: tokens.bin, the SSM checkpoints and the spec sidecar.
fn nonChunkBytes(e: *const IndexEntry) u64 {
    var n: u64 = @as(u64, e.tokens.len) * 4 + e.spec_bytes;
    for (e.ssm_bytes) |b| n += b;
    if (!e.inherited_qsa) n += e.qsa_history_bytes;
    return n;
}

fn clampAdd(base: u64, delta: i64) u64 {
    const v: i128 = @as(i128, base) + @as(i128, delta);
    return if (v < 0) 0 else @intCast(v);
}

/// Length of the longest common prefix of two token slices.
pub fn commonPrefixLen(a: []const u32, b: []const u32) usize {
    const n = @min(a.len, b.len);
    var i: usize = 0;
    while (i < n and a[i] == b[i]) : (i += 1) {}
    return i;
}

var chunk_share_env_cached: ?bool = null;
pub var chunk_share_override: ?bool = null;

/// `MLX_SERVE_SSD_CHUNK_SHARE=0` restores the write-everything commit (SSD-first arm only).
pub fn chunkShareEnabled() bool {
    if (chunk_share_override) |v| return v;
    if (chunk_share_env_cached) |v| return v;
    const v = blk: {
        const raw = std.c.getenv("MLX_SERVE_SSD_CHUNK_SHARE") orelse break :blk true;
        break :blk !std.mem.eql(u8, std.mem.sliceTo(raw, 0), "0");
    };
    chunk_share_env_cached = v;
    return v;
}

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

/// A manifest scalar narrowed to the field that will hold it. meta.json is on disk and
/// hand-editable, so a value that does not fit drops its record rather than being cast.
fn jsonInt(comptime T: type, obj: std.json.ObjectMap, key: []const u8) ?T {
    const v = obj.get(key) orelse return null;
    if (v != .integer) return null;
    return std.math.cast(T, v.integer);
}

/// The KV quant config a manifest describes, or null when it is not one this build can hold.
fn manifestQuant(obj: std.json.ObjectMap) ?kv_quant.KVQuantConfig {
    const scheme_v = obj.get("scheme") orelse return null;
    if (scheme_v != .string) return null;
    const scheme = std.meta.stringToEnum(kv_quant.Scheme, scheme_v.string) orelse return null;
    if (scheme == .off) return kv_quant.KVQuantConfig.dense;
    if (scheme != .affine) return null;
    const bits = jsonInt(u8, obj, "bits") orelse return null;
    if (bits != 4 and bits != 8) return null;
    const gs = jsonInt(u32, obj, "group_size") orelse return null;
    if (gs == 0) return null;
    var q = kv_quant.KVQuantConfig.affine(bits);
    q.group_size = gs;
    return q;
}

fn writeSpecMetaJson(a: std.mem.Allocator, w: *std.ArrayList(u8), name: []const u8, sm: SpecMeta) !void {
    try w.print(a, ",\"{s}\":{{\"base\":{d},\"step\":{d},\"layers\":{d},\"scheme\":\"{s}\",\"bits\":{d},\"group_size\":{d}", .{
        name, sm.base, sm.step, sm.layers, @tagName(sm.quant.scheme), sm.quant.bits, sm.quant.group_size,
    });
    if (sm.head) |h| {
        try w.print(a, ",\"head\":{{\"pos_base\":{d},\"ratio\":{d},\"pooled\":{s},\"rows\":{d}", .{
            h.pos_base, h.ratio, if (h.pooled) "true" else "false", h.rows,
        });
        if (h.mark_count > 0) {
            try w.appendSlice(a, ",\"marks\":[");
            for (h.marks[0..h.mark_count], 0..) |pos, i| {
                if (i > 0) try w.appendSlice(a, ",");
                try w.print(a, "{d}", .{pos});
            }
            try w.appendSlice(a, "]");
        }
        try w.appendSlice(a, "}");
    }
    try w.appendSlice(a, "}");
}

fn parseSpecMeta(obj: std.json.ObjectMap, key: []const u8) ?SpecMeta {
    const v = obj.get(key) orelse return null;
    if (v != .object) return null;
    const o = v.object;
    const base = jsonU64(o, "base") orelse return null;
    const step = jsonInt(u32, o, "step") orelse return null;
    const layers = jsonInt(u32, o, "layers") orelse return null;
    if (step == 0 or layers == 0) return null;
    const quant = manifestQuant(o) orelse return null;
    // v5 head half; absent on every earlier manifest.
    var head: ?SpecHeadMeta = null;
    if (o.get("head")) |hv| head_blk: {
        if (hv != .object) break :head_blk;
        const ho = hv.object;
        const pb = jsonInt(i32, ho, "pos_base") orelse break :head_blk;
        const ratio = jsonInt(i32, ho, "ratio") orelse break :head_blk;
        if (ratio == 0) break :head_blk;
        const pooled_v = ho.get("pooled") orelse break :head_blk;
        if (pooled_v != .bool) break :head_blk;
        const rows = jsonInt(i32, ho, "rows") orelse 0;
        var hm: SpecHeadMeta = .{ .pos_base = pb, .ratio = ratio, .pooled = pooled_v.bool, .rows = rows };
        // v8 leftovers; absent on every earlier manifest.
        if (ho.get("marks")) |mv| {
            if (mv == .array) {
                for (mv.array.items) |iv| {
                    if (iv != .integer or hm.mark_count == transformer_mod.QSA_HEAD_MARKS_MAX) break;
                    hm.marks[hm.mark_count] = std.math.cast(i32, iv.integer) orelse break;
                    hm.mark_count += 1;
                }
            }
        }
        head = hm;
    }
    return .{ .base = base, .step = step, .layers = layers, .quant = quant, .head = head };
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
    try testing.expectEqual(PersistOutcome.partial, c1);
    try testing.expectEqual(@as(u32, 128), tier.entries.items[0].kv_len);
    // Second flush continues from where it left off.
    const c2 = try tier.appendCommit(cache.entries, cache.step, cache.config, &tokens, false, null, s);
    try testing.expectEqual(PersistOutcome.partial, c2);
    try testing.expectEqual(@as(u32, 256), tier.entries.items[0].kv_len);
    // Keep flushing until complete; entry must land at the full 600.
    var guard: u32 = 0;
    while (guard < 10) : (guard += 1) {
        if (try tier.appendCommit(cache.entries, cache.step, cache.config, &tokens, false, null, s) == .persisted) break;
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

/// A test tensor of `shape` filled with `v` (f32). Owned by the caller.
fn filledArray(shape: []const c_int, v: f32, s: mlx.mlx_stream) !mlx.mlx_array {
    const scalar = mlx.mlx_array_new_float(v);
    defer _ = mlx.mlx_array_free(scalar);
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_full(&out, shape.ptr, shape.len, scalar, .float32, s));
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
    const aux_shape = [_]c_int{ 1, 256, 4 };
    const pooled_shape = [_]c_int{ 1, 64, 4 };
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
    try transformer_mod.attachQsaHistoryToLatest(&cps, &src256, s);

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
    try testing.expect(dst[2].aux_state.ctx == null);
    try testing.expectEqual(@as(c_int, 256), dst[2].qsa_hist_rows);
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

fn stAuxPooledBytes(allocator: std.mem.Allocator, io: std.Io, path: []const u8) !struct { aux: u64, pooled: u64 } {
    const raw = readFileAlloc(allocator, io, path, 4 * 1024 * 1024) orelse return error.TestUnexpectedResult;
    defer allocator.free(raw);
    if (raw.len < 8) return error.TestUnexpectedResult;
    const hdr_len = std.mem.readInt(u64, raw[0..8], .little);
    if (8 + hdr_len > raw.len) return error.TestUnexpectedResult;
    const hdr = raw[8 .. 8 + hdr_len];
    var parsed = try std.json.parseFromSlice(std.json.Value, allocator, hdr, .{});
    defer parsed.deinit();
    if (parsed.value != .object) return error.TestUnexpectedResult;
    var aux: u64 = 0;
    var pooled: u64 = 0;
    var it = parsed.value.object.iterator();
    while (it.next()) |e| {
        if (std.mem.eql(u8, e.key_ptr.*, "__metadata__")) continue;
        if (e.value_ptr.* != .object) continue;
        const o = e.value_ptr.*.object;
        const dtype = if (o.get("dtype")) |d| d.string else continue;
        const shape_v = o.get("shape") orelse continue;
        if (shape_v != .array) continue;
        var el: u64 = 1;
        for (shape_v.array.items) |x| {
            if (x != .integer) continue;
            el *= @intCast(x.integer);
        }
        const b: u64 = el * switch (dtype[0]) {
            'F' => if (dtype.len > 1 and dtype[1] == '3') @as(u64, 4) else 2,
            'B' => 2,
            'U', 'I' => if (dtype.len > 1 and dtype[1] == '8') @as(u64, 1) else 4,
            else => 1,
        };
        if (std.mem.indexOf(u8, e.key_ptr.*, ".aux") != null) aux += b;
        if (std.mem.indexOf(u8, e.key_ptr.*, ".pooled") != null) pooled += b;
    }
    return .{ .aux = aux, .pooled = pooled };
}

test "DiskTier: QSA history bytes are O(rows), not O(checkpoints x rows)" {
    const io = std.testing.io;
    const s = mlx.gpuStream();
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const base = try tmpRoot(&tmp, io, &buf);

    var tier = try DiskTier.init(testing.allocator, io, base, "fp-qsa-once", 0, 128);
    defer tier.deinit();

    var cache = try KVCache.init(testing.allocator, 3);
    defer cache.deinit();
    try fillCache(&cache, s, 3, 600, 8, 0.0, .float32);
    var tokens: [600]u32 = undefined;
    for (&tokens, 0..) |*t, i| t.* = @intCast(i + 7);

    const aux_shape = [_]c_int{ 1, 256, 8 };
    const pooled_shape = [_]c_int{ 1, 64, 8 };
    var src = buildHybridEntries(s, 200.0, 600.0);
    defer freeHybridEntries(&src);
    src[2].aux_state = makeArange(s, &aux_shape, 700.0);
    src[2].qsa_pooled = makeArange(s, &pooled_shape, 800.0);
    src[2].qsa_ratio = 4;
    var cps = [_]transformer_mod.SSMCheckpoint{
        try transformer_mod.captureSsmCheckpoint(testing.allocator, &src, 128, s),
        try transformer_mod.captureSsmCheckpoint(testing.allocator, &src, 256, s),
    };
    defer for (&cps) |*cp| cp.deinit(testing.allocator);
    try transformer_mod.attachQsaHistoryToLatest(&cps, &src, s);

    _ = try tier.appendCommit(cache.entries, cache.step, cache.config, &tokens, false, &cps, s);

    const s128 = try std.fmt.allocPrint(testing.allocator, "{s}/e{d}/s0000128.safetensors", .{ tier.root, tier.entries.items[0].id });
    defer testing.allocator.free(s128);
    const s256 = try std.fmt.allocPrint(testing.allocator, "{s}/e{d}/s0000256.safetensors", .{ tier.root, tier.entries.items[0].id });
    defer testing.allocator.free(s256);
    const qsa_path = try std.fmt.allocPrint(testing.allocator, "{s}/e{d}/qsa.safetensors", .{ tier.root, tier.entries.items[0].id });
    defer testing.allocator.free(qsa_path);
    const a128 = try stAuxPooledBytes(testing.allocator, io, s128);
    const a256 = try stAuxPooledBytes(testing.allocator, io, s256);
    try testing.expectEqual(@as(u64, 0), a128.aux + a128.pooled);
    try testing.expectEqual(@as(u64, 0), a256.aux + a256.pooled);
    const hq = try stAuxPooledBytes(testing.allocator, io, qsa_path);
    const one: u64 = 64 * 8 * 4;
    try testing.expectEqual(one, hq.aux + hq.pooled);

    var tier2 = try DiskTier.init(testing.allocator, io, base, "fp-qsa-once", 0, 128);
    defer tier2.deinit();
    var cache2 = try KVCache.init(testing.allocator, 3);
    defer cache2.deinit();
    var dst: [3]SSMCacheEntry = .{
        .{ .conv_state = mlx.mlx_array_new(), .ssm_state = mlx.mlx_array_new(), .initialized = false },
        .{ .conv_state = mlx.mlx_array_new(), .ssm_state = mlx.mlx_array_new(), .initialized = false },
        .{ .conv_state = mlx.mlx_array_new(), .ssm_state = mlx.mlx_array_new(), .initialized = false },
    };
    defer freeHybridEntries(&dst);
    try testing.expectEqual(@as(u32, 128), try tier2.restoreIntoHybrid(&cache2, &dst, 0, 128, s));
    try testing.expectEqual(@as(c_int, 128), dst[2].qsa_hist_rows);
    try testing.expectEqual(@as(c_int, 32), mlx.getShape(dst[2].qsa_pooled)[1]);
}

test "DiskTier: a mid-block checkpoint restore overlays the pooled bank onto its own leftover" {
    const io = std.testing.io;
    const s = mlx.gpuStream();
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const base = try tmpRoot(&tmp, io, &buf);

    var tier = try DiskTier.init(testing.allocator, io, base, "fp-qsa-mid", 0, 128);
    defer tier.deinit();

    var cache = try KVCache.init(testing.allocator, 3);
    defer cache.deinit();
    try fillCache(&cache, s, 3, 600, 8, 0.0, .float32);
    var tokens: [600]u32 = undefined;
    for (&tokens, 0..) |*t, i| t.* = @intCast(i + 7);

    const aux_shape = [_]c_int{ 1, 256, 8 };
    const pooled_shape = [_]c_int{ 1, 64, 8 };
    var src = buildHybridEntries(s, 200.0, 600.0);
    defer freeHybridEntries(&src);
    src[2].aux_state = makeArange(s, &aux_shape, 700.0);
    src[2].qsa_pooled = makeArange(s, &pooled_shape, 800.0);
    src[2].qsa_ratio = 4;
    // 46 is not a multiple of the ratio, so this checkpoint carries a 2-row leftover — the
    // shape `SSM_SNAPSHOT_BACKOFF` puts the always-on snapshot at.
    var cps = [_]transformer_mod.SSMCheckpoint{
        try transformer_mod.captureSsmCheckpoint(testing.allocator, &src, 46, s),
        try transformer_mod.captureSsmCheckpoint(testing.allocator, &src, 256, s),
    };
    defer for (&cps) |*cp| cp.deinit(testing.allocator);
    try transformer_mod.attachQsaHistoryToLatest(&cps, &src, s);
    _ = try tier.appendCommit(cache.entries, cache.step, cache.config, &tokens, false, &cps, s);

    var tier2 = try DiskTier.init(testing.allocator, io, base, "fp-qsa-mid", 0, 128);
    defer tier2.deinit();
    var cache2 = try KVCache.init(testing.allocator, 3);
    defer cache2.deinit();
    var dst: [3]SSMCacheEntry = .{
        .{ .conv_state = mlx.mlx_array_new(), .ssm_state = mlx.mlx_array_new(), .initialized = false },
        .{ .conv_state = mlx.mlx_array_new(), .ssm_state = mlx.mlx_array_new(), .initialized = false },
        .{ .conv_state = mlx.mlx_array_new(), .ssm_state = mlx.mlx_array_new(), .initialized = false },
    };
    defer freeHybridEntries(&dst);
    try testing.expectEqual(@as(u32, 46), try tier2.restoreIntoHybrid(&cache2, &dst, 0, 46, s));
    try testing.expect(transformer_mod.entriesHaveQsaHistory(&dst));
    try testing.expectEqual(@as(c_int, 46), dst[2].qsa_hist_rows);
    // Pooled blocks 0..46/4 come from the history file; the leftover is the checkpoint's own.
    try testing.expect(dst[2].qsa_pooled.ctx != null);
    try testing.expectEqual(@as(c_int, 11), mlx.getShape(dst[2].qsa_pooled)[1]);
    try testing.expectEqual(@as(f32, 800.0), ssmArrVal(dst[2].qsa_pooled, 0, s));
    try testing.expectEqual(@as(f32, 880.0), ssmArrVal(dst[2].qsa_pooled, 80, s));
    try testing.expectEqual(@as(c_int, 2), mlx.getShape(dst[2].aux_state)[1]);
    try testing.expectEqual(@as(f32, 1052.0), ssmArrVal(dst[2].aux_state, 0, s));
}

test "DiskTier: a GDN-layer aux window is persisted in the checkpoint file" {
    const io = std.testing.io;
    const s = mlx.gpuStream();
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const base = try tmpRoot(&tmp, io, &buf);

    var tier = try DiskTier.init(testing.allocator, io, base, "fp-ple-aux", 0, 128);
    defer tier.deinit();

    var cache = try KVCache.init(testing.allocator, 3);
    defer cache.deinit();
    try fillCache(&cache, s, 3, 600, 8, 0.0, .float32);
    var tokens: [600]u32 = undefined;
    for (&tokens, 0..) |*t, i| t.* = @intCast(i + 7);

    var src = buildHybridEntries(s, 100.0, 500.0);
    defer freeHybridEntries(&src);
    const ple_shape = [_]c_int{ 1, 3, 8 };
    src[0].aux_state = makeArange(s, &ple_shape, 900.0);
    var cps = [_]transformer_mod.SSMCheckpoint{
        try transformer_mod.captureSsmCheckpoint(testing.allocator, &src, 128, s),
    };
    defer for (&cps) |*cp| cp.deinit(testing.allocator);
    try testing.expect(cps[0].layers[0].aux_state.ctx != null);
    try testing.expect(!transformer_mod.checkpointHasQsaHistory(&cps[0]));

    _ = try tier.appendCommit(cache.entries, cache.step, cache.config, &tokens, false, &cps, s);
    const sp = try std.fmt.allocPrint(testing.allocator, "{s}/e{d}/s0000128.safetensors", .{ tier.root, tier.entries.items[0].id });
    defer testing.allocator.free(sp);
    const fam = try stAuxPooledBytes(testing.allocator, io, sp);
    try testing.expectEqual(@as(u64, 3 * 8 * 4), fam.aux);

    var cache2 = try KVCache.init(testing.allocator, 3);
    defer cache2.deinit();
    var dst: [3]SSMCacheEntry = .{
        .{ .conv_state = mlx.mlx_array_new(), .ssm_state = mlx.mlx_array_new(), .initialized = false },
        .{ .conv_state = mlx.mlx_array_new(), .ssm_state = mlx.mlx_array_new(), .initialized = false },
        .{ .conv_state = mlx.mlx_array_new(), .ssm_state = mlx.mlx_array_new(), .initialized = false },
    };
    defer freeHybridEntries(&dst);
    try testing.expectEqual(@as(u32, 128), try tier.restoreIntoHybrid(&cache2, &dst, 0, 128, s));
    try testing.expectEqual(@as(f32, 900.0), ssmArrVal(dst[0].aux_state, 0, s));
    try testing.expectEqual(@as(f32, 900.0 + 23.0), ssmArrVal(dst[0].aux_state, 23, s));
}

test "DiskTier: a pre-v7 interior restore overlays QSA history from the latest s* file" {
    const io = std.testing.io;
    const s = mlx.gpuStream();
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const base = try tmpRoot(&tmp, io, &buf);

    var tier = try DiskTier.init(testing.allocator, io, base, "fp-prev7", 0, 128);
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
    const aux_shape = [_]c_int{ 1, 256, 8 };
    const pooled_shape = [_]c_int{ 1, 64, 8 };
    src256[2].aux_state = makeArange(s, &aux_shape, 700.0);
    src256[2].qsa_pooled = makeArange(s, &pooled_shape, 800.0);
    src256[2].qsa_ratio = 4;
    var cps = [_]transformer_mod.SSMCheckpoint{
        try transformer_mod.captureSsmCheckpoint(testing.allocator, &src128, 128, s),
        try transformer_mod.captureSsmCheckpoint(testing.allocator, &src256, 256, s),
    };
    defer for (&cps) |*cp| cp.deinit(testing.allocator);
    try transformer_mod.attachQsaHistoryToLatest(&cps, &src256, s);
    _ = try tier.appendCommit(cache.entries, cache.step, cache.config, &tokens, false, &cps, s);

    const id = tier.entries.items[0].id;
    var latest = try tier.loadSsmFile(id, 256, 3);
    defer latest.deinit(testing.allocator);
    var qsa = try tier.loadQsaHistoryFile(id, 3);
    defer qsa.deinit(testing.allocator);
    for (latest.layers, qsa.layers) |*l, q| {
        if (q.aux_state.ctx != null) {
            if (l.aux_state.ctx != null) _ = mlx.mlx_array_free(l.aux_state);
            l.aux_state = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_array_set(&l.aux_state, q.aux_state));
        }
        if (q.qsa_pooled.ctx != null) {
            if (l.qsa_pooled.ctx != null) _ = mlx.mlx_array_free(l.qsa_pooled);
            l.qsa_pooled = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_array_set(&l.qsa_pooled, q.qsa_pooled));
        }
        l.qsa_ratio = q.qsa_ratio;
    }
    const dir_rel = try std.fmt.allocPrint(testing.allocator, "{s}/e{d}", .{ tier.root, id });
    defer testing.allocator.free(dir_rel);
    var interior = try transformer_mod.captureSsmCheckpoint(testing.allocator, &src128, 128, s);
    defer interior.deinit(testing.allocator);
    _ = try tier.writeSsmFile(dir_rel, &interior, s);
    test_ssm_write_qsa_aux = true;
    defer test_ssm_write_qsa_aux = false;
    _ = try tier.writeSsmFile(dir_rel, &latest, s);
    const qsa_path = try std.fmt.allocPrint(testing.allocator, "{s}/e{d}/qsa.safetensors", .{ tier.root, id });
    defer testing.allocator.free(qsa_path);
    try std.Io.Dir.deleteFileAbsolute(io, qsa_path);
    tier.entries.items[0].qsa_history_bytes = 0;
    tier.entries.items[0].qsa_history_rows = 0;
    tier.entries.items[0].inherited_qsa = false;
    try tier.writeMeta(tier.entries.items[0]);

    var cache2 = try KVCache.init(testing.allocator, 3);
    defer cache2.deinit();
    var dst: [3]SSMCacheEntry = .{
        .{ .conv_state = mlx.mlx_array_new(), .ssm_state = mlx.mlx_array_new(), .initialized = false },
        .{ .conv_state = mlx.mlx_array_new(), .ssm_state = mlx.mlx_array_new(), .initialized = false },
        .{ .conv_state = mlx.mlx_array_new(), .ssm_state = mlx.mlx_array_new(), .initialized = false },
    };
    defer freeHybridEntries(&dst);
    try testing.expectEqual(@as(u32, 128), try tier.restoreIntoHybrid(&cache2, &dst, 0, 128, s));
    try testing.expect(transformer_mod.entriesHaveQsaHistory(&dst));
    try testing.expectEqual(@as(c_int, 128), dst[2].qsa_hist_rows);
}

test "DiskTier: a pre-v7 interior restore misses when the latest s* file is gone" {
    const io = std.testing.io;
    const s = mlx.gpuStream();
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const base = try tmpRoot(&tmp, io, &buf);

    var tier = try DiskTier.init(testing.allocator, io, base, "fp-prev7-miss", 0, 128);
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
    const aux_shape = [_]c_int{ 1, 256, 8 };
    const pooled_shape = [_]c_int{ 1, 64, 8 };
    src256[2].aux_state = makeArange(s, &aux_shape, 700.0);
    src256[2].qsa_pooled = makeArange(s, &pooled_shape, 800.0);
    src256[2].qsa_ratio = 4;
    var cps = [_]transformer_mod.SSMCheckpoint{
        try transformer_mod.captureSsmCheckpoint(testing.allocator, &src128, 128, s),
        try transformer_mod.captureSsmCheckpoint(testing.allocator, &src256, 256, s),
    };
    defer for (&cps) |*cp| cp.deinit(testing.allocator);
    try transformer_mod.attachQsaHistoryToLatest(&cps, &src256, s);
    _ = try tier.appendCommit(cache.entries, cache.step, cache.config, &tokens, false, &cps, s);

    const id = tier.entries.items[0].id;
    var latest = try tier.loadSsmFile(id, 256, 3);
    defer latest.deinit(testing.allocator);
    var qsa = try tier.loadQsaHistoryFile(id, 3);
    defer qsa.deinit(testing.allocator);
    for (latest.layers, qsa.layers) |*l, q| {
        if (q.aux_state.ctx != null) {
            if (l.aux_state.ctx != null) _ = mlx.mlx_array_free(l.aux_state);
            l.aux_state = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_array_set(&l.aux_state, q.aux_state));
        }
        if (q.qsa_pooled.ctx != null) {
            if (l.qsa_pooled.ctx != null) _ = mlx.mlx_array_free(l.qsa_pooled);
            l.qsa_pooled = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_array_set(&l.qsa_pooled, q.qsa_pooled));
        }
        l.qsa_ratio = q.qsa_ratio;
    }
    const dir_rel = try std.fmt.allocPrint(testing.allocator, "{s}/e{d}", .{ tier.root, id });
    defer testing.allocator.free(dir_rel);
    var interior = try transformer_mod.captureSsmCheckpoint(testing.allocator, &src128, 128, s);
    defer interior.deinit(testing.allocator);
    _ = try tier.writeSsmFile(dir_rel, &interior, s);
    test_ssm_write_qsa_aux = true;
    defer test_ssm_write_qsa_aux = false;
    _ = try tier.writeSsmFile(dir_rel, &latest, s);
    const qsa_path = try std.fmt.allocPrint(testing.allocator, "{s}/e{d}/qsa.safetensors", .{ tier.root, id });
    defer testing.allocator.free(qsa_path);
    try std.Io.Dir.deleteFileAbsolute(io, qsa_path);
    const latest_path = try std.fmt.allocPrint(testing.allocator, "{s}/e{d}/s0000256.safetensors", .{ tier.root, id });
    defer testing.allocator.free(latest_path);
    try std.Io.Dir.deleteFileAbsolute(io, latest_path);
    tier.entries.items[0].qsa_history_bytes = 0;
    tier.entries.items[0].qsa_history_rows = 0;
    tier.entries.items[0].inherited_qsa = false;
    try tier.writeMeta(tier.entries.items[0]);

    var cache2 = try KVCache.init(testing.allocator, 3);
    defer cache2.deinit();
    var dst: [3]SSMCacheEntry = .{
        .{ .conv_state = mlx.mlx_array_new(), .ssm_state = mlx.mlx_array_new(), .initialized = false },
        .{ .conv_state = mlx.mlx_array_new(), .ssm_state = mlx.mlx_array_new(), .initialized = false },
        .{ .conv_state = mlx.mlx_array_new(), .ssm_state = mlx.mlx_array_new(), .initialized = false },
    };
    defer freeHybridEntries(&dst);
    try testing.expectError(error.DiskCacheQsaHistoryGap, tier.restoreIntoHybrid(&cache2, &dst, 0, 128, s));
    try testing.expectEqual(@as(usize, 0), cache2.step);
    try testing.expect(dst[2].aux_state.ctx == null);
}

test "DiskTier: a v7 entry with missing or short qsa.safetensors misses before KV/SSM adopt" {
    const io = std.testing.io;
    const s = mlx.gpuStream();
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const base = try tmpRoot(&tmp, io, &buf);

    var cache = try KVCache.init(testing.allocator, 3);
    defer cache.deinit();
    try fillCache(&cache, s, 3, 600, 8, 0.0, .float32);
    var tokens: [600]u32 = undefined;
    for (&tokens, 0..) |*t, i| t.* = @intCast(i + 7);

    const aux_shape = [_]c_int{ 1, 256, 8 };
    const pooled_shape = [_]c_int{ 1, 64, 8 };
    var src = buildHybridEntries(s, 200.0, 600.0);
    defer freeHybridEntries(&src);
    src[2].aux_state = makeArange(s, &aux_shape, 700.0);
    src[2].qsa_pooled = makeArange(s, &pooled_shape, 800.0);
    src[2].qsa_ratio = 4;
    var cps = [_]transformer_mod.SSMCheckpoint{
        try transformer_mod.captureSsmCheckpoint(testing.allocator, &src, 128, s),
        try transformer_mod.captureSsmCheckpoint(testing.allocator, &src, 256, s),
    };
    defer for (&cps) |*cp| cp.deinit(testing.allocator);
    try transformer_mod.attachQsaHistoryToLatest(&cps, &src, s);

    {
        var tier = try DiskTier.init(testing.allocator, io, base, "fp-qsa-gap", 0, 128);
        defer tier.deinit();
        _ = try tier.appendCommit(cache.entries, cache.step, cache.config, &tokens, false, &cps, s);
        const id = tier.entries.items[0].id;
        const qsa_path = try std.fmt.allocPrint(testing.allocator, "{s}/e{d}/qsa.safetensors", .{ tier.root, id });
        defer testing.allocator.free(qsa_path);
        try std.Io.Dir.deleteFileAbsolute(io, qsa_path);

        var cache2 = try KVCache.init(testing.allocator, 3);
        defer cache2.deinit();
        var dst: [3]SSMCacheEntry = .{
            .{ .conv_state = mlx.mlx_array_new(), .ssm_state = mlx.mlx_array_new(), .initialized = false },
            .{ .conv_state = mlx.mlx_array_new(), .ssm_state = mlx.mlx_array_new(), .initialized = false },
            .{ .conv_state = mlx.mlx_array_new(), .ssm_state = mlx.mlx_array_new(), .initialized = false },
        };
        defer freeHybridEntries(&dst);
        try testing.expectError(error.DiskCacheQsaHistoryGap, tier.restoreIntoHybrid(&cache2, &dst, 0, 128, s));
        try testing.expectEqual(@as(usize, 0), cache2.step);
        try testing.expect(dst[2].aux_state.ctx == null);
    }

    {
        var tier = try DiskTier.init(testing.allocator, io, base, "fp-qsa-short", 0, 128);
        defer tier.deinit();
        _ = try tier.appendCommit(cache.entries, cache.step, cache.config, &tokens, false, &cps, s);
        const id = tier.entries.items[0].id;
        var src64 = buildHybridEntries(s, 200.0, 600.0);
        defer freeHybridEntries(&src64);
        const aux64 = [_]c_int{ 1, 64, 8 };
        const pooled64 = [_]c_int{ 1, 16, 8 };
        src64[2].aux_state = makeArange(s, &aux64, 700.0);
        src64[2].qsa_pooled = makeArange(s, &pooled64, 800.0);
        src64[2].qsa_ratio = 4;
        var cps64 = [_]transformer_mod.SSMCheckpoint{
            try transformer_mod.captureSsmCheckpoint(testing.allocator, &src64, 64, s),
        };
        defer for (&cps64) |*cp| cp.deinit(testing.allocator);
        try transformer_mod.attachQsaHistoryToLatest(&cps64, &src64, s);
        const dir_rel = try std.fmt.allocPrint(testing.allocator, "{s}/e{d}", .{ tier.root, id });
        defer testing.allocator.free(dir_rel);
        _ = try tier.writeQsaHistoryFile(dir_rel, &cps64[0], s);

        var cache2 = try KVCache.init(testing.allocator, 3);
        defer cache2.deinit();
        var dst: [3]SSMCacheEntry = .{
            .{ .conv_state = mlx.mlx_array_new(), .ssm_state = mlx.mlx_array_new(), .initialized = false },
            .{ .conv_state = mlx.mlx_array_new(), .ssm_state = mlx.mlx_array_new(), .initialized = false },
            .{ .conv_state = mlx.mlx_array_new(), .ssm_state = mlx.mlx_array_new(), .initialized = false },
        };
        defer freeHybridEntries(&dst);
        try testing.expectError(error.DiskCacheQsaHistoryGap, tier.restoreIntoHybrid(&cache2, &dst, 0, 128, s));
        try testing.expectEqual(@as(usize, 0), cache2.step);
        try testing.expect(dst[2].aux_state.ctx == null);
    }

    {
        var tier = try DiskTier.init(testing.allocator, io, base, "fp-qsa-wt", 0, 128);
        defer tier.deinit();
        var wt_cps = [_]transformer_mod.SSMCheckpoint{
            try transformer_mod.captureSsmCheckpoint(testing.allocator, &src, 128, s),
            try transformer_mod.captureSsmCheckpoint(testing.allocator, &src, 256, s),
        };
        defer for (&wt_cps) |*cp| cp.deinit(testing.allocator);
        _ = try tier.appendCommit(cache.entries, cache.step, cache.config, &tokens, false, &wt_cps, s);
        try testing.expectEqual(@as(u32, 0), tier.entries.items[0].qsa_history_rows);

        var cache2 = try KVCache.init(testing.allocator, 3);
        defer cache2.deinit();
        var dst: [3]SSMCacheEntry = .{
            .{ .conv_state = mlx.mlx_array_new(), .ssm_state = mlx.mlx_array_new(), .initialized = false },
            .{ .conv_state = mlx.mlx_array_new(), .ssm_state = mlx.mlx_array_new(), .initialized = false },
            .{ .conv_state = mlx.mlx_array_new(), .ssm_state = mlx.mlx_array_new(), .initialized = false },
        };
        defer freeHybridEntries(&dst);
        try testing.expectError(error.DiskCacheQsaHistoryGap, tier.restoreIntoHybrid(&cache2, &dst, 0, 128, s));
        try testing.expectEqual(@as(usize, 0), cache2.step);
        try testing.expect(dst[2].aux_state.ctx == null);
    }
}

test "DiskTier: a history tensor shorter than cp_pos is a miss, not a short hit" {
    const io = std.testing.io;
    const s = mlx.gpuStream();
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const base = try tmpRoot(&tmp, io, &buf);

    var tier = try DiskTier.init(testing.allocator, io, base, "fp-qsa-short-tensor", 0, 128);
    defer tier.deinit();

    var cache = try KVCache.init(testing.allocator, 3);
    defer cache.deinit();
    try fillCache(&cache, s, 3, 600, 8, 0.0, .float32);
    var tokens: [600]u32 = undefined;
    for (&tokens, 0..) |*t, i| t.* = @intCast(i + 7);

    const aux_shape = [_]c_int{ 1, 8, 8 };
    const pooled_shape = [_]c_int{ 1, 2, 8 };
    var src = buildHybridEntries(s, 200.0, 600.0);
    defer freeHybridEntries(&src);
    src[2].aux_state = makeArange(s, &aux_shape, 700.0);
    src[2].qsa_pooled = makeArange(s, &pooled_shape, 800.0);
    src[2].qsa_ratio = 4;
    var cps = [_]transformer_mod.SSMCheckpoint{
        try transformer_mod.captureSsmCheckpoint(testing.allocator, &src, 16, s),
        try transformer_mod.captureSsmCheckpoint(testing.allocator, &src, 64, s),
    };
    defer for (&cps) |*cp| cp.deinit(testing.allocator);
    try testing.expectEqual(@as(c_int, 16), cps[0].layers[2].qsa_rows);
    try testing.expectEqual(@as(c_int, 64), cps[1].layers[2].qsa_rows);
    try transformer_mod.attachQsaHistoryToLatest(&cps, &src, s);
    try testing.expect(cps[1].layers[2].aux_state.ctx == null);
    try testing.expectEqual(@as(c_int, 64), cps[1].layers[2].qsa_rows);

    _ = try tier.appendCommit(cache.entries, cache.step, cache.config, &tokens, false, &cps, s);

    var cache2 = try KVCache.init(testing.allocator, 3);
    defer cache2.deinit();
    var dst: [3]SSMCacheEntry = .{
        .{ .conv_state = mlx.mlx_array_new(), .ssm_state = mlx.mlx_array_new(), .initialized = false },
        .{ .conv_state = mlx.mlx_array_new(), .ssm_state = mlx.mlx_array_new(), .initialized = false },
        .{ .conv_state = mlx.mlx_array_new(), .ssm_state = mlx.mlx_array_new(), .initialized = false },
    };
    defer freeHybridEntries(&dst);
    try testing.expectError(error.DiskCacheQsaHistoryGap, tier.restoreIntoHybrid(&cache2, &dst, 0, 16, s));
    try testing.expectEqual(@as(usize, 0), cache2.step);
    try testing.expect(dst[2].aux_state.ctx == null);
}

test "DiskTier: a QSA overlay error resets the half-built restore" {
    const io = std.testing.io;
    const s = mlx.gpuStream();
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const base = try tmpRoot(&tmp, io, &buf);

    var tier = try DiskTier.init(testing.allocator, io, base, "fp-qsa-ovl", 0, 128);
    defer tier.deinit();

    var cache = try KVCache.init(testing.allocator, 3);
    defer cache.deinit();
    try fillCache(&cache, s, 3, 600, 8, 0.0, .float32);
    var tokens: [600]u32 = undefined;
    for (&tokens, 0..) |*t, i| t.* = @intCast(i + 7);

    const aux_shape = [_]c_int{ 1, 256, 8 };
    const pooled_shape = [_]c_int{ 1, 64, 8 };
    var src = buildHybridEntries(s, 200.0, 600.0);
    defer freeHybridEntries(&src);
    src[2].aux_state = makeArange(s, &aux_shape, 700.0);
    src[2].qsa_pooled = makeArange(s, &pooled_shape, 800.0);
    src[2].qsa_ratio = 4;
    var cps = [_]transformer_mod.SSMCheckpoint{
        try transformer_mod.captureSsmCheckpoint(testing.allocator, &src, 256, s),
    };
    defer for (&cps) |*cp| cp.deinit(testing.allocator);
    try transformer_mod.attachQsaHistoryToLatest(&cps, &src, s);
    _ = try tier.appendCommit(cache.entries, cache.step, cache.config, &tokens, false, &cps, s);

    var cache2 = try KVCache.init(testing.allocator, 3);
    defer cache2.deinit();
    var dst: [3]SSMCacheEntry = .{
        .{ .conv_state = mlx.mlx_array_new(), .ssm_state = mlx.mlx_array_new(), .initialized = false },
        .{ .conv_state = mlx.mlx_array_new(), .ssm_state = mlx.mlx_array_new(), .initialized = false },
        .{ .conv_state = mlx.mlx_array_new(), .ssm_state = mlx.mlx_array_new(), .initialized = false },
    };
    defer freeHybridEntries(&dst);
    test_qsa_overlay_mismatch = true;
    defer test_qsa_overlay_mismatch = false;
    try testing.expectError(error.SsmCheckpointLayerMismatch, tier.restoreIntoHybrid(&cache2, &dst, 0, 256, s));
    try testing.expectEqual(@as(usize, 0), cache2.step);
}

test "DiskTier chunk share: a prefix-diverging entry hard-links qsa.safetensors" {
    const io = std.testing.io;
    const s = mlx.gpuStream();
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const base = try tmpRoot(&tmp, io, &buf);

    chunk_share_override = true;
    defer chunk_share_override = null;
    var tier = try DiskTier.init(testing.allocator, io, base, "fp-qsa-share", 0, 128);
    defer tier.deinit();
    tier.ssd_first = true;
    tier.armTestSpace(1024 * 1024 * 1024 * 1024, 2048 * 1024 * 1024 * 1024);

    var cache = try KVCache.init(testing.allocator, 3);
    defer cache.deinit();
    try fillCache(&cache, s, 3, 600, 8, 0.0, .float32);
    const toks = chunkShareTokens(520);

    const aux_shape = [_]c_int{ 1, 256, 8 };
    const pooled_shape = [_]c_int{ 1, 64, 8 };
    var src = buildHybridEntries(s, 200.0, 600.0);
    defer freeHybridEntries(&src);
    src[2].aux_state = makeArange(s, &aux_shape, 700.0);
    src[2].qsa_pooled = makeArange(s, &pooled_shape, 800.0);
    src[2].qsa_ratio = 4;
    var cps = [_]transformer_mod.SSMCheckpoint{
        try transformer_mod.captureSsmCheckpoint(testing.allocator, &src, 256, s),
    };
    defer for (&cps) |*cp| cp.deinit(testing.allocator);
    try transformer_mod.attachQsaHistoryToLatest(&cps, &src, s);

    _ = try tier.appendCommit(cache.entries, cache.step, cache.config, &toks.a, false, &cps, s);
    const donor_id = tier.entries.items[0].id;
    try testing.expect(tier.entries.items[0].qsa_history_bytes > 0);
    try testing.expectEqual(@as(u32, 256), tier.entries.items[0].qsa_history_rows);

    _ = try tier.appendCommit(cache.entries, cache.step, cache.config, &toks.b, false, &cps, s);
    try testing.expectEqual(@as(usize, 2), tier.entryCount());
    const heir = &tier.entries.items[1];
    try testing.expect(heir.inherited_qsa);
    try testing.expectEqual(@as(u8, 8), DiskTier.metaVersionFor(heir.*));

    var pbuf: [1024]u8 = undefined;
    const dp = try std.fmt.bufPrint(&pbuf, "{s}/fp-qsa-share/e{d}/qsa.safetensors", .{ base, donor_id });
    const dstat = statFile(io, dp).?;
    var hbuf: [1024]u8 = undefined;
    const hp = try std.fmt.bufPrint(&hbuf, "{s}/fp-qsa-share/e{d}/qsa.safetensors", .{ base, heir.id });
    const hstat = statFile(io, hp).?;
    try testing.expectEqual(dstat.inode, hstat.inode);
    try testing.expectEqual(@as(u64, 2), @as(u64, @intCast(hstat.nlink)));

    var cache2 = try KVCache.init(testing.allocator, 3);
    defer cache2.deinit();
    var dst: [3]SSMCacheEntry = .{
        .{ .conv_state = mlx.mlx_array_new(), .ssm_state = mlx.mlx_array_new(), .initialized = false },
        .{ .conv_state = mlx.mlx_array_new(), .ssm_state = mlx.mlx_array_new(), .initialized = false },
        .{ .conv_state = mlx.mlx_array_new(), .ssm_state = mlx.mlx_array_new(), .initialized = false },
    };
    defer freeHybridEntries(&dst);
    const m = tier.bestMatch(&toks.b, false, kv_quant.KVQuantConfig.dense).?;
    try testing.expectEqual(@as(u32, 256), try tier.restoreIntoHybrid(&cache2, &dst, m.idx, 256, s));
    try testing.expect(dst[2].aux_state.ctx == null);
    try testing.expectEqual(@as(c_int, 256), dst[2].qsa_hist_rows);
}

test "DiskTier: an inherited QSA history is dropped on extend past the common prefix" {
    const io = std.testing.io;
    const s = mlx.gpuStream();
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const base = try tmpRoot(&tmp, io, &buf);

    chunk_share_override = true;
    defer chunk_share_override = null;
    var tier = try DiskTier.init(testing.allocator, io, base, "fp-qsa-ext", 0, 128);
    defer tier.deinit();
    tier.ssd_first = true;
    tier.armTestSpace(1024 * 1024 * 1024 * 1024, 2048 * 1024 * 1024 * 1024);

    var cache = try KVCache.init(testing.allocator, 3);
    defer cache.deinit();
    try fillCache(&cache, s, 3, 600, 8, 0.0, .float32);
    const toks = chunkShareTokens(520);

    var donor_src = buildHybridEntries(s, 200.0, 600.0);
    defer freeHybridEntries(&donor_src);
    const donor_aux = [_]c_int{ 1, 600, 8 };
    const donor_pooled = [_]c_int{ 1, 150, 8 };
    donor_src[2].aux_state = makeArange(s, &donor_aux, 700.0);
    donor_src[2].qsa_pooled = makeArange(s, &donor_pooled, 800.0);
    donor_src[2].qsa_ratio = 4;
    var donor_cps = [_]transformer_mod.SSMCheckpoint{
        try transformer_mod.captureSsmCheckpoint(testing.allocator, &donor_src, 600, s),
    };
    defer for (&donor_cps) |*cp| cp.deinit(testing.allocator);
    try transformer_mod.attachQsaHistoryToLatest(&donor_cps, &donor_src, s);
    _ = try tier.appendCommit(cache.entries, cache.step, cache.config, &toks.a, false, &donor_cps, s);

    var heir_src = buildHybridEntries(s, 200.0, 600.0);
    defer freeHybridEntries(&heir_src);
    const heir_aux1 = [_]c_int{ 1, 256, 8 };
    const heir_pooled1 = [_]c_int{ 1, 64, 8 };
    heir_src[2].aux_state = makeArange(s, &heir_aux1, 700.0);
    heir_src[2].qsa_pooled = makeArange(s, &heir_pooled1, 800.0);
    heir_src[2].qsa_ratio = 4;
    var heir_cps1 = [_]transformer_mod.SSMCheckpoint{
        try transformer_mod.captureSsmCheckpoint(testing.allocator, &heir_src, 256, s),
    };
    defer for (&heir_cps1) |*cp| cp.deinit(testing.allocator);
    try transformer_mod.attachQsaHistoryToLatest(&heir_cps1, &heir_src, s);
    _ = try tier.appendCommit(cache.entries, cache.step, cache.config, &toks.b, false, &heir_cps1, s);
    try testing.expect(tier.entries.items[1].inherited_qsa);

    var heir_src2 = buildHybridEntries(s, 200.0, 600.0);
    defer freeHybridEntries(&heir_src2);
    const heir_aux2 = [_]c_int{ 1, 560, 8 };
    const heir_pooled2 = [_]c_int{ 1, 140, 8 };
    heir_src2[2].aux_state = makeArange(s, &heir_aux2, 900.0);
    heir_src2[2].qsa_pooled = makeArange(s, &heir_pooled2, 1000.0);
    heir_src2[2].qsa_ratio = 4;
    var heir_cps2 = [_]transformer_mod.SSMCheckpoint{
        try transformer_mod.captureSsmCheckpoint(testing.allocator, &heir_src2, 256, s),
        try transformer_mod.captureSsmCheckpoint(testing.allocator, &heir_src2, 560, s),
    };
    defer for (&heir_cps2) |*cp| cp.deinit(testing.allocator);
    try transformer_mod.attachQsaHistoryToLatest(&heir_cps2, &heir_src2, s);
    _ = try tier.appendCommit(cache.entries, cache.step, cache.config, &toks.b, false, &heir_cps2, s);

    const heir = &tier.entries.items[1];
    try testing.expect(!heir.inherited_qsa);
    var cache2 = try KVCache.init(testing.allocator, 3);
    defer cache2.deinit();
    var dst: [3]SSMCacheEntry = .{
        .{ .conv_state = mlx.mlx_array_new(), .ssm_state = mlx.mlx_array_new(), .initialized = false },
        .{ .conv_state = mlx.mlx_array_new(), .ssm_state = mlx.mlx_array_new(), .initialized = false },
        .{ .conv_state = mlx.mlx_array_new(), .ssm_state = mlx.mlx_array_new(), .initialized = false },
    };
    defer freeHybridEntries(&dst);
    const m = tier.bestMatch(&toks.b, false, kv_quant.KVQuantConfig.dense).?;
    try testing.expectEqual(@as(u32, 560), try tier.restoreIntoHybrid(&cache2, &dst, m.idx, 560, s));
    try testing.expect(dst[2].aux_state.ctx == null);
    try testing.expectEqual(@as(c_int, 560), dst[2].qsa_hist_rows);
}

test "DiskTier: SSM retention thins the interior, keeping both ends" {
    // Every turn adds an end-of-prompt checkpoint; unbounded, one entry grows
    // without limit. Retention keeps at most SSM_DISK_MAX_PER_ENTRY, thinning
    // the interior (#330): front-thinning end-anchored the survivors.
    const io = std.testing.io;
    const s = mlx.gpuStream();
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const base = try tmpRoot(&tmp, io, &buf);

    var tier = try DiskTier.init(testing.allocator, io, base, "fp-retain", 0, 128);
    defer tier.deinit();
    // The span-preserving policy is qwen4_exp's; the tier's default is drop-oldest (below).
    tier.cp_thin = .min_span_recency;
    tier.ssm_max_per_entry = SSM_DISK_MAX_PER_ENTRY;

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

    const e = &tier.entries.items[0];
    try testing.expectEqual(@as(usize, SSM_DISK_MAX_PER_ENTRY), e.ssm_positions.len);
    try testing.expectEqual(@as(u32, 100), e.ssm_positions[0]);
    try testing.expectEqual(@as(u32, @intCast(N * 100)), e.ssm_positions[e.ssm_positions.len - 1]);
    try testing.expect(std.mem.indexOfScalar(u32, e.ssm_positions, 200) == null);
    // The dropped position's file is gone.
    const dropped = try std.fmt.allocPrint(testing.allocator, "{s}/fp-retain/e1/s0000200.safetensors", .{base});
    defer testing.allocator.free(dropped);
    try testing.expect(statFile(io, dropped) == null);
    const kept = try std.fmt.allocPrint(testing.allocator, "{s}/fp-retain/e1/s0000100.safetensors", .{base});
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
    var complete: PersistOutcome = .partial;
    var guard: u32 = 0;
    while (guard < 40) : (guard += 1) {
        complete = try tier.appendCommit(cache.entries, cache.step, cache.config, &tokens, false, &cps, s);
        if (complete == .persisted) break;
    }
    try testing.expectEqual(PersistOutcome.persisted, complete);
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
        // The manifest stamps the lowest version that describes the entry.
        _ = std.mem.replace(u8, rewritten.items, "\"v\":6", "\"v\":3", rewritten.items);
        _ = std.mem.replace(u8, rewritten.items, "\"v\":5", "\"v\":3", rewritten.items);
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

test "modelFingerprint: rolls with --config-overrides" {
    defer model.setConfigOverrides(null);
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const base = try tmpRoot(&tmp, io, &buf);

    try tmp.dir.createDirPath(io, "model-a");
    try tmp.dir.writeFile(io, .{ .sub_path = "model-a/config.json", .data = "{\"model_type\":\"x\"}" });
    const dir_a = try std.fmt.allocPrint(testing.allocator, "{s}/model-a", .{base});
    defer testing.allocator.free(dir_a);

    model.setConfigOverrides(null);
    const fp_none = try modelFingerprint(testing.allocator, io, dir_a);
    defer testing.allocator.free(fp_none);

    const yarn = "{\"text_config\":{\"rope_parameters\":{\"rope_type\":\"yarn\",\"factor\":4.0}}}";
    model.setConfigOverrides(yarn);
    const fp_over = try modelFingerprint(testing.allocator, io, dir_a);
    defer testing.allocator.free(fp_over);
    try testing.expectEqual(@as(usize, 16), fp_over.len);
    try testing.expect(!std.mem.eql(u8, fp_none, fp_over));

    model.setConfigOverrides(yarn);
    const fp_again = try modelFingerprint(testing.allocator, io, dir_a);
    defer testing.allocator.free(fp_again);
    try testing.expectEqualStrings(fp_over, fp_again);

    // Raw-bytes pin: a whitespace-different spelling of the same JSON is a
    // different fingerprint (canonicalizing would hide a YaRN boot restoring
    // an unscaled SSD prefix if the override was re-spelled).
    const yarn_ws = "{\"text_config\": {\"rope_parameters\": {\"rope_type\": \"yarn\", \"factor\": 4.0}}}";
    model.setConfigOverrides(yarn_ws);
    const fp_ws = try modelFingerprint(testing.allocator, io, dir_a);
    defer testing.allocator.free(fp_ws);
    try testing.expect(!std.mem.eql(u8, fp_over, fp_ws));
}

test "spec meta json: the v5 head half round-trips and a v4 record parses without one" {
    var w = std.ArrayList(u8).empty;
    defer w.deinit(testing.allocator);
    const sm: SpecMeta = .{
        .base = 62_000,
        .step = 700,
        .layers = 1,
        .quant = kv_quant.KVQuantConfig.dense,
        .head = .{ .pos_base = 1, .ratio = 4, .pooled = true },
    };
    try writeSpecMetaJson(testing.allocator, &w, "mtp", sm);
    const rec = try std.fmt.allocPrint(testing.allocator, "{{\"bytes\":1{s}}}", .{w.items});
    defer testing.allocator.free(rec);
    var parsed = try std.json.parseFromSlice(std.json.Value, testing.allocator, rec, .{});
    defer parsed.deinit();
    const back = parseSpecMeta(parsed.value.object, "mtp") orelse return error.TestUnexpectedResult;
    try testing.expectEqual(sm.base, back.base);
    try testing.expectEqual(sm.step, back.step);
    const h = back.head orelse return error.TestUnexpectedResult;
    try testing.expectEqual(@as(i32, 1), h.pos_base);
    try testing.expectEqual(@as(i32, 4), h.ratio);
    try testing.expect(h.pooled);

    // A v4-shaped record (no "head") parses, with the head half absent.
    const v4 = "{\"mtp\":{\"base\":5,\"step\":9,\"layers\":1,\"scheme\":\"off\",\"bits\":0,\"group_size\":0}}";
    var p4 = try std.json.parseFromSlice(std.json.Value, testing.allocator, v4, .{});
    defer p4.deinit();
    const old = parseSpecMeta(p4.value.object, "mtp") orelse return error.TestUnexpectedResult;
    try testing.expectEqual(@as(u64, 5), old.base);
    try testing.expectEqual(@as(?SpecHeadMeta, null), old.head);
}

test "DiskTier: the qwen4 MTP head's QSA half round-trips exactly; a head-less sidecar declines the head only" {
    const io = std.testing.io;
    const s = mlx.gpuStream();
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const base = try tmpRoot(&tmp, io, &buf);

    var tier = try DiskTier.init(testing.allocator, io, base, "fp-head", 0, 128);
    defer tier.deinit();

    var cache = try KVCache.init(testing.allocator, 2);
    defer cache.deinit();
    try fillCache(&cache, s, 2, 600, 8, 0.0, .float32);
    var mtp = try KVCache.init(testing.allocator, 1);
    defer mtp.deinit();
    try fillCache(&mtp, s, 1, 600, 8, 9.5, .float32);

    var aux_src: SSMCacheEntry = .{ .conv_state = mlx.mlx_array_new(), .ssm_state = mlx.mlx_array_new(), .initialized = true };
    defer {
        _ = mlx.mlx_array_free(aux_src.conv_state);
        _ = mlx.mlx_array_free(aux_src.ssm_state);
        transformer_mod.ssmFreeQsaState(&aux_src);
    }
    aux_src.aux_state = try filledArray(&[_]c_int{ 1, 600, 8 }, 4.25, s);
    aux_src.qsa_pooled = try filledArray(&[_]c_int{ 1, 150, 8 }, -1.75, s);
    aux_src.qsa_ratio = 4;
    var head_snap = transformer_mod.ssmSnapshot(&aux_src);
    defer transformer_mod.ssmSnapshotDeinit(&head_snap);

    var tokens: [600]u32 = undefined;
    for (&tokens, 0..) |*t, i| t.* = @intCast(i + 11);
    _ = try tier.appendCommitWithSpec(
        cache.entries,
        cache.step,
        cache.config,
        &tokens,
        false,
        null,
        null,
        .{
            .entries = mtp.entries,
            .step = mtp.step,
            .config = mtp.config,
            .base_pos = 0,
            .head_aux = &head_snap,
            .head_pos_base = 1,
        },
        s,
    );

    var tier2 = try DiskTier.init(testing.allocator, io, base, "fp-head", 0, 128);
    defer tier2.deinit();
    const m = tier2.bestMatch(&tokens, false, kv_quant.KVQuantConfig.dense).?;
    var loaded = tier2.loadSpecSnap(m.idx, .mtp, 1, kv_quant.KVQuantConfig.dense) orelse
        return error.TestExpectedSpecSnap;
    defer loaded.snap.deinit();
    try testing.expectEqual(@as(usize, 600), loaded.snap.step);
    var back = loaded.head_aux orelse return error.TestExpectedHeadSnap;
    defer transformer_mod.ssmSnapshotDeinit(&back);
    try testing.expectEqual(@as(c_int, 1), loaded.head_pos_base);
    try testing.expectEqual(@as(c_int, 4), back.qsa_ratio);
    try testing.expectEqual(@as(c_int, 600), mlx.getShape(back.aux_state)[1]);
    try testing.expectEqual(@as(c_int, 150), mlx.getShape(back.qsa_pooled)[1]);
    try testing.expectEqual(@as(f32, 4.25), ssmArrVal(back.aux_state, 0, s));
    try testing.expectEqual(@as(f32, -1.75), ssmArrVal(back.qsa_pooled, 0, s));

    // Second entry, MTP history but no head: KV half loads, head half absent (a pre-v5 sidecar).
    var tokens_b: [600]u32 = undefined;
    for (&tokens_b, 0..) |*t, i| t.* = @intCast(i + 700_000);
    _ = try tier2.appendCommitWithSpec(
        cache.entries,
        cache.step,
        cache.config,
        &tokens_b,
        false,
        null,
        null,
        .{ .entries = mtp.entries, .step = mtp.step, .config = mtp.config, .base_pos = 0 },
        s,
    );
    const mb = tier2.bestMatch(&tokens_b, false, kv_quant.KVQuantConfig.dense).?;
    var kv_only = tier2.loadSpecSnap(mb.idx, .mtp, 1, kv_quant.KVQuantConfig.dense) orelse
        return error.TestExpectedSpecSnap;
    defer kv_only.snap.deinit();
    try testing.expectEqual(@as(usize, 600), kv_only.snap.step);
    try testing.expect(kv_only.head_aux == null);
}

test "DiskTier: MTP head QSA half round-trips a ring plus logical rows" {
    const io = std.testing.io;
    const s = mlx.gpuStream();
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const base = try tmpRoot(&tmp, io, &buf);

    var tier = try DiskTier.init(testing.allocator, io, base, "fp-head-ring", 0, 128);
    defer tier.deinit();

    var cache = try KVCache.init(testing.allocator, 2);
    defer cache.deinit();
    try fillCache(&cache, s, 2, 600, 8, 0.0, .float32);
    var mtp = try KVCache.init(testing.allocator, 1);
    defer mtp.deinit();
    try fillCache(&mtp, s, 1, 64, 8, 9.5, .float32);

    var aux_src: SSMCacheEntry = .{ .conv_state = mlx.mlx_array_new(), .ssm_state = mlx.mlx_array_new(), .initialized = true };
    defer {
        _ = mlx.mlx_array_free(aux_src.conv_state);
        _ = mlx.mlx_array_free(aux_src.ssm_state);
        transformer_mod.ssmFreeQsaState(&aux_src);
    }
    aux_src.aux_state = try filledArray(&[_]c_int{ 1, transformer_mod.QSA_RING_ROWS, 8 }, 4.25, s);
    aux_src.qsa_pooled = try filledArray(&[_]c_int{ 1, 16, 8 }, -1.75, s);
    aux_src.qsa_ratio = 4;
    aux_src.qsa_hist_rows = 64;
    var head_snap = transformer_mod.ssmSnapshot(&aux_src);
    defer transformer_mod.ssmSnapshotDeinit(&head_snap);
    try testing.expectEqual(@as(c_int, 64), head_snap.qsa_rows);

    var tokens: [600]u32 = undefined;
    for (&tokens, 0..) |*t, i| t.* = @intCast(i + 11);
    _ = try tier.appendCommitWithSpec(
        cache.entries,
        cache.step,
        cache.config,
        &tokens,
        false,
        null,
        null,
        .{
            .entries = mtp.entries,
            .step = mtp.step,
            .config = mtp.config,
            .base_pos = 0,
            .head_aux = &head_snap,
            .head_pos_base = 1,
        },
        s,
    );

    const m = tier.bestMatch(&tokens, false, kv_quant.KVQuantConfig.dense).?;
    var loaded = tier.loadSpecSnap(m.idx, .mtp, 1, kv_quant.KVQuantConfig.dense) orelse
        return error.TestExpectedSpecSnap;
    defer loaded.snap.deinit();
    var back = loaded.head_aux orelse return error.TestExpectedHeadSnap;
    defer transformer_mod.ssmSnapshotDeinit(&back);
    try testing.expectEqual(@as(c_int, 64), back.qsa_rows);
    try testing.expectEqual(transformer_mod.QSA_RING_ROWS, mlx.getShape(back.aux_state)[1]);
    try testing.expectEqual(@as(c_int, 16), mlx.getShape(back.qsa_pooled)[1]);
}

test "DiskTier: v8 persists the head's checkpoint leftovers; a sidecar without them loads blind" {
    // The ring on disk is 32 rows and a warm clamp lands a whole generated tail below it, so
    // the leftovers the trunk's checkpoints marked have to survive the round trip — including
    // the meta.json parse a restart reads them back through.
    const io = std.testing.io;
    const s = mlx.gpuStream();
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const base = try tmpRoot(&tmp, io, &buf);

    var tier = try DiskTier.init(testing.allocator, io, base, "fp-head-marks", 0, 128);
    defer tier.deinit();

    var cache = try KVCache.init(testing.allocator, 2);
    defer cache.deinit();
    try fillCache(&cache, s, 2, 600, 8, 0.0, .float32);
    var mtp = try KVCache.init(testing.allocator, 1);
    defer mtp.deinit();
    try fillCache(&mtp, s, 1, 64, 8, 9.5, .float32);

    var aux_src: SSMCacheEntry = .{ .conv_state = mlx.mlx_array_new(), .ssm_state = mlx.mlx_array_new(), .initialized = true };
    defer {
        _ = mlx.mlx_array_free(aux_src.conv_state);
        _ = mlx.mlx_array_free(aux_src.ssm_state);
        transformer_mod.ssmFreeQsaState(&aux_src);
    }
    aux_src.aux_state = try filledArray(&[_]c_int{ 1, transformer_mod.QSA_RING_ROWS, 8 }, 4.25, s);
    aux_src.qsa_pooled = try filledArray(&[_]c_int{ 1, 16, 8 }, -1.75, s);
    aux_src.qsa_ratio = 4;
    aux_src.qsa_hist_rows = 64;
    var head_snap = transformer_mod.ssmSnapshot(&aux_src);
    defer transformer_mod.ssmSnapshotDeinit(&head_snap);

    // Two trunk checkpoint positions, both mid-block.
    var marks: transformer_mod.QsaHeadMarkSet = .{};
    defer marks.deinit();
    marks.put(26, try filledArray(&[_]c_int{ 1, 2, 8 }, 7.5, s));
    marks.put(46, try filledArray(&[_]c_int{ 1, 2, 8 }, -3.25, s));

    var tokens: [600]u32 = undefined;
    for (&tokens, 0..) |*t, i| t.* = @intCast(i + 11);
    _ = try tier.appendCommitWithSpec(
        cache.entries,
        cache.step,
        cache.config,
        &tokens,
        false,
        null,
        null,
        .{
            .entries = mtp.entries,
            .step = mtp.step,
            .config = mtp.config,
            .base_pos = 0,
            .head_aux = &head_snap,
            .head_pos_base = 1,
            .head_marks = marks.slice(),
        },
        s,
    );

    // A restart: the positions come back through meta.json, the rows through the sidecar.
    var tier2 = try DiskTier.init(testing.allocator, io, base, "fp-head-marks", 0, 128);
    defer tier2.deinit();
    const m = tier2.bestMatch(&tokens, false, kv_quant.KVQuantConfig.dense).?;
    try testing.expectEqual(@as(u8, 8), DiskTier.metaVersionFor(tier2.entries.items[m.idx]));
    var loaded = tier2.loadSpecSnap(m.idx, .mtp, 1, kv_quant.KVQuantConfig.dense) orelse
        return error.TestExpectedSpecSnap;
    defer loaded.snap.deinit();
    var back = loaded.head_aux orelse return error.TestExpectedHeadSnap;
    defer transformer_mod.ssmSnapshotDeinit(&back);
    var back_marks = loaded.head_marks;
    defer back_marks.deinit();
    try testing.expectEqual(@as(usize, 2), back_marks.len);
    const lv26 = back_marks.find(26) orelse return error.TestExpectedHeadMark;
    const lv46 = back_marks.find(46) orelse return error.TestExpectedHeadMark;
    try testing.expectEqual(@as(c_int, 2), mlx.getShape(lv26)[1]);
    try testing.expectEqual(@as(f32, 7.5), ssmArrVal(lv26, 0, s));
    try testing.expectEqual(@as(f32, -3.25), ssmArrVal(lv46, 0, s));

    // A v5..v7 sidecar carries none: the head still restores, and clamps as it does today.
    var tokens_b: [600]u32 = undefined;
    for (&tokens_b, 0..) |*t, i| t.* = @intCast(i + 700_000);
    _ = try tier2.appendCommitWithSpec(
        cache.entries,
        cache.step,
        cache.config,
        &tokens_b,
        false,
        null,
        null,
        .{
            .entries = mtp.entries,
            .step = mtp.step,
            .config = mtp.config,
            .base_pos = 0,
            .head_aux = &head_snap,
            .head_pos_base = 1,
        },
        s,
    );
    const mb = tier2.bestMatch(&tokens_b, false, kv_quant.KVQuantConfig.dense).?;
    try testing.expectEqual(@as(u8, 5), DiskTier.metaVersionFor(tier2.entries.items[mb.idx]));
    var no_marks = tier2.loadSpecSnap(mb.idx, .mtp, 1, kv_quant.KVQuantConfig.dense) orelse
        return error.TestExpectedSpecSnap;
    defer no_marks.snap.deinit();
    var nb = no_marks.head_aux orelse return error.TestExpectedHeadSnap;
    defer transformer_mod.ssmSnapshotDeinit(&nb);
    try testing.expectEqual(@as(usize, 0), no_marks.head_marks.len);
}

test "DiskTier: SSD-first writes a checkpoint beside the chunk that closes it" {
    // Checkpoints ride outside the per-flush byte budget, so the first flush of a long hybrid
    // entry already restores; without it (arm B) the entry carries KV with no recurrent state.
    const io = std.testing.io;
    const s = mlx.gpuStream();
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const base = try tmpRoot(&tmp, io, &buf);

    var cache = try KVCache.init(testing.allocator, 3);
    defer cache.deinit();
    try fillCache(&cache, s, 3, 600, 8, 0.0, .float32);
    var tokens: [600]u32 = undefined;
    for (&tokens, 0..) |*t, i| t.* = @intCast(i + 7);
    var src128 = buildHybridEntries(s, 100.0, 500.0);
    defer freeHybridEntries(&src128);
    var cps = [_]transformer_mod.SSMCheckpoint{
        try transformer_mod.captureSsmCheckpoint(testing.allocator, &src128, 128, s),
    };
    defer for (&cps) |*cp| cp.deinit(testing.allocator);

    // Arm A: SSD-first. One flush, one chunk, and its checkpoint.
    {
        var tier = try DiskTier.init(testing.allocator, io, base, "fp-ssdfirst-cp", 0, 128);
        defer tier.deinit();
        tier.ssd_first = true;
        tier.armTestSpace(1024 * 1024 * 1024 * 1024, 2048 * 1024 * 1024 * 1024);
        tier.max_flush_bytes = 1; // bound the flush to one chunk

        const complete = try tier.appendCommit(cache.entries, cache.step, cache.config, &tokens, false, &cps, s);
        try testing.expectEqual(PersistOutcome.partial, complete); // KV is still partial
        const e = &tier.entries.items[0];
        try testing.expectEqual(@as(u32, 128), e.kv_len);
        try testing.expectEqual(@as(usize, 1), e.ssm_positions.len);
        try testing.expectEqual(@as(u32, 128), e.ssm_positions[0]);

        var cache2 = try KVCache.init(testing.allocator, 3);
        defer cache2.deinit();
        var dst: [3]SSMCacheEntry = .{
            .{ .conv_state = mlx.mlx_array_new(), .ssm_state = mlx.mlx_array_new(), .initialized = false },
            .{ .conv_state = mlx.mlx_array_new(), .ssm_state = mlx.mlx_array_new(), .initialized = false },
            .{ .conv_state = mlx.mlx_array_new(), .ssm_state = mlx.mlx_array_new(), .initialized = false },
        };
        defer freeHybridEntries(&dst);
        try testing.expectEqual(@as(u32, 128), try tier.restoreIntoHybrid(&cache2, &dst, 0, 128, s));
        try testing.expectEqual(@as(f32, 100.0), ssmArrVal(dst[0].conv_state, 0, s));
    }

    // Arm B: the chunk eats the budget, no checkpoint.
    {
        var tier = try DiskTier.init(testing.allocator, io, base, "fp-legacy-cp", 0, 128);
        defer tier.deinit();
        tier.max_flush_bytes = 1;

        _ = try tier.appendCommit(cache.entries, cache.step, cache.config, &tokens, false, &cps, s);
        const e = &tier.entries.items[0];
        try testing.expectEqual(@as(u32, 128), e.kv_len);
        try testing.expectEqual(@as(usize, 0), e.ssm_positions.len);
    }
}

test "DiskTier: an SSM checkpoint STAGES through the writer — no filesystem write on the inference thread" {
    // The checkpoint file is staged through the writer, before the meta.json that indexes it,
    // and the staged bytes are a real safetensors image (a fresh tier reads them back with
    // their `__metadata__`).
    const io = std.testing.io;
    const s = mlx.gpuStream();
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const base = try tmpRoot(&tmp, io, &buf);

    var cache = try KVCache.init(testing.allocator, 3);
    defer cache.deinit();
    try fillCache(&cache, s, 3, 600, 8, 0.0, .float32); // >= MIN_PERSIST_TOKENS
    var tokens: [600]u32 = undefined;
    for (&tokens, 0..) |*t, i| t.* = @intCast(i + 7);

    // A checkpoint with every optional part qwen4_exp carries.
    var src128 = buildHybridEntries(s, 100.0, 500.0);
    defer freeHybridEntries(&src128);
    const aux_shape = [_]c_int{ 1, 128, 4 };
    const pooled_shape = [_]c_int{ 1, 32, 4 };
    src128[2].aux_state = makeArange(s, &aux_shape, 700.0);
    src128[2].qsa_pooled = makeArange(s, &pooled_shape, 800.0);
    src128[2].qsa_ratio = 4;
    src128[1].ple_prev = .{ 42, 43, 0, 0, 0, 0, 0, 0 };
    src128[1].ple_prev_valid = true;
    var cps = [_]transformer_mod.SSMCheckpoint{
        try transformer_mod.captureSsmCheckpoint(testing.allocator, &src128, 128, s),
    };
    defer for (&cps) |*cp| cp.deinit(testing.allocator);
    try transformer_mod.attachQsaHistoryToLatest(&cps, &src128, s);

    var tier = try DiskTier.init(testing.allocator, io, base, "fp-ssd-cpstage", 0, 128);
    defer tier.deinit();
    tier.ssd_first = true;
    tier.armTestSpace(1024 * 1024 * 1024 * 1024, 2048 * 1024 * 1024 * 1024);
    tier.enableBackgroundWriter();
    try testing.expect(tier.writer != null);
    tier.writer.?.setPaused(true);
    defer tier.writer.?.setPaused(false);

    try testing.expectEqual(
        PersistOutcome.persisted,
        try tier.appendCommit(cache.entries, cache.step, cache.config, &tokens, false, &cps, s),
    );
    try testing.expectEqual(@as(usize, 1), tier.entries.items[0].ssm_positions.len);
    try testing.expectEqual(@as(u32, 128), tier.entries.items[0].ssm_positions[0]);

    // Bar 1: the checkpoint file is not on disk; the commit returned without writing it.
    try testing.expectError(
        error.FileNotFound,
        tmp.dir.statFile(io, "fp-ssd-cpstage/e1/s0000128.safetensors", .{}),
    );
    try testing.expectEqual(@as(u64, 0), tier.writer.?.filesWritten());
    try testing.expect(tier.entries.items[0].ssm_bytes[0] > 0);

    // Bar 2: staged before meta.json.
    var paths = std.ArrayList([]const u8).empty;
    defer {
        for (paths.items) |pp| testing.allocator.free(pp);
        paths.deinit(testing.allocator);
    }
    try tier.writer.?.stagedPaths(&paths, testing.allocator);
    const cp_at = for (paths.items, 0..) |pp, i| {
        if (std.mem.endsWith(u8, pp, "/s0000128.safetensors")) break i;
    } else return error.CheckpointNotStaged;
    const meta_at = for (paths.items, 0..) |pp, i| {
        if (std.mem.endsWith(u8, pp, "/meta.json")) break i;
    } else return error.MetaNotStaged;
    try testing.expect(cp_at < meta_at);

    tier.writer.?.setPaused(false);
    tier.drainWriter();
    const st = try tmp.dir.statFile(io, "fp-ssd-cpstage/e1/s0000128.safetensors", .{});
    try testing.expectEqual(tier.entries.items[0].ssm_bytes[0], st.size);

    // Bar 3: a fresh tier reads the hand-rolled image back, `__metadata__` included.
    var tier2 = try DiskTier.init(testing.allocator, io, base, "fp-ssd-cpstage", 0, 128);
    defer tier2.deinit();
    try testing.expectEqual(@as(usize, 1), tier2.entryCount());
    var cache2 = try KVCache.init(testing.allocator, 3);
    defer cache2.deinit();
    var dst: [3]SSMCacheEntry = .{
        .{ .conv_state = mlx.mlx_array_new(), .ssm_state = mlx.mlx_array_new(), .initialized = false },
        .{ .conv_state = mlx.mlx_array_new(), .ssm_state = mlx.mlx_array_new(), .initialized = false },
        .{ .conv_state = mlx.mlx_array_new(), .ssm_state = mlx.mlx_array_new(), .initialized = false },
    };
    defer freeHybridEntries(&dst);
    try testing.expectEqual(@as(u32, 128), try tier2.restoreIntoHybrid(&cache2, &dst, 0, 128, s));
    try testing.expectEqual(@as(f32, 100.0), ssmArrVal(dst[0].conv_state, 0, s));
    try testing.expectEqual(@as(f32, 500.0), ssmArrVal(dst[0].ssm_state, 0, s));
    try testing.expect(dst[2].aux_state.ctx == null);
    try testing.expectEqual(@as(c_int, 128), dst[2].qsa_hist_rows);
    try testing.expectEqual(@as(f32, 800.0 + 11.0), ssmArrVal(dst[2].qsa_pooled, 11, s));
    try testing.expectEqual(@as(c_int, 4), dst[2].qsa_ratio);
    try testing.expect(dst[1].ple_prev_valid and dst[1].ple_prev[0] == 42 and dst[1].ple_prev[1] == 43);
    try testing.expectEqual(
        try cacheValueAt(&cache, 0, 127, 3, s),
        try cacheValueAt(&cache2, 0, 127, 3, s),
    );
}

test "DiskTier: the staged encoder carries __metadata__, and refuses a value it cannot escape" {
    const io = std.testing.io;
    const s = mlx.gpuStream();
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const base = try tmpRoot(&tmp, io, &buf);

    var tier = try DiskTier.init(testing.allocator, io, base, "fp-meta", 0, 128);
    defer tier.deinit();

    var cache = try KVCache.init(testing.allocator, 1);
    defer cache.deinit();
    try fillCache(&cache, s, 1, 128, 8, 0.0, .float32);

    var list = std.ArrayList(DiskTier.NamedTensor).empty;
    defer tier.freeNamed(&list);
    try tier.appendSlice(&list, 0, "k", cache.entries[0].keys, 0, 128, s);

    const meta = [_]DiskTier.MetaPair{
        .{ .key = "layers", .value = "3" },
        .{ .key = "init", .value = "0,1,2" },
        .{ .key = "qsa_ratio", .value = "4" },
    };
    const bytes = try tier.serializeSafetensors(list.items, &meta, s);
    defer testing.allocator.free(bytes);

    try tmp.dir.writeFile(io, .{ .sub_path = "meta-probe.safetensors", .data = bytes });
    const path = try std.fmt.allocPrint(testing.allocator, "{s}/meta-probe.safetensors\x00", .{base});
    defer testing.allocator.free(path);

    const cpu = mlx.mlx_default_cpu_stream_new();
    defer _ = mlx.mlx_stream_free(cpu);
    var tensor_map = mlx.mlx_map_string_to_array_new();
    defer _ = mlx.mlx_map_string_to_array_free(tensor_map);
    var meta_map = mlx.mlx_map_string_to_string_new();
    defer _ = mlx.mlx_map_string_to_string_free(meta_map);
    try mlx.check(mlx.mlx_load_safetensors(&tensor_map, &meta_map, @ptrCast(path.ptr), cpu));
    inline for (.{ .{ "layers", "3" }, .{ "init", "0,1,2" }, .{ "qsa_ratio", "4" } }) |kv| {
        var got: [*:0]const u8 = undefined;
        try testing.expectEqual(@as(c_int, 0), mlx.mlx_map_string_to_string_get(&got, meta_map, kv[0]));
        try testing.expectEqualStrings(kv[1], std.mem.span(got));
    }
    var back = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(back);
    try testing.expectEqual(@as(c_int, 0), mlx.mlx_map_string_to_array_get(&back, tensor_map, "l0.k"));

    // A value with a quote would produce a header mlx cannot parse.
    const bad = [_]DiskTier.MetaPair{.{ .key = "init", .value = "0\",\"x" }};
    try testing.expectError(error.DiskCacheBadMetadata, tier.encodeSafetensors(list.items, &bad));
}

test "DiskTier: SSD-first stages the flush off-thread and indexes LAST" {
    // The commit returns with the whole entry still staged, and meta.json is submitted after
    // every chunk so the FIFO lands it last.
    const io = std.testing.io;
    const s = mlx.gpuStream();
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const base = try tmpRoot(&tmp, io, &buf);

    var tier = try DiskTier.init(testing.allocator, io, base, "fp-ssd-writer", 0, 128);
    defer tier.deinit();
    tier.ssd_first = true;
    tier.armTestSpace(1024 * 1024 * 1024 * 1024, 2048 * 1024 * 1024 * 1024);
    tier.enableBackgroundWriter();
    try testing.expect(tier.writer != null);
    try testing.expectEqual(SSD_FIRST_READBACK_BYTES, tier.max_flush_bytes);
    tier.writer.?.setPaused(true);
    defer tier.writer.?.setPaused(false);

    var cache = try KVCache.init(testing.allocator, 3);
    defer cache.deinit();
    try fillCache(&cache, s, 3, 640, 8, 0.0, .float32);
    var tokens: [640]u32 = undefined;
    for (&tokens, 0..) |*t, i| t.* = @intCast(i + 7);

    const complete = try tier.appendCommit(cache.entries, cache.step, cache.config, &tokens, false, null, s);
    try testing.expectEqual(PersistOutcome.persisted, complete);
    try testing.expect(tier.writer.?.pendingBytes() > 0);
    try testing.expectEqual(@as(u64, 0), tier.writer.?.filesWritten());
    try testing.expectError(error.FileNotFound, tmp.dir.statFile(io, "fp-ssd-writer/e1/meta.json", .{}));

    var paths = std.ArrayList([]const u8).empty;
    defer {
        for (paths.items) |p| testing.allocator.free(p);
        paths.deinit(testing.allocator);
    }
    try tier.writer.?.stagedPaths(&paths, testing.allocator);
    try testing.expectEqual(@as(usize, 6), paths.items.len); // 5 chunks + meta
    for (paths.items[0 .. paths.items.len - 1]) |p| {
        try testing.expect(std.mem.indexOf(u8, p, "/c0000") != null);
    }
    try testing.expect(std.mem.endsWith(u8, paths.items[paths.items.len - 1], "/meta.json"));

    tier.writer.?.setPaused(false);
    tier.drainWriter();
    try testing.expectEqual(@as(u64, 6), tier.writer.?.filesWritten());

    // The staged bytes are a real safetensors image: a fresh tier restores it.
    var tier2 = try DiskTier.init(testing.allocator, io, base, "fp-ssd-writer", 0, 128);
    defer tier2.deinit();
    try testing.expectEqual(@as(usize, 1), tier2.entryCount());
    var cache2 = try KVCache.init(testing.allocator, 3);
    defer cache2.deinit();
    try testing.expectEqual(@as(u32, 640), try tier2.restoreInto(&cache2, 0, s));
    for ([_]u32{ 0, 1, 2 }) |li| {
        for ([_]u32{ 0, 127, 128, 511, 639 }) |pos| {
            try testing.expectEqual(
                try cacheValueAt(&cache, li, pos, 3, s),
                try cacheValueAt(&cache2, li, pos, 3, s),
            );
        }
    }
}

test "DiskTier: SSD-first write-through extends without rewriting a persisted chunk" {
    // Write-through: a killed prefill leaves a restorable chunk-aligned prefix, and a persisted
    // chunk is never rewritten.
    const io = std.testing.io;
    const s = mlx.gpuStream();
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const base = try tmpRoot(&tmp, io, &buf);

    var tier = try DiskTier.init(testing.allocator, io, base, "fp-ssd-wt", 0, 128);
    defer tier.deinit();
    tier.ssd_first = true;
    tier.armTestSpace(1024 * 1024 * 1024 * 1024, 2048 * 1024 * 1024 * 1024);
    tier.enableBackgroundWriter();

    var cache = try KVCache.init(testing.allocator, 2);
    defer cache.deinit();
    var tokens: [768]u32 = undefined;
    for (&tokens, 0..) |*t, i| t.* = @intCast(i + 7);

    // Chunk boundary 1: positions [0, 640) forwarded (5 chunks of 128).
    try fillCache(&cache, s, 2, 640, 8, 0.0, .float32);
    var src640 = buildHybridEntries(s, 11.0, 22.0);
    defer freeHybridEntries(&src640);
    var cps640 = [_]transformer_mod.SSMCheckpoint{
        try transformer_mod.captureSsmCheckpoint(testing.allocator, &src640, 640, s),
    };
    defer for (&cps640) |*cp| cp.deinit(testing.allocator);
    _ = try tier.appendCommit(cache.entries, 640, cache.config, tokens[0..640], false, &cps640, s);
    tier.drainWriter();
    try testing.expectEqual(@as(usize, 1), tier.entryCount());
    try testing.expectEqual(@as(u32, 640), tier.entries.items[0].kv_len);
    {
        var c2 = try KVCache.init(testing.allocator, 2);
        defer c2.deinit();
        var dst: [3]SSMCacheEntry = .{
            .{ .conv_state = mlx.mlx_array_new(), .ssm_state = mlx.mlx_array_new(), .initialized = false },
            .{ .conv_state = mlx.mlx_array_new(), .ssm_state = mlx.mlx_array_new(), .initialized = false },
            .{ .conv_state = mlx.mlx_array_new(), .ssm_state = mlx.mlx_array_new(), .initialized = false },
        };
        defer freeHybridEntries(&dst);
        try testing.expectEqual(@as(u32, 640), try tier.restoreIntoHybrid(&c2, &dst, 0, 640, s));
    }
    tier.drainWriter(); // the restore re-indexes for LRU
    const written_after_first = tier.writer.?.filesWritten();

    // Chunk boundary 2: the prefill continues to 768. Only chunk 5 is new.
    try fillCache(&cache, s, 2, 128, 8, 640.0, .float32);
    tier.writer.?.setPaused(true);
    defer tier.writer.?.setPaused(false);
    _ = try tier.appendCommit(cache.entries, 768, cache.config, tokens[0..768], false, &cps640, s);
    var staged = std.ArrayList([]const u8).empty;
    defer {
        for (staged.items) |p| testing.allocator.free(p);
        staged.deinit(testing.allocator);
    }
    try tier.writer.?.stagedPaths(&staged, testing.allocator);
    for (staged.items) |p| {
        try testing.expect(std.mem.indexOf(u8, p, "/c000000.") == null);
        try testing.expect(std.mem.indexOf(u8, p, "/c000004.") == null);
    }
    try testing.expect(staged.items.len <= 2); // the new chunk + meta.json
    tier.writer.?.setPaused(false);
    tier.drainWriter();
    try testing.expect(tier.writer.?.filesWritten() - written_after_first <= 2);
    try testing.expectEqual(@as(usize, 1), tier.entryCount());
    try testing.expectEqual(@as(u32, 768), tier.entries.items[0].kv_len);
}

test "diskBudgetFromFreeSpace: reserve is min(64 GiB, 10% of volume); below the floor stores nothing" {
    const GB: u64 = 1 << 30;
    // 4 TB volume, 1 TB free: reserve is the 64 GiB cap.
    try testing.expectEqual(@as(?u64, 1024 * GB - 64 * GB), diskBudgetFromFreeSpace(0, 1024 * GB, 4096 * GB));
    try testing.expectEqual(@as(?u64, 100 * GB), diskBudgetFromFreeSpace(100 * GB, 1024 * GB, 4096 * GB));
    // Small volume: 10% is the binding reserve.
    try testing.expectEqual(@as(?u64, 60 * GB), diskBudgetFromFreeSpace(0, 80 * GB, 200 * GB));
    // Under the store floor: refuse, never a silent 0.
    try testing.expectEqual(@as(?u64, null), diskBudgetFromFreeSpace(0, 20 * GB, 200 * GB));
    try testing.expectEqual(@as(?u64, null), diskBudgetFromFreeSpace(500 * GB, 20 * GB, 200 * GB));
}

test "volumeSpace: the live probe is plausible or null (statfs ABI guard)" {
    // On this machine the probe must succeed: a null means the struct layout broke.
    const vs = volumeSpace("/") orelse return error.VolumeSpaceProbeFailed;
    try testing.expect(vs.total > 0);
    try testing.expect(vs.free <= vs.total);
    try testing.expect(vs.total > 1024 * 1024 * 1024); // a macOS root volume
}

test "DiskTier: SSD-first declines to store when the VOLUME is short, and says so" {
    // 10 GiB free against a 512 GiB volume leaves nothing after the reserve: the tier stores nothing.
    const io = std.testing.io;
    const s = mlx.gpuStream();
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const base = try tmpRoot(&tmp, io, &buf);

    var tier = try DiskTier.init(testing.allocator, io, base, "fp-short", 0, 128);
    defer tier.deinit();
    tier.ssd_first = true;
    tier.armTestSpace(10 * 1024 * 1024 * 1024, 512 * 1024 * 1024 * 1024);

    var cache = try KVCache.init(testing.allocator, 2);
    defer cache.deinit();
    try fillCache(&cache, s, 2, 640, 8, 0.0, .float32);
    var tokens: [640]u32 = undefined;
    for (&tokens, 0..) |*t, i| t.* = @intCast(i + 7);

    _ = try tier.appendCommit(cache.entries, 640, cache.config, &tokens, false, null, s);
    tier.drainWriter();
    try testing.expect(tier.store_declined);
    try testing.expectEqual(@as(usize, 0), tier.entryCount());
    try testing.expectEqual(@as(u64, 0), tier.total_bytes);

    tier.armTestSpace(1024 * 1024 * 1024 * 1024, 2048 * 1024 * 1024 * 1024);
    _ = try tier.appendCommit(cache.entries, 640, cache.config, &tokens, false, null, s);
    tier.drainWriter();
    try testing.expect(!tier.store_declined);
    try testing.expectEqual(@as(usize, 1), tier.entryCount());
    try testing.expectEqual(@as(u32, 640), tier.entries.items[0].kv_len);
}

test "DiskTier: entries cross the SSD-first boundary in BOTH directions (SSD-first itself bumps no manifest)" {
    // SSD-first changes when chunks are written and which checkpoints are present, never the
    // on-disk format: entries restore across both arms.
    const io = std.testing.io;
    const s = mlx.gpuStream();
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const base = try tmpRoot(&tmp, io, &buf);

    var cache = try KVCache.init(testing.allocator, 3);
    defer cache.deinit();
    try fillCache(&cache, s, 3, 600, 8, 0.0, .float32);
    var tokens: [600]u32 = undefined;
    for (&tokens, 0..) |*t, i| t.* = @intCast(i + 7);

    // Written by the legacy path, read by SSD-first.
    {
        var legacy = try DiskTier.init(testing.allocator, io, base, "fp-x-legacy", 0, 128);
        _ = try legacy.appendCommit(cache.entries, cache.step, cache.config, &tokens, false, null, s);
        legacy.deinit();

        var ssd = try DiskTier.init(testing.allocator, io, base, "fp-x-legacy", 0, 128);
        defer ssd.deinit();
        ssd.ssd_first = true;
        ssd.armTestSpace(1024 * 1024 * 1024 * 1024, 2048 * 1024 * 1024 * 1024);
        ssd.enableBackgroundWriter();
        try testing.expectEqual(@as(usize, 1), ssd.entryCount());
        var out = try KVCache.init(testing.allocator, 3);
        defer out.deinit();
        try testing.expectEqual(@as(u32, 600), try ssd.restoreInto(&out, 0, s));
        for ([_]u32{ 0, 1, 2 }) |li| for ([_]u32{ 0, 127, 128, 599 }) |pos| {
            try testing.expectEqual(try cacheValueAt(&cache, li, pos, 3, s), try cacheValueAt(&out, li, pos, 3, s));
        };
    }

    // Written by SSD-first (background writer, hand-serialized), read by the legacy path.
    {
        var ssd = try DiskTier.init(testing.allocator, io, base, "fp-x-ssd", 0, 128);
        ssd.ssd_first = true;
        ssd.armTestSpace(1024 * 1024 * 1024 * 1024, 2048 * 1024 * 1024 * 1024);
        ssd.enableBackgroundWriter();
        _ = try ssd.appendCommit(cache.entries, cache.step, cache.config, &tokens, false, null, s);
        ssd.drainWriter();
        ssd.deinit();

        var legacy = try DiskTier.init(testing.allocator, io, base, "fp-x-ssd", 0, 128);
        defer legacy.deinit();
        try testing.expect(legacy.writer == null);
        try testing.expectEqual(@as(usize, 1), legacy.entryCount());
        var out = try KVCache.init(testing.allocator, 3);
        defer out.deinit();
        try testing.expectEqual(@as(u32, 600), try legacy.restoreInto(&out, 0, s));
        for ([_]u32{ 0, 1, 2 }) |li| for ([_]u32{ 0, 127, 128, 599 }) |pos| {
            try testing.expectEqual(try cacheValueAt(&cache, li, pos, 3, s), try cacheValueAt(&out, li, pos, 3, s));
        };
        // No inherited chunks and no MTP head: stamps v4, the version an older reader accepts.
        const meta = try tmp.dir.readFileAlloc(io, "fp-x-ssd/e1/meta.json", testing.allocator, .limited(1 << 20));
        defer testing.allocator.free(meta);
        try testing.expect(std.mem.indexOf(u8, meta, "\"v\":4") != null);
        try testing.expect(std.mem.indexOf(u8, meta, "\"v\":6") == null);
    }
}

test "DiskTier: the root-wide sweep drops strays and never touches the live tier's own root" {
    // A sibling's index-less entry is a crash leftover only once it is older than
    // `STRAY_MIN_AGE_NS` (another server's flush writes meta.json last); the live tier's own
    // root is skipped entirely.
    const io = std.testing.io;
    const s = mlx.gpuStream();
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const base = try tmpRoot(&tmp, io, &buf);

    {
        var sib = try DiskTier.init(testing.allocator, io, base, "fp-sibling", 0, 128);
        defer sib.deinit();
        var cache = try KVCache.init(testing.allocator, 1);
        defer cache.deinit();
        try fillCache(&cache, s, 1, 600, 8, 0.0, .float32);
        var tokens: [600]u32 = undefined;
        for (&tokens, 0..) |*t, i| t.* = @intCast(i + 7);
        _ = try sib.appendCommit(cache.entries, cache.step, cache.config, &tokens, false, null, s);
    }
    try tmp.dir.createDirPath(io, "fp-stray/e9");
    try tmp.dir.writeFile(io, .{ .sub_path = "fp-stray/e9/c000000.safetensors", .data = "orphan" });
    const aged: std.Io.Timestamp = .{
        .nanoseconds = std.Io.Timestamp.now(io, .real).nanoseconds - 2 * @as(i96, @intCast(STRAY_MIN_AGE_NS)),
    };
    try tmp.dir.setTimestamps(io, "fp-stray/e9/c000000.safetensors", .{ .modify_timestamp = .{ .new = aged } });
    // Its young twin: index-less but written just now, so it must survive.
    try tmp.dir.createDirPath(io, "fp-inflight/e3");
    try tmp.dir.writeFile(io, .{ .sub_path = "fp-inflight/e3/c000000.safetensors", .data = "another server" });
    var live = try DiskTier.init(testing.allocator, io, base, "fp-live", 0, 128);
    defer live.deinit();
    live.ssd_first = true;
    live.armTestSpace(1024 * 1024 * 1024 * 1024, 2048 * 1024 * 1024 * 1024);
    // The live tier's own root, mid-write-through: chunks, no index yet. Staged after init
    // (init's own `scan` rightly drops an index-less entry).
    try tmp.dir.createDirPath(io, "fp-live/e1");
    try tmp.dir.writeFile(io, .{ .sub_path = "fp-live/e1/c000000.safetensors", .data = "inflight" });
    live.max_bytes = 1 << 40;
    sweepBase(testing.allocator, io, base, live.root, live.max_bytes);

    try testing.expectError(error.FileNotFound, tmp.dir.statFile(io, "fp-stray/e9/c000000.safetensors", .{}));
    try testing.expect(tmp.dir.statFile(io, "fp-inflight/e3/c000000.safetensors", .{}) catch null != null);
    try testing.expect(tmp.dir.statFile(io, "fp-sibling/e1/meta.json", .{}) catch null != null);
    try testing.expect(tmp.dir.statFile(io, "fp-live/e1/c000000.safetensors", .{}) catch null != null);

    // With a budget of zero the siblings' real entries go too, and the live root is still untouched.
    sweepBase(testing.allocator, io, base, live.root, 0);
    try testing.expectError(error.FileNotFound, tmp.dir.statFile(io, "fp-sibling/e1/meta.json", .{}));
    try testing.expect(tmp.dir.statFile(io, "fp-live/e1/c000000.safetensors", .{}) catch null != null);
    try testing.expect(tmp.dir.statFile(io, "fp-inflight/e3/c000000.safetensors", .{}) catch null != null);
}

test "DiskTier: SSM retention spacing is priced against the tier, not just capped" {
    // Span-preserving survivors sit ~L/K apart; K=16 halves the old K=8 gap at the 383k shape.
    // The bar is derived from the constant.
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const base = try tmpRoot(&tmp, io, &buf);
    var tier = try DiskTier.init(testing.allocator, io, base, "fp-spacing", 0, 128);
    defer tier.deinit();
    try testing.expectEqual(transformer_mod.ThinPolicy.oldest, tier.cp_thin);
    try testing.expectEqual(SSM_DISK_MAX_PER_ENTRY_LEGACY, tier.ssm_max_per_entry);
    tier.cp_thin = .min_span_recency;
    tier.ssm_max_per_entry = SSM_DISK_MAX_PER_ENTRY;

    const L: u32 = 383_069;
    var positions: [94]u32 = undefined;
    for (positions[0..93], 0..) |*p, i| p.* = @intCast((i + 1) * 4096);
    positions[93] = 383_039;

    const kept = try tier.ssmTargetPositions(&positions, &[_]transformer_mod.SSMCheckpoint{}, L);
    defer testing.allocator.free(kept);

    try testing.expectEqual(SSM_DISK_MAX_PER_ENTRY, kept.len);
    // Both ends survive.
    try testing.expectEqual(@as(u32, 4096), kept[0]);
    try testing.expectEqual(@as(u32, 383_039), kept[kept.len - 1]);

    var max_gap: u32 = 0;
    var i: usize = 1;
    while (i < kept.len) : (i += 1) {
        const gap = kept[i] - kept[i - 1];
        if (gap > max_gap) max_gap = gap;
    }

    // The ungated arm: every other arch keeps the previous retention (the highest N).
    {
        var legacy = try DiskTier.init(testing.allocator, io, base, "fp-spacing-legacy", 0, 128);
        defer legacy.deinit();
        try testing.expectEqual(transformer_mod.ThinPolicy.oldest, legacy.cp_thin);
        try testing.expectEqual(SSM_DISK_MAX_PER_ENTRY_LEGACY, legacy.ssm_max_per_entry);
        const old_kept = try legacy.ssmTargetPositions(&positions, &[_]transformer_mod.SSMCheckpoint{}, L);
        defer testing.allocator.free(old_kept);
        try testing.expectEqual(SSM_DISK_MAX_PER_ENTRY_LEGACY, old_kept.len);
        // End-anchored: the survivors are the last N of the input.
        try testing.expectEqualSlices(u32, positions[positions.len - SSM_DISK_MAX_PER_ENTRY_LEGACY ..], old_kept);
        try testing.expect(old_kept[0] > kept[0]);
    }
    // The newest quarter stays at capture density; the rest is spread.
    try testing.expectEqual(@as(u32, 383_039 - 380_928), kept[kept.len - 1] - kept[kept.len - 2]);
    try testing.expectEqual(@as(u32, 4096), kept[kept.len - 2] - kept[kept.len - 3]);
    try testing.expectEqual(@as(u32, 4096), kept[kept.len - 3] - kept[kept.len - 4]);
    try testing.expect(max_gap <= 40_000);
    try testing.expect(max_gap < 54_000);
    try testing.expect(kept[1] - kept[0] > 4 * 4096);
}

/// The pre-fix materializer (one `mlx_array_eval` per tensor), kept only as the byte-identity golden below.
fn materializeLegacyPerTensorForTest(tensors: []DiskTier.NamedTensor, s: mlx.mlx_stream) !void {
    for (tensors) |*t| {
        var cont = mlx.mlx_array_new();
        errdefer _ = mlx.mlx_array_free(cont);
        try mlx.check(mlx.mlx_contiguous(&cont, t.arr, false, s));
        _ = mlx.mlx_array_free(t.arr);
        t.arr = cont;
        try mlx.check(mlx.mlx_array_eval(t.arr));
    }
}

test "DiskTier: the staged serializer evals ONCE per chunk, byte-identically to the per-tensor path" {
    // One eval per chunk file (the per-tensor eval cost ~2,300 GPU syncs on a 32k warm turn), and the bytes are unchanged.
    const io = std.testing.io;
    const s = mlx.gpuStream();
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const base = try tmpRoot(&tmp, io, &buf);

    var tier = try DiskTier.init(testing.allocator, io, base, "fp-eval", 0, 128);
    defer tier.deinit();

    const qcfg = kv_quant.KVQuantConfig.affine(4);
    var cache = try KVCache.initWithConfig(testing.allocator, 3, qcfg);
    defer cache.deinit();
    try fillCache(&cache, s, 3, 256, 64, 0.0, .bfloat16);

    var a = std.ArrayList(DiskTier.NamedTensor).empty;
    defer tier.freeNamed(&a);
    var b = std.ArrayList(DiskTier.NamedTensor).empty;
    defer tier.freeNamed(&b);
    for ([_]*std.ArrayList(DiskTier.NamedTensor){ &a, &b }) |list| {
        for (cache.entries, 0..) |*e, li| {
            try testing.expect(e.initialized);
            try tier.appendSlice(list, li, "k", e.keys, 0, 128, s);
            try tier.appendSlice(list, li, "v", e.values, 0, 128, s);
            try tier.appendSlice(list, li, "ks", e.keys_scales, 0, 128, s);
            try tier.appendSlice(list, li, "kb", e.keys_biases, 0, 128, s);
            try tier.appendSlice(list, li, "vs", e.values_scales, 0, 128, s);
            try tier.appendSlice(list, li, "vb", e.values_biases, 0, 128, s);
        }
    }
    try testing.expectEqual(@as(usize, 18), a.items.len);

    const before = serialize_eval_count.load(.monotonic);
    const got = try tier.serializeSafetensors(a.items, DiskTier.no_meta, s);
    defer testing.allocator.free(got);
    try testing.expectEqual(@as(u64, 1), serialize_eval_count.load(.monotonic) - before);

    try materializeLegacyPerTensorForTest(b.items, s);
    const mid = serialize_eval_count.load(.monotonic);
    const want = try tier.encodeSafetensors(b.items, DiskTier.no_meta);
    defer testing.allocator.free(want);
    try testing.expectEqual(mid, serialize_eval_count.load(.monotonic));
    try testing.expect(got.len > 4096);
    try testing.expectEqualSlices(u8, want, got);
}

test "DiskTier: a failing synchronous writeMeta frees the final path exactly once" {
    // `writeMeta` allocates `final_path` and forks; a function-scope `errdefer` plus the
    // synchronous branch's `defer` freed it twice on any disk write failure. The missing
    // `e<id>/` directory makes `createFileAbsolute` fail, the first fallible step past the defer.
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const base = try tmpRoot(&tmp, io, &buf);

    var tier = try DiskTier.init(testing.allocator, io, base, "fp-writemeta-err", 0, 128);
    defer tier.deinit();
    try testing.expect(tier.writer == null);

    var toks = [_]u32{ 1, 2, 3 };
    var cb = [_]u64{64};
    var no_pos = [_]u32{};
    var no_sz = [_]u64{};
    const e = IndexEntry{
        .id = 777, // no e777/ directory was ever created
        .tokens = &toks,
        .kv_len = 3,
        .has_tools = false,
        .quant = kv_quant.KVQuantConfig.dense,
        .bytes = 64,
        .chunk_bytes = &cb,
        .ssm_positions = &no_pos,
        .ssm_bytes = &no_sz,
        .last_used = 0,
    };
    // The write must fail and leave exactly one free behind.
    if (tier.writeMeta(e)) |_| {
        return error.WriteMetaUnexpectedlySucceeded;
    } else |_| {}
}

test "materializeContiguous: the fresh-handle errdefer never outlives the transfer" {
    // The fresh handle's errdefer must not stay armed across `t.arr = cont`, or a later failure
    // frees the same array twice. Transliterated onto a heap allocation so the testing
    // allocator catches both a leak and a double free.
    const t = testing;

    const Item = struct { arr: *u32 };
    const S = struct {
        fn run(a: std.mem.Allocator, items: []Item, fail_at: usize) !void {
            for (items, 0..) |*it, i| {
                const cont = try a.create(u32);
                {
                    errdefer a.destroy(cont);
                    cont.* = it.arr.* + 1;
                }
                a.destroy(it.arr);
                it.arr = cont;
                if (i == fail_at) return error.Injected;
            }
        }
    };

    var items: [4]Item = undefined;
    for (&items, 0..) |*it, i| {
        it.* = .{ .arr = try testing.allocator.create(u32) };
        it.arr.* = @intCast(i);
    }
    // The caller's `defer freeNamed(&list)`: it owns every `arr`, on every outcome.
    defer for (&items) |*it| testing.allocator.destroy(it.arr);
    try t.expectError(error.Injected, S.run(testing.allocator, &items, 2));
    try t.expectEqual(@as(u32, 1), items[0].arr.*);
    try t.expectEqual(@as(u32, 3), items[2].arr.*);
    try t.expectEqual(@as(u32, 3), items[3].arr.*);
}

// ── SSD-first chunk sharing ──

/// Two 600-token sequences that agree for the first `shared` tokens and diverge after.
const ShareToks = struct { a: [600]u32, b: [600]u32 };
fn chunkShareTokens(shared: usize) ShareToks {
    var out: ShareToks = undefined;
    for (&out.a, 0..) |*t, i| t.* = @intCast(i + 7);
    for (&out.b, 0..) |*t, i| t.* = if (i < shared) @intCast(i + 7) else @intCast(9000 + i);
    return out;
}

fn chunkStat(io: std.Io, base: []const u8, fp: []const u8, id: u64, chunk: u32) ?std.Io.File.Stat {
    var buf: [1024]u8 = undefined;
    const p = std.fmt.bufPrint(&buf, "{s}/{s}/e{d}/c{d:0>6}.safetensors", .{ base, fp, id, chunk }) catch return null;
    return statFile(io, p);
}

test "commonPrefixLen: the longest shared prefix, never past the shorter slice" {
    const a = [_]u32{ 1, 2, 3, 4 };
    const b = [_]u32{ 1, 2, 9, 4, 5 };
    try testing.expectEqual(@as(usize, 2), commonPrefixLen(&a, &b));
    try testing.expectEqual(@as(usize, 4), commonPrefixLen(&a, &a));
    try testing.expectEqual(@as(usize, 0), commonPrefixLen(&a, &[_]u32{}));
    try testing.expectEqual(@as(usize, 3), commonPrefixLen(&a, a[0..3]));
}

test "DiskTier chunk share: a prefix-diverging entry hard-links the donor's whole chunks, writes only its tail, bills once, restores whole" {
    // Turn N+1's prompt diverges inside turn N's generated span: the heir links the whole
    // chunks below the common prefix and writes the rest.
    const io = std.testing.io;
    const s = mlx.gpuStream();
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const base = try tmpRoot(&tmp, io, &buf);

    chunk_share_override = true;
    defer chunk_share_override = null;
    var tier = try DiskTier.init(testing.allocator, io, base, "fp-share", 0, 128);
    defer tier.deinit();
    tier.ssd_first = true;
    tier.armTestSpace(1024 * 1024 * 1024 * 1024, 2048 * 1024 * 1024 * 1024);

    // 600 tokens => chunks 0..3 whole (512 tokens), chunk 4 partial (88).
    var cache = try KVCache.init(testing.allocator, 3);
    defer cache.deinit();
    try fillCache(&cache, s, 3, 600, 8, 0.0, .float32);
    const toks = chunkShareTokens(520); // diverges at 520: 4 whole chunks shared
    _ = try tier.appendCommit(cache.entries, cache.step, cache.config, &toks.a, false, null, s);
    try testing.expectEqual(@as(usize, 1), tier.entryCount());
    const donor_bytes = tier.total_bytes;
    const donor_id = tier.entries.items[0].id;

    _ = try tier.appendCommit(cache.entries, cache.step, cache.config, &toks.b, false, null, s);
    try testing.expectEqual(@as(usize, 2), tier.entryCount());
    const heir = &tier.entries.items[1];
    try testing.expectEqual(@as(u32, 4), heir.inherited_chunks);
    try testing.expectEqual(@as(usize, 5), heir.chunk_bytes.len);
    try testing.expectEqual(@as(u32, 600), heir.kv_len);
    // The leading files are one inode with two links; the tail is its own.
    const d0 = chunkStat(io, base, "fp-share", donor_id, 0).?;
    const h0 = chunkStat(io, base, "fp-share", heir.id, 0).?;
    try testing.expectEqual(d0.inode, h0.inode);
    try testing.expectEqual(@as(u64, 2), @as(u64, @intCast(h0.nlink)));
    const h4 = chunkStat(io, base, "fp-share", heir.id, 4).?;
    try testing.expectEqual(@as(u64, 1), @as(u64, @intCast(h4.nlink)));
    try testing.expectEqual(donor_bytes + heir.chunk_bytes[4] + 600 * 4, tier.total_bytes);
    try testing.expectEqual(heir.chunk_bytes[4] + 600 * 4, heir.bytes);

    // The heir restores through a fresh tier, and its bill survives the scan.
    var tier2 = try DiskTier.init(testing.allocator, io, base, "fp-share", 0, 128);
    defer tier2.deinit();
    try testing.expectEqual(@as(usize, 2), tier2.entryCount());
    try testing.expectEqual(tier.total_bytes, tier2.total_bytes);
    const m = tier2.bestMatch(&toks.b, false, kv_quant.KVQuantConfig.dense).?;
    try testing.expectEqual(@as(u32, 600), m.usable);
    var out = try KVCache.init(testing.allocator, 3);
    defer out.deinit();
    try tier2.restorePrefixInto(&out, m.idx, 600, s);
    try testing.expectEqual(@as(usize, 600), out.step);
    inline for (.{ 0, 300, 599 }) |pos| {
        try testing.expectEqual(try cacheValueAt(&cache, 1, pos, 3, s), try cacheValueAt(&out, 1, pos, 3, s));
    }
}

test "DiskTier chunk share: total_bytes is bytes on disk whichever holder dies first" {
    // The filesystem is the refcount: donor-then-heir and heir-then-donor both land back on the pre-commit number.
    const io = std.testing.io;
    const s = mlx.gpuStream();
    chunk_share_override = true;
    defer chunk_share_override = null;
    var cache = try KVCache.init(testing.allocator, 3);
    defer cache.deinit();
    try fillCache(&cache, s, 3, 600, 8, 0.0, .float32);
    const toks = chunkShareTokens(520);

    for ([_]bool{ true, false }) |donor_first| {
        var tmp = std.testing.tmpDir(.{ .iterate = true });
        defer tmp.cleanup();
        var buf: [512]u8 = undefined;
        const base = try tmpRoot(&tmp, io, &buf);
        var tier = try DiskTier.init(testing.allocator, io, base, "fp-order", 0, 128);
        defer tier.deinit();
        tier.ssd_first = true;
        tier.armTestSpace(1024 * 1024 * 1024 * 1024, 2048 * 1024 * 1024 * 1024);
        try testing.expectEqual(@as(u64, 0), tier.total_bytes);

        _ = try tier.appendCommit(cache.entries, cache.step, cache.config, &toks.a, false, null, s);
        _ = try tier.appendCommit(cache.entries, cache.step, cache.config, &toks.b, false, null, s);
        try testing.expectEqual(@as(usize, 2), tier.entryCount());
        const donor_idx: usize = 0;
        const heir_idx: usize = 1;
        const donor_id = tier.entries.items[donor_idx].id;
        const heir_id = tier.entries.items[heir_idx].id;
        var shared_bytes: u64 = 0;
        for (tier.entries.items[heir_idx].chunk_bytes[0..4]) |b| shared_bytes += b;
        const donor_own = tier.entries.items[donor_idx].bytes - shared_bytes; // its tail + tokens
        const heir_own = tier.entries.items[heir_idx].bytes; // its tail + tokens (links billed 0)
        try testing.expectEqual(shared_bytes + donor_own + heir_own, tier.total_bytes);

        if (donor_first) {
            tier.removeAt(donor_idx);
            try testing.expectEqual(shared_bytes + heir_own, tier.total_bytes);
            try testing.expect(chunkStat(io, base, "fp-order", heir_id, 0) != null);
            try testing.expect(chunkStat(io, base, "fp-order", donor_id, 0) == null);
            try testing.expectEqual(@as(u64, 1), @as(u64, @intCast(chunkStat(io, base, "fp-order", heir_id, 0).?.nlink)));
            tier.removeAt(0);
        } else {
            tier.removeAt(heir_idx);
            try testing.expectEqual(shared_bytes + donor_own, tier.total_bytes);
            try testing.expect(chunkStat(io, base, "fp-order", donor_id, 0) != null);
            try testing.expectEqual(@as(u64, 1), @as(u64, @intCast(chunkStat(io, base, "fp-order", donor_id, 0).?.nlink)));
            tier.removeAt(0);
        }
        try testing.expectEqual(@as(usize, 0), tier.entryCount());
        try testing.expectEqual(@as(u64, 0), tier.total_bytes);
    }
}

test "DiskTier chunk share: the legacy arm and the kill switch never link" {
    const io = std.testing.io;
    const s = mlx.gpuStream();
    var cache = try KVCache.init(testing.allocator, 3);
    defer cache.deinit();
    try fillCache(&cache, s, 3, 600, 8, 0.0, .float32);
    const toks = chunkShareTokens(520);
    // Arm 1: legacy tier, switch on. Arm 2: SSD-first, switch off.
    for ([_]struct { ssd: bool, share: bool }{ .{ .ssd = false, .share = true }, .{ .ssd = true, .share = false } }) |arm| {
        var tmp = std.testing.tmpDir(.{ .iterate = true });
        defer tmp.cleanup();
        var buf: [512]u8 = undefined;
        const base = try tmpRoot(&tmp, io, &buf);
        chunk_share_override = arm.share;
        defer chunk_share_override = null;
        var tier = try DiskTier.init(testing.allocator, io, base, "fp-nolink", 0, 128);
        defer tier.deinit();
        tier.ssd_first = arm.ssd;
        tier.armTestSpace(1024 * 1024 * 1024 * 1024, 4096 * 1024 * 1024 * 1024);
        _ = try tier.appendCommit(cache.entries, cache.step, cache.config, &toks.a, false, null, s);
        const before = tier.total_bytes;
        _ = try tier.appendCommit(cache.entries, cache.step, cache.config, &toks.b, false, null, s);
        const heir = &tier.entries.items[1];
        try testing.expectEqual(@as(u32, 0), heir.inherited_chunks);
        try testing.expectEqual(@as(u64, 1), @as(u64, @intCast(chunkStat(io, base, "fp-nolink", heir.id, 0).?.nlink)));
        var all: u64 = 600 * 4;
        for (heir.chunk_bytes) |b| all += b;
        try testing.expectEqual(before + all, tier.total_bytes);
    }
}

test "DiskTier chunk share: meta v6 carries inherited_chunks; a v5 manifest loads with none" {
    const io = std.testing.io;
    const s = mlx.gpuStream();
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const base = try tmpRoot(&tmp, io, &buf);
    chunk_share_override = true;
    defer chunk_share_override = null;
    var cache = try KVCache.init(testing.allocator, 3);
    defer cache.deinit();
    try fillCache(&cache, s, 3, 600, 8, 0.0, .float32);
    const toks = chunkShareTokens(520);
    var heir_id: u64 = 0;
    {
        var tier = try DiskTier.init(testing.allocator, io, base, "fp-v6", 0, 128);
        defer tier.deinit();
        tier.ssd_first = true;
        tier.armTestSpace(1024 * 1024 * 1024 * 1024, 2048 * 1024 * 1024 * 1024);
        _ = try tier.appendCommit(cache.entries, cache.step, cache.config, &toks.a, false, null, s);
        _ = try tier.appendCommit(cache.entries, cache.step, cache.config, &toks.b, false, null, s);
        heir_id = tier.entries.items[1].id;
    }
    var mp: [128]u8 = undefined;
    const meta_rel = try std.fmt.bufPrint(&mp, "fp-v6/e{d}/meta.json", .{heir_id});
    const meta = try tmp.dir.readFileAlloc(io, meta_rel, testing.allocator, .limited(1 << 20));
    defer testing.allocator.free(meta);
    try testing.expect(std.mem.indexOf(u8, meta, "\"v\":6") != null);
    try testing.expect(std.mem.indexOf(u8, meta, "\"inherited_chunks\":4") != null);
    {
        var tier = try DiskTier.init(testing.allocator, io, base, "fp-v6", 0, 128);
        defer tier.deinit();
        var found = false;
        for (tier.entries.items) |*e| {
            if (e.id == heir_id) {
                found = true;
                try testing.expectEqual(@as(u32, 4), e.inherited_chunks);
            }
        }
        try testing.expect(found);
    }
    // An older binary's manifest (v5, no field) still loads: nothing inherited.
    var rewritten = std.ArrayList(u8).empty;
    defer rewritten.deinit(testing.allocator);
    try rewritten.appendSlice(testing.allocator, meta);
    _ = std.mem.replace(u8, rewritten.items, "\"v\":6", "\"v\":5", rewritten.items);
    const stripped = try std.mem.replaceOwned(u8, testing.allocator, rewritten.items, "\"inherited_chunks\":4,", "");
    defer testing.allocator.free(stripped);
    try tmp.dir.writeFile(io, .{ .sub_path = meta_rel, .data = stripped });
    {
        var tier = try DiskTier.init(testing.allocator, io, base, "fp-v6", 0, 128);
        defer tier.deinit();
        for (tier.entries.items) |*e| {
            if (e.id == heir_id) try testing.expectEqual(@as(u32, 0), e.inherited_chunks);
        }
    }
}

test "DiskTier chunk share: an heir links ONLY the donor's landed chunks; the donor's queued chunks and meta drain intact" {
    // A donor mid-persist is the common case: the heir links what is on disk, writes the rest,
    // and never touches the donor's queue.
    const io = std.testing.io;
    const s = mlx.gpuStream();
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const base = try tmpRoot(&tmp, io, &buf);
    chunk_share_override = true;
    defer chunk_share_override = null;

    var tier = try DiskTier.init(testing.allocator, io, base, "fp-landed", 0, 128);
    defer tier.deinit();
    tier.ssd_first = true;
    tier.armTestSpace(1024 * 1024 * 1024 * 1024, 2048 * 1024 * 1024 * 1024);
    tier.enableBackgroundWriter();
    try testing.expect(tier.writer != null);

    var cache = try KVCache.init(testing.allocator, 3);
    defer cache.deinit();
    try fillCache(&cache, s, 3, 600, 8, 0.0, .float32);
    const toks = chunkShareTokens(520);

    // The bound goes on the donor's commits (the write-through hook's form), not on the tier;
    // the heir commits unbounded, the way `flushPendingDisk` completes an entry.
    const donor_bound: u64 = 1;

    // Donor: chunks 0 and 1 landed, chunk 2 queued (writer paused).
    _ = try tier.appendCommitBounded(cache.entries, cache.step, cache.config, &toks.a, false, null, s, donor_bound);
    tier.drainWriter();
    _ = try tier.appendCommitBounded(cache.entries, cache.step, cache.config, &toks.a, false, null, s, donor_bound);
    tier.drainWriter();
    try testing.expectEqual(@as(u32, 256), tier.entries.items[0].kv_len);
    tier.writer.?.setPaused(true);
    defer tier.writer.?.setPaused(false);
    _ = try tier.appendCommitBounded(cache.entries, cache.step, cache.config, &toks.a, false, null, s, donor_bound);
    try testing.expectEqual(@as(u32, 384), tier.entries.items[0].kv_len);
    const donor_id = tier.entries.items[0].id;
    try testing.expect(chunkStat(io, base, "fp-landed", donor_id, 2) == null); // still queued
    const dropped_before = tier.writer.?.files_dropped;

    // Heir: the overlap allows 4 whole chunks; only 2 have landed, so it links 2 and writes 2..4.
    _ = try tier.appendCommit(cache.entries, cache.step, cache.config, &toks.b, false, null, s);
    try testing.expectEqual(@as(usize, 2), tier.entryCount());
    const heir = &tier.entries.items[1];
    try testing.expectEqual(@as(u32, 2), heir.inherited_chunks);
    try testing.expectEqual(@as(usize, 5), heir.chunk_bytes.len);
    try testing.expectEqual(@as(u32, 600), heir.kv_len);

    // The donor's queue was never touched.
    try testing.expectEqual(dropped_before, tier.writer.?.files_dropped);
    tier.writer.?.setPaused(false);
    tier.drainWriter();
    try testing.expectEqual(@as(u64, 0), tier.writeErrors());
    const d2 = chunkStat(io, base, "fp-landed", donor_id, 2).?;
    try testing.expectEqual(tier.entries.items[0].chunk_bytes[2], d2.size);
    try testing.expectEqual(@as(u64, 1), @as(u64, @intCast(d2.nlink))); // the heir wrote its own chunk 2
    inline for (.{ 0, 1 }) |i| try testing.expectEqual(@as(u64, 2), @as(u64, @intCast(chunkStat(io, base, "fp-landed", heir.id, i).?.nlink)));
    inline for (.{ 2, 3, 4 }) |i| try testing.expectEqual(@as(u64, 1), @as(u64, @intCast(chunkStat(io, base, "fp-landed", heir.id, i).?.nlink)));

    var tier2 = try DiskTier.init(testing.allocator, io, base, "fp-landed", 0, 128);
    defer tier2.deinit();
    try testing.expectEqual(@as(usize, 2), tier2.entryCount());
    for (tier2.entries.items) |*e| {
        if (e.id == donor_id) try testing.expectEqual(@as(u32, 384), e.kv_len);
        if (e.id == heir.id) {
            try testing.expectEqual(@as(u32, 600), e.kv_len);
            try testing.expectEqual(@as(u32, 2), e.inherited_chunks);
        }
    }
}

test "DiskTier: deinit RETURNS with the writer paused, and lands what was queued" {
    // `DiskTier.deinit` drains before it deinits the writer; teardown must lift a pause itself.
    const io = std.testing.io;
    const s = mlx.gpuStream();
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const base = try tmpRoot(&tmp, io, &buf);

    var cache = try KVCache.init(testing.allocator, 3);
    defer cache.deinit();
    try fillCache(&cache, s, 3, 640, 8, 0.0, .float32); // > MIN_PERSIST_TOKENS
    var tokens: [640]u32 = undefined;
    for (&tokens, 0..) |*t, i| t.* = @intCast(i + 7);

    var tier = try DiskTier.init(testing.allocator, io, base, "fp-paused-deinit", 0, 128);
    tier.ssd_first = true;
    tier.armTestSpace(1024 * 1024 * 1024 * 1024, 2048 * 1024 * 1024 * 1024);
    tier.enableBackgroundWriter();
    try testing.expect(tier.writer != null);
    tier.writer.?.setPaused(true);
    // No deferred unpause on purpose: teardown is the release under test.
    _ = try tier.appendCommit(cache.entries, cache.step, cache.config, &tokens, false, null, s);
    try testing.expect(tier.writer.?.pendingBytes() > 0);

    // Not deferred: the bar is that the call returns.
    tier.deinit();

    // It lifted the pause rather than skipping the drain: the entry is really on disk.
    var tier2 = try DiskTier.init(testing.allocator, io, base, "fp-paused-deinit", 0, 128);
    defer tier2.deinit();
    try testing.expectEqual(@as(usize, 1), tier2.entryCount());
    try testing.expectEqual(@as(u32, 640), tier2.entries.items[0].kv_len);
}

test "DiskTier: an ssm/spec-only append bills the SPEC sidecar's byte delta" {
    // `appendSsmOnly` overwrote `e.spec_bytes` before taking its delta, so every commit landing
    // a spec sidecar onto a complete entry left `total_bytes` short by the sidecar (every arch).
    const io = std.testing.io;
    const s = mlx.gpuStream();
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const base = try tmpRoot(&tmp, io, &buf);

    var tier = try DiskTier.init(testing.allocator, io, base, "fp-specbill", 0, 128);
    defer tier.deinit();

    var cache = try KVCache.init(testing.allocator, 2);
    defer cache.deinit();
    try fillCache(&cache, s, 2, 600, 8, 0.0, .float32);
    var tokens: [600]u32 = undefined;
    for (&tokens, 0..) |*t, i| t.* = @intCast(i + 7);

    // Turn 1: the entry lands complete, with no spec sidecar.
    _ = try tier.appendCommit(cache.entries, cache.step, cache.config, &tokens, false, null, s);
    try testing.expectEqual(@as(usize, 1), tier.entries.items.len);
    try testing.expectEqual(@as(u64, 0), tier.entries.items[0].spec_bytes);
    const bytes_before = tier.entries.items[0].bytes;
    const total_before = tier.total_bytes;

    // Turn 2: same tokens, same KV, now carrying an MTP history snap (`appendSsmOnly`).
    var mtp = try KVCache.init(testing.allocator, 1);
    defer mtp.deinit();
    try fillCache(&mtp, s, 1, 590, 8, 9.5, .float32);
    _ = try tier.appendCommitWithSpec(
        cache.entries,
        cache.step,
        cache.config,
        &tokens,
        false,
        null,
        null,
        .{ .entries = mtp.entries, .step = mtp.step, .config = mtp.config, .base_pos = 0 },
        s,
    );

    const e = &tier.entries.items[0];
    try testing.expect(e.spec_bytes > 0); // the sidecar really was written
    try testing.expectEqual(bytes_before + e.spec_bytes, e.bytes);
    try testing.expectEqual(total_before + e.spec_bytes, tier.total_bytes);
    var own_chunks: u64 = 0;
    for (e.chunk_bytes[@min(e.inherited_chunks, e.chunk_bytes.len)..]) |b| own_chunks += b;
    try testing.expectEqual(nonChunkBytes(e) + own_chunks, e.bytes);

    // A rescan of the same root reaches the same total.
    var rescanned = try DiskTier.init(testing.allocator, io, base, "fp-specbill", 0, 128);
    defer rescanned.deinit();
    try testing.expectEqual(@as(usize, 1), rescanned.entryCount());
    try testing.expectEqual(tier.total_bytes, rescanned.total_bytes);
}

test "DiskTier: the manifest stamps the LOWEST version that describes the entry" {
    // The version is a compatibility claim: an older reader accepts 2..4 only, so a v6 stamp on
    // every entry made a downgrade discard the whole tier.
    const t = std.testing;

    // Plain entry: v4.
    var toks = [_]u32{ 1, 2, 3 };
    var cbytes = [_]u64{4096};
    var spos = [_]u32{};
    var sbytes = [_]u64{};
    const kv_only = SpecMeta{ .base = 0, .step = 600, .layers = 2, .quant = kv_quant.KVQuantConfig.dense };
    const with_head = SpecMeta{
        .base = 0,
        .step = 600,
        .layers = 1,
        .quant = kv_quant.KVQuantConfig.dense,
        .head = .{ .pos_base = 1, .ratio = 4, .pooled = true },
    };

    var e = IndexEntry{
        .id = 1,
        .tokens = &toks,
        .kv_len = 600,
        .has_tools = false,
        .quant = kv_quant.KVQuantConfig.dense,
        .bytes = 4096,
        .chunk_bytes = &cbytes,
        .ssm_positions = &spos,
        .ssm_bytes = &sbytes,
        .last_used = 1,
    };
    try t.expectEqual(@as(u8, 4), DiskTier.metaVersionFor(e));

    // A dflash sidecar is v4 shape too.
    e.spec_bytes = 4096;
    e.spec_dflash = kv_only;
    try t.expectEqual(@as(u8, 4), DiskTier.metaVersionFor(e));

    // A KV-only MTP snap is still v4; the qwen4_exp head's QSA half lifts it to v5.
    e.spec_mtp = kv_only;
    try t.expectEqual(@as(u8, 4), DiskTier.metaVersionFor(e));
    e.spec_mtp = with_head;
    try t.expectEqual(@as(u8, 5), DiskTier.metaVersionFor(e));

    // Inherited (hard-linked) chunks are v6: an older reader would bill and delete a donor's files.
    e.inherited_chunks = 4;
    try t.expectEqual(@as(u8, 6), DiskTier.metaVersionFor(e));
    e.spec_mtp = null;
    try t.expectEqual(@as(u8, 6), DiskTier.metaVersionFor(e));

    e.qsa_history_rows = 256;
    e.qsa_history_bytes = 4096;
    try t.expectEqual(@as(u8, 8), DiskTier.metaVersionFor(e));
}

test "DiskTier: a v7 full-aux QSA file serves a mid-block leftover inside a RING_ROWS ring; v>8 is refused" {
    const io = std.testing.io;
    const s = mlx.gpuStream();
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const base = try tmpRoot(&tmp, io, &buf);

    var tier = try DiskTier.init(testing.allocator, io, base, "fp-qsa-v8", 0, 128);
    defer tier.deinit();

    var cache = try KVCache.init(testing.allocator, 3);
    defer cache.deinit();
    try fillCache(&cache, s, 3, 600, 8, 0.0, .float32);
    var tokens: [600]u32 = undefined;
    for (&tokens, 0..) |*t, i| t.* = @intCast(i + 7);

    const hd: c_int = 8;
    const aux_shape = [_]c_int{ 1, 256, hd };
    const pooled_shape = [_]c_int{ 1, 64, hd };
    var src = buildHybridEntries(s, 200.0, 600.0);
    defer freeHybridEntries(&src);
    src[2].aux_state = makeArange(s, &aux_shape, 700.0);
    src[2].qsa_pooled = makeArange(s, &pooled_shape, 800.0);
    src[2].qsa_ratio = 4;
    var cps = [_]transformer_mod.SSMCheckpoint{
        try transformer_mod.captureSsmCheckpoint(testing.allocator, &src, 46, s),
        try transformer_mod.captureSsmCheckpoint(testing.allocator, &src, 256, s),
    };
    defer for (&cps) |*cp| cp.deinit(testing.allocator);
    // A v7 entry: its snaps carry recurrent state and `qsa_rows` only, and the whole raw
    // indexer history — every checkpoint's leftover included — is the entry-level file.
    for (&cps) |*cp| for (cp.layers) |*l| {
        if (l.aux_state.ctx == null) continue;
        _ = mlx.mlx_array_free(l.aux_state);
        l.aux_state = .{ .ctx = null };
    };
    _ = try tier.appendCommit(cache.entries, cache.step, cache.config, &tokens, false, &cps, s);
    const id = tier.entries.items[0].id;
    const dir_rel = try std.fmt.allocPrint(testing.allocator, "{s}/e{d}", .{ tier.root, id });
    defer testing.allocator.free(dir_rel);
    const full_layers = try testing.allocator.alloc(transformer_mod.SSMCacheEntrySnapshot, 3);
    for (full_layers) |*l| l.* = .{
        .conv_state = mlx.mlx_array_new(),
        .ssm_state = mlx.mlx_array_new(),
        .initialized = false,
    };
    full_layers[2].aux_state = makeArange(s, &aux_shape, 700.0);
    full_layers[2].qsa_pooled = makeArange(s, &pooled_shape, 800.0);
    full_layers[2].qsa_ratio = 4;
    full_layers[2].qsa_rows = 256;
    var full_cp: transformer_mod.SSMCheckpoint = .{ .pos = 256, .layers = full_layers };
    defer full_cp.deinit(testing.allocator);
    _ = try tier.writeQsaHistoryFile(dir_rel, &full_cp, s);

    // The file keeps every row it was written with; the ring truncation is the live entry's.
    var loaded = try tier.loadQsaHistoryFile(id, 3);
    defer loaded.deinit(testing.allocator);
    try testing.expectEqual(@as(c_int, 256), loaded.layers[2].qsa_rows);
    try testing.expectEqual(@as(c_int, 256), mlx.getShape(loaded.layers[2].aux_state)[1]);

    var tier2 = try DiskTier.init(testing.allocator, io, base, "fp-qsa-v8", 0, 128);
    defer tier2.deinit();
    var cache2 = try KVCache.init(testing.allocator, 3);
    defer cache2.deinit();
    var dst: [3]SSMCacheEntry = .{
        .{ .conv_state = mlx.mlx_array_new(), .ssm_state = mlx.mlx_array_new(), .initialized = false },
        .{ .conv_state = mlx.mlx_array_new(), .ssm_state = mlx.mlx_array_new(), .initialized = false },
        .{ .conv_state = mlx.mlx_array_new(), .ssm_state = mlx.mlx_array_new(), .initialized = false },
    };
    defer freeHybridEntries(&dst);
    try testing.expectEqual(@as(u32, 46), try tier2.restoreIntoHybrid(&cache2, &dst, 0, 46, s));
    try testing.expectEqual(@as(c_int, 46), dst[2].qsa_hist_rows);
    try testing.expectEqual(@as(c_int, 11), mlx.getShape(dst[2].qsa_pooled)[1]);
    try testing.expectEqual(@as(f32, 800.0), ssmArrVal(dst[2].qsa_pooled, 0, s));
    // A restore at 46 (ratio 4) seeds exactly the 2 leftover rows 44 and 45, which are the checkpoint's
    // own leftover — source rows 44 and 45.
    const leftover = mlx.getShape(dst[2].aux_state)[1];
    try testing.expectEqual(@as(c_int, 2), leftover);
    try testing.expectEqual(@as(f32, 1052.0), ssmArrVal(dst[2].aux_state, 0, s));
    try testing.expectEqual(@as(f32, 1060.0), ssmArrVal(dst[2].aux_state, hd, s));

    try tmp.dir.createDirPath(io, "fp-qsa-v8/e9");
    try tmp.dir.writeFile(io, .{
        .sub_path = "fp-qsa-v8/e9/meta.json",
        .data = "{\"v\":9,\"kv_len\":1,\"tokens\":0,\"has_tools\":false,\"scheme\":\"off\",\"bits\":0,\"group_size\":0,\"chunk_tokens\":128,\"inherited_chunks\":0,\"bytes\":0,\"chunk_bytes\":[],\"ssm\":[]}",
    });
    try testing.expect(tier.loadEntry(9) == null);
}

test "DiskTier: the per-entry checkpoint cap is gated; a legacy tier keeps 8" {
    // The cap of 16 was sized against qwen4_exp alone; every other arch keeps 8.
    const t = std.testing;
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const base = try tmpRoot(&tmp, io, &buf);

    var legacy = try DiskTier.init(testing.allocator, io, base, "fp-cap-legacy", 0, 128);
    defer legacy.deinit();
    try t.expectEqual(@as(usize, 8), SSM_DISK_MAX_PER_ENTRY_LEGACY);
    try t.expectEqual(SSM_DISK_MAX_PER_ENTRY_LEGACY, legacy.ssm_max_per_entry);

    var positions: [40]u32 = undefined;
    for (&positions, 0..) |*p, i| p.* = @intCast((i + 1) * 4096);
    const L: u32 = 40 * 4096;

    const old_kept = try legacy.ssmTargetPositions(&positions, &[_]transformer_mod.SSMCheckpoint{}, L);
    defer testing.allocator.free(old_kept);
    try t.expectEqual(SSM_DISK_MAX_PER_ENTRY_LEGACY, old_kept.len);

    var gated = try DiskTier.init(testing.allocator, io, base, "fp-cap-gated", 0, 128);
    defer gated.deinit();
    gated.cp_thin = .min_span_recency;
    gated.ssm_max_per_entry = SSM_DISK_MAX_PER_ENTRY;
    const new_kept = try gated.ssmTargetPositions(&positions, &[_]transformer_mod.SSMCheckpoint{}, L);
    defer testing.allocator.free(new_kept);
    try t.expectEqual(SSM_DISK_MAX_PER_ENTRY, new_kept.len);
    try t.expectEqual(@as(usize, 2 * SSM_DISK_MAX_PER_ENTRY_LEGACY), new_kept.len);
}

test "DiskTier: a failed background write INVALIDATES the entry it belonged to (no completion claim, restore misses)" {
    // The writer drops a failed blob, and `writeMeta` appends the index before a byte reaches
    // disk, so the RAM copy was evicted against a hole. The failed path names the entry.
    const io = std.testing.io;
    const s = mlx.gpuStream();
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const base = try tmpRoot(&tmp, io, &buf);

    var tier = try DiskTier.init(testing.allocator, io, base, "fp-poison", 0, 128);
    defer tier.deinit();
    tier.ssd_first = true;
    tier.armTestSpace(1024 * 1024 * 1024 * 1024, 2048 * 1024 * 1024 * 1024);
    tier.enableBackgroundWriter();
    try testing.expect(tier.writer != null);

    var cache = try KVCache.init(testing.allocator, 2);
    defer cache.deinit();
    try fillCache(&cache, s, 2, 600, 8, 0.0, .float32);
    var tokens: [600]u32 = undefined;
    for (&tokens, 0..) |*t, i| t.* = @intCast(i + 7);

    // Hold the writer so the failure lands strictly after the commit.
    tier.writer.?.setPaused(true);
    defer tier.writer.?.setPaused(false);
    tier.writer.?.injectFailure("c000003", .write);
    _ = try tier.appendCommit(cache.entries, cache.step, cache.config, &tokens, false, null, s);
    const dead_id = tier.entries.items[0].id;
    tier.writer.?.setPaused(false);
    tier.drainWriter();
    try testing.expect(tier.writeErrors() > 0);
    // meta.json rode the same FIFO and did land: the index still describes five whole chunks.
    try testing.expectEqual(@as(usize, 5), tier.entries.items[0].chunk_bytes.len);

    try testing.expectEqual(@as(usize, 1), tier.harvestWriteFailures());
    try testing.expect(tier.entries.items[0].poisoned);
    for (tier.entries.items[0].chunk_bytes) |b| try testing.expectEqual(@as(u64, 0), b);
    try testing.expect(!tier.holdsFullPrefix(cache.entries, cache.step, &tokens, false, cache.config));
    try testing.expect(tier.fullPrefixEntryId(cache.entries, cache.step, &tokens, false, cache.config) == null);
    try testing.expect(tier.bestMatch(&tokens, false, cache.config) == null);
    try testing.expect(!tier.entryWholeOnDisk(dead_id));
    // Attribution happens once.
    try testing.expectEqual(@as(usize, 0), tier.harvestWriteFailures());

    // The next commit reclaims the dead directory and rebuilds from scratch.
    tier.writer.?.injectFailure(null, .write);
    const out = try tier.appendCommit(cache.entries, cache.step, cache.config, &tokens, false, null, s);
    try testing.expectEqual(PersistOutcome.persisted, out);
    tier.drainWriter();
    try testing.expectEqual(@as(usize, 1), tier.entryCount());
    const live = &tier.entries.items[0];
    try testing.expect(live.id != dead_id);
    try testing.expect(!live.poisoned);
    try testing.expect(tier.entryWholeOnDisk(live.id));
    try testing.expect(tier.holdsFullPrefix(cache.entries, cache.step, &tokens, false, cache.config));

    var back = try KVCache.init(testing.allocator, 2);
    defer back.deinit();
    try testing.expectEqual(@as(u32, 600), try tier.restoreInto(&back, 0, s));
}

test "DiskTier: a poisoned entry is invisible to the hybrid lookup too" {
    // Poisoning is the tier's "this entry is dead" mark; the hybrid arm is the lookup qwen4_exp
    // actually takes, so a gap there makes the mark meaningless.
    const io = std.testing.io;
    const s = mlx.gpuStream();
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const base = try tmpRoot(&tmp, io, &buf);

    var tier = try DiskTier.init(testing.allocator, io, base, "fp-poison-hyb", 0, 128);
    defer tier.deinit();

    var cache = try KVCache.init(testing.allocator, 3);
    defer cache.deinit();
    try fillCache(&cache, s, 3, 600, 8, 0.0, .float32);
    var tokens: [600]u32 = undefined;
    for (&tokens, 0..) |*t, i| t.* = @intCast(i + 7);
    var src = buildHybridEntries(s, 100.0, 500.0);
    defer freeHybridEntries(&src);
    var cps: [1]transformer_mod.SSMCheckpoint = .{try transformer_mod.captureSsmCheckpoint(testing.allocator, &src, 512, s)};
    defer for (&cps) |*cp| cp.deinit(testing.allocator);
    _ = try tier.appendCommit(cache.entries, cache.step, cache.config, &tokens, false, &cps, s);

    try testing.expect(tier.bestHybridMatch(&tokens, false, cache.config, 600) != null);
    tier.entries.items[0].poisoned = true;
    try testing.expect(tier.bestMatch(&tokens, false, cache.config) == null);
    try testing.expect(tier.bestHybridMatch(&tokens, false, cache.config, 600) == null);
}

test "DiskTier: a manifest scalar that does not fit its field drops the record, never casts" {
    // meta.json is on disk and hand-editable; an unchecked @intCast of a corrupt scalar is
    // ReleaseFast UB.
    const io = std.testing.io;
    const s = mlx.gpuStream();
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const base = try tmpRoot(&tmp, io, &buf);

    var tier = try DiskTier.init(testing.allocator, io, base, "fp-badmeta", 0, 128);
    defer tier.deinit();

    const qcfg = kv_quant.KVQuantConfig.affine(4);
    var cache = try KVCache.initWithConfig(testing.allocator, 2, qcfg);
    defer cache.deinit();
    try fillCache(&cache, s, 2, 520, 64, 0.0, .bfloat16);
    var mtp = try KVCache.init(testing.allocator, 1);
    defer mtp.deinit();
    try fillCache(&mtp, s, 1, 520, 8, 9.5, .float32);
    var tokens: [520]u32 = undefined;
    for (&tokens, 0..) |*t, i| t.* = @intCast(i + 7);
    _ = try tier.appendCommitWithSpec(
        cache.entries,
        cache.step,
        cache.config,
        &tokens,
        false,
        null,
        null,
        .{ .entries = mtp.entries, .step = mtp.step, .config = mtp.config, .base_pos = 0 },
        s,
    );
    const id = tier.entries.items[0].id;
    const meta_path = try std.fmt.allocPrint(testing.allocator, "{s}/fp-badmeta/e{d}/meta.json", .{ base, id });
    defer testing.allocator.free(meta_path);
    const orig = readFileAlloc(testing.allocator, io, meta_path, 64 * 1024).?;
    defer testing.allocator.free(orig);
    if (tier.loadEntry(id)) |l| {
        var e = l.e;
        defer tier.freeIndexEntryOwned(&e);
        try testing.expect(e.spec_mtp != null);
    } else return error.TestUnexpectedResult;

    const cases = [_]struct { find: []const u8, replace: []const u8, entry_survives: bool }{
        .{ .find = "\"bits\":4", .replace = "\"bits\":4000", .entry_survives = false },
        .{ .find = "\"group_size\":64", .replace = "\"group_size\":68719476736", .entry_survives = false },
        .{ .find = "\"layers\":1", .replace = "\"layers\":68719476736", .entry_survives = true },
    };
    for (cases) |c| {
        const patched = try std.mem.replaceOwned(u8, testing.allocator, orig, c.find, c.replace);
        defer testing.allocator.free(patched);
        try testing.expect(!std.mem.eql(u8, patched, orig));
        try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = meta_path, .data = patched });
        if (tier.loadEntry(id)) |l| {
            var e = l.e;
            defer tier.freeIndexEntryOwned(&e);
            try testing.expect(c.entry_survives);
            // The spec half is the only part an out-of-range spec scalar costs.
            try testing.expect(e.spec_mtp == null);
        } else {
            try testing.expect(!c.entry_survives);
        }
    }
}

test "DiskTier: entryWholeOnDisk stats what the index NAMES (a truncated chunk fails it with a clean writer)" {
    // A byte can go missing with no write error at all; one stat per chunk catches it.
    const io = std.testing.io;
    const s = mlx.gpuStream();
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const base = try tmpRoot(&tmp, io, &buf);

    var tier = try DiskTier.init(testing.allocator, io, base, "fp-stat", 0, 128);
    defer tier.deinit();
    var cache = try KVCache.init(testing.allocator, 1);
    defer cache.deinit();
    try fillCache(&cache, s, 1, 600, 8, 0.0, .float32);
    var tokens: [600]u32 = undefined;
    for (&tokens, 0..) |*t, i| t.* = @intCast(i + 7);
    _ = try tier.appendCommit(cache.entries, cache.step, cache.config, &tokens, false, null, s);
    const id = tier.entries.items[0].id;
    try testing.expect(tier.entryWholeOnDisk(id));

    // Truncate one chunk behind the tier's back.
    try tmp.dir.writeFile(io, .{ .sub_path = "fp-stat/e1/c000002.safetensors", .data = "short" });
    try testing.expect(!tier.entryWholeOnDisk(id));
    try tmp.dir.deleteFile(io, "fp-stat/e1/c000002.safetensors");
    try testing.expect(!tier.entryWholeOnDisk(id));
    try testing.expect(!tier.entryWholeOnDisk(id + 999));
}
