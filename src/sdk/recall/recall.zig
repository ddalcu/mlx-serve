//! Recall: a persistent record of the token ids the server has read and
//! written, drafted from when a reply repeats text the record holds.
//!
//! A request re-sent, a reply regenerated, an agent re-copying code or a tool
//! call it wrote in an earlier session, boilerplate copying itself: the PLD
//! n-gram lookup only sees the CURRENT request's context, so all of that
//! drafts token by token. Recall keeps the token ids on disk, keyed by the
//! tokenizer that wrote them, and offers the continuation that followed the
//! same context earlier — the target model still verifies every copied
//! token, so replies are byte-identical with Recall on, off or read-only.
//!
//! What is stored, plainly: EVERY prompt and reply the server reads and
//! writes, as token ids. Ids are not encryption — the model's tokenizer
//! turns them back into their exact text — so treat the record directory as
//! you would those conversations. And a hit decodes measurably faster than a
//! miss, so with an API key active the record is loopback-only unless
//! --recall-shared opts keyed and LAN clients into sharing it: otherwise any
//! client could probe, by timing, whether some text was seen before.
//!
//! Ported from TensorFold (Apache-2.0), files `zig/src/core/lanes/recall.zig`,
//! `recall_record.zig` and `recall_store.zig` — record, pricing bands, dedup,
//! identity/lock/eviction semantics preserved; adapted here for a MULTI-model
//! server (one store per tokenizer identity, opened lazily — TensorFold serves
//! one model and refuses at startup) and for the host/engine unit split (the
//! registry is `Shared` state, `sdk/util/shared.zig`). Differences from the
//! origin, all deliberate:
//!   - `readPrompt` runs the dedup walk with resumed=0 for now (the prefix-cache
//!     resume point is not plumbed to the Generator yet); the held-run check
//!     makes a re-sent prefix add nothing, at the cost of a few thousand hash
//!     probes per turn.
//!   - recording happens at Generator.deinit for every spec mode, not just PLD.
//!   - proposals feed the PLD drafter's miss path (grafting into draft trees is
//!     a later phase; TensorFold documents the same boundary: "No served engine
//!     turns grafted lanes on yet").
//!   - an unopenable record (bad dir, identity mismatch) logs once and disables
//!     Recall for that tokenizer instead of refusing the server: the flag is a
//!     performance accelerant and must not take serving down with it.
//!
//! Wire format (mirrors TensorFold's, s/TFRECALL/MSRECALL/): chunk files
//! `%016x.ids` under a 0700 directory named by the tokenizer's identity
//! digest, each `32-byte header + 4*ids`, header = magic, version, vocab size,
//! first 16 bytes of the identity digest. An `identity` file names the
//! tokenizer and the model that first wrote the record; a different tokenizer
//! is refused by name. A `lock` file (flock) makes a second server on the same
//! directory read-only. Past the byte cap the OLDEST chunk file goes first.
//! Writes stay above a 4 GiB free-disk floor, and any write failure stops
//! adding for the rest of the run (serving continues).

const std = @import("std");
const builtin = @import("builtin");
const log = @import("../util/log.zig");
const shared_util = @import("../util/shared.zig");
const os_mutex = @import("../util/os_mutex.zig");
const io_util = @import("../util/io_util.zig");
const tokenizer_mod = @import("../tokenizer/tokenizer.zig");
const Tokenizer = tokenizer_mod.Tokenizer;

const Allocator = std.mem.Allocator;

/// The long-lived allocator for stores and the registry (freed only at exit).
const gpa = std.heap.c_allocator;

extern "c" fn flock(fd: c_int, operation: c_int) c_int;
const LOCK_EX: c_int = 2;
const LOCK_NB: c_int = 4;

// ────────────────────────── the record in RAM ──────────────────────────
// (ported from TensorFold recall_record.zig)

/// Ends a segment: no key spans it and no copy crosses it.
pub const mark: u32 = std.math.maxInt(u32);
/// Tokens a key covers: the shortest match the record finds.
pub const key_len: u32 = 8;
/// Ids a chunk holds (one chunk file's).
pub const chunk_tokens: u32 = 1 << 20;
/// Candidates a search checks at most, newest first: a lookup's time is bounded by it.
pub const max_steps: u32 = 64;
/// Tokens a match is extended back at most.
pub const max_match: u32 = 64;
/// Positions a record numbers before it stops taking ids (a reload numbers them from 0 again).
const limit: u64 = std.math.maxInt(u32) - 4 * @as(u64, chunk_tokens);

const Chunk = struct {
    ids: []u32,
    back: []u32, // each position's distance back to the previous position in its key's bucket (0: none)
};

/// A context's longest match: its length, and the position of the id that followed it.
pub const Match = struct { len: u32 = 0, at: u32 = 0 };

/// A run of ids the record holds: its length and its first position.
pub const Run = struct { len: usize = 0, at: u32 = 0 };

pub const Record = struct {
    a: Allocator,
    chunks: std.ArrayList(Chunk) = .empty,
    first: u32 = 0, // the oldest held chunk's number: positions start at first * chunk_tokens
    end: u32 = 0, // one past the newest position
    heads: []u32, // each bucket's newest position plus one (0: none)
    last: [key_len]u32 = @splat(mark), // the newest ids, the next key's
    run: u32 = 0, // ids since the last mark

    /// An empty record whose key table has 2^bucket_bits buckets (fixed: growth never rebuilds the index).
    pub fn init(a: Allocator, bucket_bits: u5) !Record {
        const heads = try a.alloc(u32, @as(usize, 1) << bucket_bits);
        @memset(heads, 0);
        return .{ .a = a, .heads = heads };
    }

    pub fn deinit(r: *Record) void {
        for (r.chunks.items) |c| r.free(c);
        r.chunks.deinit(r.a);
        r.a.free(r.heads);
    }

    fn free(r: *Record, c: Chunk) void {
        r.a.free(c.ids);
        r.a.free(c.back);
    }

    /// The oldest held position.
    pub fn base(r: *const Record) u32 {
        return r.first * chunk_tokens;
    }

    /// Ids held, marks included.
    pub fn held(r: *const Record) u32 {
        return r.end - r.base();
    }

    /// Bytes the record takes in RAM: its chunks' ids and links, and the key table.
    pub fn ramBytes(r: *const Record) u64 {
        return @as(u64, r.chunks.items.len) * chunk_tokens * 8 + @as(u64, r.heads.len) * 4;
    }

    pub fn id(r: *const Record, p: u32) u32 {
        return r.chunks.items[p / chunk_tokens - r.first].ids[p % chunk_tokens];
    }

    fn backOf(r: *const Record, p: u32) u32 {
        return r.chunks.items[p / chunk_tokens - r.first].back[p % chunk_tokens];
    }

    fn bucket(r: *const Record, key: *const [key_len]u32) usize {
        return @as(usize, @truncate(std.hash.Wyhash.hash(0x5eca11, std.mem.asBytes(key)))) & (r.heads.len - 1);
    }

    /// Whether `n` more ids fit before the positions run out.
    pub fn room(r: *const Record, n: usize) bool {
        return @as(u64, r.end) + n + 1 <= limit;
    }

    /// `ids` then a mark, each new position indexed; the caller checks `room` first.
    pub fn add(r: *Record, ids: []const u32) !void {
        for (ids) |t| try r.push(t);
        try r.push(mark);
    }

    /// One id at the end; a full newest chunk takes a new one first.
    pub fn push(r: *Record, t: u32) !void {
        const p = r.end;
        if (@as(usize, p) == @as(usize, r.base()) + r.chunks.items.len * chunk_tokens) {
            const ids = try r.a.alloc(u32, chunk_tokens);
            errdefer r.a.free(ids);
            const back = try r.a.alloc(u32, chunk_tokens);
            errdefer r.a.free(back);
            try r.chunks.append(r.a, .{ .ids = ids, .back = back });
        }
        const c = r.chunks.items[r.chunks.items.len - 1];
        const at = p % chunk_tokens;
        c.ids[at] = t;
        c.back[at] = 0;
        r.end = p + 1;
        if (t == mark) {
            r.run = 0;
            return;
        }
        std.mem.copyForwards(u32, r.last[0 .. key_len - 1], r.last[1..]);
        r.last[key_len - 1] = t;
        r.run += 1;
        if (r.run < key_len) return;
        const b = r.bucket(&r.last);
        if (r.heads[b] != 0) c.back[at] = p - (r.heads[b] - 1);
        r.heads[b] = p + 1;
    }

    /// The oldest chunk freed (never the newest): links into it end the searches that reach them.
    pub fn dropOldest(r: *Record) bool {
        if (r.chunks.items.len < 2) return false;
        r.free(r.chunks.orderedRemove(0));
        r.first += 1;
        return true;
    }

    /// Every id gone and the key table zeroed: positions start at 0 again.
    pub fn clear(r: *Record) void {
        for (r.chunks.items) |c| r.free(c);
        r.chunks.clearRetainingCapacity();
        @memset(r.heads, 0);
        r.first = 0;
        r.end = 0;
        r.run = 0;
        r.last = @splat(mark);
    }

    /// Whether the key ending at `p` holds `key`.
    fn keyAt(r: *const Record, p: u32, key: []const u32) bool {
        const from = p + 1 - key_len;
        for (key, 0..) |t, i| if (r.id(from + @as(u32, @intCast(i))) != t) return false;
        return true;
    }

    /// Positions whose keys may hold `key`, newest first, at most max_steps, each held whole.
    const Walk = struct {
        r: *const Record,
        head: u32,
        steps: u32 = 0,

        fn next(w: *Walk) ?u32 {
            if (w.steps == max_steps or w.head < w.r.base() + key_len) return null;
            w.steps += 1;
            const p = w.head - 1;
            const d = w.r.backOf(p);
            w.head = if (d == 0) 0 else p - d + 1;
            return p;
        }
    };

    fn walk(r: *const Record, key: *const [key_len]u32) Walk {
        return .{ .r = r, .head = r.heads[r.bucket(key)] };
    }

    /// How far back from the key ending at `p` the record matches the context's tail (key_len to max_match).
    fn extend(r: *const Record, p: u32, ctx: []const u32) u32 {
        var n: u32 = key_len;
        while (n < max_match and n < ctx.len and p >= r.base() + n) : (n += 1) {
            const t = r.id(p - n);
            if (t == mark or t != ctx[ctx.len - 1 - n]) break;
        }
        return n;
    }

    /// The longest match of `ctx`'s last ids with an id after it, the newest of equal length.
    pub fn longest(r: *const Record, ctx: []const u32) Match {
        if (ctx.len < key_len) return .{};
        const key = ctx[ctx.len - key_len ..][0..key_len];
        var best: Match = .{};
        var w = r.walk(key);
        while (w.next()) |p| {
            if (!r.keyAt(p, key) or p + 1 >= r.end or r.id(p + 1) == mark) continue;
            const n = r.extend(p, ctx);
            if (n <= best.len) continue;
            best = .{ .len = n, .at = p + 1 };
            if (n == max_match) break;
        }
        return best;
    }

    /// The ids from `at` up to a mark, the end or `out.len`, copied into `out`.
    pub fn copy(r: *const Record, at: u32, out: []u32) []u32 {
        var n: usize = 0;
        while (n < out.len and @as(u64, at) + n < r.end) : (n += 1) {
            const t = r.id(at + @as(u32, @intCast(n)));
            if (t == mark) break;
            out[n] = t;
        }
        return out[0..n];
    }

    /// The longest prefix of `s` the record holds in one run, and the position where it starts (len 0: none).
    pub fn heldRun(r: *const Record, s: []const u32) Run {
        if (s.len < key_len) return .{};
        const key = s[0..key_len];
        var best: Run = .{};
        var w = r.walk(key);
        while (w.next()) |p| {
            if (!r.keyAt(p, key)) continue;
            var n: usize = key_len;
            while (n < s.len) : (n += 1) {
                const q = @as(u64, p) + 1 + n - key_len;
                if (q >= r.end) break;
                const t = r.id(@intCast(q));
                if (t == mark or t != s[n]) break;
            }
            if (n <= best.len) continue;
            best = .{ .len = n, .at = p + 1 - key_len };
            if (n == s.len) break;
        }
        return best;
    }
};

// ────────────────────────── pricing ──────────────────────────
// (ported from TensorFold recall.zig)

/// Match lengths that open each band of landing rates.
pub const bands = [_]u32{ 16, 32, 64 };
/// The longest wait between probes of a band whose copies lose.
const probe_most: u32 = 1024;

/// Every stream's Recall copies by band: what landed, so a cold or noisy record stops proposing by itself.
pub const Rates = struct {
    landed: [bands.len]f64 = .{ 0.75, 2, 4.5 }, // priors: 0.6, 0.8 and 0.9 a copied id, worth half a miss each
    missed: [bands.len]f64 = .{ 0.5, 0.5, 0.5 },
    probe_every: u32 = 32, // unpriced lookups between probes
    since: u32 = 0,
    proposals: u64 = 0,
    proposed: u64 = 0,
    accepted: u64 = 0,

    pub fn bandOf(len: u32) usize {
        var b: usize = 0;
        while (b + 1 < bands.len and len >= bands[b + 1]) b += 1;
        return b;
    }

    /// A band's chance that a copied id lands when the ids before it did.
    pub fn rate(r: *const Rates, b: usize) f64 {
        return r.landed[b] / (r.landed[b] + r.missed[b]);
    }

    /// A copy of `proposed` ids landed `accepted`; a probe that landed half or more ends the backoff, else doubles it.
    pub fn observe(r: *Rates, b: usize, proposed: u64, accepted: u64, probing: bool) void {
        r.landed[b] = 0.95 * r.landed[b] + @as(f64, @floatFromInt(accepted));
        r.missed[b] = 0.95 * r.missed[b] + @as(f64, @floatFromInt(@intFromBool(accepted < proposed)));
        r.proposals += 1;
        r.proposed += proposed;
        r.accepted += accepted;
        if (probing) r.probe_every = if (2 * accepted >= proposed) 32 else @min(probe_most, 2 * r.probe_every);
    }

    /// Whether some band's copy of `n` ids is expected to land more than `head` tokens a round.
    pub fn pays(r: *Rates, n: usize, head: f64) bool {
        for (0..bands.len) |b| if (expected(r.rate(b), n) > head) return true;
        return false;
    }

    /// Whether this ask, priced out, is the next probe.
    fn probe(r: *Rates) bool {
        r.since += 1;
        if (r.since < r.probe_every) return false;
        r.since = 0;
        return true;
    }
};

/// Tokens a round with a copy of `n` ids is expected to land at `q` an id, its own token included.
pub fn expected(q: f64, n: usize) f64 {
    var total: f64 = 1;
    var reach: f64 = 1;
    for (0..n) |_| {
        reach *= q;
        total += reach;
    }
    return total;
}

pub const Params = struct {
    min_match: u32 = bands[0], // matched ids a Recall copy needs
    quiet: u64 = 4, // rounds a stream leaves Recall after a copy that landed nothing
};

/// A stream's reads of the record: copies priced against its own rounds.
pub const Reader = struct {
    st: *Store,
    params: Params = .{},
    out: std.ArrayList(u32) = .empty,
    head: f64 = 2, // tokens the stream's own rounds landed lately: the bar a copy's expected tokens must pass
    band: usize = 0, // the last copy's band
    probing: bool = false, // the last copy was a probe
    quiet_until: u64 = 0,
    memo: ?u64 = null, // the last ask's context and width, hashed: the same ask again in its round gets the same answer
    /// The winning match's length behind the last non-empty `propose` (0 after
    /// any empty return): the DFlash copy round prices and observes by it.
    last_match: u32 = 0,
    lookups: u64 = 0,
    proposals: u64 = 0,
    proposed: u64 = 0,
    accepted: u64 = 0,

    pub fn deinit(x: *Reader) void {
        x.out.deinit(gpa);
    }

    /// The record's continuation of `ctx` (owned until the next call), or empty when it is not expected to pay.
    pub fn propose(x: *Reader, ctx: []const u32, most: usize, round: u64) []const u32 {
        if (ctx.len == 0 or most < 2 or round < x.quiet_until) return &.{};
        const key = memoKey(ctx, most);
        if (x.memo) |m| if (m == key) return x.out.items;
        x.out.clearRetainingCapacity();
        x.out.ensureTotalCapacity(gpa, most) catch return &.{};
        x.memo = key;
        x.st.mutex.lock();
        defer x.st.mutex.unlock();
        const rates = &x.st.rates;
        var probing = false; // the price comes first: the record is walked only when a copy can pay, or for a probe
        if (!rates.pays(most, x.head)) {
            probing = rates.probe();
            if (!probing) return &.{};
        }
        x.lookups += 1;
        const m = x.st.record.longest(ctx);
        if (m.len < x.params.min_match) return &.{};
        const got = x.st.record.copy(m.at, x.out.allocatedSlice()[0..most]);
        if (got.len < 2) return &.{};
        const b = Rates.bandOf(m.len);
        if (!probing and expected(rates.rate(b), got.len) <= x.head) {
            probing = rates.probe();
            if (!probing) return &.{};
        }
        x.out.items.len = got.len;
        x.band = b;
        x.probing = probing;
        x.last_match = m.len;
        x.proposals += 1;
        x.proposed += got.len;
        return x.out.items;
    }

    /// What a lookup reads of `ctx` (its last max_match ids) and the width asked, hashed.
    fn memoKey(ctx: []const u32, most: usize) u64 {
        return std.hash.Wyhash.hash(most, std.mem.sliceAsBytes(ctx[ctx.len -| max_match..]));
    }

    /// The record's own landing rate for the band the last proposal matched
    /// in: the DFlash record round prices its rows from this, never the
    /// in-context copy ladder.
    pub fn landingRate(x: *const Reader) f64 {
        x.st.mutex.lock();
        defer x.st.mutex.unlock();
        return x.st.rates.rate(x.band);
    }

    /// The last copy landed `accepted` of its `proposed` ids; none landed keeps the stream off a few rounds.
    pub fn observe(x: *Reader, proposed: i64, accepted: i64, round: u64) void {
        x.memo = null;
        if (proposed <= 0) return;
        const p: u64 = @intCast(proposed);
        const a: u64 = @intCast(std.math.clamp(accepted, 0, proposed));
        x.accepted += a;
        if (a == 0) x.quiet_until = round + 1 + x.params.quiet;
        x.st.mutex.lock();
        defer x.st.mutex.unlock();
        x.st.rates.observe(x.band, p, a, x.probing);
    }

    /// A round of the stream's own drafts (a head, a lookup or none) landed `tokens`.
    pub fn noteHead(x: *Reader, tokens: usize) void {
        x.memo = null;
        x.head += 0.25 * (@as(f64, @floatFromInt(tokens)) - x.head);
    }
};

// ────────────────────────── the record on disk ──────────────────────────
// (ported from TensorFold recall_store.zig, std.c fd idiom)

const magic = "MSRECALL";
const version: u32 = 1;
/// A chunk file's header: magic, version, the vocabulary's size and the identity's first 16 bytes.
const head_bytes = 32;
/// A full chunk file's bytes: what the cap counts.
pub const chunk_bytes: u64 = head_bytes + 4 * @as(u64, chunk_tokens);
/// Writes stay above this much free disk on the record's volume, or they pause (serving continues).
const disk_floor: u64 = 4 << 30;

/// What a record's ids mean: the tokenizer's identity and vocabulary size, and the model whose server first wrote it.
pub const Identity = struct {
    digest: [32]u8,
    vocab: u32,
    name: []const u8 = "",

    /// The digest's first 8 bytes in hex: the store directory's name.
    pub fn short(i: Identity) [16]u8 {
        return std.fmt.bytesToHex(i.digest[0..8].*, .lower);
    }
};

pub const Options = struct {
    dir: []const u8,
    cap_bytes: u64,
    read_only: bool = false,
};

/// The store's state for GET /v1/recall and the startup line.
pub const Stats = struct {
    writes: bool,
    left: bool, // the last empty left a chunk file: nothing is added until one succeeds
    read_only: bool,
    ids: u64,
    disk_bytes: u64,
    ram_bytes: u64,
    cap_bytes: u64,
    added: u64,
    held: u64,
    rates: [bands.len]f64,
    probe_every: u32,
    proposals: u64,
    proposed: u64,
    accepted: u64,
    problem: []const u8,
    model: []const u8,
};

pub const Store = struct {
    io: std.Io,
    dir: [:0]u8,
    name: []u8,
    identity: Identity,
    cap_chunks: usize,
    read_only: bool, // asked to read only: never writes, never takes the lock
    writes: bool, // adds what its requests read and write (false: read only, or another server holds the lock)
    mutex: os_mutex.Mutex = .{},
    record: Record,
    rates: Rates = .{},
    lock_fd: c_int = -1,
    file: c_int = -1, // the newest chunk's file, open to append (-1: none)
    file_chunk: u32 = 0, // the record chunk that file holds
    seq0: u64 = 0, // the file number of record chunk 0
    failed: bool = false, // a write failed: this server adds nothing more
    added: u64 = 0, // ids this server added
    held: u64 = 0, // ids its requests read or wrote that the record held already
    problem_buf: [1200]u8 = undefined,
    problem: []const u8 = "", // why the last empty failed

    /// Ids written before and after each new run: keys across its joins are found, matches there reach max_match.
    pub const overlap: usize = max_match;
    /// A held run shorter than this is written again: a join costs two overlaps.
    const min_run: usize = 2 * overlap;
    /// Ids between the starts the dedup checks: runs of min_run ids or more are never missed.
    const stride: usize = 16;

    /// Everything `open` allocated, for the paths that hand it back null.
    fn freeOpen(s: *Store) void {
        s.record.deinit();
        gpa.free(s.name);
        gpa.free(s.dir);
        gpa.destroy(s);
    }

    /// The record in `o.dir` (made 0700), or null with `problem` set when another tokenizer wrote it.
    pub fn open(identity: Identity, io: std.Io, o: Options, problem: *[]const u8, a: Allocator) !?*Store {
        const s = try gpa.create(Store);
        const dir = try gpa.dupeSentinel(u8, o.dir, 0);
        const name = try gpa.dupe(u8, identity.name);
        const cap_chunks: usize = @max(2, o.cap_bytes / chunk_bytes);
        s.* = .{
            .io = io,
            .dir = dir,
            .name = name,
            .identity = .{ .digest = identity.digest, .vocab = identity.vocab, .name = name },
            .cap_chunks = cap_chunks,
            .read_only = o.read_only,
            .writes = !o.read_only,
            .record = try Record.init(gpa, bucketBits(cap_chunks)),
        };
        makePath(dir) catch {
            problem.* = std.fmt.allocPrint(a, "--recall cannot make its directory {s}", .{o.dir}) catch "--recall: directory error";
            s.freeOpen();
            return null;
        };
        if (!try s.checkIdentity(a, problem)) {
            s.freeOpen();
            return null;
        }
        if (s.writes) s.takeLock();
        errdefer {
            if (s.lock_fd >= 0) _ = std.c.close(s.lock_fd);
            s.freeOpen();
        }
        try s.load();
        return s;
    }

    pub fn close(s: *Store) void {
        s.closeFile();
        if (s.lock_fd >= 0) _ = std.c.close(s.lock_fd);
        s.record.deinit();
        gpa.free(s.name);
        gpa.free(s.dir);
        gpa.destroy(s);
    }

    fn closeFile(s: *Store) void {
        if (s.file < 0) return;
        _ = std.c.fsync(s.file);
        _ = std.c.close(s.file);
        s.file = -1;
    }

    /// The key table's size for a cap: about one bucket for every 8 ids, fixed so growth never rebuilds the index.
    fn bucketBits(chunks: usize) u5 {
        var bits: u5 = 16;
        while (bits < 28 and (@as(usize, 1) << bits) < chunks * (chunk_tokens / 8)) bits += 1;
        return bits;
    }

    /// The identity file names the tokenizer that wrote the record: another one is refused by its name.
    fn checkIdentity(s: *Store, a: Allocator, problem: *[]const u8) !bool {
        var buf: [1100]u8 = undefined;
        const path = std.fmt.bufPrintSentinel(&buf, "{s}/identity", .{s.dir}, 0) catch return error.RecallDir;
        const want = std.fmt.allocPrint(a, "mlx-serve recall {d}\ndigest {x}\nvocab {d}\nname ", .{ version, &s.identity.digest, s.identity.vocab }) catch return error.RecallWrite;
        defer a.free(want);
        const got = readSmall(a, path) catch |e| {
            if (e == error.FileNotFound) {
                if (s.writes) {
                    const full = std.mem.concat(a, u8, &.{ want, s.identity.name, "\n" }) catch return error.RecallWrite;
                    defer a.free(full);
                    writeSmall(path, full) catch {
                        problem.* = std.fmt.allocPrint(a, "--recall cannot write {s}", .{path}) catch "--recall: identity write error";
                        return false;
                    };
                }
                return true;
            }
            problem.* = std.fmt.allocPrint(a, "--recall cannot read {s}", .{path}) catch "--recall: identity read error";
            return false;
        };
        defer a.free(got);
        if (std.mem.startsWith(u8, got, want)) return true;
        var lines = std.mem.splitScalar(u8, got, '\n');
        var digest: []const u8 = "?";
        var vocab: []const u8 = "?";
        var name: []const u8 = "an unnamed model";
        while (lines.next()) |l| {
            if (std.mem.startsWith(u8, l, "digest ")) digest = l[7..@min(l.len, 7 + 16)];
            if (std.mem.startsWith(u8, l, "vocab ")) vocab = l[6..];
            if (std.mem.startsWith(u8, l, "name ") and l.len > 5) name = l[5..];
        }
        const mine = s.identity.short();
        const refusal = "--recall: {s} holds a record written under the tokenizer of {s} ({s} ids, identity {s}); " ++
            "{s}'s tokenizer is another ({d} ids, identity {s}): name another --recall-dir";
        const args = .{ s.dir, name, vocab, digest, s.identity.name, s.identity.vocab, &mine };
        problem.* = std.fmt.allocPrint(a, refusal, args) catch "--recall: tokenizer mismatch";
        return false;
    }

    /// The directory's lock: a second server with the same record reads it as it was and writes nothing.
    fn takeLock(s: *Store) void {
        var buf: [1100]u8 = undefined;
        const path = std.fmt.bufPrintSentinel(&buf, "{s}/lock", .{s.dir}, 0) catch return s.readOnly();
        const fd = std.c.open(path, .{ .ACCMODE = .RDWR, .CREAT = true, .NOFOLLOW = true }, @as(std.c.mode_t, 0o600));
        if (fd < 0) return s.readOnly();
        if (flock(fd, LOCK_EX | LOCK_NB) != 0) {
            _ = std.c.close(fd);
            return s.readOnly();
        }
        s.lock_fd = fd;
    }

    fn readOnly(s: *Store) void {
        s.writes = false;
    }

    /// Chunk files in order from the first: each full but the last, ids read up to the last mark; the rest cut away.
    fn load(s: *Store) !void {
        var seqs: std.ArrayList(u64) = .empty;
        defer seqs.deinit(gpa);
        try s.listChunks(&seqs);
        if (seqs.items.len == 0) return;
        s.seq0 = seqs.items[0];
        var pending: std.ArrayList(u32) = .empty;
        defer pending.deinit(gpa);
        const ids = try gpa.alloc(u32, 1 << 16);
        defer gpa.free(ids);
        for (seqs.items, 0..) |seq, k| {
            if (seq != s.seq0 + k) break; // a gap: the files past it go
            const n = s.readChunk(seq, &pending, ids) catch break;
            if (n < chunk_tokens) break; // a short file is the last
        }
        if (s.writes) s.cut(seqs.items);
        while (s.record.chunks.items.len > s.cap_chunks) _ = s.drop();
    }

    /// One chunk file's ids into the record a segment at a time; its count of ids (error: a bad header or id).
    fn readChunk(s: *Store, seq: u64, pending: *std.ArrayList(u32), ids: []u32) !usize {
        var buf: [1100]u8 = undefined;
        const path = try s.chunkPath(&buf, seq);
        const fd = std.c.open(path, .{ .ACCMODE = .RDONLY, .NOFOLLOW = true }, @as(std.c.mode_t, 0));
        if (fd < 0) return error.RecallRead;
        defer _ = std.c.close(fd);
        const size = std.c.lseek(fd, 0, std.c.SEEK.END);
        if (size < head_bytes) return error.RecallRead;
        var head: [head_bytes]u8 = undefined;
        if (!readAt(fd, &head, 0) or !std.mem.eql(u8, &head, &s.header())) return error.RecallRead;
        const count: usize = @intCast(@divTrunc(size - head_bytes, 4));
        if (count > chunk_tokens) return error.RecallRead;
        var at: usize = 0;
        const r = &s.record;
        while (at < count) {
            const n = @min(ids.len, count - at);
            if (!readAt(fd, std.mem.sliceAsBytes(ids[0..n]), head_bytes + 4 * at)) return error.RecallRead;
            for (ids[0..n]) |t| {
                if (t == mark) {
                    if (!r.room(pending.items.len)) return error.RecallRead;
                    try r.add(pending.items);
                    pending.clearRetainingCapacity();
                } else if (t < s.identity.vocab) try pending.append(gpa, t) else return error.RecallRead;
            }
            at += n;
        }
        return count;
    }

    /// The files cut to what the record read: the last one's torn tail truncated, the files after it deleted.
    fn cut(s: *Store, seqs: []const u64) void {
        const r = &s.record;
        const chunks: usize = (@as(usize, r.end) + chunk_tokens - 1) / chunk_tokens;
        for (seqs, 0..) |seq, k| {
            var buf: [1100]u8 = undefined;
            const path = s.chunkPath(&buf, seq) catch continue;
            if (k >= chunks or seq != s.seq0 + k) {
                _ = std.c.unlink(path);
            } else if (k + 1 == chunks) {
                const keep = head_bytes + 4 * (@as(u64, r.end) - @as(u64, k) * chunk_tokens);
                const fd = std.c.open(path, .{ .ACCMODE = .WRONLY, .NOFOLLOW = true }, @as(std.c.mode_t, 0));
                if (fd >= 0) {
                    _ = std.c.ftruncate(fd, @intCast(keep));
                    _ = std.c.close(fd);
                }
            }
        }
    }

    fn chunkPath(s: *const Store, buf: []u8, seq: u64) ![:0]const u8 {
        return std.fmt.bufPrintSentinel(buf, "{s}/{x:0>16}.ids", .{ s.dir, seq }, 0);
    }

    /// The chunk files' numbers, ascending.
    fn listChunks(s: *Store, out: *std.ArrayList(u64)) !void {
        var dir = std.Io.Dir.openDirAbsolute(s.io, s.dir, .{ .iterate = true }) catch return error.RecallRead;
        defer dir.close(s.io);
        var it = dir.iterate();
        while (it.next(s.io) catch return error.RecallRead) |ent| {
            if (ent.name.len != 20 or !std.mem.endsWith(u8, ent.name, ".ids")) continue;
            const seq = std.fmt.parseInt(u64, ent.name[0..16], 16) catch continue;
            try out.append(gpa, seq);
        }
        std.mem.sort(u64, out.items, {}, std.sort.asc(u64));
    }

    /// The oldest chunk freed and its file deleted.
    fn drop(s: *Store) bool {
        const r = &s.record;
        if (!r.dropOldest()) return false;
        if (s.writes) {
            var buf: [1100]u8 = undefined;
            if (s.chunkPath(&buf, s.seq0 + r.first - 1)) |path| {
                _ = std.c.unlink(path);
            } else |_| {}
        }
        return true;
    }

    /// A prompt read from its resume point: the ids an earlier request's state did not cover, less the overlap.
    /// `resumed` is the prefix the hot cache already covered (0 when unknown):
    /// the held-run dedup below still keeps a re-sent prefix from being written again.
    pub fn readPrompt(s: *Store, prompt: []const u32, resumed: u32) void {
        s.mutex.lock();
        defer s.mutex.unlock();
        s.add(prompt[@min(prompt.len, resumed) -| overlap ..]);
    }

    /// A finished reply, after its prompt's last `overlap` ids so the keys across the join are found.
    pub fn readReply(s: *Store, prompt: []const u32, reply: []const u32) void {
        if (reply.len == 0) return;
        const seg = std.mem.concat(gpa, u32, &.{ prompt[prompt.len -| overlap ..], reply }) catch return;
        defer gpa.free(seg);
        s.mutex.lock();
        defer s.mutex.unlock();
        s.add(seg);
    }

    /// The runs of `seg` the record lacks, each with its overlaps (the caller holds the lock).
    fn add(s: *Store, seg: []const u32) void {
        if (!s.writes or s.failed) return;
        const r = &s.record;
        var fresh: usize = 0; // where the ids not yet written start
        var i: usize = 0;
        while (i + key_len <= seg.len) {
            const run = r.heldRun(seg[i..]);
            if (run.len >= stride) {
                var j = i; // the run reaches back past i, to the last id decided
                var q = run.at;
                while (j > fresh and q > r.base() and r.id(q - 1) == seg[j - 1]) : ({
                    j -= 1;
                    q -= 1;
                }) {}
                if (i + run.len - j >= min_run) {
                    s.put(seg, fresh, j);
                    s.held += i + run.len - j;
                    fresh = i + run.len;
                    i = fresh;
                    continue;
                }
            }
            i += stride;
        }
        s.put(seg, fresh, seg.len);
    }

    /// The ids `seg[from..to]` with their overlaps, added and written behind the free-disk floor.
    fn put(s: *Store, seg: []const u32, from: usize, to: usize) void {
        if (to <= from) return;
        const piece = seg[from -| overlap .. @min(seg.len, to + overlap)];
        const r = &s.record;
        if (piece.len < key_len or !r.room(piece.len)) return;
        for (piece) |t| if (t >= s.identity.vocab) return; // a load stops at such an id: it never goes in
        const bytes = 4 * (@as(u64, piece.len) + 1) + head_bytes;
        if (!diskHasRoom(s.dir, bytes)) return;
        const start = r.end;
        r.add(piece) catch {
            s.failed = true;
            return;
        };
        const ok = s.write(start, r.end);
        if (!ok) s.failed = true;
        s.added += piece.len;
        while (r.chunks.items.len > s.cap_chunks) _ = s.drop();
    }

    /// The record's positions [start, end) appended to their chunk files.
    fn write(s: *Store, start: u32, end: u32) bool {
        const r = &s.record;
        var ids: [4096]u32 = undefined;
        var p = start;
        while (p < end) {
            const k = p / chunk_tokens;
            if (!s.fileFor(k)) return false;
            const stop = @min(end, (k + 1) * chunk_tokens);
            while (p < stop) {
                const n = @min(stop - p, ids.len);
                for (ids[0..n], 0..) |*t, i| t.* = r.id(p + @as(u32, @intCast(i)));
                writeAll(s.file, std.mem.sliceAsBytes(ids[0..n])) catch return false;
                p += n;
            }
        }
        return true;
    }

    /// The file for record chunk `k`, open to append: the one on disk, or a new one with its header.
    fn fileFor(s: *Store, k: u32) bool {
        if (s.file >= 0 and s.file_chunk == k) return true;
        s.closeFile();
        var buf: [1100]u8 = undefined;
        const path = s.chunkPath(&buf, s.seq0 + k) catch return false;
        var flags: std.c.O = .{ .ACCMODE = .WRONLY, .APPEND = true, .NOFOLLOW = true };
        var fd = std.c.open(path, flags, @as(std.c.mode_t, 0));
        if (fd < 0) {
            flags.CREAT = true;
            flags.EXCL = true;
            fd = std.c.open(path, flags, @as(std.c.mode_t, 0o600));
            if (fd < 0) return false;
            const head = s.header();
            writeAll(fd, &head) catch {
                _ = std.c.close(fd);
                return false;
            };
        }
        s.file = fd;
        s.file_chunk = k;
        return true;
    }

    /// A chunk file's header for this record: magic, version, vocabulary size, the identity's first 16 bytes.
    fn header(s: *const Store) [head_bytes]u8 {
        var head: [head_bytes]u8 = undefined;
        @memcpy(head[0..8], magic);
        std.mem.writeInt(u32, head[8..12], version, .little);
        std.mem.writeInt(u32, head[12..16], s.identity.vocab, .little);
        @memcpy(head[16..32], s.identity.digest[0..16]);
        return head;
    }

    /// Chunk files deleted; record, rates and counters emptied (identity, lock stay). Unlinked: `why` names one left.
    pub fn empty(s: *Store) error{ ReadOnly, Unlinked }!void {
        s.mutex.lock();
        defer s.mutex.unlock();
        if (!s.writes) return error.ReadOnly;
        s.closeFile();
        const r = &s.record;
        const next = s.seq0 + (@as(u64, r.end) + chunk_tokens) / chunk_tokens;
        var seqs: std.ArrayList(u64) = .empty;
        defer seqs.deinit(gpa);
        s.problem = "";
        s.listChunks(&seqs) catch s.note("{s} could not be listed", .{s.dir});
        for (seqs.items) |seq| {
            var buf: [1100]u8 = undefined;
            const path = s.chunkPath(&buf, seq) catch continue;
            const rc = std.c.unlink(path);
            if (rc != 0 and std.c._errno().* != @intFromEnum(std.c.E.NOENT) and s.problem.len == 0) {
                s.note("{s} could not be removed", .{path});
            }
        }
        r.clear();
        s.rates = .{};
        s.added = 0;
        s.held = 0;
        s.seq0 = @max(next, if (seqs.items.len > 0) seqs.items[seqs.items.len - 1] + 1 else 0);
        s.failed = s.problem.len > 0;
        if (s.failed) return error.Unlinked;
    }

    /// Why the last empty failed, for its reply.
    pub fn why(s: *const Store) []const u8 {
        return s.problem;
    }

    fn note(s: *Store, comptime fmt: []const u8, args: anytype) void {
        s.problem = std.fmt.bufPrint(&s.problem_buf, fmt, args) catch "a chunk file could not be removed";
    }

    pub fn stats(s: *Store) Stats {
        s.mutex.lock();
        defer s.mutex.unlock();
        const r = &s.record;
        var out: Stats = .{
            .writes = s.writes and !s.failed,
            .left = s.problem.len > 0,
            .read_only = s.read_only,
            .ids = r.held(),
            .disk_bytes = @as(u64, r.chunks.items.len) * head_bytes + 4 * @as(u64, r.held()),
            .ram_bytes = r.ramBytes(),
            .cap_bytes = @as(u64, s.cap_chunks) * chunk_bytes,
            .added = s.added,
            .held = s.held,
            .rates = undefined,
            .probe_every = s.rates.probe_every,
            .proposals = s.rates.proposals,
            .proposed = s.rates.proposed,
            .accepted = s.rates.accepted,
            .problem = s.problem,
            .model = s.name,
        };
        for (&out.rates, 0..) |*x, b| x.* = s.rates.rate(b);
        return out;
    }
};

/// `dir` and its missing parents, each made 0700.
fn makePath(dir: [:0]const u8) !void {
    var buf: [1100]u8 = undefined;
    if (dir.len >= buf.len) return error.RecallDir;
    @memcpy(buf[0..dir.len], dir);
    buf[dir.len] = 0;
    for (1..dir.len) |i| if (buf[i] == '/') {
        buf[i] = 0;
        _ = std.c.mkdir(@ptrCast(&buf), 0o700);
        buf[i] = '/';
    };
    const rc = std.c.mkdir(dir, 0o700);
    if (rc != 0 and std.c._errno().* != @intFromEnum(std.c.E.EXIST)) return error.RecallDir;
}

fn readSmall(a: Allocator, path: [:0]const u8) ![]u8 {
    const fd = std.c.open(path, .{ .ACCMODE = .RDONLY, .NOFOLLOW = true }, @as(std.c.mode_t, 0));
    if (fd < 0) return if (std.c._errno().* == @intFromEnum(std.c.E.NOENT)) error.FileNotFound else error.RecallRead;
    defer _ = std.c.close(fd);
    const size = std.c.lseek(fd, 0, std.c.SEEK.END);
    if (size < 0 or size > 4096) return error.RecallRead;
    const out = try a.alloc(u8, @intCast(size));
    if (!readAt(fd, out, 0)) {
        a.free(out);
        return error.RecallRead;
    }
    return out;
}

/// `bytes` as the file at `path` (0600), written whole under another name first.
fn writeSmall(path: [:0]const u8, bytes: []const u8) !void {
    var buf: [1200]u8 = undefined;
    const part = try std.fmt.bufPrintSentinel(&buf, "{s}.part", .{path}, 0);
    const flags: std.c.O = .{ .ACCMODE = .WRONLY, .CREAT = true, .TRUNC = true, .NOFOLLOW = true };
    const fd = std.c.open(part, flags, @as(std.c.mode_t, 0o600));
    if (fd < 0) return error.RecallWrite;
    {
        defer _ = std.c.close(fd);
        try writeAll(fd, bytes);
        if (std.c.fsync(fd) != 0) return error.RecallWrite;
    }
    if (std.c.rename(part, path) != 0) return error.RecallWrite;
}

fn writeAll(fd: c_int, bytes: []const u8) !void {
    var done: usize = 0;
    while (done < bytes.len) {
        const n = std.c.write(fd, bytes.ptr + done, bytes.len - done);
        if (n <= 0) {
            const e = std.c._errno().*;
            if (e == @intFromEnum(std.c.E.INTR) or e == @intFromEnum(std.c.E.AGAIN)) continue;
            return error.RecallWrite;
        }
        done += @intCast(n);
    }
}

fn readAt(fd: c_int, dest: []u8, at: u64) bool {
    var done: usize = 0;
    while (done < dest.len) {
        const n = std.c.pread(fd, dest.ptr + done, dest.len - done, @intCast(at + done));
        if (n <= 0) return false;
        done += @intCast(n);
    }
    return true;
}

/// The volume over `path` holds at least `want` bytes free, or Recall's writes pause.
/// Plain `statfs` f_bavail — conservative on purpose: the purgeable-aware probe
/// lives in the engine unit (`engines/mlx/cache/ssd/disk_tier.zig`), and the sdk
/// must not reach into it. Pausing early only leaves a shorter record.
/// macOS `struct statfs`, leading fields only; the rest is slack. Mirrors the
/// engine unit's probe (`engines/mlx/cache/ssd/disk_tier.zig`) without reaching
/// into it — the sdk must not import engine code.
const StatfsBuf = if (builtin.os.tag.isDarwin())
    extern struct {
        f_bsize: u32,
        f_iosize: i32,
        f_blocks: u64,
        f_bfree: u64,
        f_bavail: u64,
        f_files: u64,
        f_ffree: u64,
        tail: [4096]u8,
    }
else
    extern struct {
        f_type: i64,
        f_bsize: i64,
        f_blocks: u64,
        f_bfree: u64,
        f_bavail: u64,
        f_files: u64,
        f_ffree: u64,
        tail: [512]u8,
    };
extern "c" fn statfs(path: [*:0]const u8, buf: *StatfsBuf) c_int;

fn diskHasRoom(path: []const u8, want: u64) bool {
    var buf: [std.fs.max_path_bytes + 1]u8 = undefined;
    if (path.len >= buf.len) return false;
    @memcpy(buf[0..path.len], path);
    buf[path.len] = 0;
    var st: StatfsBuf = undefined;
    if (statfs(buf[0..path.len :0].ptr, &st) != 0) return false;
    const bsize: u64 = @intCast(@max(if (builtin.os.tag.isDarwin()) @as(i64, st.f_bsize) else st.f_bsize, 1));
    return st.f_bavail *| bsize >= disk_floor +| want;
}

// ────────────────────────── server wiring ──────────────────────────

/// Process-wide Recall state. `Shared` (sdk/util/shared.zig): the host holds
/// the storage and every unit reaches the same copy under `-Dunits`. The host
/// (main.zig flags, the /v1/recall endpoint) and the engine (the PLD miss path,
/// Generator.deinit) both touch it.
pub const State = struct {
    /// main's `Io`, set once at flag parse; every unit re-locals it before use
    /// (`io_util.local`, the cross-unit rule in host/AGENTS.md).
    io: ?std.Io = null,
    /// Any --recall flag sets this; every hook is a single check against it.
    enabled: bool = false,
    /// --recall-read-only: draft from the record, add nothing.
    read_only: bool = false,
    /// --recall-shared: with an API key active, let non-loopback peers use
    /// the record too. Without it, only loopback requests read or write it —
    /// a record hit decodes measurably faster, which would otherwise let any
    /// keyed or LAN client probe what other clients' conversations held.
    shared: bool = false,
    /// Bytes of record per tokenizer (chunk files' full bytes); the oldest chunk file goes first past it.
    cap_bytes: u64 = 64 << 20,
    /// Base directory (c_allocator-owned, set once from main before any request);
    /// each tokenizer's record lives in `<base>/<identity digest prefix>`.
    base_dir: []const u8 = "",
    mutex: os_mutex.Mutex = .{},
    stores: std.AutoHashMapUnmanaged([32]u8, *Store) = .empty,
    failures: std.AutoHashMapUnmanaged([32]u8, void) = .empty,
};

pub const shared_state = shared_util.Shared(State, "recall", .{});

/// Set by main.zig from `--recall-dir`. Owned by the c_allocator, process lifetime.
pub fn setBaseDir(dir: []const u8) bool {
    shared_state.get().base_dir = gpa.dupe(u8, dir) catch return false;
    return true;
}

/// Default base when --recall-dir is not passed. Caller frees.
fn defaultBaseDir() ?[]const u8 {
    const home = std.c.getenv("HOME") orelse return null;
    return std.fmt.allocPrint(gpa, "{s}/.cache/mlx-serve/recall", .{std.mem.span(home)}) catch null;
}

fn storeDir(a: Allocator, digest: [32]u8) ?[]const u8 {
    const sh_id = Identity{ .digest = digest, .vocab = 0, .name = "" };
    const sh = sh_id.short();
    var base: []const u8 = undefined;
    var base_owned = false;
    base = shared_state.get().base_dir;
    if (base.len == 0) {
        base = defaultBaseDir() orelse return null;
        base_owned = true;
    }
    defer if (base_owned) gpa.free(base);
    return std.fmt.allocPrint(a, "{s}/{s}", .{ base, &sh }) catch null;
}

/// The record for this tokenizer, opened on first use; null (logged once) when
/// the tokenizer has no identity (test/manual tokenizers: vocab 0) or its record
/// cannot be opened. The `enabled` flag is checked first, so every call site is
/// a single null-check when Recall is off.
pub fn storeFor(tok: *const Tokenizer) ?*Store {
    // The record is POSIX by design (flock, append-mode open, 0700 files); a
    // Windows build has none of it. The comptime gate also prunes the whole
    // fd layer from the Windows target's analysis (std.c's open/close/write
    // are Windows-shaped there), so recall compiles as off-with-a-reason.
    if (builtin.os.tag == .windows) return null;
    if (!shared_state.get().enabled) return null;
    if (tok.recall_vocab == 0) return null; // dummy/test tokenizer: never records
    const digest = tok.recall_identity;
    const st_state = shared_state.get();
    st_state.mutex.lock();
    defer st_state.mutex.unlock();
    if (st_state.stores.get(digest)) |st| return st;
    if (st_state.failures.contains(digest)) return null;
    const io = io_util.local(st_state.io orelse return null);
    var problem: []const u8 = "";
    defer if (problem.len > 0) gpa.free(problem);
    const dir = storeDir(gpa, digest) orelse {
        log.err("--recall: cannot build its directory path\n", .{});
        st_state.failures.put(gpa, digest, {}) catch {};
        return null;
    };
    defer gpa.free(dir);
    const identity = Identity{ .digest = digest, .vocab = tok.recall_vocab, .name = tok.recall_name };
    const st = Store.open(identity, io, .{ .dir = dir, .cap_bytes = st_state.cap_bytes, .read_only = st_state.read_only }, &problem, gpa) catch {
        log.err("--recall: cannot open the record at {s}: {s}\n", .{ dir, problem });
        st_state.failures.put(gpa, digest, {}) catch {};
        return null;
    } orelse {
        log.err("--recall: {s}\n", .{problem});
        st_state.failures.put(gpa, digest, {}) catch {};
        return null;
    };
    st_state.stores.put(gpa, digest, st) catch {
        st.close();
        return null;
    };
    log.info("recall: record for {s} at {s} (cap {d} MiB, {s})\n", .{
        tok.recall_name,
        st.dir,
        st_state.cap_bytes >> 20,
        if (st.writes) "writing" else "read-only",
    });
    return st;
}

/// The spec-gate exemption's probe: does the record hold a continuation for
/// this request's exact prompt tail? A full-tail match (the backward extension
/// reaches the whole probe window) means a reply to this prompt ending was
/// recorded before — the one case the prompt's own n-gram score cannot see.
/// Anything less (a novel prompt, an empty record, a variant that shares only
/// part of the tail) fails, so the gate treats the request exactly as upstream
/// does: arming PLD there costs its mode overhead with nothing to draft.
pub fn promptMayDraft(tok: *const Tokenizer, prompt: []const u32) bool {
    if (prompt.len < key_len) return false;
    const st = storeFor(tok) orelse return false;
    st.mutex.lock();
    defer st.mutex.unlock();
    const m = st.record.longest(prompt);
    return m.len >= @min(prompt.len, max_match);
}

/// A per-stream reader over the record for this tokenizer (null when Recall is off).
pub fn armReader(tok: *const Tokenizer) ?Reader {
    const st = storeFor(tok) orelse return null;
    return .{ .st = st };
}

/// What a finished request read and wrote: the prompt's new ids and the reply's.
/// v1 note: the hot-prefix resume point is not plumbed here yet, so `resumed`
/// is 0 — the held-run dedup keeps a re-sent prefix from being written again.
pub fn finishRequest(tok: *const Tokenizer, prompt: []const u32, reply: []const u32) void {
    const st = storeFor(tok) orelse return;
    st.readPrompt(prompt, 0);
    st.readReply(prompt, reply);
}

/// All open records' stats, for `GET /v1/recall`. Caller frees.
pub fn allStats(a: Allocator) []Stats {
    if (builtin.os.tag == .windows) return &[_]Stats{};
    const st_state = shared_state.get();
    st_state.mutex.lock();
    const n = st_state.stores.count();
    const out = a.alloc(Stats, n) catch {
        st_state.mutex.unlock();
        return &[_]Stats{};
    };
    var it = st_state.stores.iterator();
    var i: usize = 0;
    while (it.next()) |e| : (i += 1) out[i] = e.value_ptr.*.stats();
    st_state.mutex.unlock();
    return out;
}

pub const EmptyRow = struct {
    /// Both `a`-owned: the caller frees every row's `model` and `dir`.
    model: []const u8,
    dir: []const u8,
    ok: bool,
    read_only: bool,
    problem: []const u8,
};

/// `DELETE /v1/recall`: every record emptied — the open stores in memory and
/// on disk, and the records on disk no request ever opened this run (stores
/// open lazily, so a fresh server must still be able to empty what a previous
/// run left). `all_failed` maps to 409 (no record could be emptied); `rows`
/// is caller-freed.
pub fn emptyAll(a: Allocator) struct { rows: []EmptyRow, all_failed: bool } {
    if (builtin.os.tag == .windows) return .{ .rows = &[_]EmptyRow{}, .all_failed = false };
    const st_state = shared_state.get();
    var rows: std.ArrayList(EmptyRow) = .empty;
    defer rows.deinit(a);
    var failed_n: usize = 0;

    // 1. The open stores: cleared in memory, their chunk files unlinked.
    {
        st_state.mutex.lock();
        const n = st_state.stores.count();
        const list = a.alloc(*Store, n) catch {
            st_state.mutex.unlock();
            return .{ .rows = &[_]EmptyRow{}, .all_failed = n > 0 };
        };
        var it = st_state.stores.iterator();
        var i: usize = 0;
        while (it.next()) |e| : (i += 1) list[i] = e.value_ptr.*;
        st_state.mutex.unlock();
        defer a.free(list);
        for (list[0..n]) |st| {
            var row: EmptyRow = .{
                .model = a.dupe(u8, st.name) catch "",
                .dir = a.dupe(u8, st.dir) catch "",
                .ok = true,
                .read_only = st.read_only,
                .problem = "",
            };
            st.empty() catch |e| switch (e) {
                error.ReadOnly => {
                    row.ok = false;
                    row.problem = "the record is read-only";
                },
                error.Unlinked => {
                    row.ok = false;
                    row.problem = st.why();
                },
            };
            if (!row.ok) failed_n += 1;
            rows.append(a, row) catch return .{ .rows = &[_]EmptyRow{}, .all_failed = n > 0 };
        }
    }

    // 2. The records on disk no open store covers: their chunk files unlinked
    //    under the directory's own flock, so a record another server is
    //    writing (or this server holds read-only) is left alone and named.
    sweepUnopened(a, &rows, &failed_n);

    const out = rows.toOwnedSlice(a) catch return .{ .rows = &[_]EmptyRow{}, .all_failed = true };
    return .{ .rows = out, .all_failed = out.len > 0 and failed_n == out.len };
}

/// Every identity directory under the base whose digest no open store holds:
/// unlink its `*.ids` chunk files while holding its lock, and report a row.
/// The `identity` and `lock` files stay — they hold no ids and name the
/// record's tokenizer, so a later request reopens the emptied record cleanly.
fn sweepUnopened(a: Allocator, rows: *std.ArrayList(EmptyRow), failed_n: *usize) void {
    const st_state = shared_state.get();
    const io = st_state.io orelse return;
    var base: []const u8 = undefined;
    var base_owned = false;
    base = st_state.base_dir;
    if (base.len == 0) {
        base = defaultBaseDir() orelse return;
        base_owned = true;
    }
    defer if (base_owned) gpa.free(base);
    var dir = std.Io.Dir.openDirAbsolute(io, base, .{ .iterate = true }) catch return;
    defer dir.close(io);
    var it = dir.iterate();
    while (it.next(io) catch return) |ent| {
        if (ent.kind != .directory) continue;
        // The digest no open store holds: open stores were emptied above.
        var held = false;
        {
            st_state.mutex.lock();
            defer st_state.mutex.unlock();
            var sit = st_state.stores.iterator();
            while (sit.next()) |e| {
                if (std.mem.endsWith(u8, e.value_ptr.*.dir, ent.name)) held = true;
            }
        }
        if (held) continue;
        var row: EmptyRow = .{
            .model = tryNameFromIdentity(a, io, base, ent.name),
            .dir = std.fmt.allocPrint(a, "{s}/{s}", .{ base, ent.name }) catch continue,
            .ok = true,
            .read_only = false,
            .problem = "",
        };
        emptyDirChunkFiles(io, base, ent.name, &row);
        if (!row.ok) failed_n.* += 1;
        rows.append(a, row) catch {};
    }
}

/// One record directory's chunk files unlinked under its lock; a record held
/// by another server (or a lock we cannot take) is reported, not touched.
fn emptyDirChunkFiles(io: std.Io, base: []const u8, name: []const u8, row: *EmptyRow) void {
    var buf: [1100]u8 = undefined;
    const lock_path = std.fmt.bufPrintSentinel(&buf, "{s}/{s}/lock", .{ base, name }, 0) catch return;
    const lock_fd = std.c.open(lock_path, .{ .ACCMODE = .RDWR, .CREAT = true, .NOFOLLOW = true }, @as(std.c.mode_t, 0o600));
    if (lock_fd < 0) {
        row.ok = false;
        row.problem = "its directory could not be opened";
        return;
    }
    defer _ = std.c.close(lock_fd);
    if (flock(lock_fd, LOCK_EX | LOCK_NB) != 0) {
        row.ok = false;
        row.problem = "another server holds this record";
        return;
    }
    var dbuf: [1100]u8 = undefined;
    const dir_path = std.fmt.bufPrintSentinel(&dbuf, "{s}/{s}", .{ base, name }, 0) catch return;
    var dir = std.Io.Dir.openDirAbsolute(io, dir_path, .{ .iterate = true }) catch {
        row.ok = false;
        row.problem = "its directory could not be opened";
        return;
    };
    defer dir.close(io);
    var names: std.ArrayList([]u8) = .empty;
    defer {
        for (names.items) |nm| gpa.free(nm);
        names.deinit(gpa);
    }
    var it = dir.iterate();
    while (it.next(io) catch null) |ent| {
        if (!std.mem.endsWith(u8, ent.name, ".ids")) continue;
        const nm = gpa.dupe(u8, ent.name) catch continue;
        names.append(gpa, nm) catch {
            gpa.free(nm);
            continue;
        };
    }
    for (names.items) |nm| {
        var pbuf: [1100]u8 = undefined;
        const path = std.fmt.bufPrintSentinel(&pbuf, "{s}/{s}/{s}", .{ base, name, nm }, 0) catch continue;
        if (std.c.unlink(path) != 0) {
            row.ok = false;
            row.problem = "a chunk file could not be removed";
        }
    }
}

/// The `name ` line of a record directory's identity file, or the directory's
/// own name when there is none. Caller frees.
fn tryNameFromIdentity(a: Allocator, io: std.Io, base: []const u8, name: []const u8) []const u8 {
    var buf: [1100]u8 = undefined;
    const path = std.fmt.bufPrintSentinel(&buf, "{s}/{s}/identity", .{ base, name }, 0) catch return a.dupe(u8, name) catch name;
    var file = std.Io.Dir.openFileAbsolute(io, path, .{}) catch return a.dupe(u8, name) catch name;
    defer file.close(io);
    var rbuf: [4096]u8 = undefined;
    var rs = file.reader(io, &rbuf);
    const content = rs.interface.allocRemaining(a, .limited(4096)) catch return a.dupe(u8, name) catch name;
    var lines = std.mem.splitScalar(u8, content, '\n');
    while (lines.next()) |l| {
        if (std.mem.startsWith(u8, l, "name ") and l.len > 5) return a.dupe(u8, l[5..]) catch name;
    }
    return a.dupe(u8, name) catch name;
}

// ────────────────────────── tests ──────────────────────────

const testing = std.testing;
const recall_max_match: u32 = max_match;

fn openTestStore(dir: [:0]const u8) !*Store {
    var problem: []const u8 = "";
    defer if (problem.len > 0) testing.allocator.free(problem);
    const st = (try Store.open(
        .{ .digest = @splat(1), .vocab = 100_000, .name = "test-model" },
        testing.io,
        .{ .dir = dir, .cap_bytes = 4 * 1024 * 1024, .read_only = false },
        &problem,
        testing.allocator,
    )) orelse return error.TestStoreOpen;
    return st;
}

test "record: longest match, copy stops at the mark, heldRun, eviction" {
    var r = try Record.init(testing.allocator, 10);
    defer r.deinit();

    // Nothing before key_len ids.
    try testing.expectEqual(@as(u32, 0), r.longest(&[_]u32{ 3, 4, 5, 6, 7, 8 }).len);

    const a_ids = [_]u32{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
    try r.add(&a_ids);
    const ctx = [_]u32{ 2, 3, 4, 5, 6, 7, 8, 9 };
    const m = r.longest(&ctx);
    try testing.expectEqual(@as(u32, 8), m.len);
    var out: [4]u32 = undefined;
    const got = r.copy(m.at, &out);
    try testing.expectEqual(@as(usize, 1), got.len);
    try testing.expectEqual(@as(u32, 10), got[0]); // the copy stops at the segment's mark

    // heldRun finds the run again: a re-sent stream is recognized as held.
    const run = r.heldRun(&a_ids);
    try testing.expectEqual(a_ids.len, run.len);

    // Eviction: the oldest chunk goes, base advances, old matches disappear.
    while (r.chunks.items.len < 3) {
        var filler: [4096]u32 = undefined;
        for (&filler, 0..) |*t, i| t.* = @intCast(200_000 + i % 50);
        try r.add(&filler);
    }
    _ = r.dropOldest();
    try testing.expect(r.longest(&ctx).len < m.len);
}

test "store: identity roundtrip, dedup, restart, lock makes second writer read-only" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const root = buf[0..try tmp.dir.realPath(std.testing.io, &buf)];

    const d0 = try std.fmt.allocPrint(testing.allocator, "{s}/rec", .{root});
    defer testing.allocator.free(d0);
    const dir = try testing.allocator.dupeSentinel(u8, d0, 0);
    defer testing.allocator.free(dir);

    // Long-enough runs that the held-run dedup engages: a held run shorter
    // than min_run (128) is deliberately written again (a join costs two
    // overlaps), so both the prompt and the reply exceed it.
    var prompt: [160]u32 = undefined;
    for (&prompt, 0..) |*t, i| t.* = @intCast(1000 + i);
    var reply: [160]u32 = undefined;
    for (&reply, 0..) |*t, i| t.* = @intCast(5000 + i);

    const s = try openTestStore(dir);
    s.readPrompt(&prompt, 0);
    s.readReply(&prompt, &reply);
    try testing.expect(s.added > 0);

    // A second identical request adds nothing: the held-run dedup.
    const added_before = s.added;
    s.readPrompt(&prompt, 0);
    s.readReply(&prompt, &reply);
    try testing.expectEqual(added_before, s.added);

    // The reply is findable as a context continuation: ask with a prefix of
    // it, and the record answers with the ids that followed (the reply's own
    // tail is a segment end, which by design has no continuation).
    {
        s.mutex.lock();
        defer s.mutex.unlock();
        const m = s.record.longest(reply[0..100]);
        try testing.expectEqual(recall_max_match, m.len);
        var out: [8]u32 = undefined;
        const got = s.record.copy(m.at, &out);
        try testing.expectEqual(@as(usize, 8), got.len);
        try testing.expectEqual(reply[100], got[0]);
        try testing.expectEqual(reply[101], got[1]);
    }

    // A second server on the same directory — while the first still holds
    // the flock — reads the record from disk and writes nothing.
    const s2 = try openTestStore(dir);
    defer s2.close();
    try testing.expect(!s2.writes);
    {
        s2.mutex.lock();
        defer s2.mutex.unlock();
        const m = s2.record.longest(reply[0..100]);
        try testing.expectEqual(recall_max_match, m.len);
    }

    // A different tokenizer is refused by name.
    var problem: []const u8 = "";
    defer if (problem.len > 0) testing.allocator.free(problem);
    const other = try Store.open(
        .{ .digest = @splat(9), .vocab = 500, .name = "other-model" },
        testing.io,
        .{ .dir = dir, .cap_bytes = 4 * 1024 * 1024, .read_only = false },
        &problem,
        testing.allocator,
    );
    try testing.expect(other == null);
    try testing.expect(std.mem.indexOf(u8, problem, "other-model") != null);
    s.close();
}

test "rates: expected tokens, bands, probe backoff, quiet rounds" {
    const e4 = expected(0.6, 4);
    try testing.expect(e4 > 2 and e4 < 5);
    try testing.expect(expected(0.0, 8) == 1);
    try testing.expectEqual(@as(usize, 0), Rates.bandOf(16));
    try testing.expectEqual(@as(usize, 1), Rates.bandOf(32));
    try testing.expectEqual(@as(usize, 2), Rates.bandOf(1000));

    var rates = Rates{};
    // A probe fires only after probe_every unpriced asks (32 by default).
    var n: u32 = 0;
    while (n < 31) : (n += 1) try testing.expect(!rates.probe());
    try testing.expect(rates.probe());
    // A losing probe doubles the wait.
    rates.observe(0, 4, 0, true);
    try testing.expectEqual(@as(u32, 64), rates.probe_every);
    n = 0;
    while (n < 63) : (n += 1) try testing.expect(!rates.probe());
    try testing.expect(rates.probe());
    // A winning probe resets it.
    rates.observe(0, 4, 4, true);
    try testing.expectEqual(@as(u32, 32), rates.probe_every);

    // quiet: a zero-landing copy keeps the stream off for a few rounds.
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const root = buf[0..try tmp.dir.realPath(std.testing.io, &buf)];
    const d0 = try std.fmt.allocPrint(testing.allocator, "{s}/r2", .{root});
    defer testing.allocator.free(d0);
    const dir = try testing.allocator.dupeSentinel(u8, d0, 0);
    defer testing.allocator.free(dir);
    const s = try openTestStore(dir);
    defer s.close();
    var rd = Reader{ .st = s };
    defer rd.deinit();
    rd.observe(4, 0, 1);
    try testing.expectEqual(@as(u64, 1 + 1 + rd.params.quiet), rd.quiet_until);
}
