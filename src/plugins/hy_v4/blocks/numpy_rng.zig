//! NumPy's seeding chain, bit for bit: `SeedSequence(seed)` -> `PCG64` -> `Generator.integers(0, bound)`.
//!
//! Hy4-flash's Engram hash multipliers are NOT stored in the pack — transformers redraws them at load from
//! `np.random.default_rng(seed + 10007*layer_id).integers(0, bound, size=max_ngram)` — so reading a pack's
//! memories at parity needs numpy's exact draws. Every constant and step below is transcribed from numpy's
//! sources (`numpy/random/bit_generator.pyx`, `src/pcg64/pcg64.h`, `src/distributions/distributions.c`);
//! the tests pin golden draws produced by numpy itself.
//!
//! copied_from: numpy (BSD-3-Clause), algorithm only.

const std = @import("std");

// SeedSequence's mixing constants (`bit_generator.pyx`).
const INIT_A: u32 = 0x43b0d7e5;
const MULT_A: u32 = 0x931e8875;
const INIT_B: u32 = 0x8b51f9dd;
const MULT_B: u32 = 0x58f38ded;
const MIX_MULT_L: u32 = 0xca01f9dd;
const MIX_MULT_R: u32 = 0x4973f715;
const XSHIFT: u5 = 16;

const POOL: usize = 4;

fn hashmix(value_in: u32, hash_const: *u32) u32 {
    var value = value_in ^ hash_const.*;
    hash_const.* *%= MULT_A;
    value *%= hash_const.*;
    value ^= value >> XSHIFT;
    return value;
}

fn mix(x: u32, y: u32) u32 {
    var r = MIX_MULT_L *% x -% MIX_MULT_R *% y;
    r ^= r >> XSHIFT;
    return r;
}

/// `SeedSequence(entropy)` over a pool of 4 u32 words (numpy's default).
pub const SeedSequence = struct {
    pool: [POOL]u32 = @splat(0),

    pub fn init(entropy: u64) SeedSequence {
        // `_coerce_to_uint32_array`: an int becomes its u32 words, lowest first (a zero is one word).
        var words: [16]u32 = undefined;
        var n: usize = 0;
        var x = entropy;
        if (x == 0) {
            words[0] = 0;
            n = 1;
        } else while (x != 0) {
            words[n] = @truncate(x & 0xffffffff);
            n += 1;
            x >>= 32;
        }
        var hc: u32 = INIT_A;
        var pool: [POOL]u32 = @splat(0);
        for (0..POOL) |i| pool[i] = hashmix(if (i < n) words[i] else 0, &hc);
        // Mix all bits together so late bits can affect earlier bits.
        for (0..POOL) |src| {
            for (0..POOL) |dst| {
                if (src != dst) pool[dst] = mix(pool[dst], hashmix(pool[src], &hc));
            }
        }
        // Entropy past the pool mixes each word into every pool word.
        if (n > POOL) for (POOL..n) |src| {
            for (0..POOL) |dst| pool[dst] = mix(pool[dst], hashmix(words[src], &hc));
        };
        return .{ .pool = pool };
    }

    /// `generate_state(n, uint32)`: the pool cycled and hashed with INIT_B/MULT_B.
    pub fn generateState(self: SeedSequence, out: []u32) void {
        var hc: u32 = INIT_B;
        for (out, 0..) |*w, i| {
            var v = self.pool[i % POOL] ^ hc;
            hc *%= MULT_B;
            v *%= hc;
            v ^= v >> XSHIFT;
            w.* = v;
        }
    }
};

fn self_pool_init(_: *const SeedSequence, _: usize, _: u32) void {}

// PCG64 (`src/pcg64/pcg64.h`).
const MULT_LOW: u64 = 4865540595714422341;
const MULT_HIGH: u64 = 2549297995355413924;

/// PCG64's 128-bit state and odd increment, seeded from a SeedSequence's four u64 words.
pub const Pcg64 = struct {
    state: u128 = 0,
    inc: u128 = 0,
    /// The buffered upper half of the last draw (`pcg64_next32` reuses it).
    has_u32: bool = false,
    u32: u32 = 0,

    pub fn init(seed: u64) Pcg64 {
        var sq = SeedSequence.init(seed);
        var words: [8]u32 = undefined;
        sq.generateState(&words);
        // numpy views the eight u32 words as four little-endian u64, then builds the 128-bit values with
        // `(seed[0] << 64) | seed[1]`: the FIRST word is the high half.
        const val0: u64 = @as(u64, words[0]) | (@as(u64, words[1]) << 32);
        const val1: u64 = @as(u64, words[2]) | (@as(u64, words[3]) << 32);
        const val2: u64 = @as(u64, words[4]) | (@as(u64, words[5]) << 32);
        const val3: u64 = @as(u64, words[6]) | (@as(u64, words[7]) << 32);
        const initstate: u128 = (@as(u128, val0) << 64) | val1;
        const initseq: u128 = (@as(u128, val2) << 64) | val3;
        var r = Pcg64{};
        r.state = 0;
        r.inc = (initseq << 1) | 1;
        r.step();
        r.state +%= initstate;
        r.step();
        return r;
    }

    fn mult128(a: u128, b_hi: u64, b_lo: u64) u128 {
        const b: u128 = (@as(u128, b_hi) << 64) | b_lo;
        return a *% b;
    }

    fn step(r: *Pcg64) void {
        r.state = mult128(r.state, MULT_HIGH, MULT_LOW) +% r.inc;
    }

    /// `pcg64_next64`: step, then `rotr64(high ^ low, high >> 58)`.
    pub fn next64(r: *Pcg64) u64 {
        r.step();
        const hi: u64 = @truncate(r.state >> 64);
        const lo: u64 = @truncate(r.state);
        const v = hi ^ lo;
        const rot: u6 = @truncate(hi >> 58);
        return std.math.rotr(u64, v, rot);
    }

    /// `pcg64_next32`: the low half of one 64-bit draw, the upper half buffered for the next call.
    pub fn next32(r: *Pcg64) u32 {
        if (r.has_u32) {
            r.has_u32 = false;
            return r.u32;
        }
        const v = r.next64();
        r.u32 = @truncate(v >> 32);
        r.has_u32 = true;
        return @truncate(v);
    }
};

/// Lemire's bounded draw, `[0, bound)`: numpy's `random_bounded_uint64` (32-bit path below 2^32).
pub fn bounded(r: *Pcg64, bound: u64) u64 {
    if (bound == 0) return 0;
    if (bound <= 0xFFFFFFFF) {
        const rng: u32 = @truncate(bound - 1);
        if (bound == 0x100000000) return r.next32();
        const rng_excl: u64 = @as(u64, rng) + 1;
        var m: u64 = @as(u64, r.next32()) * rng_excl;
        var leftover: u32 = @truncate(m);
        if (leftover < rng_excl) {
            const threshold = (@as(u64, 0xFFFFFFFF) - rng) % rng_excl;
            while (leftover < threshold) {
                m = @as(u64, r.next32()) * rng_excl;
                leftover = @truncate(m);
            }
        }
        return m >> 32;
    }
    const rng = bound - 1;
    const rng_excl = bound;
    var m: u128 = @as(u128, r.next64()) * rng_excl;
    var leftover: u64 = @truncate(m);
    if (leftover < rng_excl) {
        const threshold = (std.math.maxInt(u64) - rng) % rng_excl;
        while (leftover < threshold) {
            m = @as(u128, r.next64()) * rng_excl;
            leftover = @truncate(m);
        }
    }
    return @truncate(m >> 64);
}

/// `np.random.default_rng(seed).integers(0, bound, size = out.len, dtype=int64)`.
pub fn integers(seed: u64, bound: u64, out: []u64) void {
    var r = Pcg64.init(seed);
    for (out) |*v| v.* = bounded(&r, bound);
}

/// The odd multipliers an Engram layer mixes an n-gram with: `drawn * 2 + 1`, bounded so a product with any
/// compressed id stays inside int64.
pub fn hashMultipliers(seed: u64, compressed_vocab: u64, max_ngram: usize, out: []u64) void {
    const half_bound = @max(@as(u64, 1), @as(u64, @intCast(std.math.maxInt(i64))) / compressed_vocab / 2);
    integers(seed, half_bound, out[0..max_ngram]);
    for (out[0..max_ngram]) |*v| v.* = v.* * 2 + 1;
}

// ── tests: golden draws from numpy ──

const testing = std.testing;

test "hy_v4 numpy rng: SeedSequence/PCG64/Lemire reproduce numpy's integers draws" {
    // Goldens from numpy 2.4.1 (`np.random.default_rng(seed).integers(0, bound, size=n, dtype=np.int64)`).
    const cases = [_]struct { seed: u64, bound: u64, want: []const u64 }{
        .{ .seed = 0, .bound = 100, .want = &.{ 85, 63, 51 } },
        .{ .seed = 1, .bound = 65536, .want = &.{ 31010, 33542, 49490, 62289 } }, // the 32-bit Lemire arm
        .{ .seed = 42, .bound = 123456789, .want = &.{ 11018636, 95550128, 80811297 } }, // 32-bit again
        .{ .seed = 1234567890123, .bound = 38170521101387, .want = &.{ 32915205067434, 9454895344647, 5926631122039 } }, // 64-bit arm
    };
    for (cases) |c| {
        var got: [4]u64 = undefined;
        integers(c.seed, c.bound, got[0..c.want.len]);
        try testing.expectEqualSlices(u64, c.want, got[0..c.want.len]);
    }
}

test "hy_v4 numpy rng: the Engram hash multipliers match numpy's per-layer draws" {
    // `seed + 10007*HF_layer_id` (1-based: pack layers 1/11/21/31 are HF ids 2/12/22/32). The bound is
    // `INT64_MAX // (max(lookup map)+1) // 2`: the pack's trained map tops at 92619 (compressed vocab 92620),
    // so it is NOT the table's 120818-row length. Multipliers are `drawn * 2 + 1`.
    const cases = [_]struct { layer: u64, comp: u64, want: [3]u64 }{
        .{ .layer = 2, .comp = 92620, .want = .{ 81853575799317, 39690085502883, 24229915018165 } },
        .{ .layer = 12, .comp = 92620, .want = .{ 19457948045719, 44729556510419, 84779352922315 } },
        .{ .layer = 22, .comp = 92620, .want = .{ 72974230057307, 21530602121221, 55998057243097 } },
        .{ .layer = 32, .comp = 92620, .want = .{ 29128509562787, 16745155128523, 22474588302733 } },
        .{ .layer = 2, .comp = 64, .want = .{ 118457471727074801, 57438995613704923, 35065230140349983 } },
    };
    for (cases) |c| {
        var got: [3]u64 = undefined;
        hashMultipliers(10007 * c.layer, c.comp, 3, &got);
        try testing.expectEqualSlices(u64, &c.want, &got);
        // Every multiplier is odd, as the reference requires.
        for (got) |m| try testing.expect(m & 1 == 1);
    }
}
