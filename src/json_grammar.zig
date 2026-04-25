//! Streaming JSON grammar state machine constrained by a `json_schema.Schema`.
//!
//! Architecture
//! ============
//! The grammar holds a single Config = a stack of Frames. Each Frame represents
//! an in-progress JSON value being parsed against a schema node. Bytes are fed in
//! one-at-a-time via `acceptByte`. The `allowedBytes` query enumerates which next
//! bytes would not be rejected — this is what the token-mask layer uses to filter
//! the LLM's vocabulary.
//!
//! Snapshots support speculative lookahead: the token-mask builder simulates each
//! candidate token's bytes through the grammar from a snapshot, accepting/rejecting
//! the token without permanently advancing real state.
//!
//! Schema construct support
//! ========================
//! Enforced strictly:
//!   * JSON syntax (RFC 8259), including \uXXXX escapes and control-char rejection
//!   * `type`: object/array/string/number/integer/boolean/null
//!   * `properties` + `required` + `additionalProperties: false`
//!   * `items` (homogeneous arrays)
//!   * `enum` / `const` (prefix-matched against canonical JSON encoding)
//!   * `minItems` / `maxItems`
//!   * `minLength` / `maxLength` on strings
//! Best-effort (relaxed) — accept any valid JSON value, rely on prompt for shape:
//!   * `anyOf` / `oneOf` — treated as "any JSON value" inside the union
//!   * `pattern`, `minimum`, `maximum`, `exclusiveMinimum/Maximum` — soft prompt-only
//!
//! Why anyOf is relaxed
//! --------------------
//! Tagged-union anyOf (e.g., `{type:"text",content}` vs `{type:"cruise_cards",query}`)
//! requires multi-config parallel parsing or eager discriminator resolution. Either
//! adds significant complexity and is the most common cause of grammar bugs. We
//! accept the relaxation: the model still sees the schema in its system prompt and
//! produces structurally-valid JSON for the union; the grammar guarantees the
//! surrounding object/array is well-formed and that all *non-anyOf* parts of the
//! response strictly conform.

const std = @import("std");
const schema_mod = @import("json_schema.zig");

pub const Schema = schema_mod.Schema;
pub const Node = schema_mod.Node;
pub const Kind = schema_mod.Kind;

// ── ByteMask ──────────────────────────────────────────────────────────────────

pub const ByteMask = struct {
    bits: [4]u64 = @splat(0),

    pub fn empty() ByteMask {
        return .{};
    }

    pub fn full() ByteMask {
        return .{ .bits = .{ ~@as(u64, 0), ~@as(u64, 0), ~@as(u64, 0), ~@as(u64, 0) } };
    }

    pub fn add(self: *ByteMask, b: u8) void {
        self.bits[b >> 6] |= (@as(u64, 1) << @intCast(b & 63));
    }

    pub fn addRange(self: *ByteMask, lo: u8, hi: u8) void {
        var b: u16 = lo;
        while (b <= hi) : (b += 1) self.add(@intCast(b));
    }

    pub fn contains(self: *const ByteMask, b: u8) bool {
        return (self.bits[b >> 6] >> @intCast(b & 63)) & 1 == 1;
    }

    pub fn isEmpty(self: *const ByteMask) bool {
        return self.bits[0] == 0 and self.bits[1] == 0 and self.bits[2] == 0 and self.bits[3] == 0;
    }
};

fn isWs(b: u8) bool {
    return b == ' ' or b == '\t' or b == '\n' or b == '\r';
}

// ── Frame state types ─────────────────────────────────────────────────────────

const StringPhase = enum { content, after_backslash, unicode_hex };

const StringState = struct {
    byte_count: u32 = 0,
    phase: StringPhase = .content,
    hex_remaining: u3 = 0,
};

const NumberPhase = enum {
    after_minus,
    leading_zero,
    int_digits,
    after_dot,
    frac_digits,
    after_exp,
    after_exp_sign,
    exp_digits,
};

const NumberState = struct {
    phase: NumberPhase,
};

const ObjectPhase = enum {
    expect_first_key_or_close,
    expect_key, // after `,`
    in_key,
    after_key,
    expect_value,
    after_value,
};

const ObjectState = struct {
    phase: ObjectPhase,
    /// Bitset of property indices already seen (max 64 properties).
    seen: u64 = 0,
    /// Buffered key being typed.
    key_buf: [256]u8 = undefined,
    key_len: u32 = 0,
    /// Resolved during `:` handling: index of property whose value we're about to read,
    /// or null for an additional-property (free-form key).
    current_prop: ?u32 = null,
};

const ArrayPhase = enum {
    expect_first_value_or_close,
    expect_value,
    after_value,
};

const ArrayState = struct {
    phase: ArrayPhase,
    count: u32 = 0,
};

const EnumState = struct {
    /// Bitset of enum_values still alive after the prefix typed so far (max 64).
    alive: u64,
    byte_count: u32 = 0,
};

const KeywordState = struct {
    target: []const u8,
    pos: u8,
};

const FrameSub = union(enum) {
    expect_value,
    in_string: StringState,
    in_number: NumberState,
    in_object: ObjectState,
    in_array: ArrayState,
    in_enum: EnumState,
    in_keyword: KeywordState,
    accepted,
};

const Frame = struct {
    schema: *const Node,
    sub: FrameSub,
};

// ── Grammar ───────────────────────────────────────────────────────────────────

const MAX_STACK = 64;

pub const Grammar = struct {
    gpa: std.mem.Allocator,
    schema: *const Schema,
    stack: std.ArrayListUnmanaged(Frame),
    /// Set true once all configurations have rejected and recovery is impossible.
    dead: bool = false,

    pub fn init(gpa: std.mem.Allocator, schema: *const Schema) std.mem.Allocator.Error!Grammar {
        var g: Grammar = .{
            .gpa = gpa,
            .schema = schema,
            .stack = .empty,
        };
        // anyOf at the root is relaxed to "any JSON" (see header doc).
        const root = if (schema.root.kind == .any_of) anyNode() else schema.root;
        try g.stack.append(gpa, .{ .schema = root, .sub = .expect_value });
        return g;
    }

    pub fn deinit(self: *Grammar) void {
        self.stack.deinit(self.gpa);
    }

    pub fn isDead(self: *const Grammar) bool {
        return self.dead;
    }

    /// True once the root value has been fully parsed. (Trailing whitespace allowed.)
    pub fn isComplete(self: *const Grammar) bool {
        return self.stack.items.len == 1 and self.stack.items[0].sub == .accepted;
    }

    /// Bitset of bytes that would be accepted by the grammar in its current state.
    /// Implemented by speculative trial: snapshot, try each byte, restore.
    pub fn allowedBytes(self: *Grammar) std.mem.Allocator.Error!ByteMask {
        var mask = ByteMask.empty();
        if (self.dead) return mask;

        const snap = try self.snapshot();
        defer self.discardSnapshot(snap);

        var b: u16 = 0;
        while (b < 256) : (b += 1) {
            const byte: u8 = @intCast(b);
            try self.restoreFrom(snap);
            if (try self.tryAdvance(byte)) {
                mask.add(byte);
            }
        }
        try self.restoreFrom(snap);
        return mask;
    }

    /// Try to consume one byte. Returns true if accepted (state advanced),
    /// false if rejected (state unchanged).
    pub fn acceptByte(self: *Grammar, byte: u8) std.mem.Allocator.Error!bool {
        if (self.dead) return false;
        const snap = try self.snapshot();
        defer self.discardSnapshot(snap);
        const ok = try self.tryAdvance(byte);
        if (!ok) try self.restoreFrom(snap);
        return ok;
    }

    /// Speculative variant of `acceptByte` that does NOT take an internal snapshot.
    /// On rejection, the grammar may be left in an indeterminate state — the caller
    /// MUST hold an outer snapshot and restore it. Used by the token-mask builder
    /// to amortise away ~100k redundant per-byte allocations.
    pub fn acceptByteFast(self: *Grammar, byte: u8) std.mem.Allocator.Error!bool {
        if (self.dead) return false;
        return try self.tryAdvance(byte);
    }

    /// Internal: advance state by one byte. May replay (e.g., when a number ends
    /// before its actual terminator byte). Returns true on success, false on reject.
    fn tryAdvance(self: *Grammar, byte: u8) std.mem.Allocator.Error!bool {
        var b: u8 = byte;
        // Bounded replay loop: consume at most a few replays per input byte.
        var iter: u32 = 0;
        while (iter < 8) : (iter += 1) {
            const r = try advanceOne(self, b);
            switch (r) {
                .consumed => return true,
                .replay => |nb| b = nb,
                .reject => return false,
            }
        }
        return false;
    }

    // ── Snapshot / restore ────────────────────────────────────────────────────
    // A snapshot is a deep copy of the stack. We use it for both speculative
    // byte trials (in allowedBytes) and for token-mask probes by external callers.

    pub const Snapshot = struct {
        stack: []Frame, // owned
        dead: bool,
    };

    pub fn snapshot(self: *const Grammar) std.mem.Allocator.Error!Snapshot {
        const copy = try self.gpa.dupe(Frame, self.stack.items);
        return .{ .stack = copy, .dead = self.dead };
    }

    pub fn restoreFrom(self: *Grammar, snap: Snapshot) std.mem.Allocator.Error!void {
        self.stack.clearRetainingCapacity();
        try self.stack.appendSlice(self.gpa, snap.stack);
        self.dead = snap.dead;
    }

    pub fn discardSnapshot(self: *const Grammar, snap: Snapshot) void {
        self.gpa.free(snap.stack);
    }

    fn top(self: *Grammar) *Frame {
        return &self.stack.items[self.stack.items.len - 1];
    }

    fn topConst(self: *const Grammar) *const Frame {
        return &self.stack.items[self.stack.items.len - 1];
    }
};

// ── Advance result ────────────────────────────────────────────────────────────

const StepResult = union(enum) {
    consumed,
    replay: u8,
    reject,
};

fn advanceOne(g: *Grammar, byte: u8) std.mem.Allocator.Error!StepResult {
    if (g.stack.items.len == 0) {
        // Past the root; only whitespace allowed.
        return if (isWs(byte)) .consumed else .reject;
    }

    if (g.stack.items.len == 1 and g.topConst().sub == .accepted) {
        return if (isWs(byte)) .consumed else .reject;
    }

    // Free whitespace handling for between-token positions.
    if (isWs(byte) and canAcceptFreeWhitespace(g.topConst())) {
        return .consumed;
    }

    if (g.stack.items.len > MAX_STACK) {
        g.dead = true;
        return .reject;
    }

    const frame = g.top();
    return switch (frame.sub) {
        .expect_value => try stepExpectValue(g, byte),
        .in_string => try stepString(g, byte),
        .in_number => try stepNumber(g, byte),
        .in_object => try stepObject(g, byte),
        .in_array => try stepArray(g, byte),
        .in_enum => try stepEnum(g, byte),
        .in_keyword => try stepKeyword(g, byte),
        .accepted => .reject,
    };
}

fn canAcceptFreeWhitespace(frame: *const Frame) bool {
    return switch (frame.sub) {
        .expect_value, .accepted => true,
        .in_string, .in_keyword, .in_enum => false,
        .in_number => false,
        .in_object => |o| switch (o.phase) {
            .expect_first_key_or_close, .expect_key, .after_key, .expect_value, .after_value => true,
            .in_key => false,
        },
        .in_array => |a| switch (a.phase) {
            .expect_first_value_or_close, .expect_value, .after_value => true,
        },
    };
}

// ── expect_value dispatch ─────────────────────────────────────────────────────

fn stepExpectValue(g: *Grammar, byte: u8) std.mem.Allocator.Error!StepResult {
    const schema = g.top().schema;
    return switch (schema.kind) {
        .any => try startAnyValue(g, byte),
        .null_ => try startKeyword(g, byte, "null"),
        .boolean => switch (byte) {
            't' => try startKeyword(g, byte, "true"),
            'f' => try startKeyword(g, byte, "false"),
            else => .reject,
        },
        .integer, .number => startNumber(g, byte),
        .string => startString(g, byte),
        .array => startArray(g, byte),
        .object => startObject(g, byte),
        .enum_ => startEnum(g, byte),
        .any_of => blk: {
            // Relaxed: accept any valid JSON value here.
            // Replace the schema with `any` and re-dispatch.
            g.top().schema = anyNode();
            break :blk try startAnyValue(g, byte);
        },
    };
}

fn startAnyValue(g: *Grammar, byte: u8) std.mem.Allocator.Error!StepResult {
    return switch (byte) {
        '"' => startString(g, byte),
        '{' => startObject(g, byte),
        '[' => startArray(g, byte),
        't' => try startKeyword(g, byte, "true"),
        'f' => try startKeyword(g, byte, "false"),
        'n' => try startKeyword(g, byte, "null"),
        '-', '0'...'9' => startNumber(g, byte),
        else => .reject,
    };
}

// ── keyword (true/false/null) ─────────────────────────────────────────────────

fn startKeyword(g: *Grammar, byte: u8, target: []const u8) std.mem.Allocator.Error!StepResult {
    if (target.len == 0 or target[0] != byte) return .reject;
    g.top().sub = .{ .in_keyword = .{ .target = target, .pos = 1 } };
    if (target.len == 1) try popAccepted(g);
    return .consumed;
}

fn stepKeyword(g: *Grammar, byte: u8) std.mem.Allocator.Error!StepResult {
    var k = g.top().sub.in_keyword;
    if (k.pos >= k.target.len) return .reject;
    if (k.target[k.pos] != byte) return .reject;
    k.pos += 1;
    g.top().sub = .{ .in_keyword = k };
    if (k.pos == k.target.len) try popAccepted(g);
    return .consumed;
}

// ── string ────────────────────────────────────────────────────────────────────

fn startString(g: *Grammar, byte: u8) StepResult {
    if (byte != '"') return .reject;
    g.top().sub = .{ .in_string = .{} };
    return .consumed;
}

fn stepString(g: *Grammar, byte: u8) std.mem.Allocator.Error!StepResult {
    var s = g.top().sub.in_string;
    const schema = g.top().schema;

    switch (s.phase) {
        .content => {
            if (byte == '"') {
                if (schema.str_min_len) |min| if (s.byte_count < min) return .reject;
                try popAccepted(g);
                return .consumed;
            }
            if (byte == '\\') {
                s.phase = .after_backslash;
                g.top().sub = .{ .in_string = s };
                return .consumed;
            }
            if (byte < 0x20) return .reject;
            if (schema.str_max_len) |max| if (s.byte_count >= max) return .reject;
            s.byte_count += 1;
            g.top().sub = .{ .in_string = s };
            return .consumed;
        },
        .after_backslash => {
            switch (byte) {
                '"', '\\', '/', 'b', 'f', 'n', 'r', 't' => {
                    if (schema.str_max_len) |max| if (s.byte_count >= max) return .reject;
                    s.byte_count += 1;
                    s.phase = .content;
                    g.top().sub = .{ .in_string = s };
                    return .consumed;
                },
                'u' => {
                    s.phase = .unicode_hex;
                    s.hex_remaining = 4;
                    g.top().sub = .{ .in_string = s };
                    return .consumed;
                },
                else => return .reject,
            }
        },
        .unicode_hex => {
            const ok = (byte >= '0' and byte <= '9') or (byte >= 'a' and byte <= 'f') or (byte >= 'A' and byte <= 'F');
            if (!ok) return .reject;
            s.hex_remaining -= 1;
            if (s.hex_remaining == 0) {
                if (schema.str_max_len) |max| if (s.byte_count >= max) return .reject;
                s.byte_count += 1;
                s.phase = .content;
            }
            g.top().sub = .{ .in_string = s };
            return .consumed;
        },
    }
}

// ── number ────────────────────────────────────────────────────────────────────

fn startNumber(g: *Grammar, byte: u8) StepResult {
    var phase: NumberPhase = undefined;
    switch (byte) {
        '-' => phase = .after_minus,
        '0' => phase = .leading_zero,
        '1'...'9' => phase = .int_digits,
        else => return .reject,
    }
    g.top().sub = .{ .in_number = .{ .phase = phase } };
    return .consumed;
}

fn stepNumber(g: *Grammar, byte: u8) std.mem.Allocator.Error!StepResult {
    var n = g.top().sub.in_number;
    const integer_only = g.top().schema.kind == .integer;

    switch (n.phase) {
        .after_minus => switch (byte) {
            '0' => n.phase = .leading_zero,
            '1'...'9' => n.phase = .int_digits,
            else => return .reject,
        },
        .leading_zero => switch (byte) {
            '.' => {
                if (integer_only) return try numberEndAndReplay(g, byte);
                n.phase = .after_dot;
            },
            'e', 'E' => {
                if (integer_only) return try numberEndAndReplay(g, byte);
                n.phase = .after_exp;
            },
            else => return try numberEndAndReplay(g, byte),
        },
        .int_digits => switch (byte) {
            '0'...'9' => {},
            '.' => {
                if (integer_only) return try numberEndAndReplay(g, byte);
                n.phase = .after_dot;
            },
            'e', 'E' => {
                if (integer_only) return try numberEndAndReplay(g, byte);
                n.phase = .after_exp;
            },
            else => return try numberEndAndReplay(g, byte),
        },
        .after_dot => switch (byte) {
            '0'...'9' => n.phase = .frac_digits,
            else => return .reject,
        },
        .frac_digits => switch (byte) {
            '0'...'9' => {},
            'e', 'E' => n.phase = .after_exp,
            else => return try numberEndAndReplay(g, byte),
        },
        .after_exp => switch (byte) {
            '+', '-' => n.phase = .after_exp_sign,
            '0'...'9' => n.phase = .exp_digits,
            else => return .reject,
        },
        .after_exp_sign => switch (byte) {
            '0'...'9' => n.phase = .exp_digits,
            else => return .reject,
        },
        .exp_digits => switch (byte) {
            '0'...'9' => {},
            else => return try numberEndAndReplay(g, byte),
        },
    }
    g.top().sub = .{ .in_number = n };
    return .consumed;
}

fn numberEndAndReplay(g: *Grammar, byte: u8) std.mem.Allocator.Error!StepResult {
    const phase = g.top().sub.in_number.phase;
    switch (phase) {
        .after_minus, .after_dot, .after_exp, .after_exp_sign => return .reject,
        else => {},
    }
    try popAccepted(g);
    return .{ .replay = byte };
}

// ── enum ──────────────────────────────────────────────────────────────────────

fn startEnum(g: *Grammar, byte: u8) StepResult {
    const schema = g.top().schema;
    var alive: u64 = 0;
    var i: u32 = 0;
    while (i < 64 and i < schema.enum_values.len) : (i += 1) {
        const ev = schema.enum_values[i].json;
        if (ev.len > 0 and ev[0] == byte) alive |= (@as(u64, 1) << @intCast(i));
    }
    if (alive == 0) return .reject;
    g.top().sub = .{ .in_enum = .{ .alive = alive, .byte_count = 1 } };
    maybeCompleteEnum(g) catch unreachable;
    return .consumed;
}

fn stepEnum(g: *Grammar, byte: u8) std.mem.Allocator.Error!StepResult {
    var e = g.top().sub.in_enum;
    const schema = g.top().schema;
    var new_alive: u64 = 0;
    var i: u32 = 0;
    while (i < 64 and i < schema.enum_values.len) : (i += 1) {
        if ((e.alive >> @intCast(i)) & 1 == 0) continue;
        const ev = schema.enum_values[i].json;
        if (ev.len > e.byte_count and ev[e.byte_count] == byte) {
            new_alive |= (@as(u64, 1) << @intCast(i));
        }
    }
    if (new_alive == 0) return .reject;
    e.alive = new_alive;
    e.byte_count += 1;
    g.top().sub = .{ .in_enum = e };
    try maybeCompleteEnum(g);
    return .consumed;
}

fn maybeCompleteEnum(g: *Grammar) std.mem.Allocator.Error!void {
    if (g.top().sub != .in_enum) return;
    const e = g.top().sub.in_enum;
    const schema = g.top().schema;
    var i: u32 = 0;
    while (i < 64 and i < schema.enum_values.len) : (i += 1) {
        if ((e.alive >> @intCast(i)) & 1 == 0) continue;
        if (schema.enum_values[i].json.len == e.byte_count) {
            try popAccepted(g);
            return;
        }
    }
}

// ── object ────────────────────────────────────────────────────────────────────

fn startObject(g: *Grammar, byte: u8) StepResult {
    if (byte != '{') return .reject;
    g.top().sub = .{ .in_object = .{ .phase = .expect_first_key_or_close } };
    return .consumed;
}

fn stepObject(g: *Grammar, byte: u8) std.mem.Allocator.Error!StepResult {
    var o = g.top().sub.in_object;
    const schema = g.top().schema;

    switch (o.phase) {
        .expect_first_key_or_close, .expect_key => |phase| {
            if (phase == .expect_first_key_or_close and byte == '}') {
                if (!requiredAllSeen(schema, o.seen)) return .reject;
                try popAccepted(g);
                return .consumed;
            }
            if (byte != '"') return .reject;
            o.phase = .in_key;
            o.key_len = 0;
            g.top().sub = .{ .in_object = o };
            return .consumed;
        },
        .in_key => {
            if (byte == '"') {
                const matched = matchPropertyName(schema, o.key_buf[0..o.key_len], o.seen);
                if (matched.matched_index == null and !schema.obj_additional) return .reject;
                o.current_prop = matched.matched_index;
                o.phase = .after_key;
                g.top().sub = .{ .in_object = o };
                return .consumed;
            }
            if (byte < 0x20) return .reject;
            // Any byte may extend the key if (a) it could match an unseen property's
            // next char OR (b) additionalProperties is allowed (free-form key).
            const matches_property = keyByteMatchesProperty(schema, o.key_buf[0..o.key_len], o.seen, byte);
            if (!matches_property and !schema.obj_additional) return .reject;
            if (o.key_len >= o.key_buf.len) return .reject;
            o.key_buf[o.key_len] = byte;
            o.key_len += 1;
            g.top().sub = .{ .in_object = o };
            return .consumed;
        },
        .after_key => {
            if (byte != ':') return .reject;
            o.phase = .expect_value;
            g.top().sub = .{ .in_object = o };
            return .consumed;
        },
        .expect_value => {
            const child_schema = if (o.current_prop) |idx|
                schema.obj_properties[idx].schema
            else
                anyNode();
            if (o.current_prop) |idx| {
                o.seen |= (@as(u64, 1) << @intCast(idx));
            }
            o.phase = .after_value;
            g.top().sub = .{ .in_object = o };
            try g.stack.append(g.gpa, .{ .schema = child_schema, .sub = .expect_value });
            return .{ .replay = byte };
        },
        .after_value => {
            if (byte == ',') {
                o.phase = .expect_key;
                g.top().sub = .{ .in_object = o };
                return .consumed;
            }
            if (byte == '}') {
                if (!requiredAllSeen(schema, o.seen)) return .reject;
                try popAccepted(g);
                return .consumed;
            }
            return .reject;
        },
    }
}

const PropertyMatchResult = struct {
    matched_index: ?u32,
};

fn matchPropertyName(schema: *const Node, name: []const u8, seen: u64) PropertyMatchResult {
    for (schema.obj_properties, 0..) |p, i| {
        if (i >= 64) break;
        if (((seen >> @intCast(i)) & 1) != 0) continue;
        if (std.mem.eql(u8, p.name, name)) return .{ .matched_index = @intCast(i) };
    }
    return .{ .matched_index = null };
}

fn keyByteMatchesProperty(schema: *const Node, prefix: []const u8, seen: u64, byte: u8) bool {
    for (schema.obj_properties, 0..) |p, i| {
        if (i >= 64) break;
        if (((seen >> @intCast(i)) & 1) != 0) continue;
        if (p.name.len <= prefix.len) continue;
        if (!std.mem.startsWith(u8, p.name, prefix)) continue;
        if (p.name[prefix.len] == byte) return true;
    }
    return false;
}

fn requiredAllSeen(schema: *const Node, seen: u64) bool {
    for (schema.obj_properties, 0..) |p, i| {
        if (i >= 64) break;
        if (!p.required) continue;
        if (((seen >> @intCast(i)) & 1) == 0) return false;
    }
    return true;
}

// ── array ─────────────────────────────────────────────────────────────────────

fn startArray(g: *Grammar, byte: u8) StepResult {
    if (byte != '[') return .reject;
    g.top().sub = .{ .in_array = .{ .phase = .expect_first_value_or_close } };
    return .consumed;
}

fn stepArray(g: *Grammar, byte: u8) std.mem.Allocator.Error!StepResult {
    var a = g.top().sub.in_array;
    const schema = g.top().schema;

    switch (a.phase) {
        .expect_first_value_or_close => {
            if (byte == ']') {
                if (schema.arr_min_items) |min| if (a.count < min) return .reject;
                try popAccepted(g);
                return .consumed;
            }
            if (schema.arr_max_items) |max| if (a.count >= max) return .reject;
            a.phase = .after_value;
            a.count += 1;
            g.top().sub = .{ .in_array = a };
            const child = schema.arr_items orelse anyNode();
            try g.stack.append(g.gpa, .{ .schema = child, .sub = .expect_value });
            return .{ .replay = byte };
        },
        .expect_value => {
            if (schema.arr_max_items) |max| if (a.count >= max) return .reject;
            a.phase = .after_value;
            a.count += 1;
            g.top().sub = .{ .in_array = a };
            const child = schema.arr_items orelse anyNode();
            try g.stack.append(g.gpa, .{ .schema = child, .sub = .expect_value });
            return .{ .replay = byte };
        },
        .after_value => {
            if (byte == ',') {
                a.phase = .expect_value;
                g.top().sub = .{ .in_array = a };
                return .consumed;
            }
            if (byte == ']') {
                if (schema.arr_min_items) |min| if (a.count < min) return .reject;
                try popAccepted(g);
                return .consumed;
            }
            return .reject;
        },
    }
}

// ── helpers ───────────────────────────────────────────────────────────────────

fn popAccepted(g: *Grammar) std.mem.Allocator.Error!void {
    g.top().sub = .accepted;
    if (g.stack.items.len <= 1) return; // root accepted, leave on stack
    _ = g.stack.pop();
    // Parent state is already in its post-value phase (e.g., object.after_value);
    // it'll naturally handle the next byte.
}

/// A "no-constraints" schema node. Used when a sub-tree is unconstrained:
///   * inside `anyOf` (relaxed branch — see header doc)
///   * for objects with no `properties`
///   * for arrays with no `items`
///   * for additional-property values when `additionalProperties: true`
///
/// `obj_additional = true` so any property name is accepted; `arr_items = null`
/// so any items kind is accepted (recurses to `any` automatically).
fn anyNode() *const Node {
    const S = struct {
        const node: Node = .{ .kind = .any, .obj_additional = true };
    };
    return &S.node;
}

// ── Tests ─────────────────────────────────────────────────────────────────────

const testing = std.testing;

fn parseSchema(gpa: std.mem.Allocator, src: []const u8) !Schema {
    const v = try std.json.parseFromSlice(std.json.Value, gpa, src, .{});
    defer v.deinit();
    return schema_mod.parse(gpa, v.value);
}

fn feed(g: *Grammar, input: []const u8) !void {
    for (input) |c| {
        const ok = try g.acceptByte(c);
        if (!ok) return error.GrammarRejected;
    }
}

test "grammar accepts simple object matching schema" {
    var schema = try parseSchema(testing.allocator,
        \\{"type":"object","properties":{"name":{"type":"string"}},"required":["name"]}
    );
    defer schema.deinit();

    var g = try Grammar.init(testing.allocator, &schema);
    defer g.deinit();

    try feed(&g, "{\"name\":\"alice\"}");
    try testing.expect(g.isComplete());
}

test "grammar rejects missing required" {
    var schema = try parseSchema(testing.allocator,
        \\{"type":"object","properties":{"name":{"type":"string"}},"required":["name"]}
    );
    defer schema.deinit();

    var g = try Grammar.init(testing.allocator, &schema);
    defer g.deinit();

    // Empty object is missing required `name`
    try testing.expectError(error.GrammarRejected, feed(&g, "{}"));
}

test "grammar rejects unknown property when additional=false" {
    var schema = try parseSchema(testing.allocator,
        \\{"type":"object","properties":{"name":{"type":"string"}},"additionalProperties":false}
    );
    defer schema.deinit();

    var g = try Grammar.init(testing.allocator, &schema);
    defer g.deinit();

    // After typing "age" through the prefix-match phase, `e"` tries to close the key — but `age`
    // isn't a known property and additional is false, so it should reject when matching.
    // (Our grammar rejects sooner — when the second char doesn't extend `name`.)
    try testing.expectError(error.GrammarRejected, feed(&g, "{\"a"));
}

test "grammar enforces enum values" {
    var schema = try parseSchema(testing.allocator,
        \\{"type":"object","properties":{"role":{"enum":["admin","user"]}},"required":["role"]}
    );
    defer schema.deinit();

    {
        var g = try Grammar.init(testing.allocator, &schema);
        defer g.deinit();
        try feed(&g, "{\"role\":\"admin\"}");
        try testing.expect(g.isComplete());
    }
    {
        var g = try Grammar.init(testing.allocator, &schema);
        defer g.deinit();
        try testing.expectError(error.GrammarRejected, feed(&g, "{\"role\":\"guest"));
    }
}

test "grammar accepts array of integers" {
    var schema = try parseSchema(testing.allocator,
        \\{"type":"array","items":{"type":"integer"}}
    );
    defer schema.deinit();

    var g = try Grammar.init(testing.allocator, &schema);
    defer g.deinit();

    try feed(&g, "[1, 2, 30, -7]");
    try testing.expect(g.isComplete());
}

test "grammar rejects float in integer array" {
    var schema = try parseSchema(testing.allocator,
        \\{"type":"array","items":{"type":"integer"}}
    );
    defer schema.deinit();

    var g = try Grammar.init(testing.allocator, &schema);
    defer g.deinit();

    try testing.expectError(error.GrammarRejected, feed(&g, "[1.5"));
}

test "grammar handles string escapes and unicode" {
    var schema = try parseSchema(testing.allocator, "{\"type\":\"string\"}");
    defer schema.deinit();

    var g = try Grammar.init(testing.allocator, &schema);
    defer g.deinit();
    try feed(&g, "\"hi\\n\\u00e9!\"");
    try testing.expect(g.isComplete());
}

test "grammar rejects unescaped control char in string" {
    var schema = try parseSchema(testing.allocator, "{\"type\":\"string\"}");
    defer schema.deinit();

    var g = try Grammar.init(testing.allocator, &schema);
    defer g.deinit();
    try testing.expectError(error.GrammarRejected, feed(&g, "\"hi\nthere\""));
}

test "grammar respects minLength and maxLength" {
    var schema = try parseSchema(testing.allocator,
        \\{"type":"string","minLength":2,"maxLength":3}
    );
    defer schema.deinit();
    {
        var g = try Grammar.init(testing.allocator, &schema);
        defer g.deinit();
        try testing.expectError(error.GrammarRejected, feed(&g, "\"a\""));
    }
    {
        var g = try Grammar.init(testing.allocator, &schema);
        defer g.deinit();
        try feed(&g, "\"abc\"");
        try testing.expect(g.isComplete());
    }
    {
        var g = try Grammar.init(testing.allocator, &schema);
        defer g.deinit();
        try testing.expectError(error.GrammarRejected, feed(&g, "\"abcd"));
    }
}

test "grammar respects minItems and maxItems" {
    var schema = try parseSchema(testing.allocator,
        \\{"type":"array","items":{"type":"integer"},"minItems":2,"maxItems":3}
    );
    defer schema.deinit();
    {
        var g = try Grammar.init(testing.allocator, &schema);
        defer g.deinit();
        try testing.expectError(error.GrammarRejected, feed(&g, "[1]"));
    }
    {
        var g = try Grammar.init(testing.allocator, &schema);
        defer g.deinit();
        try feed(&g, "[1,2,3]");
        try testing.expect(g.isComplete());
    }
    {
        var g = try Grammar.init(testing.allocator, &schema);
        defer g.deinit();
        try testing.expectError(error.GrammarRejected, feed(&g, "[1,2,3,4"));
    }
}

test "grammar accepts whitespace between tokens" {
    var schema = try parseSchema(testing.allocator,
        \\{"type":"object","properties":{"x":{"type":"integer"}},"required":["x"]}
    );
    defer schema.deinit();

    var g = try Grammar.init(testing.allocator, &schema);
    defer g.deinit();
    try feed(&g, "  {\n  \"x\" : 42\n}\n");
    try testing.expect(g.isComplete());
}

test "grammar relaxes anyOf to any-json" {
    var schema = try parseSchema(testing.allocator,
        \\{"anyOf":[{"type":"string"},{"type":"object"}]}
    );
    defer schema.deinit();

    var g = try Grammar.init(testing.allocator, &schema);
    defer g.deinit();
    try feed(&g, "[1,2,3]"); // an array — also accepted under "any"
    try testing.expect(g.isComplete());
}

test "allowedBytes initial returns object opener" {
    var schema = try parseSchema(testing.allocator,
        \\{"type":"object","properties":{"name":{"type":"string"}}}
    );
    defer schema.deinit();

    var g = try Grammar.init(testing.allocator, &schema);
    defer g.deinit();
    const mask = try g.allowedBytes();
    try testing.expect(mask.contains('{'));
    try testing.expect(mask.contains(' ')); // whitespace
    try testing.expect(!mask.contains('['));
    try testing.expect(!mask.contains('"'));
}

test "allowedBytes after { restricts to property prefixes" {
    var schema = try parseSchema(testing.allocator,
        \\{"type":"object","properties":{"name":{"type":"string"},"age":{"type":"integer"}},"additionalProperties":false}
    );
    defer schema.deinit();

    var g = try Grammar.init(testing.allocator, &schema);
    defer g.deinit();
    _ = try g.acceptByte('{');
    _ = try g.acceptByte('"');
    const mask = try g.allowedBytes();
    try testing.expect(mask.contains('n')); // start of "name"
    try testing.expect(mask.contains('a')); // start of "age"
    try testing.expect(!mask.contains('z'));
}

test "snapshot/restore roundtrip" {
    var schema = try parseSchema(testing.allocator,
        \\{"type":"object","properties":{"a":{"type":"integer"}},"required":["a"]}
    );
    defer schema.deinit();

    var g = try Grammar.init(testing.allocator, &schema);
    defer g.deinit();

    _ = try g.acceptByte('{');
    const snap = try g.snapshot();
    defer g.discardSnapshot(snap);

    _ = try g.acceptByte('"');
    _ = try g.acceptByte('a');
    _ = try g.acceptByte('"');
    try g.restoreFrom(snap);

    // After restore we're back at expect_first_key_or_close. `"` is allowed
    // (start the required key); `}` is NOT allowed because required `a` is unseen.
    const mask = try g.allowedBytes();
    try testing.expect(mask.contains('"'));
    try testing.expect(!mask.contains('}'));
}

test "cruise-app-style nested schema parses sample response" {
    const src =
        \\{
        \\  "type":"object",
        \\  "properties":{
        \\    "blocks":{
        \\      "type":"array",
        \\      "items":{
        \\        "anyOf":[
        \\          {"type":"object","properties":{"type":{"enum":["text"]},"content":{"type":"string"}},"required":["type","content"],"additionalProperties":false}
        \\        ]
        \\      }
        \\    }
        \\  },
        \\  "required":["blocks"],
        \\  "additionalProperties":false
        \\}
    ;
    var schema = try parseSchema(testing.allocator, src);
    defer schema.deinit();

    var g = try Grammar.init(testing.allocator, &schema);
    defer g.deinit();

    try feed(&g, "{\"blocks\":[{\"type\":\"text\",\"content\":\"hi\"}]}");
    try testing.expect(g.isComplete());
}
