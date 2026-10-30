//! Minimal regex engine for JSON Schema `pattern` constraints.
//!
//! Supports the subset most commonly seen in JSON Schemas:
//!   * literal characters (escaped with `\` for metachars)
//!   * `.` — any byte except newline
//!   * `[abc]` `[^abc]` `[a-z]` — character classes (with negation, ranges, escapes)
//!   * `\d` `\D` `\w` `\W` `\s` `\S` — built-in classes
//!   * `\n` `\t` `\r` `\\` `\/` `\"` — escape sequences
//!   * `*` `+` `?` `{n}` `{n,m}` — quantifiers (greedy; greedy/lazy doesn't matter for matching membership)
//!   * `|` — alternation
//!   * `(...)` — non-capturing groups (we never capture)
//!   * `^` `$` — anchors (we always full-match the whole string, so these are no-ops)
//!
//! NOT supported (errors out): backreferences, lookaround, named groups, Unicode property classes.
//!
//! The compiled form is a Thompson NFA. Matching is performed via simultaneous-states
//! simulation (no backtracking) so `step()` is O(states × bytes) per byte.
//!
//! Lifetime: the NFA borrows the arena passed to `compile`. Free the arena to free the NFA.

const std = @import("std");

pub const Error = error{
    InvalidPattern,
    UnsupportedConstruct,
    OutOfMemory,
};

const StateIndex = u32;

const Transition = union(enum) {
    /// Matches a specific byte.
    byte: u8,
    /// Matches any byte where bitset[byte/8] & (1 << byte%8) != 0.
    class: ByteClass,
    /// Matches without consuming input (epsilon).
    epsilon,
};

const ByteClass = struct {
    /// 256-bit set; one bit per byte value.
    bits: [32]u8,

    fn empty() ByteClass {
        return .{ .bits = @splat(0) };
    }

    pub fn contains(self: *const ByteClass, b: u8) bool {
        return (self.bits[b >> 3] >> @intCast(b & 7)) & 1 == 1;
    }

    fn add(self: *ByteClass, b: u8) void {
        self.bits[b >> 3] |= @as(u8, 1) << @intCast(b & 7);
    }

    fn addRange(self: *ByteClass, lo: u8, hi: u8) void {
        var b: u16 = lo;
        while (b <= hi) : (b += 1) self.add(@intCast(b));
    }

    fn invert(self: *ByteClass) void {
        for (&self.bits) |*byte| byte.* = ~byte.*;
    }
};

const State = struct {
    /// Outgoing transitions. Index 0 is "primary"; additional indices are alternatives.
    out: []Edge,
    /// True if this state is an accepting state.
    accept: bool,
};

const Edge = struct {
    transition: Transition,
    /// Target state index.
    target: StateIndex,
};

pub const Nfa = struct {
    states: []State,
    start: StateIndex,

    /// Initial state set (after applying epsilon-closure).
    pub fn startStates(self: *const Nfa, arena: std.mem.Allocator) Error!StateSet {
        var ss: StateSet = try .initEmpty(arena, self.states.len);
        try addClosure(self, &ss, arena, self.start);
        return ss;
    }

    pub fn isAccepting(self: *const Nfa, states: *const StateSet) bool {
        var it = states.iterator();
        while (it.next()) |s| {
            if (self.states[s].accept) return true;
        }
        return false;
    }

    /// Step the state set by one input byte. Returns a new (allocated) state set.
    /// Caller frees with `set.deinit(arena)`.
    pub fn step(self: *const Nfa, current: *const StateSet, byte: u8, arena: std.mem.Allocator) Error!StateSet {
        var next: StateSet = try .initEmpty(arena, self.states.len);
        var it = current.iterator();
        while (it.next()) |s_idx| {
            for (self.states[s_idx].out) |edge| {
                const matched = switch (edge.transition) {
                    .byte => |b| b == byte,
                    .class => |c| c.contains(byte),
                    .epsilon => false,
                };
                if (matched) try addClosure(self, &next, arena, edge.target);
            }
        }
        return next;
    }
};

fn addClosure(nfa: *const Nfa, set: *StateSet, arena: std.mem.Allocator, state: StateIndex) Error!void {
    if (set.contains(state)) return;
    try set.add(arena, state);
    for (nfa.states[state].out) |edge| {
        if (edge.transition == .epsilon) {
            try addClosure(nfa, set, arena, edge.target);
        }
    }
}

/// A set of NFA states. Backed by a dynamic bit set.
pub const StateSet = struct {
    bits: std.DynamicBitSetUnmanaged,

    pub fn initEmpty(arena: std.mem.Allocator, capacity: usize) Error!StateSet {
        return .{ .bits = try .initEmpty(arena, capacity) };
    }

    pub fn deinit(self: *StateSet, arena: std.mem.Allocator) void {
        self.bits.deinit(arena);
    }

    pub fn contains(self: *const StateSet, idx: StateIndex) bool {
        return self.bits.isSet(idx);
    }

    pub fn add(self: *StateSet, arena: std.mem.Allocator, idx: StateIndex) Error!void {
        _ = arena;
        self.bits.set(idx);
    }

    pub fn count(self: *const StateSet) usize {
        return self.bits.count();
    }

    pub fn isEmpty(self: *const StateSet) bool {
        return self.count() == 0;
    }

    pub fn iterator(self: *const StateSet) std.DynamicBitSetUnmanaged.Iterator(.{}) {
        return self.bits.iterator(.{});
    }
};

// ── Compiler ──────────────────────────────────────────────────────────────────

const Frag = struct {
    start: StateIndex,
    accept: StateIndex,
};

const Builder = struct {
    arena: std.mem.Allocator,
    states: std.ArrayListUnmanaged(State) = .empty,

    fn newState(self: *Builder) Error!StateIndex {
        try self.states.append(self.arena, .{ .out = &.{}, .accept = false });
        return @intCast(self.states.items.len - 1);
    }

    fn addEdge(self: *Builder, from: StateIndex, t: Transition, to: StateIndex) Error!void {
        const old = self.states.items[from].out;
        var new = try self.arena.alloc(Edge, old.len + 1);
        @memcpy(new[0..old.len], old);
        new[old.len] = .{ .transition = t, .target = to };
        self.states.items[from].out = new;
    }

    fn lit(self: *Builder, b: u8) Error!Frag {
        const s0 = try self.newState();
        const s1 = try self.newState();
        try self.addEdge(s0, .{ .byte = b }, s1);
        return .{ .start = s0, .accept = s1 };
    }

    fn class(self: *Builder, c: ByteClass) Error!Frag {
        const s0 = try self.newState();
        const s1 = try self.newState();
        try self.addEdge(s0, .{ .class = c }, s1);
        return .{ .start = s0, .accept = s1 };
    }

    fn concat(self: *Builder, a: Frag, b: Frag) Error!Frag {
        try self.addEdge(a.accept, .epsilon, b.start);
        return .{ .start = a.start, .accept = b.accept };
    }

    fn alt(self: *Builder, a: Frag, b: Frag) Error!Frag {
        const s = try self.newState();
        const e = try self.newState();
        try self.addEdge(s, .epsilon, a.start);
        try self.addEdge(s, .epsilon, b.start);
        try self.addEdge(a.accept, .epsilon, e);
        try self.addEdge(b.accept, .epsilon, e);
        return .{ .start = s, .accept = e };
    }

    fn star(self: *Builder, a: Frag) Error!Frag {
        const s = try self.newState();
        const e = try self.newState();
        try self.addEdge(s, .epsilon, a.start);
        try self.addEdge(s, .epsilon, e);
        try self.addEdge(a.accept, .epsilon, a.start);
        try self.addEdge(a.accept, .epsilon, e);
        return .{ .start = s, .accept = e };
    }

    fn plus(self: *Builder, a: Frag) Error!Frag {
        const s = try self.newState();
        const e = try self.newState();
        try self.addEdge(s, .epsilon, a.start);
        try self.addEdge(a.accept, .epsilon, a.start);
        try self.addEdge(a.accept, .epsilon, e);
        return .{ .start = s, .accept = e };
    }

    fn opt(self: *Builder, a: Frag) Error!Frag {
        const s = try self.newState();
        const e = try self.newState();
        try self.addEdge(s, .epsilon, a.start);
        try self.addEdge(s, .epsilon, e);
        try self.addEdge(a.accept, .epsilon, e);
        return .{ .start = s, .accept = e };
    }

    fn empty(self: *Builder) Error!Frag {
        const s = try self.newState();
        return .{ .start = s, .accept = s };
    }
};

const Parser = struct {
    src: []const u8,
    pos: usize = 0,
    b: *Builder,

    fn peek(self: *Parser) ?u8 {
        if (self.pos >= self.src.len) return null;
        return self.src[self.pos];
    }

    fn next(self: *Parser) ?u8 {
        if (self.pos >= self.src.len) return null;
        const c = self.src[self.pos];
        self.pos += 1;
        return c;
    }

    fn parseExpr(self: *Parser) Error!Frag {
        var left = try self.parseConcat();
        while (self.peek()) |c| {
            if (c != '|') break;
            _ = self.next();
            const right = try self.parseConcat();
            left = try self.b.alt(left, right);
        }
        return left;
    }

    fn parseConcat(self: *Parser) Error!Frag {
        var result: ?Frag = null;
        while (self.peek()) |c| {
            if (c == '|' or c == ')') break;
            const atom = try self.parseAtomQuantified();
            result = if (result) |r| try self.b.concat(r, atom) else atom;
        }
        return result orelse try self.b.empty();
    }

    fn parseAtomQuantified(self: *Parser) Error!Frag {
        const atom_start = self.pos;
        const atom = try self.parseAtom();
        const atom_end = self.pos;
        if (self.peek()) |c| {
            switch (c) {
                '*' => {
                    _ = self.next();
                    self.consumeLazyOrPossessive();
                    return try self.b.star(atom);
                },
                '+' => {
                    _ = self.next();
                    self.consumeLazyOrPossessive();
                    return try self.b.plus(atom);
                },
                '?' => {
                    _ = self.next();
                    self.consumeLazyOrPossessive();
                    return try self.b.opt(atom);
                },
                '{' => return try self.parseCounted(atom, atom_start, atom_end),
                else => {},
            }
        }
        return atom;
    }

    fn consumeLazyOrPossessive(self: *Parser) void {
        // Greedy/lazy makes no difference for membership; just accept and ignore.
        if (self.peek()) |c| if (c == '?' or c == '+') {
            _ = self.next();
        };
    }

    /// Re-parse the atom span to produce a fresh, independent NFA fragment.
    /// We use a sub-parser over the saved span — atoms are self-contained
    /// (groups balance their own parens), so this is safe.
    fn cloneAtom(self: *Parser, atom_start: usize, atom_end: usize) Error!Frag {
        var sub: Parser = .{ .src = self.src[atom_start..atom_end], .pos = 0, .b = self.b };
        return try sub.parseAtom();
    }

    fn parseCounted(self: *Parser, atom: Frag, atom_start: usize, atom_end: usize) Error!Frag {
        _ = self.next(); // consume '{'
        const n_start = self.pos;
        while (self.peek()) |c| {
            if (c >= '0' and c <= '9') _ = self.next() else break;
        }
        const n_str = self.src[n_start..self.pos];
        if (n_str.len == 0) return error.InvalidPattern;
        const n = std.fmt.parseInt(u32, n_str, 10) catch return error.InvalidPattern;

        var m: ?u32 = n;
        if (self.peek() == @as(?u8, ',')) {
            _ = self.next();
            const m_start = self.pos;
            while (self.peek()) |cc| {
                if (cc >= '0' and cc <= '9') _ = self.next() else break;
            }
            const m_str = self.src[m_start..self.pos];
            m = if (m_str.len == 0) null else (std.fmt.parseInt(u32, m_str, 10) catch return error.InvalidPattern);
        }
        if (self.next() != '}') return error.InvalidPattern;
        self.consumeLazyOrPossessive();

        // Build n required copies (the first one is the already-parsed atom).
        var result = if (n == 0) try self.b.empty() else atom;
        var i: u32 = 1;
        while (i < n) : (i += 1) {
            const cloned = try self.cloneAtom(atom_start, atom_end);
            result = try self.b.concat(result, cloned);
        }

        if (m) |max| {
            if (max < n) return error.InvalidPattern;
            const optional_count = max - n;
            var j: u32 = 0;
            while (j < optional_count) : (j += 1) {
                const cloned = try self.cloneAtom(atom_start, atom_end);
                const opt_frag = try self.b.opt(cloned);
                result = try self.b.concat(result, opt_frag);
            }
        } else {
            const tail = try self.cloneAtom(atom_start, atom_end);
            const star_frag = try self.b.star(tail);
            result = try self.b.concat(result, star_frag);
        }
        return result;
    }

    fn parseAtom(self: *Parser) Error!Frag {
        const c = self.next() orelse return error.InvalidPattern;
        switch (c) {
            '(' => {
                // Optional non-capturing prefix `?:` is allowed and ignored.
                if (self.peek() == @as(?u8, '?')) {
                    _ = self.next();
                    if (self.next() != ':') return error.UnsupportedConstruct;
                }
                const inner = try self.parseExpr();
                if (self.next() != ')') return error.InvalidPattern;
                return inner;
            },
            '[' => return try self.parseClass(),
            '.' => {
                var cls = ByteClass.empty();
                cls.invert();
                cls.bits['\n' >> 3] &= ~(@as(u8, 1) << @intCast('\n' & 7));
                return try self.b.class(cls);
            },
            '^', '$' => {
                // Anchors — treat as no-op since we always full-match.
                return try self.b.empty();
            },
            '\\' => {
                return try self.b.class(try parseEscape(self));
            },
            ')', '|', '*', '+', '?', '{', '}' => return error.InvalidPattern,
            else => {
                return try self.b.lit(c);
            },
        }
    }

    fn parseClass(self: *Parser) Error!Frag {
        var cls = ByteClass.empty();
        var negate = false;
        if (self.peek() == @as(?u8, '^')) {
            _ = self.next();
            negate = true;
        }
        var first = true;
        while (self.peek()) |c| {
            if (c == ']' and !first) break;
            first = false;
            const lo = try self.parseClassChar();
            // Range?
            if (self.peek() == @as(?u8, '-')) {
                // Look ahead — `-` followed by `]` is a literal dash
                if (self.pos + 1 < self.src.len and self.src[self.pos + 1] != ']') {
                    _ = self.next(); // consume '-'
                    const hi = try self.parseClassChar();
                    if (lo.kind == .single and hi.kind == .single) {
                        const a = lo.byte;
                        const b = hi.byte;
                        if (b < a) return error.InvalidPattern;
                        cls.addRange(a, b);
                        continue;
                    }
                    return error.InvalidPattern;
                }
            }
            switch (lo.kind) {
                .single => cls.add(lo.byte),
                .class => {
                    for (lo.class.bits, 0..) |bit_byte, idx| {
                        cls.bits[idx] |= bit_byte;
                    }
                },
            }
        }
        if (self.next() != ']') return error.InvalidPattern;
        if (negate) cls.invert();
        return try self.b.class(cls);
    }

    const ClassChar = struct {
        kind: enum { single, class },
        byte: u8 = 0,
        class: ByteClass = ByteClass.empty(),
    };

    fn parseClassChar(self: *Parser) Error!ClassChar {
        const c = self.next() orelse return error.InvalidPattern;
        if (c == '\\') {
            // Escape inside class
            const esc = self.next() orelse return error.InvalidPattern;
            return switch (esc) {
                'd', 'D', 'w', 'W', 's', 'S' => blk: {
                    var cls = ByteClass.empty();
                    fillBuiltinClass(esc, &cls);
                    break :blk .{ .kind = .class, .class = cls };
                },
                'n' => .{ .kind = .single, .byte = '\n' },
                't' => .{ .kind = .single, .byte = '\t' },
                'r' => .{ .kind = .single, .byte = '\r' },
                else => .{ .kind = .single, .byte = esc },
            };
        }
        return .{ .kind = .single, .byte = c };
    }
};

fn parseEscape(p: *Parser) Error!ByteClass {
    const c = p.next() orelse return error.InvalidPattern;
    var cls = ByteClass.empty();
    switch (c) {
        'd', 'D', 'w', 'W', 's', 'S' => fillBuiltinClass(c, &cls),
        'n' => cls.add('\n'),
        't' => cls.add('\t'),
        'r' => cls.add('\r'),
        else => cls.add(c),
    }
    return cls;
}

fn fillBuiltinClass(c: u8, cls: *ByteClass) void {
    switch (c) {
        'd' => cls.addRange('0', '9'),
        'D' => {
            cls.addRange('0', '9');
            cls.invert();
        },
        'w' => {
            cls.addRange('a', 'z');
            cls.addRange('A', 'Z');
            cls.addRange('0', '9');
            cls.add('_');
        },
        'W' => {
            cls.addRange('a', 'z');
            cls.addRange('A', 'Z');
            cls.addRange('0', '9');
            cls.add('_');
            cls.invert();
        },
        's' => {
            cls.add(' ');
            cls.add('\t');
            cls.add('\n');
            cls.add('\r');
            cls.add(0x0C);
            cls.add(0x0B);
        },
        'S' => {
            cls.add(' ');
            cls.add('\t');
            cls.add('\n');
            cls.add('\r');
            cls.add(0x0C);
            cls.add(0x0B);
            cls.invert();
        },
        else => unreachable,
    }
}

/// Compile a regex pattern. Returns an NFA owned by `arena`.
pub fn compile(arena: std.mem.Allocator, pattern: []const u8) Error!*const Nfa {
    var b: Builder = .{ .arena = arena };
    var p: Parser = .{ .src = pattern, .b = &b };
    const frag = try p.parseExpr();
    if (p.peek() != null) return error.InvalidPattern;
    b.states.items[frag.accept].accept = true;

    const nfa = try arena.create(Nfa);
    nfa.* = .{
        .states = try b.states.toOwnedSlice(arena),
        .start = frag.start,
    };
    return nfa;
}

/// Convenience: full-match a complete string against a pattern.
pub fn match(arena: std.mem.Allocator, nfa: *const Nfa, input: []const u8) Error!bool {
    var states = try nfa.startStates(arena);
    defer states.deinit(arena);
    for (input) |c| {
        const next = try nfa.step(&states, c, arena);
        states.deinit(arena);
        states = next;
        if (states.isEmpty()) return false;
    }
    return nfa.isAccepting(&states);
}

// ── Tests ─────────────────────────────────────────────────────────────────────

const testing = std.testing;

fn matchTest(pattern: []const u8, input: []const u8) !bool {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const nfa = try compile(a, pattern);
    return try match(a, nfa, input);
}

test "literal match" {
    try testing.expect(try matchTest("hello", "hello"));
    try testing.expect(!try matchTest("hello", "world"));
    try testing.expect(!try matchTest("hello", "hell"));
    try testing.expect(!try matchTest("hello", "hello!"));
}

test "alternation" {
    try testing.expect(try matchTest("cat|dog", "cat"));
    try testing.expect(try matchTest("cat|dog", "dog"));
    try testing.expect(!try matchTest("cat|dog", "fox"));
}

test "star quantifier" {
    try testing.expect(try matchTest("a*", ""));
    try testing.expect(try matchTest("a*", "aaa"));
    try testing.expect(!try matchTest("a*", "ab"));
}

test "plus quantifier" {
    try testing.expect(!try matchTest("a+", ""));
    try testing.expect(try matchTest("a+", "a"));
    try testing.expect(try matchTest("a+", "aaaa"));
}

test "optional quantifier" {
    try testing.expect(try matchTest("ab?c", "ac"));
    try testing.expect(try matchTest("ab?c", "abc"));
    try testing.expect(!try matchTest("ab?c", "abbc"));
}

test "char classes" {
    try testing.expect(try matchTest("[abc]", "a"));
    try testing.expect(try matchTest("[abc]", "b"));
    try testing.expect(!try matchTest("[abc]", "d"));
    try testing.expect(try matchTest("[a-z]+", "hello"));
    try testing.expect(!try matchTest("[a-z]+", "Hello"));
}

test "negated char class" {
    try testing.expect(try matchTest("[^abc]", "d"));
    try testing.expect(!try matchTest("[^abc]", "a"));
}

test "builtin classes" {
    try testing.expect(try matchTest("\\d+", "123"));
    try testing.expect(!try matchTest("\\d+", "abc"));
    try testing.expect(try matchTest("\\w+", "hello_world1"));
    try testing.expect(try matchTest("\\s+", "  \t"));
}

test "anchors are no-ops" {
    try testing.expect(try matchTest("^abc$", "abc"));
}

test "groups" {
    try testing.expect(try matchTest("(ab)+", "ababab"));
    try testing.expect(!try matchTest("(ab)+", "aba"));
    try testing.expect(try matchTest("(?:cat|dog)s?", "cats"));
}

test "dot doesn't match newline" {
    try testing.expect(try matchTest("a.b", "axb"));
    try testing.expect(!try matchTest("a.b", "a\nb"));
}

test "escape sequences" {
    try testing.expect(try matchTest("\\.", "."));
    try testing.expect(!try matchTest("\\.", "x"));
    try testing.expect(try matchTest("\\\\", "\\"));
}

test "counted quantifier exact" {
    try testing.expect(try matchTest("a{3}", "aaa"));
    try testing.expect(!try matchTest("a{3}", "aa"));
    try testing.expect(!try matchTest("a{3}", "aaaa"));
}

test "counted quantifier range" {
    try testing.expect(!try matchTest("a{2,4}", "a"));
    try testing.expect(try matchTest("a{2,4}", "aa"));
    try testing.expect(try matchTest("a{2,4}", "aaa"));
    try testing.expect(try matchTest("a{2,4}", "aaaa"));
    try testing.expect(!try matchTest("a{2,4}", "aaaaa"));
}

test "counted quantifier unbounded" {
    try testing.expect(try matchTest("a{2,}", "aa"));
    try testing.expect(try matchTest("a{2,}", "aaaaaaaa"));
    try testing.expect(!try matchTest("a{2,}", "a"));
}

test "common json schema patterns" {
    // Date-like
    try testing.expect(try matchTest("\\d{4}-\\d{2}-\\d{2}", "2025-04-24"));
    try testing.expect(!try matchTest("\\d{4}-\\d{2}-\\d{2}", "25-04-24"));
    // Email-ish
    try testing.expect(try matchTest("[a-z]+@[a-z]+\\.[a-z]+", "user@example.com"));
}

test "step-by-step matching tracks dead state" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const nfa = try compile(a, "abc");

    var ss = try nfa.startStates(a);
    defer ss.deinit(a);
    try testing.expect(!ss.isEmpty());
    try testing.expect(!nfa.isAccepting(&ss));

    var s2 = try nfa.step(&ss, 'a', a);
    defer s2.deinit(a);
    try testing.expect(!s2.isEmpty());
    try testing.expect(!nfa.isAccepting(&s2));

    var s3 = try nfa.step(&s2, 'x', a);
    defer s3.deinit(a);
    try testing.expect(s3.isEmpty()); // dead
}
