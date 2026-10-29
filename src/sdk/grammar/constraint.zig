//! Grammar-constrained sampling state: the JSON grammar, its token mask and the reasoning protocol that defers it.
//! Built by the request's handler, advanced by the engine per token.

const std = @import("std");
const json_grammar = @import("json_grammar.zig");
const json_schema = @import("json_schema.zig");
const token_mask = @import("token_mask.zig");
const rp_mod = @import("protocol.zig");

pub const Constraint = struct {
    grammar: *json_grammar.Grammar,
    token_bytes: *const token_mask.TokenBytes,
    mask_buf: []bool,
    /// Final-answer grammars normally apply at token 0 (thinking-off and all
    /// prompt-committed content channels). A reasoning protocol instead makes
    /// generation start in the model's channel machinery: either inside an
    /// already-open reasoning block (prompt-opened) or at the unresolved
    /// opener choice. The protocol state drives the phase switches below.
    proto: ?*const rp_mod.Protocol = null,
    pstate: rp_mod.State = .{},
    /// Where the constrained payload begins in the generated stream — the
    /// authoritative reasoning→answer boundary the response emitters route
    /// on. Set exactly once, at the transition; `null` until then.
    pending_span: ?rp_mod.ConstraintSpan = null,
};

/// A forced recovery must leave room for the whole remaining transition AND
/// at least one constrained answer token. At the ordinary completion cap the
/// caller keeps the existing length-stop behavior — no reserve is created.
pub fn forcedBoundaryCanContinue(completion_tokens: u32, max_tokens: u32, transition_tokens: usize) bool {
    return completion_tokens +| @as(u32, @intCast(transition_tokens)) + 1 <= max_tokens;
}

/// RAII bundle for grammar-constrained sampling. Owns the parsed schema,
/// grammar state machine, and per-step mask buffer. The embedded `Constraint`
/// holds pointers/slices into the surrounding struct, so this struct must NOT
/// be moved after `initFromValue`. Construct via `var sc: SchemaConstraint =
/// undefined; try sc.initFromValue(...);` and pass `&sc.constraint` to
/// `SamplingParams`.
pub const SchemaConstraint = struct {
    schema: json_schema.Schema,
    grammar: json_grammar.Grammar,
    mask_buf: []bool,
    constraint: Constraint,
    allocator: std.mem.Allocator,

    /// Initialize in-place from a JSON schema value. On failure, any partial
    /// allocations made during this call are freed and the struct is left
    /// undefined (do not call `deinit`).
    pub fn initFromValue(
        self: *SchemaConstraint,
        allocator: std.mem.Allocator,
        schema_value: std.json.Value,
        token_bytes: *const token_mask.TokenBytes,
    ) !void {
        self.allocator = allocator;
        self.schema = try json_schema.parse(allocator, schema_value);
        errdefer self.schema.deinit();

        self.grammar = try json_grammar.Grammar.init(allocator, &self.schema);
        errdefer self.grammar.deinit();

        self.mask_buf = try allocator.alloc(bool, token_bytes.bytes.len);
        errdefer allocator.free(self.mask_buf);

        self.constraint = .{
            .grammar = &self.grammar,
            .token_bytes = token_bytes,
            .mask_buf = self.mask_buf,
        };
    }

    pub fn deinit(self: *SchemaConstraint) void {
        self.allocator.free(self.mask_buf);
        self.grammar.deinit();
        self.schema.deinit();
    }

    /// Route this request through a resolved reasoning protocol. Must be
    /// called before generation starts; the protocol is a per-request value
    /// the caller keeps alive (the Constraint borrows it).
    pub fn deferWithProtocol(self: *SchemaConstraint, proto: *const rp_mod.Protocol) void {
        self.constraint.proto = proto;
        self.constraint.pstate = proto.startState();
        if (self.constraint.pstate.phase == .json_body) self.constraint.pending_span = .{ .token_index = 0, .byte_offset = 0 };
    }
};
