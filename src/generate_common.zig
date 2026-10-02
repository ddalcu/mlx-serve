const std = @import("std");
const testing = std.testing;
const mlx = @import("mlx_types.zig");
const json_schema = @import("json_schema.zig");
const json_grammar = @import("json_grammar.zig");
const token_mask = @import("token_mask.zig");
const rp_mod = @import("reasoning_protocol.zig");

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

pub const TokenLogprob = struct {
    token_id: u32,
    logprob: f32,
};

pub const LogprobResult = struct {
    token_logprob: f32, // logprob of the chosen token
    top_logprobs: []TokenLogprob, // top N alternatives (caller must free)
};

pub const ThinkBound = struct {
    budget: u32,
    opener_id: ?u32,
    closer_id: u32,
    forced: []const u32,
    in_think: bool,
    count: u32 = 0,
    cursor: usize = 0,
    fired: bool = false,

    pub fn observe(self: *ThinkBound, ids: []const u32) void {
        while (self.cursor < ids.len) : (self.cursor += 1) {
            const id = ids[self.cursor];
            if (id == self.closer_id) {
                self.in_think = false;
            } else if (self.opener_id != null and id == self.opener_id.?) {
                self.in_think = true;
                self.count = 0;
            } else if (self.in_think) {
                self.count += 1;
            }
        }
    }

    pub fn due(self: *const ThinkBound) bool {
        return !self.fired and self.in_think and self.count >= self.budget;
    }
};

pub const SamplingParams = struct {
    temperature: f32 = 1.0,
    top_p: f32 = 1.0,
    top_k: u32 = 0, // 0 = disabled
    repeat_penalty: f32 = 1.0,
    presence_penalty: f32 = 0.0, // 0.0 = disabled
    seed: ?u64 = null,
    /// Draw index under `seed`: every sample takes a fresh key.
    draw: u64 = 0,
    /// When non-null, generation is constrained to outputs that satisfy the
    /// grammar at byte level. Forces a synchronous sampling path (no lazy
    /// pipeline) since grammar advancement requires the realized token id.
    constraint: ?*Constraint = null,
    /// In-stream thinking budget (`ThinkBound`), owned by the request handler
    /// like `constraint`; null = no bound.
    think_bound: ?*ThinkBound = null,
    /// Reserved-token suppression mask: `[vocab]` bool, true = the sampler
    /// must never draw this id (reserved specials like `<|fim_hole|>`, which
    /// a degenerate distribution can rank top-5 at a collapsed position — a
    /// reserved marker in chat output is always a bug). Model-lifetime,
    /// OWNED by the Transformer (`suppress_mask`), non-owning here; wired by
    /// `Generator.initWithOptions` so every sampling path inherits it.
    /// Applied by both samplers and both stochastic-verify filters, fully
    /// lazy (`mlx_where` + -inf, no host sync); logprobs deliberately keep
    /// reading the RAW logits — the field reports the model, the mask is
    /// sampling policy. Null = no suppression (kill switch, no-template
    /// models, every non-suppressing arch).
    suppress_mask: ?mlx.mlx_array = null,
    /// Seeded draws go through `keyed_sample`: a token's noise is a hash of
    /// (seed, position, id), so a draft shares the draw its verify row makes
    /// (row-exact archs, `ModelConfig.rowExactDecode`).
    keyed: bool = false,
    /// Absolute position of generated token 0 (the prompt length): a keyed
    /// draw's position is `position_base + draw`.
    position_base: u64 = 0,

    /// Penalties read the realized history, so they keep a request off the
    /// pipelined fast path, spec verify and batched decode.
    pub fn penalized(self: SamplingParams) bool {
        return self.repeat_penalty != 1.0 or self.presence_penalty != 0.0;
    }

    /// A grammar or a penalty reshapes the logits spec verify compares against.
    pub fn shapesLogits(self: SamplingParams) bool {
        return self.constraint != null or self.penalized();
    }
};

pub const GenerationResult = struct {
    text: []u8,
    token_ids: []u32,
    prompt_tokens: u32,
    completion_tokens: u32,
    finish_reason: []const u8,
    prefill_tps: f64,
    decode_tps: f64,
    /// Wall-clock nanoseconds spent on prefill (prompt processing).
    prefill_ns: u64 = 0,
    /// Wall-clock nanoseconds spent on decode (token generation).
    decode_ns: u64 = 0,
    /// Prompt tokens served from a KV-cache prefix (hot prefix cache for MLX,
    /// persistent-session prefix reuse for llama). `prompt_tokens - cached_tokens`
    /// is what was actually run through the model this turn, so `prefill_tps`
    /// reflects real compute rather than an inflated full-prompt rate.
    cached_tokens: u32 = 0,
    logprobs: ?[]LogprobResult = null, // per-token logprobs (caller must free)
    /// Non-null only when the degenerate-tail guard cut this generation:
    /// the `finish_details.type` value emitted beside `finish_reason`
    /// ("stop" for loop cuts — see `scheduler.loopStopReason`).
    /// Static string; nothing to free.
    finish_details: ?[]const u8 = null,
    /// Absolute byte offset into `text` where the constrained JSON payload
    /// begins, when a reasoning protocol was active and the payload began.
    /// 0 = direct answer. Null when no protocol ran, the payload never began
    /// (completion cap mid-reasoning), or the boundary fell outside the
    /// emitted (loop-trimmed) tokens.
    constraint_payload_byte: ?usize = null,
};

pub fn tokensPerSec(tokens: u64, elapsed_ns: u64) f64 {
    if (elapsed_ns == 0) return 0.0;
    const tok_f: f64 = @floatFromInt(tokens);
    const ns_f: f64 = @floatFromInt(elapsed_ns);
    return tok_f * @as(f64, @floatFromInt(std.time.ns_per_s)) / ns_f;
}

pub fn prefillTokensPerSec(prompt_tokens: u32, cached_tokens: u32, prefill_ns: u64) f64 {
    const uncached: u32 = if (prompt_tokens > cached_tokens) prompt_tokens - cached_tokens else 0;
    return tokensPerSec(uncached, prefill_ns);
}

pub fn forcedBoundaryCanContinue(completion_tokens: u32, max_tokens: u32, transition_tokens: usize) bool {
    return completion_tokens +| @as(u32, @intCast(transition_tokens)) + 1 <= max_tokens;
}

pub const AdaptiveWidthState = struct {
    supporting: u8 = 0,
    /// One-way ratchet: a prefill that has stepped down never widens again.
    ratcheted: bool = false,
    transitions: u32 = 0,
    width_min: u32 = 0,
    width_max: u32 = 0,
};

test "forced recovery leaves room for the whole transition and one answer token" {
    try std.testing.expect(forcedBoundaryCanContinue(0, 2, 1));
    try std.testing.expect(forcedBoundaryCanContinue(8, 10, 1));
    try std.testing.expect(!forcedBoundaryCanContinue(0, 1, 1));
    try std.testing.expect(!forcedBoundaryCanContinue(9, 10, 1));
    // A multi-token transition plans against the WHOLE sequence.
    try std.testing.expect(forcedBoundaryCanContinue(0, 4, 3));
    try std.testing.expect(!forcedBoundaryCanContinue(0, 3, 3));
    // An in-place activation needs no transition token.
    try std.testing.expect(forcedBoundaryCanContinue(0, 1, 0));
}

test "ThinkBound: counts only tokens inside the think block and fires at the budget" {
    const OPEN: u32 = 10;
    const CLOSE: u32 = 11;
    const forced = [_]u32{ 30, 31, CLOSE, 32 };
    var tb = ThinkBound{ .budget = 3, .opener_id = OPEN, .closer_id = CLOSE, .forced = &forced, .in_think = false };

    // Content before any opener never counts.
    tb.observe(&[_]u32{ 1, 2, 3, 4, 5 });
    try testing.expect(!tb.due());

    // Opened by the model: the budget counts from the opener.
    tb.observe(&[_]u32{ 1, 2, 3, 4, 5, OPEN, 6, 7 });
    try testing.expect(!tb.due());
    tb.observe(&[_]u32{ 1, 2, 3, 4, 5, OPEN, 6, 7, 8 });
    try testing.expect(tb.due());

    // A model that closed on its own is never forced.
    var closed = ThinkBound{ .budget = 3, .opener_id = OPEN, .closer_id = CLOSE, .forced = &forced, .in_think = true };
    closed.observe(&[_]u32{ 6, 7, CLOSE, 8, 9, 10, 11 });
    try testing.expect(!closed.due());

    // Prompt-opened: the count starts at token 0 and the forced closer ends it.
    var po = ThinkBound{ .budget = 2, .opener_id = null, .closer_id = CLOSE, .forced = &forced, .in_think = true };
    po.observe(&[_]u32{ 6, 7 });
    try testing.expect(po.due());
    po.fired = true;
    po.observe(&[_]u32{ 6, 7, 30, 31, CLOSE, 32, 40 });
    try testing.expect(!po.in_think);
    try testing.expect(!po.due());
}
