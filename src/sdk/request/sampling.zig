//! What a request asks of the sampler. Built by the host's handlers, read by the engine.

const mlx = @import("../../mlx/ffi.zig");
const Constraint = @import("../grammar/constraint.zig").Constraint;
const ThinkBound = @import("forced.zig").ThinkBound;
const ToolForce = @import("forced.zig").ToolForce;

/// Sampling parameters for token generation.
pub const SamplingParams = struct {
    temperature: f32 = 1.0,
    top_p: f32 = 1.0,
    top_k: u32 = 0, // 0 = disabled
    min_p: ?f32 = null,
    repeat_penalty: f32 = 1.0,
    presence_penalty: f32 = 0.0, // 0.0 = disabled
    /// OpenAI's: subtracted once per earlier occurrence of a token (0.0 = disabled).
    frequency_penalty: f32 = 0.0,
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
    /// Forced tool-call opener (`ToolForce`), owned like `think_bound`.
    tool_force: ?*ToolForce = null,
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
        return self.repeat_penalty != 1.0 or self.presence_penalty != 0.0 or self.frequency_penalty != 0.0;
    }

    /// A grammar or a penalty reshapes the logits spec verify compares against.
    pub fn shapesLogits(self: SamplingParams) bool {
        return self.constraint != null or self.penalized();
    }
};
