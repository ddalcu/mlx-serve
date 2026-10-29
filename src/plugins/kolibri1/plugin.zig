//! Aleph Alpha Kolibri-1 (78B-A3.5B, 384 experts top-6): `Decoder(TableEmbed, RmsNorm, PostNormed(LagunaAttention),
//! PostNormed(Moe), AddNorm, LmHead)`: Laguna's attention arm without gate or sinks (NoPE full layers), sandwich norms
//! (the attention output's `post_attn_norm`, the pre-FFN `post_attention_layernorm`, the MoE output's `post_ffn_norm`)
//! and routed experts on every layer (top-k on the f32 logits + `expert_bias`, weights sigmoid(logits), an ungated
//! shared expert). Serial, as the legacy.

const std = @import("std");
const abi = @import("../../abi/plugin.zig");
const sdk = @import("../../sdk/root.zig");
const bases = @import("../../sdk/bases/bases.zig");
const family = @import("../laguna/family.zig");
const DecodeCopies = @import("../../sdk/bases/blocks/decode_copies.zig").DecodeCopies;
pub const config = @import("config.zig");

pub const plugin = config.plugin;

const b = bases.blocks;

pub const Model = bases.Decoder(.{
    .embed = b.embed.TableEmbed,
    .norm = b.norm.RmsNorm,
    .mixer = b.extras.PostNormed(DecodeCopies(b.laguna_attention.LagunaAttention(.{}))),
    .mlp = b.extras.PostNormed(b.moe.Moe),
    .seam = b.seam.AddNorm,
    .head = b.head.LmHead,
});

const T = family.Table(Model, false);

pub const table = abi.ModelTable{
    .name = abi.Str.of(plugin.name),
    .claims = config.claims,
    .parse = &config.parse,
    .free_parsed = &config.freeParsed,
    .bill = &T.bill,
    .load = &T.load,
    .unload = &T.unload,
    .forward = &T.forward,
    .cache = &.{},
    .weights = &.{},
    .spec = &.{ .trunk_weight = b.trunk_weight.trunkWeightHook(family.Loaded(Model)) },
};

/// The tiny pack the conformance suite runs: two MoE layers (a NoPE full one, a sliding one), width 64, seeded
/// random f32.
pub fn conformanceFixture(a: std.mem.Allocator) !sdk.fixture.Fixture {
    return family.conformancePack(a, .kolibri1,
        \\{"model_type":"kolibri1","hidden_size":64,"num_hidden_layers":2,"num_attention_heads":2,"num_key_value_heads":1,
        \\"head_dim":32,"rms_norm_eps":1e-6,"vocab_size":64,"max_position_embeddings":4096,"rope_theta":10000.0,
        \\"tie_word_embeddings":false,"sliding_window":4,"num_experts":4,"num_experts_per_tok":2,
        \\"moe_intermediate_size":32,"norm_topk_prob":false,"layer_types":["full_attention","sliding_attention"],
        \\"eos_token_id":2}
    );
}

// ── tests ──

test {
    _ = config;
}
