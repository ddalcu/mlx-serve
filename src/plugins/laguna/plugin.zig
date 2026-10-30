//! poolside Laguna (117.6B-A8.5B coder): `Decoder(TableEmbed, RmsNorm, LagunaAttention(.softplus_gate), Mlp, AddNorm,
//! LmHead)`: per-layer head counts, YaRN on full layers (mscale computed), sliding layers at the local base, a softplus
//! per-head output gate; a dense MLP on `mlp_only_layers` (found by weight presence), else sigmoid-routed experts
//! with a selection bias and an ungated shared expert (`family.zig`). Serial, as the legacy.

const std = @import("std");
const abi = @import("../../abi/plugin.zig");
const sdk = @import("../../sdk/root.zig");
const bases = @import("../../sdk/bases/bases.zig");
const family = @import("family.zig");
const DecodeCopies = @import("../../sdk/bases/blocks/decode_copies.zig").DecodeCopies;
pub const config = @import("config.zig");

pub const plugin = config.plugin;

const b = bases.blocks;

pub const Model = bases.Decoder(.{
    .embed = b.embed.TableEmbed,
    .norm = b.norm.RmsNorm,
    .mixer = DecodeCopies(b.laguna_attention.LagunaAttention(.{ .softplus_gate = true })),
    .mlp = family.Mlp,
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

/// The tiny pack the conformance suite runs: a dense layer then an MoE one, a YaRN full layer and a sliding one,
/// width 64, seeded random f32.
pub fn conformanceFixture(a: std.mem.Allocator) !sdk.fixture.Fixture {
    return family.conformancePack(a, .laguna,
        \\{"model_type":"laguna","hidden_size":64,"intermediate_size":128,"num_hidden_layers":2,"num_attention_heads":2,
        \\"num_key_value_heads":1,"head_dim":32,"rms_norm_eps":1e-6,"vocab_size":64,"max_position_embeddings":4096,
        \\"tie_word_embeddings":false,"gating":true,"sliding_window":4,"num_experts":4,"num_experts_per_tok":2,
        \\"moe_intermediate_size":32,"num_attention_heads_per_layer":[2,2],"eos_token_id":2,
        \\"layer_types":["full_attention","sliding_attention"],
        \\"rope_parameters":{"full_attention":{"rope_theta":500000.0,"rope_type":"yarn","factor":32.0,
        \\"original_max_position_embeddings":1024,"beta_slow":1.0,"beta_fast":32.0,"partial_rotary_factor":0.5},
        \\"sliding_attention":{"rope_type":"default","rope_theta":10000.0,"partial_rotary_factor":1.0}}}
    );
}

// ── tests ──

test {
    _ = config;
}
