//! Recursive Flint (39.8B-A2.5B): `Decoder(TableEmbed, RmsNorm, LagunaAttention(.sinks), Mlp, AddNorm, LmHead)`:
//! Laguna's attention arm without the gate, plus learned per-head sinks (`self_attn.softmax_offset`); sliding and NoPE
//! full layers; Laguna's MoE binding (`plugins/laguna/family.zig`). Batches decode, as the legacy.

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
    .mixer = DecodeCopies(b.laguna_attention.LagunaAttention(.{ .sinks = true })),
    .mlp = family.Mlp,
    .seam = b.seam.AddNorm,
    .head = b.head.LmHead,
});

const T = family.Table(Model, true);

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

/// The tiny pack the conformance suite runs: a dense layer then an MoE one, a NoPE full layer and a sliding one,
/// width 64, seeded random f32.
pub fn conformanceFixture(a: std.mem.Allocator) !sdk.fixture.Fixture {
    return family.conformancePack(a, .flint,
        \\{"model_type":"flint","hidden_size":64,"intermediate_size":128,"num_hidden_layers":2,"num_attention_heads":2,
        \\"num_key_value_heads":1,"head_dim":32,"rms_norm_eps":1e-6,"vocab_size":64,"max_position_embeddings":4096,
        \\"tie_word_embeddings":false,"rope_theta":10000.0,"sliding_window":4,"num_experts":4,"num_experts_per_tok":2,
        \\"moe_intermediate_size":32,"norm_topk_prob":true,"layer_types":["full_attention","sliding_attention"],
        \\"no_rope_layer_indices":[0],"eos_token_id":2}
    );
}

// ── tests ──

test {
    _ = config;
}
