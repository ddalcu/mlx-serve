#!/usr/bin/env python3
"""Dump reference outputs for SD 3.5's T5-XXL text encoder.

The oracle for `src/t5_encoder.zig`. Runs transformers' OWN `T5EncoderModel`
and saves the inputs, the final hidden state, and every intermediate a
24-layer tower can silently drift in:

    ids            [1, T] int32     the token ids, so both sides describe the
                                    same forward and cannot drift apart
    embedding      [1, T, d_model]  `shared(ids)`, block 0's input
    rel_bias       [1, H, T, T]     layer 0's `compute_bias` — the relative
                                    position bias EVERY layer reuses
    block.{i}.out  [1, T, d_model]  each encoder block's raw output
    last_hidden    [1, T, d_model]  after `encoder.final_layer_norm`

`rel_bias` and the per-block outputs are the point. A tower that agrees at
layer 0 and disagrees at layer 23 is a needle in a haystack; a tower whose
`rel_bias` is wrong is a one-line fix, and the relative-position bucketing is
the single most transcription-error-prone piece of T5.

Two modes:

    # tiny random-weight model of the REAL class — pins the architecture
    # without a 20 GB download. Writes a loadable component dir AND the fixture.
    python3 tests/dump_t5_fixtures.py build \\
        --dir ~/.mlx-serve/staging/t5-tiny \\
        --out ~/.mlx-serve/staging/t5_tiny_fixture.safetensors

    # real-checkpoint parity, run against SD 3.5's own text_encoder_3
    python3 tests/dump_t5_fixtures.py real \\
        --model ~/.mlx-serve/staging/sd3.5-large/text_encoder_3 \\
        --out   ~/.mlx-serve/staging/t5_real_fixture.safetensors

Then, from the repo root:

    T5_FIXTURE=~/.mlx-serve/staging/t5_tiny_fixture.safetensors \\
    T5_MODEL_DIR=~/.mlx-serve/staging/t5-tiny \\
      zig build test -Doptimize=ReleaseFast -Dtest-filter="t5 encoder parity"

Runs on CPU in float32 DELIBERATELY. MPS fp16 quietly decorrelates deep towers
(this repo has that scar from the ViT/DiT fixtures), and a reference computed
less precisely than the thing it checks is not a reference.
"""
import argparse
import os
import sys

import torch
from safetensors.torch import save_file

# The tiny geometry. Every field that decides NUMERICS is the real
# checkpoint's value (`feed_forward_proj`, the two relative-attention
# parameters, the eps, `tie_word_embeddings`); only the SIZES shrink. A fixture
# that shrinks `relative_attention_num_buckets` would stop testing the bucket
# function, which is the whole reason this fixture exists.
TINY = dict(
    vocab_size=128,
    d_model=64,
    d_ff=128,
    d_kv=16,
    num_heads=4,
    num_layers=2,
    feed_forward_proj="gated-gelu",
    layer_norm_epsilon=1e-6,
    relative_attention_num_buckets=32,
    relative_attention_max_distance=128,
    dropout_rate=0.0,
    tie_word_embeddings=False,
    is_encoder_decoder=False,
    use_cache=False,
)

# A window with real padding in it: SD 3 pads T5 to a fixed length with pad id
# 0 and passes NO attention mask, so the pad positions are attended to like any
# other token. A fixture with no padding would not catch a port that "helpfully"
# masks them.
TINY_IDS = [7, 42, 3, 99, 1, 0, 0, 0, 0, 0]

# The real tower's window. `</s>` (1) then `<pad>` (0), same as the pipeline.
REAL_IDS = [3, 9, 1974, 13, 3, 9, 1712, 1] + [0] * 8


def dump(model, ids, out):
    device = torch.device("cpu")
    model = model.to(device=device, dtype=torch.float32).eval()
    ids_t = torch.tensor([ids], dtype=torch.long, device=device)

    tensors = {"ids": ids_t.to(torch.int32).contiguous()}

    # Per-block raw outputs via hooks rather than `output_hidden_states`: that
    # tuple holds each block's INPUT plus a final-normed tail, so the last
    # block's raw output is not in it at all.
    captured = {}

    def hook(i):
        def fn(_mod, _inp, outp):
            captured[i] = (outp[0] if isinstance(outp, tuple) else outp).detach()

        return fn

    handles = [b.register_forward_hook(hook(i)) for i, b in enumerate(model.encoder.block)]
    try:
        with torch.no_grad():
            out_obj = model(input_ids=ids_t)
            emb = model.shared(ids_t)
            attn0 = model.encoder.block[0].layer[0].SelfAttention
            bias = attn0.compute_bias(ids_t.shape[1], ids_t.shape[1])
    finally:
        for h in handles:
            h.remove()

    tensors["embedding"] = emb.detach().contiguous()
    tensors["rel_bias"] = bias.detach().contiguous()
    for i, h in captured.items():
        tensors[f"block.{i}.out"] = h.contiguous()
    tensors["last_hidden"] = out_obj.last_hidden_state.detach().contiguous()

    os.makedirs(os.path.dirname(os.path.abspath(out)) or ".", exist_ok=True)
    # Cast the ACTIVATIONS to float32, never the ids. A blanket
    # `.to(torch.float32)` over every tensor turns `ids` into floats that the
    # Zig side reads back through `mlx_array_data_int32` — reinterpreting the
    # bit pattern, so every nonzero id becomes garbage while id 0 survives
    # (0.0f is all-zero bits). That failed as embedding cos = sqrt(pads/T),
    # which looks like a broken encoder rather than a broken fixture.
    saved = {
        k: (v if v.dtype in (torch.int32, torch.int64) else v.to(torch.float32))
        for k, v in tensors.items()
    }
    save_file({k: (v.to(torch.int32) if v.dtype == torch.int64 else v).contiguous()
               for k, v in saved.items()}, out)
    # Print what was actually WRITTEN, not what was built — the old loop
    # reported `ids` as int32 while saving it as float32, which is what let the
    # bug above hide in plain sight.
    for k, v in saved.items():
        dt = torch.int32 if v.dtype == torch.int64 else v.dtype
        print(f"  {k:<16} {tuple(v.shape)} {dt}")
    print(f"wrote {out}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="mode", required=True)

    b = sub.add_parser("build", help="tiny random-weight T5EncoderModel")
    b.add_argument("--dir", required=True, help="component dir to write (config + weights)")
    b.add_argument("--out", required=True)
    b.add_argument("--seed", type=int, default=0)

    r = sub.add_parser("real", help="a real text_encoder_3 dir")
    r.add_argument("--model", required=True)
    r.add_argument("--out", required=True)

    args = ap.parse_args()

    try:
        from transformers import T5Config, T5EncoderModel
    except ImportError:
        sys.exit("transformers is required to generate this fixture")

    if args.mode == "build":
        torch.manual_seed(args.seed)
        cfg = T5Config(**TINY)
        model = T5EncoderModel(cfg)
        # `T5Config`'s initializer scales weights by `initializer_factor` and
        # d_model**-0.5, which leaves several tensors near zero. Re-randomize at
        # a visible scale so a dead layer cannot pass by producing ~0 on both
        # sides — an all-zero tensor matches an all-zero tensor perfectly.
        with torch.no_grad():
            for p in model.parameters():
                if p.dim() >= 2:
                    p.normal_(0.0, 0.5)
                else:
                    p.normal_(1.0, 0.1)  # the RMS-norm gains
        d = os.path.expanduser(args.dir)
        model.save_pretrained(d, safe_serialization=True)
        print(f"wrote tiny checkpoint {d}")
        dump(model, TINY_IDS, os.path.expanduser(args.out))
    else:
        d = os.path.expanduser(args.model)
        model = T5EncoderModel.from_pretrained(d, dtype=torch.float32)
        dump(model, REAL_IDS, os.path.expanduser(args.out))


if __name__ == "__main__":
    main()
