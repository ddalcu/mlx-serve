#!/usr/bin/env python3
"""Dump HeartMuLa parity fixtures (.raw f32 / i32) for the env-gated oracles in `src/plugins/heartmula/heartmula.zig`.

USER-RUN with heartlib's own environment (torch + torchtune + vector_quantize_pytorch), fp32 on CPU, from the
UPSTREAM checkpoints (the bf16 pack is what the oracles load: `HEARTMULA_TEST_MODEL`).

Files in --out (point `HEARTMULA_FIXTURES` at it):
  tags_{a,b,c,d}.txt / lyrics_{a,b,c,d}.txt / ids_{a,b,c,d}.i32 : the prompt's text column (muq slot = 0); c has empty
                       tags, d the scripts `str.lower` maps beyond Latin (Vietnamese, fullwidth, final sigma, İ)
  c0_prefill.f32 [2, V]   : codebook-0 logits of both CFG rows after the prompt (sample a, cfg 3)
  guided.f32 [4, 8, V]    : the guided logits each of the first 4 frames samples its 8 codes from
  codes.i32 [N, 8]        : the greedy (argmax) frames, teacher-forced replay for the AR stage
  rvq_codes.i32 [8, 40] -> rvq_out.f32 [80, 512]           : RVQ dequant + cond_feature_emb + nearest x2
  dit_in.f32 [2, 96, 1024] at t=0.3 -> dit_out.f32 [2, 96, 256] : one estimator call
  eu_noise / eu_ctx [96, 256], eu_mu [96, 512] -> eu_out.f32 [96, 256] : 3 Euler steps, 24 in-context rows, CFG 1.25
  sq_in.f32 [2, 24, 128] -> sq_out.f32 [2, 46080]           : the scalar decoder (two bands)

Usage:
    ~/code/heartlib/.venv/bin/python tests/dump_heartmula_fixtures.py --ckpt ~/code/heartlib/ckpt --out DIR
    ... --replay-bf16 [--device mps]   # the bar: PyTorch bf16's own agreement with the fp32 greedy codes
"""

import argparse
import os
from pathlib import Path

os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

import numpy as np
import torch

SAMPLES = {
    "a": ("Pop, Piano, Female Vocal, Upbeat",
          "[Verse]\nNeon lights across the bay\nWe keep running anyway\n\n[Chorus]\nHold on, hold on tonight"),
    "b": ("ballad,strings,emotional", "[Intro]\n\n[Verse]\n月光洒在窗前\n我在等你回来\n[Outro]"),
    "c": ("", "[Verse]\nOne line, café, 12345 don't stop"),
    "d": ("Pop,Ballad,ＦＥＭＡＬＥ", "[Verse]\nẠnh Yêu Em ＡＢＣ\nΟΔΟΣ ΣΑΣ İstanbul Ǆ Ա Ⅷ"),
}
CFG = 3.0
FRAMES = 200


def join_tags(tags):
    """The server's `joinTags`: no space around a comma, as heartlib's README writes tags."""
    return ",".join(t.strip() for t in tags.split(",") if t.strip())


def dump(out, name, arr, dtype):
    np.ascontiguousarray(np.asarray(arr, dtype=dtype)).tofile(out / name)
    print(f"  {name} {np.asarray(arr).shape}")


def replay_bf16(args, pipe, mm):
    """Teacher-force the fp32 greedy codes through a bf16 model; print how many argmax picks agree."""
    dev = torch.device(args.device)
    codes = torch.from_numpy(np.fromfile(args.out / "codes.i32", dtype=np.int32).reshape(-1, 8)).long()
    model = mm.HeartMuLa.from_pretrained(str(args.ckpt / "HeartMuLa-oss-3B"), dtype=torch.bfloat16).to(dev).eval()
    tags, lyrics = SAMPLES["a"]
    inp = pipe.preprocess({"tags": join_tags(tags), "lyrics": lyrics}, cfg_scale=CFG)
    state = {"n": 0, "agree": 0, "frame": 0}

    def forced(logits, topk, temperature):
        i = state["n"]
        state["agree"] += int(torch.argmax(logits[0]).item() == codes[i // 8, i % 8].item())
        state["n"] += 1
        return codes[i // 8, i % 8].view(1, 1).to(dev, torch.int)

    mm.sample_topk = forced
    model.setup_caches(2)
    with torch.no_grad(), torch.autocast(device_type=dev.type, dtype=torch.bfloat16):
        model.generate_frame(tokens=inp["tokens"].to(dev), tokens_mask=inp["tokens_mask"].to(dev),
                             input_pos=inp["pos"].to(dev), temperature=1.0, topk=1, cfg_scale=CFG,
                             continuous_segments=inp["muq_embed"].to(dev), starts=inp["muq_idx"])
        for i in range(len(codes) - 1):
            pad = torch.zeros((2, 1, 9), dtype=torch.long, device=dev)
            pad[:, 0, :-1] = codes[i].to(dev)
            mask = torch.ones_like(pad, dtype=torch.bool)
            mask[..., -1] = False
            model.generate_frame(tokens=pad, tokens_mask=mask, input_pos=inp["pos"].to(dev)[..., -1:] + i + 1,
                                 temperature=1.0, topk=1, cfg_scale=CFG)
    print(f"bf16 ({args.device}) greedy replay: {state['agree']}/{state['n']} = {100 * state['agree'] / state['n']:.1f}%")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--frames", type=int, default=FRAMES)
    ap.add_argument("--replay-bf16", action="store_true")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)

    from heartlib.pipelines import music_generation as mg
    from heartlib.heartmula import modeling_heartmula as mm
    from heartlib.heartcodec.modeling_heartcodec import HeartCodec

    cpu, f32 = torch.device("cpu"), torch.float32
    pipe = mg.HeartMuLaGenPipeline.from_pretrained(str(args.ckpt), device=cpu, dtype=f32, version="3B", lazy_load=True)
    if args.replay_bf16:
        return replay_bf16(args, pipe, mm)

    for key, (tags, lyrics) in SAMPLES.items():
        (args.out / f"tags_{key}.txt").write_text(tags, encoding="utf-8")
        (args.out / f"lyrics_{key}.txt").write_text(lyrics, encoding="utf-8")
        inp = pipe.preprocess({"tags": join_tags(tags), "lyrics": lyrics}, cfg_scale=CFG)
        dump(args.out, f"ids_{key}.i32", inp["tokens"][0, :, -1].numpy(), np.int32)

    # ── LM: prefill logits, guided logits, greedy frames ──
    model = mm.HeartMuLa.from_pretrained(str(args.ckpt / "HeartMuLa-oss-3B"), dtype=f32).eval()
    tags, lyrics = SAMPLES["a"]
    inp = pipe.preprocess({"tags": join_tags(tags), "lyrics": lyrics}, cfg_scale=CFG)
    guided = []
    c0_raw = []
    orig_head = model.codebook0_head.forward
    model.codebook0_head.forward = lambda h: (c0_raw.append(orig_head(h)), c0_raw[-1])[1]

    def greedy(logits, topk, temperature):
        guided.append(logits[0].detach().clone())
        return torch.argmax(logits, dim=-1, keepdim=True).to(torch.int)

    mm.sample_topk = greedy
    model.setup_caches(2)
    frames = []
    with torch.no_grad():
        tok = model.generate_frame(tokens=inp["tokens"], tokens_mask=inp["tokens_mask"], input_pos=inp["pos"],
                                   temperature=1.0, topk=1, cfg_scale=CFG, continuous_segments=inp["muq_embed"],
                                   starts=inp["muq_idx"])
        dump(args.out, "c0_prefill.f32", c0_raw[0].numpy(), np.float32)
        frames.append(tok[0])
        for i in range(args.frames - 1):
            pad = torch.zeros((2, 1, 9), dtype=torch.long)
            pad[:, 0, :-1] = tok
            mask = torch.ones_like(pad, dtype=torch.bool)
            mask[..., -1] = False
            tok = model.generate_frame(tokens=pad, tokens_mask=mask, input_pos=inp["pos"][..., -1:] + i + 1,
                                       temperature=1.0, topk=1, cfg_scale=CFG)
            if torch.any(tok[0] >= 8193):
                break
            frames.append(tok[0])
    dump(args.out, "guided.f32", torch.stack(guided[: 4 * 8]).reshape(4, 8, -1).numpy(), np.float32)
    dump(args.out, "codes.i32", torch.stack(frames).numpy(), np.int32)
    del model

    # ── codec stages ──
    codec = HeartCodec.from_pretrained(str(args.ckpt / "HeartCodec-oss"), dtype=f32).eval()
    fm = codec.flow_matching
    g = torch.Generator().manual_seed(1)
    with torch.no_grad():
        codes = torch.randint(0, 8192, (8, 40), generator=g)
        dump(args.out, "rvq_codes.i32", codes.numpy(), np.int32)
        q = fm.vq_embed.get_output_from_indices(codes.unsqueeze(0).transpose(1, 2))
        q = torch.nn.functional.interpolate(fm.cond_feature_emb(q).permute(0, 2, 1), scale_factor=2, mode="nearest")
        dump(args.out, "rvq_out.f32", q.permute(0, 2, 1)[0].numpy(), np.float32)

        x = torch.randn(2, 96, 1024, generator=g)
        dump(args.out, "dit_in.f32", x.numpy(), np.float32)
        dump(args.out, "dit_out.f32", fm.estimator(x, timestep=torch.tensor([0.3, 0.3])).numpy(), np.float32)

        noise = torch.randn(1, 96, 256, generator=g)
        ctx = torch.zeros(1, 96, 256)
        ctx[:, :24] = torch.randn(1, 24, 256, generator=g)
        mu = torch.randn(1, 96, 512, generator=g)
        lat = fm.solve_euler(noise.clone(), ctx, 24, torch.linspace(0, 1, 4), mu, 1.25)
        lat[:, :24] = ctx[:, :24]
        for n, t in (("eu_noise", noise), ("eu_ctx", ctx), ("eu_mu", mu), ("eu_out", lat)):
            dump(args.out, f"{n}.f32", t[0].numpy(), np.float32)

        z = torch.rand(2, 24, 128, generator=g) * 2 - 1
        dump(args.out, "sq_in.f32", z.numpy(), np.float32)
        dump(args.out, "sq_out.f32", codec.scalar_model.decode(z.transpose(1, 2)).squeeze(1).numpy(), np.float32)
    print(f"export HEARTMULA_FIXTURES={args.out}")


if __name__ == "__main__":
    main()
