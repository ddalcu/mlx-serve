#!/usr/bin/env python3
"""Dump MiniMax-Music3 parity fixtures (.raw f32/i32) for the Zig oracle tests
in `src/music3.zig`.

USER-RUN (needs torch + transformers + tokenizers + mlx + numpy). Run on the
128 GB Mac: the global LLM is 8B fp32 (~32 GB resident).

TRANSCRIPTION ORACLE: the diffusers integration branch (dafe3733,
minimax-music3-integration) is not installable standalone, so the depth
decoder, condition encoder, DiT and vocoder modules below are transcribed
from `~/claude-tmp/music3/ref/*.py` (Apache-2.0, The MiniMax Team + HF Team)
into plain torch. The global LLM runs the REAL `transformers` Qwen3 code, and
`_clean_caption` / `_normalize_lyrics` are copied verbatim from the reference.

Everything runs fp32 on our DEQUANTIZED pack weights (mx.dequantize per
tensor, geometry-solved bits), so the fixtures measure the Zig port, not the
quantization. Compare with cos AND rms_ratio in float64 — a cosine alone
cannot see a scale error, and fp32 accumulation over 16M terms swamps one.

Fixture taps (all under --out, defaults ~/claude-tmp/music3/fixtures):

  text_ids.i32.raw    assembled conditional prompt token ids
  uncond_ids.i32.raw  CFG row ([1:-2] -> <|audio_cfg|>)
  last_hidden.f32.raw [2,4096]   post-prefill hidden (cond, uncond rows)
  logits0.f32.raw     [2,200000] lm_head(last_hidden), pre-mask
  ar_codes.i32.raw    [FRAMES+1,8] greedy per-frame codes (c0..c7, frame 0 first)
  ar_hiddens.f32.raw  [1,FRAMES,32768] condition stream (frame 0 skipped)
  cond_out.f32.raw    [1,L,2048] condition encoder over ar_hiddens
  dit_lat.f32.raw     [1,128,L]  seeded noise latent (channel-first)
  dit_v_t0.f32.raw    [1,128,L]  DiT velocity at t=0.0, cond = cond_out
  dit_v_t05.f32.raw   [1,128,L]  DiT velocity at t=0.5
  voc_wav.f32.raw     [1,2,L*512] vocoder(dit_lat)
  meta.json           frames/latent_len/caption/lyrics

Usage:
    uv run --with torch --with transformers --with tokenizers --with mlx \
        --with numpy python tests/dump_music3_fixtures.py \
        [--model DIR] [--out DIR] [--device cpu|mps] [--frames N] [--skip-llm]

Then:  export MUSIC3_TEST_MODEL=<pack> MUSIC3_FIXTURES=<out>  and run
`zig build test -Dtest-filter="music3 oracle"`.
"""

import argparse
import json
import math
import os
import re

import numpy as np

DEFAULT_MODEL = os.path.expanduser("~/.mlx-serve/models/ddalcu/MiniMax-Music3-MLX-Serve-8bit")
DEFAULT_OUT = os.path.expanduser("~/claude-tmp/music3/fixtures")

CAPTION = "Upbeat **synthwave** with driving bass and dreamy pads\n<|bpm 120|>"
LYRICS = "[Verse] ignored text\nneon lights across the bay [Chorus]\nwe run all night ^ we never stay"
FRAMES = 12  # emitted frames in the greedy AR fixture

AUDIO_END = 151670
AUDIO_CFG = 151654
CODE_OFFSET = 151675
SEMANTIC_VOCAB = 16384
CFG_SCALE = 1.5
TOP_K = 50

# ── prompt assembly (verbatim from ref/encoders.py) ─────────────────────────

_SPECIAL_TAG_RE = re.compile(r"<\|([^|]*)\|>")
_LEADING_TAGS_RE = re.compile(r"^[ \t]*((?:\[[^\]]+\][ \t]*)+)")


def _clean_caption(caption):
    def _rewrite_special_tag(match):
        inner = match.group(1).strip()
        parts = inner.split(None, 1)
        return f"{parts[0]} is {parts[1]}" if len(parts) == 2 else inner

    text = _SPECIAL_TAG_RE.sub(_rewrite_special_tag, caption)
    lines_out = []
    for line in text.splitlines():
        line = re.sub(r"^\s{0,3}#{1,6}\s+", "", line)
        line = re.sub(r"^\s*[*+-]\s+", "", line)
        line = re.sub(r"^\s*\*\s+", "", line)
        while "**" in line:
            updated = re.sub(r"\*\*([^*]+)\*\*", r"\1", line)
            if updated == line:
                break
            line = updated
        line = re.sub(r"(?<!\*)\*([^*\n]+)\*(?!\*)", r"\1", line)
        lines_out.append(line.rstrip())
    text = "\n".join(lines_out)
    text = re.sub(r"^\s*[-*_]{3,}\s*$", "", text, flags=re.MULTILINE)
    text = text.replace("• ", "").replace("    ", "")
    return re.sub(r"\n{2,}", "\n", text)


def _normalize_lyrics(lyrics):
    output = []
    for line in lyrics.split("\n"):
        match = _LEADING_TAGS_RE.match(line)
        output.append(match.group(1).strip() if match else line)
    text = "\n".join(output)
    text = text.replace("] ", "]\n")
    text = text.replace(" [", "\n[")
    text = text.replace(" ^ ", "\n")
    text = re.sub(r"\[([^\]]+)\]", lambda match: f"[{match.group(1).lower()}]", text)
    return f"[start]\n{text}"


def assemble_prompt(caption, lyrics):
    return (
        f"<|im_start|><|caption_start|>{_clean_caption(caption)}<|caption_end|>"
        f"<|lyrics_start|>{_normalize_lyrics(lyrics)}<|lyrics_end|><|im_end|><|audio_start|>"
    )


# ── pack loading: dequantize per weight, geometry-solved bits ───────────────


def load_dequant(path, group_size):
    import mlx.core as mx

    mx.set_default_device(mx.cpu)
    raw = mx.load(path)
    out = {}
    for name, arr in raw.items():
        if name.endswith(".scales") or name.endswith(".biases"):
            continue
        base = name[: -len(".weight")] if name.endswith(".weight") else None
        if base is not None and f"{base}.scales" in raw:
            sc = raw[f"{base}.scales"]
            in_dim = sc.shape[-1] * group_size
            bits = (arr.shape[-1] * 32) // in_dim
            d = mx.dequantize(
                arr, sc, raw[f"{base}.biases"], group_size=group_size, bits=bits
            )
            out[name] = np.array(d.astype(mx.float32))
        else:
            out[name] = np.array(arr.astype(mx.float32))
    return out


# ── transcribed reference modules (fp32) ────────────────────────────────────


def build_torch_modules(pack, cfg, device):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class RMSNorm(nn.Module):
        def __init__(self, dim, eps=1e-6):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(dim))
            self.eps = eps

        def forward(self, x):
            v = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
            return v * self.weight

    class DepthBlock(nn.Module):
        def __init__(self, dim, heads, inter):
            super().__init__()
            self.heads, self.head_dim = heads, dim // heads
            self.input_layernorm = RMSNorm(dim)
            self.post_attention_layernorm = RMSNorm(dim)
            self.to_q = nn.Linear(dim, dim, bias=False)
            self.to_k = nn.Linear(dim, dim, bias=False)
            self.to_v = nn.Linear(dim, dim, bias=False)
            self.to_out = nn.Linear(dim, dim, bias=False)
            self.gate_proj = nn.Linear(dim, inter, bias=False)
            self.up_proj = nn.Linear(dim, inter, bias=False)
            self.down_proj = nn.Linear(inter, dim, bias=False)

        def forward(self, h):
            b, t, _ = h.shape
            x = self.input_layernorm(h)
            q = self.to_q(x).view(b, t, self.heads, self.head_dim).transpose(1, 2)
            k = self.to_k(x).view(b, t, self.heads, self.head_dim).transpose(1, 2)
            v = self.to_v(x).view(b, t, self.heads, self.head_dim).transpose(1, 2)
            a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            a = a.transpose(1, 2).reshape(b, t, -1)
            h = h + self.to_out(a)
            x = self.post_attention_layernorm(h)
            return h + self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))

    class DepthDecoder(nn.Module):
        def __init__(self, c):
            super().__init__()
            dim = c["hidden_size"]
            self.audio_embeddings = nn.Embedding(7168, dim)
            self.projection = nn.Linear(dim, dim, bias=False)
            self.pos_embedding = nn.Embedding(c["max_position_embeddings"], dim)
            self.layers = nn.ModuleList(
                DepthBlock(dim, c["num_attention_heads"], c["intermediate_size"])
                for _ in range(c["num_layers"])
            )
            self.norm = RMSNorm(dim)
            self.audio_heads = nn.ModuleList(nn.Linear(dim, 1024, bias=False) for _ in range(7))

        def forward(self, e):
            pos = torch.arange(e.shape[1], device=e.device)
            h = e + self.pos_embedding(pos).unsqueeze(0)
            for layer in self.layers:
                h = layer(h)
            return self.norm(h)

    class ConditionEncoder(nn.Module):
        def __init__(self, c):
            super().__init__()
            self.c = c
            self.layer_weight_logits = nn.Parameter(torch.zeros(c["num_condition_layers"]))
            self.layer_scale = nn.Parameter(torch.ones(1))
            self.proj = nn.Conv1d(c["condition_hidden_dim"], c["out_dim"], 3, padding=1)

        def forward(self, h):
            b, f, _ = h.shape
            nl, hd = self.c["num_condition_layers"], self.c["condition_hidden_dim"]
            h = h.transpose(1, 2).reshape(b, nl, hd, f)
            w = torch.softmax(self.layer_weight_logits, dim=0)
            h = torch.einsum("blht,l->bht", h, w)
            h = self.layer_scale * h
            h = self.proj(h)
            latent_len = max(1, int(f * 44100 / 24000 * 960 / 512))
            h = F.interpolate(h, size=latent_len, mode="nearest")
            return h.transpose(1, 2)

    class DitBlock(nn.Module):
        def __init__(self, dim, heads, head_dim, ff_inner):
            super().__init__()
            self.heads, self.head_dim = heads, head_dim
            self.norm1 = nn.LayerNorm(dim)
            self.norm2 = nn.LayerNorm(dim)
            self.to_q = nn.Linear(dim, dim, bias=False)
            self.to_k = nn.Linear(dim, dim, bias=False)
            self.to_v = nn.Linear(dim, dim, bias=False)
            self.to_out = nn.Linear(dim, dim, bias=False)
            self.ff_in = nn.Linear(dim, ff_inner * 2)
            self.ff_out = nn.Linear(ff_inner, dim)

        def rope(self, x, cos, sin):
            rd = cos.shape[-1]
            r = x[..., :rd]
            h1, h2 = r.chunk(2, dim=-1)
            rot = torch.cat((-h2, h1), dim=-1)
            r = r * cos[:, None, :] + rot * sin[:, None, :]
            return torch.cat((r, x[..., rd:]), dim=-1)

        def forward(self, h, cos, sin):
            b, t, _ = h.shape
            x = self.norm1(h)
            q = self.to_q(x).view(b, t, self.heads, self.head_dim)
            k = self.to_k(x).view(b, t, self.heads, self.head_dim)
            v = self.to_v(x).view(b, t, self.heads, self.head_dim)
            q, k = self.rope(q, cos, sin), self.rope(k, cos, sin)
            import torch.nn.functional as F

            a = F.scaled_dot_product_attention(
                q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
            )
            a = a.transpose(1, 2).reshape(b, t, -1)
            h = h + self.to_out(a)
            gate_states, gate = self.ff_in(self.norm2(h)).chunk(2, dim=-1)
            return h + self.ff_out(gate_states * F.silu(gate))

    class Dit(nn.Module):
        def __init__(self, c):
            super().__init__()
            dim = c["hidden_size"]
            cc = 2 * c["in_channels"] + c["condition_dim"]
            self.fourier_weight = nn.Parameter(torch.zeros(c["fourier_embedding_dim"] // 2, 1))
            self.time_linear_1 = nn.Linear(c["fourier_embedding_dim"], dim)
            self.time_linear_2 = nn.Linear(dim, dim)
            self.preprocess_conv = nn.Conv1d(cc, cc, 1, bias=False)
            self.proj_in = nn.Linear(cc, dim, bias=False)
            self.blocks = nn.ModuleList(
                DitBlock(dim, c["num_attention_heads"], c["attention_head_dim"], c["ff_inner_dim"])
                for _ in range(c["num_layers"])
            )
            self.proj_out = nn.Linear(dim, c["in_channels"], bias=False)
            self.postprocess_conv = nn.Conv1d(c["in_channels"], c["in_channels"], 1, bias=False)
            self.rotary_dim = c["rotary_dim"]

        def forward(self, lat, t, cond):
            import torch.nn.functional as F

            zeros = torch.zeros_like(lat)
            h = torch.cat((lat, zeros, cond.transpose(1, 2)), dim=1)
            h = self.preprocess_conv(h) + h
            h = h.transpose(1, 2)
            angles = 2.0 * math.pi * t.unsqueeze(-1) @ self.fourier_weight.T
            fourier = torch.cat((angles.cos(), angles.sin()), dim=-1)
            temb = self.time_linear_2(F.silu(self.time_linear_1(fourier)))
            h = self.proj_in(h)
            h = torch.cat((temb.unsqueeze(1), h), dim=1)
            seq = h.shape[1]
            inv = 1.0 / (
                10000.0 ** (torch.arange(0, self.rotary_dim, 2, dtype=torch.float32) / self.rotary_dim)
            )
            freqs = torch.outer(torch.arange(seq, dtype=torch.float32), inv)
            freqs = torch.cat((freqs, freqs), dim=-1)
            cos, sin = freqs.cos(), freqs.sin()
            for blk in self.blocks:
                h = blk(h, cos, sin)
            h = self.proj_out(h[:, 1:])
            h = h.transpose(1, 2)
            return self.postprocess_conv(h) + h

    class Snake(nn.Module):
        def __init__(self, ch):
            super().__init__()
            self.alpha = nn.Parameter(torch.ones(1, ch, 1))

        def forward(self, x):
            return x + (self.alpha + 1e-9).reciprocal() * torch.sin(self.alpha * x).pow(2)

    from torch.nn.utils import weight_norm

    class ResUnit(nn.Module):
        def __init__(self, dim, dilation):
            super().__init__()
            pad = (7 - 1) * dilation // 2
            self.snake1 = Snake(dim)
            self.conv1 = weight_norm(nn.Conv1d(dim, dim, 7, dilation=dilation, padding=pad))
            self.snake2 = Snake(dim)
            self.conv2 = weight_norm(nn.Conv1d(dim, dim, 1))

        def forward(self, x):
            return x + self.conv2(self.snake2(self.conv1(self.snake1(x))))

    class VocBlock(nn.Module):
        def __init__(self, in_dim, out_dim, stride):
            super().__init__()
            self.snake1 = Snake(in_dim)
            self.conv_t1 = weight_norm(
                nn.ConvTranspose1d(in_dim, out_dim, 2 * stride, stride, math.ceil(stride / 2))
            )
            self.res_unit1 = ResUnit(out_dim, 1)
            self.res_unit2 = ResUnit(out_dim, 3)
            self.res_unit3 = ResUnit(out_dim, 9)

        def forward(self, x):
            return self.res_unit3(self.res_unit2(self.res_unit1(self.conv_t1(self.snake1(x)))))

    class Vocoder(nn.Module):
        def __init__(self, c):
            super().__init__()
            self.latent_channels = c["latent_channels"]
            self.dec_in_proj = nn.Conv1d(c["latent_channels"] // 2, c["decoder_input_dim"], 1)
            self.conv_in = weight_norm(
                nn.Conv1d(c["decoder_input_dim"], c["decoder_hidden_dim"], 7, padding=3)
            )
            blocks, out_dim = [], c["decoder_hidden_dim"]
            for i, stride in enumerate(c["upsampling_ratios"]):
                in_dim = c["decoder_hidden_dim"] // (2**i)
                out_dim = c["decoder_hidden_dim"] // (2 ** (i + 1))
                blocks.append(VocBlock(in_dim, out_dim, stride))
            self.blocks = nn.ModuleList(blocks)
            self.snake_out = Snake(out_dim)
            self.conv_out = weight_norm(nn.Conv1d(out_dim, 1, 7, padding=3))

        def forward(self, lat):
            b, _, length = lat.shape
            h = lat.reshape(b * 2, self.latent_channels // 2, length)
            h = self.conv_in(self.dec_in_proj(h))
            for blk in self.blocks:
                h = blk(h)
            wave = torch.tanh(self.conv_out(self.snake_out(h)))
            return wave.reshape(b, 2, -1)

    def load_into(mod, weights, rename=None):
        sd = {}
        for k, v in weights.items():
            kk = rename(k) if rename else k
            if kk is None:
                continue
            sd[kk] = torch.from_numpy(v.copy())
        missing, unexpected = mod.load_state_dict(sd, strict=False)
        assert not unexpected, f"unexpected keys: {unexpected[:5]}"
        assert not missing, f"missing keys: {missing[:5]}"
        return mod.to(device).eval()

    gs = cfg["group_size"]
    dd = load_into(
        DepthDecoder(cfg["rvq_depth_decoder"]),
        load_dequant(pack("rvq_depth_decoder.safetensors"), gs),
        rename=lambda k: k.replace(".attn.", "."),
    )
    ce = load_into(
        ConditionEncoder(cfg["condition_encoder"]),
        load_dequant(pack("condition_encoder.safetensors"), gs),
    )

    def dit_rename(k):
        k = k.replace("transformer_blocks.", "blocks.")
        k = k.replace(".attn.to_out.0.", ".to_out.").replace(".attn.", ".")
        k = k.replace("time_proj.weight", "fourier_weight")
        k = k.replace("time_embed.linear_1.", "time_linear_1.")
        k = k.replace("time_embed.linear_2.", "time_linear_2.")
        return k

    dit = load_into(Dit(cfg["transformer"]), load_dequant(pack("transformer.safetensors"), gs), dit_rename)
    voc = load_into(Vocoder(cfg["vocoder"]), load_dequant(pack("vocoder.safetensors"), gs))
    return dd, ce, dit, voc


def build_llm(pack, cfg, device):
    import torch
    from transformers import Qwen3Config, Qwen3ForCausalLM

    c = cfg["language_model"]
    hf = Qwen3Config(
        hidden_size=c["hidden_size"],
        num_hidden_layers=c["num_hidden_layers"],
        num_attention_heads=c["num_attention_heads"],
        num_key_value_heads=c["num_key_value_heads"],
        head_dim=c["head_dim"],
        intermediate_size=c["intermediate_size"],
        vocab_size=c["vocab_size"],
        rms_norm_eps=c["rms_norm_eps"],
        rope_theta=c["rope_theta"],
        max_position_embeddings=c["max_position_embeddings"],
        tie_word_embeddings=False,
        attention_bias=False,
        attention_dropout=0.0,
    )
    print("[llm] dequantizing + loading fp32 (slow, ~32 GB) ...", flush=True)
    weights = load_dequant(pack("language_model.safetensors"), cfg["group_size"])
    # Consume `weights` as we go and assign in place: one 32 GB copy, not two.
    sd = {}
    for k in list(weights.keys()):
        sd[k] = torch.from_numpy(weights.pop(k))
    model = Qwen3ForCausalLM(hf)  # real init keeps non-state-dict buffers (rotary inv_freq) alive
    missing, unexpected = model.load_state_dict(sd, strict=False, assign=True)
    assert not unexpected, unexpected[:5]
    assert not missing, missing[:5]
    return model.to(device).eval()


def guided_semantic_logits(logits, vocab_mask):
    import torch

    logits = logits.float()
    logits = logits.masked_fill(vocab_mask, -float("inf"))
    conditional, unconditional = logits[0:1], logits[1:2]
    guided = unconditional + (conditional - unconditional) * CFG_SCALE
    threshold = torch.topk(conditional, TOP_K, dim=-1).values[..., -1, None]
    guided = guided.masked_fill(conditional < threshold, -float("inf"))
    return guided.masked_fill(vocab_mask.unsqueeze(0), -float("inf"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--device", default="cpu", choices=("cpu", "mps"))
    ap.add_argument("--frames", type=int, default=FRAMES)
    ap.add_argument("--skip-llm", action="store_true", help="skip the 8B stage (redump small fixtures)")
    args = ap.parse_args()

    import torch

    torch.manual_seed(0)
    device = torch.device(args.device)
    os.makedirs(args.out, exist_ok=True)
    cfg = json.load(open(os.path.join(args.model, "config.json")))
    pack = lambda name: os.path.join(args.model, name)

    def dump(name, arr):
        arr = np.ascontiguousarray(arr)
        arr.tofile(os.path.join(args.out, name))
        print(f"  {name}: shape {arr.shape} {arr.dtype}", flush=True)

    # ── prompt ids ──
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(os.path.join(args.model, "tokenizer", "tokenizer.json"))
    prompt = assemble_prompt(CAPTION, LYRICS)
    print(f"[prompt] {prompt!r}", flush=True)
    ids = tok.encode(prompt, add_special_tokens=False).ids
    uncond = list(ids)
    for i in range(1, len(uncond) - 2):
        uncond[i] = AUDIO_CFG
    dump("text_ids.i32.raw", np.array(ids, dtype=np.int32))
    dump("uncond_ids.i32.raw", np.array(uncond, dtype=np.int32))

    # ── transcribed small modules ──
    dd, ce, dit, voc = build_torch_modules(pack, cfg, device)

    n_frames = args.frames
    if not args.skip_llm:
        llm = build_llm(pack, cfg, device)
        text_ids = torch.tensor([ids, uncond], dtype=torch.long, device=device)
        with torch.no_grad():
            emb = llm.model.embed_tokens(text_ids)
            out = llm.model(inputs_embeds=emb, use_cache=True)
            past, last_hidden = out.past_key_values, out.last_hidden_state[:, -1]
            dump("last_hidden.f32.raw", last_hidden.float().cpu().numpy())
            dump("logits0.f32.raw", llm.lm_head(last_hidden).float().cpu().numpy())

            vocab_mask = torch.ones(cfg["language_model"]["vocab_size"], dtype=torch.bool, device=device)
            vocab_mask[CODE_OFFSET : CODE_OFFSET + SEMANTIC_VOCAB] = False
            vocab_mask[AUDIO_END] = False

            # Greedy AR loop, frame-0 skip included (mirrors encoders.py:282).
            all_codes, frame_hiddens = [], []
            frame_index = 0
            while True:
                guided = guided_semantic_logits(llm.lm_head(last_hidden), vocab_mask)
                sampled = guided.argmax(dim=-1)  # greedy stand-in for _sample_top_k
                if int(sampled.item()) == AUDIO_END:
                    print(f"[ar] audio_end at frame {frame_index}", flush=True)
                    break
                semantic = (sampled - CODE_OFFSET).repeat(2)

                # depth loop, greedy (mirrors _generate_depth_codes)
                seq = [dd.projection(last_hidden).unsqueeze(1)]
                seq.append(dd.projection(llm.model.embed_tokens(semantic + CODE_OFFSET)).unsqueeze(1))
                codes, hidden_parts = [semantic], []
                for index in range(1, 8):
                    hidden = dd(torch.cat(seq, dim=1))[:, -1]
                    hidden_parts.append(hidden[:1])
                    logits = dd.audio_heads[index - 1](hidden)
                    guided_d = logits[1:2].float() + (logits[0:1].float() - logits[1:2].float()) * CFG_SCALE
                    code = guided_d.argmax(dim=-1).repeat(2)
                    codes.append(code)
                    if index < 7:
                        e = dd.audio_embeddings(code + (index - 1) * 1024)
                        seq.append(dd.projection(e).unsqueeze(1))
                frame_codes = torch.stack(codes, dim=1)  # [2,8]
                depth_hidden = torch.cat(hidden_parts, dim=-1)  # [1,28672]

                # frame 0's codes are dumped too — the replay must force them
                # to keep its state aligned even though the frame is not emitted
                all_codes.append(frame_codes[0].cpu().numpy())
                if frame_index > 0:
                    frame_hiddens.append(torch.cat((last_hidden[:1], depth_hidden), dim=-1))
                    if len(frame_hiddens) >= n_frames:
                        break

                feedback = llm.model.embed_tokens(frame_codes[:, :1] + CODE_OFFSET)
                offsets = (torch.arange(7, device=device) * 1024).unsqueeze(0)
                extra = dd.audio_embeddings(frame_codes[:, 1:] + offsets).sum(dim=1, keepdim=True)
                feedback = (feedback + extra) * 8**-0.5
                out = llm.model(inputs_embeds=feedback, past_key_values=past, use_cache=True)
                past, last_hidden = out.past_key_values, out.last_hidden_state[:, -1]
                frame_index += 1

            dump("ar_codes.i32.raw", np.stack(all_codes).astype(np.int32))
            hiddens = torch.stack(frame_hiddens, dim=1)  # [1,F,32768]
            dump("ar_hiddens.f32.raw", hiddens.float().cpu().numpy())
        del llm
    else:
        hiddens = torch.from_numpy(
            np.fromfile(os.path.join(args.out, "ar_hiddens.f32.raw"), dtype=np.float32)
        ).reshape(1, n_frames, 32768)

    # ── condition encoder / DiT / vocoder ──
    with torch.no_grad():
        cond = ce(hiddens.to(device))
        dump("cond_out.f32.raw", cond.float().cpu().numpy())
        latent_len = cond.shape[1]

        g = torch.Generator().manual_seed(7)
        lat = torch.randn(1, 128, latent_len, generator=g).to(device)
        dump("dit_lat.f32.raw", lat.float().cpu().numpy())
        for tag, tv in (("t0", 0.0), ("t05", 0.5)):
            t = torch.full((1,), tv, device=device)
            vel = dit(lat, t, cond)
            dump(f"dit_v_{tag}.f32.raw", vel.float().cpu().numpy())

        wav = voc(lat)
        dump("voc_wav.f32.raw", wav.float().cpu().numpy())

    meta = {
        "caption": CAPTION,
        "lyrics": LYRICS,
        "prompt": prompt,
        "frames": n_frames,
        "latent_len": latent_len,
        "n_text_ids": len(ids),
    }
    with open(os.path.join(args.out, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\ndone -> {args.out}", flush=True)
    print(f"export MUSIC3_TEST_MODEL={args.model}")
    print(f"export MUSIC3_FIXTURES={args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
