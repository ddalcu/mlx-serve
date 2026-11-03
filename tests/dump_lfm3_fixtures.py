#!/usr/bin/env python3
"""Reference fixtures for the lfm3 plugin, from mlx-vlm's LFM3 implementation (Blaizzy/mlx-vlm, merge efd4b9f1).

The text trunk (`--out .../tiny`): a tiny random LFM3 (conv, full and sliding attention layers, zero-tail partial
RoPE, sandwich norms, a tied head) in the LiquidAI MLX pack layout, f32 on the CPU, and its logits over a prompt
that passes the sliding window, then a few decode steps through the reference's own caches.

LFM3-VL (`--vl .../tiny_vl`): the same trunk under a tiny random tower (gated conv and axial-RoPE attention layers)
and projector. A random image through the reference's Qwen3-VL image processor and a 3-frame clip through LFM3's
video rule: their RGB bytes, pixel values, tower rows, and the trunk's logits over a prompt with the image's rows.

LFM3-Audio (`--audio .../tiny_audio`, needs mlx-audio): the trunk in the Audio pack's layout (`lfm`, `encoder`,
`depthformer`) with a tiny random speech encoder (log-mel, three stride-2 convs, a windowed LFM2 trunk) and projector.
A 0.5 s clip: its samples, log-mel, the encoder's and projector's rows, and the trunk's logits over its rows.

The codec (`--codec <LFM3-Audio pack> <dir>`, real weights: the reference fixes the backbone's shape): 40 frames of
seeded codes and mlx-vlm's f32 `load_codec(...).decode` waveform, for the env-gated codec test. `--speech <pack> <dir>`:
the card's TTS and interleaved-chat recipes through `stream_slots`, greedy over a seeded reveal order.

  <venv with mlx-vlm>/bin/python -I tests/dump_lfm3_fixtures.py --out src/plugins/lfm3/fixtures/tiny \\
      --vl src/plugins/lfm3/fixtures/tiny_vl --audio src/plugins/lfm3/fixtures/tiny_audio
"""
import argparse
import json
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx.utils import tree_flatten
from mlx_vlm.models.lfm3 import lfm3 as ref_lfm3
from mlx_vlm.models.lfm3.config import ModelConfig

CONFIG = {
    "model_type": "lfm3",
    "architectures": ["Lfm3ForCausalLM"],
    "vocab_size": 128,
    "hidden_size": 32,
    "intermediate_size": 48,
    "num_hidden_layers": 5,
    "num_attention_heads": 2,
    "num_key_value_heads": 1,
    "head_dim": 16,
    "norm_eps": 1e-5,
    "conv_L_cache": 7,
    "conv_bias": False,
    "layer_types": ["conv", "full_attention", "sliding_attention", "conv", "full_attention"],
    "rope_parameters": {
        # Small thetas so every pair turns visibly within the prompt (the zero tail and the sliding base show).
        "full_attention": {"rope_theta": 10.0, "partial_rotary_factor": 0.5, "rope_type": "default"},
        "sliding_attention": {"rope_theta": 100.0, "rope_type": "default"},
    },
    "sliding_window": 4,
    "use_sandwich_norm": True,
    "tie_word_embeddings": True,
    "max_position_embeddings": 4096,
    "bos_token_id": 1,
    "eos_token_id": 2,
    "dtype": "float32",
}
PROMPT = [1, 17, 42, 125, 3, 99, 7, 80, 64, 5, 122, 31]  # 12 tokens: past the window of 4
DECODE = [77, 40, 9, 126]


def raw(path, arr):
    path.write_bytes(np.ascontiguousarray(arr).tobytes())


def randomize(model):
    # Norm weights away from 1 so a missing or misplaced norm shows.
    params = dict(tree_flatten(model.parameters()))
    for name, v in params.items():
        if name.endswith("norm.weight"):
            params[name] = 1.0 + 0.2 * mx.random.normal(v.shape)
        else:
            params[name] = 0.3 * mx.random.normal(v.shape)
    model.load_weights(list(params.items()))
    mx.eval(model.parameters())


VL = {
    "model_type": "lfm3_vl",
    "downsample_factor": 2,
    "projector_hidden_size": 48,
    "projector_hidden_act": "gelu",
    "projector_bias": True,
    "projector_layer_norm_eps": 1e-6,
    "image_token_id": 120,
    "video_token_id": 121,
    "vision_start_token_id": 122,
    "vision_end_token_id": 123,
    "eos_token_id": 2,
    "vision_config": {
        "model_type": "lfm3_vl_vision",
        "hidden_size": 32,
        "intermediate_size": 48,
        "num_hidden_layers": 3,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "in_channels": 3,
        "patch_size": 4,
        "temporal_patch_size": 2,
        "spatial_merge_size": 2,
        "conv_kernel_size": 3,
        "norm_eps": 1e-5,
        "layer_types": ["conv", "full_attention", "conv"],
        # A small base so every pair turns across the grid.
        "rope_parameters": {"rope_type": "axial", "rope_theta": 100.0},
    },
}
PROCESSOR = {
    "image_processor": {"patch_size": 4, "temporal_patch_size": 2, "merge_size": 2, "size": {"shortest_edge": 256, "longest_edge": 4096}},
    "video_processor": {"patch_size": 4, "temporal_patch_size": 2, "merge_size": 2, "fps": 2.0, "size": {"shortest_edge": 256, "longest_edge": 8192}},
}


def dump_vl(out):
    from PIL import Image
    from mlx_vlm.models.lfm3_vl import lfm3_vl as ref_vl
    from mlx_vlm.models.lfm3_vl.config import ModelConfig as VlConfig
    from mlx_vlm.models.lfm3_vl.processing_lfm3_vl import Lfm3VlVideoProcessor
    from mlx_vlm.models.qwen3_vl.processing_qwen3_vl import Qwen3VLImageProcessor

    out.mkdir(parents=True, exist_ok=True)
    mx.random.seed(11)
    rng = np.random.default_rng(11)
    cfg = dict(VL, text_config={k: v for k, v in CONFIG.items() if k != "architectures"})
    model = ref_vl.Model(VlConfig(**{k: v for k, v in cfg.items() if k in VlConfig.__dataclass_fields__}))
    randomize(model)

    ip = PROCESSOR["image_processor"]
    image = rng.integers(0, 256, (30, 46, 3), dtype=np.uint8)
    proc = Qwen3VLImageProcessor(patch_size=4, temporal_patch_size=2, merge_size=2, min_pixels=ip["size"]["shortest_edge"], max_pixels=ip["size"]["longest_edge"])
    data = proc(images=[Image.fromarray(image)])
    pixels = np.asarray(data["pixel_values"], dtype=np.float32)
    grid = np.asarray(data["image_grid_thw"]).tolist()[0]
    rows = model.encode_images(mx.array(pixels), image_grid_thw=np.array([grid]))
    n = rows.shape[0]
    # The text template's image part, its placeholder expanded to the rows.
    prompt = [1, 17, VL["vision_start_token_id"]] + [VL["image_token_id"]] * n + [VL["vision_end_token_id"], 42, 99, 5]
    logits = model(mx.array([prompt]), pixel_values=mx.array(pixels), image_grid_thw=np.array([grid]), cache=model.make_cache()).logits[0]

    vp = PROCESSOR["video_processor"]
    clip = rng.integers(0, 256, (3, 3, 18, 26), dtype=np.uint8)
    vproc = Lfm3VlVideoProcessor(patch_size=4, temporal_patch_size=2, merge_size=2, min_pixels=vp["size"]["shortest_edge"], max_pixels=vp["size"]["longest_edge"])
    vpixels, vgrid = vproc._process_one(clip)
    vrows = model.encode_images(mx.array(vpixels), image_grid_thw=np.array([vgrid]))
    mx.eval(rows, logits, vrows)

    mx.save_safetensors(str(out / "model.safetensors"), dict(tree_flatten(model.parameters())), metadata={"format": "mlx"})
    (out / "config.json").write_text(json.dumps(cfg, indent=1))
    (out / "processor_config.json").write_text(json.dumps(PROCESSOR, indent=1))
    raw(out / "image.rgb", image)
    raw(out / "image_pixels.f32", pixels)
    raw(out / "image_rows.f32", np.array(rows.astype(mx.float32)))
    raw(out / "prefill_logits.f32", np.array(logits.astype(mx.float32)))
    raw(out / "clip.rgb", clip.transpose(0, 2, 3, 1))
    raw(out / "clip_pixels.f32", np.asarray(vpixels, dtype=np.float32))
    raw(out / "clip_rows.f32", np.array(vrows.astype(mx.float32)))
    (out / "expected.json").write_text(json.dumps({"image": [30, 46], "image_grid": grid, "prompt": prompt, "clip": [3, 18, 26], "clip_grid": list(map(int, vgrid))}))
    print(f"wrote {out}: image grid {grid} -> {n} rows, clip grid {list(vgrid)} -> {vrows.shape[0]} rows")


AUDIO = {
    "model_type": "lfm3_audio",
    "codebooks": 3,
    "audio_vocab_size": 17,
    "maskgit_steps": 3,
    "interleaved_n_text": 6,
    "interleaved_n_audio": 12,
    "audio_token_id": 110,
    "audio_out_token_id": 111,
    "audio_start_token_id": 112,
    "text_end_token_id": 113,
    "mixed_start_token_id": 114,
    "eos_token_id": [2, 113],
    "encoder": {
        "dim_in": 128,
        "dim_conv": 16,
        "dim_out": 32,
        "sliding_window": 4,
        "lfm": {
            "model_type": "lfm2", "vocab_size": 2, "hidden_size": 32, "intermediate_size": 48, "block_auto_adjust_ff_dim": False,
            "conv_bias": False, "block_multiple_of": 16, "block_ffn_dim_multiplier": 1.0,
            "num_hidden_layers": 3, "num_attention_heads": 2, "num_key_value_heads": 1, "conv_L_cache": 3, "norm_eps": 1e-5,
            "max_position_embeddings": 1024, "rope_parameters": {"rope_theta": 100.0, "rope_type": "default"},
            "layer_types": ["conv", "full_attention", "conv"],
        },
    },
    "depthformer": {
        "model_type": "lfm3", "vocab_size": 1, "hidden_size": 32, "intermediate_size": 48, "num_hidden_layers": 2,
        "num_attention_heads": 2, "num_key_value_heads": 1, "head_dim": 16, "norm_eps": 1e-5, "max_position_embeddings": 12,
        "conv_L_cache": 7, "sliding_window": 256, "use_sandwich_norm": False, "tie_word_embeddings": True,
        "layer_types": ["full_attention", "full_attention"],
        "rope_parameters": {"full_attention": {"rope_type": "default", "rope_theta": 1000000.0, "partial_rotary_factor": 1.0}},
    },
}


def dump_audio(out):
    from mlx_vlm.models.lfm3_audio import lfm3_audio as ref_audio
    from mlx_vlm.models.lfm3_audio.audio import extract_features
    from mlx_vlm.models.lfm3_audio.config import ModelConfig as AudioConfig

    out.mkdir(parents=True, exist_ok=True)
    mx.random.seed(13)
    rng = np.random.default_rng(13)
    cfg = dict(AUDIO, lfm={k: v for k, v in CONFIG.items() if k != "architectures"})
    model = ref_audio.Model(AudioConfig(**{k: v for k, v in cfg.items() if k in AudioConfig.__dataclass_fields__}))
    randomize(model)

    t = np.arange(8000) / 16000
    clip = (0.3 * np.sin(2 * np.pi * 220 * t) + 0.2 * np.sin(2 * np.pi * 1330 * t) + 0.05 * rng.standard_normal(8000)).astype(np.float32)
    feats, mask = extract_features([clip])
    enc = model.audio_tower(feats, mask.sum(-1))
    rows = model.audio_projector(enc)
    n = rows.shape[0]
    prompt = [1, 17, 42] + [AUDIO["audio_token_id"]] * n + [99, 5]
    logits = model(mx.array([prompt]), input_features=feats, feature_attention_mask=mask, cache=model.make_cache()).logits[0]
    mx.eval(enc, rows, logits)

    # One frame from a hidden row, greedy over a fixed reveal order: plain, and with the per-codebook repetition
    # penalty over two earlier frames (the reference's MaskGIT is random-ordered even when greedy).
    from mlx_vlm.models.lfm3_audio.depth import embed_frames, sample_frame
    hidden = mx.random.normal((1, CONFIG["hidden_size"]))
    order = mx.array([[2, 0, 1]])
    history = mx.array([[[3, 9, 16], [5, 9, 2]]])
    plain = sample_frame(model.depth_decoder, hidden, model.config.maskgit_schedule, 0.0, None, None, None, order)[0]
    penalized = sample_frame(model.depth_decoder, hidden, model.config.maskgit_schedule, 0.0, None, 8.0, history, order)[0]
    frame_row = embed_frames(model.codebook_embedding, plain)
    mx.eval(plain, penalized, frame_row)

    mx.save_safetensors(str(out / "model.safetensors"), dict(tree_flatten(model.parameters())), metadata={"format": "mlx"})
    (out / "config.json").write_text(json.dumps(cfg, indent=1))
    raw(out / "depth_hidden.f32", np.array(hidden))
    raw(out / "frame_row.f32", np.array(frame_row.astype(mx.float32)))
    raw(out / "clip.f32", clip)
    raw(out / "features.f32", feats[0])
    raw(out / "encoder_rows.f32", np.array(enc.astype(mx.float32)))
    raw(out / "rows.f32", np.array(rows.astype(mx.float32)))
    raw(out / "prefill_logits.f32", np.array(logits.astype(mx.float32)))
    (out / "expected.json").write_text(json.dumps({"samples": len(clip), "frames": int(feats.shape[-1]), "rows": n, "prompt": prompt,
        "order": [2, 0, 1], "history": [[3, 9, 16], [5, 9, 2]], "frame": np.array(plain).tolist(), "frame_penalized": np.array(penalized).tolist()}))
    print(f"wrote {out}: {feats.shape[-1]} mel frames -> {n} rows")


def dump_codec(pack, out):
    from mlx_vlm.models.lfm3_audio.codec import load_codec

    out.mkdir(parents=True, exist_ok=True)
    # f32 is the truth: the reference's own bf16 decode sits at cos ~0.9989 of it. 40 frames outreach the
    # decoder's left context, so a chunked stream truncates it.
    codec = load_codec(pack, dtype=mx.float32)
    rng = np.random.default_rng(17)
    codes = rng.integers(0, codec.config.codebook_size, (codec.config.n_q, 40)).astype(np.int32)
    wave = codec.decode(mx.array(codes))
    mx.eval(wave)
    # Frame after frame, as the plugin's stream takes them.
    raw(out / "codes.u32", codes.T.astype(np.uint32))
    raw(out / "wave.f32", np.array(wave, dtype=np.float32))
    print(f"wrote {out}: {codes.shape[1]} frames -> {wave.shape[0]} samples")


SPEECH_CASES = {
    "tts": ("Perform TTS in American English using a female voice.", "Hello there!"),
    "chat": ("Respond with interleaved text and audio.", "What is the capital of France? Answer in one sentence."),
}


def dump_speech(pack, out, max_tokens=96):
    """The card's recipes through the reference's `stream_slots`, greedy text and speech over a seeded reveal order."""
    from mlx_vlm import load

    # The real pack decodes on the GPU, in the pack's own numerics (the tiny fixtures run f32 on the CPU).
    mx.set_default_device(mx.gpu)
    out.mkdir(parents=True, exist_ok=True)
    model, processor = load(pack)
    tok = processor.tokenizer
    order = np.stack([np.random.default_rng(i).permutation(model.config.num_codebooks) for i in range(max_tokens)])
    raw(out / "order.u32", order.astype(np.uint32))
    for name, (system, user) in SPEECH_CASES.items():
        msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        ids = list(tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True, return_dict=True)["input_ids"])
        tokens, codes = [], []
        for token, frame in model.stream_slots(mx.array([ids]), max_tokens=max_tokens, temperature=0.0,
                                               audio_temperature=0.0, frame_order=order):
            tokens.append(token)
            # An end-of-speech frame is no speech (the reference's generate_audio drops it too).
            if frame is not None and frame[0].item() != model.config.audio_eos_code:
                codes.append(np.array(frame))
        raw(out / f"{name}_prompt.u32", np.array(ids, dtype=np.uint32))
        raw(out / f"{name}_tokens.u32", np.array(tokens, dtype=np.uint32))
        raw(out / f"{name}_codes.u32", np.array(codes, dtype=np.uint32).reshape(-1))
        print(f"{name}: {len(ids)} prompt, {len(tokens)} slots, {len(codes)} frames: {tok.decode([t for t in tokens if t != model.config.audio_out_token_id])!r}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--vl")
    ap.add_argument("--audio")
    ap.add_argument("--codec", nargs=2, metavar=("PACK", "DIR"))
    ap.add_argument("--speech", nargs=2, metavar=("PACK", "DIR"))
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    mx.set_default_device(mx.cpu)
    mx.random.seed(7)

    cfg = ModelConfig(**{k: v for k, v in CONFIG.items() if k in ModelConfig.__dataclass_fields__})
    model = ref_lfm3.Model(cfg)
    # Norm weights away from 1 so a missing or misplaced norm shows.
    params = dict(tree_flatten(model.parameters()))
    for name, v in params.items():
        if name.endswith("norm.weight"):
            params[name] = 1.0 + 0.2 * mx.random.normal(v.shape)
        else:
            params[name] = 0.3 * mx.random.normal(v.shape)
    model.load_weights(list(params.items()))
    mx.eval(model.parameters())

    cache = model.make_cache()
    prefill = model(mx.array([PROMPT]), cache=cache).logits[0]
    steps = [model(mx.array([[t]]), cache=cache).logits[0, -1] for t in DECODE]
    expected = {"prefill_logits": prefill, "decode_logits": mx.stack(steps)}
    mx.eval(expected)

    # The pack as LiquidAI's MLX packs name it.
    mx.save_safetensors(str(out / "model.safetensors"), dict(tree_flatten(model.parameters())), metadata={"format": "mlx"})
    (out / "config.json").write_text(json.dumps(CONFIG, indent=1))
    # Raw little-endian f32 rows, so the Zig test embeds them as they are.
    for name, v in expected.items():
        (out / f"{name}.f32").write_bytes(bytes(memoryview(v.astype(mx.float32))))
    (out / "tokens.json").write_text(json.dumps({"prompt": PROMPT, "decode": DECODE}))
    print(f"wrote {out}: prefill {tuple(prefill.shape)}, decode {len(DECODE)} steps")
    if a.vl:
        dump_vl(Path(a.vl))
    if a.audio:
        dump_audio(Path(a.audio))
    if a.codec:
        dump_codec(a.codec[0], Path(a.codec[1]))
    if a.speech:
        dump_speech(a.speech[0], Path(a.speech[1]))


if __name__ == "__main__":
    main()
