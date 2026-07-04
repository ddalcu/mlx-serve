#!/usr/bin/env python3
"""Dump UniRig stage-1 SKELETON parity fixtures (.raw f32 / .raw i64) for the Zig
oracle tests in `src/unirig_skeleton.zig` (Phase 3, in progress).

USER-RUN (needs torch + transformers + the 1.44 GB skeleton ckpt + a clone of the
UniRig reference repo). Runs the REFERENCE PyTorch components on deterministic
synthetic inputs and writes byte-exact golden tensors that `zig build test`
compares the Zig port against, mirroring `tests/dump_hunyuan3d_fixtures.py`. Four
oracle taps (dossier §9):

  1 UNIRIG_ENC    : michelangelo encoder latents [1,1024,512] for a fixed synthetic
                    point cloud + normals + a FIXED FPS index set (the sampling RNG
                    is factored out so the oracle isolates the encoder attention math)
  2 UNIRIG_PREFIX : output_proj(latents) + embed([bos,cls]) assembly -> [1,1026,1024]
  3 UNIRIG_STEP   : one OPT forward on the prefix -> next-token logits [267] (raw,
                    pre-grammar-mask) at the last position
  4 UNIRIG_E2E    : full GREEDY grammar-masked decode -> token id sequence (i64),
                    the integration oracle (deterministic; compare a first-N prefix)

Reference-fidelity / determinism notes:
  - fp32 on CPU for EVERYTHING (CLAUDE.md gotcha "Parity fixtures for fp16-fragile
    giants"): the model is built float32; the fp16 MLX engine (fp32-accumulating
    matmuls) is compared at cosine > 0.99. The reference trainer's bf16-mixed
    autocast is NOT reproduced — the fp32-CPU run is the ground truth.
  - torch_cluster (FPS) and flash_attn are NOT installed (heavyweight/exotic). We
    (a) stub `torch_cluster` so the GPLv3 encoder module imports, then reconstruct
    the encoder `_forward` selection with a numpy seed-0 presample + a numpy
    farthest-point sampler (random_start=False → start index 0), calling the REAL
    loaded encoder submodules for the attention math; and (b) force the encoder's
    numerically-identical non-flash path (fp32 einsum softmax). The oracle DUMPS
    the selected FPS indices, so the Zig side loads them and never needs to
    reproduce numpy's PCG64 — encoder-math parity is decoupled from sampling RNG.
  - The reference GPLv3 encoder is IMPORTED at runtime from the user's clone (same
    pattern as dump_hunyuan3d_fixtures.py importing hy3dshape) — not shipped source.

Usage:
    source /Volumes/Sandisk_1TB/hy3d-scratch/venv/bin/activate
    python3 tests/dump_unirig_fixtures.py \
        --ckpt /Volumes/Sandisk_1TB/hy3d-scratch/unirig-ckpt/skeleton/articulation-xl_quantization_256/model.ckpt \
        --repo /Volumes/Sandisk_1TB/hy3d-scratch/UniRig \
        [--out tests/fixtures/unirig] [--num-points 8192] [--max-new 256] \
        [--test-model /Volumes/Sandisk_1TB/hy3d-scratch/unirig-fp16]

Prints the `export UNIRIG_*` env block for the Zig oracles.
"""

import argparse
import os
import sys
import types

import numpy as np

SEED = 0
# tokenizer constants (tokenizer_part.py:22-45 / config.json) — kept inline so the
# dump does not depend on the reference tokenizer (which pulls in get_order/yaml).
NUM_DISCRETE = 256
TOK_BRANCH, TOK_BOS, TOK_EOS, TOK_PAD, TOK_SPRING = 256, 257, 258, 259, 260
PARTS = {261, 262}                       # body, hand
TOK_CLS_NONE = 263
CLS_VALUES = {264, 265, 266}             # vroid, mixamo, articulationxl
CLS_ARTICULATIONXL = 266
VOCAB_SIZE = 267

# encoder config (mesh_encoder block of the AR yaml)
ENC_WIDTH = 512
ENC_HEADS = 8
TOKEN_NUM = 1024
PRESAMPLE = TOKEN_NUM * 4                 # 4096
NUM_LATENTS_ARG = 512                     # ShapeAsLatentPerceiverEncoder ctor arg (unused: no_query=True)

# OPT config (unirig_ar_350m_1024_81920_float32.yaml + facebook/opt-350m base)
AR_HIDDEN = 1024
AR_LAYERS = 24
AR_HEADS = 16
AR_FFN = 4096
AR_MAX_POS = 3076


def dump_f32(path, arr):
    a = arr.detach().float().cpu().numpy() if hasattr(arr, "detach") else np.asarray(arr)
    a = np.asarray(a, dtype="<f4")
    a.ravel().tofile(path)
    print(f"[dump] {os.path.basename(path):28s} shape={tuple(a.shape)}  {a.nbytes/1e6:.2f} MB")
    return a.shape


def dump_i64(path, arr):
    a = np.asarray(arr, dtype="<i8")
    a.ravel().tofile(path)
    print(f"[dump] {os.path.basename(path):28s} shape={tuple(a.shape)}  i64 len={a.size}")
    return a.shape


# ── deterministic sampling reconstruction (torch_cluster-free) ────────────────
def fps_numpy(pts, n_sample):
    """Farthest-point sampling, random_start=False (start index 0). Returns indices
    into `pts` [M,D]. Mirrors torch_cluster.fps(ratio=..., random_start=False); the
    Zig engine implements the identical greedy algorithm, and the oracle loads these
    indices, so exact torch_cluster tie-breaking is not required."""
    m = pts.shape[0]
    sel = np.empty(n_sample, dtype=np.int64)
    dist = np.full(m, np.inf, dtype=np.float64)
    far = 0
    for i in range(n_sample):
        sel[i] = far
        d = np.sum((pts - pts[far]) ** 2, axis=1)
        dist = np.minimum(dist, d)
        far = int(np.argmax(dist))
    return sel


def select_query_indices(pc):
    """Reproduce CrossAttentionEncoder._forward's Q-point selection (inference):
    seed-0 presample of PRESAMPLE points, then FPS ratio 1/4 -> TOKEN_NUM. Returns
    absolute indices into the full point cloud (length TOKEN_NUM)."""
    n = pc.shape[0]
    rng = np.random.default_rng(seed=0)
    pre_idx = rng.choice(n, PRESAMPLE, replace=PRESAMPLE > n)      # into full pc
    pre_pts = pc[pre_idx]
    fps_idx = fps_numpy(pre_pts, TOKEN_NUM)                        # into pre_pts
    return pre_idx[fps_idx]                                        # into full pc


# ── grammar FSM (tokenizer_part.next_posible_token, ported inline) ────────────
def next_possible_tokens(ids):
    """Legal next-token id list given the sequence so far (must start with bos).
    A verbatim port of TokenizerPart.next_posible_token — deterministic, hermetic."""
    if len(ids) == 0:
        return [TOK_BOS]
    state = "expect_bos"
    for tid in ids:
        if state == "expect_bos":
            assert tid == TOK_BOS, "ids do not start with bos"
            state = "expect_cls_or_part_or_joint"
        elif state == "expect_cls_or_part_or_joint":
            if tid < NUM_DISCRETE:
                state = "expect_joint_2"
            elif tid == TOK_CLS_NONE or tid in CLS_VALUES:
                state = "expect_part_or_joint"
            else:
                state = "expect_joint"
        elif state == "expect_part_or_joint":
            state = "expect_joint_2" if tid < NUM_DISCRETE else "expect_part_or_joint"
        elif state == "expect_joint_2":
            state = "expect_joint_3"
        elif state == "expect_joint_3":
            state = "expect_branch_or_part_or_joint"
        elif state == "expect_branch_or_part_or_joint":
            if tid == TOK_BRANCH:
                state = "expect_joint"
            elif tid < NUM_DISCRETE:
                state = "expect_joint_2"
            else:
                state = "expect_joint"
        elif state == "expect_joint":
            state = "expect_joint_2"
        else:
            raise AssertionError(state)
    s = []
    def add_cls():
        s.append(TOK_CLS_NONE); s.extend(sorted(CLS_VALUES))
    def add_part():
        s.append(TOK_SPRING); s.extend(sorted(PARTS))
    def add_joint():
        s.extend(range(NUM_DISCRETE))
    if state == "expect_bos":
        s.append(TOK_BOS)
    elif state == "expect_cls_or_part_or_joint":
        add_cls(); add_part(); add_joint()
    elif state == "expect_part_or_joint":
        add_part(); add_joint(); s.append(TOK_EOS)
    elif state in ("expect_joint_2", "expect_joint_3", "expect_joint"):
        add_joint()
    elif state == "expect_branch_or_part_or_joint":
        add_joint(); add_part(); s.append(TOK_BRANCH); s.append(TOK_EOS)
    else:
        raise AssertionError(state)
    return s


# ── synthetic point cloud (deterministic) ─────────────────────────────────────
def synth_cloud(n):
    """Seeded well-distributed cloud in ~[-0.9,0.9]^3 with unit outward normals
    (radial), so FPS + the encoder attention are non-degenerate."""
    rng = np.random.default_rng(SEED)
    dirs = rng.standard_normal((n, 3)).astype(np.float32)
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True) + 1e-9
    radii = (0.5 + 0.4 * rng.random((n, 1))).astype(np.float32)
    pts = (dirs * radii).astype(np.float32)
    normals = dirs.astype(np.float32)
    return pts, normals


def build_models(ckpt_path, repo):
    """Load the reference encoder (GPLv3, imported from the clone) + a stock HF OPT
    decoder + the output_proj, all fp32 CPU eval, with the ckpt weights loaded."""
    import torch

    # stub the two heavyweight/exotic deps the reference encoder imports but we do
    # not need: torch_cluster.fps (never called — fps_numpy replaces it) and
    # lightning.pytorch (only the unused PL base classes need it;
    # ShapeAsLatentModule subclasses nn.Module).
    if "torch_cluster" not in sys.modules:
        tc = types.ModuleType("torch_cluster")
        tc.fps = lambda *a, **k: (_ for _ in ()).throw(
            RuntimeError("torch_cluster.fps stubbed in the fixture dump; use fps_numpy"))
        sys.modules["torch_cluster"] = tc
    if "lightning" not in sys.modules:
        lit = types.ModuleType("lightning")
        litp = types.ModuleType("lightning.pytorch")
        litp.LightningModule = torch.nn.Module
        lit.pytorch = litp
        sys.modules["lightning"] = lit
        sys.modules["lightning.pytorch"] = litp
    sys.path.insert(0, repo)
    from src.model.michelangelo.models.tsal.sal_perceiver import ShapeAsLatentPerceiverEncoder
    from src.model.michelangelo.models.modules.transformer_blocks import (
        QKVMultiheadAttention, QKVMultiheadCrossAttention,
    )
    from transformers import OPTConfig, OPTForCausalLM

    sd = torch.load(ckpt_path, map_location="cpu", weights_only=True, mmap=True)
    sd = sd["state_dict"] if "state_dict" in sd else sd

    # --- encoder (mesh_encoder) ---
    enc = ShapeAsLatentPerceiverEncoder(
        device="cpu", dtype="float32", num_latents=NUM_LATENTS_ARG, point_feats=3,
        embed_dim=64, num_freqs=8, include_pi=False, heads=ENC_HEADS, width=ENC_WIDTH,
        num_encoder_layers=16, use_ln_post=True, init_scale=0.25, qkv_bias=False,
        use_checkpoint=False, flash=True, supervision_type="sdf", query_method=False,
        token_num=TOKEN_NUM,
    )
    enc_sd = {k[len("model.mesh_encoder."):]: v.float()
              for k, v in sd.items() if k.startswith("model.mesh_encoder.")}
    missing, unexpected = enc.load_state_dict(enc_sd, strict=False)
    # frequencies (non-persistent buffer) is legitimately missing; nothing else may be.
    hard_missing = [m for m in missing if not m.endswith("frequencies")]
    assert not hard_missing and not unexpected, (hard_missing, unexpected)
    enc = enc.eval()
    for m in enc.modules():                              # force fp32 einsum path
        if isinstance(m, (QKVMultiheadAttention, QKVMultiheadCrossAttention)):
            m.flash = False

    # --- OPT decoder (built offline; no network) ---
    cfg = OPTConfig(
        vocab_size=VOCAB_SIZE, hidden_size=AR_HIDDEN, num_hidden_layers=AR_LAYERS,
        ffn_dim=AR_FFN, max_position_embeddings=AR_MAX_POS, num_attention_heads=AR_HEADS,
        do_layer_norm_before=True, word_embed_proj_dim=AR_HIDDEN,
        activation_function="relu", dropout=0.0, attention_dropout=0.0, layerdrop=0.0,
        _remove_final_layer_norm=False, pad_token_id=TOK_PAD, bos_token_id=TOK_BOS,
        eos_token_id=TOK_EOS,
    )
    cfg._attn_implementation = "eager"
    opt = OPTForCausalLM(cfg)
    opt_sd = {k[len("model.transformer."):]: v.float()
              for k, v in sd.items() if k.startswith("model.transformer.")}
    missing, unexpected = opt.load_state_dict(opt_sd, strict=False)
    # lm_head is tied to embed_tokens -> may report as missing/handled; nothing else may.
    hard_missing = [m for m in missing if "lm_head" not in m]
    assert not hard_missing and not unexpected, (hard_missing, unexpected)
    opt = opt.to(torch.float32).eval()

    # --- output_proj ---
    op = torch.nn.Linear(ENC_WIDTH, AR_HIDDEN)
    op.weight.data = sd["model.output_proj.weight"].float()
    op.bias.data = sd["model.output_proj.bias"].float()
    op = op.eval()
    return enc, opt, op


def run_encoder(enc, pc_t, feats_t, q_idx):
    """Reconstruct CrossAttentionEncoder._forward with a FIXED q_idx selection,
    calling the real loaded submodules. KV = full cloud (use_full_input=True),
    Q = the q_idx points. Returns latents [1,1024,512]."""
    import torch
    ce = enc.encoder
    with torch.no_grad():
        data = ce.input_proj(torch.cat([ce.fourier_embedder(pc_t), feats_t], dim=-1))
        qi = torch.from_numpy(q_idx)
        sampled_pc = pc_t[:, qi]
        sampled_feats = feats_t[:, qi]
        sampled_data = ce.input_proj(torch.cat([ce.fourier_embedder(sampled_pc), sampled_feats], dim=-1))
        latents = ce.cross_attn(sampled_data, data)
        latents = ce.self_attn(latents)
        latents = ce.ln_post(latents)
    return latents


def main():
    ap = argparse.ArgumentParser(description="Dump UniRig stage-1 skeleton parity fixtures.")
    ap.add_argument("--ckpt", required=True, help="skeleton model.ckpt")
    ap.add_argument("--repo", required=True, help="UniRig reference clone (for the GPLv3 encoder import)")
    ap.add_argument("--out", default=os.path.join("tests", "fixtures", "unirig"))
    ap.add_argument("--num-points", type=int, default=8192,
                    help="synthetic cloud size (KV length of the encoder cross-attn; >=4096)")
    ap.add_argument("--max-new", type=int, default=256, help="greedy decode cap for the E2E oracle")
    ap.add_argument("--test-model", default=None, help="converted mlx-serve dir (echoed as UNIRIG_TEST_MODEL)")
    args = ap.parse_args()

    assert args.num_points >= PRESAMPLE, f"--num-points must be >= {PRESAMPLE}"
    import torch
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    out = os.path.abspath(args.out)
    os.makedirs(out, exist_ok=True)
    print(f"[dump] ckpt={args.ckpt}\n[dump] repo={args.repo}\n[dump] out={out}  N={args.num_points}")

    enc, opt, op = build_models(args.ckpt, args.repo)

    # deterministic inputs
    pc, normals = synth_cloud(args.num_points)
    q_idx = select_query_indices(pc).astype(np.int64)
    pc_t = torch.from_numpy(pc[None])            # [1,N,3]
    feats_t = torch.from_numpy(normals[None])    # [1,N,3]

    env = {}
    dump_f32(os.path.join(out, "unirig_pc.raw"), pc)
    dump_f32(os.path.join(out, "unirig_normals.raw"), normals)
    dump_i64(os.path.join(out, "unirig_qidx.raw"), q_idx)
    env["UNIRIG_PC"] = "unirig_pc.raw"
    env["UNIRIG_NORMALS"] = "unirig_normals.raw"
    env["UNIRIG_QIDX"] = "unirig_qidx.raw"
    env["UNIRIG_N"] = str(args.num_points)

    # ── oracle 1: encoder latents [1,1024,512] ──
    latents = run_encoder(enc, pc_t, feats_t, q_idx)
    assert tuple(latents.shape) == (1, TOKEN_NUM, ENC_WIDTH), tuple(latents.shape)
    dump_f32(os.path.join(out, "unirig_enc.raw"), latents[0])
    env["UNIRIG_ENC"] = "unirig_enc.raw"

    # ── oracle 2: prefix assembly [1,1026,1024] ──
    with torch.no_grad():
        mesh_tokens = op(latents)                                    # [1,1024,1024]
        start_ids = torch.tensor([[TOK_BOS, CLS_ARTICULATIONXL]], dtype=torch.long)
        start_emb = opt.get_input_embeddings()(start_ids)            # [1,2,1024]
        prefix = torch.cat([mesh_tokens, start_emb], dim=1)          # [1,1026,1024]
    assert tuple(prefix.shape) == (1, TOKEN_NUM + 2, AR_HIDDEN), tuple(prefix.shape)
    dump_f32(os.path.join(out, "unirig_prefix.raw"), prefix[0])
    env["UNIRIG_PREFIX"] = "unirig_prefix.raw"

    # ── oracle 3: one OPT forward -> next-token logits [267] (raw) ──
    with torch.no_grad():
        out0 = opt(inputs_embeds=prefix, use_cache=True)
    step_logits = out0.logits[0, -1].float()                        # [267]
    assert tuple(step_logits.shape) == (VOCAB_SIZE,), tuple(step_logits.shape)
    dump_f32(os.path.join(out, "unirig_step_logits.raw"), step_logits)
    env["UNIRIG_STEP_LOGITS"] = "unirig_step_logits.raw"

    # ── oracle 4: full greedy grammar-masked decode -> token ids (i64) ──
    tok_embed = opt.get_input_embeddings()
    grammar_seq = [TOK_BOS, CLS_ARTICULATIONXL]     # what the FSM sees (start_tokens + generated)
    generated = []
    past = out0.past_key_values
    logits = step_logits
    with torch.no_grad():
        for _ in range(args.max_new):
            allowed = next_possible_tokens(grammar_seq)
            mask = torch.full((VOCAB_SIZE,), float("-inf"))
            mask[allowed] = 0.0
            nxt = int(torch.argmax(logits + mask).item())
            generated.append(nxt)
            grammar_seq.append(nxt)
            if nxt == TOK_EOS:
                break
            nxt_emb = tok_embed(torch.tensor([[nxt]], dtype=torch.long))
            out_i = opt(inputs_embeds=nxt_emb, past_key_values=past, use_cache=True)
            past = out_i.past_key_values
            logits = out_i.logits[0, -1].float()
    full_seq = np.array([TOK_BOS, CLS_ARTICULATIONXL] + generated, dtype=np.int64)
    dump_i64(os.path.join(out, "unirig_e2e_tokens.raw"), full_seq)
    env["UNIRIG_E2E_TOKENS"] = "unirig_e2e_tokens.raw"
    env["UNIRIG_E2E_LEN"] = str(full_seq.size)
    n_bones = sum(1 for t in generated if t < NUM_DISCRETE) // 3   # rough (ignores branch 6-coords)
    print(f"[o4] greedy decode: {len(generated)} generated tokens "
          f"(eos={'yes' if generated and generated[-1]==TOK_EOS else 'no'}), ~{n_bones} coord-triples")

    # ── env block ──
    print("\n# ── paste to run the Zig oracle tests ──")
    parts = []
    if args.test_model:
        parts.append(f"UNIRIG_TEST_MODEL={os.path.abspath(os.path.expanduser(args.test_model))}")
    for k, v in env.items():
        parts.append(f"{k}={os.path.join(out, v) if v.endswith('.raw') else v}")
    print(" \\\n".join(parts) + " \\")
    print('  zig build test -Doptimize=ReleaseFast -Dtest-filter="unirig"')


if __name__ == "__main__":
    main()
