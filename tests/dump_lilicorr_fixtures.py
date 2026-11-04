#!/usr/bin/env python3
"""Execute NVIDIA Model Optimizer's LiLiCorrHead on a tiny fp32 lattice.

Run with torch, transformers, numpy and safetensors installed:
  python tests/dump_lilicorr_fixtures.py

Only the unmodified LiLiCorrLayer/Head classes are compiled from the pinned
reference module; its unrelated DFlash/training imports are not needed.
The resulting fixture is hermetic and contains all weights and inputs.
"""

import ast
import json
from pathlib import Path
import urllib.request

import torch
from torch import nn
import torch.nn.functional as F
from safetensors.torch import save_file
from transformers.models.qwen3.modeling_qwen3 import Qwen3RMSNorm

REVISION = "6bf36980e1d73a46750c947b7ff4c3cc3fa9c34d"
SOURCE = f"https://raw.githubusercontent.com/NVIDIA/Model-Optimizer/{REVISION}/modelopt/torch/speculative/plugins/modeling_lilicorr.py"


def main():
    source = urllib.request.urlopen(SOURCE).read().decode()
    tree = ast.parse(source)
    tree.body = [node for node in tree.body if isinstance(node, ast.ClassDef)
                 and node.name in ("LiLiCorrLayer", "LiLiCorrHead")]
    scope = dict(torch=torch, nn=nn, F=F, _NORM_CLS=Qwen3RMSNorm,
                 _NUM_CANDIDATE_FEATURES=5)
    exec(compile(tree, SOURCE, "exec"), scope)

    torch.manual_seed(870)
    cfg = dict(model_hidden_size=12, hidden_size=8, num_layers=2,
               num_heads=2, mlp_ratio=2., block_size=4, candidate_topk=3,
               factor_dim=6, rms_norm_eps=1e-6, vector_eps=1e-4,
               logit_scale=8.)
    head = scope["LiLiCorrHead"](**cfg).eval()
    with torch.no_grad():
        for name, value in head.named_parameters():
            if "norm.weight" in name:
                value.copy_(1. + torch.randn_like(value) * .05)
            else:
                value.copy_(torch.randn_like(value) * .1)
    embedding = nn.Embedding(17, cfg["model_hidden_size"]).eval()
    anchor = torch.randn(1, 1, 1, cfg["model_hidden_size"])
    pass_hidden = torch.randn(1, 1, 3, cfg["model_hidden_size"])
    logits = torch.randn(1, 1, 3, 17) * 2.
    def run_case(logits):
        _, ids = torch.topk(logits, cfg["candidate_topk"], dim=-1)
        log_probs = torch.log_softmax(logits.float(), dim=-1).gather(-1, ids)
        tapped = {}
        hook = head.feature_mlp[0].register_forward_pre_hook(
            lambda module, args: tapped.update(features=args[0].detach().clone()))
        try:
            with torch.no_grad():
                start, pair = head(candidate_token_ids=ids, candidate_log_probs=log_probs,
                                   pass_hidden=pass_hidden, embed_tokens=embedding,
                                   anchor_hidden=anchor, anchor_valid=None)
                start, pair = head.compose_log_factors(start_scores=start, pair_scores=pair)
        finally:
            hook.remove()
        return {"logits": logits.reshape(1, 3, 17), "ids": ids.to(torch.int32).reshape(-1),
                "features": tapped["features"].reshape(-1), "first": start.reshape(-1), "pair": pair.reshape(-1)}

    confident_logits = torch.full_like(logits, -50.)
    confident_logits[0, 0, 0, :3] = torch.tensor([100., 0., -1.])
    confident_logits[0, 0, 1, :3] = torch.tensor([0., 90., -4.])
    confident_logits[0, 0, 2, :3] = torch.tensor([-5., 1., 110.])
    cases = {"fixture": run_case(logits), "fixture.confident": run_case(confident_logits)}
    tensors = {}
    for name, value in head.state_dict().items():
        if name in ("relative_slot_bias", "same_slot_bias"):
            continue
        name = name.replace("feature_mlp.0.", "feature_norm.")
        name = name.replace("feature_mlp.1.", "feature_mlp.up_proj.")
        name = name.replace("feature_mlp.3.", "feature_mlp.down_proj.")
        name = name.replace(".mlp.0.", ".mlp.up_proj.").replace(".mlp.2.", ".mlp.down_proj.")
        name = name.replace(".attn.in_proj_", ".attn.in_proj.")
        if name in ("slot_embedding", "rank_embedding"):
            value = value.reshape(-1, cfg["hidden_size"])
        tensors["lilicorr."+name] = value.detach().contiguous()
    tensors["lilicorr.attention_bias"] = head._attention_bias(1, device=torch.device("cpu"), dtype=torch.float32, topk=3).detach().contiguous()
    # Row zero is deliberately unrelated; the engine must skip the noise anchor.
    block_hidden = torch.cat((torch.full((1, 1, 12), 7.), pass_hidden.reshape(1, 3, 12)), dim=1)
    tensors.update({"fixture.embedding": embedding.weight.detach(), "fixture.anchor": anchor.reshape(1, 1, 12),
                    "fixture.pass_hidden": block_hidden})
    for prefix, values in cases.items():
        tensors.update({prefix+"."+key: value for key, value in values.items()})
    out = Path(__file__).resolve().parents[1]/"src"/"fixtures"/"lilicorr_fp32.safetensors"
    save_file(tensors, str(out), metadata={"reference": SOURCE, "torch": torch.__version__})
    print(json.dumps({"fixture": str(out), "reference": SOURCE, "config": cfg,
                      "first": cases["fixture"]["first"].tolist(), "pair": cases["fixture"]["pair"].tolist()}, indent=2))


if __name__ == "__main__":
    main()
