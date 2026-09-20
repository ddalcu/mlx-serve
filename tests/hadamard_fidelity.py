"""Prism Hadamard pack fidelity: the server's greedy logprobs against an f32
reference of the same pack (mlx_lm qwen3_5 + the pack's Hadamard transforms).

usage: hadamard_fidelity.py <pack dir> <port>

Prints top-1 agreement and the mean KL over each position's top-20 support.
A bf16 serve of Bonsai 2 scores KL ~1.7e-4; the pack's own f16 numerics ~3e-6.
"""
import json
import math
import sys
import urllib.request

import mlx.core as mx
import mlx_lm.models.qwen3_5 as q
import numpy as np
from mlx import nn
from tokenizers import Tokenizer

PACK, PORT = sys.argv[1], sys.argv[2]
PROMPTS = [
    "Write a Python function that merges two sorted lists into one sorted list, with a short docstring.",
    "A train leaves at 3:40 pm and travels 210 km at 84 km/h. When does it arrive? Show the steps briefly.",
    "Explain in one paragraph why the sky looks blue during the day but red at sunset.",
    "List five practical tips for writing clear commit messages.",
]


def post(path, body):
    r = urllib.request.Request(f"http://127.0.0.1:{PORT}{path}", data=json.dumps(body).encode(),
                               headers={"content-type": "application/json"})
    return json.load(urllib.request.urlopen(r, timeout=600))


def fwht(x, block, signs, inverse=False):
    shape, dtype = x.shape, x.dtype
    x = x.astype(mx.float32)
    if not inverse:
        x = x * signs
    x = mx.hadamard_transform(x.reshape(-1, block), scale=1 / math.sqrt(block)).reshape(shape)
    if inverse:
        x = x * signs
    return x.astype(dtype)


class Packed(nn.Module):
    def __init__(self, w, s, b, block, signs, embedding):
        super().__init__()
        self.weight, self.scales, self.biases, self.signs = w, s, b, signs
        self.block, self.embedding = block, embedding

    def __call__(self, x):
        if self.embedding:
            idx = x.reshape(-1)
            out = mx.dequantize(self.weight[idx], self.scales[idx], self.biases[idx], group_size=128, bits=2)
            return fwht(out.reshape(*x.shape, -1).astype(mx.float32), self.block, self.signs, inverse=True)
        return mx.quantized_matmul(fwht(x, self.block, self.signs), self.weight, self.scales.astype(x.dtype),
                                   self.biases.astype(x.dtype), transpose=True, group_size=128, bits=2)


def reference():
    cfg = json.load(open(f"{PACK}/config.json"))
    model = q.TextModel(q.TextModelArgs.from_dict(cfg["text_config"]))
    w = {k[len("language_model."):]: v for k, v in mx.load(f"{PACK}/model.safetensors").items()
         if k.startswith("language_model.")}
    for rec in cfg["modules"]:
        *path, leaf = rec["path"].split(".")
        parent = model
        for p in path:
            parent = parent[int(p)] if p.isdigit() else getattr(parent, p)
        n = rec["path"]
        setattr(parent, leaf, Packed(w.pop(n + ".weight"), w.pop(n + ".scales"), w.pop(n + ".biases"),
                                     rec["block"], w.pop(n + ".signs"), rec["embedding"]))
    model.load_weights([(k, v.astype(mx.float32)) for k, v in w.items()], strict=False)
    model.eval()
    return model


runs = []
for text in PROMPTS:
    prompt = f"<|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    pids = post("/tokenize", {"content": prompt})["tokens"]
    c = post("/v1/completions", {"model": "x", "prompt": prompt, "max_tokens": 160, "temperature": 0, "logprobs": 20})
    ch = c["choices"][0]
    ids = post("/tokenize", {"content": prompt + ch["text"]})["tokens"]
    if ids[:len(pids)] != pids or len(ids) != len(pids) + len(ch["logprobs"]["tokens"]):
        sys.exit(f"re-tokenized completion does not align: {text[:40]}")
    runs.append((pids, ids, ch["logprobs"]["top_logprobs"]))

tk = Tokenizer.from_file(f"{PACK}/tokenizer.json")
by_text = {}
for i in range(tk.get_vocab_size()):
    by_text.setdefault(tk.decode([i], skip_special_tokens=False), []).append(i)

model = reference()
top1, kls = [], []
for pids, ids, tops in runs:
    logits = model(mx.array([ids], dtype=mx.int32))[0].astype(mx.float32)
    gold = np.array(logits - mx.logsumexp(logits, axis=-1, keepdims=True))
    for j, top in enumerate(tops):
        g = gold[len(pids) - 1 + j]
        ours = {by_text[s][0]: lp for s, lp in top.items() if len(by_text.get(s, [])) == 1}
        if not ours:
            continue
        top1.append(max(ours, key=ours.get) == int(np.argmax(g)))
        sup = np.array(list(ours))
        ol = np.array([ours[t] for t in sup])
        gl = g[sup]
        ol -= np.logaddexp.reduce(ol)
        gl -= np.logaddexp.reduce(gl)
        kls.append(float(np.sum(np.exp(gl) * (gl - ol))))
print(f"TOP1={100 * np.mean(top1):.2f} KL={np.mean(kls):.3e} N={len(top1)}")
