"""Dump Laya reference fixtures from laya_mlx (ground truth for the Zig port).

Run from the laya example dir that has laya_mlx installed:
  cd /Users/sbusso/Code/dev/laya && USE_TF=0 uv run mlx-serve-laya/mlx-serve/tests/dump_laya_fixtures.py

Writes tests/fixtures/laya/:
  cases.json        6 states x 3 questions: token ids, markers, qtype, expected answers
  encoder_en_q0.npy float32 [T, 768] final-norm encoder output for case en/department
  head_en_q0.npy    float32 [T, 768] decision-head output (after type_emb + head layers)
  logits_en_q0.npy  float32 [K] raw scorer logits; act_en_q0.npy float32 [2]
  tokenizer_cases.json  HF `tokenizers` ids + decode for out-of-vocab scripts (byte fallback)
"""
import json, os, sys
import numpy as np
import mlx.core as mx
os.environ.setdefault("USE_TF", "0")
import laya_mlx
from laya_mlx.agent import collate_items

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "fixtures", "laya")
os.makedirs(OUT, exist_ok=True)

QUESTIONS = {
    "department": {"type": "choice", "instructions": "Which team should handle this?",
                   "criteria": {"billing": "invoices, payments, refunds", "technical": "bugs and outages", "sales": "pricing"}},
    "urgency": {"type": "score", "instructions": "How urgent is this?", "criteria": ["not urgent", "soon", "blocking"]},
    "churn_risk": {"type": "noul", "instructions": "Does the user threaten to cancel?"},
}
STATES = {
    "en": {"body": "I was charged twice for March. Please refund the duplicate."},
    "fr": {"body": "J'ai été facturé deux fois en mars. Merci de rembourser le doublon."},
    "hi": {"body": "मुझसे मार्च के लिए दो बार शुल्क लिया गया। कृपया डुप्लिकेट वापस करें।"},
    # Characters with no vocab entry (CJK ext-B, Buginese, Cuneiform, a 2024
    # emoji): the tokenizer's byte fallback (<0xNN> tokens) is on the path.
    "cjk_extb": {"subject": "𠀋 invoice", "body": "Name field shows 𪚥 instead of my name. Fix it, not urgent."},
    "mixed_scripts": {"body": "Réservation ᨀᨕ 👨\u200d👩\u200d👧 double booked 🫩 — refund or I cancel! 東京→Paris"},
    "cuneiform": {"body": "𒀀𒁀 renders as boxes in the PDF export, blocking our release."},
}

# HF `tokenizers` reference for the byte-fallback unit test in src/tokenizer.zig.
TOKENIZER_TEXTS = {
    "emoji": "I love it 😀🎉🚀",
    "emoji_2024": "new face 🫩 and 🪾",
    "cjk_extb": "古文字 𠀋 𪚥 test",
    "khmer": "ភាសាខ្មែរ is Khmer",
    "buginese": "Buginese ᨀᨕ script",
    "cuneiform": "𒀀𒁀𒂀 clay",
    "zwj": "family 👨\u200d👩\u200d👧\u200d👦 flag 🏳️\u200d🌈",
    "mixed": "Réservation #42: 東京 → Paris 🗼 (ok?) ᨀᨕ 𠀋",
    "combining": "e\u0301 a\u0308 z\u0335",
    "control": "tab\there\x01x",
    "private_use": "pua \ue000\uf8ff end",
}

agent = laya_mlx.load("aac6fef/laya-multilingual-mlx")
print("model dir:", agent.model_dir)
cases = []
for lang, state in STATES.items():
    items, internal = agent.prepare(state, QUESTIONS)
    result = agent.predict(state, QUESTIONS)
    for qid, item in zip(QUESTIONS, items):
        cases.append({"lang": lang, "qid": qid, "state": state, "question": QUESTIONS[qid],
                      "ids": item["ids"], "markers": item["markers"], "qtype": item["qtype"],
                      "expected": result["answers"][qid]})
    print(lang, json.dumps(result["answers"], ensure_ascii=False))
json.dump({"model_dir": str(agent.model_dir), "questions": QUESTIONS, "states": STATES, "cases": cases},
          open(os.path.join(OUT, "cases.json"), "w"), ensure_ascii=False, indent=1)

from tokenizers import Tokenizer
hf_tok = Tokenizer.from_file(os.path.join(str(agent.model_dir), "tokenizer", "tokenizer.json"))
tok_cases = []
for name, text in TOKENIZER_TEXTS.items():
    enc = hf_tok.encode(text, add_special_tokens=False)
    tok_cases.append({"name": name, "text": text, "ids": enc.ids,
                      "byte_tokens": sum(1 for t in enc.tokens if t.startswith("<0x")),
                      "decoded": hf_tok.decode(enc.ids, skip_special_tokens=False)})
    print("tok", name, len(enc.ids), "byte tokens", tok_cases[-1]["byte_tokens"])
json.dump(tok_cases, open(os.path.join(OUT, "tokenizer_cases.json"), "w"), ensure_ascii=False, indent=1)

# Intermediate tensors for case 0 (en / department), batch of one.
items, _ = agent.prepare(STATES["en"], QUESTIONS)
batch = collate_items(items[:1], agent.tok.pad_token_id)
t = {k: mx.array(v) for k, v in batch.items()}
m = agent.model
enc = m.encoder(t["input_ids"], t["attention_mask"])
h = enc + m.type_emb(t["qtype"])[:, None, :]
h = m.head(h, t["attention_mask"][:, None, None, :].astype(mx.bool_))
logits, act = m(t["input_ids"], t["attention_mask"], t["marker_pos"], t["marker_mask"], t["qtype"])
mx.eval(enc, h, logits, act)
np.save(os.path.join(OUT, "encoder_en_q0.npy"), np.asarray(enc[0].astype(mx.float32)))
np.save(os.path.join(OUT, "head_en_q0.npy"), np.asarray(h[0].astype(mx.float32)))
np.save(os.path.join(OUT, "logits_en_q0.npy"), np.asarray(logits[0]))
np.save(os.path.join(OUT, "act_en_q0.npy"), np.asarray(act[0]))
print("ids", batch["input_ids"].shape, "markers", batch["marker_pos"][0], "logits", np.asarray(logits[0]), "act", np.asarray(act[0]))
print("wrote", OUT)
