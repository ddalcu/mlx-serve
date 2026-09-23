"""Dump request-serialization fixtures from Python (ground truth for `laya.pyJson`).

The Laya prompt tokenizes `json.dumps(state, ensure_ascii=False)`, and non-string
instructions as `json.dumps(instructions)`, so both must render exactly as Python does or
the model sees different tokens for the same request.

  cd /Users/sbusso/Code/dev/laya && uv run python mlx-serve-laya/mlx-serve/tests/dump_laya_numeric_fixtures.py [model_dir]

Writes tests/fixtures/laya/numeric_cases.json:
  cases       state JSON text, its `json.dumps` text, token ids + markers of a noul question over it
  ascii_cases non-string instructions as JSON text, their `json.dumps` text (ensure_ascii), token
              ids + markers of a noul question with them over the state "x"
  rejected    state JSON texts whose numbers Python would write as Infinity (not JSON)
"""
import glob, json, os, sys

from laya_mlx.common import build_sequence
from laya_mlx.tokenizer import Tokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "fixtures", "laya", "numeric_cases.json")
model_dir = sys.argv[1] if len(sys.argv) > 1 else sorted(glob.glob(os.path.expanduser(
    "~/.cache/huggingface/hub/models--aac6fef--laya-multilingual-mlx/snapshots/*")))[0]
cfg = json.load(open(os.path.join(model_dir, "rl_agent_config.json")))
tok = Tokenizer(os.path.join(model_dir, "tokenizer"))
MAX_LEN, HEAD_MAX_LEN = cfg.get("max_len", 512), cfg.get("head_max_len", 192)

QUESTION = {"type": "noul", "instructions": "Is x greater than 0.00002?"}

# JSON texts as a client would send them.
INPUTS = {
    "exp_small": '{"x": 1e-5}',
    "fixed_boundary": '{"x": 1e-4}',
    "exp_large": '{"x": 1e16}',
    "fixed_large": '{"x": 1e15}',
    "int_beyond_i64": '{"x": 12345678901234567890}',
    "float_integral": '{"x": 1.0}',
    "float_exp_integral": '{"x": 1E5}',
    "neg_zero": '{"x": -0.0}',
    "int_neg_zero": '{"x": -0}',
    "sum_tenths": '{"x": 0.30000000000000004}',
    "huge": '{"x": 1.5e300}',
    "min_subnormal": '{"x": 5e-324}',
    "nested": '{"order": {"items": [{"qty": 2, "price": 1e-05}], "rates": [1e16, 0.0001, -0.0], "ok": true, "note": null}}',
    "dup_keys": '{"k": 1, "j": 2, "k": 3}',
    "del_char": '{"k": "a\x7fb"}',
}
ASCII_INPUTS = {
    "del_raw": '["a\x7fb"]',
    "lone_low": '["\\udc00"]',
    "lone_in_key": '{"\\udfff": "a\\ud800b"}',
    "lone_high_then_pair": '["\\ud800\\ud83d\\ude00"]',
    "nul_next_to_lone": '["\\u0000\\ud800", "x\\u0000d800"]',
    "non_ascii": '{"é": "😀"}',
}
REJECTED = ['{"x": 1e999}', '[1, [2, -1e400]]']


def case(name, text, dumps, state, ins):
    ids, markers = build_sequence(tok, state, {"t": "noul", "ins": ins, "crit": None}, MAX_LEN, HEAD_MAX_LEN)
    return {"name": name, "json": text, "dumps": dumps, "ids": ids, "markers": markers}


cases = []
for name, text in INPUTS.items():
    state = json.loads(text)
    cases.append(case(name, text, json.dumps(state, ensure_ascii=False), state, QUESTION["instructions"]))
ascii_cases = []
for name, text in ASCII_INPUTS.items():
    ins = json.dumps(json.loads(text))
    ascii_cases.append(case(name, text, ins, "x", ins))
for text in REJECTED:
    assert "Infinity" in json.dumps(json.loads(text)), text

# One case per line keeps the file reviewable.
with open(OUT, "w") as f:
    f.write('{"question": %s,\n' % json.dumps(QUESTION))
    for key, rows in (("cases", cases), ("ascii_cases", ascii_cases)):
        f.write(' "%s": [\n  ' % key + ",\n  ".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n ],\n")
    f.write(' "rejected": %s}\n' % json.dumps(REJECTED))
print("wrote", OUT)
