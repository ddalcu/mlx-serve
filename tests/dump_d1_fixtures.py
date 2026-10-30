"""Dump LiquidAI D1 reference fixtures from the card's own package (ground truth for src/plugins/d1/d1.zig).

  python -I tests/dump_d1_fixtures.py ~/.mlx-serve/models/LiquidAI/d1-3B

Loads the bf16 checkpoint on CPU in float32 with transformers (pinned 5.14.1, trust_remote_code: the card's own
modeling code). The card's prompt.py is loaded by file path, never imported from the model dir, so the run
reads nothing from the checkpoint that it executes except through the model class. Writes
tests/fixtures/d1/cases.json: per case the request, the prompt text and token ids the card builds, the readout
token groups, and the card's own answers. Refuses a dir whose config.json is not the card's LFM2-VL layout.
"""
import importlib.util
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "fixtures", "d1", "cases.json")
REPO = "LiquidAI/d1-3B"

CRIT = ["terrible, Pac-Man dies", "poor", "okay", "good, eats a pellet safely"]

CASES = [
    {
        "name": "demo_default",
        "state": "Refund me now or I cancel my subscription. Second time this month your app charged me twice.",
        "questions": {
            "team": {"type": "choice", "instructions": "Which team should handle this?",
                     "criteria": {"billing": None, "sales": None, "support": None}},
            "churn": {"type": "noul", "instructions": "Does the customer threaten to cancel?",
                      "criteria": {"false": "no threat", "true": "explicit threat"}},
            "urgency": {"type": "score", "instructions": "How urgent is this?",
                        "criteria": ["not urgent", "soon", "blocking"]},
        },
    },
    {
        "name": "described_choice_nested_state",
        "state": {"ticket": {"id": 1042, "text": "Payouts have failed for three days.\nLine two."}, "plan": "pro"},
        "questions": {
            "route": {"type": "choice", "instructions": "Which queue?",
                      "criteria": {"billing": "Payments, invoicing, refunds",
                                   "technical": "Bugs, outages, integrations",
                                   "not_urgent_queue": None}},
            "is_urgent": {"type": "noul", "instructions": "Does this convey urgency?"},
            "impact": {"type": "score", "instructions": "Rate the operational impact.",
                       "criteria": ["None", "Limited", "Critical"]},
        },
    },
    {
        "name": "pacman_directions",
        "state": {"map_around_you": ["#####", "#P.o#", "#####"], "heading": "left", "pellets_left": 12},
        "questions": {
            "up": {"type": "score", "instructions": "How good is it for Pac-Man to move up?", "criteria": CRIT},
            "left": {"type": "score", "instructions": "How good is it for Pac-Man to move left?", "criteria": CRIT},
        },
    },
    {
        "name": "string_state_many_options",
        "state": "Café “quoted” \\ tab\tend — 中文",
        "questions": {
            "lang": {"type": "choice", "instructions": "Which language is the ticket written in?",
                     "criteria": {f"lang_{i:02d}": None for i in range(30)}},
            "is_bug": {"type": "noul", "instructions": "Is it a bug?", "criteria": {}},
        },
    },
]


def load_card_prompt(model_dir):
    spec = importlib.util.spec_from_file_location("d1_card_prompt", os.path.join(model_dir, "prompt.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod  # dataclasses look their module up by name
    spec.loader.exec_module(mod)
    return mod


def revision():
    import urllib.request
    with urllib.request.urlopen(f"https://huggingface.co/api/models/{REPO}") as r:
        return json.load(r)["sha"]


def main():
    model_dir = os.path.abspath(sys.argv[1])
    with open(os.path.join(model_dir, "config.json")) as f:
        cfg = json.load(f)
    if cfg.get("auto_map", {}).get("AutoModel") != "modeling_d1.D1Model" or "quantization_config" in cfg:
        sys.exit("not the bf16 D1 checkpoint: config.json must name modeling_d1.D1Model and carry no quantization_config")

    import torch
    from transformers import AutoModel, AutoTokenizer

    card = load_card_prompt(model_dir)
    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModel.from_pretrained(model_dir, trust_remote_code=True, dtype=torch.float32).eval()
    bos, lead, style, system, option_style = tok.bos_token, "", card.DEFAULT_STATE_STYLE, card.DEFAULT_SYSTEM, "desc"

    out = []
    for case in CASES:
        state, questions = case["state"], case["questions"]
        ref = model.system_one(state, questions)
        prefix = card.prefix_text(tok, state, bos, style, system)
        prefix_ids = tok.encode(prefix, add_special_tokens=False)
        entry = {"name": case["name"], "state": state, "questions": questions, "prefix": prefix,
                 "prefix_ids": prefix_ids, "per_question": {}}
        total = len(prefix_ids)
        for qid, raw in questions.items():
            q = card.as_question(raw)
            labels = list(q.criteria.keys()) if isinstance(q, card.Choice) else []
            codes = card.aliases(tok, labels) if labels else []
            suffix = card.suffix_text(tok, q, lead, option_style)
            full = card.render(tok, state, q, bos, lead, style, system, option_style)
            suffix_ids = tok.encode(suffix, add_special_tokens=False)
            if full != prefix + suffix:
                sys.exit(f"{case['name']}/{qid}: render is not prefix + suffix")
            if tok.encode(full, add_special_tokens=False) != prefix_ids + suffix_ids:
                sys.exit(f"{case['name']}/{qid}: tokens of the whole prompt differ from prefix + suffix")
            total += len(suffix_ids)
            entry["per_question"][qid] = {
                "type": raw["type"],
                "suffix": suffix,
                "suffix_ids": suffix_ids,
                "codes": [[code, tid] for code, tid in codes],
                "groups": card.readout_ids(tok, q),
            }
        entry["input_tokens"] = total
        entry["answers"] = ref["answers"]
        out.append(entry)
        print(f"{case['name']}: {total} tokens, {len(questions)} questions")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump({"model": REPO, "revision": revision(), "precision": "float32 CPU", "cases": out},
                  f, indent=1, ensure_ascii=False)
        f.write("\n")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
