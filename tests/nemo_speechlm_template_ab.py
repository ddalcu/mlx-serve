#!/usr/bin/env python3
"""Pin src/fixtures/nemo_speechlm_chat_template.jinja against NVIDIA's own template.

  python -I tests/nemo_speechlm_template_ab.py <pack dir with the original chat_template.jinja>

Renders the shapes the server sends through the ORIGINAL template under Python jinja2 and writes the bytes to
src/fixtures/nemo_speechlm_template_cases.json; the Zig test `the embedded Nemotron Transcribe template renders
through jinja.cpp byte-exact` replays them through jinja.cpp and the embedded copy (which spells `["<|audio|>"] * n`
and `str.count` as loops and `split`: jinja.cpp has neither). Needs jinja2.
"""

import json
import sys
from pathlib import Path

import jinja2

A = {"type": "audio"}


def T(text):
    return {"type": "text", "text": text}


CALLS = [{"type": "function", "function": {"name": "f", "arguments": {"x": 1}}}]
CASES = {
    "audio then text": [{"role": "user", "content": [A, T("Transcribe it.")]}],
    "audio only": [{"role": "user", "content": [A]}],
    "two audios": [{"role": "user", "content": [A, A, T("Compare these.")]}],
    "text only list": [{"role": "user", "content": [T("hello"), T("world")]}],
    "plain string": [{"role": "user", "content": "hi there"}],
    "locator in text": [{"role": "user", "content": [A, T("What is this? <|audio|>")]}],
    "system and audio": [{"role": "system", "content": "Be brief."}, {"role": "user", "content": [A, T("Summarize.")]}],
    "vllm front locator": [{"role": "user", "content": "<|audio|>\nTranscribe."}],
    "history": [
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": "a1", "reasoning_content": "because"},
        {"role": "user", "content": [A, T("q2")]},
    ],
    "tool round trip": [
        {"role": "user", "content": "weather?"},
        {"role": "assistant", "content": "", "tool_calls": CALLS},
        {"role": "tool", "content": "sunny"},
    ],
}


def main():
    original = (Path(sys.argv[1]) / "chat_template.jinja").read_text()
    env = jinja2.Environment()

    def raise_exception(message):
        raise jinja2.exceptions.TemplateError(message)

    env.globals["raise_exception"] = raise_exception
    template = env.from_string(original)
    out = []
    for thinking in (False, True):
        for label, messages in CASES.items():
            kwargs = {} if thinking else {"enable_thinking": False}
            expected = template.render(messages=messages, add_generation_prompt=True, **kwargs)
            out.append({"label": f"{label}, thinking {'on' if thinking else 'off'}", "messages": messages, "thinking": thinking, "expected": expected})
    dest = Path(__file__).resolve().parent.parent / "src/fixtures/nemo_speechlm_template_cases.json"
    dest.write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n")
    print(f"wrote {len(out)} cases to {dest}")


if __name__ == "__main__":
    main()
