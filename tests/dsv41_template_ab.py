#!/usr/bin/env python3
"""Byte-exact A/B: a DeepSeek-V4.1 chat template vs the release's own encoder
(`encoding/encoding.py`), over the shapes the server emits.

pipenetwork's packs ship no template, so the server falls back to
`src/fixtures/dsv41_chat_template.jinja`; this pins that file (or any
candidate given with --template) against the reference, and with --dump
writes the cases plus the reference bytes for the Zig test that renders the
same template through jinja.cpp (`chat: the embedded V4.1 template ...`).

  python3 tests/dsv41_template_ab.py --encoding <DeepSeek-V4.1-Flash>/encoding \\
      [--template FILE] [--dump src/fixtures/dsv41_template_cases.json]

The server hands the template tool-call `arguments` as OBJECTS, omits
`reasoning_content` when absent, and passes `enable_thinking`, `thinking_mode`
and the effort word it mapped (`chat.dsv4EffortFor`: low|high|max).
"""
import argparse
import importlib.util
import json
import os
import sys

TEMPLATE = os.path.join(os.path.dirname(__file__), "..", "src", "fixtures", "dsv41_chat_template.jinja")

TOOLS = [{"type": "function", "function": {"name": "get_weather", "description": "Get weather", "parameters": {
    "type": "object", "properties": {"city": {"type": "string"}, "days": {"type": "integer"}}, "required": ["city"]}}}]
SIMPLE = [{"role": "user", "content": "Weather in Paris?"}]
WITH_SYSTEM = [{"role": "system", "content": "You are helpful."}, {"role": "user", "content": "Weather in Paris?"}]
TOOL_ROUND = WITH_SYSTEM + [
    {"role": "assistant", "content": "", "reasoning_content": "Need the tool.",
     "tool_calls": [{"id": "tc_0", "type": "function", "function": {"name": "get_weather", "arguments": {"city": "Paris", "days": 3}}}]},
    {"role": "tool", "content": "Sunny, 22C", "tool_call_id": "tc_0"},
]
HISTORY = [
    {"role": "user", "content": "Why is the sky blue?"},
    {"role": "assistant", "content": "Rayleigh scattering.", "reasoning_content": "Shorter wavelengths scatter more."},
    {"role": "user", "content": "And sunsets?"},
]
PARALLEL = [
    {"role": "user", "content": "Weather in Paris and Rome?"},
    {"role": "assistant", "content": "", "tool_calls": [
        {"id": "a", "type": "function", "function": {"name": "get_weather", "arguments": {"city": "Paris"}}},
        {"id": "b", "type": "function", "function": {"name": "get_weather", "arguments": {"city": "Rome"}}}]},
    {"role": "tool", "content": "Paris: sunny, 22C", "tool_call_id": "a"},
    {"role": "tool", "content": "Rome: cloudy, 19C", "tool_call_id": "b"},
]

# (label, messages, tools, thinking_mode, reasoning_effort)
CASES = [
    ("simple/chat", SIMPLE, None, "chat", "low"),
    ("simple/thinking low", SIMPLE, None, "thinking", "low"),
    ("simple/thinking high", SIMPLE, None, "thinking", "high"),
    ("simple/thinking max", SIMPLE, None, "thinking", "max"),
    ("system+tools/chat", WITH_SYSTEM, TOOLS, "chat", "low"),
    ("system+tools/thinking", WITH_SYSTEM, TOOLS, "thinking", "high"),
    ("tools, no system/thinking", SIMPLE, TOOLS, "thinking", "high"),
    ("tool round/chat", TOOL_ROUND, TOOLS, "chat", "low"),
    ("tool round/thinking", TOOL_ROUND, TOOLS, "thinking", "high"),
    ("history drops thinking/thinking", HISTORY, None, "thinking", "high"),
    ("history/chat", HISTORY, None, "chat", "low"),
    ("parallel calls/thinking", PARALLEL, TOOLS, "thinking", "high"),
]


def load_encoder(path):
    spec = importlib.util.spec_from_file_location("encoding_dsv41", os.path.join(path, "encoding.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["encoding_dsv41"] = mod
    spec.loader.exec_module(mod)
    return mod


def reference(enc, messages, tools, mode, effort):
    """Two input shapes differ from the server's (the outputs must agree):
    tools ride an (empty) leading system turn, and arguments are JSON strings."""
    msgs = json.loads(json.dumps(messages))
    if tools is not None:
        if msgs[0].get("role") == "system":
            msgs[0] = dict(msgs[0], tools=tools)
        else:
            msgs.insert(0, {"role": "system", "content": "", "tools": tools})
    for m in msgs:
        for tc in m.get("tool_calls") or []:
            if isinstance(tc["function"].get("arguments"), dict):
                tc["function"]["arguments"] = json.dumps(tc["function"]["arguments"], ensure_ascii=False)
    return enc.encode_messages(msgs, thinking_mode=mode, reasoning_effort=effort)


def context(messages, tools, mode, effort):
    # `chat.serializeExtraContext` sends both switches; a template reads either.
    return {"messages": [{k: v for k, v in m.items() if v is not None} for m in messages], "tools": tools,
            "add_generation_prompt": True, "enable_thinking": mode == "thinking", "thinking_mode": mode,
            "reasoning_effort": effort}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoding", required=True, help="dir holding the release's encoding.py")
    ap.add_argument("--template", default=TEMPLATE)
    ap.add_argument("--dump", help="write the cases and reference bytes as JSON")
    ap.add_argument("--show", action="store_true")
    a = ap.parse_args()
    enc = load_encoder(os.path.expanduser(a.encoding))
    import jinja2
    env = jinja2.Environment(trim_blocks=False, lstrip_blocks=False, extensions=["jinja2.ext.loopcontrols"])
    env.policies["json.dumps_kwargs"] = {"sort_keys": False, "ensure_ascii": False}
    env.globals["raise_exception"] = lambda msg: (_ for _ in ()).throw(RuntimeError(msg))
    tpl = env.from_string(open(os.path.expanduser(a.template)).read())
    failures, dump = 0, []
    for label, messages, tools, mode, effort in CASES:
        want = reference(enc, messages, tools, mode, effort)
        ctx = context(messages, tools, mode, effort)
        dump.append({"label": label, "messages": ctx["messages"], "tools": tools, "thinking_mode": mode,
                     "reasoning_effort": effort, "expected": want})
        got = tpl.render(**ctx)
        if got == want:
            print(f"PASS  {label}")
            continue
        failures += 1
        i = next((i for i in range(min(len(got), len(want))) if got[i] != want[i]), min(len(got), len(want)))
        print(f"FAIL  {label}: first diff at byte {i}")
        print(f"      template : {got[max(0, i - 60):i + 60]!r}")
        print(f"      reference: {want[max(0, i - 60):i + 60]!r}")
        if a.show:
            print(got, "\n----\n", want)
    print(f"\n{len(CASES) - failures}/{len(CASES)} byte-exact")
    if a.dump:
        with open(a.dump, "w") as f:
            json.dump(dump, f, indent=1, ensure_ascii=False)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
