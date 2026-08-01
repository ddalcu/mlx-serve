#!/usr/bin/env python3
"""Byte-exact A/B: our dsv4 chat template vs the release's own encoder.

The checkpoint ships NO chat_template, so `src/fixtures/dsv4_chat_template.jinja`
is our transcription of `encoding/encoding_dsv4.py` — and the converter injects
it into every mirror we build. A transcription is only worth what it is pinned
against, so this renders both sides over the shapes the server actually emits
and demands they agree BYTE FOR BYTE.

  python3 tests/dsv4_template_ab.py \
      --encoding ~/.mlx-serve/staging/DeepSeek-V4-Flash-0731/encoding

Every case here mirrors what `chat.serializeMessagesJson` +
`serializeExtraContext` hand the Jinja renderer: tool-call `arguments` as
OBJECTS (never JSON strings), `reasoning_content` present only when the client
returned it, and `thinking_mode` / `reasoning_effort` as the extra context.
"""
import argparse
import importlib.util
import json
import os
import sys

TEMPLATE = os.path.join(os.path.dirname(__file__), "..", "src", "fixtures", "dsv4_chat_template.jinja")

TOOLS = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get weather",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}, "days": {"type": "integer"}},
            "required": ["city"],
        },
    },
}]

SIMPLE = [{"role": "user", "content": "Weather in Paris?"}]
WITH_SYSTEM = [
    {"role": "system", "content": "You are helpful."},
    {"role": "user", "content": "Weather in Paris?"},
]
TOOL_ROUND = [
    {"role": "system", "content": "You are helpful."},
    {"role": "user", "content": "Weather in Paris?"},
    {"role": "assistant", "content": "", "reasoning_content": "Need the tool.",
     "tool_calls": [{"id": "tc_0", "type": "function",
                     "function": {"name": "get_weather", "arguments": {"city": "Paris", "days": 3}}}]},
    {"role": "tool", "content": "Sunny, 22C", "tool_call_id": "tc_0"},
]
HISTORY = [
    {"role": "user", "content": "Why is the sky blue?"},
    {"role": "assistant", "content": "Rayleigh scattering.",
     "reasoning_content": "Shorter wavelengths scatter more."},
    {"role": "user", "content": "And sunsets?"},
]
PARALLEL = [
    {"role": "user", "content": "Weather in Paris and Rome?"},
    {"role": "assistant", "content": "",
     "tool_calls": [
         {"id": "a", "type": "function", "function": {"name": "get_weather", "arguments": {"city": "Paris"}}},
         {"id": "b", "type": "function", "function": {"name": "get_weather", "arguments": {"city": "Rome"}}},
     ]},
    # Both results come back before the next turn — the real agent shape, and
    # the one that exercises consecutive-tool_result merging. (A conversation
    # that ENDS on an assistant turn while still asking for a generation
    # prompt is not a shape the server produces, and the reference has its own
    # `wo_eos` continuation path for it.)
    {"role": "tool", "content": "Paris: sunny, 22C", "tool_call_id": "a"},
    {"role": "tool", "content": "Rome: cloudy, 19C", "tool_call_id": "b"},
]

def dsv4_effort_for(effort):
    """Mirror of `chat.dsv4EffortFor` (keep in sync — the Zig unit test pins
    the real mapping): the server maps the CLIENT's OpenAI-vocabulary effort
    onto DeepSeek's low|high|max before the render, so the byte-pin must
    cover the MAPPED values a real request produces, not just literals."""
    if effort == "high":
        return "high"
    if effort in ("xhigh", "max"):
        return "max"
    return "low"


# (label, messages, tools, thinking_mode, reasoning_effort)
CASES = [
    ("simple/chat", SIMPLE, None, "chat", None),
    ("simple/thinking", SIMPLE, None, "thinking", None),
    ("system+tools/chat", WITH_SYSTEM, TOOLS, "chat", None),
    ("system+tools/thinking", WITH_SYSTEM, TOOLS, "thinking", None),
    ("tool round/chat", TOOL_ROUND, TOOLS, "chat", None),
    ("tool round/thinking", TOOL_ROUND, TOOLS, "thinking", None),
    ("history drop-thinking/chat", HISTORY, None, "chat", None),
    ("history drop-thinking/thinking", HISTORY, None, "thinking", None),
    ("parallel calls/chat", PARALLEL, TOOLS, "chat", None),
    # 0731's three-level effort: low adds nothing, high/max prepend a preamble
    # at index 0, and NEITHER applies outside thinking mode.
    ("effort low/thinking", WITH_SYSTEM, None, "thinking", "low"),
    ("effort high/thinking", WITH_SYSTEM, None, "thinking", "high"),
    ("effort max/thinking", WITH_SYSTEM, None, "thinking", "max"),
    ("effort high/chat (no-op)", WITH_SYSTEM, None, "chat", "high"),
    ("effort max+tools/thinking", WITH_SYSTEM, TOOLS, "thinking", "max"),
    # Client-vocabulary strings as the server actually maps them: a request
    # sending "xhigh"/"medium" reaches the template as "max"/"low".
    ("client xhigh→max/thinking", WITH_SYSTEM, None, "thinking", dsv4_effort_for("xhigh")),
    ("client medium→low/thinking", WITH_SYSTEM, None, "thinking", dsv4_effort_for("medium")),
    ("client high→high+tools/thinking", WITH_SYSTEM, TOOLS, "thinking", dsv4_effort_for("high")),
]


def load_encoder(path):
    spec = importlib.util.spec_from_file_location("encoding_dsv4", os.path.join(path, "encoding_dsv4.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["encoding_dsv4"] = mod
    spec.loader.exec_module(mod)
    return mod


def render_ours(env_tpl, messages, tools, thinking_mode, reasoning_effort):
    # The server hands the template messages whose tool-call arguments are
    # OBJECTS and omits `reasoning_content` entirely when it is absent.
    msgs = []
    for m in messages:
        d = {k: v for k, v in m.items() if v is not None}
        msgs.append(d)
    ctx = {
        "messages": msgs,
        "tools": tools,
        "add_generation_prompt": True,
        "thinking_mode": thinking_mode,
    }
    if reasoning_effort is not None:
        ctx["reasoning_effort"] = reasoning_effort
    return env_tpl.render(**ctx)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoding", required=True, help="dir holding encoding_dsv4.py")
    ap.add_argument("--show", action="store_true", help="print both renders on mismatch")
    args = ap.parse_args()
    enc = load_encoder(os.path.expanduser(args.encoding))

    import jinja2
    src = open(os.path.abspath(TEMPLATE)).read()
    env = jinja2.Environment(trim_blocks=False, lstrip_blocks=False)
    env.policies["json.dumps_kwargs"] = {"sort_keys": False}
    tpl = env.from_string(src)

    failures = 0
    for label, messages, tools, mode, effort in CASES:
        ours = render_ours(tpl, messages, tools, mode, effort)
        kwargs = {}
        if effort is not None:
            kwargs["reasoning_effort"] = effort
        theirs_msgs = json.loads(json.dumps(messages))
        # Two shape conversions, both because the reference's inputs differ
        # from what our server hands the template — the OUTPUTS are what must
        # agree:
        #  1. Tools ride the FIRST message (`msg.get("tools")`) and are only
        #     rendered from a system/developer turn — attached to a user
        #     message the reference SILENTLY DROPS them. Its canonical way to
        #     say "tools, no system prompt" is an empty system turn, which is
        #     why the template emits the `\n\n` separator unconditionally.
        #  2. `encode_arguments_to_dsml` json.loads() the arguments, i.e. the
        #     reference wants a JSON STRING; we serialize objects (the Inkling
        #     rule — a string there breaks other families). Feeding it a dict
        #     silently lands in its except branch and renders ONE parameter
        #     literally named "arguments".
        if tools is not None:
            if theirs_msgs[0].get("role") == "system":
                theirs_msgs[0] = dict(theirs_msgs[0], tools=tools)
            else:
                theirs_msgs.insert(0, {"role": "system", "content": "", "tools": tools})
        for msg in theirs_msgs:
            for tc in msg.get("tool_calls") or []:
                fn = tc["function"]
                if isinstance(fn.get("arguments"), dict):
                    fn["arguments"] = json.dumps(fn["arguments"])
        theirs = enc.encode_messages(theirs_msgs, thinking_mode=mode, **kwargs)
        if ours == theirs:
            print(f"PASS  {label}")
            continue
        failures += 1
        print(f"FAIL  {label}")
        if args.show:
            print("--- ours ---");  print(ours)
            print("--- theirs ---"); print(theirs)
        else:
            # First divergence, with a little context on both sides.
            i = next((i for i in range(min(len(ours), len(theirs))) if ours[i] != theirs[i]),
                     min(len(ours), len(theirs)))
            print(f"      first diff at byte {i}:")
            print(f"      ours  : {ours[max(0,i-60):i+60]!r}")
            print(f"      theirs: {theirs[max(0,i-60):i+60]!r}")
    print(f"\n{len(CASES)-failures}/{len(CASES)} byte-exact")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
