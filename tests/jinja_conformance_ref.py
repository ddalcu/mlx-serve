#!/usr/bin/env python3
"""Regenerate src/fixtures/jinja_conformance/*.json: Python Jinja2's renders, the references the Zig test
`jinja conformance` holds jinja.cpp to.

Reference environment, pinned exactly (refuses to run on anything else):
  Python 3.14.10, jinja2 3.1.6, markupsafe 3.0.3, transformers 5.12.1; the full lock is LOCK below.
  Templates compile through transformers' own chat-template environment (`_cached_compile_jinja_template`: its
  sandbox, trim_blocks, lstrip_blocks, loopcontrols, tojson, raise_exception, strftime_now), with ONE deviation:
  `finalize` prints None as "". jinja.cpp prints `{{ none }}` as nothing on purpose: the server sends `content: null`
  for a text-less tool-call turn, and Jinja2's "None" there is a prompt the model never saw.

  1. MLX_SERVE_JINJA_CONFORMANCE_DUMP=<abs dir> zig build test -Dtest-filter="jinja conformance"
     The host builds every template case the way a request is built and writes <dir>/<template>.inputs.json.
  2. python -I tests/jinja_conformance_ref.py <abs dir>
     Renders those inputs and tests/jinja_conformance_constructs.jsonl into src/fixtures/jinja_conformance/.

A server-built case that raises here is printed: the Zig test fails it, because a raise there is the server's silent
generic fallback.
"""

import hashlib
import json
import sys
from pathlib import Path

import jinja2
import markupsafe
import transformers
from transformers.utils.chat_template_utils import _cached_compile_jinja_template

PINS = {"python": "3.14.10", "jinja2": "3.1.6", "markupsafe": "3.0.3", "transformers": "5.12.1"}
LOCK = """
# uv venv --python 3.14.10 VENV && uv pip install --python VENV/bin/python -r <these lines>
annotated-doc==0.0.7
annotated-types==0.8.0
anyio==4.16.0
certifi==2026.11.20
click==8.5.1
filelock==4.3.2
fsspec==2026.9.0
h11==0.16.0
hf-xet==1.8.0
httpcore==1.0.9
httpx==0.29.0
huggingface-hub==1.33.0
idna==3.20
jinja2==3.1.6
markdown-it-py==4.3.0
markupsafe==3.0.3
mdurl==0.1.2
numpy==2.5.5
packaging==26.4
pydantic==2.14.4
pydantic-core==2.50.5
pygments==2.21.0
pyyaml==6.0.4
regex==2026.11.25
rich==15.0.0
safetensors==0.8.0
shellingham==1.5.4
tokenizers==0.22.2
tqdm==4.70.1
transformers==5.12.1
typer==0.29.3
typing-extensions==4.16.0
typing-inspection==0.4.5

"""

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "src" / "fixtures" / "jinja_conformance"
CONSTRUCTS = ROOT / "tests" / "jinja_conformance_constructs.jsonl"


def check_pins():
    have = {
        "python": ".".join(map(str, sys.version_info[:3])),
        "jinja2": jinja2.__version__,
        "markupsafe": markupsafe.__version__,
        "transformers": transformers.__version__,
    }
    if have != PINS:
        sys.exit(f"reference environment differs: want {PINS}, have {have}")


def compile_template(src):
    env = _cached_compile_jinja_template(src).environment
    env.finalize = lambda v: "" if v is None else v
    return env.from_string(src)


def render(template, ctx):
    try:
        return {"expected": template.render(**ctx)}
    except Exception as e:  # a raise is a result: the engine must raise too
        return {"raises": f"{type(e).__name__}: {e}"}


def write(name, doc):
    OUT.mkdir(parents=True, exist_ok=True)
    text = json.dumps(doc, ensure_ascii=False, indent=1) + "\n"
    (OUT / f"{name}.json").write_text(text, encoding="utf-8")


def fixtures(inputs_dir):
    raising = 0
    for path in sorted(Path(inputs_dir).glob("*.inputs.json")):
        doc = json.loads(path.read_text(encoding="utf-8"))
        name = doc["template"]
        src = (ROOT / "src" / "fixtures" / f"{name}.jinja").read_text(encoding="utf-8")
        if hashlib.sha256(src.encode("utf-8")).hexdigest() != doc["sha256"]:
            sys.exit(f"{name}: the dump is of another revision of the template; dump again")
        template = compile_template(src)
        cases = []
        for c in doc["cases"]:
            # jinja_render_chat's context: messages, add_generation_prompt, tools when sent, then the extra keys.
            ctx = {"messages": c["messages"], "add_generation_prompt": True}
            if c["tools"] is not None:
                ctx["tools"] = c["tools"]
            ctx.update(c["extra"])
            r = render(template, ctx)
            if c["server"] and "raises" in r:
                raising += 1
                print(f"server shape raises under Jinja2: {name}/{c['id']}: {r['raises']}", file=sys.stderr)
            cases.append({k: c[k] for k in ("id", "messages", "tools", "extra")} | r)
        write(name, {"template": name, "sha256": doc["sha256"], "cases": cases})
        print(f"{name}: {len(cases)} cases")
    return raising


def constructs():
    cases = []
    for line in CONSTRUCTS.read_text(encoding="utf-8").splitlines():
        c = json.loads(line)
        ctx = {"messages": [], "add_generation_prompt": False} | c["context"]
        try:
            template = compile_template(c["template"])
        except Exception as e:
            r = {"raises": f"{type(e).__name__}: {e}"}
        else:
            r = render(template, ctx)
        # the context as JSON text, so the test hands the engine exactly these bytes (1.0 stays a float)
        cases.append({"id": c["id"], "template": c["template"], "context": json.dumps(c["context"], ensure_ascii=False)} | r)
    write("constructs", {"cases": cases})
    print(f"constructs: {len(cases)} cases")


def main():
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    check_pins()
    raising = fixtures(sys.argv[1])
    constructs()
    if raising:
        print(f"{raising} server-built cases raise under Jinja2 (the Zig test fails them)", file=sys.stderr)


if __name__ == "__main__":
    main()
