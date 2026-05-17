#!/usr/bin/env python3
"""
Repro probe: "agent suddenly stops mid-task with no apparent reason."

Drives a faithful copy of the Swift `ChatView.runAgentLoop` against a live
mlx-serve, with canned tool outputs so it runs entirely in-memory (no real
filesystem mutations). The same system prompt and tool definitions used by
the macOS app are embedded inline so the model sees an identical request to
what it gets in production.

What it watches for, per turn:
  * completion_tokens reported by the server (`usage`)
  * length of visible content the client sees
  * length of `reasoning_content` (if any)
  * finish_reason (last seen in the stream — useful for distinguishing
    `stop` / `length` / `tool_calls`)
  * whether any tool_calls parsed

The "agent stops" pattern is flagged when a turn has:
    receivedToolCalls is empty
    AND completion_tokens >= MIN_TOKENS_FOR_SUSPICION (default 100)
    AND len(content) + len(reasoning) < MIN_VISIBLE_CHARS (default 200)
    AND finish_reason != "length" (length is its own bug, retried elsewhere)

That's the exact shape from the user's report: hundreds of tokens generated
but only ~120 chars of visible content and no tool to execute.

Usage:
    python3 tests/test_agent_stop_repro.py [--port 11234] [--max-tokens 4096]
                                           [--max-turns 15] [--task ...]
                                           [--runs 1] [--out repro.json]

Exit code 0 if no suspicious stops detected, 1 otherwise. The full captured
chunk(s) get dumped to --out (default repro_<timestamp>.json) for analysis.
"""

import argparse
import json
import os
import sys
import time
import uuid
import urllib.request
import urllib.error


# ─── Embedded copies of the app's system prompt + tool definitions ────────
# Faithful reproduction of AgentPrompt.defaultPromptFile so the server
# receives the same prompt our production client sends.
SYSTEM_PROMPT = """# System

You are an autonomous macOS agent running on Apple Silicon. Act independently to complete tasks — do not ask the user for confirmation between steps. Execute multi-step tasks without pausing. Only respond to the user when the task is fully complete or if you hit a genuine ambiguity that cannot be resolved with tools.

# Using Your Tools

IMPORTANT: Use dedicated tools instead of shell equivalents:
- readFile instead of `cat`, `head`, `tail`
- writeFile instead of `echo >` or `cat <<EOF`
- editFile instead of `sed` or `awk`
- searchFiles instead of `grep` or `rg`
- listFiles instead of `find` or `ls -R`

Use shell only for: build/test commands, git operations, process management, installing packages, and commands with no dedicated tool equivalent. Shell commands run as a login shell — your PATH includes user tools (node, npm, python, brew, etc.).

# Workspace Confinement

All file operations are confined to the working directory. Use relative paths.
Tool arguments must be valid JSON: {"key": "value"}. Every tool call MUST include at least the required parameters.

# Output Style

- Be concise. Lead with actions, not reasoning
- Don't narrate what you're about to do — just do it
- When done, briefly summarize: what changed, which files, what to verify
"""

# Pre-serialized matching AgentPrompt.toolDefinitionsJSON. We strip a couple
# of tools we never need for this scenario (browse/webSearch/saveMemory) to
# keep request size down, but the ones the user's repro actually exercises
# (shell, cwd, listFiles, readFile, writeFile, editFile) are preserved
# verbatim including the order they appear in the production string.
TOOLS_JSON = r"""[
  {"type":"function","function":{"name":"shell","description":"Run a shell command. Commands run in the current working directory (use cwd tool to change it). Example: {\"command\": \"ls -la /tmp\"}","parameters":{"type":"object","properties":{"command":{"type":"string","description":"The shell command to execute"}},"required":["command"]}}},
  {"type":"function","function":{"name":"cwd","description":"Change the working directory for all subsequent tool calls. Example: {\"path\": \"myproject/src\"}","parameters":{"type":"object","properties":{"path":{"type":"string","description":"Directory path (relative to current working directory, or absolute)"}},"required":["path"]}}},
  {"type":"function","function":{"name":"writeFile","description":"Write content to a file (overwrites). Only for SMALL files (under 100 lines). Example: {\"path\": \"src/main.swift\", \"content\": \"hello\"}","parameters":{"type":"object","properties":{"path":{"type":"string","description":"File path relative to working directory"},"content":{"type":"string","description":"File content to write"}},"required":["path","content"]}}},
  {"type":"function","function":{"name":"readFile","description":"Read a file's contents with optional line range. Example: {\"path\": \"src/main.swift\"}","parameters":{"type":"object","properties":{"path":{"type":"string","description":"File path relative to working directory"},"startLine":{"type":"string","description":"First line to read (1-based)"},"endLine":{"type":"string","description":"Last line to read"}},"required":["path"]}}},
  {"type":"function","function":{"name":"editFile","description":"Edit a file. Two modes: line-based (path, startLine, endLine, replace) or text-based (path, find, replace). Always readFile first to see line numbers.","parameters":{"type":"object","properties":{"path":{"type":"string","description":"File path"},"startLine":{"type":"string","description":"First line to replace"},"endLine":{"type":"string","description":"Last line to replace"},"find":{"type":"string","description":"Exact text to find"},"replace":{"type":"string","description":"Replacement text"}},"required":["path"]}}},
  {"type":"function","function":{"name":"listFiles","description":"List files and directories. Returns paths matching optional glob. Example: {\"path\": \"src\", \"pattern\": \"*.swift\", \"recursive\": \"true\"}","parameters":{"type":"object","properties":{"path":{"type":"string","description":"Directory to list"},"pattern":{"type":"string","description":"Glob pattern to filter"},"recursive":{"type":"string","description":"If 'true', search recursively"}},"required":[]}}}
]"""


# ─── Canned tool execution ────────────────────────────────────────────────
class FakeFS:
    """Tiny in-memory filesystem so the model sees consistent state across
    turns. Tracks files written so subsequent readFile / listFiles reflect
    what the agent did — that's what keeps the loop on a realistic
    trajectory rather than the model giving up because tools 'don't work'.
    """

    def __init__(self):
        self.cwd = "/Users/david/.mlx-serve/workspace"
        self.files = {}   # path → content
        self.dirs = set([self.cwd])

    def _abs(self, path):
        if path.startswith("/"):
            return path
        return os.path.normpath(os.path.join(self.cwd, path))

    def cwd_tool(self, args):
        path = args.get("path") or "."
        new = self._abs(path)
        self.cwd = new
        self.dirs.add(new)
        return f"Changed working directory to {new}"

    # A realistic-sized `node_modules` tree so the model's context grows at the
    # same rate as a real `npx sv create` scaffold. The user's failing turn was
    # at ~4909 prompt tokens; with synthetic tiny outputs we plateau at half
    # that and never reach the failure threshold.
    _NODE_MODULES_LISTING = "\n".join([
        "svelte.config.js",
        "vite.config.js",
        "package.json",
        "package-lock.json",
        ".gitignore",
        "README.md",
        "src/app.html",
        "src/app.d.ts",
        "src/routes/+page.svelte",
        "src/routes/+layout.svelte",
        "static/favicon.png",
        "tsconfig.json",
        "node_modules",
    ] + [
        f"node_modules/{pkg}/{f}"
        for pkg in ("@sveltejs/kit", "@sveltejs/vite-plugin-svelte", "vite", "rolldown",
                    "svelte", "vitefu", "tinyglobby", "obug", "@types/cookie",
                    "@types/estree", "acorn", "esbuild", "magic-string",
                    "rollup", "source-map-js", "tslib", "typescript",
                    "@rollup/plugin-virtual", "@rollup/pluginutils",
                    "estree-walker", "is-reference", "locate-character",
                    "kleur", "mri", "sade", "set-cookie-parser",
                    "cookie", "devalue", "import-meta-resolve",
                    "sirv", "totalist", "@sveltejs/adapter-auto")
        for f in ("LICENSE", "README.md", "package.json", "src/index.js",
                  "src/index.d.ts", "dist/index.js", "dist/index.d.ts")
    ])

    def listFiles_tool(self, args):
        path = self._abs(args.get("path") or ".")
        rec = (args.get("recursive") == "true")
        # If recursive on a path that looks like a scaffolded project, return
        # the realistic-sized listing so the model's context grows the way it
        # does in production. Filter through written files so writes still
        # surface; pad with the canned node_modules tree.
        if rec and any(s in path for s in ("ds-swift-test-todo", "agent-stop-repro", "todo")):
            written = [p[len(self.cwd)+1:] if p.startswith(self.cwd + "/") else p
                       for p in sorted(self.files)
                       if p.startswith(path.rstrip("/") + "/") or p == path]
            return "\n".join(written + [self._NODE_MODULES_LISTING])
        prefix = path.rstrip("/") + "/"
        out = []
        if not any(f.startswith(prefix) for f in self.files):
            return "(empty directory)"
        for f in sorted(self.files):
            if f.startswith(prefix):
                rel = f[len(prefix):]
                if rec or "/" not in rel:
                    out.append(rel)
        return "\n".join(out) if out else "(empty directory)"

    def readFile_tool(self, args):
        path = self._abs(args.get("path") or "")
        if path in self.files:
            text = self.files[path]
            return "\n".join(f"{i+1}| {line}" for i, line in enumerate(text.splitlines()))
        return f"Error: no such file {path}"

    def writeFile_tool(self, args):
        path = self._abs(args.get("path") or "")
        content = args.get("content") or ""
        self.files[path] = content
        return f"wrote {len(content)} bytes to {path}"

    def editFile_tool(self, args):
        path = self._abs(args.get("path") or "")
        if path not in self.files:
            return f"Error: no such file {path}"
        text = self.files[path]
        if "find" in args and "replace" in args:
            self.files[path] = text.replace(args["find"], args["replace"], 1)
            return f"edited {path} (text mode)"
        if "startLine" in args and "endLine" in args and "replace" in args:
            lines = text.splitlines()
            try:
                s = int(args["startLine"]) - 1
                e = int(args["endLine"])
            except ValueError:
                return "Error: startLine/endLine must be integers"
            lines[s:e] = (args["replace"] or "").splitlines()
            self.files[path] = "\n".join(lines)
            return f"edited {path} (lines {s+1}..{e})"
        return "Error: provide find/replace OR startLine/endLine/replace"

    # Realistic chunky outputs (matches what `npx sv create` / `npm install`
    # actually print). Padded so prompt-token totals reach the same range
    # (~4-5k) where the user's bug originally fired.
    _NPX_SV_OUTPUT = (
        "(node:65112) Warning: Setting the NODE_TLS_REJECT_UNAUTHORIZED environment variable to '0' "
        "makes TLS connections and HTTPS requests insecure by disabling certificate verification.\n"
        "(Use `node --trace-warnings ...` to show where the warning was created)\n\n"
        "HINT: Run \"sv --help\" to get the full list of commands, add-ons, and examples to one-shot "
        "and skip interactive prompts.\n"
        "┌  Welcome to the Svelte CLI! (v0.15.3)\n│\n◆  Project created\n│\n"
        "│  To skip prompts next time, run:\n"
        "●  npx sv@0.15.3 create --template minimal --types ts --install npm ds-swift-test-todo\n│\n"
        "◇  Installing dependencies with npm...\n"
        + ("[2K[1A" * 60) +
        "│\n◆  Successfully installed dependencies with npm\n│\n"
        "◇  What's next? ───────────────────────────────╮\n│                                              │\n"
        "│  📁 Project steps                            │\n│                                              │\n"
        "│    1: cd ds-swift-test-todo                  │\n│    2: npm run dev -- --open                  │\n"
        "│                                              │\n│  To close the dev server, hit Ctrl-C         │\n"
        "│                                              │\n│  Stuck? Visit us at https://svelte.dev/chat  │\n"
        "├──────────────────────────────────────────────╯\n│\n└  You're all set!\n"
    )
    _NPM_INSTALL_OUTPUT = (
        "(node:66566) Warning: Setting the NODE_TLS_REJECT_UNAUTHORIZED environment variable to '0' "
        "makes TLS connections and HTTPS requests insecure by disabling certificate verification.\n"
        "(Use `node --trace-warnings ...` to show where the warning was created)\n\n"
        "added 97 packages, and audited 154 packages in 4s\n\n"
        "25 packages are looking for funding\n  run `npm fund` for details\n\n"
        "6 vulnerabilities (3 low, 3 moderate)\n\n"
        "To address issues that do not require attention, run:\n  npm audit fix\n\n"
        "To address all issues (including breaking changes), run:\n  npm audit fix --force\n\n"
        "Run `npm audit` for details.\n"
    )
    _PRISMA_INIT_OUTPUT = (
        "(node:66600) Warning: Setting the NODE_TLS_REJECT_UNAUTHORIZED environment variable to '0' "
        "makes TLS connections and HTTPS requests insecure by disabling certificate verification.\n"
        "(Use `node --trace-warnings ...` to show where the warning was created)\n\n"
        "✔ Your Prisma schema was created at prisma/schema.prisma\n\n"
        "    schema.prisma\n  prisma.config.ts\n  .env\n\n"
        "warn You already have a .gitignore file. Don't forget to add .env in it.\n\n"
        "Next, choose how you want to set up your database:\n\n"
        "CONNECT EXISTING DATABASE:\n  1. Configure your DATABASE_URL in prisma.config.ts\n"
        "  2. Run prisma db pull to introspect your database.\n\n"
        "CREATE NEW DATABASE:\n  Local: npx prisma dev (runs Postgres locally in your terminal)\n"
        "  Cloud: npx create-db (creates a free Prisma Postgres database)\n\n"
        "Then, define your models in prisma/schema.prisma and run prisma migrate dev to apply your schema.\n\n"
        "Learn more: https://pris.ly/getting-started\n"
    )

    def shell_tool(self, args):
        cmd = args.get("command", "")
        if "npx sv create" in cmd or ("sv@" in cmd and "create" in cmd):
            return self._NPX_SV_OUTPUT
        if "npm install" in cmd or "npm i " in cmd:
            return self._NPM_INSTALL_OUTPUT
        if "npx prisma init" in cmd:
            return self._PRISMA_INIT_OUTPUT
        if "npx prisma generate" in cmd:
            return "Generated Prisma Client to ./generated/prisma"
        if "npx prisma migrate" in cmd:
            return "Applied migration"
        if cmd.startswith("ls") or cmd.startswith("cd ") or cmd.startswith("mkdir"):
            return "(ok)"
        return f"(stdout for: {cmd[:60]})"


TOOLS_DISPATCH = {
    "cwd": FakeFS.cwd_tool,
    "listFiles": FakeFS.listFiles_tool,
    "readFile": FakeFS.readFile_tool,
    "writeFile": FakeFS.writeFile_tool,
    "editFile": FakeFS.editFile_tool,
    "shell": FakeFS.shell_tool,
}


def exec_tool(fs, name, args_json):
    """Execute one parsed tool call. Returns a string (tool result content)."""
    try:
        args = json.loads(args_json) if args_json else {}
    except json.JSONDecodeError as e:
        return f"Error: malformed JSON arguments: {e}"
    fn = TOOLS_DISPATCH.get(name)
    if fn is None:
        return f"Error: unknown tool {name!r}"
    try:
        return fn(fs, args)
    except Exception as e:
        return f"Error executing {name}: {e}"


# ─── Streaming SSE client ─────────────────────────────────────────────────
def stream_chat(host, port, body, timeout=600):
    """Yield parsed SSE chunks (dict) from a streaming chat completion call."""
    url = f"http://{host}:{port}/v1/chat/completions"
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        url, data=data, method="POST",
        headers={"Content-Type": "application/json", "Connection": "close"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        if resp.status != 200:
            body = resp.read().decode("utf-8", "replace")
            raise RuntimeError(f"HTTP {resp.status}: {body[:400]}")
        for raw in resp:
            line = raw.decode("utf-8", "replace").rstrip("\r\n")
            if not line.startswith("data: "):
                continue
            payload = line[len("data: "):].strip()
            if payload == "[DONE]" or not payload:
                continue
            try:
                yield json.loads(payload)
            except json.JSONDecodeError:
                continue


def consume_one_turn(host, port, body):
    """Run one streaming chat completion. Returns a dict with the assembled
    content / reasoning / tool_calls / usage / finish_reason."""
    content = []
    reasoning = []
    tool_calls = []  # list of {id, name, args}
    pending = {}  # idx -> partial tool call
    usage = None
    timings = None
    finish_reason = None
    t0 = time.monotonic()
    last_event_at = t0
    for chunk in stream_chat(host, port, body):
        last_event_at = time.monotonic()
        if "usage" in chunk and isinstance(chunk["usage"], dict) and chunk["usage"].get("prompt_tokens") is not None:
            usage = chunk["usage"]
        if "timings" in chunk and isinstance(chunk["timings"], dict):
            timings = chunk["timings"]
        choices = chunk.get("choices") or []
        if not choices:
            continue
        ch0 = choices[0]
        delta = ch0.get("delta") or {}
        if isinstance(delta.get("content"), str):
            content.append(delta["content"])
        if isinstance(delta.get("reasoning_content"), str):
            reasoning.append(delta["reasoning_content"])
        for tc in (delta.get("tool_calls") or []):
            idx = tc.get("index", 0)
            slot = pending.setdefault(idx, {"id": "", "name": "", "args": ""})
            if "id" in tc and isinstance(tc["id"], str):
                slot["id"] = tc["id"]
            func = tc.get("function") or {}
            if isinstance(func.get("name"), str):
                slot["name"] = func["name"]
            if isinstance(func.get("arguments"), str):
                slot["args"] += func["arguments"]
        if ch0.get("finish_reason"):
            finish_reason = ch0["finish_reason"]
    elapsed = time.monotonic() - t0
    for idx in sorted(pending):
        tool_calls.append(pending[idx])
    return {
        "content": "".join(content),
        "reasoning": "".join(reasoning),
        "tool_calls": tool_calls,
        "usage": usage,
        "timings": timings,
        "finish_reason": finish_reason,
        "wall_seconds": elapsed,
        "idle_at_end_seconds": time.monotonic() - last_event_at,
    }


# ─── Bug-pattern classifier ───────────────────────────────────────────────
def classify_turn(turn, args):
    """Return one of: tool_call, clean_done, length_truncated, suspicious_stop."""
    tcs = turn["tool_calls"]
    if tcs:
        return "tool_call"
    finish = turn["finish_reason"]
    cu = (turn["usage"] or {}).get("completion_tokens") or 0
    visible = len(turn["content"]) + len(turn["reasoning"])
    if finish == "length":
        return "length_truncated"
    if cu >= args.min_tokens_for_suspicion and visible < args.min_visible_chars:
        return "suspicious_stop"
    return "clean_done"


# ─── Main loop ────────────────────────────────────────────────────────────
def run_once(args, run_idx):
    print(f"\n══════════ Run {run_idx+1}/{args.runs} (max_tokens={args.max_tokens}) ══════════")
    fs = FakeFS()
    history = [
        {"role": "system", "content": SYSTEM_PROMPT + f"\n\n# Current working directory: {fs.cwd}"},
        {"role": "user", "content": args.task},
    ]
    tools = json.loads(TOOLS_JSON)

    turns = []
    suspicious_turn = None

    for turn_i in range(args.max_turns):
        body = {
            "model": "mlx-serve",
            "messages": history,
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
            "tools": tools,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        try:
            t = consume_one_turn(args.host, args.port, body)
        except (urllib.error.URLError, urllib.error.HTTPError, RuntimeError) as e:
            print(f"  turn {turn_i+1}: HTTP error: {e}")
            return {"error": str(e), "turns": turns}

        kind = classify_turn(t, args)
        usage = t["usage"] or {}
        print(f"  turn {turn_i+1:>2}: kind={kind:>17}  "
              f"tokens={usage.get('completion_tokens','?'):>4}  "
              f"len(content)={len(t['content']):>5}  "
              f"len(reasoning)={len(t['reasoning']):>5}  "
              f"finish={t['finish_reason']!s:>10}  "
              f"tools={len(t['tool_calls'])}  "
              f"({t['wall_seconds']:.1f}s)")
        if args.verbose:
            preview = (t["content"][:120] + "…") if len(t["content"]) > 120 else t["content"]
            print(f"          content: {preview!r}")
            for tc in t["tool_calls"]:
                print(f"          → call {tc['name']}({tc['args'][:80]})")

        turns.append({
            "turn": turn_i + 1,
            "kind": kind,
            "completion_tokens": usage.get("completion_tokens"),
            "prompt_tokens": usage.get("prompt_tokens"),
            "len_content": len(t["content"]),
            "len_reasoning": len(t["reasoning"]),
            "finish_reason": t["finish_reason"],
            "wall_seconds": t["wall_seconds"],
            "tool_calls": [{"name": tc["name"], "args": tc["args"]} for tc in t["tool_calls"]],
            "content": t["content"],
            "reasoning": t["reasoning"],
            "timings": t["timings"],
        })

        if kind == "suspicious_stop":
            suspicious_turn = turns[-1]
            print(f"  ⚠ SUSPICIOUS STOP detected at turn {turn_i+1}")
            break

        if kind == "clean_done":
            print(f"  ✓ clean done at turn {turn_i+1}")
            break

        if kind == "length_truncated":
            print(f"  ⚠ length-truncated at turn {turn_i+1} (max_tokens hit)")
            break

        # tool_call branch — append assistant message + execute tools + feed back.
        assistant_msg = {
            "role": "assistant",
            "content": t["content"] or None,
            "tool_calls": [
                {"id": tc["id"] or f"call_{turn_i}_{i}",
                 "type": "function",
                 "function": {"name": tc["name"], "arguments": tc["args"]}}
                for i, tc in enumerate(t["tool_calls"])
            ],
        }
        history.append(assistant_msg)
        for tc in t["tool_calls"]:
            result = exec_tool(fs, tc["name"], tc["args"])
            history.append({
                "role": "tool",
                "tool_call_id": tc["id"] or f"call_{turn_i}",
                "content": result,
            })
        # Replicate the Swift client's nudge after a tool result
        history.append({
            "role": "user",
            "content": "Continue. If the task is complete, reply with a short plain-text summary for the user (what got done, where it lives, any caveats) — no tool calls, no JSON. If more work is needed, make the next tool call.",
        })

    return {"turns": turns, "suspicious_turn": suspicious_turn, "fs_files": list(fs.files.keys())}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=int(os.environ.get("PORT", "11234")))
    p.add_argument("--max-tokens", type=int, default=int(os.environ.get("MAX_TOKENS", "4096")))
    p.add_argument("--max-turns", type=int, default=20)
    p.add_argument("--runs", type=int, default=1, help="repeat the full run N times")
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--min-tokens-for-suspicion", type=int, default=100)
    p.add_argument("--min-visible-chars", type=int, default=200)
    p.add_argument("--task", default=(
        "Scaffold a SvelteKit todo app with Prisma ORM and SQLite, "
        "in a folder called 'agent-stop-repro-app'. Work step by step "
        "using the tools — don't pause to ask questions."
    ))
    p.add_argument("--out", default=None, help="path for the captured runs JSON dump")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    if args.out is None:
        args.out = f"/tmp/agent_stop_repro_{int(time.time())}.json"

    # Sanity: server reachable + has at least one model loaded.
    try:
        with urllib.request.urlopen(f"http://{args.host}:{args.port}/health", timeout=5) as r:
            if r.status != 200:
                print(f"server health check failed: HTTP {r.status}")
                return 2
    except Exception as e:
        print(f"server unreachable at {args.host}:{args.port}: {e}")
        return 2

    print(f"server: {args.host}:{args.port}")
    print(f"max_tokens={args.max_tokens} max_turns={args.max_turns} runs={args.runs} temp={args.temperature}")

    all_runs = []
    any_suspicious = False
    for i in range(args.runs):
        rec = run_once(args, i)
        rec["run_index"] = i
        all_runs.append(rec)
        if rec.get("suspicious_turn"):
            any_suspicious = True

    with open(args.out, "w") as f:
        json.dump({"args": vars(args), "runs": all_runs}, f, indent=2)
    print(f"\n→ wrote captured runs to {args.out}")

    print("\n══════════ Summary ══════════")
    for i, rec in enumerate(all_runs):
        if rec.get("error"):
            print(f"  run {i+1}: ERROR ({rec['error'][:80]})")
            continue
        turns = rec["turns"]
        kinds = [t["kind"] for t in turns]
        last = turns[-1] if turns else None
        last_kind = last["kind"] if last else "—"
        print(f"  run {i+1}: {len(turns)} turns, ended={last_kind}, "
              f"kinds={','.join(k[:2] for k in kinds)}")
    print()
    if any_suspicious:
        print("RESULT: REPRODUCED — at least one suspicious_stop turn detected.")
        return 1
    print("RESULT: not reproduced.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
