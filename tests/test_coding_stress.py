#!/usr/bin/env python3
"""
Coding agent stress test — pushes mlx-serve to its limits.

Asks the model to build a complex Expedia-like travel booking site with
fake data, multiple pages, search, booking, and account management.
This produces large writeFile arguments (HTML/CSS/JS/JSON) that stress:
  - Tool call argument generation under deep context
  - KV cache behavior with growing prompt
  - Token budget squeeze detection
  - writeFile content integrity (valid JSON args with large code)

Usage:
  python3 tests/test_coding_stress.py [port] [max_rounds]

Requires a running server with a model loaded.
"""

import json
import sys
import urllib.request
import time

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8080
MAX_ROUNDS = int(sys.argv[2]) if len(sys.argv) > 2 else 30
BASE = f"http://127.0.0.1:{PORT}"

GREEN = "\033[0;32m"
RED = "\033[0;31m"
YELLOW = "\033[0;33m"
CYAN = "\033[0;36m"
NC = "\033[0m"

# Stats
total_rounds = 0
valid_tool_calls = 0
empty_args = 0
invalid_json = 0
missing_required = 0
tool_counts = {}
max_prompt_tokens = 0
max_args_len = 0
budget_warnings = 0

SYSTEM_PROMPT = (
    "You are a senior full-stack developer. Build exactly what is asked. "
    "Use tools for every action — write files, run commands, read files. "
    "Write complete, working code — no placeholders or TODOs. "
    "One tool call per step. Do not summarize until the full task is done."
)

TOOLS = [
    {"type": "function", "function": {"name": "shell", "description": "Run a shell command. Example: {\"command\": \"ls -la /tmp\"}",
        "parameters": {"type": "object", "properties": {"command": {"type": "string", "description": "Shell command to execute"}}, "required": ["command"]}}},
    {"type": "function", "function": {"name": "writeFile", "description": "Write content to a file. Example: {\"path\": \"/tmp/f.txt\", \"content\": \"hello\"}",
        "parameters": {"type": "object", "properties": {"path": {"type": "string", "description": "Absolute file path"}, "content": {"type": "string", "description": "File content to write"}}, "required": ["path", "content"]}}},
    {"type": "function", "function": {"name": "readFile", "description": "Read a file's contents. Example: {\"path\": \"/tmp/f.txt\"}",
        "parameters": {"type": "object", "properties": {"path": {"type": "string", "description": "Absolute file path to read"}}, "required": ["path"]}}},
    {"type": "function", "function": {"name": "searchFiles", "description": "Search files for a text pattern (grep). Example: {\"pattern\": \"TODO\"}",
        "parameters": {"type": "object", "properties": {"pattern": {"type": "string", "description": "Text pattern to search for"}}, "required": ["pattern"]}}},
]

TASK = """Build me a travel booking website called "WanderBook" in /tmp/wanderbook/. Here's what I need:

1. Create the project directory structure
2. Generate fake data as JSON: 20 hotels across 5 cities (NYC, Paris, Tokyo, London, Sydney) with names, prices ($80-$500/night), ratings (3.5-5.0), amenities, and images (use placeholder URLs)
3. Build index.html — landing page with search form (destination, dates, guests), featured destinations grid, and navigation
4. Build search.html — results page that loads the JSON data, filters by city, sorts by price/rating, shows hotel cards with booking buttons
5. Build hotel.html — detail page for a single hotel with photo gallery, amenities list, reviews, and booking form
6. Build booking.html — checkout page with guest info form, payment form, booking summary
7. Build account.html — user dashboard with booking history, saved hotels, profile settings
8. Build styles.css — responsive design, modern look with a travel theme (blues/greens), card layouts, grid system
9. Build app.js — search filtering, form validation, localStorage for bookings/favorites, tab switching, dynamic content loading from JSON
10. Run ls -la to show all created files

Use vanilla HTML/CSS/JS only, no frameworks. Make the CSS look professional — not basic. Make the JS functional — search should actually filter, bookings should persist in localStorage."""


def mock_tool_result(name: str, args: dict) -> str:
    """Produce a mock tool result without actually executing."""
    if name == "shell":
        cmd = args.get("command", "")
        if "mkdir" in cmd:
            return f"Created directory"
        if "ls" in cmd:
            return ("total 48\n"
                    "-rw-r--r--  1 user staff  5200 Apr 7 16:00 index.html\n"
                    "-rw-r--r--  1 user staff  4800 Apr 7 16:00 search.html\n"
                    "-rw-r--r--  1 user staff  3600 Apr 7 16:00 hotel.html\n"
                    "-rw-r--r--  1 user staff  3200 Apr 7 16:00 booking.html\n"
                    "-rw-r--r--  1 user staff  2800 Apr 7 16:00 account.html\n"
                    "-rw-r--r--  1 user staff  6400 Apr 7 16:00 styles.css\n"
                    "-rw-r--r--  1 user staff  8200 Apr 7 16:00 app.js\n"
                    "-rw-r--r--  1 user staff 12000 Apr 7 16:00 hotels.json\n")
        if "cat" in cmd:
            return "File contents displayed."
        return f"Command executed: {cmd[:60]}"
    elif name == "writeFile":
        path = args.get("path", "unknown")
        content = args.get("content", "")
        return f"Wrote {len(content)} characters to {path}"
    elif name == "readFile":
        path = args.get("path", "")
        if "json" in path.lower():
            return '[{"name":"Grand Hotel","city":"NYC","price":299}]'
        return f"Contents of {path}: (file content here)"
    elif name == "searchFiles":
        return "Found 3 matches in 2 files."
    return f"Tool {name} executed."


def send_request(messages: list) -> dict:
    """Send a chat completion request and return the parsed response."""
    body = json.dumps({
        "model": "mlx-serve",
        "messages": messages,
        "tools": TOOLS,
        "max_tokens": 2048,
        "temperature": 0.3,
        "stream": False,
    }).encode()

    req = urllib.request.Request(
        f"{BASE}/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=600) as resp:
            return json.loads(resp.read())
    except Exception as e:
        return {"error": str(e)}


def validate_tool_call(tc: dict, round_num: int) -> dict:
    """Validate a single tool call and return stats."""
    global valid_tool_calls, empty_args, invalid_json, missing_required, max_args_len

    fn = tc.get("function", {})
    name = fn.get("name", "")
    args_str = fn.get("arguments", "")
    tc_id = tc.get("id", "call_0")

    result = {
        "name": name,
        "id": tc_id,
        "args_str": args_str,
        "args": {},
        "valid": False,
        "error": None,
    }

    # Track tool usage
    tool_counts[name] = tool_counts.get(name, 0) + 1

    # Check 1: args is non-empty string
    if not args_str or args_str.strip() in ("", "{}"):
        empty_args += 1
        result["error"] = "empty args"
        return result

    # Check 2: valid JSON
    try:
        args = json.loads(args_str)
        result["args"] = args
    except json.JSONDecodeError as e:
        invalid_json += 1
        result["error"] = f"invalid JSON: {str(e)[:60]}"
        return result

    # Track max args length
    if len(args_str) > max_args_len:
        max_args_len = len(args_str)

    # Check 3: required params present
    required = {
        "shell": ["command"],
        "writeFile": ["path", "content"],
        "readFile": ["path"],
        "searchFiles": ["pattern"],
    }
    if name in required:
        missing = [k for k in required[name] if k not in args or not args[k]]
        if missing:
            missing_required += 1
            result["error"] = f"missing: {', '.join(missing)}"
            return result

    valid_tool_calls += 1
    result["valid"] = True
    return result


def main():
    global total_rounds, max_prompt_tokens, budget_warnings

    # Check server health
    try:
        urllib.request.urlopen(f"{BASE}/health", timeout=5)
    except Exception:
        print(f"SKIP: Server not running on port {PORT}")
        sys.exit(0)

    print(f"{CYAN}=== Coding Agent Stress Test ==={NC}")
    print(f"Server: {BASE}")
    print(f"Max rounds: {MAX_ROUNDS}")
    print()

    # Build initial messages
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": TASK},
    ]

    for round_num in range(1, MAX_ROUNDS + 1):
        total_rounds = round_num
        start = time.time()

        resp = send_request(messages)

        if "error" in resp:
            print(f"  {RED}[{round_num:>2}] ERROR: {resp['error'][:100]}{NC}")
            break

        elapsed = time.time() - start
        choice = resp["choices"][0]
        finish = choice["finish_reason"]
        usage = resp.get("usage", {})
        pt = usage.get("prompt_tokens", 0)
        ct = usage.get("completion_tokens", 0)

        if pt > max_prompt_tokens:
            max_prompt_tokens = pt

        # Budget squeeze detection
        if ct < 20 and finish == "tool_calls":
            budget_warnings += 1

        # No tool calls — model is done or gave up
        if finish != "tool_calls":
            content = (choice.get("message", {}).get("content", "") or "")[:120]
            print(f"  {YELLOW}[{round_num:>2}] DONE ({finish}): {content}{NC}")
            print(f"       prompt={pt} comp={ct} ({elapsed:.1f}s)")
            break

        # Process tool calls
        tcs = choice.get("message", {}).get("tool_calls", [])
        if not tcs:
            print(f"  {RED}[{round_num:>2}] finish=tool_calls but no tool_calls array{NC}")
            break

        tc = tcs[0]
        result = validate_tool_call(tc, round_num)

        # Status line
        if result["valid"]:
            name = result["name"]
            args = result["args"]
            if name == "writeFile":
                content_len = len(args.get("content", ""))
                detail = f"path={args.get('path','?')}, content={content_len}ch"
            elif name == "shell":
                detail = args.get("command", "?")[:60]
            elif name == "readFile":
                detail = args.get("path", "?")
            else:
                detail = str(list(args.keys()))
            print(f"  {GREEN}[{round_num:>2}] {name}: {detail}{NC}")
        else:
            raw = result["args_str"][:80] if result["args_str"] else "(empty)"
            print(f"  {RED}[{round_num:>2}] {result['name']}: {result['error']} — raw: {raw}{NC}")

        print(f"       prompt={pt} comp={ct} ({elapsed:.1f}s)")

        # Build assistant message for history
        assistant_msg = {"role": "assistant", "content": "", "tool_calls": [tc]}
        # Include content if present
        msg_content = choice.get("message", {}).get("content", "") or ""
        if msg_content.strip():
            assistant_msg["content"] = msg_content

        messages.append(assistant_msg)

        # Mock tool result
        if result["valid"]:
            mock = mock_tool_result(result["name"], result["args"])
        else:
            # Feed back error so model can retry
            mock = f"Error: {result['error']}. Please call {result['name']} again with valid JSON arguments."

        messages.append({
            "role": "tool",
            "tool_call_id": result["id"],
            "content": mock[:500],
        })

        # Nudge
        messages.append({
            "role": "user",
            "content": "Continue. If the task is done, summarize the result. If not, take the next step.",
        })

    # Summary
    print()
    print(f"{CYAN}=== Summary ==={NC}")
    print(f"Rounds:             {total_rounds}")
    print(f"Valid tool calls:    {valid_tool_calls}")
    print(f"Empty args:          {empty_args}")
    print(f"Invalid JSON:        {invalid_json}")
    print(f"Missing required:    {missing_required}")
    print(f"Budget warnings:     {budget_warnings}")
    print(f"Max prompt tokens:   {max_prompt_tokens}")
    print(f"Max args length:     {max_args_len} chars")
    print()
    print("Tools used:")
    for name, count in sorted(tool_counts.items(), key=lambda x: -x[1]):
        print(f"  {name}: {count}")
    print()

    failure_count = empty_args + invalid_json + missing_required
    if failure_count == 0:
        print(f"{GREEN}ALL TOOL CALLS VALID{NC}")
    else:
        print(f"{RED}FAILURES: {failure_count} ({empty_args} empty, {invalid_json} bad JSON, {missing_required} missing params){NC}")

    sys.exit(1 if failure_count > 0 else 0)


if __name__ == "__main__":
    main()
