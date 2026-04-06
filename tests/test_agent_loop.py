#!/usr/bin/env python3
"""
Comprehensive agent loop stress test with streaming, thinking, tool calling,
multi-turn ReAct loops, and simulated HITL (human-in-the-loop).

Scenarios:
  1. Simple tool call (webSearch)
  2. Multi-step research (search → browse → browse → summarize)
  3. File creation pipeline (search → browse → writeFile → readFile → verify)
  4. HITL: user interrupts mid-task with clarification, agent adapts
  5. Stress: deep ReAct loop with 8+ tool calls in sequence
  6. Error recovery: tool returns error, agent retries or adapts
"""

import json
import sys
import urllib.request
import os
import time

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8080
BASE = f"http://127.0.0.1:{PORT}"
OUTPUT_DIR = "/tmp/mlx-agent-test"

GREEN = "\033[0;32m"
RED = "\033[0;31m"
YELLOW = "\033[0;33m"
CYAN = "\033[0;36m"
NC = "\033[0m"

passes = 0
fails = 0
total_tool_calls = 0
total_rounds = 0

SYSTEM_PROMPT = (
    "You are a helpful assistant. Use tools to complete tasks. "
    "Always call tools — never just describe what you would do. "
    "Complete all steps before responding."
)


TOOLS = [
    {"type": "function", "function": {"name": "webSearch", "description": "Search the web for information",
        "parameters": {"type": "object", "properties": {"query": {"type": "string", "description": "Search query"}}, "required": ["query"]}}},
    {"type": "function", "function": {"name": "browse", "description": "Browse a URL — navigate to load it, readText to extract content",
        "parameters": {"type": "object", "properties": {"action": {"type": "string", "enum": ["navigate", "readText"], "description": "navigate to load page, readText to get content"}, "url": {"type": "string", "description": "URL to browse"}}, "required": ["action"]}}},
    {"type": "function", "function": {"name": "writeFile", "description": "Write content to a file at the given path",
        "parameters": {"type": "object", "properties": {"path": {"type": "string", "description": "File path"}, "content": {"type": "string", "description": "File content"}}, "required": ["path", "content"]}}},
    {"type": "function", "function": {"name": "readFile", "description": "Read and return the content of a file",
        "parameters": {"type": "object", "properties": {"path": {"type": "string", "description": "File path to read"}}, "required": ["path"]}}},
    {"type": "function", "function": {"name": "shell", "description": "Run a shell command and return its output",
        "parameters": {"type": "object", "properties": {"command": {"type": "string", "description": "Shell command to execute"}}, "required": ["command"]}}},
]


# ── Mock tool responses ──

MOCK_RESPONSES = {
    "webSearch": {
        "top news websites 2024": "1. BBC News (bbc.com/news) - Global coverage\n2. CNN (cnn.com) - US & world news\n3. Reuters (reuters.com) - Wire service\n4. AP News (apnews.com) - Associated Press\n5. NPR (npr.org) - Public radio news\n6. Al Jazeera (aljazeera.com) - International\n7. The Guardian (theguardian.com) - UK & world",
        "python web frameworks comparison": "Top Python Web Frameworks:\n1. Django - Full-featured, batteries-included\n2. FastAPI - Modern, async, auto-docs\n3. Flask - Lightweight, flexible\n4. Starlette - ASGI framework\n5. Tornado - Async networking",
        "best restaurants tokyo": "Top Tokyo Restaurants:\n1. Sukiyabashi Jiro - Sushi (Ginza)\n2. Narisawa - Innovative Japanese (Minato)\n3. Den - Creative Japanese (Jingumae)\n4. Florilege - French-Japanese (Jingumae)\n5. Sazenka - Chinese-Japanese (Roppongi)",
        "machine learning trends 2024": "ML Trends 2024:\n1. Mixture of Experts (MoE) models gaining traction\n2. Small language models for on-device inference\n3. Multimodal AI combining vision+text+audio\n4. RAG (Retrieval Augmented Generation)\n5. AI agents with tool use capabilities",
    },
    "browse": {
        "bbc": "BBC News Headlines:\n- Climate summit reaches landmark agreement\n- UK economy grows 0.3% in Q3\n- Premier League: Arsenal beat Chelsea 2-1\n- New AI regulations proposed by EU\n- Scientists discover high-temperature superconductor",
        "cnn": "CNN Breaking News:\n- Fed signals rate cuts in 2025\n- Hurricane season: Category 4 storm approaches Florida\n- Tech layoffs continue across Silicon Valley\n- Supreme Court to hear major privacy case\n- SpaceX completes 100th Starship landing",
        "reuters": "Reuters Top Stories:\n- Global markets rally on trade deal optimism\n- OPEC+ agrees to production cut\n- Japan's GDP beats expectations\n- Pharmaceutical breakthrough in Alzheimer's treatment\n- Cybersecurity threat: major bank data breach",
        "apnews": "AP News:\n- US election polls show tight race in swing states\n- NASA's Artemis III mission on track for 2025\n- Wildfire containment reaches 80% in California\n- Immigration reform bill advances in Congress\n- Record heat wave across southern Europe",
        "npr": "NPR News:\n- Education funding debate heats up in Congress\n- New study: remote work boosts productivity 13%\n- Jazz legend Herbie Hancock announces farewell tour\n- Coral reef restoration shows promising results\n- Podcast industry revenue tops $2 billion",
        "django": "Django Documentation:\nDjango is a high-level Python web framework that encourages rapid development. Features: ORM, admin interface, URL routing, template engine, forms, authentication, caching, internationalization.",
        "fastapi": "FastAPI Documentation:\nFastAPI is a modern web framework for building APIs with Python 3.7+. Features: automatic OpenAPI docs, async support, type hints, dependency injection, OAuth2, WebSocket support.",
        "flask": "Flask Documentation:\nFlask is a lightweight WSGI web application framework. Features: built-in dev server, Jinja2 templates, RESTful request dispatching, Unicode support, extensive documentation.",
        "narisawa": "Narisawa Restaurant Tokyo:\nInnovative Japanese cuisine by Chef Yoshihiro Narisawa. 2 Michelin stars. Known for 'Satoyama' cuisine connecting forest, sea, and soil. Tasting menu: ¥32,000. Reservations required 1 month ahead.",
        "jiro": "Sukiyabashi Jiro:\n3 Michelin stars. Master sushi chef Jiro Ono, age 98. Omakase only: ¥40,000. 10-seat counter in Ginza. Featured in documentary 'Jiro Dreams of Sushi'. Reservations through hotel concierge only.",
    },
}


def mock_tool_response(tool_name, args):
    """Generate mock response for a tool call."""
    global total_tool_calls
    total_tool_calls += 1

    if tool_name == "webSearch":
        query = args.get("query", "").lower()
        for key, response in MOCK_RESPONSES["webSearch"].items():
            if any(word in query for word in key.split()):
                return response
        return f"Search results for '{args.get('query', '')}': No specific results found. Try a more specific query."

    elif tool_name == "browse":
        url = args.get("url", "").lower()
        action = args.get("action", "navigate")
        if action == "navigate":
            return f"Successfully navigated to {args.get('url', 'unknown')}. Page loaded."
        for key, response in MOCK_RESPONSES["browse"].items():
            if key in url:
                return response
        return f"Page content from {args.get('url', '')}: Generic page with navigation, articles, and footer content."

    elif tool_name == "writeFile":
        path = args.get("path", "/tmp/unknown")
        content = args.get("content", "")
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            f.write(content)
        return f"Successfully wrote {len(content)} bytes to {path}"

    elif tool_name == "readFile":
        path = args.get("path", "")
        if os.path.exists(path):
            with open(path) as f:
                return f.read()[:2000]
        return f"Error: File not found: {path}"

    elif tool_name == "shell":
        cmd = args.get("command", "")
        if "ls" in cmd:
            files = os.listdir(OUTPUT_DIR) if os.path.exists(OUTPUT_DIR) else []
            return "\n".join(files) if files else "(empty directory)"
        elif "cat" in cmd:
            return "Use readFile tool instead."
        elif "wc" in cmd:
            return "42 lines  256 words  1847 bytes"
        return f"$ {cmd}\nCommand executed successfully."

    return "OK"


def safe_parse_args(args_str):
    """Parse tool call arguments, handling both JSON and Gemma 4's custom format."""
    if not args_str:
        return {}
    try:
        return json.loads(args_str)
    except json.JSONDecodeError:
        pass
    import re
    result = {}
    for m in re.finditer(r'(\w+):<\|"\|>(.*?)<\|"\|>', args_str, re.DOTALL):
        result[m.group(1)] = m.group(2)
    if not result:
        for m in re.finditer(r'(\w+):([^,}\s]+)', args_str):
            result[m.group(1)] = m.group(2)
    return result


def send_request(messages, enable_thinking=False, max_tokens=2048):
    """Send a streaming chat completion request and parse SSE response."""
    global total_rounds
    total_rounds += 1

    body = {
        "model": "mlx-serve",
        "messages": messages,
        "tools": TOOLS,
        "max_tokens": max_tokens,
        "temperature": 0.7,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if enable_thinking:
        body["enable_thinking"] = True

    data = json.dumps(body).encode()
    req = urllib.request.Request(
        f"{BASE}/v1/chat/completions",
        data=data,
        headers={"Content-Type": "application/json"},
    )

    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            raw = resp.read().decode()
    except Exception as e:
        return {"content": f"Error: {e}", "reasoning": "", "tool_calls": None, "finish_reason": "error",
                "prompt_tokens": 0, "completion_tokens": 0}

    content_parts = []
    reasoning_parts = []
    tool_calls = []
    finish_reason = "unknown"
    prompt_tokens = 0
    completion_tokens = 0

    for line in raw.strip().split("\n"):
        if not line.startswith("data: ") or line.strip() == "data: [DONE]":
            continue
        try:
            chunk = json.loads(line[6:])
        except json.JSONDecodeError:
            continue

        choices = chunk.get("choices", [])
        if not choices:
            usage = chunk.get("usage")
            if usage:
                prompt_tokens = usage.get("prompt_tokens", 0)
                completion_tokens = usage.get("completion_tokens", 0)
            continue
        delta = choices[0].get("delta", {})
        fr = choices[0].get("finish_reason")
        if fr:
            finish_reason = fr
        if delta.get("content"):
            content_parts.append(delta["content"])
        if delta.get("reasoning_content"):
            reasoning_parts.append(delta["reasoning_content"])
        if "tool_calls" in delta:
            for tc in delta["tool_calls"]:
                idx = tc.get("index", 0)
                while len(tool_calls) <= idx:
                    tool_calls.append({"id": "", "name": "", "arguments": ""})
                if "id" in tc:
                    tool_calls[idx]["id"] = tc["id"]
                fn = tc.get("function", {})
                if "name" in fn:
                    tool_calls[idx]["name"] = fn["name"]
                if "arguments" in fn:
                    tool_calls[idx]["arguments"] += fn["arguments"]

    return {
        "content": "".join(content_parts),
        "reasoning": "".join(reasoning_parts),
        "tool_calls": tool_calls if tool_calls else None,
        "finish_reason": finish_reason,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
    }


def run_agent_loop(messages, max_rounds=15, enable_thinking_first=True):
    """Run a ReAct agent loop until the model stops calling tools or hits max rounds."""
    tool_call_log = []
    round_num = 0

    for round_num in range(max_rounds):
        thinking = enable_thinking_first and round_num == 0
        resp = send_request(messages, enable_thinking=thinking)

        if resp["finish_reason"] == "error":
            print(f"    round {round_num+1}: ERROR - {resp['content']}")
            break

        if resp["finish_reason"] == "tool_calls" and resp["tool_calls"]:
            for tc in resp["tool_calls"]:
                args = safe_parse_args(tc["arguments"])
                tool_name = tc["name"]
                tool_call_log.append(tool_name)
                result = mock_tool_response(tool_name, args)
                arg_summary = tc["arguments"][:60] + "..." if len(tc["arguments"]) > 60 else tc["arguments"]
                print(f"    round {round_num+1}: {tool_name}({arg_summary})")

                messages.append({
                    "role": "assistant", "content": None,
                    "tool_calls": [{"id": tc["id"], "type": "function",
                                     "function": {"name": tool_name, "arguments": tc["arguments"]}}]
                })
                messages.append({"role": "tool", "tool_call_id": tc["id"], "content": result})
        else:
            # Model gave a text response — done
            content_preview = resp["content"][:100].replace("\n", " ")
            print(f"    round {round_num+1}: RESPONSE ({len(resp['content'])} chars) \"{content_preview}...\"")
            messages.append({"role": "assistant", "content": resp["content"]})
            break

    return {
        "tool_calls": tool_call_log,
        "rounds": round_num + 1,
        "final_content": resp.get("content", ""),
        "has_reasoning": bool(resp.get("reasoning") if round_num == 0 else False),
        "finish_reason": resp.get("finish_reason", "unknown"),
    }


def assert_pass(desc):
    global passes
    passes += 1
    print(f"  {GREEN}PASS{NC} {desc}")


def assert_fail(desc, detail=""):
    global fails
    fails += 1
    print(f"  {RED}FAIL{NC} {desc}")
    if detail:
        print(f"    {detail}")


def check(cond, desc, detail=""):
    if cond:
        assert_pass(desc)
    else:
        assert_fail(desc, detail)
    return cond


# ════════════════════════════════════════════════════════════
# Scenario 1: Simple single tool call
# ════════════════════════════════════════════════════════════
def test_simple_tool_call():
    print(f"\n{CYAN}━━━ Scenario 1: Simple tool call ━━━{NC}")
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "Search the web for machine learning trends 2024 and tell me the top 3."},
    ]
    result = run_agent_loop(messages, max_rounds=5)
    check("webSearch" in result["tool_calls"], "called webSearch")
    check(result["final_content"] and len(result["final_content"]) > 20,
          f"gave substantive response ({len(result['final_content'])} chars)")


# ════════════════════════════════════════════════════════════
# Scenario 2: Multi-step research (search → browse → summarize)
# ════════════════════════════════════════════════════════════
def test_multi_step_research():
    print(f"\n{CYAN}━━━ Scenario 2: Multi-step research ━━━{NC}")
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "I need to compare Python web frameworks. Search for a comparison, then visit the Django and FastAPI docs pages to get their key features. Give me a brief comparison."},
    ]
    result = run_agent_loop(messages, max_rounds=10)
    check("webSearch" in result["tool_calls"], "called webSearch")
    check(result["tool_calls"].count("browse") >= 2,
          f"browsed at least 2 pages ({result['tool_calls'].count('browse')} browse calls)")
    check(len(result["tool_calls"]) >= 3, f"made 3+ tool calls total ({len(result['tool_calls'])})")


# ════════════════════════════════════════════════════════════
# Scenario 3: File creation pipeline
# ════════════════════════════════════════════════════════════
def test_file_creation():
    print(f"\n{CYAN}━━━ Scenario 3: File creation pipeline ━━━{NC}")
    output_file = f"{OUTPUT_DIR}/news_report.html"
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Search for top news websites, visit BBC and CNN to get their headlines, then write an HTML report at {output_file} with the headlines organized by source. Include proper HTML structure with <html>, <head>, <body> tags."},
    ]
    result = run_agent_loop(messages, max_rounds=12)
    check("webSearch" in result["tool_calls"], "called webSearch")
    check("browse" in result["tool_calls"], "called browse")
    wrote = "writeFile" in result["tool_calls"]
    check(wrote, "called writeFile")
    if wrote and os.path.exists(output_file):
        content = open(output_file).read()
        check(len(content) > 100, f"file has content ({len(content)} bytes)")
        check(any(t in content.lower() for t in ["<html", "<body", "<head"]),
              "file contains HTML markup")
        check(any(s in content.lower() for s in ["bbc", "cnn", "headline", "news"]),
              "file references news sources")


# ════════════════════════════════════════════════════════════
# Scenario 4: HITL — user interrupts mid-task
# ════════════════════════════════════════════════════════════
def test_hitl_interruption():
    print(f"\n{CYAN}━━━ Scenario 4: HITL — user interrupts mid-task ━━━{NC}")
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "Find me the best restaurants in Tokyo."},
    ]

    # Phase 1: Let it search
    print(f"  {YELLOW}Phase 1: Initial search{NC}")
    resp = send_request(messages)
    if resp["tool_calls"]:
        tc = resp["tool_calls"][0]
        result = mock_tool_response(tc["name"], safe_parse_args(tc["arguments"]))
        messages.append({"role": "assistant", "content": None,
                         "tool_calls": [{"id": tc["id"], "type": "function",
                                          "function": {"name": tc["name"], "arguments": tc["arguments"]}}]})
        messages.append({"role": "tool", "tool_call_id": tc["id"], "content": result})
        print(f"    {tc['name']}(...)")
        phase1_ok = True
    else:
        messages.append({"role": "assistant", "content": resp["content"]})
        phase1_ok = False

    check(phase1_ok, "phase 1: model used a tool")

    # Phase 2: User interrupts with clarification
    print(f"  {YELLOW}Phase 2: User interrupts with new requirement{NC}")
    messages.append({"role": "user", "content": "Actually, I only want sushi restaurants. Also browse the top result to get details like price and reservation info."})

    result2 = run_agent_loop(messages, max_rounds=8, enable_thinking_first=False)
    check(len(result2["tool_calls"]) >= 1,
          f"agent continued with tools after interruption ({len(result2['tool_calls'])} calls)")
    check(result2["final_content"] and ("sushi" in result2["final_content"].lower() or "jiro" in result2["final_content"].lower()),
          "final response is about sushi (adapted to user correction)")


# ════════════════════════════════════════════════════════════
# Scenario 5: Deep ReAct stress test (8+ tool calls)
# ════════════════════════════════════════════════════════════
def test_deep_react_loop():
    print(f"\n{CYAN}━━━ Scenario 5: Deep ReAct stress test ━━━{NC}")
    output_file = f"{OUTPUT_DIR}/framework_comparison.md"
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": (
            f"Do a thorough comparison of Python web frameworks. "
            f"1) Search for Python web framework comparisons. "
            f"2) Visit the documentation pages for Django, FastAPI, and Flask. "
            f"3) Write a detailed markdown comparison to {output_file} covering: features, performance, ease of use, and best use cases for each. "
            f"4) Then read the file back to verify it was written correctly. "
            f"Complete ALL steps."
        )},
    ]
    result = run_agent_loop(messages, max_rounds=15, enable_thinking_first=True)
    check(len(result["tool_calls"]) >= 5,
          f"made 5+ tool calls ({len(result['tool_calls'])} total)")
    check(result["tool_calls"].count("browse") >= 3,
          f"browsed 3+ pages ({result['tool_calls'].count('browse')} browses)")
    wrote = "writeFile" in result["tool_calls"]
    check(wrote, "wrote comparison file")
    if wrote and os.path.exists(output_file):
        content = open(output_file).read()
        check(len(content) > 200, f"comparison file has substance ({len(content)} bytes)")
        check(all(fw in content.lower() for fw in ["django", "fastapi", "flask"]),
              "file mentions all 3 frameworks")
    read = "readFile" in result["tool_calls"]
    check(read, "verified file by reading it back")


# ════════════════════════════════════════════════════════════
# Scenario 6: Error recovery
# ════════════════════════════════════════════════════════════
def test_error_recovery():
    print(f"\n{CYAN}━━━ Scenario 6: Error recovery ━━━{NC}")
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "Read the file at /tmp/mlx-agent-test/nonexistent.txt and tell me what's in it. If the file doesn't exist, create it with a summary of today's top news."},
    ]
    result = run_agent_loop(messages, max_rounds=10)
    check("readFile" in result["tool_calls"], "tried to read the file")
    # After getting a "file not found" error, should search for news and write the file
    has_recovery = ("webSearch" in result["tool_calls"] or "writeFile" in result["tool_calls"]
                    or "shell" in result["tool_calls"])
    check(has_recovery, "recovered from error (searched or wrote file)")
    check(result["final_content"] and len(result["final_content"]) > 20,
          "gave a final response after recovery")


# ════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════
if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    start_time = time.time()

    test_simple_tool_call()
    test_multi_step_research()
    test_file_creation()
    test_hitl_interruption()
    test_deep_react_loop()
    test_error_recovery()

    elapsed = time.time() - start_time
    total = passes + fails

    # Cleanup
    import shutil
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR, ignore_errors=True)

    print()
    print("═" * 55)
    print(f"  Scenarios: 6")
    print(f"  Assertions: {total} ({GREEN}{passes} passed{NC}, {RED}{fails} failed{NC})")
    print(f"  Total tool calls: {total_tool_calls}")
    print(f"  Total API rounds: {total_rounds}")
    print(f"  Elapsed: {elapsed:.0f}s")
    print("═" * 55)
    if fails == 0:
        print(f"  {GREEN}ALL {total} TESTS PASSED{NC}")
    else:
        print(f"  {RED}{fails} FAILURES{NC}")
    print("═" * 55)
    sys.exit(min(fails, 125))
