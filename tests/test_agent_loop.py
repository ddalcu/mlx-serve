#!/usr/bin/env python3
"""
Integration test: multi-turn agent loop with streaming, thinking, and mock tools.

Simulates:
  1. User asks to find top news sites, grab headlines, write an HTML report
  2. Model calls webSearch → mock results
  3. Model calls browse (multiple sites) → mock headlines
  4. Model calls writeFile → verify HTML output

Tests streaming SSE, thinking/reasoning, tool call detection,
multi-turn tool results, and final content generation.
"""

import json
import sys
import urllib.request
import os

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8080
BASE = f"http://127.0.0.1:{PORT}"
OUTPUT_FILE = "/tmp/mlx-agent-test-news.html"

# Colors
GREEN = "\033[0;32m"
RED = "\033[0;31m"
YELLOW = "\033[0;33m"
NC = "\033[0m"

passes = 0
fails = 0


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


TOOLS = [
    {"type": "function", "function": {"name": "webSearch", "description": "Search the web",
        "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}},
    {"type": "function", "function": {"name": "browse", "description": "Browse a URL and read its content",
        "parameters": {"type": "object", "properties": {"action": {"type": "string"}, "url": {"type": "string"}}, "required": ["action"]}}},
    {"type": "function", "function": {"name": "writeFile", "description": "Write content to a file",
        "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path", "content"]}}},
]

SYSTEM_PROMPT = (
    "You are a helpful assistant with access to tools. Use tools to accomplish tasks. "
    "When you need to search the web, use webSearch. To visit a page, use browse with action navigate then readText. "
    "To save files, use writeFile. Be efficient — make tool calls, don't just describe what you would do."
)

MOCK_SEARCH_RESULT = (
    "Top news websites:\n"
    "1. BBC News - https://bbc.com/news\n"
    "2. CNN - https://cnn.com\n"
    "3. Reuters - https://reuters.com\n"
    "4. AP News - https://apnews.com\n"
    "5. NPR - https://npr.org/news"
)

MOCK_BROWSE = {
    "bbc": "BBC Headlines: UK Parliament passes new climate bill. London housing prices drop 5%. Premier League results: Arsenal tops table.",
    "cnn": "CNN Headlines: Fed holds rates steady amid inflation. Hurricane watch for Gulf Coast. Tech stocks rally on AI earnings.",
    "reuters": "Reuters Headlines: Global oil prices surge 3%. EU approves new trade deal with Japan. WHO warns of new respiratory virus.",
    "apnews": "AP Headlines: Supreme Court rules on immigration case. NASA launches new Mars probe. Wildfires spread across California.",
    "npr": "NPR Headlines: Education funding bill stalls in Senate. New study links sleep to productivity. Jazz legend releases final album.",
}


def safe_parse_args(args_str):
    """Parse tool call arguments, handling both JSON and Gemma 4's custom format."""
    if not args_str:
        return {}
    try:
        return json.loads(args_str)
    except json.JSONDecodeError:
        pass
    # Gemma 4 format: key:<|"|>value<|"|> or key:value
    result = {}
    import re
    # Match key:<|"|>value<|"|> pairs
    for m in re.finditer(r'(\w+):<\|"\|>([^<]*)<\|"\|>', args_str):
        result[m.group(1)] = m.group(2)
    # Match key:value (unquoted)
    if not result:
        for m in re.finditer(r'(\w+):([^,}\s]+)', args_str):
            result[m.group(1)] = m.group(2)
    return result


def mock_browse_result(url):
    for key, content in MOCK_BROWSE.items():
        if key in url.lower():
            return content
    return f"Page content from {url}: Various news headlines and articles."


def send_streaming_request(messages, enable_thinking=False):
    """Send a streaming chat completion request and parse SSE response."""
    body = {
        "model": "mlx-serve",
        "messages": messages,
        "tools": TOOLS,
        "max_tokens": 512,
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
        return {"content": "", "reasoning": "", "tool_calls": None, "finish_reason": f"error: {e}"}

    content_parts = []
    reasoning_parts = []
    tool_calls = []
    finish_reason = "unknown"

    for line in raw.strip().split("\n"):
        if not line.startswith("data: ") or line.strip() == "data: [DONE]":
            continue
        try:
            chunk = json.loads(line[6:])
        except json.JSONDecodeError:
            continue
        choices = chunk.get("choices", [])
        if not choices:
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
    }


def run_test():
    if os.path.exists(OUTPUT_FILE):
        os.remove(OUTPUT_FILE)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": (
            f"Search the web for the top 5 news websites right now. Visit each one and "
            f"grab the headlines. Then write an HTML file at {OUTPUT_FILE} with a nice "
            f"news report containing all the headlines organized by source."
        )},
    ]

    # ── Step 1: Initial request (expect webSearch) ──
    print(f"\n{YELLOW}Step 1: Initial request (expect webSearch tool call){NC}")

    resp = send_streaming_request(messages, enable_thinking=True)

    if resp["finish_reason"] == "tool_calls":
        assert_pass("finish_reason is tool_calls")
    else:
        assert_fail("finish_reason is tool_calls", f"got: {resp['finish_reason']}")

    if resp["reasoning"]:
        assert_pass("model produced reasoning/thinking content")
    else:
        assert_fail("model produced reasoning/thinking content", "no reasoning in response")

    tc = resp["tool_calls"]
    tool_name = tc[0]["name"] if tc else ""

    if tool_name == "webSearch":
        assert_pass("model called webSearch")
    else:
        assert_fail("model called webSearch", f"got: {tool_name}")
        if not tc:
            print("  No tool call — aborting.")
            return

    # Feed mock search results
    messages.append({
        "role": "assistant", "content": None,
        "tool_calls": [{"id": tc[0]["id"], "type": "function",
                         "function": {"name": tc[0]["name"], "arguments": tc[0]["arguments"]}}]
    })
    messages.append({"role": "tool", "tool_call_id": tc[0]["id"], "content": MOCK_SEARCH_RESULT})

    # ── Step 2: Feed search results (expect browse calls) ──
    print(f"\n{YELLOW}Step 2: Feed search results (expect browse calls){NC}")

    browse_count = 0
    max_rounds = 12  # navigate+readText pairs + some slack

    for round_num in range(max_rounds):
        resp = send_streaming_request(messages)
        tc = resp["tool_calls"]
        tool_name = tc[0]["name"] if tc else ""

        if resp["finish_reason"] == "tool_calls" and tool_name == "browse":
            args = safe_parse_args(tc[0]["arguments"])
            url = args.get("url", "")
            action = args.get("action", "navigate")

            messages.append({
                "role": "assistant", "content": None,
                "tool_calls": [{"id": tc[0]["id"], "type": "function",
                                 "function": {"name": "browse", "arguments": tc[0]["arguments"]}}]
            })

            if action == "navigate":
                messages.append({"role": "tool", "tool_call_id": tc[0]["id"],
                                 "content": f"Navigated to {url}"})
            else:
                content = mock_browse_result(url)
                messages.append({"role": "tool", "tool_call_id": tc[0]["id"], "content": content})
                browse_count += 1
                print(f"  browsed: {url} ({action})")

        elif resp["finish_reason"] == "tool_calls" and tool_name == "writeFile":
            print(f"  (model skipped to writeFile after {browse_count} browse reads)")
            break

        elif resp["finish_reason"] != "tool_calls":
            # Model gave text response
            messages.append({"role": "assistant", "content": resp["content"]})
            break

        else:
            # Other tool call
            messages.append({
                "role": "assistant", "content": None,
                "tool_calls": [{"id": tc[0]["id"], "type": "function",
                                 "function": {"name": tool_name, "arguments": tc[0]["arguments"]}}]
            })
            messages.append({"role": "tool", "tool_call_id": tc[0]["id"], "content": "OK"})

        if browse_count >= 3:
            print(f"  (browsed {browse_count} sites, moving on)")
            break

    if browse_count >= 1:
        assert_pass(f"model browsed at least 1 news site ({browse_count} total)")
    else:
        assert_fail("model browsed at least 1 news site", f"browsed {browse_count}")

    # ── Step 3: Wait for writeFile call ──
    print(f"\n{YELLOW}Step 3: Wait for writeFile call{NC}")

    file_written = False
    for round_num in range(8):
        # Check if the last response already was a writeFile
        if (resp["finish_reason"] == "tool_calls" and resp["tool_calls"]
                and resp["tool_calls"][0]["name"] == "writeFile"):
            tc = resp["tool_calls"]
        else:
            resp = send_streaming_request(messages)
            tc = resp["tool_calls"]

        tool_name = tc[0]["name"] if tc else ""

        if resp["finish_reason"] == "tool_calls" and tool_name == "writeFile":
            args = safe_parse_args(tc[0]["arguments"])
            file_content = args.get("content", "")

            if file_content:
                with open(OUTPUT_FILE, "w") as f:
                    f.write(file_content)
                file_written = True

                messages.append({
                    "role": "assistant", "content": None,
                    "tool_calls": [{"id": tc[0]["id"], "type": "function",
                                     "function": {"name": "writeFile", "arguments": tc[0]["arguments"]}}]
                })
                messages.append({"role": "tool", "tool_call_id": tc[0]["id"],
                                 "content": f"File written to {OUTPUT_FILE}"})
                assert_pass("model called writeFile with content")
                break

        elif resp["finish_reason"] == "tool_calls" and tool_name == "browse":
            args = safe_parse_args(tc[0]["arguments"])
            url = args.get("url", "")
            messages.append({
                "role": "assistant", "content": None,
                "tool_calls": [{"id": tc[0]["id"], "type": "function",
                                 "function": {"name": "browse", "arguments": tc[0]["arguments"]}}]
            })
            content = mock_browse_result(url)
            messages.append({"role": "tool", "tool_call_id": tc[0]["id"], "content": content})
            resp = {"finish_reason": "", "tool_calls": None}  # Reset to fetch next

        elif resp["finish_reason"] != "tool_calls":
            messages.append({"role": "assistant", "content": resp["content"]})
            resp = {"finish_reason": "", "tool_calls": None}

        else:
            messages.append({
                "role": "assistant", "content": None,
                "tool_calls": [{"id": tc[0]["id"], "type": "function",
                                 "function": {"name": tool_name, "arguments": tc[0]["arguments"]}}]
            })
            messages.append({"role": "tool", "tool_call_id": tc[0]["id"], "content": "OK"})
            resp = {"finish_reason": "", "tool_calls": None}

    if not file_written:
        assert_fail("model called writeFile", "never called writeFile")

    # ── Step 4: Final response ──
    print(f"\n{YELLOW}Step 4: Verify final response{NC}")

    resp = send_streaming_request(messages)
    if resp["finish_reason"] != "tool_calls" and resp["content"]:
        assert_pass("model gave final content response (no more tool calls)")
    else:
        assert_fail("model gave final content response", f"finish_reason={resp['finish_reason']}")

    # ── Step 5: Validate output file ──
    print(f"\n{YELLOW}Step 5: Validate output file{NC}")

    if os.path.exists(OUTPUT_FILE):
        assert_pass(f"output file exists at {OUTPUT_FILE}")

        size = os.path.getsize(OUTPUT_FILE)
        if size > 100:
            assert_pass(f"output file is non-trivial ({size} bytes)")
        else:
            assert_fail(f"output file is non-trivial", f"only {size} bytes")

        with open(OUTPUT_FILE) as f:
            html = f.read()

        html_lower = html.lower()
        if any(tag in html_lower for tag in ["<html", "<!doctype", "<head", "<body"]):
            assert_pass("output file contains HTML markup")
        else:
            assert_fail("output file contains HTML markup")

        if any(src in html_lower for src in ["bbc", "cnn", "reuters", "ap", "npr", "headline", "news"]):
            assert_pass("output file references news sources")
        else:
            assert_fail("output file references news sources")
    else:
        assert_fail(f"output file exists at {OUTPUT_FILE}")

    # Cleanup
    if os.path.exists(OUTPUT_FILE):
        os.remove(OUTPUT_FILE)


if __name__ == "__main__":
    run_test()

    total = passes + fails
    print()
    print("═══════════════════════════════════════════")
    if fails == 0:
        print(f"  {GREEN}ALL {total} TESTS PASSED{NC}")
    else:
        print(f"  {GREEN}{passes} passed{NC}, {RED}{fails} failed{NC} (out of {total})")
    print("═══════════════════════════════════════════")
    sys.exit(fails)
