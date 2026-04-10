#!/usr/bin/env python3
"""
CLI agent that replicates MLX Core's agent loop against a running mlx-serve.

Usage:
    python3 tests/cli_agent.py [--port 8080] [--max-rounds 10] "your prompt here"

Features:
    - Streaming SSE with real-time output
    - Tool calling with mock or real execution
    - Thinking/reasoning display
    - Multi-turn ReAct loop
    - Matches the app's exact request format
"""

import json
import sys
import urllib.request
import os
import argparse
import re
import subprocess

GREEN = "\033[0;32m"
RED = "\033[0;31m"
YELLOW = "\033[0;33m"
CYAN = "\033[0;36m"
DIM = "\033[2m"
BOLD = "\033[1m"
NC = "\033[0m"

SYSTEM_PROMPT = (
    "You are a helpful macOS assistant. Use tools for tasks. "
    "Answer directly when no tools needed. For web search use webSearch tool."
)

TOOLS = [
    {"type": "function", "function": {"name": "shell", "description": "Run a command",
        "parameters": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}}},
    {"type": "function", "function": {"name": "writeFile", "description": "Write a file",
        "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path", "content"]}}},
    {"type": "function", "function": {"name": "readFile", "description": "Read a file",
        "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}},
    {"type": "function", "function": {"name": "editFile", "description": "Find and replace in file",
        "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "find": {"type": "string"}, "replace": {"type": "string"}}, "required": ["path", "find", "replace"]}}},
    {"type": "function", "function": {"name": "searchFiles", "description": "Grep for pattern",
        "parameters": {"type": "object", "properties": {"pattern": {"type": "string"}}, "required": ["pattern"]}}},
    {"type": "function", "function": {"name": "browse", "description": "Browse a URL",
        "parameters": {"type": "object", "properties": {"action": {"type": "string"}, "url": {"type": "string"}}, "required": ["action"]}}},
    {"type": "function", "function": {"name": "webSearch", "description": "Search the web",
        "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}},
]


def safe_parse_args(args_str):
    if not args_str:
        return {}
    try:
        return json.loads(args_str)
    except json.JSONDecodeError:
        result = {}
        for m in re.finditer(r'(\w+):<\|"\|>(.*?)<\|"\|>', args_str, re.DOTALL):
            result[m.group(1)] = m.group(2)
        if not result:
            for m in re.finditer(r'(\w+):([^,}\s]+)', args_str):
                result[m.group(1)] = m.group(2)
        return result


def execute_tool(name, args, mock=False):
    """Execute a tool call — mock or real."""
    if mock:
        return execute_tool_mock(name, args)
    return execute_tool_real(name, args)


def execute_tool_real(name, args):
    """Execute tool calls for real on the local system."""
    if name == "shell":
        cmd = args.get("command", "echo 'no command'")
        try:
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=30)
            output = result.stdout + result.stderr
            return output[:3000] if output else "(no output)"
        except subprocess.TimeoutExpired:
            return "Error: command timed out after 30s"
        except Exception as e:
            return f"Error: {e}"

    elif name == "writeFile":
        path = args.get("path", "/tmp/output.txt")
        content = args.get("content", "")
        try:
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            with open(path, "w") as f:
                f.write(content)
            return f"Wrote {len(content)} bytes to {path}"
        except Exception as e:
            return f"Error: {e}"

    elif name == "readFile":
        path = args.get("path", "")
        try:
            with open(path) as f:
                return f.read()[:3000]
        except Exception as e:
            return f"Error: {e}"

    elif name == "editFile":
        path = args.get("path", "")
        find = args.get("find", "")
        replace = args.get("replace", "")
        try:
            with open(path) as f:
                content = f.read()
            if find not in content:
                return f"Error: '{find}' not found in {path}"
            content = content.replace(find, replace, 1)
            with open(path, "w") as f:
                f.write(content)
            return f"Replaced in {path}"
        except Exception as e:
            return f"Error: {e}"

    elif name == "searchFiles":
        pattern = args.get("pattern", "")
        try:
            result = subprocess.run(["grep", "-rn", pattern, "."], capture_output=True, text=True, timeout=10)
            return result.stdout[:2000] if result.stdout else "No matches found"
        except Exception as e:
            return f"Error: {e}"

    elif name == "webSearch":
        query = args.get("query", "")
        return f"[Mock] Search results for '{query}': This is a mock response. In the real app, this would search DuckDuckGo."

    elif name == "browse":
        url = args.get("url", "")
        action = args.get("action", "navigate")
        if action == "navigate":
            return f"[Mock] Navigated to {url}"
        return f"[Mock] Page content from {url}: This is a mock response. In the real app, WKWebView would extract the page text."

    return f"Unknown tool: {name}"


def execute_tool_mock(name, args):
    """Mock tool execution for testing."""
    if name == "shell":
        cmd = args.get("command", "")
        if "date" in cmd:
            return "Sat Apr  5 20:00:00 EDT 2026"
        elif "ls" in cmd:
            return "file1.txt\nfile2.py\nREADME.md"
        elif "echo" in cmd:
            return cmd.replace("echo ", "").strip("'\"")
        return f"$ {cmd}\n(mock output)"
    elif name == "writeFile":
        return f"Wrote {len(args.get('content', ''))} bytes to {args.get('path', '?')}"
    elif name == "readFile":
        return f"Contents of {args.get('path', '?')}: (mock file content)"
    elif name == "webSearch":
        return f"Search results for '{args.get('query', '')}': 1. Example.com - Example result"
    elif name == "browse":
        if args.get("action") == "navigate":
            return f"Navigated to {args.get('url', '?')}"
        return f"Page text from {args.get('url', '?')}: Example page content with headlines and articles."
    return "OK"


def stream_request(base_url, messages, enable_thinking=False):
    """Send streaming request and return parsed result with live output."""
    body = {
        "model": "mlx-serve",
        "messages": messages,
        "tools": TOOLS,
        "max_tokens": 4096,
        "temperature": 0.7,
        "top_p": 0.95,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if enable_thinking:
        body["enable_thinking"] = True

    data = json.dumps(body).encode()
    req = urllib.request.Request(
        f"{base_url}/v1/chat/completions",
        data=data,
        headers={"Content-Type": "application/json"},
    )

    content_parts = []
    reasoning_parts = []
    tool_calls = []
    finish_reason = "unknown"
    prompt_tokens = 0
    completion_tokens = 0
    first_content = True
    first_reasoning = True

    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            for raw_line in resp:
                line = raw_line.decode().strip()
                if not line.startswith("data: ") or line == "data: [DONE]":
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
                    if first_content:
                        sys.stdout.write(f"{BOLD}")
                        first_content = False
                    sys.stdout.write(delta["content"])
                    sys.stdout.flush()
                    content_parts.append(delta["content"])

                if delta.get("reasoning_content"):
                    if first_reasoning:
                        sys.stdout.write(f"{DIM}💭 ")
                        first_reasoning = False
                    sys.stdout.write(delta["reasoning_content"])
                    sys.stdout.flush()
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

    except Exception as e:
        print(f"\n{RED}Error: {e}{NC}")
        return {"content": "", "reasoning": "", "tool_calls": None,
                "finish_reason": "error", "prompt_tokens": 0, "completion_tokens": 0}

    if not first_content or not first_reasoning:
        print(NC)  # Reset color

    return {
        "content": "".join(content_parts),
        "reasoning": "".join(reasoning_parts),
        "tool_calls": tool_calls if tool_calls else None,
        "finish_reason": finish_reason,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
    }


def run_agent(base_url, user_prompt, max_rounds=10, mock=False, thinking=False, interactive=False):
    """Run the full agent loop."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    print(f"\n{CYAN}{'─' * 60}{NC}")
    print(f"{BOLD}User:{NC} {user_prompt}")
    print(f"{CYAN}{'─' * 60}{NC}\n")

    total_tool_calls = 0

    for round_num in range(max_rounds):
        enable_thinking = thinking and round_num == 0

        print(f"{DIM}[Round {round_num + 1}/{max_rounds}]{NC} ", end="", flush=True)

        resp = stream_request(base_url, messages, enable_thinking=enable_thinking)

        # Pad detection
        content = resp["content"].replace("<pad>", "").strip()
        if not content and not resp["tool_calls"] and resp["finish_reason"] != "tool_calls":
            print(f"{RED}(pad-only output — model failed to respond){NC}")
            break

        if resp["finish_reason"] == "tool_calls" and resp["tool_calls"]:
            for tc in resp["tool_calls"]:
                total_tool_calls += 1
                args = safe_parse_args(tc["arguments"])
                arg_preview = json.dumps(args)
                if len(arg_preview) > 80:
                    arg_preview = arg_preview[:77] + "..."

                print(f"\n{YELLOW}🔧 {tc['name']}{NC}({arg_preview})")

                result = execute_tool(tc["name"], args, mock=mock)
                result_preview = result[:200].replace("\n", "\\n")
                print(f"{DIM}   → {result_preview}{NC}")

                messages.append({
                    "role": "assistant", "content": None,
                    "tool_calls": [{"id": tc["id"], "type": "function",
                                     "function": {"name": tc["name"], "arguments": tc["arguments"]}}]
                })
                messages.append({"role": "tool", "tool_call_id": tc["id"], "content": result})
        else:
            # Final content response
            messages.append({"role": "assistant", "content": resp["content"]})

            # HITL: ask user if they want to continue
            if interactive:
                print(f"\n{CYAN}{'─' * 60}{NC}")
                try:
                    follow_up = input(f"{BOLD}Follow-up (enter to quit):{NC} ").strip()
                    if follow_up:
                        messages.append({"role": "user", "content": follow_up})
                        print(f"{CYAN}{'─' * 60}{NC}\n")
                        continue
                except (EOFError, KeyboardInterrupt):
                    pass
            break

    print(f"\n{CYAN}{'─' * 60}{NC}")
    print(f"{DIM}Rounds: {round_num + 1} | Tool calls: {total_tool_calls} | "
          f"Prompt: {resp.get('prompt_tokens', '?')} | Completion: {resp.get('completion_tokens', '?')}{NC}")
    print(f"{CYAN}{'─' * 60}{NC}")


def main():
    parser = argparse.ArgumentParser(description="CLI agent for mlx-serve")
    parser.add_argument("prompt", nargs="?", help="User prompt")
    parser.add_argument("--port", type=int, default=8080, help="Server port (default: 8080)")
    parser.add_argument("--max-rounds", type=int, default=10, help="Max agent loop rounds")
    parser.add_argument("--mock", action="store_true", help="Use mock tool responses")
    parser.add_argument("--thinking", action="store_true", help="Enable thinking/reasoning")
    parser.add_argument("-i", "--interactive", action="store_true", help="Interactive HITL mode")

    args = parser.parse_args()
    base_url = f"http://127.0.0.1:{args.port}"

    # Check server
    try:
        urllib.request.urlopen(f"{base_url}/health", timeout=5)
    except Exception:
        print(f"{RED}Server not running on port {args.port}{NC}")
        sys.exit(1)

    if args.prompt:
        run_agent(base_url, args.prompt, args.max_rounds, args.mock, args.thinking, args.interactive)
    elif args.interactive:
        # Interactive chat loop
        print(f"{BOLD}MLX Agent CLI{NC} (port {args.port}, Ctrl+C to quit)")
        print(f"Flags: {'mock' if args.mock else 'real'} tools, {'thinking' if args.thinking else 'no thinking'}")
        while True:
            try:
                prompt = input(f"\n{BOLD}You:{NC} ").strip()
                if not prompt:
                    continue
                if prompt.lower() in ("quit", "exit", "q"):
                    break
                run_agent(base_url, prompt, args.max_rounds, args.mock, args.thinking, interactive=True)
            except (EOFError, KeyboardInterrupt):
                print()
                break
    else:
        parser.print_help()
        print(f"\nExamples:")
        print(f"  python3 tests/cli_agent.py 'What time is it?'")
        print(f"  python3 tests/cli_agent.py --mock 'Search for AI news'")
        print(f"  python3 tests/cli_agent.py --thinking 'Explain MoE models'")
        print(f"  python3 tests/cli_agent.py -i  # Interactive mode")


if __name__ == "__main__":
    main()
