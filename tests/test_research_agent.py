#!/usr/bin/env python3
"""
Stress test: agentic research + HTML report generation.

Simulates a long multi-turn conversation where the agent:
1. Searches the web for multiple topics
2. Browses several websites to gather information
3. Writes an HTML research report
4. Iterates on the report with edits
5. Verifies the final output

Exercises: webSearch, browse, writeFile, readFile, editFile, searchFiles, shell
"""

import json
import sys
import time
import urllib.request

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8090
BASE = f"http://127.0.0.1:{PORT}"
WORK_DIR = "/tmp/mlx-research-test"

GREEN = "\033[0;32m"
RED = "\033[0;31m"
YELLOW = "\033[0;33m"
CYAN = "\033[0;36m"
DIM = "\033[2m"
BOLD = "\033[1m"
NC = "\033[0m"

PASS = 0
FAIL = 0
TOTAL_TOOL_CALLS = 0
TOTAL_ROUNDS = 0
START_TIME = time.time()


def api(method, path, body=None):
    data = json.dumps(body).encode() if body else None
    req = urllib.request.Request(
        f"{BASE}{path}",
        data=data,
        headers={"Content-Type": "application/json"} if data else {},
        method=method,
    )
    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            return json.loads(resp.read())
    except Exception as e:
        return {"error": str(e)}


def send_agent(message, thinking=False, max_rounds=10):
    global TOTAL_TOOL_CALLS, TOTAL_ROUNDS
    print(f"\n{BOLD}User:{NC} {message[:120]}{'...' if len(message) > 120 else ''}")
    resp = api("POST", "/test/agent", {
        "message": message,
        "thinking": thinking,
        "max_rounds": max_rounds,
        "working_directory": WORK_DIR,
    })
    rounds = resp.get("rounds", [])
    content = resp.get("final_content", "")
    msg_count = resp.get("message_count", 0)

    for r in rounds:
        rtype = r.get("type", "?")
        if rtype == "tool_calls":
            tools = r.get("tools", "")
            print(f"  {YELLOW}Round {r['round']}: {tools}{NC}")
            TOTAL_TOOL_CALLS += tools.count("(")
        elif rtype == "content":
            preview = r.get("content", "")[:100]
            print(f"  {DIM}Round {r['round']}: {preview}...{NC}")
        elif rtype == "pad_failure":
            print(f"  {RED}Round {r['round']}: pad failure{NC}")
    TOTAL_ROUNDS += len(rounds)
    print(f"  {DIM}[{len(rounds)} rounds, {msg_count} msgs total]{NC}")
    if content:
        print(f"  {DIM}Response: {content[:150]}{'...' if len(content) > 150 else ''}{NC}")
    return resp


def check(name, condition, detail=""):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  {GREEN}PASS{NC} {name}")
    else:
        FAIL += 1
        print(f"  {RED}FAIL{NC} {name}" + (f" ({detail})" if detail else ""))


def get_history():
    return api("GET", "/test/history").get("messages", [])


def count_tool_types(history):
    """Count distinct tool types used in the conversation."""
    tools = set()
    for msg in history:
        tcs = msg.get("toolCalls", [])
        if tcs:
            for tc in tcs:
                tools.add(tc.get("name", ""))
    return tools


# ═══════════════════════════════════════════════════════
print(f"\n{CYAN}{'═' * 60}{NC}")
print(f"{BOLD}  Research Agent Stress Test{NC}")
print(f"{CYAN}{'═' * 60}{NC}")

# Setup
print(f"\n{CYAN}--- Setup ---{NC}")
status = api("GET", "/test/status")
if status.get("server") != "running":
    print(f"{RED}Server not running. Start the model first.{NC}")
    sys.exit(1)
print(f"Server: {status.get('server')} on port {status.get('port')}")

api("POST", "/test/reset")
print("Session reset.")

# Create workspace
import subprocess
subprocess.run(["rm", "-rf", WORK_DIR], capture_output=True)
subprocess.run(["mkdir", "-p", WORK_DIR], capture_output=True)

# ═══════════════════════════════════════════════════════
print(f"\n{CYAN}━━━ Phase 1: Web Research ━━━{NC}")

# Turn 1: Initial broad search
resp = send_agent(
    "I need you to research modern web frameworks for building full-stack applications. "
    "Start by searching the web for 'best web frameworks 2026 full stack comparison'. "
    "Then search for 'React vs Vue vs Svelte performance 2026'."
)
rounds = resp.get("rounds", [])
tools_used = " ".join(r.get("tools", "") for r in rounds if r.get("type") == "tool_calls")
check("Phase 1.1: used webSearch", "webSearch" in tools_used)

# Turn 2: Browse specific sites
resp = send_agent(
    "Now browse these sites to gather detailed information:\n"
    "1. Browse https://react.dev and get the key features\n"
    "2. Browse https://vuejs.org and get their selling points\n"
    "3. Browse https://svelte.dev and note what makes it different"
)
rounds = resp.get("rounds", [])
tools_used = " ".join(r.get("tools", "") for r in rounds if r.get("type") == "tool_calls")
check("Phase 1.2: used browse", "browse" in tools_used)
browse_count = tools_used.count("browse")
check("Phase 1.2: browsed multiple sites", browse_count >= 2, f"browsed {browse_count} sites")

# Turn 3: More research
resp = send_agent(
    "Search the web for 'web framework benchmark comparison throughput latency 2026' "
    "and also search for 'developer satisfaction survey web frameworks'. "
    "Browse any interesting results you find."
)
rounds = resp.get("rounds", [])
tools_used = " ".join(r.get("tools", "") for r in rounds if r.get("type") == "tool_calls")
check("Phase 1.3: continued research", len(rounds) >= 1)

# Turn 4: Backend research
resp = send_agent(
    "Now research backend frameworks too. Search for 'FastAPI vs Express vs Go Gin performance' "
    "and browse https://fastapi.tiangolo.com to get details about FastAPI's features."
)
rounds = resp.get("rounds", [])
tools_used = " ".join(r.get("tools", "") for r in rounds if r.get("type") == "tool_calls")
check("Phase 1.4: backend research", "webSearch" in tools_used or "browse" in tools_used)

# ═══════════════════════════════════════════════════════
print(f"\n{CYAN}━━━ Phase 2: Write HTML Report ━━━{NC}")

# Turn 5: Write the report
resp = send_agent(
    f"Now write a comprehensive HTML report at {WORK_DIR}/report.html that includes:\n"
    "1. An executive summary of your research findings\n"
    "2. A comparison table of frontend frameworks (React, Vue, Svelte) with columns for performance, learning curve, ecosystem, and popularity\n"
    "3. A section on backend frameworks (FastAPI, Express, Go Gin)\n"
    "4. A recommendations section\n"
    "5. Use proper HTML5 structure with embedded CSS for styling\n"
    "6. Make it visually appealing with a dark theme, good typography, and responsive layout\n"
    "Use the writeFile tool to create this report."
)
rounds = resp.get("rounds", [])
tools_used = " ".join(r.get("tools", "") for r in rounds if r.get("type") == "tool_calls")
check("Phase 2.1: wrote report file", "writeFile" in tools_used)

# Turn 6: Read and verify
resp = send_agent(
    f"Read back the report at {WORK_DIR}/report.html and tell me:\n"
    "1. Does it have all 3 frontend frameworks mentioned?\n"
    "2. Does it have a comparison table?\n"
    "3. Does it have CSS styling?\n"
    "4. How many sections does it have?"
)
rounds = resp.get("rounds", [])
tools_used = " ".join(r.get("tools", "") for r in rounds if r.get("type") == "tool_calls")
check("Phase 2.2: read report back", "readFile" in tools_used)

# ═══════════════════════════════════════════════════════
print(f"\n{CYAN}━━━ Phase 3: Edit and Improve ━━━{NC}")

# Turn 7: Edit the report
resp = send_agent(
    f"Edit the report at {WORK_DIR}/report.html to:\n"
    "1. Change the title to 'Web Framework Landscape 2026 - Comprehensive Analysis'\n"
    "2. Add a 'Methodology' section after the executive summary explaining how you gathered this data\n"
    "Use the editFile tool with find/replace to make these changes."
)
rounds = resp.get("rounds", [])
tools_used = " ".join(r.get("tools", "") for r in rounds if r.get("type") == "tool_calls")
check("Phase 3.1: edited report", "editFile" in tools_used or "writeFile" in tools_used)

# Turn 8: Search for patterns
resp = send_agent(
    f"Use the searchFiles tool to search for 'React' in {WORK_DIR}/ to verify it's mentioned in the report. "
    f"Also search for 'table' to confirm the comparison table exists. "
    f"And search for 'Methodology' to confirm our edit was applied."
)
rounds = resp.get("rounds", [])
tools_used = " ".join(r.get("tools", "") for r in rounds if r.get("type") == "tool_calls")
check("Phase 3.2: searched files", "searchFiles" in tools_used)

# Turn 9: Add more content via shell
resp = send_agent(
    f"Use the shell tool to:\n"
    f"1. Count the lines in {WORK_DIR}/report.html using 'wc -l'\n"
    f"2. Check the file size with 'ls -la {WORK_DIR}/report.html'\n"
    f"3. Search for all HTML tags used with: grep -oP '<[a-z]+' {WORK_DIR}/report.html | sort -u"
)
rounds = resp.get("rounds", [])
tools_used = " ".join(r.get("tools", "") for r in rounds if r.get("type") == "tool_calls")
check("Phase 3.3: used shell for analysis", "shell" in tools_used)

# ═══════════════════════════════════════════════════════
print(f"\n{CYAN}━━━ Phase 4: Create Additional Pages ━━━{NC}")

# Turn 10: Create a second page
resp = send_agent(
    f"Create a second HTML page at {WORK_DIR}/benchmarks.html that:\n"
    "1. Contains detailed benchmark data for the frameworks\n"
    "2. Uses the same CSS styling as the main report\n"
    "3. Links back to report.html\n"
    "4. Includes a bar chart made with pure CSS (div-based) showing relative performance scores"
)
rounds = resp.get("rounds", [])
tools_used = " ".join(r.get("tools", "") for r in rounds if r.get("type") == "tool_calls")
check("Phase 4.1: created benchmarks page", "writeFile" in tools_used)

# Turn 11: Link pages together
resp = send_agent(
    f"Edit {WORK_DIR}/report.html to add a navigation link to benchmarks.html at the top of the page. "
    f"Then read both files to verify the links work. "
    f"Also use searchFiles to find all href attributes across the {WORK_DIR}/ directory."
)
rounds = resp.get("rounds", [])
tools_used = " ".join(r.get("tools", "") for r in rounds if r.get("type") == "tool_calls")
check("Phase 4.2: linked pages", len(rounds) >= 1)

# ═══════════════════════════════════════════════════════
print(f"\n{CYAN}━━━ Phase 5: Final Verification ━━━{NC}")

# Turn 12: Final comprehensive check
resp = send_agent(
    f"Do a final check on all the files:\n"
    f"1. Use shell to run 'ls -la {WORK_DIR}/' to see all files\n"
    f"2. Read {WORK_DIR}/report.html and count sections\n"
    f"3. Read {WORK_DIR}/benchmarks.html and verify it has benchmark data\n"
    f"4. Use searchFiles to search for 'Svelte' across all files in {WORK_DIR}/\n"
    f"5. Use shell to validate the HTML: python3 -c \"from html.parser import HTMLParser; HTMLParser().feed(open('{WORK_DIR}/report.html').read()); print('Valid HTML')\""
)
rounds = resp.get("rounds", [])
check("Phase 5.1: final verification ran", len(rounds) >= 1)

# ═══════════════════════════════════════════════════════
# Verify files actually exist
print(f"\n{CYAN}━━━ File Verification ━━━{NC}")
import os
report_exists = os.path.exists(f"{WORK_DIR}/report.html")
bench_exists = os.path.exists(f"{WORK_DIR}/benchmarks.html")
check("report.html exists on disk", report_exists)
check("benchmarks.html exists on disk", bench_exists)

if report_exists:
    with open(f"{WORK_DIR}/report.html") as f:
        content = f.read()
    check("report has <html> tag", "<html" in content.lower())
    check("report has <style> or CSS", "<style" in content.lower() or "css" in content.lower())
    check("report mentions React", "react" in content.lower())
    check("report mentions Vue", "vue" in content.lower())
    check("report mentions Svelte", "svelte" in content.lower())
    check("report has <table>", "<table" in content.lower())
    check("report is substantial (>500 chars)", len(content) > 500, f"only {len(content)} chars")
    print(f"  {DIM}Report size: {len(content)} chars, {content.count(chr(10))} lines{NC}")

# Check conversation stats
history = get_history()
tool_types = count_tool_types(history)
msg_count = len(history)
print(f"\n{CYAN}━━━ Conversation Stats ━━━{NC}")
print(f"  Messages in history: {msg_count}")
print(f"  Total rounds: {TOTAL_ROUNDS}")
print(f"  Total tool calls: {TOTAL_TOOL_CALLS}")
print(f"  Distinct tools used: {tool_types}")
check("Used 5+ distinct tool types", len(tool_types) >= 5, f"used {tool_types}")
check("Made 10+ tool calls total", TOTAL_TOOL_CALLS >= 10, f"made {TOTAL_TOOL_CALLS}")
check("Conversation has 20+ messages", msg_count >= 20, f"has {msg_count}")

# ═══════════════════════════════════════════════════════
elapsed = time.time() - START_TIME
print(f"\n{CYAN}{'═' * 60}{NC}")
print(f"  {BOLD}Research Agent Test Results{NC}")
print(f"  Assertions: {PASS + FAIL} ({GREEN}{PASS} passed{NC}, {RED}{FAIL} failed{NC})")
print(f"  Tool calls: {TOTAL_TOOL_CALLS} across {TOTAL_ROUNDS} rounds")
print(f"  Messages: {msg_count}")
print(f"  Tools used: {', '.join(sorted(tool_types))}")
print(f"  Elapsed: {elapsed:.0f}s")
print(f"{CYAN}{'═' * 60}{NC}")
sys.exit(1 if FAIL > 0 else 0)
