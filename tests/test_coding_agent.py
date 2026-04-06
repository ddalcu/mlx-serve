#!/usr/bin/env python3
"""
Stress test: iterative HTML website coding with edit cycles.

Simulates a user directing the agent to build a website step by step,
requesting changes, checking output, and iterating — like a real
pair-programming session.

Exercises: writeFile, readFile, editFile, searchFiles, shell, browse, webSearch
"""

import json
import sys
import time
import os
import subprocess
import urllib.request

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8090
BASE = f"http://127.0.0.1:{PORT}"
WORK_DIR = "/tmp/mlx-coding-test"

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


def send_agent(message, max_rounds=10):
    global TOTAL_TOOL_CALLS, TOTAL_ROUNDS
    print(f"\n{BOLD}User:{NC} {message[:120]}{'...' if len(message) > 120 else ''}")
    resp = api("POST", "/test/agent", {
        "message": message,
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
            print(f"  {DIM}Round {r['round']}: {preview}{NC}")
        elif rtype == "pad_failure":
            print(f"  {RED}Round {r['round']}: pad failure{NC}")
    TOTAL_ROUNDS += len(rounds)
    print(f"  {DIM}[{len(rounds)} rounds, {msg_count} msgs total]{NC}")
    return resp


def check(name, condition, detail=""):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  {GREEN}PASS{NC} {name}")
    else:
        FAIL += 1
        print(f"  {RED}FAIL{NC} {name}" + (f" ({detail})" if detail else ""))


def file_contains(path, *terms):
    """Check if a file contains all given terms (case-insensitive)."""
    try:
        with open(path) as f:
            content = f.read().lower()
        return all(t.lower() in content for t in terms)
    except:
        return False


def file_size(path):
    try:
        return os.path.getsize(path)
    except:
        return 0


def get_history():
    return api("GET", "/test/history").get("messages", [])


def count_tool_types(history):
    tools = set()
    for msg in history:
        tcs = msg.get("toolCalls", [])
        if tcs:
            for tc in tcs:
                tools.add(tc.get("name", ""))
    return tools


# ═══════════════════════════════════════════════════════
print(f"\n{CYAN}{'═' * 60}{NC}")
print(f"{BOLD}  Coding Agent Stress Test{NC}")
print(f"{CYAN}{'═' * 60}{NC}")

# Setup
status = api("GET", "/test/status")
if status.get("server") != "running":
    print(f"{RED}Server not running.{NC}")
    sys.exit(1)
print(f"Server: {status.get('server')} on port {status.get('port')}")

api("POST", "/test/reset")
subprocess.run(["rm", "-rf", WORK_DIR], capture_output=True)
subprocess.run(["mkdir", "-p", WORK_DIR], capture_output=True)

# ═══════════════════════════════════════════════════════
print(f"\n{CYAN}━━━ Phase 1: Create Initial Website ━━━{NC}")

# Turn 1: Search for design inspiration
resp = send_agent(
    "I want to build a coffee shop website called 'Bean There, Done That'. "
    "First, search the web for 'modern coffee shop website design trends 2026' to get inspiration. "
    "Then search for 'best color palettes for coffee shop branding'."
)
rounds = resp.get("rounds", [])
tools_used = " ".join(r.get("tools", "") for r in rounds if r.get("type") == "tool_calls")
check("1.1: researched design trends", "webSearch" in tools_used)

# Turn 2: Browse for inspiration
resp = send_agent(
    "Browse https://www.starbucks.com to see how a major coffee brand structures their site. "
    "Also browse https://www.bluebottlecoffee.com for a more artisan approach. "
    "Note the layout patterns, navigation style, and color schemes they use."
)
rounds = resp.get("rounds", [])
tools_used = " ".join(r.get("tools", "") for r in rounds if r.get("type") == "tool_calls")
check("1.2: browsed real coffee sites", "browse" in tools_used)

# Turn 3: Create the main page
resp = send_agent(
    f"Now create the main HTML page at {WORK_DIR}/index.html for 'Bean There, Done That' coffee shop. Include:\n"
    "1. A hero section with the shop name and tagline 'Life's too short for bad coffee'\n"
    "2. A navigation bar with links to: Home, Menu, About, Contact\n"
    "3. An 'About Us' section describing the shop (est. 2020, locally owned, single-origin beans)\n"
    "4. A featured drinks section with 3 signature drinks\n"
    "5. A footer with address (123 Roast Ave, Portland, OR) and hours\n"
    "6. Embedded CSS with a warm color scheme: dark brown (#3E2723), cream (#FFF8E1), accent gold (#FFB300)\n"
    "7. Responsive design with flexbox layout\n"
    "8. Google Fonts link for 'Playfair Display' for headings and 'Inter' for body text\n"
    "Make it a complete, production-quality HTML5 page."
)
rounds = resp.get("rounds", [])
tools_used = " ".join(r.get("tools", "") for r in rounds if r.get("type") == "tool_calls")
check("1.3: created index.html", "writeFile" in tools_used)
check("1.3: file exists", os.path.exists(f"{WORK_DIR}/index.html"))
check("1.3: has substantial content", file_size(f"{WORK_DIR}/index.html") > 500,
      f"only {file_size(f'{WORK_DIR}/index.html')} bytes")

# ═══════════════════════════════════════════════════════
print(f"\n{CYAN}━━━ Phase 2: Review & Iterate ━━━{NC}")

# Turn 4: Read and review
resp = send_agent(
    f"Read {WORK_DIR}/index.html and give me a detailed review:\n"
    "- Does it have all the sections I asked for?\n"
    "- Is the CSS complete and using the color scheme I specified?\n"
    "- Are there any missing HTML5 semantic elements (header, nav, main, section, footer)?\n"
    "- Check if it's responsive"
)
rounds = resp.get("rounds", [])
tools_used = " ".join(r.get("tools", "") for r in rounds if r.get("type") == "tool_calls")
check("2.1: read file for review", "readFile" in tools_used)

# Turn 5: First edit - change colors
resp = send_agent(
    f"I changed my mind about the colors. Edit {WORK_DIR}/index.html to:\n"
    "1. Change the background from dark brown to a deep navy (#1A237E)\n"
    "2. Change the accent from gold to a coral (#FF7043)\n"
    "3. Keep the cream for text\n"
    "Use the editFile tool to make each color change with find/replace."
)
rounds = resp.get("rounds", [])
tools_used = " ".join(r.get("tools", "") for r in rounds if r.get("type") == "tool_calls")
check("2.2: used editFile for color changes", "editFile" in tools_used or "writeFile" in tools_used)

# Turn 6: Verify changes
resp = send_agent(
    f"Use searchFiles to search for '#1A237E' in {WORK_DIR}/ to verify the navy color was applied. "
    f"Also search for '#3E2723' to make sure the old brown color is gone. "
    f"Then read the file to confirm the changes look right."
)
rounds = resp.get("rounds", [])
tools_used = " ".join(r.get("tools", "") for r in rounds if r.get("type") == "tool_calls")
check("2.3: verified with searchFiles", "searchFiles" in tools_used or "shell" in tools_used)

# ═══════════════════════════════════════════════════════
print(f"\n{CYAN}━━━ Phase 3: Add Menu Page ━━━{NC}")

# Turn 7: Create menu page
resp = send_agent(
    f"Create a dedicated menu page at {WORK_DIR}/menu.html with:\n"
    "1. Same nav bar and styling as index.html\n"
    "2. Categories: Hot Drinks, Cold Drinks, Pastries, Light Bites\n"
    "3. At least 5 items per category with names, descriptions, and prices\n"
    "4. A special 'Barista Picks' callout section at the top\n"
    "5. Use CSS grid for the menu layout\n"
    "6. Each item should have a subtle hover effect"
)
rounds = resp.get("rounds", [])
tools_used = " ".join(r.get("tools", "") for r in rounds if r.get("type") == "tool_calls")
check("3.1: created menu.html", "writeFile" in tools_used)
check("3.1: menu.html exists", os.path.exists(f"{WORK_DIR}/menu.html"))

# Turn 8: Add prices and details
resp = send_agent(
    f"Read {WORK_DIR}/menu.html and check:\n"
    "1. Does each item have a price?\n"
    "2. Are there at least 15 menu items total?\n"
    "3. Does it have the 4 categories I requested?\n"
    "If anything is missing, fix it by editing the file."
)
rounds = resp.get("rounds", [])
check("3.2: reviewed menu page", len(rounds) >= 1)

# ═══════════════════════════════════════════════════════
print(f"\n{CYAN}━━━ Phase 4: Add Contact Page ━━━{NC}")

# Turn 9: Create contact page
resp = send_agent(
    f"Create {WORK_DIR}/contact.html with:\n"
    "1. Same nav and styling as other pages\n"
    "2. A contact form (name, email, message) with styled inputs\n"
    "3. An embedded Google Maps iframe for '123 Roast Ave, Portland, OR'\n"
    "4. Business hours in a nice table format\n"
    "5. Phone number and email address\n"
    "6. Social media icon placeholders"
)
rounds = resp.get("rounds", [])
tools_used = " ".join(r.get("tools", "") for r in rounds if r.get("type") == "tool_calls")
check("4.1: created contact.html", "writeFile" in tools_used)

# Turn 10: Cross-link all pages
resp = send_agent(
    f"Now make sure all three pages (index.html, menu.html, contact.html) link to each other correctly:\n"
    f"1. Read each file to check the nav links\n"
    f"2. Edit any files where the nav links are missing or wrong\n"
    f"3. Use searchFiles to search for 'href' across all files in {WORK_DIR}/ to verify\n"
    "Each page should link to: index.html (Home), menu.html (Menu), contact.html (Contact)"
)
rounds = resp.get("rounds", [])
tools_used = " ".join(r.get("tools", "") for r in rounds if r.get("type") == "tool_calls")
check("4.2: cross-linked pages", len(rounds) >= 1)

# ═══════════════════════════════════════════════════════
print(f"\n{CYAN}━━━ Phase 5: Polish & QA ━━━{NC}")

# Turn 11: Edit title tags
resp = send_agent(
    f"Edit the <title> tag in each HTML file to be descriptive:\n"
    f"- index.html: 'Bean There, Done That - Portland Coffee Shop'\n"
    f"- menu.html: 'Our Menu - Bean There, Done That'\n"
    f"- contact.html: 'Contact Us - Bean There, Done That'\n"
    "Use editFile for each change."
)
rounds = resp.get("rounds", [])
tools_used = " ".join(r.get("tools", "") for r in rounds if r.get("type") == "tool_calls")
check("5.1: edited title tags", "editFile" in tools_used or "writeFile" in tools_used)

# Turn 12: CSS refactoring
resp = send_agent(
    f"Search all HTML files in {WORK_DIR}/ for any inline styles (style=) and hardcoded color values. "
    "List what you find. Then extract the common CSS into a separate file "
    f"{WORK_DIR}/styles.css and update each HTML file to link to it instead of having embedded styles. "
    "Use shell to verify the file was created."
)
rounds = resp.get("rounds", [])
tools_used = " ".join(r.get("tools", "") for r in rounds if r.get("type") == "tool_calls")
check("5.2: CSS refactoring attempted", len(rounds) >= 1)

# Turn 13: Final shell-based QA
resp = send_agent(
    f"Run these shell commands to do a final QA check:\n"
    f"1. ls -la {WORK_DIR}/ to see all files and sizes\n"
    f"2. wc -l {WORK_DIR}/*.html to count lines in each file\n"
    f"3. grep -c 'Bean There' {WORK_DIR}/*.html to verify branding consistency\n"
    f"4. python3 -c \"import os; total=sum(os.path.getsize(f'{WORK_DIR}/'+f) for f in os.listdir('{WORK_DIR}')); print(f'Total size: {{total}} bytes')\"\n"
    f"5. grep -rn 'TODO\\|FIXME\\|XXX' {WORK_DIR}/ || echo 'No TODOs found'"
)
rounds = resp.get("rounds", [])
tools_used = " ".join(r.get("tools", "") for r in rounds if r.get("type") == "tool_calls")
check("5.3: ran shell QA commands", "shell" in tools_used)

# Turn 14: One more round of browsing to compare
resp = send_agent(
    "Search the web for 'html validation best practices accessibility' and browse one of the results. "
    f"Then check if our {WORK_DIR}/index.html follows basic accessibility practices "
    "(alt attributes, semantic HTML, contrast). Read the file and suggest improvements."
)
rounds = resp.get("rounds", [])
check("5.4: accessibility review", len(rounds) >= 1)

# ═══════════════════════════════════════════════════════
print(f"\n{CYAN}━━━ File Verification ━━━{NC}")

# Check all files
for fname in ["index.html", "menu.html", "contact.html"]:
    fpath = f"{WORK_DIR}/{fname}"
    exists = os.path.exists(fpath)
    check(f"{fname} exists", exists)
    if exists:
        size = file_size(fpath)
        check(f"{fname} is substantial (>300 bytes)", size > 300, f"{size} bytes")

# Content checks on index.html
if os.path.exists(f"{WORK_DIR}/index.html"):
    check("index.html has 'Bean There'", file_contains(f"{WORK_DIR}/index.html", "Bean There"))
    check("index.html has navigation", file_contains(f"{WORK_DIR}/index.html", "nav", "menu"))
    check("index.html has footer", file_contains(f"{WORK_DIR}/index.html", "footer") or
          file_contains(f"{WORK_DIR}/index.html", "portland"))

# Content checks on menu.html
if os.path.exists(f"{WORK_DIR}/menu.html"):
    check("menu.html has prices ($)", file_contains(f"{WORK_DIR}/menu.html", "$"))
    check("menu.html has drink items", file_contains(f"{WORK_DIR}/menu.html", "latte") or
          file_contains(f"{WORK_DIR}/menu.html", "espresso") or
          file_contains(f"{WORK_DIR}/menu.html", "coffee"))

# Content checks on contact.html
if os.path.exists(f"{WORK_DIR}/contact.html"):
    check("contact.html has form", file_contains(f"{WORK_DIR}/contact.html", "form", "input"))

# Check styles.css if it was created
if os.path.exists(f"{WORK_DIR}/styles.css"):
    check("styles.css was extracted", True)
    check("styles.css has content", file_size(f"{WORK_DIR}/styles.css") > 100)
else:
    check("styles.css was extracted", False, "file not created (optional)")

# Conversation stats
history = get_history()
tool_types = count_tool_types(history)
msg_count = len(history)

print(f"\n{CYAN}━━━ Conversation Stats ━━━{NC}")
print(f"  Messages in history: {msg_count}")
print(f"  Total rounds: {TOTAL_ROUNDS}")
print(f"  Total tool calls: {TOTAL_TOOL_CALLS}")
print(f"  Distinct tools used: {sorted(tool_types)}")
check("Used 5+ distinct tool types", len(tool_types) >= 5, f"used {sorted(tool_types)}")
check("Made 15+ tool calls", TOTAL_TOOL_CALLS >= 15, f"made {TOTAL_TOOL_CALLS}")
check("30+ messages in conversation", msg_count >= 30, f"has {msg_count}")
check("Used editFile at least once", "editFile" in tool_types)
check("Used searchFiles at least once", "searchFiles" in tool_types)
check("Used webSearch", "webSearch" in tool_types)
check("Used browse", "browse" in tool_types)

# ═══════════════════════════════════════════════════════
elapsed = time.time() - START_TIME
print(f"\n{CYAN}{'═' * 60}{NC}")
print(f"  {BOLD}Coding Agent Test Results{NC}")
print(f"  Assertions: {PASS + FAIL} ({GREEN}{PASS} passed{NC}, {RED}{FAIL} failed{NC})")
print(f"  Tool calls: {TOTAL_TOOL_CALLS} across {TOTAL_ROUNDS} rounds")
print(f"  Messages: {msg_count}")
print(f"  Tools used: {', '.join(sorted(tool_types))}")
print(f"  Elapsed: {elapsed:.0f}s")
print(f"{CYAN}{'═' * 60}{NC}")

# Print file tree
print(f"\n{DIM}Files created:{NC}")
for f in sorted(os.listdir(WORK_DIR)):
    size = file_size(f"{WORK_DIR}/{f}")
    print(f"  {f}: {size:,} bytes")

sys.exit(1 if FAIL > 0 else 0)
