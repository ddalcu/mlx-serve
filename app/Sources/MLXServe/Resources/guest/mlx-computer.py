#!/usr/bin/env python3
"""mlx-computer: the guest half of mlx-serve's `computer` tool.

Runs inside the Agent Sandbox desktop (XFCE on X11, DISPLAY=:0) and prints
plain text for a small local model. Perception is the AT-SPI accessibility
tree (works with any model); a screenshot is a separate action the host only
allows when the loaded model has vision.

    observe [--window focused|all] [--max N]
    read [--max CHARS]      (the focused window as TEXT: headings, paragraphs,
                             links with ids; what a page or document says)
    navigate <url or query> (Firefox: go to a URL, or web-search words; then read)
    research <query> [--sites N] [--max CHARS]
                            (search, then visit the top N result sites in
                             Firefox one after another and return a digest:
                             title, URL, excerpt per site; N up to 40)
    click <id|x,y> [--button left|right|middle] [--double]
    type <text>
    key <combo>            (xdotool key names: ctrl+l, Return, alt+F4)
    scroll <up|down> [n] [--at <id|x,y>]
    drag <from> <to>       (each an id or x,y)
    open <command ...>     (detached, in the desktop session)
    screenshot

Every action ends with a fresh `observe --window focused`, so the model sees
the result without a second round trip. Element ids are per observe: an id
from an earlier observe is refused by name, never mapped to a moved widget.

Deps: python3-pyatspi (AT-SPI), xdotool, scrot, python3-pil (optional, for a
small JPEG; without it the screenshot ships as PNG). Installed by
SandboxDesktop.provisionScript. Hermetic test: tests/guest_mlx_computer_test.py.
"""
import base64
import io
import json
import os
import subprocess
import sys
import time

CACHE_PATH = "/tmp/mlx-observe.json"
MAX_LINES = 250
MAX_DEPTH = 40
# A spreadsheet's table has a million rows: never iterate a node's children
# past this many, never descend into a huge table at all, and stop walking
# after the time budget (a walk is a round trip to the app per node).
MAX_CHILDREN = 200
HUGE_TABLE = 200
WALK_BUDGET_S = 6.0
# Roles that carry no information for the model on their own (pure layout).
SKIP_ROLES = {"filler", "panel", "separator", "scroll pane", "viewport", "layered pane",
              "root pane", "glass pane", "split pane", "tool bar", "menu bar", "section",
              "unknown", "invalid", "redundant object", "html container", "canvas",
              "image", "icon", "tree table", "table", "scroll bar"}
# Roles whose subtree is worth descending even when the node itself is skipped.
TEXT_ROLES = {"text", "entry", "password text", "paragraph", "terminal", "document text",
              "document web", "document frame", "editable text"}


class Actions:
    """Everything that touches the system, injectable for the hermetic test."""

    def __init__(self, run=None, sleep=None):
        self.run = run or self._run
        self.sleep = sleep or time.sleep

    @staticmethod
    def _run(argv, capture=True):
        env = dict(os.environ)
        env.setdefault("DISPLAY", ":0")
        r = subprocess.run(argv, capture_output=True, text=True, env=env, timeout=30)
        if r.returncode != 0 and capture:
            raise RuntimeError("%s failed: %s" % (argv[0], (r.stderr or r.stdout).strip()))
        return r.stdout

    def xdotool(self, *args):
        return self.run(["xdotool", *args])

    def screen_size(self):
        try:
            w, h = self.xdotool("getdisplaygeometry").split()
            return int(w), int(h)
        except Exception:
            return 0, 0

    def active_window_title(self):
        try:
            return self.xdotool("getactivewindow", "getwindowname").strip()
        except Exception:
            return ""

    def window_titles(self):
        """Every mapped window's title (set). The launch wait diffs this: an
        app's window is one that was not there before, never "the active
        title changed" — on a fresh desktop nothing is active, so the
        desktop itself taking focus used to pass for the launched app."""
        try:
            out = self.xdotool("search", "--onlyvisible", "--name", "", "getwindowname", "%@")
        except Exception:
            return set()
        return set(t.strip() for t in out.splitlines() if t.strip())


# ---------------------------------------------------------------- observe

def _extents(node, atspi):
    try:
        e = node.queryComponent().getExtents(atspi.DESKTOP_COORDS)
        return int(e[0]), int(e[1]), int(e[2]), int(e[3])
    except Exception:
        return None


def _showing(node, atspi):
    try:
        st = node.getState()
        return st.contains(atspi.STATE_SHOWING) and st.contains(atspi.STATE_VISIBLE)
    except Exception:
        return True


def _has_state(node, state):
    try:
        return node.getState().contains(state)
    except Exception:
        return False


def _value_text(node, role):
    if role not in TEXT_ROLES:
        return None
    try:
        t = node.queryText().getText(0, -1)
    except Exception:
        return None
    if not t:
        return None
    t = t.rstrip("\n ").replace("\n", "⏎")
    # A terminal or document is what the model came to read: keep the TAIL
    # (the latest output) and more of it; a field keeps its head.
    if role in ("terminal", "document text", "document web", "document frame"):
        return t if len(t) <= 600 else "..." + t[-597:]
    return t if len(t) <= 120 else t[:117] + "..."


def _children(node):
    try:
        n = node.childCount
    except Exception:
        return
    for i in range(min(n, MAX_CHILDREN)):
        try:
            c = node.getChildAtIndex(i)
        except Exception:
            continue
        if c is not None:
            yield c


def _on_screen(ext, screen):
    x, y, w, h = ext
    if w <= 0 or h <= 0:
        return False
    sw, sh = screen
    if sw and sh and (x + w <= 0 or y + h <= 0 or x >= sw or y >= sh):
        return False
    return True


def _huge_table(node, role):
    if role not in ("table", "tree table", "tree"):
        return False
    try:
        return node.childCount > HUGE_TABLE
    except Exception:
        return False


def _walk_window(win, atspi, screen, entries, budget, deadline=None):
    """Depth-first over one top-level window; `entries` gets dict rows."""
    stack = [(win, 0)]
    while stack:
        if deadline and time.time() > deadline:
            budget[1] += len(stack)
            break
        node, depth = stack.pop()
        if depth > MAX_DEPTH:
            continue
        try:
            role = node.getRoleName()
        except Exception:
            continue
        if not _showing(node, atspi):
            continue
        ext = _extents(node, atspi)
        name = ""
        try:
            name = (node.name or "").strip()
        except Exception:
            pass
        value = _value_text(node, role)
        huge = _huge_table(node, role)
        if huge:
            # One row for the sheet itself; its cells are reached by typing
            # into the focused one and moving with Tab / arrow keys.
            value = "(large table: type into the focused cell, move with Tab, Return and arrow keys)"
        keep = (role not in SKIP_ROLES or huge) and ext is not None and _on_screen(ext, screen) \
            and (name or value or role in TEXT_ROLES or role in ("frame", "dialog", "window"))
        if keep:
            if budget[0] <= 0:
                budget[1] += 1
            else:
                budget[0] -= 1
                entries.append({
                    "role": role, "name": name, "value": value, "ext": ext, "depth": depth,
                    "focused": _has_state(node, atspi.STATE_FOCUSED),
                    "checked": _has_state(node, atspi.STATE_CHECKED),
                })
        if huge:
            continue
        # push children in reverse so the first child pops first
        kids = list(_children(node))
        for c in reversed(kids):
            stack.append((c, depth + 1))


def _windows(desktop, atspi, which):
    """Top-level windows across every application, active one first."""
    active, others = [], []
    for app in _children(desktop):
        for win in _children(app):
            if not _showing(win, atspi):
                continue
            (active if _has_state(win, atspi.STATE_ACTIVE) else others).append(win)
    if which == "focused" and active:
        return active
    # No active window (nothing focused yet, or a panel-only desktop): every
    # window, active first, so a fresh desktop still lists its panel.
    return active + others


def format_entries(entries, elided, header):
    lines = [header]
    for i, e in enumerate(entries, 1):
        x, y, w, h = e["ext"]
        label = '"%s"' % e["name"] if e["name"] else ""
        if e["value"] is not None:
            label += (" " if label else "") + "value=%r" % e["value"]
        flags = ""
        if e["focused"]:
            flags += " *focused*"
        if e["checked"]:
            flags += " [checked]"
        lines.append("[%d] %s %s (%d,%d,%d,%d)%s" % (i, e["role"], label, x, y, w, h, flags))
    if elided:
        lines.append("... %d more elements elided (use a narrower window or scroll)" % elided)
    return "\n".join(lines)


def observe(atspi, actions, which="focused", cache_path=CACHE_PATH, max_lines=MAX_LINES):
    screen = actions.screen_size()
    title = actions.active_window_title()
    header = "screen %dx%d, active window: %s" % (screen[0], screen[1], title or "(none)")
    entries = []
    budget = [max_lines, 0]
    try:
        desktop = atspi.Registry.getDesktop(0)
    except Exception as e:  # no a11y bus yet
        return header + "\n(accessibility tree unavailable: %s; use screenshot or x,y)" % e
    deadline = time.time() + WALK_BUDGET_S
    for win in _windows(desktop, atspi, which):
        _walk_window(win, atspi, screen, entries, budget, deadline)
    cache = {str(i): {"x": e["ext"][0] + e["ext"][2] // 2, "y": e["ext"][1] + e["ext"][3] // 2,
                      "role": e["role"], "name": e["name"]}
             for i, e in enumerate(entries, 1)}
    try:
        with open(cache_path, "w") as f:
            json.dump({"at": time.time(), "elements": cache}, f)
    except OSError:
        pass
    if not entries:
        return header + "\n(no accessible elements in the %s window; use screenshot or x,y coordinates)" % which
    return format_entries(entries, budget[1], header)


# ---------------------------------------------------------------- read

# Roles whose text is the CONTENT of a page or document, in tree order.
READ_TEXT_ROLES = {"paragraph", "text", "static", "label", "list item", "table cell",
                   "column header", "row header", "caption", "terminal", "document text",
                   "editable text", "entry", "text leaf"}
READ_SKIP_SUBTREE = {"menu bar", "tool bar", "scroll bar", "status bar"}  # Firefox parents the DOCUMENT under its tab list


def _node_text(node, role):
    """Own text of a node: queryText for text-bearing roles, else the name."""
    if role in READ_TEXT_ROLES or role in TEXT_ROLES:
        try:
            t = node.queryText().getText(0, -1)
            if t and t.strip("\n ⏎\u2062\ufffc"):
                return t
        except Exception:
            pass
    try:
        return (node.name or "")
    except Exception:
        return ""


def _cell_text(cell, depth=0):
    """A cell's text: its own, else its descendants' (Firefox nests a text
    leaf or a paragraph inside the cell)."""
    try:
        role = cell.getRoleName()
    except Exception:
        role = ""
    own = _node_text(cell, role).replace("\n", " ").strip()
    if own or depth > 4:
        return own
    parts = [_cell_text(c, depth + 1) for c in _children(cell)]
    return " ".join(p for p in parts if p).strip()


def _table_rows(table):
    """Rows of a SMALL table as lists of cell strings; [] when the table is
    not row/cell shaped (a layout table) or every cell is empty."""
    rows = []
    for row in _children(table):
        try:
            role = row.getRoleName()
        except Exception:
            continue
        if role == "table row":
            cells = [_cell_text(c) for c in _children(row)
                     if getattr(c, "getRoleName", lambda: "")() in ("table cell", "column header", "row header")]
        elif role in ("table cell", "column header", "row header"):
            cells = [_cell_text(row)]
        else:
            # <thead>/<tbody> wrappers: one level down.
            sub = _table_rows(row)
            rows.extend(sub)
            continue
        if any(cells):
            rows.append([c.replace("|", "/") for c in cells])
    return rows


def read(atspi, actions, max_chars=4000, cache_path=CACHE_PATH):
    """The focused window as text. Links and buttons keep an id (click-able);
    headings get a `#`. Browser chrome (toolbars, tabs) is skipped, so a page
    reads as its content. Also refreshes the id cache like observe."""
    screen = actions.screen_size()
    title = actions.active_window_title()
    header = "screen %dx%d, active window: %s" % (screen[0], screen[1], title or "(none)")
    try:
        desktop = atspi.Registry.getDesktop(0)
    except Exception as e:
        return header + "\n(accessibility tree unavailable: %s; use screenshot)" % e
    wins = _windows(desktop, atspi, "focused")
    if not wins:
        return header + "\n(no window)"
    out, cache, seen_text = [], {}, set()
    total = [0]
    ident = [0]

    def emit(line):
        if total[0] >= max_chars:
            return False
        out.append(line)
        total[0] += len(line) + 1
        return True

    # Every focused-set window (like observe): more than one frame can carry
    # STATE_ACTIVE, and the one with the content is not always first.
    stack = [(w, 0) for w in reversed(wins)]
    truncated = False
    deadline = time.time() + WALK_BUDGET_S
    while stack and not truncated:
        if time.time() > deadline:
            truncated = True
            break
        node, depth = stack.pop()
        if depth > MAX_DEPTH:
            continue
        try:
            role = node.getRoleName()
        except Exception:
            continue
        if role in READ_SKIP_SUBTREE or not _showing(node, atspi) or _huge_table(node, role):
            continue
        ext = _extents(node, atspi)
        kids = list(_children(node))
        name = _node_text(node, role).replace("\n", " ").strip()
        line = None
        if role in ("table", "tree table"):
            # A small table (a price list, a spec sheet) reads as rows, not as
            # its cells scattered one per line; its subtree is consumed here.
            rows = _table_rows(node)
            if rows:
                for r in rows:
                    if not emit("| " + " | ".join(r) + " |"):
                        truncated = True
                        break
                if truncated:
                    break
                continue
        if role == "heading" and name:
            line = "# " + name
        elif role in ("link", "push button", "button", "toggle button", "menu item", "check box",
                      "radio button", "combo box", "page tab") and name and ext and _on_screen(ext, screen):
            ident[0] += 1
            cache[str(ident[0])] = {"x": ext[0] + ext[2] // 2, "y": ext[1] + ext[3] // 2, "role": role, "name": name}
            line = "[%d] %s: %s" % (ident[0], role, name)
        elif role in ("entry", "password text", "editable text") and ext and _on_screen(ext, screen):
            ident[0] += 1
            cache[str(ident[0])] = {"x": ext[0] + ext[2] // 2, "y": ext[1] + ext[3] // 2, "role": role, "name": name}
            line = "[%d] %s%s: %s" % (ident[0], role, " *focused*" if _has_state(node, atspi.STATE_FOCUSED) else "", name)
        elif role in READ_TEXT_ROLES and name and not kids:
            key = name[:200]
            if key not in seen_text:
                seen_text.add(key)
                line = name
        if line is not None and not emit(line):
            truncated = True
            break
        for c in reversed(kids):
            stack.append((c, depth + 1))
    try:
        with open(cache_path, "w") as f:
            json.dump({"at": time.time(), "elements": cache}, f)
    except OSError:
        pass
    body = "\n".join(out)
    if truncated:
        body += "\n... (cut at %d characters; scroll down and read again for more)" % max_chars
    if not out:
        body = "(no readable text in the focused window; try observe, screenshot, or scroll)"
    # Steer a small model: a dead page is a dead end, a results page wants a
    # click. It guesses URLs otherwise (live: DigiKey 404s, three in a row).
    low = (title or "").lower()
    if "404" in low or "not found" in low or "problem loading" in low or "server not found" in low:
        body += "\nThis page does not exist. Do NOT guess URLs: navigate with a query (search words) and click a result [id], or press key alt+Left to go back."
    elif "duckduckgo" in low:
        body += "\nNext: click a result [id] above to open that site, then read it."
    return header + "\n" + body


# ---------------------------------------------------------------- targets

class TargetError(Exception):
    pass


def resolve_target(spec, cache_path=CACHE_PATH):
    """`12` (an id from the last observe) or `640,400` → (x, y)."""
    spec = str(spec).strip()
    if "," in spec:
        try:
            x, y = spec.split(",", 1)
            return int(float(x)), int(float(y))
        except ValueError:
            raise TargetError("coordinates must be x,y (got %r)" % spec)
    if not spec.lstrip("#").isdigit():
        raise TargetError("target must be an element id from observe or x,y coordinates (got %r)" % spec)
    ident = spec.lstrip("#")
    try:
        with open(cache_path) as f:
            cache = json.load(f)
    except (OSError, ValueError):
        raise TargetError("no observe has run yet — call observe first, then use its ids")
    el = cache.get("elements", {}).get(ident)
    if el is None:
        raise TargetError("id %s is not in the last observe (it is from a previous observe, or never existed) — call observe again" % ident)
    return int(el["x"]), int(el["y"])


# ---------------------------------------------------------------- actions

def do_click(actions, target, button="left", double=False, cache_path=CACHE_PATH):
    x, y = resolve_target(target, cache_path)
    btn = {"left": "1", "middle": "2", "right": "3"}.get(button, "1")
    actions.xdotool("mousemove", "--sync", str(x), str(y))
    args = ["click"]
    if double:
        args += ["--repeat", "2", "--delay", "80"]
    actions.xdotool(*args, btn)
    return "clicked %s at %d,%d" % (button + (" double" if double else ""), x, y)


def unescape_typed(text):
    """A model writes a row as `Apple\\t1.20\\n` (the two-character escapes,
    JSON-quoted twice on the way here); turn them into the real Tab and
    Return xdotool types, unless the text already carries real ones."""
    if "\t" in text or "\n" in text:
        return text
    return text.replace("\\t", "\t").replace("\\n", "\n")


def do_type(actions, text):
    text = unescape_typed(text)
    actions.xdotool("type", "--delay", "12", "--", text)
    moves = text.count("\t") + text.count("\n")
    note = " (%d Tab/Return moves)" % moves if moves else ""
    return "typed %d characters%s" % (len(text), note)


def do_key(actions, combo):
    actions.xdotool("key", "--", *combo.split())
    return "pressed %s" % combo


def do_scroll(actions, direction, amount=3, at=None, cache_path=CACHE_PATH):
    if at:
        x, y = resolve_target(at, cache_path)
        actions.xdotool("mousemove", "--sync", str(x), str(y))
    btn = "4" if direction == "up" else "5"
    actions.xdotool("click", "--repeat", str(max(1, int(amount))), "--delay", "40", btn)
    return "scrolled %s %d" % (direction, amount)


def do_drag(actions, src, dst, cache_path=CACHE_PATH):
    x1, y1 = resolve_target(src, cache_path)
    x2, y2 = resolve_target(dst, cache_path)
    actions.xdotool("mousemove", "--sync", str(x1), str(y1))
    actions.xdotool("mousedown", "1")
    actions.xdotool("mousemove", "--sync", str(x2), str(y2))
    actions.xdotool("mouseup", "1")
    return "dragged %d,%d to %d,%d" % (x1, y1, x2, y2)


# What a desktop app needs to join the session: the display, and the
# accessibility bridge switched on (GTK, Qt and Firefox each read their own
# variable), or its window is a blank rectangle to observe.
SESSION_ENV = {"DISPLAY": ":0", "HOME": "/root", "XDG_RUNTIME_DIR": "/run/user/0",
               "GTK_MODULES": "gail:atk-bridge", "GNOME_ACCESSIBILITY": "1",
               "QT_ACCESSIBILITY": "1", "NO_AT_BRIDGE": "0", "MOZ_ENABLE_WAYLAND": "0",
               # LibreOffice exposes a tree only through its GTK plugin
               # (package libreoffice-gtk3); the generic X11 plugin is mute.
               "SAL_USE_VCLPLUGIN": "gtk3"}


def do_open(command, actions=None):
    import shlex
    import shutil
    try:
        first = shlex.split(command)[0]
    except (ValueError, IndexError):
        first = command.split()[0] if command.split() else ""
    path = os.environ.get("PATH", "/usr/local/bin:/usr/bin:/bin") + ":/usr/games:/usr/local/games"
    if first and "/" not in first and shutil.which(first, path=path) is None:
        raise TargetError("'%s' is not an installed command. Install it with the shell tool "
                          "(apt-get install -y <package>), which lists the commands a package provides." % first)
    env = dict(os.environ)
    for k, v in SESSION_ENV.items():
        env.setdefault(k, v)
    env["PATH"] = path
    before = actions.window_titles() if actions else set()
    # stderr goes to a log, not /dev/null: an app that dies at startup
    # (missing library, bad flag, a LibreOffice profile lock) used to be
    # reported as "launched" and the model looked for a window that never
    # came. The child is polled for OPEN_EXIT_WAIT_S; a window is waited for
    # up to OPEN_WINDOW_WAIT_S.
    log_path = "/tmp/mlx-open-%d.log" % os.getpid()
    try:
        log = open(log_path, "wb")
    except OSError:
        log = subprocess.DEVNULL
    child = subprocess.Popen(["setsid", "sh", "-c", command], env=env, stdin=subprocess.DEVNULL,
                             stdout=log, stderr=log, start_new_session=True)
    if log is not subprocess.DEVNULL:
        log.close()
    return _open_outcome(command, child, actions, before, log_path)


OPEN_EXIT_WAIT_S = 3.0
OPEN_WINDOW_WAIT_S = 8.0


def _stderr_tail(log_path, lines=4):
    try:
        with open(log_path, "rb") as f:
            text = f.read().decode("utf-8", "replace")
    except OSError:
        return ""
    rows = [l.strip() for l in text.splitlines() if l.strip()]
    return "\n".join(rows[-lines:])


def _open_outcome(command, child, actions, before, log_path):
    """What happened after the launch, for the model: an exit code with the
    last stderr lines, a window, or neither yet (some apps take longer)."""
    start = time.time()
    saw_window = False
    while time.time() - start < OPEN_WINDOW_WAIT_S:
        (actions.sleep if actions else time.sleep)(0.5)
        rc = child.poll() if child is not None and hasattr(child, "poll") else None
        if rc is not None and rc != 0:
            tail = _stderr_tail(log_path)
            raise TargetError("%s exited with code %d%s" % (command, rc, (": " + tail) if tail else ""))
        if actions:
            if actions.window_titles() - before:
                saw_window = True
                break
        # A launcher that exits 0 at once (a wrapper handing off to a running
        # instance, or `libreoffice` forking) still gets the window wait.
        if rc == 0 and time.time() - start >= OPEN_EXIT_WAIT_S and not actions:
            break
    if saw_window or not actions:
        return "launched: %s" % command
    tail = _stderr_tail(log_path)
    msg = ("launched %s; no window yet after %d s (some apps take longer on first start); "
           "observe again in a few seconds" % (command, int(OPEN_WINDOW_WAIT_S)))
    if tail:
        msg += "\nstderr so far: " + tail
    return msg


def _looks_like_url(s):
    s = s.strip()
    if " " in s:
        return False
    return "://" in s or s.startswith("localhost") or ("." in s and not s.endswith("."))


def do_navigate(actions, target, wait=6.0):
    """Go to a URL (or web-search plain words) in Firefox, reusing the running
    one. A 3B model cannot chain ctrl+l / type / Return / wait reliably; this
    is that chain, and it ends in `read` so the page's text is the result."""
    target = target.strip()
    if _looks_like_url(target):
        url = target if "://" in target else "https://" + target
    else:
        from urllib.parse import quote_plus
        url = "https://html.duckduckgo.com/html/?q=" + quote_plus(target)
    running = False
    try:
        running = bool(subprocess.run(["pgrep", "-x", "firefox-esr"], capture_output=True).stdout.strip()) \
            or bool(subprocess.run(["pgrep", "-x", "firefox"], capture_output=True).stdout.strip())
    except Exception:
        pass
    if running:
        try:
            wid = actions.xdotool("search", "--class", "firefox").split()[-1]
            actions.xdotool("windowactivate", "--sync", wid)
        except Exception:
            pass
        actions.xdotool("key", "--", "ctrl+l")
        actions.sleep(0.2)
        actions.xdotool("type", "--delay", "8", "--", url)
        actions.xdotool("key", "--", "Return")
    else:
        do_open("firefox-esr --new-window '%s'" % url.replace("'", "%27"))
    # Wait for a settled page title (up to `wait` seconds): Firefox in front,
    # and not one of its interstitial titles.
    end = time.time() + wait
    settled = False
    while time.time() < end and not settled:
        actions.sleep(0.5)
        title = actions.active_window_title()
        settled = bool(title) and "Firefox" in title \
            and not title.startswith(("Loading", "Restore Session", "New Tab", "Problem loading"))
    if settled:
        actions.sleep(1.0)  # let the page's tree fill in
    return "navigated to %s" % url


def _unwrap_redirect(href):
    """DuckDuckGo wraps results as https://duckduckgo.com/l/?uddg=<url>&rut=…"""
    if not href:
        return href
    if "duckduckgo.com/l/" in href and "uddg=" in href:
        from urllib.parse import parse_qs, unquote, urlparse
        q = parse_qs(urlparse(href).query)
        if q.get("uddg"):
            return unquote(q["uddg"][0])
    return href


def _result_links(atspi, actions):
    """(title, href) of the organic results on the current DuckDuckGo HTML
    page, in order: a result is a heading followed by a link with the same
    text; ads carry an 'AD' button in the heading."""
    try:
        desktop = atspi.Registry.getDesktop(0)
    except Exception:
        return []
    out, seen = [], set()
    for win in _windows(desktop, atspi, "focused"):
        stack = [win]
        last_heading = None
        while stack:
            node = stack.pop()
            try:
                role = node.getRoleName()
            except Exception:
                continue
            if role == "heading":
                try:
                    last_heading = (node.name or "").strip()
                except Exception:
                    last_heading = None
            elif role == "link" and last_heading:
                try:
                    name = (node.name or "").strip()
                except Exception:
                    name = ""
                if name and name == last_heading.replace(" AD", "").strip() and not last_heading.endswith(" AD"):
                    href = None
                    try:
                        href = node.queryHyperlink().getURI(0)
                    except Exception:
                        pass
                    href = _unwrap_redirect(href)
                    if href and href.startswith("http") and "duckduckgo.com" not in href and href not in seen:
                        seen.add(href)
                        out.append((name, href))
                    last_heading = None
            kids = list(_children(node))
            for c in reversed(kids):
                stack.append(c)
    return out


def do_research(atspi, actions, query, sites=3, max_chars=12000, cache_path=CACHE_PATH):
    """The multi-site loop a small model cannot run itself: search, then open
    each of the top `sites` results in Firefox (visible in the desktop pane)
    and read it. Excerpts share `max_chars`, so 40 sites means short ones."""
    sites = max(1, min(int(sites), 40))
    do_navigate(actions, query)
    links = _result_links(atspi, actions)
    if not links:
        return "research: no results found for %r (the search page had no result links; try other words)" % query
    links = links[:sites]
    per_site = max(300, max_chars // len(links))
    parts = ["Research: %s (%d of %d results visited)" % (query, len(links), len(links))]
    for i, (title, href) in enumerate(links, 1):
        do_navigate(actions, href, wait=8.0)
        page = read(atspi, actions, max_chars=per_site, cache_path=cache_path)
        prose = digest_excerpt(page.split("\n")[1:])
        parts.append("\n## %d. %s\n%s\n%s" % (i, title, href, prose[:per_site] or "(no readable text)"))
    return "\n".join(parts)


BANNER_WORDS = ("accept", "consent", "we use", "this site uses", "policy", "preferences", "agree")


def _is_cookie_line(line):
    """A cookie-banner sentence, not a cookie recipe: short, and 'cookie'
    beside a consent word (or the bare consent-button phrases)."""
    low = line.lower()
    if len(line) >= 200:
        return False
    if "cookie" in low:
        return any(w in low for w in BANNER_WORDS)
    return low.strip() in ("accept all", "reject all", "manage preferences", "accept all cookies")


def digest_excerpt(lines):
    """The prose of a `read` page for a research digest: no link/button
    rows, no steering notes, no cookie-banner text, and it starts at the
    first heading when there is one (what precedes it is navigation)."""
    kept = []
    for l in lines:
        if l.startswith(("Next: click", "This page does not exist", "... (cut at")):
            continue
        if l.startswith("[") and "]" in l[:5]:
            continue
        if _is_cookie_line(l):
            continue
        kept.append(l)
    first_heading = next((i for i, l in enumerate(kept) if l.startswith("# ")), None)
    if first_heading is not None and first_heading > 0:
        kept = kept[first_heading:]
    return "\n".join(kept).strip()


def encode_screenshot(png_bytes, max_width=1280):
    """PNG → (mime, bytes): a ≤max_width JPEG q70 when PIL is around, else the PNG."""
    try:
        from PIL import Image  # python3-pil
    except ImportError:
        return "image/png", png_bytes
    im = Image.open(io.BytesIO(png_bytes)).convert("RGB")
    if im.width > max_width:
        im = im.resize((max_width, int(im.height * max_width / im.width)))
    out = io.BytesIO()
    im.save(out, format="JPEG", quality=70)
    return "image/jpeg", out.getvalue()


def do_screenshot(actions, path="/tmp/mlx-shot.png"):
    actions.run(["scrot", "-o", path])
    with open(path, "rb") as f:
        png = f.read()
    mime, data = encode_screenshot(png)
    return "[screenshot:%d bytes]\ndata:%s;base64,%s" % (len(data), mime, base64.b64encode(data).decode())


# ---------------------------------------------------------------- cli

def _flag(argv, name, default=None):
    if name in argv:
        i = argv.index(name)
        if i + 1 < len(argv):
            v = argv[i + 1]
            del argv[i:i + 2]
            return v
        del argv[i]
        return default
    return default


def main(argv, atspi=None, actions=None):
    actions = actions or Actions()
    if atspi is None:
        try:
            import pyatspi  # type: ignore
            atspi = pyatspi
        except ImportError:
            atspi = None
    if not argv:
        print(__doc__.strip())
        return 2
    cmd, rest = argv[0], list(argv[1:])
    try:
        if cmd == "observe":
            which = _flag(rest, "--window", "focused")
            n = int(_flag(rest, "--max", MAX_LINES))
            if atspi is None:
                print("accessibility tree unavailable (python3-pyatspi missing); use screenshot or x,y")
                return 1
            print(observe(atspi, actions, which, max_lines=n))
            return 0
        if cmd == "screenshot":
            print(do_screenshot(actions))
            return 0
        if cmd == "read":
            n = int(_flag(rest, "--max", 4000))
            if atspi is None:
                print("accessibility tree unavailable (python3-pyatspi missing); use screenshot")
                return 1
            print(read(atspi, actions, max_chars=n))
            return 0
        if cmd == "research":
            n = int(_flag(rest, "--sites", 3))
            mx = int(_flag(rest, "--max", 12000))
            if not rest:
                raise TargetError("research needs search words")
            if atspi is None:
                print("accessibility tree unavailable (python3-pyatspi missing)")
                return 1
            print(do_research(atspi, actions, " ".join(rest), sites=n, max_chars=mx))
            return 0
        if cmd == "navigate":
            if not rest:
                raise TargetError("navigate needs a URL or search words")
            msg = do_navigate(actions, " ".join(rest))
            print(msg)
            if atspi is not None:
                print(read(atspi, actions))
            return 0
        if cmd == "click":
            button = _flag(rest, "--button", "left")
            double = "--double" in rest
            rest = [a for a in rest if a != "--double"]
            if not rest:
                raise TargetError("click needs a target (id or x,y)")
            msg = do_click(actions, rest[0], button, double)
        elif cmd == "type":
            msg = do_type(actions, " ".join(rest))
        elif cmd == "key":
            if not rest:
                raise TargetError("key needs a combo like ctrl+l or Return")
            msg = do_key(actions, " ".join(rest))
        elif cmd == "scroll":
            at = _flag(rest, "--at")
            if not rest or rest[0] not in ("up", "down"):
                raise TargetError("scroll needs a direction: up or down")
            amount = int(rest[1]) if len(rest) > 1 else 3
            msg = do_scroll(actions, rest[0], amount, at)
        elif cmd == "drag":
            if len(rest) < 2:
                raise TargetError("drag needs a from and a to (ids or x,y)")
            msg = do_drag(actions, rest[0], rest[1])
        elif cmd == "open":
            if not rest:
                raise TargetError("open needs a command")
            msg = do_open(" ".join(rest), actions)
        else:
            print("unknown action %r\n%s" % (cmd, __doc__.strip()))
            return 2
    except TargetError as e:
        print("error: %s" % e)
        return 1
    except RuntimeError as e:
        print("error: %s" % e)
        return 1
    # Fresh state after every action, without a second round trip.
    actions.sleep(0.4)
    print(msg)
    if atspi is not None:
        print(observe(atspi, actions, "focused"))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
