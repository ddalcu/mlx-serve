#!/usr/bin/env python3
"""Hermetic test for app/Sources/MLXServe/Resources/guest/mlx-computer.py.

A fake pyatspi (tree of dicts) + a recording xdotool stand-in, so the observe
format, id cache, elision cap, stale-id refusal and the action → xdotool
mapping are pinned on plain python3 on macOS. No X, no VM.

    python3 tests/guest_mlx_computer_test.py
"""
import importlib.util
import subprocess
import json
import os
import sys
import tempfile
import unittest

# The script lives under app/Sources/MLXServe/Resources, which build.sh stages
# into the .app; a __pycache__ beside it must never appear.
sys.dont_write_bytecode = True

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPT = os.path.join(HERE, "..", "app", "Sources", "MLXServe", "Resources", "guest", "mlx-computer.py")
spec = importlib.util.spec_from_file_location("mlx_computer", SCRIPT)
mc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mc)


# ------------------------------------------------------------ fake pyatspi

class State:
    def __init__(self, *names):
        self.names = set(names)

    def contains(self, s):
        return s in self.names


class Component:
    def __init__(self, ext):
        self.ext = ext

    def getExtents(self, _coords):
        return self.ext


class Text:
    def __init__(self, value):
        self.value = value

    def getText(self, a, b):
        return self.value


class Hyperlink:
    def __init__(self, uri):
        self.uri = uri

    def getURI(self, _i):
        return self.uri


class Node:
    def __init__(self, role, name="", ext=(0, 0, 10, 10), states=("showing", "visible"),
                 children=(), text=None, href=None):
        self.role, self.name, self.ext, self.states = role, name, ext, states
        self.kids, self.text, self.href = list(children), text, href

    def queryHyperlink(self):
        if self.href is None:
            raise NotImplementedError
        return Hyperlink(self.href)

    def getRoleName(self):
        return self.role

    def getState(self):
        return State(*self.states)

    def queryComponent(self):
        return Component(self.ext)

    def queryText(self):
        if self.text is None:
            raise NotImplementedError
        return Text(self.text)

    @property
    def childCount(self):
        return len(self.kids)

    def getChildAtIndex(self, i):
        return self.kids[i]


class FakeAtspi:
    DESKTOP_COORDS = 0
    STATE_SHOWING, STATE_VISIBLE, STATE_ACTIVE, STATE_FOCUSED, STATE_CHECKED = \
        "showing", "visible", "active", "focused", "checked"

    def __init__(self, desktop):
        self._desktop = desktop
        self.Registry = self

    def getDesktop(self, _i):
        return self._desktop


class FakeActions(mc.Actions):
    def __init__(self):
        self.calls = []
        self.slept = []
        super().__init__(run=self._run, sleep=self.slept.append)

    def _run(self, argv, capture=True):
        self.calls.append(argv)
        if argv[:2] == ["xdotool", "getdisplaygeometry"]:
            return "1280 800\n"
        if argv[:2] == ["xdotool", "getactivewindow"]:
            return "Terminal - root@sandbox\n"
        return ""


def desktop_with(active_children, other_children=()):
    active = Node("frame", "Terminal - root@sandbox", (0, 0, 800, 600),
                  ("showing", "visible", "active"), active_children)
    other = Node("frame", "Files", (100, 100, 500, 400), ("showing", "visible"), other_children)
    app1 = Node("application", "xfce4-terminal", children=[active])
    app2 = Node("application", "thunar", children=[other])
    return Node("desktop frame", children=[app1, app2])


class ObserveTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.cache = os.path.join(self.tmp, "observe.json")
        self.actions = FakeActions()

    def test_flattens_visible_named_elements_with_ids_and_centers(self):
        tree = desktop_with([
            Node("panel", children=[
                Node("push button", "Save", (10, 20, 100, 30)),
                Node("push button", "", (10, 60, 100, 30)),          # nameless: dropped
                Node("push button", "Hidden", (10, 90, 100, 30), states=("visible",)),  # not showing
                Node("entry", "", (10, 120, 200, 24), text="hello world"),
                Node("check box", "Wrap", (10, 150, 80, 20), states=("showing", "visible", "checked")),
            ]),
        ])
        out = mc.observe(FakeAtspi(tree), self.actions, "focused", cache_path=self.cache)
        lines = out.splitlines()
        self.assertEqual(lines[0], "screen 1280x800, active window: Terminal - root@sandbox")
        self.assertIn('[1] frame "Terminal - root@sandbox" (0,0,800,600)', lines[1])
        self.assertEqual(lines[2], '[2] push button "Save" (10,20,100,30)')
        self.assertEqual(lines[3], "[3] entry value='hello world' (10,120,200,24)")
        self.assertEqual(lines[4], '[4] check box "Wrap" (10,150,80,20) [checked]')
        self.assertEqual(len(lines), 5, out)
        cache = json.load(open(self.cache))["elements"]
        self.assertEqual(cache["2"], {"x": 60, "y": 35, "role": "push button", "name": "Save"})

    def test_focused_window_only_by_default_and_all_on_request(self):
        tree = desktop_with([Node("push button", "A", (0, 0, 10, 10))],
                            [Node("push button", "B", (0, 0, 10, 10))])
        focused = mc.observe(FakeAtspi(tree), self.actions, "focused", cache_path=self.cache)
        self.assertIn('"A"', focused)
        self.assertNotIn('"B"', focused)
        everything = mc.observe(FakeAtspi(tree), self.actions, "all", cache_path=self.cache)
        self.assertIn('"A"', everything)
        self.assertIn('"B"', everything)
        # The active window comes first in the all listing.
        self.assertLess(everything.index('"A"'), everything.index('"B"'))

    def test_elision_cap_names_how_many_were_dropped(self):
        tree = desktop_with([Node("push button", "b%d" % i, (0, i, 10, 1)) for i in range(40)])
        out = mc.observe(FakeAtspi(tree), self.actions, "focused", cache_path=self.cache, max_lines=11)
        lines = out.splitlines()
        self.assertEqual(len(lines), 1 + 11 + 1, out)
        self.assertEqual(lines[-1], "... 30 more elements elided (use a narrower window or scroll)")
        # Nothing past the cap is addressable.
        self.assertEqual(len(json.load(open(self.cache))["elements"]), 11)

    def test_off_screen_and_zero_size_elements_are_dropped(self):
        tree = desktop_with([
            Node("push button", "Gone", (2000, 0, 10, 10)),
            Node("push button", "Flat", (0, 0, 0, 0)),
            Node("push button", "Here", (0, 0, 10, 10)),
        ])
        out = mc.observe(FakeAtspi(tree), self.actions, "focused", cache_path=self.cache)
        self.assertNotIn("Gone", out)
        self.assertNotIn("Flat", out)
        self.assertIn("Here", out)

    def test_a_huge_table_is_one_row_and_never_descended(self):
        cells = [Node("table cell", "", (0, 0, 10, 10), text="x%d" % i) for i in range(5000)]
        tree = desktop_with([Node("table", "Sheet1", (0, 0, 800, 600), children=cells)])
        t0 = __import__("time").time()
        out = mc.observe(FakeAtspi(tree), self.actions, "focused", cache_path=self.cache)
        self.assertLess(__import__("time").time() - t0, 2.0)
        self.assertIn('table "Sheet1"', out)
        self.assertIn("large table", out)
        self.assertNotIn("x4999", out)

    def test_children_are_capped_per_node(self):
        many = [Node("push button", "b%d" % i, (0, 0, 10, 10)) for i in range(1000)]
        self.assertEqual(len(list(mc._children(Node("panel", children=many)))), mc.MAX_CHILDREN)

    def test_empty_tree_says_so_and_points_at_the_fallbacks(self):
        tree = Node("desktop frame", children=[])
        out = mc.observe(FakeAtspi(tree), self.actions, "focused", cache_path=self.cache)
        self.assertIn("no accessible elements", out)
        self.assertIn("screenshot", out)


class TargetTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.cache = os.path.join(self.tmp, "observe.json")
        json.dump({"at": 0, "elements": {"3": {"x": 60, "y": 35, "role": "push button", "name": "Save"}}},
                  open(self.cache, "w"))

    def test_id_resolves_to_the_cached_center(self):
        self.assertEqual(mc.resolve_target("3", self.cache), (60, 35))
        self.assertEqual(mc.resolve_target("#3", self.cache), (60, 35))

    def test_raw_coordinates_pass_through(self):
        self.assertEqual(mc.resolve_target("640,400", self.cache), (640, 400))
        self.assertEqual(mc.resolve_target(" 12.0 , 7 ", self.cache), (12, 7))

    def test_stale_id_is_refused_by_name(self):
        with self.assertRaises(mc.TargetError) as ctx:
            mc.resolve_target("12", self.cache)
        self.assertIn("previous observe", str(ctx.exception))
        self.assertIn("observe again", str(ctx.exception))

    def test_no_observe_yet_is_its_own_message(self):
        with self.assertRaises(mc.TargetError) as ctx:
            mc.resolve_target("1", os.path.join(self.tmp, "missing.json"))
        self.assertIn("call observe first", str(ctx.exception))

    def test_garbage_target_is_refused(self):
        with self.assertRaises(mc.TargetError):
            mc.resolve_target("Save", self.cache)


class ActionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.cache = os.path.join(self.tmp, "observe.json")
        json.dump({"at": 0, "elements": {"1": {"x": 100, "y": 200, "role": "push button", "name": "A"},
                                         "2": {"x": 300, "y": 400, "role": "push button", "name": "B"}}},
                  open(self.cache, "w"))
        self.actions = FakeActions()

    def xdo(self):
        return [c[1:] for c in self.actions.calls if c[0] == "xdotool"]

    def test_click_moves_then_clicks_the_mapped_button(self):
        mc.do_click(self.actions, "1", "right", cache_path=self.cache)
        self.assertEqual(self.xdo(), [["mousemove", "--sync", "100", "200"], ["click", "3"]])

    def test_double_click_repeats(self):
        mc.do_click(self.actions, "2", "left", double=True, cache_path=self.cache)
        self.assertEqual(self.xdo()[-1], ["click", "--repeat", "2", "--delay", "80", "1"])

    def test_type_uses_a_delay_and_the_end_of_options_marker(self):
        mc.do_type(self.actions, "--version")
        self.assertEqual(self.xdo(), [["type", "--delay", "12", "--", "--version"]])

    def test_key_splits_a_sequence(self):
        mc.do_key(self.actions, "ctrl+l Return")
        self.assertEqual(self.xdo(), [["key", "--", "ctrl+l", "Return"]])

    def test_scroll_at_a_target_moves_first(self):
        mc.do_scroll(self.actions, "down", 2, at="2", cache_path=self.cache)
        self.assertEqual(self.xdo(), [["mousemove", "--sync", "300", "400"],
                                      ["click", "--repeat", "2", "--delay", "40", "5"]])

    def test_drag_is_down_move_up(self):
        mc.do_drag(self.actions, "1", "640,10", cache_path=self.cache)
        self.assertEqual(self.xdo(), [["mousemove", "--sync", "100", "200"], ["mousedown", "1"],
                                      ["mousemove", "--sync", "640", "10"], ["mouseup", "1"]])

    def test_screenshot_prints_the_data_uri_the_host_attaches(self):
        png = os.path.join(self.tmp, "shot.png")
        # A 2x2 RGB PNG built by hand. With PIL present it becomes a JPEG,
        # else it ships as-is.
        import struct, zlib
        def chunk(kind, body):
            return struct.pack(">I", len(body)) + kind + body + struct.pack(">I", zlib.crc32(kind + body) & 0xffffffff)
        raw = b"".join(b"\x00" + bytes([255, 0, 0, 0, 255, 0]) for _ in range(2))
        data = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", 2, 2, 8, 2, 0, 0, 0)) \
            + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b"")
        open(png, "wb").write(data)
        out = mc.do_screenshot(self.actions, path=png)
        self.assertTrue(out.startswith("[screenshot:"), out)
        self.assertRegex(out.splitlines()[1], r"^data:image/(jpeg|png);base64,[A-Za-z0-9+/=]+$")
        self.assertEqual(self.actions.calls[0], ["scrot", "-o", png])


class TypeTests(unittest.TestCase):
    """A spreadsheet row is ONE type call: Tab moves right, Return down."""
    def test_escaped_tab_and_newline_become_real_keys(self):
        self.assertEqual(mc.unescape_typed("Apple\\t1.20\\n"), "Apple\t1.20\n")
        self.assertEqual(mc.unescape_typed("real\ttab"), "real\ttab", "real control chars stay")
        self.assertEqual(mc.unescape_typed("C:\\x\\r"), "C:\\x\\r", "other backslashes untouched")

    def test_type_reports_the_moves(self):
        actions = FakeActions()
        self.assertEqual(mc.do_type(actions, "Apple\\t1.20\\n"), "typed 11 characters (2 Tab/Return moves)")
        self.assertEqual(actions.calls[-1], ["xdotool", "type", "--delay", "12", "--", "Apple\t1.20\n"])
        self.assertEqual(mc.do_type(actions, "hello"), "typed 5 characters")


class OpenTests(unittest.TestCase):
    def test_open_refuses_a_command_that_is_not_installed(self):
        with self.assertRaises(mc.TargetError) as ctx:
            mc.do_open("definitely-not-a-command-xyz --flag", FakeActions())
        self.assertIn("not an installed command", str(ctx.exception))
        self.assertIn("apt-get install", str(ctx.exception))

    class Child:
        """A fake Popen: `rc` is what poll() reports (None = still running)."""
        def __init__(self, rc=None):
            self.rc = rc

        def poll(self):
            return self.rc

    def _open(self, actions, child, log=""):
        real_popen = subprocess.Popen

        def fake_popen(*a, **k):
            # The child's stderr lands in the log file do_open hands it.
            if log and hasattr(k.get("stderr"), "write"):
                k["stderr"].write(log.encode())
            return child
        subprocess.Popen = fake_popen
        real_wait = mc.OPEN_WINDOW_WAIT_S
        mc.OPEN_WINDOW_WAIT_S = 0.05  # the fake sleep does not pass time; bound the loop
        try:
            return mc.do_open("sh -c true", actions)
        finally:
            subprocess.Popen = real_popen
            mc.OPEN_WINDOW_WAIT_S = real_wait

    def test_open_waits_for_a_new_window(self):
        actions = FakeActions()
        sets = iter([{"Desktop"}, {"Desktop"}, {"Desktop"}, {"Desktop", "Terminal"}])
        actions.window_titles = lambda: next(sets, {"Desktop", "Terminal"})
        msg = self._open(actions, self.Child(None))
        self.assertEqual(msg, "launched: sh -c true")
        self.assertGreaterEqual(len(actions.slept), 2, "polled until a NEW window appeared")

    def test_open_does_not_mistake_the_desktop_taking_focus_for_the_app(self):
        # Live 2026-09-06: a fresh desktop has no active window; "Desktop"
        # becoming active passed for LibreOffice, which had exited 1.
        actions = FakeActions()
        actions.window_titles = lambda: {"Desktop", "xfce4-panel"}
        titles = iter(["", "Desktop"])
        actions.active_window_title = lambda: next(titles, "Desktop")
        msg = self._open(actions, self.Child(None))
        self.assertIn("no window yet", msg)

    def test_open_reports_a_child_that_exited_with_an_error_and_its_stderr(self):
        actions = FakeActions()
        actions.window_titles = lambda: {"Desktop"}
        with self.assertRaises(mc.TargetError) as ctx:
            self._open(actions, self.Child(127), log="sh: 1: soffice: not found\n")
        self.assertIn("exited with code 127", str(ctx.exception))
        self.assertIn("soffice: not found", str(ctx.exception), "the last stderr lines ride the error")

    def test_open_says_no_window_yet_instead_of_a_bare_launched(self):
        actions = FakeActions()
        actions.window_titles = lambda: {"Desktop"}  # never changes
        msg = self._open(actions, self.Child(None), log="javaldx: Could not find a Java Runtime\n")
        self.assertTrue(msg.startswith("launched sh -c true; no window yet after"), msg)
        self.assertIn("observe again", msg)
        self.assertIn("javaldx", msg, "stderr so far is shown so the model knows what it is waiting on")


class CliTests(unittest.TestCase):
    def test_an_action_ends_with_a_fresh_observe(self):
        tree = desktop_with([Node("push button", "Save", (10, 20, 100, 30))])
        actions = FakeActions()
        tmp = tempfile.mkdtemp()
        mc.CACHE_PATH = os.path.join(tmp, "observe.json")
        import contextlib, io
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = mc.main(["key", "ctrl+l"], atspi=FakeAtspi(tree), actions=actions)
        self.assertEqual(rc, 0)
        out = buf.getvalue()
        self.assertIn("pressed ctrl+l", out)
        self.assertIn('[2] push button "Save"', out)
        self.assertTrue(actions.slept, "actions settle before the follow-up observe")

    def test_stale_id_is_an_error_line_not_a_traceback(self):
        tmp = tempfile.mkdtemp()
        mc.CACHE_PATH = os.path.join(tmp, "observe.json")
        json.dump({"at": 0, "elements": {}}, open(mc.CACHE_PATH, "w"))
        import contextlib, io
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = mc.main(["click", "7"], atspi=FakeAtspi(Node("desktop frame")), actions=FakeActions())
        self.assertEqual(rc, 1)
        self.assertTrue(buf.getvalue().startswith("error: id 7"), buf.getvalue())

    def test_unknown_action_prints_usage(self):
        import contextlib, io
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = mc.main(["dance"], atspi=FakeAtspi(Node("desktop frame")), actions=FakeActions())
        self.assertEqual(rc, 2)
        self.assertIn("observe", buf.getvalue())



class ReadTests(unittest.TestCase):
    """`read` is the research primitive: a page as text, links keep ids."""
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.cache = os.path.join(self.tmp, "observe.json")
        self.actions = FakeActions()

    def page(self):
        return desktop_with([
            Node("tool bar", children=[Node("push button", "Reload", (0, 0, 10, 10))]),   # chrome: skipped
            Node("document web", "Wiki", (0, 100, 1280, 600), children=[
                Node("heading", "Raspberry Pi 5", (10, 110, 300, 30)),
                Node("paragraph", "", (10, 150, 600, 40), text="It costs $60 and has 4 GB RAM."),
                Node("paragraph", "", (10, 200, 600, 40), text="It costs $60 and has 4 GB RAM."),  # dedup
                Node("link", "Buy now", (10, 260, 80, 20)),
                Node("entry", "Search", (10, 300, 200, 24), states=("showing", "visible", "focused")),
            ]),
        ])

    def test_reads_headings_text_and_clickable_links(self):
        out = mc.read(FakeAtspi(self.page()), self.actions, cache_path=self.cache)
        lines = out.splitlines()
        self.assertEqual(lines[1], "# Raspberry Pi 5")
        self.assertEqual(lines[2], "It costs $60 and has 4 GB RAM.")
        self.assertEqual(lines[3], "[1] link: Buy now")
        self.assertEqual(lines[4], "[2] entry *focused*: Search")
        self.assertEqual(len(lines), 5, out)
        self.assertNotIn("Reload", out, "browser chrome is not page content")
        cache = json.load(open(self.cache))["elements"]
        self.assertEqual(cache["1"]["name"], "Buy now")

    def test_read_is_capped_and_says_so(self):
        big = desktop_with([Node("paragraph", "", (0, i, 10, 1), text="line %d " % i + "x" * 50) for i in range(200)])
        out = mc.read(FakeAtspi(big), self.actions, max_chars=500, cache_path=self.cache)
        self.assertLess(len(out), 700)
        self.assertIn("cut at 500 characters", out)

    def test_a_dead_page_and_a_results_page_say_what_to_do_next(self):
        actions = FakeActions()
        actions.run = lambda argv, capture=True: ("1280 800\n" if argv[1] == "getdisplaygeometry"
                                                  else "404 | DigiKey — Mozilla Firefox\n")
        out = mc.read(FakeAtspi(desktop_with([])), actions, cache_path=self.cache)
        self.assertIn("does not exist", out)
        self.assertIn("alt+Left", out)
        actions.run = lambda argv, capture=True: ("1280 800\n" if argv[1] == "getdisplaygeometry"
                                                  else "pi at DuckDuckGo — Mozilla Firefox\n")
        out = mc.read(FakeAtspi(desktop_with([Node("link", "Buy a Pi", (0, 0, 10, 10))])), actions, cache_path=self.cache)
        self.assertIn("[1] link: Buy a Pi", out)
        self.assertIn("click a result [id]", out)

    def test_a_small_table_reads_as_rows(self):
        table = Node("table", "", (0, 0, 400, 100), children=[
            Node("table row", children=[Node("column header", "Model"), Node("column header", "Price")]),
            Node("table row", children=[
                Node("table cell", "", children=[Node("text leaf", "", text="Pi 5 4GB")]),
                Node("table cell", "$60")]),
            Node("table row", children=[Node("table cell", "Pi 5 8GB"), Node("table cell", "$80 | ex VAT")]),
        ])
        page = desktop_with([Node("document web", "Prices", (0, 100, 1280, 600), children=[
            Node("heading", "Prices", (10, 110, 300, 30)), table])])
        out = mc.read(FakeAtspi(page), self.actions, cache_path=self.cache)
        lines = out.splitlines()
        self.assertEqual(lines[1:], ["# Prices", "| Model | Price |", "| Pi 5 4GB | $60 |", "| Pi 5 8GB | $80 / ex VAT |"], out)

    def test_a_huge_table_is_still_skipped(self):
        cells = [Node("table row", children=[Node("table cell", "x")]) for _ in range(mc.HUGE_TABLE + 1)]
        page = desktop_with([Node("table", "", (0, 0, 400, 100), children=cells)])
        out = mc.read(FakeAtspi(page), self.actions, cache_path=self.cache)
        self.assertNotIn("| x |", out)

    def test_empty_read_points_at_the_fallbacks(self):
        out = mc.read(FakeAtspi(desktop_with([])), self.actions, cache_path=self.cache)
        self.assertIn("no readable text", out)


class ResearchTests(unittest.TestCase):
    """The multi-site loop lives in code: organic results (not ads) are
    collected by href, each is visited in Firefox and read into a digest."""
    def results_page(self):
        return desktop_with([Node("document web", "ddg", (0, 100, 1280, 600), children=[
            Node("heading", "Shop Pi 5 - Amazon AD", (0, 0, 10, 10)),
            Node("link", "Shop Pi 5 - Amazon", (0, 0, 10, 10), href="https://ads.example/x"),
            Node("heading", "Buy a Raspberry Pi 5", (0, 0, 10, 10)),
            Node("link", "Buy a Raspberry Pi 5", (0, 0, 10, 10), href="https://www.raspberrypi.com/products/raspberry-pi-5/"),
            Node("link", "www.raspberrypi.com", (0, 0, 10, 10), href="https://www.raspberrypi.com/"),
            Node("heading", "Raspberry Pi 5 - Wikipedia", (0, 0, 10, 10)),
            Node("link", "Raspberry Pi 5 - Wikipedia", (0, 0, 10, 10), href="https://en.wikipedia.org/wiki/Raspberry_Pi_5"),
        ])])

    def test_digest_drops_cookie_banners_and_starts_at_the_first_heading(self):
        lines = ["Home", "Products", "We use cookies to improve your experience. Accept all",
                 "[1] link: Skip to content", "# Raspberry Pi 5", "The Pi 5 has a 2.4 GHz CPU.",
                 "This site uses cookies", "More text.", "Next: click a result [id] above to open that site, then read it."]
        self.assertEqual(mc.digest_excerpt(lines), "# Raspberry Pi 5\nThe Pi 5 has a 2.4 GHz CPU.\nMore text.")
        # No heading: nothing is cut from the front.
        self.assertEqual(mc.digest_excerpt(["Plain text.", "More."]), "Plain text.\nMore.")

    def test_duckduckgo_redirects_are_unwrapped(self):
        self.assertEqual(mc._unwrap_redirect("https://duckduckgo.com/l/?uddg=https%3A%2F%2Fwww.raspberrypi.com%2Fproducts%2F&rut=abc"),
                         "https://www.raspberrypi.com/products/")
        self.assertEqual(mc._unwrap_redirect("https://example.com/a"), "https://example.com/a")

    def test_result_links_skip_ads_and_secondary_links(self):
        links = mc._result_links(FakeAtspi(self.results_page()), FakeActions())
        self.assertEqual(links, [("Buy a Raspberry Pi 5", "https://www.raspberrypi.com/products/raspberry-pi-5/"),
                                 ("Raspberry Pi 5 - Wikipedia", "https://en.wikipedia.org/wiki/Raspberry_Pi_5")])

    def test_research_visits_each_result_and_digests_it(self):
        visited = []
        real_nav, real_read = mc.do_navigate, mc.read
        mc.do_navigate = lambda actions, target, wait=6.0: visited.append(target)
        mc.read = lambda atspi, actions, max_chars=4000, cache_path=None: \
            "screen 1280x800, active window: X\n# Heading\n[1] link: nav\nThe Pi 5 costs $60.\nNext: click a result [id]"
        try:
            out = mc.do_research(FakeAtspi(self.results_page()), FakeActions(), "raspberry pi 5", sites=1)
        finally:
            mc.do_navigate, mc.read = real_nav, real_read
        self.assertEqual(visited, ["raspberry pi 5", "https://www.raspberrypi.com/products/raspberry-pi-5/"])
        self.assertIn("## 1. Buy a Raspberry Pi 5", out)
        self.assertIn("The Pi 5 costs $60.", out)
        self.assertNotIn("[1] link", out, "link rows are not prose")
        self.assertNotIn("Next: click", out)

    def test_research_caps_sites_at_40(self):
        real_nav = mc.do_navigate
        mc.do_navigate = lambda *a, **k: None
        try:
            out = mc.do_research(FakeAtspi(self.results_page()), FakeActions(), "x", sites=99)
        finally:
            mc.do_navigate = real_nav
        self.assertIn("(2 of 2 results visited)", out)


class NavigateTests(unittest.TestCase):
    def test_url_detection(self):
        self.assertTrue(mc._looks_like_url("https://example.com"))
        self.assertTrue(mc._looks_like_url("wikipedia.org/wiki/Pi"))
        self.assertTrue(mc._looks_like_url("localhost:8080"))
        self.assertFalse(mc._looks_like_url("raspberry pi 5 price"))
        self.assertFalse(mc._looks_like_url("doom"))

    def test_query_becomes_a_duckduckgo_html_search_and_a_running_firefox_is_reused(self):
        actions = FakeActions()
        real_run = subprocess.run
        subprocess.run = lambda argv, **kw: type("R", (), {"stdout": b"123\n"})()  # firefox running
        try:
            msg = mc.do_navigate(actions, "raspberry pi 5 price", wait=0)
        finally:
            subprocess.run = real_run
        self.assertIn("html.duckduckgo.com/html/?q=raspberry+pi+5+price", msg)
        xdo = [c[1:] for c in actions.calls if c[0] == "xdotool"]
        self.assertIn(["key", "--", "ctrl+l"], xdo)
        self.assertTrue(any(c[:3] == ["type", "--delay", "8"] for c in xdo), xdo)
        self.assertIn(["key", "--", "Return"], xdo)

if __name__ == "__main__":
    unittest.main(verbosity=1)
