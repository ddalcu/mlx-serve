#!/usr/bin/env python3
"""The house-rule lint over src/ (plan section 10); `zig build lint` runs it, `zig build lint-test` its tests.

  python3 scripts/lint.py [--all] [--json PATH] [--strict]

Every rule reports its violations; the exit status is 0 unless --strict and a rule named in
scripts/lint_enforced.txt has a violation. Run from the repo root. Reuses census.py's line parser.
"""
import bisect, collections, fnmatch, glob, json, os, re, subprocess, sys, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import census  # noqa: E402

# The one list of media plugins: the 16 media backends plus the per-request job plugins (.bb/decisions.md P0); they
# reach only sdk/, mlx/, abi/. Moves to src/abi/media.zig's registry once plugins register (P3).
MEDIA_PLUGINS = {"flux2", "krea", "mage_flow", "qwen_image", "moss_image", "ltx", "minimax_h3", "hunyuan3d",
                 "trellis2", "pixal3d", "acestep", "music3", "stable_audio3", "yue2", "qwen3_tts", "kokoro",
                 "laya", "kev", "d1", "clef", "nemotron_diarization"}

SRC = "src"
SOFT_CAP = census.SOFT_CAP
DETAIL_CAP = 20
PINS = "scripts/lint_pins.txt"
ENFORCED = "scripts/lint_enforced.txt"

# Spellings of a plugin's model in code besides its folder name (compared as lowercase word runs).
PLUGIN_ALIASES = {"qwen4_exp": ["qwen4"], "glm5_next": ["glm5", "glm"], "deepseek_v41": ["dsv41", "deepseek41"],
                  "deepseek_v4": ["dsv4", "deepseek"], "mimo_v2": ["mimo"], "nemotron_h": ["nemotron"],
                  "minimax_h3": ["h3"], "prism_hadamard_qwen35": ["prism", "bonsai"], "muse_glimmer": ["muse"],
                  "flux2": ["flux"], "moss_image": ["moss"]}
# Model names with no plugin folder yet: they still make an `is<Model>()` call a model-identity check.
EXTRA_MODELS = ["inkling", "embeddinggemma", "embeddinggemma2", "gemma3", "qwen3", "laguna", "kolibri1", "flint",
                "spark25", "k2horizon", "hyv3", "bailing", "minicpm", "gptoss", "mistral"]

RULES = [
    ("host-imports-engine", "src/host/ imports a file under src/engines/"),
    ("host-imports-plugin", "src/host/ imports a file under src/plugins/"),
    ("engine-imports-plugin", "src/engines/mlx/ imports a file under src/plugins/"),
    ("plugin-imports-core", "a plugin imports a file outside src/sdk/, src/mlx/, src/abi/, src/plugins/"),
    ("media-plugin-imports-engine", "a media plugin imports a file under src/engines/"),
    ("model-type-compare", "a model_type string compare outside src/plugins/"),
    ("model-predicate", "an is<Model>() call outside src/plugins/"),
    ("plugin-name", "a plugin identity (code identifier or model-type string) in src/engines/ or src/host/"),
    ("raw-mlx-c", "raw mlx-c (`mlx.mlx_*`) outside src/mlx/"),
    ("env-read", "an environment read (env API, or an MLX_SERVE_ name passed to an env helper) outside src/env.zig"),
    ("env-mention", "any other MLX_SERVE_ string outside src/env.zig (logs, test names)"),
    ("inline-kernel", "Metal/CUDA source in a Zig multiline string (one per string)"),
    ("file-header", "a .zig file that does not open with `//!`"),
    ("history-comment", "a comment carrying \"PR #\", an ISO date, a P1 move marker or \"this/it/we/code used to\""),
    ("source-scan-test", "a test that @embedFile's a .zig file, directly or through a const"),
    ("graph-scope", "a fn taking *Graph and returning an Array without a scope()"),
    ("pin-unmatched", "a scripts/lint_pins.txt line no file matches (a `!` line: one some file holds)"),
    ("file-over-cap", f"a file over {SOFT_CAP} production lines"),
]
RULE_IDS = [r for r, _ in RULES]

LEX = re.compile(r'"(?:[^"\\]|\\.)*"?|\'(?:[^\'\\]|\\.)*\'?|//.*')
IMPORT = re.compile(r'(?:(?:pub\s+)?const\s+(\w+)\s*=\s*)?@import\("([^"]+\.zig)"\)')
WORD = re.compile(r'[A-Z]{2,}\d*(?=[A-Z][a-z]|[^A-Za-z]|$)|[A-Z]?[a-z]+\d*|[A-Z]\d*|\d+')
IDENT = re.compile(r'\b[A-Za-z_]\w*')
IS_CALL = re.compile(r'\bis([A-Z0-9]\w*)\s*\(')
ENV = re.compile(r'\b(?:get|set|unset)env\s*\(|\b(?:get|has)Env\w*\s*\(|\bEnvMap\b|\benviron\b')
# An MLX_SERVE_ name handed to an env helper (`envFlagCached(&x, "MLX_SERVE_…")`, `readEnvUsize("MLX_SERVE_…")`).
ENV_HELPER = re.compile(r'\b\w*(?:env|Env)\w*\s*\((?:[^()"]|"(?:[^"\\]|\\.)*")*"MLX_SERVE_')
HISTORY = re.compile(r'\bPR ?#\d|\b(?:19|20)\d\d-(?:0[1-9]|1[0-2])-(?:0[1-9]|[12]\d|3[01])\b'
                     r'|\b[Ss]plit out of\b|\(P1\b|\b(?:[Tt]his|[Ii]t|[Ww]e|[Cc]ode) used to\b')
# Words that make a name a chat/tool dialect fact rather than a model identity (GLM `<arg_key>` tags, muse channels).
DIALECT = {"tool", "tools", "tag", "tags", "dialect", "arg", "args", "key", "call", "calls", "think", "channel",
           "channels", "header", "template", "eom", "msg", "message", "harmony", "hermes", "reasoning", "opener"}
MODEL_TYPE_STR = re.compile(r'^[A-Za-z][\w.\-]*$')
FILE_EXT = re.compile(r'\.[a-z]{2,8}$')
GRAPH_PARAM = re.compile(r'(\w+)\s*:\s*\*(?:\w+\.)*Graph\b')
RETURNS_ARRAY = re.compile(r'\)\s*(?:\w+)?!?(?:\w+\.)*Array\s*\{')
FOLD = str.maketrans("", "", "_.")
FORK = re.compile(r'^//!\s*copied_from:\s*(\S+)')


class Src:
    """One Zig file: lines, test flags, and per line the code (strings blanked), string contents and comment."""

    def __init__(self, path, text):
        self.path = path
        lines = text.split("\n")
        if lines and lines[-1] == "":
            lines.pop()
        self.lines = lines
        self.is_test, self.tests = parse_tests(lines)
        self.code, self.strs, self.comment, self.multiline = [], [], [], []
        for l in lines:
            ml = bool(census.MULTILINE.match(l))
            self.multiline.append(ml)
            if ml:
                self.code.append(""); self.strs.append([l.lstrip()[2:]]); self.comment.append("")
                continue
            if '"' not in l and "'" not in l and "//" not in l:
                self.code.append(l); self.strs.append([]); self.comment.append("")
                continue
            code, strs, com, last = [], [], "", 0
            for m in LEX.finditer(l):
                t = m.group(0)
                code.append(l[last:m.start()])
                if t.startswith("//"):
                    com = t[2:]
                    last = len(l)
                    break
                if t[0] == '"':
                    strs.append(t[1:-1] if len(t) > 1 and t.endswith('"') else t[1:])
                code.append(" ")
                last = m.end()
            code.append(l[last:])
            self.code.append("".join(code)); self.strs.append(strs); self.comment.append(com)
        self.imports = []  # (line index, alias or None, resolved path)
        for i, (l, c) in enumerate(zip(lines, self.code)):
            if "@import" in c:
                for m in IMPORT.finditer(l):
                    self.imports.append((i, m.group(1), os.path.normpath(os.path.join(os.path.dirname(path), m.group(2)))))

    @property
    def prod_lines(self):
        return len(self.lines) - sum(self.is_test)


def parse_tests(lines):
    """census.parse over lines in memory: (is_test per line, test blocks as (start, end))."""
    is_test, blocks, i = [False] * len(lines), [], 0
    while i < len(lines):
        if census.TESTS_MARKER.match(lines[i]):
            for k in range(i, len(lines)):
                is_test[k] = True
        m = census.TEST_OPEN.match(lines[i])
        if m:
            e = census.block_end(lines, i, m.group(1))
            blocks.append((i, e))
            for k in range(i, e + 1):
                is_test[k] = True
            i = e + 1
            continue
        i += 1
    return is_test, blocks


def under(path, *dirs):
    return any(path == d or path.startswith(d.rstrip("/") + "/") for d in dirs)


def plugin_of(path):
    parts = path.split("/")
    return parts[2] if len(parts) > 3 and parts[:2] == ["src", "plugins"] else None


def words(s):
    return [w.lower() for w in WORD.findall(s)]


def runs_matching(ws, tokens):
    """Tokens equal to a run of up to four consecutive words; a token of three characters or fewer (`h3`, `ltx`)
    only as one whole word, so `h_3d` is not `h3`."""
    hits = set()
    for i in range(len(ws)):
        acc = ""
        for j in range(i, min(i + 4, len(ws))):
            acc += ws[j]
            if acc in tokens and (j == i or len(acc) > 3):
                hits.add(acc)
    return hits


class Ctx:
    """What rules read beyond one file: plugin names and their model tokens."""

    def __init__(self, plugins):
        self.token_plugin = {}
        for p in plugins:
            for t in [p.replace("_", "")] + PLUGIN_ALIASES.get(p, []):
                self.token_plugin[t] = p
        self.model_tokens = set(self.token_plugin) | set(EXTRA_MODELS)
        alts = sorted(self.token_plugin, key=len, reverse=True)
        # A cheap prefilter over a whole file: a line names a plugin only if a token appears in it once
        # lowercased with `_` and `.` dropped.
        self.prefilter = re.compile("|".join(re.escape(t) for t in alts)) if alts else None


# ── rules: each takes (Src, Ctx) and yields (line index, message) ──

def r_imports(pred, label):
    def rule(f, ctx):
        for i, _, dest in f.imports:
            if pred(f.path, dest):
                yield i, f"{label} {dest}"
    return rule


def host_engine(src, dest):
    return under(src, "src/host") and under(dest, "src/engines")


def host_plugin(src, dest):
    return under(src, "src/host") and under(dest, "src/plugins")


def engine_plugin(src, dest):
    return under(src, "src/engines/mlx") and under(dest, "src/plugins")


def plugin_core(src, dest):
    return plugin_of(src) is not None and not under(dest, "src/sdk", "src/mlx", "src/abi", "src/plugins")


def media_engine(src, dest):
    return plugin_of(src) in MEDIA_PLUGINS and under(dest, "src/engines")


def r_model_type_compare(f, ctx):
    if plugin_of(f.path):
        return
    for i, l in enumerate(f.lines):
        if "model_type" in f.code[i]:
            uncommented = l[:len(l) - len(f.comment[i]) - 2] if f.comment[i] else l
            if census.mem_calls_on(uncommented, "model_type"):
                yield i, l.strip()[:120]


def r_model_predicate(f, ctx):
    if plugin_of(f.path):
        return
    for i, c in enumerate(f.code):
        if "is" not in c:
            continue
        for m in IS_CALL.finditer(c):
            hit = runs_matching(words(m.group(1)), ctx.model_tokens)
            if hit:
                yield i, f"is{m.group(1)}() names {', '.join(sorted(hit))}"


def ident_identity(t, ctx, code_idents):
    """Plugin tokens a code identifier names as an identity. Facts are not identity: SCREAMING_CASE constants
    (`MUSE_EOM_TAG`), snake_case names that only start with a model (`diffusion_max_steps`), dialect names
    (`museThinkOpenerAt`, `glmArgKey`), and a bare two-character variable (`d1`)."""
    if t.upper() == t:
        return set()
    ws = words(t)
    if DIALECT.intersection(ws):
        return set()
    toks = runs_matching(ws, ctx.token_plugin)
    if "_" in t and t == t.lower() and t.replace("_", "") not in ctx.token_plugin:
        toks = {k for k in toks if not any("".join(ws[:n]) == k for n in range(1, 5))}
    return {k for k in toks if not (len(k) <= 2 and t.lower() == k and t in code_idents)}


def str_identity(s, ctx):
    """Plugin tokens a string literal names as an identity: a whole model-type-like token (`"qwen3_5_moe"`), never
    a sentence, a log label, a path, a file name or a SCREAMING_CASE name."""
    s = s.strip()
    if not MODEL_TYPE_STR.match(s) or FILE_EXT.search(s) or s.upper() == s:
        return set()
    ws = words(s)
    return set() if DIALECT.intersection(ws) else runs_matching(ws, ctx.token_plugin)


def r_plugin_name(f, ctx):
    if not under(f.path, "src/engines", "src/host") or ctx.prefilter is None:
        return
    folded = "\n".join(f.lines).lower().translate(FOLD)
    starts = [0] + [m.end() for m in re.finditer("\n", folded)]
    candidates = sorted({bisect.bisect_right(starts, m.start()) - 1 for m in ctx.prefilter.finditer(folded)})
    for i in candidates:
        code_idents = IDENT.findall(f.code[i])
        toks = set()
        for t in code_idents:
            toks |= ident_identity(t, ctx, code_idents)
        if not f.multiline[i]:
            for st in f.strs[i]:
                toks |= str_identity(st, ctx)
        if toks:
            yield i, f"{', '.join(sorted({ctx.token_plugin[k] for k in toks}))}: {f.lines[i].strip()[:100]}"


def r_raw_mlx_c(f, ctx):
    if under(f.path, "src/mlx"):
        return
    aliases = {"mlx"} | {a for _, a, d in f.imports if a and d == "src/mlx/ffi.zig"}
    pat = re.compile(r'\b(?:' + "|".join(sorted(aliases)) + r')\.mlx_\w+')
    for i, c in enumerate(f.code):
        if ".mlx_" in c:
            n = len(pat.findall(c))
            if n:
                yield i, f"{n} raw mlx-c reference{'s' if n > 1 else ''}"


# The guest agent is its own binary (its own module root, the VM's environment), so it cannot reach src/env.zig.
ENV_EXEMPT = {"src/env.zig", "src/host/sandbox/vz_agent.zig"}


def env_lines(f):
    """(line index, is a read) for every line outside ENV_EXEMPT touching the environment or an MLX_SERVE_ name."""
    if f.path in ENV_EXEMPT:
        return
    for i, c in enumerate(f.code):
        named = not f.multiline[i] and any("MLX_SERVE_" in s for s in f.strs[i])
        if ENV.search(c):
            yield i, True
        elif named:
            l = f.lines[i]
            yield i, bool(ENV_HELPER.search(l[:len(l) - len(f.comment[i]) - 2] if f.comment[i] else l))


def r_env_read(f, ctx):
    for i, read in env_lines(f):
        if read:
            yield i, f.lines[i].strip()[:120]


def r_env_mention(f, ctx):
    for i, read in env_lines(f):
        if not read:
            yield i, f.lines[i].strip()[:120]


def r_inline_kernel(f, ctx):
    i = 0
    while i < len(f.lines):
        if f.multiline[i]:
            j = i
            while j < len(f.lines) and f.multiline[j]:
                j += 1
            if census.KERNEL.search("\n".join(f.lines[i:j])):
                yield i, f"{j - i} lines"
            i = j
        else:
            i += 1


def r_file_header(f, ctx):
    if not f.lines or not f.lines[0].startswith("//!"):
        yield 0, "no //! header"


def r_history_comment(f, ctx):
    for i, com in enumerate(f.comment):
        if com:
            m = HISTORY.search(com)
            if m:
                yield i, f"\"{m.group(0)}\": {com.lstrip('/!').strip()[:100]}"


def r_source_scan_test(f, ctx):
    if not f.tests:
        return
    embed_names = set()
    for l, t in zip(f.lines, f.is_test):
        if not t and "@embedFile" in l:
            embed_names.update(re.findall(r'(?:const|var)\s+(\w+)\s*(?::[^=]*)?=\s*@embedFile\("[^"]+\.zig"\)', l))
    name_re = re.compile(r'\b(?:' + "|".join(sorted(embed_names)) + r')\b') if embed_names else None
    for s, e in f.tests:
        body = "\n".join(f.lines[s:e + 1])
        if re.search(r'@embedFile\("[^"]+\.zig"\)', body) or (name_re and name_re.search(body)):
            yield s, f.lines[s].strip()[:100]


def r_graph_scope(f, ctx):
    if under(f.path, "src/mlx"):
        return
    for i, l in enumerate(f.lines):
        m = census.FN_OPEN.match(l)
        if not m or f.is_test[i]:
            continue
        sig_end = i
        while sig_end < len(f.lines) - 1 and not f.lines[sig_end].rstrip().endswith("{"):
            sig_end += 1
        sig = " ".join(f.lines[i:sig_end + 1])
        g = GRAPH_PARAM.search(sig)
        if not g or not RETURNS_ARRAY.search(sig):
            continue
        body = "\n".join(f.code[sig_end:census.block_end(f.lines, i, m.group(1)) + 1])
        if re.search(rf'\b{g.group(1)}\b', body.split("{", 1)[-1]) and ".scope()" not in body:
            yield i, f"fn {m.group(2)}"


def r_file_over_cap(f, ctx):
    if f.prod_lines > SOFT_CAP:
        yield 0, f"{f.prod_lines} production lines"


FILE_RULES = {
    "host-imports-engine": r_imports(host_engine, "imports"),
    "host-imports-plugin": r_imports(host_plugin, "imports"),
    "engine-imports-plugin": r_imports(engine_plugin, "imports"),
    "plugin-imports-core": r_imports(plugin_core, "imports"),
    "media-plugin-imports-engine": r_imports(media_engine, "imports"),
    "model-type-compare": r_model_type_compare,
    "model-predicate": r_model_predicate,
    "plugin-name": r_plugin_name,
    "raw-mlx-c": r_raw_mlx_c,
    "env-read": r_env_read,
    "env-mention": r_env_mention,
    "inline-kernel": r_inline_kernel,
    "file-header": r_file_header,
    "history-comment": r_history_comment,
    "source-scan-test": r_source_scan_test,
    "graph-scope": r_graph_scope,
    "file-over-cap": r_file_over_cap,
}


def read_pins(text):
    """Pin lines as (line number, glob, needle, rule); `#` lines and blank lines skipped."""
    pins = []
    for n, l in enumerate(text.split("\n"), 1):
        if not l.strip() or l.lstrip().startswith("#"):
            continue
        parts = l.split("\t")
        pins.append((n, *(parts + ["", "", ""])[:3]))
    return pins


def check_pins(pins, files):
    """(pin line, message) for each pin whose glob matches no file holding its needle, or, for a `!needle`, some file
    that holds it; files: path → text."""
    for n, pat, needle, rule in pins:
        forbid = needle.startswith("!")
        needle = needle[1:] if forbid else needle
        if not pat or not needle:
            yield n, f"malformed pin (want glob<TAB>needle<TAB>rule): {pat!r}"
            continue
        hits = [p for p, t in files.items() if fnmatch.fnmatch(p, pat) and needle in t]
        if forbid and hits:
            yield n, f"{hits[0]}: {needle!r} is forbidden ({rule or 'no rule'})"
        elif not forbid and not hits:
            yield n, f"{pat}: {needle!r} not found ({rule or 'no rule'})"


def folder_of(path):
    """The reporting folder: src/<area>/<name> under engines, plugins, host, sdk; else src/<area>."""
    parts = path.split("/")
    if len(parts) <= 2:
        return "src"
    if parts[1] in ("engines", "plugins", "host", "sdk") and len(parts) > 3:
        return "/".join(parts[:3])
    return "/".join(parts[:2])


def fork_age(path, line):
    """(ISO date, age in days) of a line from git blame; ('uncommitted', 0) when it is not in a commit yet."""
    try:
        out = subprocess.run(["git", "blame", "--porcelain", "-L", f"{line + 1},{line + 1}", "--", path],
                             capture_output=True, text=True, timeout=10).stdout
        t = int(re.search(r'^author-time (\d+)', out, re.M).group(1))
        if out.startswith("0" * 40):
            return "uncommitted", 0
        return time.strftime("%Y-%m-%d", time.localtime(t)), int((time.time() - t) // 86400)
    except (AttributeError, OSError, subprocess.SubprocessError):
        return "unknown", -1


def forks(f):
    for i, l in enumerate(f.lines):
        if not l.startswith("//!"):
            break
        m = FORK.match(l)
        if m:
            yield i, m.group(1)


def run(files, pins_text, enforced):
    """Lint every file. files: path → text. Returns the report dict."""
    plugins = sorted({plugin_of(p) for p in files if plugin_of(p)})
    ctx = Ctx(plugins)
    viol = {r: [] for r in RULE_IDS}  # (path, line index, message, in test)
    prod_by_folder, fork_list = collections.Counter(), []
    for path in sorted(files):
        f = Src(path, files[path])
        prod_by_folder[folder_of(path)] += f.prod_lines
        for rid, rule in FILE_RULES.items():
            for i, msg in rule(f, ctx):
                viol[rid].append((path, i, msg, f.is_test[i] if i < len(f.is_test) else False))
        for i, src in forks(f):
            fork_list.append((path, i, src))
    for n, msg in check_pins(read_pins(pins_text), files):
        viol["pin-unmatched"].append((PINS, n - 1, msg, False))
    return {"viol": viol, "prod_by_folder": prod_by_folder, "forks": fork_list, "enforced": enforced,
            "files": len(files)}


def load_tree():
    files = {}
    for p in census.zig_files(SRC):
        files[p] = open(p, errors="replace").read()
    pins = open(PINS).read() if os.path.exists(PINS) else ""
    enforced = []
    if os.path.exists(ENFORCED):
        enforced = [l.strip() for l in open(ENFORCED) if l.strip() and not l.lstrip().startswith("#")]
    return files, pins, enforced


def report(res, seconds, show_all, json_path):
    viol, enforced = res["viol"], res["enforced"]
    print(f"mlx-serve lint: {len(RULES)} rules over {res['files']} files of src/ in {seconds:.2f} s; "
          f"enforced: {', '.join(enforced) or 'none (report-only)'}")
    print(f"\n {'#':>2}  {'rule':<28} {'total':>7} {'prod':>7}  {'enf':<3}  description")
    for n, (rid, desc) in enumerate(RULES, 1):
        v = viol[rid]
        print(f" {n:>2}  {rid:<28} {len(v):>7} {sum(1 for x in v if not x[3]):>7}  "
              f"{'yes' if rid in enforced else '':<3}  {desc}")

    per_folder = collections.defaultdict(collections.Counter)
    for rid, v in viol.items():
        for path, *_ in v:
            per_folder[folder_of(path) if path.startswith("src/") else path][rid] += 1
    print("\n== violations per folder (columns = rule #)")
    print(f"{'folder':<34}" + "".join(f"{n:>6}" for n in range(1, len(RULES) + 1)))
    for folder in sorted(per_folder):
        print(f"{folder:<34}" + "".join(f"{per_folder[folder][r] or '.':>6}" for r in RULE_IDS))

    for n, (rid, desc) in enumerate(RULES, 1):
        v = viol[rid]
        if not v:
            continue
        shown = v if show_all else v[:DETAIL_CAP]
        print(f"\n== {n}. {rid}: {len(v)} ({desc})")
        by_file = collections.Counter(x[0] for x in v)
        if len(by_file) > 1:
            top = by_file.most_common(None if show_all else 10)
            print("  per file: " + ", ".join(f"{p} {c}" for p, c in top) + (" ..." if len(top) < len(by_file) else ""))
        for path, i, msg, tst in shown:
            print(f"  {path}:{i + 1}{' [test]' if tst else ''}  {msg}")
        if len(shown) < len(v):
            print(f"  ... {len(v) - len(shown)} more (--all)")

    print(f"\n== forks (`//! copied_from:`): {len(res['forks'])}")
    fork_rows = []
    for path, i, src in res["forks"]:
        date, age = fork_age(path, i)
        fork_rows.append({"file": path, "line": i + 1, "copied_from": src, "date": date, "age_days": age})
        print(f"  {path}:{i + 1}  from {src}  {date} ({age} days)")

    print("\n== ratchet: production lines per folder")
    for folder, n in sorted(res["prod_by_folder"].items(), key=lambda kv: (-kv[1], kv[0])):
        print(f"{n:9d}  {folder}")

    if json_path:
        out = {"files": res["files"], "seconds": round(seconds, 3), "enforced": enforced,
               "rules": {rid: {"description": desc, "total": len(viol[rid]),
                               "prod": sum(1 for x in viol[rid] if not x[3]), "enforced": rid in enforced,
                               "by_folder": {fo: c[rid] for fo, c in sorted(per_folder.items()) if c[rid]},
                               "by_file": dict(sorted(collections.Counter(x[0] for x in viol[rid]).items()))}
                         for rid, desc in RULES},
               "forks": fork_rows, "prod_lines_by_folder": dict(sorted(res["prod_by_folder"].items()))}
        os.makedirs(os.path.dirname(os.path.abspath(json_path)), exist_ok=True)
        with open(json_path, "w") as fh:
            json.dump(out, fh, indent=1)


def failed_rules(res):
    """Enforced rules with at least one violation: what --strict fails on."""
    return [r for r in res["enforced"] if res["viol"][r]]


def main(argv):
    show_all, strict, json_path = "--all" in argv, "--strict" in argv, None
    if "--json" in argv:
        json_path = argv[argv.index("--json") + 1]
    t0 = time.time()
    files, pins, enforced = load_tree()
    unknown = [r for r in enforced if r not in RULE_IDS]
    if unknown:
        print(f"lint: unknown rule in {ENFORCED}: {', '.join(unknown)}", file=sys.stderr)
        return 2
    res = run(files, pins, enforced)
    report(res, time.time() - t0, show_all, json_path)
    failed = failed_rules(res)
    if strict and failed:
        print(f"\nlint: FAILED (strict): {', '.join(failed)}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
