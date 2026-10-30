#!/usr/bin/env python3
"""Source census for the refactor scoreboard; scripts/loc.sh and scripts/test_census.sh are its entry points.

  python3 scripts/census.py loc    # lines per file/folder (prod vs test), model-identity checks, kernel text
  python3 scripts/census.py tests  # tests per file, source-scan tests, env-gated tests

Run from the repo root. A test line is inside a `test ... {` block or after a `// ── Tests` marker.
"""
import os, re, sys, collections

ROOTS = {"src": "src"}
SOFT_CAP = 1500
TEST_OPEN = re.compile(r'^(\s*)test( "([^"]*)"| \{| ([A-Za-z_][\w.]*) \{)')
TESTS_MARKER = re.compile(r'^// ── [Tt]ests\b')
FN_OPEN = re.compile(r'^(\s*)(?:pub )?(?:inline )?fn (\w+)\(')
MULTILINE = re.compile(r'^\s*\\\\')
# Metal or CUDA source inside a Zig multiline string: an API word, a preprocessor line, or a typed C declaration.
KERNEL = re.compile(r'thread_position_in_grid|threadgroup|simd_|simdgroup|threadIdx|blockIdx|__global__|__device__'
                    r'|kernel void|metal::|#include|#define|#pragma|_Pragma|\[\[|static_cast<|bfloat16_t|half4|float4|uint3'
                    r'|\b(uint|int|long|short|ushort|float|half|bool|auto|size_t|uint32_t|int64_t)\s+\w+\s*(=|\[|;)'
                    r'|for \((uint|int|auto|size_t|short|ushort|long)')
MEM_CALL = re.compile(r'mem\.(eql|startsWith|endsWith|indexOf)\(u8,')


def zig_files(root):
    for d, _, fs in sorted(os.walk(root)):
        for f in sorted(fs):
            if f.endswith(".zig"):
                yield os.path.join(d, f)


def block_end(lines, i, indent):
    """Index of the line closing the block opened at line i (zig fmt: the closer is `indent}`)."""
    s = lines[i].rstrip()
    if s.endswith("}") and s.count("{") == s.count("}"):
        return i
    for j in range(i + 1, len(lines)):
        if lines[j].rstrip() == indent + "}":
            return j
    return len(lines) - 1


def parse(path):
    """(lines, is_test per line, test blocks as (start, end, name))."""
    lines = open(path, errors="replace").read().split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    is_test = [False] * len(lines)
    blocks = []
    i = 0
    while i < len(lines):
        if TESTS_MARKER.match(lines[i]):
            for k in range(i, len(lines)):
                is_test[k] = True
        m = TEST_OPEN.match(lines[i])
        if m:
            e = block_end(lines, i, m.group(1))
            blocks.append((i, e, m.group(3) or m.group(4) or ""))
            for k in range(i, e + 1):
                is_test[k] = True
            i = e + 1
            continue
        i += 1
    return lines, is_test, blocks


def mem_calls_on(line, needle):
    """mem.eql/startsWith/endsWith/indexOf calls on this line whose arguments name `needle` and a string literal."""
    n = 0
    for m in MEM_CALL.finditer(line):
        depth, j, in_str = 1, m.end(), False
        while j < len(line) and depth:
            c = line[j]
            if in_str:
                if c == "\\": j += 1
                elif c == '"': in_str = False
            elif c == '"': in_str = True
            elif c == "(": depth += 1
            elif c == ")": depth -= 1
            j += 1
        args = line[m.end():j]
        if needle in args and '"' in args:
            n += 1
    return n


def kernel_lines(lines, is_test):
    """Lines of multiline-string runs (production only) that read as Metal/CUDA source."""
    n, i = 0, 0
    while i < len(lines):
        if MULTILINE.match(lines[i]) and not is_test[i]:
            j = i
            while j < len(lines) and MULTILINE.match(lines[j]):
                j += 1
            if KERNEL.search("\n".join(lines[i:j])):
                n += j - i
            i = j
        else:
            i += 1
    return n


def model_config():
    """(field names, predicate names) of `ModelConfig` in src/sdk/config/model.zig. A predicate is a method that mutates nothing."""
    lines = open("src/sdk/config/model.zig").read().split("\n")
    s = lines.index("pub const ModelConfig = struct {")
    e = next(i for i in range(s, len(lines)) if lines[i] == "};")
    body = lines[s + 1:e]
    fields = [m.group(1) for l in body if (m := re.match(r'^    ([a-z_]\w*): ', l))]
    methods = [m.group(1) for l in body if (m := re.match(r'^    (?:pub )?fn (\w+)\(', l))]
    preds = [x for x in methods if not re.match(r'(apply|ensure|merge|add|populate|deinit)', x)]
    return fields, methods, preds


def table(title, counter, total_label="total"):
    print(f"\n== {title}")
    for k, v in sorted(counter.items(), key=lambda kv: (-kv[1], kv[0])):
        if v:
            print(f"{v:8d}  {k}")
    print(f"{sum(counter.values()):8d}  {total_label} ({sum(1 for v in counter.values() if v)} files)")


def identity(title, prod, every):
    """Model-identity checks per file: production lines, and every line (tests included)."""
    print(f"\n== {title}: production, all lines incl. tests")
    for p in sorted(every, key=lambda k: (-prod[k], -every[k], k)):
        if every[p]:
            print(f"{prod[p]:8d} {every[p]:8d}  {p}")
    for label, keep in (("total", lambda k: True), ("outside src/sdk/config/model.zig", lambda k: k != "src/sdk/config/model.zig")):
        a = sum(v for k, v in prod.items() if keep(k)); b = sum(v for k, v in every.items() if keep(k))
        fa = sum(1 for k, v in prod.items() if keep(k) and v); fb = sum(1 for k, v in every.items() if keep(k) and v)
        print(f"{a:8d} {b:8d}  {label} ({fa} / {fb} files)")


def cmd_loc():
    per_file, per_folder, per_root, kernels = {}, collections.Counter(), {}, collections.Counter()
    fields, methods, preds = model_config()
    pred_re = re.compile(r'\.(' + "|".join(preds) + r')\(')
    checks = {"pred": (collections.Counter(), collections.Counter()), "mt": (collections.Counter(), collections.Counter())}
    for name, root in ROOTS.items():
        pr = te = 0
        for p in zig_files(root):
            lines, is_test, _ = parse(p)
            t = sum(is_test)
            per_file[p] = (len(lines) - t, t)
            pr += len(lines) - t; te += t
            per_folder[(os.path.dirname(p), "prod")] += len(lines) - t
            per_folder[(os.path.dirname(p), "test")] += t
            kernels[p] = kernel_lines(lines, is_test)
            if name == "src":
                for l, tst in zip(lines, is_test):
                    if l.lstrip().startswith("//"):
                        continue
                    for key, n in (("pred", len(pred_re.findall(l))), ("mt", mem_calls_on(l, "model_type"))):
                        checks[key][1][p] += n
                        if not tst:
                            checks[key][0][p] += n
        per_root[name] = (pr, te)

    print("== roots: production, test, total lines of Zig")
    for k, (p, t) in per_root.items():
        print(f"{p:9d} {t:9d} {p + t:9d}  {k}")
    print("\n== folders: production, test")
    for d in sorted({d for d, _ in per_folder}):
        print(f"{per_folder[(d, 'prod')]:9d} {per_folder[(d, 'test')]:9d}  {d}")
    print("\n== files: production, test (by production lines)")
    for p, (a, b) in sorted(per_file.items(), key=lambda kv: -kv[1][0]):
        print(f"{a:9d} {b:9d}  {p}")
    big = [(a, p) for p, (a, _) in per_file.items() if a > SOFT_CAP]
    print(f"\n== files over {SOFT_CAP} production lines: {len(big)} "
          f"(src: {sum(1 for _, p in big if p.startswith('src/'))})")
    for a, p in sorted(big, reverse=True):
        print(f"{a:9d}  {p}")
    print(f"\n== ModelConfig: {len(fields)} fields, {len(methods)} methods, {len(preds)} predicates (non-mutating methods)")
    print("predicates: " + " ".join(preds))
    identity("ModelConfig predicate calls in src", *checks["pred"])
    identity("model_type string compares in src (mem.eql/startsWith/endsWith/indexOf)", *checks["mt"])
    print("\n== kernel text inside Zig: Metal/CUDA lines in multiline strings, production")
    for name, root in ROOTS.items():
        print(f"{sum(v for p, v in kernels.items() if p.startswith(root + '/')):8d}  {name}")
    table("kernel text per file (all roots)", kernels)


def cmd_tests():
    tests, scans, gated = collections.Counter(), collections.Counter(), collections.Counter()
    env_names = collections.Counter()
    scan_list = []
    for root in ROOTS.values():
        for p in zig_files(root):
            lines, is_test, blocks = parse(p)
            text_in = lambda s, e: "\n".join(lines[s:e + 1])
            # Declarations holding Zig source (`const src = @embedFile("x.zig")`) and fns that read the environment.
            embed_names = set(re.findall(r'(?:const|var)\s+(\w+)\s*(?::[^=]*)?=\s*@embedFile\("[^"]+\.zig"\)',
                                         "\n".join(l for l, t in zip(lines, is_test) if not t)))
            env_fns = set()
            for i, l in enumerate(lines):
                m = FN_OPEN.match(l)
                if m and "getenv(" in text_in(i, block_end(lines, i, m.group(1))):
                    env_fns.add(m.group(2))
            env_call = re.compile(r'\b(' + "|".join(env_fns) + r')\(') if env_fns else None
            for s, e, name in blocks:
                body = text_in(s, e)
                tests[p] += 1
                if re.search(r'@embedFile\("[^"]+\.zig"\)', body) or any(re.search(rf'\b{n}\b', body) for n in embed_names):
                    scans[p] += 1
                    scan_list.append(f"{p}:{s + 1}  {name}")
                if "SkipZigTest" in body and ("getenv(" in body or (env_call and env_call.search(body))):
                    gated[p] += 1
                    for n in re.findall(r'getenv\("(\w+)"\)', body):
                        env_names[n] += 1
    print("== per root: tests, source-scan, env-gated")
    for name, root in ROOTS.items():
        f = lambda c: sum(v for k, v in c.items() if k.startswith(root + "/"))
        print(f"{f(tests):8d} {f(scans):8d} {f(gated):8d}  {name}")
    table("tests per file", tests)
    table("source-scan tests (a test that @embedFile's a .zig file, directly or through a const)", scans)
    for x in scan_list:
        print("  " + x)
    table("env-gated tests (skip unless an env var is set, directly or through a helper fn)", gated)
    print("\n== env names read directly inside gated tests")
    for n, c in env_names.most_common():
        print(f"{c:8d}  {n}")


if __name__ == "__main__":
    {"loc": cmd_loc, "tests": cmd_tests}[sys.argv[1] if len(sys.argv) > 1 else "loc"]()
