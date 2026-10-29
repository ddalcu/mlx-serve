#!/usr/bin/env python3
"""Machine code (__text bytes) per source file and per generic function family, from a Mach-O binary's symbols.

  python3 scripts/symsize.py <binary> [repo-root] [--json out.json]

A function's size is the gap to the next symbol, so inlined code counts toward its caller. A family is a function
name with Zig's instance suffixes (`__anon_123`, `__func_45`) and generic arguments `(…)` removed; a family with more
than FLAG instances is flagged (plan L12).
"""
import collections, json, os, re, subprocess, sys

FLAG = 50
TOPS = ["src"]
SUFFIX = re.compile(r"__(func|anon|struct|union|enum|opaque)_\d+")


def text_section(binp):
    out = subprocess.check_output(["otool", "-l", binp], text=True).splitlines()
    i = next(i for i, l in enumerate(out) if "sectname __text" in l)
    return int(out[i + 2].split()[1], 16), int(out[i + 3].split()[1], 16)


def functions(binp):
    addr, size = text_section(binp)
    end = addr + size
    syms = []
    for l in subprocess.check_output(["nm", "-n", binp], text=True).splitlines():
        p = l.split(None, 2)
        if len(p) == 3 and p[1] in "tT" and addr <= int(p[0], 16) < end:
            syms.append((int(p[0], 16), p[2]))
    return size, [(n, (syms[i + 1][0] if i + 1 < len(syms) else end) - a) for i, (a, n) in enumerate(syms)]


def strip_args(s):
    """`a.F(x,(y)).g` -> `a.F(…).g`."""
    out, depth = [], 0
    for c in s:
        if c == "(":
            if depth == 0:
                out.append("(…)")
            depth += 1
        elif c == ")":
            depth = max(depth - 1, 0)
        elif depth == 0:
            out.append(c)
    return "".join(out)


def family(n):
    n = n[1:] if n.startswith("_") else n
    if n.startswith(("_Z", "-[", "+[")):
        return None  # C++ / Objective-C
    return strip_args(SUFFIX.sub("", n))


def module_map(repo):
    ours = {}
    for top in TOPS:
        base = os.path.join(repo, top)
        for root, _, files in os.walk(base):
            for f in files:
                if f.endswith(".zig"):
                    rel = os.path.relpath(os.path.join(root, f), base)[:-4]
                    ours.setdefault(rel.replace("/", "."), top + "/" + rel + ".zig")
    return ours


def source_of(n, ours):
    n = n[1:] if n.startswith("_") else n
    if n.startswith(("_Z", "-[", "+[")):
        return "C++ / Objective-C (jinja, xatlas, libc++, llama shim, Metal glue)"
    base = SUFFIX.sub("", n)
    if base.startswith(("Io.Writer.print", "fmt.")):
        return "std: fmt / Writer.print (monomorphized)"
    if "." not in base:
        return "C (ds4, stb, msv shims, compiler_rt)"
    parts = strip_args(base).split(".")
    for k in range(len(parts), 0, -1):
        if ".".join(parts[:k]) in ours:
            return ours[".".join(parts[:k])]
    return "std: " + parts[0]


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    jout = sys.argv[sys.argv.index("--json") + 1] if "--json" in sys.argv else None
    if jout in args:
        args.remove(jout)
    binp, repo = args[0], (args[1] if len(args) > 1 else os.path.join(os.path.dirname(__file__), ".."))
    text, fns = functions(binp)
    ours = module_map(repo)
    files, fams = collections.defaultdict(lambda: [0, 0]), collections.defaultdict(lambda: [0, 0])
    for n, s in fns:
        f = files[source_of(n, ours)]; f[0] += s; f[1] += 1
        k = family(n)
        if k is not None:
            g = fams[k]; g[0] += s; g[1] += 1
    mb = lambda b: f"{b / 1e6:7.3f} MB {100 * b / text:5.1f}%"
    print(f"__text {text / 1e6:.2f} MB, {len(fns)} functions")

    wp = fams.get("Io.Writer.print", [0, 0])
    fmt = [sum(v[i] for k, v in fams.items() if k.startswith(("Io.Writer.print", "fmt."))) for i in (0, 1)]
    log = [sum(v[i] for k, v in fams.items() if k.startswith("log.")) for i in (0, 1)]
    print(f"\n== Writer.print: {wp[1]} instances, {mb(wp[0])} of __text")
    print(f"== Io.Writer.print* + fmt.*: {fmt[1]} functions, {mb(fmt[0])}")
    print(f"== log.* (src/sdk/util/log.zig and std.log): {log[1]} functions, {mb(log[0])}")
    for k, (s, c) in sorted(fams.items(), key=lambda kv: -kv[1][1]):
        if k.startswith("log.") and c > 1:
            print(f"   {c:6d}  {mb(s)}  {k}")

    multi = sorted(((k, s, c) for k, (s, c) in fams.items() if c > 1), key=lambda r: -r[1])
    print(f"\n== generic families with more than {FLAG} instances (flagged)")
    for k, s, c in sorted(multi, key=lambda r: -r[2]):
        if c > FLAG:
            print(f"{c:6d}  {mb(s)}  {k}")
    print("\n== largest generic families by bytes (top 40)")
    for k, s, c in multi[:40]:
        print(f"{c:6d}  {mb(s)}  {k}")
    print("\n== per source file: bytes, share, functions")
    for k, (s, c) in sorted(files.items(), key=lambda kv: -kv[1][0]):
        print(f"{mb(s)}  {c:6d}  {k}")
    if jout:
        json.dump({"text_bytes": text, "files": files, "families": fams}, open(jout, "w"), indent=0)


if __name__ == "__main__":
    main()
