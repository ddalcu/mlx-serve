#!/usr/bin/env python3
"""docs/plugin-abi.md from src/abi/*.zig: every public type, field and function with its doc comment.

  python3 scripts/plugin_abi_doc.py            # write the doc
  python3 scripts/plugin_abi_doc.py --check    # exit 1 when the doc is not what the sources produce
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "plugin-abi.md"
# Reading order: the tables a plugin exports first, the shared vocabulary they use last.
FILES = ["plugin.zig", "media.zig", "quant.zig", "host.zig", "engine.zig", "common.zig", "dylib.zig"]
TYPE = re.compile(r"^pub const (\w+) = (extern struct|extern union|packed struct(?:\(\w+\))?|enum(?:\(\w+\))?|struct) \{")
ALIAS = re.compile(r"^pub const (\w+)(?::\s*([^=]+))?\s*=\s*(.+);$")
FIELD = re.compile(r"^(\w+):\s*(.+?)(?:\s*=\s*(.+?))?,\s*(?://\s*(.*))?$")
MEMBER = re.compile(r"^(\w+)(?:\s*=\s*([^,]+))?,\s*(?://\s*(.*))?$")
FN = re.compile(r"^pub fn (\w+)(\(.*)\{$")


def cell(s):
    return (s or "").replace("|", "\\|").strip()


def depth_change(line):
    code = re.sub(r'"(?:\\.|[^"\\])*"', '""', line.split("//")[0])
    return code.count("{") - code.count("}")


def parse(path):
    """[(kind, name, detail, doc, rows)] in source order; rows = [(name, type, default, doc)]."""
    lines = path.read_text().splitlines()
    header = [l[3:].strip() for l in lines if l.startswith("//!")]
    items, doc, i = [], [], 0
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith("///"):
            doc.append(line[3:].strip())
            i += 1
            continue
        m = TYPE.match(line)
        if m:
            name, kind = m.group(1), m.group(2)
            rows, depth, fdoc, j = [], depth_change(line), [], i + 1
            if depth == 0:  # one line: `enum(u8) { a, b = 2 };` or `extern struct { a: u32 = 0, b: f32 };`
                inline = line[line.index("{") + 1:line.rindex("}")]
                for v in (x.strip() for x in re.split(r",(?![^(]*\))", inline) if x.strip()):
                    if kind.startswith("enum"):
                        rows.append((v.split("=")[0].strip(), "", v.split("=")[1].strip() if "=" in v else None, ""))
                    elif (f := FIELD.match(v + ",")):
                        rows.append((f.group(1), f.group(2), f.group(3), ""))
            while j < len(lines) and depth > 0:
                inner = lines[j].strip()
                if depth == 1 and inner.startswith("///"):
                    fdoc.append(inner[3:].strip())
                elif depth == 1 and inner and not inner.startswith(("//", "pub fn", "fn ", "const ", "pub const", "comptime", "}")):
                    f = FIELD.match(inner) if not kind.startswith("enum") else None
                    e = MEMBER.match(inner) if kind.startswith("enum") else None
                    if f:
                        rows.append((f.group(1), f.group(2), f.group(3), " ".join(fdoc + ([f.group(4)] if f.group(4) else []))))
                    elif e:
                        rows.append((e.group(1), "", e.group(2), " ".join(fdoc + ([e.group(3)] if e.group(3) else []))))
                    fdoc = []
                elif depth == 1 and not inner.startswith("///"):
                    fdoc = []
                depth += depth_change(lines[j])
                j += 1
            items.append(("type", name, kind, " ".join(doc), rows))
            doc, i = [], j
            continue
        m = FN.match(line)
        if m:
            sig = m.group(2).rstrip()
            items.append(("fn", m.group(1), sig, " ".join(doc), []))
            depth, j = depth_change(line), i + 1
            while j < len(lines) and depth > 0:
                depth += depth_change(lines[j])
                j += 1
            doc, i = [], j
            continue
        m = ALIAS.match(line)
        if m:
            items.append(("const", m.group(1), (m.group(2) or "").strip() + " = " + m.group(3).strip() if m.group(2) else m.group(3).strip(), " ".join(doc), []))
        doc = [] if not line.startswith("///") else doc
        i += 1
    return " ".join(header), items


def render():
    out = [
        "# Plugin ABI reference",
        "",
        "Generated from `src/abi/*.zig` by `scripts/plugin_abi_doc.py`; do not edit by hand (`tests/test_plugin_abi_doc.sh`",
        "fails when it is stale). The same sources ship in the plugin kit (`src/abi/`), and `include/mlx_serve_plugin.h`",
        "mirrors them for C, C++ and Rust. Guides: [plugins.md](plugins.md), [model](plugin-model.md),",
        "[media](plugin-media.md), [quant](plugin-quant.md).",
        "",
        "Every struct opens with `size`: one passed by pointer grows at its end (an older plugin's shorter struct still",
        "reads), one held by value or as an array element is frozen per `version`. A field added later defaults to the",
        "behaviour before it, so a plugin that never sets it is unchanged.",
        "",
    ]
    for fname in FILES:
        header, items = parse(ROOT / "src" / "abi" / fname)
        out += [f"## `{fname}`", ""]
        if header:
            out += [header, ""]
        consts = [it for it in items if it[0] == "const"]
        if consts:
            out += ["| Name | Value | |", "|---|---|---|"]
            out += [f"| `{n}` | `{cell(d)}` | {cell(doc)} |" for _, n, d, doc, _ in consts]
            out.append("")
        for kind, name, detail, doc, rows in items:
            if kind == "type":
                out += [f"### `{name}` ({detail})", ""]
                if doc:
                    out += [doc, ""]
                if rows:
                    enum = detail.startswith("enum")
                    out += (["| Member | Value | |", "|---|---|---|"] if enum else ["| Field | Type | Default | |", "|---|---|---|---|"])
                    for fn, ft, fd, fdoc in rows:
                        out.append(f"| `{fn}` | `{cell(fd)}` | {cell(fdoc)} |" if enum and fd else
                                   f"| `{fn}` | | {cell(fdoc)} |" if enum else
                                   f"| `{fn}` | `{cell(ft)}` | {'`' + cell(fd) + '`' if fd else ''} | {cell(fdoc)} |")
                    out.append("")
            elif kind == "fn":
                out += [f"### `{name}{detail}`", ""]
                if doc:
                    out += [doc, ""]
    return "\n".join(out).rstrip() + "\n"


def main():
    text = render()
    if "--check" in sys.argv[1:]:
        if not OUT.exists() or OUT.read_text() != text:
            print(f"{OUT.relative_to(ROOT)} is stale: run python3 scripts/plugin_abi_doc.py", file=sys.stderr)
            return 1
        return 0
    OUT.write_text(text)
    print(OUT.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    sys.exit(main())
