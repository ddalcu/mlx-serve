#!/usr/bin/env python3
"""Pack the -Dunits plugin groups so plugins that compile the same code share an object.

Build one object per plugin first (Debug, macOS, nothing is linked):
  zig build units -Dunits=true -Dstrip -Dunit-groups=99 --prefix /tmp/units-solo
then run:  python3 scripts/unit_groups.py /tmp/units-solo/units
and paste the printed table over `packed_groups` in build/plugins.zig.

A group's cost is the __text bytes of the union of its plugins' functions (identical code in two plugins compiles
once). For 8, 4 and 3 requested groups it packs the largest plugin first into the group where it adds the least,
capped at the largest round-robin group's cost, so the longest compile never grows; the packing uses as few groups as
fit (8 requested fills 4 at this writing).
"""
import os, re, subprocess, sys
from collections import Counter

LINE = re.compile(r"^([0-9a-f]+) \(__TEXT,__text\) (?:non-)?external (?:\[[^\]]*\] )?(\S.*)$")
ANON = re.compile(r"__(func|struct|enum|union|anon|opaque)_\d+")


def functions(obj):
    rows = []
    for ln in subprocess.run(["xcrun", "nm", "-nm", obj], capture_output=True, text=True, check=True).stdout.splitlines():
        m = LINE.match(ln)
        if m:
            rows.append((int(m.group(1), 16), m.group(2)))
    out = Counter()
    for (a, n), nxt in zip(rows, rows[1:] + [(rows[-1][0], "")]):
        name = n[2:] if n.startswith("l_") else n.lstrip("_")
        if not name.startswith("tmp"):
            out[(ANON.sub(r"__\1", name), nxt[0] - a)] += 1
    return out


def cost(sets, group):
    seen = Counter()
    for p in group:
        seen |= sets[p]
    return sum(k[1] * c for k, c in seen.items())


def pack(sets, n):
    names = sorted(sets)
    cap = max(cost(sets, names[i::n]) for i in range(n))
    groups, cur = [[] for _ in range(n)], [0] * n
    for p in sorted(names, key=lambda p: -cost(sets, [p])):
        c, g = min((c > cap, c - cur[g], c, g) for g, c in ((g, cost(sets, groups[g] + [p])) for g in range(n)))[2:]
        groups[g].append(p)
        cur[g] = c
    return [sorted(g) for g in groups if g]


def main():
    units = sys.argv[1]
    sets = {}
    for fn in os.listdir(units):
        g = fn[:-2]
        if not g.isdigit():
            continue
        root = open(f"src/unit_{g}.zig").read()
        plugins = re.findall(r"plugins/([a-z0-9_]+)/plugin\.zig", root)
        if len(plugins) != 1:
            sys.exit(f"unit {g} holds {len(plugins)} plugins: build with -Dunit-groups at least the plugin count")
        sets[plugins[0]] = functions(os.path.join(units, fn))
    print("const packed_groups = [_][]const []const []const u8{")
    for n in (8, 4, 3):
        print("    &.{")
        for g in pack(sets, n):
            print("        &.{ " + ", ".join(f'"{p}"' for p in g) + " },")
        print("    },")
    print("};")


if __name__ == "__main__":
    main()
