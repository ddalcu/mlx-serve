#!/usr/bin/env python3
"""The units' duplicated state (-Dunits). Every unit object compiles its own copy of the globals its code reaches.
The host, the MLX engine and the module archs' unit each own their state; a plugin unit owns none, and a global two
of them hold must be `Shared` (src/sdk/util/shared.zig). This lists the internal data symbols of each object, threadlocals included (a
Debug build keeps their names) and fails on one in a plugin unit or in two owners that scripts/unit_globals_ok.txt
does not allow, by pattern (`~regex`) or by name.

  python3 scripts/unit_globals.py <unit .o file or dir>...      (zig build units-check -Dunits=true runs this)
"""
import os, re, subprocess, sys

def allowed_set(path):
    pats, names = [], set()
    for line in open(path):
        line = line.split('#', 1)[0].strip()
        if not line: continue
        if line.startswith('~'): pats.append(re.compile(line[1:]))
        else: names.add(line)
    return pats, names

def duplicated(nm_out, pats, names):
    """The internal data symbols of one object's `nm -m` listing that are neither Shared storage nor allowed."""
    for l in nm_out.splitlines():
        # A threadlocal is a `__thread_vars` descriptor under the variable's own name (its `$tlv$init` image beside it).
        if not re.search(r'\(__DATA,__(data|bss|common|thread_vars)\)', l) or 'non-external' not in l: continue
        sym = l.split('non-external ', 1)[1].strip().lstrip('_')
        # A unit resolves Shared state to the host's copy, so it never holds the storage itself.
        if not (sym.startswith(('sdk.util.shared.Shared(', 'sdk.util.shared.SharedThreadLocal(')) and sym.endswith(').storage')):
            if sym in names or any(p.search(sym) for p in pats): continue
        yield sym

OWNERS = ('host', 'engine-mlx', 'engine-modules')

def main(argv):
    objs = []
    for a in argv[1:]:
        objs += [os.path.join(a, f) for f in sorted(os.listdir(a)) if f.endswith('.o')] if os.path.isdir(a) else [a]
    pats, names = allowed_set(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'unit_globals_ok.txt'))
    held = {}
    for o in objs:
        out = subprocess.run(['nm', '-m', o], capture_output=True, text=True, check=True).stdout
        for sym in duplicated(out, pats, names): held.setdefault(sym, []).append(os.path.basename(o)[:-2])
    bad = {s: u for s, u in held.items() if len(u) > 1 or u[0] not in OWNERS}
    for sym, units in sorted(bad.items()):
        print(f'{sym}: a copy in {len(units)} unit(s) ({", ".join(units[:3])}{"…" if len(units) > 3 else ""})')
    if bad:
        print(f'\nunit_globals: {len(bad)} global(s) duplicated across units. Make each one '
              'Shared (src/sdk/util/shared.zig); list one that is safe to copy in scripts/unit_globals_ok.txt.', file=sys.stderr)
        return 1
    return 0

if __name__ == '__main__':
    sys.exit(main(sys.argv))
