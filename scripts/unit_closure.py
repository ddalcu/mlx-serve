#!/usr/bin/env python3
"""Each plugin's file closure under -Dunits. Zig keys a compilation on every file its imports reach, test-block
imports included, so a plugin unit recompiles whenever any file in that closure changes. A plugin's closure may hold
only `src/sdk/`, `src/mlx/`, `src/abi/`, `src/kernels/`, `src/env.zig`, its own folder and the plugins it remixes
(another plugin's folder, whose imports are held to the same bar); this prints every edge that leaves it
(file:line -> target) and fails when any plugin has one.

  python3 scripts/unit_closure.py [plugin ...]      (no arguments: every registered plugin)
  python3 scripts/unit_closure.py --summary         (one line per plugin)
  python3 scripts/unit_closure.py --except a,b      (skips plugins a and b; an exit from inside their folders is only noted)
"""
import os, re, sys

IMP = re.compile(r'@import\("([^"]+\.zig)"\)')
ALLOWED = ('src/sdk/', 'src/mlx/', 'src/abi/', 'src/kernels/', 'src/plugins/')
_cache = {}


def imports(p):
    if p not in _cache:
        try:
            s = open(p, errors='ignore').read()
        except FileNotFoundError:
            s = ''
        _cache[p] = [(os.path.normpath(os.path.join(os.path.dirname(p), m.group(1))), s.count('\n', 0, m.start()) + 1)
                     for m in IMP.finditer(s)]
    return _cache[p]


def plugin_of(path):
    parts = path.split('/')
    return parts[2] if len(parts) > 3 and parts[1] == 'plugins' else None


def exits(plugin, skipped=()):
    start = f'src/plugins/{plugin}/plugin.zig'
    seen, stack, out, notes = {start}, [start], [], []
    while stack:
        p = stack.pop()
        for t, line in imports(p):
            if not t.startswith('src/') or not os.path.exists(t):
                continue
            # The registry imports every plugin: reaching it puts them all in the closure.
            if not (t.startswith(ALLOWED) or t == 'src/env.zig') or t == 'src/plugins/registry.zig':
                (notes if plugin_of(p) in skipped else out).append((p, line, t))
                continue
            if t not in seen:
                seen.add(t)
                stack.append(t)
    return seen, out, notes


def plugins():
    d = 'src/plugins'
    return sorted(n for n in os.listdir(d) if not n.startswith('_') and os.path.exists(f'{d}/{n}/plugin.zig')
                  and not os.path.exists(f'{d}/{n}/build.zig'))


def main(argv):
    args, skipped = argv[1:], set()
    if '--except' in args:
        i = args.index('--except')
        skipped = set(args[i + 1].split(','))
        del args[i:i + 2]
    summary = '--summary' in args
    names = [n for n in ([a for a in args if not a.startswith('--')] or plugins()) if n not in skipped]
    bad = 0
    for n in names:
        seen, out, notes = exits(n, skipped)
        bad += bool(out)
        if summary:
            print(f'{len(out):4} exits, {len(seen):4} files  {n}')
            continue
        for p, line, t in out:
            print(f'{p}:{line}: {n} reaches {t}')
        for p, line, t in notes:
            print(f'{p}:{line}: note: {n} reaches skipped {t}')
    if bad:
        print(f'\nunit_closure: {bad} plugin(s) reach outside sdk/, mlx/, abi/, kernels/, env.zig and the plugin folders.',
              file=sys.stderr)
    return 1 if bad else 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
