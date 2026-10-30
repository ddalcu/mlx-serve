#!/usr/bin/env python3
"""The compiler's peak memory per compile unit (`-Dunits`), read from `zig build --summary all`; fails when any unit
peaks over the cap. Run on a clean cache so every unit actually compiles.

  python3 scripts/unit_memory.py [--cap-gb 3] [-- extra zig build args...]
"""
import re, subprocess, sys

UNIT = re.compile(r'compile obj (\S+).*?MaxRSS:(\d+(?:\.\d+)?)([KMG])')
SCALE = {'K': 1 / (1 << 20), 'M': 1 / 1024, 'G': 1.0}


def main(argv):
    cap = 3.0
    extra = []
    if '--cap-gb' in argv:
        i = argv.index('--cap-gb')
        cap = float(argv[i + 1])
        argv = argv[:i] + argv[i + 2:]
    if '--' in argv:
        extra = argv[argv.index('--') + 1:]
    cmd = ['./.zig-toolchain/zig', 'build', 'units', '-Dunits=true', '--summary', 'all'] + extra
    out = subprocess.run(cmd, capture_output=True, text=True)
    text = out.stdout + out.stderr
    peaks = {}
    for m in UNIT.finditer(text):
        gb = float(m.group(2)) * SCALE[m.group(3)]
        peaks[m.group(1)] = max(peaks.get(m.group(1), 0.0), gb)
    if out.returncode != 0 or not peaks:
        sys.stderr.write(text[-4000:])
        print('unit_memory: the units build failed or reported no unit')
        return 1
    over = {k: v for k, v in peaks.items() if v > cap}
    for k, v in sorted(peaks.items(), key=lambda kv: -kv[1]):
        print(f'{k:16} {v:5.2f} GB{"  OVER" if k in over else ""}')
    if over:
        print(f'unit_memory: {len(over)} unit(s) over {cap} GB')
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
