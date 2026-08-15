#!/usr/bin/env python3
"""Regenerate the README star chart from the GitHub API.

Usage: python3 scripts/star-chart.py
Needs the `gh` CLI authenticated. Writes docs/star-history.svg + docs/star-history-dark.svg.
"""
import subprocess, sys, math
from datetime import datetime, timedelta, timezone
from pathlib import Path

REPO = "ddalcu/mlx-serve"
OUT = Path(__file__).resolve().parent.parent / "docs"

W, H = 800, 340
ML, MR, MT, MB = 56, 24, 36, 40  # margins


def fetch_starred_at():
    out = subprocess.run(
        ["gh", "api", "-H", "Accept: application/vnd.github.star+json",
         "--paginate", f"repos/{REPO}/stargazers?per_page=100",
         "--jq", ".[].starred_at"],
        capture_output=True, text=True, check=True).stdout
    return sorted(datetime.fromisoformat(l.replace("Z", "+00:00"))
                  for l in out.splitlines() if l.strip())


def daily_cumulative(times):
    now = datetime.now(timezone.utc)
    day0 = times[0].date()
    days = (now.date() - day0).days + 1
    pts, i = [], 0
    for d in range(days):
        cutoff = datetime.combine(day0 + timedelta(days=d + 1), datetime.min.time(), timezone.utc)
        while i < len(times) and times[i] < cutoff:
            i += 1
        pts.append((datetime.combine(day0 + timedelta(days=d), datetime.min.time(), timezone.utc), i))
    pts[-1] = (now, len(times))
    return pts


def nice_step(span, target_ticks):
    raw = span / target_ticks
    mag = 10 ** math.floor(math.log10(raw))
    for m in (1, 2, 2.5, 5, 10):
        if raw <= m * mag:
            return m * mag
    return 10 * mag


def month_ticks(t0, t1):
    ticks, y, m = [], t0.year, t0.month
    while True:
        m += 1
        if m > 12:
            m, y = 1, y + 1
        t = datetime(y, m, 1, tzinfo=timezone.utc)
        if t > t1:
            return ticks
        ticks.append(t)


def render(pts, theme):
    dark = theme == "dark"
    c_text = "#8b949e" if dark else "#57606a"
    c_title = "#e6edf3" if dark else "#24292f"
    c_grid = "#30363d" if dark else "#d8dee4"
    c_line = "#f7a41d" if dark else "#e3a008"
    t0, t1 = pts[0][0], pts[-1][0]
    total = pts[-1][1]
    ymax = math.ceil(total * 1.12 / 50) * 50
    step = int(nice_step(ymax, 5))

    def X(t):
        return ML + (t - t0) / (t1 - t0) * (W - ML - MR)

    def Y(v):
        return H - MB - v / ymax * (H - MT - MB)

    s = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" font-family="-apple-system,\'Segoe UI\',Helvetica,Arial,sans-serif">']
    s.append(f'<defs><linearGradient id="g" x1="0" y1="0" x2="0" y2="1">'
             f'<stop offset="0" stop-color="{c_line}" stop-opacity="0.28"/>'
             f'<stop offset="1" stop-color="{c_line}" stop-opacity="0"/></linearGradient></defs>')
    s.append(f'<text x="{ML}" y="22" font-size="14" font-weight="600" fill="{c_title}">GitHub stars</text>')
    s.append(f'<text x="{W - MR}" y="22" font-size="12" text-anchor="end" fill="{c_text}">{REPO} · {t1:%b %-d, %Y}</text>')
    for v in range(0, ymax + 1, step):
        y = Y(v)
        s.append(f'<line x1="{ML}" y1="{y:.1f}" x2="{W - MR}" y2="{y:.1f}" stroke="{c_grid}" stroke-width="1"/>')
        s.append(f'<text x="{ML - 8}" y="{y + 4:.1f}" font-size="12" text-anchor="end" fill="{c_text}">{v}</text>')
    for t in month_ticks(t0, t1):
        s.append(f'<text x="{X(t):.1f}" y="{H - MB + 20}" font-size="12" text-anchor="middle" fill="{c_text}">{t:%b}</text>')
    line = " ".join(f'{X(t):.1f},{Y(v):.1f}' for t, v in pts)
    s.append(f'<polygon points="{ML},{Y(0):.1f} {line} {X(t1):.1f},{Y(0):.1f}" fill="url(#g)"/>')
    s.append(f'<polyline points="{line}" fill="none" stroke="{c_line}" stroke-width="2.5" '
             f'stroke-linejoin="round" stroke-linecap="round"/>')
    ex, ey = X(t1), Y(total)
    s.append(f'<circle cx="{ex:.1f}" cy="{ey:.1f}" r="4" fill="{c_line}"/>')
    s.append(f'<text x="{ex - 8:.1f}" y="{ey - 10:.1f}" font-size="13" font-weight="600" text-anchor="end" fill="{c_title}">{total} ★</text>')
    s.append('</svg>')
    return "\n".join(s)


def main():
    times = fetch_starred_at()
    if not times:
        sys.exit("no stargazers returned")
    pts = daily_cumulative(times)
    (OUT / "star-history.svg").write_text(render(pts, "light"))
    (OUT / "star-history-dark.svg").write_text(render(pts, "dark"))
    print(f"{pts[-1][1]} stars -> docs/star-history{{,-dark}}.svg")


if __name__ == "__main__":
    main()
