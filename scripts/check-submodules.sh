#!/usr/bin/env bash
# Verify every git submodule is checked out at the commit the superproject pins
# (its recorded gitlink), and snap it back if it drifted.
#
# Why this exists: a submodule can silently drift off its pin — a stray
# `git checkout` inside lib/<sub>, an interrupted bump, a rebase that moved the
# gitlink but not the working tree. A drifted ENGINE submodule (lib/ds4,
# lib/mlx-src, lib/mlxc-src) either fails to compile with a cryptic missing-file
# error, or worse compiles against a different struct ABI and corrupts at
# runtime. This is the one guard that catches it before the build does.
#
# Modes:
#   (default)  check-only: warn on drift, exit 0. Never mutates — safe for the
#              inner `zig build` loop and for a bump you haven't committed yet.
#   --fix      auto-fix: snap a CLEAN drifted submodule back to its pin with
#              `git submodule update --init`. A submodule with uncommitted file
#              edits is never touched (warn + continue) so local work survives.
#
# No-op (exit 0) when this isn't a git checkout (source tarball / CI export) or
# git is unavailable — the vendored sources are already whatever they are.

set -u

FIX=0
[ "${1:-}" = "--fix" ] && FIX=1

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT" || exit 0

command -v git >/dev/null 2>&1 || exit 0
git rev-parse --is-inside-work-tree >/dev/null 2>&1 || exit 0

# Submodule paths straight from .gitmodules — the single source of truth.
paths="$(git config -f .gitmodules --get-regexp path 2>/dev/null | awk '{print $2}')"
[ -n "$paths" ] || exit 0

short() { printf '%.12s' "$1"; }

for p in $paths; do
    # The pin = the gitlink in the index (a staged, not-yet-committed bump
    # counts), which equals HEAD's when nothing is staged.
    pinned="$(git ls-files -s -- "$p" 2>/dev/null | awk '{print $2}')"
    [ -n "$pinned" ] || continue

    # Not initialized yet (fresh clone) — `git submodule update --init` fetches
    # it; in check-only mode just point at the fix.
    if [ ! -e "$p/.git" ]; then
        if [ "$FIX" = 1 ]; then
            echo "[submodules] $p not initialized — fetching pin $(short "$pinned")..."
            git submodule update --init -- "$p" \
                || echo "[submodules] WARN: could not init $p — run: git submodule update --init $p"
        else
            echo "[submodules] WARN: $p not initialized (pinned $(short "$pinned")) — run: git submodule update --init $p"
        fi
        continue
    fi

    have="$(git -C "$p" rev-parse HEAD 2>/dev/null)"
    [ "$have" = "$pinned" ] && continue   # in sync — stay quiet

    dirty=""
    [ -n "$(git -C "$p" status --porcelain 2>/dev/null)" ] && dirty=1

    if [ -n "$dirty" ]; then
        # Uncommitted file edits inside the submodule — never clobber them,
        # even in --fix mode. git submodule update would refuse anyway.
        echo "[submodules] WARN: $p is at $(short "$have") but pinned $(short "$pinned"), and has LOCAL EDITS — leaving it. Commit/stash inside $p, or: git submodule update --init $p"
    elif [ "$FIX" = 1 ]; then
        echo "[submodules] $p drifted $(short "$have") -> pin $(short "$pinned"); snapping back..."
        git submodule update --init -- "$p" \
            || echo "[submodules] WARN: could not update $p — fix manually: git submodule update --init $p"
    else
        echo "[submodules] WARN: $p is at $(short "$have") but pinned $(short "$pinned") — run: git submodule update --init $p  (./app/build.sh auto-fixes)"
    fi
done

exit 0
