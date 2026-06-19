#!/usr/bin/env bash
# release.sh — trigger the GitHub "Release" workflow for the version documented
# at the top of CHANGELOG.md, but ONLY if that version matches the version the
# workflow will actually cut.
#
# The Release workflow computes its version as CalVer YY.M.N where N is one past
# the last GitHub release for the current YY.M (see .github/workflows/release.yml
# "Extract version"). If the newest CHANGELOG entry names a different version,
# you'd ship release notes that don't match the tag — or forget to write them.
# This guard refuses to dispatch unless the two agree.
#
# Usage:
#   ./release.sh            # verify match, confirm, then dispatch
#   ./release.sh -y         # skip the confirmation prompt
#   ./release.sh --dry-run  # print what it would do, never dispatch
#
# Env overrides: CHANGELOG, WORKFLOW (default release.yml), REF (default main).

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CHANGELOG="${CHANGELOG:-$REPO_ROOT/CHANGELOG.md}"
WORKFLOW="${WORKFLOW:-release.yml}"
REF="${REF:-main}"

usage() {
  sed -n '2,18p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
}

# Newest version documented in CHANGELOG.md, sans leading 'v' (e.g. "26.6.11").
# Empty if the file has no "## vX.Y.Z" heading. Single awk pass (no pipe, so a
# `set -o pipefail` caller can't trip on SIGPIPE).
changelog_top_version() {
  awk '
    /^##[[:space:]]*v[0-9]+\.[0-9]+\.[0-9]+/ {
      line = $0
      sub(/^##[[:space:]]*v/, "", line)
      match(line, /^[0-9]+\.[0-9]+\.[0-9]+/)
      print substr(line, RSTART, RLENGTH)
      exit
    }
  ' "$1"
}

# Version the Release workflow will produce on a fresh dispatch: CalVer YY.M.N
# with N = (max N among existing vYY.M.* GitHub releases) + 1. Mirrors the
# "Extract version" step in .github/workflows/release.yml verbatim.
computed_release_version() {
  local prefix last_n
  prefix="$(date -u +%y.%-m)"
  last_n="$(gh release list --limit 100 --json tagName \
    --jq "[.[] | .tagName | select(test(\"^v${prefix}\\\\.[0-9]+$\")) | sub(\"^v${prefix}\\\\.\"; \"\") | tonumber] | max // 0")"
  echo "${prefix}.$((last_n + 1))"
}

main() {
  set -euo pipefail

  local dry_run=0 assume_yes=0
  for arg in "$@"; do
    case "$arg" in
      --dry-run)   dry_run=1 ;;
      -y|--yes)    assume_yes=1 ;;
      -h|--help)   usage; return 0 ;;
      *) echo "release.sh: unknown argument '$arg'" >&2; usage >&2; return 2 ;;
    esac
  done

  local cl_version computed
  cl_version="$(changelog_top_version "$CHANGELOG")"
  if [ -z "$cl_version" ]; then
    echo "release.sh: no '## vX.Y.Z' entry found at the top of $CHANGELOG" >&2
    return 1
  fi
  computed="$(computed_release_version)"

  echo "CHANGELOG top entry : v$cl_version"
  echo "Workflow will cut   : v$computed   (CalVer: last release + 1)"

  if [ "$cl_version" != "$computed" ]; then
    {
      echo "release.sh: version mismatch — refusing to trigger a release."
      echo "  CHANGELOG.md top entry : v$cl_version"
      echo "  workflow would release : v$computed"
      echo "Make the CHANGELOG heading match the next release version (or cut the"
      echo "pending release first), then re-run."
    } >&2
    return 1
  fi

  echo "✓ versions match (v$computed)"

  if [ "$dry_run" -eq 1 ]; then
    echo "[dry-run] would run: gh workflow run $WORKFLOW --ref $REF"
    return 0
  fi

  if [ "$assume_yes" -ne 1 ]; then
    read -r -p "Trigger the Release workflow for v$computed on '$REF'? [y/N] " reply
    case "$reply" in
      [yY] | [yY][eE][sS]) ;;
      *) echo "aborted."; return 0 ;;
    esac
  fi

  gh workflow run "$WORKFLOW" --ref "$REF"
  echo "✓ dispatched v$computed on '$REF'."
  echo "  Watch:  gh run watch \$(gh run list --workflow=$WORKFLOW --limit 1 --json databaseId --jq '.[0].databaseId')"
}

# Only run main when executed directly — sourcing (e.g. tests) just loads the
# functions.
if [ "${BASH_SOURCE[0]}" = "${0}" ]; then
  main "$@"
fi
