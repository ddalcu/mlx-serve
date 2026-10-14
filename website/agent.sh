#!/bin/sh
# A coding agent for your mlx-serve: pick one (the ones you have are marked), it is installed with its
# own installer when missing, then the server's own setup script points it at the server, adds the
# mlx-serve skill (apps, games, films made on that server) and starts it.
#
#   curl -fsSL https://mlxserve.com/agent.sh | sh                          asks for the server and the agent
#   curl -fsSL https://mlxserve.com/agent.sh | sh -s -- 192.168.1.20 pi    or takes them (MLX_SERVE_URL, MLX_SERVE_AGENT)
#
# Run it again to come back: Enter keeps the server and the agent you chose last, and edits you made
# to the skill or to the agent's settings are kept. The config and the skill come from the server
# (the script the web console's Code Launcher serves), so they always match the mlx-serve you run.

CONF=$HOME/.mlx-serve/agent.conf

# id (the server's /launch name) | name | command | its installer | the shell that runs it.
# An agent without an installer here is offered once it is installed.
AGENTS='pi|pi|pi|https://pi.dev/install.sh|sh
claude|Claude Code|claude|https://claude.ai/install.sh|bash
omp|oh-my-pi|omp|https://omp.sh/install|sh
opencode2|OpenCode 2|opencode2||
grok|Grok|grok||'

die() {
  printf 'error: %s\n' "$*" >&2
  exit 1
}

# 192.168.1.20 -> http://192.168.1.20:11234; host:port and http(s):// URLs are kept.
normalize() {
  case $1 in
    http://* | https://*) u=$1 ;;
    \[*\]) u=http://$1:11234 ;;
    *:*) u=http://$1 ;;
    *) u=http://$1:11234 ;;
  esac
  printf '%s' "${u%/}"
}

# OpenCode 2 may answer to `opencode`: the same version probe the server's setup script runs.
installed() {
  command -v "$2" >/dev/null 2>&1 && return 0
  [ "$1" = opencode2 ] || return 1
  case $(opencode --version 2>/dev/null) in
    [2-9].* | v[2-9].* | [1-9][0-9].* | v[1-9][0-9].*) return 0 ;;
  esac
  return 1
}

# Installers put commands in user bin dirs that this shell may not have on PATH yet.
find_command() {
  command -v "$1" >/dev/null 2>&1 && return 0
  for d in "$HOME/.local/bin" "$HOME/.pi/agent/bin" "$HOME/.bun/bin" "$HOME/bin"; do
    if [ -x "$d/$1" ]; then
      PATH=$d:$PATH
      export PATH
      return 0
    fi
  done
  return 1
}

# ask <question> <default>: the answer from the terminal, or the default when there is none
ask() {
  answer=
  if (: </dev/tty) 2>/dev/null; then
    printf '%s [%s]: ' "$1" "$2" >/dev/tty
    read -r answer </dev/tty || answer=
  fi
  printf '%s' "${answer:-$2}"
}

main() {
  set -e
  last_url=$(sed -n 's/^url=//p' "$CONF" 2>/dev/null | head -n 1)
  last_agent=$(sed -n 's/^agent=//p' "$CONF" 2>/dev/null | head -n 1)

  url=${1:-${MLX_SERVE_URL:-}}
  [ -n "$url" ] || url=$(ask 'mlx-serve address' "${last_url:-localhost:11234}")
  url=$(normalize "$url")
  printf 'mlx-serve: %s\n' "$url" >&2
  curl -fsS -m 5 "$url/health" >/dev/null 2>&1 ||
    die "no mlx-serve answers at $url. Is MLX Core running there? From another machine, use that Mac's address and port 11234."

  # n|id|name|command|installer|shell|state per agent on offer; the default is the last one used,
  # else the first one installed, else the first on offer
  offered='' ids='' n=0 last_n='' installed_n=''
  while IFS='|' read -r id name cmd installer shell; do
    if installed "$id" "$cmd"; then
      state=installed
    elif [ -n "$installer" ]; then
      state="not installed: its installer runs first"
    else
      continue
    fi
    n=$((n + 1))
    offered="$offered$n|$id|$name|$cmd|$installer|$shell|$state
"
    ids="$ids $id"
    [ "$id" != "$last_agent" ] || last_n=$n
    [ "$state" != installed ] || [ -n "$installed_n" ] || installed_n=$n
  done <<EOF
$AGENTS
EOF

  want=${2:-${MLX_SERVE_AGENT:-}}
  if [ -z "$want" ]; then
    if (: </dev/tty) 2>/dev/null; then
      printf 'Coding agents:\n' >/dev/tty
      printf '%s' "$offered" | while IFS='|' read -r i _ name _ _ _ state; do
        printf '  %s) %-12s %s\n' "$i" "$name" "$state"
      done >/dev/tty
    fi
    want=$(ask 'Agent' "${last_n:-${installed_n:-1}}")
  fi
  pick=$(printf '%s' "$offered" | while IFS='|' read -r i id rest; do
    if [ "$want" = "$i" ] || [ "$want" = "$id" ]; then
      printf '%s|%s' "$id" "$rest"
      break
    fi
  done)
  [ -n "$pick" ] || die "'$want' is not on offer here; choose one of:$ids"
  IFS='|' read -r id name cmd installer shell state <<EOF
$pick
EOF

  if [ "$state" != installed ]; then
    printf 'installing %s with its own installer (%s)\n' "$name" "$installer" >&2
    tmp=$(mktemp)
    curl -fsSL "$installer" -o "$tmp" || die "could not download $name's installer from $installer"
    # stdout through cat: an installer that offers to start its agent asks only on a terminal, and
    # the agent starts below, configured for the server
    { "$shell" "$tmp" </dev/null || echo "$?" >"$tmp.failed"; } | cat
    if [ -f "$tmp.failed" ]; then
      rm -f "$tmp" "$tmp.failed"
      die "$name's installer failed; fix what it reported, then run this again."
    fi
    rm -f "$tmp"
    find_command "$cmd" || die "$name was installed but its \`$cmd\` command is not on PATH: open a new terminal and run this again."
  fi

  resp=$(curl -sS -m 30 -w '\n%{http_code}' "$url/launch?agent=$id&base=$url") || die "could not fetch $name's setup from $url"
  code=$(printf '%s' "$resp" | tail -n 1)
  body=$(printf '%s' "$resp" | sed '$d')
  if [ "$code" != 200 ]; then
    msg=$(printf '%s' "$body" | sed -n 's/.*"message": *"\([^"]*\)".*/\1/p' | head -n 1)
    case $msg in
      "Unknown endpoint") die "this mlx-serve is too old for this installer: update it (the app updates itself; or brew upgrade mlx-serve), then run this again." ;;
      "no such chat model"*) die "$msg: download a chat model in MLX Core (with LAN sharing on, share one), then run this again." ;;
      "") die "$url answered HTTP $code" ;;
      *) die "$msg" ;;
    esac
  fi
  mkdir -p "$HOME/.mlx-serve"
  printf 'url=%s\nagent=%s\n' "$url" "$id" >"$CONF"
  launch=$HOME/.mlx-serve/launch-$id.sh
  printf '%s\n' "$body" >"$launch"
  printf 'starting %s (run this command again to come back)\n' "$name" >&2
  # The agent takes the terminal itself as stdin. Agents built on Bun (Claude Code, oh-my-pi,
  # OpenCode) and Grok cannot watch a reopened /dev/tty on macOS, which is what a setup script
  # piped into sh falls back to.
  if [ ! -t 0 ] && [ -t 2 ]; then exec sh "$launch" 0<&2; fi
  if [ ! -t 0 ] && [ -t 1 ]; then exec sh "$launch" 0<&1; fi
  exec sh "$launch"
}

main "$@"
