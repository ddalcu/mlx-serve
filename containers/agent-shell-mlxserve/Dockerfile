# syntax=docker/dockerfile:1
#
# Lightweight agentic-dev shell: Debian-slim (glibc) + Node.js/npm + Python3/pip + apt,
# plus a lean set of everyday agent CLI tools (process / search / network / files / editing).
# Design notes:
#   * glibc base  -> prebuilt Python wheels & npm native modules "just work" (no musl pain).
#   * Node copied from the official image, then `strip`ped in a throwaway stage (~20MB saved)
#     so the final image never contains binutils.
#   * apt is kept so an agent can install more tools on demand (`apt-get update && apt-get install ...`).
#   * Baked-in tooling (all via the single apt layer, --no-install-recommends):
#       - git, curl, ca-certificates  : fetch/clone (universal, cost RAM to install at runtime)
#       - procps                      : ps, pgrep, pkill, top, free, uptime, vmstat, watch
#       - psmisc                      : killall, fuser, pstree
#       - ripgrep                     : rg (fast recursive search agents lean on constantly)
#       - lsof                        : open files / what's listening on a port (sandbox port maps)
#       - iproute2                    : ss, ip (modern net tooling; the sandbox init uses `ip link`)
#       - less                        : pager expected by git log / man
#       - nano                        : small everyday editor (vim is heavier; skipped)
#       - xz-utils                    : tar.xz is the default compression for half the
#                                       ecosystem's release tarballs (node, hermes's
#                                       managed Node — issue #150); tar shells out to xz
#       - dropbear-bin                : sshd for the Sandbox terminal's agent CLI sessions
#                                       (mlx-serve's /.vz-init starts `dropbear -R -s -p 22`;
#                                       -bin only = no init scripts, ~500 KB). The app PROBES
#                                       for it and reports a stale cached image when absent —
#                                       never remove it without changing that transport.
#   * "Practical minimum" cleanup: --no-install-recommends, drop apt lists / docs / man / locales.

ARG DEBIAN_TAG=trixie-slim
ARG NODE_TAG=24-trixie-slim

##############################################################################
# Stage 1: source of a known-good Node.js + npm (same Debian base = ABI match)
##############################################################################
FROM node:${NODE_TAG} AS nodesrc
# npm: track latest. node:24 bundles npm 11.16.0, inside the 11.10-11.16 band
# hermes's installer rejects (those npms ignore min-release-age-exclude; issue
# #150) — with a current npm the baked Node passes hermes's toolchain check and
# it skips downloading its own managed Node. Upgraded HERE, not in the final
# image, so the node_modules COPY below carries no dead old-npm lower layer.
RUN npm install -g --no-fund --no-audit npm@latest

##############################################################################
# Stage 2: strip the Node binary (binutils lives only here, never shipped)
##############################################################################
FROM debian:${DEBIAN_TAG} AS strip
RUN apt-get update \
 && apt-get install -y --no-install-recommends binutils \
 && rm -rf /var/lib/apt/lists/*
COPY --from=nodesrc /usr/local/bin/node /node
RUN strip --strip-unneeded /node

##############################################################################
# Stage 3: the final image
##############################################################################
FROM debian:${DEBIAN_TAG}

LABEL org.opencontainers.image.title="agent-shell-mlxserve" \
      org.opencontainers.image.description="mlx-serve Agent Sandbox guest: Node.js + Python3 + apt, pre-baked MCP servers, dropbear sshd for in-guest agent CLI sessions"

ENV LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    # Container is disposable => let `pip install` write to the system env directly.
    PIP_BREAK_SYSTEM_PACKAGES=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# --- Node.js (stripped binary + npm/npx/corepack from the official image) ---
COPY --from=strip   /node                          /usr/local/bin/node
COPY --from=nodesrc /usr/local/lib/node_modules    /usr/local/lib/node_modules
RUN ln -s /usr/local/lib/node_modules/npm/bin/npm-cli.js      /usr/local/bin/npm  \
 && ln -s /usr/local/lib/node_modules/npm/bin/npx-cli.js      /usr/local/bin/npx  \
 && ln -s /usr/local/lib/node_modules/corepack/dist/corepack.js /usr/local/bin/corepack

# --- Python3 + pip + a lean set of agent CLI tools via apt, then aggressive
#     (apt-preserving) cleanup ---
# These are preinstalled on purpose: under the mlx-serve sandbox the rootfs runs
# demand-paged from the base image (guest RAM is workload headroom), so a runtime
# `apt install` costs *guest RAM* and is slow — bake the universal tools in.
# process/search/network/file tooling agents reach for constantly (ps/pgrep/pkill,
# killall/pstree, rg, lsof, ss/ip, less, nano) is included so common commands don't
# silently fail (e.g. the sandbox init's `ip link` needs iproute2).
# The EXTERNALLY-MANAGED removal is what actually makes `pip install` work: Debian's
# PEP 668 marker otherwise blocks it, and the usual PIP_BREAK_SYSTEM_PACKAGES env is
# DROPPED by the sandbox's guest init, so we clear the marker in the filesystem.
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
        python3 \
        python3-pip \
        python3-venv \
        git \
        curl \
        ca-certificates \
        libatomic1 \
        procps \
        psmisc \
        ripgrep \
        lsof \
        iproute2 \
        less \
        nano \
        xz-utils \
        dropbear-bin \
 && rm -f /usr/lib/python3.*/EXTERNALLY-MANAGED \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/* \
           /usr/share/doc/* \
           /usr/share/man/* \
           /usr/share/locale/* \
           /root/.cache

# --- Pre-baked MCP servers (Mac App Store build) ---
# The mlx-serve Agent Sandbox runs stdio MCP servers INSIDE this guest. On the
# Developer ID build they're fetched with `npx` on first use; the Mac App Store
# build cannot download code that adds agent tools (App Review guideline 2.5.2),
# so the common ones are baked in here, pinned by exact version. `npx <name>`
# then resolves the global install instead of hitting the npm registry.
#
# This is the same move iSH made: its first approved build shipped with no live
# package fetch. `apt`/`npm`/`pip` for the USER's own project stay live — that
# doesn't change the app's functionality; a downloaded agent tool does.
ARG MCP_FS_VERSION=2026.7.4
ARG MCP_GITHUB_VERSION=2025.4.8
RUN npm install -g --no-fund --no-audit \
        @modelcontextprotocol/server-filesystem@${MCP_FS_VERSION} \
        @modelcontextprotocol/server-github@${MCP_GITHUB_VERSION} \
 && npm cache clean --force \
 && rm -rf /root/.npm

# --- non-root user (opt-in: kept as root by default so `apt`/global installs work) ---
RUN useradd --create-home --shell /bin/bash agent
WORKDIR /workspace

# Fail the build early if stripping broke Node or a runtime/tool is missing.
# Tools with clean `--version` exits are checked directly; those with quirky exit
# codes / stderr-only output (killall, lsof) are checked via `command -v`, and
# `ss` falls back to `ip -V` so a version-string oddity can't fail the build.
RUN node --version && npm --version && python3 --version && python3 -m pip --version \
 && git --version && curl --version \
 && ps --version && rg --version && less --version && nano --version \
 && command -v pgrep && command -v pkill \
 && command -v killall && command -v pstree \
 && command -v lsof \
 && { ss --version || ip -V; } \
 && xz --version \
 && command -v dropbear \
 && test ! -e /usr/lib/python3.*/EXTERNALLY-MANAGED  # pip must be unblocked

CMD ["/bin/bash"]
