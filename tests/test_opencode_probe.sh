#!/bin/bash
# Exercise production OpenCode detection with isolated login shells and a fixture HTTP server.
set -eu
: "${TMPDIR:?set TMPDIR to a workspace scratch directory}"
python3 - "${1:-./zig-out/bin/mlx-serve}" "$TMPDIR" <<'PY'
import http.server
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading

binary = Path(sys.argv[1]).resolve()
scratch = Path(sys.argv[2]).resolve()
scratch.mkdir(parents=True, exist_ok=True)
if not binary.is_file():
    raise SystemExit(f"FAIL: binary not found: {binary}")

class Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        bodies = {
            "/health": {"status": "ok"},
            "/v1/models": {"data": [{"id": "fixture-model", "loaded": True,
                "capabilities": ["chat"], "meta": {"context_length": 24576}}]},
            "/metrics.json": {},
        }
        body = json.dumps(bodies.get(self.path, {})).encode()
        self.send_response(200 if self.path in bodies else 404)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass

server = http.server.HTTPServer(("127.0.0.1", 0), Handler)
thread = threading.Thread(target=server.serve_forever, daemon=True)
thread.start()
base = f"http://127.0.0.1:{server.server_port}"

cases = [
    ("canonical-v1", "opencode", "1.18.34", 0, False, 1, "v1", None),
    ("canonical-v2", "opencode", "2.0.20", 0, False, 1, "v2", None),
    ("canonical-v3", "opencode", "3.0.0", 0, False, 1, "v2", None),
    ("canonical-v10", "opencode", "10.2.0", 0, False, 1, "v2", None),
    ("canonical-missing", "opencode", None, 0, False, 1, None, "not installed"),
    ("canonical-empty", "opencode", "", 0, False, 1, None, "could not determine"),
    ("canonical-127-empty", "opencode", "", 127, False, 1, None, "could not determine"),
    ("canonical-nonzero", "opencode", "2.0.20", 1, False, 1, None, "could not determine"),
    ("canonical-unparsed", "opencode", "dev", 0, False, 1, None, "could not determine"),
    ("canonical-marker-missing", "opencode", "2.0.20", 0, False, 1, None, "could not determine"),
    ("canonical-shell-failed", "opencode", "2.0.20", 0, False, 1, None, "could not determine"),
    ("alias-marker-missing", "opencode2", "2.0.20", 0, False, 2, None, "could not check the legacy"),
    ("alias-marker-invalid", "opencode2", "2.0.20", 0, False, 2, None, "could not check the legacy"),
    ("alias-shell-failed", "opencode2", "2.0.20", 0, False, 2, None, "could not check the legacy"),
    ("alias-v2", "opencode2", "2.0.20", 0, True, 1, "v2", None),
    ("alias-v3", "opencode2", "3.0.0", 0, False, 1, "v2", None),
    ("alias-v10", "opencode2", "10.2.0", 0, False, 1, "v2", None),
    ("alias-v1-fallback", "opencode2", "1.18.34", 0, True, 2, "legacy", None),
    ("alias-missing-fallback", "opencode2", None, 0, True, 2, "legacy", None),
    ("alias-failed-fallback", "opencode2", "", 127, True, 2, "legacy", None),
    ("alias-unparsed-fallback", "opencode2", "dev", 0, True, 2, "legacy", None),
    ("alias-v1-no-legacy", "opencode2", "1.18.34", 0, False, 2, None, "no OpenCode v2 binary"),
    ("alias-missing-no-legacy", "opencode2", None, 0, False, 2, None, "no OpenCode v2 binary"),
]
failures = []
try:
    for name, agent, version, rc, legacy, count, profile, error in cases:
        with tempfile.TemporaryDirectory(prefix=f"opencode-{name}-", dir=scratch) as tmp:
            root = Path(tmp)
            home = root / "home"
            bindir = root / "bin"
            home.mkdir()
            bindir.mkdir()
            calls = root / "shell-calls"
            version_calls = root / "version-calls"
            (home / ".zprofile").write_text(
                'export PATH="$OC_FIXTURE_BIN:/usr/bin:/bin"\n'
                'printf "%s\\n" "$ZSH_EXECUTION_STRING" >> "$OC_SHELL_CALLS"\n'
                + ('printf "banner node 18.2.0\\n"\n' if name.endswith("marker-missing") or name.endswith("marker-invalid")
                    else 'printf "banner node 18.2.0\\nMLXOCV=0 18.2.0\\n"\n')
                + ('exit 0\n' if name.endswith("marker-missing") else '')
                + ('printf "MLXOCL=invalid\\n"; exit 0\n' if name.endswith("marker-invalid") else '')
                + ('exit 2\n' if name.endswith("shell-failed") else '')
            )
            if version is not None:
                executable = bindir / "opencode"
                executable.write_text('#!/bin/sh\n'
                    'test "$1" = --version || exit 99\n'
                    'printf "version\\n" >> "$OC_VERSION_CALLS"\n'
                    'printf "%s\\n" "$OC_FIXTURE_VERSION"\n'
                    'exit "$OC_FIXTURE_RC"\n')
                executable.chmod(0o755)
            if legacy:
                executable = bindir / "opencode2"
                executable.write_text('#!/bin/sh\nexit 99\n')
                executable.chmod(0o755)
            env = os.environ.copy()
            env.update(HOME=str(home), ZDOTDIR=str(home), TMPDIR=str(root),
                PATH="/usr/bin:/bin", XDG_CONFIG_HOME=str(home / ".config"),
                OC_FIXTURE_BIN=str(bindir), OC_SHELL_CALLS=str(calls),
                OC_VERSION_CALLS=str(version_calls), OC_FIXTURE_VERSION=version or "",
                OC_FIXTURE_RC=str(rc))
            result = subprocess.run([str(binary), "launch", agent, "--print", "--no-start",
                "--url", base], env=env, text=True, capture_output=True, timeout=20)
            errors = []
            observed = calls.read_text().splitlines() if calls.exists() else []
            if len(observed) != count:
                errors.append(f"shell count {len(observed)} != {count}")
            expected_versions = int(version is not None and name not in
                ("canonical-marker-missing", "canonical-shell-failed", "alias-marker-missing", "alias-marker-invalid", "alias-shell-failed"))
            observed_versions = len(version_calls.read_text().splitlines()) if version_calls.exists() else 0
            if observed_versions != expected_versions:
                errors.append(f"version calls {observed_versions} != {expected_versions}")
            legacy_calls = sum("command -v opencode2" in command for command in observed)
            if legacy_calls != int(count == 2):
                errors.append(f"legacy probes {legacy_calls} != {int(count == 2)}")
            config = home / ".mlx-serve" / "opencode2" / "opencode" / "cli.json"
            if profile:
                if result.returncode != 0:
                    errors.append(f"exit {result.returncode} != 0")
                expected_bin = "opencode2" if profile == "legacy" else "opencode"
                invocation = (f"\n{expected_bin} --model mlx/fixture-model" if profile == "v1"
                    else f"\n{expected_bin} --standalone")
                if invocation not in result.stdout:
                    errors.append(f"missing invocation {invocation!r}")
                if profile == "v1":
                    if config.exists() or "XDG_CONFIG_HOME" in result.stdout:
                        errors.append("v1 used v2 config")
                else:
                    if not config.exists():
                        errors.append("v2 config missing")
                    else:
                        parsed = json.loads(config.read_text())
                        if parsed.get("plugins", [{}])[-1].get("options", {}).get("metricsUrl") != base + "/metrics.json":
                            errors.append("wrong metrics URL")
                    if 'export XDG_CONFIG_HOME="$HOME/.mlx-serve/opencode2"' not in result.stdout:
                        errors.append("dedicated XDG directory missing")
                if profile != "legacy" and f"detected OpenCode {version}" not in result.stderr:
                    errors.append("detected version omitted or replaced by banner")
            else:
                if result.returncode != 1 or error not in result.stderr:
                    errors.append(f"expected exit 1 with {error!r}, got {result.returncode}")
                if (home / ".mlx-serve").exists() or result.stdout:
                    errors.append("failure wrote config or printed launch script")
            if errors:
                failures.append(name)
                print(f"FAIL: {name}: {'; '.join(errors)}")
                print(result.stderr)
            else:
                print(f"PASS: {name}: shells={len(observed)}, legacy={legacy_calls}, versions={observed_versions}")
finally:
    server.shutdown()
    server.server_close()
    thread.join()
print(f"{len(cases) - len(failures)}/{len(cases)} passed, {len(failures)} failed")
sys.exit(bool(failures))
PY
