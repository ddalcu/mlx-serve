#!/usr/bin/env python3
"""Tests of scripts/lint.py's rules (one violation and one clean case each, on in-memory files) and of
scripts/unit_globals.py's scan of a unit object's symbols.

  python3 scripts/lint_test.py   # or: zig build lint-test
"""
import os, sys, unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lint  # noqa: E402
import unit_globals  # noqa: E402

HDR = "//! A file.\n"
# Plugin folders the name rules know about; each fixture tree carries them.
PLUGINS = {f"src/plugins/{p}/{p}.zig": HDR for p in ("qwen4_exp", "minimax_h3", "d1", "krea")}


def count(rule, files, pins=""):
    return len(lint.run({**PLUGINS, **files}, pins, [])["viol"][rule])


class Boundaries(unittest.TestCase):
    def test_host_imports_engine(self):
        self.assertEqual(count("host-imports-engine", {"src/host/a/x.zig": HDR + 'const m = @import("../../engines/mlx/model.zig");\n'}), 1)
        self.assertEqual(count("host-imports-engine", {"src/host/a/x.zig": HDR + 'const e = @import("../../abi/engine.zig");\n'}), 0)

    def test_host_imports_plugin(self):
        self.assertEqual(count("host-imports-plugin", {"src/host/a/x.zig": HDR + 'const k = @import("../../plugins/krea/krea.zig");\n'}), 1)
        self.assertEqual(count("host-imports-plugin", {"src/host/a/x.zig": HDR + 'const l = @import("../log.zig");\n'}), 0)

    def test_engine_imports_plugin(self):
        self.assertEqual(count("engine-imports-plugin", {"src/engines/mlx/x.zig": HDR + 'const q = @import("../../plugins/qwen4_exp/qwen4_exp.zig");\n'}), 1)
        self.assertEqual(count("engine-imports-plugin", {"src/engines/mlx/x.zig": HDR + 'const b = @import("../../sdk/blocks/b.zig");\n'}), 0)

    def test_plugin_imports_core(self):
        self.assertEqual(count("plugin-imports-core", {"src/plugins/krea/a.zig": HDR + 'const l = @import("../../host/log.zig");\n'}), 1)
        ok = HDR + 'const s = @import("../../sdk/x.zig");\nconst o = @import("../d1/d1.zig");\nconst m = @import("../../mlx/ffi.zig");\nconst std = @import("std");\n'
        self.assertEqual(count("plugin-imports-core", {"src/plugins/krea/a.zig": ok}), 0)

    def test_media_plugin_imports_engine(self):
        imp = HDR + 'const m = @import("../../engines/mlx/model.zig");\n'
        self.assertEqual(count("media-plugin-imports-engine", {"src/plugins/krea/a.zig": imp}), 1)
        self.assertEqual(count("media-plugin-imports-engine", {"src/plugins/qwen4_exp/a.zig": imp}), 0)


class ModelNames(unittest.TestCase):
    def test_model_type_compare(self):
        line = 'fn f(c: C) bool { return std.mem.eql(u8, c.model_type, "qwen4_exp"); }\n'
        self.assertEqual(count("model-type-compare", {"src/host/x.zig": HDR + line}), 1)
        self.assertEqual(count("model-type-compare", {"src/plugins/krea/x.zig": HDR + line}), 0)
        self.assertEqual(count("model-type-compare", {"src/host/x.zig": HDR + "// " + line}), 0)

    def test_model_predicate(self):
        self.assertEqual(count("model-predicate", {"src/host/x.zig": HDR + "const a = cfg.isQwen4();\nconst b = cfg.isInkling();\n"}), 2)
        self.assertEqual(count("model-predicate", {"src/host/x.zig": HDR + "const a = cfg.isMoe();\n"}), 0)

    def test_plugin_name(self):
        self.assertEqual(count("plugin-name", {"src/engines/mlx/x.zig": HDR + "fn forwardQwen4With() void {}\n"}), 1)
        self.assertEqual(count("plugin-name", {"src/host/x.zig": HDR + 'const t = "minimax_h3";\n'}), 1)
        self.assertEqual(count("plugin-name", {"src/host/x.zig": HDR + "const m = .qwen4_exp;\nconst f = c.qwen4_exp;\n"}), 2)
        clean = HDR + "var h_3d = 0;\nconst d1 = 2;\nconst k = 0xD1B54A32;\n// qwen4 in a comment\n"
        self.assertEqual(count("plugin-name", {"src/engines/mlx/x.zig": clean}), 0)
        # Facts, not identity: named constants, config-field prefixes, dialect names, log labels, file names.
        facts = HDR + ('const QWEN4_NGRAM_ROWS = 1;\nconst s = c.qwen4_ngram_size;\nfn qwen4ToolCallTag() void {}\n'
                       'log.info("[qwen4] engaged", .{});\nconst p = "qwen4.metal";\nconst e = "QWEN4_PROFILE";\n')
        self.assertEqual(count("plugin-name", {"src/host/x.zig": facts}), 0)
        self.assertEqual(count("plugin-name", {"src/sdk/x.zig": HDR + "fn forwardQwen4With() void {}\n"}), 0)


class OneWay(unittest.TestCase):
    def test_raw_mlx_c(self):
        body = 'const mlx = @import("../mlx/ffi.zig");\nconst m = @import("../mlx/ffi.zig");\n_ = mlx.mlx_add(&o, a, b, s);\n_ = m.mlx_free(o);\n'
        self.assertEqual(count("raw-mlx-c", {"src/sdk/x.zig": HDR + body}), 2)
        self.assertEqual(count("raw-mlx-c", {"src/mlx/ops.zig": HDR + body}), 0)
        llama = 'const ffi = @import("ffi.zig");\nconst r = ffi.mlx_llama_open(p);\n'
        self.assertEqual(count("raw-mlx-c", {"src/engines/llama/bridge.zig": HDR + llama}), 0)

    def test_env_read(self):
        reads = HDR + 'const v = std.c.getenv("HOME");\nconst n = readEnvUsize("MLX_SERVE_X", 0);\n_ = setenv("MLX_SERVE_Y", "1", 1);\n'
        self.assertEqual(count("env-read", {"src/host/x.zig": reads}), 3)
        self.assertEqual(count("env-mention", {"src/host/x.zig": reads}), 0)
        self.assertEqual(count("env-read", {"src/env.zig": HDR + 'const v = std.c.getenv("MLX_SERVE_X");\n'}), 0)
        self.assertEqual(count("env-read", {"src/host/x.zig": HDR + "// MLX_SERVE_X=0 turns it off\n"}), 0)

    def test_env_mention(self):
        mention = HDR + 'log.info("engaged (MLX_SERVE_X=0 restores)", .{});\n'
        self.assertEqual(count("env-mention", {"src/host/x.zig": mention}), 1)
        self.assertEqual(count("env-read", {"src/host/x.zig": mention}), 0)
        self.assertEqual(count("env-mention", {"src/env.zig": mention}), 0)

    def test_inline_kernel(self):
        self.assertEqual(count("inline-kernel", {"src/sdk/x.zig": HDR + "const s =\n    \\\\kernel void f(uint i [[thread_position_in_grid]]) {}\n;\n"}), 1)
        self.assertEqual(count("inline-kernel", {"src/sdk/x.zig": HDR + "const s =\n    \\\\usage: mlx-serve run MODEL\n;\n"}), 0)


class Hygiene(unittest.TestCase):
    def test_file_header(self):
        self.assertEqual(count("file-header", {"src/host/x.zig": "const a = 1;\n"}), 1)
        self.assertEqual(count("file-header", {"src/host/x.zig": HDR}), 0)

    def test_history_comment(self):
        bad = HDR + ("// fixed in PR #12\n/// live 2026-10-01 on the Ultra\nconst a = 1; // this used to crash\n"
                     "// It used to leak\n//! Split out of scheduler.zig.\n// moved here (P1)\n")
        self.assertEqual(count("history-comment", {"src/host/x.zig": bad}), 6)
        ok = HDR + ('// the table is used to index rows\n// Used to thread triples through the view\n'
                    '// messages used to produce the reply\nconst d = "2026-10-01";\n')
        self.assertEqual(count("history-comment", {"src/host/x.zig": ok}), 0)

    def test_source_scan_test(self):
        bad = HDR + 'const src = @embedFile("x.zig");\ntest "scan" {\n    _ = src;\n}\ntest "direct" {\n    _ = @embedFile("y.zig");\n}\n'
        self.assertEqual(count("source-scan-test", {"src/host/x.zig": bad}), 2)
        ok = HDR + 'test "fixture" {\n    _ = @embedFile("fixtures/a.json");\n}\n'
        self.assertEqual(count("source-scan-test", {"src/host/x.zig": ok}), 0)

    def test_graph_scope(self):
        bad = HDR + "fn f(g: *Graph, x: Array) !Array {\n    return g.add(x, x);\n}\n"
        self.assertEqual(count("graph-scope", {"src/sdk/x.zig": bad}), 1)
        ok = HDR + "fn f(parent: *ops.Graph, x: Array) !Array {\n    var g = parent.scope();\n    defer g.deinit();\n    return parent.keep(try g.add(x, x));\n}\n"
        self.assertEqual(count("graph-scope", {"src/sdk/x.zig": ok}), 0)


class PinsForksRatchets(unittest.TestCase):
    def test_pins(self):
        pins = "# glob<TAB>needle<TAB>rule\nsrc/host/*.zig\tneedle_here\tL4 guard\n"
        self.assertEqual(count("pin-unmatched", {"src/host/x.zig": HDR}, pins), 1)
        self.assertEqual(count("pin-unmatched", {"src/host/x.zig": HDR + "// needle_here\n"}, pins), 0)
        self.assertEqual(count("pin-unmatched", {"src/host/x.zig": HDR}, "src/host/*.zig\n"), 1)  # malformed

    def test_forbid_pins(self):
        # A `!` needle forbids: the pin fails when any file matching the glob holds the rest.
        pins = "src/host/*.zig\t!cache.step +=\tone step mover\n"
        self.assertEqual(count("pin-unmatched", {"src/host/x.zig": HDR}, pins), 0)
        self.assertEqual(count("pin-unmatched", {"src/host/x.zig": HDR + "cache.step += 1;\n"}, pins), 1)
        self.assertEqual(count("pin-unmatched", {"src/engines/x.zig": HDR + "cache.step += 1;\n"}, pins), 0)

    def test_file_over_cap(self):
        prod = HDR + "const a = 1;\n" * lint.SOFT_CAP
        self.assertEqual(count("file-over-cap", {"src/host/x.zig": prod}), 1)
        tests = HDR + "test \"t\" {\n" + "    _ = 1;\n" * lint.SOFT_CAP + "}\n"
        self.assertEqual(count("file-over-cap", {"src/host/x.zig": tests}), 0)

    def test_forks(self):
        res = lint.run({"src/plugins/krea/attn.zig": "//! Krea's attention.\n//! copied_from: sdk/blocks/attention/sdpa.zig\n"}, "", [])
        self.assertEqual([(p, src) for p, _, src in res["forks"]], [("src/plugins/krea/attn.zig", "sdk/blocks/attention/sdpa.zig")])
        self.assertEqual(lint.run({"src/plugins/krea/a.zig": HDR + "// copied_from: x\n"}, "", [])["forks"], [])

    def test_strict_fails_only_enforced_rules(self):
        res = lint.run({"src/host/x.zig": "const a = 1;\n"}, "", ["file-header", "env-read"])
        self.assertEqual(lint.failed_rules(res), ["file-header"])
        self.assertEqual(lint.failed_rules(lint.run({"src/host/x.zig": "const a = 1;\n"}, "", [])), [])


class UnitGlobals(unittest.TestCase):
    def test_threadlocals_are_unit_state(self):
        nm = "\n".join([
            "0000000000ab7c68 (__DATA,__thread_vars) non-external _sdk.media.progress.stream_keepalive",
            "00000000013be348 (__DATA,__thread_bss) non-external _sdk.media.progress.stream_keepalive$tlv$init",
            "0000000000ab7c90 (__DATA,__bss) non-external _sdk.util.log.sink",
            "0000000000ab7ca0 (__DATA,__data) external _mlx_serve_unit_table",
        ])
        self.assertEqual(list(unit_globals.duplicated(nm, [], {"sdk.util.log.sink"})), ["sdk.media.progress.stream_keepalive"])


if __name__ == "__main__":
    unittest.main()
