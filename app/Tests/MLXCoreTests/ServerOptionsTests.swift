import XCTest
@testable import MLXCore

/// Unit tests for `ServerOptions.toCLIArgs()` — the Settings UI relies on
/// this to translate user choices into the actual `mlx-serve` CLI invocation.
/// A wrong arg here would silently launch the server with the wrong config,
/// so we lock down the contract.
final class ServerOptionsTests: XCTestCase {
    func testDefaultsProduceCanonicalArgs() {
        let opts = ServerOptions()
        let args = opts.toCLIArgs()

        // Always present.
        XCTAssertEqual(args.first, "--serve")
        XCTAssertTrue(contains(args, flag: "--port", value: "11234"))
        XCTAssertTrue(contains(args, flag: "--host", value: "0.0.0.0"))
        XCTAssertTrue(contains(args, flag: "--log-level", value: "info"))
        // Default `ctxSize == 0` means "Auto" — server picks the memory-bounded
        // safe ceiling at startup. The CLI flag is omitted entirely in that
        // case (the server's own `getEffectiveContextLength` runs).
        XCTAssertFalse(contains(args, flag: "--ctx-size"))

        // Spec-decode: PLD default-on.
        XCTAssertTrue(args.contains("--pld"))
        XCTAssertFalse(args.contains("--no-pld"))
        XCTAssertTrue(contains(args, flag: "--pld-draft-len", value: "5"))
        XCTAssertTrue(contains(args, flag: "--pld-key-len", value: "3"))

        // Off-by-default flags omit themselves.
        XCTAssertFalse(args.contains("--no-vision"))
        XCTAssertFalse(args.contains("--drafter"))
        XCTAssertFalse(contains(args, flag: "--timeout"))   // 300 = default, not emitted
    }

    /// SYNC GUARD (2026-06): every "emit-only-when-non-default" launch flag
    /// assumes a specific SERVER-side default. If the Swift default — or the
    /// emit guard — drifts from it, the flag is omitted and the server silently
    /// runs ITS default. That's exactly the prefix-cache-entries (1-vs-32, OOM)
    /// and llama-cache-entries (1-vs-4) surprises. For a fresh ServerOptions(),
    /// NONE of these flags may appear: the app default must equal the server
    /// default. The comments name the Zig source of truth — change both together.
    /// See CLAUDE.md "ServerOptions defaults must mirror the Zig server defaults".
    func testDefaultsMatchDocumentedServerDefaults() {
        let d = ServerOptions()
        // Each Swift default MUST equal the server default it mirrors. The
        // llama-cache-entries bug was a SILENT one: Swift defaulted to 1, the
        // server to 4, and the flag was omitted (1 != 1 is false), so the
        // server ran 4 while the UI showed 1 — an emit-check alone can't catch
        // that. These value assertions can; update both sides together.
        XCTAssertEqual(d.ctxSize, 0)                  // main.zig ctx_size
        XCTAssertEqual(d.requestTimeout, 300)         // main.zig timeout
        XCTAssertEqual(d.noVision, false)             // main.zig no_vision
        XCTAssertEqual(d.maxConcurrent, 1)            // server.zig max_concurrent
        XCTAssertEqual(d.kvQuant, .off)               // server.zig kv-quant
        XCTAssertEqual(d.prefixCacheMem, "2GB")       // server.zig prefix_cache_mem_bytes
        XCTAssertEqual(d.tokenizeCacheEntries, 4)     // server.zig tokenize_cache_entries
        XCTAssertEqual(d.llamaKvQuant, .off)          // server.zig llama_kv_quant
        XCTAssertEqual(d.llamaCacheEntries, 4)        // server.zig llama_cache_entries
        XCTAssertEqual(d.skipMemPreflight, false)     // scheduler.zig skip_mem_preflight
        XCTAssertEqual(d.ssdStreaming, false)         // main.zig ds4_ssd_streaming

        // Corollary: with defaults matching the server, a fresh launch omits
        // every match-default flag (each guard fires only on a real change).
        let args = d.toCLIArgs(physicalMemoryBytes: 64 * Self.GiB)
        for flag in ["--ctx-size", "--timeout", "--no-vision", "--max-concurrent",
                     "--kv-quant", "--prefix-cache-mem", "--tokenize-cache-entries",
                     "--llama-kv-quant", "--llama-cache-entries", "--skip-mem-preflight",
                     "--ssd-streaming", "--top-k", "--drafter"] {
            XCTAssertFalse(args.contains(flag),
                "\(flag) appeared at default — its Swift default or emit-guard drifted from the server")
        }
        // ALWAYS-emitted flags (authoritative from the app) MUST be present.
        XCTAssertTrue(args.contains("--prefix-cache-entries"))
    }

    /// 4096 was too low: a single thinking trace + agentic answer routinely
    /// tripped `finish_reason: "length"` and surfaced the truncation notice.
    /// The default per-turn output budget must be generous — the server still
    /// clamps it to the live context window, so a high value can't overflow.
    func testDefaultMaxTokensIsGenerous() {
        XCTAssertGreaterThanOrEqual(ServerOptions().defaultMaxTokens, 16384)
    }

    func testPLDOffUsesNoPldFlag() {
        var opts = ServerOptions()
        opts.enablePLD = false
        let args = opts.toCLIArgs()
        XCTAssertTrue(args.contains("--no-pld"))
        XCTAssertFalse(args.contains("--pld"))
    }

    func testCustomPortAndCtxSizeAreEmitted() {
        var opts = ServerOptions()
        opts.port = 8080
        opts.ctxSize = 65536
        let args = opts.toCLIArgs()
        XCTAssertTrue(contains(args, flag: "--port", value: "8080"))
        XCTAssertTrue(contains(args, flag: "--ctx-size", value: "65536"))
    }

    func testCtxSizeZeroOmitsFlag() {
        var opts = ServerOptions()
        opts.ctxSize = 0
        let args = opts.toCLIArgs()
        XCTAssertFalse(contains(args, flag: "--ctx-size"))
    }

    func testDrafterPathPullsInBlockSize() {
        var opts = ServerOptions()
        opts.drafterPath = "/tmp/gemma-4-E4B-it-assistant-bf16"
        opts.draftBlockSize = 6
        let args = opts.toCLIArgs()
        XCTAssertTrue(contains(args, flag: "--drafter", value: "/tmp/gemma-4-E4B-it-assistant-bf16"))
        XCTAssertTrue(contains(args, flag: "--draft-block-size", value: "6"))
    }

    func testEmptyDrafterPathOmitsBothFlags() {
        let opts = ServerOptions()  // drafterPath = ""
        let args = opts.toCLIArgs()
        XCTAssertFalse(args.contains("--drafter"))
        XCTAssertFalse(args.contains("--draft-block-size"))
    }

    func testCustomTimeoutIsEmittedWhenNonDefault() {
        var opts = ServerOptions()
        opts.requestTimeout = 600
        XCTAssertTrue(contains(opts.toCLIArgs(), flag: "--timeout", value: "600"))

        opts.requestTimeout = 0  // unlimited
        XCTAssertTrue(contains(opts.toCLIArgs(), flag: "--timeout", value: "0"))
    }

    // MARK: - Host / port (Settings UI fields)

    /// `parsePort` backs the Settings port text field — it must accept exactly
    /// what a TCP listen can bind and reject everything else, because an
    /// invalid value that slipped through would silently launch on a port the
    /// rest of the app (health checks, chat client) isn't watching.
    func testParsePortAcceptsValidPorts() {
        XCTAssertEqual(ServerOptions.parsePort("11234"), 11234)
        XCTAssertEqual(ServerOptions.parsePort(" 8080 "), 8080)   // trims whitespace
        XCTAssertEqual(ServerOptions.parsePort("1"), 1)
        XCTAssertEqual(ServerOptions.parsePort("65535"), 65535)
    }

    func testParsePortRejectsJunk() {
        XCTAssertNil(ServerOptions.parsePort(""))
        XCTAssertNil(ServerOptions.parsePort("0"))      // 0 = kernel-assigned ephemeral; client couldn't find the server
        XCTAssertNil(ServerOptions.parsePort("65536"))
        XCTAssertNil(ServerOptions.parsePort("-1"))
        XCTAssertNil(ServerOptions.parsePort("80x"))
        XCTAssertNil(ServerOptions.parsePort("abc"))
        XCTAssertNil(ServerOptions.parsePort("11 234"))
    }

    /// The host field is free text in Settings; a cleared field must not
    /// launch `--host ""` (the server would fail to bind).
    func testEmptyHostFallsBackToBindAll() {
        var opts = ServerOptions()
        opts.host = "   "
        XCTAssertTrue(contains(opts.toCLIArgs(), flag: "--host", value: "0.0.0.0"))
    }

    func testCustomHostIsEmitted() {
        var opts = ServerOptions()
        opts.host = "127.0.0.1"
        XCTAssertTrue(contains(opts.toCLIArgs(), flag: "--host", value: "127.0.0.1"))
    }

    func testNoVisionFlag() {
        var opts = ServerOptions()
        opts.noVision = true
        XCTAssertTrue(opts.toCLIArgs().contains("--no-vision"))
    }

    func testServerLaunchEqualsIgnoresPerRequestFields() {
        var a = ServerOptions()
        var b = ServerOptions()
        // Still purely per-request: max tokens, penalties, spec-decode
        // TriStates. (Sampling defaults — temperature/top-p/top-k — became
        // launch flags in 2026-06 so external clients like Claude Code
        // inherit them; covered by testSamplingDefaultsAffectRestartDetection.)
        b.defaultMaxTokens = 8192
        b.perRequestEnablePLD = .off
        b.defaultRepeatPenalty = 1.1
        XCTAssertTrue(a.serverLaunchEquals(b),
                     "Per-request defaults must NOT trigger restart")

        a.port = 9000
        XCTAssertFalse(a.serverLaunchEquals(b),
                      "Server-launch fields MUST trigger restart")
    }

    func testTriStateMaps() {
        XCTAssertNil(ServerOptions.TriState.auto.asOptionalBool)
        XCTAssertEqual(ServerOptions.TriState.on.asOptionalBool, true)
        XCTAssertEqual(ServerOptions.TriState.off.asOptionalBool, false)
    }

    func testRoundTripCodable() throws {
        var opts = ServerOptions()
        opts.port = 9999
        opts.drafterPath = "/x/y/z"
        opts.defaultTemperature = 0.42
        opts.perRequestEnableDrafter = .off

        let data = try JSONEncoder().encode(opts)
        let decoded = try JSONDecoder().decode(ServerOptions.self, from: data)
        XCTAssertEqual(opts, decoded)
    }

    // MARK: - GGUF + common-engine flags

    func testLlamaKvQuantOmittedAtDefault() {
        let args = ServerOptions().toCLIArgs()
        XCTAssertFalse(args.contains("--llama-kv-quant"),
                      "default (.off) must NOT emit the flag so existing CLI invocations stay byte-identical")
    }

    func testLlamaKvQuantQ8EmitsFlag() {
        var opts = ServerOptions()
        opts.llamaKvQuant = .q8
        let args = opts.toCLIArgs()
        XCTAssertTrue(contains(args, flag: "--llama-kv-quant", value: "q8"))
    }

    func testLlamaKvQuantQ4EmitsFlag() {
        var opts = ServerOptions()
        opts.llamaKvQuant = .q4
        let args = opts.toCLIArgs()
        XCTAssertTrue(contains(args, flag: "--llama-kv-quant", value: "q4"))
    }

    func testLlamaCacheEntriesOmittedAtDefault() {
        // Default (4) MUST equal the server's `llama_cache_entries` default so
        // omitting the flag runs the same value — the prior Swift default of 1
        // silently ran the server's 4 (mismatch surprise).
        let args = ServerOptions().toCLIArgs()
        XCTAssertFalse(args.contains("--llama-cache-entries"))
        XCTAssertEqual(ServerOptions().llamaCacheEntries, 4,
                       "must mirror server.zig llama_cache_entries default")
    }

    func testLlamaCacheEntriesEmitsWhenChanged() {
        var opts = ServerOptions()
        opts.llamaCacheEntries = 1   // off the (4) default → must be emitted
        XCTAssertTrue(contains(opts.toCLIArgs(), flag: "--llama-cache-entries", value: "1"))
        opts.llamaCacheEntries = 8
        XCTAssertTrue(contains(opts.toCLIArgs(), flag: "--llama-cache-entries", value: "8"))
    }

    func testTokenizeCacheEntriesOmittedAtDefault() {
        let args = ServerOptions().toCLIArgs()
        XCTAssertFalse(args.contains("--tokenize-cache-entries"),
                      "default (4) must NOT emit — matches server-side default")
    }

    // MARK: - Prefix cache: RAM clamp + always-emit (16 GB OOM fix)

    private static let GiB: UInt64 = 1_073_741_824

    /// Regression: the app only emitted `--prefix-cache-entries` when the value
    /// differed from 1, but the SERVER default is 32 — so the flag was never
    /// sent and a 32-entry cache silently launched, filling 16 GB Macs. The
    /// flag must now ALWAYS be emitted so the server's 32 can't leak.
    func testPrefixCacheEntriesAlwaysEmitted() {
        let args = ServerOptions().toCLIArgs(physicalMemoryBytes: 64 * Self.GiB)
        XCTAssertTrue(args.contains("--prefix-cache-entries"),
                      "must always emit so the server's default-32 never leaks through")
    }

    func testPrefixCacheEntriesClampedOnLowRAM() {
        var opts = ServerOptions()
        opts.prefixCacheEntries = 8
        // 16 GB Mac → capped to 1.
        XCTAssertTrue(contains(opts.toCLIArgs(physicalMemoryBytes: 16 * Self.GiB),
                               flag: "--prefix-cache-entries", value: "1"))
        // 32 GB → capped to 8 (== request here, so unchanged).
        XCTAssertTrue(contains(opts.toCLIArgs(physicalMemoryBytes: 32 * Self.GiB),
                               flag: "--prefix-cache-entries", value: "8"))
        // 64 GB → uncapped.
        opts.prefixCacheEntries = 16
        XCTAssertTrue(contains(opts.toCLIArgs(physicalMemoryBytes: 64 * Self.GiB),
                               flag: "--prefix-cache-entries", value: "16"))
    }

    func testRamCappedPrefixCacheEntriesTiers() {
        // ≤18 GB → 1
        XCTAssertEqual(ServerOptions.ramCappedPrefixCacheEntries(8, physicalMemoryBytes: 16 * Self.GiB), 1)
        XCTAssertEqual(ServerOptions.ramCappedPrefixCacheEntries(32, physicalMemoryBytes: 18 * Self.GiB), 1)
        // ≤36 GB → 8
        XCTAssertEqual(ServerOptions.ramCappedPrefixCacheEntries(32, physicalMemoryBytes: 32 * Self.GiB), 8)
        XCTAssertEqual(ServerOptions.ramCappedPrefixCacheEntries(4, physicalMemoryBytes: 32 * Self.GiB), 4,
                       "clamp is a ceiling, never raises a smaller request")
        // big RAM → uncapped
        XCTAssertEqual(ServerOptions.ramCappedPrefixCacheEntries(16, physicalMemoryBytes: 128 * Self.GiB), 16)
        // explicit disable is preserved on every machine
        XCTAssertEqual(ServerOptions.ramCappedPrefixCacheEntries(0, physicalMemoryBytes: 16 * Self.GiB), 0)
    }

    func testPrefixCacheDisableSurvivesClamp() {
        var opts = ServerOptions()
        opts.prefixCacheEntries = 0
        XCTAssertTrue(contains(opts.toCLIArgs(physicalMemoryBytes: 16 * Self.GiB),
                               flag: "--prefix-cache-entries", value: "0"))
    }

    func testTokenizeCacheEntriesEmitsWhenChanged() {
        var opts = ServerOptions()
        opts.tokenizeCacheEntries = 0
        var args = opts.toCLIArgs()
        XCTAssertTrue(contains(args, flag: "--tokenize-cache-entries", value: "0"))
        opts.tokenizeCacheEntries = 16
        args = opts.toCLIArgs()
        XCTAssertTrue(contains(args, flag: "--tokenize-cache-entries", value: "16"))
    }

    func testServerLaunchEqualsCoversNewFields() {
        var a = ServerOptions()
        var b = ServerOptions()
        // Each new field flipping must trigger a restart.
        b.llamaKvQuant = .q4
        XCTAssertFalse(a.serverLaunchEquals(b))
        b = ServerOptions()
        b.llamaCacheEntries = 2   // off the default (4)
        XCTAssertFalse(a.serverLaunchEquals(b))
        b = ServerOptions()
        b.tokenizeCacheEntries = 0
        XCTAssertFalse(a.serverLaunchEquals(b))
        // Sanity: untouched defaults are equal.
        a = ServerOptions(); b = ServerOptions()
        XCTAssertTrue(a.serverLaunchEquals(b))
    }

    // MARK: - Log level

    func testLogLevelDefaultIsInfo() {
        let args = ServerOptions().toCLIArgs()
        XCTAssertTrue(contains(args, flag: "--log-level", value: "info"))
    }

    func testCustomLogLevelEmitsCLIFlag() {
        for lvl in ServerOptions.LogLevel.allCases {
            var opts = ServerOptions()
            opts.logLevel = lvl
            XCTAssertTrue(
                contains(opts.toCLIArgs(), flag: "--log-level", value: lvl.rawValue),
                "logLevel=\(lvl.rawValue) must emit --log-level \(lvl.rawValue)"
            )
        }
    }

    func testLogLevelChangeTriggersRestart() {
        var a = ServerOptions()
        var b = ServerOptions()
        b.logLevel = .debug
        XCTAssertFalse(a.serverLaunchEquals(b),
                       "Switching log level must require a server restart")
        a.logLevel = .debug
        XCTAssertTrue(a.serverLaunchEquals(b))
    }

    func testLogLevelHasHumanReadableLabel() {
        // The Settings picker shows these — empty labels would render blank rows.
        for lvl in ServerOptions.LogLevel.allCases {
            XCTAssertFalse(lvl.label.isEmpty,
                           "\(lvl.rawValue) needs a label for the Settings picker")
        }
    }

    // MARK: - Engine inference

    func testEngineFromArchitecture() {
        // The Settings UI hides MLX-only sections when engine != .mlx and
        // surfaces the GGUF section when engine == .llama. The discriminator
        // is the `architecture` string the server reports for the active
        // model, derived from `model_type` in config.json (or the GGUF stub).
        var info = ModelInfo(name: "x", quantBits: 4, layers: 0,
                             hiddenSize: 0, vocabSize: 0,
                             contextLength: 0, modelMaxTokens: 0,
                             architecture: "gguf")
        XCTAssertEqual(info.engine, .llama)
        info.architecture = "deepseek_v4"
        XCTAssertEqual(info.engine, .dsv4)
        info.architecture = "gemma4"
        XCTAssertEqual(info.engine, .mlx)
        info.architecture = "qwen3_5_moe"
        XCTAssertEqual(info.engine, .mlx)
        info.architecture = ""  // older server build that omits the field
        XCTAssertEqual(info.engine, .mlx, "empty arch must default to .mlx (the most common path)")
    }

    // MARK: helpers

    private func contains(_ args: [String], flag: String, value: String? = nil) -> Bool {
        guard let i = args.firstIndex(of: flag) else { return false }
        guard let value else { return true }
        let next = i + 1
        return next < args.count && args[next] == value
    }
}

extension ServerOptionsTests {
    /// The Settings temperature must reach third-party clients (Claude Code
    /// omits sampling params entirely, so the server-launch default is the
    /// only channel). Top-p rides along; top-k 0 means "no opinion" and must
    /// be OMITTED so the model's generation_config.json recommendation
    /// (Qwen 3.6: top_k=20, Gemma 4: 64) stays in effect.
    func testSamplingDefaultsReachLaunchArgs() {
        var opts = ServerOptions()
        opts.defaultTemperature = 0.7
        opts.defaultTopP = 0.95
        opts.defaultTopK = 0
        let args = opts.toCLIArgs()
        XCTAssertTrue(contains(args, flag: "--temp", value: "0.7"))
        XCTAssertTrue(contains(args, flag: "--top-p", value: "0.95"))
        XCTAssertFalse(args.contains("--top-k"))

        opts.defaultTopK = 40
        XCTAssertTrue(contains(opts.toCLIArgs(), flag: "--top-k", value: "40"))
    }

    /// Changing a sampling default must trip the restart detector — these now
    /// affect the launched process, not just the app's own request bodies.
    func testSamplingDefaultsAffectRestartDetection() {
        let base = ServerOptions()
        var changed = base
        changed.defaultTemperature = 0.42
        XCTAssertFalse(base.serverLaunchEquals(changed))
    }
}

extension ServerOptionsTests {
    /// CHARACTERIZATION GUARD for the migration-safe `init(from:)`: it decodes
    /// key-by-key with `decodeIfPresent`, so a field added to the struct + CodingKeys
    /// but FORGOTTEN in `init(from:)` would silently never load (decoded = default),
    /// with no compiler error. Setting EVERY field to a non-default and asserting a
    /// full round-trip catches exactly that — a forgotten key makes the decoded
    /// value differ from the encoded one.
    func testEveryFieldRoundTripsThroughCustomDecoder() throws {
        var o = ServerOptions()
        o.host = "127.0.0.1"
        o.port = 9999
        o.ctxSize = 65536
        o.noVision = true
        o.logLevel = .debug
        o.requestTimeout = 600
        o.enablePLD = false
        o.pldDraftLen = 7
        o.pldKeyLen = 4
        o.drafterPath = "/x/y/drafter"
        o.draftBlockSize = 8
        o.maxConcurrent = 4
        o.kvQuant = .int8
        o.prefixCacheEntries = 3   // off the default (8) so the round-trip moves it
        o.prefixCacheMem = "4GB"
        o.skipMemPreflight = true
        o.ssdStreaming = true
        o.llamaKvQuant = .q8
        o.llamaCacheEntries = 2   // off the default (4) so the round-trip moves it
        o.tokenizeCacheEntries = 16
        o.defaultMaxTokens = 8192
        o.defaultTemperature = 0.42
        o.defaultTopP = 0.5
        o.defaultTopK = 40
        o.defaultRepeatPenalty = 1.2
        o.defaultPresencePenalty = 0.3
        o.defaultReasoningBudget = 2048
        o.defaultEnableThinking = true
        o.perRequestEnablePLD = .on
        o.perRequestEnableDrafter = .off
        o.telegram = .init(enabled: true, botToken: "1:abc", agentMode: true,
                           useMCP: true, enableThinking: true, allowedChatIds: [7, 8])
        o.sandbox = .init(enabled: true, baseImage: "alpine")

        XCTAssertNotEqual(o, ServerOptions(), "sanity: every field moved off its default")
        let decoded = try JSONDecoder().decode(ServerOptions.self, from: try JSONEncoder().encode(o))
        XCTAssertEqual(o, decoded, "a field missing from the custom init(from:) would revert to its default here")
    }

    /// Slider arithmetic leaves float dirt (0.8 − 0.1 = 0.7000000000000001);
    /// argv must carry the clean decimal (seen verbatim in `ps` output live).
    func testSamplingFlagFormattingIsClean() {
        var opts = ServerOptions()
        opts.defaultTemperature = 0.8 - 0.1
        XCTAssertTrue(contains(opts.toCLIArgs(), flag: "--temp", value: "0.7"))
    }
}

extension ServerOptionsTests {
    // MARK: - Skip-memory-preflight CLI flag
    //
    // The MLX loader's free-RAM pre-flight is now toggled by the
    // `--skip-mem-preflight` CLI flag (was the MLX_SERVE_SKIP_MEM_PREFLIGHT env
    // var) — same shape as every other launch knob, so it shows in --help, in
    // `ps`, and in the launch-command echo at the top of the server log.

    func testSkipMemPreflightDefaultsOff() {
        XCTAssertFalse(ServerOptions().skipMemPreflight,
                       "the safety check must stay on by default")
    }

    func testSkipMemPreflightOmittedByDefault() {
        XCTAssertFalse(ServerOptions().toCLIArgs().contains("--skip-mem-preflight"),
                       "default (off) must not emit the flag so the pre-flight runs")
    }

    func testSkipMemPreflightEmitsFlagWhenOn() {
        var opts = ServerOptions()
        opts.skipMemPreflight = true
        XCTAssertTrue(opts.toCLIArgs().contains("--skip-mem-preflight"))
    }

    /// It used to be an env var; the clean break makes it a normal CLI flag.
    func testSkipMemPreflightTakesNoValueArg() {
        var opts = ServerOptions()
        opts.skipMemPreflight = true
        let args = opts.toCLIArgs()
        guard let i = args.firstIndex(of: "--skip-mem-preflight") else {
            return XCTFail("flag missing")
        }
        // It's a bare boolean flag — the next token must be another flag (or end),
        // never a value the server would mis-parse as a model path etc.
        if i + 1 < args.count {
            XCTAssertTrue(args[i + 1].hasPrefix("--"),
                          "boolean flag must not be followed by a value token")
        }
    }

    func testSkipMemPreflightChangeTriggersRestart() {
        var a = ServerOptions()
        var b = ServerOptions()
        b.skipMemPreflight = true
        XCTAssertFalse(a.serverLaunchEquals(b),
                       "toggling the memory pre-flight must require a server restart")
        a.skipMemPreflight = true
        XCTAssertTrue(a.serverLaunchEquals(b))
    }
}

extension ServerOptionsTests {
    // MARK: - ds4 SSD weight-streaming CLI flag (--ssd-streaming, issue #39)
    //
    // DeepSeek-V4-Flash can be larger than RAM; --ssd-streaming makes the
    // embedded ds4 engine stream expert weights from SSD instead of holding the
    // full model resident (skips warmup + residency). ds4-only — the MLX and
    // llama.cpp engines ignore it. Same bare-boolean shape as --skip-mem-preflight,
    // so it shows in --help, in `ps`, and in the launch-command echo.

    func testSsdStreamingDefaultsOff() {
        // Mirrors main.zig `var ds4_ssd_streaming: bool = false` — the Swift
        // default must equal the server default (see the defaults-mirror gotcha).
        XCTAssertFalse(ServerOptions().ssdStreaming,
                       "ds4 SSD streaming must be opt-in")
    }

    func testSsdStreamingOmittedByDefault() {
        XCTAssertFalse(ServerOptions().toCLIArgs().contains("--ssd-streaming"),
                       "default (off) must not emit the flag")
    }

    func testSsdStreamingEmitsFlagWhenOn() {
        var opts = ServerOptions()
        opts.ssdStreaming = true
        XCTAssertTrue(opts.toCLIArgs().contains("--ssd-streaming"))
    }

    func testSsdStreamingTakesNoValueArg() {
        var opts = ServerOptions()
        opts.ssdStreaming = true
        let args = opts.toCLIArgs()
        guard let i = args.firstIndex(of: "--ssd-streaming") else {
            return XCTFail("flag missing")
        }
        // Bare boolean flag — the next token must be another flag (or end),
        // never a value the server would mis-parse.
        if i + 1 < args.count {
            XCTAssertTrue(args[i + 1].hasPrefix("--"),
                          "boolean flag must not be followed by a value token")
        }
    }

    func testSsdStreamingChangeTriggersRestart() {
        var a = ServerOptions()
        var b = ServerOptions()
        b.ssdStreaming = true
        XCTAssertFalse(a.serverLaunchEquals(b),
                       "toggling ds4 SSD streaming must require a server restart")
        a.ssdStreaming = true
        XCTAssertTrue(a.serverLaunchEquals(b))
    }

    /// The ds4 Settings section renders via `if let m = serverFlagFields["ssdStreaming"]`
    /// — a missing key would silently drop the toggle from the UI with no error.
    /// Pin the metadata's presence + restart flag (the testable slice of the
    /// otherwise-untestable SwiftUI wiring).
    func testSsdStreamingHasSettingsMetadata() {
        guard let field = ServerOptions.serverFlagFields["ssdStreaming"] else {
            return XCTFail("ssdStreaming metadata missing — the Settings toggle would silently disappear")
        }
        XCTAssertFalse(field.title.isEmpty)
        XCTAssertFalse(field.explainer.isEmpty)
        XCTAssertTrue(field.needsRestart, "ssd-streaming is a launch flag — must flag a restart")
    }
}

extension ServerOptionsTests {
    // MARK: - Agent sandbox (contain) — app-side setting, NOT a server-launch flag
    //
    // The sandbox routes the agent's shell commands into an isolated Linux guest
    // (the embedded `contain` library) instead of running them on host macOS. It
    // lives on the app side (the tool executor reads it), so — like `telegram` —
    // it must never reach the server CLI or trip the restart banner.

    func testSandboxDefaultsOff() {
        let s = ServerOptions().sandbox
        XCTAssertFalse(s.enabled, "the sandbox must be opt-in (off by default)")
        XCTAssertEqual(s.baseImage, "ddalcu/agent-shell",
                       "default base image must be arm64 — the HVF guest can't run an amd64-only image")
    }

    func testSandboxIsNotAServerLaunchFlag() {
        var opts = ServerOptions()
        opts.sandbox.enabled = true
        opts.sandbox.baseImage = "ubuntu:24.04"
        XCTAssertTrue(ServerOptions().serverLaunchEquals(opts),
                      "toggling the sandbox is app-side — it must not require a server restart")
        let args = opts.toCLIArgs()
        XCTAssertFalse(args.contains { $0.contains("sandbox") || $0.contains("contain") },
                       "the sandbox setting must never reach the server CLI")
    }

    func testSandboxLegacyBlobDecodesToDefaults() throws {
        // A config.json written before the sandbox field existed must decode with
        // sandbox defaulted (off), never throw and reset the user's whole config.
        let legacy = #"{"host":"0.0.0.0","port":11234}"#.data(using: .utf8)!
        let decoded = try JSONDecoder().decode(ServerOptions.self, from: legacy)
        XCTAssertFalse(decoded.sandbox.enabled)
        XCTAssertEqual(decoded.sandbox.baseImage, "ddalcu/agent-shell")
    }

    func testSandboxNetworkDefaultsOnAndDecodesLegacyBlobs() throws {
        // Network + live port mapping is the useful default for an agent that
        // builds and runs servers; the toggle exists to opt back into isolation.
        XCTAssertTrue(ServerOptions.SandboxConfig().network)
        // A blob written before the field existed must default it, not throw.
        let legacy = #"{"sandbox":{"enabled":true,"baseImage":"alpine"}}"#.data(using: .utf8)!
        let decoded = try JSONDecoder().decode(ServerOptions.self, from: legacy)
        XCTAssertTrue(decoded.sandbox.network)
        // And a stored `false` must round-trip.
        var o = ServerOptions()
        o.sandbox.network = false
        let rt = try JSONDecoder().decode(ServerOptions.self, from: try JSONEncoder().encode(o))
        XCTAssertFalse(rt.sandbox.network)
    }

    func testSandboxRoundTripsThroughDecoder() throws {
        var o = ServerOptions()
        o.sandbox = .init(enabled: true, baseImage: "python:3.12")
        let decoded = try JSONDecoder().decode(ServerOptions.self, from: try JSONEncoder().encode(o))
        XCTAssertEqual(decoded.sandbox, o.sandbox,
                       "a field missing from SandboxConfig.init(from:) would revert to its default here")
    }

    func testSandboxTrimmedBaseImageStripsWhitespace() {
        var s = ServerOptions.SandboxConfig()
        s.baseImage = "  nikolaik/python-nodejs\n"
        XCTAssertEqual(s.trimmedBaseImage, "nikolaik/python-nodejs")
    }

    func testSandboxResetToDefaultsClearsIt() {
        // Settings' "Reset to Defaults" assigns ServerOptions(); the sandbox must
        // come back off with the default image.
        var o = ServerOptions()
        o.sandbox = .init(enabled: true, baseImage: "alpine")
        o = ServerOptions()
        XCTAssertFalse(o.sandbox.enabled)
        XCTAssertEqual(o.sandbox.baseImage, "ddalcu/agent-shell")
    }
}
