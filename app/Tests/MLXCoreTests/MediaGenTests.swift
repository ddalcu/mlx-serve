import XCTest
import Foundation

// =============================================================================
// MARK: - Replicas from MediaGen.swift / PythonManager.swift
//
// MLXCore is an executable target — tests can't @testable-import it. Same
// replica pattern as HFSearchTests / ToolKeyOrderTests: copy the pure-logic
// pieces here verbatim and assert on them.
// =============================================================================

private enum QualityPresetReplica: String, CaseIterable {
    case fast = "Fast"
    case good = "Good"
    case quality = "Quality"
    case superQuality = "Super Quality"
}

private enum GenEventReplica: Equatable {
    case progress(step: Int, total: Int, message: String)
    case complete(path: String)
    case log(String)

    static func parse(_ line: String) -> GenEventReplica? {
        guard let data = line.data(using: .utf8),
              let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let type = obj["type"] as? String else { return nil }
        switch type {
        case "progress":
            let step = (obj["step"] as? Int) ?? 0
            let total = (obj["total"] as? Int) ?? 0
            let msg = (obj["message"] as? String) ?? ""
            return .progress(step: step, total: total, message: msg)
        case "complete":
            guard let path = obj["path"] as? String else { return nil }
            return .complete(path: path)
        case "error":
            let msg = (obj["message"] as? String) ?? "unknown error"
            return .log("ERROR: " + msg)
        default:
            return nil
        }
    }
}

/// Replica of `VideoModelPreset.frameLadder` — the 8N+1 ladder LTX-2.3
/// requires for its temporal patch encoder. Starts at 9 (the smallest
/// valid clip) and strides by 16, matching the documented LTX-2.3 ladder.
/// Off-grid frame counts get rejected by the pipeline with a confusing
/// error, so we lock the picker to this ladder.
private func frameLadder(maxFrames: Int) -> [Int] {
    var values: [Int] = []
    var n = 9
    while n <= maxFrames {
        values.append(n)
        n += 16
    }
    if !values.contains(maxFrames) { values.append(maxFrames) }
    return values
}

/// Replica of `RAMChecker.safeFrameCap` — bounds frame count by free RAM
/// at a given resolution. Coefficient 12.0 reflects ltx-2-mlx's memory
/// profile (VAE staging pushes unified memory harder than the old
/// diffusers path); minimum 9 matches the smallest valid 8N+1 clip.
private func safeFrameCap(modelRAMGB: Int, modelMaxFrames: Int, width: Int, height: Int, available: Int) -> Int {
    let pixelMP = Double(width * height) / 1_000_000.0
    let headroom = max(0, available - modelRAMGB)
    let perHundred = max(2.0, pixelMP * 12.0)
    let framesByRAM = Int((Double(headroom) / perHundred) * 100)
    return min(modelMaxFrames, max(9, framesByRAM))
}

// =============================================================================
// MARK: - Replica of PythonManager's venv import-probe
//
// `checkPackages()` builds one `python -c` program from `requiredImports` and
// treats a nonzero exit as "venv not ready → show the install pane". The bug:
// a bare `import ltx_pipelines_mlx` succeeds even after the git-pinned package
// drifts and stops exporting `TextToVideoPipeline`, so the app reported
// `.ready` and let the user click Generate — which then died at runtime with
// "cannot import name 'TextToVideoPipeline'". The probe must verify the exact
// symbols the generation script imports, not just that the module loads.
// Kept verbatim-in-sync with PythonManager.requiredImports / importProbeScript().
// =============================================================================

/// Pipeline symbols `VideoGenService.script` imports from `ltx_pipelines_mlx`
/// (ltx-2-mlx ≥0.14 names).
private let ltxPipelineSymbolsReplica = ["TI2VidOneStagePipeline", "TI2VidTwoStagesPipeline", "TI2VidTwoStagesHQPipeline"]

private let requiredImportsReplica: [String] = [
    "import mflux",
    "import ltx_core_mlx",
    "from ltx_pipelines_mlx import " + ltxPipelineSymbolsReplica.joined(separator: ", "),
    "from mlx_audio.tts.generate import generate_audio",
    "import PIL",
    "import safetensors",
    "import huggingface_hub",
]

private func importProbeScriptReplica() -> String {
    requiredImportsReplica.joined(separator: "\n")
}

/// The single import statement that probes `ltx_pipelines_mlx`, isolated so a
/// test can run it against a throwaway package without needing mflux/PIL/etc.
private func ltxProbeStatementReplica() -> String {
    requiredImportsReplica.first { $0.contains("ltx_pipelines_mlx") }!
}

/// Replica of `VideoGenService.indicatesVenvNeedsReinstall`.
private func indicatesVenvNeedsReinstallReplica(_ message: String) -> Bool {
    message.localizedCaseInsensitiveContains("import failed")
}

/// Replica of `VideoGenService.decodeFrames` — the native `/v1/video/generations`
/// rgb8 wire format (frames/height/width/fps + base64 data, with a length check).
/// MLXCore is an executable target so tests can't import the real method; this
/// pins the contract the server (`src/ltx_server.zig`) emits.
struct DecodedFramesReplica: Equatable {
    var rgb: Data; var frames: Int; var height: Int; var width: Int; var fps: Int
}
func decodeFramesReplica(_ body: Data) -> DecodedFramesReplica? {
    guard let obj = try? JSONSerialization.jsonObject(with: body) as? [String: Any],
          let format = obj["format"] as? String, format == "rgb8",
          let frames = obj["frames"] as? Int,
          let height = obj["height"] as? Int,
          let width = obj["width"] as? Int,
          let b64 = obj["data"] as? String,
          let rgb = Data(base64Encoded: b64),
          rgb.count == frames * height * width * 3
    else { return nil }
    let fps = (obj["fps"] as? Int) ?? 24
    return DecodedFramesReplica(rgb: rgb, frames: frames, height: height, width: width, fps: fps)
}

final class MediaGenTests: XCTestCase {

    // MARK: - Native video wire format

    func testDecodeFramesParsesValidBody() {
        let rgb = Data([10, 20, 30, 40, 50, 60]) // 1 frame, 1x2, 3ch
        let b64 = rgb.base64EncodedString()
        let body = "{\"frames\":1,\"height\":1,\"width\":2,\"fps\":24,\"format\":\"rgb8\",\"data\":\"\(b64)\"}".data(using: .utf8)!
        let d = decodeFramesReplica(body)
        XCTAssertEqual(d, DecodedFramesReplica(rgb: rgb, frames: 1, height: 1, width: 2, fps: 24))
    }

    func testDecodeFramesRejectsLengthMismatchAndWrongFormat() {
        // declared 1x2x3=6 bytes but only 3 provided
        let short = "{\"frames\":1,\"height\":1,\"width\":2,\"fps\":24,\"format\":\"rgb8\",\"data\":\"\(Data([1,2,3]).base64EncodedString())\"}".data(using: .utf8)!
        XCTAssertNil(decodeFramesReplica(short))
        let wrongFmt = "{\"frames\":1,\"height\":1,\"width\":2,\"fps\":24,\"format\":\"png\",\"data\":\"\(Data([1,2,3,4,5,6]).base64EncodedString())\"}".data(using: .utf8)!
        XCTAssertNil(decodeFramesReplica(wrongFmt))
    }

    // MARK: - Quality preset

    func testQualityPresetExposesFourTiers() {
        XCTAssertEqual(
            QualityPresetReplica.allCases.map(\.rawValue),
            ["Fast", "Good", "Quality", "Super Quality"]
        )
    }

    // MARK: - LTX frame ladder

    /// Every value in the ladder must satisfy `(n - 1) % 8 == 0` per LTX's
    /// temporal patch requirement. Off-by-one would produce a non-obvious
    /// runtime error from the pipeline.
    func testFrameLadderIsAll8NPlus1() {
        for cap in [49, 97, 161, 193] {
            for f in frameLadder(maxFrames: cap) {
                XCTAssertEqual((f - 1) % 8, 0, "Frame count \(f) is not 8N+1")
            }
        }
    }

    func testFrameLadderRespectsCap() {
        let ladder = frameLadder(maxFrames: 193)
        XCTAssertEqual(ladder.first, 9)
        XCTAssertTrue(ladder.allSatisfy { $0 <= 193 })
        XCTAssertTrue(ladder.contains(193))
    }

    func testFrameLadderIsMonotonicallyIncreasing() {
        let ladder = frameLadder(maxFrames: 193)
        XCTAssertEqual(ladder, ladder.sorted())
        XCTAssertEqual(Set(ladder).count, ladder.count, "Ladder should have no duplicates")
    }

    // MARK: - RAM cap

    /// Plenty of RAM headroom should let the user pick the model's max.
    /// (LTX 2.3 Q4 defaults: 24 GB model, 193-frame cap.)
    func testSafeFrameCapWithAmpleRAM() {
        let cap = safeFrameCap(modelRAMGB: 24, modelMaxFrames: 193, width: 704, height: 480, available: 128)
        XCTAssertEqual(cap, 193, "With ample RAM the cap should hit the model max")
    }

    /// Just-enough RAM should still allow at least the smallest ladder
    /// step — the gate doesn't refuse to generate altogether.
    func testSafeFrameCapWithMinimalRAM() {
        let cap = safeFrameCap(modelRAMGB: 24, modelMaxFrames: 193, width: 704, height: 480, available: 25)
        XCTAssertGreaterThanOrEqual(cap, 9, "Even tight RAM must allow the smallest ladder step")
    }

    /// Going to a higher resolution should not increase the cap at the
    /// same available RAM — pixel cost is in the formula.
    func testSafeFrameCapDecreasesWithResolution() {
        let lowRes = safeFrameCap(modelRAMGB: 24, modelMaxFrames: 193, width: 480, height: 704, available: 48)
        let highRes = safeFrameCap(modelRAMGB: 24, modelMaxFrames: 193, width: 768, height: 512, available: 48)
        XCTAssertGreaterThanOrEqual(lowRes, highRes)
    }

    // MARK: - Pipeline mode invariants

    /// Every canonical quality preset in the app's catalog must resolve to a
    /// known ltx-2-mlx pipeline mode, and its numFrames must satisfy the
    /// 8N+1 LTX ladder invariant. Drifting from either silently breaks
    /// generation at the Python boundary.
    func testEveryQualityModeIsKnown() {
        // Replica of the `VideoModelPreset.ltx23Q4` profile table — kept in
        // sync with MediaGen.swift. If these diverge, the canonical source
        // in MediaGen.swift is the one we actually ship; update it here.
        let profiles: [(mode: String, numFrames: Int)] = [
            ("oneStage",   49),
            ("oneStage",   97),
            ("twoStage",   97),
            ("twoStageHQ", 97),
        ]
        let validModes: Set<String> = ["oneStage", "twoStage", "twoStageHQ"]
        for p in profiles {
            XCTAssertTrue(validModes.contains(p.mode), "Unknown pipeline mode: \(p.mode)")
            XCTAssertEqual((p.numFrames - 1) % 8, 0,
                           "Frame count \(p.numFrames) for mode \(p.mode) is not 8N+1")
        }
    }

    // MARK: - GenEvent wire contract

    func testGenEventProgressParse() {
        let line = #"{"type":"progress","step":3,"total":10,"message":"Step 3/10"}"#
        XCTAssertEqual(
            GenEventReplica.parse(line),
            .progress(step: 3, total: 10, message: "Step 3/10")
        )
    }

    func testGenEventCompleteParse() {
        let line = #"{"type":"complete","path":"/tmp/out.png"}"#
        XCTAssertEqual(GenEventReplica.parse(line), .complete(path: "/tmp/out.png"))
    }

    func testGenEventErrorBecomesLog() {
        let line = #"{"type":"error","message":"torch broke"}"#
        guard case let .log(msg) = GenEventReplica.parse(line)! else {
            XCTFail("Expected .log for error event"); return
        }
        XCTAssertTrue(msg.hasPrefix("ERROR: "), "log line should be prefixed: \(msg)")
        XCTAssertTrue(msg.contains("torch broke"))
    }

    func testGenEventReturnsNilForNonJSON() {
        XCTAssertNil(GenEventReplica.parse("not json"))
        XCTAssertNil(GenEventReplica.parse("{}"))
        XCTAssertNil(GenEventReplica.parse(#"{"type":"unknown"}"#))
        XCTAssertNil(GenEventReplica.parse(#"{"type":"complete"}"#))
    }

    // MARK: - frame_rate / fps plumbing

    /// Replica of the arg vector `VideoGenService.buildArgs` produces (fps
    /// portion). ltx-2-mlx's `generate_and_save` makes `frame_rate` a required
    /// keyword arg, so the CLI args MUST carry an fps the script forwards —
    /// dropping it is a hard TypeError the moment generation starts.
    private func videoBuildArgsReplica(fps: Int, numFrames: Int, steps: Int) -> [String] {
        [
            "--frames", String(numFrames),
            "--fps", String(fps),
            "--steps", String(steps),
        ]
    }

    func testBuildArgsForwardsFps() {
        let args = videoBuildArgsReplica(fps: 24, numFrames: 97, steps: 8)
        guard let i = args.firstIndex(of: "--fps") else {
            return XCTFail("video args must include --fps so the script can pass the mandatory frame_rate")
        }
        XCTAssertEqual(args[args.index(after: i)], "24",
            "fps value must follow the --fps flag")
    }

    // MARK: - venv import-probe (ltx symbol drift)

    /// Regression for the bug report: a venv whose `ltx_pipelines_mlx` module
    /// imports fine but no longer exports `TextToVideoPipeline` (upstream
    /// drift) must be detected as NOT ready, so the UI re-offers the installer
    /// instead of letting generation fail at runtime. Runs the real probe
    /// statement against a throwaway package with the symbol missing.
    func testProbeDetectsMissingPipelineSymbol() throws {
        guard let py = Self.findPython3() else { throw XCTSkip("python3 not available") }
        let dir = try Self.makeFakeLtxPackage(symbols: [])
        defer { try? FileManager.default.removeItem(atPath: dir) }
        let status = Self.runPythonProbe(py, script: ltxProbeStatementReplica(), pythonPath: dir)
        XCTAssertNotEqual(status, 0,
            "A venv whose ltx_pipelines_mlx lacks TextToVideoPipeline must probe as NOT ready")
    }

    /// The positive case: when every pipeline symbol is exported, the probe
    /// passes and the venv counts as ready.
    func testProbePassesWhenPipelineSymbolsPresent() throws {
        guard let py = Self.findPython3() else { throw XCTSkip("python3 not available") }
        let dir = try Self.makeFakeLtxPackage(symbols: ltxPipelineSymbolsReplica)
        defer { try? FileManager.default.removeItem(atPath: dir) }
        let status = Self.runPythonProbe(py, script: ltxProbeStatementReplica(), pythonPath: dir)
        XCTAssertEqual(status, 0, "A venv exporting all pipeline symbols must probe as ready")
    }

    /// The full probe must check ltx at the symbol level, as a newline-joined
    /// multi-statement program — not the old single comma `import` that masked
    /// the drift.
    func testImportProbeIsSymbolLevelForLtx() {
        let script = importProbeScriptReplica()
        XCTAssertTrue(script.contains("from ltx_pipelines_mlx import \(ltxPipelineSymbolsReplica.first!)"),
            "Probe must verify the pipeline symbols, not just that the module loads:\n\(script)")
        XCTAssertTrue(script.contains("\n"),
            "Probe must be a newline-joined multi-statement program")
        XCTAssertFalse(script.contains("import mflux, "),
            "Probe should not collapse modules into a single comma import")
    }

    /// The runtime-recovery predicate: a failed required import (the bug's
    /// "ltx-2-mlx import failed: … Re-run the installer." message) must trigger
    /// re-detection; an ordinary per-request error must not.
    func testReinstallPredicateMatchesImportFailures() {
        XCTAssertTrue(indicatesVenvNeedsReinstallReplica(
            "ltx-2-mlx import failed: cannot import name 'TextToVideoPipeline'. Re-run the installer."))
        XCTAssertTrue(indicatesVenvNeedsReinstallReplica("mflux import failed: No module named 'mflux'"))
        XCTAssertFalse(indicatesVenvNeedsReinstallReplica("Model download failed: 404"))
        XCTAssertFalse(indicatesVenvNeedsReinstallReplica("ffmpeg not found on PATH."))
    }

    /// End-to-end against the real venv (skipped where none exists, e.g. CI):
    /// the symbols the app probes must actually be exported by the installed
    /// `ltx_pipelines_mlx`. This is what caught the 0.14.9 rename
    /// (TextToVideoPipeline → TI2VidOneStagePipeline) that made a freshly
    /// installed venv still fail the import check.
    func testRealVenvPassesFullImportProbe() throws {
        let venvPython = NSString(string: "~/.mlx-serve/venv/bin/python").expandingTildeInPath
        guard FileManager.default.isExecutableFile(atPath: venvPython) else {
            throw XCTSkip("no venv at ~/.mlx-serve/venv")
        }
        let status = Self.runPythonProbe(venvPython, script: importProbeScriptReplica(), pythonPath: "")
        XCTAssertEqual(status, 0,
            "The installed venv must pass the same import probe the app runs:\n\(importProbeScriptReplica())")
    }

    // MARK: - Probe test helpers

    private static func findPython3() -> String? {
        for p in ["/opt/homebrew/bin/python3", "/usr/local/bin/python3", "/usr/bin/python3"]
        where FileManager.default.isExecutableFile(atPath: p) { return p }
        return nil
    }

    /// Create a throwaway `ltx_pipelines_mlx` package exporting `symbols`;
    /// returns the directory to place on PYTHONPATH.
    private static func makeFakeLtxPackage(symbols: [String]) throws -> String {
        let root = (NSTemporaryDirectory() as NSString)
            .appendingPathComponent("ltxprobe-" + UUID().uuidString)
        let pkg = (root as NSString).appendingPathComponent("ltx_pipelines_mlx")
        try FileManager.default.createDirectory(atPath: pkg, withIntermediateDirectories: true)
        let body = symbols.map { "\($0) = object" }.joined(separator: "\n") + "\n"
        try body.write(toFile: (pkg as NSString).appendingPathComponent("__init__.py"),
                       atomically: true, encoding: .utf8)
        return root
    }

    private static func runPythonProbe(_ python: String, script: String, pythonPath: String) -> Int32 {
        let proc = Process()
        proc.executableURL = URL(fileURLWithPath: python)
        proc.arguments = ["-c", script]
        var env = ProcessInfo.processInfo.environment
        env["PYTHONPATH"] = pythonPath
        proc.environment = env
        proc.standardOutput = Pipe()
        proc.standardError = Pipe()
        do { try proc.run() } catch { return -1 }
        proc.waitUntilExit()
        return proc.terminationStatus
    }
}
