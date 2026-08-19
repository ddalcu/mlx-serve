import XCTest
@testable import MLXCore

/// The app keeps its own copy of the server's media-architecture table, and
/// that copy drifted: `minimax_music3`, `kokoro` and bare `mageflow` were
/// missing, so a MiniMax Music 3 checkpoint on disk was labelled a red
/// "Unsupported" on the same screen that offers to download it (#228).
///
/// Zig already pins its two copies against each other (`gen.media_model_types`
/// vs `model_discovery.isMediaModelType`). Nothing pinned Swift, which is
/// exactly why the drift went unnoticed — so the guard reads the Zig array.
final class MediaModalityParityTests: XCTestCase {

    private var repoRoot: URL {
        URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()  // MLXCoreTests
            .deletingLastPathComponent()  // Tests
            .deletingLastPathComponent()  // app
            .deletingLastPathComponent()  // repo root
    }

    /// `media_model_types` out of src/gen.zig, which is the source of truth.
    private func zigMediaModelTypes() throws -> Set<String> {
        let text = try String(contentsOf: repoRoot.appendingPathComponent("src/gen.zig"), encoding: .utf8)
        guard let start = text.range(of: "pub const media_model_types = [_][]const u8{"),
              let end = text.range(of: "};", range: start.upperBound..<text.endIndex) else {
            XCTFail("could not find media_model_types in src/gen.zig")
            return []
        }
        let body = text[start.upperBound..<end.lowerBound]
        return Set(body.split(whereSeparator: { $0 == "," || $0 == "\n" })
            .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
            .filter { $0.hasPrefix("\"") }
            .map { $0.trimmingCharacters(in: CharacterSet(charactersIn: "\"")) })
    }

    func testEveryMediaArchitectureTheServerKnowsIsKnownToTheApp() throws {
        let zig = try zigMediaModelTypes()
        XCTAssertGreaterThanOrEqual(zig.count, 11, "parsed too few types — did the Zig array move?")
        for type in zig {
            // Prefix entries in Zig ("flux2", "krea", "mage_flow", "hunyuan3d")
            // are matched by prefix on both sides, so the bare stem must pass.
            XCTAssertTrue(isMediaModelType(type),
                          "\(type) is a media architecture on the server but reads as unsupported in the app")
        }
    }

    func testTheThreeThatDriftedAreCoveredByName() {
        // Named individually so a future prefix-vs-exact refactor cannot make
        // the loop above pass while these regress.
        XCTAssertTrue(isMediaModelType("minimax_music3"))
        XCTAssertTrue(isMediaModelType("kokoro"))
        XCTAssertTrue(isMediaModelType("mageflow"))
        XCTAssertTrue(isMediaModelType("mage_flow_turbo"))
    }

    func testChatArchitecturesStillDoNotReadAsMedia() {
        for t in ["qwen3", "llama", "gemma3", "bailing_hybrid", "bert", "minimax_text"] {
            XCTAssertFalse(isMediaModelType(t), "\(t) must not read as media")
        }
    }

    func testAMediaModelIsStillNeverChatPickable() {
        // Widening the media list also widens the chat-picker EXCLUSION
        // (`isChatPickable` reads it), which is the direction we want: today
        // minimax_music3 is excluded only by accident, because it fails the
        // architecture gate entirely. It must stay excluded for the right
        // reason — `useModelAndAwaitReady` would push a diffusion checkpoint
        // through the text loader.
        for t in ["minimax_music3", "kokoro", "acestep", "flux2-klein-4b", "qwen3_tts"] {
            let m = LocalModel(id: "test:\(t)", name: t, path: "/tmp/\(t)",
                               sizeFormatted: "1 GB", modelType: t, source: .mlxServe, kind: .base)
            XCTAssertFalse(m.isChatPickable, "\(t) must not be chat-pickable")
        }
    }
}

/// Which create pane a media checkpoint belongs to. Mirrors Zig's
/// `gen.modalityFromType`, which the app had no equivalent of — the browser
/// could tell that a model was media, but not what KIND, so it could not offer
/// to open it anywhere.
final class MediaModalityRoutingTests: XCTestCase {

    func testEveryMediaArchitectureRoutesToItsPane() {
        let cases: [(String, MediaModality)] = [
            ("flux2-klein-4b", .image), ("krea2_turbo", .image),
            ("mage_flow", .image), ("mageflow", .image),
            ("qwen3_tts", .voice), ("kokoro", .voice),
            ("acestep", .music), ("minimax_music3", .music),
            ("AudioVideo", .video), ("minimax_h3", .video),
            ("hunyuan3d_2_1", .mesh), ("hunyuan3d_2_1_paint", .mesh),
        ]
        for (type, want) in cases {
            XCTAssertEqual(MediaModality(modelType: type), want, "\(type)")
        }
    }

    func testANonMediaArchitectureHasNoModality() {
        for t in ["qwen3", "llama", "bert", ""] {
            XCTAssertNil(MediaModality(modelType: t), "\(t)")
        }
    }

    func testMusicAndVoiceSplitEvenThoughTheServerCallsThemBothAudio() {
        // Zig's modalityFromType collapses these to `.audio` because they share
        // one engine slot. The APP has to tell them apart: they are two tabs of
        // one pane, so a Use button that only knew "audio" would drop a music
        // model on the Voice tab.
        XCTAssertEqual(MediaModality(modelType: "acestep")?.experiment, .audio)
        XCTAssertEqual(MediaModality(modelType: "qwen3_tts")?.experiment, .audio)
        XCTAssertEqual(MediaModality(modelType: "acestep")?.audioTab, .music)
        XCTAssertEqual(MediaModality(modelType: "qwen3_tts")?.audioTab, .voice)
        // Non-audio modalities have no tab to pick.
        XCTAssertNil(MediaModality(modelType: "flux2-klein-4b")?.audioTab)
        XCTAssertEqual(MediaModality(modelType: "minimax_h3")?.experiment, .video)
        XCTAssertEqual(MediaModality(modelType: "hunyuan3d_2_1")?.experiment, .model3d)
    }

    func testEveryModalityNamesThePaneItOpens() {
        for m in MediaModality.allCases {
            XCTAssertFalse(m.paneName.isEmpty, "\(m) has no pane name for the Use button")
        }
    }
}

/// The media Use button must never reach the chat load path.
final class MediaUseButtonSourceTests: XCTestCase {

    func testTheMediaUseButtonNeverStartsTheServerOnTheModel() throws {
        // `useModelAndAwaitReady` → `server.start(modelPath:)` loads a path as
        // the server's PRIMARY CHAT model. Pointing that at a diffusion
        // checkpoint runs it through the text loader — the exact failure
        // `isChatPickable` exists to prevent, and the tempting one-line "fix"
        // for this issue. The media button opens a pane instead; the pane
        // loads the model the way it always has.
        let root = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent().deletingLastPathComponent()
            .deletingLastPathComponent().deletingLastPathComponent()
        let text = try String(
            contentsOf: root.appendingPathComponent("app/Sources/MLXServe/Views/ModelBrowserView.swift"),
            encoding: .utf8)
        guard let start = text.range(of: "private struct UseMediaModelButton"),
              let end = text.range(of: "\n}", range: start.upperBound..<text.endIndex) else {
            XCTFail("UseMediaModelButton not found — did it move?")
            return
        }
        let body = String(text[start.lowerBound..<end.upperBound])
        XCTAssertFalse(body.contains("useModelAndAwaitReady"),
                       "the media Use button must not load the model as the chat model")
        XCTAssertFalse(body.contains("server.start"),
                       "the media Use button must not start the server on the model path")
        XCTAssertTrue(body.contains("showCreate"),
                      "the media Use button should route through AppState.showCreate")
    }
}
