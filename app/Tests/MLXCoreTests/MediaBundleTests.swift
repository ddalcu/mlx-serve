import XCTest
@testable import MLXCore

/// Media-model download bundles: pull ONLY the files each engine reads (FLUX
/// weight subdirs, TTS speech_tokenizer, LTX's 3 safetensors — not its ~50 GB
/// of LoRAs/upscalers), and group cross-model dependencies (LTX → Gemma-3-12B).
final class MediaBundleTests: XCTestCase {

    // MARK: - File selection

    func testLtxAllowlistKeepsOnlyEngineSafetensors() {
        let entries: [[String: Any]] = [
            ["path": "config.json", "type": "file", "size": 4000],
            ["path": "embedded_config.json", "type": "file", "size": 8000],
            ["path": "transformer-dev.safetensors", "type": "file", "size": 11_000_000_000],
            ["path": "connector.safetensors", "type": "file", "size": 5_900_000_000],
            ["path": "vae_decoder.safetensors", "type": "file", "size": 777_000_000],
            // The ~50 GB we must NOT pull:
            ["path": "ltx-2.3-22b-distilled-lora-384.safetensors", "type": "file", "size": 7_100_000_000],
            ["path": "transformer-distilled.safetensors", "type": "file", "size": 11_000_000_000],
            ["path": "spatial_upscaler_x1_5_v1_0.safetensors", "type": "file", "size": 1_000_000_000],
            ["path": "vae_encoder.safetensors", "type": "file", "size": 608_000_000],
            ["path": "README.md", "type": "file", "size": 100],
        ]
        let sel = FileSelection(keepSafetensors: [
            "transformer-dev.safetensors", "connector.safetensors", "vae_decoder.safetensors",
        ])
        let picked = Set(DownloadManager.selectNeededFiles(from: entries, selection: sel).map(\.0))
        // Keeps config jsons + exactly the 3 engine safetensors.
        XCTAssertTrue(picked.contains("config.json"))
        XCTAssertTrue(picked.contains("embedded_config.json"))
        XCTAssertTrue(picked.contains("transformer-dev.safetensors"))
        XCTAssertTrue(picked.contains("connector.safetensors"))
        XCTAssertTrue(picked.contains("vae_decoder.safetensors"))
        XCTAssertEqual(picked.filter { $0.hasSuffix(".safetensors") }.count, 3)
        XCTAssertFalse(picked.contains("ltx-2.3-22b-distilled-lora-384.safetensors"))
        XCTAssertFalse(picked.contains("transformer-distilled.safetensors"))
        XCTAssertFalse(picked.contains("spatial_upscaler_x1_5_v1_0.safetensors"))
        XCTAssertFalse(picked.contains("README.md"))
    }

    func testRecursiveKeepsWeightSubdirsThatChatDefaultDrops() {
        let entries: [[String: Any]] = [
            ["path": "config.json", "type": "file", "size": 4000],
            ["path": "transformer/0.safetensors", "type": "file", "size": 5_000_000_000],
            ["path": "transformer/model.safetensors.index.json", "type": "file", "size": 4000],
            ["path": "vae/0.safetensors", "type": "file", "size": 300_000_000],
            ["path": "text_encoder/0.safetensors", "type": "file", "size": 4_000_000_000],
            ["path": "tokenizer/tokenizer.json", "type": "file", "size": 4000],
            ["path": "tokenizer/chat_template.jinja", "type": "file", "size": 4000],
            ["path": "README.md", "type": "file", "size": 100],
        ]
        let recursive = Set(DownloadManager.selectNeededFiles(from: entries, selection: FileSelection(recursive: true)).map(\.0))
        XCTAssertTrue(recursive.contains("transformer/0.safetensors"))
        XCTAssertTrue(recursive.contains("vae/0.safetensors"))
        XCTAssertTrue(recursive.contains("text_encoder/0.safetensors"))
        XCTAssertTrue(recursive.contains("tokenizer/tokenizer.json"))
        XCTAssertTrue(recursive.contains("tokenizer/chat_template.jinja"))
        XCTAssertFalse(recursive.contains("README.md"))
        // The chat default (top-level + mtp/ only) would MISS the FLUX subdirs —
        // exactly the bug that made app-side FLUX downloads unloadable.
        let chat = Set(DownloadManager.selectNeededFiles(from: entries).map(\.0))
        XCTAssertTrue(chat.contains("config.json"))
        XCTAssertFalse(chat.contains("transformer/0.safetensors"))
    }

    func testChatDefaultUnchangedTopLevelPlusMtp() {
        let entries: [[String: Any]] = [
            ["path": "config.json", "type": "file", "size": 4000],
            ["path": "model.safetensors", "type": "file", "size": 5_000_000_000],
            ["path": "mtp/weights.safetensors", "type": "file", "size": 100_000_000],
            ["path": "original/model.safetensors", "type": "file", "size": 5_000_000_000],
        ]
        let picked = Set(DownloadManager.selectNeededFiles(from: entries).map(\.0))
        XCTAssertTrue(picked.contains("config.json"))
        XCTAssertTrue(picked.contains("model.safetensors"))
        XCTAssertTrue(picked.contains("mtp/weights.safetensors"))   // the one nested exception
        XCTAssertFalse(picked.contains("original/model.safetensors"))
    }

    // MARK: - Readiness

    func testComponentReadyNeedsMarkersAndSafetensors() throws {
        let fm = FileManager.default
        let root = NSTemporaryDirectory() + "mediatest-\(UUID().uuidString)"
        let modelDir = (root as NSString).appendingPathComponent("author/name")
        try fm.createDirectory(atPath: modelDir, withIntermediateDirectories: true)
        defer { try? fm.removeItem(atPath: root) }

        let comp = MediaComponent(repo: "author/name", selection: .chatDefault, readyMarkers: ["config.json"])
        // Empty dir (no config.json) — existingModelDir won't even resolve it.
        XCTAssertFalse(DownloadManager.componentReady(comp, modelsRoot: root))
        // config.json present but NO weights → still not ready.
        fm.createFile(atPath: (modelDir as NSString).appendingPathComponent("config.json"), contents: Data("{}".utf8))
        XCTAssertFalse(DownloadManager.componentReady(comp, modelsRoot: root))
        // A safetensors lands → ready.
        fm.createFile(atPath: (modelDir as NSString).appendingPathComponent("model.safetensors"), contents: Data([0, 1, 2]))
        XCTAssertTrue(DownloadManager.componentReady(comp, modelsRoot: root))
    }

    // MARK: - Bundle mappings

    func testLtxBundleBundlesGemmaDependency() {
        let b = VideoModelPreset.ltx23Q4.bundle
        XCTAssertEqual(b.components.count, 2)
        XCTAssertEqual(b.primaryRepo, "dgrauet/ltx-2.3-mlx-q4")
        XCTAssertEqual(b.dependencyRepos, ["mlx-community/gemma-3-12b-it-4bit"])
        XCTAssertEqual(b.components[0].selection.keepSafetensors?.count, 3)   // allowlist
    }

    func testFluxAndTtsBundlesAreRecursiveSingleComponent() {
        let f = ImageModelPreset.flux2Klein4B_Q4.bundle
        XCTAssertEqual(f.components.count, 1)
        XCTAssertTrue(f.components[0].selection.recursive)
        XCTAssertTrue(f.components[0].readyMarkers.contains("transformer"))

        let t = AudioModelPreset.qwen3TTS06B.bundle
        XCTAssertEqual(t.components.count, 1)
        XCTAssertTrue(t.components[0].selection.recursive)
        XCTAssertTrue(t.components[0].readyMarkers.contains("speech_tokenizer"))
    }
}
