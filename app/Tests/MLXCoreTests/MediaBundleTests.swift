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
            ["path": "audio_vae.safetensors", "type": "file", "size": 106_000_000],
            ["path": "vocoder.safetensors", "type": "file", "size": 258_000_000],
            // VAE encoder (~0.6 GB) → image-to-video first-frame conditioning.
            ["path": "vae_encoder.safetensors", "type": "file", "size": 608_000_000],
            // Two-stage + proper one-stage pipelines (server-side):
            ["path": "transformer-distilled.safetensors", "type": "file", "size": 11_000_000_000],
            ["path": "spatial_upscaler_x2_v1_1.safetensors", "type": "file", "size": 1_000_000_000],
            // The rest of the ~50 GB we must NOT pull:
            ["path": "ltx-2.3-22b-distilled-lora-384.safetensors", "type": "file", "size": 7_100_000_000],
            ["path": "spatial_upscaler_x1_5_v1_0.safetensors", "type": "file", "size": 1_000_000_000],
            ["path": "README.md", "type": "file", "size": 100],
        ]
        // Use the REAL bundle's selection so the test can't drift from production.
        let sel = MediaBundle.ltx(repo: "owner/ltx", displayName: "LTX").components.first!.selection
        let picked = Set(DownloadManager.selectNeededFiles(from: entries, selection: sel).map(\.0))
        // Keeps config jsons + exactly the 8 engine safetensors (incl. audio VAE +
        // encoder + the two-stage distilled transformer + x2 upscaler).
        XCTAssertTrue(picked.contains("config.json"))
        XCTAssertTrue(picked.contains("embedded_config.json"))
        XCTAssertTrue(picked.contains("transformer-dev.safetensors"))
        XCTAssertTrue(picked.contains("connector.safetensors"))
        XCTAssertTrue(picked.contains("vae_decoder.safetensors"))
        XCTAssertTrue(picked.contains("audio_vae.safetensors"))  // sound: VAE
        XCTAssertTrue(picked.contains("vocoder.safetensors"))    // sound: vocoder
        XCTAssertTrue(picked.contains("vae_encoder.safetensors")) // image-to-video
        XCTAssertTrue(picked.contains("transformer-distilled.safetensors"))    // two-stage stage 2 / one-stage
        XCTAssertTrue(picked.contains("spatial_upscaler_x2_v1_1.safetensors")) // two-stage upscale
        XCTAssertEqual(picked.filter { $0.hasSuffix(".safetensors") }.count, 8)
        XCTAssertFalse(picked.contains("ltx-2.3-22b-distilled-lora-384.safetensors"))
        XCTAssertFalse(picked.contains("spatial_upscaler_x1_5_v1_0.safetensors"))
        XCTAssertFalse(picked.contains("README.md"))
    }

    func testLtxAudioFilesAreOptionalNotReadyMarkers() {
        // The audio VAE + vocoder are allowlisted (pulled when the repo ships
        // them) but must NOT gate readiness — a video-only checkpoint still
        // downloads + plays.
        let ltx = MediaBundle.ltx(repo: "owner/ltx", displayName: "LTX")
        let primary = ltx.components.first!
        // The audio VAE/vocoder, the VAE encoder (image-to-video), and the
        // two-stage weights (distilled transformer + x2 upscaler) are
        // allowlisted but optional — none gate readiness, so existing
        // dev-only installs keep working.
        for f in ["audio_vae.safetensors", "vocoder.safetensors", "vae_encoder.safetensors",
                  "transformer-distilled.safetensors", "spatial_upscaler_x2_v1_1.safetensors"] {
            XCTAssertTrue(primary.selection.keepSafetensors?.contains(f) ?? false,
                          "\(f) must be in the download allowlist")
            XCTAssertFalse(primary.readyMarkers.contains(f),
                           "\(f) must NOT be a ready marker (optional feature)")
        }
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
        XCTAssertEqual(b.components[0].selection.keepSafetensors?.count, 8)   // allowlist (incl. audio VAE + vocoder + image encoder + two-stage weights)
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

    func testDefaultTtsPresetIsEightBitAndBundleRecursive() {
        // 8-bit is the default voice model (smaller download, lower RAM); the
        // bf16 presets stay in the catalog as fidelity fallbacks.
        let d = AudioModelPreset.all.first
        XCTAssertEqual(d?.id, AudioModelPreset.qwen3TTS06B8bit.id)
        XCTAssertTrue(d?.repo.hasSuffix("-8bit") ?? false)
        // Same repo layout as bf16 (config + model + speech_tokenizer/) — the
        // recursive TTS bundle factory applies unchanged.
        let b = AudioModelPreset.qwen3TTS06B8bit.bundle
        XCTAssertEqual(b.components.count, 1)
        XCTAssertTrue(b.components[0].selection.recursive)
        XCTAssertTrue(b.components[0].readyMarkers.contains("speech_tokenizer"))
        XCTAssertEqual(AudioModelPreset.qwen3TTS17B8bit.repo, "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-8bit")
    }

    func testKreaBundleIsSinglePublicRecursiveComponent() {
        let k = ImageModelPreset.krea2Turbo.bundle
        // One public component, no gated dependency, recursive (pulls the weight subdirs).
        XCTAssertEqual(k.components.count, 1)
        XCTAssertTrue(k.dependencyRepos.isEmpty)
        XCTAssertTrue(k.components[0].selection.recursive)
        XCTAssertNil(k.components[0].selection.keepSafetensors)
        // Transformer is a TOP-LEVEL FILE (not a `transformer/` subdir like FLUX),
        // plus the three Qwen subdirs + config.
        let m = k.components[0].readyMarkers
        XCTAssertTrue(m.contains("transformer_mixed_4_8.safetensors"))
        XCTAssertFalse(m.contains("transformer"))
        for marker in ["config.json", "vae", "text_encoder", "tokenizer"] {
            XCTAssertTrue(m.contains(marker), "missing readyMarker \(marker)")
        }
    }

    func testNsfwClassifierProvisioningDefaults() {
        // Shared content-filter classifier: the original public Apache-2.0 repo.
        XCTAssertEqual(DownloadManager.nsfwClassifierRepo, "Falconsai/nsfw_image_detection")
        // Safe mode is ON by default on a generation request.
        let r = ImageGenRequest(model: .krea2Turbo, prompt: "x", width: 512, height: 512, steps: 8, guidance: 0)
        XCTAssertTrue(r.safeMode)
    }

    // MARK: - 3D (Hunyuan3D) bundle + local-repo readiness

    func testModel3DBundleIsOneRecursiveHFRepoWithAllStages() {
        let b = Model3DModelPreset.hunyuan3d21_8bit.bundle
        // ONE published HF repo carries all three stages (shape at the root,
        // paint/ + unirig/ subdirs) — a single download, no dependency repos.
        XCTAssertEqual(b.components.count, 1)
        XCTAssertTrue(b.dependencyRepos.isEmpty)
        XCTAssertEqual(b.primaryRepo, "ddalcu/Hunyuan3D-2.1-MLX-Serve-8bit")
        let comp = b.components[0]
        // Recursive so the paint/ + unirig/ subdirs ride the same pull.
        XCTAssertTrue(comp.selection.recursive)
        // Allowlist covers exactly the seven engine weights across the stages.
        XCTAssertEqual(comp.selection.keepSafetensors?.count, 7)
        for f in ["dit.safetensors", "conditioner.safetensors", "vae.safetensors",
                  "unet.safetensors", "unet_dual.safetensors", "dino.safetensors",
                  "skeleton.safetensors"] {
            XCTAssertTrue(comp.selection.keepSafetensors?.contains(f) ?? false, "missing allowlist \(f)")
        }
        // Ready markers span all three stages so a partial pull never reads
        // ready (texture/rig would 400 at request time).
        for marker in ["config.json", "dit.safetensors", "conditioner.safetensors", "vae.safetensors",
                       "paint/config.json", "paint/unet.safetensors", "paint/unet_dual.safetensors",
                       "paint/dino.safetensors", "paint/vae.safetensors",
                       "unirig/config.json", "unirig/skeleton.safetensors"] {
            XCTAssertTrue(comp.readyMarkers.contains(marker), "missing readyMarker \(marker)")
        }
    }

    func testModel3DReadinessRequiresAllThreeStages() throws {
        // A shape-only dir (partial download / the pre-combined local layout)
        // must NOT read as ready — texture/rig requests would 400.
        let fm = FileManager.default
        let root = NSTemporaryDirectory() + "hy3dtest-\(UUID().uuidString)"
        let modelDir = (root as NSString).appendingPathComponent("ddalcu/Hunyuan3D-2.1-MLX-Serve-8bit")
        try fm.createDirectory(atPath: modelDir, withIntermediateDirectories: true)
        defer { try? fm.removeItem(atPath: root) }

        let comp = Model3DModelPreset.hunyuan3d21_8bit.bundle.components[0]
        // Nothing present → not ready.
        XCTAssertFalse(DownloadManager.componentReady(comp, modelsRoot: root))
        // Shape stage alone → still not ready (paint/unirig markers missing).
        fm.createFile(atPath: (modelDir as NSString).appendingPathComponent("config.json"), contents: Data("{}".utf8))
        for f in ["dit.safetensors", "conditioner.safetensors", "vae.safetensors"] {
            fm.createFile(atPath: (modelDir as NSString).appendingPathComponent(f), contents: Data([0, 1, 2]))
        }
        XCTAssertFalse(DownloadManager.componentReady(comp, modelsRoot: root))
        // Paint stage lands → still waiting on unirig.
        try fm.createDirectory(atPath: (modelDir as NSString).appendingPathComponent("paint"), withIntermediateDirectories: true)
        for f in ["paint/config.json", "paint/unet.safetensors", "paint/unet_dual.safetensors",
                  "paint/dino.safetensors", "paint/vae.safetensors"] {
            fm.createFile(atPath: (modelDir as NSString).appendingPathComponent(f), contents: Data([0, 1, 2]))
        }
        XCTAssertFalse(DownloadManager.componentReady(comp, modelsRoot: root))
        // UniRig stage lands → ready.
        try fm.createDirectory(atPath: (modelDir as NSString).appendingPathComponent("unirig"), withIntermediateDirectories: true)
        for f in ["unirig/config.json", "unirig/skeleton.safetensors"] {
            fm.createFile(atPath: (modelDir as NSString).appendingPathComponent(f), contents: Data([0, 1, 2]))
        }
        XCTAssertTrue(DownloadManager.componentReady(comp, modelsRoot: root))
        // Removing one stage weight breaks readiness again.
        try fm.removeItem(atPath: (modelDir as NSString).appendingPathComponent("paint/unet.safetensors"))
        XCTAssertFalse(DownloadManager.componentReady(comp, modelsRoot: root))
    }

    func testHunyuanPresetDownloadsFromHF() {
        // Published combined repo — the pane shows the standard download bar,
        // not the "convert locally" hint.
        XCTAssertFalse(Model3DModelPreset.hunyuan3d21_8bit.isLocalOnly)
        XCTAssertTrue(Model3DModelPreset.all.contains(.hunyuan3d21_8bit))
    }

    func testKreaPresetIsDistilledTurboDefaults() {
        let p = ImageModelPreset.krea2Turbo
        XCTAssertEqual(p.variant, .krea2Turbo)
        XCTAssertEqual(p.defaultQuality, .good)
        // Distilled Turbo: 8 steps, no CFG.
        XCTAssertEqual(p.settings(.good).steps, 8)
        XCTAssertEqual(p.settings(.good).guidance, 0.0)
        // Surfaced in the catalog so the picker shows it.
        XCTAssertTrue(ImageModelPreset.all.contains(p))
        // Resolutions are all multiples of 16 in [256, 2048] (the Krea size gate).
        for r in p.resolutions {
            XCTAssertEqual(r.width % 16, 0)
            XCTAssertEqual(r.height % 16, 0)
            XCTAssertTrue(r.width >= 256 && r.width <= 2048 && r.height >= 256 && r.height <= 2048)
        }
    }
}
