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

    /// Regression: Mage-Flow ships the diffusers layout — weight SUBDIRS and a
    /// `model_index.json` root, NO top-level `config.json`. `existingModelDir`
    /// only counted a dir as holding a model on `config.json` OR a `.gguf`, so a
    /// fully-downloaded Mage-Flow never resolved: the picker showed "Download"
    /// forever, and a click reverted instantly (files present → skip → re-check
    /// still false). Readiness must key on the diffusers root marker too.
    func testMageFlowDiffusersLayoutReadyWithoutRootConfig() throws {
        let fm = FileManager.default
        let root = NSTemporaryDirectory() + "mageflowtest-\(UUID().uuidString)"
        let comp = ImageModelPreset.mageFlowTurbo.bundle.components[0]
        let modelDir = (root as NSString).appendingPathComponent(comp.repo)
        try fm.createDirectory(atPath: modelDir, withIntermediateDirectories: true)
        defer { try? fm.removeItem(atPath: root) }

        // Empty dir → not ready.
        XCTAssertFalse(DownloadManager.componentReady(comp, modelsRoot: root))
        // The diffusers root marker + all four weight subdirs, plus a real
        // safetensors — exactly the on-disk shape, and crucially NO config.json.
        fm.createFile(atPath: (modelDir as NSString).appendingPathComponent("model_index.json"), contents: Data("{}".utf8))
        for sub in ["transformer", "vae", "text_encoder", "scheduler"] {
            try fm.createDirectory(atPath: (modelDir as NSString).appendingPathComponent(sub), withIntermediateDirectories: true)
        }
        fm.createFile(atPath: (modelDir as NSString).appendingPathComponent("transformer/diffusion_pytorch_model.safetensors"), contents: Data([0, 1, 2]))
        XCTAssertFalse(fm.fileExists(atPath: (modelDir as NSString).appendingPathComponent("config.json")),
                       "the fixture must have NO root config.json — that's the whole point")
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

    /// 2.5 ships its own text encoder, so its bundle must NOT carry the shared
    /// Gemma-3 chat model as a dependency (8 GB fetched for something the
    /// server never opens) — and it must reach INTO the pack for the encoder
    /// it does use, which needs a recursive fetch and a directory ready
    /// marker. Every one of those follows from `shipsOwnTextEncoder`, which is
    /// why the flag is what the bundle switches on.
    func testLtx25ShipsItsOwnEncoderAndPullsNoSharedGemma() {
        let b = VideoModelPreset.ltx25Q4.bundle
        XCTAssertTrue(VideoModelPreset.ltx25Q4.shipsOwnTextEncoder)
        XCTAssertEqual(b.components.count, 1)
        XCTAssertEqual(b.primaryRepo, "ddalcu/LTX-2.5-MLX-Serve-4bit")
        XCTAssertEqual(b.dependencyRepos, [])
        XCTAssertFalse(b.dependencyRepos.contains(MediaBundle.ltxGemmaRepo))
        XCTAssertTrue(b.components[0].selection.recursive)
        // The encoder's weights are `model.safetensors` — an allowlist without
        // it downloads a pack whose text path cannot load.
        XCTAssertEqual(b.components[0].selection.keepSafetensors?.contains("model.safetensors"), true)
        XCTAssertTrue(b.components[0].readyMarkers.contains(MediaBundle.ltx25TextEncoderDir))

        // 2.3 keeps the shared encoder — the flag is what separates them, not
        // the backend (both are `.ltx` and share the whole request surface).
        XCTAssertFalse(VideoModelPreset.ltx23Q4.shipsOwnTextEncoder)
        XCTAssertEqual(VideoModelPreset.ltx23Q4.bundle.dependencyRepos, [MediaBundle.ltxGemmaRepo])
        XCTAssertEqual(VideoModelPreset.ltx25Q4.backend, VideoModelPreset.ltx23Q4.backend)

        // First run pulls nothing extra, so the two size figures agree. On 2.3
        // they must NOT — that difference is the shared encoder.
        XCTAssertEqual(VideoModelPreset.ltx25Q4.approxDownloadGB,
                       VideoModelPreset.ltx25Q4.approxFirstRunDownloadGB)
        XCTAssertNotEqual(VideoModelPreset.ltx23Q4.approxDownloadGB,
                          VideoModelPreset.ltx23Q4.approxFirstRunDownloadGB)
    }

    /// Every LTX pack that carries its own text encoder must ride the `ltx25`
    /// bundle, and every LTX bundle must pull BOTH transformer variants — the
    /// two-stage tiers need `dev` for stage 1 and `distilled` for the refine,
    /// and a pack missing one 400s at generate time after a 40-60 GB download.
    /// Written over `all` so a preset added later (an 8-bit pack, a 2.6) is
    /// covered the day it lands rather than the day someone remembers.
    func testEveryLtxPackPullsBothTransformersAndItsOwnEncoder() {
        var checked = 0
        for preset in VideoModelPreset.all where preset.backend == .ltx {
            checked += 1
            let files = Set(preset.bundle.components.flatMap { Array($0.selection.keepSafetensors ?? []) })
            XCTAssertTrue(files.contains("transformer-dev.safetensors"), "\(preset.id) never fetches the dev transformer")
            XCTAssertTrue(files.contains("transformer-distilled.safetensors"), "\(preset.id) never fetches the distilled transformer")
            if preset.shipsOwnTextEncoder {
                XCTAssertTrue(preset.bundle.id.hasPrefix("ltx25:"),
                              "\(preset.id) ships its own encoder but uses \(preset.bundle.id) — the shared Gemma-3 fetch is 8 GB it never opens")
                XCTAssertEqual(preset.bundle.components.count, 1, "\(preset.id) should have no second component")
                XCTAssertTrue(preset.bundle.components[0].readyMarkers.contains(MediaBundle.ltx25TextEncoderDir),
                              "\(preset.id) reads ready without its text encoder")
            }
        }
        XCTAssertGreaterThanOrEqual(checked, 2, "no LTX preset found — guard is vacuous")
    }

    /// LTX-2.5's own pipeline defaults denoise a 1920x1088 canvas; our ladder
    /// stopped at 768x512, which is 0.39 MP against the reference's 2.09 MP.
    /// That is the whole "looks softer than the published clips" gap before
    /// anything about quantization: a canvas the model was not asked to fill
    /// cannot be sharpened by steps, guidance or a wider quant.
    ///
    /// Pins the ladder REACHES the reference canvas rather than pinning the
    /// exact list — a future ladder may re-space its rungs, but dropping the
    /// top one silently puts every user back on a preview-sized render.
    func testTheLtxLadderReachesTheReferenceCanvas() {
        for preset in VideoModelPreset.all where preset.backend == .ltx {
            let best = preset.resolutions.map { $0.width * $0.height }.max() ?? 0
            XCTAssertGreaterThanOrEqual(
                best, 1920 * 1088,
                "\(preset.id) tops out at \(best) px — below LTX's own 1920x1088 default canvas")
            // And the rungs must be distinct enough to be worth offering: at
            // least four separate pixel counts, or the menu is decoration.
            let areas = Set(preset.resolutions.map { $0.width * $0.height })
            XCTAssertGreaterThanOrEqual(areas.count, 4, "\(preset.id) ladder has \(areas.count) distinct sizes")
        }
    }

    /// The one-stage tiers run the DISTILLED transformer, whose sigma table is
    /// fixed at 8 steps — the server clamps anything else and logs it. A tier
    /// asking for 12 is a dead knob: it reads as "more steps than Fast" in the
    /// pane's own hint while both tiers run the identical schedule.
    func testOneStageTiersAskForTheStepCountTheDistilledScheduleActuallyRuns() {
        for preset in VideoModelPreset.all where preset.backend == .ltx {
            for q in QualityPreset.allCases {
                let s = preset.settings(q)
                guard s.mode == .oneStage else { continue }
                XCTAssertEqual(s.steps, 8,
                               "\(preset.id) \(q.label): one-stage runs the fixed 8-step distilled table, tier asks \(s.steps)")
            }
        }
    }

    /// A two-stage tier denoises at HALF the requested size and upscales, so on
    /// a small canvas "Quality" is a 384x256 render — worse than the one-stage
    /// tier it sits below in the menu. The pane must be able to say so, which
    /// means the rule is a function, not a comment.
    func testTwoStageTierWarnsUntilTheCanvasIsBigEnoughToHalve() {
        // 768x512 halves to 384x256 — below the model's smallest offered rung.
        XCTAssertNotNil(VideoModelPreset.ltx25Q4.twoStageCanvasNote(width: 768, height: 512))
        XCTAssertNotNil(VideoModelPreset.ltx25Q4.twoStageCanvasNote(width: 1024, height: 576))
        // 1600x896 halves to 800x448, 1920x1088 to 960x544 — both at or above
        // the smallest canvas the picker offers, so the tier pays for itself.
        XCTAssertNil(VideoModelPreset.ltx25Q4.twoStageCanvasNote(width: 1600, height: 896))
        XCTAssertNil(VideoModelPreset.ltx25Q4.twoStageCanvasNote(width: 1920, height: 1088))
        // H3 has no two-stage pipeline, so it never carries the note.
        XCTAssertNil(VideoModelPreset.minimaxH3.twoStageCanvasNote(width: 768, height: 512))
    }

    /// The default canvas is a per-MAC decision: 1920x1088 is right on this
    /// 128 GB machine and unusable on a 16 GB one, and a single static default
    /// has to be sized for the smallest Mac — which is how everyone ended up
    /// rendering previews. Mirrors `RecommendedModelPick.starterPick`: a pure
    /// function of physical memory, so it is testable off-machine.
    func testDefaultCanvasScalesWithTheMacsMemory() {
        let p = VideoModelPreset.ltx25Q4
        // 16 GB cannot hold the 24 GB pack at all — it falls back to the
        // smallest rung rather than to a canvas it definitely cannot render.
        let small = p.recommendedResolution(totalGB: 16)
        let mid   = p.recommendedResolution(totalGB: 36)
        let big   = p.recommendedResolution(totalGB: 128)
        XCTAssertEqual(small, p.resolutions.min { $0.width * $0.height < $1.width * $1.height })
        XCTAssertLessThanOrEqual(small.width * small.height, 768 * 512,
                                 "16 GB Mac must not default to a canvas it cannot hold")
        XCTAssertGreaterThan(big.width * big.height, small.width * small.height,
                             "a 128 GB Mac defaults to the same canvas as a 16 GB one")
        XCTAssertGreaterThanOrEqual(mid.width * mid.height, small.width * small.height)
        // Every pick must be a rung the picker actually offers, or the menu
        // renders blank on first launch.
        for r in [small, mid, big] {
            XCTAssertTrue(p.resolutions.contains(r), "\(r.label) is not on the ladder")
        }
        // And the auto-pick stays inside what the frame ladder can serve: a
        // default nobody can render at the default length is not a default.
        let frames = p.settings(p.defaultQuality).numFrames
        for (gb, r) in [(36, mid), (128, big)] {
            XCTAssertGreaterThanOrEqual(
                RAMChecker.safeFrameCap(model: p, width: r.width, height: r.height, available: gb), frames,
                "\(gb) GB: default canvas \(r.label) cannot hold \(frames) frames")
        }
    }

    /// The whole RGB volume comes back as ONE base64 blob (the server
    /// base64s `frames.rgb` into the JSON body and the app decodes it in
    /// memory), so a frame count is only offerable if its payload is. At
    /// 1920x1088 a 193-frame clip is 1.2 GB of raw RGB — 1.6 GB base64, held
    /// twice on each side. The ladder must shorten as the canvas grows.
    ///
    /// Hard cap rather than the existing soft RAM warning: an over-budget
    /// pick does not run slowly, it hangs and then dies.
    func testTheFrameLadderShortensAsTheCanvasGrows() {
        let p = VideoModelPreset.ltx25Q4
        let small = p.frameOptions(width: 768, height: 512)
        let big = p.frameOptions(width: 1920, height: 1088)
        XCTAssertEqual(small.last, p.maxFrames, "768x512 is nowhere near the payload budget")
        XCTAssertLessThan(big.last ?? 0, small.last ?? 0, "1920x1088 must offer fewer frames than 768x512")
        for opts in [small, big] {
            XCTAssertFalse(opts.isEmpty)
            for n in opts { XCTAssertEqual((n - 1) % 8, 0, "\(n) is off LTX's 8N+1 ladder") }
        }
        // Every offered combination stays inside the budget it was cut for.
        for r in p.resolutions {
            guard let longest = p.frameOptions(width: r.width, height: r.height).last else {
                return XCTFail("\(r.label) offers no frame counts")
            }
            XCTAssertLessThanOrEqual(longest * r.width * r.height * 3, VideoModelPreset.maxFramePayloadBytes,
                                     "\(r.label) x \(longest)f exceeds the response-payload budget")
        }
        // A canvas so large nothing fits still offers the ladder's first rung
        // rather than an empty picker.
        XCTAssertFalse(p.frameOptions(width: 4096, height: 4096).isEmpty)
    }

    /// Every LTX resolution must survive every QUALITY TIER, and two of the
    /// four tiers run a two-stage pipeline whose stage 1 is HALF resolution —
    /// so the server needs both edges divisible by 64 (the latent grid is /32,
    /// halved). 704x480 and 480x704 are only /32, so picking Quality or Super
    /// Quality on them earned a 400 with no way to tell from the pane which
    /// combination was the bad one. The default was one of them.
    ///
    /// A resolution offered on a tier that refuses it is the dead-control
    /// class: the pane must not present a combination the server rejects.
    func testEveryLtxResolutionSurvivesTheTwoStagePipelines() {
        // The server gates on the PIPELINE, not the backend, so this asks every
        // video preset the same question and only holds those that actually
        // offer a two-stage tier to the /64 rule. A future backend that adopts
        // two-stage is covered the day it does; H3 (one-stage only) is not
        // constrained by a rule that cannot apply to it.
        var checked = 0
        for preset in VideoModelPreset.all {
            let twoStageTiers = QualityPreset.allCases.filter {
                preset.settings($0).mode != .oneStage
            }
            guard !twoStageTiers.isEmpty else { continue }
            checked += 1
            // Every resolution offered must be legal for those tiers.
            for r in preset.resolutions {
                XCTAssertEqual(r.width % 64, 0,
                               "\(preset.id): \(r.label) width \(r.width) is not /64 — 400s on \(twoStageTiers.map(\.label))")
                XCTAssertEqual(r.height % 64, 0,
                               "\(preset.id): \(r.label) height \(r.height) is not /64 — 400s on \(twoStageTiers.map(\.label))")
            }
            XCTAssertEqual(preset.defaultResolution.width % 64, 0, "\(preset.id) default resolution is not /64")
            XCTAssertEqual(preset.defaultResolution.height % 64, 0, "\(preset.id) default resolution is not /64")
        }
        // Both LTX presets offer two-stage; a zero here means the loop went
        // vacuous and the guard stopped guarding anything.
        XCTAssertGreaterThanOrEqual(checked, 2, "no video preset offers a two-stage tier — guard is vacuous")
    }

    /// The published repo's ACTUAL file tree (ddalcu/LTX-2.5-MLX-Serve-4bit),
    /// run through the real bundle selection. A 2.5 pack is only useful if the
    /// download brings the whole engine AND the in-pack text encoder — and a
    /// download that quietly misses one file fails 36 GB later, at load, with
    /// a missing-weight error nobody can map back to the allowlist.
    func testLtx25BundlePullsEveryFileThePublishedRepoNeeds() {
        let entries: [[String: Any]] = [
            ["path": ".gitattributes", "type": "file", "size": 1674],
            ["path": "LICENSE.md", "type": "file", "size": 30938],
            ["path": "README.md", "type": "file", "size": 5000],
            ["path": "ltx-acceptable-use-policy-snapshot-2026-08-12.pdf", "type": "file", "size": 110423],
            ["path": "config.json", "type": "file", "size": 1039],
            ["path": "embedded_config.json", "type": "file", "size": 2529],
            ["path": "quantize_config.json", "type": "file", "size": 100],
            ["path": "split_model.json", "type": "file", "size": 412],
            ["path": "spatial_upscaler_x2_v1_1_config.json", "type": "file", "size": 275],
            ["path": "temporal_upscaler_x2_v1_0_config.json", "type": "file", "size": 273],
            ["path": "transformer-distilled.safetensors", "type": "file", "size": 11_320_068_903],
            ["path": "transformer-dev.safetensors", "type": "file", "size": 11_320_068_903],
            ["path": "connector.safetensors", "type": "file", "size": 6_344_495_770],
            ["path": "vae_decoder.safetensors", "type": "file", "size": 814_349_515],
            ["path": "vae_encoder.safetensors", "type": "file", "size": 637_885_335],
            ["path": "audio_vae.safetensors", "type": "file", "size": 106_509_020],
            ["path": "vocoder.safetensors", "type": "file", "size": 258_314_115],
            ["path": "spatial_upscaler_x2_v1_1.safetensors", "type": "file", "size": 995_745_061],
            ["path": "temporal_upscaler_x2_v1_0.safetensors", "type": "file", "size": 261_945_581],
            ["path": "gemma4-12b-ltx-v1/model.safetensors", "type": "file", "size": 6_699_162_168],
            ["path": "gemma4-12b-ltx-v1/config.json", "type": "file", "size": 4438],
            ["path": "gemma4-12b-ltx-v1/tokenizer.json", "type": "file", "size": 32_169_626],
            ["path": "gemma4-12b-ltx-v1/tokenizer_config.json", "type": "file", "size": 3736],
            ["path": "gemma4-12b-ltx-v1/generation_config.json", "type": "file", "size": 255],
            ["path": "gemma4-12b-ltx-v1/chat_template.jinja", "type": "file", "size": 18683],
            ["path": "gemma4-12b-ltx-v1/processor_config.json", "type": "file", "size": 1382],
        ]
        let sel = VideoModelPreset.ltx25Q4.bundle.components.first!.selection
        let picked = Set(DownloadManager.selectNeededFiles(from: entries, selection: sel).map(\.0))

        // Everything the LTX engine opens by name.
        for f in ["config.json", "transformer-distilled.safetensors", "transformer-dev.safetensors",
                  "connector.safetensors", "vae_decoder.safetensors", "vae_encoder.safetensors",
                  "audio_vae.safetensors", "vocoder.safetensors",
                  "spatial_upscaler_x2_v1_1.safetensors", "temporal_upscaler_x2_v1_0.safetensors"] {
            XCTAssertTrue(picked.contains(f), "engine file \(f) would not be downloaded")
        }
        // The in-pack text encoder: weights AND the tokenizer the server loads
        // beside them. Missing either is a pack that downloads and cannot encode.
        for f in ["gemma4-12b-ltx-v1/model.safetensors", "gemma4-12b-ltx-v1/config.json",
                  "gemma4-12b-ltx-v1/tokenizer.json", "gemma4-12b-ltx-v1/tokenizer_config.json"] {
            XCTAssertTrue(picked.contains(f), "text-encoder file \(f) would not be downloaded")
        }
        // Every ready marker must be satisfiable from what was picked, or the
        // pane offers Download forever on a complete install.
        for marker in VideoModelPreset.ltx25Q4.bundle.components.first!.readyMarkers {
            let satisfied = picked.contains(marker) || picked.contains { $0.hasPrefix(marker + "/") }
            XCTAssertTrue(satisfied, "ready marker \(marker) is never downloaded")
        }
        XCTAssertEqual(picked.filter { $0.hasSuffix(".safetensors") }.count, 10)
    }

    /// The encoder subdir is a three-way contract — the server resolves it,
    /// the bundle fetches it, the ready marker checks it — with no compiler
    /// between the Swift and Zig halves. Renaming it on one side makes every
    /// 2.5 pack fail to load with a message about a missing encoder, so the
    /// name is scanned out of the server source (same shape as the
    /// `turbo_lora.safetensors` guard).
    func testLtx25TextEncoderDirMatchesTheServersOwnConstant() throws {
        let root = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent().deletingLastPathComponent()
            .deletingLastPathComponent().deletingLastPathComponent()
        let src = try String(contentsOf: root.appendingPathComponent("src/ltx_video.zig"), encoding: .utf8)
        XCTAssertTrue(src.contains("\"\(MediaBundle.ltx25TextEncoderDir)\""),
                      "src/ltx_video.zig does not name \(MediaBundle.ltx25TextEncoderDir) — the app would fetch an encoder the server never looks for")
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

    /// Kokoro's bundle is NOT the TTS bundle. Reusing `tts(...)` demanded a
    /// `speech_tokenizer/` Kokoro does not have, so a fully downloaded Kokoro
    /// read as permanently incomplete — the pane would offer Download forever
    /// and the voice would never enable. The dictionaries live in `g2p/`, so
    /// the pull has to be recursive too.
    func testKokoroBundleIsNotTheTtsBundle() {
        let b = AudioModelPreset.kokoro82M.bundle
        XCTAssertEqual(b.components.count, 1)
        XCTAssertTrue(b.components[0].selection.recursive, "g2p/ is a subdir — a shallow pull misses the phonemizer")
        let m = b.components[0].readyMarkers
        for marker in ["config.json", "model.safetensors", "voices.safetensors", "g2p"] {
            XCTAssertTrue(m.contains(marker), "missing readyMarker \(marker)")
        }
        XCTAssertFalse(m.contains("speech_tokenizer"),
                       "Kokoro has no speech_tokenizer — requiring it makes a complete download read as incomplete forever")
        XCTAssertEqual(b.primaryRepo, "ddalcu/Kokoro-82M-MLX-Serve")
    }

    /// Class guard: the `.audio` slot hosts two architectures with different
    /// repo shapes, and `supportsCloning` is the discriminator. A third audio
    /// backend must not silently inherit the wrong bundle.
    func testAudioBundleDispatchFollowsTheDeclaredCapability() {
        for p in AudioModelPreset.allIncludingVoiceOnly {
            let m = p.bundle.components[0].readyMarkers
            if p.supportsCloning {
                XCTAssertTrue(m.contains("speech_tokenizer"), "\(p.id) clones — it needs the codec tokenizer")
            } else {
                XCTAssertTrue(m.contains("g2p"), "\(p.id) can't clone — it needs the phonemizer dictionaries")
            }
        }
    }

    /// Kokoro is voice-mode only. `.all` is what the MEDIA panes offer, and
    /// both AudioGenView's reference-clip control and VideoGenView's "Speak
    /// text" composer send `ref_audio` — which Kokoro answers with a named 400.
    /// Keeping it out of `.all` makes that unreachable by construction; a
    /// future "tidy up the catalog" that re-adds it would only show up as a
    /// runtime 400 in the Video pane.
    func testKokoroIsAbsentFromTheMediaGenCatalog() {
        XCTAssertFalse(AudioModelPreset.all.contains { $0.id == AudioModelPreset.kokoro82M.id })
        XCTAssertTrue(AudioModelPreset.all.allSatisfy(\.supportsCloning),
                      "every preset a media pane can pick must accept ref_audio")
        XCTAssertTrue(AudioModelPreset.allIncludingVoiceOnly.contains { $0.id == AudioModelPreset.kokoro82M.id },
                      "still downloadable from the model browser")
    }

    /// The readiness contract the download bar reads, against a real temp dir:
    /// the complete Kokoro layout is ready, and dropping `g2p/` is not.
    func testKokoroReadinessNeedsTheG2pDictionaries() throws {
        let fm = FileManager.default
        let root = NSTemporaryDirectory() + "kokorotest-\(UUID().uuidString)"
        let modelDir = (root as NSString).appendingPathComponent("ddalcu/Kokoro-82M-MLX-Serve")
        try fm.createDirectory(atPath: modelDir, withIntermediateDirectories: true)
        defer { try? fm.removeItem(atPath: root) }

        let comp = AudioModelPreset.kokoro82M.bundle.components[0]
        XCTAssertFalse(DownloadManager.componentReady(comp, modelsRoot: root))

        func file(_ name: String) -> String { (modelDir as NSString).appendingPathComponent(name) }
        fm.createFile(atPath: file("config.json"), contents: Data("{}".utf8))
        fm.createFile(atPath: file("model.safetensors"), contents: Data([0, 1, 2]))
        fm.createFile(atPath: file("voices.safetensors"), contents: Data([0, 1, 2]))
        try fm.createDirectory(atPath: file("g2p"), withIntermediateDirectories: true)
        XCTAssertTrue(DownloadManager.componentReady(comp, modelsRoot: root))

        // The phonemizer dictionaries are load-bearing — without them the
        // engine has no text→IPA path, so a g2p-less dir is NOT ready.
        try fm.removeItem(atPath: file("g2p"))
        XCTAssertFalse(DownloadManager.componentReady(comp, modelsRoot: root))
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

    func testMageFlowBundleIsSinglePublicRecursiveComponentDiffusersLayout() {
        let b = ImageModelPreset.mageFlowTurbo.bundle
        // One public component, no gated dependency, recursive (pulls the weight subdirs).
        XCTAssertEqual(b.components.count, 1)
        XCTAssertTrue(b.dependencyRepos.isEmpty)
        XCTAssertTrue(b.components[0].selection.recursive)
        // The README `assets/` images are junk — excluded from the pull.
        XCTAssertTrue(b.components[0].selection.excludeSubstrings.contains("assets/"))
        // Diffusers layout: the root marker is `model_index.json`, NOT config.json.
        let m = b.components[0].readyMarkers
        XCTAssertTrue(m.contains("model_index.json"))
        XCTAssertFalse(m.contains("config.json"))
        for marker in ["transformer", "vae", "text_encoder", "scheduler"] {
            XCTAssertTrue(m.contains(marker), "missing readyMarker \(marker)")
        }
    }

    func testMageFlowEditPresetIsEditCapableAndBundlesLikeTurbo() {
        let p = ImageModelPreset.mageFlowEditTurbo
        // The Edit checkpoint is the ONE image preset (besides FLUX) that can do
        // reference-image instruction edits — the picker lights up on this flag.
        XCTAssertTrue(p.supportsReferenceEdit)
        XCTAssertFalse(ImageModelPreset.mageFlowTurbo.supportsReferenceEdit)
        // Mage-Flow uses the final hidden state — no rebalance UI.
        XCTAssertEqual(p.condWeightCount, 0)
        XCTAssertTrue(ImageModelPreset.all.contains { $0.variant == .mageFlowEditTurbo })
        XCTAssertEqual(p.repo, "mage-flow-community/Mage-Flow-Edit-Turbo")
        // Same diffusers-layout bundle shape as Turbo.
        let b = p.bundle
        XCTAssertEqual(b.components.count, 1)
        XCTAssertTrue(b.dependencyRepos.isEmpty)
        XCTAssertTrue(b.components[0].selection.recursive)
        let m = b.components[0].readyMarkers
        XCTAssertTrue(m.contains("model_index.json"))
        XCTAssertFalse(m.contains("config.json"))
        for marker in ["transformer", "vae", "text_encoder", "scheduler"] {
            XCTAssertTrue(m.contains(marker), "missing readyMarker \(marker)")
        }
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

    /// Class guard: the image catalog is FLUX.2 + Krea only. FLUX.1
    /// (schnell/dev) was dropped as a picker option — a future preset must
    /// not reintroduce the `.flux1` variant tag or a "FLUX.1" name.
    func testImageCatalogHasNoFlux1Entries() {
        for p in ImageModelPreset.all {
            XCTAssertNotEqual(p.variant.rawValue, "flux1", p.id)
            XCTAssertFalse(p.name.contains("FLUX.1"), p.id)
        }
    }

    func testKreaPresetIsDistilledTurboDefaults() {
        let p = ImageModelPreset.krea2Turbo
        XCTAssertEqual(p.variant, .krea2Turbo)
        XCTAssertEqual(p.defaultQuality, .good)
        // Distilled Turbo: 8 steps (CFG isn't modelled — no image backend
        // reads a guidance field, so the tier carries a step count only).
        XCTAssertEqual(p.settings(.good).steps, 8)
        // Surfaced in the catalog so the picker shows it.
        XCTAssertTrue(ImageModelPreset.all.contains(p))
        // Resolutions are all multiples of 16 in [256, 2048] (the Krea size gate).
        for r in p.resolutions {
            XCTAssertEqual(r.width % 16, 0)
            XCTAssertEqual(r.height % 16, 0)
            XCTAssertTrue(r.width >= 256 && r.width <= 2048 && r.height >= 256 && r.height <= 2048)
        }
    }

    /// The only MLX build of FLUX.2-klein 9B (`mlx-community/flux2-klein-9b-4bit`)
    /// ships NO root config.json — same repo layout as the 4B minus that one
    /// file. Requiring it as a ready marker is the Kokoro-vs-tts bug exactly:
    /// a fully downloaded model that reads as permanently incomplete, so the
    /// pane offers Download forever and Generate never enables.
    func testKlein9BBundleDoesNotRequireARootConfigItNeverShips() {
        let nine = ImageModelPreset.flux2Klein9B_Q4.bundle
        XCTAssertEqual(nine.components.count, 1)
        XCTAssertEqual(nine.primaryRepo, "mlx-community/flux2-klein-9b-4bit")
        XCTAssertTrue(nine.components[0].selection.recursive)
        XCTAssertFalse(nine.components[0].readyMarkers.contains("config.json"))
        // The weight subdirs still have to be there — readiness stays real.
        for marker in ["transformer", "vae", "text_encoder", "tokenizer"] {
            XCTAssertTrue(nine.components[0].readyMarkers.contains(marker), "missing marker \(marker)")
        }
        // The 4B DOES ship one, and keeps checking for it.
        XCTAssertTrue(ImageModelPreset.flux2Klein4B_Q4.bundle.components[0].readyMarkers.contains("config.json"))
    }

    /// Third bite of the configless-repo class, and the one the marker test
    /// above CANNOT see: dropping `config.json` from the ready markers isn't
    /// enough, because `existingModelDir` never resolves the directory in the
    /// first place — it counted a dir as holding a model on `config.json`,
    /// `model_index.json` or a `.gguf`, and klein 9B ships NONE of the three.
    /// So a complete 8.9 GB download still read as absent: the pane offered
    /// "Download (~10 GB)" forever and a click flickered and reverted (files
    /// present → size-matched skip → instant finish → re-check still false).
    /// Readiness has to accept the weight-subdir shape itself.
    func testKlein9BOnDiskWithNoRootJsonAtAllReadsAsReady() throws {
        let fm = FileManager.default
        let root = NSTemporaryDirectory() + "klein9b-\(UUID().uuidString)"
        let modelDir = (root as NSString).appendingPathComponent("mlx-community/flux2-klein-9b-4bit")
        try fm.createDirectory(atPath: modelDir, withIntermediateDirectories: true)
        defer { try? fm.removeItem(atPath: root) }

        let comp = ImageModelPreset.flux2Klein9B_Q4.bundle.components[0]
        XCTAssertFalse(DownloadManager.componentReady(comp, modelsRoot: root))

        // The real on-disk shape: four weight subdirs, weights inside them, and
        // no root marker file of any kind.
        for sub in ["transformer", "vae", "text_encoder", "tokenizer"] {
            try fm.createDirectory(atPath: (modelDir as NSString).appendingPathComponent(sub), withIntermediateDirectories: true)
        }
        for w in ["transformer/0.safetensors", "vae/0.safetensors", "text_encoder/0.safetensors"] {
            fm.createFile(atPath: (modelDir as NSString).appendingPathComponent(w), contents: Data([0, 1, 2]))
        }
        for absent in ["config.json", "model_index.json"] {
            XCTAssertFalse(fm.fileExists(atPath: (modelDir as NSString).appendingPathComponent(absent)),
                           "the fixture must ship NO \(absent) — that's the whole point")
        }
        XCTAssertTrue(DownloadManager.componentReady(comp, modelsRoot: root))
    }

    /// The flip side: the weight-subdir shape must not make a HALF-downloaded
    /// repo read as present. A bare `transformer/` with nothing in it is a
    /// download that got as far as creating the folder.
    func testEmptyWeightSubdirsAreNotAModel() throws {
        let fm = FileManager.default
        let root = NSTemporaryDirectory() + "klein9bpartial-\(UUID().uuidString)"
        let modelDir = (root as NSString).appendingPathComponent("mlx-community/flux2-klein-9b-4bit")
        defer { try? fm.removeItem(atPath: root) }
        for sub in ["transformer", "vae"] {
            try fm.createDirectory(atPath: (modelDir as NSString).appendingPathComponent(sub), withIntermediateDirectories: true)
        }
        XCTAssertNil(DownloadManager.existingModelDir(rootDir: root, repoId: "mlx-community/flux2-klein-9b-4bit"))
        // An in-flight transfer writes `.partial`, which is not a weight either.
        fm.createFile(atPath: (modelDir as NSString).appendingPathComponent("transformer/0.safetensors.partial"), contents: Data([0]))
        XCTAssertNil(DownloadManager.existingModelDir(rootDir: root, repoId: "mlx-community/flux2-klein-9b-4bit"))
    }

    /// 9B is a bigger klein, not a different backend: it goes through the same
    /// FLUX engine, so it inherits the same capabilities — including reference
    /// editing, which is what a klein is for.
    func testKlein9BIsInTheCatalogWithFluxCapabilities() {
        let nine = ImageModelPreset.flux2Klein9B_Q4
        XCTAssertTrue(ImageModelPreset.all.contains(nine))
        XCTAssertEqual(nine.variant, .flux2Klein9B)
        XCTAssertTrue(nine.supportsReferenceEdit)
        XCTAssertTrue(nine.supportsImg2Img)
        XCTAssertTrue(nine.supportsLoRA)
        XCTAssertFalse(nine.stepsAreFixed)
        XCTAssertEqual(nine.condWeightCount, 3)   // FLUX concatenates 3 tapped encoder layers
        // Catalog order is cheapest → heaviest; 9B is heavier than the 4B and
        // lighter than Krea.
        let ids = ImageModelPreset.all.map(\.id)
        XCTAssertLessThan(ids.firstIndex(of: ImageModelPreset.flux2Klein4B_Q4.id)!,
                          ids.firstIndex(of: nine.id)!)
        XCTAssertLessThan(ids.firstIndex(of: nine.id)!,
                          ids.firstIndex(of: ImageModelPreset.krea2Turbo.id)!)
    }

    // MARK: - approxSizeLabel

    func testApproxSizeLabelRoundsWholeAtOneGBAndAbove() {
        XCTAssertEqual(ImageModelPreset.flux2Klein4B_Q4.bundle.approxSizeLabel, "~5 GB")
        XCTAssertEqual(ImageModelPreset.krea2Turbo.bundle.approxSizeLabel, "~15 GB")
        XCTAssertEqual(AudioModelPreset.qwen3TTS06B8bit.bundle.approxSizeLabel, "~2 GB")
    }

    /// A model whose bundle rounds to "~0 GB" below 1 GB would read as free —
    /// keep one decimal under the 1 GB floor. Built directly (no current
    /// catalog entry is this small) so the assertion can't silently stop
    /// exercising the branch if the catalog changes.
    func testApproxSizeLabelKeepsOneDecimalBelowOneGB() {
        let bundle = MediaBundle.tts(repo: "org/tiny-tts", displayName: "Tiny TTS", sizeGB: 0.4)
        XCTAssertEqual(bundle.approxSizeLabel, "~0.4 GB")
    }

    // MARK: - MediaModelPreset (Media tab generic surface)

    /// Every catalog the Media tab renders must be non-empty and free of
    /// duplicate ids — the tab is a `ForEach` over each catalog keyed by id.
    func testMediaCatalogsAreNonEmptyWithUniqueIds() {
        func assertUnique<P: MediaModelPreset>(_ presets: [P], _ label: String) {
            XCTAssertFalse(presets.isEmpty, label)
            let ids = presets.map(\.id)
            XCTAssertEqual(ids.count, Set(ids).count, "\(label) has a duplicate id")
        }
        assertUnique(ImageModelPreset.all, "image")
        assertUnique(AudioModelPreset.allIncludingVoiceOnly, "audio")
        assertUnique(VideoModelPreset.all, "video")
        assertUnique(MusicModelPreset.all, "music")
    }

    /// Every preset resolves to a real bundle with a primary repo and a
    /// non-empty size label — the two things the generic Media row renders.
    func testEveryMediaPresetResolvesADisplayableBundle() {
        func assertDisplayable<P: MediaModelPreset>(_ presets: [P], _ label: String) {
            for p in presets {
                XCTAssertFalse(p.bundle.primaryRepo.isEmpty, "\(label)/\(p.id)")
                XCTAssertFalse(p.bundle.approxSizeLabel.isEmpty, "\(label)/\(p.id)")
            }
        }
        assertDisplayable(ImageModelPreset.all, "image")
        assertDisplayable(AudioModelPreset.allIncludingVoiceOnly, "audio")
        assertDisplayable(VideoModelPreset.all, "video")
        assertDisplayable(MusicModelPreset.all, "music")
    }

    /// Every preset needs a real, non-stub plain-English description — the
    /// Media pane renders it under the model name, same idea as
    /// `RecommendedModelPick.blurb`.
    func testEveryMediaPresetHasNonEmptyDescription() {
        func assertDescribed<P: MediaModelPreset>(_ presets: [P], _ label: String) {
            for p in presets {
                XCTAssertGreaterThan(p.description.count, 20, "\(label)/\(p.id) description reads as a stub")
            }
        }
        assertDescribed(ImageModelPreset.all, "image")
        assertDescribed(AudioModelPreset.allIncludingVoiceOnly, "audio")
        assertDescribed(VideoModelPreset.all, "video")
        assertDescribed(MusicModelPreset.all, "music")
    }

    // MARK: - MediaModelPreset.meetsSystemRequirements

    /// `approxRAMGB` is already the full peak-footprint figure (not raw
    /// weight size), so the comparison needs no extra overhead multiplier —
    /// unlike `RecommendedModelPick.meetsSystemRequirements`.
    func testMediaPresetMeetsRequirementsWhenRamCoversTheFootprint() {
        XCTAssertTrue(ImageModelPreset.flux2Klein4B_Q4.meetsSystemRequirements(physicalMemoryBytes: 16 * 1_073_741_824))
    }

    func testMediaPresetDoesNotMeetRequirementsWhenTooBig() {
        // LTX: approxRAMGB 24 — an 8 GB Mac falls short.
        XCTAssertFalse(VideoModelPreset.ltx23Q4.meetsSystemRequirements(physicalMemoryBytes: 8 * 1_073_741_824))
    }

    /// The boundary is exact (`>=`), not a soft margin — a Mac with exactly
    /// the footprint's RAM counts as meeting it.
    func testMediaPresetRequirementsBoundaryIsInclusive() {
        let bytes = UInt64(ImageModelPreset.krea2Turbo.approxRAMGB) * 1_073_741_824
        XCTAssertTrue(ImageModelPreset.krea2Turbo.meetsSystemRequirements(physicalMemoryBytes: bytes))
    }
    func testMiniMaxH3FourBitPresetIsALowRAMAlternative() {
        let all = VideoModelPreset.all
        XCTAssertTrue(all.contains(where: { $0.id == VideoModelPreset.minimaxH3.id }))
        guard let q4 = all.first(where: { $0.repo == "ddalcu/MiniMax-H3-FL2VA-MLX-Serve-4bit" }) else {
            return XCTFail("no 4-bit H3 preset in VideoModelPreset.all")
        }
        XCTAssertNotEqual(q4.id, VideoModelPreset.minimaxH3.id)
        // The point of the pack: it fits small Macs. Staged residency peaks at
        // ~24.5 GB billed, so the guidance must sit well under the 8-bit's 44.
        XCTAssertLessThanOrEqual(q4.approxRAMGB, 28)
        XCTAssertLessThanOrEqual(q4.approxDownloadGB, 41)
        // Same engine, same recipe surface as the 8-bit preset.
        XCTAssertEqual(q4.backend, .minimaxH3)
        XCTAssertTrue(q4.supportsFastRecipe)
        XCTAssertTrue(q4.generatesAudio)
        XCTAssertTrue(q4.supportsLoRA)
        XCTAssertTrue(q4.supportsTurbo)
        // Bundle rides the SAME minimax factory keyed on the 4-bit repo.
        XCTAssertEqual(q4.bundle.id, "minimax-h3:ddalcu/MiniMax-H3-FL2VA-MLX-Serve-4bit")
    }

}
