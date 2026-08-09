import XCTest
@testable import MLXCore

/// Custom (user-added) media models: any checkpoint the server discovers in a
/// model root with a supported media arch shows up in its pane's picker as an
/// "On This Mac" row, synthesized from the matching family preset — same knobs
/// and capability declarations, its own id/repo. The source of truth is
/// `/v1/models` (`ServerManager.allModels`), the same list the LAN rows read.
final class CustomMediaModelsTests: XCTestCase {

    private func info(_ name: String, arch: String, caps: [String],
                      lan: String? = nil) -> ModelInfo {
        ModelInfo(name: name, quantBits: 4, layers: 0, hiddenSize: 0,
                  vocabSize: 0, contextLength: 0, modelMaxTokens: 0,
                  architecture: arch, capabilities: caps, lanPeer: lan)
    }

    // MARK: - Arch → family mapping

    /// Every media arch the server serves maps to the family preset whose
    /// capability declarations describe that backend — and nothing else does.
    /// Kokoro stays nil: the audio pane's catalog is cloning-capable ONLY
    /// (`ref_audio` on Kokoro is a named 400), and voice mode names its preset
    /// directly, so a custom Kokoro must not enter the pane through this door.
    func testEverySupportedMediaArchMapsToItsOwnFamilyPreset() {
        let models = [
            info("me/flux-finetune", arch: "flux2-klein-4b", caps: ["image"]),
            info("me/flux-big", arch: "flux2-klein-9b", caps: ["image"]),
            info("me/krea-custom", arch: "krea2_turbo", caps: ["image"]),
            info("me/Mage-Flow-Custom", arch: "mage_flow", caps: ["image"]),
            info("me/ltx-finetune", arch: "AudioVideo", caps: ["video"]),
            info("me/h3-requant", arch: "minimax_h3", caps: ["video"]),
            info("me/tts-finetune", arch: "qwen3_tts", caps: ["audio"]),
            info("me/kokoro-custom", arch: "kokoro", caps: ["audio"]),
            info("me/acestep-custom", arch: "acestep", caps: ["audio", "music"]),
            info("me/mesh-custom", arch: "hunyuan3d_2_1", caps: ["3d"]),
            info("me/chat-model", arch: "gemma4", caps: ["chat"]),
        ]

        let images = CustomMediaModels.imagePresets(from: models)
        XCTAssertEqual(images.map(\.id),
                       ["me/Mage-Flow-Custom", "me/flux-big", "me/flux-finetune", "me/krea-custom"])
        XCTAssertEqual(images.first { $0.id == "me/flux-finetune" }?.variant, .flux2Klein4B)
        XCTAssertEqual(images.first { $0.id == "me/flux-big" }?.variant, .flux2Klein9B)
        XCTAssertEqual(images.first { $0.id == "me/krea-custom" }?.variant, .krea2Turbo)
        XCTAssertEqual(images.first { $0.id == "me/Mage-Flow-Custom" }?.variant, .mageFlowTurbo)

        let videos = CustomMediaModels.videoPresets(from: models)
        XCTAssertEqual(videos.map(\.id), ["me/h3-requant", "me/ltx-finetune"])
        XCTAssertEqual(videos.first { $0.id == "me/ltx-finetune" }?.backend, .ltx)
        XCTAssertEqual(videos.first { $0.id == "me/h3-requant" }?.backend, .minimaxH3)
        // A custom H3 is treated as the FL2VA family: `tasks` (the ref2va
        // discriminator) is a config fact /v1/models doesn't carry, and
        // reference fields sent to an fl2va DiT are the 400 class.
        XCTAssertEqual(videos.first { $0.id == "me/h3-requant" }?.supportsReferences, false)

        // Audio pane: cloning-capable TTS only — no Kokoro, no ACE-Step (music
        // advertises "audio" too, but it answers a different endpoint).
        XCTAssertEqual(CustomMediaModels.audioPresets(from: models).map(\.id),
                       ["me/tts-finetune"])
        XCTAssertEqual(CustomMediaModels.musicPresets(from: models).map(\.id),
                       ["me/acestep-custom"])
        XCTAssertEqual(CustomMediaModels.meshPresets(from: models).map(\.id),
                       ["me/mesh-custom"])
    }

    /// The server gates Mage-Flow edit capability on the DIRECTORY NAME
    /// (`dirIsEdit`: "mage-flow-edit"/"mageflow-edit", case-insensitive) — the
    /// checkpoints are byte-identical, so the id is the only signal. The
    /// synthesized preset must mirror that or the pane offers edit controls
    /// the server won't honor (or hides ones it would).
    func testMageFlowEditIsKeyedOnTheDirNameLikeTheServer() {
        let models = [
            info("me/My-Mage-Flow-Edit-4bit", arch: "mage_flow", caps: ["image"]),
            info("me/mageflow-edit-tuned", arch: "mageflow", caps: ["image"]),
            info("me/mage-flow-plain", arch: "mage_flow", caps: ["image"]),
        ]
        let presets = CustomMediaModels.imagePresets(from: models)
        XCTAssertEqual(presets.first { $0.id == "me/My-Mage-Flow-Edit-4bit" }?.variant, .mageFlowEditTurbo)
        XCTAssertEqual(presets.first { $0.id == "me/mageflow-edit-tuned" }?.variant, .mageFlowEditTurbo)
        XCTAssertEqual(presets.first { $0.id == "me/mage-flow-plain" }?.variant, .mageFlowTurbo)
    }

    // MARK: - What gets a row

    /// Catalog repos never duplicate into custom rows (they're already preset
    /// rows), LAN entries never show under "On This Mac" (they have their own
    /// section), and the synthesized preset carries the family's knobs with
    /// its own id — which is also the repo, since a discovered id IS the
    /// on-disk `<org>/<name>` dir every resolver reads.
    func testCustomRowsAreLocalNonCatalogEntriesWithFamilyKnobs() {
        let models = [
            info(ImageModelPreset.krea2Turbo.repo, arch: "krea2_turbo", caps: ["image"]),
            info("me/krea-remix", arch: "krea2_turbo", caps: ["image"]),
            info("peer-flux@studio", arch: "flux2-klein-4b", caps: ["image"], lan: "studio"),
        ]
        let presets = CustomMediaModels.imagePresets(from: models)
        XCTAssertEqual(presets.map(\.id), ["me/krea-remix"])

        let custom = presets[0]
        XCTAssertEqual(custom.repo, "me/krea-remix")
        XCTAssertEqual(custom.name, "me/krea-remix")
        XCTAssertEqual(custom.resolutions, ImageModelPreset.krea2Turbo.resolutions)
        XCTAssertEqual(custom.settings(.good).steps, ImageModelPreset.krea2Turbo.settings(.good).steps)
        XCTAssertEqual(custom.approxRAMGB, ImageModelPreset.krea2Turbo.approxRAMGB)
    }

    /// End to end from the wire: a real `/v1/models` entry (verbatim server
    /// JSON shape, live 2026-08-09) parses into the ModelInfo the pickers
    /// read and produces its "On This Mac" row.
    func testServerListEntryParsesIntoAVideoRow() {
        let entry: [String: Any] = [
            "id": "antocorr/MiniMax-H3-FL2VA-MLX-Serve-2bit-text-encoder",
            "loaded": false, "state": "unloaded", "bytes_resident": 0,
            "capabilities": ["video"], "input_modalities": ["text"],
            "meta": ["architecture": "minimax_h3", "engine": "mlx",
                     "quantization": "4-bit", "context_length": 0],
        ]
        let m = APIClient.parseModelInfo(entry)
        XCTAssertEqual(m.architecture, "minimax_h3")
        XCTAssertNil(m.lanPeer)
        XCTAssertEqual(CustomMediaModels.videoPresets(from: [m]).map(\.id),
                       ["antocorr/MiniMax-H3-FL2VA-MLX-Serve-2bit-text-encoder"])
    }

    // MARK: - Settings resolution

    /// A persisted custom pick resolves against the live model list; with the
    /// list unavailable (server down) it falls back to the catalog default
    /// exactly like an unknown preset id always has.
    func testResolvedModelFindsCustomsAndFallsBackWhenAbsent() {
        var s = ImageGenSettings()
        s.modelId = "me/krea-remix"
        let models = [info("me/krea-remix", arch: "krea2_turbo", caps: ["image"])]
        XCTAssertEqual(s.resolvedModel(models: models).id, "me/krea-remix")
        XCTAssertEqual(s.resolvedModel(models: models).variant, .krea2Turbo)
        XCTAssertEqual(s.resolvedModel(models: []).id, ImageModelPreset.flux2Klein4B_Q4.id)
        // The parameterless var stays the catalog-only resolution.
        XCTAssertEqual(s.resolvedModel.id, ImageModelPreset.flux2Klein4B_Q4.id)
    }

    /// A LAN pick of a peer's CUSTOM model adopts the family preset by its
    /// base id — the pane gates canvases, frame ladders and request fields on
    /// the resolved preset, and keeping the previous pick's knobs is the
    /// H3-sent-LTX-shapes class (off-canvas sizes, frames below the trained
    /// floor).
    func testLanCustomVideoPickAdoptsItsFamilyPreset() {
        var s = VideoGenSettings()
        s.modelId = "lan:me/h3-requant@studio"
        let models = [info("me/h3-requant@studio", arch: "minimax_h3",
                           caps: ["video"], lan: "studio")]
        let resolved = s.resolvedModel(models: models)
        XCTAssertEqual(resolved.backend, .minimaxH3)
        XCTAssertEqual(resolved.frameOptions, VideoModelPreset.minimaxH3.frameOptions)
        // Unknown remote id with no matching entry: unchanged old behavior.
        XCTAssertEqual(s.resolvedModel(models: []).id, VideoModelPreset.ltx23Q4.id)
    }
}
