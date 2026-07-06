import Foundation

/// Which files of a HuggingFace repo a download should pull. Media models are
/// NOT the flat single-dir shape chat models are: FLUX ships weight SUBDIRS
/// (`transformer/`, `vae/`, …), TTS ships `speech_tokenizer/`, and LTX ships
/// ~50 GB of files (LoRAs, upscalers, alternate transformers) the engine never
/// reads. This lets each download pull EXACTLY what's needed — no more.
struct FileSelection: Equatable {
    /// Descend into subdirectories (FLUX/TTS). When false, only top-level files
    /// + the `mtp/` sidecar are kept (the chat-model default).
    var recursive: Bool = false
    /// Skip any file whose path contains one of these (belt-and-suspenders for
    /// junk a recursive scan would otherwise grab).
    var excludeSubstrings: [String] = []
    /// When non-nil, among `.safetensors` files keep ONLY these basenames. The
    /// LTX allowlist (`transformer-dev`/`connector`/`vae_decoder`) — skips the
    /// LoRAs, upscalers, and alternate transformers. Non-safetensors (json/txt)
    /// follow the normal extension rule.
    var keepSafetensors: Set<String>? = nil

    /// Chat-model default: top-level files + `mtp/`, all needed extensions.
    static let chatDefault = FileSelection()
}

/// One downloadable piece of a media bundle: a HF repo + how to pull it + how
/// to tell it's fully present on disk.
struct MediaComponent: Equatable {
    let repo: String
    let selection: FileSelection
    /// Relative paths (file OR dir) that must exist for this component to be
    /// "ready". Combined with a generic "has at least one .safetensors" check
    /// so a config-only partial download never reads as ready.
    let readyMarkers: [String]

    static func == (l: MediaComponent, r: MediaComponent) -> Bool { l.repo == r.repo }
}

/// A media model + its dependencies, downloaded as a unit. Today: FLUX/TTS are
/// single-component; LTX is `[ltx, gemma-3-12b]` (the text encoder, which is
/// also selectable as a chat model). Designed to grow — new bundles just add
/// components / a new factory.
struct MediaBundle: Identifiable, Equatable {
    let id: String
    let displayName: String
    /// Primary model first, then dependencies.
    let components: [MediaComponent]
    let sizeEstimateGB: Double

    var primaryRepo: String { components.first!.repo }
    var dependencyRepos: [String] { Array(components.dropFirst().map(\.repo)) }

    static func == (l: MediaBundle, r: MediaBundle) -> Bool { l.id == r.id }
}

// MARK: - Bundle factories (per modality)

extension MediaBundle {
    /// FLUX (mflux): one repo with weight subdirs (`transformer/`, `vae/`,
    /// `text_encoder/`, `tokenizer/`). Recursive download; ready when all four
    /// subdirs + config are present.
    static func flux(repo: String, displayName: String, sizeGB: Double) -> MediaBundle {
        MediaBundle(
            id: "flux:\(repo)",
            displayName: displayName,
            components: [
                MediaComponent(
                    repo: repo,
                    selection: FileSelection(recursive: true),
                    readyMarkers: ["config.json", "transformer", "vae", "text_encoder", "tokenizer"]
                ),
            ],
            sizeEstimateGB: sizeGB
        )
    }

    /// Qwen3-TTS: top-level model + `speech_tokenizer/` subdir (the codec
    /// decoder reads `<dir>/speech_tokenizer/`). Recursive download.
    static func tts(repo: String, displayName: String, sizeGB: Double) -> MediaBundle {
        MediaBundle(
            id: "tts:\(repo)",
            displayName: displayName,
            components: [
                MediaComponent(
                    repo: repo,
                    selection: FileSelection(recursive: true),
                    readyMarkers: ["config.json", "speech_tokenizer"]
                ),
            ],
            sizeEstimateGB: sizeGB
        )
    }

    /// LTX-Video: pull ONLY the safetensors the engine reads (allowlist) plus
    /// the small json configs — the repo also carries ~50 GB of LoRAs /
    /// upscalers / alternate transformers we never touch. Depends on the
    /// Gemma-3-12B text encoder (a normal chat model the app downloads).
    ///
    /// `audio_vae.safetensors` + `vocoder.safetensors` (the audio VAE + BigVGAN
    /// vocoder, ~0.37 GB together) are allowlisted so the generated video gets a
    /// SOUND track — the `dgrauet/ltx-2.3-mlx-q4` repo ships both. They're
    /// deliberately NOT ready markers: a checkpoint without them still completes
    /// and plays (silently). The server loads both from the model dir.
    static func ltx(repo: String, displayName: String) -> MediaBundle {
        MediaBundle(
            id: "ltx:\(repo)",
            displayName: displayName,
            components: [
                MediaComponent(
                    repo: repo,
                    selection: FileSelection(keepSafetensors: [
                        "transformer-dev.safetensors", "connector.safetensors", "vae_decoder.safetensors",
                        "audio_vae.safetensors", "vocoder.safetensors",
                        // VAE encoder (~0.6 GB) → image-to-video first-frame conditioning.
                        // Not a ready marker (like the audio files): I2V is optional.
                        "vae_encoder.safetensors",
                        // Two-stage + proper one-stage pipelines (~11 GB + ~1 GB):
                        // the distilled transformer + x2 spatial upscaler the server's
                        // `pipeline: two_stage[_hq]` modes read. Allowlisted like the
                        // VAE encoder — NOT ready markers, so existing dev-only
                        // installs keep working (readiness/gating unchanged).
                        "transformer-distilled.safetensors", "spatial_upscaler_x2_v1_1.safetensors",
                    ]),
                    readyMarkers: [
                        "config.json", "transformer-dev.safetensors",
                        "connector.safetensors", "vae_decoder.safetensors",
                    ]
                ),
                ltxGemmaComponent,
            ],
            // ~18 GB (3 LTX) + ~0.6 GB (VAE encoder) + ~0.37 GB (audio VAE + vocoder)
            // + ~12 GB (distilled transformer + x2 upscaler) + ~8 GB (Gemma-3-12B 4-bit).
            sizeEstimateGB: 39
        )
    }

    /// Krea-2-Turbo (mlx-serve bundle): ONE public repo, assembled so the engine
    /// loads it directly — a top-level transformer file + `vae/`/`text_encoder/`/
    /// `tokenizer/` subdirs + `config.json`. Recursive download (no auth, no
    /// gated base repo); ready when the transformer file + three subdirs + config
    /// are present. Unlike FLUX the transformer is a top-level FILE, not a
    /// `transformer/` subdir — hence its own readyMarkers.
    static func krea(repo: String, displayName: String, sizeGB: Double) -> MediaBundle {
        MediaBundle(
            id: "krea:\(repo)",
            displayName: displayName,
            components: [
                MediaComponent(
                    repo: repo,
                    selection: FileSelection(recursive: true),
                    readyMarkers: ["config.json", "transformer_mixed_4_8.safetensors", "vae", "text_encoder", "tokenizer"]
                ),
            ],
            sizeEstimateGB: sizeGB
        )
    }

    /// Hunyuan3D (shape stage): a flat model dir — `config.json` + the three
    /// engine safetensors (`dit`, `conditioner`, `vae`). Non-recursive with a
    /// safetensors allowlist so a future published HF repo pulls ONLY those
    /// three. Ready when all four markers are present. For a `local/`
    /// (convert-on-device) repo there's no download — readiness checks disk
    /// presence either way, so local and published repos share this factory.
    static func model3d(repo: String, displayName: String, sizeGB: Double) -> MediaBundle {
        MediaBundle(
            id: "model3d:\(repo)",
            displayName: displayName,
            components: [
                MediaComponent(
                    repo: repo,
                    // Recursive: the combined repo ships the paint (texture)
                    // stage in `paint/` and the UniRig auto-rig stage in
                    // `unirig/` beside the root shape weights — one pull
                    // lights up all three. Allowlist = exactly the seven
                    // engine weights (extras in the repo never download).
                    selection: FileSelection(recursive: true, keepSafetensors: [
                        "dit.safetensors", "conditioner.safetensors", "vae.safetensors",
                        "unet.safetensors", "unet_dual.safetensors", "dino.safetensors",
                        "skeleton.safetensors",
                    ]),
                    // All three stages must be present — a partial pull that
                    // reads "ready" would 400 on texture/rig requests.
                    readyMarkers: [
                        "config.json", "dit.safetensors",
                        "conditioner.safetensors", "vae.safetensors",
                        "paint/config.json", "paint/unet.safetensors",
                        "paint/unet_dual.safetensors", "paint/dino.safetensors",
                        "paint/vae.safetensors",
                        "unirig/config.json", "unirig/skeleton.safetensors",
                    ]
                ),
            ],
            sizeEstimateGB: sizeGB
        )
    }

    /// ACE-Step music (text2music): a flat converted dir — `config.json` +
    /// `model.safetensors` (DiT + condition encoder) + `vae.safetensors`
    /// (Oobleck) + the `text_encoder/` Qwen3-Embedding subdir. Single
    /// self-contained repo, no external-component dependencies (the simplest
    /// bundle yet). Local-convert repos share this factory with any future
    /// published one (readiness checks disk presence either way).
    static func music(repo: String, displayName: String, sizeGB: Double) -> MediaBundle {
        MediaBundle(
            id: "music:\(repo)",
            displayName: displayName,
            components: [
                MediaComponent(
                    repo: repo,
                    selection: FileSelection(recursive: true, keepSafetensors: [
                        "model.safetensors", "vae.safetensors",
                    ]),
                    readyMarkers: [
                        "config.json", "model.safetensors", "vae.safetensors",
                        "text_encoder/config.json", "text_encoder/model.safetensors",
                        "text_encoder/tokenizer.json",
                    ]
                ),
            ],
            sizeEstimateGB: sizeGB
        )
    }

    /// The Gemma-3-12B text encoder LTX needs — also a standalone chat model.
    /// Standard MLX layout (config + tokenizer + sharded safetensors).
    static let ltxGemmaRepo = "mlx-community/gemma-3-12b-it-4bit"
    static let ltxGemmaComponent = MediaComponent(
        repo: ltxGemmaRepo,
        selection: .chatDefault,
        readyMarkers: ["config.json", "tokenizer.json"]
    )
}

// MARK: - Preset → bundle

extension ImageModelPreset {
    var bundle: MediaBundle {
        switch variant {
        case .krea2Turbo:
            return .krea(repo: repo, displayName: name, sizeGB: Double(approxDownloadGB))
        default:
            return .flux(repo: repo, displayName: name, sizeGB: Double(approxDownloadGB))
        }
    }
}

extension AudioModelPreset {
    var bundle: MediaBundle {
        .tts(repo: repo, displayName: name, sizeGB: approxDownloadGB)
    }
}

extension VideoModelPreset {
    var bundle: MediaBundle {
        .ltx(repo: repo, displayName: name)
    }
}

extension Model3DModelPreset {
    var bundle: MediaBundle {
        .model3d(repo: repo, displayName: name, sizeGB: approxDownloadGB)
    }
}

extension MusicModelPreset {
    var bundle: MediaBundle {
        .music(repo: repo, displayName: name, sizeGB: approxDownloadGB)
    }
}
