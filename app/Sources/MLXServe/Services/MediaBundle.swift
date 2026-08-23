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
    /// When set, pull ONLY this immediate subfolder's files and write them at
    /// the destination ROOT. A multi-variant MLX repo ships one complete model
    /// per quant subfolder (see `MlxVariant`); each is fetched into its own
    /// model dir, so the prefix must come off on the way to disk.
    var subfolder: String? = nil

    /// Chat-model default: top-level files + `mtp/`, all needed extensions.
    static let chatDefault = FileSelection()

    /// One quant subfolder of a multi-variant MLX repo.
    static func mlxVariant(_ folder: String) -> FileSelection {
        FileSelection(subfolder: folder)
    }

    /// Where a fetched file lands, relative to the destination dir. Only a
    /// variant selection rewrites anything — everything else is pass-through,
    /// so the `mtp/` sidecar keeps its directory.
    func localPath(forRemote path: String) -> String {
        guard let subfolder, path.hasPrefix(subfolder + "/") else { return path }
        return String(path.dropFirst(subfolder.count + 1))
    }
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

    /// "~15 GB" / "~2.5 GB" — whole numbers at 1 GB+, one decimal below (a
    /// TTS model rounding to "~0 GB" would be a lie). Shared by
    /// `BundleDownloadBar` and the Model Browser's Media pane so the two
    /// surfaces can't drift onto different rounding rules.
    ///
    /// A .5 keeps its decimal, and ties round UP.
    ///
    /// `%.0f`/`%.1f` are round-half-to-EVEN, which made the 2.0 GB and 2.5 GB
    /// Qwen3-TTS presets both print "~2 GB" — two different downloads wearing
    /// one label — and rendered Kokoro's 0.35 GB as "~0.3 GB". Both errors run
    /// the same way: a size shown SMALLER than the bytes about to be fetched.
    /// For a download prompt that is the misleading direction, so ties go up.
    var approxSizeLabel: String { Self.sizeLabel(forGB: sizeEstimateGB) }

    /// Static so the rounding rule is testable without building a bundle.
    static func sizeLabel(forGB gb: Double) -> String {
        if gb < 1 { return String(format: "~%.1f GB", roundUpHalf(gb * 10) / 10) }
        let rounded = roundUpHalf(gb * 2) / 2
        return rounded == rounded.rounded()
            ? String(format: "~%.0f GB", rounded)
            : String(format: "~%.1f GB", rounded)
    }

    /// `.rounded(.toNearestOrAwayFromZero)`, spelled out because the whole
    /// point of this helper is that the DEFAULT tie rule was the bug.
    private static func roundUpHalf(_ v: Double) -> Double {
        v.rounded(.toNearestOrAwayFromZero)
    }

    static func == (l: MediaBundle, r: MediaBundle) -> Bool { l.id == r.id }
}

// MARK: - Bundle factories (per modality)

extension MediaBundle {
    /// FLUX (mflux): one repo with weight subdirs (`transformer/`, `vae/`,
    /// `text_encoder/`, `tokenizer/`). Recursive download; ready when all four
    /// subdirs are present, plus the root config.json where the conversion
    /// ships one.
    ///
    /// `hasRootConfig` is not a preference — it is a fact about the repo.
    /// `mlx-community/flux2-klein-9b-4bit` ships no config.json, and demanding
    /// one would make a complete 10 GB download read as permanently incomplete
    /// (the Kokoro-reusing-the-TTS-bundle bug: the pane offers Download
    /// forever). The server classifies that repo from the DiT's weight names
    /// instead, so nothing downstream needs the file either.
    static func flux(repo: String, displayName: String, sizeGB: Double, hasRootConfig: Bool = true) -> MediaBundle {
        let subdirs = ["transformer", "vae", "text_encoder", "tokenizer"]
        return MediaBundle(
            id: "flux:\(repo)",
            displayName: displayName,
            components: [
                MediaComponent(
                    repo: repo,
                    selection: FileSelection(recursive: true),
                    readyMarkers: hasRootConfig ? ["config.json"] + subdirs : subdirs
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

    /// Kokoro: top-level weights + the `g2p/` dictionary subdir, so the download
    /// must be RECURSIVE.
    ///
    /// Its ready markers are NOT the TTS ones. `tts(...)` requires
    /// `speech_tokenizer`, which Kokoro does not have — reusing that bundle made
    /// a fully downloaded Kokoro read as permanently incomplete, so the pane
    /// would offer Download forever and never enable the voice.
    /// `voices.safetensors` and `g2p` are both listed because either one missing
    /// breaks the engine AT LOAD (no voices, or no phonemizer) rather than
    /// degrading gracefully.
    static func kokoro(repo: String, displayName: String, sizeGB: Double) -> MediaBundle {
        MediaBundle(
            id: "kokoro:\(repo)",
            displayName: displayName,
            components: [
                MediaComponent(
                    repo: repo,
                    selection: FileSelection(recursive: true),
                    readyMarkers: ["config.json", "model.safetensors", "voices.safetensors", "g2p"]
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
    /// MiniMax-H3: ONE self-contained repo — DiT, text encoder, both VAEs and
    /// the tokenizer. Upstream splits these across `Comfy-Org` (weights, no
    /// tokenizer) and `MiniMaxAI` (tokenizer); our converted mirror bundles
    /// them so there is no second component to keep in sync.
    ///
    /// `audio_vae.safetensors` is allowlisted but is NOT a ready marker, the
    /// same call the LTX bundle makes: without it the server still generates,
    /// the clip is just silent.
    static func minimaxH3(repo: String, displayName: String) -> MediaBundle {
        MediaBundle(
            id: "minimax-h3:\(repo)",
            displayName: displayName,
            components: [
                MediaComponent(
                    repo: repo,
                    selection: FileSelection(keepSafetensors: [
                        "transformer.safetensors", "text_encoder.safetensors",
                        "video_vae.safetensors", "audio_vae.safetensors",
                        // The Turbo distillation adapter (~744 MB): 4-step
                        // sampling instead of 30. Allowlisted but NOT a ready
                        // marker, the same call `audio_vae` makes — a pack
                        // downloaded before it shipped must keep reading as
                        // complete rather than offering a 69 GB re-download.
                        // Those installs get it on demand instead, see
                        // `TurboLoraFetch`.
                        TurboLoraFetch.fileName,
                    ]),
                    readyMarkers: [
                        "config.json", "transformer.safetensors",
                        "text_encoder.safetensors", "video_vae.safetensors",
                        // The tokenizer is a ready marker BECAUSE it ships in
                        // this repo: without it there is no prompt to encode,
                        // and upstream does not provide one alongside the
                        // weights.
                        "tokenizer.json",
                    ]
                ),
            ],
            sizeEstimateGB: 70 // 69 + the Turbo adapter
        )
    }

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

    /// LTX 2.5: same engine files as 2.3, but the text encoder lives INSIDE
    /// the pack (`gemma4-12b-ltx-v1/`) instead of being the shared Gemma-3
    /// chat download — so there is no second component, and the fetch has to
    /// be recursive to reach the encoder's own tokenizer + config.
    ///
    /// The safetensors allowlist still applies (by BASENAME), so `model` is on
    /// it for the encoder's weights; the upscalers ride along like 2.3's. The
    /// encoder DIRECTORY is a ready marker: without it the server has no text
    /// path at all, which is a harder failure than 2.3's missing-I2V case.
    static func ltx25(repo: String, displayName: String, sizeGB: Double) -> MediaBundle {
        MediaBundle(
            id: "ltx25:\(repo)",
            displayName: displayName,
            components: [
                MediaComponent(
                    repo: repo,
                    selection: FileSelection(
                        recursive: true,
                        excludeSubstrings: [".cache/"],
                        keepSafetensors: [
                            "transformer-distilled.safetensors", "transformer-dev.safetensors",
                            "connector.safetensors", "vae_decoder.safetensors", "vae_encoder.safetensors",
                            "audio_vae.safetensors", "vocoder.safetensors",
                            "spatial_upscaler_x2_v1_1.safetensors", "temporal_upscaler_x2_v1_0.safetensors",
                            // LTX's own DiffVAE decoder — the 8-bit pack ships
                            // it, the 4-bit one does not, and the allowlist is
                            // by basename so a pack without it just has one
                            // fewer file to fetch.
                            "vae_diffusion_decoder.safetensors",
                            // The in-pack Gemma-4 text encoder's weights.
                            "model.safetensors",
                        ]
                    ),
                    readyMarkers: [
                        "config.json", "transformer-distilled.safetensors",
                        "connector.safetensors", "vae_decoder.safetensors",
                        ltx25TextEncoderDir,
                    ]
                ),
            ],
            sizeEstimateGB: sizeGB
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

    /// ACE-Step music: a flat converted dir — `config.json` +
    /// `model.safetensors` (DiT + condition encoder) + `vae.safetensors`
    /// (Oobleck) + `fsq.safetensors` (cover-mode tokenizer; fetched on demand
    /// into packs that predate it) + the `text_encoder/` Qwen3-Embedding subdir. Single
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
                        "model.safetensors", "vae.safetensors", "fsq.safetensors",
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

    /// MiniMax Music 3: a flat converted dir — `config.json` + five component
    /// safetensors (LLM, depth decoder, DiT, condition encoder, vocoder) +
    /// the `tokenizer/` subdir the engine reads (`music_tokenizer/` rides
    /// along via the recursive scan). The vocoder is the completeness marker
    /// (written LAST by the converter — mirrors the server's
    /// `requiredMediaMarker`).
    static func music3(repo: String, displayName: String, sizeGB: Double) -> MediaBundle {
        MediaBundle(
            id: "music3:\(repo)",
            displayName: displayName,
            components: [
                MediaComponent(
                    repo: repo,
                    selection: FileSelection(recursive: true, keepSafetensors: [
                        "language_model.safetensors", "rvq_depth_decoder.safetensors",
                        "transformer.safetensors", "condition_encoder.safetensors",
                        "vocoder.safetensors",
                    ]),
                    readyMarkers: [
                        "config.json", "language_model.safetensors",
                        "rvq_depth_decoder.safetensors", "transformer.safetensors",
                        "condition_encoder.safetensors", "vocoder.safetensors",
                        "tokenizer/tokenizer.json",
                    ]
                ),
            ],
            sizeEstimateGB: sizeGB
        )
    }

    /// Mage-Flow (diffusers layout): one repo with weight subdirs
    /// (`transformer/`, `vae/`, `text_encoder/`, `scheduler/`) and NO root
    /// config.json — detection keys on `model_index.json`. Recursive download
    /// excluding the README `assets/` images. Ready when the index + all four
    /// component subdirs are present.
    static func mageFlow(repo: String, displayName: String, sizeGB: Double) -> MediaBundle {
        MediaBundle(
            id: "mageflow:\(repo)",
            displayName: displayName,
            components: [
                MediaComponent(
                    repo: repo,
                    selection: FileSelection(recursive: true, excludeSubstrings: ["assets/"]),
                    readyMarkers: ["model_index.json", "transformer", "vae", "text_encoder", "scheduler"]
                ),
            ],
            sizeEstimateGB: sizeGB
        )
    }

    /// The subdirectory LTX 2.5 ships its own text encoder in. Cross-pinned
    /// with the server's `ltx_video.LtxVersion.textEncoderSubdir` — the server
    /// resolves the encoder from this exact path, so a rename here silently
    /// makes every 2.5 pack unloadable.
    static let ltx25TextEncoderDir = "gemma4-12b-ltx-v1"

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
        case .mageFlowTurbo, .mageFlowEditTurbo:
            return .mageFlow(repo: repo, displayName: name, sizeGB: Double(approxDownloadGB))
        case .flux2Klein9B:
            // The one MLX conversion of klein 9B ships no root config.json.
            return .flux(repo: repo, displayName: name, sizeGB: Double(approxDownloadGB), hasRootConfig: false)
        default:
            return .flux(repo: repo, displayName: name, sizeGB: Double(approxDownloadGB))
        }
    }
}

extension AudioModelPreset {
    /// The `.audio` slot hosts TWO architectures with different repo shapes, so
    /// this dispatches instead of assuming Qwen3-TTS. `supportsCloning` is the
    /// discriminator the preset already declares.
    var bundle: MediaBundle {
        supportsCloning
            ? .tts(repo: repo, displayName: name, sizeGB: approxDownloadGB)
            : .kokoro(repo: repo, displayName: name, sizeGB: approxDownloadGB)
    }
}

extension VideoModelPreset {
    var bundle: MediaBundle {
        switch backend {
        case .ltx:
            // A pack carrying its own encoder must NOT also pull the shared
            // Gemma-3 chat model: 8 GB fetched for a component it never opens.
            return shipsOwnTextEncoder
                ? .ltx25(repo: repo, displayName: name, sizeGB: Double(approxDownloadGB))
                : .ltx(repo: repo, displayName: name)
        case .minimaxH3: return .minimaxH3(repo: repo, displayName: name)
        }
    }
}

extension Model3DModelPreset {
    var bundle: MediaBundle {
        .model3d(repo: repo, displayName: name, sizeGB: approxDownloadGB)
    }
}

extension MusicModelPreset {
    var bundle: MediaBundle {
        switch family {
        case .acestep: return .music(repo: repo, displayName: name, sizeGB: approxDownloadGB)
        case .minimaxMusic3: return .music3(repo: repo, displayName: name, sizeGB: approxDownloadGB)
        }
    }
}

// MARK: - Media pane generic surface

/// Common surface every media-gen preset (image/audio/video/music) exposes to
/// the Model Browser's Media tab, so ONE generic row renders all four
/// modalities instead of four near-duplicate views. `Model3DModelPreset`
/// deliberately does NOT conform — the Media tab covers exactly the four
/// modalities the user asked for; 3D stays its own thing for now.
/// What every media preset — INCLUDING 3D — can say about its cost. Split out
/// of `MediaModelPreset` so a Create pane's picker can rank all five catalogues
/// (`MediaModelPicks`) without dragging 3D into the Model Browser's Media tab,
/// which is what conforming it to the fuller protocol below would do.
protocol MediaModelSizing: Identifiable where ID == String {
    var name: String { get }
    /// Peak unified-memory footprint, GB — already the full RAM-needed
    /// figure (not raw weight size), unlike `RecommendedModelPick.sizeGB`,
    /// so `meetsSystemRequirements` below needs no extra overhead multiplier.
    var approxRAMGB: Int { get }
}

extension MediaModelSizing {
    /// Whether this Mac's physical RAM covers what the model needs. A soft
    /// signal for the UI (show a warning, never a download/use gate) — same
    /// "warn, don't block" policy as `RecommendedModelPick.meetsSystemRequirements`
    /// and `ImageGenView`'s oversized-model alert.
    func meetsSystemRequirements(physicalMemoryBytes: UInt64) -> Bool {
        physicalMemoryBytes >= UInt64(approxRAMGB) * 1_073_741_824
    }
}

protocol MediaModelPreset: MediaModelSizing, Hashable {
    var bundle: MediaBundle { get }
    /// Plain-English explanation shown under the model in the Media pane —
    /// the same idea as `RecommendedModelPick.blurb`.
    var description: String { get }
}

extension ImageModelPreset: MediaModelPreset {}
extension AudioModelPreset: MediaModelPreset {}
extension VideoModelPreset: MediaModelPreset {}
extension MusicModelPreset: MediaModelPreset {}
// Sizing only — see `MediaModelSizing`: 3D stays out of the Media tab.
extension Model3DModelPreset: MediaModelSizing {}
