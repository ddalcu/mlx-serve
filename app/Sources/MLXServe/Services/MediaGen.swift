import Foundation
import SwiftUI

/// Shared types for the image/video generation pipeline.
///
/// Models are trained at fixed resolution buckets and have an opinionated
/// step/CFG sweet spot per "speed vs. quality" tradeoff. The UI exposes a
/// Quality picker (Fast / Good / Quality / Super Quality) plus a model-
/// specific resolution dropdown. Anything more granular lives behind the
/// Advanced disclosure.

// MARK: - Quality preset

/// Industry-standard tier names. Each model defines its own concrete
/// step/guidance numbers per tier so a "Fast" on FLUX.2-klein doesn't mean
/// the same as "Fast" on FLUX.2-dev.
enum QualityPreset: String, CaseIterable, Identifiable, Codable {
    case fast = "Fast"
    case good = "Good"
    case quality = "Quality"
    case superQuality = "Super Quality"

    var id: String { rawValue }
    var label: String { rawValue }
}

// MARK: - Resolution buckets

/// Resolutions the model was trained on. Picking off-grid values usually
/// works on FLUX/LTX but produces visible artefacts, so we pin the picker
/// to known-good buckets and let users override via Advanced.
struct ResolutionOption: Hashable, Identifiable {
    let width: Int
    let height: Int
    let label: String   // e.g. "1024 × 1024 (square)"

    var id: String { "\(width)x\(height)" }
}

// MARK: - Image presets

/// mflux variant — picks the model class and `ModelConfig` factory the
/// Python script will use. Both run on MLX with native 4/8-bit quantization.
enum FluxVariant: String, Hashable, Codable {
    case flux2Klein4B     // FLUX.2-klein 4B params — uses Flux2Klein, ModelConfig.flux2_klein_4b()
    case flux2Klein9B     // FLUX.2-klein 9B params — uses Flux2Klein, ModelConfig.flux2_klein_9b()
    case krea2Turbo       // Krea-2-Turbo single-stream MMDiT — served by the krea image backend
}

struct ImageQualitySettings: Hashable {
    let steps: Int
    let guidance: Double
}

struct ImageModelPreset: Identifiable, Hashable {
    let id: String
    let name: String
    let variant: FluxVariant
    /// `ModelConfig` factory name — sets the model architecture (e.g.
    /// "schnell", "dev", "flux2_klein_4b"). Weights themselves are loaded
    /// from `repo`; the architecture must match what's stored there.
    let configName: String
    /// Pre-quantized mflux-format HuggingFace mirror. Required and
    /// non-gated — every preset ships with one we've verified is open.
    /// Loaded via `snapshot_download` + `model_path`, so weights download
    /// directly with no HF login or license-accept step.
    let repo: String
    let approxDownloadGB: Int
    let approxRAMGB: Int
    let resolutions: [ResolutionOption]
    let defaultResolution: ResolutionOption
    let qualityProfiles: [QualityPreset: ImageQualitySettings]
    let defaultQuality: QualityPreset
    /// Plain-English explanation shown under the model in the Media pane.
    let description: String

    static func == (lhs: Self, rhs: Self) -> Bool { lhs.id == rhs.id }
    func hash(into hasher: inout Hasher) { hasher.combine(id) }

    func settings(_ quality: QualityPreset) -> ImageQualitySettings {
        qualityProfiles[quality] ?? qualityProfiles[defaultQuality]!
    }

    // FLUX is trained at ~1 MP across a stable bucket of aspect ratios.
    // The architecture is shared across versions, so the bucket is too.
    private static let fluxResolutions: [ResolutionOption] = [
        .init(width: 1024, height: 1024, label: "1024 × 1024 (square)"),
        .init(width: 1152, height: 896,  label: "1152 × 896 (landscape 4:3)"),
        .init(width: 896,  height: 1152, label: "896 × 1152 (portrait 3:4)"),
        .init(width: 1216, height: 832,  label: "1216 × 832 (landscape 3:2)"),
        .init(width: 832,  height: 1216, label: "832 × 1216 (portrait 2:3)"),
        .init(width: 1344, height: 768,  label: "1344 × 768 (landscape 16:9)"),
        .init(width: 768,  height: 1344, label: "768 × 1344 (portrait 9:16)"),
        .init(width: 1536, height: 640,  label: "1536 × 640 (cinematic)"),
    ]

    /// FLUX.2-klein 4B 4-bit. Smallest footprint, fastest download.
    static let flux2Klein4B_Q4 = ImageModelPreset(
        id: "mflux/flux2-klein-4b-q4",
        name: "FLUX.2-klein 4B 4-bit (~5 GB)",
        variant: .flux2Klein4B,
        configName: "flux2_klein_4b",
        repo: "Runpod/FLUX.2-klein-4B-mflux-4bit",
        approxDownloadGB: 5,
        approxRAMGB: 8,
        resolutions: fluxResolutions,
        defaultResolution: fluxResolutions[0],
        qualityProfiles: [
            .fast:         .init(steps: 4,  guidance: 1.0),
            .good:         .init(steps: 8,  guidance: 1.0),
            .quality:      .init(steps: 12, guidance: 1.5),
            .superQuality: .init(steps: 20, guidance: 1.5),
        ],
        defaultQuality: .good,
        description: "A fast, lightweight image generator — great for everyday text-to-image and quick edits without a huge download."
    )

    // Krea-2-Turbo accepts any multiple of 16 in [256, 2048]; offer a few
    // common buckets (the server resolves/clamps anything off-grid).
    private static let kreaResolutions: [ResolutionOption] = [
        .init(width: 1024, height: 1024, label: "1024 × 1024 (square)"),
        .init(width: 768,  height: 768,  label: "768 × 768 (square, fast)"),
        .init(width: 512,  height: 512,  label: "512 × 512 (fast, low RAM)"),
        .init(width: 1024, height: 1536, label: "1024 × 1536 (portrait 2:3)"),
        .init(width: 1536, height: 1024, label: "1536 × 1024 (landscape 3:2)"),
        .init(width: 1344, height: 768,  label: "1344 × 768 (landscape 16:9)"),
        .init(width: 768,  height: 1344, label: "768 × 1344 (portrait 9:16)"),
    ]

    /// Krea-2-Turbo — single-download mlx-serve bundle (transformer mixed-4/8 +
    /// 8-bit Qwen3-VL encoder + Qwen-Image VAE + tokenizer). Distilled Turbo:
    /// 8-step flow-matching, no CFG. Served by the native `krea` image backend
    /// (auto-detected from `config.json` `model_type`).
    ///
    /// NOTE: `repo` must point at the PUBLIC bundle you upload. Defaulted to the
    /// `ddalcu` namespace — change it to wherever you publish.
    static let krea2Turbo = ImageModelPreset(
        id: "krea/krea-2-turbo-mlx-serve",
        name: "Krea 2 Turbo mixed-4/8 (~15 GB)",
        variant: .krea2Turbo,
        configName: "krea2_turbo",
        repo: "ddalcu/Krea-2-Turbo-MLX-Serve-mixed-4-8",
        approxDownloadGB: 15,
        approxRAMGB: 24,
        resolutions: kreaResolutions,
        defaultResolution: kreaResolutions[0],
        qualityProfiles: [
            // Distilled Turbo: guidance 0 always; steps beyond ~8 add little.
            .fast:         .init(steps: 6,  guidance: 0.0),
            .good:         .init(steps: 8,  guidance: 0.0),
            .quality:      .init(steps: 12, guidance: 0.0),
            .superQuality: .init(steps: 16, guidance: 0.0),
        ],
        defaultQuality: .good,
        description: "A larger, high-fidelity image model tuned for photorealistic results in just a few steps — best when quality matters more than download size."
    )

    /// Catalog ordered cheapest → heaviest. Default (`first`) is FLUX.2-klein
    /// 4B Q4 — smallest download.
    static let all: [ImageModelPreset] = [
        .flux2Klein4B_Q4, .krea2Turbo,
    ]
}

// MARK: - Video presets

/// Pipeline shape — ltx-2-mlx exposes three. One-stage is fastest. Two-stage
/// uses dev transformer + distilled LoRA for ~10× the quality at ~10× the
/// runtime. Two-stage HQ uses a higher-quality stage 1.
enum VideoPipelineMode: String, Hashable, Codable {
    case oneStage      // TI2VidOneStagePipeline,    num_steps configurable
    case twoStage      // TI2VidTwoStagesPipeline,   stage1_steps configurable
    case twoStageHQ    // TI2VidTwoStagesHQPipeline, stage1_steps configurable
}

struct VideoQualitySettings: Hashable {
    let mode: VideoPipelineMode
    /// num_steps for oneStage, stage1_steps for two-stage modes.
    let steps: Int
    /// CFG scale, only used by two-stage modes.
    let cfgScale: Double
    /// Spatial-temporal guidance. Only used by two-stage modes. Official
    /// defaults: 1.0 for twoStage, 0.0 for twoStageHQ.
    let stgScale: Double
    /// Suggested frame count — must satisfy (n-1) % 8 == 0.
    let numFrames: Int
}

struct VideoModelPreset: Identifiable, Hashable {
    let id: String
    let name: String
    let repo: String                          // open HF mirror
    let approxDownloadGB: Int                 // weights only
    let approxFirstRunDownloadGB: Int         // + Gemma text encoder
    let approxRAMGB: Int
    let resolutions: [ResolutionOption]
    let defaultResolution: ResolutionOption
    let fps: Int
    let qualityProfiles: [QualityPreset: VideoQualitySettings]
    let defaultQuality: QualityPreset
    let maxFrames: Int
    let frameOptions: [Int]
    /// Plain-English explanation shown under the model in the Media pane.
    let description: String

    static func == (lhs: Self, rhs: Self) -> Bool { lhs.id == rhs.id }
    func hash(into hasher: inout Hasher) { hasher.combine(id) }

    func settings(_ q: QualityPreset) -> VideoQualitySettings {
        qualityProfiles[q] ?? qualityProfiles[defaultQuality]!
    }

    /// Frame count that COVERS an attached audio clip: the smallest ladder
    /// value ≥ duration×fps, capped at the model's max (a longer clip is
    /// trimmed to the video by the server). nil for a zero/invalid duration.
    func framesCovering(durationSeconds: Double) -> Int? {
        guard durationSeconds > 0, let cap = frameOptions.last else { return nil }
        let needed = Int((durationSeconds * Double(fps)).rounded(.up))
        return frameOptions.first(where: { $0 >= needed }) ?? cap
    }

    /// LTX-2.3 trained resolutions. README default is 480×704 (portrait).
    private static let ltxResolutions: [ResolutionOption] = [
        .init(width: 704,  height: 480, label: "704 × 480 (landscape 3:2)"),
        .init(width: 480,  height: 704, label: "480 × 704 (portrait 3:4) — default"),
        .init(width: 768,  height: 512, label: "768 × 512 (landscape 3:2)"),
        .init(width: 512,  height: 768, label: "512 × 768 (portrait 2:3)"),
    ]

    /// LTX-2.3 frame ladder — every valid `8N+1` count from 9 up to
    /// `maxFrames`. 193 is the practical cap (≈8s at 24 fps); beyond that
    /// needs a 64 GB+ Mac. The preset defaults (49, 97) must land on this
    /// ladder or the Frames picker renders blank.
    private static func frameLadder(maxFrames: Int) -> [Int] {
        var values: [Int] = []
        var n = 9
        while n <= maxFrames { values.append(n); n += 8 }
        if !values.contains(maxFrames) { values.append(maxFrames) }
        return values
    }

    static let ltx23Q4: VideoModelPreset = {
        let cap = 193
        return VideoModelPreset(
            id: "dgrauet/ltx-2.3-mlx-q4",
            name: "LTX-Video 2.3 Q4 (with audio, ~26 GB)",
            repo: "dgrauet/ltx-2.3-mlx-q4",
            // Bundle pulls ONLY the 3 safetensors the engine reads (~18 GB) —
            // not the repo's ~50 GB of LoRAs/upscalers/alt transformers — plus
            // the ~8 GB Gemma-3-12B text encoder.
            approxDownloadGB: 18,
            approxFirstRunDownloadGB: 26,
            approxRAMGB: 24,
            resolutions: ltxResolutions,
            defaultResolution: ltxResolutions[0],
            fps: 24,
            qualityProfiles: [
                .fast:         .init(mode: .oneStage,   steps: 8,  cfgScale: 1.0, stgScale: 0.0, numFrames: 49),
                .good:         .init(mode: .oneStage,   steps: 12, cfgScale: 1.0, stgScale: 0.0, numFrames: 97),
                .quality:      .init(mode: .twoStage,   steps: 30, cfgScale: 3.0, stgScale: 1.0, numFrames: 97),
                .superQuality: .init(mode: .twoStageHQ, steps: 15, cfgScale: 3.0, stgScale: 0.0, numFrames: 97),
            ],
            defaultQuality: .good,
            maxFrames: cap,
            frameOptions: frameLadder(maxFrames: cap),
            description: "Generates short video clips from a text prompt (and optionally a starting image or audio track), with sound built in. The heaviest model here — it also pulls a Gemma text encoder on first use."
        )
    }()

    static let all: [VideoModelPreset] = [.ltx23Q4]
}

// MARK: - Audio presets (TTS / voice cloning)

/// A neural text-to-speech model served by mlx-serve's NATIVE Qwen3-TTS engine
/// (`src/tts.zig`). Only the `qwen3_tts` architecture is supported — the engine
/// dispatches on `config.json`'s `model_type`, so non-Qwen3-TTS checkpoints
/// (e.g. the old gpt2-based MOSS-TTS) can't load and aren't offered here.
///
/// We deliberately don't surface the macOS system voices here — those live in
/// Voice mode. This panel is neural-only.
struct AudioModelPreset: Identifiable, Hashable {
    let id: String
    let name: String
    /// Open `mlx-community` Qwen3-TTS repo (downloaded via DownloadManager).
    let repo: String
    /// Rough on-disk weight size, GB (first-run download). Shown in the picker.
    let approxDownloadGB: Double
    /// Peak unified-memory footprint, GB — drives the soft RAM gate.
    let approxRAMGB: Int
    /// Suggested reference-clip length for good cloning, in seconds. Surfaced
    /// as a hint next to the record button.
    let recommendedRefSeconds: Int
    /// Plain-English explanation shown under the model in the Media pane.
    let description: String

    static func == (lhs: Self, rhs: Self) -> Bool { lhs.id == rhs.id }
    func hash(into hasher: inout Hasher) { hasher.combine(id) }

    /// Qwen3-TTS 0.6B (Base) 8-bit — the lightest supported model. Default.
    /// Affine 8-bit talker + code predictor; the codec decoder and speaker
    /// encoder stay unquantized, so cloning fidelity is unchanged.
    static let qwen3TTS06B8bit = AudioModelPreset(
        id: "mlx-audio/qwen3-tts-0.6b-base-8bit",
        name: "Qwen3-TTS 0.6B 8-bit (balanced, ~2 GB)",
        repo: "mlx-community/Qwen3-TTS-12Hz-0.6B-Base-8bit",
        approxDownloadGB: 2.0,
        approxRAMGB: 3,
        recommendedRefSeconds: 8,
        description: "The lightest voice model — quick to generate speech and clone a voice from a short reference clip, with a small memory footprint."
    )

    /// Qwen3-TTS 0.6B (Base) bf16 — full-precision fidelity fallback.
    static let qwen3TTS06B = AudioModelPreset(
        id: "mlx-audio/qwen3-tts-0.6b-base",
        name: "Qwen3-TTS 0.6B bf16 (full precision, ~2.5 GB)",
        repo: "mlx-community/Qwen3-TTS-12Hz-0.6B-Base-bf16",
        approxDownloadGB: 2.5,
        approxRAMGB: 4,
        recommendedRefSeconds: 8,
        description: "The same small voice model at full precision — slightly more accurate output than the 8-bit build, at a bit more memory."
    )

    /// Qwen3-TTS 1.7B (Base) 8-bit — the quality pick: ~30% smaller download
    /// than bf16 and lower RAM, with near-identical output.
    static let qwen3TTS17B8bit = AudioModelPreset(
        id: "mlx-audio/qwen3-tts-1.7b-base-8bit",
        name: "Qwen3-TTS 1.7B 8-bit (quality, ~3.1 GB)",
        repo: "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-8bit",
        approxDownloadGB: 3.1,
        approxRAMGB: 5,
        recommendedRefSeconds: 8,
        description: "A larger voice model for more natural, expressive speech — 8-bit keeps the download and memory reasonable."
    )

    /// Qwen3-TTS 1.7B (Base) bf16 — highest fidelity here; best for
    /// expressive, long-form cloning when the Mac has the headroom.
    static let qwen3TTS17B = AudioModelPreset(
        id: "mlx-audio/qwen3-tts-1.7b-base",
        name: "Qwen3-TTS 1.7B bf16 (max fidelity, ~4.5 GB)",
        repo: "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16",
        approxDownloadGB: 4.5,
        approxRAMGB: 8,
        recommendedRefSeconds: 8,
        description: "The highest-fidelity voice model here, at full precision — best for expressive, long-form narration when you have the RAM to spare."
    )

    /// Catalog ordered lightest → heaviest. Default (`first`) is the 8-bit
    /// Qwen3-TTS 0.6B; bf16 builds stay as fidelity fallbacks.
    /// Only `qwen3_tts` models — the native engine can't serve other TTS archs.
    static let all: [AudioModelPreset] = [.qwen3TTS06B8bit, .qwen3TTS06B, .qwen3TTS17B8bit, .qwen3TTS17B]
}

// MARK: - 3D presets (image → mesh)

/// A single-image-to-3D model served by mlx-serve's NATIVE Hunyuan3D engine.
/// The engine dispatches on `config.json`'s `model_type`, so only converted
/// Hunyuan3D checkpoints load here.
///
/// ONE combined HF repo carries both stages: shape weights at the root and
/// the paint (texture) stage in `paint/` — a single download lights up
/// shape + texture (the server resolves the subdir via
/// `gen.findStageModelDir`). A `local/` repo prefix still marks a
/// convert-on-device build (`tests/convert_hunyuan3d_weights.py` et al.),
/// for which the pane shows a "convert locally" hint instead of a Download
/// button.
struct Model3DModelPreset: Identifiable, Hashable {
    let id: String
    let name: String
    /// Model directory under `~/.mlx-serve/models`. A `local/` prefix marks a
    /// convert-on-device model (no HF pull); any other prefix is a normal repo.
    let repo: String
    /// Peak unified-memory footprint, GB — drives the soft RAM gate. The paint
    /// stage is the peak (shape frees before it loads).
    let approxRAMGB: Int
    /// Full bundle download size, GB (shape + paint).
    let approxDownloadGB: Double

    static func == (lhs: Self, rhs: Self) -> Bool { lhs.id == rhs.id }
    func hash(into hasher: inout Hasher) { hasher.combine(id) }

    /// True when the model has no published HF repo yet and must be converted
    /// locally — the pane shows a "convert locally" hint instead of a download
    /// button while its weights are absent.
    var isLocalOnly: Bool { repo.hasPrefix("local/") }

    /// Hunyuan3D 2.1, 8-bit — the combined shape + paint repo.
    static let hunyuan3d21_8bit = Model3DModelPreset(
        id: "hunyuan3d-2-1-8bit",
        name: "Hunyuan3D 2.1 (8-bit)",
        repo: "ddalcu/Hunyuan3D-2.1-MLX-Serve-8bit",
        approxRAMGB: 5,
        approxDownloadGB: 8.5
    )

    /// Catalog. One entry today; grows as more 3D checkpoints convert.
    static let all: [Model3DModelPreset] = [.hunyuan3d21_8bit]
}

/// ACE-Step music-generation checkpoints (the second audio backend beside
/// Qwen3-TTS). Same local-convert convention as `Model3DModelPreset`.
struct MusicModelPreset: Identifiable, Hashable {
    let id: String
    let name: String
    /// Model directory under `~/.mlx-serve/models`. A `local/` prefix marks a
    /// convert-on-device model (no HF pull); any other prefix is a normal repo.
    let repo: String
    /// Peak unified-memory footprint, GB — drives the soft RAM gate
    /// (DiT + Qwen3-Embedding text encoder + Oobleck VAE resident together).
    let approxRAMGB: Int
    /// On-disk weight size, GB.
    let approxDownloadGB: Double
    /// Turbo checkpoints are distillation-fixed at 8 steps — not user-editable
    /// (the LTX distilled-sigmas convention).
    let fixedSteps: Int
    /// Whether the checkpoint conditions on lyrics (all ACE-Step 1.5 do).
    let supportsLyrics: Bool
    /// Plain-English explanation shown under the model in the Media pane.
    let description: String

    static func == (lhs: Self, rhs: Self) -> Bool { lhs.id == rhs.id }
    func hash(into hasher: inout Hasher) { hasher.combine(id) }

    /// True when the model has no published HF repo yet and must be converted
    /// locally — the pane shows a "convert locally" hint instead of a download
    /// button while its weights are absent.
    var isLocalOnly: Bool { repo.hasPrefix("local/") }

    /// ACE-Step v1.5 XL Turbo, 8-bit — 4B-class DiT, 8-step distilled.
    /// Published converted repo (DiT+encoders, Oobleck VAE, Qwen3-Embedding
    /// text encoder in one bundle) — one-click download in the Music tab.
    static let acestepXLTurbo8bit = MusicModelPreset(
        id: "acestep-v15-xl-turbo-8bit",
        name: "ACE-Step 1.5 XL Turbo (8-bit)",
        repo: "ddalcu/ACE-Step-1.5-XL-Turbo-MLX-Serve-8bit",
        approxRAMGB: 9,
        approxDownloadGB: 6.3,
        fixedSteps: 8,
        supportsLyrics: true,
        description: "Generates full songs — instrumental or with sung lyrics — from a style description in just 8 steps. One self-contained download."
    )

    /// Catalog. One entry today (the XL Turbo build); grows as more ACE-Step
    /// variants convert.
    static let all: [MusicModelPreset] = [.acestepXLTurbo8bit]
}

/// Dropdown catalogs for the Music tab's advanced options. Users shouldn't
/// have to know the server's value grammar — every entry here is a value the
/// engine accepts verbatim (languages ⊆ the reference VALID_LANGUAGES, bpm in
/// [30,300], keyscales in note+accidental+mode form, time signatures in
/// {2,3,4,6}). "Auto" rows map to nil/"" → the field is omitted from the
/// request and the model decides.
enum MusicOptions {
    /// (label, language code). Codes are the reference pipeline's VALID_LANGUAGES.
    static let languages: [(label: String, code: String)] = [
        ("Auto", "unknown"),
        ("English", "en"), ("Spanish", "es"), ("French", "fr"),
        ("German", "de"), ("Italian", "it"), ("Portuguese", "pt"),
        ("Japanese", "ja"), ("Korean", "ko"), ("Chinese", "zh"),
        ("Cantonese", "yue"), ("Russian", "ru"), ("Hindi", "hi"),
        ("Arabic", "ar"), ("Dutch", "nl"), ("Polish", "pl"),
        ("Turkish", "tr"), ("Vietnamese", "vi"), ("Swedish", "sv"),
    ]

    /// (label, bpm). Labels carry the genre anchor so non-musicians can pick.
    static let bpms: [(label: String, bpm: Int)] = [
        ("60 — slow ballad", 60),
        ("75 — downtempo", 75),
        ("85 — hip-hop", 85),
        ("95 — groove", 95),
        ("105 — mid-tempo pop", 105),
        ("120 — pop / house", 120),
        ("128 — EDM", 128),
        ("140 — trap / techno", 140),
        ("160 — punk / footwork", 160),
        ("174 — drum & bass", 174),
    ]

    /// All 24 conventional keys (12 pitch classes × major/minor).
    static let keyscales: [String] = {
        let notes = ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]
        return notes.map { "\($0) major" } + notes.map { "\($0) minor" }
    }()

    /// (label, wire value). The engine takes the beats-per-bar number.
    static let timeSignatures: [(label: String, value: String)] = [
        ("4/4", "4"), ("3/4", "3"), ("2/4", "2"), ("6/8", "6"),
    ]
}

/// A reusable named prompt for the Music tab's Examples menus — a built-in
/// starter or a user-saved one (persisted via `MusicPromptLibrary`). Used for
/// BOTH the style-prompt and lyrics fields; `title` is the display + dedup key.
struct MusicPrompt: Codable, Equatable, Identifiable {
    let title: String
    let body: String
    var id: String { title }

    /// Built-in style-prompt starters (genre / mood / instrumentation).
    static let builtinStyles: [MusicPrompt] = [
        MusicPrompt(title: "Synthwave",
            body: "Upbeat 80s synthwave with a driving analog bass line, dreamy lush pads, punchy retro drum machine, and a catchy soaring lead synth melody. Retro-futuristic, neon night-drive mood."),
        MusicPrompt(title: "Lo-fi study beats",
            body: "Chill lo-fi hip hop with a dusty vinyl texture, warm mellow Rhodes piano chords, soft boom-bap drums, gentle tape saturation, and a relaxed nostalgic late-night mood."),
        MusicPrompt(title: "Epic orchestral",
            body: "Epic cinematic orchestral trailer music with thunderous taiko drums, soaring string ostinatos, heroic brass fanfares, and a choir swelling to a triumphant climax."),
        MusicPrompt(title: "Modern pop",
            body: "Bright modern pop with punchy drums, shimmering synths, a bouncy bass groove, and a catchy female vocal hook. Radio-ready, feel-good summer energy."),
        MusicPrompt(title: "Acoustic folk",
            body: "Intimate acoustic folk ballad with fingerpicked guitar, soft warm male vocals, gentle harmonies, and a touch of cello. Quiet fireside atmosphere, honest and tender."),
        MusicPrompt(title: "Stadium rock",
            body: "High-energy stadium rock anthem with crunchy electric guitars, pounding drums, a driving bass, and powerful raspy male vocals building into a huge singalong chorus."),
        MusicPrompt(title: "Deep house",
            body: "Warm deep house with a rolling four-on-the-floor kick, deep sub bass, silky filtered chords, soft vocal chops, and a hypnotic late-night club groove."),
        MusicPrompt(title: "Jazz trio",
            body: "Relaxed late-night jazz trio with brushed drums, walking upright bass, and warm improvised piano. Smoky lounge atmosphere, tender and swinging."),
        MusicPrompt(title: "Trap",
            body: "Hard-hitting trap beat with booming 808 bass, crisp rattling hi-hats, dark atmospheric bells, and punchy snappy snares. Confident, cinematic, modern."),
        MusicPrompt(title: "Cinematic ambient",
            body: "Slow cinematic ambient with evolving warm synth pads, distant piano, soft field-recording textures, and a gentle emotional swell. Spacious, reflective, calm."),
    ]

    /// Built-in lyric templates — ORIGINAL, royalty-free skeletons across the
    /// song types people reach for most, meant as editable starting points.
    /// (These are original placeholder lyrics, NOT reproductions of any
    /// existing song — real lyrics are copyrighted and users add their own.)
    static let builtinLyrics: [MusicPrompt] = [
        MusicPrompt(title: "Pop hook", body: """
            [Verse]
            Woke up with the sunlight spilling on the floor
            Got that feeling something good is knocking at my door
            Phone down, head up, stepping into gold
            Every little moment worth a hundred more

            [Chorus]
            We're alive tonight, hearts on fire
            Dancing till the stars retire
            Turn it up, don't let it fade
            This is the memory we made
            """),
        MusicPrompt(title: "Acoustic ballad", body: """
            [Verse]
            The kettle hums a tired song, the winter's at the door
            Your letters in a shoebox that I don't read anymore
            But the garden that we planted still comes up every spring
            Some things keep their promises without us doing anything

            [Chorus]
            So I'll leave the porch light burning, like the old days
            Half the town away from you, and half a life too late
            If you ever wander home, you won't have to knock
            The door was never locked
            """),
        MusicPrompt(title: "Rock anthem", body: """
            [Verse]
            Concrete under worn-out shoes, we've been running all our lives
            Every no we ever heard just sharpened up our knives
            They said settle, we said never, wrote it on the wall

            [Chorus]
            We are the thunder, hear us roar
            Kicking down that closed door
            Louder than they've ever known
            Tonight we take the throne
            """),
        MusicPrompt(title: "Love song", body: """
            [Verse]
            You found me in the noise of an ordinary day
            Quiet as a Sunday, you just took my breath away
            No grand parade, no fireworks, no scene
            Just your hand in mine and everything between

            [Chorus]
            And I'd choose you, I'd choose you again
            Every version of this world I'm ever in
            Come the highs, come the lows, come whatever's true
            I would still, I would always choose you
            """),
        MusicPrompt(title: "Breakup song", body: """
            [Verse]
            I still take the long way, drive right past your street
            Left your hoodie in the closet, couldn't fold it up so neat
            Everybody says that time is gonna set me free
            But the clocks all move so slow when you're not here with me

            [Chorus]
            So I'm learning how to miss you and let you go
            Two things I never thought that I could hold
            You were half of every plan I ever made
            Now I'm building something new out of the shade
            """),
        MusicPrompt(title: "Party anthem", body: """
            [Verse]
            Lights down low, the whole room starting to move
            Bassline hitting like it's got something to prove
            No worries left outside on the floor
            Hands to the ceiling, then we ask for more

            [Chorus]
            Turn it up, turn it up, let the whole night ring
            We came here to dance, we came here to sing
            Nobody's tired, nobody's slow
            Tonight we let it all go
            """),
    ]
}

/// Pure helpers for the Music tab's saved-prompt library. Deduped by title
/// (case-sensitive), newest-first; uncapped (prompts are small text).
enum MusicPromptStore {
    /// Auto-title from a body: the first non-empty, non-`[section]` line,
    /// collapsed and capped at 40 chars. "" → "Untitled".
    static func autoTitle(from body: String) -> String {
        let lines = body.split(whereSeparator: \.isNewline).map { $0.trimmingCharacters(in: .whitespaces) }
        let first = lines.first { line in
            !line.isEmpty && !(line.hasPrefix("[") && line.hasSuffix("]"))
        } ?? ""
        let collapsed = first.split(whereSeparator: { $0 == " " || $0 == "\t" }).joined(separator: " ")
        if collapsed.isEmpty { return "Untitled" }
        if collapsed.count > 40 {
            return String(collapsed.prefix(40)).trimmingCharacters(in: .whitespaces) + "…"
        }
        return collapsed
    }

    /// Front-insert `prompt`, removing any earlier entry with the same title.
    static func adding(_ prompt: MusicPrompt, to list: [MusicPrompt]) -> [MusicPrompt] {
        var out = list
        out.removeAll { $0.title == prompt.title }
        out.insert(prompt, at: 0)
        return out
    }

    static func removing(title: String, from list: [MusicPrompt]) -> [MusicPrompt] {
        list.filter { $0.title != title }
    }
}
// MARK: - Requests

struct ImageGenRequest {
    var model: ImageModelPreset
    var prompt: String
    var negativePrompt: String = ""
    var seed: Int = -1
    var width: Int
    var height: Int
    var steps: Int
    var guidance: Double
    /// Keep the model resident after this generation (default off → unload
    /// when done, freeing GPU memory). On → instant reuse for the next gen.
    var keepResident: Bool = false
    /// Apply the server's NSFW content filter (on by default). Off → sends
    /// `"safety": false` so the server skips it for this request.
    var safeMode: Bool = true
    /// Image-to-image: path to a source PNG/JPEG. The server resizes it to the
    /// requested resolution, VAE-encodes it, and partially renoises.
    var initImagePath: String? = nil
    /// How far to renoise the source (1 = ignore it, low = small change).
    /// Only meaningful with `initImagePath` in variation mode.
    var strength: Double = 0.6
    /// Instruction editing (FLUX.2 only): condition on the source as a clean
    /// in-context reference — "make the hair blue" keeps the same person.
    /// false = variation (renoise) mode.
    var editMode: Bool = false
    /// Extra in-context reference images for edit mode (FLUX.2
    /// multi-reference): "replace the face in image 1 with the face from
    /// image 2". Sent as `ref_images` beside the primary source; the server
    /// takes at most 3.
    var refImagePaths: [String] = []
    /// Conditioning rebalance (Advanced): global multiplier on the prompt
    /// embeddings. 1.0 = off.
    var condGain: Double = 1.0
    /// Conditioning rebalance (Advanced): per-tapped-encoder-layer weights as
    /// the user typed them — comma/space separated, `condWeightCount` values
    /// (12 for Krea, 3 for FLUX). Empty = off.
    var condWeightsText: String = ""
    /// Style LoRA (Advanced): absolute path to a .safetensors adapter applied
    /// to the DiT at runtime. nil = none.
    var loraPath: String? = nil
    /// LoRA strength multiplier (on top of the file's own alpha/rank scale).
    var loraScale: Double = 1.0
}

extension ImageModelPreset {
    /// Number of tapped text-encoder layers the backend fuses — the count
    /// `cond_weights` must supply (Krea stacks 12 layers; FLUX concatenates 3).
    var condWeightCount: Int { variant == .krea2Turbo ? 12 : 3 }

    /// Instruction editing (in-context reference conditioning) is a trained
    /// FLUX.2 capability; Krea can only do renoise variations.
    var supportsReferenceEdit: Bool { variant == .flux2Klein4B || variant == .flux2Klein9B }
}

extension ImageGenRequest {
    /// Number of values `condWeightsText` must supply — one per tapped text
    /// encoder layer (Krea stacks 12 layers; FLUX concatenates 3).
    var condWeightCount: Int { model.condWeightCount }

    /// Parse a comma/space-separated weights string → doubles. Empty tokens
    /// are skipped; any unparseable token (or no tokens) → nil.
    static func parseCondWeights(_ text: String) -> [Double]? {
        let tokens = text.split(whereSeparator: { $0 == "," || $0.isWhitespace })
        guard !tokens.isEmpty else { return nil }
        var out: [Double] = []
        out.reserveCapacity(tokens.count)
        for t in tokens {
            guard let v = Double(t), v.isFinite else { return nil }
            out.append(v)
        }
        return out
    }
}

struct VideoGenRequest {
    var model: VideoModelPreset
    var prompt: String
    var seed: Int = 42
    var width: Int
    var height: Int
    var numFrames: Int
    var fps: Int
    var mode: VideoPipelineMode
    var steps: Int
    var cfgScale: Double
    var stgScale: Double = 0.0
    /// Optional first-frame image for image-to-video conditioning — supported
    /// by every pipeline mode (the server VAE-encodes it and pins it as the
    /// clean first latent frame).
    var firstFrameImagePath: String? = nil
    /// Optional speech/audio clip for audio-to-video: the soundtrack is frozen
    /// as conditioning (voices, lip sync and performance follow it) and the
    /// ORIGINAL clip is muxed into the mp4. Any AVFoundation-readable format;
    /// forces a two-stage pipeline.
    var audioPath: String? = nil
    /// Keep the model resident after this generation (default off → unload).
    var keepResident: Bool = false
    /// Style LoRA (Advanced): absolute path to a .safetensors adapter applied
    /// to the DiT at runtime. nil = none.
    var loraPath: String? = nil
    /// LoRA strength multiplier (on top of the file's own alpha/rank scale).
    var loraScale: Double = 1.0
}

struct AudioGenRequest {
    var model: AudioModelPreset
    /// The text to speak.
    var text: String
    /// Path to a normalized 24 kHz mono WAV of the voice to clone. `nil` falls
    /// back to the model's default voice (no cloning).
    var refAudioPath: String? = nil
    /// Transcript of the reference clip. Optional — supplying it can make voice
    /// cloning more stable.
    var refText: String = ""
    /// Playback speed multiplier.
    var speed: Double = 1.0
    /// Sampling temperature — higher is more expressive/varied.
    var temperature: Double = 0.7
    /// Keep the model resident after this generation (default off → unload).
    var keepResident: Bool = false
}

struct MusicGenRequest {
    var model: MusicModelPreset
    /// Style/genre/mood description — the "in the style of…" prompt.
    var prompt: String
    /// Optional lyrics; empty → the server's "[Instrumental]" convention.
    var lyrics: String = ""
    /// Vocal language code ("en", "zh", …) — only meaningful with lyrics.
    var vocalLanguage: String = "en"
    /// Optional musical metadata; nil/empty → the model decides ("N/A").
    var bpm: Int? = nil
    var keyscale: String = ""
    var timesignature: String = ""
    /// Track length in seconds (server-valid 10–600).
    var durationSeconds: Int = 60
    /// -1 = fresh random seed per generation.
    var seed: Int = -1
    /// Keep the model resident after this generation (default off → unload).
    var keepResident: Bool = false
}

struct Model3DGenRequest {
    var model: Model3DModelPreset
    /// Path to the source photo (PNG/JPEG). The subject is cut out and
    /// composited on white before encoding (the reference pipeline's rembg step).
    var photoPath: String
    /// Denoising steps for the shape flow.
    var steps: Int = 30
    /// Classifier-free guidance scale.
    var guidanceScale: Double = 5.0
    /// Marching-cubes octree resolution — higher = finer mesh, more memory/time.
    var octreeResolution: Int = 384
    /// Generation seed. -1 → a random seed is drawn per request.
    var seed: Int = -1
    /// Keep the model resident after this generation (default off → unload).
    var keepResident: Bool = false
    /// Run the P2 paint stage (full PBR texture) after shape generation.
    /// Off by default until the paint port is validated end to end.
    var texture: Bool = false
}

// MARK: - RAM checks

enum RAMChecker {
    /// Total physical memory in GB. Used for the rough "do you have enough
    /// RAM for this model" gate shown before generation starts.
    static var totalGB: Int {
        Int(ProcessInfo.processInfo.physicalMemory / (1024 * 1024 * 1024))
    }

    /// Free + inactive memory in GB (pages we can reclaim without paging).
    /// Close enough for the "do you have headroom" gate before kicking a
    /// multi-GB model load.
    ///
    /// Was a `/usr/bin/vm_stat` subprocess; now `host_statistics64` directly
    /// (`SystemMetrics.availableBytes`), which vm_stat is itself a printer for.
    /// The binary isn't reachable inside the App Sandbox container.
    static var availableGB: Int {
        let bytes = SystemMetrics.availableBytes()
        guard bytes > 0 else { return totalGB } // kernel query failed; assume headroom
        return Int(bytes / (1024 * 1024 * 1024))
    }

    /// Frame count an LTX run can safely fit at the chosen resolution.
    /// Linear in pixels × frames after a fixed model-load cost.
    static func safeFrameCap(model: VideoModelPreset, width: Int, height: Int, available: Int) -> Int {
        // Model load alone takes `approxRAMGB`. Each megapixel × 100 frames
        // costs roughly 12 GB on top of that for ltx-2-mlx — the VAE decode
        // staging pushes memory harder than the diffusers path did.
        let pixelMP = Double(width * height) / 1_000_000.0
        let headroom = max(0, available - model.approxRAMGB)
        let perHundred = max(2.0, pixelMP * 12.0)
        let framesByRAM = Int((Double(headroom) / perHundred) * 100)
        return min(model.maxFrames, max(9, framesByRAM))
    }
}
