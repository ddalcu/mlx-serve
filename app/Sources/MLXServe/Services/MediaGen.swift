import Foundation
import SwiftUI

/// Shared types for the image/video generation pipeline.
///
/// Models are trained at fixed resolution buckets and have an opinionated
/// step-count sweet spot per "speed vs. quality" tradeoff. The UI exposes a
/// Quality picker (Fast / Good / Quality / Super Quality) plus a model-
/// specific resolution dropdown. Anything more granular lives behind the
/// Advanced disclosure.

// MARK: - Quality preset

/// Industry-standard tier names. Each model defines its own concrete step
/// count per tier, so a "Fast" on FLUX.2-klein doesn't mean the same as "Fast"
/// on FLUX.2-dev. There is no CFG here: no image backend reads a guidance
/// field, so carrying one only invited the UI to show a knob that does nothing.
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

    /// Sentinel: send NO `size`, so the server keeps the reference image's own
    /// resolution (the edit pipeline's `max_size = source size` default). An
    /// editor that returns a different shape than it was given is wrong by
    /// default; a fixed bucket is the override, not the other way round.
    static let matchSource = ResolutionOption(width: 0, height: 0, label: "Match source (keep the original size)")

    var isMatchSource: Bool { width == 0 && height == 0 }

    /// Sentinel: the menu row that reveals the width/height fields. Carries no
    /// size of its own — the pane holds the typed values, because they must
    /// survive a preset switch the way the reference lists do.
    static let custom = ResolutionOption(width: -1, height: -1, label: "Custom…")

    var isCustom: Bool { width == -1 && height == -1 }
}

// MARK: - Custom resolution

/// The resolution grid a backend actually samples on.
///
/// This MIRRORS `clampFluxDim` / `clampKreaDim` in `src/gen.zig` — the server
/// stays the authority and rewrites anything off-grid regardless of what the
/// app sends. The mirror exists so the pane can say what will happen BEFORE
/// the request, instead of the user reading a size back off a finished image.
/// Documented duplication in the `isMediaModelType` / `modalityFromType`
/// mould; `CustomResolutionTests` is what keeps the two from drifting.
struct ResolutionGrid: Hashable {
    /// Every dimension must be a multiple of this.
    let alignment: Int
    let minDim: Int
    let maxDim: Int

    /// Round onto the grid the way the server does — UP, never to nearest
    /// (`((v + 31) / 32) * 32`). Rounding the friendly way would print a hint
    /// naming a size the server does not generate.
    func snap(_ v: Int) -> Int {
        guard v > 0 else { return minDim }
        return ((v + alignment - 1) / alignment) * alignment
    }

    /// Classify a typed size. In-range-but-off-grid is a CORRECTION (the model
    /// can nearly do it, so do it and say so); out-of-range is a REFUSAL, since
    /// silently clamping 4000 to 1536 hands back a picture a third of the
    /// requested size with nothing explaining why.
    func resolve(width: Int, height: Int) -> CustomResolution {
        for v in [width, height] where v <= 0 {
            return .invalid(message: "Width and height must be whole numbers above zero.")
        }
        for v in [width, height] where v < minDim || v > maxDim {
            return .invalid(message: "This model samples between \(minDim) and \(maxDim) px per side. \(v) is outside that.")
        }
        let w = snap(width), h = snap(height)
        guard w != width || h != height else { return .ok(width: width, height: height) }
        return .corrected(width: w, height: h,
                          note: "Rounded to \(w) × \(h) — this model samples in steps of \(alignment) px.")
    }
}

/// What the pane does with a typed size.
enum CustomResolution: Equatable {
    /// Already on the grid — send it as typed.
    case ok(width: Int, height: Int)
    /// Nudged onto the grid. `note` is the small hint shown under the fields.
    case corrected(width: Int, height: Int, note: String)
    /// Cannot be honored at all; Generate stays disabled and `message` says why.
    case invalid(message: String)

    /// The size to actually send, or nil when there is nothing to send.
    var size: (width: Int, height: Int)? {
        switch self {
        case let .ok(w, h), let .corrected(w, h, _): return (w, h)
        case .invalid: return nil
        }
    }

    var isValid: Bool { size != nil }

    /// The line shown under the fields — nil when there is nothing to say.
    var hint: String? {
        switch self {
        case .ok: return nil
        case let .corrected(_, _, note): return note
        case let .invalid(message): return message
        }
    }
}

// MARK: - Image presets

/// mflux variant — picks the model class and `ModelConfig` factory the
/// Python script will use. Both run on MLX with native 4/8-bit quantization.
enum FluxVariant: String, Hashable, Codable {
    case flux2Klein4B     // FLUX.2-klein 4B params — uses Flux2Klein, ModelConfig.flux2_klein_4b()
    case flux2Klein9B     // FLUX.2-klein 9B params — uses Flux2Klein, ModelConfig.flux2_klein_9b()
    case krea2Turbo       // Krea-2-Turbo single-stream MMDiT — served by the krea image backend
    case mageFlowTurbo    // Microsoft Mage-Flow-Turbo double-stream flow DiT — served by the mage_flow backend
    case mageFlowEditTurbo // Microsoft Mage-Flow-Edit-Turbo — same arch, edit-trained; multi-reference in-context editor
}

struct ImageQualitySettings: Hashable {
    let steps: Int
}

struct ImageModelPreset: Identifiable, Hashable {
    var id: String
    var name: String
    let variant: FluxVariant
    /// `ModelConfig` factory name — sets the model architecture (e.g.
    /// "schnell", "dev", "flux2_klein_4b"). Weights themselves are loaded
    /// from `repo`; the architecture must match what's stored there.
    let configName: String
    /// Pre-quantized mflux-format HuggingFace mirror. Required and
    /// non-gated — every preset ships with one we've verified is open.
    /// Loaded via `snapshot_download` + `model_path`, so weights download
    /// directly with no HF login or license-accept step.
    var repo: String
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
        // The server has always taken this (`clampFluxDim` accepts any multiple
        // of 32 from 256 up) — it was a missing menu row, not a missing
        // capability. Below the ~1MP trained scale, so quality falls off; it is
        // here for speed and for Macs that cannot hold a 1024² working set.
        .init(width: 512,  height: 512,  label: "512 × 512 (fast, low RAM)"),
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
            .fast:         .init(steps: 4),
            .good:         .init(steps: 8),
            .quality:      .init(steps: 12),
            .superQuality: .init(steps: 20),
        ],
        defaultQuality: .good,
        description: "A fast, lightweight image generator — great for everyday text-to-image and quick edits without a huge download."
    )

    /// FLUX.2-klein 9B 4-bit. Same architecture as the 4B — the whole delta is
    /// six numbers (8 double / 24 single blocks, 32 heads, wider joint dim, and
    /// a Qwen3-8B text encoder instead of the 4B's), which the engine reads off
    /// the checkpoint rather than assuming. Twice the download for a
    /// meaningfully stronger model at the same 4-step-ish schedule.
    ///
    /// `mlx-community/flux2-klein-9b-4bit` is the only MLX conversion of it and
    /// ships NO root config.json — hence its own bundle markers, and the
    /// server-side weight-name fingerprint that makes the dir discoverable.
    static let flux2Klein9B_Q4 = ImageModelPreset(
        id: "mflux/flux2-klein-9b-q4",
        name: "FLUX.2-klein 9B 4-bit (~10 GB)",
        variant: .flux2Klein9B,
        configName: "flux2_klein_9b",
        repo: "mlx-community/flux2-klein-9b-4bit",
        approxDownloadGB: 10,
        approxRAMGB: 16,
        resolutions: fluxResolutions,
        defaultResolution: fluxResolutions[0],
        qualityProfiles: [
            .fast:         .init(steps: 4),
            .good:         .init(steps: 8),
            .quality:      .init(steps: 12),
            .superQuality: .init(steps: 20),
        ],
        defaultQuality: .good,
        description: "The bigger FLUX.2-klein — stronger prompt following and detail than the 4B, with the same fast schedule and the same instruction editing. Twice the download and memory."
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
            // Distilled Turbo: steps beyond ~8 add little.
            .fast:         .init(steps: 6),
            .good:         .init(steps: 8),
            .quality:      .init(steps: 12),
            .superQuality: .init(steps: 16),
        ],
        defaultQuality: .good,
        description: "A larger, high-fidelity image model tuned for photorealistic results in just a few steps — best when quality matters more than download size."
    )

    /// Mage-Flow is genuinely native-resolution: the model card claims 512-2048
    /// on ANY aspect ratio, "including extreme 4:1", and that holds — measured
    /// on an M-series Mac, cost tracks MEGAPIXELS only, not shape (1024², 2048×512
    /// and 512×2048 all land within a few hundred ms of each other at 4 steps).
    /// So the menu goes all the way to 2048 and includes the panoramas; the time
    /// hints are measured, since 2048² costs ~8× a 1024².
    private static let mageFlowResolutions: [ResolutionOption] = [
        .init(width: 1024, height: 1024, label: "1024 × 1024 (square) · ~6s"),
        .init(width: 768,  height: 768,  label: "768 × 768 (square, fast)"),
        .init(width: 512,  height: 512,  label: "512 × 512 (fastest) · ~3s"),
        .init(width: 1024, height: 1536, label: "1024 × 1536 (portrait 2:3)"),
        .init(width: 1536, height: 1024, label: "1536 × 1024 (landscape 3:2) · ~13s"),
        .init(width: 1344, height: 768,  label: "1344 × 768 (landscape 16:9)"),
        .init(width: 768,  height: 1344, label: "768 × 1344 (portrait 9:16)"),
        .init(width: 2048, height: 1152, label: "2048 × 1152 (16:9, large)"),
        .init(width: 2048, height: 2048, label: "2048 × 2048 (max) · ~50s"),
        .init(width: 2048, height: 512,  label: "2048 × 512 (panorama 4:1)"),
        .init(width: 512,  height: 2048, label: "512 × 2048 (tall 1:4)"),
    ]

    /// Mage-Flow-Turbo — the official MIT diffusers repo (no login / license
    /// step; Microsoft renamed the org to `mage-flow-community`, old
    /// `microsoft/` dirs on disk keep working — nothing keys on the org).
    /// Native double-stream flow DiT + Qwen3-VL text encoder + DiCo VAE,
    /// served by the native `mage_flow` backend (auto-detected from
    /// `model_index.json`, not `config.json`). Distilled Turbo: 4-step flow
    /// matching, guidance 1.0 (no CFG). Runs bf16 (DiT+encoder) + f32 VAE.
    static let mageFlowTurbo = ImageModelPreset(
        id: "mage-flow-community/mage-flow-turbo",
        name: "Mage-Flow Turbo (~17 GB)",
        variant: .mageFlowTurbo,
        configName: "mage_flow",
        repo: "mage-flow-community/Mage-Flow-Turbo",
        approxDownloadGB: 17,
        approxRAMGB: 16,
        resolutions: mageFlowResolutions,
        defaultResolution: mageFlowResolutions[0],
        qualityProfiles: [
            // Distillation-fixed at 4 steps (`stepsAreFixed`), so every tier is
            // 4: measured, 8 steps costs 2× and 12 costs 4× for a DIFFERENT
            // image, not a better one. The quality picker hides for this preset.
            .fast:         .init(steps: 4),
            .good:         .init(steps: 4),
            .quality:      .init(steps: 4),
            .superQuality: .init(steps: 4),
        ],
        defaultQuality: .good,
        description: "Microsoft's native-resolution image model — crisp, photorealistic results in just 4 steps, at any size from 512 to 2048 and any aspect ratio. Open (MIT) with no login or license step."
    )

    /// Microsoft Mage-Flow-Edit-Turbo — same architecture as Turbo, but
    /// edit-trained: a multi-reference in-context editor (change / compose from
    /// one or more reference images + a text instruction). Distilled 4-step,
    /// guidance 1.0. Same MIT diffusers layout (`model_index.json`), served by
    /// the `mage_flow` backend (the Edit weights light up `supportsEdit`).
    static let mageFlowEditTurbo = ImageModelPreset(
        id: "mage-flow-community/mage-flow-edit-turbo",
        name: "Mage-Flow Edit Turbo (~17 GB)",
        variant: .mageFlowEditTurbo,
        configName: "mage_flow",
        repo: "mage-flow-community/Mage-Flow-Edit-Turbo",
        approxDownloadGB: 17,
        approxRAMGB: 16,
        resolutions: mageFlowResolutions,
        defaultResolution: mageFlowResolutions[0],
        qualityProfiles: [
            // Same distillation-fixed 4 steps as Turbo.
            .fast:         .init(steps: 4),
            .good:         .init(steps: 4),
            .quality:      .init(steps: 4),
            .superQuality: .init(steps: 4),
        ],
        defaultQuality: .good,
        description: "Microsoft's native-resolution image EDITOR — change or compose images from one or more references plus a text instruction, in just 4 steps. Keeps the source's own resolution by default. Open (MIT)."
    )

    /// Our 8-bit mirrors of the two Mage-Flow releases: same architecture, same
    /// 4-step schedule, roughly half the download and half the resident weights.
    /// They reuse the bf16 sibling's `variant` because quantization changes the
    /// checkpoint, not the capability set — the engine's `MfLinear` picks
    /// dense-vs-quantized per tensor at load, so nothing else has to know.
    static let mageFlowTurbo8bit = ImageModelPreset(
        id: "ddalcu/mage-flow-turbo-8bit",
        name: "Mage-Flow Turbo 8-bit (~9 GB)",
        variant: .mageFlowTurbo,
        configName: "mage_flow",
        repo: "ddalcu/Mage-Flow-Turbo-MLX-Serve-8bit",
        approxDownloadGB: 9,
        approxRAMGB: 10,
        resolutions: mageFlowResolutions,
        defaultResolution: mageFlowResolutions[0],
        qualityProfiles: [
            .fast:         .init(steps: 4),
            .good:         .init(steps: 4),
            .quality:      .init(steps: 4),
            .superQuality: .init(steps: 4),
        ],
        defaultQuality: .good,
        description: "Microsoft's native-resolution image model, quantized to 8-bit — the same crisp 4-step results at half the download and memory. Open (MIT) with no login or license step."
    )

    /// 8-bit mirror of the editor. The repo name keeps "Mage-Flow-Edit" because
    /// the engine gates edit capability on the DIRECTORY NAME (`dirIsEdit`), and
    /// the download dir is the repo id.
    static let mageFlowEditTurbo8bit = ImageModelPreset(
        id: "ddalcu/mage-flow-edit-turbo-8bit",
        name: "Mage-Flow Edit Turbo 8-bit (~10 GB)",
        variant: .mageFlowEditTurbo,
        configName: "mage_flow",
        repo: "ddalcu/Mage-Flow-Edit-Turbo-MLX-Serve-8bit",
        approxDownloadGB: 10,
        approxRAMGB: 11,
        resolutions: mageFlowResolutions,
        defaultResolution: mageFlowResolutions[0],
        qualityProfiles: [
            .fast:         .init(steps: 4),
            .good:         .init(steps: 4),
            .quality:      .init(steps: 4),
            .superQuality: .init(steps: 4),
        ],
        defaultQuality: .good,
        description: "Microsoft's native-resolution image EDITOR, quantized to 8-bit — change or compose from one or more references in 4 steps, at half the download and memory. Open (MIT)."
    )

    /// Catalog ordered cheapest → heaviest. Default (`first`) is FLUX.2-klein
    /// 4B Q4 — smallest download.
    static let all: [ImageModelPreset] = [
        .flux2Klein4B_Q4,                              // 5
        .mageFlowTurbo8bit, .mageFlowEditTurbo8bit,    // 9, 10
        .flux2Klein9B_Q4,                              // 10
        .krea2Turbo,                                   // 15
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

/// Which server-side engine arm serves this preset. Explicit rather than
/// sniffed from the id: the download bundle and the request surface both
/// dispatch on it, and a magic-string check in two places is how they drift.
enum VideoBackendKind: String, Hashable {
    case ltx
    case minimaxH3
}

struct VideoModelPreset: Identifiable, Hashable {
    var id: String
    var name: String
    var repo: String                          // open HF mirror
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

    // What the BACKEND can actually do. Declared, never inferred — the pane
    // gates on these instead of assuming every video model is LTX-shaped.
    // Mage-Flow shipped with five dead image controls before its preset started
    // declaring capabilities; these exist so the video pane cannot repeat it.
    // Defaults describe LTX, so existing presets are unchanged.

    /// Which engine arm serves it.
    var backend: VideoBackendKind = .ltx
    /// Runtime LoRA adapters (stacked: several attach at once and their
    /// deltas sum). Every video backend takes them — H3's are the same
    /// `lora_paths`/`lora_scales` fields, resolved against its own module
    /// names, and they stack with the engine-owned Turbo adapter.
    var supportsLoRA: Bool = true
    /// Classifier-free guidance. H3 is CFG-DISTILLED — there is no guidance
    /// pass to scale, so a CFG slider would be a dead control.
    var supportsCFG: Bool = true
    /// One-stage / two-stage / two-stage-HQ pipelines. LTX-only.
    var supportsPipelineModes: Bool = true
    /// Audio-to-video conditioning from an attached clip. H3 GENERATES its
    /// soundtrack jointly with the frames and takes no audio input on FL2VA.
    var supportsAudioInput: Bool = true
    /// Whether the model produces its own soundtrack.
    var generatesAudio: Bool = false
    /// Server-side fast recipe (H3: step cache + attention broadcast, 2.83x
    /// at 768p). DEFAULT-ON server-side; the pane offers a max-quality
    /// opt-out that sends "fast": false.
    var supportsFastRecipe: Bool = false
    /// ref2va reference conditioning (images / videos / audio the generation
    /// follows). Only the REF2VA pack has it — the FL2VA DiT would generate
    /// while ignoring every reference, so the server 400s and the pane hides
    /// the controls AND stops sending the fields.
    var supportsReferences: Bool = false
    /// Engine-owned Turbo distillation LoRA (larryvrh/MiniMax-H3-Turbo-Lora):
    /// 4-step sampling instead of 30, the server's fast recipe off. Needs
    /// `turbo_lora.safetensors` beside the pack's weights — absent, the server
    /// answers a named 400 saying where to get it. FL2VA packs only until the
    /// LoRA has been eyeballed on the REF2VA DiT.
    var supportsTurbo: Bool = false
    /// Chained-window long clips (server `chain_windows`): N windows joined by
    /// fl2va keyframe conditioning, so the REF2VA pack cannot serve it.
    var supportsChainedWindows: Bool = false
    /// Last-frame keyframe conditioning (server `last_frame_image`). The other
    /// half of fl2va — first-LAST frame to video+audio — where the first frame
    /// is the geometry anchor (plain stretch) and the last is a follower
    /// (aspect-preserving center-cover). LTX pins the last LATENT frame the
    /// same way it pins the first (both anchors resized to the canvas). A
    /// reference has no keyframe row to anchor, so on H3 this rides the same
    /// partition complement as Turbo and chaining.
    var supportsLastFrame: Bool = false
    /// Denoising-step range the Steps slider offers. LTX's is the default; a
    /// backend whose floor is higher declares it, because a slider that goes
    /// somewhere the model does not work is a dead range, not a fast option.
    var stepsRange: ClosedRange<Int> = 4...50
    /// One sentence under the Steps slider. Per-backend for the same reason —
    /// LTX's "runs well from ~8" is wrong advice on any other engine.
    var stepsHelp: String = "More steps refine the video further at the cost of speed. ~8 is fast, ~30 is the reference default."
    /// Lowest step count we have a QUALITY verdict on. Below it the model
    /// still runs — the server has no such floor — but it needs a distilled
    /// few-step adapter to look like anything. This is advice, not a gate:
    /// `stepsRange` stays as wide as the server's own acceptance, because a
    /// pane that refuses what the API takes makes its own LoRA slot useless
    /// (#254). 0 = the whole range is tested, so nothing is ever said.
    var testedStepsFloor: Int = 0
    /// What to say below it. The wording is the PRESET's, not the helper's:
    /// which adapter unlocks the low end, and what goes wrong without one, is
    /// a fact about this checkpoint.
    var testedStepsFloorNote: String = ""
    /// Lowest frame count we have a verdict on, same contract as
    /// `testedStepsFloor`. 0 = no advisory.
    var testedFrameFloor: Int = 0
    /// What to say below it, appended to the clip's own length.
    var testedFrameFloorNote: String = ""

    /// Advice under the Steps slider when the user has gone below the tested
    /// floor. `distilled` = a few-step adapter is engaged (the engine-owned
    /// Turbo toggle, or any attached Style LoRA) — that is what the low end is
    /// FOR, so warning there would argue against the setup the user just made.
    func stepsAdvisory(steps: Int, distilled: Bool) -> String? {
        guard testedStepsFloor > 0, steps < testedStepsFloor, !distilled else { return nil }
        return testedStepsFloorNote
    }

    /// Advice under the Frames slider for a clip shorter than the model's own
    /// stated range. Short clips stay OFFERED: the engine generates them, and
    /// a 1-second test is how you find a step count without paying for a
    /// 20-minute render.
    func framesAdvisory(_ frames: Int) -> String? {
        guard testedFrameFloor > 0, frames < testedFrameFloor else { return nil }
        let seconds = Double(frames) / Double(max(1, fps))
        return String(format: "%.1f s — ", seconds) + testedFrameFloorNote
    }
    /// Weights resident DURING sampling, in GB. Distinct from `approxRAMGB`,
    /// which is the staged LOAD peak (`max(TE, DiT) + VAEs`): by the time the
    /// DiT is stepping, the text encoder has been run and freed. Only the H3
    /// memory model reads it; 0 means "not modelled".
    var ditResidentGB: Double = 0
    /// The staged LOAD peak from the pack's actual file sizes, in GiB.
    /// `approxRAMGB` is the rounded-up version of this and drives the coarse
    /// "does this Mac have enough RAM at all" alert, where erring high is free.
    /// The frame model is not that: rounding 22.8 up to 26 costs a 32 GB Mac
    /// the 4-bit pack it exists for. 0 falls back to `approxRAMGB`.
    var stagedPeakGB: Double = 0
    /// The pack carries its OWN text encoder. LTX 2.3 borrows the shared
    /// `mlx-community/gemma-3-12b-it-4bit` download (also a chat model, so it
    /// is a separate bundle component); 2.5 ships a Lightricks fine-tune of
    /// Gemma-4-12B in its own subdirectory, which the server resolves from
    /// inside the pack rather than from `~/.mlx-serve/models`. Declared,
    /// because it decides the download bundle's SHAPE — a 2.5 pack that also
    /// pulled the shared encoder would fetch 8 GB it never opens, and one
    /// that didn't ship its own would generate against the wrong encoder.
    var shipsOwnTextEncoder: Bool = false
    /// LTX's own DiffVAE decoder (`vae_diffusion_decoder.safetensors`), which
    /// their published clips are decoded with — a small NA diffusion model that
    /// denoises the pixels the plain conv decoder interpolates. DECLARED per
    /// preset, because only the 8-bit pack ships the file: the 4-bit one does
    /// not, and a toggle that always 400s is worse than no toggle. The field is
    /// omitted from the request entirely when this is false.
    var supportsDiffusionDecoder: Bool = false

    static func == (lhs: Self, rhs: Self) -> Bool { lhs.id == rhs.id }
    func hash(into hasher: inout Hasher) { hasher.combine(id) }

    func settings(_ q: QualityPreset) -> VideoQualitySettings {
        qualityProfiles[q] ?? qualityProfiles[defaultQuality]!
    }

    /// The grid this backend samples on. Unlike the image side this is a
    /// FUNCTION, not a property, because LTX's alignment depends on the
    /// PIPELINE: `handleVideo` refuses a non-/32 canvas outright ("width and
    /// height must be multiples of 32"), and a two-stage pipeline denoises at
    /// half the canvas so it demands /64 as its own named 400. Same model, two
    /// grids — a constant could only ever be right for one of them.
    ///
    /// Video is also where this matters most: the image backends silently
    /// rewrite an off-grid size, so a bad guess costs a slightly different
    /// picture. Here it costs the whole generation.
    ///
    /// The ceiling is the largest canvas the preset itself ships; the real
    /// bound past that is memory and time, which `H3Plan`/`H3TimeEstimate` and
    /// the frame ladder already model and surface.
    func resolutionGrid(twoStage: Bool) -> ResolutionGrid {
        let maxDim = resolutions.map { max($0.width, $0.height) }.max() ?? 1024
        switch backend {
        // H3 has no two-stage pipeline at all, and its own fastest canvases
        // (544, 672, 960) are /32 and not /64 — applying LTX's two-stage grid
        // here would refuse the model's own shipped rows.
        case .minimaxH3:
            return ResolutionGrid(alignment: 32, minDim: 256, maxDim: maxDim)
        case .ltx:
            return ResolutionGrid(alignment: twoStage ? 64 : 32, minDim: 256, maxDim: maxDim)
        }
    }

    /// The picker's rows plus the Custom… sentinel, which goes last so the
    /// tuned canvases stay the obvious pick.
    func resolutionOptions() -> [ResolutionOption] { resolutions + [.custom] }

    /// Which prompt FORMAT this model expects — the one chokepoint the pane's
    /// examples, placeholder, hint and tips link all read. Derived from the
    /// capabilities the preset already declares, never sniffed from the id.
    ///
    /// H3 is not prompted in prose: MiniMax's own pipeline expands the user's
    /// request into a labelled document (H3-Context-IR, API-only) before the
    /// DiT ever sees it, and we send the prompt verbatim. The REF2VA pack takes
    /// the six-section full-reference format, the FL2VA packs the three-field
    /// base format. See `H3PromptExamples`.
    var promptFormat: VideoPromptFormat {
        switch backend {
        case .ltx: return .ltx
        case .minimaxH3: return supportsReferences ? .h3Reference : .h3Base
        }
    }

    /// Frame count that COVERS an attached audio clip: the smallest ladder
    /// value ≥ duration×fps, capped at the model's max (a longer clip is
    /// trimmed to the video by the server). nil for a zero/invalid duration.
    func framesCovering(durationSeconds: Double) -> Int? {
        guard durationSeconds > 0, let cap = frameOptions.last else { return nil }
        let needed = Int((durationSeconds * Double(fps)).rounded(.up))
        return frameOptions.first(where: { $0 >= needed }) ?? cap
    }

    /// LTX trained resolutions, every edge divisible by **64**.
    ///
    /// Not cosmetic: the Quality and Super Quality tiers run two-stage
    /// pipelines whose first stage is HALF resolution, and the latent grid is
    /// /32 — so a full-resolution edge that is only /32 leaves stage 1 on a
    /// fractional grid and the server refuses it by name. The list used to
    /// carry 704×480 and 480×704 (and defaulted to one of them), so picking
    /// Quality on a fresh install was an immediate 400.
    ///
    /// 448 replaces 480 as the short edge: it is the nearest /64 value, so the
    /// two landscape/portrait pairs keep their place in the list. Pinned by
    /// `testEveryLtxResolutionSurvivesTheTwoStagePipelines`.
    private static let ltxResolutions: [ResolutionOption] = [
        .init(width: 704,  height: 448, label: "704 × 448 (landscape 14:9) — fastest"),
        .init(width: 448,  height: 704, label: "448 × 704 (portrait 9:14)"),
        .init(width: 768,  height: 512, label: "768 × 512 (landscape 3:2)"),
        .init(width: 512,  height: 768, label: "512 × 768 (portrait 2:3)"),
        .init(width: 1024, height: 576, label: "1024 × 576 (landscape 16:9)"),
        .init(width: 576,  height: 1024, label: "576 × 1024 (portrait 9:16)"),
        .init(width: 1600, height: 896, label: "1600 × 896 (landscape 16:9) — recommended"),
        .init(width: 896,  height: 1600, label: "896 × 1600 (portrait 9:16)"),
        .init(width: 1920, height: 1088, label: "1920 × 1088 (landscape 16:9) — LTX's own canvas, slowest"),
        .init(width: 1088, height: 1920, label: "1088 × 1920 (portrait 9:16) — slowest"),
    ]

    /// Ceiling on ONE generation's raw RGB volume. The server base64s the whole
    /// `frames.rgb` buffer into a single JSON body and the app decodes it in
    /// memory, so both sides hold the volume plus its base64 at once — a
    /// 193-frame 1920×1088 clip is 1.2 GB raw, 1.6 GB encoded, and that is a
    /// hang followed by a death rather than a slow render. 768 MB is twice the
    /// largest volume we already ship (H3's 124f at 1344×768 = 384 MB).
    /// Lifting it is a TRANSPORT change (stream the frames, or mux server-side)
    /// — not a number to raise on its own.
    static let maxFramePayloadBytes = 768 * 1024 * 1024

    /// Frame counts offerable at this canvas: the `8N+1` ladder trimmed to what
    /// one response can carry. Always returns at least the first rung, so the
    /// picker can never render blank.
    func frameOptions(width: Int, height: Int, chainWindows: Int = 1) -> [Int] {
        let perFrame = max(1, width * height * 3)
        let budget = Self.maxFramePayloadBytes / perFrame
        // Chained windows deliver `w*n - (w-1)` frames in ONE response (#283).
        let w = max(1, chainWindows)
        let fits = frameOptions.filter { $0 * w - (w - 1) <= budget }
        return fits.isEmpty ? Array(frameOptions.prefix(1)) : fits
    }

    /// Ceiling for the AUTO-picked default canvas. This is a TIME budget, not a
    /// memory one: the RAM rule below would hand a 64 GB Mac 1920×1088, and the
    /// default tier is the one people press without thinking. 1920×1088 stays
    /// one click away in the picker for finals.
    private static let autoCanvasCapPixels = 1600 * 896

    /// Stage 1 of a two-stage render must itself be a canvas this model would
    /// be asked to fill: the pipeline denoises at HALF the requested size and
    /// upscales, so "Quality" at 768×512 is a 384×256 render — visibly softer
    /// than the one-stage tier sitting above it in the same menu. The smallest
    /// canvas we offer is the floor, which makes 1408×896 the smallest size
    /// where the tier pays for itself (and 1600×896 the first one on the
    /// ladder). nil = no note; the backend without a two-stage tier never
    /// carries one.
    func twoStageCanvasNote(width: Int, height: Int) -> String? {
        guard QualityPreset.allCases.contains(where: { settings($0).mode != .oneStage }) else { return nil }
        guard let smallest = resolutions.map({ $0.width * $0.height }).min() else { return nil }
        let halfArea = (width / 2) * (height / 2)
        guard halfArea < smallest else { return nil }
        return "Quality and Super Quality denoise at half this size (\(width / 2) × \(height / 2)) and upscale — "
             + "below 1600 × 896 they can look softer than the one-stage tiers, not sharper."
    }

    /// Default canvas for THIS Mac. A single static default has to be safe on
    /// the smallest supported machine, which is how a 128 GB Mac ended up
    /// rendering 0.39 MP previews while LTX's own pipeline defaults to 2.09 MP.
    /// Pure function of physical memory (mirrors `RecommendedModelPick`), so it
    /// is testable off-machine: the largest rung that can still hold the
    /// default tier's frame count, capped by `autoCanvasCapPixels`, and the
    /// smallest rung when the model does not comfortably fit at all.
    func recommendedResolution(totalGB: Int) -> ResolutionOption {
        let frames = settings(defaultQuality).numFrames
        let area = { (r: ResolutionOption) in r.width * r.height }
        let fits = resolutions.filter {
            area($0) <= Self.autoCanvasCapPixels
                && RAMChecker.safeFrameCap(model: self, width: $0.width, height: $0.height,
                                           available: totalGB) >= frames
        }
        // Ties are the portrait twin of the same canvas — resolve them by
        // LADDER ORDER (landscape first), never by whichever the sort emitted
        // last, or a fresh install opens in portrait for no stated reason.
        let target = fits.map(area).max() ?? resolutions.map(area).min()
        guard let want = target else { return defaultResolution }
        return resolutions.first { area($0) == want } ?? defaultResolution
    }

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
                // One-stage runs the distilled transformer, whose sigma table
                // is fixed at 8 — the server clamps anything else and logs it,
                // so asking for 12 only made the pane's hint lie.
                .good:         .init(mode: .oneStage,   steps: 8,  cfgScale: 1.0, stgScale: 0.0, numFrames: 97),
                .quality:      .init(mode: .twoStage,   steps: 30, cfgScale: 3.0, stgScale: 1.0, numFrames: 97),
                .superQuality: .init(mode: .twoStageHQ, steps: 15, cfgScale: 3.0, stgScale: 0.0, numFrames: 97),
            ],
            defaultQuality: .good,
            maxFrames: cap,
            frameOptions: frameLadder(maxFrames: cap),
            description: "Generates short video clips from a text prompt (and optionally a starting image or audio track), with sound built in. The heaviest model here — it also pulls a Gemma text encoder on first use.",
            supportsLastFrame: true
        )
    }()

    /// LTX-2.5 (August 2026). The DiT is 2.3's key template minus the 96
    /// video-FF biases plus one `keyframes_abs_pos_embedding`, and the
    /// connector, both VAEs, the vocoder and both upscalers re-ship
    /// byte-identical — so every resolution, frame and quality number here is
    /// deliberately 2.3's. What changed is the text encoder: a Lightricks
    /// fine-tune of Gemma-4-12B, shipped INSIDE the pack, which is why this
    /// one pulls nothing extra on first run.
    static let ltx25Q4: VideoModelPreset = {
        let cap = 193
        return VideoModelPreset(
            id: "ddalcu/LTX-2.5-MLX-Serve-4bit",
            name: "LTX-Video 2.5 Q4 (with audio, ~36 GB)",
            repo: "ddalcu/LTX-2.5-MLX-Serve-4bit",
            // Self-contained: the ~6.7 GB encoder is part of the pack, so the
            // first-run figure is the same number rather than +8 GB.
            approxDownloadGB: 36,
            approxFirstRunDownloadGB: 36,
            approxRAMGB: 24,
            resolutions: ltxResolutions,
            defaultResolution: ltxResolutions[0],
            fps: 24,
            qualityProfiles: [
                .fast:         .init(mode: .oneStage,   steps: 8,  cfgScale: 1.0, stgScale: 0.0, numFrames: 49),
                // One-stage runs the distilled transformer, whose sigma table
                // is fixed at 8 — the server clamps anything else and logs it,
                // so asking for 12 only made the pane's hint lie.
                .good:         .init(mode: .oneStage,   steps: 8,  cfgScale: 1.0, stgScale: 0.0, numFrames: 97),
                .quality:      .init(mode: .twoStage,   steps: 30, cfgScale: 3.0, stgScale: 1.0, numFrames: 97),
                .superQuality: .init(mode: .twoStageHQ, steps: 15, cfgScale: 3.0, stgScale: 0.0, numFrames: 97),
            ],
            defaultQuality: .good,
            maxFrames: cap,
            frameOptions: frameLadder(maxFrames: cap),
            description: "The newest LTX. Generates short video clips with sound from a text prompt, and optionally a starting image or audio track. Ships its own text encoder, so nothing extra downloads on first use.",
            supportsLastFrame: true,
            shipsOwnTextEncoder: true
        )
    }()

    /// LTX-2.5 at 8 bits — the quality pack. Same weights, same recipe, same
    /// files as the 4-bit one; only the DiT/text-encoder width differs.
    ///
    /// Affine 4-bit group-64 injects ~9.9% relative noise into every one of the
    /// 1632 quantized linears (measured off the packs: quantizer step over
    /// per-group weight std); 8 bits is ~0.6%, and the same clip at the same
    /// seed keeps a face, legs and fur the 4-bit render loses. It costs 3.5% of
    /// generation time (measured 175 s vs 169 s, 97f at 768x512 one-stage) — the
    /// DiT is compute-bound at these token counts, so the wider weights are
    /// nearly free. 4-bit stays for Macs that cannot hold this one; every
    /// community LTX int4 pack uses ConvRot/W4A8 rather than plain affine.
    static let ltx25Q8: VideoModelPreset = {
        let cap = 193
        return VideoModelPreset(
            id: "ddalcu/LTX-2.5-MLX-Serve-8bit",
            name: "LTX-Video 2.5 (8-bit, best quality, ~59 GB)",
            repo: "ddalcu/LTX-2.5-MLX-Serve-8bit",
            approxDownloadGB: 59,
            approxFirstRunDownloadGB: 59,
            approxRAMGB: 42,
            resolutions: ltxResolutions,
            defaultResolution: ltxResolutions[2],
            fps: 24,
            qualityProfiles: [
                .fast:         .init(mode: .oneStage,   steps: 8,  cfgScale: 1.0, stgScale: 0.0, numFrames: 49),
                .good:         .init(mode: .oneStage,   steps: 8,  cfgScale: 1.0, stgScale: 0.0, numFrames: 97),
                .quality:      .init(mode: .twoStage,   steps: 30, cfgScale: 3.0, stgScale: 1.0, numFrames: 97),
                .superQuality: .init(mode: .twoStageHQ, steps: 15, cfgScale: 3.0, stgScale: 0.0, numFrames: 97),
            ],
            defaultQuality: .good,
            maxFrames: cap,
            frameOptions: frameLadder(maxFrames: cap),
            description: "The sharpest LTX we ship. Same model as the 4-bit pack at twice the weight precision — noticeably more detail in faces, fur and fine texture, for about the same generation time. Needs a Mac with plenty of memory.",
            supportsLastFrame: true,
            shipsOwnTextEncoder: true,
            supportsDiffusionDecoder: true
        )
    }()

    /// The server sends width/height straight to the DiT with no resampling,
    /// so these must already be on the model's trained canvas — no smaller
    /// tier exists (MiniMax's H3-Base generates natively at 768p; 2K needs a
    /// separate regenerate stage we haven't converted). Computed via
    /// `adapt_canvas`/`minimax_h3.adaptCanvas`'s own math (768px short edge,
    /// ≤768×1344 area, /32) over MiniMax's stated six aspect ratios (21:9 to
    /// 9:16). 1344×768 is the reference node's default and the confirmed-good
    /// rap demo's resolution. Speed labels are pixel-count ratios vs the
    /// smallest option (768×768 = 0.59MP), squared — the DiT attends over one
    /// packed sequence, so cost is roughly quadratic in pixel count: 4:3/3:4
    /// (0.79MP) ≈1.8x, and 16:9/9:16/21:9 all land on the same area cap
    /// (1.03MP) ≈3.1x. Estimates, not measured per-resolution timings.
    /// Speed labels are RELATIVE TO 960x544, the cheapest canvas on the list.
    ///
    /// Cost is not proportional to pixels: the DiT runs full bidirectional
    /// attention over the packed [text | cond | audio | video] sequence, so the
    /// projections scale with the row count and attention scales with its
    /// SQUARE. Measured on an M5 Max at a matched 124 frames, 1344x768 costs
    /// 2.9x the time of 960x544 for 1.98x the pixels — the gap is the S² term.
    /// The remaining ratios come from that same model (row count = w/32 × h/32
    /// per latent frame), which reproduces the measured pair within 6%.
    ///
    /// 960x544 is below the model's 768 native short edge, which is why it is
    /// offered as the long-form canvas rather than made the default: at 124
    /// frames 1344x768 resolves genuinely more detail (+25% high-frequency
    /// energy at matched display size). It is what makes the top of the 17k+5
    /// ladder — 362 frames, 15 s in ONE generation — practical at all.
    private static let h3Resolutions: [ResolutionOption] = [
        .init(width: 1344, height: 768,  label: "1344 × 768 (16:9 widescreen) — most detail, 2.9x slower"),
        .init(width: 960,  height: 544,  label: "960 × 544 (16:9 widescreen) — fastest, best for long clips"),
        .init(width: 768,  height: 768,  label: "768 × 768 (square) — 1.2x slower"),
        .init(width: 1024, height: 768,  label: "1024 × 768 (4:3 landscape) — 1.8x slower"),
        .init(width: 768,  height: 1024, label: "768 × 1024 (3:4 portrait) — 1.8x slower"),
        .init(width: 544,  height: 960,  label: "544 × 960 (9:16 portrait) — fastest, best for long clips"),
        .init(width: 768,  height: 1344, label: "768 × 1344 (9:16 portrait) — 2.9x slower"),
        .init(width: 1536, height: 672,  label: "1536 × 672 (21:9 cinematic) — 2.9x slower"),
    ]

    /// H3's frame ladder is `17k + 5`, NOT LTX's `8N + 1` (its VAE folds 17
    /// source frames into 5 latent tokens) — offering a count off it means the
    /// server silently snaps it. The floor is the ENGINE's (5, from
    /// `temporalShape`'s `max(5, length)`), not the reference node's
    /// trained-range start: 124 frames at 960x544 is a 20-minute job, so
    /// "shorter is off-distribution" made the cheapest way to try a prompt or
    /// a step count — a one-second clip — unreachable at any setting, and left
    /// the server's OWN default length (56) off the slider. Below the stated
    /// range the pane says so (`testedFrameFloor`).
    private static func h3FrameLadder(minFrames: Int, maxFrames: Int) -> [Int] {
        var out: [Int] = []
        var n = minFrames
        while n <= maxFrames {
            out.append(n)
            n += 17
        }
        return out
    }

    /// Both H3 packs share everything except repo/size/RAM: same engine, same
    /// frame ladder, same fast recipe. One factory so they cannot drift.
    private static func minimaxH3Preset(repo: String, name: String,
                                        downloadGB: Int, ramGB: Int, ditGB: Double, stagedGB: Double,
                                        description: String,
                                        supportsReferences: Bool = false) -> VideoModelPreset {
        // The model's own range: MiniMax states 4-15 s at 24 fps, and the
        // reference node's trained range is ~124-362 on the 17k+5 ladder. The
        // SLIDER goes down to the engine's floor anyway and warns below the
        // stated range; the quality tiers below still start at 124, so the
        // short end is somewhere you steer to, never somewhere you land.
        //
        // The ceiling used to be 209 (the rap demo, the longest clip we had
        // shipped a verdict on) with a comment calling 362 "untested-by-us" —
        // stale by the time it was read: 362 frames at 960x544 ran in ONE
        // generation on an M5 Max at 139.6 s/step. Length is bounded by MEMORY
        // and TIME, and both are now modelled and shown (`H3Plan`,
        // `H3TimeEstimate`) instead of hidden behind a cap.
        let minF = 5
        let cap = 362
        // Where the tiers and the stale-value clamp start: the reference
        // node's trained-range start, and the lowest rung at or above
        // MiniMax's stated 4 s (107 = 4.5 s; 90 = 3.75 s is under it).
        let tierMin = 124
        let statedMin = 107
        // What the quality TIERS pick. Deliberately not `cap`: the ladder going
        // to 362 is the slider's reach, but a preset is a default, and at
        // 1344x768 the top of the ladder is a five-hour job. Picking "Quality"
        // must not silently start one — the user gets there by moving the
        // slider, with the estimate under it saying what it costs.
        let tierMax = 209
        return VideoModelPreset(
            id: repo,
            name: name,
            repo: repo,
            approxDownloadGB: downloadGB,
            // Self-contained: weights, both VAEs and the tokenizer ship in the
            // one repo, so there is no separate text-encoder pull.
            approxFirstRunDownloadGB: downloadGB,
            // The text encoder and the DiT are staged sequentially (they cannot
            // both be resident), so the peak is the larger of the two plus
            // activations, not their sum.
            approxRAMGB: ramGB,
            resolutions: h3Resolutions,
            defaultResolution: h3Resolutions[0],
            fps: 24,
            // .good = the eyeballed capstone A/B; .quality = the rap demo's
            // longer 209-frame run.
            qualityProfiles: [
                .fast:         .init(mode: .oneStage, steps: 16, cfgScale: 1.0, stgScale: 0.0, numFrames: tierMin),
                .good:         .init(mode: .oneStage, steps: 30, cfgScale: 1.0, stgScale: 0.0, numFrames: tierMin),
                .quality:      .init(mode: .oneStage, steps: 30, cfgScale: 1.0, stgScale: 0.0, numFrames: tierMax),
                .superQuality: .init(mode: .oneStage, steps: 50, cfgScale: 1.0, stgScale: 0.0, numFrames: tierMax),
            ],
            defaultQuality: .good,
            maxFrames: cap,
            frameOptions: h3FrameLadder(minFrames: minF, maxFrames: cap),
            description: description,
            backend: .minimaxH3,
            // Both partitions take stacked adapters: the server resolves them
            // against H3's own module names and sums them with Turbo. No
            // community H3 LoRA exists yet beyond the distillation, so this is
            // a capability, not a recommendation.
            supportsLoRA: true,
            supportsCFG: false,
            supportsPipelineModes: false,
            supportsAudioInput: false,
            generatesAudio: true,
            supportsFastRecipe: true,
            supportsReferences: supportsReferences,
            // Turbo and chaining both ride fl2va machinery (the LoRA is
            // untested on the REF2VA DiT; a reference has no keyframe row to
            // chain through), so both flags are the partition's complement —
            // derived here so the two fl2va presets cannot drift apart.
            supportsTurbo: !supportsReferences,
            supportsChainedWindows: !supportsReferences,
            supportsLastFrame: !supportsReferences,
            // MiniMax publishes no step count at all — no default, no range, no
            // maximum — so this range is OURS. It used to START at 16, the
            // lowest tier we had a verdict on, which locked out every distilled
            // few-step adapter the pane's own LoRA slot can load — including
            // REF2VA Turbo distillations, on the one pack whose Turbo toggle
            // does not exist (#254). The floor is now the server's (4, turbo's
            // own), and 16 survives as `testedStepsFloor` advice. Below ~6 the
            // fast recipe's warmup and tail windows still cover the whole
            // schedule, so an undistilled run there pays full price per step
            // while looking like the cheap option — which is what the note says.
            stepsRange: 4...50,
            stepsHelp: "More steps mean more detail and steadier motion, and cost time in direct proportion. 16 is the fast tier, 30 is the default, 50 is for a final render. Fewer than 16 only works with a distilled few-step adapter.",
            testedStepsFloor: 16,
            testedStepsFloorNote: "Under 16 steps this model needs a distilled few-step adapter — the engine-owned Turbo LoRA, or a community one attached under Style LoRAs. Without one the picture is rough and the soundtrack usually comes out garbled, and the fast recipe means those steps are not much cheaper each.",
            testedFrameFloor: statedMin,
            testedFrameFloorNote: "below MiniMax's stated 4-second minimum. Good for trying a prompt or a step count cheaply; motion and the soundtrack degrade outside the trained range.",
            ditResidentGB: ditGB,
            stagedPeakGB: stagedGB
        )
    }

    static let minimaxH3: VideoModelPreset = minimaxH3Preset(
        repo: "ddalcu/MiniMax-H3-FL2VA-MLX-Serve-8bit",
        name: "MiniMax-H3 (Hailuo 3.0) 8-bit — video + native audio",
        downloadGB: 69,
        ramGB: 44,
        ditGB: 20.46,
        stagedGB: 38.24,
        description: "Generates video with a matching stereo soundtrack from one prompt — the sound is produced jointly with the picture, not dubbed on after. Describe the scene, then the audio you want after 'overall_soundscape:'. Clips run from 5 to 15 seconds. It is slow, and how slow depends heavily on the settings — the estimate under the Generate button is live, and 960 × 544 is several times quicker than the widescreen canvas."
    )

    /// The 4-bit pack: same speed (the DiT is compute-bound), ~40 GB download
    /// and a ~25 GB staged peak — the pick for 32-48 GB Macs.
    static let minimaxH3Q4: VideoModelPreset = minimaxH3Preset(
        repo: "ddalcu/MiniMax-H3-FL2VA-MLX-Serve-4bit",
        name: "MiniMax-H3 (Hailuo 3.0) 4-bit — video + native audio, low RAM",
        downloadGB: 40,
        ramGB: 26,
        ditGB: 11.11,
        stagedGB: 22.82,
        description: "The 4-bit build of MiniMax-H3: same generation speed, a 40 GB download instead of 69, and it fits comfortably on 32 GB Macs. Slightly softer detail than the 8-bit build — pick that one if you have 48 GB or more."
    )

    /// The REF2VA partition: same engine and geometry, a DiT trained to follow
    /// reference images / videos / audio for character, style and scene
    /// continuity. It cannot do first/last-frame conditioning, and FL2VA cannot
    /// do references — they are two checkpoints, not two modes of one.
    static let minimaxH3Ref2VA: VideoModelPreset = minimaxH3Preset(
        repo: "ddalcu/MiniMax-H3-REF2VA-MLX-Serve-8bit",
        name: "MiniMax-H3 (Hailuo 3.0) REF2VA 8-bit — references + native audio",
        downloadGB: 69,
        ramGB: 44,
        ditGB: 20.46,
        stagedGB: 38.24,
        description: "The reference-conditioned MiniMax-H3: attach up to 9 images, 3 clips and 3 audio references and the generation follows them for character, style and scene continuity. Refer to them in the prompt as <Picture 1>, <Video 1>, <Audio 1>. Same speed and soundtrack as the standard pack; every reference adds tokens to every sampling step, so more references means slower.",
        supportsReferences: true
    )

    // 8-bit first: it is the one to reach for on a Mac that can hold it, and
    // the picker features the first entry of each capability group.
    static let all: [VideoModelPreset] = [.ltx25Q8, .ltx25Q4, .ltx23Q4, .minimaxH3, .minimaxH3Q4, .minimaxH3Ref2VA]
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
    var id: String
    var name: String
    /// Open `mlx-community` Qwen3-TTS repo (downloaded via DownloadManager).
    var repo: String
    /// Rough on-disk weight size, GB (first-run download). Shown in the picker.
    let approxDownloadGB: Double
    /// Peak unified-memory footprint, GB — drives the soft RAM gate.
    let approxRAMGB: Int
    /// Suggested reference-clip length for good cloning, in seconds. Surfaced
    /// as a hint next to the record button.
    let recommendedRefSeconds: Int
    /// Plain-English explanation shown under the model in the Media pane.
    let description: String
    /// Whether the backend can clone from a reference clip. Kokoro CANNOT —
    /// asking it to is a named 400, not a silently plain-voiced answer — so the
    /// preset declares it and the UI hides the clip control rather than
    /// offering a dead one (the `ImageModelPreset.supportsImg2Img` rule).
    var supportsCloning: Bool = true
    /// Built-in voice ids, empty when the backend has none. A comma-separated
    /// selection BLENDS them server-side.
    var builtInVoices: [String] = []

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

    /// Kokoro-82M — the fast path. Non-autoregressive, so it is ~17x realtime
    /// against Qwen3-TTS's ~1.3x, at a fifth of the download and a tenth of the
    /// RAM. No cloning; instead 54 built-in voices that BLEND by naming several
    /// (`"af_bella,af_sky"`).
    ///
    /// VOICE MODE ONLY — deliberately absent from `all` (see below). Everything
    /// that speaks with it names this preset directly.
    static let kokoro82M = AudioModelPreset(
        id: "kokoro/kokoro-82m",
        name: "Kokoro 82M (fastest, ~330 MB)",
        repo: "ddalcu/Kokoro-82M-MLX-Serve",
        approxDownloadGB: 0.35,
        approxRAMGB: 1,
        recommendedRefSeconds: 0,
        description: "A tiny, very fast voice model with 54 built-in voices you can blend together. No cloning — pick or mix a voice instead.",
        supportsCloning: false,
        builtInVoices: kokoroVoices
    )

    /// The 54 published Kokoro voices. Prefix = language + gender
    /// (a=American, b=British, e/f/h/i/j/p/z = other languages; f/m = female/male).
    static let kokoroVoices: [String] = [
        "af_alloy", "af_aoede", "af_bella", "af_heart", "af_jessica", "af_kore",
        "af_nicole", "af_nova", "af_river", "af_sarah", "af_sky",
        "am_adam", "am_echo", "am_eric", "am_fenrir", "am_liam", "am_michael",
        "am_onyx", "am_puck", "am_santa",
        "bf_alice", "bf_emma", "bf_isabella", "bf_lily",
        "bm_daniel", "bm_fable", "bm_george", "bm_lewis",
        "ef_dora", "em_alex", "em_santa", "ff_siwis",
        "hf_alpha", "hf_beta", "hm_omega", "hm_psi",
        "if_sara", "im_nicola",
        "jf_alpha", "jf_gongitsune", "jf_nezumi", "jf_tebukuro", "jm_kumo",
        "pf_dora", "pm_alex", "pm_santa",
        "zf_xiaobei", "zf_xiaoni", "zf_xiaoxiao", "zf_xiaoyi",
        "zm_yunjian", "zm_yunxi", "zm_yunxia", "zm_yunyang",
    ]

    /// Presets the MEDIA panes may offer, ordered lightest → heaviest. Default
    /// (`first`) is the 8-bit Qwen3-TTS 0.6B; bf16 builds stay as fidelity
    /// fallbacks.
    ///
    /// Cloning-capable ONLY: AudioGenView's reference-clip control and
    /// AudioGenService's `ref_audio` both assume it, and Kokoro answers
    /// `ref_audio` with a named 400. Keeping Kokoro out makes that impossible
    /// BY CONSTRUCTION rather than by list ordering.
    static let all: [AudioModelPreset] = [.qwen3TTS06B8bit, .qwen3TTS06B, .qwen3TTS17B8bit, .qwen3TTS17B]

    /// Every audio preset including voice-mode-only backends — for the model
    /// browser and the catalogue guards, never for a media pane's picker.
    static let allIncludingVoiceOnly: [AudioModelPreset] = all + [.kokoro82M]
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
    var id: String
    var name: String
    /// Model directory under `~/.mlx-serve/models`. A `local/` prefix marks a
    /// convert-on-device model (no HF pull); any other prefix is a normal repo.
    var repo: String
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

/// Which music ENGINE a checkpoint drives. The two families share the
/// endpoint and nothing else: ACE-Step reads the whole musical-metadata knob
/// set, MiniMax Music 3 rejects every one of those fields BY NAME and
/// requires lyrics — so the family gates the FIELDS (request body + sidecar),
/// not just the pane's controls.
enum MusicEngineFamily {
    case acestep
    case minimaxMusic3
}

/// Music-generation checkpoints (ACE-Step + MiniMax Music 3, the music arms
/// of the audio backend). Same local-convert convention as `Model3DModelPreset`.
struct MusicModelPreset: Identifiable, Hashable {
    var id: String
    var name: String
    /// Model directory under `~/.mlx-serve/models`. A `local/` prefix marks a
    /// convert-on-device model (no HF pull); any other prefix is a normal repo.
    var repo: String
    /// Engine family — decides the knob set, duration range and bundle layout.
    var family: MusicEngineFamily = .acestep
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

    /// timesignature/vocal_language — ACE-Step conditioning fields MiniMax's
    /// card documents no equivalent for, so the server still names each a 400
    /// on Music 3 and the pane hides them there.
    var supportsMusicalMeta: Bool { family == .acestep }

    /// Tempo and key, which BOTH engines support — they are conditioning
    /// fields on ACE-Step and caption text on Music 3 (Global Metadata on
    /// MiniMax's card; its example caption reads "BPM: 96. Key: C major."").
    /// The pane used to hide them on Music 3 along with the two genuinely
    /// unsupported knobs, which read as "this model can't do tempo".
    var supportsTempoAndKey: Bool { true }
    /// Music 3 is lyric-conditioned; the server 400s empty lyrics. ACE-Step
    /// defaults empty lyrics to "[Instrumental]".
    var requiresLyrics: Bool { family == .minimaxMusic3 }
    /// Server-valid duration bounds (ACE [10,600]; Music 3 [1,360], floored
    /// at 5 for a usable slider).
    var durationRange: ClosedRange<Double> {
        family == .acestep ? 10...600 : 5...360
    }

    /// Music 3 takes `steps` in [4,100] — the flow-match refinement passes.
    /// ACE-Step Turbo is distillation-fixed at 8 and the server IGNORES the
    /// field there, so exposing it would be a control that visibly does
    /// nothing. `fixedSteps` stays the per-checkpoint default either way.
    var supportsSteps: Bool { family == .minimaxMusic3 }
    var stepsRange: ClosedRange<Int> { 4...100 }
    /// Reference audio (server `ref_audio`, #259): ACE-Step feeds a 30 s
    /// window of the clip's VAE latent into its timbre slot — ONE pooled
    /// token among hundreds of lyric/text tokens, so it is style/timbre
    /// guidance, never a cover (that mode needs the FSQ codes we don't
    /// ship). Music 3 has no such slot and names the field a 400. Gates the
    /// control AND the field.
    var supportsReferenceAudio: Bool { family == .acestep }
    /// Source-audio tasks (server `task` + `src_audio`): ACE-Step's `cover`
    /// (the source's melody/structure through its FSQ codes, new caption) and
    /// `complete` (vocal-to-BGM: the source latent as context, an instrument
    /// list in the instruction). Music 3 names both fields a 400. Gates the
    /// mode control AND every field it brings.
    var supportsSourceAudio: Bool { family == .acestep }

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

    /// MiniMax Music 3, 8-bit — hierarchical AR (8B LLM + depth decoder)
    /// driving a flow-matching DiT; full songs with sung lyrics at 44.1 kHz.
    static let miniMaxMusic3_8bit = MusicModelPreset(
        id: "minimax-music3-8bit",
        name: "MiniMax Music 3 (8-bit)",
        repo: "ddalcu/MiniMax-Music3-MLX-Serve-8bit",
        family: .minimaxMusic3,
        approxRAMGB: 20,
        approxDownloadGB: 13.6,
        fixedSteps: 30,
        supportsLyrics: true,
        description: "MiniMax's full-song model: an 8B language model writes the music frame by frame from your style prompt and lyrics, then a diffusion decoder renders it. Slower than ACE-Step, strongest vocals."
    )

    /// Catalog, best-first per family.
    static let all: [MusicModelPreset] = [.acestepXLTurbo8bit, .miniMaxMusic3_8bit]
}

extension MusicGenRequest {
    /// Is the lyrics requirement met? Music 3 is lyric-conditioned and the
    /// server 400s an empty block — but ticking instrumental LIFTS that, or the
    /// checkbox would be unreachable on the one model that most needs it.
    /// The pane's Generate gate and the service's pre-flight read this ONE
    /// answer so they cannot disagree about what is sendable.
    static func lyricsSatisfied(model: MusicModelPreset, lyrics: String,
                                instrumental: Bool) -> Bool {
        if !model.requiresLyrics || instrumental { return true }
        return !lyrics.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
    }
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

    /// The section tags the checkpoints were trained on, verbatim from
    /// MiniMaxAI/MiniMax-Music3's model card. Nothing in the app listed them,
    /// so the vocabulary was guesswork — the helper text said "like [verse] or
    /// [chorus]" and left the other seven undiscoverable.
    static let sectionTags: [String] = [
        "[intro]", "[verse]", "[pre-chorus]", "[chorus]", "[post-chorus]",
        "[bridge]", "[instrumental]", "[solo]", "[outro]",
    ]
    static let sectionTagHint: String = sectionTags.joined(separator: " ")

    /// What the server accepts for `bpm`. The pane used to offer only the ten
    /// anchors below, so 92 was unaskable while the chat tool could send it.
    static let bpmRange: ClosedRange<Int> = 30...300

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

    /// The character a key is conventionally associated with, so the dropdown
    /// reads like the BPM one (whose labels carry a genre anchor) instead of
    /// asking a non-musician to pick between 24 bare note names.
    ///
    /// These are the common Western associations, not physics — equal
    /// temperament makes every major key acoustically identical to every other,
    /// and the associations survive from unequal historical tunings plus
    /// instrument register and repertoire. Useful as a nudge, not a rule; only
    /// the twelve most-used keys get one, the rest show bare.
    /// One word each. Two-word moods made the longest row
    /// ("A minor — natural, plain sad") wider than any sane menu, and a
    /// truncated label is worse than no label.
    static let keyMoods: [String: String] = [
        "C major": "open",
        "G major": "pastoral",
        "D major": "triumphant",
        "A major": "sunny",
        "E major": "brilliant",
        "F major": "gentle",
        "Bb major": "brassy",
        "Eb major": "heroic",
        "A minor": "plain sad",
        "E minor": "wistful",
        "D minor": "solemn",
        "C minor": "stormy",
        "B minor": "yearning",
        "F# minor": "moody",
        "G minor": "restless",
    ]

    /// "C major — plain, open", or just "C major" where we have no association.
    static func keyLabel(_ key: String) -> String {
        guard let mood = keyMoods[key] else { return key }
        return "\(key) — \(mood)"
    }

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

    /// Style starters for the pane's Examples menu, per engine family.
    /// ACE-Step reads a one-line genre/mood description and takes tempo, key
    /// and meter from its own controls. Music 3 has none of those controls
    /// (the server 400s the fields) and was trained on a structured caption —
    /// so the caption is the ONLY place its tempo, key and arrangement can be
    /// stated, and a one-liner leaves every one of those to the model.
    static func builtinStyles(for family: MusicEngineFamily) -> [MusicPrompt] {
        family == .minimaxMusic3 ? music3Styles : builtinStyles
    }

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

    /// MiniMax Music 3 captions — the checkpoint's own three-block format
    /// (Global Metadata / Vocal Details / Arrangement) with its labelled
    /// lines, written out in full so a user can edit one line and re-run.
    static let music3Styles: [MusicPrompt] = [
        MusicPrompt(title: "Synthwave", body: """
            Global Metadata
            Basic Attributes: bpm is 118. key is F, and scale is minor. Retro Synthwave / Outrun.
            Global Emotional Progression: Opens cool and mechanical, warms through the first chorus into something wistful and wide-eyed, peaks in a euphoric final chorus, then empties out into calm.
            Application Scenarios & Imagery: A night drive along an empty coastal highway, neon signs smearing past the windscreen.
            Sonics & Production Profile: Wide analog stereo field, warm saturated low end, glassy chorused highs, gated reverb on the snare, a light tape flutter across the whole mix.

            Vocal Details
            Vocal Gender & Timbre: A single female lead, breathy and cool in a comfortable mid register, slightly detached.
            Vocal Style: Long sustained phrases riding over the pulse, conversational in the verses and fully open in the choruses.
            Harmony/Backing Vocals: Octave-up doubles and soft airy thirds in the choruses only.
            Vocal FX: Lush plate reverb and a slow quarter-note delay, always kept intelligible.

            Arrangement
            Instrument Lifecycle Description (Primary/Secondary Layering):
            Primary: An arpeggiated analog synth runs from the first bar to the last, joined by a soaring lead synth in the choruses.
            Secondary: Analog bass and a punchy drum machine enter at the first verse, warm pads sit underneath, extra percussion arrives for the final chorus.
            Groove & Foundation Progression: A steady eighth-note pulse under a driving four-on-the-floor kick, with the bass locking to the arpeggio and opening into longer notes in the chorus.
            Embellishments, Textures & Spatial FX: Slow filter sweeps open each section, a snare fill lifts the pre-chorus, the bridge drops the drums for a single held pad, and the outro strips back to the arpeggio and tape hiss.
            """),
        MusicPrompt(title: "Lo-fi study beats", body: """
            Global Metadata
            Basic Attributes: bpm is 82. key is Eb, and scale is major. Lo-fi Hip Hop / Chillhop.
            Global Emotional Progression: Nostalgic and unhurried from the first bar, drifting a little brighter in the middle and settling back into stillness.
            Application Scenarios & Imagery: Rain on a window at 2am, a desk lamp, a half-finished page.
            Sonics & Production Profile: Narrow warm mix, rolled-off highs, dusty vinyl crackle throughout, gentle tape saturation and soft sidechain breathing on the pads.

            Vocal Details
            Vocal Gender & Timbre: Largely instrumental — the Rhodes piano carries the melody.
            Vocal Style: Occasional wordless female vocal fragments, soft and half-buried.
            Harmony/Backing Vocals: None beyond those fragments.
            Vocal FX: Heavy low-pass and long reverb, used as texture rather than as a lead.

            Arrangement
            Instrument Lifecycle Description (Primary/Secondary Layering):
            Primary: Warm mellow Rhodes chords with a loose swung feel, present the whole track.
            Secondary: Soft boom-bap drums and a round upright bass enter after the intro; a muted guitar figure joins in the middle third and drops away for the outro.
            Groove & Foundation Progression: Laid-back swung sixteenths, snare slightly behind the beat, bass walking simply under the chord changes without ever pushing the tempo.
            Embellishments, Textures & Spatial FX: Vinyl crackle and distant room tone run underneath, a tape-stop marks the halfway point, and the final bars fade to crackle alone.
            """),
        MusicPrompt(title: "Epic orchestral", body: """
            Global Metadata
            Basic Attributes: bpm is 96. key is D, and scale is minor. Cinematic Orchestral / Trailer.
            Global Emotional Progression: Ominous and sparse at the start, gathering tension in waves, breaking into a triumphant, overwhelming climax before a quiet, exhausted resolution.
            Application Scenarios & Imagery: An army cresting a ridge at dawn; the last shot before the credits.
            Sonics & Production Profile: Huge concert-hall depth, deep sub-heavy percussion, wide string sections panned across the stage, generous natural reverb tails and enormous dynamic range.

            Vocal Details
            Vocal Gender & Timbre: A mixed choir, no soloist — deep basses under bright sopranos.
            Vocal Style: Wordless sustained vowels in the build, shifting to hard rhythmic syllables at the climax.
            Harmony/Backing Vocals: Dense block harmony doubling the brass line an octave above.
            Vocal FX: Nothing beyond the hall — the space is the effect.

            Arrangement
            Instrument Lifecycle Description (Primary/Secondary Layering):
            Primary: Low strings open alone with a repeating ostinato that survives every section and returns at the end.
            Secondary: Taiko drums and low brass enter for the first build, horns and full strings for the second, choir and cymbals at the climax, solo cello for the resolution.
            Groove & Foundation Progression: A driving ostinato pulse that doubles in rhythmic density each build, with the percussion moving from a slow half-time thud into hammered eighth notes.
            Embellishments, Textures & Spatial FX: Rising string runs and braams mark each transition, a full silence lands one bar before the climax, and the last thirty seconds decay into a single sustained note.
            """),
        MusicPrompt(title: "Modern pop", body: """
            Global Metadata
            Basic Attributes: bpm is 112. key is A, and scale is major. Contemporary Pop / Dance Pop.
            Global Emotional Progression: Bright and confident from the first line, building anticipation through the pre-chorus and landing on an open, celebratory chorus that gets bigger every time it returns.
            Application Scenarios & Imagery: Windows down in late summer, a city that stays warm after dark.
            Sonics & Production Profile: Loud, tight and radio-forward — punchy compressed drums, a clean wide top end, shimmering synths and a bass that sits right under the vocal.

            Vocal Details
            Vocal Gender & Timbre: A female lead, bright and forward with a light rasp at the top of her range.
            Vocal Style: Rhythmic and clipped in the verses, sustained and belted in the choruses.
            Harmony/Backing Vocals: Stacked thirds and fifths on every chorus line, plus short answering ad-libs between phrases.
            Vocal FX: Tight doubling, a clean short slap delay and a bright plate — polished but not processed into glass.

            Arrangement
            Instrument Lifecycle Description (Primary/Secondary Layering):
            Primary: A bouncy synth bass and the lead vocal drive the whole song.
            Secondary: Muted plucks and finger snaps carry the verses, full drums and shimmering pad chords land at each chorus, a guitar counter-line appears only in the last chorus.
            Groove & Foundation Progression: A four-on-the-floor kick with an offbeat bass bounce, the verse groove stripped to kick and snaps and the chorus opening into full kit with a clap layer.
            Embellishments, Textures & Spatial FX: Riser sweeps and a drum drop-out set up each chorus, the bridge cuts to vocal and pad alone, and the final chorus adds an octave-up vocal stack.
            """),
        MusicPrompt(title: "Acoustic folk", body: """
            Global Metadata
            Basic Attributes: bpm is 72. key is G, and scale is major. Acoustic Folk / Singer-Songwriter.
            Global Emotional Progression: Intimate and tentative at first, growing in warmth and certainty as the arrangement fills, then returning to the quiet it started in.
            Application Scenarios & Imagery: A wooden room, one microphone, rain outside and a fire that has been going a while.
            Sonics & Production Profile: Close, dry and honest — natural room sound, minimal compression, full mid-range, breaths and string noise left in.

            Vocal Details
            Vocal Gender & Timbre: A male lead, soft and warm in a low-to-mid register, a little air on every phrase.
            Vocal Style: Storytelling delivery, close to the microphone, gentle and unhurried, lifting only slightly at the emotional peak.
            Harmony/Backing Vocals: A single female harmony a third above, entering at the second chorus and staying to the end.
            Vocal FX: A short natural room reverb, nothing else.

            Arrangement
            Instrument Lifecycle Description (Primary/Secondary Layering):
            Primary: Fingerpicked steel-string acoustic guitar from the first note to the last.
            Secondary: An upright bass joins the first chorus, a cello enters in the second verse and holds long notes underneath, brushed drums arrive only for the final chorus.
            Groove & Foundation Progression: A steady fingerpicked pattern acts as the pulse, with the bass outlining the roots and the brushes adding a soft swing at the end rather than a beat.
            Embellishments, Textures & Spatial FX: Occasional harmonics and a slide into each chorus, one bar of guitar alone before the last verse, and a final chord left to ring out into the room.
            """),
        MusicPrompt(title: "Trap", body: """
            Global Metadata
            Basic Attributes: bpm is 140. key is C#, and scale is minor. Trap / Dark Cinematic Hip Hop.
            Global Emotional Progression: Cold and menacing from the opening bar, tightening as the hats accelerate, opening into a wide, cinematic hook and dropping back to near silence at the close.
            Application Scenarios & Imagery: Empty parking garage, headlights, rain on concrete.
            Sonics & Production Profile: Sub-heavy and modern — long distorted 808s, sharp transient drums, a dark and mostly empty midrange, wide reverb on the melodic layer only.

            Vocal Details
            Vocal Gender & Timbre: A male lead, low and half-spoken, with a relaxed confident delivery.
            Vocal Style: Rhythmic triplet phrasing in the verses, longer melodic lines on the hook.
            Harmony/Backing Vocals: Doubled hook lines an octave down and short ad-libs answering the ends of phrases.
            Vocal FX: Light autotune on the hook only, tight doubling and a dark reverb.

            Arrangement
            Instrument Lifecycle Description (Primary/Secondary Layering):
            Primary: A booming 808 bass and a dark bell melody carry the track end to end.
            Secondary: Rattling hi-hats and snappy snares enter after the intro, a low string pad joins for the hook, and a detuned pluck answers the vocal in the second verse.
            Groove & Foundation Progression: Half-time snare on the third beat against triplet hat rolls that double in speed into each hook, the 808 sliding between roots rather than repeating them.
            Embellishments, Textures & Spatial FX: Reverse cymbals mark transitions, the beat cuts out entirely for two bars before the final hook, and the outro leaves the bell melody alone in the reverb.
            """),
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
    var seed: Int = -1
    var width: Int
    var height: Int
    var steps: Int
    /// Keep the model resident after this generation (default off → unload
    /// when done, freeing GPU memory). On → instant reuse for the next gen.
    var keepResident: Bool = false
    /// Set when the pane picked a network model: the LAN routing id
    /// (`<model>@<peer>`). The gen service then skips local resolve/load/
    /// unload — the hosting Mac loads on demand and manages its own memory.
    var lanModelId: String? = nil
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
    /// Style LoRAs (Advanced): stacked `.safetensors` adapters applied to the
    /// DiT at runtime (sent as `lora_paths`/`lora_scales` — mirrors mflux).
    /// Empty = none. Rows with an empty `path` are dropped before sending.
    var loras: [LoraAdapter] = []
}

extension ImageModelPreset {
    /// Number of tapped text-encoder layers the backend fuses — the count
    /// `cond_weights` must supply (Krea stacks 12 layers; FLUX concatenates 3;
    /// Mage-Flow uses the single final hidden state — no rebalance, matching
    /// gen.zig `condWeightCount` == 0, so the rebalance UI hides).
    var condWeightCount: Int {
        switch variant {
        case .krea2Turbo: return 12
        case .mageFlowTurbo, .mageFlowEditTurbo: return 0
        default: return 3
        }
    }

    /// The grid this backend samples on, mirroring `gen.zig`'s per-backend
    /// clamps. Derived from the variant, never from the id: a new preset of an
    /// existing variant inherits the right grid instead of defaulting to one.
    var resolutionGrid: ResolutionGrid {
        switch variant {
        // `clampKreaDim` — VAE ×8 + DiT patch ×2. Mage-Flow is native-resolution
        // with a ×16 VAE downsample and shares the same clamp server-side.
        case .krea2Turbo, .mageFlowTurbo, .mageFlowEditTurbo:
            return ResolutionGrid(alignment: 16, minDim: 256, maxDim: 2048)
        // `clampFluxDim` — klein's /32 crop granularity, 1536 covering the
        // widest preset edge.
        case .flux2Klein4B, .flux2Klein9B:
            return ResolutionGrid(alignment: 32, minDim: 256, maxDim: 1536)
        }
    }

    /// Instruction editing (in-context reference conditioning) is a trained
    /// capability: FLUX.2-klein, and the Mage-Flow-Edit checkpoint. Krea and
    /// Mage-Flow Turbo (txt2img) can only do renoise variations.
    var supportsReferenceEdit: Bool {
        variant == .flux2Klein4B || variant == .flux2Klein9B || variant == .mageFlowEditTurbo
    }

    // ── Capability flags: what the Advanced panel is allowed to offer ──
    // Each mirrors a CONCRETE server-side fact, because a control the backend
    // ignores is worse than a missing one — the user pays for a setting that
    // silently does nothing (Mage-Flow shipped with five of them).

    /// Renoise variations (`image` + `strength`) need the backend's VAE encoder
    /// — `gen.ImageEngine.supportsImg2Img`. Mage-Flow has no variation path at
    /// all: sending a source without `mode:"edit"` is a 400.
    var supportsImg2Img: Bool {
        switch variant {
        case .mageFlowTurbo, .mageFlowEditTurbo: return false
        default: return true
        }
    }

    /// Runtime LoRA adapters attach to the DiT (`gen.ImageEngine.setLora`).
    /// Mage-Flow has no LoRA path, so a picked adapter matches 0 modules → 400.
    var supportsLoRA: Bool {
        switch variant {
        case .mageFlowTurbo, .mageFlowEditTurbo: return false
        default: return true
        }
    }

    /// Steps are a real knob only where the schedule isn't distilled shut.
    /// Mage-Flow Turbo is distillation-fixed at 4: measured on an M-series Mac,
    /// 8 steps costs 2× and 12 costs 4× for a DIFFERENT image, not a better one.
    var stepsAreFixed: Bool {
        switch variant {
        case .mageFlowTurbo, .mageFlowEditTurbo: return true
        default: return false
        }
    }

    /// The fixed step count for a distilled preset (its `.good` profile).
    var fixedSteps: Int { settings(.good).steps }

    /// Starter prompts for the Examples menu, for the mode the pane is in.
    /// An instruction-tuned editor's repertoire IS its prompt vocabulary — the
    /// only way to discover that it can produce a depth map or de-rain a photo
    /// is to be shown the sentence that does it.
    func promptExamples(editing: Bool) -> [ImagePromptExampleGroup] {
        guard editing && supportsReferenceEdit else { return ImagePromptExamples.textToImage }
        switch variant {
        case .mageFlowEditTurbo: return ImagePromptExamples.mageFlowEdit
        default: return ImagePromptExamples.genericEdit
        }
    }

    /// The resolution menu for the current mode. In EDIT mode an edit-capable
    /// model offers "Match source" first — the reference pipeline's default and
    /// the only choice that can't distort the picture the user handed over.
    func resolutionOptions(editMode: Bool) -> [ResolutionOption] {
        let base = (editMode && supportsReferenceEdit) ? [.matchSource] + resolutions : resolutions
        // Last, so the fixed buckets stay the obvious pick and the fields only
        // appear for someone who went looking for them.
        return base + [.custom]
    }

    /// Keep a persisted selection valid when the mode changes (leaving edit mode
    /// removes "Match source" from the menu — a stale selection would otherwise
    /// leave the picker showing nothing).
    func validResolution(_ current: ResolutionOption, editMode: Bool) -> ResolutionOption {
        let opts = resolutionOptions(editMode: editMode)
        return opts.contains(current) ? current : (editMode && supportsReferenceEdit ? .matchSource : defaultResolution)
    }
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
    /// Optional last-frame image (H3 fl2va): the frame the clip lands on. The
    /// first frame sets the geometry and this one follows it, so a pair with
    /// mismatched aspects still produces a clean landing.
    var lastFrameImagePath: String? = nil
    /// Optional speech/audio clip for audio-to-video: the soundtrack is frozen
    /// as conditioning (voices, lip sync and performance follow it) and the
    /// ORIGINAL clip is muxed into the mp4. Any AVFoundation-readable format;
    /// forces a two-stage pipeline.
    var audioPath: String? = nil
    /// Keep the model resident after this generation (default off → unload).
    var keepResident: Bool = false
    /// Max-quality opt-out of the server's fast recipe ("fast": false — every
    /// forward dense, ~2.8x slower at 768p, "just a smidge better").
    var bestQuality: Bool = false
    /// Decode the final latent with LTX's DiffVAE instead of the conv decoder
    /// ("decoder": "diffusion"). Only sent on a preset that declares it.
    var diffusionDecoder: Bool = false
    /// Set when the pane picked a network model: the LAN routing id
    /// (`<model>@<peer>`). The gen service then skips local resolve/load/
    /// unload — the hosting Mac loads on demand and manages its own memory.
    var lanModelId: String? = nil
    /// Style LoRAs (Advanced): stacked `.safetensors` adapters applied to the
    /// DiT at runtime (sent as `lora_paths`/`lora_scales` — mirrors mflux).
    /// Empty = none. Rows with an empty `path` are dropped before sending.
    var loras: [LoraAdapter] = []
    /// Turbo distillation LoRA (H3 fl2va): 4-step sampling, fast recipe off.
    var turbo: Bool = false
    /// Chained windows (H3 fl2va): number of `numFrames`-frame windows joined
    /// by keyframe conditioning. 1 = a single ordinary window.
    var chainWindows: Int = 1
    /// Steps for LTX's two-stage refine pass (`stage2_steps`). 0 = Auto, which
    /// is the server's own "all 3" — sent only when the user picks a number,
    /// so the absent field keeps meaning the default everywhere.
    var stage2Steps: Int = 0
    /// Audio-guidance strength for audio-to-video (`cfg_audio_scale`). Only
    /// meaningful with a clip attached: without one there is no audio guider
    /// to scale.
    var cfgAudioScale: Double = 7.0
    /// ref2va references (REF2VA pack only). Images the generation follows for
    /// character/style, clips for motion and scene, audio for voice and score.
    /// Paths, resolved to base64 at request time.
    var refImagePaths: [String] = []
    var refVideoPaths: [String] = []
    var refAudioPaths: [String] = []
    /// How a reference IMAGE is sized. `.match` scales it (down only) to the
    /// generation's pixel area; `.max` uses a 2048 short edge for identity
    /// fidelity and is several times slower — reference tokens ride through
    /// every sampling step.
    var refImageSize: RefImageSizing = .match
    /// Per-step latent previews (`preview`). Off by default: it is not free
    /// (an x0 solve and a host copy of the previewed frames, per step), so the
    /// absent field keeps meaning "no previews" for every other client too.
    var livePreview: Bool = false
}

/// ref2va references already encoded for the wire. Kept apart from
/// `VideoGenRequest` (which holds PATHS) so `requestBody` stays pure — reading
/// files and pulling frames out of a movie is the caller's job.
struct VideoRefPayloads {
    /// One reference clip: its base64 PNG frames and, optionally, the base64
    /// WAV soundtrack that belongs to it. The soundtrack rides ON the clip
    /// rather than in a parallel array, so a silent clip cannot shift the
    /// pairing of the ones that have sound.
    struct Video {
        var frames: [String]
        var audio: String?
    }
    var images: [String] = []
    var videos: [Video] = []
    var audios: [String] = []

    var isEmpty: Bool { images.isEmpty && videos.isEmpty && audios.isEmpty }
}

/// `ref_image_size` on the wire.
enum RefImageSizing: String, CaseIterable, Hashable {
    case match, max

    var label: String {
        switch self {
        case .match: return "Match generation size"
        case .max:   return "Maximum detail (several times slower)"
        }
    }
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
    /// Max-quality opt-out of the server's fast recipe ("fast": false — every
    /// forward dense, ~2.8x slower at 768p, "just a smidge better").
    var bestQuality: Bool = false
    /// Set when the pane picked a network model: the LAN routing id
    /// (`<model>@<peer>`). The gen service then skips local resolve/load/
    /// unload — the hosting Mac loads on demand and manages its own memory.
    var lanModelId: String? = nil
}

struct MusicGenRequest {
    var model: MusicModelPreset
    /// Style/genre/mood description — the "in the style of…" prompt.
    var prompt: String
    /// Optional lyrics; empty → the server's "[Instrumental]" convention.
    var lyrics: String = ""
    /// Wordless track. Sent as `instrumental: true` with the lyrics field
    /// OMITTED — the server names the pair a 400 on both backends, and on
    /// Music 3 an omitted lyrics field is the only spelling it accepts.
    var instrumental: Bool = false
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
    /// Flow-match refinement passes; nil = the server's own default. Only sent
    /// on backends whose `supportsSteps` is true.
    var steps: Int? = nil
    /// Keep the model resident after this generation (default off → unload).
    var keepResident: Bool = false
    /// Max-quality opt-out of the server's fast recipe ("fast": false — every
    /// forward dense, ~2.8x slower at 768p, "just a smidge better").
    var bestQuality: Bool = false
    /// Optional reference clip (48 kHz stereo WAV prepared by
    /// `AudioReference.referenceWav`) whose style/timbre the track follows.
    /// Only sent where `supportsReferenceAudio`.
    var refAudioPath: String? = nil
    /// What the track is made FROM. Anything but `.text2music` needs
    /// `srcAudioPath` and is only sent where `supportsSourceAudio`.
    var task: MusicTask = .text2music
    /// The cover / complete source (48 kHz stereo WAV, full length — the
    /// server makes the track exactly as long as this clip).
    var srcAudioPath: String? = nil
    /// Cover: share of the denoise steps that follow the source (1 = all).
    var coverStrength: Double = 1.0
    /// Cover: start from a blend with the source instead of pure noise (0 = off).
    var coverNoiseStrength: Double = 0.0
    /// Complete: instruments to add, from `MusicTask.trackClasses`; empty =
    /// the model decides.
    var trackClasses: [String] = []
    /// Set when the pane picked a network model: the LAN routing id
    /// (`<model>@<peer>`). The gen service then skips local resolve/load/
    /// unload — the hosting Mac loads on demand and manages its own memory.
    var lanModelId: String? = nil
}

/// ACE-Step generation tasks (server `task`). Raw values are the wire spelling.
enum MusicTask: String, CaseIterable, Codable {
    case text2music, cover, complete

    var label: String {
        switch self {
        case .text2music: return "Text to music"
        case .cover: return "Cover"
        case .complete: return "Vocal to BGM"
        }
    }
    var needsSource: Bool { self != .text2music }

    /// The instrument vocabulary the `complete` instruction accepts (the
    /// server refuses anything else by name).
    static let trackClasses = ["woodwinds", "brass", "fx", "synth", "strings", "percussion",
                               "keyboard", "guitar", "bass", "drums", "backing_vocals", "vocals"]
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
    /// Max-quality opt-out of the server's fast recipe ("fast": false — every
    /// forward dense, ~2.8x slower at 768p, "just a smidge better").
    var bestQuality: Bool = false
    /// Set when the pane picked a network model: the LAN routing id
    /// (`<model>@<peer>`). The gen service then skips local resolve/load/
    /// unload — the hosting Mac loads on demand and manages its own memory.
    var lanModelId: String? = nil
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

    /// Frame count a run can safely fit at the chosen resolution.
    ///
    /// The formula below is LTX's — a fixed load cost plus roughly 12 GB per
    /// megapixel per 100 frames of VAE decode staging — and it was applied to
    /// EVERY video backend. On H3 that is not an approximation, it is a
    /// different model: the frame-dependent term there is the fast recipe's
    /// attention-broadcast cache, so the LTX formula computed 677 frames on a
    /// 128 GB Mac (warning never fires) and 32 on a 48 GB one (below H3's own
    /// 124-frame floor, so it fires always). A backend with its own memory
    /// model answers for itself.
    static func safeFrameCap(model: VideoModelPreset, width: Int, height: Int, available: Int,
                             fast: Bool = true) -> Int {
        if model.backend == .minimaxH3 {
            return H3Plan.frameCap(model: model, width: width, height: height,
                                   availableGB: available, fast: fast)
        }
        let pixelMP = Double(width * height) / 1_000_000.0
        let headroom = max(0, available - model.approxRAMGB)
        let perHundred = max(2.0, pixelMP * 12.0)
        let framesByRAM = Int((Double(headroom) / perHundred) * 100)
        return min(model.maxFrames, max(9, framesByRAM))
    }
}
