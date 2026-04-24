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
/// Python script will use. Both run on MLX with native 4/8-bit
/// quantization; `flux2Klein` is newer and smaller (4B params) than the
/// 12B FLUX.1 transformer.
enum FluxVariant: String, Hashable, Codable {
    case flux1            // FLUX.1 schnell / dev — uses Flux1 class
    case flux2Klein4B     // FLUX.2-klein 4B params — uses Flux2Klein, ModelConfig.flux2_klein_4b()
    case flux2Klein9B     // FLUX.2-klein 9B params — uses Flux2Klein, ModelConfig.flux2_klein_9b()
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
        defaultQuality: .good
    )

    /// FLUX.1-schnell 4-bit — Apache 2.0 distilled, 1–4 step inference.
    static let schnellQ4 = ImageModelPreset(
        id: "mflux/schnell-q4",
        name: "FLUX.1-schnell 4-bit (~10 GB)",
        variant: .flux1,
        configName: "schnell",
        repo: "dhairyashil/FLUX.1-schnell-mflux-4bit",
        approxDownloadGB: 10,
        approxRAMGB: 8,
        resolutions: fluxResolutions,
        defaultResolution: fluxResolutions[0],
        qualityProfiles: [
            .fast:         .init(steps: 2,  guidance: 0.0),
            .good:         .init(steps: 4,  guidance: 0.0),
            .quality:      .init(steps: 6,  guidance: 0.0),
            .superQuality: .init(steps: 12, guidance: 0.0),
        ],
        defaultQuality: .good
    )

    /// FLUX.1-dev 4-bit — non-commercial license, higher quality than schnell.
    static let devQ4 = ImageModelPreset(
        id: "mflux/dev-q4",
        name: "FLUX.1-dev 4-bit (~10 GB)",
        variant: .flux1,
        configName: "dev",
        repo: "dhairyashil/FLUX.1-dev-mflux-4bit",
        approxDownloadGB: 10,
        approxRAMGB: 12,
        resolutions: fluxResolutions,
        defaultResolution: fluxResolutions[0],
        qualityProfiles: [
            .fast:         .init(steps: 14, guidance: 3.0),
            .good:         .init(steps: 20, guidance: 3.5),
            .quality:      .init(steps: 28, guidance: 3.5),
            .superQuality: .init(steps: 50, guidance: 4.5),
        ],
        defaultQuality: .quality
    )

    /// FLUX.1-schnell 8-bit — higher fidelity than Q4.
    static let schnellQ8 = ImageModelPreset(
        id: "mflux/schnell-q8",
        name: "FLUX.1-schnell 8-bit (~18 GB)",
        variant: .flux1,
        configName: "schnell",
        repo: "dhairyashil/FLUX.1-schnell-mflux-8bit",
        approxDownloadGB: 18,
        approxRAMGB: 12,
        resolutions: fluxResolutions,
        defaultResolution: fluxResolutions[0],
        qualityProfiles: [
            .fast:         .init(steps: 2,  guidance: 0.0),
            .good:         .init(steps: 4,  guidance: 0.0),
            .quality:      .init(steps: 6,  guidance: 0.0),
            .superQuality: .init(steps: 12, guidance: 0.0),
        ],
        defaultQuality: .good
    )

    /// FLUX.1-dev 8-bit.
    static let devQ8 = ImageModelPreset(
        id: "mflux/dev-q8",
        name: "FLUX.1-dev 8-bit (~18 GB)",
        variant: .flux1,
        configName: "dev",
        repo: "dhairyashil/FLUX.1-dev-mflux-8bit",
        approxDownloadGB: 18,
        approxRAMGB: 16,
        resolutions: fluxResolutions,
        defaultResolution: fluxResolutions[0],
        qualityProfiles: [
            .fast:         .init(steps: 14, guidance: 3.0),
            .good:         .init(steps: 20, guidance: 3.5),
            .quality:      .init(steps: 28, guidance: 3.5),
            .superQuality: .init(steps: 50, guidance: 4.5),
        ],
        defaultQuality: .quality
    )

    /// Catalog ordered cheapest → heaviest. Default (`first`) is FLUX.2-klein
    /// 4B Q4 — smallest download.
    static let all: [ImageModelPreset] = [
        .flux2Klein4B_Q4, .schnellQ4, .devQ4, .schnellQ8, .devQ8,
    ]
}

// MARK: - Video presets

/// Pipeline shape — ltx-2-mlx exposes three. One-stage is fastest. Two-stage
/// uses dev transformer + distilled LoRA for ~10× the quality at ~10× the
/// runtime. Two-stage HQ uses a higher-quality stage 1.
enum VideoPipelineMode: String, Hashable, Codable {
    case oneStage      // TextToVideoPipeline, num_steps configurable
    case twoStage      // TwoStagePipeline,  stage1_steps configurable
    case twoStageHQ    // TwoStageHQPipeline, stage1_steps configurable
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

    static func == (lhs: Self, rhs: Self) -> Bool { lhs.id == rhs.id }
    func hash(into hasher: inout Hasher) { hasher.combine(id) }

    func settings(_ q: QualityPreset) -> VideoQualitySettings {
        qualityProfiles[q] ?? qualityProfiles[defaultQuality]!
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
            name: "LTX-Video 2.3 Q4 (with audio, ~41 GB)",
            repo: "dgrauet/ltx-2.3-mlx-q4",
            approxDownloadGB: 41,
            approxFirstRunDownloadGB: 49,    // + ~8 GB Gemma 3 12B 4-bit
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
            frameOptions: frameLadder(maxFrames: cap)
        )
    }()

    static let all: [VideoModelPreset] = [.ltx23Q4]
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
    /// Optional first-frame image for image-to-video conditioning (2-stage
    /// pipelines only — the distilled 1-stage pipeline doesn't accept it).
    var firstFrameImagePath: String? = nil
}

// MARK: - RAM checks

enum RAMChecker {
    /// Total physical memory in GB. Used for the rough "do you have enough
    /// RAM for this model" gate shown before generation starts.
    static var totalGB: Int {
        Int(ProcessInfo.processInfo.physicalMemory / (1024 * 1024 * 1024))
    }

    /// Free + inactive memory in GB (pages we can reclaim without paging).
    /// Approximation via `vm_stat`. Close enough for the "do you have
    /// headroom" gate before kicking a multi-GB model load.
    static var availableGB: Int {
        let proc = Process()
        proc.executableURL = URL(fileURLWithPath: "/usr/bin/vm_stat")
        let pipe = Pipe()
        proc.standardOutput = pipe
        proc.standardError = Pipe()
        do { try proc.run() } catch { return totalGB }
        proc.waitUntilExit()
        guard let text = String(data: pipe.fileHandleForReading.readDataToEndOfFile(), encoding: .utf8) else { return totalGB }

        var pageSize: Int = 4096
        if let match = text.range(of: #"page size of (\d+)"#, options: .regularExpression) {
            let s = String(text[match])
            let digits = s.filter { $0.isNumber }
            if let n = Int(digits) { pageSize = n }
        }

        func pages(for key: String) -> Int {
            guard let line = text.split(separator: "\n").first(where: { $0.contains(key) }) else { return 0 }
            let digits = line.filter { $0.isNumber }
            return Int(digits) ?? 0
        }
        let free = pages(for: "Pages free")
        let inactive = pages(for: "Pages inactive")
        let bytes = (free + inactive) * pageSize
        return bytes / (1024 * 1024 * 1024)
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
