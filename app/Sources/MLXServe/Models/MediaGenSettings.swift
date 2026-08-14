import Foundation

/// One stacked LoRA adapter: a `.safetensors` path plus its strength. Several
/// can attach at once (mirrors mflux's `lora_paths`/`lora_scales`, sent to
/// the server as JSON arrays of the same names) — their effects sum at
/// generation time rather than merging into the base weights, so order
/// doesn't matter. `id` is NOT persisted (see `CodingKeys`) — it exists only
/// to give SwiftUI's `ForEach` stable row identity while the list is edited.
struct LoraAdapter: Codable, Equatable, Identifiable {
    var id: UUID = UUID()
    /// Absolute path to a `.safetensors` adapter. Empty rows (mid-edit, e.g.
    /// right after tapping "+") are dropped before the request is sent.
    var path: String = ""
    /// Strength multiplier on top of the file's own alpha/rank scale.
    var scale: Double = 1.0

    private enum CodingKeys: String, CodingKey { case path, scale }
}

/// Keys for the pre-multi-LoRA single `loraPath`/`loraScale` fields, kept
/// only so `ImageGenSettings`/`VideoGenSettings` can migrate an old
/// UserDefaults blob into the new `loras` array. Not tied to any stored
/// property, so it can't be part of either struct's synthesized CodingKeys.
private enum LegacyLoraCodingKeys: String, CodingKey { case loraPath, loraScale }

/// Bounds for a drag-resizable prompt editor. The height is persisted, so a
/// value dragged on a taller window — or a garbage one — must never come back
/// as an editor too small to type in or taller than the pane.
enum PromptEditorHeight {
    static let minHeight: Double = 70
    static let maxHeight: Double = 600
    static let defaultHeight: Double = 110

    static func clamp(_ h: Double) -> Double {
        guard h.isFinite else { return defaultHeight }
        return Swift.min(maxHeight, Swift.max(minHeight, h))
    }
}

/// Sticky last-used settings for the three media-generation panels.
///
/// The Image/Audio/Video windows keep their controls as view `@State`, so a
/// user's chosen model / quality / resolution / steps / seed was forgotten the
/// moment the window closed. These structs persist that choice to UserDefaults
/// (Codable JSON), mirroring `ServerOptions`: a no-arg init seeds the views'
/// current defaults, `load()`/`save()` round-trip under a distinct key, and a
/// migration-safe `init(from:)` (every key `decodeIfPresent`) keeps old blobs
/// valid as new fields ship — without it the compiler-synthesized decode throws
/// on the first missing key and `load()`'s `try?` silently resets everything.
///
/// Presets (`ImageModelPreset` / `AudioModelPreset` / `VideoModelPreset`) and
/// `ResolutionOption` are NOT Codable but have stable string `id`s, so we
/// persist the id and reconstruct via `.all.first { $0.id == }` with the preset
/// default as the unknown-id fallback. The prompt and transient inputs
/// (reference audio, first-frame image) are deliberately NOT persisted.

// MARK: - Image

struct ImageGenSettings: Codable, Equatable {
    var modelId: String = ImageModelPreset.flux2Klein4B_Q4.id
    var quality: QualityPreset = .good
    var resolutionId: String = ImageModelPreset.flux2Klein4B_Q4.defaultResolution.id
    var steps: Int = 8
    var seed: Int = -1
    var safeMode: Bool = true
    var keepResident: Bool = false
    /// img2img renoise strength (the source image path itself is transient —
    /// not persisted, like video's first-frame).
    var strength: Double = 0.6
    /// Source-image mode: instruction edit (FLUX.2) vs renoise variation.
    var editMode: Bool = true
    /// Conditioning rebalance (Advanced): global gain + weights text.
    var condGain: Double = 1.0
    var condWeightsText: String = ""
    /// Style LoRAs (Advanced): sticky stack of adapter path + strength pairs.
    /// Empty = none attached.
    var loras: [LoraAdapter] = []

    private static let storageKey = "imageGenSettings"

    static func load() -> ImageGenSettings {
        guard let data = UserDefaults.standard.data(forKey: storageKey),
              let v = try? JSONDecoder().decode(ImageGenSettings.self, from: data) else {
            return ImageGenSettings()
        }
        return v
    }

    func save() {
        guard let data = try? JSONEncoder().encode(self) else { return }
        UserDefaults.standard.set(data, forKey: Self.storageKey)
    }
}

extension ImageGenSettings {
    /// The persisted model, or the catalog default when the id is unknown
    /// (uninstalled / renamed preset).
    var resolvedModel: ImageModelPreset { resolvedModel(models: []) }

    /// Same, but also resolving custom (user-added) models against the live
    /// `/v1/models` list — a custom id with the list unavailable (server
    /// down) falls back like any unknown id.
    func resolvedModel(models: [ModelInfo]) -> ImageModelPreset {
        ImageModelPreset.all.first { $0.id == modelId }
            ?? CustomMediaModels.imagePreset(for: modelId, from: models)
            ?? .flux2Klein4B_Q4
    }

    /// The persisted resolution revalidated against `m`'s buckets — unknown ids
    /// (e.g. carried over from a different model) fall back to the model default.
    func resolvedResolution(for m: ImageModelPreset) -> ResolutionOption {
        m.resolutions.first { $0.id == resolutionId } ?? m.defaultResolution
    }

    /// Migration-safe decode (see type doc). Declared in an extension so the
    /// memberwise / no-arg initializers + `encode(to:)` stay synthesized.
    init(from decoder: Decoder) throws {
        self.init()
        let c = try decoder.container(keyedBy: CodingKeys.self)
        if let v = try c.decodeIfPresent(String.self, forKey: .modelId) { modelId = v }
        if let v = try c.decodeIfPresent(QualityPreset.self, forKey: .quality) { quality = v }
        if let v = try c.decodeIfPresent(String.self, forKey: .resolutionId) { resolutionId = v }
        if let v = try c.decodeIfPresent(Int.self, forKey: .steps) { steps = v }
        if let v = try c.decodeIfPresent(Int.self, forKey: .seed) { seed = v }
        if let v = try c.decodeIfPresent(Bool.self, forKey: .safeMode) { safeMode = v }
        if let v = try c.decodeIfPresent(Bool.self, forKey: .keepResident) { keepResident = v }
        if let v = try c.decodeIfPresent(Double.self, forKey: .strength) { strength = v }
        if let v = try c.decodeIfPresent(Bool.self, forKey: .editMode) { editMode = v }
        if let v = try c.decodeIfPresent(Double.self, forKey: .condGain) { condGain = v }
        if let v = try c.decodeIfPresent(String.self, forKey: .condWeightsText) { condWeightsText = v }
        if let v = try c.decodeIfPresent([LoraAdapter].self, forKey: .loras), !v.isEmpty {
            loras = v
        } else {
            // Pre-multi-LoRA blob: a single "loraPath"/"loraScale" pair, not
            // backed by a stored property anymore, so read it via its own key.
            let legacy = try decoder.container(keyedBy: LegacyLoraCodingKeys.self)
            let lp = try legacy.decodeIfPresent(String.self, forKey: .loraPath) ?? ""
            let ls = try legacy.decodeIfPresent(Double.self, forKey: .loraScale) ?? 1.0
            if !lp.isEmpty { loras = [LoraAdapter(path: lp, scale: ls)] }
        }
    }
}

// MARK: - Audio

struct AudioGenSettings: Codable, Equatable {
    var modelId: String = AudioModelPreset.qwen3TTS06B8bit.id
    var speed: Double = 1.0
    var temperature: Double = 0.7
    var keepResident: Bool = false

    private static let storageKey = "audioGenSettings"

    static func load() -> AudioGenSettings {
        guard let data = UserDefaults.standard.data(forKey: storageKey),
              let v = try? JSONDecoder().decode(AudioGenSettings.self, from: data) else {
            return AudioGenSettings()
        }
        return v
    }

    func save() {
        guard let data = try? JSONEncoder().encode(self) else { return }
        UserDefaults.standard.set(data, forKey: Self.storageKey)
    }
}

extension AudioGenSettings {
    var resolvedModel: AudioModelPreset { resolvedModel(models: []) }

    func resolvedModel(models: [ModelInfo]) -> AudioModelPreset {
        AudioModelPreset.all.first { $0.id == modelId }
            ?? CustomMediaModels.audioPreset(for: modelId, from: models)
            ?? .qwen3TTS06B8bit
    }

    init(from decoder: Decoder) throws {
        self.init()
        let c = try decoder.container(keyedBy: CodingKeys.self)
        if let v = try c.decodeIfPresent(String.self, forKey: .modelId) { modelId = v }
        if let v = try c.decodeIfPresent(Double.self, forKey: .speed) { speed = v }
        if let v = try c.decodeIfPresent(Double.self, forKey: .temperature) { temperature = v }
        if let v = try c.decodeIfPresent(Bool.self, forKey: .keepResident) { keepResident = v }
    }
}

// MARK: - Music

struct MusicGenSettings: Codable, Equatable {
    var modelId: String = MusicModelPreset.acestepXLTurbo8bit.id
    var durationSeconds: Int = 60
    var vocalLanguage: String = "en"
    var keepResident: Bool = false

    private static let storageKey = "musicGenSettings"

    static func load() -> MusicGenSettings {
        guard let data = UserDefaults.standard.data(forKey: storageKey),
              let v = try? JSONDecoder().decode(MusicGenSettings.self, from: data) else {
            return MusicGenSettings()
        }
        return v
    }

    func save() {
        guard let data = try? JSONEncoder().encode(self) else { return }
        UserDefaults.standard.set(data, forKey: Self.storageKey)
    }
}

extension MusicGenSettings {
    var resolvedModel: MusicModelPreset { resolvedModel(models: []) }

    func resolvedModel(models: [ModelInfo]) -> MusicModelPreset {
        MusicModelPreset.all.first { $0.id == modelId }
            ?? CustomMediaModels.musicPreset(for: modelId, from: models)
            ?? .acestepXLTurbo8bit
    }

    init(from decoder: Decoder) throws {
        self.init()
        let c = try decoder.container(keyedBy: CodingKeys.self)
        if let v = try c.decodeIfPresent(String.self, forKey: .modelId) { modelId = v }
        if let v = try c.decodeIfPresent(Int.self, forKey: .durationSeconds) { durationSeconds = v }
        if let v = try c.decodeIfPresent(String.self, forKey: .vocalLanguage) { vocalLanguage = v }
        if let v = try c.decodeIfPresent(Bool.self, forKey: .keepResident) { keepResident = v }
    }
}

// MARK: - Video

struct VideoGenSettings: Codable, Equatable {
    var modelId: String = VideoModelPreset.ltx23Q4.id
    var quality: QualityPreset = .good
    var resolutionId: String = VideoModelPreset.ltx23Q4.defaultResolution.id
    var numFrames: Int = 97
    var fps: Int = 24
    var mode: VideoPipelineMode = .oneStage
    var steps: Int = 12
    var cfgScale: Double = 1.0
    var stgScale: Double = 0.0
    var seed: Int = 42
    var keepResident: Bool = false
    /// Max-quality opt-out of the server's fast recipe (H3).
    var bestQuality: Bool = false
    /// Decode with LTX's DiffVAE instead of the conv decoder (8-bit LTX-2.5 only).
    var diffusionDecoder: Bool = false
    /// Turbo distillation LoRA (H3 fl2va): 4-step sampling.
    var turbo: Bool = false
    /// Style LoRAs (Advanced): sticky stack of adapter path + strength pairs.
    /// Empty = none attached.
    var loras: [LoraAdapter] = []
    /// Height of the drag-resizable prompt editor.
    var promptHeight: Double = PromptEditorHeight.defaultHeight

    private static let storageKey = "videoGenSettings"

    static func load() -> VideoGenSettings {
        guard let data = UserDefaults.standard.data(forKey: storageKey),
              let v = try? JSONDecoder().decode(VideoGenSettings.self, from: data) else {
            return VideoGenSettings()
        }
        return v
    }

    func save() {
        guard let data = try? JSONEncoder().encode(self) else { return }
        UserDefaults.standard.set(data, forKey: Self.storageKey)
    }
}

extension VideoGenSettings {
    /// A persisted LAN pick ("lan:<model>@<peer>") whose base id matches a
    /// local preset resolves to THAT preset — the pane gates ladders,
    /// resolutions and request capability-gating on this value, and the old
    /// blanket LTX fallback sent a remote H3 off-canvas sizes and frame
    /// counts below its trained floor.
    var resolvedModel: VideoModelPreset { resolvedModel(models: []) }

    func resolvedModel(models: [ModelInfo]) -> VideoModelPreset {
        if let local = VideoModelPreset.all.first(where: { $0.id == modelId }) { return local }
        // A custom pick (local or a peer's) adopts its family preset the same
        // way — the pane gates canvases, frame ladders and request fields on
        // the resolved value, so an unmatched custom must not keep another
        // backend's knobs.
        if let lan = LanPick.lanId(modelId) {
            let base = LanPick.base(of: lan)
            if let matched = VideoModelPreset.all.first(where: { $0.id == base }) { return matched }
            if let custom = CustomMediaModels.videoPreset(for: base, from: models) { return custom }
        } else if let custom = CustomMediaModels.videoPreset(for: modelId, from: models) {
            return custom
        }
        return .ltx23Q4
    }

    /// A persisted pick wins; with nothing saved the canvas is sized for THIS
    /// Mac rather than for the smallest supported one (see
    /// `VideoModelPreset.recommendedResolution`) — a static default meant a
    /// 128 GB machine opened on a preview-sized render.
    func resolvedResolution(for m: VideoModelPreset) -> ResolutionOption {
        m.resolutions.first { $0.id == resolutionId }
            ?? m.recommendedResolution(totalGB: RAMChecker.totalGB)
    }

    init(from decoder: Decoder) throws {
        self.init()
        let c = try decoder.container(keyedBy: CodingKeys.self)
        if let v = try c.decodeIfPresent(String.self, forKey: .modelId) { modelId = v }
        if let v = try c.decodeIfPresent(QualityPreset.self, forKey: .quality) { quality = v }
        if let v = try c.decodeIfPresent(String.self, forKey: .resolutionId) { resolutionId = v }
        if let v = try c.decodeIfPresent(Int.self, forKey: .numFrames) { numFrames = v }
        if let v = try c.decodeIfPresent(Int.self, forKey: .fps) { fps = v }
        if let v = try c.decodeIfPresent(VideoPipelineMode.self, forKey: .mode) { mode = v }
        if let v = try c.decodeIfPresent(Int.self, forKey: .steps) { steps = v }
        if let v = try c.decodeIfPresent(Double.self, forKey: .cfgScale) { cfgScale = v }
        if let v = try c.decodeIfPresent(Double.self, forKey: .stgScale) { stgScale = v }
        if let v = try c.decodeIfPresent(Int.self, forKey: .seed) { seed = v }
        if let v = try c.decodeIfPresent(Bool.self, forKey: .keepResident) { keepResident = v }
        if let v = try c.decodeIfPresent(Bool.self, forKey: .bestQuality) { bestQuality = v }
        if let v = try c.decodeIfPresent(Bool.self, forKey: .turbo) { turbo = v }
        if let v = try c.decodeIfPresent(Double.self, forKey: .promptHeight) {
            promptHeight = PromptEditorHeight.clamp(v)
        }
        if let v = try c.decodeIfPresent([LoraAdapter].self, forKey: .loras), !v.isEmpty {
            loras = v
        } else {
            let legacy = try decoder.container(keyedBy: LegacyLoraCodingKeys.self)
            let lp = try legacy.decodeIfPresent(String.self, forKey: .loraPath) ?? ""
            let ls = try legacy.decodeIfPresent(Double.self, forKey: .loraScale) ?? 1.0
            if !lp.isEmpty { loras = [LoraAdapter(path: lp, scale: ls)] }
        }
    }
}

// MARK: - 3D

struct Model3DGenSettings: Codable, Equatable {
    var modelId: String = Model3DModelPreset.hunyuan3d21_8bit.id
    var steps: Int = 30
    var guidance: Double = 5.0
    /// Marching-cubes octree resolution (128 / 256 / 384 — the reference
    /// default, affordable since the FlashVDM hierarchical volume decode).
    var resolution: Int = 384
    var keepResident: Bool = false
    /// Slowly rotate + "breathe" the previewed model on a turntable.
    var turntable: Bool = true
    /// P2 paint stage (full PBR texture). Off until validated end to end.
    var texture: Bool = false

    private static let storageKey = "model3dGenSettings"

    static func load() -> Model3DGenSettings {
        guard let data = UserDefaults.standard.data(forKey: storageKey),
              let v = try? JSONDecoder().decode(Model3DGenSettings.self, from: data) else {
            return Model3DGenSettings()
        }
        return v
    }

    func save() {
        guard let data = try? JSONEncoder().encode(self) else { return }
        UserDefaults.standard.set(data, forKey: Self.storageKey)
    }
}

extension Model3DGenSettings {
    var resolvedModel: Model3DModelPreset { resolvedModel(models: []) }

    func resolvedModel(models: [ModelInfo]) -> Model3DModelPreset {
        Model3DModelPreset.all.first { $0.id == modelId }
            ?? CustomMediaModels.meshPreset(for: modelId, from: models)
            ?? .hunyuan3d21_8bit
    }

    init(from decoder: Decoder) throws {
        self.init()
        let c = try decoder.container(keyedBy: CodingKeys.self)
        if let v = try c.decodeIfPresent(String.self, forKey: .modelId) { modelId = v }
        if let v = try c.decodeIfPresent(Int.self, forKey: .steps) { steps = v }
        if let v = try c.decodeIfPresent(Double.self, forKey: .guidance) { guidance = v }
        // Legacy migration: pre-FlashVDM builds persisted a 380 "fine" option.
        if let v = try c.decodeIfPresent(Int.self, forKey: .resolution) { resolution = v == 380 ? 384 : v }
        if let v = try c.decodeIfPresent(Bool.self, forKey: .keepResident) { keepResident = v }
        if let v = try c.decodeIfPresent(Bool.self, forKey: .turntable) { turntable = v }
        if let v = try c.decodeIfPresent(Bool.self, forKey: .texture) { texture = v }
    }
}
