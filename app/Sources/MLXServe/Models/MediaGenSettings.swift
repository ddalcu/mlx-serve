import Foundation

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
    var guidance: Double = 1.0
    var seed: Int = -1
    var negativePrompt: String = ""
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
    /// Style LoRA (Advanced): sticky adapter path ("" = none) + strength.
    var loraPath: String = ""
    var loraScale: Double = 1.0

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
    var resolvedModel: ImageModelPreset {
        ImageModelPreset.all.first { $0.id == modelId } ?? .flux2Klein4B_Q4
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
        if let v = try c.decodeIfPresent(Double.self, forKey: .guidance) { guidance = v }
        if let v = try c.decodeIfPresent(Int.self, forKey: .seed) { seed = v }
        if let v = try c.decodeIfPresent(String.self, forKey: .negativePrompt) { negativePrompt = v }
        if let v = try c.decodeIfPresent(Bool.self, forKey: .safeMode) { safeMode = v }
        if let v = try c.decodeIfPresent(Bool.self, forKey: .keepResident) { keepResident = v }
        if let v = try c.decodeIfPresent(Double.self, forKey: .strength) { strength = v }
        if let v = try c.decodeIfPresent(Bool.self, forKey: .editMode) { editMode = v }
        if let v = try c.decodeIfPresent(Double.self, forKey: .condGain) { condGain = v }
        if let v = try c.decodeIfPresent(String.self, forKey: .condWeightsText) { condWeightsText = v }
        if let v = try c.decodeIfPresent(String.self, forKey: .loraPath) { loraPath = v }
        if let v = try c.decodeIfPresent(Double.self, forKey: .loraScale) { loraScale = v }
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
    var resolvedModel: AudioModelPreset {
        AudioModelPreset.all.first { $0.id == modelId } ?? .qwen3TTS06B8bit
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
    var resolvedModel: MusicModelPreset {
        MusicModelPreset.all.first { $0.id == modelId } ?? .acestepXLTurbo8bit
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
    /// Style LoRA (Advanced): sticky adapter path ("" = none) + strength.
    var loraPath: String = ""
    var loraScale: Double = 1.0

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
    var resolvedModel: VideoModelPreset {
        VideoModelPreset.all.first { $0.id == modelId } ?? .ltx23Q4
    }

    func resolvedResolution(for m: VideoModelPreset) -> ResolutionOption {
        m.resolutions.first { $0.id == resolutionId } ?? m.defaultResolution
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
        if let v = try c.decodeIfPresent(String.self, forKey: .loraPath) { loraPath = v }
        if let v = try c.decodeIfPresent(Double.self, forKey: .loraScale) { loraScale = v }
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
    var resolvedModel: Model3DModelPreset {
        Model3DModelPreset.all.first { $0.id == modelId } ?? .hunyuan3d21_8bit
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
