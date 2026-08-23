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
    /// The size last typed into the Custom… fields. Kept even while a fixed
    /// bucket is selected, so switching back to Custom restores what you had —
    /// the same convention as the H3 reference lists surviving a preset switch.
    var customWidth: Int = 1024
    var customHeight: Int = 1024

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
        // Custom is a sentinel, not a row in `resolutions` — the size it means
        // lives in `customWidth`/`customHeight`, so it has to be recognised
        // here or a saved custom pick reopens on the model's default.
        if resolutionId == ResolutionOption.custom.id { return .custom }
        return m.resolutions.first { $0.id == resolutionId } ?? m.defaultResolution
    }

    /// The same answer as `resolvedResolution`, with the Custom sentinel turned
    /// into the size it stands for. Every consumer that wants NUMBERS reads this;
    /// only the pane's picker wants the sentinel, because it has to select the
    /// "Custom…" row.
    ///
    /// This exists because `MediaToolArgs.resolution` returns `saved` VERBATIM
    /// when the model named no size — its own `width > 0` filters only guard the
    /// pixel/aspect paths — so handing it the sentinel puts -1 × -1 on the wire.
    /// A size the current model cannot honor falls back to that model's default:
    /// the settings blob is shared across presets, so a 2048 saved on Krea is
    /// simply not a thing FLUX can be asked for.
    func concreteResolution(for m: ImageModelPreset) -> ResolutionOption {
        let picked = resolvedResolution(for: m)
        guard picked.isCustom else { return picked }
        guard let size = m.resolutionGrid.resolve(width: customWidth, height: customHeight).size else {
            return m.defaultResolution
        }
        return ResolutionOption(width: size.width, height: size.height,
                                label: "\(size.width) × \(size.height)")
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
        if let v = try c.decodeIfPresent(Int.self, forKey: .customWidth) { customWidth = v }
        if let v = try c.decodeIfPresent(Int.self, forKey: .customHeight) { customHeight = v }
    }
}

// MARK: - Audio

struct AudioGenSettings: Codable, Equatable {
    var modelId: String = AudioModelPreset.qwen3TTS06B8bit.id
    var speed: Double = 1.0
    var temperature: Double = 0.7
    var keepResident: Bool = false
    /// The DRAFT, not just the knobs: the pane unmounts on every navigation,
    /// so without these a trip to Chat for a copy-paste wiped the text and
    /// the attached reference clip (live 2026-08-22).
    var text: String = ""
    var refAudioPath: String? = nil
    var refText: String = ""

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
        if let v = try c.decodeIfPresent(String.self, forKey: .text) { text = v }
        refAudioPath = try c.decodeIfPresent(String.self, forKey: .refAudioPath)
        if let v = try c.decodeIfPresent(String.self, forKey: .refText) { refText = v }
    }
}

// MARK: - Music

struct MusicGenSettings: Codable, Equatable {
    var modelId: String = MusicModelPreset.acestepXLTurbo8bit.id
    var durationSeconds: Int = 60
    var vocalLanguage: String = "en"
    var keepResident: Bool = false
    // Everything below used to live in plain `@State`. The pane is UNMOUNTED
    // when you navigate away from the Audio page, so these reset on every
    // visit, not just across launches — the image and video panes persist
    // their seeds and this one did not.
    var bpm: Int? = nil
    var keyscale: String = ""
    var timesignature: String = ""
    var seed: Int = -1
    var steps: Int? = nil
    var instrumental: Bool = false
    /// Advanced starts OPEN. Collapsed-by-default is why tempo, key, seed and
    /// steps read as missing features — they were one unlabeled chevron away.
    var showAdvanced: Bool = true
    /// The draft (see `AudioGenSettings.text`).
    var prompt: String = ""
    var lyrics: String = ""
    var refAudioPath: String? = nil
    /// Source-audio task state (ACE-Step cover / complete), part of the draft.
    var task: MusicTask = .text2music
    var srcAudioPath: String? = nil
    var coverStrength: Double = 1.0
    var coverNoiseStrength: Double = 0.0
    var trackClasses: [String] = []

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

    /// Every key optional so a blob written by an older build still decodes —
    /// a throwing decode would silently reset the pane to defaults for every
    /// existing install. Absent keys keep the property defaults, which is also
    /// how `showAdvanced` ends up OPEN for people upgrading from a build that
    /// never wrote it.
    init(from decoder: Decoder) throws {
        self.init()
        let c = try decoder.container(keyedBy: CodingKeys.self)
        if let v = try c.decodeIfPresent(String.self, forKey: .modelId) { modelId = v }
        if let v = try c.decodeIfPresent(Int.self, forKey: .durationSeconds) { durationSeconds = v }
        if let v = try c.decodeIfPresent(String.self, forKey: .vocalLanguage) { vocalLanguage = v }
        if let v = try c.decodeIfPresent(Bool.self, forKey: .keepResident) { keepResident = v }
        bpm = try c.decodeIfPresent(Int.self, forKey: .bpm)
        if let v = try c.decodeIfPresent(String.self, forKey: .keyscale) { keyscale = v }
        if let v = try c.decodeIfPresent(String.self, forKey: .timesignature) { timesignature = v }
        if let v = try c.decodeIfPresent(Int.self, forKey: .seed) { seed = v }
        steps = try c.decodeIfPresent(Int.self, forKey: .steps)
        if let v = try c.decodeIfPresent(Bool.self, forKey: .instrumental) { instrumental = v }
        if let v = try c.decodeIfPresent(Bool.self, forKey: .showAdvanced) { showAdvanced = v }
        if let v = try c.decodeIfPresent(String.self, forKey: .prompt) { prompt = v }
        if let v = try c.decodeIfPresent(String.self, forKey: .lyrics) { lyrics = v }
        refAudioPath = try c.decodeIfPresent(String.self, forKey: .refAudioPath)
        if let v = try c.decodeIfPresent(MusicTask.self, forKey: .task) { task = v }
        srcAudioPath = try c.decodeIfPresent(String.self, forKey: .srcAudioPath)
        if let v = try c.decodeIfPresent(Double.self, forKey: .coverStrength) { coverStrength = v }
        if let v = try c.decodeIfPresent(Double.self, forKey: .coverNoiseStrength) { coverNoiseStrength = v }
        if let v = try c.decodeIfPresent([String].self, forKey: .trackClasses) { trackClasses = v }
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
    /// Steps in LTX's two-stage refine pass. 0 = Auto (the server's own "all 3").
    var stage2Steps: Int = 0
    /// Audio-guidance strength for audio-to-video. The LTX reference default.
    var cfgAudioScale: Double = 7.0
    /// Chained windows (H3 fl2va): 1 = a single ordinary window.
    var chainWindows: Int = 1
    /// Style LoRAs (Advanced): sticky stack of adapter path + strength pairs.
    /// Empty = none attached.
    var loras: [LoraAdapter] = []
    /// Height of the drag-resizable prompt editor.
    var promptHeight: Double = PromptEditorHeight.defaultHeight
    /// The size last typed into the Custom… fields, kept across preset switches
    /// like the image pane's.
    var customWidth: Int = 704
    var customHeight: Int = 448

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
        if resolutionId == ResolutionOption.custom.id { return .custom }
        return m.resolutions.first { $0.id == resolutionId }
            ?? m.recommendedResolution(totalGB: RAMChecker.totalGB)
    }

    /// `resolvedResolution` with the Custom sentinel turned into the size it
    /// stands for — read by every consumer that wants NUMBERS. Same hazard as
    /// the image side: `MediaToolArgs.resolution` hands `saved` back verbatim
    /// when the model names no size, so the sentinel's -1 would ride the wire
    /// from the chat's `generate_video`.
    ///
    /// Resolved against the ONE-STAGE grid: the saved size is a canvas, and the
    /// pipeline is chosen by the quality tier at request time. A two-stage tier
    /// tightens the grid to /64 in the pane, where the user can see the note.
    func concreteResolution(for m: VideoModelPreset) -> ResolutionOption {
        let picked = resolvedResolution(for: m)
        guard picked.isCustom else { return picked }
        guard let size = m.resolutionGrid(twoStage: false)
            .resolve(width: customWidth, height: customHeight).size else {
            return m.recommendedResolution(totalGB: RAMChecker.totalGB)
        }
        return ResolutionOption(width: size.width, height: size.height,
                                label: "\(size.width) × \(size.height)")
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
        if let v = try c.decodeIfPresent(Int.self, forKey: .customWidth) { customWidth = v }
        if let v = try c.decodeIfPresent(Int.self, forKey: .customHeight) { customHeight = v }
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
