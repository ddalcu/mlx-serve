import Foundation

/// One avatar persona: what the model is told to be (`systemPrompt`), the voice
/// it clones (`voiceClipPath` → a normalized 24 kHz mono WAV fed to Qwen3-TTS as
/// `ref_audio`), and the 3D model that speaks it (`glbPath`, usually one of the
/// user's 3D generations). All optional except name/prompt so a persona works
/// text-only, default-voice, and mesh-less while you fill the pieces in.
///
/// Codable (persisted via `AvatarPersonaStore`). Migration-safe decode — every
/// key `decodeIfPresent` — so a persona blob from an older build stays valid as
/// new fields ship (same discipline as `MediaGenSettings`).
struct AvatarPersona: Codable, Equatable, Identifiable {
    var id: UUID = UUID()
    var name: String = "Avatar"
    var systemPrompt: String = AvatarPersona.defaultSystemPrompt
    /// Path to a normalized 24 kHz mono WAV of the voice to clone. `nil` → the
    /// TTS model's default voice.
    var voiceClipPath: String? = nil
    /// Path to the `.glb` mesh that visually represents this persona. `nil` → the
    /// window shows a "pick a 3D model" placeholder.
    var glbPath: String? = nil
    /// A folder of documents to ground answers in (mini-RAG, same engine the
    /// chat's "attach a folder" uses). `nil` → no retrieval, the persona answers
    /// from the model's own knowledge.
    var docFolderPath: String? = nil

    static let defaultSystemPrompt = """
    You are a warm, natural-sounding virtual avatar having a spoken conversation. \
    Keep replies short and conversational — a sentence or two — because everything \
    you say is read aloud. Never use Markdown, bullet points, code blocks, or URLs; \
    speak in plain sentences.
    """

    /// Migration-safe decode (see type doc). Declared in an extension keeps the
    /// memberwise / no-arg initializers + `encode(to:)` synthesized.
}

extension AvatarPersona {
    init(from decoder: Decoder) throws {
        self.init()
        let c = try decoder.container(keyedBy: CodingKeys.self)
        if let v = try c.decodeIfPresent(UUID.self, forKey: .id) { id = v }
        if let v = try c.decodeIfPresent(String.self, forKey: .name) { name = v }
        if let v = try c.decodeIfPresent(String.self, forKey: .systemPrompt) { systemPrompt = v }
        if let v = try c.decodeIfPresent(String.self, forKey: .voiceClipPath) { voiceClipPath = v }
        if let v = try c.decodeIfPresent(String.self, forKey: .glbPath) { glbPath = v }
        if let v = try c.decodeIfPresent(String.self, forKey: .docFolderPath) { docFolderPath = v }
    }
}

/// The saved personas plus the current selection — persisted to UserDefaults as
/// one Codable blob, mirroring `Model3DGenSettings.load()/save()`. The avatar
/// window owns a copy, edits it, and calls `save()`; `selectedPersona` always
/// returns a usable persona (falls back to the first, then to a fresh default),
/// so the engine never has to handle an empty catalog.
struct AvatarPersonaStore: Codable, Equatable {
    var personas: [AvatarPersona] = [AvatarPersona()]
    var selectedId: UUID? = nil

    private static let storageKey = "avatarPersonas"

    static func load() -> AvatarPersonaStore {
        guard let data = UserDefaults.standard.data(forKey: storageKey),
              let v = try? JSONDecoder().decode(AvatarPersonaStore.self, from: data) else {
            return AvatarPersonaStore()
        }
        return v.normalized()
    }

    func save() {
        guard let data = try? JSONEncoder().encode(self) else { return }
        UserDefaults.standard.set(data, forKey: Self.storageKey)
    }

    /// The selected persona, or the first one, or a fresh default — never nil.
    var selectedPersona: AvatarPersona {
        if let id = selectedId, let p = personas.first(where: { $0.id == id }) { return p }
        return personas.first ?? AvatarPersona()
    }

    /// Insert or replace `persona` (matched by id) and select it. Returns a new
    /// store — value semantics keep the mutation explicit at the call site.
    func upserting(_ persona: AvatarPersona) -> AvatarPersonaStore {
        var s = self
        if let idx = s.personas.firstIndex(where: { $0.id == persona.id }) {
            s.personas[idx] = persona
        } else {
            s.personas.append(persona)
        }
        s.selectedId = persona.id
        return s.normalized()
    }

    /// Remove `id`; if it was selected (or the catalog empties), re-point the
    /// selection so `selectedPersona` stays meaningful.
    func removing(_ id: UUID) -> AvatarPersonaStore {
        var s = self
        s.personas.removeAll { $0.id == id }
        return s.normalized()
    }

    /// Guarantee a non-empty catalog and a valid selection.
    func normalized() -> AvatarPersonaStore {
        var s = self
        if s.personas.isEmpty { s.personas = [AvatarPersona()] }
        if s.selectedId == nil || !s.personas.contains(where: { $0.id == s.selectedId }) {
            s.selectedId = s.personas.first?.id
        }
        return s
    }

    init() {}

    init(from decoder: Decoder) throws {
        self.init()
        let c = try decoder.container(keyedBy: CodingKeys.self)
        if let v = try c.decodeIfPresent([AvatarPersona].self, forKey: .personas) { personas = v }
        if let v = try c.decodeIfPresent(UUID.self, forKey: .selectedId) { selectedId = v }
    }
}
