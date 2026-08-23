import Foundation

/// Per-tab composer drafts. `ChatDetailView` is REUSED across tabs (no
/// `.id(sessionId)`), so without a keyed store the half-typed message in one
/// chat silently rode along when you switched to another — and sending it there.
/// Stash on tab switch, restore on return; a sent (or cleared) field must not
/// resurrect, so whitespace-only text stores as nothing.
struct ComposerDrafts {
    private var storage: [UUID: String] = [:]

    mutating func stash(_ text: String, for id: UUID) {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else {
            storage.removeValue(forKey: id)
            return
        }
        storage[id] = text
    }

    func restore(for id: UUID) -> String {
        storage[id] ?? ""
    }

    mutating func clear(for id: UUID) {
        storage.removeValue(forKey: id)
    }
}

/// Persists beside the chat history, so a half-typed message outlives a relaunch.
/// UUID keys encode as their string forms; the whole store is one small file.
extension ComposerDrafts: Codable {
    private enum CodingKeys: String, CodingKey {
        case drafts
    }

    init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        storage = try c.decodeIfPresent([UUID: String].self, forKey: .drafts) ?? [:]
    }

    func encode(to encoder: Encoder) throws {
        var c = encoder.container(keyedBy: CodingKeys.self)
        try c.encode(storage, forKey: .drafts)
    }
}
