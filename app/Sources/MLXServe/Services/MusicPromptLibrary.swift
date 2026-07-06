import Foundation

/// Persisted store of the user's saved style prompts and lyrics for the Music
/// tab's Examples menus. Thin glue over `MusicPromptStore` (the pure list
/// logic) + UserDefaults; deduped by title, newest-first, uncapped.
@MainActor
final class MusicPromptLibrary: ObservableObject {
    @Published private(set) var savedStyles: [MusicPrompt] = []
    @Published private(set) var savedLyrics: [MusicPrompt] = []

    private let defaults: UserDefaults
    private let key: String

    /// `defaults`/`suiteName` are injectable so tests don't touch `.standard`.
    init(defaults: UserDefaults = .standard, key: String = "musicPromptLibrary") {
        self.defaults = defaults
        self.key = key
        load()
    }

    private struct Persisted: Codable {
        var styles: [MusicPrompt]
        var lyrics: [MusicPrompt]
    }

    func saveStyle(title: String, body: String) {
        guard let p = makePrompt(title: title, body: body) else { return }
        savedStyles = MusicPromptStore.adding(p, to: savedStyles)
        persist()
    }

    func saveLyrics(title: String, body: String) {
        guard let p = makePrompt(title: title, body: body) else { return }
        savedLyrics = MusicPromptStore.adding(p, to: savedLyrics)
        persist()
    }

    func deleteStyle(title: String) {
        savedStyles = MusicPromptStore.removing(title: title, from: savedStyles)
        persist()
    }

    func deleteLyrics(title: String) {
        savedLyrics = MusicPromptStore.removing(title: title, from: savedLyrics)
        persist()
    }

    /// Empty title/body are ignored; a blank title falls back to an auto-title.
    private func makePrompt(title: String, body: String) -> MusicPrompt? {
        let trimmedBody = body.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmedBody.isEmpty else { return nil }
        var t = title.trimmingCharacters(in: .whitespacesAndNewlines)
        if t.isEmpty { t = MusicPromptStore.autoTitle(from: body) }
        return MusicPrompt(title: t, body: body)
    }

    private func load() {
        guard let data = defaults.data(forKey: key),
              let p = try? JSONDecoder().decode(Persisted.self, from: data) else { return }
        savedStyles = p.styles
        savedLyrics = p.lyrics
    }

    private func persist() {
        let p = Persisted(styles: savedStyles, lyrics: savedLyrics)
        if let data = try? JSONEncoder().encode(p) {
            defaults.set(data, forKey: key)
        }
    }
}
