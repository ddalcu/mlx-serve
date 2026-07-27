import AppKit
import Foundation

/// Uploaded voice clips, kept as a small named library instead of one file.
///
/// The Settings clip has always been a single `voice-clips/voice-clone.wav` that
/// every upload OVERWRITES — fine when the app had one voice, wrong the moment
/// agents each pick their own: a second upload silently cost you the first, and
/// re-recording in Settings would have changed an agent's voice under it. Clips
/// now live beside it under their own names, so several agents can point at
/// different ones and you can come back to a clip later.
///
/// Same folder on purpose (one place to look, one thing to back up); the global
/// clip is filtered OUT of the library listing so it can't show up twice.
enum VoiceClipLibrary {

    struct Clip: Identifiable, Equatable, Hashable {
        var name: String
        var path: String
        var id: String { path }
    }

    /// Extensions we store. Everything is normalized to wav on install, so this
    /// is really "what a previous install could have left here".
    static let audioExtensions: Set<String> = ["wav", "mp3", "m4a", "aiff", "aif", "caf", "flac"]

    static var directory: String { VoiceCloneClipStore.directory }

    /// The global Settings clip's basename — never a library entry.
    static let globalClipBasename = "voice-clone"

    // MARK: - Naming (pure)

    /// A filename-safe label derived from the picked file's own name, so a clip
    /// reads as "Morgan Freeman" rather than a hash.
    static func basename(for sourceName: String) -> String {
        let stem = (sourceName as NSString).deletingPathExtension
        var cleaned = stem.components(separatedBy: CharacterSet(charactersIn: "/\\:.*?\"<>|"))
            .joined(separator: "-")
            .trimmingCharacters(in: .whitespacesAndNewlines)
        while cleaned.contains("--") { cleaned = cleaned.replacingOccurrences(of: "--", with: "-") }
        cleaned = cleaned.trimmingCharacters(in: CharacterSet(charactersIn: "-"))
        if cleaned.isEmpty { return "Voice clip" }
        // Taking the global clip's name would put an agent and the Settings
        // picker on the same file.
        if cleaned == globalClipBasename { return "\(cleaned)-clip" }
        return cleaned
    }

    /// `morgan` → `morgan-2` when taken. Keeps the user's own name visible
    /// instead of appending a UUID.
    static func uniqueBasename(_ base: String, existing: [String]) -> String {
        guard existing.contains(base) else { return base }
        var n = 2
        while existing.contains("\(base)-\(n)") { n += 1 }
        return "\(base)-\(n)"
    }

    static func displayName(forPath path: String) -> String {
        ((path as NSString).lastPathComponent as NSString).deletingPathExtension
    }

    // MARK: - Listing

    static func clips(in directory: String = VoiceClipLibrary.directory) -> [Clip] {
        let files = (try? FileManager.default.contentsOfDirectory(atPath: directory)) ?? []
        return files
            .filter { audioExtensions.contains(($0 as NSString).pathExtension.lowercased()) }
            .filter { ($0 as NSString).deletingPathExtension != globalClipBasename }
            .map { Clip(name: ($0 as NSString).deletingPathExtension,
                        path: (directory as NSString).appendingPathComponent($0)) }
            .sorted { $0.name.localizedCaseInsensitiveCompare($1.name) == .orderedAscending }
    }

    // MARK: - Install

    /// Normalize `source` (24 kHz mono wav — what the TTS reference path wants)
    /// and store it under its own name. `normalize` is injected so the naming and
    /// collision rules are testable without an audio file.
    @discardableResult
    static func install(_ source: URL,
                        in directory: String = VoiceClipLibrary.directory,
                        normalize: (URL) throws -> URL = AudioReference.normalizedReferenceWav(fromFile:)) throws -> Clip {
        let normalized = try normalize(source)
        try FileManager.default.createDirectory(atPath: directory, withIntermediateDirectories: true)
        let taken = clips(in: directory).map(\.name)
        let name = uniqueBasename(basename(for: source.lastPathComponent), existing: taken)
        let dest = (directory as NSString).appendingPathComponent("\(name).wav")
        try? FileManager.default.removeItem(atPath: dest)
        try FileManager.default.copyItem(at: normalized, to: URL(fileURLWithPath: dest))
        return Clip(name: name, path: dest)
    }

    /// Pick an audio file and add it to the library. nil when the user cancels.
    @MainActor
    static func pickAndInstall() throws -> Clip? {
        let panel = NSOpenPanel()
        panel.allowedContentTypes = [.audio, .wav, .mp3, .mpeg4Audio, .aiff]
        panel.canChooseFiles = true
        panel.canChooseDirectories = false
        panel.allowsMultipleSelection = false
        panel.prompt = "Add Voice"
        guard AppActivation.runModal(panel) == .OK, let url = panel.url else { return nil }
        return try install(url)
    }

    /// The folder itself — clips are deleted/renamed in Finder, like skills.
    @MainActor
    static func revealInFinder() {
        try? FileManager.default.createDirectory(atPath: directory, withIntermediateDirectories: true)
        NSWorkspace.shared.open(URL(fileURLWithPath: directory))
    }
}
