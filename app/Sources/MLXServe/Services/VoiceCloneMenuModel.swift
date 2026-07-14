import AppKit
import Foundation

/// Pure decisions behind the voice picker's clone integration (tray panel +
/// voice overlay): when the cloned voice is what actually speaks, what the
/// collapsed picker shows, and why the clone rows are disabled. The picker
/// treats "My voice" as a first-class voice beside the Apple system voices —
/// previously it showed only the Apple fallback ("Jamie") even while every
/// sentence was synthesized in the cloned voice.
enum VoiceCloneMenuModel {

    /// The clone speaks only when a clip is set, the user hasn't switched
    /// back to a system voice, AND the Qwen3-TTS model is on disk (without it
    /// every sentence silently falls back to the system voice — the UI must
    /// not claim otherwise).
    static func cloneIsActive(clipPath: String, cloneEnabled: Bool, ttsModelDownloaded: Bool) -> Bool {
        !clipPath.isEmpty && cloneEnabled && ttsModelDownloaded
    }

    /// Longest clip name the tray will render. The label is a FILENAME the user
    /// picked, so it can be arbitrarily long — and the menu-bar panel is a fixed
    /// narrow column that a 200-character name blows out sideways, dragging the
    /// whole tray layout with it. Clamp at the source: both places that render
    /// the name go through the helpers below.
    static let maxClipLabelLength = 10

    /// The clip name as the tray may show it: never longer than
    /// `maxClipLabelLength`, with an ellipsis marking what was cut. The stored
    /// `voiceCloneLabel` keeps its full value — this is display-only.
    static func clipDisplayName(_ label: String, maxLength: Int = maxClipLabelLength) -> String {
        let trimmed = label.trimmingCharacters(in: .whitespacesAndNewlines)
        guard trimmed.count > maxLength, maxLength > 0 else { return trimmed }
        return String(trimmed.prefix(maxLength)) + "…"
    }

    /// Collapsed picker label: the clip's name while the clone is active,
    /// otherwise the system voice that will actually speak.
    static func collapsedLabel(clipPath: String, cloneEnabled: Bool, ttsModelDownloaded: Bool,
                               cloneLabel: String, systemVoiceName: String?) -> String {
        if cloneIsActive(clipPath: clipPath, cloneEnabled: cloneEnabled,
                         ttsModelDownloaded: ttsModelDownloaded) {
            let name = clipDisplayName(cloneLabel)
            return name.isEmpty ? "My voice" : name
        }
        return systemVoiceName ?? "Voice"
    }

    /// Menu-row title for the clone entry.
    static func cloneItemTitle(label: String) -> String {
        let name = clipDisplayName(label)
        return name.isEmpty ? "My voice" : "My voice — \(name)"
    }

    /// Why the clone rows are disabled; nil when cloning can work. Having no
    /// clip is NOT a reason — that's what "Choose audio file…" is for.
    static func cloneUnavailableReason(ttsModelDownloaded: Bool) -> String? {
        ttsModelDownloaded ? nil
            : "Requires the Qwen3-TTS voice model — download it from the Audio tile in the menu bar."
    }

    // MARK: - Disk seams (not unit-tested)

    /// Is the Audio pane's TTS model on disk? Cheap directory check; cache the
    /// result per panel appearance — don't call per render.
    @MainActor
    static func ttsModelDownloaded() -> Bool {
        ServerManager.resolveModelDir(repo: AudioGenSettings.load().resolvedModel.repo) != nil
    }

    /// The Settings ▸ Voice "Choose file…" flow, reused by the picker menu:
    /// NSOpenPanel → normalize to 24 kHz mono WAV → persist to the stable
    /// clip location. Returns (path, displayLabel); nil on user cancel.
    @MainActor
    static func pickAndPersistClip() throws -> (path: String, label: String)? {
        let panel = NSOpenPanel()
        panel.allowedContentTypes = [.audio, .wav, .mp3, .mpeg4Audio, .aiff]
        panel.canChooseFiles = true
        panel.canChooseDirectories = false
        guard AppActivation.runModal(panel) == .OK, let url = panel.url else { return nil }
        let normalized = try AudioReference.normalizedReferenceWav(fromFile: url)
        return (VoiceCloneClipStore.persist(normalized), url.lastPathComponent)
    }
}
