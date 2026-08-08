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

    /// Kokoro speaks only when it is the selected engine AND its checkpoint is
    /// on disk. Same honesty rule as the clone: without the model every sentence
    /// silently falls back to the system voice, so the menu must not tick it.
    static func kokoroIsActive(engine: VoiceEngine, kokoroDownloaded: Bool) -> Bool {
        engine == .kokoro && kokoroDownloaded
    }

    /// What the collapsed tray label shows, given the whole engine choice. One
    /// function so the label can never disagree with what will actually speak.
    static func collapsedLabel(engine: VoiceEngine,
                               clipPath: String,
                               cloneEnabled: Bool,
                               ttsModelDownloaded: Bool,
                               kokoroDownloaded: Bool,
                               kokoroVoice: String,
                               cloneLabel: String,
                               systemVoiceName: String?) -> String {
        if kokoroIsActive(engine: engine, kokoroDownloaded: kokoroDownloaded) {
            return clipDisplayName(KokoroVoiceCatalog.blendDisplayName(for: kokoroVoice))
        }
        if engine == .clone,
           cloneIsActive(clipPath: clipPath, cloneEnabled: cloneEnabled,
                         ttsModelDownloaded: ttsModelDownloaded) {
            let name = clipDisplayName(cloneLabel)
            return name.isEmpty ? "My voice" : name
        }
        return systemVoiceName ?? "Voice"
    }

    /// Why the Kokoro rows are disabled; nil when it can speak.
    static func kokoroUnavailableReason(kokoroDownloaded: Bool) -> String? {
        kokoroDownloaded ? nil
            : "Requires the Kokoro voice model — download it in Settings ▸ Voice."
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

    /// Is a voice backend COMPLETE on disk? The bundle's ready markers, not a
    /// bare `config.json` check: the app downloads these itself now, and
    /// `config.json` is one of the first files a 345 MB pull lands — so a
    /// dir-exists test says "available" while the weights and (for Kokoro) the
    /// `g2p/` dictionaries are still in flight, and picking a voice then gets
    /// silence. Same answer the Settings download bar gives, so the two
    /// surfaces can't disagree. Cache per panel appearance — these stat the
    /// disk, don't call them per render.
    @MainActor
    private static func bundleOnDisk(_ preset: AudioModelPreset) -> Bool {
        preset.bundle.components.allSatisfy {
            DownloadManager.componentReady($0, roots: ModelRoots().ownedRoots)
        }
    }

    /// Which cloning model the voice path will ACTUALLY load: the Audio pane's
    /// configured preset when it's on disk, else any other cloning-capable one
    /// that is. nil when none are downloaded.
    ///
    /// Resolving against the disk rather than trusting the setting is the fix for
    /// a silent failure: the default is the 0.6B repo, so a machine holding only
    /// the 1.7B variants had `synthesize` return nil and every cloned sentence
    /// come out in the system voice while the picker still showed the clip as
    /// active. Kokoro is never a candidate — it cannot clone (`ref_audio` is a
    /// named 400 there), which is exactly why `AudioModelPreset.all` is the
    /// cloning-capable catalog. Pure, with the disk check injected.
    static func resolvedCloneModel(configured: AudioModelPreset,
                                   isDownloaded: (AudioModelPreset) -> Bool) -> AudioModelPreset? {
        if configured.supportsCloning, isDownloaded(configured) { return configured }
        return AudioModelPreset.all.first { $0.supportsCloning && isDownloaded($0) }
    }

    /// Can a cloned voice actually speak right now? The picker's enabled state and
    /// the synthesizer read THIS, so the UI can't tick a voice that won't work.
    static func cloneAvailable(configured: AudioModelPreset,
                               isDownloaded: (AudioModelPreset) -> Bool) -> Bool {
        resolvedCloneModel(configured: configured, isDownloaded: isDownloaded) != nil
    }

    /// The live-disk resolution (production seam).
    @MainActor
    static func resolvedCloneModel() -> AudioModelPreset? {
        resolvedCloneModel(configured: AudioGenSettings.load().resolvedModel,
                           isDownloaded: bundleOnDisk)
    }

    /// Is a cloning backend ready? (Was "is the configured one on disk", which
    /// ignored every other downloaded variant.)
    @MainActor
    static func ttsModelDownloaded() -> Bool {
        resolvedCloneModel() != nil
    }

    /// Is the Kokoro checkpoint ready?
    @MainActor
    static func kokoroModelDownloaded() -> Bool {
        bundleOnDisk(AudioModelPreset.kokoro82M)
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
