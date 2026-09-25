import Foundation

/// Settings edits that must outlive a query edit.
///
/// The form's lazy and eager containers are different types, so a filter edit
/// rebuilds everything below them and takes any `@State` down there with it.
/// Rows read these instead, from `SettingsView`, which spans both containers.
final class SettingsFormState: ObservableObject {
    /// `providers.json` as it was when the pane opened, plus every edit since.
    @Published var providerEntries: [ProviderEntry] = ProvidersFile.load()

    /// Each row's raw model-id field text. The field holds what is being typed
    /// (a trailing comma included); the entry holds the parsed ids.
    @Published var providerModelText: [UUID: String] = [:]

    /// Port field text. Nil until the field has been seen — an empty string is
    /// a state the field allows (nothing is committed from it).
    @Published var portText: String?

    /// Embedded-engine rows, read once from `mlx-serve --version` — a
    /// print-and-exit that never binds a port, so they show with the server
    /// stopped.
    @Published var engineVersions: [EngineVersion] = []

    /// Held, never built here: an `AudioRecorder` owns an audio graph, and
    /// opening Settings must not touch one. Kept so a query edit cannot tear
    /// down a running capture.
    @Published private(set) var recorder: AudioRecorder?

    /// The recorder, built on the first record.
    func audioRecorder() -> AudioRecorder {
        if let recorder { return recorder }
        let made = AudioRecorder()
        recorder = made
        return made
    }
}
