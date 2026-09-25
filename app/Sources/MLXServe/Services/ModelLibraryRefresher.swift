import Foundation

/// Runs the model-library scan off the main thread and publishes the newest result.
///
/// The walk reads every served root, and `AppState.refreshModels()` is reached
/// from UI actions (a section switch, a finished transfer, a picker opening), so
/// it must not run on the main actor.
@MainActor
final class ModelLibraryRefresher {
    typealias Scan = @Sendable (DownloadManager.LocalScanInputs) -> [LocalModel]
    typealias Apply = @MainActor ([LocalModel]) -> Void

    private var generation = 0

    /// The scan itself, off the main actor, with no publishing. An owner whose
    /// next step needs the list — launch deciding whether to preload a model —
    /// awaits this and applies the result itself.
    func scan(
        inputs: DownloadManager.LocalScanInputs,
        scan: @escaping Scan = DownloadManager.discoverLocalModels
    ) async -> [LocalModel] {
        await Task.detached(priority: .utility) { scan(inputs) }.value
    }

    /// Returns immediately; `apply` runs on the main actor once the scan lands.
    /// A scan superseded by a newer one is dropped rather than applied.
    func refresh(
        inputs: DownloadManager.LocalScanInputs,
        scan: @escaping Scan = DownloadManager.discoverLocalModels,
        apply: @escaping Apply
    ) {
        generation &+= 1
        let generation = self.generation
        Task { [weak self] in
            guard let self else { return }
            // Detached: a plain `Task` inherits the main actor and puts the walk back on it.
            let models = await Task.detached(priority: .utility) { scan(inputs) }.value
            guard !Task.isCancelled, self.generation == generation else { return }
            apply(models)
        }
    }
}
