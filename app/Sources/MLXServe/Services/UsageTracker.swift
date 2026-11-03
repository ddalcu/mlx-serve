import Foundation

/// The running tally behind the chat window's status strip and the usage dashboard: polls the server's counters
/// while it runs, turns what each look adds into a per-day, per-model history on disk, and publishes the live rates.
@MainActor
final class UsageTracker: ObservableObject {
    /// The newest reading of the server's counters; nil while it is not answering.
    @Published private(set) var live: ThroughputSnapshot?
    @Published private(set) var decodeTPS: Double?
    @Published private(set) var prefillTPS: Double?
    @Published private(set) var summary = UsageSummary()

    private var ledger = UsageLedger()
    private var store: UsageStore?
    private var storeOpened = false
    private let path: String
    private let today: () -> String
    private let fetch: () async -> ThroughputSnapshot?
    private var poll: Task<Void, Never>?

    /// The file is opened by the first thing that needs it, so a tracker nobody asks anything of costs no disk.
    init(path: String = UsageStore.defaultPath, today: @escaping () -> String = { UsageDay.key(Date()) },
         fetch: @escaping () async -> ThroughputSnapshot?) {
        self.path = path
        self.today = today
        self.fetch = fetch
    }

    /// How often to look: every second while a request is served, so the rates and the prefill bar move, else at rest.
    nonisolated static func interval(busy: Bool) -> Duration { busy ? .seconds(1) : .seconds(2) }

    func start() {
        guard poll == nil else { return }
        poll = Task { [weak self] in
            while !Task.isCancelled {
                guard let self else { return }
                let snapshot = await self.fetch()
                guard !Task.isCancelled else { return }
                if let snapshot { self.ingest(snapshot) } else { self.clearLive() }
                try? await Task.sleep(for: Self.interval(busy: snapshot?.busy ?? false))
            }
        }
    }

    func stop() {
        poll?.cancel()
        poll = nil
        clearLive()
    }

    /// Reads the stored history into `summary`.
    func load() {
        guard let store = openStore() else { return }
        summary = UsageSummary(rows: (try? store.rows()) ?? [])
    }

    func ingest(_ snapshot: ThroughputSnapshot) {
        if let previous = live {
            decodeTPS = snapshot.decodeTPS(since: previous)
            prefillTPS = snapshot.prefillTPS(since: previous)
        }
        live = snapshot
        let deltas = ledger.ingest(snapshot, day: today())
        guard !deltas.isEmpty, let store = openStore() else { return }
        try? store.add(deltas)
        load()
    }

    /// Forgets one model's history, or all of it.
    func reset(model: String?) {
        guard let store = openStore() else { return }
        try? store.reset(model: model)
        load()
    }

    private func clearLive() {
        live = nil
        decodeTPS = nil
        prefillTPS = nil
    }

    private func openStore() -> UsageStore? {
        if !storeOpened {
            storeOpened = true
            store = try? UsageStore(path: path)
        }
        return store
    }
}
