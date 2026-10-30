import AppKit
import Combine
import Foundation

enum MonitoringError: Error, Equatable {
    case authentication, disabled, unsupported, oversized, storageVersion
    case http(Int)
}

struct MonitoringEndpoint: Equatable, Sendable {
    var url: URL
    var apiKey = ""
    var storageKey: String {
        var parts = URLComponents(url: url, resolvingAgainstBaseURL: false)!
        parts.user = nil; parts.password = nil; parts.query = nil; parts.fragment = nil
        parts.host = parts.host?.lowercased()
        if parts.path == "/" { parts.path = "" }
        return parts.url!.absoluteString.trimmingCharacters(in: CharacterSet(charactersIn: "/"))
    }
}

enum MonitoringHTTP {
    private final class RedirectPolicy: NSObject, URLSessionTaskDelegate, @unchecked Sendable {
        func urlSession(_ session: URLSession, task: URLSessionTask,
                        willPerformHTTPRedirection response: HTTPURLResponse, newRequest request: URLRequest,
                        completionHandler: @escaping (URLRequest?) -> Void) { completionHandler(nil) }
    }
    private static let session: URLSession = {
        let config = URLSessionConfiguration.ephemeral
        config.timeoutIntervalForRequest = 5
        config.timeoutIntervalForResource = 5
        config.waitsForConnectivity = false
        config.requestCachePolicy = .reloadIgnoringLocalCacheData
        return URLSession(configuration: config, delegate: RedirectPolicy(), delegateQueue: nil)
    }()

    static func check(status: Int, data: Data) throws {
        switch status {
        case 200: break
        case 401, 403: throw MonitoringError.authentication
        case 503: throw MonitoringError.disabled
        case 404, 405, 501: throw MonitoringError.unsupported
        default: throw MonitoringError.http(status)
        }
        guard data.count <= 2 * 1024 * 1024 else { throw MonitoringError.oversized }
    }

    static func load(_ endpoint: MonitoringEndpoint, path: String) async throws -> Data {
        let url = URL(string: endpoint.storageKey)!.appendingPathComponent(String(path.drop(while: { $0 == "/" })))
        var request = URLRequest(url: url)
        request.setValue("application/json", forHTTPHeaderField: "Accept")
        if !endpoint.apiKey.isEmpty { request.setValue("Bearer \(endpoint.apiKey)", forHTTPHeaderField: "Authorization") }
        let (data, response) = try await session.data(for: request)
        guard let response = response as? HTTPURLResponse else { throw MonitoringError.unsupported }
        try check(status: response.statusCode, data: data)
        return data
    }
}

struct MonitoringArchive: Codable {
    var version = 1
    var servers: [String: MonitoringHistory] = [:]
    mutating func compact(at now: TimeInterval) {
        for key in Array(servers.keys) {
            servers[key]?.compact(at: now)
            if servers[key]?.samples.isEmpty == true && servers[key]?.requests.isEmpty == true { servers.removeValue(forKey: key) }
        }
        let oldest = servers.keys.sorted {
            (servers[$0]?.samples.last?.time ?? 0, $0) < (servers[$1]?.samples.last?.time ?? 0, $1)
        }
        for key in oldest.prefix(max(0, oldest.count - 8)) { servers.removeValue(forKey: key) }
    }
}

final class MonitoringPersistence: @unchecked Sendable {
    static let maximumBytes = 32 * 1024 * 1024
    let url: URL
    private let queue = DispatchQueue(label: "mlx-serve.monitor-history", qos: .utility)
    init(url: URL = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
        .appendingPathComponent("MLX Core/monitor-history.json")) { self.url = url }

    func load() async throws -> MonitoringArchive {
        try await withCheckedThrowingContinuation { continuation in
            queue.async { [url] in
                do {
                    guard FileManager.default.fileExists(atPath: url.path) else {
                        continuation.resume(returning: MonitoringArchive()); return
                    }
                    let handle = try FileHandle(forReadingFrom: url)
                    defer { try? handle.close() }
                    let data = try handle.read(upToCount: Self.maximumBytes + 1) ?? Data()
                    guard data.count <= Self.maximumBytes else { throw MonitoringError.oversized }
                    let archive = try JSONDecoder().decode(MonitoringArchive.self, from: data)
                    guard archive.version == 1 else { throw MonitoringError.storageVersion }
                    continuation.resume(returning: archive)
                } catch { continuation.resume(throwing: error) }
            }
        }
    }

    func enqueue(_ archive: MonitoringArchive, completion: @escaping @Sendable (Error?) -> Void) {
        queue.async { [self] in
            do { try write(archive); completion(nil) }
            catch { completion(error) }
        }
    }

    func save(_ archive: MonitoringArchive) async throws {
        try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<Void, Error>) in
            enqueue(archive) { error in
                if let error { continuation.resume(throwing: error) }
                else { continuation.resume() }
            }
        }
    }

    func flush(_ archive: MonitoringArchive) throws {
        final class ResultBox: @unchecked Sendable { var error: Error? }
        let result = ResultBox(), done = DispatchSemaphore(value: 0)
        enqueue(archive) { result.error = $0; done.signal() }
        done.wait()
        if let error = result.error { throw error }
    }
    private func write(_ archive: MonitoringArchive) throws {
        try FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
        let data = try JSONEncoder().encode(archive)
        guard data.count <= Self.maximumBytes else { throw MonitoringError.oversized }
        try data.write(to: url, options: .atomic)
    }
}

@MainActor
final class MonitoringStore: ObservableObject {
    enum Status: Equatable {
        case stopped, connecting, live, disabled, authentication, unavailable, retrying
        var text: String {
            switch self {
            case .stopped: L10n.text("Server stopped")
            case .connecting: L10n.text("Connecting to metrics…")
            case .live: L10n.text("Live")
            case .disabled: L10n.text("Metrics disabled. Enable Metrics panel in Server settings and restart the server.")
            case .authentication: L10n.text("Monitoring authentication failed. Check the server API key.")
            case .unavailable: L10n.text("Metrics are unavailable on this server.")
            case .retrying: L10n.text("Cannot reach server metrics. Retrying…")
            }
        }
    }
    typealias Transport = @Sendable (MonitoringEndpoint, String) async throws -> Data
    @Published private(set) var snapshot: MonitoringSnapshot?
    @Published private(set) var status: Status = .stopped
    @Published private(set) var history = MonitoringHistory()
    @Published private(set) var rates = MonitoringRates()
    @Published private(set) var decodePoints: [MonitoringPoint] = []
    @Published private(set) var prefillPoints: [MonitoringPoint] = []
    @Published private(set) var models: [ModelInfo]?
    @Published private(set) var now: TimeInterval
    @Published private(set) var serverName = ""
    @Published private(set) var storageWarning: String?
    private let persistence: MonitoringPersistence
    private let transport: Transport
    private let clock: @Sendable () -> TimeInterval
    private let pollInterval: TimeInterval
    private let modelInterval: TimeInterval
    private var endpoint: MonitoringEndpoint?
    private var generation = 0
    private var archive = MonitoringArchive()
    private var loading: Task<MonitoringArchive, Error>?
    private var maintenanceTask: Task<Void, Never>?
    private var loaded = false
    private var writable = true
    private var lastSave: TimeInterval = 0
    private var liveHistory = MonitoringLiveHistory()
    private var afterFailure = false
    private var metricsTask: Task<Void, Never>?
    private var modelTask: Task<Void, Never>?
    private var observations = Set<AnyCancellable>()

    init(persistence: MonitoringPersistence = MonitoringPersistence(),
         transport: @escaping Transport = { try await MonitoringHTTP.load($0, path: $1) },
         clock: @escaping @Sendable () -> TimeInterval = { Date().timeIntervalSince1970 },
         pollInterval: TimeInterval = 1, modelInterval: TimeInterval = 15) {
        self.persistence = persistence; self.transport = transport; self.clock = clock
        self.pollInterval = pollInterval; self.modelInterval = modelInterval; now = clock()
        NotificationCenter.default.publisher(for: NSApplication.willTerminateNotification).sink { [weak self] _ in
            MainActor.assumeIsolated { self?.finish() }
        }.store(in: &observations)
    }

    func observe(_ server: ServerManager) {
        startMaintenance()
        Publishers.CombineLatest3(server.$status, server.$port, server.$lastLaunchedOptions)
            .sink { [weak self, weak server] _ in
                Task { @MainActor [weak self, weak server] in
                    guard let self, let server else { return }
                    let active = server.status == .running
                    self.setEndpoint(active ? MonitoringEndpoint(url: server.api.serverURL(port: server.port, path: ""),
                        apiKey: server.lastLaunchedOptions?.apiKey ?? "") : nil)
                }
            }.store(in: &observations)
    }

    func setEndpoint(_ next: MonitoringEndpoint?) {
        guard endpoint != next else { return }
        persist()
        generation += 1
        metricsTask?.cancel(); modelTask?.cancel()
        endpoint = next
        snapshot = nil; models = nil; rates = MonitoringRates()
        liveHistory = MonitoringLiveHistory(); decodePoints = []; prefillPoints = []
        now = clock(); afterFailure = false
        guard let next else { status = .stopped; return }
        startMaintenance()
        serverName = next.storageKey
        status = .connecting
        history = archive.servers[next.storageKey] ?? MonitoringHistory()
        let token = generation
        metricsTask = Task { [weak self] in await self?.collect(next, token: token) }
        modelTask = Task { [weak self] in await self?.collectModels(next, token: token) }
    }

    func retry() {
        guard let current = endpoint else { return }
        setEndpoint(nil); setEndpoint(current)
    }

    private func startMaintenance() {
        guard maintenanceTask == nil else { return }
        maintenanceTask = Task { [weak self] in
            await self?.restoreHistory()
            while !Task.isCancelled {
                do { try await Task.sleep(nanoseconds: 60_000_000_000) } catch { return }
                guard self != nil else { return }
                self?.persist()
            }
        }
    }

    func restoreHistory() async {
        guard !loaded else { return }
        if loading == nil { loading = Task { [persistence] in try await persistence.load() } }
        do {
            var saved = try await loading!.value
            guard !loaded else { return }
            saved.compact(at: clock())
            saved.servers.merge(archive.servers) { _, current in current }
            archive = saved
            if let key = endpoint?.storageKey ?? saved.servers.keys.max(by: {
                (saved.servers[$0]?.samples.last?.time ?? 0) < (saved.servers[$1]?.samples.last?.time ?? 0)
            }) {
                history = saved.servers[key] ?? MonitoringHistory()
                if serverName.isEmpty { serverName = key }
            }
            loaded = true; loading = nil
            persist()
        } catch {
            guard !loaded else { return }
            loaded = true; loading = nil; writable = false
            storageWarning = L10n.text("History could not be loaded. Live monitoring continues without changing the saved file.")
        }
    }

    private func collect(_ endpoint: MonitoringEndpoint, token: Int) async {
        await restoreHistory()
        while !Task.isCancelled && token == generation {
            let started = clock()
            do {
                let data = try await transport(endpoint, "/metrics.json")
                guard token == generation, !Task.isCancelled else { return }
                let feed = try MonitoringSnapshot.parse(data)
                now = clock()
                rates = liveHistory.accept(feed, at: now)
                decodePoints = liveHistory.decodePoints; prefillPoints = liveHistory.prefillPoints
                history.accept(feed, at: now, afterFailure: afterFailure)
                afterFailure = false
                snapshot = feed; status = .live
                if now - lastSave >= 15 { persist() }
            } catch {
                guard token == generation, !Task.isCancelled else { return }
                snapshot = nil; rates = MonitoringRates(); now = clock()
                liveHistory = MonitoringLiveHistory(); afterFailure = true
                switch error as? MonitoringError {
                case .authentication: status = .authentication; return
                case .disabled: status = .disabled; return
                case .unsupported, .oversized: status = .unavailable; return
                default: status = .retrying
                }
            }
            do { try await Task.sleep(nanoseconds: UInt64(max(0.01, pollInterval - (clock() - started)) * 1_000_000_000)) }
            catch { return }
        }
    }

    private func collectModels(_ endpoint: MonitoringEndpoint, token: Int) async {
        while !Task.isCancelled && token == generation {
            do {
                let data = try await transport(endpoint, "/v1/models")
                guard token == generation, !Task.isCancelled else { return }
                guard let raw = try JSONSerialization.jsonObject(with: data) as? [String: Any], raw["data"] is [[String: Any]] else {
                    throw MonitoringError.unsupported
                }
                models = APIClient.parseModelsBody(data).models
            } catch {
                guard token == generation else { return }
                models = nil
            }
            do { try await Task.sleep(nanoseconds: UInt64(max(0.01, modelInterval) * 1_000_000_000)) }
            catch { return }
        }
    }

    private func persist() {
        guard loaded, writable else { return }
        let now = clock()
        self.now = now
        history.compact(at: now)
        if let endpoint { archive.servers[endpoint.storageKey] = history }
        archive.compact(at: now)
        let copy = archive
        lastSave = now
        persistence.enqueue(copy) { [weak self] error in
            guard error != nil else { return }
            Task { @MainActor [weak self] in
                self?.storageWarning = L10n.text("History could not be saved. Live monitoring continues.")
            }
        }
    }

    private func finish() {
        metricsTask?.cancel(); modelTask?.cancel()
        guard loaded, writable else { return }
        if let endpoint { archive.servers[endpoint.storageKey] = history }
        archive.compact(at: clock())
        try? persistence.flush(archive)
    }

    deinit { metricsTask?.cancel(); modelTask?.cancel(); maintenanceTask?.cancel(); loading?.cancel() }
}
