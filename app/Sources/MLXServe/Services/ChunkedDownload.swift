import Foundation

// MARK: - Errors

/// Thrown when a transfer stops making progress. The retry loop treats it as a
/// transient failure and resumes from the sidecar rather than starting over.
struct DownloadStallError: Error, LocalizedError {
    var errorDescription: String? { "Download stalled — server stopped sending data" }
}

/// Thrown when a ranged chunk request comes back as a full `200` body, or when
/// the origin's `Content-Range` total contradicts the size we planned against:
/// either way every chunk boundary is void, so the transfer falls back to one
/// stream instead of assembling a corrupt file.
struct DownloadRangeUnsupportedError: Error, LocalizedError {
    var errorDescription: String? { "Server ignored the Range header" }
}

// MARK: - Chunk planning

enum DownloadChunking {
    /// Connections opened per file. Measured against the HF CDN on a
    /// single-stream-limited line (2026-07-25, same file, same minute): 1 conn
    /// 22.6 MB/s, 8 conns 41.5 MB/s, 16 conns 46.3 MB/s. Ollama also defaults
    /// to 16.
    ///
    /// This buys NOTHING when the LINE is the limiter rather than the stream —
    /// re-measured 2026-07-26 at 1 conn 55.8 MB/s vs 8 conns 53.7 MB/s, a wash.
    /// It's insurance for the constrained case, not a speedup everywhere, so
    /// don't quote it as one without naming the link it was measured on.
    static let defaultConnections = 16
    /// Never split below this. A chunk this small spends more on connection
    /// setup and the `resolve/main` redirect than the parallelism saves.
    static let minChunkBytes: Int64 = 8 << 20
    static let maxConnections = 16
    private static let connectionsDefaultsKey = "downloadConnections"

    struct Chunk: Equatable {
        var start: Int64
        var end: Int64          // inclusive
        var length: Int64 { end - start + 1 }
    }

    static func clampConnections(_ n: Int) -> Int { min(max(n, 1), maxConnections) }

    /// Escape hatch with no UI: `defaults write <bundle-id> downloadConnections <n>`.
    static func configuredConnections(_ defaults: UserDefaults = .standard) -> Int {
        let raw = defaults.integer(forKey: connectionsDefaultsKey)
        return raw > 0 ? clampConnections(raw) : defaultConnections
    }

    /// Split `fileSize` into at most `connections` contiguous chunks of at least
    /// `minChunkBytes` each. An EMPTY result means "fetch it as one stream" —
    /// an unknown size, a file too small to be worth splitting, or one
    /// connection. Callers branch on that rather than on a size threshold of
    /// their own.
    static func plan(fileSize: Int64, connections: Int, minChunkBytes: Int64 = minChunkBytes) -> [Chunk] {
        guard fileSize > 0, minChunkBytes > 0 else { return [] }
        let affordable = max(fileSize / minChunkBytes, 1)
        let count = Int(min(Int64(max(connections, 1)), affordable))
        guard count > 1 else { return [] }

        let base = fileSize / Int64(count)
        var out: [Chunk] = []
        var start: Int64 = 0
        for i in 0..<count {
            // The last chunk absorbs the remainder, so the plan always ends
            // exactly at EOF.
            let end = (i == count - 1) ? fileSize - 1 : start + base - 1
            out.append(Chunk(start: start, end: end))
            start = end + 1
        }
        return out
    }
}

// MARK: - Resume state

/// Per-chunk byte counts for an in-flight multi-connection transfer, persisted
/// beside the `.partial` as `<name>.partial.parts`.
///
/// With N connections writing at their own offsets the `.partial` file's SIZE
/// is no longer the contiguous prefix, so it can't drive resume the way the
/// single-stream path used it. This sidecar can. It is written AFTER the bytes
/// land, so it may under-report (harmless: those bytes are refetched and
/// rewritten at the same offsets) but never over-report — which would leave a
/// hole in a file we'd then call complete.
struct ChunkedResumeState: Codable, Equatable {
    struct Entry: Codable, Equatable {
        var start: Int64
        var end: Int64          // inclusive
        var done: Int64 = 0
        var length: Int64 { end - start + 1 }
        var isComplete: Bool { done >= length }
    }

    var fileSize: Int64
    var chunks: [Entry]

    init(fileSize: Int64, chunks: [Entry]) {
        self.fileSize = fileSize
        self.chunks = chunks
    }

    init(fileSize: Int64, chunks: [DownloadChunking.Chunk]) {
        self.init(fileSize: fileSize, chunks: chunks.map { Entry(start: $0.start, end: $0.end) })
    }

    var completedBytes: Int64 { chunks.reduce(0) { $0 + $1.done } }
    var isFinished: Bool { chunks.allSatisfy { $0.isComplete } }

    /// A state is usable only when it describes the SAME file laid out
    /// contiguously from 0 to EOF. A size change upstream, or a truncated or
    /// hand-edited sidecar, means the `.partial` bytes can't be trusted.
    func isValid(forSize size: Int64) -> Bool {
        guard fileSize == size, size > 0, !chunks.isEmpty else { return false }
        var cursor: Int64 = 0
        for c in chunks {
            guard c.start == cursor, c.end >= c.start, c.done >= 0, c.done <= c.length else { return false }
            cursor = c.end + 1
        }
        return cursor == size
    }

    /// Build a plan that KEEPS an already-downloaded contiguous prefix — a
    /// `.partial` left by the old single-stream path, or by a build before this
    /// one — as a completed leading chunk, splitting only the remainder. Without
    /// this, upgrading mid-download would throw away however many GB were
    /// already on disk.
    static func planAdopting(prefix: Int64, fileSize: Int64, connections: Int,
                             minChunkBytes: Int64 = DownloadChunking.minChunkBytes) -> ChunkedResumeState {
        guard fileSize > 0 else { return ChunkedResumeState(fileSize: fileSize, chunks: [Entry]()) }
        let kept = max(min(prefix, fileSize), 0)
        guard kept > 0 else {
            return ChunkedResumeState(fileSize: fileSize,
                                      chunks: DownloadChunking.plan(fileSize: fileSize, connections: connections,
                                                                    minChunkBytes: minChunkBytes))
        }
        var entries = [Entry(start: 0, end: kept - 1, done: kept)]
        let remaining = fileSize - kept
        if remaining > 0 {
            let sub = DownloadChunking.plan(fileSize: remaining, connections: connections, minChunkBytes: minChunkBytes)
            if sub.isEmpty {
                entries.append(Entry(start: kept, end: fileSize - 1))
            } else {
                entries.append(contentsOf: sub.map { Entry(start: $0.start + kept, end: $0.end + kept) })
            }
        }
        return ChunkedResumeState(fileSize: fileSize, chunks: entries)
    }

    static func sidecarPath(forPartial partialPath: String) -> String { partialPath + ".parts" }

    static func load(forPartial partialPath: String, expectedSize: Int64) -> ChunkedResumeState? {
        guard let data = FileManager.default.contents(atPath: sidecarPath(forPartial: partialPath)),
              let state = try? JSONDecoder().decode(ChunkedResumeState.self, from: data),
              state.isValid(forSize: expectedSize) else { return nil }
        return state
    }

    func save(forPartial partialPath: String) {
        guard let data = try? JSONEncoder().encode(self) else { return }
        // Atomic: a torn sidecar strands a multi-GB partial.
        try? data.write(to: URL(fileURLWithPath: Self.sidecarPath(forPartial: partialPath)), options: .atomic)
    }

    static func remove(forPartial partialPath: String) {
        try? FileManager.default.removeItem(atPath: sidecarPath(forPartial: partialPath))
    }
}

// MARK: - Positional file

/// A file opened for positional writes. `pwrite` is thread-safe and carries its
/// own offset, so N chunk connections fill their own regions of the same
/// `.partial` concurrently with no seek races — the reason this isn't a
/// `FileHandle`, whose write cursor is shared state.
final class PositionalFile: @unchecked Sendable {
    private let fd: Int32
    private let lock = NSLock()
    private var closed = false

    init(path: String) throws {
        let opened = open(path, O_WRONLY | O_CREAT, 0o644)
        guard opened >= 0 else { throw Self.posixError() }
        fd = opened
    }

    func write(_ data: Data, at offset: Int64) throws {
        var cursor = offset
        try data.withUnsafeBytes { (raw: UnsafeRawBufferPointer) in
            guard var ptr = raw.baseAddress else { return }
            var remaining = raw.count
            while remaining > 0 {
                let written = pwrite(fd, ptr, remaining, off_t(cursor))
                if written < 0 {
                    if errno == EINTR { continue }
                    throw Self.posixError()
                }
                guard written > 0 else { throw Self.posixError(EIO) }
                remaining -= written
                ptr = ptr.advanced(by: written)
                cursor += Int64(written)
            }
        }
    }

    func truncate(to length: Int64) throws {
        guard ftruncate(fd, off_t(length)) == 0 else { throw Self.posixError() }
    }

    func close() {
        lock.lock()
        defer { lock.unlock() }
        guard !closed else { return }
        closed = true
        _ = Darwin.close(fd)
    }

    deinit { close() }

    private static func posixError(_ code: Int32 = errno) -> NSError {
        NSError(domain: NSPOSIXErrorDomain, code: Int(code),
                userInfo: [NSLocalizedDescriptionKey: String(cString: strerror(code))])
    }
}

// MARK: - Shared session

enum DownloadSession {
    /// ONE session for every model transfer. Keep-alive is the point: a repo's
    /// config/tokenizer/index files stop re-paying TCP + TLS + the
    /// `resolve/main` 302 each, and a big file's chunks reuse warmed
    /// connections. The per-host cap has to clear our connection count —
    /// URLSession's default is 6, which would silently queue chunks 7 and 8.
    /// A `var` only so tests can point the whole download loop at a stub
    /// origin; nothing in the app reassigns it.
    static var shared: URLSession = makeDefault()

    static func makeDefault() -> URLSession {
        let config = URLSessionConfiguration.default
        config.timeoutIntervalForRequest = 60
        config.timeoutIntervalForResource = 7200
        config.httpMaximumConnectionsPerHost = DownloadChunking.maxConnections + 4
        config.requestCachePolicy = .reloadIgnoringLocalCacheData
        config.urlCache = nil
        return URLSession(configuration: config)
    }
}

// MARK: - Downloader

/// Multi-connection ranged file transfer.
///
/// Opens up to `connections` HTTP range requests against the same URL, each
/// writing its own region of the `.partial` at its own offset, and persists
/// per-chunk progress so an interrupted run resumes every chunk in place.
/// Falls back to a single stream when the file is too small to split, the size
/// is unknown, or the origin doesn't honour `Range`.
final class ChunkedFileDownloader: @unchecked Sendable {
    private let url: URL
    private let partialPath: String
    private let fileSize: Int64
    private let session: URLSession
    private let headers: [String: String]
    private let connections: Int
    private let minChunkBytes: Int64

    /// `(bytesOfThisFile, bytesPerSecond)`.
    var onProgress: ((Int64, Double) -> Void)?

    private let lock = NSLock()
    private var state: ChunkedResumeState?      // nil ⇒ single-stream mode
    private var streamBytes: Int64 = 0          // single-stream counter
    private var tasks: [URLSessionTask] = []
    private var cancelled = false
    private var stalled = false
    private var lastFlush = Date.distantPast
    private var lastReport = Date.distantPast
    private var startedAt = Date()
    private var baseBytes: Int64 = 0            // bytes already on disk when this run began

    private var stallTimer: DispatchSourceTimer?
    private var stallCheckBytes: Int64 = 0
    private var slowSince: Date?
    private static let stallSpeedThreshold: Double = 10_000     // 10 KB/s, aggregate
    private static let stallTimeout: TimeInterval = 30

    init(url: URL, partialPath: String, fileSize: Int64, session: URLSession = DownloadSession.shared,
         headers: [String: String] = [:], connections: Int = DownloadChunking.defaultConnections,
         minChunkBytes: Int64 = DownloadChunking.minChunkBytes) {
        self.url = url
        self.partialPath = partialPath
        self.fileSize = fileSize
        self.session = session
        self.headers = headers
        self.connections = DownloadChunking.clampConnections(connections)
        self.minChunkBytes = minChunkBytes
    }

    /// Bytes already on disk for this file — the sidecar's total when a
    /// multi-connection transfer was interrupted, else the plain `.partial`
    /// size. Drives the "Resuming from …" status text.
    var resumableBytesOnDisk: Int64 {
        Self.resumableBytes(partialPath: partialPath, fileSize: fileSize)
    }

    static func resumableBytes(partialPath: String, fileSize: Int64) -> Int64 {
        if let state = ChunkedResumeState.load(forPartial: partialPath, expectedSize: fileSize) {
            return state.completedBytes
        }
        // No sidecar ⇒ nothing chunked was in flight, so the file size IS the
        // contiguous prefix (the single-stream path's own resume rule).
        guard !FileManager.default.fileExists(atPath: ChunkedResumeState.sidecarPath(forPartial: partialPath)) else { return 0 }
        return partialFileSize(partialPath)
    }

    private static func partialFileSize(_ path: String) -> Int64 {
        guard let attrs = try? FileManager.default.attributesOfItem(atPath: path),
              let size = attrs[.size] as? Int64 else { return 0 }
        return size
    }

    func run() async throws {
        let plan = resolvePlan()
        // Every byte is already on disk — the process died between the last
        // write and the rename. Re-streaming a finished multi-GB shard because
        // its `.partial` never got committed is the expensive way to be wrong.
        if fileSize > 0, !plan.chunks.isEmpty, plan.isFinished, plan.completedBytes >= fileSize {
            report(plan.completedBytes)
            return
        }
        if plan.chunks.count > 1 {
            do {
                try await runChunked(plan)
                return
            } catch let error as DownloadRangeUnsupportedError {
                _ = error
                // The plan is void: the origin either ignored Range or disagrees
                // about the file's size. Drop everything the plan produced and
                // take the origin at its word.
                ChunkedResumeState.remove(forPartial: partialPath)
                try? FileManager.default.removeItem(atPath: partialPath)
            }
        } else {
            ChunkedResumeState.remove(forPartial: partialPath)
        }
        try await runSingleStream(resumeFrom: Self.partialFileSize(partialPath))
    }

    // MARK: Plan resolution

    private func resolvePlan() -> ChunkedResumeState {
        if let saved = ChunkedResumeState.load(forPartial: partialPath, expectedSize: fileSize) {
            return saved
        }
        let sidecar = ChunkedResumeState.sidecarPath(forPartial: partialPath)
        if FileManager.default.fileExists(atPath: sidecar) {
            // A sidecar that no longer validates means the bytes below it belong
            // to a different artifact — start clean rather than stitch two
            // versions of a weight file together.
            ChunkedResumeState.remove(forPartial: partialPath)
            try? FileManager.default.removeItem(atPath: partialPath)
            return ChunkedResumeState(fileSize: fileSize,
                                      chunks: DownloadChunking.plan(fileSize: fileSize, connections: connections,
                                                                    minChunkBytes: minChunkBytes))
        }
        // No sidecar: any `.partial` here is a contiguous prefix from the
        // single-stream path. Keep it.
        return ChunkedResumeState.planAdopting(prefix: Self.partialFileSize(partialPath), fileSize: fileSize,
                                               connections: connections, minChunkBytes: minChunkBytes)
    }

    // MARK: Chunked transfer

    private func runChunked(_ initial: ChunkedResumeState) async throws {
        let file = try PositionalFile(path: partialPath)
        defer { file.close() }

        lock.withLock {
            state = initial
            baseBytes = initial.completedBytes
            startedAt = Date()
        }
        initial.save(forPartial: partialPath)

        startStallDetection()
        defer {
            stopStallDetection()
            flushState()
        }

        do {
            try await withTaskCancellationHandler {
                try await withThrowingTaskGroup(of: Void.self) { group in
                    for (index, entry) in initial.chunks.enumerated() where !entry.isComplete {
                        group.addTask { [weak self] in
                            guard let self else { return }
                            try await self.runChunk(index: index, file: file)
                        }
                    }
                    try await group.waitForAll()
                }
            } onCancel: {
                cancelAll()
            }
        } catch {
            throw mapTransportError(error)
        }

        let done = lock.withLock { state?.completedBytes ?? 0 }
        guard done >= fileSize else {
            // A chunk closed early. The sidecar banked what landed, so the
            // caller's retry resumes exactly the hole.
            throw URLError(.networkConnectionLost)
        }
    }

    private func runChunk(index: Int, file: PositionalFile) async throws {
        let entry = lock.withLock { state?.chunks[index] }
        guard let entry, !entry.isComplete else { return }
        let offset = entry.start + entry.done

        var request = URLRequest(url: url)
        for (key, value) in headers { request.setValue(value, forHTTPHeaderField: key) }
        request.setValue("bytes=\(offset)-\(entry.end)", forHTTPHeaderField: "Range")

        let sink = ChunkSink(file: file, offset: offset, requiresRangeResponse: true)
        sink.onTotalSize = { [weak self] total in self?.verifyTotalSize(total) }
        sink.onBytes = { [weak self] count in self?.chunkAdvanced(index: index, by: count) }
        try await perform(request, sink: sink)
    }

    /// The origin's `Content-Range` total is authoritative. If it contradicts
    /// the size we planned against, every boundary past chunk 0 is wrong.
    private func verifyTotalSize(_ total: Int64) -> Error? {
        guard total > 0, total != fileSize else { return nil }
        return DownloadRangeUnsupportedError()
    }

    private func chunkAdvanced(index: Int, by count: Int64) {
        lock.lock()
        state?.chunks[index].done += count
        let total = state?.completedBytes ?? 0
        let snapshot = state
        var flush = false
        let now = Date()
        if now.timeIntervalSince(lastFlush) > 2 {
            lastFlush = now
            flush = true
        }
        lock.unlock()

        if flush { snapshot?.save(forPartial: partialPath) }
        report(total)
    }

    private func flushState() {
        let snapshot = lock.withLock { state }
        snapshot?.save(forPartial: partialPath)
    }

    // MARK: Single-stream transfer

    private func runSingleStream(resumeFrom: Int64) async throws {
        let file = try PositionalFile(path: partialPath)
        defer { file.close() }

        // A resume offset past EOF would earn a 416 forever — only trust a
        // prefix that's actually shorter than the file.
        let offset = (fileSize > 0 && resumeFrom >= fileSize) ? 0 : max(resumeFrom, 0)

        lock.withLock {
            state = nil
            streamBytes = offset
            baseBytes = offset
            startedAt = Date()
        }

        var request = URLRequest(url: url)
        for (key, value) in headers { request.setValue(value, forHTTPHeaderField: key) }
        if offset > 0 { request.setValue("bytes=\(offset)-", forHTTPHeaderField: "Range") }

        let sink = ChunkSink(file: file, offset: offset, requiresRangeResponse: false)
        sink.onRestart = { [weak self] in self?.streamRestarted() }
        sink.onBytes = { [weak self] count in self?.streamAdvanced(by: count) }

        startStallDetection()
        defer { stopStallDetection() }

        do {
            try await withTaskCancellationHandler {
                try await perform(request, sink: sink)
            } onCancel: {
                cancelAll()
            }
        } catch {
            throw mapTransportError(error)
        }
    }

    private func streamRestarted() {
        lock.lock()
        streamBytes = 0
        baseBytes = 0
        startedAt = Date()
        lock.unlock()
    }

    private func streamAdvanced(by count: Int64) {
        lock.lock()
        streamBytes += count
        let total = streamBytes
        lock.unlock()
        report(total)
    }

    // MARK: Transport plumbing

    private func perform(_ request: URLRequest, sink: ChunkSink) async throws {
        let task = session.dataTask(with: request)
        task.delegate = sink

        let alreadyCancelled = lock.withLock { () -> Bool in
            tasks.append(task)
            return cancelled
        }
        if alreadyCancelled { task.cancel() }

        try await withTaskCancellationHandler {
            try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<Void, Error>) in
                // Set before `resume()`, so no callback can beat it.
                sink.onComplete = { error in
                    if let error { continuation.resume(throwing: error) } else { continuation.resume() }
                }
                task.resume()
            }
        } onCancel: {
            task.cancel()
        }
    }

    private func cancelAll() {
        lock.lock()
        cancelled = true
        let snapshot = tasks
        lock.unlock()
        snapshot.forEach { $0.cancel() }
    }

    /// A stalled transfer reaches us as a plain URLSession cancellation (we
    /// cancelled it). Only the flag can tell that apart from the user hitting
    /// Cancel, which must stay a cancellation all the way up.
    private func mapTransportError(_ error: Error) -> Error {
        let stalledOut = lock.withLock { stalled }
        guard stalledOut, !Task.isCancelled else { return error }
        return DownloadStallError()
    }

    private func currentBytes() -> Int64 {
        lock.withLock { state?.completedBytes ?? streamBytes }
    }

    private func report(_ total: Int64) {
        lock.lock()
        let now = Date()
        guard now.timeIntervalSince(lastReport) > 0.25 else { lock.unlock(); return }
        lastReport = now
        let elapsed = now.timeIntervalSince(startedAt)
        let base = baseBytes
        lock.unlock()

        let speed = elapsed > 0 ? Double(total - base) / elapsed : 0
        onProgress?(total, max(speed, 0))
    }

    // MARK: Stall detection

    private func startStallDetection() {
        stallCheckBytes = currentBytes()
        slowSince = nil
        let timer = DispatchSource.makeTimerSource(queue: .global(qos: .utility))
        timer.schedule(deadline: .now() + 5, repeating: 5)
        timer.setEventHandler { [weak self] in self?.checkForStall() }
        timer.resume()
        stallTimer = timer
    }

    private func stopStallDetection() {
        stallTimer?.cancel()
        stallTimer = nil
    }

    private func checkForStall() {
        let current = currentBytes()
        let recentSpeed = Double(current - stallCheckBytes) / 5.0
        stallCheckBytes = current

        // Push the real rate even when nothing is arriving — otherwise the UI
        // keeps showing the last healthy speed on a dead connection.
        onProgress?(current, max(recentSpeed, 0))

        guard recentSpeed < Self.stallSpeedThreshold else {
            slowSince = nil
            return
        }
        if slowSince == nil {
            slowSince = Date()
        } else if Date().timeIntervalSince(slowSince!) > Self.stallTimeout {
            lock.withLock { stalled = true }
            cancelAll()
            stopStallDetection()
        }
    }
}

// MARK: - Per-task delegate

/// Writes one connection's bytes at its own offset. One sink serves exactly one
/// `URLSessionDataTask` (assigned via `task.delegate`), so its own callbacks are
/// already serialized and it needs no lock of its own.
private final class ChunkSink: NSObject, URLSessionDataDelegate {
    private let file: PositionalFile
    private var offset: Int64
    private let requiresRangeResponse: Bool
    private var failure: Error?
    private var finished = false

    var onBytes: ((Int64) -> Void)?
    var onComplete: ((Error?) -> Void)?
    /// Called when the origin ignored a resume `Range` on the single-stream
    /// path — the file starts over from byte 0.
    var onRestart: (() -> Void)?
    /// The total file size the origin reports in `Content-Range`; returning an
    /// error aborts the chunk.
    var onTotalSize: ((Int64) -> Error?)?

    init(file: PositionalFile, offset: Int64, requiresRangeResponse: Bool) {
        self.file = file
        self.offset = offset
        self.requiresRangeResponse = requiresRangeResponse
    }

    func urlSession(_ session: URLSession, dataTask: URLSessionDataTask,
                    didReceive response: URLResponse,
                    completionHandler: @escaping (URLSession.ResponseDisposition) -> Void) {
        let status = (response as? HTTPURLResponse)?.statusCode ?? 0
        switch status {
        case 206:
            if let total = Self.contentRangeTotal(response), let error = onTotalSize?(total) {
                failure = error
                completionHandler(.cancel)
                return
            }
            completionHandler(.allow)
        case 200:
            guard !requiresRangeResponse else {
                failure = DownloadRangeUnsupportedError()
                completionHandler(.cancel)
                return
            }
            // Origin ignored our resume Range and is sending the whole file.
            offset = 0
            onRestart?()
            do {
                try file.truncate(to: 0)
            } catch {
                failure = error
                completionHandler(.cancel)
                return
            }
            completionHandler(.allow)
        default:
            failure = URLError(.badServerResponse, userInfo: [NSLocalizedDescriptionKey: "HTTP \(status)"])
            completionHandler(.cancel)
        }
    }

    func urlSession(_ session: URLSession, dataTask: URLSessionDataTask, didReceive data: Data) {
        guard failure == nil else { return }
        do {
            try file.write(data, at: offset)
        } catch {
            failure = error
            dataTask.cancel()
            return
        }
        offset += Int64(data.count)
        onBytes?(Int64(data.count))
    }

    func urlSession(_ session: URLSession, task: URLSessionTask, didCompleteWithError error: Error?) {
        guard !finished else { return }
        finished = true
        // The file handle is SHARED across chunks — the downloader closes it.
        onComplete?(failure ?? error)
    }

    /// `Content-Range: bytes 0-99/12345` → 12345. nil for `*`.
    private static func contentRangeTotal(_ response: URLResponse) -> Int64? {
        guard let http = response as? HTTPURLResponse,
              let header = http.value(forHTTPHeaderField: "Content-Range"),
              let slash = header.lastIndex(of: "/") else { return nil }
        return Int64(header[header.index(after: slash)...].trimmingCharacters(in: .whitespaces))
    }
}
