import XCTest
@testable import MLXCore

/// Drives the REAL `DownloadManager.download(repoId:)` loop — tree listing,
/// per-file retry, resume, commit, cancel — against a stub Hugging Face origin.
///
/// `ChunkedDownloadTests` proves the transport assembles bytes correctly; this
/// proves the loop AROUND it still behaves once files arrive over N connections
/// instead of one. Those are different bugs: a transport that works and a
/// commit path that leaves a `.partial` behind both "download fine".
@MainActor
final class DownloadManagerTransferTests: XCTestCase {
    private var tempRoot: String!
    private var savedSession: URLSession!

    private let repoId = "acme/fixture-model"

    override func setUpWithError() throws {
        tempRoot = (NSTemporaryDirectory() as NSString)
            .appendingPathComponent("mlx-serve-dm-tests-\(UUID().uuidString)")
        try FileManager.default.createDirectory(atPath: tempRoot, withIntermediateDirectories: true)
        savedSession = DownloadSession.shared
        let config = URLSessionConfiguration.ephemeral
        config.protocolClasses = [HuggingFaceStubProtocol.self]
        config.httpMaximumConnectionsPerHost = 20
        DownloadSession.shared = URLSession(configuration: config)
        HuggingFaceStubProtocol.reset()
    }

    override func tearDownWithError() throws {
        DownloadSession.shared = savedSession
        HuggingFaceStubProtocol.reset()
        try? FileManager.default.removeItem(atPath: tempRoot)
    }

    // MARK: - Happy path

    func testAWholeRepoLandsIntactWithNoLeftovers() async throws {
        let files = Self.fixtureFiles()
        HuggingFaceStubProtocol.serve(repo: repoId, files: files)
        let manager = DownloadManager(modelsRoot: tempRoot)

        await manager.download(repoId: repoId)

        XCTAssertEqual(manager.downloads[repoId]?.status, .completed,
                       "error: \(manager.downloads[repoId]?.error ?? "-")")
        let dir = DownloadManager.newLayoutDir(rootDir: tempRoot, repoId: repoId)
        for (name, body) in files {
            XCTAssertEqual(try Data(contentsOf: URL(fileURLWithPath: (dir as NSString).appendingPathComponent(name))),
                           body, "\(name) did not reassemble")
        }
        XCTAssertTrue(Self.strays(in: dir).isEmpty, "left behind: \(Self.strays(in: dir))")
        XCTAssertTrue(manager.isReady(repoId))
    }

    func testTheBigFileActuallyUsedManyConnections() async throws {
        // Guard against the feature silently becoming a no-op: the plan is
        // derived from the tree listing's size, and a listing that reports 0
        // would quietly single-stream everything.
        HuggingFaceStubProtocol.serve(repo: repoId, files: Self.fixtureFiles())
        await DownloadManager(modelsRoot: tempRoot).download(repoId: repoId)

        let shardRequests = HuggingFaceStubProtocol.requests
            .filter { $0.path.hasSuffix("model.safetensors") }
        XCTAssertGreaterThan(shardRequests.count, 1, "the shard was fetched as a single stream")
        XCTAssertTrue(shardRequests.allSatisfy { $0.range != nil })
    }

    // MARK: - Failure and resume

    func testATransientFailureIsRetriedAndTheFileStillLands() async throws {
        let files = Self.fixtureFiles()
        // Every chunk of the shard dies once, then the origin behaves.
        HuggingFaceStubProtocol.serve(repo: repoId, files: files, failFirstNShardChunks: 4)
        let manager = DownloadManager(modelsRoot: tempRoot)

        await manager.download(repoId: repoId)

        XCTAssertEqual(manager.downloads[repoId]?.status, .completed,
                       "error: \(manager.downloads[repoId]?.error ?? "-")")
        let dir = DownloadManager.newLayoutDir(rootDir: tempRoot, repoId: repoId)
        XCTAssertEqual(try Data(contentsOf: URL(fileURLWithPath: (dir as NSString).appendingPathComponent("model.safetensors"))),
                       files.first { $0.0 == "model.safetensors" }!.1)
        XCTAssertTrue(Self.strays(in: dir).isEmpty, "left behind: \(Self.strays(in: dir))")
    }

    func testASecondRunSkipsWhatIsAlreadyOnDisk() async throws {
        HuggingFaceStubProtocol.serve(repo: repoId, files: Self.fixtureFiles())
        let manager = DownloadManager(modelsRoot: tempRoot)
        await manager.download(repoId: repoId)

        HuggingFaceStubProtocol.clearRequests()
        await manager.download(repoId: repoId)

        XCTAssertEqual(manager.downloads[repoId]?.status, .completed)
        let fileFetches = HuggingFaceStubProtocol.requests.filter { !$0.path.contains("/api/") }
        XCTAssertTrue(fileFetches.isEmpty, "re-fetched: \(fileFetches.map(\.path))")
    }

    // MARK: - Cancel

    func testCancelDuringATransferLeavesNoFootprint() async throws {
        HuggingFaceStubProtocol.serve(repo: repoId, files: Self.fixtureFiles(), throttle: true)
        let manager = DownloadManager(modelsRoot: tempRoot)

        let finished = expectation(description: "download settled")
        manager.start(repoId: repoId) { finished.fulfill() }
        try await Task.sleep(nanoseconds: 150_000_000)
        manager.cancel(repoId)
        await fulfillment(of: [finished], timeout: 30)

        XCTAssertNil(manager.downloads[repoId], "a cancelled download must not leave a row")
        let dir = DownloadManager.newLayoutDir(rootDir: tempRoot, repoId: repoId)
        XCTAssertFalse(FileManager.default.fileExists(atPath: dir), "cancel must leave zero footprint")
    }

    // MARK: - Every entry point uses the same transport

    /// The app has SIX download entry points (`start`, both `startGguf`
    /// overloads, `startBundle`, `download`, `downloadGguf`) reached from ~14
    /// call sites across the views. They all funnel into `download` or
    /// `downloadGguf`, and those are the only two callers of `transferFile`.
    ///
    /// A seventh entry point that hand-rolled its own `dataTask` would work
    /// perfectly and silently opt out of chunking, the shared session and the
    /// token — nothing would fail, it would just be slow again. Source-audit it.
    func testEveryTransferGoesThroughTheChunkedDownloader() throws {
        let source = try String(contentsOf: Self.downloadManagerSource, encoding: .utf8)

        XCTAssertEqual(source.components(separatedBy: "try await transferFile(").count - 1, 2,
                       "expected exactly the two file loops to transfer bytes")
        for banned in ["dataTask", "downloadTask", "FileHandle(forWritingAtPath:", "URLSession.shared"] {
            XCTAssertFalse(source.contains(banned),
                           "\(banned) in DownloadManager bypasses DownloadSession.shared / the chunked path")
        }
        // Anything that reaches Hugging Face carries the token.
        XCTAssertFalse(source.contains("URLRequest(url:") && !source.contains("hfApiRequest"),
                       "a bare URLRequest to HF skips the Authorization header")
    }

    /// The same class guard one level up: no OTHER service may fetch from
    /// huggingface.co on its own session — that's how a surface silently loses
    /// the token and starts eating rate limits.
    func testNoServiceFetchesHuggingFaceOutsideTheSharedSession() throws {
        let fm = FileManager.default
        let walker = try XCTUnwrap(fm.enumerator(at: Self.sourcesRoot, includingPropertiesForKeys: nil))
        var offenders: [String] = []
        // FILE scope, not line scope: the URL is almost always built a line or
        // two above the fetch, so a line-level check passes vacuously.
        for case let url as URL in walker where url.pathExtension == "swift" {
            let text = try String(contentsOf: url, encoding: .utf8)
            guard text.contains("huggingface.co"), text.contains("URLSession.shared") else { continue }
            offenders.append(url.lastPathComponent)
        }
        XCTAssertTrue(offenders.isEmpty,
                      "fetch HF off the shared session (no token → rate limits, no keep-alive): \(offenders)")
    }

    private static var sourcesRoot: URL {
        URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()    // MLXCoreTests
            .deletingLastPathComponent()    // Tests
            .deletingLastPathComponent()    // app
            .appendingPathComponent("Sources/MLXServe")
    }

    private static var downloadManagerSource: URL {
        sourcesRoot.appendingPathComponent("Services/DownloadManager.swift")
    }

    // MARK: - Helpers

    /// A repo shaped like a real one: small metadata files plus one shard big
    /// enough that the planner splits it (the 8 MB floor is the real one).
    private static func fixtureFiles() -> [(String, Data)] {
        [
            ("config.json", Data(#"{"model_type":"llama"}"#.utf8)),
            ("tokenizer.json", Data(#"{"version":"1.0"}"#.utf8)),
            ("model.safetensors", pseudoRandom(bytes: 40 << 20)),
        ]
    }

    /// Anything that isn't a committed file — a stranded `.partial`, an orphan
    /// chunk sidecar.
    private static func strays(in dir: String) -> [String] {
        let entries = (try? FileManager.default.contentsOfDirectory(atPath: dir)) ?? []
        return entries.filter { $0.contains(".partial") }
    }

    private static func pseudoRandom(bytes count: Int) -> Data {
        var out = Data(count: count)
        var seed: UInt64 = 0xD1B54A32D192ED03
        out.withUnsafeMutableBytes { raw in
            let p = raw.bindMemory(to: UInt8.self)
            for i in 0..<count {
                seed = seed &* 6364136223846793005 &+ 1442695040888963407
                p[i] = UInt8((seed >> 33) & 0xFF)
            }
        }
        return out
    }
}

// MARK: - Stub Hugging Face origin

/// Answers the two endpoints the download loop speaks: the recursive tree
/// listing and `resolve/main/<path>` with real Range semantics.
final class HuggingFaceStubProtocol: URLProtocol {
    struct Recorded { let path: String; let range: String? }

    private static let lock = NSLock()
    private static var repos: [String: [(String, Data)]] = [:]
    private static var throttle = false
    private static var shardChunkFailuresLeft = 0
    private static var recorded: [Recorded] = []

    static func serve(repo: String, files: [(String, Data)], throttle: Bool = false,
                      failFirstNShardChunks: Int = 0) {
        lock.lock(); defer { lock.unlock() }
        self.repos = repo.isEmpty ? [:] : [repo: files]
        self.throttle = throttle
        self.shardChunkFailuresLeft = failFirstNShardChunks
        self.recorded = []
    }

    /// Several repos at once — for a download that pulls a companion (a Gemma 4
    /// model and its assistant drafter).
    static func serve(repos: [String: [(String, Data)]]) {
        lock.lock(); defer { lock.unlock() }
        self.repos = repos
        self.throttle = false
        self.shardChunkFailuresLeft = 0
        self.recorded = []
    }

    static func reset() { serve(repo: "", files: []) }

    static func clearRequests() {
        lock.lock(); defer { lock.unlock() }
        recorded = []
    }

    static var requests: [Recorded] {
        lock.lock(); defer { lock.unlock() }
        return recorded
    }

    override class func canInit(with request: URLRequest) -> Bool { request.url?.host == "huggingface.co" }
    override class func canonicalRequest(for request: URLRequest) -> URLRequest { request }

    private var stopped = false
    override func stopLoading() { stopped = true }

    override func startLoading() {
        guard let url = request.url else { return }
        let path = url.path
        let range = request.value(forHTTPHeaderField: "Range")
        Self.lock.lock()
        Self.recorded.append(Recorded(path: path, range: range))
        let repos = Self.repos
        let throttle = Self.throttle
        Self.lock.unlock()

        if path.hasPrefix("/api/models/") {
            // /api/models/<org>/<name>/tree/main
            let repo = path.dropFirst("/api/models/".count).split(separator: "/").prefix(2).joined(separator: "/")
            guard let files = repos[repo] else {
                send(status: 404, body: Data(), headers: [:])
                return
            }
            let entries = files.map { ["type": "file", "path": $0.0, "size": $0.1.count] as [String: Any] }
            send(status: 200, body: try! JSONSerialization.data(withJSONObject: entries), headers: [:])
            return
        }

        guard let (repo, files) = repos.first(where: { path.hasPrefix("/\($0.key)/resolve/main/") }),
              let body = files.first(where: { $0.0 == String(path.dropFirst("/\(repo)/resolve/main/".count)) })?.1 else {
            send(status: 404, body: Data(), headers: [:])
            return
        }

        // Injected transient failure, spent one chunk at a time.
        if path.hasSuffix("model.safetensors") {
            let shouldFail: Bool = {
                Self.lock.lock(); defer { Self.lock.unlock() }
                guard Self.shardChunkFailuresLeft > 0 else { return false }
                Self.shardChunkFailuresLeft -= 1
                return true
            }()
            if shouldFail {
                client?.urlProtocol(self, didFailWithError: URLError(.networkConnectionLost))
                return
            }
        }

        if request.httpMethod == "HEAD" {
            send(status: 200, body: Data(), headers: ["Content-Length": "\(body.count)"], bodyless: true)
            return
        }

        var status = 200
        var slice = body
        var headers: [String: String] = [:]
        if let range, let (start, end) = Self.parseRange(range, size: body.count) {
            status = 206
            slice = body.subdata(in: start..<(end + 1))
            headers["Content-Range"] = "bytes \(start)-\(end)/\(body.count)"
        }
        headers["Content-Length"] = "\(slice.count)"
        send(status: status, body: slice, headers: headers, throttle: throttle)
    }

    private func send(status: Int, body: Data, headers: [String: String],
                      bodyless: Bool = false, throttle: Bool = false) {
        guard let url = request.url,
              let response = HTTPURLResponse(url: url, statusCode: status, httpVersion: "HTTP/1.1",
                                             headerFields: headers) else { return }
        client?.urlProtocol(self, didReceive: response, cacheStoragePolicy: .notAllowed)
        if !bodyless {
            var i = 0
            let step = 64 << 10
            while i < body.count {
                if stopped { return }
                let j = min(i + step, body.count)
                client?.urlProtocol(self, didLoad: body.subdata(in: i..<j))
                i = j
                if throttle { Thread.sleep(forTimeInterval: 0.02) }
            }
        }
        client?.urlProtocolDidFinishLoading(self)
    }

    private static func parseRange(_ header: String, size: Int) -> (Int, Int)? {
        guard header.hasPrefix("bytes="), size > 0 else { return nil }
        let parts = header.dropFirst("bytes=".count).split(separator: "-", omittingEmptySubsequences: false)
        guard let start = Int(parts.first ?? ""), start < size else { return nil }
        let end = parts.count > 1 ? (Int(parts[1]) ?? size - 1) : size - 1
        return (start, min(end, size - 1))
    }
}
