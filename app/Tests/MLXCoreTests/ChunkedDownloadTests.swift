import XCTest
@testable import MLXCore

/// Multi-connection ranged downloads. Planning and resume bookkeeping are pure;
/// the transport is exercised end to end against a `URLProtocol` stub that
/// speaks HTTP Range, so the tests prove the ASSEMBLED BYTES — a chunk writing
/// at the wrong offset is exactly the bug that arithmetic-only tests miss.
final class ChunkedDownloadTests: XCTestCase {
    private var tempRoot: String!

    override func setUpWithError() throws {
        tempRoot = (NSTemporaryDirectory() as NSString)
            .appendingPathComponent("mlx-serve-chunk-tests-\(UUID().uuidString)")
        try FileManager.default.createDirectory(atPath: tempRoot, withIntermediateDirectories: true)
        RangeStubProtocol.reset()
    }

    override func tearDownWithError() throws {
        try? FileManager.default.removeItem(atPath: tempRoot)
        RangeStubProtocol.reset()
    }

    // MARK: - Planning

    func testPlanSkipsAFileTooSmallToSplit() {
        // 4 MB with an 8 MB floor can't afford two chunks — one stream.
        XCTAssertTrue(DownloadChunking.plan(fileSize: 4 << 20, connections: 8, minChunkBytes: 8 << 20).isEmpty)
    }

    func testPlanCapsChunkCountByTheMinimumChunkSize() {
        // 40 MB / 8 MB = 5 affordable chunks even though 8 connections are allowed.
        let chunks = DownloadChunking.plan(fileSize: 40 << 20, connections: 8, minChunkBytes: 8 << 20)
        XCTAssertEqual(chunks.count, 5)
    }

    func testPlanUsesEveryConnectionOnALargeFile() {
        let chunks = DownloadChunking.plan(fileSize: 4 << 30, connections: 8, minChunkBytes: 8 << 20)
        XCTAssertEqual(chunks.count, 8)
    }

    func testPlanCoversEveryByteExactlyOnce() {
        let size: Int64 = 100_000_003   // deliberately not divisible
        let chunks = DownloadChunking.plan(fileSize: size, connections: 8, minChunkBytes: 8 << 20)
        XCTAssertEqual(chunks.first?.start, 0)
        XCTAssertEqual(chunks.last?.end, size - 1)
        XCTAssertEqual(chunks.reduce(Int64(0)) { $0 + $1.length }, size, "chunks must sum to the file")
        for (a, b) in zip(chunks, chunks.dropFirst()) {
            XCTAssertEqual(b.start, a.end + 1, "no gaps, no overlaps")
        }
    }

    func testPlanReturnsNothingForAnUnknownSize() {
        // HEAD gave us nothing — we can't range-split what we can't measure.
        XCTAssertTrue(DownloadChunking.plan(fileSize: 0, connections: 8, minChunkBytes: 8 << 20).isEmpty)
        XCTAssertTrue(DownloadChunking.plan(fileSize: -1, connections: 8, minChunkBytes: 8 << 20).isEmpty)
    }

    func testPlanReturnsNothingForASingleConnection() {
        XCTAssertTrue(DownloadChunking.plan(fileSize: 4 << 30, connections: 1, minChunkBytes: 8 << 20).isEmpty)
    }

    func testConnectionCountIsClamped() {
        XCTAssertEqual(DownloadChunking.clampConnections(0), 1)
        XCTAssertEqual(DownloadChunking.clampConnections(-3), 1)
        XCTAssertEqual(DownloadChunking.clampConnections(8), 8)
        XCTAssertEqual(DownloadChunking.clampConnections(999), DownloadChunking.maxConnections)
    }

    // MARK: - Resume state

    func testResumeStateRoundTripsThroughTheSidecar() throws {
        let partial = tmp("a.partial")
        var st = ChunkedResumeState(fileSize: 300, chunks: [
            .init(start: 0, end: 99), .init(start: 100, end: 199), .init(start: 200, end: 299),
        ])
        st.chunks[1].done = 40
        st.save(forPartial: partial)

        let back = ChunkedResumeState.load(forPartial: partial, expectedSize: 300)
        XCTAssertEqual(back, st)
        XCTAssertEqual(back?.completedBytes, 40)
    }

    func testResumeStateRejectsASizeMismatch() throws {
        // The upstream file changed — the bytes on disk describe a different
        // artifact, so the partial must not be adopted.
        let partial = tmp("b.partial")
        ChunkedResumeState(fileSize: 300, chunks: [.init(start: 0, end: 299)]).save(forPartial: partial)
        XCTAssertNil(ChunkedResumeState.load(forPartial: partial, expectedSize: 301))
    }

    func testResumeStateRejectsAGappedOrOverlongPlan() {
        let gapped = ChunkedResumeState(fileSize: 300, chunks: [.init(start: 0, end: 99), .init(start: 150, end: 299)])
        XCTAssertFalse(gapped.isValid(forSize: 300))

        var overrun = ChunkedResumeState(fileSize: 300, chunks: [.init(start: 0, end: 299)])
        overrun.chunks[0].done = 400
        XCTAssertFalse(overrun.isValid(forSize: 300), "a chunk cannot have written more than its own range")
    }

    func testAdoptedPrefixBecomesACompletedLeadingChunk() {
        // A `.partial` left by the old single-stream path (or an earlier run):
        // keep the bytes, split only what's left.
        let st = ChunkedResumeState.planAdopting(prefix: 500, fileSize: 4500, connections: 4, minChunkBytes: 1000)
        XCTAssertTrue(st.isValid(forSize: 4500))
        XCTAssertEqual(st.completedBytes, 500)
        XCTAssertEqual(st.chunks.first, .init(start: 0, end: 499, done: 500))
        XCTAssertEqual(st.chunks.count, 5, "prefix + 4 chunks over the remaining 4000 bytes")
    }

    func testAdoptedPrefixCoveringTheWholeFileIsComplete() {
        let st = ChunkedResumeState.planAdopting(prefix: 4500, fileSize: 4500, connections: 4, minChunkBytes: 1000)
        XCTAssertTrue(st.isValid(forSize: 4500))
        XCTAssertEqual(st.completedBytes, 4500)
    }

    // MARK: - Transport

    func testParallelChunksAssembleTheFileByteForByte() async throws {
        let blob = Self.pseudoRandom(bytes: 1 << 20)
        RangeStubProtocol.configure(blob: blob)
        let partial = tmp("weights.safetensors.partial")

        let d = downloader(partial: partial, size: Int64(blob.count), connections: 8, minChunk: 64 << 10)
        try await d.run()

        XCTAssertEqual(try Data(contentsOf: URL(fileURLWithPath: partial)), blob)
        XCTAssertEqual(RangeStubProtocol.requestedRanges.count, 8, "one request per connection")
        XCTAssertTrue(RangeStubProtocol.requestedRanges.allSatisfy { $0?.hasPrefix("bytes=") == true })
    }

    func testSmallFileTakesTheSingleStreamPath() async throws {
        let blob = Self.pseudoRandom(bytes: 4096)
        RangeStubProtocol.configure(blob: blob)
        let partial = tmp("config.json.partial")

        let d = downloader(partial: partial, size: Int64(blob.count), connections: 8, minChunk: 64 << 10)
        try await d.run()

        XCTAssertEqual(try Data(contentsOf: URL(fileURLWithPath: partial)), blob)
        XCTAssertEqual(RangeStubProtocol.requestedRanges, [nil], "no Range header on a cold single stream")
    }

    func testAShortChunkFailsTheTransferInsteadOfLeavingAHole() async throws {
        // Every chunk closes cleanly after 16 KB of its range. Nothing errors at
        // the socket level, so only the completeness check stands between us and
        // a "finished" file full of holes.
        let blob = Self.pseudoRandom(bytes: 1 << 20)
        let size = Int64(blob.count)
        let partial = tmp("short.partial")
        RangeStubProtocol.configure(blob: blob, cutAfter: 16 << 10, cutIsClean: true)

        do {
            try await downloader(partial: partial, size: size, connections: 4, minChunk: 64 << 10).run()
            XCTFail("a short transfer must not report success")
        } catch {}
    }

    func testInterruptedTransferResumesOnlyTheMissingBytes() async throws {
        let blob = Self.pseudoRandom(bytes: 1 << 20)
        let size = Int64(blob.count)
        let partial = tmp("shard.partial")

        // First pass: every chunk stops 16 KB in.
        RangeStubProtocol.configure(blob: blob, cutAfter: 16 << 10, cutIsClean: true)
        do {
            try await downloader(partial: partial, size: size, connections: 4, minChunk: 64 << 10).run()
            XCTFail("expected the interrupted transfer to throw")
        } catch {}

        let saved = try XCTUnwrap(ChunkedResumeState.load(forPartial: partial, expectedSize: size))
        XCTAssertEqual(saved.chunks.count, 4)
        XCTAssertEqual(saved.completedBytes, 4 * (16 << 10), "each chunk banked what it actually wrote")

        // Second pass: resume. Every request must start where its chunk stopped.
        RangeStubProtocol.configure(blob: blob)
        let resumed = downloader(partial: partial, size: size, connections: 4, minChunk: 64 << 10)
        XCTAssertEqual(resumed.resumableBytesOnDisk, 4 * (16 << 10))
        try await resumed.run()

        XCTAssertEqual(try Data(contentsOf: URL(fileURLWithPath: partial)), blob)
        let starts = RangeStubProtocol.requestedRanges.compactMap { Self.rangeStart($0) }.sorted()
        XCTAssertEqual(starts, saved.chunks.map { $0.start + $0.done }.sorted(),
                       "a resumed chunk must refetch nothing it already banked")
    }

    func testAChunkDyingMidFlightBanksWhatItWrote() async throws {
        // The socket-error flavour: the group fails fast, so siblings get
        // cancelled — but nothing that reached disk may be lost.
        let blob = Self.pseudoRandom(bytes: 1 << 20)
        let size = Int64(blob.count)
        let partial = tmp("dropped.partial")
        RangeStubProtocol.configure(blob: blob, cutAfter: 16 << 10)

        do {
            try await downloader(partial: partial, size: size, connections: 4, minChunk: 64 << 10).run()
            XCTFail("expected the dropped connection to throw")
        } catch {}

        let saved = try XCTUnwrap(ChunkedResumeState.load(forPartial: partial, expectedSize: size))
        XCTAssertGreaterThan(saved.completedBytes, 0)
        for chunk in saved.chunks {
            XCTAssertTrue(chunk.done == 0 || chunk.done == 16 << 10, "banked \(chunk.done)")
        }

        RangeStubProtocol.configure(blob: blob)
        try await downloader(partial: partial, size: size, connections: 4, minChunk: 64 << 10).run()
        XCTAssertEqual(try Data(contentsOf: URL(fileURLWithPath: partial)), blob)
    }

    func testAFullyWrittenPartialIsNotRefetched() async throws {
        // The process died between the last byte and the rename. Re-streaming a
        // finished multi-GB shard is the expensive way to be wrong.
        let blob = Self.pseudoRandom(bytes: 1 << 20)
        let partial = tmp("finished.partial")
        try blob.write(to: URL(fileURLWithPath: partial))
        RangeStubProtocol.configure(blob: blob)

        try await downloader(partial: partial, size: Int64(blob.count), connections: 4, minChunk: 64 << 10).run()

        XCTAssertEqual(RangeStubProtocol.requestedRanges, [], "nothing left to fetch")
        XCTAssertEqual(try Data(contentsOf: URL(fileURLWithPath: partial)), blob)
    }

    func testAPartialFromTheOldSingleStreamPathKeepsItsBytes() async throws {
        // Upgrading mid-download must not throw away what's already on disk.
        let blob = Self.pseudoRandom(bytes: 1 << 20)
        let prefix = 256 << 10
        let partial = tmp("legacy.partial")
        try blob.subdata(in: 0..<prefix).write(to: URL(fileURLWithPath: partial))
        RangeStubProtocol.configure(blob: blob)

        try await downloader(partial: partial, size: Int64(blob.count), connections: 4, minChunk: 64 << 10).run()

        XCTAssertEqual(try Data(contentsOf: URL(fileURLWithPath: partial)), blob)
        let starts = RangeStubProtocol.requestedRanges.compactMap { Self.rangeStart($0) }
        XCTAssertFalse(starts.contains(0), "byte 0 was already ours — refetching it wastes the prefix")
        XCTAssertEqual(starts.min(), Int64(prefix))
    }

    func testFallsBackToOneStreamWhenTheServerIgnoresRange() async throws {
        let blob = Self.pseudoRandom(bytes: 1 << 20)
        RangeStubProtocol.configure(blob: blob, supportsRanges: false)
        let partial = tmp("norange.partial")

        let d = downloader(partial: partial, size: Int64(blob.count), connections: 4, minChunk: 64 << 10)
        try await d.run()

        // The file still lands intact, and the chunk sidecar is gone (the plan
        // it described is void).
        XCTAssertEqual(try Data(contentsOf: URL(fileURLWithPath: partial)), blob)
        XCTAssertFalse(FileManager.default.fileExists(atPath: ChunkedResumeState.sidecarPath(forPartial: partial)))
    }

    func testAContentRangeTotalThatContradictsTheHeadSizeFallsBack() async throws {
        // HF's tree listing and the CDN disagreeing about the size would make
        // every chunk boundary wrong — take the origin's word and stream it.
        let blob = Self.pseudoRandom(bytes: 1 << 20)
        RangeStubProtocol.configure(blob: blob)
        let partial = tmp("badsize.partial")

        let d = downloader(partial: partial, size: Int64(blob.count) + 4096, connections: 4, minChunk: 64 << 10)
        try await d.run()

        XCTAssertEqual(try Data(contentsOf: URL(fileURLWithPath: partial)), blob)
    }

    func testHttpErrorSurfacesTheStatusCode() async throws {
        RangeStubProtocol.configure(blob: Data(), status: 401)
        let partial = tmp("gated.partial")
        do {
            try await downloader(partial: partial, size: 4096, connections: 4, minChunk: 64 << 10).run()
            XCTFail("expected a 401 to throw")
        } catch {
            XCTAssertTrue(error.localizedDescription.contains("401"), "got: \(error.localizedDescription)")
        }
    }

    func testCancellationStopsTheTransfer() async throws {
        let blob = Self.pseudoRandom(bytes: 1 << 20)
        RangeStubProtocol.configure(blob: blob, throttleSlices: true)
        let partial = tmp("cancel.partial")
        let d = downloader(partial: partial, size: Int64(blob.count), connections: 4, minChunk: 64 << 10)

        let task = Task { try await d.run() }
        try await Task.sleep(nanoseconds: 80_000_000)
        task.cancel()
        do {
            try await task.value
            // A very fast machine may legitimately finish first.
        } catch {
            XCTAssertTrue(DownloadManager.isCancellation(error), "got: \(error)")
        }
    }

    // MARK: - Live measurement (opt-in)

    /// A/B against the real HF CDN, off by default because it moves ~650 MB:
    ///
    ///     MLX_SERVE_LIVE_DOWNLOAD=1 swift test --filter testLiveHuggingFaceThroughput
    ///
    /// Both legs run in the same minute over the same file so the comparison
    /// isn't reading line drift as a win.
    func testLiveHuggingFaceThroughput() async throws {
        try XCTSkipUnless(ProcessInfo.processInfo.environment["MLX_SERVE_LIVE_DOWNLOAD"] == "1",
                          "set MLX_SERVE_LIVE_DOWNLOAD=1 to measure against the real CDN")
        let url = URL(string: "https://huggingface.co/ddalcu/Kokoro-82M-MLX-Serve/resolve/main/model.safetensors")!
        let size: Int64 = 324_613_768

        // Alternate the legs rather than running each in a block — a block
        // schedule reads line drift as a win (or a loss).
        var rates: [Int: [Double]] = [:]
        for connections in [8, 1, 8, 1] {
            let partial = tmp("live-\(connections).partial")
            try? FileManager.default.removeItem(atPath: partial)
            ChunkedResumeState.remove(forPartial: partial)

            let downloader = ChunkedFileDownloader(
                url: url, partialPath: partial, fileSize: size,
                session: DownloadSession.shared, headers: DownloadManager.hfHeaders(),
                connections: connections
            )
            let started = Date()
            try await downloader.run()
            let seconds = Date().timeIntervalSince(started)
            let rate = Double(size) / seconds / 1e6
            rates[connections, default: []].append(rate)
            print("[live] \(connections) conn \(String(format: "%.1f", rate)) MB/s")
            fflush(stdout)

            XCTAssertEqual(Self.partialSize(partial), size, "\(connections) conns produced a short file")
            try? FileManager.default.removeItem(atPath: partial)
            ChunkedResumeState.remove(forPartial: partial)
        }
        let best = { (n: Int) in rates[n]!.max()! }
        print("[live] best 1 conn \(String(format: "%.1f", best(1))) MB/s, "
              + "best 8 conns \(String(format: "%.1f", best(8))) MB/s "
              + "(\(String(format: "%.2f", best(8) / best(1)))x)")

        // Integrity at the SHIPPING connection count, against the real CDN:
        // a chunk landing at the wrong offset is silent otherwise.
        let partial = tmp("live-default.partial")
        try? FileManager.default.removeItem(atPath: partial)
        ChunkedResumeState.remove(forPartial: partial)
        let shipped = ChunkedFileDownloader(
            url: url, partialPath: partial, fileSize: size,
            session: DownloadSession.shared, headers: DownloadManager.hfHeaders(),
            connections: DownloadChunking.configuredConnections()
        )
        try await shipped.run()
        let (reference, _) = try await URLSession.shared.data(from: url)
        XCTAssertEqual(try Data(contentsOf: URL(fileURLWithPath: partial)), reference,
                       "\(DownloadChunking.configuredConnections()) chunks did not reassemble the CDN's bytes")
        try? FileManager.default.removeItem(atPath: partial)
        ChunkedResumeState.remove(forPartial: partial)
    }

    /// The other half of the fix: a repo's config/tokenizer/index files used to
    /// get a FRESH `URLSession` each, re-paying TCP + TLS + the `resolve/main`
    /// 302 every time. Same opt-in env var.
    func testLiveSharedSessionHandshakeCost() async throws {
        try XCTSkipUnless(ProcessInfo.processInfo.environment["MLX_SERVE_LIVE_DOWNLOAD"] == "1",
                          "set MLX_SERVE_LIVE_DOWNLOAD=1 to measure against the real CDN")
        let repo = "mlx-community/Qwen3-8B-4bit"
        let files = ["config.json", "tokenizer_config.json", "special_tokens_map.json", "added_tokens.json",
                     "model.safetensors.index.json", "vocab.json", "merges.txt", "README.md",
                     ".gitattributes", "tokenizer.json"]

        func fetch(sharedSession: Bool) async throws -> TimeInterval {
            let started = Date()
            for name in files {
                let url = URL(string: "https://huggingface.co/\(repo)/resolve/main/\(name)")!
                let session: URLSession
                if sharedSession {
                    session = DownloadSession.shared
                } else {
                    // What every file used to get.
                    let config = URLSessionConfiguration.default
                    config.timeoutIntervalForRequest = 60
                    session = URLSession(configuration: config)
                }
                _ = try await session.data(for: URLRequest(url: url))
                if !sharedSession { session.finishTasksAndInvalidate() }
            }
            return Date().timeIntervalSince(started)
        }

        _ = try await fetch(sharedSession: true)          // warm the route
        var shared: [TimeInterval] = []
        var perFile: [TimeInterval] = []
        for _ in 0..<3 {
            perFile.append(try await fetch(sharedSession: false))
            shared.append(try await fetch(sharedSession: true))
        }
        print(String(format: "[live] %d files — session per file %.2fs, shared session %.2fs (%.2fx)",
                     files.count, perFile.min()!, shared.min()!, perFile.min()! / shared.min()!))
        fflush(stdout)
    }

    private static func partialSize(_ path: String) -> Int64 {
        (try? FileManager.default.attributesOfItem(atPath: path)[.size] as? Int64) as? Int64 ?? 0
    }

    // MARK: - Hugging Face auth

    func testTokenComesFromTheEnvironmentFirst() {
        let token = DownloadManager.hfToken(environment: ["HF_TOKEN": "hf_env"], home: tempRoot)
        XCTAssertEqual(token, "hf_env")
    }

    func testTokenFallsBackToTheCliLoginFile() throws {
        // A Finder-launched app has NO shell environment, so the file
        // `huggingface-cli login` writes is the one that actually works.
        let dir = ((tempRoot as NSString).appendingPathComponent(".cache") as NSString)
            .appendingPathComponent("huggingface")
        try FileManager.default.createDirectory(atPath: dir, withIntermediateDirectories: true)
        try "hf_from_file\n".write(toFile: (dir as NSString).appendingPathComponent("token"),
                                   atomically: true, encoding: .utf8)

        XCTAssertEqual(DownloadManager.hfToken(environment: [:], home: tempRoot), "hf_from_file")
    }

    func testNoTokenYieldsNoAuthorizationHeader() {
        XCTAssertNil(DownloadManager.hfToken(environment: ["HF_TOKEN": "   "], home: tempRoot))
        XCTAssertTrue(DownloadManager.hfHeaders(environment: [:], home: tempRoot).isEmpty)
        XCTAssertEqual(DownloadManager.hfHeaders(environment: ["HF_TOKEN": "abc"], home: tempRoot),
                       ["Authorization": "Bearer abc"])
    }

    // MARK: - Helpers

    private func tmp(_ name: String) -> String {
        (tempRoot as NSString).appendingPathComponent(name)
    }

    private func downloader(partial: String, size: Int64, connections: Int, minChunk: Int64) -> ChunkedFileDownloader {
        ChunkedFileDownloader(
            url: URL(string: "https://stub.test/model/weights")!,
            partialPath: partial,
            fileSize: size,
            session: Self.stubSession(),
            headers: [:],
            connections: connections,
            minChunkBytes: minChunk
        )
    }

    private static func stubSession() -> URLSession {
        let c = URLSessionConfiguration.ephemeral
        c.protocolClasses = [RangeStubProtocol.self]
        c.httpMaximumConnectionsPerHost = 16
        return URLSession(configuration: c)
    }

    /// Deterministic pseudo-random bytes — a reproducible failure beats a
    /// pretty one, and a constant fill would hide an offset bug.
    private static func pseudoRandom(bytes count: Int) -> Data {
        var out = Data(count: count)
        var seed: UInt64 = 0x9E3779B97F4A7C15
        out.withUnsafeMutableBytes { raw in
            let p = raw.bindMemory(to: UInt8.self)
            for i in 0..<count {
                seed = seed &* 6364136223846793005 &+ 1442695040888963407
                p[i] = UInt8truncating(seed >> 33)
            }
        }
        return out
    }

    private static func UInt8truncating(_ v: UInt64) -> UInt8 { UInt8(v & 0xFF) }

    private static func rangeStart(_ header: String?) -> Int64? {
        guard let h = header, h.hasPrefix("bytes=") else { return nil }
        return Int64(h.dropFirst("bytes=".count).split(separator: "-").first ?? "")
    }
}

// MARK: - Range-speaking stub origin

/// A `URLProtocol` that serves a fixed blob with real `Range` semantics, so the
/// downloader is tested against the protocol it actually uses rather than a
/// mock of our own code.
final class RangeStubProtocol: URLProtocol {
    private static let lock = NSLock()
    private static var blob = Data()
    private static var supportsRanges = true
    private static var status = 200
    private static var cutAfter: Int?
    private static var cutIsClean = false
    private static var throttleSlices = false
    private static var requested: [String?] = []

    /// `cutAfter` truncates every response body after N bytes; `cutIsClean`
    /// picks whether that looks like a dropped socket or a well-formed short
    /// body (the case only the completeness check can catch).
    static func configure(blob: Data, supportsRanges: Bool = true, status: Int = 200,
                          cutAfter: Int? = nil, cutIsClean: Bool = false, throttleSlices: Bool = false) {
        lock.lock(); defer { lock.unlock() }
        self.blob = blob
        self.supportsRanges = supportsRanges
        self.status = status
        self.cutAfter = cutAfter
        self.cutIsClean = cutIsClean
        self.throttleSlices = throttleSlices
        self.requested = []
    }

    static func reset() { configure(blob: Data()) }

    static var requestedRanges: [String?] {
        lock.lock(); defer { lock.unlock() }
        return requested
    }

    private static func snapshot() -> (Data, Bool, Int, Int?, Bool, Bool) {
        lock.lock(); defer { lock.unlock() }
        return (blob, supportsRanges, status, cutAfter, cutIsClean, throttleSlices)
    }

    private static func record(_ range: String?) {
        lock.lock(); defer { lock.unlock() }
        requested.append(range)
    }

    override class func canInit(with request: URLRequest) -> Bool { request.url?.host == "stub.test" }
    override class func canonicalRequest(for request: URLRequest) -> URLRequest { request }

    override func startLoading() {
        let header = request.value(forHTTPHeaderField: "Range")
        Self.record(header)
        let (blob, ranges, status, cutAfter, cutIsClean, throttle) = Self.snapshot()

        guard status == 200 else {
            let resp = HTTPURLResponse(url: request.url!, statusCode: status, httpVersion: "HTTP/1.1", headerFields: [:])!
            client?.urlProtocol(self, didReceive: resp, cacheStoragePolicy: .notAllowed)
            client?.urlProtocolDidFinishLoading(self)
            return
        }

        var code = 200
        var body = blob
        var headers: [String: String] = [:]
        if ranges, let h = header, let (s, e) = Self.parseRange(h, size: blob.count) {
            code = 206
            body = blob.subdata(in: s..<(e + 1))
            headers["Content-Range"] = "bytes \(s)-\(e)/\(blob.count)"
        }
        headers["Content-Length"] = "\(body.count)"
        let resp = HTTPURLResponse(url: request.url!, statusCode: code, httpVersion: "HTTP/1.1", headerFields: headers)!
        client?.urlProtocol(self, didReceive: resp, cacheStoragePolicy: .notAllowed)

        if let cut = cutAfter, body.count > cut {
            client?.urlProtocol(self, didLoad: body.subdata(in: 0..<cut))
            if cutIsClean {
                client?.urlProtocolDidFinishLoading(self)
            } else {
                // Let the loader hand the bytes to the delegate before the
                // socket dies — a real drop doesn't un-deliver what arrived.
                Thread.sleep(forTimeInterval: 0.05)
                client?.urlProtocol(self, didFailWithError: URLError(.networkConnectionLost))
            }
            return
        }

        var i = 0
        let step = 32 << 10
        while i < body.count {
            if stopped { return }
            let j = min(i + step, body.count)
            client?.urlProtocol(self, didLoad: body.subdata(in: i..<j))
            i = j
            if throttle { Thread.sleep(forTimeInterval: 0.01) }
        }
        client?.urlProtocolDidFinishLoading(self)
    }

    private var stopped = false
    override func stopLoading() { stopped = true }

    /// `bytes=S-E` / `bytes=S-`, clamped to the blob.
    private static func parseRange(_ header: String, size: Int) -> (Int, Int)? {
        guard header.hasPrefix("bytes="), size > 0 else { return nil }
        let spec = header.dropFirst("bytes=".count)
        let parts = spec.split(separator: "-", omittingEmptySubsequences: false)
        guard let start = Int(parts.first ?? ""), start < size else { return nil }
        let end = parts.count > 1 ? (Int(parts[1]) ?? size - 1) : size - 1
        return (start, min(end, size - 1))
    }
}
