import XCTest
@testable import MLXCore

/// The off-main library scan behind `AppState.refreshModels()`.
final class ModelLibraryRefresherTests: XCTestCase {
    private var tempRoot: String!

    override func setUpWithError() throws {
        tempRoot = (NSTemporaryDirectory() as NSString)
            .appendingPathComponent("mlx-serve-refresher-\(UUID().uuidString)")
        try FileManager.default.createDirectory(atPath: tempRoot, withIntermediateDirectories: true)
    }

    override func tearDownWithError() throws {
        try? FileManager.default.removeItem(atPath: tempRoot)
    }

    private var emptyInputs: DownloadManager.LocalScanInputs {
        DownloadManager.LocalScanInputs(
            ownedRoots: [tempRoot], lmStudioRoot: nil, huggingFaceRoot: nil,
            customRoot: nil, toolRoots: [], inFlightDirs: []
        )
    }

    private func makeFakeModel(at path: String) throws {
        try FileManager.default.createDirectory(atPath: path, withIntermediateDirectories: true)
        try "{}".write(toFile: (path as NSString).appendingPathComponent("config.json"),
                       atomically: true, encoding: .utf8)
    }

    // MARK: - The awaited first scan

    /// The launch plan reads the library the moment this returns, so `scan`
    /// must hand back the walk's result rather than an empty list: awaiting a
    /// scan that published later would be the same race as not awaiting at all.
    @MainActor
    func testAwaitingAScanReturnsTheWalkResult() async throws {
        let refresher = ModelLibraryRefresher()
        try makeFakeModel(at: (tempRoot as NSString).appendingPathComponent("gemma-4-e4b-it-4bit"))

        let models = await refresher.scan(inputs: emptyInputs)

        XCTAssertEqual(models.count, 1)
        XCTAssertTrue(models.first?.path.hasSuffix("gemma-4-e4b-it-4bit") ?? false)
    }

    // MARK: - Ordering

    @MainActor
    func testTheScanRunsOffTheMainThreadAndAppliesOnIt() async {
        let refresher = ModelLibraryRefresher()
        let scannedOnMain = ThreadBox()
        let appliedOnMain = ThreadBox()

        refresher.refresh(inputs: emptyInputs) { _ in
            scannedOnMain.value = Thread.isMainThread
            return []
        } apply: { _ in
            appliedOnMain.value = Thread.isMainThread
        }

        let landed = await waitUntil { appliedOnMain.value }
        XCTAssertTrue(landed)
        XCTAssertEqual(scannedOnMain.value, false, "the walk must not run on the main thread")
        XCTAssertEqual(appliedOnMain.value, true, "the result lands on the main actor")
    }

    @MainActor
    func testOnlyTheNewestScanIsApplied() async {
        let refresher = ModelLibraryRefresher()
        let releaseFirst = DispatchSemaphore(value: 0)
        var applied: [String] = []

        // A slow first scan, then a fast second one while the first is still in
        // the walk: the second is the state, and the first must not overwrite it.
        refresher.refresh(inputs: emptyInputs) { _ in
            _ = releaseFirst.wait(timeout: .now() + 5)
            return [fakeModel("stale")]
        } apply: { models in
            applied = models.map(\.id)
        }
        refresher.refresh(inputs: emptyInputs) { _ in [fakeModel("latest")] } apply: { models in
            applied = models.map(\.id)
        }

        let secondLanded = await waitUntil { !applied.isEmpty }
        XCTAssertTrue(secondLanded)
        XCTAssertEqual(applied, ["latest"])

        // Let the superseded walk finish; its result must still be dropped.
        releaseFirst.signal()
        _ = await waitUntil(timeout: 0.5) { applied != ["latest"] }
        XCTAssertEqual(applied, ["latest"], "a superseded scan must never be applied")
    }

    // MARK: - The captured inputs are what the scan reads

    func testTheStaticScanUsesOnlyItsCapturedInputs() throws {
        try makeFakeModel(at: (tempRoot as NSString).appendingPathComponent("acme/demo"))

        let inputs = DownloadManager.LocalScanInputs(
            ownedRoots: [tempRoot], lmStudioRoot: nil, huggingFaceRoot: nil,
            customRoot: nil, toolRoots: [], inFlightDirs: []
        )
        let scanned = DownloadManager.discoverLocalModels(inputs)
        XCTAssertEqual(scanned.map(\.id), ["mlxServe:acme/demo"])

        // Nothing captured: no root is read, so an empty library is the answer
        // even though the same directory is sitting on disk.
        let unscanned = DownloadManager.discoverLocalModels(
            DownloadManager.LocalScanInputs(
                ownedRoots: [], lmStudioRoot: nil, huggingFaceRoot: nil,
                customRoot: nil, toolRoots: [], inFlightDirs: []
            )
        )
        XCTAssertTrue(unscanned.isEmpty)
    }
}

/// A `Bool` a background scan can hand back to the test, with the lock the
/// concurrency checker requires.
private final class ThreadBox: @unchecked Sendable {
    private let lock = NSLock()
    private var stored = false

    var value: Bool {
        get { lock.withLock { stored } }
        set { lock.withLock { stored = newValue } }
    }
}

/// A minimal model for the ordering test, built off the main actor.
private func fakeModel(_ id: String) -> LocalModel {
    LocalModel(id: id, name: id, path: "/tmp/\(id)", sizeFormatted: "", modelType: "qwen3",
               source: .custom, kind: .base)
}

/// Polls until `condition` holds. The refresher has no "wait for it" API by
/// design — production callers must not wait.
@MainActor
private func waitUntil(timeout: TimeInterval = 5, _ condition: @MainActor () -> Bool) async -> Bool {
    let deadline = Date().addingTimeInterval(timeout)
    while Date() < deadline {
        if condition() { return true }
        try? await Task.sleep(nanoseconds: 10_000_000)
    }
    return condition()
}
