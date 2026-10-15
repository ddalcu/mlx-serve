import XCTest
@testable import MLXCore

/// `GET /v1/load-progress` as the app reads and words it (`LoadProgress`).
final class LoadProgressTests: XCTestCase {
    private let body = """
    {"loading":[
      {"id":"Qwen3.8-27B","path":"/m/Qwen3.8-27B","progress":{"phase":"reading_weights","done":5000000000,"total":20000000000,"compiling":false,"elapsed_s":3,"eta_s":9}},
      {"id":"Qwen3.5-0.8B","path":"/m/Qwen3.5-0.8B","progress":{"phase":"ane_programs","done":9,"total":28,"compiling":true,"elapsed_s":40,"eta_s":130}}
    ]}
    """

    func testParsesEveryLoadingModelByPath() throws {
        let all = LoadProgress.parse(Data(body.utf8))
        XCTAssertEqual(all.count, 2)
        let gpu = try XCTUnwrap(all["/m/Qwen3.8-27B"])
        XCTAssertEqual(gpu.phase, .readingWeights)
        XCTAssertEqual(gpu.fraction ?? -1, 0.25, accuracy: 1e-9)
        XCTAssertEqual(gpu.eta, 9)
        let ane = try XCTUnwrap(all["/m/Qwen3.5-0.8B"])
        XCTAssertTrue(ane.compiling)
        XCTAssertEqual(ane.done, 9)
        XCTAssertTrue(LoadProgress.parse(Data(#"{"loading":[]}"#.utf8)).isEmpty)
        XCTAssertTrue(LoadProgress.parse(Data("not json".utf8)).isEmpty)
    }

    func testOnlyCountedPhasesHaveAFraction() {
        var p = LoadProgress(phase: .warmingUp, done: 0, total: 0, compiling: false, eta: nil)
        XCTAssertNil(p.fraction)
        p = LoadProgress(phase: .starting, done: 0, total: 0, compiling: false, eta: nil)
        XCTAssertNil(p.fraction)
        p = LoadProgress(phase: .anePrograms, done: 30, total: 28, compiling: false, eta: nil)
        XCTAssertEqual(p.fraction, 1)
    }

    func testTheLabelSaysWhatIsHappeningHowFarAndHowLong() {
        XCTAssertEqual(LoadProgress(phase: .readingWeights, done: 5_000_000_000, total: 20_000_000_000, compiling: false, eta: 9).label,
                       "Reading weights · 5.0 of 20.0 GB · about 9 s left")
        XCTAssertEqual(LoadProgress(phase: .anePrograms, done: 9, total: 28, compiling: true, eta: 130).label,
                       "Compiling for the Neural Engine · 9 of 28 · about 3 min left")
        XCTAssertEqual(LoadProgress(phase: .anePrograms, done: 9, total: 28, compiling: false, eta: nil).label,
                       "Loading from the Neural Engine cache · 9 of 28")
        XCTAssertEqual(LoadProgress(phase: .warmingUp, done: 0, total: 0, compiling: false, eta: nil).label, "Warming up")
        XCTAssertEqual(LoadProgress(phase: .starting, done: 0, total: 0, compiling: false, eta: nil).label, "Starting")
    }
}

final class LoadProgressLookupTests: XCTestCase {
    private let one = LoadProgress(phase: .anePrograms, done: 1, total: 4, compiling: false, eta: nil)
    private let two = LoadProgress(phase: .readingWeights, done: 1, total: 4, compiling: false, eta: nil)

    /// The registry may spell a path differently (a symlink resolved, a trailing slash): with one
    /// load running, it is the load being waited on.
    func testFindsTheLoadByPathOrTheOnlyOneRunning() {
        XCTAssertEqual(LoadProgress.lookup(["/a": one, "/b": two], path: "/b"), two)
        XCTAssertEqual(LoadProgress.lookup(["/real/a": one], path: "/link/a"), one)
        XCTAssertNil(LoadProgress.lookup(["/a": one, "/b": two], path: "/c"))
        XCTAssertNil(LoadProgress.lookup([:], path: "/a"))
    }
}

final class LoadReadyNotificationTests: XCTestCase {
    /// Only a long load, finished while the app is in the background, earns a notification.
    func testOnlyALongLoadInTheBackgroundNotifies() {
        XCTAssertTrue(LoadProgress.notifiesReady(after: 40, appActive: false))
        XCTAssertFalse(LoadProgress.notifiesReady(after: 40, appActive: true))
        XCTAssertFalse(LoadProgress.notifiesReady(after: 5, appActive: false))
    }
}

final class LoadProgressPollingTests: XCTestCase {
    /// Every load shows, any client's: a slow look while idle, fast while anything loads.
    func testPollsFastWhileLoadingAndSlowlyOtherwise() {
        XCTAssertEqual(ServerManager.loadProgressInterval(loading: true), .milliseconds(400))
        XCTAssertEqual(ServerManager.loadProgressInterval(loading: false), .seconds(2))
    }
}

final class ModelListRefreshTests: XCTestCase {
    /// The tray re-reads `/v1/models` while open only when the server says it would read
    /// differently (`/props` `registry_generation`); the first reading only records it.
    func testOnlyAMovedGenerationRereadsTheModelList() {
        XCTAssertTrue(ServerManager.modelListMoved(seen: 5, reported: 6))
        XCTAssertFalse(ServerManager.modelListMoved(seen: 5, reported: 5))
        XCTAssertFalse(ServerManager.modelListMoved(seen: nil, reported: 5))
        XCTAssertFalse(ServerManager.modelListMoved(seen: 5, reported: nil))
    }
}
