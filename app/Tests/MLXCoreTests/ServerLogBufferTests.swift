import XCTest
import Combine
@testable import MLXCore

/// Unit tests for the pure buffer-trim helper that backs `ServerManager`'s
/// throttled log pipeline. Behavior we care about:
///   - keep at most N characters of tail content,
///   - never grow beyond N regardless of how many chunks we ingest,
///   - leave shorter buffers untouched,
///   - empty input is a no-op,
///   - cap of 0 clears (degenerate but well-defined).
///
/// Throttling (debounce timer, main-actor flush) is exercised by the full
/// app in practice; the pure helper here is the testable surface.
final class ServerLogBufferTests: XCTestCase {

    func testShortBufferIsUnchanged() {
        var buf = "hello world"
        ServerManager.trimLogTail(&buf, toAtMost: 65_536)
        XCTAssertEqual(buf, "hello world")
    }

    func testTrimsToTailWhenOverCap() {
        var buf = String(repeating: "x", count: 100) + "TAIL"
        ServerManager.trimLogTail(&buf, toAtMost: 4)
        XCTAssertEqual(buf, "TAIL")
    }

    func testExactlyAtCapNoTrim() {
        var buf = "abcd"
        ServerManager.trimLogTail(&buf, toAtMost: 4)
        XCTAssertEqual(buf, "abcd")
    }

    func testEmptyBufferStaysEmpty() {
        var buf = ""
        ServerManager.trimLogTail(&buf, toAtMost: 10)
        XCTAssertEqual(buf, "")
    }

    func testZeroCapClearsBuffer() {
        var buf = "anything"
        ServerManager.trimLogTail(&buf, toAtMost: 0)
        XCTAssertEqual(buf, "")
    }

    func testRepeatedAppendsStayBounded() {
        // Simulates the readability handler: many small chunks arriving over
        // time. The buffer must never grow past the cap.
        var buf = ""
        let cap = 1024
        for i in 0..<10_000 {
            buf.append("chunk-\(i)\n")
            ServerManager.trimLogTail(&buf, toAtMost: cap)
            XCTAssertLessThanOrEqual(buf.count, cap)
        }
        // Final tail must contain the most recent chunk in full.
        XCTAssertTrue(buf.hasSuffix("chunk-9999\n"),
                      "Latest chunk must survive trimming; got tail: \(buf.suffix(40))")
    }

    // MARK: - ThrottledLogBuffer

    func testThrottledBufferAppendAndSnapshot() {
        let buf = ThrottledLogBuffer(maxBytes: 100)
        buf.append("hello ")
        buf.append("world")
        XCTAssertEqual(buf.snapshot(), "hello world")
    }

    func testThrottledBufferRespectsMaxBytes() {
        let buf = ThrottledLogBuffer(maxBytes: 5)
        buf.append("12345")
        buf.append("6789")
        // Last 5 characters of "123456789" = "56789"
        XCTAssertEqual(buf.snapshot(), "56789")
    }

    func testThrottledBufferClear() {
        let buf = ThrottledLogBuffer(maxBytes: 100)
        buf.append("dirty")
        buf.clear()
        XCTAssertEqual(buf.snapshot(), "")
        // Still appendable after clear.
        buf.append("fresh")
        XCTAssertEqual(buf.snapshot(), "fresh")
    }

    func testThrottledBufferConcurrentAppendsAreSafe() {
        // Hammer the lock from multiple queues — the stderr `readabilityHandler`
        // doesn't promise serial delivery across spawn/restart boundaries, and
        // future call sites might fan out further. If the lock is correct this
        // never deadlocks or trips a TSan / sanitizer condition, and the final
        // size stays bounded.
        let buf = ThrottledLogBuffer(maxBytes: 8_192)
        let group = DispatchGroup()
        let workers = 8
        let perWorker = 1_000
        for w in 0..<workers {
            DispatchQueue.global().async(group: group) {
                for i in 0..<perWorker {
                    buf.append("w\(w)-\(i)\n")
                }
            }
        }
        group.wait()
        XCTAssertLessThanOrEqual(buf.snapshot().count, 8_192)
    }

    // MARK: - LogPoller
    //
    // LogPoller is the pull-based bridge from `ThrottledLogBuffer` (the
    // off-main, lock-guarded source of truth) to the SwiftUI views that
    // render the log. Crucially: nothing in `ServerManager` is `@Published`
    // for the log, so ChatView/Settings/etc. never re-render when stderr
    // arrives. Only the view that *holds* the poller does, and only on
    // its own tick. These tests pin that contract.

    @MainActor
    func testPollerStartsEmpty() {
        let poller = LogPoller(interval: 0.5, snapshot: { "" })
        XCTAssertEqual(poller.text, "")
    }

    @MainActor
    func testPollerRefreshPullsFromSnapshot() {
        var source = "first"
        let poller = LogPoller(interval: 0.5, snapshot: { source })
        poller.refresh()
        XCTAssertEqual(poller.text, "first")
        source = "second"
        poller.refresh()
        XCTAssertEqual(poller.text, "second")
    }

    @MainActor
    func testPollerSkipsAssignmentWhenUnchanged() {
        // Skipping the assignment when nothing changed avoids a wasted
        // @Published cycle, which would re-render the log window for no
        // visible reason.
        let poller = LogPoller(interval: 0.5, snapshot: { "stable" })
        poller.refresh()
        var publishCount = 0
        let cancel = poller.$text.sink { _ in publishCount += 1 }
        defer { cancel.cancel() }
        // Re-publishing the initial state happens once on subscribe; we
        // want NO additional publishes on no-op refreshes.
        let baseline = publishCount
        poller.refresh()
        poller.refresh()
        poller.refresh()
        XCTAssertEqual(publishCount, baseline,
                       "Identical snapshots must not retrigger @Published")
    }

    @MainActor
    func testPollerStopsTicking() {
        var calls = 0
        let poller = LogPoller(interval: 0.05) {
            calls += 1
            return "snap-\(calls)"
        }
        poller.start()
        // Let a few ticks fire.
        let exp = expectation(description: "ticks")
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.18) {
            poller.stop()
            exp.fulfill()
        }
        wait(for: [exp], timeout: 1.0)
        let callsAtStop = calls
        // No more ticks after stop().
        let exp2 = expectation(description: "no-more-ticks")
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.2) {
            exp2.fulfill()
        }
        wait(for: [exp2], timeout: 1.0)
        XCTAssertEqual(calls, callsAtStop,
                       "Stopped poller must not invoke its snapshot closure")
    }
}
