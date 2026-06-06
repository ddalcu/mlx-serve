import XCTest
@testable import MLXCore

/// Pins the streamed-token coalescer that fixes the tray-popover freeze.
///
/// Regression context: the chat/agent engine used to apply every streamed token
/// straight into `AppState.chatSessions`, firing `AppState.objectWillChange`
/// once per token. While the `MenuBarExtra(.window)` tray popover is open during
/// the assistant's answer (the voice-mode case), that per-token churn re-renders
/// the popover dozens-to-hundreds of times a second and saturates the main run
/// loop — starving SwiftUI `Button` hit-testing exactly like a continuous
/// animation does, so the voice **Stop** button goes dead while the AppKit
/// `Menu`/`Picker` keep working. Coalescing the deltas to a bounded cadence
/// restores idle run-loop time between flushes so the buttons respond again.
///
/// The clock is injected via `now`, so the batching is fully deterministic with
/// no real time involved.
final class StreamCoalescerTests: XCTestCase {

    func testFirstDeltaFlushesImmediately() {
        var c = StreamCoalescer(interval: 0.05)
        // Text must start appearing right away — the throttle only batches the
        // *subsequent* deltas, it doesn't delay the first one.
        XCTAssertEqual(c.add(content: "H", now: 0.0)?.content, "H")
    }

    func testDeltasWithinIntervalAccumulate() {
        var c = StreamCoalescer(interval: 0.05)
        _ = c.add(content: "H", now: 0.0)
        XCTAssertNil(c.add(content: "e", now: 0.01))
        XCTAssertNil(c.add(content: "l", now: 0.02))
        XCTAssertNil(c.add(content: "l", now: 0.04))
    }

    func testBatchedFlushAfterInterval() {
        var c = StreamCoalescer(interval: 0.05)
        _ = c.add(content: "H", now: 0.0)
        _ = c.add(content: "e", now: 0.01)
        _ = c.add(content: "l", now: 0.02)
        _ = c.add(content: "l", now: 0.04)
        // 0.06 - 0.0 >= 0.05 → flush everything buffered since the first flush.
        XCTAssertEqual(c.add(content: "o", now: 0.06)?.content, "ello")
    }

    func testDrainReturnsRemainderThenNil() {
        var c = StreamCoalescer(interval: 0.05)
        _ = c.add(content: "H", now: 0.0)
        XCTAssertNil(c.add(content: "i", now: 0.01))
        XCTAssertEqual(c.drain()?.content, "i")
        XCTAssertNil(c.drain())
    }

    func testDrainOnEmptyIsNil() {
        var c = StreamCoalescer(interval: 0.05)
        XCTAssertNil(c.drain())
    }

    func testReasoningCoalescesIndependently() {
        var c = StreamCoalescer(interval: 0.05)
        XCTAssertEqual(c.add(reasoning: "x", now: 0.0)?.reasoning, "x")
        XCTAssertNil(c.add(reasoning: "y", now: 0.01))
        XCTAssertEqual(c.drain()?.reasoning, "y")
    }

    func testContentAndReasoningBatchTogether() {
        var c = StreamCoalescer(interval: 0.05)
        _ = c.add(content: "a", reasoning: "1", now: 0.0)   // first flush
        XCTAssertNil(c.add(content: "b", reasoning: "2", now: 0.01))
        let batch = c.drain()
        XCTAssertEqual(batch?.content, "b")
        XCTAssertEqual(batch?.reasoning, "2")
    }

    /// The load-bearing invariant: a long, fast answer must NOT produce one UI
    /// update per token. 100 tokens arriving every 10 ms across ~1 s, throttled
    /// at 50 ms, should collapse to roughly 20 flushes — never one per token.
    func testFastBurstCollapsesToBoundedFlushCount() {
        var c = StreamCoalescer(interval: 0.05)
        var flushes = 0
        var reassembled = ""
        for i in 0..<100 {
            if let batch = c.add(content: "x", now: Double(i) * 0.01) {
                flushes += 1
                reassembled += batch.content
            }
        }
        if let tail = c.drain() { flushes += 1; reassembled += tail.content }

        XCTAssertLessThanOrEqual(flushes, 25, "per-token UI churn would be ~100 flushes")
        XCTAssertGreaterThan(flushes, 1, "must still stream incrementally, not buffer the whole answer")
        // No token is ever dropped or duplicated — the batched text reassembles
        // to exactly the original stream.
        XCTAssertEqual(reassembled, String(repeating: "x", count: 100))
    }
}
