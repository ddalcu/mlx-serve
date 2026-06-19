import XCTest
import Combine
import AppKit
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

    // MARK: - Launch command echo (top of the server log)
    //
    // The log window now prints the exact invocation above the server's own
    // first line so the user can see and reproduce how mlx-serve was launched.

    func testLaunchCommandLineJoinsBinaryAndArgs() {
        let cmd = ServerManager.launchCommandLine(
            binaryPath: "/Applications/MLX Core.app/Contents/MacOS/mlx-serve",
            args: ["--model", "/Volumes/Models/Qwen", "--serve", "--port", "11234"])
        // Binary path has a space → quoted; the whole invocation is present.
        XCTAssertTrue(cmd.contains("\"/Applications/MLX Core.app/Contents/MacOS/mlx-serve\""))
        XCTAssertTrue(cmd.contains("--model"))
        XCTAssertTrue(cmd.contains("--port 11234"))
    }

    func testLaunchCommandLineQuotesArgsWithSpaces() {
        let cmd = ServerManager.launchCommandLine(
            binaryPath: "/usr/bin/mlx-serve",
            args: ["--model", "/Volumes/My Drive/model dir"])
        XCTAssertTrue(cmd.contains("\"/Volumes/My Drive/model dir\""),
                      "a path with spaces must be quoted so the line is copy-pasteable")
        XCTAssertFalse(cmd.contains("/usr/bin/mlx-serve\""), "a space-free binary path stays unquoted")
    }

    /// Every launch knob — including --skip-mem-preflight — is a CLI flag now,
    /// so it rides through `args` into the echoed command (no special-casing).
    func testLaunchCommandLineEchoesSkipPreflightFlag() {
        let cmd = ServerManager.launchCommandLine(
            binaryPath: "/usr/bin/mlx-serve",
            args: ["--serve", "--skip-mem-preflight"])
        XCTAssertTrue(cmd.contains("--skip-mem-preflight"))
        XCTAssertFalse(cmd.contains("MLX_SERVE_SKIP_MEM_PREFLIGHT"),
                       "the env var is gone — it must not reappear in the echoed command")
    }

    // MARK: - Crash summary (menu-bar error text)
    //
    // The menu bar used to show `String(stderr.suffix(200))`, which on a crash
    // lands mid-native-backtrace ("…wqthread + 8") and hides the real cause.
    // `summarizeCrash` extracts the meaningful line instead.

    func testSummarizeCrashSurfacesGpuOOM() {
        let log = """
        MoE routing compiled (kernel fusion enabled)
        GDN gate compiled (kernel fusion enabled)
        25  libsystem_pthread.dylib  0x0000  start_wqthread + 8
        libc++abi: terminating due to uncaught exception of type std::runtime_error: [METAL] Command buffer execution failed: Insufficient Memory (00000008:kIOGPUCommandBufferCallbackErrorOutOfMemory)
        """
        let msg = ServerManager.summarizeCrash(log, exitCode: 134)
        XCTAssertTrue(msg.localizedCaseInsensitiveContains("GPU memory"), "got: \(msg)")
        XCTAssertFalse(msg.contains("wqthread"), "must not surface a backtrace frame: \(msg)")
    }

    func testSummarizeCrashSurfacesPreflightRefusal() {
        let log = "Loading...\nInsufficient memory to load model: weights ~41.8 GB but only 12.0 GB free. Close other models/apps..."
        let msg = ServerManager.summarizeCrash(log, exitCode: 1)
        XCTAssertTrue(msg.hasPrefix("Insufficient memory to load model"), "got: \(msg)")
    }

    func testSummarizeCrashStripsCppPreambleForGenericFatal() {
        let log = "boot\nlibc++abi: terminating due to uncaught exception of type std::runtime_error: something specific went wrong"
        let msg = ServerManager.summarizeCrash(log, exitCode: 134)
        XCTAssertEqual(msg, "something specific went wrong")
    }

    func testSummarizeCrashMissingWeight() {
        let log = "Loading model.safetensors...\nMISSING WEIGHT: model.layers.0.foo.weight"
        let msg = ServerManager.summarizeCrash(log, exitCode: 1)
        XCTAssertTrue(msg.contains("MISSING WEIGHT"), "got: \(msg)")
    }

    func testSummarizeCrashEmptyFallsBackToExitCode() {
        XCTAssertEqual(ServerManager.summarizeCrash("   \n  ", exitCode: 9), "exit code 9")
    }

    // MARK: - Crash log view (wrapping, not horizontal scroll)
    //
    // The crash alert used to configure its log text view for horizontal
    // scrolling (`isHorizontallyResizable = true`, infinite container width),
    // so long backtrace / exception lines ran off the right edge and the
    // tail — where the real cause lives ("…execution failed: Insufficient
    // Memory") — got cut off even when scrolling. The fix wraps to the
    // visible width. These tests pin the wrapping configuration.

    @MainActor
    func testCrashLogViewWrapsInsteadOfScrollingHorizontally() {
        let scroll = ServerManager.makeCrashLogScrollView(log: "hello", width: 640, height: 280)
        XCTAssertFalse(scroll.hasHorizontalScroller,
                       "Crash log must not offer a horizontal scroller")
        XCTAssertTrue(scroll.hasVerticalScroller)
        guard let textView = scroll.documentView as? NSTextView else {
            return XCTFail("documentView must be an NSTextView")
        }
        XCTAssertFalse(textView.isHorizontallyResizable,
                       "A horizontally-resizable text view scrolls instead of wrapping")
        XCTAssertEqual(textView.textContainer?.widthTracksTextView, true,
                       "Container width must track the text view so lines wrap to it")
        // Container width must be bounded to the visible width, not infinite.
        if let w = textView.textContainer?.containerSize.width {
            XCTAssertLessThan(w, CGFloat.greatestFiniteMagnitude,
                              "An infinite container width is exactly the horizontal-scroll bug")
        }
    }

    @MainActor
    func testCrashLogViewIsSelectableAndHoldsFullLog() {
        let log = String(repeating: "frame line that is quite long\n", count: 50)
        let scroll = ServerManager.makeCrashLogScrollView(log: log, width: 640, height: 280)
        guard let textView = scroll.documentView as? NSTextView else {
            return XCTFail("documentView must be an NSTextView")
        }
        XCTAssertTrue(textView.isSelectable, "User must be able to select & copy the log")
        XCTAssertFalse(textView.isEditable)
        XCTAssertEqual(textView.string, log, "Full log must be present for copy")
    }
}
