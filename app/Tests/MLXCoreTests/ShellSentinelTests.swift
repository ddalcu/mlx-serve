import XCTest
@testable import MLXCore

/// Pure-protocol tests for the console sentinel used to drive a shell inside the
/// agent sandbox. The guest runs an interactive /bin/sh over the console;
/// we frame each command so we can (a) find the end of THIS command's output
/// even when it contains newlines/prompts and (b) recover its exit code. These
/// tests exercise the framing + parsing with synthetic console text — no guest,
/// no HVF — so they run anywhere in CI.
final class ShellSentinelTests: XCTestCase {
    let nonce = "abc123"

    func testFrameCarriesCommandAndEchoProofExitMarker() {
        let bytes = ShellSentinel.frame(command: "ls -la", nonce: nonce, seq: 7)
        let s = String(decoding: bytes, as: UTF8.self)
        XCTAssertTrue(s.contains("ls -la"), "the command must be sent verbatim")
        XCTAssertTrue(s.contains("\"$?\""), "the marker must print the command's own exit status")
        // Echo-proof: the assembled prefix must NOT appear literally in what we
        // send (else an echoing tty reflects it and scan locks onto the `%d`).
        XCTAssertFalse(s.contains("__CTN_\(nonce)_EXIT7="),
                       "assembled marker must not appear literally in the frame")
        XCTAssertTrue(s.contains("__CTN_%s_EXIT%s="), "marker is assembled from %s args")
        XCTAssertTrue(s.contains("'\(nonce)'") && s.contains("'7'"), "nonce + seq passed as args")
    }

    func testScanSkipsEchoedMarkerAndFindsRealOne() {
        // Echo on: the console holds the ECHOED printf (…EXIT5=%d__) BEFORE the
        // real executed marker (…EXIT5=0__). scan must skip the non-numeric one.
        let text = "printf '\\n__CTN_\(nonce)_EXIT5=%d__\\n' \"$?\"\nhi\n\n__CTN_\(nonce)_EXIT5=0__\n"
        let r = ShellSentinel.scan(text, nonce: nonce, seq: 5)
        XCTAssertEqual(r?.code, 0, "must skip the echoed %d marker and match the real digit one")
    }

    func testScanReturnsNilBeforeMarkerArrives() {
        let text = "some output\nstill running"
        XCTAssertNil(ShellSentinel.scan(text, nonce: nonce, seq: 1))
    }

    func testScanReturnsNilWhenDigitsNotYetComplete() {
        // Prefix present but the closing "__" hasn't streamed in yet.
        let text = "out\n__CTN_\(nonce)_EXIT1=0"
        XCTAssertNil(ShellSentinel.scan(text, nonce: nonce, seq: 1),
                     "an incomplete marker (no closing __) must not be parsed")
    }

    func testScanRecoversOutputAndExitCode() {
        // `echo hi` prints "hi\n"; then our printf injects "\n" + marker. We strip
        // ONLY the injected newline, so the command's own trailing "\n" survives —
        // faithful to what `Process` stdout would capture on the host path.
        let text = "hi\n\n__CTN_\(nonce)_EXIT3=0__\n"
        let r = ShellSentinel.scan(text, nonce: nonce, seq: 3)
        XCTAssertEqual(r?.output, "hi\n", "keep the command's own newline; drop only the one we injected")
        XCTAssertEqual(r?.code, 0)
    }

    func testScanRecoversNonZeroExitCode() {
        let text = "boom\n\n__CTN_\(nonce)_EXIT4=127__\n"
        let r = ShellSentinel.scan(text, nonce: nonce, seq: 4)
        XCTAssertEqual(r?.output, "boom\n")
        XCTAssertEqual(r?.code, 127)
    }

    func testScanStripsOnlyInjectedNewlineWhenCommandOutputHasNone() {
        // `printf hi` prints "hi" (no newline); guest -> "hi" + injected "\n" + marker.
        let text = "hi\n__CTN_\(nonce)_EXIT6=0__\n"
        let r = ShellSentinel.scan(text, nonce: nonce, seq: 6)
        XCTAssertEqual(r?.output, "hi", "no command newline to keep — only the injected one is removed")
    }

    func testScanIgnoresMarkerForADifferentSeq() {
        // A stale marker from a previous (e.g. timed-out) command must not be
        // mistaken for the current one.
        let text = "leftover\n\n__CTN_\(nonce)_EXIT2=0__\n"
        XCTAssertNil(ShellSentinel.scan(text, nonce: nonce, seq: 5),
                     "only the marker for the requested seq may match")
    }

    func testScanIgnoresMarkerForADifferentNonce() {
        let text = "x\n\n__CTN_otherNonce_EXIT1=0__\n"
        XCTAssertNil(ShellSentinel.scan(text, nonce: nonce, seq: 1))
    }

    func testReadyProbeIsEchoProof() {
        // The probe must NOT contain the assembled marker literally — it builds
        // it via `printf %s <nonce>` so that terminal-echoed input (before
        // stty -echo takes effect) can't false-trigger readiness. Only the shell
        // actually running the printf produces the assembled marker.
        let probe = String(decoding: ShellSentinel.readyProbe(nonce: nonce), as: UTF8.self)
        XCTAssertFalse(probe.contains("__CTN_\(nonce)_READY__"),
                       "assembled marker must not appear literally in the probe (echo would false-match)")
        XCTAssertTrue(probe.contains("__CTN_%s_READY__"), "marker is assembled from %s + nonce arg")
        XCTAssertTrue(probe.contains("'\(nonce)'"), "nonce is passed as a printf argument")
        XCTAssertTrue(probe.contains("stty -echo"), "the ready probe should also quiet terminal echo")
    }

    func testReadyDetectionMatchesAssembledMarker() {
        XCTAssertTrue(ShellSentinel.isReady("banner\n__CTN_\(nonce)_READY__\n", nonce: nonce))
        XCTAssertFalse(ShellSentinel.isReady("still booting...", nonce: nonce))
    }

    func testMultilineCommandOutputSurvives() {
        // Output with embedded blank lines/newlines must not confuse the parser —
        // that's the whole point of the trailing marker.
        let body = "line1\n\nline3\n"
        let text = body + "\n__CTN_\(nonce)_EXIT9=0__\n"
        let r = ShellSentinel.scan(text, nonce: nonce, seq: 9)
        XCTAssertEqual(r?.output, "line1\n\nline3\n", "only the injected newline stripped; interior + own newlines preserved")
        XCTAssertEqual(r?.code, 0)
    }

    // MARK: terminal control-sequence sanitizing

    private let ESC = "\u{1B}"

    func testSanitizeStripsColorAndKeepsText() {
        // curl -I header line: `\e[1mX-Powered-By\e[0m: Express`
        let raw = "\(ESC)[1mX-Powered-By\(ESC)[0m: Express"
        XCTAssertEqual(TerminalOutput.sanitize(raw), "X-Powered-By: Express")
    }

    func testSanitizeCollapsesNpmSpinnerStorm() {
        // npm's progress: repeated `\e[1G\e[0K<frame>` (cursor col 1 + clear
        // line), with braille spinner frames, ending in the real message.
        let ESC = self.ESC
        var raw = ""
        for f in ["⠙", "⠹", "⠸", "⠼"] { raw += "\(ESC)[1G\(ESC)[0K\(f)" }
        raw += "\(ESC)[1G\(ESC)[0Kadded 67 packages in 3s\n"
        let out = TerminalOutput.sanitize(raw)
        // Only the final overwrite of the line survives — no braille, no escapes.
        XCTAssertEqual(out, "added 67 packages in 3s\n")
        XCTAssertFalse(out.contains("⠙"))
        XCTAssertFalse(out.contains("\(ESC)"))
        XCTAssertFalse(out.contains("[1G"))
    }

    func testSanitizeHandlesBareCarriageReturnProgress() {
        // A `\r`-overwriting progress bar (no escapes): last write wins per line.
        let raw = "downloading  10%\rdownloading  55%\rdownloading 100%\ndone\n"
        XCTAssertEqual(TerminalOutput.sanitize(raw), "downloading 100%\ndone\n")
    }

    func testSanitizeStripsOscTitleAndLoneEscapes() {
        let ESC = self.ESC
        let raw = "\(ESC)]0;my title\u{07}hello\(ESC)(Bworld"
        XCTAssertEqual(TerminalOutput.sanitize(raw), "helloworld")
    }

    func testSanitizeIsIdentityOnPlainText() {
        let plain = "just plain output\nwith two lines\n"
        XCTAssertEqual(TerminalOutput.sanitize(plain), plain)
    }

    func testSanitizeNormalizesCrlfWithoutEatingLines() {
        // The guest tty is ONLCR: every \n arrives as \r\n. The trailing \r must
        // NOT be treated as a line-overwrite (that ran multiple lines together).
        let raw = "ROUTED_12\r\nLinux\r\n/workspace\r\n"
        XCTAssertEqual(TerminalOutput.sanitize(raw), "ROUTED_12\nLinux\n/workspace\n")
    }
}
