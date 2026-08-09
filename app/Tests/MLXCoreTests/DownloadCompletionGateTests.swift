import XCTest
@testable import MLXCore

/// DownloadManager's `onFinish` hooks run on completion, FAILURE and cancel
/// alike (documented on `start`/`startBundle` — the caller decides what a
/// failure means). Two callers forgot: the held create-prompt ran its
/// generation against a model that never landed (clearing the held prompt its
/// own card promised to keep), and the welcome sheet's "Get" dismissed into a
/// chat with no model. Any completion hook that triggers a follow-on action
/// must re-check the bytes actually arrived.
final class DownloadCompletionGateTests: XCTestCase {

    private func source(_ relativePath: String) throws -> String {
        let url = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()  // MLXCoreTests
            .deletingLastPathComponent()  // Tests
            .deletingLastPathComponent()  // app
            .appendingPathComponent(relativePath)
        return try String(contentsOf: url, encoding: .utf8)
    }

    /// Slice a function body out of a source file (from its declaration to the
    /// next top-level-looking declaration) so the assertion can demand the
    /// readiness check INSIDE the right closure's neighborhood.
    private func body(of marker: String, in source: String, until end: String) throws -> String {
        let start = try XCTUnwrap(source.range(of: marker), "\(marker) not found").upperBound
        let tail = source[start...]
        let stop = try XCTUnwrap(tail.range(of: end), "\(end) not found after \(marker)").lowerBound
        return String(tail[..<stop])
    }

    func testHeldCreatePromptCompletionReChecksTheModelIsOnDisk() throws {
        let chat = try source("Sources/MLXServe/Views/ChatView.swift")
        let fn = try body(of: "private func startHeldDownload",
                          in: chat, until: "private func generateInChat")
        XCTAssertTrue(fn.contains("createModelReady("), """
            startHeldDownload's completion must gate on createModelReady — \
            startBundle calls onFinish on failure and cancel too, and a held \
            prompt whose download failed stays held (the card says so).
            """)
    }

    func testWelcomeGetCompletionReChecksTheDownloadLanded() throws {
        let welcome = try source("Sources/MLXServe/Views/WelcomeView.swift")
        let fn = try body(of: "private func startDownload",
                          in: welcome, until: "\n}")
        XCTAssertTrue(fn.contains("isReady("), """
            WelcomeModelRow's Get completion must gate on downloads.isReady — \
            a failed download must re-offer Get/Resume, not dismiss the \
            welcome sheet into a chat with no model.
            """)
    }
}
