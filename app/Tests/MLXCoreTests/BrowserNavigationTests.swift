import XCTest
import AppKit
@testable import MLXCore

final class BrowserURLResolutionTests: XCTestCase {
    func testTargetsResolveToTheSchemeTheyMean() throws {
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        try "<html></html>".write(to: dir.appendingPathComponent("index.html"), atomically: true, encoding: .utf8)
        let wd = dir.path
        let cases: [(String, String)] = [
            ("https://example.com", "https://example.com"),
            ("http://example.com/x", "http://example.com/x"),
            ("example.com", "https://example.com"),
            ("localhost:3000", "http://localhost:3000"),
            ("127.0.0.1:8080/app", "http://127.0.0.1:8080/app"),
            ("file:///tmp/a.html", "file:///tmp/a.html"),
            ("/tmp/a.html", "file:///tmp/a.html"),
            ("index.html", "file://\(wd)/index.html"),
            ("./index.html", "file://\(wd)/index.html"),
        ]
        for (input, want) in cases {
            XCTAssertEqual(BrowserManager.resolveURL(input, workingDirectory: wd)?.absoluteString, want, input)
        }
        XCTAssertEqual(BrowserManager.resolveURL("missing.html", workingDirectory: wd)?.absoluteString, "https://missing.html")
        XCTAssertNil(BrowserManager.resolveURL("", workingDirectory: nil))
    }
}

/// The URL bar reads what the manager publishes: link clicks, back/forward and
/// same-document pushState all have to land there, not only tool navigations.
@MainActor
final class BrowserPublishedStateTests: XCTestCase {
    func testUrlAndTitleFollowEveryNavigationIncludingPushState() async throws {
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        let file = dir.appendingPathComponent("p.html")
        try "<html><head><title>Pushy</title></head><body></body></html>".write(to: file, atomically: true, encoding: .utf8)
        let m = BrowserManager.shared
        _ = try await m.load(file)
        XCTAssertEqual(m.currentURL, file.absoluteString)
        XCTAssertEqual(m.pageTitle, "Pushy")
        _ = try await m.evaluateJS("history.pushState({}, '', '?tab=2'); document.title = 'Pushy 2'; 1")
        for _ in 0..<20 where m.currentURL != file.absoluteString + "?tab=2" {
            try await Task.sleep(nanoseconds: 50_000_000)
        }
        XCTAssertEqual(m.currentURL, file.absoluteString + "?tab=2")
        XCTAssertEqual(m.pageTitle, "Pushy 2")
    }
}
