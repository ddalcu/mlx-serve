import XCTest

/// Pins the VideoGenView → VideoGenRequest audio wiring. SwiftUI views can't
/// be instantiated in tests, and an attached clip that never reaches the
/// request is exactly the "settings silently dropped from the wire" bug class
/// that hit pipeline/cfg/stg — so this reads the view source directly (the
/// InfoPlistTests / VideoGenDialogueExampleTests pattern).
final class VideoGenA2VWiringTests: XCTestCase {
    private func videoGenViewSource() throws -> String {
        let url = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent() // MLXCoreTests
            .deletingLastPathComponent() // Tests
            .deletingLastPathComponent() // app root
            .appendingPathComponent("Sources/MLXServe/Views/VideoGenView.swift")
        return try String(contentsOf: url, encoding: .utf8)
    }

    func testAttachedAudioReachesTheRequest() throws {
        let source = try videoGenViewSource()
        XCTAssertTrue(
            source.contains("audioPath: audioURL?.path"),
            "VideoGenView must pass the attached clip into VideoGenRequest.audioPath — otherwise a2vid silently degrades to generated audio"
        )
    }

    func testSpeechSectionExists() throws {
        let source = try videoGenViewSource()
        XCTAssertTrue(source.contains("Speech & sound"),
                      "The Speech & sound (audio-to-video) section is missing from the video pane")
        XCTAssertTrue(source.contains("framesCovering"),
                      "Attaching a clip should auto-suggest a frame count that covers it")
    }
}
