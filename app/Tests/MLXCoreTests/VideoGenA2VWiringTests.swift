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
        // Capability-gated on purpose: the clip must reach the request on a
        // backend that takes audio input (LTX), and must NOT reach the
        // transcode — whose failure is a hard error — on one that generates
        // its own soundtrack (MiniMax-H3).
        XCTAssertTrue(
            source.contains("audioPath: model.supportsAudioInput ? audioURL?.path : nil"),
            "VideoGenView must pass the attached clip into VideoGenRequest.audioPath (gated on supportsAudioInput) — otherwise a2vid silently degrades to generated audio, or a stale clip hard-errors an H3 generate"
        )
    }

    func testSpeechSectionExists() throws {
        let source = try videoGenViewSource()
        XCTAssertTrue(source.contains("Speech & sound"),
                      "The Speech & sound (audio-to-video) section is missing from the video pane")
        XCTAssertTrue(source.contains("framesCovering"),
                      "Attaching a clip should auto-suggest a frame count that covers it")
    }

    /// The live preview is opt-in, which is only true if the pane HAS the
    /// control and every half of it is wired: a `@State` nobody persists resets
    /// each launch, and a toggle that never reaches the request is the
    /// silently-dropped-setting class this file exists for.
    func testLivePreviewToggleIsPresentAndFullyWired() throws {
        let source = try videoGenViewSource()
        XCTAssertTrue(source.contains("isOn: $livePreview"),
                      "The video pane must offer a live-preview toggle — the server default is off, so without a control the feature is unreachable")
        XCTAssertTrue(source.contains("livePreview: livePreview"),
                      "The toggle must reach VideoGenRequest.livePreview, or it changes nothing on the wire")
        XCTAssertTrue(source.contains("livePreview = s.livePreview"),
                      "The toggle must hydrate from VideoGenSettings, or it reads off after every launch")
        XCTAssertTrue(source.contains("s.livePreview = livePreview"),
                      "The toggle must persist into VideoGenSettings, or it forgets the user's choice")
    }
}
