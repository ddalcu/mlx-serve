import XCTest
import SwiftUI
import AppKit
@testable import MLXCore

/// The Modern look of the generated-audio player: the setting that picks it, and what it draws.
final class ModernAudioPlayerTests: XCTestCase {

    private var suite: String!
    private var defaults: UserDefaults!

    override func setUp() {
        suite = "mlx-core-player-style-\(UUID().uuidString)"
        defaults = UserDefaults(suiteName: suite)
    }

    override func tearDown() {
        defaults.removePersistentDomain(forName: suite)
    }

    // MARK: - The setting

    func testMlxAmpIsTheDefault() {
        XCTAssertEqual(AudioPlayerStyle.defaultStyle, .mlxAmp)
        XCTAssertEqual(stored().wrappedValue, .mlxAmp)
    }

    func testTheStoredLookIsTheOneShown() {
        defaults.set(AudioPlayerStyle.modern.rawValue, forKey: InterfacePrefKey.audioPlayerStyle)
        XCTAssertEqual(stored().wrappedValue, .modern)
    }

    /// A value a later build wrote falls back to the default rather than to nothing.
    func testAnUnknownValueFallsBackToMlxAmp() {
        defaults.set("cassette", forKey: InterfacePrefKey.audioPlayerStyle)
        XCTAssertEqual(stored().wrappedValue, .mlxAmp)
    }

    /// The setting the way the three views read it.
    private func stored() -> AppStorage<AudioPlayerStyle> {
        AppStorage(wrappedValue: AudioPlayerStyle.defaultStyle, InterfacePrefKey.audioPlayerStyle, store: defaults)
    }

    // MARK: - What the deck shows

    func testTheInfoLineNamesRateChannelsAndBitrate() {
        let stereo = AudioClipPlayer.Info(seconds: 120, sampleRate: 48_000, channels: 2, kbps: 1536)
        XCTAssertEqual(ModernAudioPlayer.infoLine(stereo), "48 kHz · Stereo · 1536 kbps")
        let mono = AudioClipPlayer.Info(seconds: 4, sampleRate: 44_100, channels: 1, kbps: 705)
        XCTAssertEqual(ModernAudioPlayer.infoLine(mono), "44.1 kHz · Mono · 705 kbps")
        XCTAssertEqual(ModernAudioPlayer.infoLine(nil), "")
    }

    func testTheVolumeGlyphFollowsTheLevel() {
        XCTAssertEqual(ModernAudioPlayer.volumeSymbol(0), "speaker.slash.fill")
        XCTAssertEqual(ModernAudioPlayer.volumeSymbol(0.2), "speaker.wave.1.fill")
        XCTAssertEqual(ModernAudioPlayer.volumeSymbol(0.5), "speaker.wave.2.fill")
        XCTAssertEqual(ModernAudioPlayer.volumeSymbol(0.9), "speaker.wave.3.fill")
    }

    func testADeckKeyIsGreyVioletWhenOnAndFaintWhileLocked() {
        XCTAssertEqual(ModernAudioPlayer.keyInk(enabled: true, lit: false, pressed: false), .secondaryLabelColor)
        XCTAssertEqual(ModernAudioPlayer.keyInk(enabled: true, lit: false, pressed: true), .labelColor)
        XCTAssertEqual(ModernAudioPlayer.keyInk(enabled: true, lit: true, pressed: false), ModernSpectrum.violetNS)
        XCTAssertEqual(ModernAudioPlayer.keyInk(enabled: false, lit: false, pressed: false), .quaternaryLabelColor)
        XCTAssertEqual(ModernAudioPlayer.keyInk(enabled: false, lit: true, pressed: false).alphaComponent, 0.4, accuracy: 0.001)
    }

    func testAPointerOnTheSeekBarClampsToTheTrack() {
        let track = 40.0...240.0
        XCTAssertEqual(ModernAudioPlayer.seekFraction(x: 140, track: track), 0.5)
        XCTAssertEqual(ModernAudioPlayer.seekFraction(x: 10, track: track), 0)
        XCTAssertEqual(ModernAudioPlayer.seekFraction(x: 400, track: track), 1)
    }

    /// A click on the clocks beside the seek bar is not a seek; one on the track (or its thumb's reach) is.
    @MainActor
    func testTheSeekBarTakesClicksOnTheTrackOnly() {
        let size = CGSize(width: 300, height: 20)
        let hit = ModernLiveDrawing.seekHitRect(size: size)
        let track = ModernLiveDrawing.track(width: size.width)
        XCTAssertFalse(hit.contains(CGPoint(x: 4, y: 10)), "the elapsed clock")
        XCTAssertFalse(hit.contains(CGPoint(x: 296, y: 10)), "the total clock")
        XCTAssertTrue(hit.contains(CGPoint(x: track.lowerBound - 4, y: 10)), "the thumb at the start")
        XCTAssertTrue(hit.contains(CGPoint(x: 150, y: 10)), "the middle of the track")
    }

    /// A click outside the canvas's pointer region does nothing, drag included; one inside drags anywhere.
    @MainActor
    func testACanvasClickOutsideItsPointerRegionIsIgnoredWithItsDrag() throws {
        let window = NSWindow(contentRect: NSRect(x: 0, y: 0, width: 200, height: 20), styleMask: [.borderless],
                              backing: .buffered, defer: false)
        let view = LiveCanvasView(frame: NSRect(x: 0, y: 0, width: 200, height: 20))
        window.contentView = view
        var hits: [CGFloat] = []
        view.onPointer = { point, _ in hits.append(point.x) }
        view.pointerRegion = { size in CGRect(x: 50, y: 0, width: 100, height: size.height) }
        func event(_ type: NSEvent.EventType, _ x: CGFloat) throws -> NSEvent {
            try XCTUnwrap(NSEvent.mouseEvent(with: type, location: NSPoint(x: x, y: 10), modifierFlags: [], timestamp: 0,
                                             windowNumber: window.windowNumber, context: nil, eventNumber: 0,
                                             clickCount: 1, pressure: 1))
        }
        view.mouseDown(with: try event(.leftMouseDown, 20))
        view.mouseDragged(with: try event(.leftMouseDragged, 100))
        XCTAssertEqual(hits, [])
        view.mouseDown(with: try event(.leftMouseDown, 100))
        view.mouseDragged(with: try event(.leftMouseDragged, 190))
        XCTAssertEqual(hits, [100, 190])
    }

    /// The history row's waveform moves only while its track plays, not while it is paused.
    func testAHistoryRowMovesOnlyWhileItsTrackPlays() {
        XCTAssertTrue(ModernPlaylist.moves("a.wav", current: "a.wav", paused: false))
        XCTAssertFalse(ModernPlaylist.moves("a.wav", current: "a.wav", paused: true))
        XCTAssertFalse(ModernPlaylist.moves("b.wav", current: "a.wav", paused: false))
    }

    /// The analyzer's bands, with each neighbour pair's midpoint between them: a wide screen reads as one smooth spectrum.
    func testTheSpectrumDrawsEachBandAndTheMidpointsBetween() {
        XCTAssertEqual(ModernSpectrum.bars([0, 1, 0.5]), [0, 0.5, 1, 0.75, 0.5])
        XCTAssertEqual(ModernSpectrum.bars([0.3]), [0.3])
        XCTAssertEqual(ModernSpectrum.bars([]), [])
    }
}
