import XCTest

// =============================================================================
// MARK: - Replica of GenExperiment (Views/StatusMenuView.swift)
//
// MLXCore is an executable target — tests can't @testable-import it, so the
// pure model behind the menu's "Experiments" section is replicated verbatim
// here (same pattern as MediaGenTests). Keep in sync with the production enum.
// =============================================================================

private enum GenExperimentReplica: String, CaseIterable, Identifiable {
    case image, video, audio

    var id: String { rawValue }

    var icon: String {
        switch self {
        case .image: "photo.on.rectangle.angled"
        case .video: "film.stack"
        case .audio: "waveform"
        }
    }

    var title: String {
        switch self {
        case .image: "ImageGen"
        case .video: "VideoGen"
        case .audio: "AudioGen"
        }
    }

    var help: String {
        switch self {
        case .image: "Image Generation (FLUX.2 / Krea-2)"
        case .video: "Video Generation (LTX-Video 2.3)"
        case .audio: "Audio Generation — neural TTS & voice cloning"
        }
    }
}

final class GenExperimentTests: XCTestCase {

    /// The section must contain exactly the three media tools, in display
    /// order. Adding a fourth gen feature without surfacing it here is the
    /// regression this pins.
    func testSectionHasExactlyTheThreeMediaToolsInOrder() {
        XCTAssertEqual(GenExperimentReplica.allCases.map(\.title),
                       ["ImageGen", "VideoGen", "AudioGen"])
    }

    /// Every tile has a non-empty tooltip naming its modality.
    func testHelpTextPresentPerTool() {
        for e in GenExperimentReplica.allCases {
            XCTAssertFalse(e.help.isEmpty, "\(e.title) needs a tooltip")
        }
    }

    /// Icons are the picker's visual identity — distinct per tool.
    func testIconsAreUnique() {
        let icons = GenExperimentReplica.allCases.map(\.icon)
        XCTAssertEqual(Set(icons).count, icons.count, "Each experiment needs a distinct icon")
    }
}
