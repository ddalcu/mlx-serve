import XCTest
@testable import MLXCore

final class ActivityPhrasesTests: XCTestCase {
    func testHiddenPhrasesPreserveMetricsAndElapsedTime() {
        XCTAssertEqual(
            GeneratingIndicator.statusSegments(showActivityPhrases: false, phrase: "brain-tickling"),
            [.gpu, .memory, .elapsed]
        )
    }

    func testVisiblePhraseFollowsMemoryAndPrecedesElapsedTime() {
        XCTAssertEqual(
            GeneratingIndicator.statusSegments(showActivityPhrases: true, phrase: "brain-tickling"),
            [.gpu, .memory, .activityPhrase("brain-tickling"), .elapsed]
        )
    }

    func testSegmentIdentitySurvivesPhraseVisibilityAndRotation() {
        let visible = GeneratingIndicator.statusSegments(showActivityPhrases: true, phrase: "first")
        let hidden = GeneratingIndicator.statusSegments(showActivityPhrases: false, phrase: "first")
        let rotated = GeneratingIndicator.statusSegments(showActivityPhrases: true, phrase: "second")
        XCTAssertEqual(visible.last?.id, hidden.last?.id)
        XCTAssertEqual(visible.map(\.id), rotated.map(\.id))
        XCTAssertEqual(Set(visible.map(\.id)).count, visible.count)
        XCTAssertEqual(Set(hidden.map(\.id)).count, hidden.count)
    }
}
