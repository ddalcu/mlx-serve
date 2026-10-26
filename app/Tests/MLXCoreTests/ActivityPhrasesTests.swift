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
}
