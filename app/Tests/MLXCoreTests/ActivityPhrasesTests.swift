import XCTest
@testable import MLXCore

final class ActivityPhrasesTests: XCTestCase {
    func testDefaultAndPersistenceAcrossIndicators() {
        let defaults = UserDefaults.standard
        let key = InterfacePrefKey.showActivityPhrases
        let previous = defaults.object(forKey: key)
        defer {
            if let previous { defaults.set(previous, forKey: key) }
            else { defaults.removeObject(forKey: key) }
        }
        defaults.removeObject(forKey: key)

        let indicator = GeneratingIndicator()
        XCTAssertTrue(indicator.showActivityPhrases)
        indicator.showActivityPhrases = false
        XCTAssertEqual(defaults.object(forKey: key) as? Bool, false)
        XCTAssertFalse(GeneratingIndicator().showActivityPhrases)
        indicator.showActivityPhrases = true
        XCTAssertTrue(GeneratingIndicator().showActivityPhrases)
    }
}
