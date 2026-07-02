import XCTest
@testable import MLXCore

/// The toolbar's working-folder chip shows only the folder NAME (the full path
/// lives in the tooltip) — a full home-dir path ate toolbar width and leaked
/// into screenshots. Pure helper test; the SwiftUI shell is not testable.
final class WorkingDirectoryIndicatorTests: XCTestCase {
    func testDisplayNameShowsOnlyTheFolderName() {
        XCTAssertEqual(WorkingDirectoryIndicator.displayName("/Users/d/projects/myapp"), "myapp")
        XCTAssertEqual(WorkingDirectoryIndicator.displayName("/Users/d/projects/myapp/"), "myapp",
                       "trailing slash must not blank the name")
        XCTAssertEqual(WorkingDirectoryIndicator.displayName("workspace"), "workspace")
        XCTAssertEqual(WorkingDirectoryIndicator.displayName("/"), "/",
                       "filesystem root has no parent to abbreviate to")
    }
}
