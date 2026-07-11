import XCTest
@testable import MLXCore

/// Pins the URL that the tray's Endpoints accordion "open in browser" root item
/// navigates to. The construction is a pure static helper so it's testable
/// without rendering the SwiftUI view.
final class EndpointsSectionTests: XCTestCase {

    func testRootURLPointsAtServerRoot() {
        XCTAssertEqual(
            EndpointsSection.rootURL("http://localhost:11234")?.absoluteString,
            "http://localhost:11234/"
        )
    }

    /// `baseURL` may or may not already carry a trailing slash; either way the
    /// root link must resolve to exactly one — never `…//`.
    func testRootURLNormalizesToSingleTrailingSlash() {
        XCTAssertEqual(
            EndpointsSection.rootURL("http://localhost:8080/")?.absoluteString,
            "http://localhost:8080/"
        )
    }

    /// The `/v1/` base is what most people paste into an OpenAI-compatible
    /// client, so the accordion offers it as its own copy row. Same
    /// trailing-slash tolerance as `rootURL`.
    func testV1BaseURLIsTheClientBase() {
        XCTAssertEqual(
            EndpointsSection.v1BaseURL("http://localhost:11234"),
            "http://localhost:11234/v1/"
        )
        XCTAssertEqual(
            EndpointsSection.v1BaseURL("http://localhost:8080/"),
            "http://localhost:8080/v1/"
        )
    }
}
