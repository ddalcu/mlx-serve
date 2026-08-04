import XCTest
@testable import MLXCore

/// The About section's outbound links: release notes, the repo, and the
/// author's X account.
///
/// Pure data so the destinations are testable — a Link whose URL is built at
/// the call site is a dead button nothing catches (`URL(string:)!` on a typo
/// would crash the pane instead, which is worse). The section is deliberately
/// NOT part of Updates: that whole section is gated on `selfUpdate`, so on a
/// Mac App Store build these links would never render at all.
final class CommunityLinksTests: XCTestCase {

    private func source(_ relativePath: String) throws -> String {
        let url = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()  // MLXCoreTests
            .deletingLastPathComponent()  // Tests
            .deletingLastPathComponent()  // app
            .appendingPathComponent(relativePath)
        return try String(contentsOf: url, encoding: .utf8)
    }

    // MARK: Catalog

    func testEveryLinkHasARealDestinationAndSomethingToRead() {
        XCTAssertFalse(CommunityLinks.all.isEmpty)
        for item in CommunityLinks.all {
            XCTAssertFalse(item.title.isEmpty, "\(item.id) has no title")
            XCTAssertFalse(item.explainer.isEmpty, "\(item.id) has no explainer")
            XCTAssertFalse(item.actionLabel.isEmpty, "\(item.id) has no button label")
            let scheme = item.url.scheme
            XCTAssertEqual(scheme, "https", "\(item.id) must be https, got \(scheme ?? "nil")")
            XCTAssertNotNil(item.url.host, "\(item.id) has no host")
        }
    }

    func testDestinationsAreTheOnesWeAdvertise() {
        let byId = Dictionary(uniqueKeysWithValues: CommunityLinks.all.map { ($0.id, $0.url.absoluteString) })
        XCTAssertEqual(byId["releases"], "https://github.com/ddalcu/mlx-serve/releases")
        XCTAssertEqual(byId["star"], "https://github.com/ddalcu/mlx-serve")
        XCTAssertEqual(byId["x"], "https://x.com/ddalcu")
    }

    /// The releases link is built from `UpdateChecker.repo`, the same constant
    /// the updater fetches against — a hardcoded second copy is how the pane
    /// ends up pointing at a repo the app no longer updates from.
    func testReleasesLinkFollowsTheRepoTheUpdaterUses() {
        let releases = CommunityLinks.all.first { $0.id == "releases" }
        XCTAssertEqual(releases?.url.absoluteString,
                       "https://github.com/\(UpdateChecker.repo)/releases")
    }

    func testIdsAreUnique() {
        let ids = CommunityLinks.all.map(\.id)
        XCTAssertEqual(Set(ids).count, ids.count, "duplicate link id")
    }

    // MARK: Placement

    /// The whole point of a separate category: Updates is `selfUpdate`-gated,
    /// so App Store users would never see these.
    func testAboutIsVisibleEvenWhenSelfUpdateIsOff() {
        let mas = SettingsCategory.visible(engine: nil, selfUpdate: false)
        XCTAssertTrue(mas.contains(.about))
        XCTAssertFalse(mas.contains(.updates))
    }

    func testAboutRendersLast() {
        let all = SettingsCategory.visible(engine: nil, selfUpdate: true)
        XCTAssertEqual(all.last, .about)
    }

    // MARK: Wiring

    /// A catalog nothing renders still passes every assertion above.
    func testSettingsViewRendersTheAboutSectionFromTheCatalog() throws {
        let src = try source("Sources/MLXServe/Views/SettingsView.swift")
        XCTAssertTrue(src.contains("category: .about"),
                      "SettingsView no longer builds the About section")
        XCTAssertTrue(src.contains("CommunityLinks.all"),
                      "the About section must iterate the shared catalog")
    }
}
