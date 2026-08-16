import XCTest

/// SwaTex ships KaTeX fonts as a SwiftPM resource bundle. A source build can
/// compile perfectly while a hand-assembled .app crashes or draws blanks if
/// that bundle is not copied, so dependency parity and both Developer ID
/// packaging paths are part of the feature contract.
final class LaTeXPackagingTests: XCTestCase {
    private var repositoryRoot: URL {
        URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent() // MLXCoreTests
            .deletingLastPathComponent() // Tests
            .deletingLastPathComponent() // app
            .deletingLastPathComponent() // repository
    }

    private func source(_ relativePath: String) throws -> String {
        try String(
            contentsOf: repositoryRoot.appendingPathComponent(relativePath),
            encoding: .utf8
        )
    }

    func testSwaTexDependencyIsDeclaredInBothAppManifests() throws {
        let package = try source("app/Package.swift")
        let project = try source("app/project.yml")

        for manifest in [package, project] {
            XCTAssertTrue(manifest.contains("https://github.com/PhraseHQ/SwaTex"))
            XCTAssertTrue(manifest.contains("SwaTexRender"))
            XCTAssertTrue(manifest.contains("0.5.0"))
            XCTAssertTrue(manifest.contains("0.6.0"))
        }
    }

    func testSwaTexFontBundleShipsInBothDeveloperIDPackagingPaths() throws {
        for path in ["app/build.sh", ".github/workflows/release.yml"] {
            let packaging = try source(path)
            XCTAssertTrue(
                packaging.contains("SwaTex_SwaTexRender.bundle"),
                "\(path) must copy SwaTex's bundled KaTeX fonts into Contents/Resources"
            )
        }
    }

    func testBuildPipelinesKeepEachPackagesDeclaredSwiftLanguageMode() throws {
        let globalSwift5Override = "-Xswiftc -swift-version -Xswiftc 5"
        for path in [
            "app/build.sh",
            "app/CLAUDE.md",
            ".github/workflows/ci.yml",
            ".github/workflows/release.yml",
        ] {
            XCTAssertFalse(
                try source(path).contains(globalSwift5Override),
                "\(path) must not force Swift 6 dependencies such as SwaTex into Swift 5 mode"
            )
        }
    }

    func testSwaTexAndBundledFontLicensesAreAttributed() throws {
        let notice = try source("NOTICE")
        for attribution in ["SwaTex", "RaTeX", "KaTeX", "SIL Open Font License"] {
            XCTAssertTrue(notice.contains(attribution), "NOTICE is missing \(attribution)")
        }
    }
}
