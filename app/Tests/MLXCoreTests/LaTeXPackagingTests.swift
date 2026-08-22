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

    /// Copying the bundle is only half of it: SwiftPM's own `Bundle.module`
    /// looks for it beside the .app, which codesign refuses to seal, so every
    /// hand-assembled build trapped on the first equation while this file's
    /// older "the bundle is copied somewhere" assertion stayed green
    /// (issue #233). Every path that compiles the app patches the lookup.
    func testEveryBuildPathPatchesSwaTexFontLookupBeforeCompiling() throws {
        for path in ["app/build.sh", ".github/workflows/release.yml", ".github/workflows/ci.yml"] {
            let script = try source(path)
            guard let patch = script.range(of: "patch-swatex-font-lookup.sh") else {
                return XCTFail("\(path) must patch SwaTexRender's font lookup")
            }
            guard let build = script.range(of: "swift build") else {
                return XCTFail("\(path) does not build the app")
            }
            XCTAssertTrue(
                patch.lowerBound < build.lowerBound,
                "\(path) must patch the lookup before compiling it"
            )
        }
    }

    /// The patch has to fit the source it edits — a silently unapplied one
    /// ships a build that crashes exactly where it did before.
    func testTheFontLookupPatchAppliesToTheCheckedOutSwaTexAndIsIdempotent() throws {
        let checkout = repositoryRoot
            .appendingPathComponent("app/.build/checkouts/SwaTex")
        let provider = checkout
            .appendingPathComponent("Sources/SwaTexRender/KaTeXFontProvider.swift")
        try XCTSkipUnless(
            FileManager.default.fileExists(atPath: provider.path),
            "SwaTex is not checked out"
        )

        let staged = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("swatex-patch-\(UUID().uuidString)")
        let stagedProvider = staged.appendingPathComponent("Sources/SwaTexRender/KaTeXFontProvider.swift")
        try FileManager.default.createDirectory(
            at: stagedProvider.deletingLastPathComponent(),
            withIntermediateDirectories: true
        )
        defer { try? FileManager.default.removeItem(at: staged) }

        // The checkout may already carry the patch from a previous build, so
        // stage the pristine upstream text the patch is written against.
        var pristine = try String(contentsOf: provider, encoding: .utf8)
        if let marker = pristine.range(of: "// Added by mlx-serve") {
            pristine = String(pristine[pristine.startIndex..<marker.lowerBound])
            pristine = pristine.replacingOccurrences(
                of: "let url = mlxServeKaTeXFontURL(name),",
                with: "let url = Bundle.module.url(\n"
                    + "                forResource: name, withExtension: \"ttf\", subdirectory: \"Fonts\"),"
            )
        }
        try pristine.write(to: stagedProvider, atomically: true, encoding: .utf8)

        for pass in 1...2 {
            let result = try Self.run(
                repositoryRoot.appendingPathComponent("scripts/patch-swatex-font-lookup.sh").path,
                staged.path
            )
            XCTAssertEqual(result, 0, "patch pass \(pass) failed")
        }

        let patched = try String(contentsOf: stagedProvider, encoding: .utf8)
        XCTAssertTrue(patched.contains("mlxServeKaTeXFontURL"))
        XCTAssertFalse(
            patched.contains("Bundle.module.url("),
            "the trapping accessor must not remain on the font path"
        )
        XCTAssertEqual(
            patched.components(separatedBy: "func mlxServeKaTeXFontURL").count - 1,
            1,
            "a second run must not append the helper again"
        )
    }

    private static func run(_ script: String, _ argument: String) throws -> Int32 {
        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/bin/bash")
        process.arguments = [script, argument]
        process.standardOutput = Pipe()
        process.standardError = Pipe()
        try process.run()
        process.waitUntilExit()
        return process.terminationStatus
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
