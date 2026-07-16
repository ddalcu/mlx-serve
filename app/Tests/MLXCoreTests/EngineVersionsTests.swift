import XCTest
@testable import MLXCore

/// `EngineVersions.parse` turns `mlx-serve --version` stdout into named rows so
/// Settings can show engine versions WITHOUT booting the server (the app spawns
/// `mlx-serve --version` — a print-and-exit that never binds a port or loads a
/// model). Pins the parse contract against the exact Zig output shape
/// (src/version.zig): one `name value` line per component, version may contain
/// spaces (`ggml 0.16.0 (47c786924)`).
final class EngineVersionsTests: XCTestCase {

    private let sample = """
    mlx-serve 26.7.9
    mlx 0.32.0
    mlx-c 0.6.0_3
    ggml 0.16.0 (47c786924)
    llama.cpp b9999
    gguf 3
    ds4 80ebbc3
    """

    func testParsesEveryComponentInOrder() {
        let rows = EngineVersions.parse(sample)
        XCTAssertEqual(rows.map(\.name), ["mlx-serve", "mlx", "mlx-c", "ggml", "llama.cpp", "gguf", "ds4"])
        XCTAssertEqual(rows.map(\.version), ["26.7.9", "0.32.0", "0.6.0_3", "0.16.0 (47c786924)", "b9999", "3", "80ebbc3"])
    }

    func testVersionKeepsInternalSpaces() {
        // Everything after the first token is the version — the ggml commit
        // parenthetical must survive.
        let ggml = EngineVersions.parse(sample).first { $0.name == "ggml" }
        XCTAssertEqual(ggml?.version, "0.16.0 (47c786924)")
    }

    func testEngineRowsDropsTheAppItself() {
        // The "Installed version" row already shows the app version; the Engines
        // list is the embedded libraries only.
        let engines = EngineVersions.engineRows(from: sample)
        XCTAssertFalse(engines.contains { $0.name == "mlx-serve" })
        XCTAssertEqual(engines.first?.name, "mlx")
    }

    func testToleratesBlankAndMalformedLines() {
        let text = """

        mlx 0.32.0

        bare-token-no-version
        ds4 80ebbc3
        """
        let rows = EngineVersions.parse(text)
        XCTAssertEqual(rows.map(\.name), ["mlx", "ds4"], "blank + single-token lines are skipped")
    }

    func testEmptyInputYieldsNothing() {
        XCTAssertTrue(EngineVersions.parse("").isEmpty)
        XCTAssertTrue(EngineVersions.parse("   \n\n").isEmpty)
    }

    func testRowsAreUniquelyIdentifiedByName() {
        let rows = EngineVersions.parse(sample)
        XCTAssertEqual(Set(rows.map(\.id)).count, rows.count)
    }
}
