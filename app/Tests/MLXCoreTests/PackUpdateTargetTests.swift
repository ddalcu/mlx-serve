import XCTest
@testable import MLXCore

final class PackUpdateTargetTests: XCTestCase {
    private func model(_ name: String = "org/m", source: LocalModelSource = .mlxServe, type: String = "qwen3",
                       quantFile: String? = nil) -> LocalModel {
        LocalModel(id: "\(source.rawValue):\(name)", name: name, path: "/tmp/root/\(name)",
                   sizeFormatted: "1 GB", modelType: type, source: source, kind: .base, quantFile: quantFile)
    }

    private func entry(_ path: String, _ size: Int) -> [String: Any] { ["path": path, "type": "file", "size": size] }

    private func tempDir() throws -> String {
        let dir = (NSTemporaryDirectory() as NSString).appendingPathComponent("pack-target-\(UUID().uuidString)")
        try FileManager.default.createDirectory(atPath: dir, withIntermediateDirectories: true)
        addTeardownBlock { try? FileManager.default.removeItem(atPath: dir) }
        return dir
    }

    func testAnotherToolsFolderIsNeverChecked() {
        for source in [LocalModelSource.lmStudio, .huggingFace, .mtplx, .osaurus] {
            guard case .skipped = PackUpdateCheck.target(for: model(source: source), marker: nil, drafterOff: false) else {
                return XCTFail("\(source) must be skipped")
            }
        }
        XCTAssertEqual(PackUpdateCheck.target(for: model(source: .custom), marker: nil, drafterOff: false),
                       .check(repo: "org/m", selection: .chat(drafter: true)))
    }

    func testTheMarkerWinsOverTheName() {
        XCTAssertEqual(PackUpdateCheck.target(for: model(), marker: PackSource(repo: "real/repo"), drafterOff: true),
                       .check(repo: "real/repo", selection: .chat(drafter: false)))
        XCTAssertEqual(PackUpdateCheck.target(for: model("org/m-4bit"), marker: PackSource(repo: "org/m", subfolder: "4bit"), drafterOff: false),
                       .check(repo: "org/m", selection: .variant("4bit")))
        XCTAssertEqual(PackUpdateCheck.target(for: model("org/m-2bit"), marker: PackSource(repo: "org/m", revision: "2bit"), drafterOff: true),
                       .check(repo: "org/m", revision: "2bit", selection: .chat(drafter: false)), "a branch download checks its branch")
    }

    func testAGgufQuantChecksOnlyItsOwnShards() {
        let m = model(type: "llama", quantFile: "m-Q8_0-00001-of-00002.gguf")
        XCTAssertEqual(PackUpdateCheck.target(for: m, marker: nil, drafterOff: false),
                       .check(repo: "org/m", selection: .gguf("m-Q8_0-00001-of-00002.gguf")))
        let entries = [entry("m-Q4_K_M.gguf", 4), entry("Q8_0/m-Q8_0-00001-of-00002.gguf", 8),
                       entry("Q8_0/m-Q8_0-00002-of-00002.gguf", 9), entry("README.md", 1)]
        XCTAssertEqual(PackUpdateCheck.wanted(entries, selection: .gguf("m-Q8_0-00001-of-00002.gguf")).map(\.0),
                       ["Q8_0/m-Q8_0-00001-of-00002.gguf", "Q8_0/m-Q8_0-00002-of-00002.gguf"])
    }

    func testAMediaModelChecksItsFamilySelection() {
        guard case .check(let repo, _, .media(let sel)) = PackUpdateCheck.target(for: model(type: "flux2"), marker: nil, drafterOff: false) else {
            return XCTFail("a FLUX pack resolves to its bundle's selection")
        }
        XCTAssertEqual(repo, "org/m")
        XCTAssertTrue(sel.recursive, "FLUX keeps its weight subdirs")
    }

    func testMissingOnHuggingFaceIsALocalCopy() throws {
        let dir = try tempDir()
        XCTAssertEqual(PackUpdateCheck.classify(status: 404, wanted: [], inDir: dir), .localCopy)
        XCTAssertEqual(PackUpdateCheck.classify(status: 401, wanted: [], inDir: dir), .localCopy)
        XCTAssertEqual(PackUpdateCheck.classify(status: 200, wanted: [], inDir: dir), .localCopy, "a repo with none of our files is not our source")
        XCTAssertEqual(PackUpdateCheck.classify(status: 503, wanted: [("config.json", 1)], inDir: dir), .failed)
    }

    func testAddedAndReplacedFilesAreCountedApart() throws {
        let dir = try tempDir()
        try Data(count: 10).write(to: URL(fileURLWithPath: (dir as NSString).appendingPathComponent("config.json")))
        try Data(count: 3).write(to: URL(fileURLWithPath: (dir as NSString).appendingPathComponent("model.safetensors")))
        XCTAssertEqual(PackUpdateCheck.classify(status: 200, wanted: [("config.json", 10)], inDir: dir), .upToDate)
        XCTAssertEqual(PackUpdateCheck.classify(status: 200, wanted: [("config.json", 10), ("drafter/model.safetensors", 30)], inDir: dir),
                       .update(PackUpdate(files: 1, bytes: 30, replaces: 0)))
        XCTAssertEqual(PackUpdateCheck.classify(status: 200, wanted: [("config.json", 10), ("model.safetensors", 20)], inDir: dir),
                       .update(PackUpdate(files: 1, bytes: 20, replaces: 1)))
    }

    func testTheSourceMarkerRoundTrips() throws {
        let dir = try tempDir()
        XCTAssertNil(PackSource.read(dir: dir))
        PackSource(repo: "org/m", subfolder: "4bit").write(dir: dir)
        XCTAssertEqual(PackSource.read(dir: dir), PackSource(repo: "org/m", subfolder: "4bit"))
        PackSource(repo: "org/m", revision: "2bit", commit: "abc").write(dir: dir)
        XCTAssertEqual(PackSource.read(dir: dir)?.source, HFSource(repo: "org/m", revision: "2bit"))
        XCTAssertEqual(PackSource.read(dir: dir)?.commit, "abc")
        XCTAssertEqual(try JSONDecoder().decode(PackSource.self, from: Data(#"{"repo":"org/m"}"#.utf8)).source, HFSource(repo: "org/m"),
                       "a marker written before branches reads as main")
    }

    func testTheSweepAlertListsOnlyModelsWithAnUpdate() {
        let a = model("org/a"), b = model("org/b"), c = model("org/c")
        let checks: [String: UpdateCheck] = [
            a.id: UpdateCheck(result: .update(PackUpdate(files: 3, bytes: 3_850_000_000)), repo: "org/a"),
            b.id: UpdateCheck(result: .upToDate, repo: "org/b"),
            c.id: UpdateCheck(result: .update(PackUpdate(files: 1, bytes: 1_000, replaces: 1)), repo: "org/c"),
        ]
        let items = PackUpdateCheck.available(checks, models: [c, b, a])
        XCTAssertEqual(items.map(\.model.name), ["org/a", "org/c"])
        let text = PackUpdateCheck.sweepMessage(items)
        XCTAssertTrue(text.contains("org/a") && text.contains("org/c") && !text.contains("org/b"), text)
        XCTAssertTrue(text.contains("replace"), "a replacing update is named before Update All: \(text)")
        XCTAssertTrue(PackUpdateCheck.available([b.id: checks[b.id]!], models: [a, b, c]).isEmpty)
    }
}
