import XCTest
@testable import MLXCore

/// Bar: a row says "In use" only for a model the running server holds, whoever loaded it.
final class ModelUseStateTests: XCTestCase {
    private let a = "/Users/x/.lmstudio/models/org/a"
    private let b = "/Users/x/.lmstudio/models/org/b"

    private func row(_ name: String, loaded: Bool, state: String? = nil, lanPeer: String? = nil) -> ModelInfo {
        ModelInfo(name: name, quantBits: 4, layers: 1, hiddenSize: 1, vocabSize: 1, contextLength: 1,
                  modelMaxTokens: 1, loaded: loaded, state: state ?? (loaded ? "ready" : "unloaded"), lanPeer: lanPeer)
    }

    func testAModelAnotherClientSwitchedAwayFromIsNotInUse() {
        let rows = [row("org/a", loaded: false), row("org/b", loaded: true)]
        XCTAssertEqual(ModelUseState.resolve(path: a, selectedPath: a, serverStatus: .running, rows: rows), .idle)
        XCTAssertEqual(ModelUseState.resolve(path: b, selectedPath: a, serverStatus: .running, rows: rows), .inUse)
    }

    func testEveryResidentModelIsInUse() {
        let rows = [row("org/a", loaded: true), row("org/b", loaded: true)]
        XCTAssertEqual(ModelUseState.resolve(path: a, selectedPath: a, serverStatus: .running, rows: rows), .inUse)
        XCTAssertEqual(ModelUseState.resolve(path: b, selectedPath: a, serverStatus: .running, rows: rows), .inUse)
    }

    func testAModelTheServerIsLoadingSaysLoading() {
        let rows = [row("org/a", loaded: false, state: "loading")]
        XCTAssertEqual(ModelUseState.resolve(path: a, selectedPath: a, serverStatus: .running, rows: rows), .loading)
    }

    func testAPeerOnTheLanDoesNotMakeALocalModelInUse() {
        let rows = [row("org/a", loaded: false), row("org/a", loaded: true, lanPeer: "peer")]
        XCTAssertEqual(ModelUseState.resolve(path: a, selectedPath: a, serverStatus: .running, rows: rows), .idle)
    }

    func testAGgufQuantReadsItsRepoFoldersRow() {
        let q4 = "/m/Qwen/Qwen2.5-0.5B-Instruct-GGUF/qwen2.5-0.5b-instruct-q4_k_m.gguf"
        let q8 = "/m/Qwen/Qwen2.5-0.5B-Instruct-GGUF/Q8_0/qwen2.5-0.5b-instruct-q8_0.gguf"
        let loaded = [row("Qwen/Qwen2.5-0.5B-Instruct-GGUF", loaded: true)]
        XCTAssertEqual(ModelUseState.resolve(path: q8, selectedPath: a, serverStatus: .running, rows: loaded), .inUse)
        XCTAssertEqual(ModelUseState.resolve(path: q4, selectedPath: q8, serverStatus: .running, rows: loaded), .idle,
                       "the app's pick is the quant the folder's row loaded")
        XCTAssertEqual(ModelUseState.resolve(path: q8, selectedPath: q8, serverStatus: .running, rows: loaded), .inUse)
        let evicted = [row("Qwen/Qwen2.5-0.5B-Instruct-GGUF", loaded: false)]
        XCTAssertEqual(ModelUseState.resolve(path: q8, selectedPath: q8, serverStatus: .running, rows: evicted), .idle)
    }

    func testAGgufMatchesOnlyItsRepoFolderNotAFolderFurtherUp() {
        let quant = "/Volumes/org/a/models/other/repo/x-q4_k_m.gguf"
        let rows = [row("org/a", loaded: true), row("other/repo", loaded: false)]
        XCTAssertEqual(ModelUseState.resolve(path: quant, selectedPath: b, serverStatus: .running, rows: rows), .idle)
        let noRepoRow = [row("org/a", loaded: true)]
        XCTAssertEqual(ModelUseState.resolve(path: quant, selectedPath: b, serverStatus: .running, rows: noRepoRow), .idle)
    }

    func testWithoutAServerRowTheSelectionDecides() {
        XCTAssertEqual(ModelUseState.resolve(path: a, selectedPath: a, serverStatus: .running, rows: []), .inUse)
        XCTAssertEqual(ModelUseState.resolve(path: b, selectedPath: a, serverStatus: .running, rows: []), .idle)
        XCTAssertEqual(ModelUseState.resolve(path: a, selectedPath: a, serverStatus: .starting, rows: []), .loading)
        XCTAssertEqual(ModelUseState.resolve(path: a, selectedPath: a, serverStatus: .stopped, rows: []), .selected)
    }

    func testAStoppedServerIgnoresItsLastRows() {
        let rows = [row("org/b", loaded: true)]
        XCTAssertEqual(ModelUseState.resolve(path: b, selectedPath: a, serverStatus: .stopped, rows: rows), .idle)
        XCTAssertEqual(ModelUseState.resolve(path: a, selectedPath: a, serverStatus: .stopped, rows: rows), .selected)
    }
}
