import XCTest
@testable import MLXCore

final class HFSourceTests: XCTestCase {
    private func entry(_ path: String, _ size: Int) -> [String: Any] { ["path": path, "type": "file", "size": size] }
    private func branch(_ name: String, _ bytes: Int64 = 1) -> HFBranch { HFBranch(name: name, commit: nil, bytes: bytes) }

    func testAnIdAPageAndABranchURLParse() {
        XCTAssertEqual(HFSource.parse("org/repo"), HFSource(repo: "org/repo"))
        XCTAssertEqual(HFSource.parse("org/repo@2.81bpw"), HFSource(repo: "org/repo", revision: "2.81bpw"))
        XCTAssertEqual(HFSource.parse("org/repo@feature/x:Q4_K_M"), HFSource(repo: "org/repo", revision: "feature/x"),
                       "the CLI's quant tag rides after the revision")
        XCTAssertEqual(HFSource.parse("org/repo@8acadd0"), HFSource(repo: "org/repo", revision: "8acadd0"))
        XCTAssertEqual(HFSource.parse(" https://huggingface.co/org/repo/ "), HFSource(repo: "org/repo"))
        XCTAssertEqual(HFSource.parse("https://huggingface.co/org/repo/tree/2.81bpw"), HFSource(repo: "org/repo", revision: "2.81bpw"))
        XCTAssertEqual(HFSource.parse("hf.co/org/repo/tree/feature%2Fx/sub/dir?x=1"), HFSource(repo: "org/repo", revision: "feature/x"))
        XCTAssertEqual(HFSource.parse("https://huggingface.co/org/repo/blob/main/README.md"), HFSource(repo: "org/repo"))
    }

    func testTextThatNamesNoRepoIsASearch() {
        for text in ["", "gemma", "gemma 4/it", "https://example.com/org/repo", "org/", "org/repo@"] {
            XCTAssertNil(HFSource.parse(text), text)
        }
    }

    func testEveryURLCarriesTheEncodedRevision() {
        let branch = HFSource(repo: "org/repo", revision: "feature/x")
        XCTAssertEqual(branch.treeURL.absoluteString, "https://huggingface.co/api/models/org/repo/tree/feature%2Fx?recursive=true")
        XCTAssertEqual(branch.fileURL("sub/model.safetensors").absoluteString,
                       "https://huggingface.co/org/repo/resolve/feature%2Fx/sub/model.safetensors")
        XCTAssertEqual(HFSource(repo: "org/repo").fileURL("README.md").absoluteString,
                       "https://huggingface.co/org/repo/resolve/main/README.md")
        XCTAssertEqual(HFSource.refsURL(repo: "org/repo").absoluteString, "https://huggingface.co/api/models/org/repo/refs")
        XCTAssertEqual(ModelCard.readmeURL(repoId: "org/repo", revision: "2.81bpw")?.absoluteString,
                       "https://huggingface.co/org/repo/resolve/2.81bpw/README.md")
    }

    func testMainKeepsItsFolderAndABranchGetsItsOwn() {
        XCTAssertEqual(HFSource(repo: "org/repo").localRepoId, "org/repo")
        XCTAssertEqual(HFSource(repo: "org/repo", revision: "2.81bpw").localRepoId, "org/repo-2.81bpw")
        XCTAssertEqual(HFSource(repo: "org/repo", revision: "feature/x@y").localRepoId, "org/repo-feature-x-y",
                       "no `/` or `@` reaches a folder name or a model id")
    }

    func testABranchIsPricedByWhatItsDownloadFetches() {
        let entries = [entry("README.md", 9), entry("config.json", 10), entry("model.safetensors", 1000),
                       entry("receipts/run.json", 5), entry("original/model.safetensors", 999)]
        XCTAssertEqual(HFBranch.priced(name: "2bit", commit: "c2", entries: entries), HFBranch(name: "2bit", commit: "c2", bytes: 1010))
        XCTAssertNil(HFBranch.priced(name: "main", commit: nil, entries: [entry("README.md", 9), entry("config.json", 3)]),
                     "a card-only branch holds no model")
    }

    func testRefsParse() {
        let json = #"{"tags":[],"branches":[{"name":"2.9bpw","ref":"refs/heads/2.9bpw","targetCommit":"c1"},{"name":"main","targetCommit":"c0"}]}"#
        let refs = HFBranch.refs(from: Data(json.utf8))
        XCTAssertEqual(refs.map(\.name), ["2.9bpw", "main"])
        XCTAssertEqual(refs.map(\.commit), ["c1", "c0"])
    }

    func testMainLeadsOnlyWhenItHoldsWeights() {
        let all = HFBranch.ordered([branch("2.9bpw"), branch("main"), branch("1.68bpw")])
        XCTAssertEqual(all.map(\.name), ["main", "1.68bpw", "2.9bpw"])
        XCTAssertEqual(HFBranch.preselect(all, requested: nil), "main")
        XCTAssertEqual(HFBranch.preselect(all, requested: "2.9bpw"), "2.9bpw", "a pasted tree URL picks its branch")

        let branchesOnly = HFBranch.ordered([branch("2.9bpw"), branch("1.68bpw")])
        XCTAssertEqual(HFBranch.preselect(branchesOnly, requested: nil), "1.68bpw")
        XCTAssertEqual(HFBranch.preselect(branchesOnly, requested: "gone"), "1.68bpw")
        XCTAssertNil(HFBranch.preselect([], requested: nil))
    }

    func testALoneMainShowsNoPicker() {
        XCTAssertFalse(HFBranch.needsPicker([]))
        XCTAssertFalse(HFBranch.needsPicker([branch("main")]))
        XCTAssertTrue(HFBranch.needsPicker([branch("2bit")]))
        XCTAssertTrue(HFBranch.needsPicker([branch("main"), branch("2bit")]))
    }

    func testAPastedIdIsLookedUpWithTheSearchFields() throws {
        let url = try XCTUnwrap(HFSearchService.lookupURL(repo: "org/repo"))
        XCTAssertEqual(url.path, "/api/models/org/repo")
        let fields = URLComponents(url: url, resolvingAgainstBaseURL: false)?.queryItems?.filter { $0.name == "expand[]" }.map(\.value)
        XCTAssertEqual(fields?.contains("config"), true)
        XCTAssertEqual(fields?.contains("safetensors"), true)
    }
}

/// Read-only against a real branch-only repo: refs, each branch's tree, and one
/// branch's card, never a weight file. `MLX_SERVE_LIVE_BRANCHES=1` runs it.
@MainActor
final class HFBranchLiveTests: XCTestCase {
    private let repoId = "OpensourceWTF/DeepSeek-V4.1-Flash-1017-streaming-repack-exl3"

    func testABranchOnlyRepoListsItsBranchesWithSizes() async throws {
        try XCTSkipUnless(ProcessInfo.processInfo.environment["MLX_SERVE_LIVE_BRANCHES"] == "1")
        let manager = DownloadManager(modelsRoot: NSTemporaryDirectory())
        let branches = await manager.listBranches(repoId: repoId)
        for b in branches { print("[live] \(b.name)  \(MemoryInfo.format(b.bytes))  \(b.commit ?? "-")") }

        XCTAssertEqual(Set(branches.map(\.name)), ["1.68bpw", "2.25bpw", "2.68bpw", "2.81bpw", "2.9bpw"], "main holds only the card")
        XCTAssertTrue(branches.allSatisfy { $0.bytes > 100_000_000_000 }, "each branch is a 100+ GB pack")
        XCTAssertEqual(HFBranch.preselect(branches, requested: HFSource.parse("\(repoId)@2.81bpw")?.revision), "2.81bpw")

        let readme = try XCTUnwrap(ModelCard.readmeURL(repoId: repoId, revision: "2.81bpw"))
        let (data, response) = try await DownloadSession.shared.data(for: DownloadManager.hfApiRequest(readme))
        XCTAssertEqual((response as? HTTPURLResponse)?.statusCode, 200)
        print("[live] 2.81bpw README: \(data.count) bytes")
    }
}
