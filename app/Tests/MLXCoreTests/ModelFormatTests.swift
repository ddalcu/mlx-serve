import XCTest
@testable import MLXCore

/// Tests for the Model Browser's weight-format filter (MLX / GGUF / Both) and the
/// GGUF download helpers it drives.
final class ModelFormatTests: XCTestCase {

    // MARK: - Format → HF filter tags

    func testFilterTagsPerFormat() {
        XCTAssertEqual(ModelFormat.mlx.filterTags, ["mlx"])
        XCTAssertEqual(ModelFormat.gguf.filterTags, ["gguf"])
        // Both queries each tag and merges client-side.
        XCTAssertEqual(ModelFormat.both.filterTags, ["mlx", "gguf"])
    }

    func testAllCasesAndLabels() {
        XCTAssertEqual(ModelFormat.allCases, [.mlx, .gguf, .both])
        XCTAssertEqual(ModelFormat.mlx.label, "MLX")
        XCTAssertEqual(ModelFormat.gguf.label, "GGUF")
        XCTAssertEqual(ModelFormat.both.label, "Both")
    }

    // MARK: - Search URL carries the chosen filter

    func testSearchURLUsesTheGivenFilter() throws {
        for tag in ["mlx", "gguf"] {
            let url = try XCTUnwrap(HFSearchService.searchURL(query: "qwen", filter: tag, skip: 0, limit: 50))
            let comps = try XCTUnwrap(URLComponents(url: url, resolvingAgainstBaseURL: false))
            let items = comps.queryItems ?? []
            XCTAssertTrue(items.contains(URLQueryItem(name: "filter", value: tag)), "filter=\(tag) missing")
            XCTAssertTrue(items.contains(URLQueryItem(name: "search", value: "qwen")))
            XCTAssertTrue(items.contains(URLQueryItem(name: "expand[]", value: "tags")))
        }
    }

    func testSearchURLOmitsEmptyQuery() throws {
        let url = try XCTUnwrap(HFSearchService.searchURL(query: "   ", filter: "gguf", skip: 0, limit: 50))
        let comps = try XCTUnwrap(URLComponents(url: url, resolvingAgainstBaseURL: false))
        XCTAssertFalse((comps.queryItems ?? []).contains { $0.name == "search" })
    }

    // MARK: - GGUF repo detection (tags)

    func testIsGgufRepoFromTags() {
        func model(tags: [String]?) -> HFModel {
            HFModel(id: "x/y", downloads: 0, likes: 0, lastModified: nil, tags: tags, safetensors: nil, pipelineTag: nil)
        }
        XCTAssertTrue(model(tags: ["gguf", "qwen2"]).isGgufRepo)
        XCTAssertTrue(model(tags: ["GGUF"]).isGgufRepo) // case-insensitive
        XCTAssertFalse(model(tags: ["qwen2", "text-generation"]).isGgufRepo)
        XCTAssertFalse(model(tags: nil).isGgufRepo)
    }

    // MARK: - Quant label extraction for the picker

    func testQuantLabelExtractsQuantToken() {
        XCTAssertEqual(DownloadManager.quantLabel(forFilename: "Qwen2.5-7B-Instruct-Q4_K_M.gguf"), "Q4_K_M")
        XCTAssertEqual(DownloadManager.quantLabel(forFilename: "model-IQ2_XXS.gguf"), "IQ2_XXS")
        XCTAssertEqual(DownloadManager.quantLabel(forFilename: "Falcon-7B-F16.gguf"), "F16")
        // No recognizable quant token → extension-stripped basename.
        XCTAssertEqual(DownloadManager.quantLabel(forFilename: "weird-model.gguf"), "weird-model")
    }
}
