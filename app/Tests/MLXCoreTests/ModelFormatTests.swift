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

    // MARK: - HF "Unsupported architecture" regression

    /// LM Studio community GGUF repacks frequently tag themselves only with
    /// `gguf` / `llama-cpp` / `base_model:...` — they don't inherit the
    /// upstream architecture-family tag. Pre-fix, the Model Browser flagged
    /// `lmstudio-community/gemma-4-E4B-it-GGUF` as "Unsupported architecture"
    /// because no tag matched `gemma*`. The embedded llama.cpp engine
    /// handles whatever architecture the .gguf declares, so this is a false
    /// negative we must NOT regress on.
    func testGgufRepoWithoutFamilyTagIsStillSupported() {
        func model(id: String, tags: [String]?) -> HFModel {
            HFModel(id: id, downloads: 0, likes: 0, lastModified: nil, tags: tags, safetensors: nil, pipelineTag: nil)
        }
        // LMS-community GGUF repacks — no `gemma*` tag.
        XCTAssertTrue(model(
            id: "lmstudio-community/gemma-4-E4B-it-GGUF",
            tags: ["gguf", "llama-cpp", "base_model:google/gemma-4-E4B-it"]
        ).isSupportedArchitecture)
        XCTAssertTrue(model(
            id: "lmstudio-community/gemma-4-E2B-it-GGUF",
            tags: ["gguf"]
        ).isSupportedArchitecture)
        // Bartowski-style GGUF repack — same shape.
        XCTAssertTrue(model(
            id: "bartowski/Qwen2.5-7B-Instruct-GGUF",
            tags: ["gguf", "llama-cpp", "imatrix"]
        ).isSupportedArchitecture)

        // Sanity: non-GGUF with an unknown tag set should still flag as
        // unsupported — the GGUF permissive pass only fires on .gguf repos.
        XCTAssertFalse(model(
            id: "someone/random-arch",
            tags: ["text-generation", "totally-unknown-arch"]
        ).isSupportedArchitecture)

        // Non-GGUF with a known family tag is supported as before.
        XCTAssertTrue(model(
            id: "google/gemma-4-E4B-it",
            tags: ["gemma4", "text-generation"]
        ).isSupportedArchitecture)
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
