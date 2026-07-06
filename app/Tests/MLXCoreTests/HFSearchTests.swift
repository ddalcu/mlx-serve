import XCTest
import Foundation
@testable import MLXCore

// =============================================================================
// MARK: - Type replicas for testing (matches HFModels.swift)
// =============================================================================

private struct TestHFSafetensors: Codable {
    let parameters: [String: Int64]?
    let total: Int64?
}

private let testCompatiblePipelineTags: Set<String> = [
    "text-generation", "image-text-to-text", "any-to-any",
]

private struct TestHFModel: Identifiable, Codable {
    let id: String
    let downloads: Int?
    let likes: Int?
    let lastModified: String?
    let tags: [String]?
    let safetensors: TestHFSafetensors?
    let pipelineTag: String?

    enum CodingKeys: String, CodingKey {
        case id, downloads, likes, lastModified, tags, safetensors
        case pipelineTag = "pipeline_tag"
    }

    var isCompatible: Bool {
        guard let tag = pipelineTag, !tag.isEmpty else { return true }
        return testCompatiblePipelineTags.contains(tag)
    }

    var hasVision: Bool {
        let tag = pipelineTag ?? ""
        return tag == "image-text-to-text" || tag == "any-to-any"
    }

    var hasToolCalling: Bool {
        let lower = id.lowercased()
        let isInstructTuned = lower.contains("-it") || lower.contains("-instruct") || lower.contains("-chat")
        guard isInstructTuned else { return false }
        let toolFamilies = ["gemma-4", "gemma-3", "qwen3", "qwen2.5", "llama-3", "mistral"]
        return toolFamilies.contains { lower.contains($0) }
    }

    var author: String {
        id.split(separator: "/").first.map(String.init) ?? ""
    }
    var modelName: String {
        id.split(separator: "/").last.map(String.init) ?? id
    }
    // Delegates to the real parser so this replica can't drift from it.
    var quantization: String? { HFModel.quantizationLabel(forId: id) }
    var estimatedSizeBytes: Int64 {
        guard let params = safetensors?.parameters else { return 0 }
        var total: Int64 = 0
        for (dtype, count) in params {
            let bytesPerParam: Double
            switch dtype.uppercased() {
            case "F64": bytesPerParam = 8
            case "F32", "U32", "I32": bytesPerParam = 4
            case "F16", "BF16", "U16", "I16": bytesPerParam = 2
            case "I8", "U8": bytesPerParam = 1
            case let d where d.contains("4"): bytesPerParam = 0.5
            default: bytesPerParam = 2
            }
            total += Int64(Double(count) * bytesPerParam)
        }
        return total
    }
    var modelSize: String {
        let name = modelName
        let pattern = #"(?:^|[-_])[Ee]?(\d+(?:\.\d+)?[BbMm])(?![Ii])"#
        guard let regex = try? NSRegularExpression(pattern: pattern),
              let match = regex.firstMatch(in: name, range: NSRange(name.startIndex..., in: name)),
              let range = Range(match.range(at: 1), in: name) else {
            return "\u{2014}"
        }
        return String(name[range]).uppercased()
    }
    var lastModifiedDate: Date? {
        guard let lastModified else { return nil }
        let formatter = ISO8601DateFormatter()
        formatter.formatOptions = [.withInternetDateTime, .withFractionalSeconds]
        return formatter.date(from: lastModified) ?? ISO8601DateFormatter().date(from: lastModified)
    }
}

// =============================================================================
// MARK: - Unit tests
// =============================================================================

final class HFModelTests: XCTestCase {

    func testQuantizationParsing() {
        XCTAssertEqual(TestHFModel.make(id: "x/model-4bit").quantization, "4-bit")
        XCTAssertEqual(TestHFModel.make(id: "x/model-8bit").quantization, "8-bit")
        XCTAssertEqual(TestHFModel.make(id: "x/model-bf16").quantization, "BF16")
        XCTAssertEqual(TestHFModel.make(id: "x/model-fp16").quantization, "FP16")
        XCTAssertEqual(TestHFModel.make(id: "x/model-3bit").quantization, "3-bit")
        XCTAssertNil(TestHFModel.make(id: "x/plain-model").quantization)
    }

    func testAuthorAndModelName() {
        let m = TestHFModel.make(id: "mlx-community/gemma-4-e2b-it-4bit")
        XCTAssertEqual(m.author, "mlx-community")
        XCTAssertEqual(m.modelName, "gemma-4-e2b-it-4bit")
    }

    func testSizeEstimation_BF16AndU32() {
        let m = TestHFModel.make(
            id: "test/model",
            safetensors: TestHFSafetensors(
                parameters: ["BF16": 631_148_099, "U32": 579_616_768],
                total: 1_210_764_867
            )
        )
        // BF16: 631M * 2 = 1.26 GB, U32: 579M * 4 = 2.32 GB → ~3.58 GB
        let sizeGB = Double(m.estimatedSizeBytes) / (1024 * 1024 * 1024)
        XCTAssertGreaterThan(sizeGB, 3.0)
        XCTAssertLessThan(sizeGB, 4.0)
    }

    func testSizeEstimation_NoSafetensors() {
        let m = TestHFModel.make(id: "test/model")
        XCTAssertEqual(m.estimatedSizeBytes, 0)
    }

    func testDateParsing() {
        let m = TestHFModel.make(id: "x/m", lastModified: "2026-04-13T13:07:28.000Z")
        XCTAssertNotNil(m.lastModifiedDate)
        let m2 = TestHFModel.make(id: "x/m")
        XCTAssertNil(m2.lastModifiedDate)
    }

    func testDecodeRealAPIShape() throws {
        let json = """
        [
            {
                "_id": "69cea456",
                "id": "mlx-community/gemma-4-e2b-it-4bit",
                "lastModified": "2026-04-13T13:07:28.000Z",
                "downloads": 132195,
                "likes": 7,
                "tags": ["mlx", "safetensors"],
                "safetensors": {
                    "parameters": {"BF16": 631148099, "U32": 579616768},
                    "total": 1210764867
                }
            },
            {
                "_id": "abc",
                "id": "mlx-community/minimal-model"
            }
        ]
        """
        let models = try JSONDecoder().decode([TestHFModel].self, from: json.data(using: .utf8)!)
        XCTAssertEqual(models.count, 2)
        XCTAssertEqual(models[0].downloads, 132195)
        XCTAssertEqual(models[0].quantization, "4-bit")
        XCTAssertGreaterThan(models[0].estimatedSizeBytes, 0)
        XCTAssertNil(models[1].downloads)
        XCTAssertEqual(models[1].estimatedSizeBytes, 0)
    }

    func testClientSideSort_Downloads() {
        let models = [
            TestHFModel.make(id: "x/a", downloads: 100),
            TestHFModel.make(id: "x/b", downloads: 500),
            TestHFModel.make(id: "x/c", downloads: 50),
        ]
        let descending = models.sorted { ($0.downloads ?? 0) > ($1.downloads ?? 0) }
        XCTAssertEqual(descending.map(\.id), ["x/b", "x/a", "x/c"])
        let ascending = models.sorted { ($0.downloads ?? 0) < ($1.downloads ?? 0) }
        XCTAssertEqual(ascending.map(\.id), ["x/c", "x/a", "x/b"])
    }

    func testCompatibility_TextGeneration() {
        XCTAssertTrue(TestHFModel.make(id: "x/m", pipelineTag: "text-generation").isCompatible)
        XCTAssertTrue(TestHFModel.make(id: "x/m", pipelineTag: "image-text-to-text").isCompatible)
        XCTAssertTrue(TestHFModel.make(id: "x/m", pipelineTag: "any-to-any").isCompatible)
    }

    func testCompatibility_NilPipelineIsCompatible() {
        XCTAssertTrue(TestHFModel.make(id: "x/m").isCompatible)
        XCTAssertTrue(TestHFModel.make(id: "x/m", pipelineTag: "").isCompatible)
    }

    func testVisionCapability() {
        XCTAssertTrue(TestHFModel.make(id: "x/m", pipelineTag: "image-text-to-text").hasVision)
        XCTAssertTrue(TestHFModel.make(id: "x/m", pipelineTag: "any-to-any").hasVision)
        XCTAssertFalse(TestHFModel.make(id: "x/m", pipelineTag: "text-generation").hasVision)
        XCTAssertFalse(TestHFModel.make(id: "x/m").hasVision)
    }

    func testToolCallingCapability() {
        // Instruction-tuned from known families → has tool calling
        XCTAssertTrue(TestHFModel.make(id: "mlx-community/gemma-4-e2b-it-4bit").hasToolCalling)
        XCTAssertTrue(TestHFModel.make(id: "mlx-community/qwen3-8b-instruct").hasToolCalling)
        XCTAssertTrue(TestHFModel.make(id: "mlx-community/llama-3-8b-instruct").hasToolCalling)
        // Not instruction-tuned → no tool calling
        XCTAssertFalse(TestHFModel.make(id: "mlx-community/gemma-4-e2b-4bit").hasToolCalling)
        // Unknown family → no tool calling
        XCTAssertFalse(TestHFModel.make(id: "mlx-community/custom-model-it").hasToolCalling)
    }

    func testCompatibility_UnsupportedPipelines() {
        XCTAssertFalse(TestHFModel.make(id: "x/m", pipelineTag: "text-to-speech").isCompatible)
        XCTAssertFalse(TestHFModel.make(id: "x/m", pipelineTag: "automatic-speech-recognition").isCompatible)
        XCTAssertFalse(TestHFModel.make(id: "x/m", pipelineTag: "image-classification").isCompatible)
    }

    func testClientSideSort_EstimatedSize() {
        let small = TestHFModel.make(id: "x/small", safetensors: TestHFSafetensors(parameters: ["BF16": 1_000_000], total: 1_000_000))
        let large = TestHFModel.make(id: "x/large", safetensors: TestHFSafetensors(parameters: ["BF16": 1_000_000_000], total: 1_000_000_000))
        let none = TestHFModel.make(id: "x/none")
        let sorted = [small, large, none].sorted { $0.estimatedSizeBytes > $1.estimatedSizeBytes }
        XCTAssertEqual(sorted.map(\.id), ["x/large", "x/small", "x/none"])
    }

    // MARK: - Model size parsing from name

    func testModelSize_CommonPatterns() {
        XCTAssertEqual(TestHFModel.make(id: "x/gemma-4-31b-it-4bit").modelSize, "31B")
        XCTAssertEqual(TestHFModel.make(id: "x/gemma-4-e2b-it-4bit").modelSize, "2B")
        XCTAssertEqual(TestHFModel.make(id: "x/gemma-4-e4b-it-4bit").modelSize, "4B")
        XCTAssertEqual(TestHFModel.make(id: "x/gemma-4-26b-a4b-it-4bit").modelSize, "26B")
        XCTAssertEqual(TestHFModel.make(id: "x/Kokoro-82M-bf16").modelSize, "82M")
        XCTAssertEqual(TestHFModel.make(id: "x/parakeet-tdt-0.6b-v2").modelSize, "0.6B")
        XCTAssertEqual(TestHFModel.make(id: "x/LFM2-24B-A2B-MLX-4bit").modelSize, "24B")
        XCTAssertEqual(TestHFModel.make(id: "x/DeepSeek-R1-0528-Qwen3-8B-MLX-4bit").modelSize, "8B")
        XCTAssertEqual(TestHFModel.make(id: "x/LFM2.5-1.2B-Instruct-MLX-8bit").modelSize, "1.2B")
    }

    func testModelSize_DoesNotMatchQuantBits() {
        // "8bit" should NOT be parsed as "8B"
        XCTAssertEqual(TestHFModel.make(id: "x/Qwen3-Coder-Next-8bit").modelSize, "\u{2014}")
        XCTAssertEqual(TestHFModel.make(id: "x/GLM-4.7-Flash-MLX-8bit").modelSize, "\u{2014}")
    }

    func testModelSize_NoMatch() {
        XCTAssertEqual(TestHFModel.make(id: "x/Kimi-K2.5").modelSize, "\u{2014}")
    }
}

// =============================================================================
// MARK: - Integration tests (hits real HuggingFace API)
// =============================================================================

final class HFSearchIntegrationTests: XCTestCase {

    private func buildURL(search: String? = nil, limit: Int = 5) -> URL {
        var components = URLComponents(string: "https://huggingface.co/api/models")!
        var items: [URLQueryItem] = [
            URLQueryItem(name: "filter", value: "mlx"),
            URLQueryItem(name: "sort", value: "downloads"),
            URLQueryItem(name: "direction", value: "-1"),
            URLQueryItem(name: "limit", value: "\(limit)"),
        ]
        if let search {
            items.append(URLQueryItem(name: "search", value: search))
        }
        for field in ["safetensors", "lastModified", "likes", "downloads", "tags"] {
            items.append(URLQueryItem(name: "expand[]", value: field))
        }
        components.queryItems = items
        return components.url!
    }

    func testFetchMLXModels_ReturnsResults() async throws {
        let url = buildURL()
        let (data, response) = try await URLSession.shared.data(from: url)
        let http = response as! HTTPURLResponse
        XCTAssertEqual(http.statusCode, 200)

        let models = try JSONDecoder().decode([TestHFModel].self, from: data)
        XCTAssertGreaterThanOrEqual(models.count, 1, "Should find at least 1 MLX model")
        XCTAssertFalse(models[0].id.isEmpty)
        XCTAssertNotNil(models[0].downloads, "expand[]=downloads should populate field")
    }

    func testSearchGemma_ReturnsGemmaModels() async throws {
        let url = buildURL(search: "gemma")
        let (data, _) = try await URLSession.shared.data(from: url)
        let models = try JSONDecoder().decode([TestHFModel].self, from: data)
        XCTAssertGreaterThanOrEqual(models.count, 1, "Searching 'gemma' should find MLX models")
        for m in models {
            XCTAssertTrue(m.id.lowercased().contains("gemma"), "\(m.id) should contain 'gemma'")
        }
    }

    func testSafetensorsExpand_PopulatesSizeData() async throws {
        let url = buildURL(search: "gemma-4-e2b-it-4bit", limit: 10)
        let (data, _) = try await URLSession.shared.data(from: url)
        let models = try JSONDecoder().decode([TestHFModel].self, from: data)

        if let target = models.first(where: { $0.id == "mlx-community/gemma-4-e2b-it-4bit" }) {
            XCTAssertNotNil(target.safetensors, "Known model should have safetensors metadata")
            let sizeGB = Double(target.estimatedSizeBytes) / (1024 * 1024 * 1024)
            XCTAssertGreaterThan(sizeGB, 2.0, "gemma-4-e2b-it-4bit should be > 2 GB")
            XCTAssertLessThan(sizeGB, 5.0, "gemma-4-e2b-it-4bit should be < 5 GB")
        }
    }

    func testPagination_SkipWorks() async throws {
        // Fetch page 1
        let url1 = buildURL(limit: 3)
        let (data1, _) = try await URLSession.shared.data(from: url1)
        let page1 = try JSONDecoder().decode([TestHFModel].self, from: data1)

        // Fetch page 2 with skip=3
        var components = URLComponents(string: "https://huggingface.co/api/models")!
        components.queryItems = [
            URLQueryItem(name: "filter", value: "mlx"),
            URLQueryItem(name: "sort", value: "downloads"),
            URLQueryItem(name: "direction", value: "-1"),
            URLQueryItem(name: "limit", value: "3"),
            URLQueryItem(name: "skip", value: "3"),
            URLQueryItem(name: "expand[]", value: "downloads"),
        ]
        let (data2, _) = try await URLSession.shared.data(from: components.url!)
        let page2 = try JSONDecoder().decode([TestHFModel].self, from: data2)

        XCTAssertGreaterThanOrEqual(page1.count, 1)
        XCTAssertGreaterThanOrEqual(page2.count, 1)
        // Pages should not overlap
        let page1Ids = Set(page1.map(\.id))
        let page2Ids = Set(page2.map(\.id))
        XCTAssertTrue(page1Ids.isDisjoint(with: page2Ids), "Page 1 and 2 should have different models")
    }
}

// MARK: - Test helper

private extension TestHFModel {
    static func make(
        id: String,
        downloads: Int? = nil,
        likes: Int? = nil,
        lastModified: String? = nil,
        safetensors: TestHFSafetensors? = nil,
        pipelineTag: String? = nil
    ) -> TestHFModel {
        TestHFModel(id: id, downloads: downloads, likes: likes, lastModified: lastModified, tags: nil, safetensors: safetensors, pipelineTag: pipelineTag)
    }
}

// =============================================================================
// MARK: - HF tree-API fallback-size parsing
//
// Reproduces the bug where GGUF repos showed "Unknown" in the Model Browser's
// RAM Est column: the original `fetchFallbackSizes` only summed `.safetensors`
// files, so any GGUF-only repo fell through with `fallbackSizeBytes = nil`
// and `estimatedSizeBytes = 0`. These tests pin the new `parseFallbackSize`
// path that picks up GGUF quants and returns a min/max range when the repo
// ships more than one. Uses `@testable import MLXCore` so the real function
// is under test rather than a replica.
// =============================================================================

final class HFFallbackSizeTests: XCTestCase {
    private typealias Entry = HFSearchService.TreeFileEntry

    func testSafetensorsSum_winsWhenPresent() {
        let files: [Entry] = [
            .init(path: "config.json", size: 1_000),
            .init(path: "model-00001-of-00002.safetensors", size: 2_000_000_000),
            .init(path: "model-00002-of-00002.safetensors", size: 3_000_000_000),
            .init(path: "model.gguf", size: 4_000_000_000),  // also present — must lose
        ]
        XCTAssertEqual(HFSearchService.parseFallbackSize(files: files), .safetensorsSum(5_000_000_000))
    }

    func testGgufRange_acrossMultipleQuants() {
        let files: [Entry] = [
            .init(path: "README.md", size: 5_000),
            .init(path: "gemma-4-E4B-Q2_K.gguf", size: 1_700_000_000),
            .init(path: "gemma-4-E4B-Q4_K_M.gguf", size: 2_600_000_000),
            .init(path: "gemma-4-E4B-Q5_K_M.gguf", size: 3_100_000_000),
            .init(path: "gemma-4-E4B-Q8_0.gguf", size: 4_500_000_000),
            .init(path: "gemma-4-E4B-BF16.gguf", size: 8_500_000_000),
        ]
        XCTAssertEqual(
            HFSearchService.parseFallbackSize(files: files),
            .ggufRange(min: 1_700_000_000, max: 8_500_000_000)
        )
    }

    func testGgufSingle_collapsesToSingleVariant() {
        let files: [Entry] = [
            .init(path: "config.json", size: 1_000),
            .init(path: "Qwen3.5-0.8B-Q4_K_M.gguf", size: 500_000_000),
        ]
        XCTAssertEqual(HFSearchService.parseFallbackSize(files: files), .ggufSingle(500_000_000))
    }

    func testMmprojSidecar_isExcluded() {
        // Gemma 4 VL / Qwen 3.6 VL repos ship a `mmproj-*.gguf` next to the
        // LLM quant. The sidecar is a CLIP vision encoder, not a loadable
        // LLM — must NOT enter the range.
        let files: [Entry] = [
            .init(path: "gemma-4-E4B-Q4_K_M.gguf", size: 2_600_000_000),
            .init(path: "gemma-4-E4B-BF16.gguf", size: 8_500_000_000),
            .init(path: "mmproj-gemma-4-E4B-F16.gguf", size: 200_000_000),
        ]
        XCTAssertEqual(
            HFSearchService.parseFallbackSize(files: files),
            .ggufRange(min: 2_600_000_000, max: 8_500_000_000)
        )
    }

    func testTinyFiles_areExcluded() {
        // LFS pointer stubs occasionally show up with non-zero but absurdly-small
        // sizes — those would skew the min downward if counted.
        let files: [Entry] = [
            .init(path: "tiny-pointer.gguf", size: 500),                    // < 1 MB
            .init(path: "real-Q4_K_M.gguf", size: 2_600_000_000),
            .init(path: "real-BF16.gguf", size: 8_500_000_000),
        ]
        XCTAssertEqual(
            HFSearchService.parseFallbackSize(files: files),
            .ggufRange(min: 2_600_000_000, max: 8_500_000_000)
        )
    }

    func testSubdirGguf_isExcluded() {
        // Split-shard layouts (`shard/part-001.gguf`) can't be reassembled by
        // the single-file download path — don't include them in the row's
        // displayed range, which advertises top-level pickable files.
        let files: [Entry] = [
            .init(path: "shard/part-001.gguf", size: 5_000_000_000),
            .init(path: "shard/part-002.gguf", size: 5_000_000_000),
            .init(path: "Q4_K_M.gguf", size: 2_600_000_000),
        ]
        XCTAssertEqual(HFSearchService.parseFallbackSize(files: files), .ggufSingle(2_600_000_000))
    }

    func testEmptyOrNoLLMArtifacts_returnsNil() {
        XCTAssertNil(HFSearchService.parseFallbackSize(files: []))
        let onlyDocs: [Entry] = [
            .init(path: "README.md", size: 5_000),
            .init(path: "LICENSE", size: 1_000),
        ]
        XCTAssertNil(HFSearchService.parseFallbackSize(files: onlyDocs))
    }

    func testTreeEntries_filtersNonFilesAndUnreadableSizes() {
        let raw: [[String: Any]] = [
            ["path": "Q4.gguf", "type": "file", "size": 2_600_000_000],
            ["path": "subdir", "type": "directory", "size": 0],       // not a file
            ["path": "Q8.gguf", "type": "file"],                       // no size
            ["path": "Q5.gguf", "type": "file", "size": 3_100_000_000],
        ]
        let entries = HFSearchService.treeEntries(from: raw)
        XCTAssertEqual(entries.count, 2)
        XCTAssertEqual(Set(entries.map(\.path)), ["Q4.gguf", "Q5.gguf"])
    }
}

// MARK: - HFSearchService.needsFallbackFetch — the gate that re-broke GGUF
//
// Regression for the bug that surfaced after the parseFallbackSize work:
// fetchFallbackSizes' caller hard-coded `!$0.isGgufRepo`, so GGUF rows
// never reached the new branch and the column kept rendering "Unknown".
// Pin the gate's contract: GGUF rows with no estimated size MUST be
// fetched. If anyone re-adds an `isGgufRepo` exclusion here, this test
// goes red and points them at the right code path.

final class HFFallbackFetchGateTests: XCTestCase {
    private func gguf(id: String) -> HFModel {
        // GGUF repo shape as the HF API returns it: `tags: ["gguf"]`, no
        // `safetensors` block, no parameters → estimatedSizeBytes == 0.
        HFModel(id: id, downloads: 100, likes: 10, lastModified: nil,
                tags: ["gguf"], safetensors: nil, pipelineTag: "text-generation")
    }

    private func mlxNoSafetensors(id: String) -> HFModel {
        HFModel(id: id, downloads: 100, likes: 10, lastModified: nil,
                tags: ["mlx"], safetensors: nil, pipelineTag: "text-generation")
    }

    private func mlxWithParams(id: String) -> HFModel {
        let st = HFSafetensors(parameters: ["BF16": 7_000_000_000], total: nil)
        return HFModel(id: id, downloads: 100, likes: 10, lastModified: nil,
                       tags: ["mlx"], safetensors: st, pipelineTag: "text-generation")
    }

    func testGgufRow_isFetched() {
        // The original bug: this returned false because of `!$0.isGgufRepo`,
        // leaving every GGUF row in the Model Browser stuck at "Unknown".
        XCTAssertTrue(HFSearchService.needsFallbackFetch(gguf(id: "unsloth/gemma-4-E4B-it-GGUF")))
    }

    func testMlxRow_withoutMetadata_isFetched() {
        XCTAssertTrue(HFSearchService.needsFallbackFetch(mlxNoSafetensors(id: "mlx-community/some-7B")))
    }

    func testMlxRow_withParameters_isNotRefetched() {
        XCTAssertFalse(HFSearchService.needsFallbackFetch(mlxWithParams(id: "mlx-community/has-params")))
    }

    func testIncompatibleRow_isSkipped() {
        // `text-to-image` and similar pipeline tags aren't loadable by
        // mlx-serve — burning a tree-API request on them just wastes a
        // round trip.
        let m = HFModel(id: "x/diffusion", downloads: nil, likes: nil,
                        lastModified: nil, tags: ["diffusers"], safetensors: nil,
                        pipelineTag: "text-to-image")
        XCTAssertFalse(HFSearchService.needsFallbackFetch(m))
    }
}

// MARK: - HFModel non-affine quantization gate
//
// Since v26.6.10 (issue #24) the server LOADS nvfp4/mxfp4/mxfp8 safetensors
// checkpoints (per-weight quant-mode resolution in transformer.zig), so the
// Model Browser must offer them — the old client-side "Unsupported
// quantization" gate mirrored a server discovery gate that no longer exists.
// Only formats the server still rejects (model.zig parse: anything outside
// affine/nvfp4/mxfp4/mxfp8, e.g. mxfp6) stay flagged. GGUF repos were always
// exempt: llama.cpp loads mxfp4 (GPT-OSS) natively.

final class HFModelQuantGateTests: XCTestCase {
    private func mlx(id: String, tags: [String]? = nil, pipeline: String? = "text-generation") -> HFModel {
        HFModel(id: id, downloads: 100, likes: 1, lastModified: nil,
                tags: tags, safetensors: nil, pipelineTag: pipeline)
    }

    func testEncoderEmbeddingRepoIsCompatibleAndSupported() {
        // Encoder-only embedding repos (BERT family) are served via
        // /v1/embeddings and power GPU folder indexing — the browser must
        // offer them. Real shape of mlx-community/bge-small-en-v1.5-8bit:
        // pipeline "feature-extraction", tags include "bert".
        let m = HFModel(id: "mlx-community/bge-small-en-v1.5-8bit",
                        downloads: 186, likes: 1, lastModified: nil,
                        tags: ["sentence-transformers", "safetensors", "bert",
                               "feature-extraction", "sentence-similarity", "mlx"],
                        safetensors: nil, pipelineTag: "feature-extraction")
        XCTAssertTrue(m.isCompatible, "feature-extraction pipeline must be browsable")
        XCTAssertTrue(m.isSupportedArchitecture, "bert tag is a supported family")
        XCTAssertNil(m.incompatibleReason)
        // sentence-similarity is the other pipeline tag embedding repos use.
        let m2 = HFModel(id: "mlx-community/all-MiniLM-L6-v2-8bit",
                         downloads: 81, likes: 1, lastModified: nil,
                         tags: ["bert", "mlx"], safetensors: nil,
                         pipelineTag: "sentence-similarity")
        XCTAssertTrue(m2.isCompatible)
        XCTAssertTrue(m2.isSupportedArchitecture)
    }

    func testLocalBertModelIsSupportedModelType() {
        XCTAssertTrue(supportedModelTypes.contains("bert"),
                      "local encoder checkpoints (config.json model_type=bert) must not be flagged unsupported")
    }

    func testDiffusionGemmaIsSupportedModelType() {
        // DiffusionGemma (block diffusion, config.json model_type=diffusion_gemma)
        // is served by the canvas-denoising loop since v26.6.10 — the browser
        // must not flag mlx-community/diffusiongemma-* as unsupported.
        XCTAssertTrue(supportedModelTypes.contains("diffusion_gemma"),
                      "diffusion_gemma checkpoints are served (src/diffusion.zig) and must not be flagged unsupported")
    }

    func testNvfp4_servedSinceV26_6_10_notFlagged() {
        let m = mlx(id: "mlx-community/Qwen3-30B-A3B-nvfp4", tags: ["mlx", "qwen3"])
        XCTAssertNil(m.unsupportedQuantization)
        XCTAssertNil(m.incompatibleReason)
        XCTAssertEqual(m.quantization, "NVFP4")
    }

    func testMxfp4Mxfp8_servedSinceV26_6_10_notFlagged() {
        XCTAssertNil(mlx(id: "x/model-mxfp4").unsupportedQuantization)
        XCTAssertNil(mlx(id: "x/model-MXFP8-it").unsupportedQuantization)
        XCTAssertEqual(mlx(id: "x/model-mxfp4").quantization, "MXFP4")
        XCTAssertEqual(mlx(id: "x/model-MXFP8-it").quantization, "MXFP8")
    }

    func testMxfp6_stillUnsupported_flagged() {
        // model.zig rejects quantization.mode outside {affine, nvfp4, mxfp4,
        // mxfp8} — mxfp6 repos must keep the un-downloadable flag.
        let m = mlx(id: "x/model-mxfp6")
        XCTAssertEqual(m.unsupportedQuantization, "MXFP6")
        XCTAssertEqual(m.incompatibleReason, "Unsupported quantization (MXFP6)")
    }

    func testAffineQuant_notFlagged() {
        XCTAssertNil(mlx(id: "mlx-community/gemma-4-e2b-it-4bit").unsupportedQuantization)
        XCTAssertNil(mlx(id: "x/model-8bit").unsupportedQuantization)
        XCTAssertNil(mlx(id: "x/model-bf16").incompatibleReason)
    }

    func testGgufMxfp_notFlagged() {
        // GPT-OSS-style GGUF mxfp4 is served by the embedded llama.cpp engine —
        // must NOT be flagged, or we'd hide a loadable download.
        let g = HFModel(id: "lmstudio-community/gpt-oss-20b-MXFP4-GGUF", downloads: 1, likes: 1,
                        lastModified: nil, tags: ["gguf"], safetensors: nil, pipelineTag: "text-generation")
        XCTAssertNil(g.unsupportedQuantization)
        XCTAssertNil(g.incompatibleReason)
    }

    func testArchitectureReasonTakesPrecedence() {
        // An unsupported architecture is the more fundamental blocker — it wins
        // the surfaced reason even when the name also carries an mxfp6 marker.
        let m = HFModel(id: "x/some-diffusion-mxfp6", downloads: 1, likes: 1, lastModified: nil,
                        tags: ["diffusers"], safetensors: nil, pipelineTag: nil)
        XCTAssertEqual(m.incompatibleReason, "Unsupported architecture")
    }
}

// MARK: - HFModel.quantization label parsing
//
// The badge column hardcoded only {3,4,6,8}-bit, so 2/5/9-bit MLX repos and
// GGUF qN_/iqN_ ids showed no quant badge at all. These pin the generalized
// width parsing against the real HFModel.

final class HFModelQuantizationLabelTests: XCTestCase {
    private func m(_ id: String) -> HFModel {
        HFModel(id: id, downloads: nil, likes: nil, lastModified: nil,
                tags: nil, safetensors: nil, pipelineTag: nil)
    }

    func testCommonMlxWidths() {
        XCTAssertEqual(m("x/model-3bit").quantization, "3-bit")
        XCTAssertEqual(m("x/model-4bit").quantization, "4-bit")
        XCTAssertEqual(m("x/model-6bit").quantization, "6-bit")
        XCTAssertEqual(m("x/model-8bit").quantization, "8-bit")
    }

    func testUncommonWidths_previouslyDropped() {
        // The bug: any width outside {3,4,6,8} showed no badge.
        XCTAssertEqual(m("x/model-2bit").quantization, "2-bit")
        XCTAssertEqual(m("x/model-5bit").quantization, "5-bit")
        XCTAssertEqual(m("x/model-9bit").quantization, "9-bit")
        XCTAssertEqual(m("mlx-community/Foo-2-bit").quantization, "2-bit")
    }

    func testFpDtypes() {
        XCTAssertEqual(m("x/model-bf16").quantization, "BF16")
        XCTAssertEqual(m("x/model-fp16").quantization, "FP16")
    }

    func testGgufStyleQuants() {
        XCTAssertEqual(m("x/model-Q2_K").quantization, "2-bit")
        XCTAssertEqual(m("x/model-Q5_K_M").quantization, "5-bit")
        XCTAssertEqual(m("x/model-Q6_K").quantization, "6-bit")
        XCTAssertEqual(m("x/model-IQ3_M").quantization, "3-bit")
    }

    func testFractionalWidth_notTruncated() {
        // "3.5bit" must not be misread as "5-bit".
        XCTAssertEqual(m("x/model-3.5bit").quantization, "3.5-bit")
    }

    func testNoQuant_returnsNil() {
        XCTAssertNil(m("x/plain-model").quantization)
        XCTAssertNil(m("mlx-community/Qwen3-30B-A3B").quantization)
    }

    func testServedFpModes_badgeTheFormat() {
        // nvfp4/mxfp4/mxfp8 are loadable since v26.6.10 — badge the format
        // name, not a bare bit width (and not nothing, as pre-support).
        XCTAssertEqual(m("x/Qwen3-30B-nvfp4").quantization, "NVFP4")
        XCTAssertEqual(m("x/model-mxfp4").quantization, "MXFP4")
        XCTAssertEqual(m("x/gemma-4-E4B-it-qat-nvfp4").quantization, "NVFP4")
    }

    func testUnloadableFpModes_noBadge() {
        // mxfp6 stays outside the server's set — it surfaces via
        // incompatibleReason, never as a quant badge.
        XCTAssertNil(m("x/model-mxfp6").quantization)
    }

    func testGgufRepo_showsMulti() {
        // GGUF repos host many quant files; the repo ID has no single quant.
        // The column should show "Multi" rather than "—".
        let gguf = HFModel(id: "bartowski/Meta-Llama-3.1-8B-Instruct-GGUF",
                           downloads: nil, likes: nil, lastModified: nil,
                           tags: ["gguf"], safetensors: nil, pipelineTag: nil)
        XCTAssertEqual(gguf.quantization, "Multi")
    }

    func testGgufRepo_specificQuant_preservedOverMulti() {
        // A rare single-quant GGUF repo whose ID encodes the quant (e.g. Q4_K_M)
        // should still show the specific label, not "Multi".
        let gguf = HFModel(id: "user/model-Q4_K_M-GGUF",
                           downloads: nil, likes: nil, lastModified: nil,
                           tags: ["gguf"], safetensors: nil, pipelineTag: nil)
        XCTAssertEqual(gguf.quantization, "4-bit")
    }
}

// MARK: - HFModel.ramEstimate range surfacing

final class HFModelRamEstimateTests: XCTestCase {
    func testRamEstimate_rangeFromGgufFields() {
        // Two quants populated → ramEstimate shows the formatted range with
        // the same ×1.2 overhead as the single-value path.
        var m = HFModel(id: "u/gemma-4-E4B-it-GGUF", downloads: nil, likes: nil,
                        lastModified: nil, tags: ["gguf"], safetensors: nil, pipelineTag: nil)
        m.ggufMinSizeBytes = 1_700_000_000   // ≈ 1.58 GB
        m.ggufMaxSizeBytes = 8_500_000_000   // ≈ 7.91 GB
        let s = m.ramEstimate
        XCTAssertTrue(s.contains("GB"), "expected GB unit, got \(s)")
        XCTAssertTrue(s.contains("\u{2013}"), "expected en-dash range separator, got \(s)")
        XCTAssertNotEqual(s, "Unknown")
    }

    func testRamEstimate_singleGgufStillUsesFallbackPath() {
        // Single-quant GGUF: parseFallbackSize records .ggufSingle, the
        // service sets fallbackSizeBytes only, ramEstimate returns the
        // single-value format.
        var m = HFModel(id: "u/Q4-only-GGUF", downloads: nil, likes: nil,
                        lastModified: nil, tags: ["gguf"], safetensors: nil, pipelineTag: nil)
        m.fallbackSizeBytes = 2_600_000_000
        XCTAssertFalse(m.ramEstimate.contains("\u{2013}"))
        XCTAssertNotEqual(m.ramEstimate, "Unknown")
    }

    func testRamEstimateBytes_usesMaxForConservativeFitness() {
        // Range repo with max ≈ 8 GB: fitness must compare against the
        // conservative-high number so a 6 GB Mac doesn't see "fits".
        var m = HFModel(id: "u/range", downloads: nil, likes: nil,
                        lastModified: nil, tags: ["gguf"], safetensors: nil, pipelineTag: nil)
        m.ggufMinSizeBytes = 1_700_000_000
        m.ggufMaxSizeBytes = 8_500_000_000
        // 8.5 GB × 1.2 ≈ 10.2 GB. Allow ±1% slack for the float math.
        let expected: Int64 = Int64(Double(8_500_000_000) * 1.2)
        XCTAssertEqual(m.ramEstimateBytes, expected)
    }

    func testRamEstimate_unknownWhenNothingPopulated() {
        let m = HFModel(id: "u/empty", downloads: nil, likes: nil,
                        lastModified: nil, tags: nil, safetensors: nil, pipelineTag: nil)
        XCTAssertEqual(m.ramEstimate, "Unknown")
        XCTAssertEqual(m.ramEstimateBytes, 0)
    }
}

// MARK: - MemoryInfo.formatRange

final class MemoryInfoFormatRangeTests: XCTestCase {
    func testGbRange_sameUnit() {
        // 1.58 GB to 7.91 GB → "1.6–7.9 GB" (rounded to 1 decimal place).
        let s = MemoryInfo.formatRange(1_700_000_000, 8_500_000_000)
        XCTAssertTrue(s.hasSuffix(" GB"), "got \(s)")
        XCTAssertTrue(s.contains("\u{2013}"), "got \(s)")
    }

    func testMbRange_whenBothUnderGb() {
        let s = MemoryInfo.formatRange(200_000_000, 800_000_000)
        XCTAssertTrue(s.hasSuffix(" MB"), "got \(s)")
        XCTAssertTrue(s.contains("\u{2013}"), "got \(s)")
    }

    func testSingleValue_whenMinEqualsMax() {
        // Degenerate range collapses to the single-value formatter — no dash.
        let s = MemoryInfo.formatRange(2_600_000_000, 2_600_000_000)
        XCTAssertFalse(s.contains("\u{2013}"), "got \(s)")
    }

    func testReversedArgs_areNormalized() {
        let a = MemoryInfo.formatRange(8_500_000_000, 1_700_000_000)
        let b = MemoryInfo.formatRange(1_700_000_000, 8_500_000_000)
        XCTAssertEqual(a, b)
    }
}

// =============================================================================
// MARK: - Model browser search defaults
// =============================================================================

@MainActor
final class HFSearchDefaultsTests: XCTestCase {
    /// The format filter defaults to MLX — the native engine's format — not
    /// "Both": mixing GGUF into the default results buries the recommended
    /// MLX builds. GGUF stays one click away in the segmented picker.
    func testFormatFilterDefaultsToMlx() {
        XCTAssertEqual(HFSearchService().format, .mlx)
    }
}
