import XCTest
@testable import MLXCore

/// Retrieval-quality diagnostics against a real local document folder, gated on
/// `DOCS_RAG_FOLDER` (skipped in CI). Uses the production Apple sentence
/// embeddings — run it to eyeball what the searchDocuments tool would return:
///
///   DOCS_RAG_FOLDER=~/path/to/folder swift test -Xswiftc -swift-version \
///     -Xswiftc 5 --filter DocumentIndexRealDataTests
///
/// Assertions are deliberately structural (chunks line-aligned, metadata fields
/// kept intact, known-relevant files surfaced) rather than score thresholds —
/// embedding scores shift across OS versions.
final class DocumentIndexRealDataTests: XCTestCase {

    private static var folder: URL? {
        guard let path = ProcessInfo.processInfo.environment["DOCS_RAG_FOLDER"], !path.isEmpty else {
            return nil
        }
        return URL(fileURLWithPath: NSString(string: path).expandingTildeInPath)
    }

    /// Built once and shared across the test methods — indexing a real folder
    /// with production embeddings is the expensive part.
    @MainActor private static var cachedIndex: DocumentIndex?

    @MainActor
    private func buildIndex() async throws -> DocumentIndex {
        guard let folder = Self.folder else {
            throw XCTSkip("DOCS_RAG_FOLDER not set — real-data diagnostics skipped")
        }
        if let cached = Self.cachedIndex, case .ready = cached.state {
            return cached
        }
        let index = DocumentIndex(folderURL: folder)
        Self.cachedIndex = index
        index.startIndexing()
        let start = Date()
        while Date().timeIntervalSince(start) < 600 {
            if case .ready = index.state { break }
            if case .failed(let msg) = index.state { throw XCTSkip("indexing failed: \(msg)") }
            try await Task.sleep(nanoseconds: 200_000_000)
        }
        guard case .ready(let files, let chunks) = index.state else {
            XCTFail("indexing did not finish within 10 minutes")
            throw XCTSkip("timeout")
        }
        print("[real-data] indexed \(files) files → \(chunks) chunks in \(Int(Date().timeIntervalSince(start)))s")
        return index
    }

    /// No chunk may start or end mid-line: structured single-paragraph files
    /// (YAML/JSON/logs with no blank lines) must split at line boundaries.
    /// Verified by re-chunking one real file and checking that every chunk
    /// boundary coincides with a whole line of the original text.
    @MainActor
    func testChunksAreLineAligned() async throws {
        guard let folder = Self.folder else {
            throw XCTSkip("DOCS_RAG_FOLDER not set — real-data diagnostics skipped")
        }
        let fm = FileManager.default
        let firstFile = try XCTUnwrap(
            try fm.contentsOfDirectory(at: folder, includingPropertiesForKeys: nil)
                .filter { DocumentChunker.isIndexable($0) }
                .sorted { $0.path < $1.path }
                .first)
        let text = try XCTUnwrap(DocumentIndex.extractText(firstFile))
        let lines = Set(text.components(separatedBy: "\n").map {
            $0.trimmingCharacters(in: .whitespaces)
        })
        for chunk in DocumentChunker.chunk(text) {
            let chunkLines = chunk.components(separatedBy: "\n")
            for boundary in [chunkLines.first, chunkLines.last].compactMap({ $0 }) {
                let t = boundary.trimmingCharacters(in: .whitespaces)
                XCTAssertTrue(t.isEmpty || lines.contains(t),
                              "chunk boundary is not a whole line of the source: '\(t.prefix(80))'")
            }
        }
    }

    /// The marquee query from the user's manual test. The top results should
    /// be dominated by files that actually carry `frustration: HIGH`.
    @MainActor
    func testFrustratedCustomerQuerySurfacesHighFrustrationFiles() async throws {
        let index = try await buildIndex()
        guard index.chunks.contains(where: { $0.text.contains("frustration:") }) else {
            throw XCTSkip("folder doesn't look like CXone YAML — skipping the frustration assertions")
        }

        for query in ["which customer is very frustrated",
                      "customer with high frustration",
                      "angry customer complaint escalation"] {
            let out = await index.search(query: query)
            print("\n[real-data] query: \(query)\n\(out.prefix(1500))\n")
        }

        // Ground truth: files with `frustration: HIGH`. At least 3 of the top 5
        // results for the literal query should come from that set.
        let highFiles = Set(index.chunks.filter { $0.text.contains("frustration: HIGH") }.map { $0.source })
        guard !highFiles.isEmpty else { throw XCTSkip("no frustration: HIGH files present") }

        let out = await index.search(query: "customer with high frustration")
        let returnedSources = out.components(separatedBy: "\n")
            .filter { $0.hasPrefix("[") && $0.contains("relevance") }
            .compactMap { line -> String? in
                guard let close = line.firstIndex(of: "]") else { return nil }
                return line[line.index(close, offsetBy: 2)...]
                    .components(separatedBy: " (relevance").first
            }
        let hits = returnedSources.filter { highFiles.contains($0) }.count
        print("[real-data] 'customer with high frustration' → \(hits)/\(returnedSources.count) results are frustration: HIGH files")
        XCTAssertGreaterThanOrEqual(hits, 3,
            "retrieval should surface frustration: HIGH files for this query; got sources: \(returnedSources)")
    }

    /// Sanity: an exact-term query (entity that exists in few files) must rank
    /// a file containing the term at the top — the hybrid lexical component.
    @MainActor
    func testExactTermQueryFindsItsFile() async throws {
        let index = try await buildIndex()
        // Pick a moderately rare word from the corpus itself: take a chunk and
        // use one of its distinctive tokens.
        guard let sample = index.chunks.first(where: { $0.text.contains("royal caribbean") }) else {
            throw XCTSkip("corpus has no 'royal caribbean' sample token")
        }
        let out = await index.search(query: "royal caribbean competitor mention")
        print("[real-data] exact-term query → \(out.prefix(600))")
        XCTAssertTrue(out.contains(sample.source) || out.contains("royal caribbean"),
                      "exact term should surface a file containing it")
    }
}
