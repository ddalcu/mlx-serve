import XCTest
@testable import MLXCore

/// Mini-RAG pipeline: folder → chunks → embeddings → searchDocuments tool.
/// Pure pieces (chunker, retrieval scoring, formatting) are tested with a
/// deterministic fake embedder; the searchDocuments tool dispatch is tested
/// through AgentEngine.executeToolCall like the MCP routing tests above it.
final class DocumentIndexTests: XCTestCase {

    // MARK: - Chunker

    func testShortTextIsOneChunk() {
        let chunks = DocumentChunker.chunk("Hello world.\n\nSecond paragraph.")
        XCTAssertEqual(chunks, ["Hello world.\n\nSecond paragraph."])
    }

    func testEmptyAndWhitespaceTextYieldNoChunks() {
        XCTAssertEqual(DocumentChunker.chunk(""), [])
        XCTAssertEqual(DocumentChunker.chunk("  \n\n \t"), [])
    }

    func testLongTextSplitsAtParagraphsWithinMaxChars() {
        let para = String(repeating: "word ", count: 60).trimmingCharacters(in: .whitespaces) // ~300 chars
        let text = Array(repeating: para, count: 10).joined(separator: "\n\n")               // ~3000 chars
        let chunks = DocumentChunker.chunk(text, maxChars: 1000, overlapChars: 0)
        XCTAssertGreaterThan(chunks.count, 1)
        for c in chunks {
            XCTAssertLessThanOrEqual(c.count, 1000, "chunk exceeds maxChars: \(c.count)")
            XCTAssertFalse(c.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
        }
        // Nothing lost: every paragraph appears in some chunk.
        XCTAssertTrue(chunks.allSatisfy { $0.contains("word") })
    }

    func testConsecutiveChunksShareOverlap() {
        // Distinct numbered paragraphs so we can detect the carried-over tail.
        let paras = (1...10).map { "Paragraph \($0) " + String(repeating: "alpha\($0) ", count: 30) }
        let text = paras.joined(separator: "\n\n")
        let chunks = DocumentChunker.chunk(text, maxChars: 600, overlapChars: 120)
        XCTAssertGreaterThan(chunks.count, 1)
        for i in 1..<chunks.count {
            // The head of chunk i must repeat text that already appeared in chunk i-1.
            let head = String(chunks[i].prefix(40))
            XCTAssertTrue(chunks[i - 1].contains(head.trimmingCharacters(in: .whitespacesAndNewlines)),
                          "chunk \(i) does not overlap its predecessor")
        }
    }

    func testGiantSingleParagraphIsHardSplit() {
        let text = String(repeating: "x", count: 5000)
        let chunks = DocumentChunker.chunk(text, maxChars: 1200, overlapChars: 0)
        XCTAssertGreaterThan(chunks.count, 1)
        for c in chunks { XCTAssertLessThanOrEqual(c.count, 1200) }
    }

    /// Structured files (YAML/JSON/logs) have no blank lines, so the whole file
    /// is one "paragraph" — it must split at LINE boundaries, never mid-line.
    /// Regression: CXone YAML transcripts came back as mid-word windows
    /// ("act Themes/Accounts…") because oversized paragraphs were split at
    /// fixed character offsets.
    func testNoBlankLineFileSplitsAtLineBoundaries() {
        let lines = (1...120).map { "key\($0): some value words for line number \($0)" }
        let text = lines.joined(separator: "\n")           // no blank lines anywhere
        let chunks = DocumentChunker.chunk(text, maxChars: 600, overlapChars: 100)
        XCTAssertGreaterThan(chunks.count, 1)
        let lineSet = Set(lines)
        for chunk in chunks {
            XCTAssertLessThanOrEqual(chunk.count, 600)
            for boundary in [chunk.components(separatedBy: "\n").first!,
                             chunk.components(separatedBy: "\n").last!] {
                XCTAssertTrue(lineSet.contains(boundary),
                              "chunk boundary is not a whole source line: '\(boundary)'")
            }
        }
        // Nothing dropped: every line appears in at least one chunk.
        for line in lines {
            XCTAssertTrue(chunks.contains { $0.contains(line) }, "lost line: \(line)")
        }
    }

    /// The metadata line a question hinges on must survive chunking intact.
    func testMetadataFieldLineSurvivesChunking() {
        var lines = (1...80).map { "entityValue\($0): something descriptive \($0)" }
        lines.insert("frustration: HIGH", at: 40)
        let chunks = DocumentChunker.chunk(lines.joined(separator: "\n"), maxChars: 500, overlapChars: 80)
        XCTAssertTrue(chunks.contains { $0.contains("frustration: HIGH") },
                      "the frustration: HIGH line must appear unbroken in a chunk")
    }

    func testIndexableFileFilter() {
        func indexable(_ name: String) -> Bool {
            DocumentChunker.isIndexable(URL(fileURLWithPath: "/tmp/docs/\(name)"))
        }
        XCTAssertTrue(indexable("transcript.txt"))
        XCTAssertTrue(indexable("notes.md"))
        XCTAssertTrue(indexable("report.PDF"))
        XCTAssertTrue(indexable("data.json"))
        XCTAssertTrue(indexable("config.yaml"))
        XCTAssertTrue(indexable("table.csv"))
        XCTAssertFalse(indexable("photo.png"))
        XCTAssertFalse(indexable("archive.zip"))
        XCTAssertFalse(indexable("binary"))
        XCTAssertFalse(indexable(".DS_Store"))
        XCTAssertFalse(indexable(".hidden.txt"))
    }

    // MARK: - Retrieval scoring

    func testCosineSimilarity() {
        XCTAssertEqual(DocumentRetrieval.cosine([1, 0], [0, 1]), 0, accuracy: 1e-9)
        XCTAssertEqual(DocumentRetrieval.cosine([1, 2, 3], [1, 2, 3]), 1, accuracy: 1e-9)
        XCTAssertEqual(DocumentRetrieval.cosine([1, 0], [-1, 0]), -1, accuracy: 1e-9)
        XCTAssertEqual(DocumentRetrieval.cosine([0, 0], [1, 1]), 0, accuracy: 1e-9) // zero vector guard
    }

    func testSearchRanksByVectorSimilarity() {
        let chunks = [
            DocumentRetrieval.Chunk(text: "about cats", source: "cats.txt", vector: [1, 0, 0]),
            DocumentRetrieval.Chunk(text: "about dogs", source: "dogs.txt", vector: [0, 1, 0]),
            DocumentRetrieval.Chunk(text: "about fish", source: "fish.txt", vector: [0, 0, 1]),
        ]
        let results = DocumentRetrieval.search(chunks: chunks, query: "q",
                                               queryVector: [0.1, 0.9, 0.1], topK: 2)
        XCTAssertEqual(results.count, 2)
        XCTAssertEqual(results[0].source, "dogs.txt")
    }

    func testSearchFallsBackToLexicalWhenNoQueryVector() {
        let chunks = [
            DocumentRetrieval.Chunk(text: "we discussed pricing and the budget", source: "a.txt", vector: nil),
            DocumentRetrieval.Chunk(text: "lunch plans for tuesday", source: "b.txt", vector: nil),
        ]
        let results = DocumentRetrieval.search(chunks: chunks, query: "what was the pricing budget?",
                                               queryVector: nil, topK: 1)
        XCTAssertEqual(results.first?.source, "a.txt")
    }

    /// Hybrid scoring: an exact-term (lexical) hit must be able to beat a
    /// slightly-better embedding score. Pure cosine ranked semantically-vague
    /// metadata above chunks containing the literal query terms.
    func testHybridScoringBoostsExactTermMatches() {
        let chunks = [
            // Cosine favors A (0.6 vs 0.5), but B contains the query's words.
            DocumentRetrieval.Chunk(text: "general account discussion notes", source: "a.txt",
                                    vector: [0.6, 0.8]),
            DocumentRetrieval.Chunk(text: "frustration: HIGH customer call", source: "b.txt",
                                    vector: [0.5, 0.866]),
        ]
        let results = DocumentRetrieval.search(chunks: chunks, query: "high frustration customer",
                                               queryVector: [1, 0], topK: 2)
        XCTAssertEqual(results.first?.source, "b.txt",
                       "lexical term hits must outweigh a small cosine edge")
    }

    /// Light stemming: query "frustrated" must match "frustration" in text —
    /// inflection differences shouldn't zero out the lexical signal.
    func testLexicalScoreMatchesInflectedForms() {
        let score = DocumentRetrieval.lexicalScore(
            query: "frustrated customer",
            text: "sentiment: NEGATIVE\nfrustration: HIGH\ncustomer: yes")
        XCTAssertEqual(score, 1.0, accuracy: 0.01,
                       "'frustrated'→'frustration' and 'customer'→'customer' should both match")
    }

    /// IDF weighting: a query term that appears in EVERY chunk ("frustration:"
    /// is a key in every CXone file) carries no signal — the rare value token
    /// ("HIGH") must dominate. Without IDF, 'customer with high frustration'
    /// returned 0/5 frustration: HIGH files on the real corpus because flat
    /// term counting let the ubiquitous terms drown the discriminating one.
    func testLexicalScoringWeightsRareTermsHigher() {
        // 9 chunks all say "frustration: NONE", one says "frustration: HIGH".
        // Every chunk shares the ubiquitous stems; only one has the rare one.
        var chunks = (0..<9).map {
            DocumentRetrieval.Chunk(text: "customer record \($0)\nfrustration: NONE\ncsat: 5",
                                    source: "none\($0).yaml", vector: nil)
        }
        chunks.append(DocumentRetrieval.Chunk(text: "customer record x\nfrustration: HIGH\ncsat: 0",
                                              source: "high.yaml", vector: nil))
        let results = DocumentRetrieval.search(chunks: chunks, query: "customer with high frustration",
                                               queryVector: nil, topK: 3)
        XCTAssertEqual(results.first?.source, "high.yaml",
                       "the chunk with the rare discriminating term must rank first")
        // And the gap must be material, not a tie-break: ubiquitous terms
        // ("customer", "frustration") should contribute ~nothing.
        if results.count >= 2 {
            XCTAssertGreaterThan(results[0].score, results[1].score * 2,
                                 "rare-term hit should dominate, got \(results.map(\.score))")
        }
    }

    /// Result diversity: one file's many chunks must not monopolize the top-K —
    /// "which customer…" questions need excerpts from multiple files.
    func testSearchCapsExcerptsPerSource() {
        var chunks = (0..<8).map {
            DocumentRetrieval.Chunk(text: "dominant chunk \($0)", source: "big.yaml", vector: [1, 0])
        }
        chunks.append(DocumentRetrieval.Chunk(text: "other relevant file", source: "other.yaml",
                                              vector: [0.9, 0.436]))
        let results = DocumentRetrieval.search(chunks: chunks, query: "q", queryVector: [1, 0], topK: 5)
        XCTAssertEqual(results.filter { $0.source == "big.yaml" }.count, 2,
                       "at most 2 excerpts per source file")
        XCTAssertTrue(results.contains { $0.source == "other.yaml" },
                      "the second file must surface despite lower scores")
    }

    func testSearchOnEmptyIndexReturnsNothing() {
        XCTAssertTrue(DocumentRetrieval.search(chunks: [], query: "q", queryVector: [1], topK: 5).isEmpty)
    }

    func testSearchRespectsTopK() {
        let chunks = (0..<10).map { DocumentRetrieval.Chunk(text: "t\($0)", source: "s\($0)", vector: [Double($0)]) }
        XCTAssertEqual(DocumentRetrieval.search(chunks: chunks, query: "q", queryVector: [1], topK: 3).count, 3)
    }

    // MARK: - Result formatting

    func testFormatResultsEmpty() {
        let out = DocumentRetrieval.formatResults([], query: "pricing")
        XCTAssertTrue(out.contains("No"), "empty result should explain nothing matched: \(out)")
        XCTAssertTrue(out.contains("pricing"))
    }

    func testFormatResultsListsSourcesAndText() {
        let results = [
            DocumentRetrieval.SearchResult(text: "the budget was approved", source: "march/meeting.txt", score: 0.91),
            DocumentRetrieval.SearchResult(text: "pricing tiers draft", source: "pricing.pdf", score: 0.74),
        ]
        let out = DocumentRetrieval.formatResults(results, query: "budget")
        XCTAssertTrue(out.contains("march/meeting.txt"))
        XCTAssertTrue(out.contains("the budget was approved"))
        XCTAssertTrue(out.contains("pricing.pdf"))
        XCTAssertTrue(out.contains("[1]"))
        XCTAssertTrue(out.contains("[2]"))
    }

    // MARK: - Tool definition & wiring

    func testSearchDocumentsToolKindExists() {
        XCTAssertNotNil(AgentToolKind(rawValue: "searchDocuments"))
    }

    func testSearchDocumentsToolJSONIsValidAndRequiresQuery() throws {
        let data = AgentPrompt.searchDocumentsToolJSON.data(using: .utf8)!
        let arr = try XCTUnwrap(try JSONSerialization.jsonObject(with: data) as? [[String: Any]])
        XCTAssertEqual(arr.count, 1)
        let fn = try XCTUnwrap(arr[0]["function"] as? [String: Any])
        XCTAssertEqual(fn["name"] as? String, "searchDocuments")
        let params = try XCTUnwrap(fn["parameters"] as? [String: Any])
        XCTAssertEqual(params["required"] as? [String], ["query"])
    }

    @MainActor
    func testMissingQueryParamIsValidated() {
        XCTAssertEqual(AgentEngine.missingRequiredParams(for: "searchDocuments", arguments: [:]), ["query"])
        XCTAssertEqual(AgentEngine.missingRequiredParams(for: "searchDocuments", arguments: ["query": "x"]), [])
    }

    func testCombinedToolsJSONWithDocs() throws {
        // Docs only (plain chat + attached folder): just the search tool.
        let docsOnly = ChatTurnEngine.combinedToolsJSON(
            agentMode: false, mcpToolsJSON: nil, docsToolJSON: AgentPrompt.searchDocumentsToolJSON)
        let docsArr = try XCTUnwrap(try JSONSerialization.jsonObject(
            with: docsOnly!.data(using: .utf8)!) as? [[String: Any]])
        XCTAssertEqual(docsArr.count, 1)

        // Agent mode + docs: built-ins plus the search tool.
        let merged = ChatTurnEngine.combinedToolsJSON(
            agentMode: true, mcpToolsJSON: nil, docsToolJSON: AgentPrompt.searchDocumentsToolJSON)
        let mergedArr = try XCTUnwrap(try JSONSerialization.jsonObject(
            with: merged!.data(using: .utf8)!) as? [[String: Any]])
        let names = mergedArr.compactMap { ($0["function"] as? [String: Any])?["name"] as? String }
        XCTAssertTrue(names.contains("searchDocuments"))
        XCTAssertTrue(names.contains("shell"))

        // No sources at all → nil (existing behavior).
        XCTAssertNil(ChatTurnEngine.combinedToolsJSON(agentMode: false, mcpToolsJSON: nil, docsToolJSON: nil))
    }

    // MARK: - Tool dispatch through AgentEngine

    @MainActor
    func testSearchDocumentsWithoutIndexReturnsError() async {
        let rep = AgentEngine.RepetitionTracker()
        var wd: String? = nil
        let tc = APIClient.ToolCall(id: "1", name: "searchDocuments",
                                    arguments: ["query": "budget"], rawArguments: "{}")
        let r = await AgentEngine.executeToolCall(
            tc, workingDirectory: &wd, repetition: rep, iteration: 0,
            agentMemory: AgentMemory(), documentIndex: nil)
        XCTAssertTrue(r.output.hasPrefix("Error:"), "expected error without an index, got: \(r.output)")
    }

    @MainActor
    func testSearchDocumentsToolReturnsRankedExcerpts() async {
        let index = DocumentIndex(folderURL: URL(fileURLWithPath: "/tmp/docs"),
                                  makeEmbedder: { FakeEmbedder() })
        index.finishForTesting(chunks: [
            DocumentRetrieval.Chunk(text: "the cat sat on the mat", source: "cats.txt",
                                    vector: FakeEmbedder().vector(for: "the cat sat on the mat")),
            DocumentRetrieval.Chunk(text: "stocks fell sharply today", source: "market.txt",
                                    vector: FakeEmbedder().vector(for: "stocks fell sharply today")),
        ], files: 2)

        let rep = AgentEngine.RepetitionTracker()
        var wd: String? = nil
        let tc = APIClient.ToolCall(id: "1", name: "searchDocuments",
                                    arguments: ["query": "cat mat"], rawArguments: "{}")
        let r = await AgentEngine.executeToolCall(
            tc, workingDirectory: &wd, repetition: rep, iteration: 0,
            agentMemory: AgentMemory(), documentIndex: index)
        XCTAssertTrue(r.output.contains("cats.txt"), "expected the cat chunk first: \(r.output)")
        // The top result must be the cat chunk, not the market one.
        let catPos = r.output.range(of: "cats.txt")!.lowerBound
        let marketPos = r.output.range(of: "market.txt")?.lowerBound
        if let marketPos { XCTAssertLessThan(catPos, marketPos) }
    }

    @MainActor
    func testSearchWhileIndexingTellsModelToWait() async {
        let index = DocumentIndex(folderURL: URL(fileURLWithPath: "/tmp/docs"),
                                  makeEmbedder: { FakeEmbedder() })
        // state stays .indexing — never finished
        let out = await index.search(query: "anything")
        XCTAssertTrue(out.lowercased().contains("index"), "should say indexing is in progress: \(out)")
    }

    // MARK: - End-to-end folder indexing (real files, fake embedder)

    @MainActor
    func testIndexFolderEndToEnd() async throws {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("doc-index-test-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        try "alice: the deploy is scheduled for friday\nbob: ok, after the budget review"
            .write(to: dir.appendingPathComponent("chat1.txt"), atomically: true, encoding: .utf8)
        try "{\"topic\": \"quarterly pricing\", \"decision\": \"raise tier two\"}"
            .write(to: dir.appendingPathComponent("notes.json"), atomically: true, encoding: .utf8)
        try Data([0xFF, 0xD8, 0xFF]).write(to: dir.appendingPathComponent("photo.jpg")) // skipped

        let index = DocumentIndex(folderURL: dir, makeEmbedder: { FakeEmbedder() })
        index.startIndexing()
        // Poll until ready (file IO + fake embedding is fast).
        for _ in 0..<200 {
            if case .ready = index.state { break }
            if case .failed(let msg) = index.state { return XCTFail("indexing failed: \(msg)") }
            try await Task.sleep(nanoseconds: 20_000_000)
        }
        guard case .ready(let files, let chunks) = index.state else {
            return XCTFail("index never became ready: \(index.state)")
        }
        XCTAssertEqual(files, 2, "jpg must be skipped")
        XCTAssertGreaterThanOrEqual(chunks, 2)

        let out = await index.search(query: "deploy friday")
        XCTAssertTrue(out.contains("chat1.txt"), "search should surface the transcript: \(out)")
    }

    /// Indexing must fan out across multiple workers, each with its own
    /// embedder instance — single-threaded NLEmbedding measured 48ms/chunk,
    /// i.e. 7+ minutes for a 500-file folder. One embedder = one worker.
    @MainActor
    func testIndexingParallelizesAcrossWorkers() async throws {
        try XCTSkipIf(ProcessInfo.processInfo.activeProcessorCount < 4, "needs a multicore host")
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("doc-index-par-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        for i in 0..<12 {
            try "file \(i) content about topic number \(i)"
                .write(to: dir.appendingPathComponent("f\(i).txt"), atomically: true, encoding: .utf8)
        }

        let counter = EmbedderFactoryCounter()
        let index = DocumentIndex(folderURL: dir, makeEmbedder: {
            counter.increment()
            return FakeEmbedder()
        })
        index.startIndexing()
        for _ in 0..<200 {
            if case .ready = index.state { break }
            try await Task.sleep(nanoseconds: 20_000_000)
        }
        guard case .ready(let files, _) = index.state else {
            return XCTFail("index never became ready: \(index.state)")
        }
        XCTAssertEqual(files, 12)
        XCTAssertGreaterThan(counter.count, 1,
                             "indexing must create one embedder per worker (parallel), got \(counter.count)")
    }

    @MainActor
    func testIndexFolderWithNoDocumentsFails() async throws {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("doc-index-empty-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        try Data([0x00]).write(to: dir.appendingPathComponent("blob.bin"))

        let index = DocumentIndex(folderURL: dir, makeEmbedder: { FakeEmbedder() })
        index.startIndexing()
        for _ in 0..<200 {
            if case .failed = index.state { break }
            try await Task.sleep(nanoseconds: 20_000_000)
        }
        guard case .failed = index.state else {
            return XCTFail("expected .failed for a folder with no indexable documents, got \(index.state)")
        }
    }

    // MARK: - ServerEmbedding (GPU via /v1/embeddings)

    func testServerEmbeddingChunksRequestsAtMaxBatch() async {
        let rec = TransportRecorder()
        let e = ServerEmbedding(model: "minilm-8bit", transport: rec.transport)
        let texts = (0..<130).map { "text number \($0)" }
        let vecs = await e.vectors(for: texts)
        XCTAssertEqual(vecs.count, 130)
        XCTAssertEqual(rec.batches.map(\.count), [64, 64, 2])
        // Order preserved end-to-end (vector encodes the text length).
        for (t, v) in zip(texts, vecs) {
            XCTAssertEqual(v?[0], Double(t.count))
        }
    }

    func testServerEmbeddingDegradesToNilOnTransportFailure() async {
        let rec = TransportRecorder()
        rec.failAll = true
        let e = ServerEmbedding(model: "minilm-8bit", transport: rec.transport)
        let vecs = await e.vectors(for: ["a", "b", "c"])
        XCTAssertEqual(vecs.count, 3)
        XCTAssertTrue(vecs.allSatisfy { $0 == nil }, "failed transport must degrade to lexical-only, not crash")
    }

    func testServerEmbeddingPrefixesBgeQueriesOnly() async {
        let bgeRec = TransportRecorder()
        _ = await ServerEmbedding(model: "bge-small-en-v1.5-8bit", transport: bgeRec.transport)
            .queryVector(for: "what was the budget?")
        XCTAssertEqual(bgeRec.texts.count, 1)
        XCTAssertTrue(bgeRec.texts[0].hasPrefix("Represent this sentence for searching relevant passages: "),
                      "BGE queries need the instruction prefix: \(bgeRec.texts[0])")
        XCTAssertTrue(bgeRec.texts[0].hasSuffix("what was the budget?"))

        // Passages are never prefixed, and symmetric models aren't either.
        let bgePassageRec = TransportRecorder()
        _ = await ServerEmbedding(model: "bge-small-en-v1.5-8bit", transport: bgePassageRec.transport)
            .vectors(for: ["a passage"])
        XCTAssertEqual(bgePassageRec.texts, ["a passage"])
        let plainRec = TransportRecorder()
        _ = await ServerEmbedding(model: "all-MiniLM-L6-v2-8bit", transport: plainRec.transport)
            .queryVector(for: "what was the budget?")
        XCTAssertEqual(plainRec.texts, ["what was the budget?"])
    }

    func testPickEmbeddingModelPrefersLoadedEncoder() {
        func model(_ name: String, embeddings: Bool, loaded: Bool) -> ModelInfo {
            ModelInfo(name: name, quantBits: 0, layers: 0, hiddenSize: 0, vocabSize: 0,
                      contextLength: 0, modelMaxTokens: 0,
                      supportsEmbeddings: embeddings, loaded: loaded)
        }
        XCTAssertNil(ServerEmbedding.pickEmbeddingModel([model("gemma", embeddings: false, loaded: true)]))
        // A stub encoder is picked (server hot-loads on first request) …
        XCTAssertEqual(ServerEmbedding.pickEmbeddingModel([
            model("gemma", embeddings: false, loaded: true),
            model("bge-stub", embeddings: true, loaded: false),
        ]), "bge-stub")
        // … but an already-loaded encoder wins over a stub.
        XCTAssertEqual(ServerEmbedding.pickEmbeddingModel([
            model("bge-stub", embeddings: true, loaded: false),
            model("minilm-loaded", embeddings: true, loaded: true),
        ]), "minilm-loaded")
    }

    func testParseModelInfoReadsEmbeddingsCapabilityFromStub() {
        // Shape of an UNLOADED encoder stub from /v1/models (arch-hint path).
        let stub: [String: Any] = [
            "id": "bge-small-en-v1.5-8bit", "loaded": false, "state": "unloaded",
            "capabilities": ["embeddings"], "meta": ["architecture": "bert"],
        ]
        let info = APIClient.parseModelInfo(stub)
        XCTAssertTrue(info.supportsEmbeddings)
        XCTAssertFalse(info.loaded)
        // Chat models don't grow the capability.
        let chat: [String: Any] = ["id": "gemma", "capabilities": ["chat", "tool_use"]]
        XCTAssertFalse(APIClient.parseModelInfo(chat).supportsEmbeddings)
    }

    /// End-to-end: when the GPU probe yields a ServerEmbedding, indexing
    /// embeds every chunk through it (no CPU embedder), and the QUERY rides
    /// the same embedder so vectors stay in one space.
    @MainActor
    func testIndexingUsesGpuEmbedderForChunksAndQuery() async throws {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("doc-index-gpu-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        for i in 0..<6 {
            try "deploy notes for service \(i): roll out on friday"
                .write(to: dir.appendingPathComponent("f\(i).txt"), atomically: true, encoding: .utf8)
        }

        let rec = TransportRecorder()
        let index = DocumentIndex(
            folderURL: dir,
            makeEmbedder: { XCTFail("CPU embedder must not be built when the GPU probe succeeds"); return nil },
            gpuEmbedderProbe: { ServerEmbedding(model: "bge-test", transport: rec.transport) }
        )
        index.startIndexing()
        for _ in 0..<200 {
            if case .ready = index.state { break }
            try await Task.sleep(nanoseconds: 20_000_000)
        }
        guard case .ready(let files, let chunks) = index.state else {
            return XCTFail("index never became ready: \(index.state)")
        }
        XCTAssertEqual(files, 6)
        XCTAssertEqual(rec.texts.count, chunks, "every chunk must be embedded via the server transport")

        let before = rec.texts.count
        _ = await index.search(query: "deploy friday")
        XCTAssertEqual(rec.texts.count, before + 1, "the query must use the same (GPU) embedder as the index")
        XCTAssertTrue(rec.texts.last?.contains("deploy friday") ?? false)
    }
}

/// Thread-safe call counter for the embedder factory (invoked off-main by
/// the indexing workers).
private final class EmbedderFactoryCounter: @unchecked Sendable {
    private let lock = NSLock()
    private var _count = 0
    var count: Int { lock.lock(); defer { lock.unlock() }; return _count }
    func increment() { lock.lock(); _count += 1; lock.unlock() }
}

/// Deterministic bag-of-words embedder: a tiny fixed vocabulary mapped to
/// orthogonal-ish dimensions, so vector search ranks by real token overlap
/// without depending on the OS NLEmbedding asset being present on CI.
private struct FakeEmbedder: TextEmbedding {
    private static let vocab = ["cat", "mat", "stocks", "fell", "deploy", "friday",
                                "pricing", "budget", "alice", "bob"]
    func vector(for text: String) -> [Double]? {
        let words = Set(text.lowercased().components(separatedBy: CharacterSet.alphanumerics.inverted))
        var v = Self.vocab.map { words.contains($0) ? 1.0 : 0.0 }
        v.append(0.001) // never the zero vector
        return v
    }

    func vectors(for texts: [String]) async -> [[Double]?] {
        texts.map { vector(for: $0) }
    }
}

/// Thread-safe recorder for ServerEmbedding's transport closure: batch
/// sizes, embedded texts, and an optional failure mode.
private final class TransportRecorder: @unchecked Sendable {
    private let lock = NSLock()
    private var _batches: [[String]] = []
    var failAll = false
    var batches: [[String]] { lock.lock(); defer { lock.unlock() }; return _batches }
    var texts: [String] { batches.flatMap { $0 } }

    private func record(_ texts: [String]) -> Bool {
        lock.lock(); defer { lock.unlock() }
        _batches.append(texts)
        return failAll
    }

    func transport(_ texts: [String], _ model: String) async throws -> [[Double]] {
        if record(texts) { throw URLError(.cannotConnectToHost) }
        // Distinct per-text vectors so order round-trips visibly.
        return texts.map { [Double($0.count), 1.0] }
    }
}
