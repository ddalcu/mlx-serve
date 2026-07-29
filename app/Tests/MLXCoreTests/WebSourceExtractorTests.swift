import XCTest
@testable import MLXCore

/// Recovers the pages a web-search answer was built from, so the reply can cite
/// them.
///
/// The sources are only in the HIDDEN tool-result messages — the transcript
/// filters those out, so without this the user reads a confident answer with no
/// way to see where it came from.
final class WebSourceExtractorTests: XCTestCase {

    /// A real `WebSearchHandler` payload: a header line, then title / url /
    /// snippet triples separated by blank lines.
    private let searchOutput = """
    Search results for 'sveltekit blog':

    Introduction • SvelteKit Docs
    https://svelte.dev/docs/kit/introduction
    SvelteKit is a framework for rapidly developing robust apps.

    Build a SvelteKit Markdown Blog
    https://joyofcode.xyz/sveltekit-markdown-blog
    Learn how to build a blog using SvelteKit and MDsveX.

    YusufCeng1z/sveltekit-shadcn-blog-starter
    https://github.com/YusufCeng1z/sveltekit-shadcn-blog-starter
    A blog starter template.
    """

    // MARK: - Parsing one result payload

    func testParsesEveryTitleUrlPair() {
        let out = WebSourceExtractor.sources(fromSearchOutput: searchOutput)
        XCTAssertEqual(out.count, 3)
        XCTAssertEqual(out[0].title, "Introduction • SvelteKit Docs")
        XCTAssertEqual(out[0].url, "https://svelte.dev/docs/kit/introduction")
        XCTAssertEqual(out[2].title, "YusufCeng1z/sveltekit-shadcn-blog-starter")
    }

    func testDomainIsStrippedOfSchemeAndWww() {
        // The domain is the row's right-hand label; "https://www." in there is
        // noise on every single row.
        XCTAssertEqual(WebSourceExtractor.sources(fromSearchOutput: searchOutput)[0].domain, "svelte.dev")
        XCTAssertEqual(WebSource(title: "t", url: "https://www.example.co.uk/a/b").domain, "example.co.uk")
        XCTAssertEqual(WebSource(title: "t", url: "http://Example.COM").domain, "example.com")
    }

    func testHeaderLineIsNotMistakenForAResult() {
        let out = WebSourceExtractor.sources(fromSearchOutput: searchOutput)
        XCTAssertFalse(out.contains { $0.title.hasPrefix("Search results for") })
    }

    func testTheFallbackPageDumpYieldsNoSources() {
        // When DuckDuckGo returns nothing the handler falls back to raw page
        // text. Presenting that as "sources" would be inventing citations.
        let dump = "Search results for 'x':\n\nSome page text\nmore text\nand more"
        XCTAssertTrue(WebSourceExtractor.sources(fromSearchOutput: dump).isEmpty)
    }

    func testABlockWithoutAUrlIsSkipped() {
        let mixed = """
        Search results for 'x':

        Real Title
        https://real.example/page
        snippet

        Not a result
        just some prose
        """
        let out = WebSourceExtractor.sources(fromSearchOutput: mixed)
        XCTAssertEqual(out.map(\.url), ["https://real.example/page"])
    }

    func testNonHttpUrlsAreRejected() {
        // A javascript: or file: line must never become a clickable source row.
        let payload = "Search results for 'x':\n\nTitle\njavascript:alert(1)\nsnippet"
        XCTAssertTrue(WebSourceExtractor.sources(fromSearchOutput: payload).isEmpty)
    }

    func testTruncatedOutputStillYieldsWhatItHas() {
        // Tool output is capped before it reaches the transcript, so the last
        // block is routinely cut mid-snippet.
        let cut = """
        Search results for 'x':

        First
        https://a.example/1
        snippet

        Second
        https://b.example/2
        """
        XCTAssertEqual(WebSourceExtractor.sources(fromSearchOutput: cut).count, 2)
    }

    // MARK: - Attributing sources to a reply

    private func message(_ role: ChatMessage.Role, _ content: String,
                         tool: String? = nil) -> ChatMessage {
        var m = ChatMessage(role: role, content: content)
        if let tool {
            m.toolName = tool
            m.toolCallId = "call_\(tool)"
        }
        return m
    }

    func testSourcesComeFromTheSameTurnOnly() {
        // An earlier turn's search must not be cited by a later, unrelated
        // answer — the walk stops at the previous user message.
        let messages = [
            message(.user, "old question"),
            message(.system, searchOutput, tool: "webSearch"),
            message(.assistant, "old answer"),
            message(.user, "new question"),
            message(.assistant, "new answer"),
        ]
        XCTAssertTrue(WebSourceExtractor.sources(forMessageId: messages[4].id, in: messages).isEmpty)
        XCTAssertEqual(WebSourceExtractor.sources(forMessageId: messages[2].id, in: messages).count, 3)
    }

    func testSourcesAcrossSeveralSearchesInOneTurnAreCombined() {
        let second = """
        Search results for 'more':

        Another
        https://c.example/x
        snippet
        """
        let messages = [
            message(.user, "q"),
            message(.system, searchOutput, tool: "webSearch"),
            message(.system, second, tool: "webSearch"),
            message(.assistant, "answer"),
        ]
        XCTAssertEqual(WebSourceExtractor.sources(forMessageId: messages[3].id, in: messages).count, 4)
    }

    func testDuplicateUrlsAreCollapsedKeepingFirstOrder() {
        // Two searches routinely surface the same top hit; listing it twice
        // makes the count wrong.
        let repeated = "Search results for 'x':\n\nIntroduction • SvelteKit Docs\nhttps://svelte.dev/docs/kit/introduction\ns"
        let messages = [
            message(.user, "q"),
            message(.system, searchOutput, tool: "webSearch"),
            message(.system, repeated, tool: "webSearch"),
            message(.assistant, "answer"),
        ]
        let out = WebSourceExtractor.sources(forMessageId: messages[3].id, in: messages)
        XCTAssertEqual(out.count, 3)
        XCTAssertEqual(out[0].url, "https://svelte.dev/docs/kit/introduction")
    }

    func testOtherToolsAreIgnored() {
        let messages = [
            message(.user, "q"),
            message(.system, searchOutput, tool: "readFile"),
            message(.assistant, "answer"),
        ]
        XCTAssertTrue(WebSourceExtractor.sources(forMessageId: messages[2].id, in: messages).isEmpty)
    }

    func testUnknownMessageIdYieldsNothing() {
        XCTAssertTrue(WebSourceExtractor.sources(forMessageId: UUID(), in: []).isEmpty)
    }

    func testATurnWithNoSearchHasNoChip() {
        let messages = [message(.user, "hi"), message(.assistant, "hello")]
        XCTAssertTrue(WebSourceExtractor.sources(forMessageId: messages[1].id, in: messages).isEmpty)
    }
}
