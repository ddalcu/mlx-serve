import XCTest
@testable import MLXCore

/// Unit tests for `SettingsSearch` — the pure matcher behind the Settings
/// screen's filter field. SwiftUI rows/sections can't be exercised directly,
/// so all of the filtering *decision* lives here and is pinned by these tests;
/// the views only ask "does this row match?" and hide themselves.
final class SettingsSearchTests: XCTestCase {

    // MARK: - Empty query shows everything

    func testBlankQueryMatchesEverything() {
        XCTAssertTrue(SettingsSearch.matches(query: "", in: ["Host", "Bind address"]))
        XCTAssertTrue(SettingsSearch.matches(query: "   ", in: ["Host", "Bind address"]))
        XCTAssertTrue(SettingsSearch.matches(query: "\n\t", in: []))
    }

    // MARK: - Label matching

    func testMatchesOnLabelCaseInsensitively() {
        let hay = ["Context size", "How much of the model's context window to use."]
        XCTAssertTrue(SettingsSearch.matches(query: "context", in: hay))
        XCTAssertTrue(SettingsSearch.matches(query: "CONTEXT", in: hay))
        XCTAssertTrue(SettingsSearch.matches(query: "CoNtExT SiZe", in: hay))
    }

    func testMatchesOnPartialWord() {
        // Typing "vis" should already surface "Disable vision".
        XCTAssertTrue(SettingsSearch.matches(query: "vis", in: ["Disable vision", "Skip the SigLIP encoder."]))
    }

    // MARK: - Description matching

    func testMatchesOnExplainerNotJustTitle() {
        // "Prometheus" appears only in the explainer of the metrics toggle.
        let hay = ["Metrics", "Expose Prometheus metrics at /metrics and a live panel."]
        XCTAssertTrue(SettingsSearch.matches(query: "prometheus", in: hay))
    }

    func testMatchesOnSlashPrefixedPaths() {
        let hay = ["Metrics", "Expose Prometheus metrics at /metrics."]
        XCTAssertTrue(SettingsSearch.matches(query: "/metrics", in: hay))
    }

    // MARK: - Multi-token queries are AND, and may span label + description

    func testAllTokensMustMatch() {
        let hay = ["Prefix cache entries", "How many KV snapshots to keep across requests."]
        XCTAssertTrue(SettingsSearch.matches(query: "prefix cache", in: hay))
        XCTAssertTrue(SettingsSearch.matches(query: "cache prefix", in: hay), "token order must not matter")
        XCTAssertFalse(SettingsSearch.matches(query: "prefix telegram", in: hay))
    }

    func testTokensMaySpanLabelAndExplainer() {
        // "kv" is in the explainer, "prefix" in the title — a query using both must hit.
        let hay = ["Prefix cache entries", "How many KV snapshots to keep across requests."]
        XCTAssertTrue(SettingsSearch.matches(query: "prefix kv", in: hay))
    }

    func testCollapsesRepeatedWhitespaceBetweenTokens() {
        let hay = ["Prefix cache entries", "KV snapshots."]
        XCTAssertTrue(SettingsSearch.matches(query: "  prefix    cache  ", in: hay))
    }

    // MARK: - Misses

    func testNonMatchingQueryReturnsFalse() {
        XCTAssertFalse(SettingsSearch.matches(query: "telegram", in: ["Host", "Bind address."]))
    }

    func testEmptyHaystackNeverMatchesANonBlankQuery() {
        XCTAssertFalse(SettingsSearch.matches(query: "host", in: []))
        XCTAssertFalse(SettingsSearch.matches(query: "host", in: ["", "  "]))
    }

    // MARK: - Normalization

    func testDiacriticsAreFolded() {
        XCTAssertTrue(SettingsSearch.matches(query: "resume", in: ["Résumé download", ""]))
        XCTAssertTrue(SettingsSearch.matches(query: "résumé", in: ["Resume download", ""]))
    }

    func testCurlyAndStraightApostrophesAreInterchangeable() {
        // Explainers in ServerOptions use curly apostrophes ("model's"); a user
        // types the straight one.
        XCTAssertTrue(SettingsSearch.matches(query: "model's", in: ["Context size", "Pegs it to the model\u{2019}s window."]))
        XCTAssertTrue(SettingsSearch.matches(query: "model\u{2019}s", in: ["Context size", "Pegs it to the model's window."]))
    }

    // MARK: - Real metadata is reachable (class guard)

    /// Every server-flag row the Settings screen renders must be findable by
    /// typing its own title. Guards against a field whose title/explainer is
    /// empty, or a future normalization change that drops characters.
    func testEverySeverFlagFieldIsFoundByItsOwnTitle() {
        for (key, field) in ServerOptions.serverFlagFields {
            XCTAssertFalse(field.title.isEmpty, "\(key) has an empty title")
            XCTAssertTrue(
                SettingsSearch.matches(query: field.title, in: [field.title, field.explainer]),
                "field '\(key)' (title: \(field.title)) is not findable by its own title"
            )
        }
    }

    // MARK: - Section filtering

    func testBlankQueryNeverCollapsesASection() {
        let f = SettingsSearch.section(query: "", title: "Server")
        XCTAssertFalse(f.collapsed(visibleRows: 0))
        XCTAssertEqual(f.childQuery, "")
    }

    /// A hit on the section TITLE shows the WHOLE section: searching "telegram"
    /// must reveal the bot-token row, whose own text never says "telegram".
    func testTitleHitClearsTheQueryForChildRows() {
        let f = SettingsSearch.section(query: "telegram", title: "Messaging — Telegram bot")
        XCTAssertTrue(f.headerMatches)
        XCTAssertEqual(f.childQuery, "", "children of a matching title must see a blank query")
        XCTAssertFalse(f.collapsed(visibleRows: 0))

        // A row that would otherwise be filtered out now survives.
        XCTAssertTrue(SettingsSearch.matches(query: f.childQuery, in: ["Bot token", "Paste the token @BotFather gives you."]))
    }

    /// Section SUBTITLES are long prose that name half the app ("…and the
    /// cross-request hot prefix cache…"). Matching a section on its subtitle
    /// would dump every unrelated row in that section on screen — live-caught
    /// with "prefix cache" pulling in Concurrent requests and KV quantization.
    /// Only the title opens a section wholesale.
    func testSubtitleTextDoesNotOpenTheWholeSection() {
        let f = SettingsSearch.section(query: "prefix cache", title: "MLX Performance")
        XCTAssertFalse(f.headerMatches, "a subtitle word must not open the section")
        XCTAssertEqual(f.childQuery, "prefix cache", "rows must keep filtering themselves")

        // The unrelated neighbours in that section stay hidden…
        XCTAssertFalse(SettingsSearch.matches(
            query: f.childQuery,
            in: ["Concurrent requests", "Continuous batching: how many chat requests share one forward pass."]))
        // …while the rows the user asked for survive.
        XCTAssertTrue(SettingsSearch.matches(
            query: f.childQuery,
            in: ["Prefix cache entries", "Hot prefix cache size: how many KV snapshots to keep."]))
    }

    func testTitleMissPassesTheQueryThroughToRows() {
        let f = SettingsSearch.section(query: "port", title: "Voice")
        XCTAssertFalse(f.headerMatches)
        XCTAssertEqual(f.childQuery, "port")
    }

    func testSectionCollapsesOnlyWhenNothingInsideSurvived() {
        let f = SettingsSearch.section(query: "port", title: "Voice")
        XCTAssertTrue(f.collapsed(visibleRows: 0))
        XCTAssertFalse(f.collapsed(visibleRows: 1))
    }

    /// Collapse must be a pure function of the CURRENT row count, never sticky.
    /// The view keeps a collapsed section's rows in the tree precisely so the
    /// count can climb back above zero; if collapse latched, editing the query
    /// could never bring the section back.
    func testCollapseIsNotSticky() {
        let f = SettingsSearch.section(query: "cache", title: "Voice")
        XCTAssertTrue(f.collapsed(visibleRows: 0))
        XCTAssertFalse(f.collapsed(visibleRows: 2), "a section must re-open as soon as a row matches again")
        XCTAssertTrue(f.collapsed(visibleRows: 0))
    }

    func testWhitespaceOnlyQueryIsTreatedAsNoFilter() {
        let f = SettingsSearch.section(query: "   ", title: "Voice")
        XCTAssertFalse(f.collapsed(visibleRows: 0))
    }

    /// A couple of realistic searches against the shipped metadata — the thing
    /// a user actually types.
    func testRealisticQueriesHitTheRightFields() {
        func hay(_ key: String) -> [String] {
            guard let f = ServerOptions.serverFlagFields[key] else { return [] }
            return [f.title, f.explainer]
        }
        XCTAssertTrue(SettingsSearch.matches(query: "port", in: hay("port")))
        XCTAssertTrue(SettingsSearch.matches(query: "api key", in: hay("apiKey")))
        XCTAssertTrue(SettingsSearch.matches(query: "devices", in: hay("apiKey")), "explainer text must be searchable")
        XCTAssertFalse(SettingsSearch.matches(query: "devices", in: hay("port")))
    }
}
