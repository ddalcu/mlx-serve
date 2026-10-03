import Foundation

/// Pure text matcher behind the Settings screen's filter field.
///
/// Every `SettingsRow` asks this whether it should stay on screen, and every
/// `SettingsSection` asks it whether its own header matches (a header hit shows
/// the whole section). Keeping the decision here — rather than inline in the
/// SwiftUI bodies — is what makes the filter testable; see `SettingsSearchTests`.
///
/// Semantics: the query is split on whitespace and *every* token must appear
/// somewhere in the row's haystack (its label plus its description), so
/// "prefix cache" narrows rather than widens. Matching is case- and
/// diacritic-insensitive, and curly/straight apostrophes are interchangeable
/// because the shipped explainers use the curly form ("the model's window")
/// while users type the straight one.
enum SettingsSearch {

    /// True when `query` should keep a row whose searchable text is `haystack`
    /// (conventionally `[title, explainer]`). A blank query keeps everything.
    static func matches(query: String, in haystack: [String]) -> Bool {
        let needles = tokens(query)
        if needles.isEmpty { return true }
        let hay = normalize((haystack + haystack.map(L10n.text)).joined(separator: " "))
        if hay.isEmpty { return false }
        return needles.allSatisfy { hay.contains($0) }
    }

    /// The query split into normalized, non-empty search tokens. Empty for a
    /// blank query.
    static func tokens(_ query: String) -> [String] {
        normalize(query)
            .split(whereSeparator: { $0.isWhitespace })
            .map(String.init)
    }

    /// Lowercase, diacritic-folded, with typographic apostrophes flattened to
    /// the ASCII form so `model's` and `model’s` are the same string.
    static func normalize(_ s: String) -> String {
        s.replacingOccurrences(of: "\u{2019}", with: "'")
            .folding(options: [.diacriticInsensitive, .caseInsensitive], locale: nil)
    }

    /// How one Settings section behaves under the current filter.
    ///
    /// Matched on the TITLE only. Section subtitles are long prose that name
    /// half the app — "MLX Performance" is subtitled "…and the cross-request hot
    /// prefix cache…", so including it would make a search for "prefix cache"
    /// open that whole section, Concurrent requests and KV quantization
    /// included. Rows carry the searchable substance; the title is just the
    /// wholesale switch.
    static func section(query: String, title: String) -> SectionFilter {
        let headerMatches = matches(query: query, in: [title])
        return SectionFilter(
            filtering: !tokens(query).isEmpty,
            headerMatches: headerMatches,
            // A title hit shows the entire section — its rows must not filter
            // themselves out. Searching "telegram" should surface the bot-token
            // row even though that row's own text never says "telegram".
            childQuery: headerMatches ? "" : query
        )
    }

    /// The two decisions a `SettingsSection` makes: what query to hand its rows,
    /// and whether to hide its own header + card.
    struct SectionFilter: Equatable {
        /// A filter is active (the query has at least one token).
        let filtering: Bool
        /// The section's own title/subtitle satisfied the query.
        let headerMatches: Bool
        /// The query the section's rows should filter themselves against.
        let childQuery: String

        /// Hide the header and card when a filter is active and nothing inside
        /// survived it.
        ///
        /// Deliberately a pure function of the *current* row count rather than a
        /// latch: the view keeps a collapsed section's rows in the tree (they
        /// render as nothing but keep publishing their count) so that editing
        /// the query can bring the section straight back. A sticky collapse
        /// would strand it hidden forever.
        func collapsed(visibleRows: Int) -> Bool {
            filtering && !headerMatches && visibleRows == 0
        }
    }
}
