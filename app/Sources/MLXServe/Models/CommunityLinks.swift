import Foundation

/// The About section's outbound links, as data.
///
/// Pure so the destinations are testable and live in one place: a `Link` whose
/// URL is spelled at the call site is a dead button that compiles, renders and
/// looks correct. The releases URL is derived from `UpdateChecker.repo` — the
/// same constant the updater fetches against — so the pane can never point at
/// a repo the app no longer updates from.
enum CommunityLinks {

    struct Item: Identifiable, Equatable {
        let id: String
        let title: String
        let explainer: String
        /// The button's text. Carries the ↗ suffix used elsewhere in Settings
        /// for anything that leaves the app.
        let actionLabel: String
        let url: URL
    }

    static let all: [Item] = [
        Item(
            id: "releases",
            title: "Release notes",
            explainer: "What changed in each version, including the one you're running.",
            actionLabel: "View Releases ↗",
            url: URL(string: "https://github.com/\(UpdateChecker.repo)/releases")!
        ),
        Item(
            id: "star",
            title: "Star the project on GitHub",
            explainer: "mlx-serve is free and open source. A star helps other people find it.",
            actionLabel: "Open GitHub ↗",
            url: URL(string: "https://github.com/\(UpdateChecker.repo)")!
        ),
        Item(
            id: "x",
            title: "Follow @ddalcu on X",
            explainer: "Release notes, benchmarks and what's coming next. Questions and bug reports are welcome there too.",
            actionLabel: "Open X ↗",
            url: URL(string: "https://x.com/ddalcu")!
        ),
    ]
}
