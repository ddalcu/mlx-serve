import Foundation

/// One row of the ⌘L model switcher.
///
/// `tag` is a `ChatModelSelection` tag — a local path, or `lan:<id@peer>` — so
/// the palette never decides what picking a row MEANS. That answer already
/// lives in one place, shared with the tray and the composer's pill, and a
/// per-surface copy is how a picker silently stops honouring a LAN selection.
struct ModelPaletteRow: Identifiable, Equatable {
    let tag: String
    /// The readable name (`ModelDisplayName`), same as the pill shows.
    let title: String
    /// The line under it: quant, size, engine, or the Mac it lives on.
    let detail: String
    /// Group heading — a `LocalModelSource` title, or `networkSection`.
    let section: String
    /// Everything a query is matched against, lowercased once at build time.
    /// Deliberately NOT the filesystem path (see `ModelPalette.filtered`).
    let searchText: String

    var id: String { tag }
}

/// The ⌘L model switcher: which models this chat can be pointed at, filtered by
/// typing, chosen with the arrow keys.
///
/// It offers exactly what the composer's pill offers — same `isChatPickable`
/// filter, same `lanModels(capability: "chat")` question, same group order —
/// because two pickers a keystroke apart that list different models is the
/// worse half of having two.
enum ModelPalette {

    /// Heading for models shared by other Macs, spelled as the pill spells it.
    static let networkSection = "On Your Network"

    // MARK: - Rows

    static func rows(local: [LocalModel], lan: [ModelInfo]) -> [ModelPaletteRow] {
        let pickable = local.filter(\.isChatPickable)
        // The pill's own duplicate rule: two rows reading identically make the
        // list a coin flip, so a shared label earns its engine in the detail.
        let dupNames = LocalModel.duplicateNames(in: pickable)
        var out: [ModelPaletteRow] = []

        for source in LocalModelSource.allCases {
            for model in pickable where model.source == source {
                out.append(row(for: model, section: source.sectionTitle,
                               needsEngine: dupNames.contains(model.displayLabel)))
            }
        }
        for peer in lan where peer.lanAdvertises("chat") {
            out.append(row(forLan: peer))
        }
        return out
    }

    private static func row(for model: LocalModel, section: String,
                            needsEngine: Bool) -> ModelPaletteRow {
        var detail: [String] = []
        // A GGUF repo is a shelf: the quant IS which model this row loads, so
        // it comes first.
        if let quantFile = model.quantFile {
            detail.append(model.quantLabel ?? DownloadManager.quantLabel(forFilename: quantFile))
        } else if let quant = model.quantization {
            detail.append(quant)
        }
        detail.append(model.sizeFormatted)
        if needsEngine { detail.append(model.engine.shortLabel) }

        let title = ModelDisplayName.pretty(model.name)
        return ModelPaletteRow(
            tag: ChatModelSelection.tag(localPath: model.path, lanChatModelId: nil),
            title: title,
            detail: detail.joined(separator: " · "),
            section: section,
            // The repo id rides along: "e4b", "4bit" and "mlx-community" are
            // all things people type, and the readable title has dropped some
            // of them on purpose. The section HEADING does not: it is true of
            // every row in its group, so matching it is the same "answers all
            // of them" failure as matching the path — and "MLX-Serve Models"
            // shares a word with most of the ids under it.
            searchText: [title, model.name, detail.joined(separator: " ")]
                .joined(separator: " ").lowercased())
    }

    private static func row(forLan peer: ModelInfo) -> ModelPaletteRow {
        let title = ModelDisplayName.pretty(peer.name)
        let host = peer.lanPeer ?? ""
        let detail = host.isEmpty ? "On your network" : "On \(host)"
        return ModelPaletteRow(
            tag: ChatModelSelection.tag(localPath: "", lanChatModelId: peer.name),
            title: title,
            detail: detail,
            section: networkSection,
            searchText: [title, peer.name, host]
                .joined(separator: " ").lowercased())
    }

    // MARK: - Filtering

    /// Rows matching `query`. EVERY whitespace-separated word must land, so a
    /// second word narrows instead of starting a new search ("qwen 8bit" is how
    /// you tell two quants of one repo apart); an empty query hides nothing.
    ///
    /// Matched against `searchText`, which excludes the on-disk PATH on
    /// purpose: every model under `~/.mlx-serve/models` shares most of one, so
    /// matching it makes "models", "users" and the account name match
    /// everything — a filter that answers "all of them" has stopped working.
    static func filtered(_ rows: [ModelPaletteRow], query: String) -> [ModelPaletteRow] {
        let words = query.lowercased().split(whereSeparator: \.isWhitespace)
        guard !words.isEmpty else { return rows }
        return rows.filter { row in words.allSatisfy { row.searchText.contains($0) } }
    }

    // MARK: - Selection

    /// Where the selection opens: on the model already answering, so ⌘L then
    /// Return is a no-op rather than a switch to whatever sorts first. A
    /// current model the list does not hold (a peer that went away) falls to
    /// the top; an empty list selects nothing.
    static func selection(in rows: [ModelPaletteRow], current: String) -> Int? {
        guard !rows.isEmpty else { return nil }
        return rows.firstIndex { $0.tag == current } ?? 0
    }

    /// Arrow-key movement. It CLAMPS: wrapping means holding ↓ past the last
    /// model puts you silently back on the first, and Return then loads a model
    /// nobody was looking at. No selection yet ⇒ the first key lands on the top
    /// row, which is what the list already highlights.
    static func move(_ index: Int?, by delta: Int, count: Int) -> Int? {
        guard count > 0 else { return nil }
        guard let index else { return 0 }
        return min(max(index + delta, 0), count - 1)
    }

    /// The selection as it must be READ, after typing has shrunk the list under
    /// it — an index into rows that no longer exist is the out-of-bounds trap.
    static func clamped(_ index: Int?, count: Int) -> Int? {
        guard count > 0, let index else { return nil }
        return min(max(index, 0), count - 1)
    }

    /// The tag a row index names, or nil when it names none.
    static func tag(at index: Int?, in rows: [ModelPaletteRow]) -> String? {
        guard let index, rows.indices.contains(index) else { return nil }
        return rows[index].tag
    }
}
