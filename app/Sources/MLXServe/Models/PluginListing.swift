import Foundation

/// Models ▸ Plugins as a list and a detail (`PluginsPane`): every plugin once, installed ones first with their
/// catalog entry, the state its row shows, the packs it serves, and which page the detail column shows.
enum PluginListing {
    struct Item: Identifiable, Equatable {
        let name: String
        var receipt: OutsidePlugins.Receipt?
        var entry: PluginCatalog.Entry?
        var id: String { name }
        var kind: String { entry?.kind ?? receipt?.kind ?? "" }
    }

    typealias Lists = (installed: [Item], available: [Item])

    static func items(receipts: [OutsidePlugins.Receipt], catalog: [PluginCatalog.Entry]) -> Lists {
        let byName = Dictionary(catalog.map { ($0.name, $0) }, uniquingKeysWith: { first, _ in first })
        let installed = receipts.map { Item(name: $0.name, receipt: $0, entry: byName[$0.name]) }
        let names = Set(receipts.map(\.name))
        return (installed, catalog.filter { !names.contains($0.name) }.map { Item(name: $0.name, entry: $0) })
    }

    enum State: Equatable {
        case on
        case off
        case updateAvailable(String)
        case available
        case needsNewer(String)
    }

    /// An installed plugin runs only when loading is on and its own switch is; an update shows only when this server
    /// can run it.
    static func state(_ item: Item, switches: OutsidePlugins.Switches, running: String?) -> State {
        let compatible = item.entry.map { PluginCatalog.runs(on: running, requirement: $0.requiresMlxServe) } ?? true
        guard let receipt = item.receipt else {
            return compatible ? .available : .needsNewer(item.entry?.requiresMlxServe ?? "")
        }
        if let entry = item.entry, compatible, PluginCatalog.compare(receipt.version, entry.version) == .orderedAscending {
            return .updateAvailable(entry.version)
        }
        return switches.enabled && switches.isOn(item.name) ? .on : .off
    }

    struct Pack: Equatable, Identifiable {
        let text: String
        let repo: String?
        var id: String { repo ?? text }
    }

    /// A catalog `models` line, `<what> (<org>/<name>)`: the trailing parentheses are a repo only in that shape.
    static func packs(_ entry: PluginCatalog.Entry) -> [Pack] {
        entry.models.map { line in
            guard line.hasSuffix(")"), let open = line.range(of: " (", options: .backwards) else { return Pack(text: line, repo: nil) }
            let inside = String(line[open.upperBound..<line.index(before: line.endIndex)])
            guard inside.range(of: #"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$"#, options: .regularExpression) != nil else {
                return Pack(text: line, repo: nil)
            }
            return Pack(text: String(line[..<open.lowerBound]), repo: inside)
        }
    }

    enum Selection: Hashable {
        case plugin(String)
        case agent
    }

    /// The selection, kept on a listed plugin: a removed or unlisted one falls back to the first row, and with nothing
    /// listed the agent page is what is left.
    static func resolve(_ selection: Selection?, _ lists: Lists) -> Selection {
        let names = (lists.installed + lists.available).map(\.name)
        switch selection {
        case .agent: return .agent
        case .plugin(let name) where names.contains(name): return .plugin(name)
        default: return names.first.map(Selection.plugin) ?? .agent
        }
    }
}
