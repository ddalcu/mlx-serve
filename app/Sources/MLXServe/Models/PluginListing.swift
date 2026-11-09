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
        var title: String { entry.flatMap { $0.title.isEmpty ? nil : $0.title } ?? name }
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
        /// Installed, but the server refused it as built for another mlx-serve: a rebuild with this kit loads it.
        case needsRebuild
        /// The running server has not applied the switch or the options: it loads plugins only at its start.
        case restartNeeded
    }

    /// An installed plugin runs only when loading is on and its own switch is; an update shows only when this server
    /// can run it. `serverUp`: `server` is what a running server reported, so a plugin missing from it is not loaded.
    static func state(_ item: Item, switches: OutsidePlugins.Switches, running: String?,
                      server: [String: PluginCatalog.ServerStatus] = [:], serverUp: Bool = false) -> State {
        let compatible = item.entry.map { PluginCatalog.runs(on: running, requirement: $0.requiresMlxServe) } ?? true
        guard let receipt = item.receipt else {
            return compatible ? .available : .needsNewer(item.entry?.requiresMlxServe ?? "")
        }
        if let entry = item.entry, compatible, PluginCatalog.compare(receipt.version, entry.version) == .orderedAscending {
            return .updateAvailable(entry.version)
        }
        if server[item.name]?.rebuild == true { return .needsRebuild }
        let wanted = switches.enabled && switches.isOn(item.name)
        if serverUp, wanted != (server[item.name]?.loaded == true) { return .restartNeeded }
        if serverUp, wanted, OutsidePlugins.applied(receipt, switches) != (server[item.name]?.options ?? [:]) { return .restartNeeded }
        return wanted ? .on : .off
    }

    enum StepStatus: Equatable { case waiting, active, done }

    struct Step: Equatable {
        let title: String
        let status: StepStatus
    }

    /// An install as the three steps a person follows; the log stays behind Details.
    static func installSteps(_ phase: PluginManager.Phase) -> [Step] {
        let current: Int
        var waitingOnUser = false
        switch phase {
        case .fetching: current = 0
        case .awaitingConfirmation: current = 1; waitingOnUser = true
        case .building: current = 1
        case .installing: current = 2
        case .done: current = 3
        case .idle, .failed: return []
        }
        let titles = [L10n.text("Download the source"),
                      L10n.text("Compile and test it on this Mac"),
                      L10n.text("Install and turn it on")]
        return titles.enumerated().map { i, title in
            Step(title: title, status: i < current ? .done : (i == current && !waitingOnUser ? .active : .waiting))
        }
    }

    /// The installed plugins a rebuild would bring back that the catalog can build (Rebuild all).
    static func rebuildable(_ installed: [Item], server: [String: PluginCatalog.ServerStatus]) -> [PluginCatalog.Entry] {
        installed.compactMap { server[$0.name]?.rebuild == true ? $0.entry : nil }
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
