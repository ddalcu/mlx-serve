import AppKit

/// The Cluster page's Thunderbolt buttons. Pairing gives the cable its address through
/// `ThunderboltAddressJob` (the first time, behind one password prompt), sets the role, and
/// restarts the server when this Mac leads; the Mac that joins dials the leader's end of the
/// cable, so it needs no name.
@MainActor
enum ThunderboltPairing {
    /// This Mac leads a pair that runs `model` (a path from the library).
    static func lead(model: String, appState: AppState, ports: [RdmaPort]) async -> String? {
        if let err = await giveAddress(.host, ports: ports) { return err }
        var o = appState.serverOptions
        o.clusterRole = .host
        o.clusterDevice = ""
        o.clusterCoordinator = ""
        o.clusterModelPath = model
        appState.serverOptions = o
        restartServer(appState)
        return nil
    }

    /// This Mac joins the Mac that leads at the other end of the cable; a Mac that led
    /// restarts its server without the pair.
    static func join(appState: AppState, ports: [RdmaPort]) async -> String? {
        if let err = await giveAddress(.worker, ports: ports) { return err }
        let led = appState.serverOptions.clusterRole == .host
        var o = appState.serverOptions
        o.clusterRole = .worker
        o.clusterDevice = ""
        o.clusterCoordinator = ""
        appState.serverOptions = o
        if led { restartServer(appState) }
        return nil
    }

    /// The address a restart took away, for the role this Mac already has.
    static func reconnect(_ role: ClusterRole, appState: AppState, ports: [RdmaPort]) async -> String? {
        if let err = await giveAddress(role, ports: ports) { return err }
        if role == .host { restartServer(appState) }
        return nil
    }

    static func unpair(appState: AppState) {
        let led = appState.serverOptions.clusterRole == .host
        appState.serverOptions.clusterRole = .off
        _ = ThunderboltAddressJob.apply(.off)
        if led { restartServer(appState) }
    }

    /// Sets the role on the cable and waits for the job to give the port its address: the
    /// server listens on it as soon as it restarts.
    private static func giveAddress(_ role: ClusterRole, ports: [RdmaPort]) async -> String? {
        guard ports.contains(where: \.active) else { return L10n.text("Connect the Thunderbolt cable first.") }
        if let err = ThunderboltAddressJob.apply(role) { return err }
        for _ in 0..<40 {
            let up = await Task.detached { ClusterSetup.ports() }.value.first(where: \.active)
            if let up, ClusterSetup.ipv4(on: up.interface) == ClusterSetup.linkAddress(role) { return nil }
            try? await Task.sleep(for: .milliseconds(250))
        }
        return L10n.text("The cable did not get its address. Unplug it, plug it back into a Thunderbolt 5 port and try again.")
    }

    private static func restartServer(_ appState: AppState) {
        appState.server.stop()
        appState.startServer(loadingSelection: false)
    }
}

/// While this Mac has no pair, watches for a Mac on the network that leads one and waits:
/// with a Thunderbolt cable in, MLX Serve comes forward and asks to join it, once per leader.
@MainActor
final class PairInvites {
    private var asked: Set<String> = []
    private var loop: Task<Void, Never>?

    func start(appState: AppState) {
        loop = Task { [weak self, weak appState] in
            while !Task.isCancelled {
                try? await Task.sleep(for: .seconds(10))
                guard let self, let appState else { return }
                await self.check(appState)
            }
        }
    }

    /// The leader to ask about: one still waiting that was not asked about while it waited.
    nonisolated static func next(waiting: Set<String>, asked: inout Set<String>) -> String? {
        asked.formIntersection(waiting)
        return waiting.subtracting(asked).sorted().first
    }

    private func check(_ appState: AppState) async {
        guard appState.serverOptions.clusterRole == .off else { asked.removeAll(); return }
        await appState.server.refreshModelsIfMoved()
        let waiting = Set(appState.server.lanPeers.filter(\.value.waitsForPartner).keys)
        guard let leader = Self.next(waiting: waiting, asked: &asked) else { return }
        let ports = await Task.detached { ClusterSetup.ports() }.value
        guard ports.contains(where: \.active) else { return }
        asked.insert(leader)

        NSApplication.shared.activate(ignoringOtherApps: true)
        let alert = NSAlert()
        alert.messageText = L10n.format("%@ wants to pair over Thunderbolt", leader)
        alert.informativeText = L10n.text("Join it, and the two Macs work on the same jobs together.")
        alert.addButton(withTitle: L10n.text("Join"))
        alert.addButton(withTitle: L10n.text("Not now"))
        guard alert.runModal() == .alertFirstButtonReturn, appState.serverOptions.clusterRole == .off else { return }
        if let err = await ThunderboltPairing.join(appState: appState, ports: ports) {
            let failed = NSAlert()
            failed.messageText = L10n.text("Could not join")
            failed.informativeText = err
            failed.runModal()
        }
    }
}
