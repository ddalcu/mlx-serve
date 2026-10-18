import Foundation

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

    /// This Mac joins the Mac that leads at the other end of the cable.
    static func join(appState: AppState, ports: [RdmaPort]) async -> String? {
        if let err = await giveAddress(.worker, ports: ports) { return err }
        var o = appState.serverOptions
        o.clusterRole = .worker
        o.clusterDevice = ""
        o.clusterCoordinator = ""
        appState.serverOptions = o
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
