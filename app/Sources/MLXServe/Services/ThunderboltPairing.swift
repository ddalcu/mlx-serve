import Foundation

/// The Cluster page's Thunderbolt buttons. Pairing sets the cable's address (macOS asks for
/// the password), the role, and restarts the server when this Mac leads; the Mac that
/// joins dials the leader's end of the cable, so it needs no name.
@MainActor
enum ThunderboltPairing {
    /// This Mac leads a pair that runs `model` (a path from the library).
    static func lead(model: String, appState: AppState, ports: [RdmaPort]) -> String? {
        if let err = giveAddress(.host, ports: ports) { return err }
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
    static func join(appState: AppState, ports: [RdmaPort]) -> String? {
        if let err = giveAddress(.worker, ports: ports) { return err }
        var o = appState.serverOptions
        o.clusterRole = .worker
        o.clusterDevice = ""
        o.clusterCoordinator = ""
        appState.serverOptions = o
        return nil
    }

    /// The address a restart took away, for the role this Mac already has.
    static func reconnect(_ role: ClusterRole, appState: AppState, ports: [RdmaPort]) -> String? {
        if let err = giveAddress(role, ports: ports) { return err }
        if role == .host { restartServer(appState) }
        return nil
    }

    static func unpair(appState: AppState) {
        let led = appState.serverOptions.clusterRole == .host
        appState.serverOptions.clusterRole = .off
        if led { restartServer(appState) }
    }

    private static func giveAddress(_ role: ClusterRole, ports: [RdmaPort]) -> String? {
        guard let p = ports.first(where: \.active) else { return L10n.text("Connect the Thunderbolt cable first.") }
        guard ClusterSetup.ipv4(on: p.interface) != ClusterSetup.linkAddress(role) else { return nil }
        return ClusterSetup.assignAddress(interface: p.interface, role: role)
    }

    private static func restartServer(_ appState: AppState) {
        appState.server.stop()
        appState.startServer(loadingSelection: false)
    }
}
