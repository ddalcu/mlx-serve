import Foundation

/// What selecting an agent means for the model the server is running.
///
/// An agent may pin a model; "Current" (nil) means whatever is selected right
/// now. The one outcome that must never happen silently is a multi-GB download,
/// so a model that isn't on disk makes the agent UNAVAILABLE — greyed in the
/// picker with a Download button, and declined out loud when the switch came
/// from speech (silently answering as whoever was active is the failure the user
/// can't see).
///
/// Pure so the Agents window, the chat picker and the tray picker can't disagree.
enum AgentModelSwitch {

    enum Decision: Equatable {
        /// Already the active model, or the agent doesn't pin one.
        case noChange
        /// Load this local path (`AppState.useModelAndAwaitReady`).
        case load(path: String)
        /// A LAN `id@peer` — the server resolves it; no local load, no download.
        case lan(id: String)
        /// Pinned model isn't downloaded. Offer the download; never start one.
        case needsDownload(path: String)
        /// Pinned model can't be reached at all (an offline peer).
        case unavailable(reason: String)
    }

    /// A LAN model id, `<model>@<peer>` — the same shape the server's own
    /// mirroring uses. A local path always starts with `/` or `~`.
    static func isLanId(_ value: String) -> Bool {
        !value.hasPrefix("/") && !value.hasPrefix("~") && value.contains("@")
    }

    static func decide(modelPath: String?,
                       selectedModelPath: String,
                       downloadedPaths: [String],
                       lanModelIds: [String]) -> Decision {
        guard let pinned = modelPath?.trimmingCharacters(in: .whitespacesAndNewlines),
              !pinned.isEmpty else { return .noChange }
        if pinned == selectedModelPath { return .noChange }
        if isLanId(pinned) {
            guard lanModelIds.contains(pinned) else {
                return .unavailable(reason: "\(pinned) isn't on the network right now — the peer sharing it is offline or LAN discovery is off.")
            }
            return .lan(id: pinned)
        }
        guard downloadedPaths.contains(pinned) else { return .needsDownload(path: pinned) }
        return .load(path: pinned)
    }

    /// Can the user pick this agent right now? A pinned model that isn't there
    /// makes the agent unusable, so the row is greyed rather than selectable-then-
    /// broken.
    static func isSelectable(_ decision: Decision) -> Bool {
        switch decision {
        case .noChange, .load, .lan: return true
        case .needsDownload, .unavailable: return false
        }
    }

    /// What to SAY when a spoken switch can't happen. nil when the switch is
    /// fine — the caller then just switches.
    static func spokenDecline(agentName: String, decision: Decision) -> String? {
        switch decision {
        case .noChange, .load, .lan:
            return nil
        case .needsDownload:
            return "\(agentName) needs its model downloaded first."
        case .unavailable:
            return "\(agentName) can't run right now — its model isn't reachable."
        }
    }

    /// Label for the model row in the editor and the picker.
    static func displayName(for modelPath: String?, localModels: [LocalModel]) -> String {
        guard let modelPath, !modelPath.isEmpty else { return "Current" }
        if let match = localModels.first(where: { $0.path == modelPath }) { return match.name }
        if isLanId(modelPath) { return modelPath }
        return (modelPath as NSString).lastPathComponent
    }
}
