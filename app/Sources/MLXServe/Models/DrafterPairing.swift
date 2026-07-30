import Foundation

/// Which assistant drafter the app should launch with for a given model.
///
/// The drafter is paired to ONE Gemma 4 size and arrives with it
/// (`DownloadManager.companionDrafterRepo`), so pairing is something the app
/// decides rather than something the user configures — the Settings switch is
/// there to turn it OFF, and that off has to stick.
///
/// That is the whole reason this isn't just "fill in `drafterPath` when it's
/// empty": empty means both "never decided" and "the user switched it off", and
/// a default that can't tell those apart re-enables itself at the next model
/// switch, silently, forever.
enum DrafterPairing {

    /// The `drafterPath` to launch with. `""` = no drafter.
    ///
    /// - Parameters:
    ///   - modelPath: the model about to be served (a directory path; only its
    ///     last component is read).
    ///   - optedOut: `ServerOptions.drafterOptOut` — the user turned it off.
    ///   - onDiskPath: where the paired drafter lives, nil when not downloaded.
    ///
    /// The MoE Gemma 4 is excluded even when its drafter IS on disk (an older
    /// build's Drafters pane could have fetched it): verify pays expert routing
    /// there, so the drafter costs decode speed rather than buying it — the
    /// server defaults it off on MoE targets for the same reason. Turning it on
    /// by hand in Settings still works; we just never reach for it ourselves.
    static func decide(modelPath: String, optedOut: Bool, onDiskPath: String?) -> String {
        guard !optedOut,
              DownloadManager.companionDrafterRepo(forRepoId: modelPath) != nil,
              let onDiskPath, !onDiskPath.isEmpty else { return "" }
        return onDiskPath
    }
}
