import Foundation

/// One selectable `.gguf` file in a GGUF repo — a quant.
struct GgufQuant: Identifiable, Equatable, Hashable {
    /// Basename as it lives on disk / in the HF tree (`Qwen3.5-4B-Q4_K_M.gguf`).
    let filename: String
    /// Short human label for menus (`Q4_K_M`).
    let label: String

    var id: String { filename }
}

/// The Discover row's GGUF action menu, as data.
///
/// A GGUF repo is not one model: it ships a folder of quants and the user picks
/// which to run. The row therefore never collapses into a terminal "on disk"
/// state — having `Q4_K_M` says nothing about whether you also want `Q8_0`, so
/// the menu keeps offering the ones you don't have, alongside the ones you do.
enum GgufQuantMenuModel {

    struct Menu: Equatable {
        /// Quants present on this Mac — offer Use / Delete.
        let onDisk: [GgufQuant]
        /// Quants the repo publishes that we don't have yet — offer Download.
        let available: [GgufQuant]
    }

    /// Split a repo's published `.gguf` list against what's on disk.
    ///
    /// An on-disk quant the repo no longer lists (re-quantized, renamed) stays
    /// in `onDisk`: it's still a real file the server can load, and hiding it
    /// would strand it with no way to select or delete it.
    static func build(remote: [String], onDisk: [String]) -> Menu {
        let have = Set(onDisk)
        let quant = { (f: String) in GgufQuant(filename: f, label: DownloadManager.quantLabel(forFilename: f)) }

        let onDiskQuants = onDisk
            .filter { DownloadManager.isSupportedGguf($0) }
            .sorted()
            .map(quant)

        let availableQuants = remote
            .filter { DownloadManager.isSupportedGguf($0) && !have.contains($0) }
            .sorted()
            .map(quant)

        return Menu(onDisk: onDiskQuants, available: availableQuants)
    }

    /// The menu's button title. Owning quants outranks a failed/partial transfer
    /// of a *different* quant — the row must keep reporting what you actually
    /// have, not the state of the last thing you clicked.
    static func buttonLabel(onDisk: [GgufQuant], failed: Bool, hasPartial: Bool) -> String {
        if onDisk.count == 1 { return "✓ \(onDisk[0].label)" }
        if onDisk.count > 1 { return "✓ \(onDisk.count) on disk" }
        if failed { return "Retry" }
        return hasPartial ? "Resume" : "Download"
    }
}
