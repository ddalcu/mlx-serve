import SwiftUI

/// Shared plumbing for offering LAN-discovered models beside a media pane's
/// local presets in ONE Picker. Selection (and the persisted `modelId`) is a
/// string: a preset id for local models, `"lan:<model>@<peer>"` for network
/// ones — the suffix form the local server proxies to the hosting Mac.
enum LanPick {
    static let prefix = "lan:"

    /// The LAN routing id inside a selection/persisted value, or nil for locals.
    static func lanId(_ value: String) -> String? {
        value.hasPrefix(prefix) ? String(value.dropFirst(prefix.count)) : nil
    }

    /// The `modelId` to persist for the current pane state.
    static func persisted(lanModel: String?, presetId: String) -> String {
        lanModel.map { prefix + $0 } ?? presetId
    }

    /// The peer name inside a LAN routing id ("model@peer" → "peer").
    static func peer(of id: String) -> String {
        guard let at = id.lastIndex(of: "@") else { return id }
        return String(id[id.index(after: at)...])
    }

    /// `peer(of:)`'s mirror: the model id without the peer suffix.
    static func base(of id: String) -> String {
        guard let at = id.lastIndex(of: "@") else { return id }
        return String(id[..<at])
    }

    // The Picker-binding form (`selection`) and the `LanModelPickerRows`
    // section retired with the panes' old radio pickers: `MediaModelChooser`
    // owns both halves of the choice now, including the preset adoption a LAN
    // pick performs (its "On Your Network" section + `onSelectLan`), pinned by
    // `MediaModelChooserTests.testLanPickAdoptsCatalogThenCustomFamilyPreset`.
}
