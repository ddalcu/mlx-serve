import AppKit

/// The one place an `NSOpenPanel` is built (pinned by OpenPanelTests): every
/// picker shows hidden files, because dotfolders are exactly what people
/// point an agent at.
enum OpenPanel {
    static func make() -> NSOpenPanel {
        let panel = NSOpenPanel()
        panel.showsHiddenFiles = true
        return panel
    }
}
