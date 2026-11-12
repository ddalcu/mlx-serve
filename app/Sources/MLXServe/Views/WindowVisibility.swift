import SwiftUI
import AppKit

/// Tells its view whether the window it sits in can be seen: false while that window is closed, minimised or
/// fully covered, where `onDisappear` never fires.
struct WindowVisibility: NSViewRepresentable {
    let onChange: (Bool) -> Void

    func makeNSView(context: Context) -> Probe {
        let probe = Probe()
        probe.onChange = onChange
        return probe
    }

    func updateNSView(_ probe: Probe, context: Context) { probe.onChange = onChange }

    final class Probe: NSView {
        var onChange: (Bool) -> Void = { _ in }
        private var observer: NSObjectProtocol?

        deinit { observer.map(NotificationCenter.default.removeObserver) }

        override func viewDidMoveToWindow() {
            super.viewDidMoveToWindow()
            observer.map(NotificationCenter.default.removeObserver)
            observer = window.map {
                NotificationCenter.default.addObserver(forName: NSWindow.didChangeOcclusionStateNotification, object: $0,
                                                       queue: .main) { [weak self] _ in self?.report() }
            }
            report()
        }

        private func report() { onChange(window?.occlusionState.contains(.visible) ?? false) }
    }
}
