import AppKit

/// Starts the app's own bundle again and ends this process — for a change
/// that only lands on the next launch: a catalog resolves once per process
/// (Settings ▸ Interface ▸ Language), and `UpdateChecker` runs the same dance
/// over the bundle it just swapped in place.
enum AppRelaunch {
    /// The detached child outlives this process; the sleep lets it quit fully
    /// before `open` starts the bundle. The path rides as a positional
    /// argument — a bundle path containing a quote, `$` or a space is never
    /// script text, and only this app's own bundle is ever reopened.
    static func relaunch(at bundlePath: String) {
        let relaunch = Process()
        relaunch.executableURL = URL(fileURLWithPath: "/bin/sh")
        relaunch.arguments = ["-c", "sleep 1; /usr/bin/open \"$@\"", "sh", bundlePath]
        try? relaunch.run()
        NSApplication.shared.terminate(nil)
    }
}
