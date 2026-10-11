import Foundation
import AppKit

/// Picks the catalog the whole app reads (Settings ▸ Interface ▸ Language).
///
/// `AppleLanguages` in the app's OWN domain is the per-app language override
/// macOS resolves a bundle from — the same key System Settings writes — and it
/// is the only lever that reaches every reader at once: `L10n`, and the
/// `Text("…")`/`Label`/`Button` literals SwiftUI resolves through CFBundle.
/// (Swapping `Bundle.main`'s class reaches `L10n` and nothing else, which
/// ships a screen that is half one language and half the other.)
///
/// A catalog is resolved once per process, so the row's change lands on the
/// next start — the app starts itself again rather than showing half of each.
enum AppLanguageOverride {
    private static let preferredLanguagesKey = "AppleLanguages"

    /// The catalog the stored preference asks for. The app ships English (the
    /// development region, so its keys are their own text) and Simplified
    /// Chinese, so those are the only two answers — a Mac set to any other
    /// language reads English.
    static func targetLanguage(preference: String?, preferredLanguages: [String]) -> String {
        (AppLanguage(rawValue: preference ?? "") ?? .system).catalog(preferredLanguages: preferredLanguages)
    }

    /// The Mac's own languages, read from the GLOBAL domain on purpose:
    /// `Locale.preferredLanguages` answers with this app's override once one
    /// has been written, and "follow the system" has to mean the system.
    static var systemLanguages: [String] {
        let global = UserDefaults.standard.persistentDomain(forName: UserDefaults.globalDomain)
        return (global?[preferredLanguagesKey] as? [String]) ?? Locale.preferredLanguages
    }

    /// Reads the stored preference and writes the override. Runs at the top of
    /// `main()`, before the first string of the launch is localized.
    static func install() {
        let language = targetLanguage(preference: UserDefaults.standard.string(forKey: InterfacePrefKey.language),
                                      preferredLanguages: systemLanguages)
        UserDefaults.standard.set([language], forKey: preferredLanguagesKey)
    }

    /// Starts the app again and quits this one — the change the user just made
    /// is picked up by the new process's `install()`. Same shape as
    /// `UpdateChecker`'s relaunch: the detached child outlives this process,
    /// and the sleep lets it quit fully before `open` starts the bundle.
    static func relaunch() {
        let restarted = Process()
        restarted.executableURL = URL(fileURLWithPath: "/bin/sh")
        restarted.arguments = ["-c", "sleep 1; /usr/bin/open \"\(Bundle.main.bundlePath)\""]
        try? restarted.run()
        NSApplication.shared.terminate(nil)
    }
}
