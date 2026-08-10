import AppKit
import Darwin

/// Which running apps are using the most memory right now — surfaced when a
/// model load is refused for memory, so the user can see what to quit instead
/// of guessing. Pure ranking/formatting (testable); the live enumeration reads
/// `NSWorkspace` + each app's `phys_footprint`.
enum RunningAppsMemory {
    struct AppMem: Equatable, Identifiable {
        let name: String
        let bytes: Int64
        var id: String { name }
        var label: String { MemoryInfo.format(bytes) }
    }

    /// Pure: sort by memory descending and keep the top `limit`.
    static func rank(_ apps: [AppMem], limit: Int) -> [AppMem] {
        Array(apps.sorted { $0.bytes > $1.bytes }.prefix(max(0, limit)))
    }

    /// Pure: one line for a crash alert — "Figma 1.5 GB · Xcode 1.0 GB".
    static func summaryLine(_ apps: [AppMem]) -> String {
        apps.map { "\($0.name) \($0.label)" }.joined(separator: " · ")
    }

    /// Pure: combined footprint of the listed apps — what quitting them frees.
    static func totalBytes(_ apps: [AppMem]) -> Int64 {
        apps.reduce(0) { $0 + $1.bytes }
    }

    /// The top user-facing apps by memory footprint (impure — reads live procs).
    /// Only `.regular` apps (the ones a user recognizes and can quit) above
    /// `minBytes`, so the list is short and actionable. `mlx-serve`'s own
    /// footprint isn't here — it's not a `.regular` app and it's the thing
    /// trying to load anyway.
    static func topApps(limit: Int = 4, minBytes: Int64 = 300 * 1024 * 1024) -> [AppMem] {
        var mems: [AppMem] = []
        for app in NSWorkspace.shared.runningApplications where app.activationPolicy == .regular {
            guard let name = app.localizedName, app.processIdentifier > 0 else { continue }
            if let bytes = physFootprint(pid: app.processIdentifier), bytes >= minBytes {
                mems.append(AppMem(name: name, bytes: bytes))
            }
        }
        return rank(mems, limit: limit)
    }

    /// A pid's physical memory footprint (the same number Activity Monitor's
    /// "Memory" column shows). nil if the process can't be read — a sandboxed
    /// (App Store) build may be denied here, in which case the caller just omits
    /// the app list and shows the generic guidance.
    private static func physFootprint(pid: pid_t) -> Int64? {
        var info = rusage_info_v2()
        let result = withUnsafeMutablePointer(to: &info) { ptr -> Int32 in
            ptr.withMemoryRebound(to: rusage_info_t?.self, capacity: 1) { rebound in
                proc_pid_rusage(pid, RUSAGE_INFO_V2, rebound)
            }
        }
        guard result == 0 else { return nil }
        return Int64(bitPattern: info.ri_phys_footprint)
    }
}
