import Foundation

/// Persisted access to a user-picked file or folder under the App Sandbox.
///
/// An `NSOpenPanel` grant lasts only for the current launch. To reopen a
/// user-chosen path after a relaunch — the voice-clone clip, an attached
/// document folder — the app must store a SECURITY-SCOPED BOOKMARK and, before
/// touching the path, call `startAccessingSecurityScopedResource()`.
///
/// Outside the sandbox (the Developer ID build) bookmarks still resolve fine;
/// the start/stop calls are harmless no-ops there, so the same code path works
/// for both builds.
///
/// The bookmark blobs live in `UserDefaults` keyed by a caller-chosen name. The
/// pure key/encode/decode logic is unit-tested; the AppKit resolution is thin.
enum SecurityScopedBookmark {

    /// Namespaced UserDefaults key for a bookmark slot.
    static func defaultsKey(_ name: String) -> String { "securityBookmark.\(name)" }

    /// Create a bookmark for `url` and persist it under `name`. Call right after
    /// the user picks the path in an open panel, while access is still granted.
    @discardableResult
    static func store(_ url: URL, name: String,
                      defaults: UserDefaults = .standard) -> Bool {
        do {
            let data = try url.bookmarkData(options: .withSecurityScope,
                                            includingResourceValuesForKeys: nil,
                                            relativeTo: nil)
            defaults.set(data, forKey: defaultsKey(name))
            return true
        } catch {
            // Non-sandboxed builds can't create a security-scoped bookmark for
            // some paths; that's fine — the path is directly accessible there.
            return false
        }
    }

    /// Resolve the bookmark stored under `name`. Returns the URL and whether a
    /// security scope was started (the caller must balance it with `stop`).
    /// A stale bookmark is refreshed transparently.
    static func resolve(name: String,
                        defaults: UserDefaults = .standard) -> (url: URL, started: Bool)? {
        guard let data = defaults.data(forKey: defaultsKey(name)) else { return nil }
        var stale = false
        guard let url = try? URL(resolvingBookmarkData: data,
                                 options: .withSecurityScope,
                                 relativeTo: nil,
                                 bookmarkDataIsStale: &stale) else { return nil }
        let started = url.startAccessingSecurityScopedResource()
        if stale { store(url, name: name, defaults: defaults) }
        return (url, started)
    }

    /// Run `body` with the bookmarked path accessible, balancing the scope.
    static func withResolved<T>(name: String,
                                defaults: UserDefaults = .standard,
                                _ body: (URL) throws -> T) rethrows -> T? {
        guard let (url, started) = resolve(name: name, defaults: defaults) else { return nil }
        defer { if started { url.stopAccessingSecurityScopedResource() } }
        return try body(url)
    }

    static func clear(name: String, defaults: UserDefaults = .standard) {
        defaults.removeObject(forKey: defaultsKey(name))
    }

    static func has(name: String, defaults: UserDefaults = .standard) -> Bool {
        defaults.data(forKey: defaultsKey(name)) != nil
    }

    // MARK: - Per-session slots

    /// The session's agent working folder (the WorkspacePicker pick).
    static func workingFolderName(_ sessionId: UUID) -> String {
        "workingFolder.\(sessionId.uuidString)"
    }

    /// The session's attached document folder (mini RAG).
    static func attachedFolderName(_ sessionId: UUID) -> String {
        "attachedFolder.\(sessionId.uuidString)"
    }

    // MARK: - Launch-lifetime access

    /// Live grants started this launch, keyed by bookmark name.
    private static let startedLock = NSLock()
    private static var started: [String: URL] = [:]

    /// Resolve the bookmark and START its security scope, holding it for the
    /// rest of the launch. For paths used across a whole agent turn (shell/file
    /// tools run for minutes), a per-call `withResolved` balance would drop
    /// access mid-use. Idempotent per name, so repeated turns can't leak one
    /// kernel scope each — the grant count is bounded by the number of distinct
    /// bookmarks touched this launch.
    @discardableResult
    static func startAccessOnce(name: String, defaults: UserDefaults = .standard) -> URL? {
        startedLock.lock()
        if let url = started[name] { startedLock.unlock(); return url }
        startedLock.unlock()

        guard let (url, startedScope) = resolve(name: name, defaults: defaults) else { return nil }
        startedLock.lock()
        if let existing = started[name] {
            // Raced with another caller: keep the first grant, balance ours.
            startedLock.unlock()
            if startedScope { url.stopAccessingSecurityScopedResource() }
            return existing
        }
        started[name] = url
        startedLock.unlock()
        return url
    }
}
