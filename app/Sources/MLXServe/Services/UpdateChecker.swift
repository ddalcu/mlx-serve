import AppKit
import Foundation

/// A newer app release discovered on GitHub, ready to download + install.
struct AppUpdate: Equatable {
    let version: String        // normalized, no "v" prefix — e.g. "26.9.0"
    let tagName: String        // "v26.9.0"
    let dmgURL: URL            // the MLXCore.dmg release asset
    let releaseNotes: String   // release body markdown ("" when absent)
    let releasePageURL: URL?   // https://github.com/…/releases/tag/v26.9.0
}

/// Self-updater backed directly by the GitHub Releases API — no Sparkle, no
/// appcast to host. `releases/latest` is fetched once a day (and on demand),
/// the `vYY.M.N` CalVer tag is compared against `CFBundleShortVersionString`,
/// and installing downloads the notarized `MLXCore.dmg` asset, mounts it,
/// swaps the installed bundle atomically (rename old aside → rename new in),
/// and relaunches. Dev builds that don't run from a replaceable `.app`
/// (swift run, xctest) fall back to opening the DMG in Finder.
///
/// All decision logic (version compare, release-JSON parsing, asset pick,
/// throttle, bundle discovery) is in pure `static` helpers pinned by
/// `UpdateCheckerTests`; only the fetch/mount/swap plumbing is live-only.
@MainActor
final class UpdateChecker: ObservableObject {
    nonisolated static let repo = "ddalcu/mlx-serve"
    nonisolated static let latestReleaseURL = URL(string: "https://api.github.com/repos/\(repo)/releases/latest")!
    /// The app-installer asset uploaded by the release flow (build.sh phase 7).
    nonisolated static let preferredAssetName = "MLXCore.dmg"
    nonisolated static let autoCheckInterval: TimeInterval = 24 * 3600
    nonisolated static let lastCheckKey = "lastUpdateCheckAt"
    nonisolated static let autoCheckEnabledKey = "autoCheckUpdates"

    enum Phase: Equatable {
        case idle
        case checking
        /// Fraction 0…1 of the DMG downloaded (0 until the first chunk lands).
        case downloading(Double)
        case installing
        /// Transient after a user-initiated check that found nothing.
        case upToDate
        case failed(String)
    }

    @Published var available: AppUpdate?
    @Published var phase: Phase = .idle

    /// Invoked right before terminate+relaunch so the owner can stop the
    /// server child process (AppState wires `server.stop()` here).
    var willRelaunch: (() -> Void)?

    private var timer: Timer?

    /// Default ON; `UserDefaults.bool` alone can't express that, so absence
    /// of the key means enabled.
    var autoCheckEnabled: Bool {
        get {
            UserDefaults.standard.object(forKey: Self.autoCheckEnabledKey) == nil
                || UserDefaults.standard.bool(forKey: Self.autoCheckEnabledKey)
        }
        set {
            objectWillChange.send()
            UserDefaults.standard.set(newValue, forKey: Self.autoCheckEnabledKey)
        }
    }

    var currentVersion: String {
        Bundle.main.object(forInfoDictionaryKey: "CFBundleShortVersionString") as? String ?? "0.0.0"
    }

    // MARK: - Pure logic (unit-tested)

    /// "v26.7.1" / "26.7.1" → [26, 7, 1]. Strictly numeric dot components —
    /// anything else ("latest", "v26.7.1-beta") is nil so a surprising tag can
    /// never announce an update.
    nonisolated static func parseCalVer(_ s: String) -> [Int]? {
        let bare = s.hasPrefix("v") ? String(s.dropFirst()) : s
        guard !bare.isEmpty else { return nil }
        let parts = bare.split(separator: ".", omittingEmptySubsequences: false)
        var out: [Int] = []
        for part in parts {
            guard !part.isEmpty, part.allSatisfy(\.isNumber), let n = Int(part) else { return nil }
            out.append(n)
        }
        return out
    }

    /// Numeric component-wise compare (lexicographic would sort v26.10 below
    /// v26.9); shorter versions zero-pad. Malformed input is never newer.
    nonisolated static func isNewer(_ remote: String, than local: String) -> Bool {
        guard var r = parseCalVer(remote), var l = parseCalVer(local) else { return false }
        let n = max(r.count, l.count)
        r.append(contentsOf: repeatElement(0, count: n - r.count))
        l.append(contentsOf: repeatElement(0, count: n - l.count))
        for i in 0..<n where r[i] != l[i] { return r[i] > l[i] }
        return false
    }

    private struct Release: Decodable {
        struct Asset: Decodable {
            let name: String
            let browser_download_url: String
        }
        let tag_name: String
        let html_url: String?
        let body: String?
        let draft: Bool?
        let prerelease: Bool?
        let assets: [Asset]?
    }

    /// GitHub `releases/latest` JSON → an update, or nil when there's nothing
    /// to do: tag not newer than `currentVersion` (dev builds run pre-bumped
    /// versions and must stay silent), draft/prerelease, or no .dmg asset.
    nonisolated static func findUpdate(inReleaseJSON data: Data, currentVersion: String) -> AppUpdate? {
        guard let release = try? JSONDecoder().decode(Release.self, from: data) else { return nil }
        if release.draft == true || release.prerelease == true { return nil }
        guard isNewer(release.tag_name, than: currentVersion) else { return nil }
        let assets = release.assets ?? []
        guard let asset = assets.first(where: { $0.name == preferredAssetName })
            ?? assets.first(where: { $0.name.hasSuffix(".dmg") }),
            let dmgURL = URL(string: asset.browser_download_url)
        else { return nil }
        let version = release.tag_name.hasPrefix("v")
            ? String(release.tag_name.dropFirst()) : release.tag_name
        return AppUpdate(
            version: version,
            tagName: release.tag_name,
            dmgURL: dmgURL,
            releaseNotes: release.body ?? "",
            releasePageURL: release.html_url.flatMap(URL.init(string:)))
    }

    /// Daily throttle for the silent background check. A `lastCheck` in the
    /// future (clock stepped backward) checks immediately instead of wedging
    /// the throttle shut until the wall clock catches up.
    nonisolated static func shouldAutoCheck(
        now: Date, lastCheck: Date?, interval: TimeInterval = autoCheckInterval
    ) -> Bool {
        guard let last = lastCheck else { return true }
        if last > now { return true }
        return now.timeIntervalSince(last) >= interval
    }

    /// The one real `.app` bundle at the top of a mounted DMG (skips the
    /// /Applications symlink and metadata entries).
    nonisolated static func findAppBundle(inMountedDMG dir: URL) -> URL? {
        let items = (try? FileManager.default.contentsOfDirectory(
            at: dir, includingPropertiesForKeys: [.isDirectoryKey, .isSymbolicLinkKey])) ?? []
        return items.first { item in
            guard item.pathExtension == "app" else { return false }
            let values = try? item.resourceValues(forKeys: [.isDirectoryKey, .isSymbolicLinkKey])
            return values?.isDirectory == true && values?.isSymbolicLink != true
        }
    }

    /// Where the running app can be replaced in place: a real `.app` bundle
    /// directory whose parent is writable (the installed-in-/Applications
    /// case). Dev binaries under .build and the xctest host return nil — the
    /// installer then falls back to opening the DMG for a manual drag.
    nonisolated static func installTarget(for bundleURL: URL) -> URL? {
        guard bundleURL.pathExtension == "app" else { return nil }
        var isDir: ObjCBool = false
        guard FileManager.default.fileExists(atPath: bundleURL.path, isDirectory: &isDir),
              isDir.boolValue,
              FileManager.default.isWritableFile(
                  atPath: bundleURL.deletingLastPathComponent().path)
        else { return nil }
        return bundleURL
    }

    // MARK: - Checking

    /// Kick off the launch-time check plus a low-frequency re-check timer.
    /// Each tick re-applies the persisted daily throttle, so the network is
    /// hit at most once per `autoCheckInterval` regardless of app relaunches.
    func startAutoCheck() {
        // Never under the xctest host — unit tests must not touch the network.
        guard ProcessInfo.processInfo.environment["XCTestConfigurationFilePath"] == nil else { return }
        maybeAutoCheck()
        let t = Timer(timeInterval: 6 * 3600, repeats: true) { _ in
            Task { @MainActor [weak self] in self?.maybeAutoCheck() }
        }
        t.tolerance = 600
        RunLoop.main.add(t, forMode: .common)
        timer = t
    }

    private func maybeAutoCheck() {
        guard autoCheckEnabled else { return }
        let last = UserDefaults.standard.object(forKey: Self.lastCheckKey) as? Date
        guard Self.shouldAutoCheck(now: Date(), lastCheck: last) else { return }
        Task { await self.check(userInitiated: false) }
    }

    /// Fetch `releases/latest` and publish any newer release. Silent checks
    /// swallow network errors (offline is normal); user-initiated checks
    /// surface them plus an explicit "up to date".
    @discardableResult
    func check(userInitiated: Bool = false) async -> AppUpdate? {
        // The Mac App Store updates the app itself; a self-updater that
        // downloads and swaps the bundle is forbidden (App Review 2.5.2). No
        // network check, no install path.
        guard BuildFeatures.current.selfUpdate else { return nil }
        switch phase {
        case .checking, .downloading, .installing: return available
        default: break
        }
        phase = .checking
        UserDefaults.standard.set(Date(), forKey: Self.lastCheckKey)
        do {
            var request = URLRequest(url: Self.latestReleaseURL)
            request.setValue("application/vnd.github+json", forHTTPHeaderField: "Accept")
            let (data, response) = try await URLSession.shared.data(for: request)
            if let http = response as? HTTPURLResponse, http.statusCode != 200 {
                throw UpdateError.badStatus(http.statusCode)
            }
            let update = Self.findUpdate(inReleaseJSON: data, currentVersion: currentVersion)
            available = update
            phase = (update == nil && userInitiated) ? .upToDate : .idle
            return update
        } catch {
            phase = userInitiated ? .failed(error.localizedDescription) : .idle
            return nil
        }
    }

    // MARK: - Download + install

    /// Download the DMG, verify the bundled app's version, swap the installed
    /// bundle, and relaunch. Any failure lands in `.failed`; the
    /// not-replaceable case (dev build) opens the DMG in Finder instead.
    func downloadAndInstall() async {
        guard let update = available else { return }
        switch phase {
        case .downloading, .installing: return
        default: break
        }
        phase = .downloading(0)
        do {
            let dmg = try await download(update)
            phase = .installing
            try await install(fromDMG: dmg, update: update)
        } catch UpdateError.notReplaceable(let dmg) {
            // Running outside a writable .app (dev build): hand the DMG to
            // Finder so the user can drag it into /Applications themselves.
            NSWorkspace.shared.open(dmg)
            phase = .idle
        } catch {
            phase = .failed(error.localizedDescription)
        }
    }

    private nonisolated func download(_ update: AppUpdate) async throws -> URL {
        let dest = FileManager.default.temporaryDirectory
            .appendingPathComponent("MLXCore-update-\(update.version).dmg")
        try? FileManager.default.removeItem(at: dest)
        let (bytes, response) = try await URLSession.shared.bytes(from: update.dmgURL)
        if let http = response as? HTTPURLResponse, http.statusCode != 200 {
            throw UpdateError.badStatus(http.statusCode)
        }
        let total = response.expectedContentLength // -1 when unknown
        FileManager.default.createFile(atPath: dest.path, contents: nil)
        let handle = try FileHandle(forWritingTo: dest)
        defer { try? handle.close() }
        var buffer = Data()
        buffer.reserveCapacity(1 << 20)
        var written: Int64 = 0
        var lastReported = -1.0
        for try await byte in bytes {
            buffer.append(byte)
            if buffer.count >= 1 << 20 {
                try handle.write(contentsOf: buffer)
                written += Int64(buffer.count)
                buffer.removeAll(keepingCapacity: true)
                if total > 0 {
                    let fraction = Double(written) / Double(total)
                    if fraction - lastReported >= 0.01 { // ≤100 UI updates total
                        lastReported = fraction
                        await MainActor.run { self.phase = .downloading(fraction) }
                    }
                }
            }
        }
        try handle.write(contentsOf: buffer)
        return dest
    }

    private nonisolated func install(fromDMG dmg: URL, update: AppUpdate) async throws {
        let mountPoint = FileManager.default.temporaryDirectory
            .appendingPathComponent("mlxcore-update-mount-\(update.version)")
        try await Self.run("/usr/bin/hdiutil",
                           ["attach", dmg.path, "-nobrowse", "-readonly",
                            "-mountpoint", mountPoint.path])
        var mounted = true
        defer {
            if mounted {
                Task.detached {
                    try? await Self.run("/usr/bin/hdiutil", ["detach", mountPoint.path, "-force"])
                }
            }
        }

        guard let newApp = Self.findAppBundle(inMountedDMG: mountPoint) else {
            throw UpdateError.noAppInDMG
        }
        // Sanity: the DMG really contains the version we announced. Guards
        // against a mis-uploaded asset replacing the app with something else.
        let plist = NSDictionary(contentsOf: newApp.appendingPathComponent("Contents/Info.plist"))
        if let v = plist?["CFBundleShortVersionString"] as? String, v != update.version {
            throw UpdateError.versionMismatch(expected: update.version, found: v)
        }

        guard let target = Self.installTarget(for: Bundle.main.bundleURL) else {
            throw UpdateError.notReplaceable(dmg: dmg)
        }
        let parent = target.deletingLastPathComponent()
        let baseName = target.deletingPathExtension().lastPathComponent

        // Copy out of the read-only mount with ditto (preserves the code
        // signature, xattrs, and symlinks) into a hidden sibling, so the
        // final swap is two same-volume renames — atomic, no torn bundle.
        let staged = parent.appendingPathComponent(".\(baseName)-update-\(update.version).app")
        try? FileManager.default.removeItem(at: staged)
        try await Self.run("/usr/bin/ditto", [newApp.path, staged.path])

        let old = parent.appendingPathComponent(".\(baseName)-previous.app")
        try? FileManager.default.removeItem(at: old)
        try FileManager.default.moveItem(at: target, to: old)
        do {
            try FileManager.default.moveItem(at: staged, to: target)
        } catch {
            try? FileManager.default.moveItem(at: old, to: target) // roll back
            throw error
        }
        // The running executable's files stay open (APFS) — safe to delete.
        try? FileManager.default.removeItem(at: old)

        try? await Self.run("/usr/bin/hdiutil", ["detach", mountPoint.path, "-force"])
        mounted = false

        await MainActor.run { self.willRelaunch?() }
        // Detached child survives our exit; the sleep lets this process quit
        // fully before `open` starts the new bundle.
        let relaunch = Process()
        relaunch.executableURL = URL(fileURLWithPath: "/bin/sh")
        relaunch.arguments = ["-c", "sleep 1; /usr/bin/open \"\(target.path)\""]
        try? relaunch.run()
        await MainActor.run { NSApplication.shared.terminate(nil) }
    }

    /// Run a tool to completion without blocking the cooperative pool.
    private nonisolated static func run(_ tool: String, _ args: [String]) async throws {
        try await withCheckedThrowingContinuation { (cont: CheckedContinuation<Void, Error>) in
            let p = Process()
            p.executableURL = URL(fileURLWithPath: tool)
            p.arguments = args
            let errPipe = Pipe()
            p.standardOutput = Pipe()
            p.standardError = errPipe
            p.terminationHandler = { proc in
                if proc.terminationStatus == 0 {
                    cont.resume()
                } else {
                    let stderr = String(
                        data: errPipe.fileHandleForReading.readDataToEndOfFile(),
                        encoding: .utf8) ?? ""
                    cont.resume(throwing: UpdateError.toolFailed(
                        tool: (tool as NSString).lastPathComponent,
                        status: proc.terminationStatus,
                        stderr: stderr.trimmingCharacters(in: .whitespacesAndNewlines)))
                }
            }
            do { try p.run() } catch { cont.resume(throwing: error) }
        }
    }
}

enum UpdateError: LocalizedError {
    case badStatus(Int)
    case noAppInDMG
    case versionMismatch(expected: String, found: String)
    case notReplaceable(dmg: URL)
    case toolFailed(tool: String, status: Int32, stderr: String)

    var errorDescription: String? {
        switch self {
        case .badStatus(let code):
            return "GitHub returned HTTP \(code)"
        case .noAppInDMG:
            return "The downloaded disk image contains no app bundle"
        case .versionMismatch(let expected, let found):
            return "Downloaded app is v\(found), expected v\(expected)"
        case .notReplaceable:
            return "This build can't self-update — install from the opened disk image"
        case .toolFailed(let tool, let status, let stderr):
            return "\(tool) failed (\(status))\(stderr.isEmpty ? "" : ": \(stderr)")"
        }
    }
}
