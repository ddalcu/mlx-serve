import Foundation

/// Installs an `mlx-serve` symlink onto the user's PATH so the CLI works from
/// Terminal (`mlx-serve run gemma4-e4b`, `mlx-serve --serve ...`).
///
/// Target preference: an existing home bin dir that is already on the user's
/// shell PATH (`~/.local/bin`, then `~/bin`) — a plain symlink, no admin
/// prompt. Otherwise `/usr/local/bin` behind a one-time admin authorization
/// (osascript `with administrator privileges`, the Docker Desktop / VS Code
/// pattern). Shell dotfiles are never edited.
///
/// The symlink points INTO the app bundle (`Contents/MacOS/mlx-serve`); dyld
/// resolves `@executable_path/../Frameworks` from the real path after
/// following the symlink, so the bundled dylibs load — verified live.
enum CLIInstaller {
    static let linkName = "mlx-serve"
    static let adminDirectory = "/usr/local/bin"

    struct Target: Equatable {
        let directory: String
        let requiresAdmin: Bool
    }

    enum Status: Equatable {
        case installed(linkPath: String)
        case notInstalled
    }

    enum InstallError: LocalizedError {
        case binaryNotFound
        case adminFailed(String)
        case unavailableInThisBuild

        var errorDescription: String? {
            switch self {
            case .binaryNotFound: return "mlx-serve binary not found in the app bundle."
            case .adminFailed(let msg): return msg
            case .unavailableInThisBuild:
                return "CLI install isn't available in the App Store build (it can't symlink outside the sandbox). Download the CLI from GitHub, or use the direct-download build of the app."
            }
        }
    }

    /// The App Store build can't symlink onto the user's PATH — writing to a
    /// shared location and the `osascript` admin prompt are both disallowed.
    static func requireInstallable() throws {
        guard BuildFeatures.current.cliInstaller else { throw InstallError.unavailableInThisBuild }
    }

    // MARK: - Pure decision logic

    /// Home bin dirs we're willing to symlink into, most-preferred first.
    static func homeBinCandidates(home: String) -> [String] {
        [home + "/.local/bin", home + "/bin"]
    }

    /// Pick the install target. A home candidate wins only when the directory
    /// EXISTS and is already on the shell PATH — a link in a dir the shell
    /// never searches would be dead weight, and we refuse to edit dotfiles to
    /// fix that. Otherwise fall back to /usr/local/bin (admin).
    static func selectTarget(home: String, existingDirs: Set<String>, pathEntries: [String]) -> Target {
        let onPath = Set(pathEntries.map(normalizeDir))
        for candidate in homeBinCandidates(home: home) {
            if existingDirs.contains(candidate), onPath.contains(candidate) {
                return Target(directory: candidate, requiresAdmin: false)
            }
        }
        return Target(directory: adminDirectory, requiresAdmin: true)
    }

    static func normalizeDir(_ dir: String) -> String {
        var d = dir
        while d.count > 1, d.hasSuffix("/") { d.removeLast() }
        return d
    }

    static func parsePathEntries(_ raw: String) -> [String] {
        raw.split(separator: ":").map(String.init).filter { !$0.isEmpty }
    }

    private static let pathMarkerBegin = "__MLX_PATH_BEGIN__"
    private static let pathMarkerEnd = "__MLX_PATH_END__"

    /// The command we ask the login shell to run. Markers isolate the PATH
    /// from any banner/echo noise an interactive rc file prints.
    static let pathProbeCommand =
        "printf '\\n\(pathMarkerBegin)%s\(pathMarkerEnd)\\n' \"$PATH\""

    static func extractPath(fromShellOutput output: String) -> String? {
        guard let begin = output.range(of: pathMarkerBegin),
              let end = output.range(of: pathMarkerEnd, range: begin.upperBound..<output.endIndex)
        else { return nil }
        return String(output[begin.upperBound..<end.lowerBound])
    }

    static func shellQuote(_ s: String) -> String {
        "'" + s.replacingOccurrences(of: "'", with: "'\\''") + "'"
    }

    static func adminInstallShellCommand(binarySource: String) -> String {
        "mkdir -p \(adminDirectory) && ln -sf \(shellQuote(binarySource)) \(adminDirectory)/\(linkName)"
    }

    // MARK: - Status + install

    /// Scan the candidate dirs (home bins + /usr/local/bin) for an `mlx-serve`
    /// symlink pointing at OUR binary. A foreign link (e.g. a Homebrew
    /// install) is not ours and reports notInstalled.
    static func status(binarySource: String, home: String, fm: FileManager = .default) -> Status {
        let normalizedSource = (binarySource as NSString).standardizingPath
        for dir in homeBinCandidates(home: home) + [adminDirectory] {
            let link = dir + "/" + linkName
            guard let dest = try? fm.destinationOfSymbolicLink(atPath: link) else { continue }
            let resolved = dest.hasPrefix("/")
                ? dest
                : (dir as NSString).appendingPathComponent(dest)
            if (resolved as NSString).standardizingPath == normalizedSource {
                return .installed(linkPath: link)
            }
        }
        return .notInstalled
    }

    /// Create (or repair) the symlink in a user-writable dir. Removes a stale
    /// or dangling link/file at the target name first.
    @discardableResult
    static func installIntoHomeBin(directory: String, binarySource: String) throws -> String {
        try requireInstallable()
        let fm = FileManager.default
        try fm.createDirectory(atPath: directory, withIntermediateDirectories: true)
        let link = directory + "/" + linkName
        // fileExists follows symlinks (false for a dangling link), so probe
        // the link itself too.
        if fm.fileExists(atPath: link) || (try? fm.destinationOfSymbolicLink(atPath: link)) != nil {
            try fm.removeItem(atPath: link)
        }
        try fm.createSymbolicLink(atPath: link, withDestinationPath: binarySource)
        return link
    }

    /// Symlink into /usr/local/bin via osascript with an admin prompt.
    /// Runs the Process off the main thread (call from a background task).
    @discardableResult
    static func installWithAdmin(binarySource: String) throws -> String {
        try requireInstallable()
        let shellCmd = adminInstallShellCommand(binarySource: binarySource)
        let escaped = shellCmd
            .replacingOccurrences(of: "\\", with: "\\\\")
            .replacingOccurrences(of: "\"", with: "\\\"")
        let script = "do shell script \"\(escaped)\" with administrator privileges"

        let p = Process()
        p.executableURL = URL(fileURLWithPath: "/usr/bin/osascript")
        p.arguments = ["-e", script]
        let errPipe = Pipe()
        p.standardOutput = Pipe()
        p.standardError = errPipe
        try p.run()
        p.waitUntilExit()
        if p.terminationStatus != 0 {
            let msg = String(data: errPipe.fileHandleForReading.readDataToEndOfFile(),
                             encoding: .utf8) ?? ""
            // User hitting Cancel on the password dialog lands here too.
            throw InstallError.adminFailed(
                msg.contains("User canceled") ? "Authorization was cancelled."
                    : "Install failed: \(msg.trimmingCharacters(in: .whitespacesAndNewlines))")
        }
        return adminDirectory + "/" + linkName
    }

    // MARK: - Environment probes (side-effecting, thin)

    /// Locate the mlx-serve binary to link against. Mirrors
    /// `ServerManager.resolveBinaryPath` (bundled first, dev zig-out second).
    static func resolveBinarySource() -> String? {
        if let execURL = Bundle.main.executableURL {
            let bundled = execURL.deletingLastPathComponent()
                .appendingPathComponent(linkName).path
            if FileManager.default.fileExists(atPath: bundled) { return bundled }
        }
        let repoRoot = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent() // Services/
            .deletingLastPathComponent() // MLXServe/
            .deletingLastPathComponent() // Sources/
            .deletingLastPathComponent() // app/
            .path
        let zigOut = (repoRoot as NSString).appendingPathComponent("zig-out/bin/" + linkName)
        if FileManager.default.fileExists(atPath: zigOut) { return zigOut }
        return nil
    }

    /// The user's SHELL PATH, not this GUI process's (Finder-launched apps get
    /// a minimal PATH that never contains ~/.local/bin). Asks the login+
    /// interactive shell to print $PATH between markers; falls back to the
    /// process PATH if the shell misbehaves or hangs.
    static func userShellPathEntries() -> [String] {
        let fallback = parsePathEntries(ProcessInfo.processInfo.environment["PATH"] ?? "")
        let shell = ProcessInfo.processInfo.environment["SHELL"] ?? "/bin/zsh"

        let p = Process()
        p.executableURL = URL(fileURLWithPath: shell)
        p.arguments = ["-l", "-i", "-c", pathProbeCommand]
        let outPipe = Pipe()
        p.standardOutput = outPipe
        p.standardError = Pipe()
        p.standardInput = FileHandle.nullDevice
        do { try p.run() } catch { return fallback }

        // Watchdog: a pathological rc file must not wedge the app.
        let deadline = DispatchWorkItem { if p.isRunning { p.terminate() } }
        DispatchQueue.global().asyncAfter(deadline: .now() + 5, execute: deadline)
        let data = outPipe.fileHandleForReading.readDataToEndOfFile()
        p.waitUntilExit()
        deadline.cancel()

        guard let out = String(data: data, encoding: .utf8),
              let path = extractPath(fromShellOutput: out),
              !path.isEmpty
        else { return fallback }
        return parsePathEntries(path)
    }

    /// One-shot probe for the UI: what's the current state, and where would
    /// an install go? Spawns the user's shell — call off the main thread.
    enum Probe: Equatable {
        case installed(linkPath: String)
        case available(Target)
        case binaryMissing
    }

    static func probe() -> Probe {
        guard let source = resolveBinarySource() else { return .binaryMissing }
        let home = NSHomeDirectory()
        if case .installed(let link) = status(binarySource: source, home: home) {
            return .installed(linkPath: link)
        }
        let fm = FileManager.default
        var isDir: ObjCBool = false
        let existing = Set(homeBinCandidates(home: home).filter {
            fm.fileExists(atPath: $0, isDirectory: &isDir) && isDir.boolValue
        })
        return .available(selectTarget(home: home,
                                       existingDirs: existing,
                                       pathEntries: userShellPathEntries()))
    }
}
