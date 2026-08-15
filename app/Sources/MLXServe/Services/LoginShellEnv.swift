import Foundation

/// The user's SHELL environment, not this process's.
///
/// A bundle launched from Finder gets a minimal environment that never carries
/// anything set in `~/.zshrc` / `~/.zprofile` — so `HF_HOME`, the variable
/// people actually use to move the Hugging Face cache off the boot drive, is
/// invisible to `ProcessInfo.processInfo.environment` in the shipped app while
/// working fine when the binary is run from a terminal. Asking the login+
/// interactive shell to print the values is the only way to see them.
///
/// Spawning a shell is slow and can be wedged by a pathological rc file, so
/// `values(of:)` is watchdogged and callers prime it OFF the main thread.
enum LoginShellEnv {

    static func beginMarker(_ name: String) -> String { "__MLX_ENV_\(name)_BEGIN__" }
    static func endMarker(_ name: String) -> String { "__MLX_ENV_\(name)_END__" }

    /// What we ask the shell to run. Markers isolate each value from the
    /// banner/echo noise an interactive rc file prints.
    static func probeCommand(_ names: [String]) -> String {
        names.map { name in
            "printf '\\n\(beginMarker(name))%s\(endMarker(name))\\n' \"$\(name)\""
        }.joined(separator: "; ")
    }

    /// Values found between the markers. A name whose value is EMPTY is
    /// omitted, not returned as `""` — an unset variable must never shadow one
    /// the process itself carries.
    static func parse(_ names: [String], fromShellOutput output: String) -> [String: String] {
        var out: [String: String] = [:]
        for name in names {
            guard let begin = output.range(of: beginMarker(name)),
                  let end = output.range(of: endMarker(name), range: begin.upperBound..<output.endIndex)
            else { continue }
            let value = String(output[begin.upperBound..<end.lowerBound])
                .trimmingCharacters(in: .whitespacesAndNewlines)
            if !value.isEmpty { out[name] = value }
        }
        return out
    }

    /// Fill in only what the process environment LACKS. A launch that does
    /// carry the variable (terminal, or an explicit override) is authoritative.
    static func merge(shell: [String: String], into process: [String: String]) -> [String: String] {
        var out = process
        for (key, value) in shell where (out[key]?.isEmpty ?? true) {
            out[key] = value
        }
        return out
    }

    /// Run the probe. Blocking — never call this on the main thread.
    static func values(of names: [String]) -> [String: String] {
        guard !names.isEmpty else { return [:] }
        let shell = ProcessInfo.processInfo.environment["SHELL"] ?? "/bin/zsh"

        let p = Process()
        p.executableURL = URL(fileURLWithPath: shell)
        p.arguments = ["-l", "-i", "-c", probeCommand(names)]
        let outPipe = Pipe()
        p.standardOutput = outPipe
        p.standardError = Pipe()
        p.standardInput = FileHandle.nullDevice
        do { try p.run() } catch { return [:] }

        let deadline = DispatchWorkItem { if p.isRunning { p.terminate() } }
        DispatchQueue.global().asyncAfter(deadline: .now() + 5, execute: deadline)
        let data = outPipe.fileHandleForReading.readDataToEndOfFile()
        p.waitUntilExit()
        deadline.cancel()

        guard let text = String(data: data, encoding: .utf8) else { return [:] }
        return parse(names, fromShellOutput: text)
    }

    // MARK: - Hugging Face

    /// Everything that can move the HF cache or carry its token.
    static let huggingFaceNames = ["HF_HUB_CACHE", "HF_HOME", "XDG_CACHE_HOME", "HF_TOKEN"]

    private static let lock = NSLock()
    nonisolated(unsafe) private static var cachedHF: [String: String] = [:]

    /// The HF variables, process environment first and the login shell filling
    /// the gaps. Cheap and non-blocking: it returns the process environment
    /// until `primeHuggingFace()` has run.
    static func huggingFaceEnvironment(
        process: [String: String] = ProcessInfo.processInfo.environment
    ) -> [String: String] {
        lock.lock()
        let shellValues = cachedHF
        lock.unlock()
        return merge(shell: shellValues, into: process)
    }

    /// Spawn the shell once and cache what it says. Blocking — off-main only.
    ///
    /// Gated on `customModelFolders`: under MAS the app is sandboxed, so a
    /// cache outside the container is unreadable no matter what the shell says,
    /// and `~` already resolves to the container's own `.cache`. Nothing to
    /// find, so nothing is spawned.
    static func primeHuggingFace() {
        guard BuildFeatures.current.customModelFolders else { return }
        let found = values(of: huggingFaceNames)
        lock.lock()
        cachedHF = found
        lock.unlock()
    }
}
