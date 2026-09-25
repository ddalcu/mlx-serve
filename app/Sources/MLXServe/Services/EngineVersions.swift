import Foundation

/// One `name → version` row from `mlx-serve --version`.
struct EngineVersion: Identifiable, Equatable, Sendable {
    let name: String
    let version: String
    var id: String { name }
}

/// Reads the app + embedded-engine versions from `mlx-serve --version` WITHOUT
/// booting the server. `--version` is a print-and-exit path (no port bind, no
/// model load), so `probe` spawns the bundled binary as a one-shot subprocess
/// and parses its output — letting Settings show MLX / mlx-c / ggml / llama.cpp
/// / GGUF / ds4 versions even when the server isn't running. The Zig side
/// (src/version.zig) is the source of the output format.
enum EngineVersions {

    /// Parse `--version` stdout: one `name value` line per component. The first
    /// whitespace-delimited token is the component name; the remainder (trimmed)
    /// is its version, which may contain spaces (`ggml 0.16.0 (47c786924)`).
    /// Blank lines and single-token lines are skipped. Pure + testable.
    static func parse(_ text: String) -> [EngineVersion] {
        text.split(whereSeparator: \.isNewline).compactMap { rawLine -> EngineVersion? in
            let line = rawLine.trimmingCharacters(in: .whitespaces)
            guard let spaceIdx = line.firstIndex(where: \.isWhitespace) else { return nil }
            let name = String(line[..<spaceIdx])
            let version = line[line.index(after: spaceIdx)...].trimmingCharacters(in: .whitespaces)
            guard !name.isEmpty, !version.isEmpty else { return nil }
            return EngineVersion(name: name, version: version)
        }
    }

    /// The embedded-engine rows only — drops the `mlx-serve` app row (already
    /// shown as "Installed version" in Settings).
    static func engineRows(from text: String) -> [EngineVersion] {
        parse(text).filter { $0.name != "mlx-serve" }
    }

    static func naxAvailable(in versions: [EngineVersion]) -> Bool {
        versions.first { $0.name == "nax" }?.version
            .split(whereSeparator: \.isWhitespace).first == "on"
    }

    /// Human label for a `--version` component name (Settings row title).
    /// Unknown names fall back to the raw token so a new Zig line still
    /// renders (ugly beats invisible).
    static func displayLabel(_ name: String) -> String {
        switch name {
        case "mlx": return "MLX"
        case "mlx-c": return "mlx-c"
        case "nax": return "M5 Neural Accelerators"
        case "ggml": return "ggml"
        case "llama.cpp": return "llama.cpp"
        case "gguf": return "GGUF format"
        case "ds4": return "ds4"
        default: return name
        }
    }

    /// Explainer text under a component row; empty for unknown names.
    static func explainer(_ name: String) -> String {
        switch name {
        case "mlx": return "Apple's MLX array framework — the native engine for `.safetensors` models."
        case "mlx-c": return "The C bindings the server links MLX through (pinned submodule revision)."
        case "nax": return "Whether MLX dispatches to the M5 GPU's neural accelerators (NAX). The bundled MLX always ships the NAX kernels; \"on\" needs an M5-class GPU on macOS 26.2+."
        case "ggml": return "The tensor library under the llama.cpp GGUF engine (with its short commit)."
        case "llama.cpp": return "Pinned llama.cpp release serving `.gguf` models. Ships inside the app download."
        case "gguf": return "GGUF file-format version the engine reads."
        case "ds4": return "Embedded ds4 engine (DeepSeek-V4-Flash), pinned commit."
        default: return ""
        }
    }

    /// Spawn `<binaryPath> --version` as a one-shot subprocess and parse it.
    /// Returns [] if the binary is missing or the run fails. Runs off the main
    /// thread; the process prints and exits immediately (no server, no port).
    static func probe(binaryPath: String) async -> [EngineVersion] {
        guard FileManager.default.fileExists(atPath: binaryPath) else { return [] }
        return await withCheckedContinuation { continuation in
            DispatchQueue.global(qos: .utility).async {
                let proc = Process()
                proc.executableURL = URL(fileURLWithPath: binaryPath)
                proc.arguments = ["--version"]
                let out = Pipe()
                proc.standardOutput = out
                proc.standardError = Pipe()   // swallow any stderr noise
                do {
                    try proc.run()
                } catch {
                    continuation.resume(returning: [])
                    return
                }
                let data = out.fileHandleForReading.readDataToEndOfFile()
                proc.waitUntilExit()
                let text = String(data: data, encoding: .utf8) ?? ""
                continuation.resume(returning: parse(text))
            }
        }
    }
}
