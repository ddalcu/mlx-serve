import Foundation

/// Drives native media generation through an on-demand `mlx-serve --serve`
/// subprocess instead of the Python venv. Each generation model (Qwen3-TTS for
/// audio, FLUX.2-klein for image) is served single-model by its own serve-mode
/// instance — the same self-contained request→file path the embedded ds4/llama
/// engines use — kept warm per model dir and driven over HTTP.
///
/// This replaces `PythonManager.runScript` for audio + image. Video stays on
/// Python until its native engine lands.
@MainActor
final class NativeGenServer {
    static let shared = NativeGenServer()

    private var proc: Process?
    private var port: Int = 0
    private var servedDir: String = ""
    private(set) var lastLog: String = ""

    enum GenServerError: LocalizedError {
        case binaryMissing(String)
        case modelMissing(String)
        case exited(String)
        case timeout
        case http(Int, String)

        var errorDescription: String? {
            switch self {
            case .binaryMissing(let p): return "mlx-serve binary not found at \(p)."
            case .modelMissing(let r): return "Model \(r) is not downloaded. Download it first."
            case .exited(let s): return "Generation server exited: \(s)"
            case .timeout: return "Generation server did not become ready in time."
            case .http(let c, let m): return "Server returned HTTP \(c): \(m)"
            }
        }
    }

    /// Ensure a serve-mode instance is running for `dir`; returns its port,
    /// reusing a warm one for the same model.
    func ensure(modelDir dir: String) async throws -> Int {
        if let p = proc, p.isRunning, servedDir == dir { return port }
        stop()

        let bin = Self.resolveBinaryPath()
        guard FileManager.default.fileExists(atPath: bin) || bin == "mlx-serve" else {
            throw GenServerError.binaryMissing(bin)
        }
        let chosen = Self.freePort()
        let p = Process()
        p.executableURL = URL(fileURLWithPath: bin)
        p.arguments = ["--model", dir, "--serve", "--host", "127.0.0.1", "--port", String(chosen)]
        // Finder-launched apps inherit a minimal PATH; prepend Homebrew so any
        // child lookups match the user's shell (mirrors PythonManager).
        var env = ProcessInfo.processInfo.environment
        let brew = ["/opt/homebrew/bin", "/usr/local/bin"]
        let existing = env["PATH"] ?? "/usr/bin:/bin:/usr/sbin:/sbin"
        env["PATH"] = (brew + [existing]).joined(separator: ":")
        p.environment = env

        let pipe = Pipe()
        p.standardOutput = pipe
        p.standardError = pipe
        pipe.fileHandleForReading.readabilityHandler = { [weak self] h in
            let d = h.availableData
            guard !d.isEmpty, let s = String(data: d, encoding: .utf8) else { return }
            Task { @MainActor in
                self?.lastLog = String((self?.lastLog ?? "" + s).suffix(8000))
            }
        }
        try p.run()
        proc = p
        port = chosen
        servedDir = dir
        try await waitHealthy(timeout: 240)
        return chosen
    }

    /// POST a JSON body, returning the raw response bytes (WAV, or JSON).
    func post(path: String, json: [String: Any]) async throws -> Data {
        var req = URLRequest(url: URL(string: "http://127.0.0.1:\(port)\(path)")!)
        req.httpMethod = "POST"
        req.setValue("application/json", forHTTPHeaderField: "Content-Type")
        req.httpBody = try JSONSerialization.data(withJSONObject: json)
        req.timeoutInterval = 900
        let (data, resp) = try await URLSession.shared.data(for: req)
        let code = (resp as? HTTPURLResponse)?.statusCode ?? -1
        guard code == 200 else {
            throw GenServerError.http(code, String(data: data.prefix(400), encoding: .utf8) ?? "")
        }
        return data
    }

    /// POST with `stream:true` and yield each parsed SSE `data:` event as a
    /// dictionary (`{type:"progress"|"complete"|"error", …}`). The connection is
    /// torn down when the consuming task is cancelled.
    nonisolated func postStream(path: String, json: [String: Any], port: Int) -> AsyncThrowingStream<[String: Any], Error> {
        AsyncThrowingStream { continuation in
            let task = Task {
                do {
                    var body = json
                    body["stream"] = true
                    var req = URLRequest(url: URL(string: "http://127.0.0.1:\(port)\(path)")!)
                    req.httpMethod = "POST"
                    req.setValue("application/json", forHTTPHeaderField: "Content-Type")
                    req.setValue("text/event-stream", forHTTPHeaderField: "Accept")
                    req.httpBody = try JSONSerialization.data(withJSONObject: body)
                    req.timeoutInterval = 900
                    let (bytes, resp) = try await URLSession.shared.bytes(for: req)
                    let code = (resp as? HTTPURLResponse)?.statusCode ?? -1
                    guard code == 200 else { throw GenServerError.http(code, "stream start failed") }
                    for try await line in bytes.lines {
                        guard line.hasPrefix("data: ") else { continue }
                        let payload = String(line.dropFirst(6))
                        if let d = payload.data(using: .utf8),
                           let obj = try? JSONSerialization.jsonObject(with: d) as? [String: Any] {
                            continuation.yield(obj)
                        }
                    }
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
            continuation.onTermination = { _ in task.cancel() }
        }
    }

    func stop() {
        proc?.terminate()
        proc = nil
        port = 0
        servedDir = ""
    }

    private func waitHealthy(timeout: TimeInterval) async throws {
        let deadline = Date().addingTimeInterval(timeout)
        while Date() < deadline {
            if let p = proc, !p.isRunning {
                throw GenServerError.exited(String(lastLog.suffix(600)))
            }
            if await healthOK() { return }
            try? await Task.sleep(nanoseconds: 500_000_000)
        }
        throw GenServerError.timeout
    }

    private func healthOK() async -> Bool {
        guard let url = URL(string: "http://127.0.0.1:\(port)/health") else { return false }
        var r = URLRequest(url: url)
        r.timeoutInterval = 2
        guard let (_, resp) = try? await URLSession.shared.data(for: r) else { return false }
        return (resp as? HTTPURLResponse)?.statusCode == 200
    }

    // MARK: - Resolution helpers

    /// Bundled `Contents/MacOS/mlx-serve`, else dev `zig-out/bin/mlx-serve`,
    /// else bare name (mirrors ServerManager.resolveBinaryPath).
    static func resolveBinaryPath() -> String {
        if let exec = Bundle.main.executableURL {
            let bundled = exec.deletingLastPathComponent().appendingPathComponent("mlx-serve").path
            if FileManager.default.fileExists(atPath: bundled) { return bundled }
        }
        let repoRoot = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent() // Services
            .deletingLastPathComponent() // MLXServe
            .deletingLastPathComponent() // Sources
            .deletingLastPathComponent() // app
            .path
        let zigOut = (repoRoot as NSString).appendingPathComponent("zig-out/bin/mlx-serve")
        if FileManager.default.fileExists(atPath: zigOut) { return zigOut }
        return "mlx-serve"
    }

    /// Resolve a HuggingFace repo id to a local model directory: the app's
    /// `~/.mlx-serve/models/<repo>` download, else the HuggingFace cache snapshot
    /// (mlx-serve's loader follows the cache's symlinks). nil if neither exists.
    static func resolveModelDir(repo: String) -> String? {
        let fm = FileManager.default
        let modelsRoot = NSString(string: "~/.mlx-serve/models").expandingTildeInPath
        if let dir = DownloadManager.existingModelDir(rootDir: modelsRoot, repoId: repo),
           fm.fileExists(atPath: (dir as NSString).appendingPathComponent("config.json")) {
            return dir
        }
        // HF cache: ~/.cache/huggingface/hub/models--<owner>--<name>/snapshots/<latest>
        let slug = repo.replacingOccurrences(of: "/", with: "--")
        let snaps = NSString(string: "~/.cache/huggingface/hub/models--\(slug)/snapshots").expandingTildeInPath
        if let entries = try? fm.contentsOfDirectory(atPath: snaps) {
            for e in entries.sorted(by: >) {
                let dir = (snaps as NSString).appendingPathComponent(e)
                if fm.fileExists(atPath: (dir as NSString).appendingPathComponent("config.json")) {
                    return dir
                }
            }
        }
        return nil
    }

    /// A free localhost TCP port (bind to 0, read the assignment, release).
    static func freePort() -> Int {
        let fd = socket(AF_INET, SOCK_STREAM, 0)
        guard fd >= 0 else { return 8830 }
        defer { close(fd) }
        var addr = sockaddr_in()
        addr.sin_family = sa_family_t(AF_INET)
        addr.sin_addr.s_addr = inet_addr("127.0.0.1")
        addr.sin_port = 0
        let bound = withUnsafePointer(to: &addr) {
            $0.withMemoryRebound(to: sockaddr.self, capacity: 1) {
                bind(fd, $0, socklen_t(MemoryLayout<sockaddr_in>.size))
            }
        }
        guard bound == 0 else { return 8830 }
        var len = socklen_t(MemoryLayout<sockaddr_in>.size)
        withUnsafeMutablePointer(to: &addr) {
            _ = $0.withMemoryRebound(to: sockaddr.self, capacity: 1) {
                getsockname(fd, $0, &len)
            }
        }
        return Int(UInt16(bigEndian: addr.sin_port))
    }
}
