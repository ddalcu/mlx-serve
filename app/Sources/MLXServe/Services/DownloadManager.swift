import Foundation

@MainActor
class DownloadManager: ObservableObject {
    @Published var downloads: [String: DownloadState] = [:]

    struct DownloadState {
        var progress: Double = 0
        var status: Status = .idle
        var statusText: String = ""
        var error: String?
        var currentFile: String = ""
        var fileIndex: Int = 0
        var fileCount: Int = 0
        var bytesPerSecond: Double = 0
        var fileProgress: Double = 0

        enum Status: Equatable {
            case idle, downloading, completed, failed
        }

        var speedFormatted: String {
            if bytesPerSecond > 1_000_000 {
                return String(format: "%.1f MB/s", bytesPerSecond / 1_000_000)
            } else if bytesPerSecond > 1_000 {
                return String(format: "%.0f KB/s", bytesPerSecond / 1_000)
            }
            return ""
        }

        var percentFormatted: String {
            String(format: "%.0f%%", fileProgress * 100)
        }
    }

    let modelsDir: String = {
        let path = NSString(string: "~/.mlx-serve/models").expandingTildeInPath
        try? FileManager.default.createDirectory(atPath: path, withIntermediateDirectories: true)
        return path
    }()

    /// Check if a model has all required files for loading.
    /// Verifies: config.json, tokenizer files, chat template, and ALL safetensors shards.
    func isReady(_ repoId: String) -> Bool {
        let name = repoId.split(separator: "/").last.map(String.init) ?? repoId
        let modelDir = (modelsDir as NSString).appendingPathComponent(name)
        let fm = FileManager.default

        // Must have config.json
        guard fm.fileExists(atPath: (modelDir as NSString).appendingPathComponent("config.json")) else { return false }

        // Must have tokenizer (tokenizer.json or tokenizer.model)
        let hasTokenizer = fm.fileExists(atPath: (modelDir as NSString).appendingPathComponent("tokenizer.json"))
            || fm.fileExists(atPath: (modelDir as NSString).appendingPathComponent("tokenizer.model"))
        guard hasTokenizer else { return false }

        guard let entries = try? fm.contentsOfDirectory(atPath: modelDir) else { return false }
        let safetensors = entries.filter { $0.hasSuffix(".safetensors") }

        // Must have at least one safetensors file
        guard !safetensors.isEmpty else { return false }

        // If sharded (model.safetensors.index.json exists), check all shards are present
        let indexPath = (modelDir as NSString).appendingPathComponent("model.safetensors.index.json")
        if fm.fileExists(atPath: indexPath) {
            if let data = fm.contents(atPath: indexPath),
               let index = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
               let weightMap = index["weight_map"] as? [String: String] {
                let requiredShards = Set(weightMap.values)
                for shard in requiredShards {
                    let shardPath = (modelDir as NSString).appendingPathComponent(shard)
                    guard fm.fileExists(atPath: shardPath) else { return false }
                    // Check it's not a zero-byte stub
                    let size = (try? fm.attributesOfItem(atPath: shardPath)[.size] as? UInt64) ?? 0
                    guard size > 0 else { return false }
                }
            }
        } else {
            // Single-file model — check the safetensors file is non-trivial
            guard let first = safetensors.first else { return false }
            let fullPath = (modelDir as NSString).appendingPathComponent(first)
            let size = (try? fm.attributesOfItem(atPath: fullPath)[.size] as? UInt64) ?? 0
            guard size > 1_000_000 else { return false }
        }

        return true
    }

    func modelPath(for repoId: String) -> String {
        let name = repoId.split(separator: "/").last.map(String.init) ?? repoId
        return (modelsDir as NSString).appendingPathComponent(name)
    }

    func download(repoId: String) async {
        let name = repoId.split(separator: "/").last.map(String.init) ?? repoId
        let destDir = (modelsDir as NSString).appendingPathComponent(name)

        downloads[repoId] = DownloadState(status: .downloading, statusText: "Fetching file list...")

        do {
            try FileManager.default.createDirectory(atPath: destDir, withIntermediateDirectories: true)

            let listURL = URL(string: "https://huggingface.co/api/models/\(repoId)/tree/main")!
            let (listData, _) = try await URLSession.shared.data(from: listURL)
            guard let files = try JSONSerialization.jsonObject(with: listData) as? [[String: Any]] else {
                throw URLError(.cannotParseResponse)
            }

            let neededExtensions = Set(["json", "safetensors", "jinja", "model", "txt"])
            let neededFiles = files.compactMap { file -> (String, Int64)? in
                guard let path = file["path"] as? String,
                      let ftype = file["type"] as? String, ftype == "file" else { return nil }
                let ext = (path as NSString).pathExtension
                if neededExtensions.contains(ext) || path == "chat_template.jinja" {
                    let size = file["size"] as? Int64 ?? (file["size"] as? Int).map { Int64($0) } ?? 0
                    return (path, size)
                }
                return nil
            }

            let totalSize = neededFiles.reduce(Int64(0)) { $0 + $1.1 }
            var downloadedSize: Int64 = 0

            // Pre-check disk space
            let destURL = URL(fileURLWithPath: destDir)
            if let values = try? destURL.resourceValues(forKeys: [.volumeAvailableCapacityForImportantUsageKey]),
               let available = values.volumeAvailableCapacityForImportantUsage,
               available < totalSize {
                throw NSError(domain: NSCocoaErrorDomain, code: NSFileWriteOutOfSpaceError, userInfo: [
                    NSLocalizedDescriptionKey: "Not enough disk space. Need \(formatBytes(totalSize)) but only \(formatBytes(Int64(available))) available."
                ])
            }

            for (idx, (filePath, fileSize)) in neededFiles.enumerated() {
                let destPath = (destDir as NSString).appendingPathComponent(filePath)
                let partialPath = destPath + ".partial"

                // Create subdirectories if needed
                let parentDir = (destPath as NSString).deletingLastPathComponent
                try? FileManager.default.createDirectory(atPath: parentDir, withIntermediateDirectories: true)

                // Skip if already exists with right size
                if let attrs = try? FileManager.default.attributesOfItem(atPath: destPath),
                   let existingSize = attrs[.size] as? Int64,
                   existingSize == fileSize && fileSize > 0 {
                    downloadedSize += fileSize
                    downloads[repoId]?.progress = totalSize > 0 ? Double(downloadedSize) / Double(totalSize) : 0
                    downloads[repoId]?.statusText = "Skipped \(filePath) (exists)"
                    downloads[repoId]?.fileIndex = idx + 1
                    downloads[repoId]?.fileCount = neededFiles.count
                    continue
                }

                let sizeStr = formatBytes(fileSize)
                downloads[repoId]?.currentFile = (filePath as NSString).lastPathComponent
                downloads[repoId]?.fileIndex = idx + 1
                downloads[repoId]?.fileCount = neededFiles.count
                downloads[repoId]?.fileProgress = 0
                downloads[repoId]?.bytesPerSecond = 0
                downloads[repoId]?.statusText = "\(filePath) (\(sizeStr))"

                let fileURL = URL(string: "https://huggingface.co/\(repoId)/resolve/main/\(filePath)")!
                let maxRetries = 20

                for attempt in 0..<maxRetries {
                    try Task.checkCancellation()

                    // Check for existing partial download
                    let existingBytes: Int64
                    if let attrs = try? FileManager.default.attributesOfItem(atPath: partialPath),
                       let size = attrs[.size] as? Int64, size > 0 {
                        existingBytes = size
                        downloads[repoId]?.statusText = "Resuming \(filePath) from \(formatBytes(existingBytes))..."
                        downloads[repoId]?.fileProgress = fileSize > 0 ? Double(existingBytes) / Double(fileSize) : 0
                    } else {
                        existingBytes = 0
                    }

                    // Create or open partial file
                    let fm = FileManager.default
                    if !fm.fileExists(atPath: partialPath) {
                        fm.createFile(atPath: partialPath, contents: nil)
                    }
                    guard let fileHandle = FileHandle(forWritingAtPath: partialPath) else {
                        throw URLError(.cannotCreateFile)
                    }
                    try fileHandle.seekToEnd()

                    var request = URLRequest(url: fileURL)
                    if existingBytes > 0 {
                        request.setValue("bytes=\(existingBytes)-", forHTTPHeaderField: "Range")
                    }

                    do {
                        try await downloadToFile(
                            request: request,
                            fileHandle: fileHandle,
                            repoId: repoId,
                            fileSize: fileSize,
                            existingBytes: existingBytes,
                            baseDownloaded: downloadedSize,
                            totalSize: totalSize
                        )

                        // Success — move partial to final destination
                        try? fm.removeItem(atPath: destPath)
                        try fm.moveItem(atPath: partialPath, toPath: destPath)
                        break
                    } catch is CancellationError {
                        // Keep partial file for future resume
                        throw CancellationError()
                    } catch {
                        // Partial file stays on disk — next attempt resumes from it
                        if attempt < maxRetries - 1 {
                            let isStall = error is DownloadStallError
                            let delay = isStall ? 2.0 : Double(attempt + 1) * 2.0
                            let reason = isStall ? "Download stalled" : "Connection lost"
                            downloads[repoId]?.statusText = "\(reason), retrying in \(Int(delay))s... (\(attempt + 2)/\(maxRetries))"
                            try await Task.sleep(nanoseconds: UInt64(delay * 1_000_000_000))
                        } else {
                            throw error
                        }
                    }
                }

                downloadedSize += fileSize
                downloads[repoId]?.progress = totalSize > 0 ? Double(downloadedSize) / Double(totalSize) : 0
                downloads[repoId]?.fileProgress = 1.0
            }

            downloads[repoId] = DownloadState(progress: 1.0, status: .completed, statusText: "Complete",
                                               fileIndex: neededFiles.count, fileCount: neededFiles.count)
        } catch {
            downloads[repoId] = DownloadState(status: .failed, error: error.localizedDescription)
        }
    }

    /// Stream download directly to a file on disk (survives interruptions).
    /// Uses dataTask so bytes are written as they arrive — the .partial file always
    /// reflects how far we got, enabling Range-header resume on retry.
    private func downloadToFile(
        request: URLRequest,
        fileHandle: FileHandle,
        repoId: String,
        fileSize: Int64,
        existingBytes: Int64,
        baseDownloaded: Int64,
        totalSize: Int64
    ) async throws {
        let delegate = StreamingDelegate(fileHandle: fileHandle, existingBytes: existingBytes)
        let config = URLSessionConfiguration.default
        config.timeoutIntervalForRequest = 60
        config.timeoutIntervalForResource = 7200
        let session = URLSession(configuration: config, delegate: delegate, delegateQueue: nil)
        defer { session.finishTasksAndInvalidate() }

        try await withTaskCancellationHandler {
            try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<Void, Error>) in
                delegate.onProgress = { [weak self] fileBytesTotal, speed in
                    let fileProgress = fileSize > 0 ? Double(fileBytesTotal) / Double(fileSize) : 0
                    let overallDownloaded = baseDownloaded + fileBytesTotal
                    Task { @MainActor [weak self] in
                        self?.downloads[repoId]?.fileProgress = fileProgress
                        self?.downloads[repoId]?.bytesPerSecond = speed
                        self?.downloads[repoId]?.progress = totalSize > 0 ? Double(overallDownloaded) / Double(totalSize) : 0
                    }
                }
                delegate.onComplete = { error in
                    if let error { continuation.resume(throwing: error) }
                    else { continuation.resume() }
                }
                session.dataTask(with: request).resume()
            }
        } onCancel: {
            session.invalidateAndCancel()
        }
    }

    /// Check whether a model has .partial files from an interrupted download.
    func hasPartialDownload(_ repoId: String) -> Bool {
        let name = repoId.split(separator: "/").last.map(String.init) ?? repoId
        let modelDir = (modelsDir as NSString).appendingPathComponent(name)
        guard let entries = try? FileManager.default.contentsOfDirectory(atPath: modelDir) else { return false }
        return entries.contains { $0.hasSuffix(".partial") }
    }

    func discoverLocalModels() -> [LocalModel] {
        guard let entries = try? FileManager.default.contentsOfDirectory(atPath: modelsDir) else {
            return []
        }
        return entries.compactMap { name in
            guard !name.hasPrefix(".") else { return nil }
            var dirPath = (modelsDir as NSString).appendingPathComponent(name)
            dirPath = (dirPath as NSString).resolvingSymlinksInPath

            let configPath = (dirPath as NSString).appendingPathComponent("config.json")
            guard FileManager.default.fileExists(atPath: configPath) else { return nil }

            let dirEntries = (try? FileManager.default.contentsOfDirectory(atPath: dirPath)) ?? []
            let hasSafetensors = dirEntries.contains { $0.hasSuffix(".safetensors") && !$0.hasSuffix(".index.json") }
            guard hasSafetensors else { return nil }

            let size = directorySize(dirPath)
            return LocalModel(
                id: name,
                name: name,
                path: dirPath,
                sizeFormatted: MemoryInfo.format(Int64(size))
            )
        }.sorted { $0.name < $1.name }
    }

    func removeIncomplete(repoId: String) {
        let name = repoId.split(separator: "/").last.map(String.init) ?? repoId
        let destDir = (modelsDir as NSString).appendingPathComponent(name)
        try? FileManager.default.removeItem(atPath: destDir)
        downloads.removeValue(forKey: repoId)
    }

    private func directorySize(_ path: String) -> UInt64 {
        guard let enumerator = FileManager.default.enumerator(atPath: path) else { return 0 }
        var total: UInt64 = 0
        while let file = enumerator.nextObject() as? String {
            let fullPath = (path as NSString).appendingPathComponent(file)
            if let attrs = try? FileManager.default.attributesOfItem(atPath: fullPath),
               let size = attrs[.size] as? UInt64 {
                total += size
            }
        }
        return total
    }

    private func formatBytes(_ bytes: Int64) -> String {
        if bytes > 1_000_000_000 { return String(format: "%.1f GB", Double(bytes) / 1e9) }
        if bytes > 1_000_000 { return String(format: "%.0f MB", Double(bytes) / 1e6) }
        return "\(bytes) B"
    }
}

// MARK: - Streaming Download Delegate

/// Thrown when a download stalls (speed below threshold for too long) — triggers auto-retry.
private struct DownloadStallError: Error, LocalizedError {
    var errorDescription: String? { "Download stalled — server stopped sending data" }
}

/// Writes received data directly to a file handle as it arrives.
/// If the server returns 200 instead of 206, truncates the file (Range was ignored).
private class StreamingDelegate: NSObject, URLSessionDataDelegate {
    let fileHandle: FileHandle
    var existingBytes: Int64
    var onProgress: ((Int64, Double) -> Void)?   // fileBytesTotal, speed
    var onComplete: ((Error?) -> Void)?
    private var bytesReceived: Int64 = 0
    private var statusCode: Int = 0
    private var writeError: Error?
    private let startTime = Date()
    private var lastProgressUpdate = Date.distantPast

    // Stall detection — cancels the task if speed stays below threshold
    private weak var activeTask: URLSessionDataTask?
    private(set) var stalledOut = false
    private var stallTimer: DispatchSourceTimer?
    private var stallCheckBytes: Int64 = 0
    private var slowSince: Date?
    private static let stallSpeedThreshold: Double = 10_000  // 10 KB/s
    private static let stallTimeout: TimeInterval = 30

    init(fileHandle: FileHandle, existingBytes: Int64) {
        self.fileHandle = fileHandle
        self.existingBytes = existingBytes
    }

    deinit {
        stallTimer?.cancel()
    }

    private func startStallDetection(task: URLSessionDataTask) {
        activeTask = task
        stallCheckBytes = bytesReceived
        let timer = DispatchSource.makeTimerSource(queue: .global(qos: .utility))
        timer.schedule(deadline: .now() + 5, repeating: 5)
        timer.setEventHandler { [weak self] in
            self?.checkForStall()
        }
        timer.resume()
        stallTimer = timer
    }

    private func checkForStall() {
        let currentBytes = bytesReceived
        let bytesSinceCheck = currentBytes - stallCheckBytes
        let recentSpeed = Double(bytesSinceCheck) / 5.0  // ~5s interval
        stallCheckBytes = currentBytes

        // Push real-time speed to UI (prevents stale speed when data stops flowing)
        onProgress?(existingBytes + currentBytes, recentSpeed)

        if recentSpeed < Self.stallSpeedThreshold {
            if slowSince == nil {
                slowSince = Date()
            } else if Date().timeIntervalSince(slowSince!) > Self.stallTimeout {
                stalledOut = true
                activeTask?.cancel()
                stallTimer?.cancel()
            }
        } else {
            slowSince = nil
        }
    }

    func urlSession(_ session: URLSession, dataTask: URLSessionDataTask,
                    didReceive response: URLResponse,
                    completionHandler: @escaping (URLSession.ResponseDisposition) -> Void) {
        statusCode = (response as? HTTPURLResponse)?.statusCode ?? 0
        if statusCode == 200 {
            // Server ignored Range header — sending full file; start over
            existingBytes = 0
            do {
                try fileHandle.truncate(atOffset: 0)
                try fileHandle.seek(toOffset: 0)
            } catch {
                writeError = error
                completionHandler(.cancel)
                return
            }
            startStallDetection(task: dataTask)
            completionHandler(.allow)
        } else if statusCode == 206 {
            startStallDetection(task: dataTask)
            completionHandler(.allow)
        } else {
            completionHandler(.cancel)
        }
    }

    func urlSession(_ session: URLSession, dataTask: URLSessionDataTask, didReceive data: Data) {
        guard writeError == nil else { return }
        do {
            try fileHandle.write(contentsOf: data)
        } catch {
            writeError = error
            dataTask.cancel()
            return
        }
        bytesReceived += Int64(data.count)
        let now = Date()
        guard now.timeIntervalSince(lastProgressUpdate) > 0.25 else { return }
        lastProgressUpdate = now
        let elapsed = now.timeIntervalSince(startTime)
        let speed = elapsed > 0 ? Double(bytesReceived) / elapsed : 0
        onProgress?(existingBytes + bytesReceived, speed)
    }

    func urlSession(_ session: URLSession, task: URLSessionTask, didCompleteWithError error: Error?) {
        stallTimer?.cancel()
        try? fileHandle.close()
        let effectiveError = writeError ?? error
        if stalledOut {
            onComplete?(DownloadStallError())
        } else if effectiveError != nil {
            onComplete?(effectiveError)
        } else if statusCode != 0 && statusCode != 200 && statusCode != 206 {
            onComplete?(URLError(.badServerResponse, userInfo: [
                NSLocalizedDescriptionKey: "HTTP \(statusCode)"
            ]))
        } else {
            onComplete?(nil)
        }
    }
}
