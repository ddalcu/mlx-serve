import Foundation

@MainActor
class DownloadManager: ObservableObject {
    @Published var downloads: [String: DownloadState] = [:]

    struct DownloadState {
        var progress: Double = 0
        var status: Status = .idle
        var statusText: String = ""
        var error: String?

        enum Status: Equatable {
            case idle, downloading, completed, failed
        }
    }

    let modelsDir: String = {
        let path = NSString(string: "~/.mlx-serve/models").expandingTildeInPath
        try? FileManager.default.createDirectory(atPath: path, withIntermediateDirectories: true)
        return path
    }()

    /// A model is ready to use when it has config.json AND at least one .safetensors file
    func isReady(_ repoId: String) -> Bool {
        let name = repoId.split(separator: "/").last.map(String.init) ?? repoId
        let modelDir = (modelsDir as NSString).appendingPathComponent(name)
        let configPath = (modelDir as NSString).appendingPathComponent("config.json")
        guard FileManager.default.fileExists(atPath: configPath) else { return false }

        // Must have at least one safetensors file > 1MB
        guard let entries = try? FileManager.default.contentsOfDirectory(atPath: modelDir) else { return false }
        return entries.contains { entry in
            guard entry.hasSuffix(".safetensors") else { return false }
            let fullPath = (modelDir as NSString).appendingPathComponent(entry)
            let size = (try? FileManager.default.attributesOfItem(atPath: fullPath)[.size] as? UInt64) ?? 0
            return size > 1_000_000
        }
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

            // List files from HuggingFace API
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

            for (idx, (filePath, fileSize)) in neededFiles.enumerated() {
                let destPath = (destDir as NSString).appendingPathComponent(filePath)

                // Skip if already exists with right size
                if let attrs = try? FileManager.default.attributesOfItem(atPath: destPath),
                   let existingSize = attrs[.size] as? Int64,
                   existingSize == fileSize && fileSize > 0 {
                    downloadedSize += fileSize
                    downloads[repoId]?.progress = totalSize > 0 ? Double(downloadedSize) / Double(totalSize) : 0
                    downloads[repoId]?.statusText = "Skipped \(filePath) (exists)"
                    continue
                }

                let sizeStr = fileSize > 1_000_000_000
                    ? String(format: "%.1f GB", Double(fileSize) / 1e9)
                    : fileSize > 1_000_000
                        ? String(format: "%.0f MB", Double(fileSize) / 1e6)
                        : "\(fileSize) B"
                downloads[repoId]?.statusText = "Downloading \(filePath) (\(sizeStr)) [\(idx+1)/\(neededFiles.count)]"

                let fileURL = URL(string: "https://huggingface.co/\(repoId)/resolve/main/\(filePath)")!

                // Use a custom session with longer timeout for large files
                let config = URLSessionConfiguration.default
                config.timeoutIntervalForRequest = 3600
                config.timeoutIntervalForResource = 3600
                let session = URLSession(configuration: config)

                let (tempURL, response) = try await session.download(from: fileURL)

                // Verify download succeeded
                if let httpResponse = response as? HTTPURLResponse, httpResponse.statusCode != 200 {
                    throw URLError(.badServerResponse, userInfo: [
                        NSLocalizedDescriptionKey: "HTTP \(httpResponse.statusCode) for \(filePath)"
                    ])
                }

                try? FileManager.default.removeItem(atPath: destPath)
                try FileManager.default.moveItem(at: tempURL, to: URL(fileURLWithPath: destPath))

                downloadedSize += fileSize
                downloads[repoId]?.progress = totalSize > 0 ? Double(downloadedSize) / Double(totalSize) : 0
            }

            downloads[repoId] = DownloadState(progress: 1.0, status: .completed, statusText: "Complete")
        } catch {
            downloads[repoId] = DownloadState(status: .failed, error: error.localizedDescription)
        }
    }

    /// Discover models that are fully downloaded (have config + safetensors)
    func discoverLocalModels() -> [LocalModel] {
        guard let entries = try? FileManager.default.contentsOfDirectory(atPath: modelsDir) else {
            return []
        }
        return entries.compactMap { name in
            guard !name.hasPrefix(".") else { return nil }
            var dirPath = (modelsDir as NSString).appendingPathComponent(name)

            // Resolve symlinks
            dirPath = (dirPath as NSString).resolvingSymlinksInPath

            let configPath = (dirPath as NSString).appendingPathComponent("config.json")
            guard FileManager.default.fileExists(atPath: configPath) else { return nil }

            // Must have at least one .safetensors file
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

    /// Remove an incomplete download
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
}
