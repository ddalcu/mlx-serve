import Foundation

/// What of a pack sits in memory once loaded: DeepSeek-V4.1 reads its Engram tables from disk
/// (`model.dsv41DropsKey`), so they come off its size. Read from the shards' headers, cached.
enum ResidentSize {
    nonisolated(unsafe) private static var cache: [String: Int64] = [:]
    private static let lock = NSLock()

    static func bytes(of m: LocalModel) -> Int64 {
        guard m.modelType == "deepseek_v41" else { return m.sizeBytes }
        lock.lock()
        defer { lock.unlock() }
        if let hit = cache[m.path] { return hit }
        let off = engramBytes(inDir: m.path)
        let resident = max(m.sizeBytes - off, 0)
        cache[m.path] = resident
        return resident
    }

    /// The bytes of every `.engram.embed.` tensor across the pack's `.safetensors` shards.
    static func engramBytes(inDir dir: String) -> Int64 {
        let files = (try? FileManager.default.contentsOfDirectory(atPath: dir))?.filter { $0.hasSuffix(".safetensors") } ?? []
        return files.reduce(0) { $0 + engramBytes(inShard: (dir as NSString).appendingPathComponent($1)) }
    }

    static func engramBytes(inShard path: String) -> Int64 {
        guard let h = FileHandle(forReadingAtPath: path) else { return 0 }
        defer { try? h.close() }
        guard let lenData = try? h.read(upToCount: 8), lenData.count == 8 else { return 0 }
        let len = lenData.withUnsafeBytes { $0.loadUnaligned(as: UInt64.self) }
        guard len < 256 << 20, let json = try? h.read(upToCount: Int(len)),
              let header = (try? JSONSerialization.jsonObject(with: json)) as? [String: Any] else { return 0 }
        var total: Int64 = 0
        for (name, v) in header where name.contains(".engram.embed.") {
            if let offs = (v as? [String: Any])?["data_offsets"] as? [NSNumber], offs.count == 2 {
                total += offs[1].int64Value - offs[0].int64Value
            }
        }
        return total
    }
}
