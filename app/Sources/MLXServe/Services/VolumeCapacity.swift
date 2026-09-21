import Foundation

/// Free bytes on the volume holding a path. Important-usage capacity (counts
/// purgeable space) is APFS-only and reads 0 on exFAT/NTFS/SMB, so fall back to
/// plain capacity. Twin of `kv_disk_cache.volumeSpace` on the engine side.
enum VolumeCapacity {
    static func available(atPath path: String) -> Int64? {
        let v = try? URL(fileURLWithPath: path).resourceValues(forKeys: [
            .volumeAvailableCapacityForImportantUsageKey,
            .volumeAvailableCapacityKey,
            .volumeTotalCapacityKey,
        ])
        return resolve(important: v?.volumeAvailableCapacityForImportantUsage,
                       plain: v?.volumeAvailableCapacity.map(Int64.init),
                       total: v?.volumeTotalCapacity.map(Int64.init))
    }

    static func resolve(important: Int64?, plain: Int64?, total: Int64?) -> Int64? {
        if let important, important > 0, total.map({ important <= $0 }) ?? true { return important }
        return plain
    }
}
