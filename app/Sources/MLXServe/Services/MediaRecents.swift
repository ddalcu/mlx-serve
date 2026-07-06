import Foundation

/// Shared recent-generations list logic for the audio history shelves
/// (Voice + Music). Newest-first, deduped, and deliberately UNCAPPED — every
/// generation stays in the list across launches; the files on disk are the
/// source of truth and the only size limit.
enum MediaRecents {

    /// Insert `path` at the front, removing any earlier occurrence.
    static func inserting(_ path: String, into list: [String]) -> [String] {
        var out = list
        out.removeAll { $0 == path }
        out.insert(path, at: 0)
        return out
    }

    /// Rebuild the list from the day-bucketed output layout
    /// (`<root>/<yyyy-MM-dd>/<file><suffix>`), newest first.
    static func scan(root: String, suffix: String) -> [String] {
        let fm = FileManager.default
        guard let days = try? fm.contentsOfDirectory(atPath: root) else { return [] }
        var paths: [(String, Date)] = []
        for day in days {
            let dayDir = (root as NSString).appendingPathComponent(day)
            guard let files = try? fm.contentsOfDirectory(atPath: dayDir) else { continue }
            for f in files where f.hasSuffix(suffix) {
                let full = (dayDir as NSString).appendingPathComponent(f)
                let date = (try? fm.attributesOfItem(atPath: full)[.modificationDate] as? Date) ?? .distantPast
                paths.append((full, date))
            }
        }
        return paths.sorted { $0.1 > $1.1 }.map(\.0)
    }
}
