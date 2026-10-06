import Foundation

/// Output roots for natively-generated media. All three modalities (image,
/// audio, video) are produced by the embedded `mlx-serve` engine — there is no
/// Python venv anymore; this is just where the results are written.
enum MediaStorage {
    static let imagesRoot: String = make("images")
    static let videosRoot: String = make("videos")
    static let audiosRoot: String = make("audio")
    static let musicRoot: String = make("music")
    static let soundRoot: String = make("sound")
    static let models3dRoot: String = make("models3d")

    /// `<root>/<yyyy-MM-dd>/<yyyy-MM-dd_HH-mm-ss>_<slug>.<ext>`, the day dir
    /// created. The slug is the prompt's first 40 lowercase alphanumerics.
    static func datedPath(root: String, prompt: String, ext: String, now: Date = Date()) -> String {
        let df = DateFormatter()
        df.dateFormat = "yyyy-MM-dd"
        let dayDir = (root as NSString).appendingPathComponent(df.string(from: now))
        try? FileManager.default.createDirectory(atPath: dayDir, withIntermediateDirectories: true)
        df.dateFormat = "yyyy-MM-dd_HH-mm-ss"
        let slug = prompt
            .lowercased()
            .replacingOccurrences(of: #"[^a-z0-9]+"#, with: "-", options: .regularExpression)
            .trimmingCharacters(in: CharacterSet(charactersIn: "-"))
            .prefix(40)
        return (dayDir as NSString).appendingPathComponent("\(df.string(from: now))_\(slug).\(ext)")
    }

    private static func make(_ name: String) -> String {
        let dir = NSString(string: "~/.mlx-serve/generations/\(name)").expandingTildeInPath
        try? FileManager.default.createDirectory(atPath: dir, withIntermediateDirectories: true)
        return dir
    }
}
