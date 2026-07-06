import Foundation

/// Output roots for natively-generated media. All three modalities (image,
/// audio, video) are produced by the embedded `mlx-serve` engine — there is no
/// Python venv anymore; this is just where the results are written.
enum MediaStorage {
    static let imagesRoot: String = make("images")
    static let videosRoot: String = make("videos")
    static let audiosRoot: String = make("audio")
    static let musicRoot: String = make("music")
    static let models3dRoot: String = make("models3d")

    private static func make(_ name: String) -> String {
        let dir = NSString(string: "~/.mlx-serve/generations/\(name)").expandingTildeInPath
        try? FileManager.default.createDirectory(atPath: dir, withIntermediateDirectories: true)
        return dir
    }
}
