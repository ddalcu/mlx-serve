import Foundation

/// One random UUID per install, made at launch and kept in defaults. It links
/// what one install sends (shared benchmark rows today) without naming the Mac
/// or its owner.
enum InstallId {
    static let key = "installId"

    static func current(_ defaults: UserDefaults = .standard) -> String {
        if let id = defaults.string(forKey: key), UUID(uuidString: id) != nil { return id }
        let id = UUID().uuidString
        defaults.set(id, forKey: key)
        return id
    }
}
