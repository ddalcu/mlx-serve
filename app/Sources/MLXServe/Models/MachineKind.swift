import AppKit
import UniformTypeIdentifiers

/// What kind of machine a cluster member is, from its `/v1/models` "machine": the icon
/// macOS itself draws for it, and a word for it.
struct MachineKind: Equatable {
    let os: String
    /// The device model id ("Mac16,5", "iPhone17,1"); empty when it has none.
    let model: String

    static let unknown = MachineKind(os: "", model: "")
    static let mac = MachineKind(os: "macos", model: "")
    static let thisMac = MachineKind(os: "macos", model: sysctlString("hw.model") ?? "")

    init(os: String, model: String) {
        self.os = os
        self.model = model
    }

    var label: String {
        if model.hasPrefix("iPhone") { return "iPhone" }
        if model.hasPrefix("iPad") { return "iPad" }
        switch os {
        case "macos": return "Mac"
        case "ios": return "iPhone"
        case "linux": return L10n.text("Linux PC")
        case "windows": return L10n.text("Windows PC")
        default: return L10n.text("Computer")
        }
    }

    /// The exact model's icon when macOS knows the model, else its family's.
    var icon: NSImage {
        let byModel = model.isEmpty ? nil
            : UTType(tag: model, tagClass: UTTagClass(rawValue: "com.apple.device-model-code"), conformingTo: nil)
        let family: String = switch label {
        case "iPhone": "com.apple.iphone"
        case "iPad": "com.apple.ipad"
        case "Mac": "com.apple.mac"
        default: "public.generic-pc"
        }
        return NSWorkspace.shared.icon(for: byModel ?? UTType(family) ?? .item)
    }

    fileprivate static func sysctlString(_ name: String) -> String? {
        var len = 0
        guard sysctlbyname(name, nil, &len, nil, 0) == 0, len > 0 else { return nil }
        var buf = [CChar](repeating: 0, count: len)
        guard sysctlbyname(name, &buf, &len, nil, 0) == 0 else { return nil }
        return String(cString: buf)
    }
}

/// A LAN peer as this Mac's server reports it in `/v1/models` "lan_peers".
struct LanPeer: Equatable {
    let kind: MachineKind
    /// The model of a Thunderbolt pair it leads; empty when it leads none.
    let leads: String
    let joined: Bool
    var url: URL? = nil
    /// Its free memory, as its server reports it (0 when it does not).
    var freeBytes: Int64 = 0

    /// It leads a pair and waits for a Mac to join it.
    var waitsForPartner: Bool { !leads.isEmpty && !joined }
}
