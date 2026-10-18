import AppKit
import UniformTypeIdentifiers

/// What kind of machine a cluster member is, from its `/health` ("os", "model"): the icon
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

    init(health: Data) {
        let json = (try? JSONSerialization.jsonObject(with: health)) as? [String: Any]
        self.init(os: json?["os"] as? String ?? "", model: json?["model"] as? String ?? "")
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

    private static func sysctlString(_ name: String) -> String? {
        var len = 0
        guard sysctlbyname(name, nil, &len, nil, 0) == 0, len > 0 else { return nil }
        var buf = [CChar](repeating: 0, count: len)
        guard sysctlbyname(name, &buf, &len, nil, 0) == 0 else { return nil }
        return String(cString: buf)
    }
}
