import Foundation

/// Outside plugins: the dylibs `mlx-serve plugin install` put in ~/.mlx-serve/plugins/, each beside its install
/// receipt `<name>.json`, and the Settings switches the server reads once at startup from ~/.mlx-serve/plugins.json
/// (`{"enabled": Bool, "disabled": [name], "options": {name: {variable: value}}}`, src/host/plugins/loader.zig). Off
/// unless the user turns it on.
enum OutsidePlugins {
    /// A setting a plugin reads from its environment (`abi/dylib.zig` `Option`): the server sets the saved value before
    /// it loads the plugin, over one the shell set.
    struct Option: Decodable, Equatable, Identifiable {
        enum Kind: String, Decodable { case bool, number, text }
        var name: String
        var kind: Kind
        /// The plugin's own value when none is saved; "" when it names none.
        var defaultValue: String
        var help: String
        var id: String { name }

        init(name: String, kind: Kind, defaultValue: String = "", help: String = "") {
            self.name = name; self.kind = kind; self.defaultValue = defaultValue; self.help = help
        }

        init(from decoder: Decoder) throws {
            let c = try decoder.container(keyedBy: CodingKeys.self)
            self.init(name: try c.decode(String.self, forKey: .name), kind: try c.decode(Kind.self, forKey: .kind),
                      defaultValue: try c.decodeIfPresent(String.self, forKey: .defaultValue) ?? "",
                      help: try c.decodeIfPresent(String.self, forKey: .help) ?? "")
        }

        private enum CodingKeys: String, CodingKey {
            case name, kind, help
            case defaultValue = "default"
        }

        /// Whether the server sets `value` (src/abi/dylib.zig `valueOk`): `0` or `1`, a plain decimal number, or text
        /// up to 1 KiB.
        func accepts(_ value: String) -> Bool {
            switch kind {
            case .bool: return value == "0" || value == "1"
            case .number: return value.range(of: #"\A-?[0-9]+(\.[0-9]+)?([eE][-+]?[0-9]+)?\z"#, options: .regularExpression) != nil
            case .text: return value.utf8.count <= 1024 && !value.contains("\0")
            }
        }
    }

    /// What `plugin install` recorded for one plugin.
    struct Receipt: Decodable, Equatable, Identifiable {
        var name: String
        var kind: String
        var version: String
        var sha256: String
        var signature: String
        var source: String
        /// The `model_type`s it declares: packs of these types are the plugin's.
        var modelTypes: [String]
        var options: [Option]
        var id: String { name }

        init(name: String, kind: String, version: String = "", sha256: String, signature: String = "", source: String = "",
             modelTypes: [String] = [], options: [Option] = []) {
            self.name = name; self.kind = kind; self.version = version
            self.sha256 = sha256; self.signature = signature; self.source = source; self.modelTypes = modelTypes
            self.options = options
        }

        init(from decoder: Decoder) throws {
            let c = try decoder.container(keyedBy: CodingKeys.self)
            self.init(name: try c.decode(String.self, forKey: .name),
                      kind: try c.decode(String.self, forKey: .kind),
                      version: try c.decodeIfPresent(String.self, forKey: .version) ?? "",
                      sha256: try c.decode(String.self, forKey: .sha256),
                      signature: try c.decodeIfPresent(String.self, forKey: .signature) ?? "",
                      source: try c.decodeIfPresent(String.self, forKey: .source) ?? "",
                      modelTypes: try c.decodeIfPresent([String].self, forKey: .modelTypes) ?? [],
                      options: (try? c.decodeIfPresent([Option].self, forKey: .options)) ?? [])
        }

        private enum CodingKeys: String, CodingKey {
            case name, kind, version, sha256, signature, source, options
            case modelTypes = "model_types"
        }
    }

    /// The model types the switched-on plugins serve, split by what they serve: a model plugin's packs are chat
    /// models, a media plugin's are not.
    struct Served: Equatable {
        var chat: Set<String> = []
        var media: Set<String> = []

        init(receipts: [Receipt], switches: Switches) {
            guard switches.enabled else { return }
            for r in receipts where switches.isOn(r.name) {
                switch r.kind {
                case "model": chat.formUnion(r.modelTypes)
                case "media": media.formUnion(r.modelTypes)
                default: break
                }
            }
        }
    }

    /// What this Mac's plugins serve, read once: the server loads plugins only at its start. None on the Mac App
    /// Store build, which loads no plugins.
    static let served = Served(receipts: BuildFeatures.current.isMAS ? [] : receipts(), switches: loadSwitches())

    struct Switches: Equatable {
        var enabled = false
        var disabled: [String] = []
        /// Each plugin's saved option values, by variable.
        var options: [String: [String: String]] = [:]

        func isOn(_ name: String) -> Bool { !disabled.contains(name) }

        /// `value` nil goes back to the plugin's own default.
        mutating func setOption(_ plugin: String, _ name: String, _ value: String?) {
            var values = options[plugin] ?? [:]
            values[name] = value
            options[plugin] = values.isEmpty ? nil : values
        }

        mutating func set(_ name: String, on: Bool) {
            disabled.removeAll { $0 == name }
            if !on { disabled.append(name) }
        }

        /// Switching a plugin on means it runs: loading comes on with it. Loading coming on must not also start the
        /// installed plugins the user never chose, so those are switched off first.
        func turningOn(_ names: [String], installed: [String]) -> Switches {
            var s = self
            if !s.enabled {
                for other in installed where !names.contains(other) && !s.disabled.contains(other) { s.disabled.append(other) }
            }
            s.enabled = true
            s.disabled.removeAll { names.contains($0) }
            return s
        }
    }

    static let defaultDir = NSString(string: "~/.mlx-serve/plugins").expandingTildeInPath
    static let defaultSettingsPath = NSString(string: "~/.mlx-serve/plugins.json").expandingTildeInPath

    /// Every installed plugin (a receipt beside its dylib), by name.
    static func receipts(inDir dir: String = defaultDir) -> [Receipt] {
        let files = (try? FileManager.default.contentsOfDirectory(atPath: dir)) ?? []
        return files.filter { $0.hasSuffix(".json") }.compactMap { file -> Receipt? in
            let stem = String(file.dropLast(".json".count))
            let base = dir as NSString
            guard FileManager.default.fileExists(atPath: base.appendingPathComponent(stem + ".dylib")),
                  let data = FileManager.default.contents(atPath: base.appendingPathComponent(file)) else { return nil }
            return try? JSONDecoder().decode(Receipt.self, from: data)
        }.sorted { $0.name < $1.name }
    }

    /// A missing or unreadable file reads as off, as the server reads it.
    static func loadSwitches(path: String = defaultSettingsPath) -> Switches {
        guard let data = FileManager.default.contents(atPath: path),
              let object = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else { return Switches() }
        return Switches(enabled: object["enabled"] as? Bool ?? false, disabled: object["disabled"] as? [String] ?? [],
                        options: optionValues(object["options"]))
    }

    /// Each plugin's string values; anything else is skipped, as the server skips it.
    private static func optionValues(_ raw: Any?) -> [String: [String: String]] {
        var out: [String: [String: String]] = [:]
        for (plugin, values) in raw as? [String: Any] ?? [:] {
            let strings = (values as? [String: Any] ?? [:]).compactMapValues { $0 as? String }
            if !strings.isEmpty { out[plugin] = strings }
        }
        return out
    }

    /// The values the server sets for `receipt`'s plugin (src/host/plugins/loader.zig `applyOptions`): saved, declared,
    /// and accepted by their kind.
    static func applied(_ receipt: Receipt, _ switches: Switches) -> [String: String] {
        let saved = switches.options[receipt.name] ?? [:]
        var out: [String: String] = [:]
        for o in receipt.options { if let v = saved[o.name], o.accepts(v) { out[o.name] = v } }
        return out
    }

    /// Writes the switches, keeping any other key the file holds.
    static func save(_ switches: Switches, path: String = defaultSettingsPath) throws {
        var object: [String: Any] = [:]
        if let data = FileManager.default.contents(atPath: path),
           let existing = try? JSONSerialization.jsonObject(with: data) as? [String: Any] { object = existing }
        object["enabled"] = switches.enabled
        object["disabled"] = switches.disabled
        object["options"] = switches.options.isEmpty ? nil : switches.options
        try FileManager.default.createDirectory(atPath: (path as NSString).deletingLastPathComponent, withIntermediateDirectories: true)
        let data = try JSONSerialization.data(withJSONObject: object, options: [.prettyPrinted, .sortedKeys])
        try data.write(to: URL(fileURLWithPath: path), options: .atomic)
    }

    /// The row's second line: kind and version, who signed the file, and the start of its hash.
    static func summary(_ r: Receipt) -> String {
        L10n.format("%@ %@ · signed: %@ · sha256 %@",
                    r.kind, r.version.isEmpty ? "–" : r.version,
                    r.signature.isEmpty ? "unknown" : r.signature, String(r.sha256.prefix(12)))
    }
}
