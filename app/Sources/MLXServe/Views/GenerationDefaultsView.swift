import SwiftUI

struct GenerationDefaultsRows: View {
    @Binding var profile: GenerationDefaults
    var inheritance = "Inherit"
    var fields = GenerationField.allCases
    var allowsForce = true
    var inherited = GenerationDefaults()

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                VStack(alignment: .leading, spacing: 4) {
                    Text("All parameters").font(.app(.body))
                    Text(L10n.format("Off: %@ · On: custom value", L10n.text(inheritance))
                         + (allowsForce ? L10n.text(" · Force: ignore client override") : ""))
                        .font(.app(.caption2)).foregroundStyle(.secondary)
                }
                .frame(maxWidth: .infinity, alignment: .leading)
                GenerationModeSwitch(value: profile.mode(fields: fields), allowsForce: allowsForce,
                                     inheritance: inheritance, title: "All parameters") {
                    profile.setMode($0, fields: fields, inherited: inherited)
                }
            }
            Divider()
            ForEach(fields) { field in
                SearchableRow(searchText: [field.title, field.parameterName, field.help, inheritance, "Force", "Ignore client override"]) {
                    GenerationParameterRow(field: field, mode: profile.mode(field),
                        value: value(field), hasInheritedValue: inherited.rules[field.rawValue] != nil,
                        inheritance: inheritance, allowsForce: allowsForce,
                        setMode: { profile.setMode($0, field: field, inherited: inherited.rules[field.rawValue]?.value) },
                        setValue: { set($0, field) })
                }
            }
        }
    }

    private func value(_ field: GenerationField) -> GenerationDefaults.Value {
        profile.rules[field.rawValue]?.value ?? inherited.rules[field.rawValue]?.value ?? field.defaultValue
    }

    private func set(_ value: GenerationDefaults.Value, _ field: GenerationField) {
        let lock = profile.rules[field.rawValue]?.ignoreClient ?? false
        profile.rules[field.rawValue] = .init(value: value, ignoreClient: lock)
    }
}

struct GenerationParameterRow: View {
    let field: GenerationField
    let mode: GenerationDefaults.Mode
    let value: GenerationDefaults.Value
    let hasInheritedValue: Bool
    let inheritance: String
    let allowsForce: Bool
    let setMode: (GenerationDefaults.Mode) -> Void
    let setValue: (GenerationDefaults.Value) -> Void

    var body: some View {
        HStack(alignment: .top, spacing: 18) {
            VStack(alignment: .leading, spacing: 5) {
                (Text(L10n.text(field.title)).font(.app(.body))
                 + Text(" · ").font(.app(.caption)).foregroundColor(.secondary)
                 + Text(verbatim: field.parameterName).font(.app(.caption).monospaced()).foregroundColor(.secondary))
                Text(verbatim: field.help)
                    .font(.app(.caption2)).foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
            }
            .frame(maxWidth: .infinity, alignment: .leading)
            VStack(alignment: .trailing, spacing: 5) {
                if mode == .inherited && !hasInheritedValue {
                    Text(L10n.text(inheritance)).font(.app(.caption)).foregroundStyle(.secondary)
                        .frame(height: 22)
                } else {
                    valueControl
                }
                if field.showsSlider {
                    // A Form reserves a label column for an unlabeled Slider; hide it so the
                    // track spans the column its end labels sit under.
                    slider.labelsHidden().accessibilityLabel(L10n.text(field.title))
                    if let guidance = field.guidance {
                        HStack {
                            Text(L10n.text(guidance.low))
                            Spacer()
                            Text(L10n.text(guidance.high))
                        }
                        .font(.app(.caption2)).foregroundStyle(.secondary)
                    }
                }
            }
            .frame(width: 148)
            .disabled(mode == .inherited)
            GenerationModeSwitch(value: mode, allowsForce: allowsForce,
                                 inheritance: inheritance, title: field.title, onChange: setMode)
        }
        .padding(.vertical, 7)
    }

    @ViewBuilder
    private var slider: some View {
        let binding = Binding(get: { field.sliderPosition(number) },
                              set: { setValue(.number(field.sliderNumber($0))) })
        if let presets = field.presets {
            Slider(value: binding, in: 0...Double(presets.count - 1), step: 1)
        } else {
            Slider(value: binding, in: field.sliderRange)
        }
    }

    private var number: Double {
        if case .number(let n) = value { return n }
        return 0
    }

    @ViewBuilder
    private var valueControl: some View {
        switch field {
        case .thinking:
            Picker("", selection: Binding(
                get: { if case .boolean(let v) = value { return v }; return true },
                set: { setValue(.boolean($0)) })) {
                Text("On").font(.app(.body)).tag(true)
                Text("Off").font(.app(.body)).tag(false)
            }
            .labelsHidden().fixedSize()
        case .effort:
            Picker("", selection: Binding(
                get: { if case .text(let v) = value { return v }; return "low" },
                set: { setValue(.text($0)) })) {
                ForEach(["none", "minimal", "low", "medium", "high", "xhigh", "max"], id: \.self) { word in
                    Text(verbatim: word).font(.app(.body)).tag(word)
                }
            }
            .labelsHidden().fixedSize()
        default:
            TextField("", value: Binding(get: { number },
                set: { if let value = field.numberValue($0) { setValue(value) } }),
                format: .number.grouping(.never).precision(.fractionLength(0...2)))
                .textFieldStyle(.roundedBorder).multilineTextAlignment(.trailing)
                // Fixed: a width that follows the value reflows the row while the slider drags.
                .frame(width: 76)
                .font(.app(.body).monospacedDigit())
                .accessibilityLabel(L10n.text(field.title))
        }
    }
}

struct GenerationModeSwitch: View {
    let value: GenerationDefaults.Mode?
    let allowsForce: Bool
    let inheritance: String
    let title: String
    let onChange: (GenerationDefaults.Mode) -> Void
    @Environment(\.accessibilityReduceMotion) private var reduceMotion

    private var stateLabel: String {
        switch value {
        case .inherited: "Off"
        case .enabled: "On"
        case .forced: "Force"
        case nil: "Mixed"
        }
    }

    private var ballColor: Color {
        switch value {
        case .inherited, nil: .gray
        case .enabled: .accentColor
        case .forced: .orange
        }
    }

    private var modes: [GenerationDefaults.Mode] {
        allowsForce ? [.inherited, .enabled, .forced] : [.inherited, .enabled]
    }

    var body: some View {
        VStack(spacing: 3) {
            HStack(spacing: 0) {
                ForEach(modes, id: \.rawValue) { mode in
                    Button { onChange(mode) } label: {
                        Circle()
                            .fill(Color.secondary.opacity(value == mode ? 0 : 0.35))
                            .frame(width: 6, height: 6)
                            .frame(width: 20, height: 28)
                            .contentShape(Rectangle())
                    }
                    .buttonStyle(.plain)
                    .help(help(mode))
                    .accessibilityLabel(L10n.text(title) + ": " + help(mode))
                    .accessibilityValue(value == mode ? L10n.text("Selected") : "")
                }
            }
            .background {
                Capsule().fill(Color.secondary.opacity(0.2)).frame(height: 2).padding(.horizontal, 10)
            }
            .overlay(alignment: .leading) {
                Circle().fill(ballColor).frame(width: 14, height: 14)
                    .offset(x: 3 + CGFloat(value?.rawValue ?? 0) * 20)
                    .opacity(value == nil ? 0 : 1)
                    .animation(reduceMotion ? nil : .easeInOut(duration: 0.16), value: value)
                    .allowsHitTesting(false)
                    .accessibilityHidden(true)
            }
            .padding(.horizontal, 3)
            .background(Color.secondary.opacity(0.1), in: Capsule())
            .overlay(Capsule().strokeBorder(Color.secondary.opacity(0.2), lineWidth: 1))
            Text(L10n.text(stateLabel)).font(.app(.caption2))
                .foregroundStyle(value == .forced ? Color.orange : Color.secondary)
                .frame(height: 14)
                .frame(maxWidth: .infinity)
                .accessibilityHidden(true)
        }
        .frame(width: CGFloat(modes.count * 20 + 6))
        .accessibilityElement(children: .contain)
        .accessibilityLabel(L10n.text(title))
        .accessibilityValue(L10n.text(stateLabel))
    }

    private func help(_ mode: GenerationDefaults.Mode) -> String {
        switch mode {
        case .inherited: L10n.format("Off: %@", L10n.text(inheritance))
        case .enabled: L10n.text("On: use custom value")
        case .forced: L10n.text("Force: ignore client override")
        }
    }
}

struct GlobalGenerationDefaultsView: View {
    @EnvironmentObject var appState: AppState
    @State private var profile = GenerationDefaults()
    @State private var loaded = false
    @State private var error: String?

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Defaults for app chats and external API clients. Model rules override global rules. Applies to the next request.")
                .font(.app(.caption)).foregroundStyle(.secondary)
            GenerationDefaultsRows(profile: $profile, inheritance: "Model default")
                .disabled(!loaded)
            if let error {
                Text(verbatim: error).font(.app(.caption)).foregroundStyle(.red)
                Button("Reload settings") { load() }.font(.app(.body))
            }
        }
        .onAppear(perform: load)
        .onReceive(NotificationCenter.default.publisher(for: GenerationDefaultsFile.changed)) { _ in
            guard let saved = try? GenerationDefaultsFile.load(), saved != profile else { return }
            profile = saved
        }
        .onChange(of: profile) { _, value in
            guard loaded else { return }
            do { try GenerationDefaultsFile.save(value); error = nil }
            catch { self.error = "Could not save generation defaults: \(error.localizedDescription)" }
        }
        .disabled(!loaded && error == nil)
    }

    private func load() {
        loaded = false
        do {
            try GenerationDefaultsFile.migrate(appState.serverOptions)
            profile = try GenerationDefaultsFile.load()
            error = nil
            loaded = true
        } catch { self.error = "Could not read generation defaults: \(error.localizedDescription)" }
    }
}
