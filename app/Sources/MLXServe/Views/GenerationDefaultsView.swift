import SwiftUI

struct GenerationDefaultsRows: View {
    @Binding var profile: GenerationDefaults
    var inheritance = "Inherit"
    var fields = GenerationField.allCases
    var showsClientLocks = true

    var body: some View {
        HStack {
            MixedCheckbox(title: L10n.format("All: %@", L10n.text(inheritance)), value: profile.inheritanceState(fields: fields)) {
                profile.setAllInherited($0, fields: fields)
            }
            Spacer()
            if showsClientLocks {
                MixedCheckbox(title: L10n.format("All: %@", L10n.text("Ignore client override")), value: profile.clientLockState(fields: fields)) {
                    profile.setAllClientLocks($0, fields: fields)
                }
                .disabled(fields.allSatisfy { profile.rules[$0.rawValue] == nil })
            }
        }
        Divider()
        ForEach(fields) { field in
            SearchableRow(searchText: [field.title, showsClientLocks ? field.help : field.clientHelp, inheritance] + (showsClientLocks ? ["Ignore client override"] : [])) {
                VStack(alignment: .leading, spacing: 6) {
                    HStack(alignment: .firstTextBaseline, spacing: 10) {
                        Text(L10n.text(field.title)).font(.app(.body))
                        Spacer(minLength: 8)
                        Toggle(L10n.text(inheritance), isOn: inheritanceBinding(field))
                            .toggleStyle(.checkbox).font(.app(.caption))
                        if field.showsSlider {
                            Slider(value: Binding(
                                get: { if case .number(let n) = value(field) { return n }; return 0 },
                                set: { set(.number($0), field) }), in: field.range)
                                .frame(maxWidth: 150)
                                .disabled(profile.rules[field.rawValue] == nil)
                        }
                        valueControl(field)
                            .disabled(profile.rules[field.rawValue] == nil)
                    }
                    HStack {
                        Text(showsClientLocks ? L10n.text(field.help) : L10n.text(field.clientHelp)).font(.app(.caption2)).foregroundStyle(.secondary)
                        Spacer(minLength: 8)
                        if showsClientLocks {
                            Toggle("Ignore client override", isOn: lockBinding(field))
                                .toggleStyle(.checkbox).font(.app(.caption2))
                                .disabled(profile.rules[field.rawValue] == nil)
                        }
                    }
                }
                .padding(.vertical, 5)
            }
        }
    }

    private func inheritanceBinding(_ field: GenerationField) -> Binding<Bool> {
        Binding(get: { profile.rules[field.rawValue] == nil },
                set: { profile.setInherited($0, field: field) })
    }

    private func lockBinding(_ field: GenerationField) -> Binding<Bool> {
        Binding(get: { profile.rules[field.rawValue]?.ignoreClient ?? false },
                set: { profile.rules[field.rawValue]?.ignoreClient = $0 })
    }

    @ViewBuilder
    private func valueControl(_ field: GenerationField) -> some View {
        switch field {
        case .thinking:
            Picker("", selection: Binding(
                get: { if case .boolean(let v) = value(field) { return v }; return true },
                set: { set(.boolean($0), field) })) {
                    Text("On").tag(true)
                    Text("Off").tag(false)
                }
                .labelsHidden().frame(width: 100).font(.app(.body))
        case .effort:
            Picker("", selection: Binding(
                get: { if case .text(let v) = value(field) { return v }; return "low" },
                set: { set(.text($0), field) })) {
                    ForEach(["none", "minimal", "low", "medium", "high", "xhigh", "max"], id: \.self) { word in
                        Text(verbatim: word).tag(word)
                    }
                }
                .labelsHidden().frame(width: 110).font(.app(.body))
        default:
            TextField("", value: Binding(
                get: { if case .number(let v) = value(field) { return v }; return 0 },
                set: { n in
                    guard n.isFinite, field.range.contains(n) else { return }
                    let integer = [.topK, .maxTokens, .budget].contains(field)
                    set(.number(integer ? n.rounded() : n), field)
                }), format: .number)
                .textFieldStyle(.roundedBorder).multilineTextAlignment(.trailing)
                .frame(width: 95).font(.app(.body).monospacedDigit())
        }
    }

    private func value(_ field: GenerationField) -> GenerationDefaults.Value {
        profile.rules[field.rawValue]?.value ?? field.defaultValue
    }
    private func set(_ value: GenerationDefaults.Value, _ field: GenerationField) {
        let lock = profile.rules[field.rawValue]?.ignoreClient ?? false
        profile.rules[field.rawValue] = .init(value: value, ignoreClient: lock)
    }
}

struct MixedCheckbox: NSViewRepresentable {
    let title: String
    let value: Bool?
    var onChange: (Bool) -> Void
    @Environment(\.isEnabled) private var enabled

    func makeNSView(context: Context) -> MixedCheckboxButton {
        MixedCheckboxButton(title: title, value: value, onChange: onChange)
    }

    func updateNSView(_ button: MixedCheckboxButton, context: Context) {
        button.title = title
        button.isEnabled = enabled
        button.onChange = onChange
        button.setValue(value)
    }

    func sizeThatFits(_ proposal: ProposedViewSize, nsView: MixedCheckboxButton, context: Context) -> CGSize? {
        nsView.intrinsicContentSize
    }
}

final class MixedCheckboxButton: NSButton {
    var onChange: (Bool) -> Void
    private var value: Bool?

    init(title: String, value: Bool?, onChange: @escaping (Bool) -> Void) {
        self.value = value
        self.onChange = onChange
        super.init(frame: .zero)
        self.title = title
        setButtonType(.switch)
        allowsMixedState = true
        font = AppType.system(.caption)
        target = self
        action = #selector(toggleAll)
        setValue(value)
    }

    required init?(coder: NSCoder) { fatalError("not used") }

    func setValue(_ value: Bool?) {
        self.value = value
        state = value.map { $0 ? .on : .off } ?? .mixed
    }

    @objc private func toggleAll() {
        let selected = value != true
        setValue(selected)
        onChange(selected)
    }
}

struct GlobalGenerationDefaultsView: View {
    @EnvironmentObject var appState: AppState
    @State private var profile = GenerationDefaults()
    @State private var loaded = false
    @State private var error: String?

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Defaults for app chats and external API clients. Explicit client values win unless locked. Model rules override global rules. Applies to the next request.")
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
