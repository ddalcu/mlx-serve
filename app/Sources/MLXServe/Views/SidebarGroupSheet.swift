import SwiftUI

/// The small name prompt behind "New Group…" and "Rename Group…".
struct SidebarGroupSheet: View {
    let title: LocalizedStringKey
    let action: LocalizedStringKey
    @State var name: String
    let onSubmit: (String) -> Void
    @Environment(\.dismiss) private var dismiss

    private var isBlank: Bool { name.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty }

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text(title).font(.app(.headline))
            TextField("Group Name", text: $name)
                .textFieldStyle(.roundedBorder)
                .onSubmit(submit)
            HStack {
                Spacer()
                Button("Cancel", role: .cancel) { dismiss() }
                    .keyboardShortcut(.cancelAction)
                Button(action, action: submit)
                    .keyboardShortcut(.defaultAction)
                    .disabled(isBlank)
            }
        }
        .padding(20)
        .frame(width: 300)
    }

    private func submit() {
        guard !isBlank else { return }
        onSubmit(name)
        dismiss()
    }
}
