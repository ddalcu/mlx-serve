import SwiftUI

/// The conversation label is independent of selection actions and persistent chat state.
struct SidebarSessionLabel<Status: View>: View {
    let title: String
    let subject: String?
    let agentSymbol: String?
    let isExternalBridge: Bool
    let isSelected: Bool
    @ViewBuilder var status: () -> Status

    var body: some View {
        HStack(alignment: .firstTextBaseline, spacing: 4) {
            HStack(alignment: .firstTextBaseline, spacing: 4) {
                if isExternalBridge {
                    Image(systemName: "paperplane.fill")
                        .font(.app(.caption2))
                        .foregroundStyle(isSelected ? Color.white.opacity(0.85) : Color.accentColor)
                        .help("Telegram conversation (view only)")
                }
                if let agentSymbol {
                    Image(systemName: agentSymbol)
                        .font(.app(.caption2))
                        .foregroundStyle(Color.accentColor)
                } else if !isExternalBridge {
                    Image(systemName: "bubble.left")
                        .font(.app(.subheadline, weight: .medium))
                        .foregroundStyle(.secondary)
                }
            }
            // One slot contains both bridge and agent symbols without shifting the text.
            .frame(width: 32)
            VStack(alignment: .leading, spacing: 2) {
                HStack(spacing: 4) {
                    Text(title)
                        .font(.app(.rowTitle))
                        .lineLimit(1)
                        .foregroundStyle(.primary)
                    status()
                }
                if let subject {
                    Text(subject)
                        .font(.app(.caption2))
                        .lineLimit(1)
                        .truncationMode(.tail)
                        .foregroundStyle(.secondary)
                }
            }
        }
    }
}
