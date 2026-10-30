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
                if let agentSymbol {
                    Image(systemName: agentSymbol)
                        .font(.app(.caption2))
                        .foregroundStyle(Color.accentColor)
                } else if isExternalBridge {
                    Image(systemName: "paperplane.fill")
                        .font(.app(.caption2))
                        .foregroundStyle(isSelected ? Color.white.opacity(0.85) : Color.accentColor)
                        .help("Telegram conversation (view only)")
                } else {
                    Image(systemName: "bubble.left")
                        .font(.app(.subheadline, weight: .medium))
                        .foregroundStyle(.secondary)
                }
            }
            .frame(width: 16)
            .overlay(alignment: .bottomTrailing) {
                if isExternalBridge && agentSymbol != nil {
                    Image(systemName: "paperplane.circle.fill")
                        .font(.app(.caption2))
                        .foregroundStyle(isSelected ? Color.white.opacity(0.85) : Color.accentColor)
                        .scaleEffect(0.7, anchor: .bottomTrailing)
                        .offset(x: 2, y: 3)
                        .help("Telegram conversation (view only)")
                }
            }
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
