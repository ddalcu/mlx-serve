import SwiftUI

/// The note waiting for the agent's next step, above the composer. Styled
/// like the composer's attachment chips: the icon, the text, and the ways
/// out (edit, drop, or send now once the chat is idle).
struct SteeringNoteRow: View {
    let note: String
    /// The chat is idle, so the note can only leave as a turn of its own.
    let canSendNow: Bool
    let onEdit: () -> Void
    let onRemove: () -> Void
    let onSendNow: () -> Void

    var body: some View {
        HStack(spacing: 8) {
            Image(systemName: "arrow.turn.down.right")
                .font(.system(size: 16, weight: .semibold))
                .foregroundStyle(.white)
                .frame(width: 32, height: 32)
                .background(Color.accentColor.opacity(0.85))
                .clipShape(RoundedRectangle(cornerRadius: 6))
            VStack(alignment: .leading, spacing: 1) {
                Text(canSendNow ? "Note not sent yet" : "Sent at the agent's next step")
                    .font(.caption2)
                    .foregroundStyle(.secondary)
                Text(note)
                    .font(.caption.weight(.medium))
                    .lineLimit(2)
                    .truncationMode(.tail)
            }
            Spacer(minLength: 4)
            if canSendNow {
                Button("Send", action: onSendNow)
                    .controlSize(.small)
                    .help("Send the note as the next message")
            }
            Button(action: onEdit) {
                Image(systemName: "pencil.circle.fill")
                    .font(.system(size: 14))
                    .foregroundStyle(.secondary)
            }
            .buttonStyle(.plain)
            .help("Edit the note in the composer; nothing is sent while you edit")
            Button(action: onRemove) {
                Image(systemName: "xmark.circle.fill")
                    .font(.system(size: 14))
                    .foregroundStyle(.secondary)
            }
            .buttonStyle(.plain)
            .help("Drop the note")
        }
        .padding(.horizontal, 8)
        .padding(.vertical, 4)
        .frame(maxWidth: 480, minHeight: 44, alignment: .leading)
        .background(Color.secondary.opacity(0.15))
        .clipShape(RoundedRectangle(cornerRadius: 10))
        .frame(maxWidth: .infinity, alignment: .leading)
    }
}
