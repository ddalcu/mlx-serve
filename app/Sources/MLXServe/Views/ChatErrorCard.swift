import SwiftUI

/// The transcript's failure row.
///
/// Replaces `[Error: …]` glued onto the assistant's own text, which read as
/// something the model had said and gave the user nothing to do about it. The
/// card states what happened in the app's own voice and — for the one failure
/// that is fixable from here — offers the setting that fixes it.
struct ChatErrorCard: View {
    let notice: ChatErrorNotice
    /// Opens Settings at the context-size control. Passed in rather than reached
    /// for so the card stays renderable outside the chat window.
    let onIncreaseContext: () -> Void

    var body: some View {
        HStack(alignment: .top, spacing: 10) {
            Image(systemName: "exclamationmark.circle")
                .font(.system(size: 15, weight: .medium))
                .foregroundStyle(.red)
                .padding(.top, 1)

            VStack(alignment: .leading, spacing: 6) {
                Text(notice.headline)
                    .font(.callout.weight(.semibold))
                    .foregroundStyle(.red)
                Text(notice.detail)
                    .font(.callout)
                    .foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
                    .textSelection(.enabled)

                if notice.offersContextAction {
                    Button(action: onIncreaseContext) {
                        HStack(spacing: 6) {
                            Image(systemName: "arrow.up.forward.square")
                                .font(.system(size: 11, weight: .medium))
                            Text("Increase Context Size")
                                .font(.callout.weight(.medium))
                        }
                        .padding(.horizontal, 12)
                        .padding(.vertical, 7)
                        .contentShape(Capsule())
                    }
                    .buttonStyle(.plain)
                    .background(Color.primary.opacity(0.08))
                    .clipShape(Capsule())
                    .overlay(Capsule().stroke(Color.primary.opacity(0.15), lineWidth: 1))
                    .padding(.top, 2)
                    .help("Open Settings → Context size. A larger window needs more memory and takes effect when the server restarts.")
                }
            }
            Spacer(minLength: 0)
        }
        .padding(14)
        .background(Color.red.opacity(0.10))
        .clipShape(RoundedRectangle(cornerRadius: ChatMetrics.bubbleCornerRadius))
        .overlay(
            RoundedRectangle(cornerRadius: ChatMetrics.bubbleCornerRadius)
                .stroke(Color.red.opacity(0.25), lineWidth: 1)
        )
        .contextMenu {
            Button("Copy Error") {
                NSPasteboard.general.clearContents()
                NSPasteboard.general.setString(notice.message, forType: .string)
            }
        }
    }
}
