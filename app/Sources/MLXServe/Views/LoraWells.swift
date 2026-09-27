import SwiftUI

/// The Style LoRAs block the Image and Video panes share: one well per
/// attached adapter with its scale, and the add well under the last one.
enum LoraWells {
    /// Thinner than the image wells' 84: two lines of text, not a thumbnail.
    static let minHeight: CGFloat = 64
}

/// Click-only: a LoRA is picked from a file panel, and a well that accepts a
/// drag is a promise this one does not keep.
struct LoraAddWell: View {
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            MediaWellAction(title: "Choose .safetensors…",
                            systemImage: "paintpalette",
                            caption: "Add LoRA adapter for custom style. Several can stack at once.")
            .padding(.horizontal, 12)
            .padding(.vertical, 8)
            .frame(maxWidth: .infinity, minHeight: LoraWells.minHeight, alignment: .center)
            .background(MediaDropWellBackground(isTargeted: false))
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
    }
}

/// One attached adapter: its file name, its scale and the way out.
struct LoraAdapterRow: View {
    @Binding var lora: LoraAdapter
    let remove: () -> Void

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack(spacing: 8) {
                Image(systemName: "paintpalette")
                    .foregroundStyle(.secondary)
                Text(URL(fileURLWithPath: lora.path).lastPathComponent)
                    .font(.app(.caption))
                    .lineLimit(1)
                    .truncationMode(.middle)
                    .help(lora.path)
                Spacer()
                Button(action: remove) {
                    Image(systemName: "xmark.circle.fill")
                }
                .buttonStyle(.borderless)
                .foregroundStyle(.secondary)
                .help("Remove this LoRA")
            }
            HStack(spacing: 8) {
                Text("Scale").font(.app(.caption))
                Slider(value: $lora.scale, in: 0...2, step: 0.05)
                // Fixed width: a readout that sizes to its digits drags the
                // slider's right edge every time the value crosses a width.
                Text(String(format: "%.2f", lora.scale))
                    .font(.app(.caption).monospacedDigit())
                    .foregroundStyle(.secondary)
                    .frame(width: 34, alignment: .trailing)
            }
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 8)
        .frame(maxWidth: .infinity, minHeight: LoraWells.minHeight, alignment: .leading)
        .background(MediaDropWellBackground(isTargeted: false))
    }
}
