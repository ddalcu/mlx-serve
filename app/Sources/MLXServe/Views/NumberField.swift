import SwiftUI

/// A small typed number box, clamped to a range.
///
/// Sliders are fine for exploring and terrible for landing on a value: the
/// music pane's duration was a `Slider(step: 5)` with a read-only label, so
/// asking for 95 seconds meant dragging, and BPM was ten fixed menu entries
/// against a server that takes 30 to 300. Both are numbers people already know
/// and want to type.
///
/// Reading is deliberately forgiving — it shares `SeedText.parse`, so a value
/// pasted out of a caption ("~128 bpm", "1:30") still lands. Clamping happens
/// in the reader, so an out-of-range number becomes the nearest legal one
/// instead of travelling to the server and earning a 400.
struct NumberField: View {
    let range: ClosedRange<Int>
    @Binding var value: Int
    var placeholder: String = ""
    var width: CGFloat = 70
    var help: String? = nil

    @State private var text: String = ""
    @FocusState private var focused: Bool

    var body: some View {
        TextField(placeholder, text: $text)
            .textFieldStyle(.roundedBorder)
            .font(.caption.monospacedDigit())
            .frame(width: width)
            .focused($focused)
            .multilineTextAlignment(.trailing)
            .onChange(of: text) { _, t in
                if let v = SeedText.parse(t, in: range) { value = v }
            }
            // On blur, repaint from the value so an abandoned half-edit ("9")
            // shows what will actually run rather than the user's scratch.
            .onChange(of: focused) { _, isFocused in
                if !isFocused { text = String(clamped(value)) }
            }
            // A value changed from OUTSIDE (a slider drag, hydration, a model
            // switch that re-clamped the range) has to show up in the box.
            .onChange(of: value) { _, v in
                guard !focused else { return }
                text = String(clamped(v))
            }
            .onAppear { text = String(clamped(value)) }
            .help(help ?? "")
    }

    private func clamped(_ v: Int) -> Int {
        min(range.upperBound, max(range.lowerBound, v))
    }
}

/// The same box, but empty means "let the model decide" — the state the music
/// pane's Auto rows used to express. A separate type rather than an optional
/// binding on `NumberField` so the non-optional case cannot accidentally
/// acquire an empty state it has no meaning for.
struct OptionalNumberField: View {
    let range: ClosedRange<Int>
    @Binding var value: Int?
    var placeholder: String = "Auto"
    var width: CGFloat = 70
    var help: String? = nil

    @State private var text: String = ""
    @FocusState private var focused: Bool

    var body: some View {
        TextField(placeholder, text: $text)
            .textFieldStyle(.roundedBorder)
            .font(.caption.monospacedDigit())
            .frame(width: width)
            .focused($focused)
            .multilineTextAlignment(.trailing)
            .onChange(of: text) { _, t in
                if let v = SeedText.parse(t, in: range) {
                    value = v
                } else if t.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                    // Clearing the box IS the Auto row: the field is omitted
                    // from the request and the model picks.
                    value = nil
                }
            }
            .onChange(of: focused) { _, isFocused in
                if !isFocused { text = value.map(String.init) ?? "" }
            }
            .onChange(of: value) { _, v in
                guard !focused else { return }
                text = v.map(String.init) ?? ""
            }
            .onAppear { text = value.map(String.init) ?? "" }
            .help(help ?? "")
    }
}
