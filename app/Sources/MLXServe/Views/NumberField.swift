import SwiftUI

/// Whether a change to the bound value has to be painted into the box.
enum NumberFieldRepaint {
    /// The box owns its text while it is being typed into: repainting from the
    /// value mid-edit moves the caret, which is why the focused case used to
    /// decline outright. But a value that does NOT match what the box says was
    /// not produced by the box — it came from a menu, a slider, a dice roll —
    /// and a choice that does not show up reads as a choice that did not land.
    static func shouldRepaint(text: String, value: Int?, range: ClosedRange<Int>, focused: Bool) -> Bool {
        guard focused else { return true }
        return SeedText.parse(text, in: range) != value
    }
}

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
        TextField(L10n.text(placeholder), text: $text)
            .textFieldStyle(.roundedBorder)
            // Body, not caption: a bezeled field is as tall as its font, and a
            // caption-sized box sat visibly below the pickers beside it.
            .font(.app(.body).monospacedDigit())
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
                guard NumberFieldRepaint.shouldRepaint(text: text, value: clamped(v),
                                                       range: range, focused: focused) else { return }
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
        TextField(L10n.text(placeholder), text: $text)
            .textFieldStyle(.roundedBorder)
            // Body, not caption: a bezeled field is as tall as its font, and a
            // caption-sized box sat visibly below the pickers beside it.
            .font(.app(.body).monospacedDigit())
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
                guard NumberFieldRepaint.shouldRepaint(text: text, value: v,
                                                       range: range, focused: focused) else { return }
                text = v.map(String.init) ?? ""
            }
            .onAppear { text = value.map(String.init) ?? "" }
            .help(help ?? "")
    }
}
