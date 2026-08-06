import SwiftUI

/// Reading and showing a seed the user typed or PASTED.
///
/// The image and video panes rendered their seed as a `Stepper` whose label was
/// a static `Text`, so there was no text entry at all — the only way to reach a
/// value was ±1 clicks, and a pasted 7-digit seed was 3.8 million of them. The
/// music pane had a plain `TextField` all along; this is that, shared, so the
/// three cannot drift.
enum SeedText {

    /// Read a seed out of arbitrary pasted text, or nil when there is nothing
    /// readable in it.
    ///
    /// Deliberately forgiving: seeds get pasted out of captions, filenames and
    /// chat messages ("Seed: 3,847,592"), and a field that rejects those is
    /// refusing exactly the input it exists to accept. A leading `-` is a SIGN
    /// only where the range allows one — elsewhere it is a filename separator.
    ///
    /// nil (rather than 0) for unreadable input, so a caller can leave its
    /// current value alone: clearing the box to type a new number is the normal
    /// way to use it, and snapping to 0 mid-edit fights the user.
    static func parse(_ text: String, in range: ClosedRange<Int>) -> Int? {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        let negative = range.lowerBound < 0 && trimmed.hasPrefix("-")
        let body = Array(negative ? String(trimmed.dropFirst()) : trimmed)

        // The LONGEST run of digits, with `,` bridged when it sits between two
        // of them. Keeping every digit in the string instead would let
        // "video_1234_768p.mp4" donate the 4 out of "mp4" and read as
        // 12347684; taking the FIRST run instead would read "3,847,592" as 3.
        var best = "", current = ""
        for (i, c) in body.enumerated() {
            if c.isASCII && c.isNumber {
                current.append(c)
                continue
            }
            let bridges = c == "," && !current.isEmpty
                && i + 1 < body.count && body[i + 1].isASCII && body[i + 1].isNumber
            if bridges { continue }
            if current.count > best.count { best = current }
            current = ""
        }
        if current.count > best.count { best = current }

        guard !best.isEmpty, let magnitude = Int(best) else { return nil }
        let value = negative ? -magnitude : magnitude
        return min(range.upperBound, max(range.lowerBound, value))
    }

    /// What the field shows for a value it already holds. A negative sentinel
    /// renders EMPTY so the placeholder ("random") does the explaining — a
    /// literal "-1" sitting in a seed box reads as a broken value.
    /// Clamps first, so a stored value the range cannot hold shows what will
    /// actually RUN — a `-1` restored into a pane with no random path would
    /// otherwise render blank and read as random when it is not.
    static func format(_ value: Int, in range: ClosedRange<Int>) -> String {
        let clamped = min(range.upperBound, max(range.lowerBound, value))
        guard clamped >= 0 else { return "" }
        return String(clamped)
    }

    /// A fresh seed for the dice.
    ///
    /// 32-bit, because that is the range seeds are shared in and a 19-digit one
    /// is unpasteable in practice even though `Int` would hold it. Never the
    /// negative sentinel: a dice that rolls "surprise me" has not rolled
    /// anything, and the point of the button is a number you can read off and
    /// paste back.
    static func randomSeed(in range: ClosedRange<Int>) -> Int {
        let lo = max(0, range.lowerBound)
        let hi = min(range.upperBound, Int(UInt32.max))
        guard lo < hi else { return lo }
        return Int.random(in: lo...hi)
    }

    /// The value an EMPTY box means, or nil when empty means "leave it alone".
    ///
    /// Where the range has a negative half, empty IS the random sentinel — the
    /// mirror of `format` rendering it blank. Without this, clearing the image
    /// pane's seed box would silently keep the last number instead of going
    /// back to random, which is the behaviour it shipped with.
    static func emptyValue(in range: ClosedRange<Int>) -> Int? {
        range.lowerBound < 0 ? range.lowerBound : nil
    }
}

/// A seed box: type it, or paste it.
///
/// Keeps the typed STRING as the source of truth while the field has focus, so
/// the box can be emptied and retyped; the bound value only moves when the text
/// reads as a number. On blur the text is re-formatted from the value, so an
/// abandoned half-edit ("Seed: ") shows the seed that will actually be used
/// rather than the user's scratch.
struct SeedField: View {
    let label: String
    let placeholder: String
    let range: ClosedRange<Int>
    @Binding var value: Int
    var help: String? = nil

    @State private var text: String = ""
    @FocusState private var focused: Bool

    var body: some View {
        VStack(alignment: .leading, spacing: 2) {
            Text(label).font(.caption)
            HStack(spacing: 6) {
                field
                Button {
                    roll()
                } label: {
                    Image(systemName: "die.face.5")
                }
                .buttonStyle(.borderless)
                .help("Roll a new seed")
                .accessibilityLabel("Roll a new seed")
            }
        }
    }

    /// Both halves in one place: clicking the dice while the field has focus
    /// must still repaint it, and the `value` observer below deliberately
    /// declines to touch the text while the user is mid-edit.
    private func roll() {
        let v = SeedText.randomSeed(in: range)
        value = v
        text = SeedText.format(v, in: range)
    }

    private var field: some View {
        TextField(placeholder, text: $text)
                .textFieldStyle(.roundedBorder)
                .font(.caption.monospacedDigit())
                .frame(width: 160)
                .focused($focused)
                .onChange(of: text) { _, t in
                    if let v = SeedText.parse(t, in: range) {
                        value = v
                    } else if t.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty,
                              let empty = SeedText.emptyValue(in: range) {
                        value = empty
                    }
                }
                .onChange(of: focused) { _, isFocused in
                    if !isFocused { text = SeedText.format(value, in: range) }
                }
                // A value changed from OUTSIDE the field (hydration, a preset
                // switch) must show up in it.
                .onChange(of: value) { _, v in
                    guard !focused else { return }
                    text = SeedText.format(v, in: range)
                }
                .onAppear { text = SeedText.format(value, in: range) }
                .help(help ?? "")
    }
}
