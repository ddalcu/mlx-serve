import Foundation

/// Pure logic behind the menu bar's steering number box (`SteeringQuickBox`).
enum SteeringQuickSet {
    /// The server's own bound (`steering.SCALE_LIMIT`). Values outside it are a named 400,
    /// so refusing here keeps the box from sending a request that cannot succeed.
    static let limit: Double = 100

    /// What the user typed, or nil when it is not a scale the server would accept.
    /// Rejects empty, junk, and non-finite — NOT zero, which is how you turn the arm off.
    static func parse(_ text: String) -> Double? {
        let t = text.trimmingCharacters(in: .whitespaces)
        guard !t.isEmpty, let v = Double(t), v.isFinite, abs(v) <= limit else { return nil }
        return v
    }

    /// One step of the stepper. Coarse enough to be useful, fine enough that the step
    /// before the cliff is still reachable.
    private static let step: Double = 0.1

    /// `value + step * count`, clamped to the server's bound and rounded to one decimal so
    /// repeated stepping cannot accumulate binary-float dust into the box.
    static func stepped(_ value: Double, by count: Int) -> Double {
        (clamped(value + step * Double(count)) * 10).rounded() / 10
    }

    /// What to send for a typed ffn. Both arms at zero is the server's own "off", so a
    /// box set to 0 turns steering off rather than sending a no-op edit.
    static func override(name: String, ffn: Double, attn: Double) -> SteeringOverride {
        if ffn == 0, attn == 0 { return .off }
        return .configured(name: name, ffn: ffn, attn: attn)
    }

    /// A scale the server will accept. Outside the bound `validateScale` fails at load
    /// and `model_settings.fromValue` drops the whole steering key, so the model loads
    /// unsteered with nothing to explain it; non-finite becomes 0 (off).
    static func clamped(_ v: Double) -> Double {
        guard v.isFinite else { return 0 }
        return min(max(v, -limit), limit)
    }

    /// `/props` describes ONE model (the app asks for the resident chat model), but the
    /// tray draws a row per resident model. The box belongs only to the row the fetch
    /// described; other rows configure steering through Model Settings.
    static func ownsLiveSteering(row: String, liveChatModel: String?) -> Bool {
        guard let live = liveChatModel, !live.isEmpty, !row.isEmpty else { return false }
        return row == live
    }

    /// Mirrors the server's `steering.archSupported`.
    static func archSupportsSteering(_ architecture: String) -> Bool {
        architecture.hasPrefix("qwen4_exp")
    }

    /// The bank a model will arm on load, read from the settings file (the tray lists
    /// only resident models).
    static func configuredBank(path: String, file: ModelSettingsFile) -> String? {
        guard case .configured(let name, let ffn, let attn)? = file.override(for: path)?.steering,
              ffn != 0 || attn != 0 else { return nil }
        return name
    }

    /// The server reports nothing at 0/0 (`Banks.reserve` refuses scale 0), so the bank
    /// remembered this session keeps the box on screen for the next value.
    static func visibleBank(live: String?, remembered: String?) -> String? {
        if let l = live, !l.isEmpty { return l }
        if let r = remembered, !r.isEmpty { return r }
        return nil
    }

    /// The "steer X on load" badge: a bank saved for the selected model, shown until the
    /// live box draws for that model — including at 0/0, when the server reports nothing
    /// but the box stays up on the remembered bank. Never both for one model.
    static func loadBadge(configured: String?, boxDrawn: Bool) -> String? {
        guard let c = configured, !boxDrawn else { return nil }
        return c
    }
}
