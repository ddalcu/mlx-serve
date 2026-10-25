import Foundation

/// MOSS-Image layout prompts, ported from the authors' prompt enhancer
/// (OpenMOSS/MOSS-Image `skills/moss-image-prompt-enhancer/scripts/compile_prompt.py`,
/// Apache-2.0; see NOTICE). A chat model drafts boxes as top-left pixel
/// `xyxy`; the model reads `abs_centralized` `yxyx` JSON on one line, without
/// the canvas size. Pure — Enhance applies it to the reply.
enum MossLayout {
    struct Canvas: Equatable {
        let width: Int
        let height: Int
    }

    /// The prompt a draft compiles to: a caption draft's caption, or the layout
    /// JSON on `canvas`; nil when `reply` is neither (plain prose stays as is).
    /// Boxes drawn on another canvas are scaled to this one, clipped to its
    /// edges, and dropped when nothing is left: unlike the authors' agent loop,
    /// a local chat model gets no second try.
    static func compile(_ reply: String, canvas: Canvas) -> String? {
        guard let data = reply.data(using: .utf8),
              let draft = (try? JSONSerialization.jsonObject(with: data)) as? [String: Any] else { return nil }
        if draft["items"] == nil, let caption = draft["caption"] as? String {
            return caption.trimmingCharacters(in: .whitespacesAndNewlines)
        }
        // Already in the model's form: nothing to convert.
        if draft["bbox_format"] as? String == "yxyx" { return reply.trimmingCharacters(in: .whitespacesAndNewlines) }
        guard let summary = draft["summary"] as? [String: Any], summary["caption"] is String,
              let items = draft["items"] as? [[String: Any]] else { return nil }
        let draftW = (draft["width"] as? NSNumber)?.doubleValue ?? Double(canvas.width)
        let draftH = (draft["height"] as? NSNumber)?.doubleValue ?? Double(canvas.height)
        guard draftW > 0, draftH > 0 else { return nil }
        let sx = Double(canvas.width) / draftW, sy = Double(canvas.height) / draftH
        let W = Double(canvas.width), H = Double(canvas.height)

        var compiled: [String] = []
        for item in items {
            guard let label = item["label"] as? String, let caption = item["caption"] as? String,
                  let raw = item["bbox"] as? [NSNumber], raw.count == 4 else { continue }
            let v = raw.map(\.doubleValue)
            let box = [min(max(v[0] * sx, 0), W), min(max(v[1] * sy, 0), H),
                       min(max(v[2] * sx, 0), W), min(max(v[3] * sy, 0), H)]
            guard let centered = centeredBox(box, width: canvas.width, height: canvas.height) else { continue }
            var fields = [("label", json(label)), ("caption", json(caption))]
            if let text = item["text"] as? String { fields.append(("text", json(text))) }
            for key in ["tags", "palette"] {
                if let list = item[key] as? [String] { fields.append((key, json(list))) }
            }
            fields.append(("bbox", "[" + centered.map(String.init).joined(separator: ",") + "]"))
            compiled.append(object(fields))
        }
        guard !compiled.isEmpty else { return nil }

        var head: [(String, String)] = []
        for key in ["meta", "canvas", "caption", "palette"] {
            if let value = summary[key] as? String { head.append((key, json(value))) }
            if let list = summary[key] as? [String] { head.append((key, json(list))) }
        }
        return object([("summary", object(head)), ("items", "[" + compiled.joined(separator: ",") + "]"),
                       ("bbox_coord_space", json("abs_centralized")), ("bbox_format", json("yxyx"))])
    }

    /// `[x1, y1, x2, y2]` top-left pixels → centered `[y1, x1, y2, x2]`,
    /// enclosing fractional edges first (floor the min, ceil the max); nil
    /// when the box has no area.
    static func centeredBox(_ b: [Double], width: Int, height: Int) -> [Int]? {
        // Rounded to 9 places first, as the reference does, so 1.00000000001 is 1.
        func r(_ v: Double) -> Double { (v * 1e9).rounded() / 1e9 }
        let out = [Int(r(b[1]).rounded(.down)) - height / 2, Int(r(b[0]).rounded(.down)) - width / 2,
                   Int(r(b[3]).rounded(.up)) - height / 2, Int(r(b[2]).rounded(.up)) - width / 2]
        guard b[0] < b[2], b[1] < b[3], out[0] < out[2], out[1] < out[3] else { return nil }
        return out
    }

    /// The model was shown keys in the authors' order, so objects are written by hand.
    private static func object(_ fields: [(String, String)]) -> String {
        "{" + fields.map { json($0.0) + ":" + $0.1 }.joined(separator: ",") + "}"
    }

    private static func json(_ value: Any) -> String {
        let data = (try? JSONSerialization.data(withJSONObject: value, options: [.fragmentsAllowed, .withoutEscapingSlashes])) ?? Data()
        return String(decoding: data, as: UTF8.self)
    }
}
