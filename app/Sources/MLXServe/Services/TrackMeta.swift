import Foundation
import AppKit

/// What the Modern player says about a generated clip: a title a person would
/// give it, the model that made it and when. Read from the `<clip>.txt`
/// sidecar and the file's own name, cached like `AudioClipPlayer.info`.
struct TrackMeta: Equatable {
    let title: String
    /// The model's short name, "" when the sidecar does not say.
    let model: String
    /// When the clip was written, from its name; nil for a file named by hand.
    let date: Date?

    private static var cache: [String: TrackMeta] = [:]

    @MainActor
    static func of(_ path: String) -> TrackMeta {
        if let hit = cache[path] { return hit }
        let txt = (path as NSString).deletingPathExtension + ".txt"
        let meta = parse(sidecar: try? String(contentsOfFile: txt, encoding: .utf8), path: path)
        cache[path] = meta
        return meta
    }

    static func parse(sidecar: String?, path: String) -> TrackMeta {
        let stem = ((path as NSString).lastPathComponent as NSString).deletingPathExtension
        let body = sidecar ?? ""
        let prompt = section("Style prompt", in: body) ?? section("Text", in: body) ?? ""
        return TrackMeta(title: title(prompt: prompt) ?? title(stem: stem),
                         model: modelName(sidecarValue("model", in: body) ?? ""),
                         date: date(stem: stem))
    }

    // MARK: - Title

    static let titleLimit = 60

    /// The prompt's first real line, cut to its first sentence: "Global
    /// Metadata" style headers and "Label:" prefixes are skipped. Music 3's
    /// caption format ends its "Basic Attributes" line with the genre, which
    /// names a track better than its bpm.
    static func title(prompt: String) -> String? {
        let lines = prompt.split(whereSeparator: \.isNewline)
            .map { $0.trimmingCharacters(in: .whitespaces) }
            .filter { !$0.isEmpty }
        let body = lines.count > 1 ? lines.filter { !isHeader($0) } : lines
        guard var line = body.first ?? lines.first else { return nil }
        var label = ""
        if let colon = line.firstIndex(of: ":"), line[..<colon].split(separator: " ").count <= 3,
           !line[line.index(after: colon)...].trimmingCharacters(in: .whitespaces).isEmpty {
            label = String(line[..<colon])
            line = line[line.index(after: colon)...].trimmingCharacters(in: .whitespaces)
        }
        let sentences = line.components(separatedBy: CharacterSet(charactersIn: ".!?;"))
            .map { $0.trimmingCharacters(in: .whitespaces) }
            .filter { !$0.isEmpty }
        line = (label == "Basic Attributes" ? sentences.last : sentences.first) ?? ""
        return line.isEmpty ? nil : capitalized(clip(line))
    }

    /// The prompt slug from `MediaStorage.datedPath`, read back as words. The
    /// slug stops at 40 characters, so a cut last word is dropped.
    static func title(stem: String) -> String {
        let slug = stem.replacingOccurrences(of: #"^\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}_?"#, with: "", options: .regularExpression)
        var words = slug.split(separator: "-").map(String.init)
        guard !words.isEmpty else { return stem }
        let cut = slug.count >= 40 && words.count > 1
        if cut { words.removeLast() }
        return capitalized(words.joined(separator: " ")) + (cut ? "…" : "")
    }

    private static func isHeader(_ line: String) -> Bool {
        line.split(separator: " ").count <= 3 && line.rangeOfCharacter(from: .punctuationCharacters) == nil
    }

    private static func clip(_ text: String) -> String {
        guard text.count > titleLimit else { return text }
        let head = text.prefix(titleLimit)
        let word = head.lastIndex(of: " ").map { head[..<$0] } ?? head
        return word.trimmingCharacters(in: .punctuationCharacters.union(.whitespaces)) + "…"
    }

    private static func capitalized(_ text: String) -> String {
        text.prefix(1).uppercased() + text.dropFirst()
    }

    // MARK: - Sidecar fields

    /// "org/Repo-Name-8bit" → "Repo-Name-8bit": the folder is noise in a row.
    static func modelName(_ raw: String) -> String {
        (raw as NSString).lastPathComponent
    }

    private static func section(_ name: String, in body: String) -> String? {
        guard let start = body.range(of: "# \(name)\n") else { return nil }
        let rest = body[start.upperBound...]
        let end = rest.range(of: "\n\n")?.lowerBound ?? rest.endIndex
        let text = rest[..<end].trimmingCharacters(in: .whitespacesAndNewlines)
        return text.isEmpty ? nil : text
    }

    private static func sidecarValue(_ key: String, in body: String) -> String? {
        for line in body.split(separator: "\n") where line.hasPrefix("\(key): ") {
            return String(line.dropFirst(key.count + 2)).trimmingCharacters(in: .whitespaces)
        }
        return nil
    }

    // MARK: - Date

    /// `MediaStorage.datedPath` writes the stamp with the user's calendar, so it
    /// is read back with the same one.
    static func date(stem: String, calendar: Calendar = .current) -> Date? {
        guard stem.count >= 19 else { return nil }
        if stampFormatter.calendar != calendar {
            stampFormatter.calendar = calendar
            stampFormatter.timeZone = calendar.timeZone
        }
        return stampFormatter.date(from: String(stem.prefix(19)))
    }

    /// The history groups every row by day on each redraw: one formatter, not one per row.
    private static let stampFormatter: DateFormatter = {
        let df = DateFormatter()
        df.locale = Locale(identifier: "en_US_POSIX")
        df.dateFormat = "yyyy-MM-dd_HH-mm-ss"
        return df
    }()

    // MARK: - Cover

    /// Two hues picked from the file name: the same clip always gets the same
    /// cover, and neighbours in the list rarely share one.
    static func coverHues(_ path: String) -> (Double, Double) {
        var hash: UInt64 = 0xcbf2_9ce4_8422_2325
        for byte in ((path as NSString).lastPathComponent).utf8 {
            hash = (hash ^ UInt64(byte)) &* 0x0000_0100_0000_01B3
        }
        let first = Double(hash % 360) / 360
        let spread = 0.08 + Double((hash >> 16) % 100) / 100 * 0.14
        return (first, (first + spread).truncatingRemainder(dividingBy: 1))
    }

    static func coverColors(_ path: String) -> (NSColor, NSColor) {
        let (a, b) = coverHues(path)
        return (NSColor(hue: a, saturation: 0.62, brightness: 0.78, alpha: 1),
                NSColor(hue: b, saturation: 0.72, brightness: 0.46, alpha: 1))
    }
}

/// The history in day groups, newest first, the way the list shows it.
struct TrackDay: Identifiable, Equatable {
    let id: String
    let label: String
    let paths: [String]

    /// Consecutive paths that share a day form one group; a clip with no date
    /// joins "Earlier".
    static func group(_ paths: [String], date: (String) -> Date?, now: Date = Date(), calendar: Calendar = .current) -> [TrackDay] {
        var days: [TrackDay] = []
        for path in paths {
            let day = date(path).map { calendar.startOfDay(for: $0) }
            let id = day.map { "\($0.timeIntervalSinceReferenceDate)" } ?? "earlier"
            if let last = days.last, last.id == id {
                days[days.count - 1] = TrackDay(id: id, label: last.label, paths: last.paths + [path])
            } else {
                days.append(TrackDay(id: id, label: label(day, now: now, calendar: calendar), paths: [path]))
            }
        }
        return days
    }

    static func label(_ day: Date?, now: Date, calendar: Calendar) -> String {
        guard let day else { return L10n.text("Earlier") }
        if calendar.isDate(day, inSameDayAs: now) { return L10n.text("Today") }
        if let yesterday = calendar.date(byAdding: .day, value: -1, to: now), calendar.isDate(day, inSameDayAs: yesterday) {
            return L10n.text("Yesterday")
        }
        let sameYear = calendar.component(.year, from: day) == calendar.component(.year, from: now)
        var style = Date.FormatStyle(date: .omitted, time: .omitted).weekday(.wide).month(.abbreviated).day()
        if !sameYear { style = style.year() }
        style.calendar = calendar
        return day.formatted(style)
    }
}
