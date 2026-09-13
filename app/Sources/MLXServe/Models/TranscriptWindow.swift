import Foundation

/// How much of a conversation the transcript lays out when it opens.
///
/// The transcript is laid out whole (exact heights are what every scroll
/// decision rests on), so its cost is the number of rendered replies: about
/// 10 ms per markdown layout. A chat of 500 turns opened in 2.5 s. The last
/// `rowLimit` rows open at once; the rest wait behind "Show earlier messages".
enum TranscriptWindow {

    /// Rows, not messages: a tool call's four messages are one row.
    static let rowLimit = 150

    /// Index of the first row laid out for a chat that has just opened.
    static func firstRow(total: Int, limit: Int = rowLimit) -> Int {
        max(0, total - limit)
    }

    /// The cut is an INDEX fixed when the chat opens, never "the last N": a
    /// streaming reply appends rows, and moving the cut with them would push
    /// the oldest visible row out from under a reader who is looking at it. A
    /// transcript that got shorter than its cut is cut like one that just
    /// opened.
    static func clamp(first: Int, total: Int, limit: Int = rowLimit) -> Int {
        first < total ? first : firstRow(total: total, limit: limit)
    }
}
