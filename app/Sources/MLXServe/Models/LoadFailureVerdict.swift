import Foundation

/// Whether a failed hot-switch load may fall back to a server restart. A load the
/// server REFUSED by name (no SSD budget, budget below the resident set, budget over
/// the wired limit, MTP on a streamed model) would fail the same way at boot, so a
/// restart only turns a 503 into a dead server. The verdict reads the 503 body's
/// `error.type`, never its prose; anything unparseable keeps the restart.
enum LoadFailureVerdict {
    /// The `type` strings `server.zig` emits for those refusals; duplicated across the
    /// Zig/Swift boundary and pinned by nothing but this list.
    static let namedRefusalTypes: Set<String> = [
        "expert_streaming_required",
        "ssd_budget_below_resident",
        "ssd_budget_exceeds_wired_limit",
        "expert_streaming_mtp_unsupported",
        "expert_streaming_unsupported_layout",
        "expert_slab_import_copied",
    ]

    static func errorType(fromBody data: Data) -> String? {
        guard let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let err = obj["error"] as? [String: Any],
              let type = err["type"] as? String, !type.isEmpty else { return nil }
        return type
    }

    static func shouldRestartAfterLoadFailure(body: Data) -> Bool {
        guard let type = errorType(fromBody: body) else { return true }
        return !namedRefusalTypes.contains(type)
    }
}
