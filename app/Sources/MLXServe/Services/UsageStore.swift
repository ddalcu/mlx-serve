import Foundation

/// The dashboard's history in `~/.mlx-serve/usage.db`: what each model did on each local calendar day. A file of its
/// own, so wiping the chats or the models never wipes the tally, and resetting the tally never touches a chat.
final class UsageStore {
    private let db: SQLiteDatabase

    static var defaultPath: String { ChatStore.inAppFolder("usage.db") }

    init(path: String = UsageStore.defaultPath) throws {
        db = try SQLiteDatabase(path: path)
        try db.execute("""
            CREATE TABLE IF NOT EXISTS usage_daily (
                day TEXT NOT NULL,
                model TEXT NOT NULL,
                requests INTEGER NOT NULL,
                prompt_tokens INTEGER NOT NULL,
                cached_tokens INTEGER NOT NULL,
                generation_tokens INTEGER NOT NULL,
                prefill_ns INTEGER NOT NULL,
                decode_ns INTEGER NOT NULL,
                PRIMARY KEY (day, model)
            )
            """)
        guard try db.userVersion() < 1 else { return }
        try db.transaction {
            guard try db.userVersion() < 1 else { return }   // another process may have added them since
            try db.execute("""
                ALTER TABLE usage_daily ADD COLUMN peak_decode_tps REAL NOT NULL DEFAULT 0;
                ALTER TABLE usage_daily ADD COLUMN peak_prefill_tps REAL NOT NULL DEFAULT 0;
                """)
            try db.setUserVersion(1)
        }
    }

    /// Adds each delta to its day's row for that model, all or nothing.
    func add(_ deltas: [UsageDelta]) throws {
        guard !deltas.isEmpty else { return }
        try db.transaction {
            for d in deltas {
                let c = d.counters
                try db.run("""
                    INSERT INTO usage_daily
                        (day, model, requests, prompt_tokens, cached_tokens, generation_tokens, prefill_ns, decode_ns,
                         peak_decode_tps, peak_prefill_tps)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT (day, model) DO UPDATE SET
                        requests = requests + excluded.requests,
                        prompt_tokens = prompt_tokens + excluded.prompt_tokens,
                        cached_tokens = cached_tokens + excluded.cached_tokens,
                        generation_tokens = generation_tokens + excluded.generation_tokens,
                        prefill_ns = prefill_ns + excluded.prefill_ns,
                        decode_ns = decode_ns + excluded.decode_ns,
                        peak_decode_tps = MAX(peak_decode_tps, excluded.peak_decode_tps),
                        peak_prefill_tps = MAX(peak_prefill_tps, excluded.peak_prefill_tps)
                    """, [.text(d.day), .text(d.model), .integer(Int(c.requests)), .integer(Int(c.promptTokens)),
                          .integer(Int(c.cachedTokens)), .integer(Int(c.generatedTokens)),
                          .integer(Int(c.prefillNs)), .integer(Int(c.decodeNs)),
                          .real(c.peakDecodeTPS), .real(c.peakPrefillTPS)])
            }
        }
    }

    /// Every row, oldest day first.
    func rows() throws -> [UsageRow] {
        var out: [UsageRow] = []
        try db.query("""
            SELECT day, model, requests, prompt_tokens, cached_tokens, generation_tokens, prefill_ns, decode_ns,
                   peak_decode_tps, peak_prefill_tps
            FROM usage_daily ORDER BY day, model
            """) { r in
            out.append(UsageRow(day: r.text(0) ?? "", model: r.text(1) ?? "",
                                counters: UsageCounters(requests: Int64(r.integer(2)), promptTokens: Int64(r.integer(3)),
                                                        cachedTokens: Int64(r.integer(4)), generatedTokens: Int64(r.integer(5)),
                                                        prefillNs: Int64(r.integer(6)), decodeNs: Int64(r.integer(7)),
                                                        peakDecodeTPS: r.real(8), peakPrefillTPS: r.real(9))))
        }
        return out
    }

    /// Forgets one model's history, or everything's when `model` is nil.
    func reset(model: String?) throws {
        if let model {
            try db.run("DELETE FROM usage_daily WHERE model = ?", [.text(model)])
        } else {
            try db.run("DELETE FROM usage_daily")
        }
    }
}
