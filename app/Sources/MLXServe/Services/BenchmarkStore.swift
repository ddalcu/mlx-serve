import Foundation

/// Local benchmark history and the community database client.
///
/// The community backend is Firebase Realtime Database over plain HTTPS — no
/// SDK, two calls. RTDB was picked over Firestore precisely because its REST
/// shape is ordinary JSON: the website reads the same URL with a bare `fetch`
/// and no client library at all.
///
/// Phase 1 has no tamper protection by design. That makes the READER the place
/// robustness has to live: rows decode individually so one junk write can't
/// blank the board, and unknown fields are ignored so a client shipped today
/// keeps reading rows written by a client shipped later.
enum BenchmarkStore {

    // MARK: - Shared coders

    /// ISO8601 dates. Swift's default is a reference-epoch Double, which the
    /// website would render as 2001 — and the website is the whole point of
    /// picking a plain-JSON database.
    static let encoder: JSONEncoder = {
        let e = JSONEncoder()
        e.dateEncodingStrategy = .iso8601
        e.outputFormatting = [.withoutEscapingSlashes]
        return e
    }()

    static let decoder: JSONDecoder = {
        let d = JSONDecoder()
        d.dateDecodingStrategy = .iso8601
        return d
    }()

    // MARK: - Local history

    static var historyURL: URL {
        let base = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("MLX Core", isDirectory: true)
        try? FileManager.default.createDirectory(at: base, withIntermediateDirectories: true)
        return base.appendingPathComponent("benchmarks.json")
    }

    static func loadLocal() -> [BenchmarkResult] {
        guard let data = try? Data(contentsOf: historyURL) else { return [] }
        return decodeRows(data)
    }

    static func saveLocal(_ rows: [BenchmarkResult]) {
        guard let data = try? encoder.encode(rows) else { return }
        try? data.write(to: historyURL, options: .atomic)
    }

    @discardableResult
    static func appendLocal(_ rows: [BenchmarkResult]) -> [BenchmarkResult] {
        let merged = merged(loadLocal(), adding: rows)
        saveLocal(merged)
        return merged
    }

    /// Newest first, deduped by id so a resubmitted session appears once.
    static func merged(_ existing: [BenchmarkResult], adding: [BenchmarkResult]) -> [BenchmarkResult] {
        var byId: [String: BenchmarkResult] = [:]
        for row in existing + adding { byId[row.id] = row }
        return byId.values.sorted { $0.date > $1.date }
    }

    /// Decodes a plain JSON array, skipping rows that don't parse.
    private static func decodeRows(_ data: Data) -> [BenchmarkResult] {
        guard let items = try? JSONSerialization.jsonObject(with: data) as? [Any] else { return [] }
        return items.compactMap { item in
            guard let itemData = try? JSONSerialization.data(withJSONObject: item) else { return nil }
            return try? decoder.decode(BenchmarkResult.self, from: itemData)
        }
        .filter(\.isLadderRow)
    }

    // MARK: - Shared marker

    /// Row ids already sent to the community database. Local only: a field
    /// on the row would ride the POST and the rules reject unknown fields,
    /// so it lives beside the history instead of in it. Per ROW, not per
    /// session: rows POST one at a time, and a retry after a failure halfway
    /// must send only what did not land, or the board counts the rest twice.
    static let sharedKey = "benchmarkSharedRows"

    static func sharedRowIds(_ defaults: UserDefaults = .standard) -> Set<String> {
        Set(defaults.stringArray(forKey: sharedKey) ?? [])
    }

    static func markShared(_ rowIds: [String], defaults: UserDefaults = .standard) {
        var ids = sharedRowIds(defaults)
        ids.formUnion(rowIds)
        defaults.set(Array(ids).sorted(), forKey: sharedKey)
    }

    /// A session is shared once every row worth sharing has landed.
    static func isShared(_ session: BenchmarkSession, defaults: UserDefaults = .standard) -> Bool {
        let rows = session.rungs.filter(\.isPublishable)
        guard !rows.isEmpty else { return false }
        let ids = sharedRowIds(defaults)
        return rows.allSatisfy { ids.contains($0.id) }
    }

    /// The publishable rows of `rows` not yet sent.
    static func unsent(_ rows: [BenchmarkResult], defaults: UserDefaults = .standard) -> [BenchmarkResult] {
        let ids = sharedRowIds(defaults)
        return rows.filter { $0.isPublishable && !ids.contains($0.id) }
    }

    // MARK: - Community database

    /// Firebase Realtime Database. Public by design: this identifies the
    /// database, and what may be written to it is enforced by the database's
    /// own security rules (see docs/benchmarks-setup.md).
    static let communityBaseURL = "https://mlxserve-default-rtdb.firebaseio.com"

    /// Newest N rows. `orderBy="$key"` sorts by push key, which RTDB generates
    /// in timestamp order, so `limitToLast` is "most recent".
    static func communityFetchURL(limit: Int) -> URL? {
        var components = URLComponents(string: "\(communityBaseURL)/results.json")
        components?.queryItems = [
            URLQueryItem(name: "orderBy", value: "\"$key\""),
            URLQueryItem(name: "limitToLast", value: String(limit)),
        ]
        return components?.url
    }

    static func communitySubmitURL() -> URL? {
        URL(string: "\(communityBaseURL)/results.json")
    }

    /// Parses an RTDB collection response.
    ///
    /// A POST creates a child under a generated push key, so the collection
    /// reads back as an OBJECT keyed by those pushes, never an array. Rows are
    /// decoded one at a time: decoding the collection as a unit would let a
    /// single malformed write hide every real result.
    static func decodeCommunity(_ data: Data) -> [BenchmarkResult] {
        guard !data.isEmpty,
              let object = try? JSONSerialization.jsonObject(with: data),
              let dict = object as? [String: Any] else { return [] }
        return dict.values.compactMap { value in
            guard let rowData = try? JSONSerialization.data(withJSONObject: value) else { return nil }
            return try? decoder.decode(BenchmarkResult.self, from: rowData)
        }
        .filter(\.isLadderRow)
        .sorted { $0.date > $1.date }
    }

    // MARK: - Aggregation

    /// The grouping that decides which rows may share a median: one machine
    /// × model × settings, every rung.
    ///
    /// GPU cores are part of the key because a 32-core and a 40-core M4 Max
    /// report the same chip string and do not share a decode speed. The
    /// settings SIGNATURE is part of it because a kv-quant 8 row and a dense
    /// row measured the same rung on the same Mac and describe different
    /// speeds. Engine version is deliberately NOT in the key: fragmenting by
    /// release would leave every cell at n=1 forever.
    static func familyKey(_ row: BenchmarkResult) -> String {
        [row.modelId, row.settingsSignature,
         row.hardware.chip, String(row.hardware.gpuCores), String(row.hardware.ramGB)]
            .joined(separator: "|")
    }

    /// A family of cells: one machine × model × settings, every rung.
    ///
    /// `samples(at:)` is displayed next to every figure: a rung built from
    /// one submission is a data point, not a benchmark, and the reader has to
    /// be able to tell the difference.
    struct CellFamily: Identifiable, Hashable {
        struct Rung: Hashable {
            var targetTokens: Int
            var decodeTps: Double
            var prefillTps: Double
            var ceilingDecodeTps: Double
            var ttftMs: Double
            var sampleCount: Int
        }

        var id: String
        var modelId: String
        var settings: [String: String]
        var isLossy: Bool
        var hardware: BenchmarkHardware
        var rungs: [Rung]
        var sessionCount: Int
        /// The newest session behind the row.
        var latestDate: Date

        func rung(at target: Int) -> Rung? { rungs.first { $0.targetTokens == target } }
        func decode(at target: Int) -> Double? { rung(at: target)?.decodeTps }
        func samples(at target: Int) -> Int { rung(at: target)?.sampleCount ?? 0 }
    }

    /// Per-rung medians per family, sorted by the fastest smallest rung.
    static func aggregateSessions(_ rows: [BenchmarkResult]) -> [CellFamily] {
        var groups: [String: [BenchmarkResult]] = [:]
        for row in rows { groups[familyKey(row), default: []].append(row) }

        return groups.compactMap { key, members -> CellFamily? in
            guard let first = members.first else { return nil }
            var byRung: [Int: [BenchmarkResult]] = [:]
            for row in members { byRung[row.effectiveTargetTokens, default: []].append(row) }
            let rungs = byRung.keys.sorted().map { target -> CellFamily.Rung in
                let rows = byRung[target] ?? []
                return CellFamily.Rung(
                    targetTokens: target,
                    decodeTps: BenchmarkStats.median(rows.map(\.decodeTps)),
                    prefillTps: BenchmarkStats.median(rows.map(\.prefillTps)),
                    ceilingDecodeTps: BenchmarkStats.median(rows.compactMap(\.ceilingDecodeTps).filter { $0 > 0 }),
                    ttftMs: BenchmarkStats.median(rows.map(\.ttftMs)),
                    sampleCount: rows.count)
            }
            return CellFamily(
                id: key,
                modelId: first.modelId,
                settings: first.settings ?? [:],
                isLossy: members.contains { $0.isLossy },
                hardware: first.hardware,
                rungs: rungs,
                sessionCount: Set(members.map(\.sessionId)).count,
                latestDate: members.map(\.date).max() ?? .distantPast)
        }
        .sorted { ($0.rungs.first?.decodeTps ?? 0) > ($1.rungs.first?.decodeTps ?? 0) }
    }
}

// MARK: - Network

/// Failures talking to the community database.
///
/// Its own type rather than `APIError`: that one's messages all say "from
/// mlx-serve" and point at the server log, which is actively misleading when
/// the thing that failed is a Firebase request (a live 404 rendered as
/// "HTTP 404 from mlx-servecommunity fetch failed").
enum BenchmarkCommunityError: LocalizedError {
    /// 404 — the Realtime Database hasn't been created yet. Much the most
    /// likely failure, so it gets the actionable message.
    case notConfigured
    /// 401/403 — the database exists but its rules reject us.
    case denied
    case http(Int)

    var errorDescription: String? {
        switch self {
        case .notConfigured:
            return "The community database hasn't been created yet. See docs/benchmarks-setup.md for the one-time Firebase setup."
        case .denied:
            return "The community database rejected the request. Its security rules may not be deployed yet — see docs/benchmarks-setup.md."
        case .http(let code):
            return "The community database returned HTTP \(code). Try again in a moment."
        }
    }

    static func from(status: Int) -> BenchmarkCommunityError {
        switch status {
        case 404: return .notConfigured
        case 401, 403: return .denied
        default: return .http(status)
        }
    }
}

/// Submission and fetch. Deliberately a thin wrapper: two requests, no SDK.
actor BenchmarkCommunityClient {

    private let session: URLSession
    init(session: URLSession = .shared) { self.session = session }

    /// Fetch the newest community rows.
    func fetch(limit: Int = 500) async throws -> [BenchmarkResult] {
        guard let url = BenchmarkStore.communityFetchURL(limit: limit) else { return [] }
        let (data, response) = try await session.data(from: url)
        guard let http = response as? HTTPURLResponse, http.statusCode == 200 else {
            throw BenchmarkCommunityError.from(status: (response as? HTTPURLResponse)?.statusCode ?? -1)
        }
        return BenchmarkStore.decodeCommunity(data)
    }

    /// What a submission managed: every row that landed, then the failure
    /// that stopped it, if any. Landed rows are the caller's to remember, so a
    /// retry sends the rest and never the same row twice.
    struct Outcome {
        var sentIds: [String] = []
        var error: Error?
    }

    /// Publish rows one at a time. Opt-in only — never called without an
    /// explicit press of Share.
    func submit(_ rows: [BenchmarkResult]) async -> Outcome {
        var outcome = Outcome()
        guard let url = BenchmarkStore.communitySubmitURL() else { return outcome }
        for row in rows {
            do {
                var request = URLRequest(url: url)
                request.httpMethod = "POST"
                request.setValue("application/json", forHTTPHeaderField: "Content-Type")
                request.httpBody = try BenchmarkStore.encoder.encode(row)
                let (_, response) = try await session.data(for: request)
                guard let http = response as? HTTPURLResponse, http.statusCode == 200 else {
                    throw BenchmarkCommunityError.from(status: (response as? HTTPURLResponse)?.statusCode ?? -1)
                }
                outcome.sentIds.append(row.id)
            } catch {
                outcome.error = error
                break
            }
        }
        return outcome
    }
}
