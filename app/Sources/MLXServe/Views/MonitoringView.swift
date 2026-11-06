import SwiftUI
import Charts

enum MonitorFormat {
    static func number(_ value: Double?) -> String {
        guard let value, value.isFinite else { return "—" }
        return value.formatted(.number.precision(.fractionLength(0...2)))
    }
    static func bytes(_ value: Double?) -> String {
        guard let value, value.isFinite, value >= 0, value < Double(Int64.max) else { return "—" }
        return ByteCountFormatter.string(fromByteCount: Int64(value), countStyle: .memory)
    }
    static func modelSize(_ model: ModelInfo) -> String {
        bytes(model.bytesResident > 0 ? Double(model.bytesResident) : model.bytesOnDisk.map { Double($0) })
    }
    static func phase(_ phase: String) -> String {
        switch phase {
        case "prefill": L10n.text("Prefilling")
        case "decode": L10n.text("Decoding")
        case "cached": L10n.text("In cache")
        default: phase
        }
    }
    static func capability(_ key: String) -> String {
        switch key {
        case "chat": L10n.text("Chat")
        case "tool_use": L10n.text("Tool use")
        case "streaming": L10n.text("Streaming")
        case "vision": L10n.text("Vision")
        case "thinking": L10n.text("Thinking")
        case "json_schema": L10n.text("Structured output")
        case "audio": L10n.text("Audio")
        case "music": L10n.text("Music")
        case "image": L10n.text("Image")
        case "video": L10n.text("Video")
        case "transcription": L10n.text("Transcription")
        case "alignment": L10n.text("Alignment")
        case "diarization": L10n.text("Speaker diarization")
        case "reranking": L10n.text("Reranking")
        default: key
        }
    }
    static func state(_ state: String?, loaded: Bool) -> String {
        switch state {
        case "ready": L10n.text("Ready")
        case "loading": L10n.text("Loading")
        case "unloaded": L10n.text("Unloaded")
        case "evicting": L10n.text("Evicting")
        case "error": L10n.text("Error")
        case .none: L10n.text(loaded ? "Loaded" : "Unloaded")
        default: state ?? "—"
        }
    }
    static func color(_ percent: Double) -> Color { percent >= 90 ? .red : percent >= 70 ? .orange : .accentColor }
}

struct MonitoringView: View {
    @ObservedObject var store: MonitoringStore
    var openSettings: () -> Void
    @State private var range: MonitoringRange = .live
    private var from: TimeInterval { store.now - range.seconds }
    private var snapshot: MonitoringSnapshot? { store.snapshot }
    private func counter(_ key: String) -> Double? { snapshot?.counters[key] }
    private func gauge(_ key: String) -> Double? { snapshot?.gauges[key] }
    private func average(_ key: String) -> Double? { snapshot?.averageMilliseconds(key) }
    private func fmt(_ value: Double?) -> String { MonitorFormat.number(value) }

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 20) {
                HStack {
                    VStack(alignment: .leading, spacing: 6) {
                        Text(L10n.text("Monitoring")).font(.app(.title))
                        Text(store.serverName.isEmpty ? L10n.text("Local Server") : store.serverName)
                            .font(.app(.caption)).foregroundStyle(.secondary).textSelection(.enabled)
                    }
                    Spacer()
                    Text(store.status == .live ? L10n.text("Live") : store.status == .connecting ? L10n.text("Connecting") : L10n.text("Unavailable"))
                        .font(.app(.caption)).padding(8).background(.quaternary, in: Capsule())
                }
                if store.status != .live {
                    HStack {
                        Text(store.status.text).font(.app(.body)).foregroundStyle(.secondary)
                        Spacer()
                        Button(L10n.text("Settings"), action: openSettings)
                        if store.status != .stopped { Button(L10n.text("Retry")) { store.retry() } }
                    }
                }
                if let warning = store.storageWarning {
                    Text(warning).font(.app(.caption)).foregroundStyle(.orange)
                }
                LazyVGrid(columns: [GridItem(.adaptive(minimum: 200), spacing: 12)], spacing: 12) {
                    ForEach(cards) { card in MonitoringMetricCard(card: card) }
                }
                Picker(L10n.text("Monitoring range"), selection: $range) {
                    ForEach(MonitoringRange.allCases) { Text($0.title).tag($0) }
                }
                .pickerStyle(.segmented).frame(maxWidth: 520).accessibilityLabel(L10n.text("Monitoring range"))
                Text("\(Date(timeIntervalSince1970: from).formatted(date: range == .day ? .abbreviated : .omitted, time: .standard)) – \(Date(timeIntervalSince1970: store.now).formatted(date: range == .day ? .abbreviated : .omitted, time: .standard))")
                    .font(.app(.caption)).foregroundStyle(.secondary)
                LazyVGrid(columns: [GridItem(.adaptive(minimum: 340), spacing: 12)], spacing: 12) {
                    MonitoringChart(title: range == .live ? L10n.text("Decode tok/s") : L10n.text("Generated tok/s"),
                        points: range == .live ? store.decodePoints : series(["generation_tokens_total"]), from: from, to: store.now)
                    MonitoringChart(title: L10n.text("Prefill tok/s"),
                        points: range == .live ? store.prefillPoints : series(["prefill_tokens_total"]), from: from, to: store.now)
                    MonitoringChart(title: L10n.text("Successful requests / min"),
                        points: series(["requests_success_total"], scale: 60), from: from, to: store.now)
                    MonitoringChart(title: L10n.text("Failed + rejected + cancelled / min"),
                        points: series(["requests_failed_total", "requests_rejected_total", "requests_cancelled_total"], scale: 60), from: from, to: store.now)
                }
                Text(L10n.text("In this window (observed): ") + summary(store.history.totals(from: from, to: store.now)))
                    .font(.app(.caption)).foregroundStyle(.secondary)
                Text(L10n.text("Since server startup: ") + summary(snapshot?.counters ?? [:]))
                    .font(.app(.caption)).foregroundStyle(.secondary)
                sessionsTable
                modelTotalsTable
                requestTable
                Text(L10n.text("History records observations while this app is running. Up to 24 hours are saved locally."))
                    .font(.app(.caption)).foregroundStyle(.secondary)
                modelsTable
            }
            .padding(24)
        }
        .scrollEdgeEffectStyle(.soft, for: .top)
        .navigationTitle(L10n.text("Monitoring"))
    }

    private func series(_ keys: [String], scale: Double = 1) -> [MonitoringPoint] {
        store.history.series(keys: keys, from: from, to: store.now).map {
            MonitoringPoint(time: $0.time, value: $0.value.map { $0 * scale }, segment: $0.segment)
        }
    }

    private func summary(_ values: [String: Double]) -> String {
        L10n.format("%@ ok · %@ failed · %@ rejected · %@ cancelled", fmt(values["requests_success_total"]),
            fmt(values["requests_failed_total"]), fmt(values["requests_rejected_total"]), fmt(values["requests_cancelled_total"]))
    }

    private var cards: [MonitoringMetricCard.Card] {
        let prefillTime = snapshot?.histograms["prefill_time_seconds"]?.sum
        let prefillAverage = (prefillTime ?? 0) > 0 ? counter("prefill_tokens_total").map { $0 / prefillTime! } : nil
        let prefillProgress = (gauge("requests_prefilling") ?? 0) > 0
            ? MonitoringSnapshot.ratio(gauge("prefill_tokens_live"), gauge("prefill_tokens_expected")) : nil
        return [
            .init(title: "Decode", value: fmt(store.rates.decode), unit: "tok/s",
                detail: L10n.format("%@ ms average · %@", fmt(average("decode_time_seconds")),
                    L10n.text(gauge("generation_tokens_live") == nil ? "completed-token fallback" : "includes in-flight tokens"))),
            .init(title: "Prefill", value: fmt(store.rates.prefill), unit: "tok/s",
                detail: ((gauge("requests_prefilling") ?? 0) > 0 ? L10n.text("Prefilling") + " · " : "")
                    + L10n.format("%@ forwarded · %@ tok/s average · %@ ms average", fmt(gauge("prefill_tokens_live")), fmt(prefillAverage), fmt(average("prefill_time_seconds"))), progress: prefillProgress),
            .init(title: "Running requests", value: fmt(gauge("requests_running")), unit: "",
                detail: L10n.format("%@ waiting · %@ req/s · %@ cancelled", fmt(gauge("requests_waiting")), fmt(store.rates.requests), fmt(counter("requests_cancelled_total")))),
            .init(title: "Time to first token", value: fmt(average("time_to_first_token_seconds")), unit: "ms avg",
                detail: L10n.format("%@ ms end-to-end average · since startup", fmt(average("e2e_request_latency_seconds")))),
            .init(title: "Cache hit rate", value: fmt(MonitoringSnapshot.ratio(counter("prefix_cache_hits_total"), counter("prefix_cache_queries_total"))), unit: "%",
                detail: L10n.format("%@ / %@ queries · %@%% tokens reused", fmt(counter("prefix_cache_hits_total")), fmt(counter("prefix_cache_queries_total")),
                    fmt(MonitoringSnapshot.ratio(counter("prefix_cache_tokens_total"), counter("prompt_tokens_total"))))),
            .init(title: "GPU utilization", value: fmt(gauge("gpu_utilization_pct")), unit: "%",
                detail: L10n.text(gauge("gpu_utilization_pct") == nil ? "Current utilization unavailable" : (gauge("gpu_utilization_pct") ?? 0) >= 90 ? "Critical (≥90%)" : (gauge("gpu_utilization_pct") ?? 0) >= 70 ? "High (≥70%)" : "Current utilization"), progress: gauge("gpu_utilization_pct")),
            .init(title: "Process memory", value: fmt(gauge("memory_mb")), unit: "MB",
                detail: L10n.format("Physical footprint · MLX %@ active · %@ pool", MonitorFormat.bytes(gauge("mlx_active_bytes")), MonitorFormat.bytes(gauge("mlx_cache_bytes")))),
            .init(title: "Generated total", value: fmt(gauge("generation_tokens_live") ?? counter("generation_tokens_total")), unit: "tokens",
                detail: L10n.format("%@ successful requests", fmt(counter("requests_success_total"))))
        ]
    }

    private var sessionsTable: some View {
        let sessions = (snapshot?.sessions ?? []).sorted { a, b in a.model == b.model ? (a.phase != "cached" && b.phase == "cached") : a.model < b.model }
        let rows = sessions.map { session in
            [session.model, MonitorFormat.phase(session.phase), fmt(session.context)
                + ((session.capacity ?? 0) > 0 ? " / \(fmt(session.capacity)) · \(fmt(MonitoringSnapshot.ratio(session.context, session.capacity)))%" : ""),
             fmt(session.cached), session.phase == "cached" ? "—" : fmt(session.generated), MonitorFormat.bytes(session.stateBytes)]
        }
        return MonitoringTable(title: "Sessions", columns: ["Model", "Phase", "Context", "Cached", "Generated", "KV + state"],
            rows: rows, meters: sessions.map { MonitoringSnapshot.ratio($0.context, $0.capacity) }, meterColumn: 2,
            emptyText: snapshot?.sessions == nil ? "Live sessions unavailable" : "No sessions")
    }

    private var modelTotalsTable: some View {
        let totals = store.history.modelTotals(from: from, to: store.now)
        let rows = totals.keys.sorted().map { key in
            [key == "unattributed" ? L10n.text("Unattributed") : key, fmt(totals[key]?["requests_success_total"]),
             fmt(totals[key]?["generation_tokens_total"]), fmt(totals[key]?["prefill_tokens_total"])]
        }
        return MonitoringTable(title: "By model", columns: ["Model", "Requests", "Generated", "Prefill"], rows: rows)
    }

    private var requestTable: some View {
        let rows = store.history.visibleRequests(from: from, to: store.now).map { request in
            [Date(timeIntervalSince1970: request.startTime).formatted(date: .numeric, time: .standard), request.client, request.model,
             request.endTime != nil ? L10n.text("No longer observed") : (store.now - request.lastTime > 2 ? L10n.text("Last seen") + " · " : "") + MonitorFormat.phase(request.phase),
             fmt((request.endTime ?? request.lastTime) - request.startTime) + " s", fmt(request.context), fmt(request.cached), fmt(request.generated)]
        }
        return MonitoringTable(title: "Request history", columns: ["Started", "Client", "Model", "Phase / observation", "Duration", "Context", "Cached", "Generated"], rows: rows)
    }

    private var modelsTable: some View {
        let models = (store.models ?? []).sorted { $0.loaded == $1.loaded ? $0.name < $1.name : $0.loaded }
        let rows = models.map { model in
            [model.name, model.capabilities.map(MonitorFormat.capability).joined(separator: ", "),
             MonitorFormat.modelSize(model),
             MonitorFormat.state(model.state, loaded: model.loaded)]
        }
        return MonitoringTable(title: "Models", columns: ["Model", "Capabilities", "Size", "State"], rows: rows,
            emptyText: store.models == nil ? "Model list unavailable" : "No models discovered")
    }
}

private struct MonitoringMetricCard: View {
    struct Card: Identifiable {
        var title: String
        var value: String
        var unit: String
        var detail: String
        var progress: Double? = nil
        var id: String { title }
    }
    let card: Card
    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            Text(L10n.text(card.title)).font(.app(.body)).foregroundStyle(.secondary)
            HStack(alignment: .firstTextBaseline, spacing: 5) {
                Text(card.value).font(.app(.title)).monospacedDigit()
                Text(L10n.text(card.unit)).font(.app(.caption)).foregroundStyle(.secondary)
            }
            Text(card.detail).font(.app(.caption)).foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
            Spacer(minLength: 0)
            if let progress = card.progress {
                ProgressView(value: min(100, max(0, progress)), total: 100)
                    .tint(MonitorFormat.color(progress)).accessibilityValue("\(MonitorFormat.number(progress))%")
            }
        }
        .frame(maxWidth: .infinity, minHeight: 110, alignment: .topLeading).padding(16)
        .background(.quaternary.opacity(0.3), in: RoundedRectangle(cornerRadius: 12))
        .overlay(RoundedRectangle(cornerRadius: 12).strokeBorder(.quaternary))
    }
}

private struct MonitoringChart: View {
    let title: String
    let points: [MonitoringPoint]
    let from: TimeInterval
    let to: TimeInterval
    @State private var selected: Date?
    private var inspected: MonitoringPoint? {
        guard let selected else { return points.last }
        return points.min { abs($0.date.timeIntervalSince(selected)) < abs($1.date.timeIntervalSince(selected)) }
    }
    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text(title).font(.app(.body))
            Text(inspected.map { $0.date.formatted(date: .omitted, time: .standard) + " · " + MonitorFormat.number($0.value) } ?? "—")
                .font(.app(.caption)).foregroundStyle(.secondary).monospacedDigit()
            Chart {
                ForEach(points) { point in
                    if let value = point.value {
                        LineMark(x: .value(L10n.text("Time"), point.date), y: .value(title, value), series: .value("segment", point.segment))
                            .foregroundStyle(Color.accentColor).lineStyle(StrokeStyle(lineWidth: 2))
                    }
                }
                if let inspected, let value = inspected.value {
                    PointMark(x: .value(L10n.text("Time"), inspected.date), y: .value(title, value)).foregroundStyle(Color.accentColor)
                }
            }
            .chartXScale(domain: Date(timeIntervalSince1970: from)...Date(timeIntervalSince1970: max(from + 1, to)))
            .chartYScale(domain: 0...max(1, (points.compactMap(\.value).max() ?? 0) * 1.1))
            .chartXSelection(value: $selected).frame(height: 130)
            .focusable().onKeyPress(.leftArrow) { inspect(-1); return .handled }
            .onKeyPress(.rightArrow) { inspect(1); return .handled }
            .accessibilityLabel(title).accessibilityValue(MonitorFormat.number(inspected?.value))
        }
        .padding(16).background(.quaternary.opacity(0.3), in: RoundedRectangle(cornerRadius: 12))
        .overlay(RoundedRectangle(cornerRadius: 12).strokeBorder(.quaternary))
    }
    private func inspect(_ direction: Int) {
        guard !points.isEmpty else { return }
        let current = inspected.flatMap { point in points.firstIndex { $0.id == point.id } } ?? (points.count - 1)
        selected = points[min(points.count - 1, max(0, current + direction))].date
    }
}

private struct MonitoringTable: View {
    var title: String
    var columns: [String]
    var rows: [[String]]
    var meters: [Double?] = []
    var meterColumn: Int? = nil
    var emptyText = "No observations in this range"
    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            Text(L10n.text(title)).font(.app(.headline))
            if rows.isEmpty {
                Text(L10n.text(emptyText)).font(.app(.body)).foregroundStyle(.secondary).padding(.vertical, 12)
            } else {
                ScrollView(.horizontal) {
                    Grid(alignment: .leading, horizontalSpacing: 20, verticalSpacing: 12) {
                        GridRow {
                            ForEach(Array(columns.enumerated()), id: \.offset) { _, column in
                                Text(L10n.text(column)).font(.app(.caption)).foregroundStyle(.secondary)
                            }
                        }
                        ForEach(rows.indices, id: \.self) { row in
                            GridRow {
                                ForEach(columns.indices, id: \.self) { column in
                                    VStack(alignment: .leading, spacing: 5) {
                                        Text(rows[row][column]).font(.app(.body)).textSelection(.enabled).lineLimit(2)
                                        if column == meterColumn, meters.indices.contains(row), let percent = meters[row] {
                                            ProgressView(value: min(100, max(0, percent)), total: 100).tint(MonitorFormat.color(percent))
                                        }
                                    }
                                    .frame(minWidth: column == 0 ? 220 : 100, maxWidth: column == 0 ? 320 : 240, alignment: .leading)
                                }
                            }
                        }
                    }.padding(14)
                }
                .overlay(RoundedRectangle(cornerRadius: 10).strokeBorder(.quaternary))
            }
        }
    }
}
