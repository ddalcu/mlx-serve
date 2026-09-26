import SwiftUI

/// The Laya Decisions window: shows the pane for `appState.decisionsModelPath`.
struct LayaDecisionsWindow: View {
    @EnvironmentObject var appState: AppState
    var body: some View {
        if let path = appState.decisionsModelPath {
            LayaDecisionsPane(modelPath: path).id(path)
        } else {
            Text("Pick a Laya model in Models \u{2192} Downloaded and press Use.")
                .font(.app(.callout))
                .padding(40)
        }
    }
}

/// Demo page for a Laya typed-decision model: a state, a few questions, one
/// `POST /v1/decisions`, the answers as probability bars. Opened by the Use
/// button on a `laya` row; loads the model the way the gen panes do.
struct LayaDecisionsPane: View {
    let modelPath: String
    @EnvironmentObject var server: ServerManager

    @State private var state = "Refund me now or I cancel my subscription. Second time this month your app charged me twice."
    @State private var questions: [Question] = [
        Question(name: "team", type: .choice, instructions: "Which team should handle this?", criteria: "billing, sales, support"),
        Question(name: "churn", type: .noul, instructions: "Does the customer threaten to cancel?", criteria: "no threat, explicit threat"),
        Question(name: "urgency", type: .score, instructions: "How urgent is this?", criteria: "not urgent, soon, blocking"),
    ]
    @State private var answers: [(name: String, summary: String, bars: [(String, Double)], confidence: Double?)] = []
    @State private var rawResponse = ""
    @State private var error: String?
    @State private var latencyMs: Double?
    @State private var busy = false
    @State private var showRaw = false

    struct Question: Identifiable {
        let id = UUID()
        var name: String
        var type: Kind
        var instructions: String
        /// choice: comma list; noul: optional "false label, true label"; score: rung labels low to high.
        var criteria: String

        enum Kind: String, CaseIterable { case choice, noul, score }

        var json: [String: Any]? {
            let parts = criteria.split(separator: ",").map { $0.trimmingCharacters(in: .whitespaces) }.filter { !$0.isEmpty }
            var q: [String: Any] = ["type": type.rawValue, "instructions": instructions]
            switch type {
            case .choice:
                guard parts.count >= 2 else { return nil }
                q["criteria"] = parts
            case .noul:
                guard parts.isEmpty || parts.count == 2 else { return nil }
                if parts.count == 2 { q["criteria"] = ["false": parts[0], "true": parts[1]] }
            case .score:
                guard parts.count >= 2 else { return nil }
                q["criteria"] = parts
            }
            return q
        }
    }

    private var requestBody: [String: Any] {
        var qs: [String: Any] = [:]
        for q in questions where !q.name.isEmpty { if let j = q.json { qs[q.name] = j } }
        return ["model": (modelPath as NSString).lastPathComponent, "state": state, "questions": qs]
    }

    private var requestJSON: String {
        (try? JSONSerialization.data(withJSONObject: requestBody, options: [.prettyPrinted, .sortedKeys]))
            .flatMap { String(data: $0, encoding: .utf8) } ?? "{}"
    }

    var body: some View {
        HSplitView {
            ScrollView {
                VStack(alignment: .leading, spacing: 16) {
                    Text("Laya Decisions").font(.app(.title2).bold())
                    Text((modelPath as NSString).lastPathComponent).font(.app(.caption)).foregroundStyle(.secondary)

                    Text("State").font(.app(.headline))
                    TextEditor(text: $state).font(.app(.body)).frame(minHeight: 80)
                        .overlay(RoundedRectangle(cornerRadius: 6).stroke(.quaternary))

                    HStack {
                        Text("Questions").font(.app(.headline))
                        Spacer()
                        Button { questions.append(Question(name: "q\(questions.count + 1)", type: .choice, instructions: "", criteria: "yes, no")) } label: { Image(systemName: "plus") }
                            .buttonStyle(.plain)
                    }
                    ForEach($questions) { $q in
                        VStack(alignment: .leading, spacing: 6) {
                            HStack {
                                TextField("name", text: $q.name).frame(width: 120).font(.app(.body))
                                Picker("", selection: $q.type) {
                                    ForEach(Question.Kind.allCases, id: \.self) { Text($0.rawValue) }.font(.app(.body))
                                }.frame(width: 100)
                                Spacer()
                                Button { questions.removeAll { $0.id == q.id } } label: { Image(systemName: "minus.circle") }
                                    .buttonStyle(.plain).foregroundStyle(.secondary).font(.app(.body))
                            }
                            TextField("instructions", text: $q.instructions).font(.app(.body))
                            TextField(criteriaHint(q.type), text: $q.criteria)
                                .foregroundStyle(q.json == nil ? .red : .primary).font(.app(.body))
                        }
                        .padding(8)
                        .background(RoundedRectangle(cornerRadius: 6).fill(.quaternary.opacity(0.4)))
                    }

                    HStack {
                        Button(busy ? "Asking…" : "Ask") { Task { await ask() } }
                            .keyboardShortcut(.return, modifiers: .command)
                            .disabled(busy || questions.allSatisfy { $0.json == nil })
                        if let latencyMs { Text(String(format: "%.0f ms", latencyMs)).font(.app(.body)).foregroundStyle(.secondary) }
                        Spacer()
                        Toggle("Raw JSON", isOn: $showRaw).toggleStyle(.checkbox).font(.app(.body))
                    }

                    if let error { Text(error).foregroundStyle(.red).textSelection(.enabled) }

                    ForEach(answers, id: \.name) { a in
                        VStack(alignment: .leading, spacing: 4) {
                            HStack {
                                Text(a.name).font(.app(.headline))
                                Text(a.summary).foregroundStyle(.secondary)
                                if let c = a.confidence { Text(String(format: "confidence %.2f", c)).font(.app(.caption)).foregroundStyle(.tertiary) }
                            }
                            ForEach(a.bars, id: \.0) { label, p in
                                HStack {
                                    Text(label).frame(width: 110, alignment: .trailing).font(.app(.caption))
                                    GeometryReader { g in
                                        RoundedRectangle(cornerRadius: 3).fill(.tint).frame(width: max(2, g.size.width * p))
                                    }.frame(height: 10)
                                    Text(String(format: "%.1f%%", p * 100)).font(.app(.caption).monospacedDigit()).frame(width: 50, alignment: .trailing)
                                }
                            }
                        }
                        .padding(8)
                        .background(RoundedRectangle(cornerRadius: 6).fill(.quaternary.opacity(0.4)))
                    }

                    if showRaw {
                        Text("Request").font(.app(.headline))
                        codeBlock(requestJSON)
                        if !rawResponse.isEmpty {
                            Text("Response").font(.app(.headline))
                            codeBlock(rawResponse)
                        }
                    }
                }
                .padding(20)
                .frame(maxWidth: .infinity, alignment: .leading)
            }
            .frame(minWidth: 420)

            ScrollView { docs.padding(20) }
                .frame(minWidth: 280, idealWidth: 340, maxWidth: 420)
        }
    }

    private func criteriaHint(_ k: Question.Kind) -> String {
        switch k {
        case .choice: return "criteria: option, option, …"
        case .noul: return "criteria (optional): false label, true label"
        case .score: return "criteria: rung labels, low to high"
        }
    }

    private func codeBlock(_ s: String) -> some View {
        Text(s).font(.app(.caption, design: .monospaced)).textSelection(.enabled)
            .frame(maxWidth: .infinity, alignment: .leading).padding(8)
            .background(RoundedRectangle(cornerRadius: 6).fill(.quaternary.opacity(0.4)))
    }

    private var docs: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("What this is").font(.app(.headline))
            Text("Laya is not a chat model. It reads a piece of text (the state) and answers typed questions about it in one forward pass, with calibrated probabilities. A few milliseconds per request, so it suits routing, triage, moderation and scoring.")
            Text("Question types").font(.app(.headline))
            Text("**choice** picks one of your options.\n`\"criteria\": [\"billing\", \"sales\"]`")
            Text("**noul** is a yes/no; the answer is P(true). Criteria are optional labels for each side.\n`\"criteria\": {\"false\": \"no threat\", \"true\": \"explicit threat\"}`")
            Text("**score** is an ordinal over labelled rungs, low to high; the answer is the expected rung index plus per-rung probabilities.\n`\"criteria\": [\"not urgent\", \"soon\", \"blocking\"]`")
            Text("Every question needs `instructions`. The state can be a string or a JSON object.")
            Text("API").font(.app(.headline))
            Text("`POST /v1/decisions` with `model`, `state` and `questions`. Chat endpoints refuse this model and point here.")
            codeBlock("curl -X POST http://localhost:\(server.port)/v1/decisions \\\n  -H 'content-type: application/json' \\\n  -d '\(requestJSON.replacingOccurrences(of: "\n", with: "").replacingOccurrences(of: "  ", with: ""))'")
            Text("Answers carry the chosen value, per-option `probabilities`, a `confidence` and an `action.act_probability` (how sure the model is that acting on the answer is right).")
        }
        .font(.app(.callout))
        .frame(maxWidth: .infinity, alignment: .leading)
    }

    private func ask() async {
        busy = true; error = nil; answers = []; latencyMs = nil
        defer { busy = false }
        do {
            let dir = modelPath
            let port = try await server.ensureRunning(forGenModelDir: dir)
            let info = try await server.loadModel(id: dir)
            var body = requestBody
            body["model"] = info.name
            var req = URLRequest(url: URL(string: "http://localhost:\(port)/v1/decisions")!)
            req.httpMethod = "POST"
            req.setValue("application/json", forHTTPHeaderField: "Content-Type")
            req.httpBody = try JSONSerialization.data(withJSONObject: body)
            let t0 = Date()
            let (data, resp) = try await URLSession.shared.data(for: req)
            latencyMs = Date().timeIntervalSince(t0) * 1000
            rawResponse = (try? JSONSerialization.jsonObject(with: data)).flatMap {
                try? JSONSerialization.data(withJSONObject: $0, options: [.prettyPrinted, .sortedKeys])
            }.flatMap { String(data: $0, encoding: .utf8) } ?? String(decoding: data, as: UTF8.self)
            guard (resp as? HTTPURLResponse)?.statusCode == 200 else {
                error = rawResponse; return
            }
            answers = parse(data)
        } catch {
            self.error = error.localizedDescription
        }
    }

    private func parse(_ data: Data) -> [(name: String, summary: String, bars: [(String, Double)], confidence: Double?)] {
        guard let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let ans = obj["answers"] as? [String: [String: Any]] else { return [] }
        return ans.keys.sorted().compactMap { name in
            let a = ans[name]!
            let conf = a["confidence"] as? Double
            switch a["type"] as? String {
            case "choice":
                let probs = (a["probabilities"] as? [String: Double]) ?? [:]
                return (name, "→ \(a["choice"] as? String ?? "?")", probs.sorted { $0.value > $1.value }.map { ($0.key, $0.value) }, conf)
            case "noul":
                let p = a["noul"] as? Double ?? 0
                return (name, p >= 0.5 ? "→ true" : "→ false", [("true", p), ("false", 1 - p)], conf)
            case "score":
                let probs = (a["probabilities"] as? [String: Double]) ?? [:]
                let legend = (a["legend"] as? [String: String]) ?? [:]
                let bars = probs.sorted { (Int($0.key) ?? 0) < (Int($1.key) ?? 0) }.map { (legend[$0.key] ?? $0.key, $0.value) }
                let val = (a["score"] as? Double).map { String(format: "%.2f", $0) } ?? "?"
                return (name, "→ score \(val) of 0…\(max(0, probs.count - 1))", bars, conf)
            default:
                return (name, "\(a)", [], conf)
            }
        }
    }
}
