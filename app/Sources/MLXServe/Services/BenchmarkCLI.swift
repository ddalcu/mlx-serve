import Combine
import Foundation

/// `MLXCore bench`: the Benchmarks window's ladder from a shell, against a server
/// that is already up, so a Mac reached over ssh can post the same rows to the
/// board. Nothing leaves the machine without `--share`.
enum BenchmarkCLI {

    struct Options: Equatable {
        var port: UInt16 = 11234
        var model: String?
        var note: String?
        var share = false
    }

    struct UsageError: Error, Equatable {
        let message: String
    }

    static let usage = """
        usage: MLXCore bench [--port N] [--model ID] [--note TEXT] [--share]

        Runs the Benchmarks ladder against the server on 127.0.0.1:N (default 11234)
        and prints one row per rung. --model picks a resident chat model by id;
        without it, the first one, as the window does. --share posts the rows to
        the community board, as the window's Share button does.
        """

    static func parse(_ args: [String]) -> Result<Options, UsageError> {
        var options = Options()
        var i = 0
        while i < args.count {
            let flag = args[i]
            if flag == "--share" {
                options.share = true
                i += 1
                continue
            }
            guard ["--port", "--model", "--note"].contains(flag) else {
                return .failure(.init(message: "unknown argument \(flag)"))
            }
            guard i + 1 < args.count else { return .failure(.init(message: "\(flag) needs a value")) }
            let value = args[i + 1]
            switch flag {
            case "--port":
                guard let port = UInt16(value), port > 0 else {
                    return .failure(.init(message: "--port takes 1-65535, got \(value)"))
                }
                options.port = port
            case "--model": options.model = value
            default: options.note = BenchmarkResult.cleanNote(value)
            }
            i += 2
        }
        return .success(options)
    }

    /// The model a row will name: resident here, serving chat, and not a LAN
    /// peer (whose speed is another Mac's). `ServerManager.residentChatModel`
    /// makes the same choice for the window.
    static func pickModel(_ models: [ModelInfo], requested: String?) -> ModelInfo? {
        let resident = models.filter { $0.servesChat && $0.loaded && $0.lanPeer == nil }
        guard let requested else { return resident.first }
        return resident.first { $0.name == requested }
    }

    static func main(_ args: [String]) -> Never {
        switch parse(args) {
        case .failure(let error):
            warn("\(error.message)\n\n\(usage)")
            exit(2)
        case .success(let options):
            Task { @MainActor in exit(await run(options)) }
            dispatchMain()
        }
    }

    @MainActor
    static func run(_ options: Options) async -> Int32 {
        guard let models = await listModels(port: options.port) else {
            warn("nothing answered on 127.0.0.1:\(options.port)")
            return 1
        }
        guard let model = pickModel(models, requested: options.model) else {
            let what = options.model.map { "\($0) is not a resident chat model" } ?? "no chat model is loaded"
            warn("\(what) on 127.0.0.1:\(options.port)")
            return 1
        }
        let ladder = BenchmarkSuite.ladder
        if case .contextTooSmall(let have, let need) = LadderPreflight.decide(contextLength: model.contextLength, ladder: ladder) {
            warn("\(model.name) serves \(have) tokens of context; the ladder needs \(need)")
            return 1
        }

        let runner = BenchmarkRunner()
        let progress = runner.$phase.removeDuplicates().sink { warn("  \(describe($0))") }
        warn("benchmarking \(model.name) on 127.0.0.1:\(options.port)")
        let rows = await runner.run(ladder: ladder, modelId: model.name, port: options.port, note: options.note)
        progress.cancel()

        rows.forEach { print(line($0)) }
        if let drift = rows.first {
            print("drift: " + BenchmarkDrift.summary(first: drift.decodeTps, last: drift.driftDecodeTps, percent: drift.driftPercent))
        }
        if case .failed(let message) = runner.phase { warn("stopped: \(message)") }
        guard !rows.isEmpty else { return 1 }
        _ = BenchmarkStore.appendLocal(rows)
        return options.share ? await share(rows) : 0
    }

    /// `/v1/models` with a short timeout. `APIClient`'s session waits for
    /// connectivity, so a port with nothing on it would hang the shell instead of failing.
    private static func listModels(port: UInt16) async -> [ModelInfo]? {
        guard let url = URL(string: "http://127.0.0.1:\(port)/v1/models") else { return nil }
        var request = URLRequest(url: url)
        request.timeoutInterval = 10
        guard let (data, _) = try? await URLSession.shared.data(for: request),
              let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let entries = json["data"] as? [[String: Any]] else { return nil }
        return entries.map(APIClient.parseModelInfo)
    }

    private static func share(_ rows: [BenchmarkResult]) async -> Int32 {
        let outcome = await BenchmarkCommunityClient().submit(BenchmarkStore.unsent(rows))
        BenchmarkStore.markShared(outcome.sentIds)
        if let error = outcome.error {
            warn("shared \(outcome.sentIds.count) of \(rows.count) rows, then: \(error.localizedDescription)")
            return 1
        }
        print("shared \(outcome.sentIds.count) rows")
        return 0
    }

    private static func line(_ r: BenchmarkResult) -> String {
        let rung = BenchmarkSuite.title(forTarget: r.effectiveTargetTokens).padding(toLength: 5, withPad: " ", startingAt: 0)
        let figures = String(format: "prefill %8.1f t/s  decode %6.1f t/s  ceiling %6.1f t/s  ttft %7.0f ms  runs %d",
                             r.prefillTps, r.decodeTps, r.ceilingDecodeTps ?? 0, r.ttftMs, r.runs)
        return "\(rung) \(figures)  context \(r.contextUsed == true ? "used" : "ignored")"
    }

    private static func describe(_ phase: BenchmarkRunner.Phase) -> String {
        switch phase {
        case .idle: return "idle"
        case .calibrating: return "calibrating"
        case .warmup(let rung): return "\(rung): warmup"
        case .running(let rung, let run, let of): return "\(rung): run \(run)/\(of)"
        case .drift(let run, let of): return "drift check \(run)/\(of)"
        case .stopping: return "stopping"
        case .cancelled: return "cancelled"
        case .done: return "done"
        case .failed(let message): return "failed: \(message)"
        }
    }

    private static func warn(_ text: String) {
        FileHandle.standardError.write(Data((text + "\n").utf8))
    }
}
