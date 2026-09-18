import Foundation

/// Runs one climb of the context ladder against the server as configured.
///
/// The methodology (corpus, byte fit, cache-bust lead-in) lives in
/// `BenchmarkCorpus`; the settings capture in `BenchmarkSettings`. This type
/// owns the IO and the two live checks that can only be made while a run is
/// happening:
///
///  * a CODING run whose prompt hit the KV prefix cache is DISCARDED — its
///    prefill figure measured a cache lookup;
///  * the COUNTING run on the same archive is kept whatever the cache did —
///    its prefix hit is the point, it measures decode over the same prefix.
@MainActor
final class BenchmarkRunner: ObservableObject {

    enum Phase: Equatable {
        case idle
        case calibrating
        case warmup(rung: String)
        case running(rung: String, run: Int, of: Int)
        /// The first rung again, after the ladder: the drift check.
        case drift(run: Int, of: Int)
        case stopping
        case cancelled
        case done
        case failed(String)
    }

    struct Progress: Equatable {
        var completed: Int
        var total: Int
        var fraction: Double { total > 0 ? Double(completed) / Double(total) : 0 }
    }

    @Published private(set) var phase: Phase = .idle
    @Published private(set) var progress = Progress(completed: 0, total: 0)
    /// Coding runs thrown away because the prompt hit the prefix cache.
    @Published private(set) var discardedRuns = 0
    /// Rungs completed so far, so the charts fill in rung by rung.
    @Published private(set) var completedRungs: [BenchmarkResult] = []

    private let api: APIClient
    init(api: APIClient = APIClient()) { self.api = api }

    private var cancelRequested = false

    /// Stop after the request in flight. Completed rungs are kept; the
    /// drift check is skipped because a cut ladder has no "end".
    func cancel() {
        guard case .stopping = phase else {
            cancelRequested = true
            phase = .stopping
            return
        }
    }

    /// Measure the ladder. Returns one result per completed rung; a coding
    /// request that fails ends the ladder and the completed rungs are kept
    /// with the server's error surfaced through `phase`.
    func run(
        ladder: [BenchmarkSuite],
        modelId: String,
        port: UInt16,
        note: String? = nil,
        hardware: BenchmarkHardware = SystemMetrics.benchmarkHardware()
    ) async -> [BenchmarkResult] {
        let sessionId = UUID().uuidString
        let tag = String(UInt64(Date().timeIntervalSince1970 * 1000), radix: 36)
        var seq = 0

        discardedRuns = 0
        completedRungs = []
        cancelRequested = false
        // 1 calibration + per rung: 1 warmup + runs × (coding + counting),
        // + the first rung's runs again at the end for the drift check.
        progress = Progress(completed: 0,
                            total: 1 + ladder.reduce(0) { $0 + $1.warmups + $1.runs * 2 }
                                + (ladder.first?.runs ?? 0))

        let props = (try? await api.fetchPropsRaw(port: port, model: modelId)) ?? [:]
        let settings = BenchmarkSettings.flatten(props: props)
        let engineVersion = settings["version"] ?? "unknown"

        // One throwaway request in exactly the ladder's shape, generating a
        // single token, so even the first rung is sized against this tokenizer
        // instead of a guess.
        phase = .calibrating
        var fits: [BenchmarkCorpus.LadderFit] = []
        seq += 1
        let calibration = try? await request(
            archive: BenchmarkCorpus.buildCodeContextWithConstant(bytes: BenchmarkCorpus.calibrationFillerBytes),
            tag: tag, seq: seq, instruction: BenchmarkCorpus.codeInstruction,
            maxTokens: 1, model: modelId, port: port)
        if let calibration, calibration.promptTokens > 0 {
            fits.append(.init(bytes: BenchmarkCorpus.calibrationFillerBytes, tokens: calibration.promptTokens))
        }
        progress.completed += 1

        var results: [BenchmarkResult] = []
        var failure: String?
        // The first rung's archive and decode, kept for the drift re-measure.
        var firstArchive: String?
        var firstDecode: Double?

        rungs: for rung in ladder {
            if cancelRequested { break }
            let fillerBytes = BenchmarkCorpus.fillerBytesFor(
                target: rung.targetTokens, fits: fits, fixedChars: BenchmarkCorpus.rungFixedChars)
            let archive = BenchmarkCorpus.buildCodeContextWithConstant(bytes: fillerBytes)

            for _ in 0..<rung.warmups {
                if cancelRequested { break rungs }
                seq += 1
                phase = .warmup(rung: rung.title)
                _ = try? await request(archive: archive, tag: tag, seq: seq,
                                       instruction: BenchmarkCorpus.codeInstruction,
                                       maxTokens: rung.genTokens, model: modelId, port: port)
                progress.completed += 1
            }

            var prefill: [Double] = [], decode: [Double] = [], ttft: [Double] = [], ceiling: [Double] = []
            var promptTokens = 0, completionTokens = 0
            var contextUsed = false

            for run in 0..<rung.runs {
                if cancelRequested { break rungs }
                seq += 1
                phase = .running(rung: rung.title, run: run + 1, of: rung.runs)

                let coding: APIClient.CompletionTimings
                do {
                    coding = try await request(archive: archive, tag: tag, seq: seq,
                                               instruction: BenchmarkCorpus.codeInstruction,
                                               maxTokens: rung.genTokens, model: modelId, port: port,
                                               returnsContent: true)
                } catch {
                    failure = error.localizedDescription
                    break rungs
                }
                progress.completed += 1

                if LadderSample.keep(kind: .coding, promptTokens: coding.promptTokens,
                                     cachedTokens: coding.cachedTokens,
                                     completionTokens: coding.completionTokens) {
                    prefill.append(coding.prefillTps)
                    decode.append(coding.decodeTps)
                    ttft.append(coding.ttftMs)
                    promptTokens = coding.promptTokens
                    completionTokens = coding.completionTokens
                    if BenchmarkCorpus.usedPlantedConstant(coding.content) { contextUsed = true }
                } else {
                    discardedRuns += 1
                }

                // Same archive, same tag and seq: a prefix hit by design.
                if let counting = try? await request(archive: archive, tag: tag, seq: seq,
                                                     instruction: BenchmarkCorpus.countInstruction,
                                                     maxTokens: rung.genTokens, model: modelId, port: port),
                   counting.decodeTps > 0,
                   LadderSample.keep(kind: .counting, promptTokens: counting.promptTokens,
                                     cachedTokens: counting.cachedTokens,
                                     completionTokens: counting.completionTokens) {
                    ceiling.append(counting.decodeTps)
                }
                progress.completed += 1
            }

            if promptTokens > 0 { fits.append(.init(bytes: fillerBytes, tokens: promptTokens)) }

            let row = BenchmarkResult(
                sessionId: sessionId,
                suiteId: rung.id,
                modelId: modelId,
                engineVersion: engineVersion,
                prefillTps: BenchmarkStats.median(prefill),
                decodeTps: BenchmarkStats.median(decode),
                ttftMs: BenchmarkStats.median(ttft),
                promptTokens: promptTokens,
                completionTokens: completionTokens,
                runs: decode.count,
                spreadPercent: BenchmarkStats.spreadPercent(decode),
                hardware: hardware,
                targetTokens: rung.targetTokens,
                ceilingDecodeTps: BenchmarkStats.median(ceiling),
                contextUsed: contextUsed,
                settings: settings,
                note: note
            )
            // A rung whose every coding run was discarded measured nothing.
            if row.isPublishable {
                results.append(row)
                completedRungs = results
                if firstArchive == nil { firstArchive = archive; firstDecode = row.decodeTps }
            }
        }

        // Drift: the smallest rung's coding scenario once more, minutes of
        // sustained load later, same bytes, fresh cache-bust tag. Only after
        // a ladder that ran to the end — a failed climb has no "end".
        if failure == nil, !cancelRequested, let firstArchive, let first = ladder.first {
            var again: [Double] = []
            for run in 0..<first.runs {
                seq += 1
                phase = .drift(run: run + 1, of: first.runs)
                if let t = try? await request(archive: firstArchive, tag: tag, seq: seq,
                                              instruction: BenchmarkCorpus.codeInstruction,
                                              maxTokens: first.genTokens, model: modelId, port: port),
                   LadderSample.keep(kind: .coding, promptTokens: t.promptTokens,
                                     cachedTokens: t.cachedTokens, completionTokens: t.completionTokens) {
                    again.append(t.decodeTps)
                }
                progress.completed += 1
            }
            let last = again.isEmpty ? nil : BenchmarkStats.median(again)
            if let percent = BenchmarkDrift.percent(first: firstDecode, last: last) {
                for i in results.indices {
                    results[i].driftDecodeTps = last
                    results[i].driftPercent = percent
                }
                completedRungs = results
            }
        }

        phase = failure.map { .failed($0) } ?? (cancelRequested ? .cancelled : .done)
        return results
    }

    private func request(
        archive: String, tag: String, seq: Int, instruction: String,
        maxTokens: Int, model: String, port: UInt16, returnsContent: Bool = false
    ) async throws -> APIClient.CompletionTimings {
        try await api.benchmarkCompletion(
            port: port,
            model: model,
            prompt: BenchmarkCorpus.prompt(archive: archive, tag: tag, seq: seq, instruction: instruction),
            maxTokens: maxTokens,
            returnsContent: returnsContent
        )
    }
}
