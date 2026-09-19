import Foundation

/// The context-ladder workload: a port of llmprobe's `src/bench/corpus.ts`
/// and the ladder constants of its `src/bench/index.ts` (Apache-2.0).
///
/// The filler is synthetic TypeScript rather than prose: varied identifiers,
/// imports, types and comments — the attention pattern of a real agent's
/// context window. Summarising one sentence stamped out N times is highly
/// predictable output, and a rung baseline built that way already collects a
/// speculation boost. Deterministic, so a rung is reproducible and the
/// byte-to-token fit stays stable across runs.
///
/// The corpus is pure ASCII, so the TypeScript's UTF-16 `.slice` and this
/// file's byte slicing agree. `BenchmarkCorpusTests` pins a SHA-256 of the
/// 4096-byte output against the value node computed from the TypeScript.
enum BenchmarkCorpus {

    // MARK: - Constants (llmprobe index.ts)

    /// Long enough for the decode rate to mean something: a speculator's
    /// acceptance pattern is a distribution, and 64 tokens is too short a
    /// sample to read one out of.
    static let contextGenTokens = 192
    /// Opening guess only; every rung re-fits this against reported usage.
    static let bytesPerToken = 4.0
    /// How far the fitted filler may stray from the flat guess, per token.
    static let minFitBytesPerToken = 2.0
    static let maxFitBytesPerToken = 8.0
    /// Roughly what a chat template costs on top of the message text.
    static let templateTokens = 10
    /// Filler for the one throwaway request that calibrates the ladder.
    static let calibrationFillerBytes = 2048

    static let plantedConstant = "RETRY_BUDGET_MS"
    static let plantedValue = 7413

    static let plantedModule = """
        // src/config/retry-budget.ts
        /** Total wall clock a retry loop may spend before giving up. */
        export const \(plantedConstant) = \(plantedValue);


        """

    /// What a rung asks for, over the synthetic codebase. The task has to reach
    /// into the context (the planted constant), so this measures long-context
    /// work rather than generation with a large irrelevant prefix attached.
    static let codeInstruction =
        "Implement and export `withRetryBudget<T>(fn: () => Promise<T>): Promise<T>` " +
        "for the codebase above. Retry with exponential backoff, give up once " +
        "\(plantedConstant) has elapsed, and follow the conventions of the " +
        "surrounding modules. Reply with only the TypeScript."

    /// Maximally predictable output the prompt does not contain — the ceiling
    /// on what speculation can do at this context length.
    static let countInstruction =
        "Ignore the code above. Count from 1 to 200, one number per line, with no other output."

    /// Everything in a rung prompt that is not filler: the cache-bust tag, the
    /// planted module, the instruction and their separators.
    static let rungFixedChars = 120 + codeInstruction.count + 30

    // MARK: - Corpus (llmprobe corpus.ts)

    private static let domains = [
        "order", "invoice", "shipment", "ledger", "session",
        "device", "channel", "tariff", "manifest", "quota",
    ]

    private static let verbs = [
        "resolve", "normalise", "validate", "reconcile", "expand", "prune", "merge", "index",
    ]

    private static func capitalise(_ s: String) -> String {
        s.prefix(1).uppercased() + s.dropFirst()
    }

    private static func article(_ word: String) -> String {
        "aeiou".contains(word.lowercased().prefix(1)) ? "an" : "a"
    }

    /// Four shapes in rotation, verbatim from the TypeScript.
    static func module(_ i: Int) -> String {
        let domain = domains[i % domains.count]
        let verb = verbs[(i / domains.count) % verbs.count]
        let name = capitalise(domain)
        let fn = "\(verb)\(name)"

        switch i % 4 {
        case 0:
            return """
                // src/\(domain)/\(verb).ts
                import { Clock } from "../runtime/clock";
                import type { \(name)Record } from "./types";

                /** \(capitalise(verb)) \(article(domain)) \(domain) against the ledger, dropping expired entries. */
                export function \(fn)(input: \(name)Record, clock: Clock): \(name)Record | null {
                  const cutoff = clock.now() - input.windowMs;
                  if (input.updatedAt < cutoff) return null;
                  return { ...input, \(verb)dAt: clock.now(), revision: input.revision + 1 };
                }


                """
        case 1:
            return """
                // src/\(domain)/types.ts
                export interface \(name)Record {
                  id: string;
                  revision: number;
                  updatedAt: number;
                  windowMs: number;
                  tags: readonly string[];
                }

                export const EMPTY_\(domain.uppercased()): \(name)Record = {
                  id: "",
                  revision: 0,
                  updatedAt: 0,
                  windowMs: \(1000 + i * 7),
                  tags: [],
                };


                """
        case 2:
            return """
                // src/\(domain)/\(verb)-batch.ts
                import { \(fn) } from "./\(verb)";
                import type { \(name)Record } from "./types";

                /**
                 * Batch form of `\(fn)`. Returns only the records that survived, so callers
                 * can compare lengths rather than scanning for nulls.
                 */
                export async function \(fn)Batch(
                  records: readonly \(name)Record[],
                  concurrency = \(2 + (i % 6)),
                ): Promise<\(name)Record[]> {
                  const out: \(name)Record[] = [];
                  for (let i = 0; i < records.length; i += concurrency) {
                    const slice = records.slice(i, i + concurrency);
                    out.push(...slice.filter((r) => r.revision >= 0));
                  }
                  return out;
                }


                """
        default:
            return """
                // src/\(domain)/\(verb).test.ts
                import { describe, expect, test } from "vitest";
                import { \(fn) } from "./\(verb)";

                describe("\(fn)", () => {
                  test("drops a record older than its window", () => {
                    const clock = { now: () => \(10_000 + i * 13) };
                    const stale = { id: "\(domain)-\(i)", revision: 1, updatedAt: 0, windowMs: 5, tags: [] };
                    expect(\(fn)(stale, clock)).toBeNull();
                  });
                });


                """
        }
    }

    /// Deterministic TypeScript source of exactly `bytes` bytes.
    static func buildCodeContext(bytes: Int) -> String {
        guard bytes > 0 else { return "" }
        var out = ""
        var i = 0
        while out.utf8.count < bytes {
            out += module(i)
            i += 1
        }
        return String(decoding: out.utf8.prefix(bytes), as: UTF8.self)
    }

    /// `bytes` of source with the config module buried past the midpoint,
    /// on a module boundary — the blank line between files. A bare newline
    /// lands inside a JSDoc block often enough that the constant ends up
    /// commented out.
    static func buildCodeContextWithConstant(bytes: Int) -> String {
        let filler = buildCodeContext(bytes: bytes)
        let half = filler.utf8.count / 2
        let start = filler.utf8.index(filler.utf8.startIndex, offsetBy: half)
        let cut: String.Index
        if let gap = filler.range(of: "\n\n", range: start..<filler.endIndex) {
            cut = gap.upperBound
        } else {
            cut = filler.endIndex
        }
        return String(filler[..<cut]) + plantedModule + String(filler[cut...])
    }

    /// Did the answer actually use the value it had to go and find?
    static func usedPlantedConstant(_ text: String) -> Bool {
        text.contains(plantedConstant) || text.contains(String(plantedValue))
    }

    // MARK: - Filler fit (llmprobe index.ts)

    /// One (filler bytes → tokens the engine counted) observation.
    struct LadderFit: Equatable {
        var bytes: Int
        var tokens: Int
    }

    /// How much filler to write to land on `target` input tokens.
    ///
    /// Two observations give a straight line through the most recent pair;
    /// they carry identical non-filler overhead, so the intercept absorbs it
    /// exactly. One observation cannot separate slope from intercept, so it
    /// subtracts the overhead it knows about (`fixedChars` + the template).
    static func fillerBytesFor(target: Int, fits: [LadderFit], fixedChars: Int) -> Int {
        let t = Double(target)
        func clamp(_ bytes: Double) -> Int {
            Int(min(t * maxFitBytesPerToken, max(t * minFitBytesPerToken, bytes)).rounded())
        }

        let usable = fits.filter { $0.bytes > 0 && $0.tokens > 0 }
        guard let last = usable.last else { return clamp(t * bytesPerToken) }
        let prev = usable.count >= 2 ? usable[usable.count - 2] : nil

        if prev == nil || prev!.tokens == last.tokens {
            let fillerTokens = last.tokens - templateTokens
            if fillerTokens <= 0 { return clamp(t * bytesPerToken) }
            let perToken = Double(last.bytes + fixedChars) / Double(fillerTokens)
            return clamp((t - Double(templateTokens)) * perToken - Double(fixedChars))
        }

        let slope = Double(last.bytes - prev!.bytes) / Double(last.tokens - prev!.tokens)
        return clamp(Double(last.bytes) + slope * (t - Double(last.tokens)))
    }

    // MARK: - Prompt assembly

    /// The cache-bust lead-in sits at the very FRONT of the prompt: prefix
    /// caches match from token 0. Unique per session (`tag`) and per measured
    /// run (`seq`); the coding and counting requests of ONE run share both, so
    /// the counting run's prefix hit is the decode-only measurement it is
    /// meant to be.
    static func prompt(archive: String, tag: String, seq: Int, instruction: String) -> String {
        "[probe \(tag)-\(seq)] \(archive)\n\n\(instruction)"
    }
}
