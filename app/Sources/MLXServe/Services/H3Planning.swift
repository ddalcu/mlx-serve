import Foundation
import IOKit

/// MiniMax-H3's memory and time models.
///
/// H3 is the first backend where the option a user picks changes the answer by
/// an ORDER of magnitude — 124 frames at 960x544 is a 20-minute job and 362 at
/// 1344x768 is a three-hour one on the same Mac — so "generate and find out" is
/// not an acceptable interface. Everything here is derived from the engine's
/// own row arithmetic (`minimax_h3.zig`) and calibrated against measured runs;
/// `H3PlanningTests` pins it to those measurements.
///
/// Both models are estimates and are labelled as such in the UI. The memory one
/// errs HIGH on purpose: an over-estimate costs a warning the user can ignore,
/// an under-estimate costs an uncatchable Metal OOM.
enum H3Plan {

    // The DiT's own geometry, from the pack's config.json. Not tunable.
    static let hiddenSize = 5376
    static let blocks = 50
    /// Latent pixels per patch axis: the VAE compresses 16x and the DiT
    /// patchifies 2x2 on top, so one row covers a 32x32 pixel tile.
    static let pixelsPerRowAxis = 32
    static let fps = 24
    /// Audio latents run at 40 Hz and the stream is STEREO — two rows each.
    static let audioLatentFps = 40
    static let audioChannels = 2

    /// `videoLatentT` in the engine: 17 source frames fold into 5 latent ones,
    /// which is why the frame ladder is 17k+5.
    static func latentT(frames: Int) -> Int {
        if frames <= 5 { return 2 }
        return ((frames - 5) / 17) * 5 + 2
    }

    /// Rows in the packed `[text | cond | audio | video]` sequence — the one
    /// number both models are built on. `promptTokens` and `keyframes` are the
    /// small terms; they matter at 256px, where they are most of the sequence.
    static func rows(width: Int, height: Int, frames: Int, promptTokens: Int, keyframes: Int = 0) -> Int {
        let frameRows = (width / pixelsPerRowAxis) * (height / pixelsPerRowAxis)
        let video = latentT(frames: frames) * frameRows
        let audioT = Int((Double(frames) / Double(fps) * Double(audioLatentFps)).rounded())
        return video + audioT * audioChannels + promptTokens + keyframes * frameRows
    }

    /// The fast recipe's attention-broadcast cache: one `[S, hidden]` bf16 per
    /// block, held for the whole run. At long frame counts THIS is what runs a
    /// Mac out of memory, not the weights — 58 GB at 1344x768 x 362 frames.
    /// Linear in rows: the [S, S] score matrix is never materialized (SDPA is
    /// fused), which is the only reason a 108k-row sequence is possible at all.
    static func pabCacheBytes(rows: Int) -> UInt64 {
        UInt64(rows) * UInt64(hiddenSize) * 2 * UInt64(blocks)
    }

    /// Per-step transients (qkv, attention output, the 14336-wide MLP
    /// intermediate, the residual stream). Estimated from the tensor widths
    /// rather than measured, and rounded UP for the reason in the type comment.
    static let activationBytesPerRow: UInt64 = 96 * 1024

    /// The Turbo LoRA rides resident beside the DiT when engaged (744 MB bf16
    /// on disk, rounded up — the server bills the real file the same way).
    static let turboLoraBytes: UInt64 = 800 * 1024 * 1024

    /// Peak unified memory for one generation, in bytes.
    ///
    /// During sampling the text encoder is already freed, so the resident set is
    /// the DiT plus the cache plus activations. The staged LOAD peak
    /// (`max(TE, DiT) + VAEs`, the preset's `approxRAMGB`) is the floor — a run
    /// that samples cheaply still had to get the weights in.
    static func peakBytes(model: VideoModelPreset, width: Int, height: Int,
                          frames: Int, fast: Bool, turbo: Bool = false,
                          promptTokens: Int = 250) -> UInt64 {
        let gb: UInt64 = 1024 * 1024 * 1024
        let stagedGB = model.stagedPeakGB > 0 ? model.stagedPeakGB : Double(model.approxRAMGB)
        let loadPeak = UInt64(stagedGB * Double(gb))
        guard model.backend == .minimaxH3 else { return UInt64(model.approxRAMGB) * gb }
        let r = rows(width: width, height: height, frames: frames, promptTokens: promptTokens)
        // Turbo forces the fast recipe off server-side, so its callers pass
        // fast=false and the broadcast cache never bills; the LoRA itself
        // rides beside the DiT.
        let sampling = UInt64(Double(model.ditResidentGB) * Double(gb))
            + (fast ? pabCacheBytes(rows: r) : 0)
            + (turbo ? turboLoraBytes : 0)
            + UInt64(r) * activationBytesPerRow
        return max(loadPeak, sampling)
    }

    /// How much of a Mac's memory a generation may plan to use. 0.80 is not a
    /// guess: on the 48 GB machine in #126 the MLX wired limit came out at
    /// 38338 MB, i.e. 78% of total, and that limit is the real ceiling.
    static let usableFraction = 0.80

    static func budgetBytes(availableGB: Int) -> UInt64 {
        UInt64(Double(availableGB) * usableFraction * 1024 * 1024 * 1024)
    }

    /// Whether ONE specific configuration fits. Distinct from `frameCap` on
    /// purpose: the cap has a floor (below the ladder's start the model is
    /// off-distribution, so a smaller number is not a usable answer), which
    /// means "cap == floor" is ambiguous between "124 frames fits" and
    /// "nothing fits". A Mac too small to load the pack at all would otherwise
    /// see no warning at exactly 124 frames.
    static func fits(model: VideoModelPreset, width: Int, height: Int,
                     frames: Int, fast: Bool, turbo: Bool = false, availableGB: Int) -> Bool {
        peakBytes(model: model, width: width, height: height, frames: frames, fast: fast, turbo: turbo)
            <= budgetBytes(availableGB: availableGB)
    }

    /// Longest clip that fits `availableGB`, snapped DOWN to the model's own
    /// frame ladder. Never returns less than the ladder floor — ask `fits`
    /// whether that floor is itself reachable.
    static func frameCap(model: VideoModelPreset, width: Int, height: Int,
                         availableGB: Int, fast: Bool = true) -> Int {
        let budget = budgetBytes(availableGB: availableGB)
        var best = model.frameOptions.first ?? 124
        for n in model.frameOptions {
            if peakBytes(model: model, width: width, height: height, frames: n, fast: fast) <= budget { best = n }
        }
        return best
    }
}

// MARK: - Reference limits

/// ref2va's file limits, mirroring `minimax_h3.zig`'s `MAX_REF_*`.
///
/// The three per-type caps sum to 15 and MiniMax's real limit is 12 files
/// ACROSS all of them, so a set can clear every per-type cap and still be
/// refused. Building that set in the picker and finding out at generate time is
/// the failure this exists to prevent.
enum H3RefLimits {
    static let images = 9
    static let videos = 3
    static let audios = 3
    static let total = 12

    /// How many more files of one type may be attached. The tighter of that
    /// type's own cap and what is left of the combined budget.
    static func remaining(perType: Int, current: Int, totalAttached: Int) -> Int {
        max(0, min(perType - current, total - totalAttached))
    }

    /// Shown only once the combined cap actually binds — before that it is
    /// noise, and after it an empty list with no Add button is unexplained.
    static func totalNote(attached: Int) -> String? {
        guard attached >= total else { return nil }
        return "\(total) of \(total) reference files — the model takes at most \(total) across images, clips and audio."
    }
}

// MARK: - Hardware

/// What the time model needs to know about this Mac. A struct rather than a
/// bare number so a test can name a machine it is not running on.
struct H3Hardware: Equatable {
    /// GPU cores. The one figure that tracks H3 throughput: the DiT step is at
    /// the compute roofline (SDPA and the linears both measured ~13 TFLOPS
    /// against a ~15 ceiling), so it is not bandwidth- or memory-bound.
    let gpuCores: Int
    /// For display ("estimated for M4 Max"). Empty when we could not name it.
    let label: String

    /// The machine the anchor measurements were taken on.
    static let anchor = H3Hardware(gpuCores: 40, label: "M4 Max")

    /// Scan: GPU core count from the IORegistry, chip name from sysctl. Falls
    /// back to the anchor's core count rather than to zero — an estimate scaled
    /// by an unknown is worse than an estimate that says which Mac it assumed.
    static let current: H3Hardware = {
        H3Hardware(gpuCores: scanGpuCores() ?? anchor.gpuCores, label: scanChipName())
    }()

    private static func scanGpuCores() -> Int? {
        var iter: io_iterator_t = 0
        guard let matching = IOServiceMatching("AGXAccelerator") else { return nil }
        guard IOServiceGetMatchingServices(kIOMainPortDefault, matching, &iter) == KERN_SUCCESS else { return nil }
        defer { IOObjectRelease(iter) }
        let entry = IOIteratorNext(iter)
        guard entry != 0 else { return nil }
        defer { IOObjectRelease(entry) }
        // The count lives on the accelerator or on its parent depending on the
        // SoC generation, so walk up once rather than trusting one level.
        for target in [entry, parentOf(entry)] where target != 0 {
            defer { if target != entry { IOObjectRelease(target) } }
            if let ref = IORegistryEntryCreateCFProperty(target, "gpu-core-count" as CFString,
                                                        kCFAllocatorDefault, 0)?.takeRetainedValue(),
               let n = (ref as? NSNumber)?.intValue, n > 0 {
                return n
            }
        }
        return nil
    }

    private static func parentOf(_ entry: io_registry_entry_t) -> io_registry_entry_t {
        var parent: io_registry_entry_t = 0
        guard IORegistryEntryGetParentEntry(entry, kIOServicePlane, &parent) == KERN_SUCCESS else { return 0 }
        return parent
    }

    private static func scanChipName() -> String {
        var size = 0
        guard sysctlbyname("machdep.cpu.brand_string", nil, &size, nil, 0) == 0, size > 0 else { return "" }
        var buf = [CChar](repeating: 0, count: size)
        guard sysctlbyname("machdep.cpu.brand_string", &buf, &size, nil, 0) == 0 else { return "" }
        // "Apple M4 Max" -> "M4 Max"
        let raw = String(cString: buf).trimmingCharacters(in: .whitespaces)
        return raw.hasPrefix("Apple ") ? String(raw.dropFirst(6)) : raw
    }
}

// MARK: - Time

/// Where an estimate came from. A number measured on this Mac and a number
/// extrapolated from someone else's are different claims and must read
/// differently — a bare "~3 h" on a job this long reads as a promise.
enum H3EstimateSource: Equatable {
    case hardwareModel(H3Hardware)
    case ownHistory(runs: Int)
}

enum H3TimeEstimate {

    // ── The per-step cost curve, fitted on M4 Max dense (fast recipe OFF)
    // measurements at two geometries: 36.6 s/step at 864x480 x 73f (S=9,404,
    // the ablation ladder's dq-gemm baseline) and 275.7 s/step at 1344x768 x
    // 124f (S=37,960, the acceptance run's flat cadence).
    //
    // The square term is the full attention over the packed sequence and it is
    // NOT optional: at a matched 124 frames, 1344x768 costs 2.9x the time of
    // 960x544 for 1.98x the pixels. A linear-in-pixels model under-promises the
    // wide canvas by about half.
    private static let stepLinear = 0.0027817
    private static let stepQuadratic = 1.1806e-7

    /// Fixed stages, from the profile: weights ~36 s cold, text encode ~15 s
    /// cold, audio decode under a second.
    private static let fixedSeconds = 55.0
    /// Tiled VAE decode: 2.4 min at 1344x768 x 124 frames, scaled by pixels.
    private static let vaeAnchorSeconds = 144.0
    private static let vaeAnchorPixels = 1344.0 * 768.0 * 124.0

    /// Dense seconds for one denoising step at `rows`, on the anchor machine.
    static func denseStepSeconds(rows: Int) -> Double {
        let s = Double(rows)
        return stepLinear * s + stepQuadratic * s * s
    }

    /// What the fast recipe costs as a fraction of the dense schedule.
    ///
    /// Mirrors the engine's own broadcast schedule (`attnBroadcastRefresh`:
    /// warmup and tail always refresh, every k-th step in between) rather than
    /// applying a flat 1/2.83, because those windows SCALE with the schedule —
    /// at 6 steps or fewer they cover the whole run and the recipe is a no-op.
    /// An estimate using the flat factor would promise a short run 3x faster
    /// than it can possibly go.
    static func fastFactor(steps: Int, k: Int = 2) -> Double {
        guard steps > 0 else { return 1.0 }
        let warmup = scaledGate(steps: steps, at30: 4)
        let tail = scaledGate(steps: steps, at30: 2)
        var refresh = 0
        for i in 0..<steps {
            if i < warmup || i + tail >= steps || (i - warmup) % k == 0 { refresh += 1 }
        }
        let broadcast = steps - refresh
        // A broadcast step skips the attention branch: measured 63 s against a
        // full step's 280 s at the capstone geometry.
        let schedule = (Double(refresh) + 0.225 * Double(broadcast)) / Double(steps)
        // Velocity caching (TeaCache-style) then skips a further share of what
        // is left — it never touches the first two steps or the last, so short
        // schedules get little of it. 0.55 reproduces the capstone's measured
        // 14-of-30 cached steps.
        let cacheable = Double(max(0, steps - 3)) / Double(steps)
        return schedule * (1.0 - 0.55 * cacheable)
    }

    /// The engine's `scaledGate`: a window that is `at30` steps wide on the
    /// 30-step schedule, scaled down and clamped at 1.
    private static func scaledGate(steps: Int, at30: Int) -> Int {
        max(1, min(at30, steps * at30 / 30))
    }

    /// End-to-end seconds for one generation, including load, text encode and
    /// VAE decode — a "sampling only" number is not what a user is waiting for.
    static func seconds(model: VideoModelPreset, width: Int, height: Int, frames: Int,
                        steps: Int, fast: Bool, hardware: H3Hardware = .current,
                        promptTokens: Int = 250) -> Double {
        let r = H3Plan.rows(width: width, height: height, frames: frames, promptTokens: promptTokens)
        let perStep = denseStepSeconds(rows: r) * (fast ? fastFactor(steps: steps) : 1.0)
        let vae = vaeAnchorSeconds * (Double(width) * Double(height) * Double(frames)) / vaeAnchorPixels
        let anchorSeconds = perStep * Double(steps) + fixedSeconds + vae
        let cores = hardware.gpuCores > 0 ? Double(hardware.gpuCores) : Double(H3Hardware.anchor.gpuCores)
        return anchorSeconds * (Double(H3Hardware.anchor.gpuCores) / cores)
    }

    /// Seconds still to go, from the step cadence of a run in flight, or nil
    /// when there is not enough to say.
    ///
    /// Step 0 carries graph build and Metal JIT and is thrown away — including
    /// it inflated a 30-step estimate by minutes. The rest is a MEDIAN, not a
    /// mean or the last lap: under the fast recipe a velocity-cached step takes
    /// ~0.02 s, and one of those in the trailing window makes the whole
    /// remaining run look instant.
    static func liveEta(stepDurations: [Double], totalSteps: Int) -> Double? {
        let laps = Array(stepDurations.dropFirst())
        guard laps.count >= 2, totalSteps > 0 else { return nil }
        let remaining = totalSteps - stepDurations.count
        guard remaining > 0 else { return 0 }
        let sorted = laps.sorted()
        let median = sorted.count % 2 == 1
            ? sorted[sorted.count / 2]
            : (sorted[sorted.count / 2 - 1] + sorted[sorted.count / 2]) / 2
        return median * Double(remaining)
    }

    /// "about 50 min" / "about 3 h 20 min", with where the number came from.
    /// Prefers this Mac's own history when it has any.
    static func describeBest(model: VideoModelPreset, width: Int, height: Int, frames: Int,
                             steps: Int, fast: Bool,
                             history: H3RunHistory = .load()) -> String {
        let anchorSeconds = seconds(model: model, width: width, height: height, frames: frames,
                                    steps: steps, fast: fast, hardware: .anchor)
        if history.speedFactor != nil {
            return describe(seconds: history.apply(toAnchorSeconds: anchorSeconds),
                            source: .ownHistory(runs: history.runs))
        }
        return describe(seconds: seconds(model: model, width: width, height: height, frames: frames,
                                         steps: steps, fast: fast),
                        source: .hardwareModel(.current))
    }

    static func describe(seconds: Double, source: H3EstimateSource) -> String {
        let provenance: String
        switch source {
        case .hardwareModel(let hw):
            provenance = hw.label.isEmpty ? "estimated" : "estimated for \(hw.label)"
        case .ownHistory(let runs):
            provenance = runs == 1 ? "based on your last run" : "based on your last \(runs) runs"
        }
        return "\(duration(seconds)) — \(provenance)"
    }

    /// Coarse on purpose: a job measured in hours does not have a meaningful
    /// seconds digit, and printing one invites the user to time us.
    static func duration(_ seconds: Double) -> String {
        guard seconds.isFinite, seconds > 0 else { return "unknown" }
        if seconds < 90 { return "about \(Int(seconds.rounded())) sec" }
        let minutes = Int((seconds / 60).rounded())
        if minutes < 60 { return "about \(minutes) min" }
        let h = minutes / 60
        let m = minutes % 60
        return m == 0 ? "about \(h) h" : "about \(h) h \(m) min"
    }
}

// MARK: - Live cadence

/// Turns a run's SSE progress events into step laps.
///
/// Progress events are not one-per-step-in-order: a stage label change repeats
/// a step number, and the load / encode stages report before sampling starts.
/// Only a step that moves FORWARD closes a lap.
struct H3StepClock {
    private var lastStamp: Double?
    private var lastStep: Int = -1
    private(set) var durations: [Double] = []

    init() {}

    mutating func observe(step: Int, at now: Double = ProcessInfo.processInfo.systemUptime) {
        defer { if step > lastStep { lastStep = step; lastStamp = now } }
        guard step > lastStep, let previous = lastStamp else { return }
        durations.append(max(0, now - previous))
    }

    /// Seconds remaining, or nil while there is not enough to say. See
    /// `H3TimeEstimate.liveEta` for why the first lap is discarded.
    func eta(totalSteps: Int) -> Double? {
        H3TimeEstimate.liveEta(stepDurations: durations, totalSteps: totalSteps)
    }
}

// MARK: - Own-history calibration

/// Completed-run timings for THIS Mac, so the estimate stops being someone
/// else's measurement after the first generation.
///
/// One scalar is fitted — how much slower or faster this machine is than the
/// anchor — rather than a per-geometry table: the shape of the cost curve is
/// already known and measured, so a table would need a run at every geometry
/// before it could say anything, while a factor generalizes from run one.
struct H3RunHistory: Codable {
    /// measured / predicted-on-anchor, one per completed run.
    private(set) var records: [Double]
    /// Old records fall off, or a machine never re-learns after an OS or model
    /// update changes what it can do.
    static let keptRuns = 20
    private static let defaultsKey = "h3RunSpeedRatios"

    init(records: [Double]) { self.records = records }

    var runs: Int { records.count }

    mutating func record(predictedOnAnchor: Double, measured: Double) {
        guard predictedOnAnchor > 0, measured > 0, measured.isFinite else { return }
        records.append(measured / predictedOnAnchor)
        if records.count > Self.keptRuns { records.removeFirst(records.count - Self.keptRuns) }
    }

    /// MEDIAN, never a mean: a thermally-throttled run, or one that shared the
    /// GPU with a resident chat model, is several times slow and would own the
    /// estimate for every run after it.
    var speedFactor: Double? {
        guard !records.isEmpty else { return nil }
        let s = records.sorted()
        return s.count % 2 == 1 ? s[s.count / 2] : (s[s.count / 2 - 1] + s[s.count / 2]) / 2
    }

    func apply(toAnchorSeconds seconds: Double) -> Double {
        seconds * (speedFactor ?? 1.0)
    }

    // MARK: Persistence

    static func load(defaults: UserDefaults = .standard) -> H3RunHistory {
        guard let raw = defaults.array(forKey: defaultsKey) as? [Double] else { return H3RunHistory(records: []) }
        return H3RunHistory(records: raw)
    }

    func save(defaults: UserDefaults = .standard) {
        defaults.set(records, forKey: Self.defaultsKey)
    }

    /// Record one finished run against what the anchor model predicted for it.
    static func remember(model: VideoModelPreset, width: Int, height: Int, frames: Int,
                         steps: Int, fast: Bool, measuredSeconds: Double,
                         defaults: UserDefaults = .standard) {
        guard model.backend == .minimaxH3 else { return }
        let predicted = H3TimeEstimate.seconds(model: model, width: width, height: height, frames: frames,
                                               steps: steps, fast: fast, hardware: .anchor)
        var h = load(defaults: defaults)
        h.record(predictedOnAnchor: predicted, measured: measuredSeconds)
        h.save(defaults: defaults)
    }
}

// MARK: - Turbo LoRA acquisition

/// Where the Turbo distillation adapter comes from, and whether this request
/// needs one fetched.
///
/// The adapter is allowlisted in the H3 bundle, so a pack downloaded from now
/// on arrives with it. Everyone who already has the pack does NOT — and they
/// are precisely the people who will turn Turbo on. Re-downloading 69 GB to
/// collect one 744 MB file is not an answer, so the app fetches that file into
/// the pack it already has, from the same repo the pack came from.
enum TurboLoraFetch {
    /// The name the SERVER resolves (`<pack dir>/turbo_lora.safetensors`) and
    /// the bundle allowlists. One constant so a rename cannot land the file
    /// where nothing reads it.
    static let fileName = "turbo_lora.safetensors"

    /// Roughly what it costs, for the sentence shown before it starts. The
    /// file is 744 MB; this is only ever prose.
    static let approxMB = 744

    enum Decision: Equatable {
        /// The adapter is on disk — generate.
        case ready
        /// Missing locally: pull it from the pack's own repo first.
        case fetch
        /// Turbo isn't wanted, or this preset cannot use it at all.
        case notNeeded
        /// The model is on another Mac. Its pack is not ours to complete; that
        /// server answers its own named 400 (which the app now surfaces).
        case unavailableRemotely
    }

    static func decide(turboRequested: Bool, backendSupportsTurbo: Bool,
                       isRemote: Bool, fileOnDisk: Bool) -> Decision {
        guard turboRequested, backendSupportsTurbo else { return .notNeeded }
        if isRemote { return .unavailableRemotely }
        return fileOnDisk ? .ready : .fetch
    }

    /// Whether the adapter is already sitting in `modelDir`.
    static func isOnDisk(modelDir: String?) -> Bool {
        guard let modelDir else { return false }
        return FileManager.default.fileExists(atPath: (modelDir as NSString).appendingPathComponent(fileName))
    }
}
