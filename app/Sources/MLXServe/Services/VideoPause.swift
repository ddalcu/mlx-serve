import Foundation

/// The Pause control of a running video. The server only holds a job at its next step
/// boundary and says so on the stream (`paused`, `resumed`), so the button has a
/// "pausing" state the user can withdraw before the step ends.
struct VideoPause: Equatable {
    enum State: Equatable { case running, pausing, paused }
    private(set) var state: State = .running
    /// When the server confirmed the hold (system uptime); nil while it runs.
    private var heldSince: Double?

    mutating func request() { if state == .running { state = .pausing } }
    mutating func withdraw() { if state == .pausing { state = .running } }
    mutating func resume() { state = .running }

    /// The server held the job. A pause withdrawn meanwhile stays withdrawn: the server
    /// lets go as soon as the withdrawal lands.
    mutating func serverPaused(at now: Double) {
        heldSince = now
        if state == .pausing { state = .paused }
    }

    /// The server let the job go; returns how long it was held, for the clocks to leave out.
    mutating func serverResumed(at now: Double) -> Double {
        defer { heldSince = nil }
        if state == .paused { state = .running }
        guard let since = heldSince else { return 0 }
        return max(0, now - since)
    }
}
