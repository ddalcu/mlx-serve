import Foundation

/// A slow read held for `hold` seconds, for values a view body asks for on
/// every evaluation but that can still change while the app runs.
final class HeldValue<Value> {
    private let hold: TimeInterval
    private let now: () -> Date
    private let read: () -> Value
    private let lock = NSLock()
    private var cached: (value: Value, at: Date)?

    init(hold: TimeInterval, now: @escaping () -> Date = Date.init, read: @escaping () -> Value) {
        self.hold = hold
        self.now = now
        self.read = read
    }

    var value: Value {
        lock.lock(); defer { lock.unlock() }
        let t = now()
        if let cached, t.timeIntervalSince(cached.at) < hold { return cached.value }
        let v = read()
        cached = (v, t)
        return v
    }
}
