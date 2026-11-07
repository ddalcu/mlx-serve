import AppKit

/// `kill` (SIGTERM), a closed terminal (SIGHUP) and ^C (SIGINT) end the process without AppKit's terminate, so
/// `willTerminate` never runs and the server this app started lives on, orphaned, holding its model. Caught here,
/// they quit the way ⌘Q does.
enum QuitSignals {
    static let quitting: [Int32] = [SIGTERM, SIGHUP, SIGINT]

    /// Keep the returned sources alive: a released source stops delivering.
    static func install(_ signals: [Int32] = quitting,
                        onSignal: @escaping () -> Void = { NSApp.terminate(nil) }) -> [DispatchSourceSignal] {
        signals.map { sig in
            signal(sig, SIG_IGN)
            let source = DispatchSource.makeSignalSource(signal: sig, queue: .main)
            source.setEventHandler(handler: onSignal)
            source.resume()
            return source
        }
    }
}
