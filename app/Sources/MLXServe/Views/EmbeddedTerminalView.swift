import SwiftUI
import SwiftTerm

/// Gutter balancing (pure — EmbeddedTerminalLayoutTests). SwiftTerm draws its
/// columns from x=0 but stops them short of the right edge by the scroller
/// reservation (its overlay NSScroller is never hidden), so an un-inset
/// terminal reads as "more margin on the right". Mirroring the reservation on
/// the left evens the gutters; the residual difference is the ≤1-cell column
/// quantization every terminal app has.
enum EmbeddedTerminalLayout {
    static func terminalFrame(in bounds: CGRect, scrollerReservation: CGFloat) -> CGRect {
        CGRect(x: bounds.minX + scrollerReservation,
               y: bounds.minY,
               width: max(0, bounds.width - scrollerReservation),
               height: max(0, bounds.height))
    }
}

/// A real terminal emulator embedded in SwiftUI, hosting one PTY-backed
/// process that a `Handle` owns.
///
/// THE SwiftTerm SEAM: this is the only file in the app that imports
/// SwiftTerm. Everything else deals in argv + an exit callback, so a
/// libghostty-backed implementation can replace this single file later.
///
/// LIFETIME: the process belongs to the `Handle`, not to the view. The view
/// only re-parents the handle's terminal into whatever window is showing it,
/// and dismantling it un-parents — it NEVER terminates (a dismantle used to
/// SIGTERM the ssh, so closing the window killed every live session).
struct EmbeddedTerminalView: NSViewRepresentable {
    let handle: Handle

    func makeNSView(context: Context) -> PaddedTerminalContainer {
        PaddedTerminalContainer(handle: handle)
    }

    func updateNSView(_ view: PaddedTerminalContainer, context: Context) {
        // SwiftUI reuses this view across sibling terminal rows (the detail
        // column shows `.terminal(a)` then `.terminal(b)` — same view type,
        // same position), so makeNSView runs once and the container would keep
        // showing the FIRST session's terminal. Point it at the current handle.
        view.handle = handle
    }

    static func dismantleNSView(_ view: PaddedTerminalContainer, coordinator: ()) {
        // Only OUR child: SwiftUI dismantles lazily, and by then another host
        // (a "Move Tab to New Window" pop-out) may already have adopted the
        // terminal — removing it here left that window showing bare ground.
        view.release()
    }

    /// Owns the terminal view and the process it spawned, for as long as the
    /// session lives — independent of any window.
    final class Handle {
        let terminalView: LocalProcessTerminalView
        private let delegate: ProcessDelegate

        /// - Parameters:
        ///   - executable: absolute path (sandbox sessions spawn `/usr/bin/ssh`).
        ///   - onExit: called on the main thread when the child exits. nil exit
        ///     code = the PTY/IO layer died (e.g. the guest was stopped).
        init(executable: String, args: [String], onExit: @escaping (Int32?) -> Void) {
            terminalView = LocalProcessTerminalView(frame: .zero)
            delegate = ProcessDelegate(onExit: onExit)
            terminalView.processDelegate = delegate
            // Default environment (TERM=xterm-256color etc.) — ssh needs nothing
            // from the host env; every path it uses arrives via argv.
            terminalView.startProcess(executable: executable, args: args)
        }

        /// SIGTERM to the spawned process, which drops the PTY and fires onExit.
        func terminate() { terminalView.terminate() }

        /// Paint a theme (the 16 ANSI slots + text) on `background`. Live:
        /// SwiftTerm recomputes its palette and redraws.
        func apply(theme: TerminalTheme, background: TerminalTheme.RGB) {
            terminalView.installColors(theme.ansi.map(Self.color))
            terminalView.nativeForegroundColor = Self.nsColor(theme.foreground)
            terminalView.nativeBackgroundColor = Self.nsColor(background)
            terminalView.caretColor = Self.nsColor(theme.foreground)
            terminalView.needsDisplay = true
            terminalView.superview?.needsLayout = true
        }

        private static func color(_ c: TerminalTheme.RGB) -> SwiftTerm.Color {
            SwiftTerm.Color(red: UInt16(c.r) * 257, green: UInt16(c.g) * 257, blue: UInt16(c.b) * 257)
        }

        private static func nsColor(_ c: TerminalTheme.RGB) -> NSColor {
            NSColor(srgbRed: CGFloat(c.r) / 255, green: CGFloat(c.g) / 255, blue: CGFloat(c.b) / 255, alpha: 1)
        }
    }

    /// Hosts the terminal with a left inset mirroring SwiftTerm's right-side
    /// scroller reservation (see EmbeddedTerminalLayout), painted in the
    /// terminal's own background color so the strip reads as margin, not seam.
    ///
    /// ONE terminal view, several possible hosts (the chat window's detail
    /// column, a pop-out window, a stale container SwiftUI has not dismantled
    /// yet): a host adopts the terminal only while it is IN A WINDOW. A
    /// detached container updating itself used to steal the view back from
    /// the pop-out, which then showed bare ground and never repainted.
    final class PaddedTerminalContainer: NSView {
        var handle: Handle {
            didSet { adoptIfVisible() }
        }
        private var terminal: LocalProcessTerminalView { handle.terminalView }

        init(handle: Handle) {
            self.handle = handle
            super.init(frame: .zero)
            wantsLayer = true
        }

        required init?(coder: NSCoder) { nil }

        override func viewDidMoveToWindow() {
            super.viewDidMoveToWindow()
            adoptIfVisible()
        }

        private func adoptIfVisible() {
            guard window != nil else { return }
            // Drop whatever we were showing (a previous handle's terminal).
            for sub in subviews where sub !== terminal { sub.removeFromSuperview() }
            if terminal.superview !== self {
                terminal.removeFromSuperview()
                addSubview(terminal)
            }
            needsLayout = true
            terminal.needsDisplay = true
            DispatchQueue.main.async { [weak self] in
                guard let self, self.terminal.superview === self else { return }
                self.window?.makeFirstResponder(self.terminal)
            }
        }

        func release() {
            if terminal.superview === self { terminal.removeFromSuperview() }
        }

        override func layout() {
            super.layout()
            // The same width SwiftTerm reserves for its scroller strip
            // (scrollerStyle is public; the reservation itself is not).
            let reservation = NSScroller.scrollerWidth(for: .regular,
                                                       scrollerStyle: terminal.scrollerStyle)
            terminal.frame = EmbeddedTerminalLayout.terminalFrame(in: bounds,
                                                                  scrollerReservation: reservation)
            layer?.backgroundColor = terminal.nativeBackgroundColor.cgColor
        }
    }

    private final class ProcessDelegate: NSObject, LocalProcessTerminalViewDelegate {
        private let onExit: (Int32?) -> Void
        private var exited = false

        init(onExit: @escaping (Int32?) -> Void) { self.onExit = onExit }

        func sizeChanged(source: LocalProcessTerminalView, newCols: Int, newRows: Int) {}
        func setTerminalTitle(source: LocalProcessTerminalView, title: String) {}
        func hostCurrentDirectoryUpdate(source: TerminalView, directory: String?) {}
        func processTerminated(source: TerminalView, exitCode: Int32?) {
            // Dedup: terminate() + the IO teardown can both land here.
            guard !exited else { return }
            exited = true
            let cb = onExit
            DispatchQueue.main.async { cb(exitCode) }
        }
    }
}
