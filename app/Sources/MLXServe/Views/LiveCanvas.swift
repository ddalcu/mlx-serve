import SwiftUI
import AppKit

/// An AppKit view that redraws on its own timer, so something moving costs a
/// draw of this view, not a layout of the window. The timer runs only while
/// `shouldRun` holds and the window can be seen, at `fps` read as it starts.
final class LiveCanvasView: NSView {
    var render: @MainActor (CGContext, CGSize) -> Void = { _, _ in }
    var shouldRun: @MainActor () -> Bool = { false }
    var fps: @MainActor () -> Double = { 0 }
    /// A click or drag at this point, for a view that takes them (a seek bar).
    var onPointer: (@MainActor (CGPoint, CGSize) -> Void)?
    /// Where a click may start (nil = anywhere); its drag then goes where it likes.
    var pointerRegion: (@MainActor (CGSize) -> CGRect)?

    private var timer: Timer?
    private var tracking = false
    private var observers: [NSObjectProtocol] = []

    override var isFlipped: Bool { true }
    override func acceptsFirstMouse(for event: NSEvent?) -> Bool { true }

    override init(frame: NSRect) {
        super.init(frame: frame)
        wantsLayer = true
        layerContentsRedrawPolicy = .onSetNeedsDisplay
    }

    required init?(coder: NSCoder) { fatalError("not used") }

    deinit {
        timer?.invalidate()
        observers.forEach(NotificationCenter.default.removeObserver)
    }

    override func draw(_ dirtyRect: NSRect) {
        guard let cg = NSGraphicsContext.current?.cgContext else { return }
        render(cg, bounds.size)
    }

    /// System colours resolve at draw time, so a light/dark switch needs a redraw.
    override func viewDidChangeEffectiveAppearance() {
        super.viewDidChangeEffectiveAppearance()
        needsDisplay = true
    }

    override func viewDidMoveToWindow() {
        super.viewDidMoveToWindow()
        observers.forEach(NotificationCenter.default.removeObserver)
        observers = []
        let center = NotificationCenter.default
        // A slower machine state, or a window that is hidden or covered, changes what is worth drawing.
        for name in [Notification.Name.NSProcessInfoPowerStateDidChange, ProcessInfo.thermalStateDidChangeNotification] {
            observers.append(center.addObserver(forName: name, object: nil, queue: .main) { [weak self] _ in self?.reschedule() })
        }
        if let window {
            observers.append(center.addObserver(forName: NSWindow.didChangeOcclusionStateNotification, object: window, queue: .main) { [weak self] _ in
                self?.reschedule()
            })
        }
        reschedule()
    }

    /// Redraw once, then run the timer only if something is moving and the window can be seen.
    func reschedule() {
        timer?.invalidate()
        timer = nil
        needsDisplay = true
        guard let window, window.occlusionState.contains(.visible), shouldRun() else { return }
        let rate = fps()
        guard rate > 0 else { return }
        let t = Timer(timeInterval: 1 / rate, repeats: true) { [weak self] _ in
            guard let self else { return }
            self.needsDisplay = true
            if !self.shouldRun() { self.reschedule() }
        }
        RunLoop.main.add(t, forMode: .common)
        timer = t
    }

    private func point(_ event: NSEvent) -> CGPoint { convert(event.locationInWindow, from: nil) }
    override func mouseDown(with event: NSEvent) {
        let p = point(event)
        tracking = pointerRegion.map { $0(bounds.size).contains(p) } ?? true
        if tracking { onPointer?(p, bounds.size); needsDisplay = true }
    }

    override func mouseDragged(with event: NSEvent) {
        guard tracking else { return }
        onPointer?(point(event), bounds.size)
        needsDisplay = true
    }
}

/// SwiftUI's door to `LiveCanvasView`, sized by its frame. SwiftUI hands it new
/// closures when the state they read changes; between those, only its own
/// timer draws.
struct LiveCanvas: NSViewRepresentable {
    let shouldRun: @MainActor () -> Bool
    let fps: @MainActor () -> Double
    var onPointer: (@MainActor (CGPoint, CGSize) -> Void)?
    var pointerRegion: (@MainActor (CGSize) -> CGRect)?
    let render: @MainActor (CGContext, CGSize) -> Void

    func makeNSView(context: Context) -> LiveCanvasView { configured(LiveCanvasView(frame: .zero)) }

    func updateNSView(_ view: LiveCanvasView, context: Context) {
        _ = configured(view)
        view.reschedule()
    }

    private func configured(_ view: LiveCanvasView) -> LiveCanvasView {
        view.render = render
        view.shouldRun = shouldRun
        view.fps = fps
        view.onPointer = onPointer
        view.pointerRegion = pointerRegion
        return view
    }
}
