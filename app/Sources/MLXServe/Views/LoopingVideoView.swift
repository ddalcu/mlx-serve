import AVFoundation
import AppKit
import SwiftUI

/// A silent, looping, chrome-free video — a demo, not a player.
struct LoopingVideoView: NSViewRepresentable {
    let url: URL
    var gravity: AVLayerVideoGravity = .resizeAspect

    func makeNSView(context: Context) -> NSView {
        let player = AVQueuePlayer()
        player.isMuted = true
        // Held by the coordinator: a looper nobody retains stops looping when
        // it is collected, which reads as "played once and froze".
        context.coordinator.looper = AVPlayerLooper(player: player,
                                                    templateItem: AVPlayerItem(url: url))
        context.coordinator.player = player

        let view = PlayerContainerView()
        view.wantsLayer = true
        let layer = AVPlayerLayer(player: player)
        layer.videoGravity = gravity
        view.playerLayer = layer
        view.layer?.addSublayer(layer)
        player.play()
        return view
    }

    func updateNSView(_ nsView: NSView, context: Context) {
        (nsView as? PlayerContainerView)?.playerLayer?.videoGravity = gravity
    }

    static func dismantleNSView(_ nsView: NSView, coordinator: Coordinator) {
        coordinator.player?.pause()
        coordinator.player = nil
        coordinator.looper = nil
    }

    func makeCoordinator() -> Coordinator { Coordinator() }

    final class Coordinator {
        var player: AVQueuePlayer?
        var looper: AVPlayerLooper?
    }

    /// A sublayer gets no autoresizing, so the layer is sized by hand or it
    /// stays at its birth size while the panel grows. Actions disabled: the
    /// implicit animation makes it chase the view during a live resize.
    final class PlayerContainerView: NSView {
        var playerLayer: AVPlayerLayer?

        override func layout() {
            super.layout()
            CATransaction.begin()
            CATransaction.setDisableActions(true)
            playerLayer?.frame = bounds
            CATransaction.commit()
        }
    }
}

/// Where a bundled asset lives, across the three shapes this app is built in: a
/// signed .app, an SPM resource bundle, and a dev build running from source.
/// ONE lookup — `WelcomeView` and the menu-bar icon had their own copies, and
/// the dev branch of one pointed at `app/<name>` instead of the Resources
/// folder, which is why a stray duplicate asset had to sit there to satisfy it.
enum BundledAsset {
    static func url(_ name: String) -> URL? {
        let candidates = [
            Bundle.main.resourceURL?.appendingPathComponent(name),
            Bundle.main.bundleURL.appendingPathComponent("MLXCore_MLXCore.bundle/Resources/\(name)"),
            URL(fileURLWithPath: #filePath)
                .deletingLastPathComponent()   // Views
                .deletingLastPathComponent()   // MLXServe
                .appendingPathComponent("Resources/\(name)"),
        ]
        return candidates.compactMap { $0 }.first { FileManager.default.fileExists(atPath: $0.path) }
    }

    static func image(_ name: String) -> NSImage? {
        url(name).flatMap { NSImage(contentsOf: $0) }
    }
}
