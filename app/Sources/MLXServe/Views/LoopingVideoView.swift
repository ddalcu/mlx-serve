import AVFoundation
import AppKit
import SwiftUI

/// A silent, looping, chrome-free video — a demo, not a player.
///
/// `AVPlayerView` is the wrong tool here: it brings transport controls, a
/// scrubber and a focus ring for something the user is meant to watch rather
/// than operate. This is a bare `AVPlayerLayer` in an `NSView`, driven by an
/// `AVQueuePlayer` + `AVPlayerLooper` — the looper is what makes the wrap
/// seamless, where the usual "seek to zero on `didPlayToEndTime`" leaves a
/// visible hitch every cycle.
///
/// Muted at the player AND with the item's audio tracks disabled: a welcome
/// screen that starts making noise on launch is the thing everyone remembers
/// about an app, and only one of those two is enough on its own for a file
/// whose audio track is absent anyway.
struct LoopingVideoView: NSViewRepresentable {
    let url: URL
    /// Fill the frame (cropping) or fit inside it. A demo of a UI is usually
    /// `.resizeAspect` — cropping a screen recording cuts off the thing being
    /// demonstrated.
    var gravity: AVLayerVideoGravity = .resizeAspect

    func makeNSView(context: Context) -> NSView {
        let view = PlayerContainerView()
        let item = AVPlayerItem(url: url)
        let queue = AVQueuePlayer(items: [item])
        queue.isMuted = true
        queue.actionAtItemEnd = .advance
        // Held on the coordinator: an AVPlayerLooper that nobody retains stops
        // looping the moment it is collected, which reads as "the video played
        // once and froze".
        context.coordinator.looper = AVPlayerLooper(player: queue, templateItem: item)
        context.coordinator.player = queue

        let layer = AVPlayerLayer(player: queue)
        layer.videoGravity = gravity
        view.playerLayer = layer
        view.wantsLayer = true
        view.layer?.addSublayer(layer)
        queue.play()
        return view
    }

    func updateNSView(_ nsView: NSView, context: Context) {
        context.coordinator.player?.isMuted = true
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

    /// The layer has to be resized by hand: a sublayer added to a view's layer
    /// gets no autoresizing, so without this the video stays at its birth size
    /// while the panel around it grows.
    final class PlayerContainerView: NSView {
        var playerLayer: AVPlayerLayer?

        override func layout() {
            super.layout()
            CATransaction.begin()
            // The implicit animation makes the layer chase the view a frame
            // behind during a live window resize.
            CATransaction.setDisableActions(true)
            playerLayer?.frame = bounds
            CATransaction.commit()
        }
    }
}

/// Where a bundled non-image asset lives, across the three shapes this app is
/// built in: a signed .app, an SPM resource bundle, and a dev build running
/// from source. Mirrors `WelcomeView.loadBundledImage`'s candidate list — the
/// two are the same question asked about different file types.
enum BundledAsset {
    static func url(_ name: String) -> URL? {
        let candidates: [URL?] = [
            Bundle.main.resourceURL?.appendingPathComponent(name),
            Bundle.main.bundleURL.appendingPathComponent("MLXCore_MLXCore.bundle/Resources/\(name)"),
            URL(fileURLWithPath: #filePath)
                .deletingLastPathComponent()          // Views
                .deletingLastPathComponent()          // MLXServe
                .appendingPathComponent("Resources/\(name)"),
        ]
        for case let url? in candidates where FileManager.default.fileExists(atPath: url.path) {
            return url
        }
        return nil
    }
}
