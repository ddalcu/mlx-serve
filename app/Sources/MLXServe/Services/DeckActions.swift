import AppKit

/// A key on the generated-audio player's deck, in either look (MLX-Amp, Modern).
enum DeckKey: CaseIterable {
    case previous, playPause, stop, next, shuffle, repeatAll, export, reveal, radio
    /// A row of the history: a click plays or stops it.
    case historyRow

    /// Where a press goes. A running AI Radio owns playback: it locks the keys
    /// that would pick the next track and takes Stop (asked first) and Next.
    func route(radioOn: Bool) -> DeckRoute {
        switch self {
        case .radio: return .radio
        case .playPause, .export, .reveal: return .own
        case .stop, .next: return radioOn ? .radio : .own
        case .previous, .shuffle, .repeatAll, .historyRow: return radioOn ? .locked : .own
        }
    }

    func isLocked(radioOn: Bool) -> Bool { route(radioOn: radioOn) == .locked }
}

enum DeckRoute: Equatable {
    /// The key does its own thing to the player or the file.
    case own
    /// The station takes the press.
    case radio
    case locked
}

/// What the deck's keys do: both looks bind their keys here, so a change to
/// the radio's locks or redirects lands in both.
@MainActor
struct DeckActions {
    let player: AudioClipPlayer
    let radio: AIRadio
    let path: String
    /// The pane's status replaces the track on the title (generating, failed,
    /// empty): Play then starts the track rather than resuming it.
    let status: MlxAmpPlayer.Status?
    let onRadioStart: () -> Void
    /// Asked first: stopping a station discards the track being composed.
    let onRadioStop: () -> Void
    /// The list Previous/Next step through; nil = the player's own queue.
    var queue: [String]? = nil

    var active: Bool { status == nil && player.playingPath == path }
    var playing: Bool { active && !player.isPaused }

    func isLocked(_ key: DeckKey) -> Bool { key.isLocked(radioOn: radio.isOn) }

    func press(_ key: DeckKey) {
        switch (key, key.route(radioOn: radio.isOn)) {
        case (_, .locked): return
        case (.radio, _): if radio.isOn { onRadioStop() } else { onRadioStart() }
        case (.stop, .radio): onRadioStop()
        case (.next, .radio): radio.skip()
        case (.previous, _): player.advance(-1, from: path, in: queue)
        case (.playPause, _): if active { player.togglePause() } else if !path.isEmpty { player.play(path) }
        case (.stop, _): player.stop()
        case (.next, _): player.advance(1, from: path, in: queue)
        case (.shuffle, _): player.shuffle.toggle()
        case (.repeatAll, _): player.autoNext.toggle()
        case (.export, _): if !path.isEmpty { AudioExport.exportAndReveal(path) }
        case (.reveal, _): if !path.isEmpty { NSWorkspace.shared.activateFileViewerSelecting([URL(fileURLWithPath: path)]) }
        // A row plays or stops through its list's own callbacks; here it only locks.
        case (.historyRow, _): break
        }
    }
}
