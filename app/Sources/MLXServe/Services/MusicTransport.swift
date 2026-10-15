import AppKit
import Combine
import MediaPlayer

/// Previous / Play-Pause / Next for the Music tab's track from outside the pane:
/// the tray row and macOS Now Playing (media keys, Control Center). Presses go
/// through `DeckActions`, so AI Radio's locks hold here too, and Previous/Next
/// step through the Music list whichever tab last set the player's queue.
@MainActor
final class MusicTransport: ObservableObject {
    static let shared = MusicTransport()

    struct Track: Equatable {
        let path: String
        let title: String
        let playing: Bool
        let radioOn: Bool
    }

    /// A music track playing, paused, restored from the last run, or the last one of a station on air; else nil.
    @Published private(set) var track: Track?

    private let player = AudioClipPlayer.shared
    private let radio = AIRadio.shared
    private var music: MusicGenService?
    private var observer: AnyCancellable?
    private var commandsRegistered = false

    func attach(_ music: MusicGenService) {
        self.music = music
        // objectWillChange fires before the value lands: read it a turn later.
        observer = player.objectWillChange.merge(with: radio.objectWillChange, music.objectWillChange)
            .receive(on: RunLoop.main)
            .sink { [weak self] in self?.refresh() }
        if player.loadedPath == nil, let path = UserDefaults.standard.string(forKey: Self.savedPath),
           music.recent.contains(path) {
            player.restore(path, at: UserDefaults.standard.double(forKey: Self.savedSeconds))
        }
        // Synchronous delivery: termination may never turn the runloop again.
        NotificationCenter.default.addObserver(forName: NSApplication.willTerminateNotification, object: nil, queue: nil) { _ in
            MainActor.assumeIsolated { MusicTransport.shared.save() }
        }
    }

    private static let savedPath = "music.resumePath"
    private static let savedSeconds = "music.resumeSeconds"

    /// The Music track loaded now and how far into it, for the next launch.
    private func save() {
        guard let music, let path = player.loadedPath, music.recent.contains(path) else { return }
        UserDefaults.standard.set(path, forKey: Self.savedPath)
        UserDefaults.standard.set(player.elapsed, forKey: Self.savedSeconds)
    }

    func press(_ key: DeckKey) {
        guard let track, let music else { return }
        DeckActions(player: player, radio: radio, path: track.path, status: nil,
                    onRadioStart: {}, onRadioStop: {}, queue: music.recent).press(key)
    }

    private func refresh() {
        var next: Track?
        if let music, let path = player.loadedPath, player.playingPath != nil || radio.isOn || player.resumeAt != nil,
           music.recent.contains(path) {
            next = Track(path: path, title: TrackMeta.of(path).title,
                         playing: player.playingPath == path && !player.isPaused, radioOn: radio.isOn)
        }
        guard next != track else { return }
        track = next
        save()
        publishNowPlaying()
    }

    private func publishNowPlaying() {
        let center = MPNowPlayingInfoCenter.default()
        // A restored track waits for a click: launch must not take the media keys from other apps.
        guard let track, player.playingPath != nil || track.radioOn else {
            center.nowPlayingInfo = nil
            center.playbackState = .stopped
            return
        }
        // Registered on the first track, so the media keys stay with other apps until music plays here.
        registerCommands()
        MPRemoteCommandCenter.shared().previousTrackCommand.isEnabled = !DeckKey.previous.isLocked(radioOn: track.radioOn)
        let meta = TrackMeta.of(track.path)
        var info: [String: Any] = [
            MPMediaItemPropertyTitle: track.title,
            MPNowPlayingInfoPropertyElapsedPlaybackTime: player.elapsed,
            MPNowPlayingInfoPropertyPlaybackRate: track.playing ? 1.0 : 0.0,
        ]
        if !meta.model.isEmpty { info[MPMediaItemPropertyArtist] = meta.model }
        if track.radioOn { info[MPMediaItemPropertyAlbumTitle] = "AI Radio" }
        if let seconds = AudioClipPlayer.info(track.path)?.seconds { info[MPMediaItemPropertyPlaybackDuration] = seconds }
        center.nowPlayingInfo = info
        center.playbackState = track.playing ? .playing : .paused
    }

    private func registerCommands() {
        guard !commandsRegistered else { return }
        commandsRegistered = true
        let commands = MPRemoteCommandCenter.shared()
        func bind(_ command: MPRemoteCommand, _ key: DeckKey, when: @escaping (Track) -> Bool = { _ in true }) {
            command.addTarget { [weak self] _ in
                guard let self, let track = self.track, when(track) else { return .noActionableNowPlayingItem }
                self.press(key)
                return .success
            }
        }
        bind(commands.togglePlayPauseCommand, .playPause)
        bind(commands.playCommand, .playPause) { !$0.playing }
        bind(commands.pauseCommand, .playPause) { $0.playing }
        bind(commands.nextTrackCommand, .next)
        bind(commands.previousTrackCommand, .previous)
    }
}
