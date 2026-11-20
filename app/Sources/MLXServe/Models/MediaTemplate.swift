import Foundation

/// The gallery's sections, in order.
enum TemplateKind: CaseIterable {
    case video, image, audio

    var title: String {
        switch self {
        case .video: "Video"
        case .image: "Image"
        case .audio: "Audio"
        }
    }
}

/// One-tap media for people who will never write a prompt: a few blanks, the
/// words a chat model writes from them, and the user's own photos. Videos are
/// the six-section REF2VA document FastH3 Omni renders with the photos as
/// `<Picture n>`; pictures are a Qwen-Image edit of one photo; songs are sung
/// by ACE-Step and shared as a video of the photo.
enum MediaTemplate: String, CaseIterable, Identifiable {
    case santa, pets, reaction, party, hug
    case portrait, headshot, poster, figure
    case birthday, theme, ballad, roast

    var id: String { rawValue }

    var kind: TemplateKind {
        switch self {
        case .santa, .pets, .reaction, .party, .hug: .video
        case .portrait, .headshot, .poster, .figure: .image
        case .birthday, .theme, .ballad, .roast: .audio
        }
    }

    /// References + native audio in 8 steps.
    static let videoModel = VideoModelPreset.fastH3Omni
    /// Edits a photo into a new scene and keeps the face, in 8 steps.
    static let imageModel = ImageModelPreset.qwenImage21Turbo_4bit
    /// Sings the lyrics it is given, in 8 steps.
    static let songModel = MusicModelPreset.acestepXLTurbo8bit

    /// A clip's or a song's length; a picture has none. Omni was trained on 5,
    /// 10 and 15 second clips, and a reaction is a beat, not a scene.
    var seconds: Int {
        switch self {
        case .reaction: 5
        case .santa, .pets, .party, .hug: 10
        case .portrait, .headshot, .poster, .figure: 0
        case .birthday, .ballad: 45
        case .theme, .roast: 40
        }
    }

    /// A reaction is shared as a GIF in a chat and a song's cover is an album
    /// square, where square reads best.
    var defaultShape: ShareShape {
        switch self {
        case .reaction, .headshot, .birthday, .theme, .ballad, .roast: .square
        default: .story
        }
    }

    /// Word-timed captions from the soundtrack: only where someone talks.
    var captions: Bool { [.santa, .pets, .hug].contains(self) }

    /// A reaction is a meme: it also saves as a looping GIF.
    var makesGIF: Bool { self == .reaction }

    var title: String {
        switch self {
        case .santa: "A Message from Santa"
        case .pets: "Pet Talk"
        case .reaction: "Reaction"
        case .party: "Dance Party"
        case .hug: "Hug Your Younger Self"
        case .portrait: "Royal Portrait"
        case .headshot: "Corporate Headshot"
        case .poster: "Movie Poster"
        case .figure: "Action Figure"
        case .birthday: "Birthday Song"
        case .theme: "Theme Song"
        case .ballad: "Power Ballad"
        case .roast: "Roast Rap"
        }
    }

    var blurb: String {
        switch self {
        case .santa: "Santa says their name and holds up their photo by the fire."
        case .pets: "Your pets host a podcast, read the news or confess on the couch."
        case .reaction: "Anyone in your cast, mind blown, rolling their eyes or saying say less. Add meme text, save a GIF."
        case .party: "Up to four of your cast dancing their hearts out at a packed house party."
        case .hug: "You, today, kneel down and hug the child in an old photo of you."
        case .portrait: "Anyone in your cast as a pompous noble in a Renaissance oil painting, with meme text."
        case .headshot: "Your pet or your friend in a suit, in a very serious LinkedIn headshot."
        case .poster: "Anyone in your cast as the hero of a blockbuster action movie."
        case .figure: "Anyone in your cast as a boxed collectible action figure, with their own accessories."
        case .birthday: "A sing-along birthday song with their name in it, as a video of their photo."
        case .theme: "An 80s sitcom theme song all about them."
        case .ballad: "An epic power ballad about something small that went wrong."
        case .roast: "A playful hip-hop roast of someone, all in good fun."
        }
    }

    var symbol: String {
        switch self {
        case .santa: "gift"
        case .pets: "pawprint"
        case .reaction: "face.smiling"
        case .party: "figure.socialdance"
        case .hug: "heart"
        case .portrait: "crown"
        case .headshot: "briefcase"
        case .poster: "film"
        case .figure: "figure.stand"
        case .birthday: "birthday.cake"
        case .theme: "tv"
        case .ballad: "music.mic"
        case .roast: "flame"
        }
    }

    /// The muted sample loop in the app's Resources.
    var sampleName: String { "template-\(rawValue)" }
    /// The clip of the option picked in `fill`, for the stage; nil for a template without options.
    func choiceSample(_ fill: TemplateFill) -> String? {
        for slot in slots { if case .choice = slot.kind { return sampleName(choice: fill.text(slot.key)) } }
        return nil
    }

    /// The bundled clip showing one option of a picker: "template-reaction-mind-blown".
    func sampleName(choice: String) -> String {
        sampleName + "-" + choice.lowercased().split { !$0.isLetter && !$0.isNumber }.joined(separator: "-")
    }

    var slots: [TemplateSlot] {
        switch self {
        case .santa:
            [TemplateSlot(key: "child", label: "Who is it for?", kind: .cast),
             TemplateSlot(key: "age", label: "How old are they?", kind: .text(example: "6")),
             TemplateSlot(key: "deed", label: "Something good they did this year",
                          kind: .text(example: "helped their little sister learn to ride a bike")),
             TemplateSlot(key: "wish", label: "What are they hoping for?", kind: .text(example: "a telescope"))]
        case .pets:
            [TemplateSlot(key: "a", label: "Who's talking?", kind: .cast),
             TemplateSlot(key: "b", label: "And who answers?", kind: .cast, optional: true),
             TemplateSlot(key: "topic", label: "What about?", kind: .text(example: "why the couch cushion is missing")),
             TemplateSlot(key: "format", label: "Where?", kind: .choice(PetFormat.allCases.map(\.label)))]
        case .reaction:
            [TemplateSlot(key: "who", label: "Who's reacting?", kind: .cast),
             TemplateSlot(key: "reaction", label: "Reaction", kind: .choice(Reaction.allCases.map(\.label))),
             TemplateSlot(key: "top", label: "Meme text on top", kind: .text(example: "when the WiFi drops mid-game"),
                          optional: true)]
        case .party:
            [TemplateSlot(key: "a", label: "Who dances out front?", kind: .cast),
             TemplateSlot(key: "b", label: "Dancing behind", kind: .cast, optional: true),
             TemplateSlot(key: "c", label: "Dancing behind", kind: .cast, optional: true),
             TemplateSlot(key: "d", label: "Dancing behind", kind: .cast, optional: true)]
        case .hug:
            [TemplateSlot(key: "now", label: "You, today", kind: .cast),
             TemplateSlot(key: "young", label: "You as a child", kind: .cast),
             TemplateSlot(key: "whisper", label: "What do you whisper?", kind: .text(example: "I'm so proud of you."))]
        case .portrait:
            memeSlots(top: "his royal highness", bottom: "duke of the couch")
        case .headshot:
            memeSlots(top: "chief snack officer", bottom: "10 years of nap experience")
        case .poster:
            memeSlots(top: "this summer, one good boy", bottom: "the last nap")
        case .figure:
            [TemplateSlot(key: "who", label: "Who's in it?", kind: .cast),
             TemplateSlot(key: "gear", label: "Their accessories", kind: .text(example: "a coffee mug, a laptop and a tiny dog bed")),
             TemplateSlot(key: "top", label: "Top text", kind: .text(example: "limited edition"), optional: true),
             TemplateSlot(key: "bottom", label: "Bottom text", kind: .text(example: "batteries not included"), optional: true)]
        case .birthday:
            [TemplateSlot(key: "who", label: "Who's it for?", kind: .cast),
             TemplateSlot(key: "love", label: "Something they love", kind: .text(example: "pizza and long walks"))]
        case .theme:
            [TemplateSlot(key: "who", label: "Whose theme song?", kind: .cast),
             TemplateSlot(key: "known", label: "What are they known for?", kind: .text(example: "stealing socks and snoring"))]
        case .ballad:
            [TemplateSlot(key: "who", label: "Whose heartbreak?", kind: .cast),
             TemplateSlot(key: "what", label: "What went wrong?", kind: .text(example: "my coffee went cold"))]
        case .roast:
            [TemplateSlot(key: "who", label: "Who's getting roasted?", kind: .cast),
             TemplateSlot(key: "tease", label: "Roast them about", kind: .text(example: "always being late"))]
        }
    }

    private func memeSlots(top: String, bottom: String) -> [TemplateSlot] {
        [TemplateSlot(key: "who", label: "Who's in it?", kind: .cast),
         TemplateSlot(key: "top", label: "Top text", kind: .text(example: top), optional: true),
         TemplateSlot(key: "bottom", label: "Bottom text", kind: .text(example: bottom), optional: true)]
    }

    /// Required text slots start on their example, so a first try needs nothing typed.
    func defaultFill() -> TemplateFill {
        var fill = TemplateFill()
        for slot in slots {
            switch slot.kind {
            case .text(let example): if !slot.optional { fill.answers[slot.key] = example }
            case .choice(let options): fill.answers[slot.key] = options.first
            case .cast: break
            }
        }
        return fill
    }

    /// The label of the first required slot left empty, or nil when the form is complete.
    func missing(_ fill: TemplateFill) -> String? {
        slots.first { slot in
            guard !slot.optional else { return false }
            if case .cast = slot.kind { return fill.cast[slot.key] == nil }
            return fill.text(slot.key).isEmpty
        }?.label
    }

    /// The photos sent as references, in `<Picture n>` order.
    func cast(_ fill: TemplateFill) -> [CastMember] {
        slots.compactMap { slot in
            if case .cast = slot.kind { return fill.cast[slot.key] }
            return nil
        }
    }

    // MARK: - Lines

    /// The spoken lines, in the order they are said.
    func lineKeys(_ fill: TemplateFill) -> [String] {
        switch self {
        case .santa: ["santa"]
        case .pets: fill.cast["b"] == nil ? ["a"] : ["a", "b"]
        case .reaction: Reaction(label: fill.text("reaction")).line == nil ? [] : ["line"]
        case .party, .portrait, .headshot, .poster, .figure: []
        case .hug: ["whisper"]
        case .birthday, .theme, .ballad, .roast: ["title", "lyrics"]
        }
    }

    /// A line the template speaks that is still blank.
    func linesMissing(_ fill: TemplateFill, _ lines: [String: String]) -> Bool {
        lineKeys(fill).contains { (lines[$0] ?? "").trimmingCharacters(in: .whitespacesAndNewlines).isEmpty }
    }

    /// Who says a line, as the review step names them.
    func speaker(of key: String, _ fill: TemplateFill) -> String {
        switch key {
        case "santa": "Santa"
        case "whisper": fill.cast["now"]?.name ?? "You"
        case "line": fill.cast["who"]?.name ?? "Them"
        case "title": "Title"
        case "lyrics": "Lyrics"
        default: fill.cast[key]?.name ?? key
        }
    }

    /// A template whose only line is the user's own needs no chat model.
    var writesOwnLines: Bool { kind == .image || [.reaction, .party, .hug].contains(self) }

    func ownLines(_ fill: TemplateFill) -> [String: String] {
        switch self {
        case .hug: ["whisper": fill.text("whisper")]
        case .reaction: Reaction(label: fill.text("reaction")).line.map { ["line": $0] } ?? [:]
        default: [:]
        }
    }

    /// The brief the chat model writes the lines from.
    func ask(_ fill: TemplateFill) -> String {
        let keys = lineKeys(fill)
        let rules = " No stage directions, no emoji, no quotation marks. Answer with only a JSON object with the string keys "
            + keys.joined(separator: ", ") + "."
        switch self {
        case .santa:
            let m = fill.cast["child"] ?? CastMember.placeholder(.girl)
            return "Write what Santa Claus says, looking into the camera, in a personal video message to \(m.name), "
                + "a \(m.kind.noun) aged \(fill.text("age")). Something good they did this year: \(fill.text("deed")). "
                + "What they are hoping for: \(fill.text("wish")). Start with a warm greeting that uses the name, "
                + "mention the good deed, hint about the wish without promising it. Warm and jolly, at most "
                + "\(TemplateText.wordBudget(seconds: seconds) - 2) words of spoken English." + rules
        case .pets:
            let format = PetFormat(label: fill.text("format"))
            var brief = keys.count == 1 ? "Write one line for a funny talking video. "
                : "Write a two-line exchange for a funny talking video. "
            for key in keys {
                guard let m = fill.cast[key] else { continue }
                brief += "\(m.name) (key \(key)) is a \(m.kind.noun), \(m.voice.trait). "
            }
            brief += "Setting: \(format.brief). Topic: \(fill.text("topic")). "
            brief += keys.count == 1 ? "" : "The first speaks first and the second replies with the punchline. "
            return brief + "Each line at most 12 words of spoken English." + rules
        case .birthday, .theme, .ballad, .roast:
            return songAsk(fill)
        case .reaction, .party, .hug, .portrait, .headshot, .poster, .figure:
            return ""
        }
    }

    /// Lines to start from when no chat model answers; the review step edits them.
    func fallbackLines(_ fill: TemplateFill) -> [String: String] {
        switch self {
        case .santa:
            let name = fill.cast["child"]?.name ?? "my friend"
            return ["santa": "Ho ho ho, \(name)! I heard you \(fill.text("deed")). Keep being kind, and watch for a little surprise."]
        case .pets:
            var lines = ["a": "Okay, I can explain. It's about \(fill.text("topic"))."]
            if fill.cast["b"] != nil { lines["b"] = "Nobody asked, but go on." }
            return lines
        case .birthday, .theme, .ballad, .roast:
            return songFallback(fill)
        case .reaction, .party, .hug, .portrait, .headshot, .poster, .figure:
            return ownLines(fill)
        }
    }

    // MARK: - The H3 document

    func document(_ fill: TemplateFill, lines: [String: String]) -> String {
        let spoken = lines.mapValues(TemplateText.clean)
        switch self {
        case .santa: return santaDocument(fill, spoken)
        case .pets: return petsDocument(fill, spoken)
        case .reaction: return reactionDocument(fill, spoken)
        case .party: return partyDocument(fill)
        case .hug: return hugDocument(fill, spoken)
        case .portrait, .headshot, .poster, .figure, .birthday, .theme, .ballad, .roast: return ""
        }
    }

    private func definition(_ i: Int, _ m: CastMember, extra: String = "") -> String {
        let looks = TemplateText.clean(m.looks)
        return "<Subject \(i)> is the \(m.kind.noun) in <Picture \(i)>\(extra)\(looks.isEmpty ? "" : ", \(looks)")."
    }

    private func keeps(_ m: CastMember) -> String {
        let looks = TemplateText.clean(m.looks)
        return (m.kind.isPerson ? "\(m.kind.their) face, hair and clothing are retained"
            : "\(m.kind.their) face, colouring, markings and anything \(m.kind.they) wear\(m.kind.they == "they" ? "" : "s"), like a collar or bandana, are retained") + (looks.isEmpty ? "" : ", including \(looks)")
    }

    private func santaDocument(_ fill: TemplateFill, _ lines: [String: String]) -> String {
        let child = fill.cast["child"] ?? CastMember.placeholder(.girl)
        return """
        subject_definitions:
        \(definition(1, child))

        summary:
        [reference generation] The target video is a single shot of Santa Claus by his workshop fireplace, holding up a framed photograph of <Subject 1> and speaking a personal message to them.

        retention_analysis:
        <Subject 1> (appears in [Shot 1] as the framed photograph Santa holds): fully_preserved - \(keeps(child)) inside the photograph.

        detailed_description:
        The target video is in a warm, magical cinematic style with flickering firelight and soft twinkling lights.
        [Shot 1] A medium shot with a very slow push in frames Santa Claus (S1), a jolly older man with a full, fluffy white beard, rosy cheeks, round gold-rimmed spectacles and a red velvet suit with white fur trim, seated in a carved wooden armchair beside a crackling stone fireplace in his cozy workshop. Shelves of hand-painted wooden toys and strings of warm twinkling lights fill the background, and a few snowflakes drift past a frosted window. He holds up a small framed photograph of <Subject 1> toward the camera with both hands, glances at it fondly, then looks straight into the lens with twinkling eyes and says in a deep, warm, jolly voice, <d>[English] \(lines["santa"] ?? "")</d> He chuckles softly, lowers the photograph to his knee and gives a slow, knowing wink as the firelight dances across his face.

        overall_soundscape:
        The steady crackle and pop of the fireplace, a faint jingle of sleigh bells somewhere outside, the soft creak of the wooden armchair and Santa's warm, close voice.

        non_diegetic_music:
        A gentle music box playing a slow festive melody, very quiet beneath the dialogue.
        """
    }

    private func petsDocument(_ fill: TemplateFill, _ lines: [String: String]) -> String {
        let a = fill.cast["a"] ?? CastMember.placeholder(.dog)
        let b = fill.cast["b"]
        let format = PetFormat(label: fill.text("format"))
        let pair = b != nil
        var defs = definition(1, a)
        var keep = "<Subject 1> (appears in [Shot 1]): fully_preserved - \(keeps(a))."
        var shot = "[Shot 1] A medium static shot at eye level frames <Subject 1> (S1)"
        var action = "<Subject 1> leans \(format.toward) with \(a.kind.isPerson ? "eyebrows" : "ears perked and eyebrows") raised, and says in \(a.voice.phrase), <d>[English] \(lines["a"] ?? "")</d>"
        if let b {
            defs += "\n" + definition(2, b)
            keep += "\n<Subject 2> (appears in [Shot 1]): fully_preserved - \(keeps(b))."
            shot += " on the left and <Subject 2> (S2) on the right"
            action += " <Subject 2> slowly turns toward <Subject 1>, narrows \(b.kind.their) eyes and replies in \(b.voice.phrase), <d>[English] \(lines["b"] ?? "")</d> <Subject 1> tilts \(a.kind.their) head"
        } else {
            shot += " in the center of the frame"
            action += " <Subject 1> tilts \(a.kind.their) head and blinks slowly"
        }
        let hosts = pair ? "<Subject 1> and <Subject 2>" : "<Subject 1>"
        return """
        subject_definitions:
        \(defs)

        summary:
        [reference generation] The target video is a single shot of \(hosts) \(format.summary), \(pair ? "each speaking one line" : "speaking one line").

        retention_analysis:
        \(keep)

        detailed_description:
        The target video is in a realistic cinematic style with \(format.light).
        \(shot), \(format.set(pair: pair)) \(action) while the camera holds steady.

        overall_soundscape:
        \(format.sound), and the close, clear \(pair ? "voices of both hosts" : "voice of the host").

        non_diegetic_music:
        N/A
        """
    }

    private func reactionDocument(_ fill: TemplateFill, _ lines: [String: String]) -> String {
        let who = fill.cast["who"] ?? CastMember.placeholder(.dog)
        let reaction = Reaction(label: fill.text("reaction"))
        return """
        subject_definitions:
        \(definition(1, who))

        summary:
        [reference generation] The target video is a single short reaction shot of <Subject 1> \(reaction.summary), made to be shared as a reaction meme.

        retention_analysis:
        <Subject 1> (appears in [Shot 1]): fully_preserved - \(keeps(who)).

        detailed_description:
        The target video is in \(reaction.style).
        [Shot 1] \(reaction.shot(who, line: lines["line"] ?? ""))

        overall_soundscape:
        \(reaction.sound)

        non_diegetic_music:
        \(reaction.music)
        """
    }

    private func partyDocument(_ fill: TemplateFill) -> String {
        let dancers = cast(fill)
        let all = dancers.isEmpty ? [CastMember.placeholder(.dog)] : dancers
        let subjects = all.indices.map { "<Subject \($0 + 1)>" }
        let behind = subjects.count == 1 ? "the whole crowd dances along behind"
            : TemplateText.list(Array(subjects.dropFirst())) + (subjects.count == 2 ? " dances" : " dance")
                + " right behind, copying the moves and grinning"
        let upright = all.indices.filter { !all[$0].kind.isPerson }.map { subjects[$0] }
        let pets = upright.isEmpty ? "" : ", " + TemplateText.list(upright)
            + (upright.count == 1 ? " dancing upright on its hind legs" : " dancing upright on their hind legs")
        return """
        subject_definitions:
        \(all.enumerated().map { definition($0.offset + 1, $0.element) }.joined(separator: "\n"))

        summary:
        [reference generation] The target video is a single shot of \(TemplateText.list(subjects)) dancing together in the middle of a packed, lively house party.

        retention_analysis:
        \(all.enumerated().map { "<Subject \($0.offset + 1)> (appears in [Shot 1]): fully_preserved - \(keeps($0.element))." }.joined(separator: "\n"))

        detailed_description:
        The target video is in a bright, fun, realistic style with colourful party lights.
        [Shot 1] A medium wide shot with a slow push in, from the front of a living room decorated for a party, with balloons, streamers and a glowing disco ball. <Subject 1> dances out in front with big, confident, funny moves, swinging arms and hips to the beat, while \(behind)\(pets). A crowd of party guests packed in on both sides cheers, claps and dances along. Everyone stays in full view in the middle of the frame. Confetti drifts through the coloured light.

        overall_soundscape:
        A lively party crowd cheering and clapping on the beat, happy shouts and the shuffle of dancing feet on a wooden floor.

        non_diegetic_music:
        Upbeat, funky dance-party music with a driving four-on-the-floor beat, slap bass and bright brass stabs, fast tempo, loud and joyful.
        """
    }

    private func hugDocument(_ fill: TemplateFill, _ lines: [String: String]) -> String {
        let now = fill.cast["now"] ?? CastMember.placeholder(.woman)
        let young = fill.cast["young"] ?? CastMember.placeholder(.girl)
        let they = now.kind.they.capitalized
        let little = young.kind.isPerson ? "the child" : "the young \(young.kind.noun)"
        let embrace = now.kind.isPerson
            ? "<Subject 1> opens \(now.kind.their) arms and \(little) walks into them; they hold each other tightly, eyes closing."
            : "<Subject 1> nuzzles \(little) gently and they cuddle up close, eyes closing."
        return """
        subject_definitions:
        \(definition(1, now))
        \(definition(2, young, extra: young.kind.isPerson ? ", the same person as a young child" : ", the same \(young.kind.noun) when young"))

        summary:
        [reference generation] The target video is a single emotional shot of <Subject 1> kneeling down to hug <Subject 2>, \(now.kind.their) younger self, in a sunlit backyard.

        retention_analysis:
        <Subject 1> (appears in [Shot 1]): fully_preserved - \(keeps(now)).
        <Subject 2> (appears in [Shot 1]): fully_preserved - \(keeps(young)).

        detailed_description:
        The target video is in a soft, dreamy cinematic style with warm golden-hour backlight.
        [Shot 1] A medium shot with a slow push in frames <Subject 2> standing in tall green grass in a quiet backyard, against a wooden fence and a blur of trees. Golden late-afternoon sun shines from behind and lights the edges of \(little)'s \(young.kind.isPerson ? "hair" : "fur"). <Subject 1> (S1) steps into the frame from the right, stops and looks at \(little) for a moment with glistening eyes, then lowers down to \(little)'s eye level. <Subject 2> looks up shyly, then brightens. \(embrace) \(they) whispers softly, <d>[English] \(lines["whisper"] ?? "")</d> The camera keeps pushing in slowly on the embrace as the low sun flares softly behind them and a few seed heads drift through the light.

        overall_soundscape:
        Gentle evening birdsong, a soft rustle of grass as \(now.kind.they) kneels, the faint rustle of clothing during the hug and a quiet, close whisper.

        non_diegetic_music:
        Solo piano, slow tempo, soft and tender, a simple melody that swells gently as they embrace.
        """
    }
}

/// One blank on a template's form.
struct TemplateSlot: Identifiable, Equatable {
    enum Kind: Equatable {
        /// A photo from the cast: anyone in it, person or pet.
        case cast
        case text(example: String)
        case choice([String])
    }

    let key: String
    let label: String
    let kind: Kind
    var optional = false

    var id: String { key }
}

/// What the user put in a template's blanks.
struct TemplateFill: Equatable {
    var cast: [String: CastMember] = [:]
    var answers: [String: String] = [:]

    /// An answer as it may appear in the document.
    func text(_ key: String) -> String { TemplateText.clean(answers[key] ?? "") }

    /// The picks as the cast has them now: a replaced photo or an edit is
    /// taken up, and a member removed since is dropped.
    func resolved(in members: [CastMember]) -> TemplateFill {
        TemplateFill(cast: cast.compactMapValues { m in members.first { $0.id == m.id } }, answers: answers)
    }
}

/// The reactions the Reaction template offers.
enum Reaction: CaseIterable {
    case mindBlown, eyeRoll, sayLess

    init(label: String) { self = Self.allCases.first { $0.label == label } ?? .mindBlown }

    var label: String {
        switch self {
        case .mindBlown: "Mind blown"
        case .eyeRoll: "Eye roll"
        case .sayLess: "Say less"
        }
    }

    /// The one thing said, if anything.
    var line: String? {
        switch self {
        case .mindBlown: "Whoa."
        case .eyeRoll: nil
        case .sayLess: "Say less."
        }
    }

    var summary: String {
        switch self {
        case .mindBlown: "being so mind-blown that the world around them dissolves into a swirling galaxy"
        case .eyeRoll: "giving a slow, deadpan, unimpressed eye roll"
        case .sayLess: "grinning a wide, knowing grin and nodding slowly before saying say less"
        }
    }

    var style: String {
        switch self {
        case .mindBlown: "a surreal, cosmic style with glowing purple and blue nebula light"
        case .eyeRoll: "a realistic cinematic style with cool, soft, even light"
        case .sayLess: "a realistic cinematic style with warm, soft light"
        }
    }

    /// The shot, framed the way the meme is known.
    func shot(_ m: CastMember, line: String) -> String {
        let hands = m.kind.isPerson ? "hands" : "paws"
        let their = m.kind.their
        switch self {
        case .mindBlown:
            return "A close-up with a slow push in frames <Subject 1> (S1) against a dark background. <Subject 1> stares straight into the lens, then both \(hands) rise to press against the sides of \(their) head as \(their) eyes go wide and \(their) jaw drops in pure amazement. The room behind melts away into a deep, swirling galaxy of purple and blue nebula and drifting stars, and glittering golden stardust streams off \(their) head and shoulders and floats outward into space as <Subject 1> breathes out in \(m.voice.phrase), <d>[English] \(line)</d> The galaxy keeps turning slowly behind \(their) awestruck face."
        case .eyeRoll:
            let pose = m.kind.isPerson ? "standing with \(their) arms crossed" : "sitting very upright with \(their) front paws together"
            return "A static medium close-up, framed from the chest up, holds on <Subject 1> \(pose) against a plain, dark grey wall. <Subject 1> listens to someone off camera with a flat, unimpressed expression, then gives a huge, exaggerated, unmistakable eye roll: \(their) eyes roll slowly all the way up toward the ceiling and over to the side, the pupils sliding up under the upper eyelids, while \(their) head tips back and turns away in pure exasperation. Then \(their) eyes drop back to the person off camera with a flat, deadpan stare. The camera does not move."
        case .sayLess:
            return "A static medium close-up, framed from the chest up, holds on <Subject 1> (S1) against a dark background with soft, warm, out-of-focus lights. <Subject 1> leans in toward the lens with a wide, knowing, mischievous grin and raised eyebrows, and nods big, slow, exaggerated nods, the whole head dipping deep down and back up three times, eyes twinkling. Still grinning, <Subject 1> says in \(m.voice.phrase), <d>[English] \(line)</d> and keeps nodding, the grin growing even wider."
        }
    }

    var sound: String {
        switch self {
        case .mindBlown: "A deep, swelling cosmic whoosh and a soft, glittering shimmer as the stardust streams away, under a breathy, amazed voice."
        case .eyeRoll: "Quiet room tone and one faint, unimpressed exhale through the nose."
        case .sayLess: "Quiet room tone and a short, confident voice close to the camera."
        }
    }

    var music: String {
        switch self {
        case .mindBlown: "A single, slow, ethereal synth swell with a choir pad, rising as the galaxy opens up."
        case .eyeRoll, .sayLess: "N/A"
        }
    }
}

/// Where Pet Talk is set.
enum PetFormat: CaseIterable {
    case podcast, news, couch

    init(label: String) { self = Self.allCases.first { $0.label == label } ?? .podcast }

    var label: String {
        switch self {
        case .podcast: "Podcast"
        case .news: "News desk"
        case .couch: "Couch confession"
        }
    }

    var brief: String {
        switch self {
        case .podcast: "a cozy home podcast"
        case .news: "the evening news, read from a TV anchor desk"
        case .couch: "a reality-show confession to the camera from the living-room couch"
        }
    }

    var summary: String {
        switch self {
        case .podcast: "hosting a cozy home podcast"
        case .news: "presenting the evening news from a TV studio anchor desk"
        case .couch: "sitting on a living-room couch for a reality-show style confession to the camera"
        }
    }

    var toward: String {
        switch self {
        case .podcast: "toward the microphone"
        case .news: "toward the camera with a serious anchor expression"
        case .couch: "toward the camera as if confessing a secret"
        }
    }

    var light: String {
        switch self {
        case .podcast: "warm practical studio light"
        case .news: "bright, even broadcast studio light"
        case .couch: "warm, soft afternoon window light"
        }
    }

    var sound: String {
        switch self {
        case .podcast: "Quiet, intimate studio room tone, the soft creak of a chair"
        case .news: "The hushed hum of a television studio, a faint click of studio equipment"
        case .couch: "Quiet living-room tone, the soft rustle of the couch cushions"
        }
    }

    func set(pair: Bool) -> String {
        let seat = pair ? "sitting side by side" : "sitting"
        switch self {
        case .podcast:
            return "\(seat) behind a wooden desk in a cozy home podcast studio, \(pair ? "each with" : "with") a black broadcast microphone on a boom arm in front of them. Warm tungsten lamps glow against grey acoustic foam panels behind them, and a small potted plant sits at the end of the desk."
        case .news:
            return "\(seat) behind a glossy curved news anchor desk in a bright television studio, a large screen behind them showing a soft-focus city skyline at dusk."
        case .couch:
            return "\(seat) on a soft grey couch in a cozy living room, a throw blanket bunched beside them, a lamp glowing on a side table and family photos blurred on the wall behind."
        }
    }
}

/// Text rules shared by the template documents and their review step.
enum TemplateText {
    /// Words a clip of `seconds` carries at a natural speaking pace.
    static let wordsPerSecond = 2.6

    static func wordBudget(seconds: Int) -> Int { Int(Double(seconds) * wordsPerSecond) }

    static func words(in lines: [String: String]) -> Int {
        lines.values.reduce(0) { $0 + $1.split(whereSeparator: \.isWhitespace).count }
    }

    /// H3's frame ladder is 17k+5 at 24 fps: the first rung that covers `seconds`.
    static func frames(seconds: Int, fps: Int = 24) -> Int {
        let k = Int((Double(seconds * fps - 5) / 17).rounded(.up))
        return 17 * max(1, k) + 5
    }

    /// A user's or a model's words as they may sit in the document: no markup,
    /// no quotes, one line.
    static func clean(_ text: String) -> String {
        let stripped = text.filter { !"<>\"\u{201C}\u{201D}".contains($0) }
        return stripped.split(whereSeparator: \.isWhitespace).joined(separator: " ")
    }

    /// The lines out of a chat reply: the outermost JSON object, every key a
    /// non-blank string, or nil.
    static func parseLines(_ reply: String, keys: [String]) -> [String: String]? {
        guard let open = reply.firstIndex(of: "{"), let close = reply.lastIndex(of: "}"), open < close,
              let obj = try? JSONSerialization.jsonObject(with: Data(reply[open...close].utf8)) as? [String: Any]
        else { return nil }
        var out: [String: String] = [:]
        for key in keys {
            guard let line = (obj[key] as? String).map(key == "lyrics" ? cleanLines : clean), !line.isEmpty else { return nil }
            out[key] = line
        }
        return out
    }

    /// "A, B and C", in English whatever the Mac's language: it goes into an English prompt.
    static func list(_ items: [String]) -> String {
        items.count < 2 ? items.joined() : items.dropLast().joined(separator: ", ") + " and " + items[items.count - 1]
    }

    /// `clean` line by line, keeping the breaks and one blank line between sections.
    static func cleanLines(_ text: String) -> String {
        text.components(separatedBy: .newlines).map(clean)
            .reduce(into: [String]()) { out, line in if !(line.isEmpty && (out.last ?? "").isEmpty) { out.append(line) } }
            .joined(separator: "\n").trimmingCharacters(in: .whitespacesAndNewlines)
    }

    static let writerSystem = "You write short, natural spoken lines for personal videos."
}

/// The canvas, by where the video is going.
enum ShareShape: String, CaseIterable, Identifiable {
    case story, square, wide

    var id: String { rawValue }

    var label: String {
        switch self {
        case .story: "Phone story"
        case .square: "Square post"
        case .wide: "Wide"
        }
    }

    var width: Int {
        switch self {
        case .story: 544
        case .square: 736
        case .wide: 960
        }
    }

    var height: Int {
        switch self {
        case .story: 960
        case .square: 736
        case .wide: 544
        }
    }
}

/// Whether a template can be made now, and where. This Mac when it holds the
/// model and has the memory; otherwise a Mac on the network serving the same
/// pack. A Mac that cannot hold the model is never offered a download it could
/// never use.
enum TemplateReadiness: Equatable {
    /// `seconds` is the time estimate where one is known.
    case ready(seconds: Double?)
    /// The LAN routing id ("model@peer") of the Mac that makes it.
    case onPeer(String)
    case needsDownload
    case tooBig(neededGB: Int)

    static func of(onDisk: Bool, neededBytes: UInt64, usableBytes: UInt64, seconds: Double?,
                   peer: String? = nil) -> TemplateReadiness {
        let fits = usableBytes == 0 || neededBytes <= usableBytes
        if onDisk, fits { return .ready(seconds: seconds) }
        if let peer { return .onPeer(peer) }
        return fits ? .needsDownload : .tooBig(neededGB: Int((Double(neededBytes) / Double(1 << 30)).rounded(.up)))
    }

    /// The first Mac on the network serving the pack `t` is made on.
    static func peer(for t: MediaTemplate, in lanModels: [ModelInfo]) -> String? {
        peer(repo: t.modelRepo, capability: t.lanCapability, in: lanModels)
    }

    static func peer(repo: String, capability: String, in lanModels: [ModelInfo]) -> String? {
        let name = (repo as NSString).lastPathComponent
        return lanModels.first { m in
            m.lanAdvertises(capability) && (LanPick.base(of: m.name) as NSString).lastPathComponent == name
        }?.name
    }
}
