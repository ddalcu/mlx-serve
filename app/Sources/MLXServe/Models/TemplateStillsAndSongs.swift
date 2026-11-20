import Foundation

/// The picture and song templates: the model each kind is made on, the edit
/// prompt that puts a cast member into a new scene, and the brief a chat model
/// writes song lyrics from.
extension MediaTemplate {
    var modelRepo: String { bundle.primaryRepo }

    var bundle: MediaBundle {
        switch kind {
        case .video: Self.videoModel.bundle
        case .image: Self.imageModel.bundle
        case .audio: Self.songModel.bundle
        }
    }

    /// What a Mac on the network advertises for the kind (`ModelInfo.lanAdvertises`).
    var lanCapability: String {
        switch kind {
        case .video: "video"
        case .image: "image"
        case .audio: "music"
        }
    }

    // MARK: - Pictures

    /// The edit instruction: the scene, and the person or pet from the photo
    /// kept as they are. Never the meme text: the model would misspell it, so
    /// it is burned on afterwards.
    func imagePrompt(_ fill: TemplateFill) -> String {
        let keep = Self.keep(fill.cast["who"])
        switch self {
        case .portrait:
            return "Turn this into a grand Renaissance oil painting: \(keep), posed as a pompous royal noble in a velvet robe with an ermine collar, a jewelled crown and a gold medallion, seated on a red velvet throne in a candlelit palace hall, dramatic chiaroscuro lighting, visible brush strokes, ornate gold frame."
        case .headshot:
            return "Turn this into a professional corporate headshot: \(keep), wearing a navy business suit, crisp white shirt and a red tie, sitting upright and confident, soft grey studio backdrop, flattering studio lighting, sharp focus, LinkedIn profile photo."
        case .poster:
            return "Turn this into a blockbuster action movie poster: \(keep), as the hero in a heroic low-angle pose wearing a black tactical jacket and aviator sunglasses, a huge fiery explosion and a helicopter behind, dramatic orange and teal lighting, cinematic, no text."
        case .figure:
            let gear = TemplateText.clean(fill.text("gear"))
            return "Turn this into a product photo of a collectible plastic action figure in its retail blister pack: \(keep), as a glossy toy action figure standing in a moulded clear plastic bubble on a bright colourful cardboard backing card, with small plastic accessories in their own slots beside it\(gear.isEmpty ? "" : ": \(gear)"), studio product lighting, sharp focus, no text, no logos."
        default:
            return ""
        }
    }

    /// Who the edit keeps, and what of them.
    private static func keep(_ who: CastMember?) -> String {
        let m = who ?? CastMember.placeholder(.dog)
        let looks = TemplateText.clean(m.looks)
        return "the same \(m.kind.noun) from the photo, keeping \(m.kind.their) exact face"
            + (m.kind.isPerson ? " and hair" : ", fur colours and markings")
            + (looks.isEmpty ? "" : ", including \(looks)")
    }

    // MARK: - Songs

    /// The edit that makes a song's cover, or nil to use their photo as it is.
    func songCoverPrompt(_ fill: TemplateFill) -> String? {
        guard self == .roast else { return nil }
        return "Turn this into a hip-hop album cover: \(Self.keep(fill.cast["who"])), as a confident rapper wearing a backwards cap, oversized sunglasses, a hoodie and heavy gold chains, holding a microphone and striking a swagger pose, a graffiti wall and city lights behind, purple and gold neon lighting, gritty and bold, no text."
    }

    /// ACE-Step's style caption: genre, instruments, voice and tempo.
    func songStyle(_ fill: TemplateFill) -> String {
        let voice = fill.cast["who"].map { [.woman, .girl].contains($0.kind) ? "female" : "male" } ?? "male"
        switch self {
        case .birthday:
            return "upbeat feel-good pop birthday song, bright acoustic guitar, hand claps, bouncy bass, party sing-along chorus, joyful female lead vocal, 120 bpm"
        case .theme:
            return "1980s TV sitcom opening theme song, cheesy saxophone, bright synth, slap bass, upbeat drums, warm male lead vocal with backing choir, nostalgic, 118 bpm"
        case .roast:
            return "playful old-school hip-hop roast track, punchy boom-bap drums, funky bass, record scratches, confident witty \(voice) rap vocal, crowd laughing and cheering, 95 bpm"
        case .ballad:
            return "over-the-top 1980s power ballad, dramatic piano intro, huge drums, soaring electric guitar solo, epic emotional \(voice) rock vocal, slow 70 bpm"
        default:
            return ""
        }
    }

    func songAsk(_ fill: TemplateFill) -> String {
        let m = fill.cast["who"] ?? CastMember.placeholder(.dog)
        let about: String
        switch self {
        case .birthday:
            about = "a happy birthday song for \(m.name), who loves \(fill.text("love"))"
        case .theme:
            about = "a 1980s TV sitcom theme song about \(m.name), a \(m.kind.noun) known for \(fill.text("known"))"
        case .roast:
            about = "a playful hip-hop roast rap about \(m.name), teasing them for \(fill.text("tease")), all in good fun with no swearing or insults about looks"
        default:
            about = "an over-the-top dramatic power ballad, sung by \(m.name) in the first person, about this tragedy: \(fill.text("what"))"
        }
        return "Write the lyrics of \(about). Use the name \(m.name) in the chorus. Two sections, each starting with its tag on its own line: "
            + "[verse] with 4 lines, then [chorus] with 4 lines. Short, simple lines that rhyme, funny and family friendly. "
            + "Answer with only a JSON object with the string keys title and lyrics, where lyrics uses \\n line breaks."
    }

    func songFallback(_ fill: TemplateFill) -> [String: String] {
        let name = fill.cast["who"]?.name ?? "my friend"
        switch self {
        case .birthday:
            return ["title": "Happy Birthday \(name)", "lyrics": """
                [verse]
                Wake up \(name), the sun is out
                Candles waiting, there is no doubt
                \(fill.text("love").capitalizedFirst) and all your friends
                This is the day that never ends

                [chorus]
                Happy birthday \(name), hey hey hey
                This is your day, your day
                Cake on the table, friends at the door
                Happy birthday \(name), let's sing it some more
                """]
        case .theme:
            return ["title": "It's \(name)!", "lyrics": """
                [verse]
                Every morning, every night
                Someone's up to something, that's right
                Known for \(fill.text("known"))
                The whole street knows the sound

                [chorus]
                \(name)! Here comes \(name)!
                Nobody does it quite the same
                When the doorbell rings, you know the rest
                It's \(name), the very best
                """]
        case .roast:
            return ["title": "The \(name) Roast", "lyrics": """
                [verse]
                Yo, gather round, I got a story to tell
                About \(name), you know the legend well
                \(fill.text("tease").capitalizedFirst), every single day
                We love you anyway, but hey, hey, hey

                [chorus]
                \(name), \(name), what you gonna do
                The whole crew laughing, and it's all about you
                \(name), \(name), take a bow
                You got roasted, and you're famous now
                """]
        default:
            return ["title": "The Ballad of \(name)", "lyrics": """
                [verse]
                I had it all this morning, I thought it would last
                But \(fill.text("what")), and it happened so fast

                [chorus]
                Oh why, oh why, does it hurt me so
                \(name) is alone with nowhere to go
                The world kept turning, the sky turned blue
                But my heart is broken, what can I do
                """]
        }
    }
}

extension ShareShape {
    /// A picture's pixels: the shape at a 1024 long side, on the editors' 32-pixel grid.
    var imageSize: (width: Int, height: Int) {
        let long = 1024.0, w = Double(width), h = Double(height)
        let snap = { (v: Double) in Int((v / 32).rounded()) * 32 }
        return w >= h ? (1024, snap(long * h / w)) : (snap(long * w / h), 1024)
    }
}

private extension String {
    var capitalizedFirst: String { prefix(1).uppercased() + dropFirst() }
}
