import Foundation

/// A person or pet the templates can put in a video: a name, what they are,
/// how they sound, and one photo.
struct CastMember: Codable, Identifiable, Equatable, Hashable {
    var id = UUID()
    var name: String
    var kind: CastKind
    var voice: CastVoice
    /// Optional words about how they look; they ride the document beside the photo.
    var looks: String = ""
    /// The photo's file name in the cast folder.
    var photo: String
    /// A real person is in the cast only with their say-so.
    var consent = false
    var createdAt = Date()

    /// Stands in for an unfilled slot, so a document always renders.
    static func placeholder(_ kind: CastKind) -> CastMember {
        CastMember(name: kind.label, kind: kind, voice: .default(for: kind), photo: "")
    }
}

enum CastKind: String, Codable, CaseIterable, Identifiable {
    case dog, cat, pet, woman, man, girl, boy

    var id: String { rawValue }

    var label: String {
        switch self {
        case .dog: "Dog"
        case .cat: "Cat"
        case .pet: "Other pet"
        case .woman: "Woman"
        case .man: "Man"
        case .girl: "Girl"
        case .boy: "Boy"
        }
    }

    var isPerson: Bool { ![.dog, .cat, .pet].contains(self) }

    var noun: String { self == .pet ? "pet" : rawValue }

    var they: String {
        switch self {
        case .woman, .girl: "she"
        case .man, .boy: "he"
        default: "they"
        }
    }

    var their: String {
        switch self {
        case .woman, .girl: "her"
        case .man, .boy: "his"
        default: "their"
        }
    }
}

/// How a cast member sounds when a template makes them talk.
enum CastVoice: String, Codable, CaseIterable, Identifiable {
    case goofy, sassy, grumpy, posh, sweet, calm

    var id: String { rawValue }

    var label: String {
        switch self {
        case .goofy: "Goofy guy"
        case .sassy: "Sassy lady"
        case .grumpy: "Grumpy grandpa"
        case .posh: "Posh butler"
        case .sweet: "Sweet little kid"
        case .calm: "Calm narrator"
        }
    }

    /// How the document says it.
    var phrase: String {
        switch self {
        case .goofy: "a warm, goofy, earnest male voice"
        case .sassy: "a dry, sassy, unimpressed female voice"
        case .grumpy: "a gruff, grumpy old man's voice"
        case .posh: "a posh, plummy British butler's voice"
        case .sweet: "a sweet, high little child's voice"
        case .calm: "a calm, warm narrator's voice"
        }
    }

    /// How the chat model is told to write for them.
    var trait: String {
        switch self {
        case .goofy: "goofy, earnest and always hungry"
        case .sassy: "dry, sarcastic and unimpressed"
        case .grumpy: "grumpy and fed up with everything"
        case .posh: "posh, proper and easily scandalised"
        case .sweet: "sweet, excitable and very innocent"
        case .calm: "calm, wise and a little smug"
        }
    }

    static func `default`(for kind: CastKind) -> CastVoice {
        switch kind {
        case .dog: .goofy
        case .cat: .sassy
        case .girl, .boy: .sweet
        default: .calm
        }
    }
}
