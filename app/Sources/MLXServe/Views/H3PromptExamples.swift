import Foundation

/// Prompt guidance for the video pane, per BACKEND.
///
/// The pane used to give LTX advice on every model: a 4–8 sentence hint, a link
/// to LTX's prompting guide and one static LTX-shaped Examples list — shown
/// unchanged while a MiniMax-H3 model was selected. That is the stale
/// cross-model link class, and here it costs more than a wrong URL: H3 is not
/// prompted in prose at all.
///
/// MiniMax runs a hosted prompt-expansion stage (H3-Context-IR) in front of the
/// DiT. It ships no weights, so mlx-serve sends the user's prompt VERBATIM to
/// the text encoder — which means the document H3 was trained and evaluated on
/// is something the user has to write. MiniMax publishes two writing guides in
/// `MiniMaxAI/MiniMax-H3` under `docs/` (the "Prompt tips" link points there);
/// the two formats are:
///
///   base (T2VA / I2VA / FL2VA / L2VA) — three core fields, label inline:
///       integrated_multimodal_description: [Shot 1] …
///       overall_soundscape: …
///       non_diegetic_music: …
///
///   full-reference (ref2va) — six sections, label on its own line, in order:
///       subject_definitions / summary / retention_analysis /
///       detailed_description / overall_soundscape / non_diegetic_music
///
/// A user cannot guess `integrated_multimodal_description:` from anything, so
/// the placeholder names the labels and the examples are written in the format
/// VERBATIM — the same rule the image pane's control-map prompts carry
/// (`ImagePromptExamples`): a reword is a quality change, not a copy edit.
struct VideoPromptExample: Hashable {
    let title: String
    let body: String
}

/// Which prompt format the selected model expects. Derived from the preset's
/// declared capabilities (`VideoModelPreset.promptFormat`), never sniffed from
/// an id.
enum VideoPromptFormat: Hashable {
    case ltx
    /// MiniMax-H3 T2VA / I2VA — the FL2VA packs.
    case h3Base
    /// MiniMax-H3 full-reference — the REF2VA pack.
    case h3Reference
}

enum H3PromptExamples {

    // MARK: - The format contracts

    /// Section labels the base format requires, in the order they must appear.
    static let baseSections = [
        "integrated_multimodal_description:",
        "overall_soundscape:",
        "non_diegetic_music:",
    ]

    /// Section labels the full-reference format requires, in order.
    static let referenceSections = [
        "subject_definitions:",
        "summary:",
        "retention_analysis:",
        "detailed_description:",
        "overall_soundscape:",
        "non_diegetic_music:",
    ]

    static func sections(for format: VideoPromptFormat) -> [String] {
        switch format {
        case .ltx: return []
        case .h3Base: return baseSections
        case .h3Reference: return referenceSections
        }
    }

    /// Do the format's labels all appear, in order? Substring search rather
    /// than line parsing: `detailed_description:` is followed by prose on the
    /// next line, and users paste with their own wrapping.
    static func carriesSections(_ text: String, _ labels: [String]) -> Bool {
        var cursor = text.startIndex
        for label in labels {
            guard let r = text.range(of: label, range: cursor..<text.endIndex) else { return false }
            cursor = r.upperBound
        }
        return true
    }

    // MARK: - Examples

    static func examples(for format: VideoPromptFormat) -> [VideoPromptExample] {
        switch format {
        case .ltx: return ltx
        case .h3Base: return h3Base
        case .h3Reference: return h3Reference
        }
    }

    /// Canonical LTX-style example prompts. Dense, cinematographer-style,
    /// covering four common shot types. The dialogue example matters most: LTX
    /// only generates speech when the spoken words appear in quotes (short
    /// phrases, acting directions between them, per the official prompting
    /// guide) — without it the soundtrack is ambient noise only.
    static let ltx: [VideoPromptExample] = [
        .init(title: "Talking character (dialogue)",
              body: "Medium close-up of a woman in her thirties with short auburn hair, seated at a kitchen table in warm morning light. She looks into the camera and says warmly, \"Good morning. I made coffee — it's still hot.\" She pauses, glancing toward the window, then adds with a small smile, \"Come sit with me for a minute.\" Her voice is clear and natural, speaking English. Soft room tone with a faint clink of a cup. The camera holds steady at eye level."),
        .init(title: "Cinematic character",
              body: "Medium shot of a young woman with dark curly hair and freckles, wearing a beige wool coat, walking slowly down a rain-slicked cobblestone street at dusk. She holds a folded paper map in one hand and glances up at the glowing shop windows. The camera tracks her from the side at eye level, then slowly dollies in as she stops. Warm amber light spills from the windows onto the wet stones, contrasting with the deep blue-grey sky. Light rain falls continuously, catching the light."),
        .init(title: "Nature aerial",
              body: "A wide aerial shot sweeps low over a pine forest at sunrise, mist clinging to the treetops in thick white ribbons. The camera glides forward steadily, revealing a narrow river cutting through the valley below, its surface catching the gold of the early sun. A flock of birds lifts off in a loose spiral. Lighting is soft, warm, and directional from the right. Colors are saturated emerald greens and amber golds."),
        .init(title: "Product close-up",
              body: "Close-up of hands in a sunlit kitchen kneading bread dough on a floured wooden counter. The camera holds steady at a low angle, focused tight on the rhythmic press-and-fold motion. Flour dust rises and catches in the shaft of morning light from a window on the left. The hands belong to an older man in a rolled blue shirt, skin weathered and dusted white. Warm natural backlight, muted earth tones."),
    ]

    /// MiniMax-H3 base format. THREE OF THESE FOUR BODIES ARE VERBATIM
    /// H3-Context-IR OUTPUT — MiniMax's own expansion stage, called with a
    /// one-line request and its reply shipped unedited (2026-08-06). That is
    /// the point: the model was trained on this stage's output, so its own
    /// prose is a better example than any transcription of the guide.
    ///
    /// What they show that hand-writing kept missing: the body runs ~1000-1500
    /// characters for a 5-second shot; soundscapes are physical and specific
    /// ("rhythmic, heavy thuds of the dough impacting the board", not "kitchen
    /// sounds"); the shot opens with style plus a named camera behaviour
    /// ("static shot", "slow push in"); and `non_diegetic_music` states
    /// instrumentation, tempo and dynamics.
    ///
    /// WORDS verbatim, LAYOUT ours: the expansion stage runs each field as one
    /// inline line, which lands in the prompt box as a single ~1500-character
    /// wall — the format is only useful if you can see its parts. Label on its
    /// own line and a blank line between fields, matching `h3Reference` below,
    /// which is also how MiniMax's written guide shows the format.
    ///
    /// The keyframe example is OURS — that mode needs a reference image, which
    /// Context-IR takes as a URL we have nothing to point at.
    static let h3Base: [VideoPromptExample] = [
        .init(title: "Character speaking (dialogue)",
              body: """
integrated_multimodal_description:
[Shot 1] Cinematic, medium close-up, static shot. An on-screen woman in her early thirties (S1) sits in the center of the frame at a rustic wooden kitchen table. She has warm brown eyes, loose wavy brunette hair tied in a messy bun, and wears an oversized, cream-colored chunky knit sweater. The modern farmhouse kitchen behind her features white subway tile, a silver espresso machine, and a hanging pothos plant softly blurred in the background. Bright, warm morning sunlight pours in from an off-frame window to the left, casting a soft, golden glow across the side of her face and illuminating the steam rising from a white ceramic mug resting on the table. Looking directly into the camera lens with a relaxed, welcoming smile, the woman (S1) leans forward slightly, gently resting her forearms on the table, and says in a warm, conversational tone, <d>[English] Good morning. I made coffee. Come sit with me.</d> As she finishes speaking, she tilts her head slightly and maintains her gaze, her shoulders visibly relaxing.

overall_soundscape:
Faint, continuous ambient chirping of birds sets a morning tone, accompanied by the distinct rustle of heavy knit fabric as the woman shifts forward. A subtle, dull thud is heard as her forearms rest against the wooden table, followed by her clear, foreground spoken dialogue.

non_diegetic_music:
Solo acoustic guitar, slow tempo, playing a sparse and repetitive fingerpicking pattern with soft dynamics that sits quietly beneath the dialogue.
"""),
        .init(title: "Scene with its own soundtrack",
              body: """
integrated_multimodal_description:
[Shot 1] Cinematic, low aerial wide shot, the camera continuously moving forward while it pedestals up and rolls gently clockwise over a rugged, dark volcanic coastline. Warm, low-angle golden hour sunlight streams in from the left, casting long shadows and highlighting the jagged textures of parallel black basalt ribs that stretch from the bottom foreground out into the vast, deep blue-green ocean. Heavy, rhythmic ocean swells roll in from the background, colliding violently with the solid rock formations. The continuous impact sends explosive bursts of bright white surf and fine, golden-lit mist high into the air. As the camera steadily climbs and banks, the turbulent white water churns vigorously, spilling back over the dark, grooved stones into the sea, revealing a wider expanse of the jagged shoreline fading into the horizon under the vibrant sunset light.

overall_soundscape:
A pronounced, continuous roar of heavy wind rushing past the microphone, layered over the deep, booming thuds of massive ocean swells crashing against solid rock, followed immediately by the loud, hissing splash of churning surf and water continuously cascading back into the sea.

non_diegetic_music:
N/A
"""),
        .init(title: "Close-up with texture and sound",
              body: """
integrated_multimodal_description:
[Shot 1] Cinematic, close-up shot with a slow push in, framing a thick, worn oak wooden counter generously dusted with fine white flour. An older man's weathered, wrinkled hands, emerging from faded blue rolled-up linen sleeves, dominate the center of the frame. The hands rhythmically press into a smooth, elastic ball of bread dough with the heels of the palms, folding the soft mass over itself and giving it a quarter turn with steady, practiced precision. Strong, warm morning sunlight streams in diagonally from the upper left, casting deep, textured shadows and intensely backlighting the scene. The directional light catches vibrant, powdery clouds of flour dust that float and swirl gracefully in the air above the counter. With each firm push of the dough, a small puff of dry flour scatters across the wooden surface, and the elastic dough slightly springs back as the hands momentarily release their pressure, while the softly blurred background implies the rustic warmth of a traditional kitchen.

overall_soundscape:
Rhythmic, heavy thuds of the dough impacting the solid wooden board dominate the foreground, accompanied by the dry, powdery friction of skin sliding over scattered flour. A faint, slightly sticky stretching sound is clearly heard each time the dough is folded back on itself, layered over a quiet, airy room tone.

non_diegetic_music:
Solo acoustic guitar, slow tempo, featuring warm, steady fingerpicking that gently underscores the steady rhythm of the kneading.
"""),
        .init(title: "From an attached first frame",
              body: """
For the target video, at 0.00 seconds into the target video, <Picture 1> (from [Shot 1]) is fully referenced.

integrated_multimodal_description:
[Shot 1] Cinematic, medium shot with a slow push in, opening on the exact composition, subjects and lighting of <Picture 1> and holding their appearance throughout. The subject begins to move within the frame while the light direction, colour palette and background stay continuous with the reference image. The camera pushes in with small amplitude at slow speed, closing the distance without cutting, so the framing tightens gradually around the subject as the action develops. Small physical details continue moving throughout: fabric settles, hair shifts, and the light on the subject's face changes slightly as the distance closes. The shot holds its final framing for the last moment without a cut.

overall_soundscape:
A quiet, continuous room tone matching the location, with the close, dry sounds of the subject's own movement in the foreground: fabric brushing, a soft footfall, and the faint rustle of clothing as the pose changes.

non_diegetic_music:
N/A
"""),
    ]

    /// MiniMax-H3 full-reference (REF2VA). Six sections in order. These four are
    /// OURS — Context-IR takes references as URLs, and we have nothing public to
    /// point it at — but they are written to the profile its real output shows
    /// (`h3Base` above): `detailed_description` is the bulk of the document at
    /// ~1100-1300 characters, camera behaviour is named rather than implied, and
    /// the soundscape is physical.
    ///
    /// The section that matters most is `retention_analysis`: it is the only
    /// place the model is told what to DO with a reference, and its markers are
    /// fixed strings — visible content takes `fully_preserved` /
    /// `partially_preserved` / `attribute_transfer` / `weak_reference`, audio
    /// takes `fully_copy` / `partially_copy` / `reference` / `weak_reference`.
    /// Attaching a clip and never stating its role leaves the model
    /// conditioning rows with nothing said about them. Measured 2026-08-06:
    /// with a visual reference attached, output moves toward it (colour
    /// histogram correlation 0.81-0.95 against 0.56-0.67 without), and the
    /// effect scales with how many reference rows are attached.
    static let h3Reference: [VideoPromptExample] = [
        .init(title: "Keep a subject (image reference)",
              body: """
subject_definitions:
<Subject 1> is the young woman in <Picture 1>, with long dark hair, a blue cardigan and a thin silver necklace.

summary:
[reference generation] The target video is a single shot of <Subject 1> reading at a cafe window as afternoon light moves across the table.

retention_analysis:
<Subject 1> (appears in [Shot 1]): fully_preserved - her face, long dark hair, blue cardigan and silver necklace are retained.

detailed_description:
The target video is in a realistic cinematic style with warm afternoon window light.
[Shot 1] A medium shot with a slow push in frames <Subject 1> seated beside a cafe window at a small marble table, an open paperback held in both hands. Warm directional light enters from the left through the glass and falls across the table, the rim of a white cup and the side of her face, leaving the room behind her in soft shadow. The thin silver necklace catches the light each time she shifts her weight. She turns a page with her right hand, then glances up toward the street where blurred figures pass outside the glass. Steam rises steadily from the cup beside her and drifts through the shaft of light. She settles back into the chair, rests one forearm on the marble and returns to the page. The camera pushes in with small amplitude at slow speed, tightening from her shoulders to her face and hands while the background compresses behind her. Toward the end of the shot the light shifts perceptibly warmer as cloud clears outside, deepening the contrast on the knit of the cardigan.

overall_soundscape:
Quiet cafe room tone with the dry rustle of a turning page close to the camera, distant cutlery against ceramic, the low hiss and clunk of an espresso machine further back, and muffled street traffic through the glass.

non_diegetic_music:
N/A
"""),
        .init(title: "Follow a clip's look and pacing (video reference)",
              body: """
subject_definitions:
<Subject 1> is the coastal landscape in <Video 1>, with dark ridged rock running seaward, white surf along its edges and deep blue-green water beyond.
<Video 1> is the reference for the target video's camera movement and pacing, a continuous aerial pass with no cuts.

summary:
[reference generation] The target video is a low aerial shot moving forward over <Subject 1> at golden hour, following the continuous unbroken camera movement of <Video 1>.

retention_analysis:
<Subject 1> (appears in [Shot 1]): fully_preserved - the dark ridged rock, the white surf line along its edges, the deep blue-green water and the low warm light are retained.
<Video 1> (camera movement and pacing structure): weak_reference - only the continuous single-take aerial pacing is followed; the target video is not an edit of it.

detailed_description:
The target video is in a realistic aerial cinematography style with warm low-angle golden-hour light.
[Shot 1] A low aerial tracking shot glides forward over <Subject 1>, whose parallel ridges run seaward into deep blue-green water. Warm directional sunlight enters from the upper left and rakes across the wet rock, picking out the wide flat tops of the ridges and throwing long shadows into the channels between them. White surf breaks along each edge in sequence, foaming through the gaps and draining back in bright threads as the next set arrives. The camera pedestals up steadily and rolls a few degrees counterclockwise at slow speed, opening the frame onto more open water while the ridges continue past the lower edge. Swell moves through the channels in a slow rhythm and spray lifts briefly where a larger set collides with the outermost rock, catching the light as fine golden mist that drifts inland before falling back. Between sets the water drains off the flat tops in thin bright sheets, leaving the rock dark and wet where the light no longer catches it. The low warm light holds through the end of the shot without a cut.

overall_soundscape:
A continuous roar of wind across the microphone over the deep booming impact of heavy swell against solid rock, each impact followed by the loud hiss of churning surf draining back through the channels, with distant gulls high above.

non_diegetic_music:
N/A
"""),
        .init(title: "Reuse a soundtrack (audio reference)",
              body: """
subject_definitions:
<Audio 1> is the ambience reference for the target video, containing rain on glass, distant traffic and a low interior room tone.

summary:
[audio reference] The target video is a single interior shot of a rain-streaked window at night, with its ambience following <Audio 1>.

retention_analysis:
<Audio 1>: reference - the target video's ambience follows the rain density, traffic distance and room tone of <Audio 1> without copying the original signal.

detailed_description:
The target video is in a realistic cinematic style with low-key interior light.
[Shot 1] A medium static shot holds on a rain-streaked window at night, the room dark except for a warm lamp out of frame to the right that catches the sill and one edge of the glass. Streetlights outside break into soft coloured circles through the water, amber nearest the camera and cold white further off. Individual drops gather at the top of the pane, hold, then run down in irregular lines that swallow smaller drops as they go, leaving bright trails that slowly refill. A car passes below and its headlights sweep across the glass, briefly resolving the pattern of water into sharp points before the room returns to its darker state. Behind the glass the room stays almost black, with only the lamp's warm edge picking out the sill and a single reflected highlight that shifts as the water moves. The camera then pushes in with small amplitude at slow speed until the reflections fill the frame and the window frame itself falls outside it, the coloured circles growing and softening as focus settles on the nearest running drops, holding on the moving water through the end of the shot.

overall_soundscape:
Steady rain against glass in the near foreground with an irregular tap as heavier drops strike the sill, a low continuous interior room tone beneath it, and the muffled wet hiss of passing traffic rising and falling in the distance, following the texture and density of <Audio 1>.

non_diegetic_music:
N/A
"""),
        .init(title: "Subject + clip + audio (all three)",
              body: """
subject_definitions:
<Subject 1> is the young man in <Picture 1>, with short wavy brown hair, a dark-grey hooded jacket and a canvas bag over one shoulder.
<Video 1> is the reference for the target video's camera movement and pacing, a handheld tracking shot that follows its subject from the side.
<Audio 1> is the ambience reference for the target video, containing wet street traffic, footsteps and light rain.

summary:
[reference generation + audio reuse] The target video follows <Subject 1> walking along a rain-slicked street at dusk, using the handheld side-tracking movement of <Video 1> and reusing the ambience of <Audio 1>.

retention_analysis:
<Subject 1> (appears in [Shot 1]): fully_preserved - his face, short wavy brown hair, dark-grey hooded jacket and canvas bag are retained.
<Video 1> (camera movement and pacing structure): weak_reference - only the handheld side-tracking movement and its unhurried pacing are followed; the target video is not an edit of it.
<Audio 1>: partially_copy - the wet-street traffic and rain ambience is copied as the target video's ambience bed, with the subject's own footsteps added.

detailed_description:
The target video is in a realistic cinematic style with cool dusk light and warm shop windows.
[Shot 1] A handheld tracking shot follows <Subject 1> from the side at eye level as he walks along a rain-slicked pavement at dusk, hands pushed into his pockets and the canvas bag swinging slightly against his hip. Warm amber light spills from shop windows onto the wet stones and pools in the standing water, contrasting with the deep blue-grey of the sky above the rooftops. Fine rain falls continuously, visible where it crosses the lit windows. He glances toward the glass without slowing, his breath faintly visible in the cold. The camera keeps pace beside him with small amplitude, the frame drifting a few degrees as the operator walks, then falls back slightly as he passes a lit doorway so that the warm interior light crosses his face and shoulders. A cyclist passes behind him in the opposite direction. The shot holds on him walking away into the deeper part of the street as the shop lights fall behind.

overall_soundscape:
The copied ambience layer from <Audio 1> - the wet hiss of tyres on soaked asphalt and light rain across the whole frame - continues throughout, with his own footfalls landing close to the camera in shallow water and the soft creak of the canvas strap against his jacket.

non_diegetic_music:
N/A
"""),
    ]

    // MARK: - Placeholder + hint + tips link

    /// Placeholder shown in the empty prompt field. For H3 it NAMES the section
    /// labels: nothing else in the UI can tell the user that the model wants
    /// `integrated_multimodal_description:`, and a format they can't guess is
    /// worse than one they merely write badly.
    static func placeholder(for format: VideoPromptFormat) -> String {
        switch format {
        case .ltx:
            return "Describe your shot like a cinematographer — subject, action, camera movement, lighting, setting. 4–8 sentences. Put spoken dialogue in quotes to make characters talk. Click Examples above for a starting point."
        // Kept to a few lines: this sits INSIDE the 110pt prompt editor, so a
        // section-per-line list is clipped exactly where the last labels are.
        // The labels are named here; the Examples menu carries the full shape.
        case .h3Base:
            return "MiniMax-H3 expects three labelled fields — integrated_multimodal_description: (the shot, its style, action and camera movement), overall_soundscape: (ambience and physical sound), non_diegetic_music: (score only the audience hears, or N/A). Click Examples above for the exact shape."
        case .h3Reference:
            return "MiniMax-H3 REF2VA expects six labelled sections in order — subject_definitions:, summary:, retention_analysis: (what to DO with each reference), detailed_description:, overall_soundscape:, non_diegetic_music:. Refer to attachments as <Picture 1>, <Video 1>, <Audio 1>. Click Examples above."
        }
    }

    /// The soft caption under the prompt field. Two distinct cases, because
    /// they need different fixes: a prompt that is too SHORT, and an H3 prompt
    /// that is long enough but carries none of the format's labels.
    ///
    /// The label check is deliberately weak — the FIRST label only — so a user
    /// part-way through writing the document isn't nagged, and a prompt written
    /// in the format is never flagged.
    static func hint(for format: VideoPromptFormat, prompt: String) -> String? {
        let words = prompt.split(whereSeparator: { $0.isWhitespace || $0.isNewline }).count
        guard words > 0 else { return nil }
        switch format {
        case .ltx:
            guard words < 15 else { return nil }
            return "LTX-Video performs best with detailed 4–8 sentence prompts. Try Examples or Prompt tips above."
        case .h3Base:
            if let first = baseSections.first, !prompt.contains(first) {
                return "MiniMax-H3 was trained on labelled prompts. Start with “\(first)”, then overall_soundscape: and non_diegetic_music:. Try Examples or Prompt tips above."
            }
            guard words < 25 else { return nil }
            return "MiniMax-H3 was trained on detailed shot descriptions — style, action, camera movement and sound. Try Examples above."
        case .h3Reference:
            if let first = referenceSections.first, !prompt.contains(first) {
                return "MiniMax-H3 REF2VA was trained on six labelled sections starting with “\(first)”. retention_analysis: is what states each reference's role. Try Examples or Prompt tips above."
            }
            if !prompt.contains("retention_analysis:") {
                return "Add retention_analysis: — it is the only place the model is told what to do with each reference."
            }
            guard words < 25 else { return nil }
            return "MiniMax-H3 REF2VA was trained on detailed shot descriptions. Try Examples above."
        }
    }

    /// Where "Prompt tips" points. LTX's own guide for LTX; for H3 the two
    /// writing guides MiniMax ships, copied into the repo — a link to LTX's
    /// site while an H3 model is selected is advice for a different engine.
    static func tipsURL(for format: VideoPromptFormat) -> URL {
        switch format {
        case .ltx:
            return URL(string: "https://docs.ltx.video/api-documentation/prompting-guide")!
        case .h3Base, .h3Reference:
            // MiniMax's OWN guides, not a copy of them in our repo: they are
            // the spec for a stage we do not run, they get updated by the
            // people who run it, and redistributing someone else's docs to
            // save a click is how a link goes stale.
            return URL(string: "https://huggingface.co/MiniMaxAI/MiniMax-H3/tree/main/docs")!
        }
    }
}
