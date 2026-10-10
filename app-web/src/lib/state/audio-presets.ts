import { t, N } from "../i18n/i18n";
const kokoroVoices = [
  "af_alloy",
  "af_aoede",
  "af_bella",
  "af_heart",
  "af_jessica",
  "af_kore",
  "af_nicole",
  "af_nova",
  "af_river",
  "af_sarah",
  "af_sky",
  "am_adam",
  "am_echo",
  "am_eric",
  "am_fenrir",
  "am_liam",
  "am_michael",
  "am_onyx",
  "am_puck",
  "am_santa",
  "bf_alice",
  "bf_emma",
  "bf_isabella",
  "bf_lily",
  "bm_daniel",
  "bm_fable",
  "bm_george",
  "bm_lewis",
  "ef_dora",
  "em_alex",
  "em_santa",
  "ff_siwis",
  "hf_alpha",
  "hf_beta",
  "hm_omega",
  "hm_psi",
  "if_sara",
  "im_nicola",
  "jf_alpha",
  "jf_gongitsune",
  "jf_nezumi",
  "jf_tebukuro",
  "jm_kumo",
  "pf_dora",
  "pm_alex",
  "pm_santa",
  "zf_xiaobei",
  "zf_xiaoni",
  "zf_xiaoxiao",
  "zf_xiaoyi",
  "zm_yunjian",
  "zm_yunxi",
  "zm_yunxia",
  "zm_yunyang",
];
const trackClasses = [
  "woodwinds",
  "brass",
  "fx",
  "synth",
  "strings",
  "percussion",
  "keyboard",
  "guitar",
  "bass",
  "drums",
  "backing_vocals",
  "vocals",
];
const sectionTags = [
  "[intro]",
  "[verse]",
  "[pre-chorus]",
  "[chorus]",
  "[post-chorus]",
  "[bridge]",
  "[instrumental]",
  "[solo]",
  "[outro]",
];
const musicStyles = [
  {
    title: N("Synthwave"),
    body: "Upbeat 80s synthwave with a driving analog bass line, dreamy lush pads, punchy retro drum machine, and a catchy soaring lead synth melody. Retro-futuristic, neon night-drive mood.",
  },
  {
    title: N("Lo-fi study beats"),
    body: "Chill lo-fi hip hop with a dusty vinyl texture, warm mellow Rhodes piano chords, soft boom-bap drums, gentle tape saturation, and a relaxed nostalgic late-night mood.",
  },
  {
    title: N("Epic orchestral"),
    body: "Epic cinematic orchestral trailer music with thunderous taiko drums, soaring string ostinatos, heroic brass fanfares, and a choir swelling to a triumphant climax.",
  },
  {
    title: N("Modern pop"),
    body: "Bright modern pop with punchy drums, shimmering synths, a bouncy bass groove, and a catchy female vocal hook. Radio-ready, feel-good summer energy.",
  },
  {
    title: N("Acoustic folk"),
    body: "Intimate acoustic folk ballad with fingerpicked guitar, soft warm male vocals, gentle harmonies, and a touch of cello. Quiet fireside atmosphere, honest and tender.",
  },
  {
    title: N("Stadium rock"),
    body: "High-energy stadium rock anthem with crunchy electric guitars, pounding drums, a driving bass, and powerful raspy male vocals building into a huge singalong chorus.",
  },
  {
    title: N("Deep house"),
    body: "Warm deep house with a rolling four-on-the-floor kick, deep sub bass, silky filtered chords, soft vocal chops, and a hypnotic late-night club groove.",
  },
  {
    title: N("Jazz trio"),
    body: "Relaxed late-night jazz trio with brushed drums, walking upright bass, and warm improvised piano. Smoky lounge atmosphere, tender and swinging.",
  },
  {
    title: N("Trap"),
    body: "Hard-hitting trap beat with booming 808 bass, crisp rattling hi-hats, dark atmospheric bells, and punchy snappy snares. Confident, cinematic, modern.",
  },
  {
    title: N("Cinematic ambient"),
    body: "Slow cinematic ambient with evolving warm synth pads, distant piano, soft field-recording textures, and a gentle emotional swell. Spacious, reflective, calm.",
  },
];
const music3Styles = [
  {
    title: N("Synthwave"),
    body: "Global Metadata\nBasic Attributes: bpm is 118. key is F, and scale is minor. Retro Synthwave / Outrun.\nGlobal Emotional Progression: Opens cool and mechanical, warms through the first chorus into something wistful and wide-eyed, peaks in a euphoric final chorus, then empties out into calm.\nApplication Scenarios & Imagery: A night drive along an empty coastal highway, neon signs smearing past the windscreen.\nSonics & Production Profile: Wide analog stereo field, warm saturated low end, glassy chorused highs, gated reverb on the snare, a light tape flutter across the whole mix.\n\nVocal Details\nVocal Gender & Timbre: A single female lead, breathy and cool in a comfortable mid register, slightly detached.\nVocal Style: Long sustained phrases riding over the pulse, conversational in the verses and fully open in the choruses.\nHarmony/Backing Vocals: Octave-up doubles and soft airy thirds in the choruses only.\nVocal FX: Lush plate reverb and a slow quarter-note delay, always kept intelligible.\n\nArrangement\nInstrument Lifecycle Description (Primary/Secondary Layering):\nPrimary: An arpeggiated analog synth runs from the first bar to the last, joined by a soaring lead synth in the choruses.\nSecondary: Analog bass and a punchy drum machine enter at the first verse, warm pads sit underneath, extra percussion arrives for the final chorus.\nGroove & Foundation Progression: A steady eighth-note pulse under a driving four-on-the-floor kick, with the bass locking to the arpeggio and opening into longer notes in the chorus.\nEmbellishments, Textures & Spatial FX: Slow filter sweeps open each section, a snare fill lifts the pre-chorus, the bridge drops the drums for a single held pad, and the outro strips back to the arpeggio and tape hiss.",
  },
  {
    title: N("Lo-fi study beats"),
    body: "Global Metadata\nBasic Attributes: bpm is 82. key is Eb, and scale is major. Lo-fi Hip Hop / Chillhop.\nGlobal Emotional Progression: Nostalgic and unhurried from the first bar, drifting a little brighter in the middle and settling back into stillness.\nApplication Scenarios & Imagery: Rain on a window at 2am, a desk lamp, a half-finished page.\nSonics & Production Profile: Narrow warm mix, rolled-off highs, dusty vinyl crackle throughout, gentle tape saturation and soft sidechain breathing on the pads.\n\nVocal Details\nVocal Gender & Timbre: Largely instrumental — the Rhodes piano carries the melody.\nVocal Style: Occasional wordless female vocal fragments, soft and half-buried.\nHarmony/Backing Vocals: None beyond those fragments.\nVocal FX: Heavy low-pass and long reverb, used as texture rather than as a lead.\n\nArrangement\nInstrument Lifecycle Description (Primary/Secondary Layering):\nPrimary: Warm mellow Rhodes chords with a loose swung feel, present the whole track.\nSecondary: Soft boom-bap drums and a round upright bass enter after the intro; a muted guitar figure joins in the middle third and drops away for the outro.\nGroove & Foundation Progression: Laid-back swung sixteenths, snare slightly behind the beat, bass walking simply under the chord changes without ever pushing the tempo.\nEmbellishments, Textures & Spatial FX: Vinyl crackle and distant room tone run underneath, a tape-stop marks the halfway point, and the final bars fade to crackle alone.",
  },
  {
    title: N("Epic orchestral"),
    body: "Global Metadata\nBasic Attributes: bpm is 96. key is D, and scale is minor. Cinematic Orchestral / Trailer.\nGlobal Emotional Progression: Ominous and sparse at the start, gathering tension in waves, breaking into a triumphant, overwhelming climax before a quiet, exhausted resolution.\nApplication Scenarios & Imagery: An army cresting a ridge at dawn; the last shot before the credits.\nSonics & Production Profile: Huge concert-hall depth, deep sub-heavy percussion, wide string sections panned across the stage, generous natural reverb tails and enormous dynamic range.\n\nVocal Details\nVocal Gender & Timbre: A mixed choir, no soloist — deep basses under bright sopranos.\nVocal Style: Wordless sustained vowels in the build, shifting to hard rhythmic syllables at the climax.\nHarmony/Backing Vocals: Dense block harmony doubling the brass line an octave above.\nVocal FX: Nothing beyond the hall — the space is the effect.\n\nArrangement\nInstrument Lifecycle Description (Primary/Secondary Layering):\nPrimary: Low strings open alone with a repeating ostinato that survives every section and returns at the end.\nSecondary: Taiko drums and low brass enter for the first build, horns and full strings for the second, choir and cymbals at the climax, solo cello for the resolution.\nGroove & Foundation Progression: A driving ostinato pulse that doubles in rhythmic density each build, with the percussion moving from a slow half-time thud into hammered eighth notes.\nEmbellishments, Textures & Spatial FX: Rising string runs and braams mark each transition, a full silence lands one bar before the climax, and the last thirty seconds decay into a single sustained note.",
  },
  {
    title: N("Modern pop"),
    body: "Global Metadata\nBasic Attributes: bpm is 112. key is A, and scale is major. Contemporary Pop / Dance Pop.\nGlobal Emotional Progression: Bright and confident from the first line, building anticipation through the pre-chorus and landing on an open, celebratory chorus that gets bigger every time it returns.\nApplication Scenarios & Imagery: Windows down in late summer, a city that stays warm after dark.\nSonics & Production Profile: Loud, tight and radio-forward — punchy compressed drums, a clean wide top end, shimmering synths and a bass that sits right under the vocal.\n\nVocal Details\nVocal Gender & Timbre: A female lead, bright and forward with a light rasp at the top of her range.\nVocal Style: Rhythmic and clipped in the verses, sustained and belted in the choruses.\nHarmony/Backing Vocals: Stacked thirds and fifths on every chorus line, plus short answering ad-libs between phrases.\nVocal FX: Tight doubling, a clean short slap delay and a bright plate — polished but not processed into glass.\n\nArrangement\nInstrument Lifecycle Description (Primary/Secondary Layering):\nPrimary: A bouncy synth bass and the lead vocal drive the whole song.\nSecondary: Muted plucks and finger snaps carry the verses, full drums and shimmering pad chords land at each chorus, a guitar counter-line appears only in the last chorus.\nGroove & Foundation Progression: A four-on-the-floor kick with an offbeat bass bounce, the verse groove stripped to kick and snaps and the chorus opening into full kit with a clap layer.\nEmbellishments, Textures & Spatial FX: Riser sweeps and a drum drop-out set up each chorus, the bridge cuts to vocal and pad alone, and the final chorus adds an octave-up vocal stack.",
  },
  {
    title: N("Acoustic folk"),
    body: "Global Metadata\nBasic Attributes: bpm is 72. key is G, and scale is major. Acoustic Folk / Singer-Songwriter.\nGlobal Emotional Progression: Intimate and tentative at first, growing in warmth and certainty as the arrangement fills, then returning to the quiet it started in.\nApplication Scenarios & Imagery: A wooden room, one microphone, rain outside and a fire that has been going a while.\nSonics & Production Profile: Close, dry and honest — natural room sound, minimal compression, full mid-range, breaths and string noise left in.\n\nVocal Details\nVocal Gender & Timbre: A male lead, soft and warm in a low-to-mid register, a little air on every phrase.\nVocal Style: Storytelling delivery, close to the microphone, gentle and unhurried, lifting only slightly at the emotional peak.\nHarmony/Backing Vocals: A single female harmony a third above, entering at the second chorus and staying to the end.\nVocal FX: A short natural room reverb, nothing else.\n\nArrangement\nInstrument Lifecycle Description (Primary/Secondary Layering):\nPrimary: Fingerpicked steel-string acoustic guitar from the first note to the last.\nSecondary: An upright bass joins the first chorus, a cello enters in the second verse and holds long notes underneath, brushed drums arrive only for the final chorus.\nGroove & Foundation Progression: A steady fingerpicked pattern acts as the pulse, with the bass outlining the roots and the brushes adding a soft swing at the end rather than a beat.\nEmbellishments, Textures & Spatial FX: Occasional harmonics and a slide into each chorus, one bar of guitar alone before the last verse, and a final chord left to ring out into the room.",
  },
  {
    title: N("Trap"),
    body: "Global Metadata\nBasic Attributes: bpm is 140. key is C#, and scale is minor. Trap / Dark Cinematic Hip Hop.\nGlobal Emotional Progression: Cold and menacing from the opening bar, tightening as the hats accelerate, opening into a wide, cinematic hook and dropping back to near silence at the close.\nApplication Scenarios & Imagery: Empty parking garage, headlights, rain on concrete.\nSonics & Production Profile: Sub-heavy and modern — long distorted 808s, sharp transient drums, a dark and mostly empty midrange, wide reverb on the melodic layer only.\n\nVocal Details\nVocal Gender & Timbre: A male lead, low and half-spoken, with a relaxed confident delivery.\nVocal Style: Rhythmic triplet phrasing in the verses, longer melodic lines on the hook.\nHarmony/Backing Vocals: Doubled hook lines an octave down and short ad-libs answering the ends of phrases.\nVocal FX: Light autotune on the hook only, tight doubling and a dark reverb.\n\nArrangement\nInstrument Lifecycle Description (Primary/Secondary Layering):\nPrimary: A booming 808 bass and a dark bell melody carry the track end to end.\nSecondary: Rattling hi-hats and snappy snares enter after the intro, a low string pad joins for the hook, and a detuned pluck answers the vocal in the second verse.\nGroove & Foundation Progression: Half-time snare on the third beat against triplet hat rolls that double in speed into each hook, the 808 sliding between roots rather than repeating them.\nEmbellishments, Textures & Spatial FX: Reverse cymbals mark transitions, the beat cuts out entirely for two bars before the final hook, and the outro leaves the bell melody alone in the reverb.",
  },
];
const musicLyrics = [
  {
    title: N("Pop hook"),
    body: "[Verse]\nWoke up with the sunlight spilling on the floor\nGot that feeling something good is knocking at my door\nPhone down, head up, stepping into gold\nEvery little moment worth a hundred more\n\n[Chorus]\nWe're alive tonight, hearts on fire\nDancing till the stars retire\nTurn it up, don't let it fade\nThis is the memory we made",
  },
  {
    title: N("Acoustic ballad"),
    body: "[Verse]\nThe kettle hums a tired song, the winter's at the door\nYour letters in a shoebox that I don't read anymore\nBut the garden that we planted still comes up every spring\nSome things keep their promises without us doing anything\n\n[Chorus]\nSo I'll leave the porch light burning, like the old days\nHalf the town away from you, and half a life too late\nIf you ever wander home, you won't have to knock\nThe door was never locked",
  },
  {
    title: N("Rock anthem"),
    body: "[Verse]\nConcrete under worn-out shoes, we've been running all our lives\nEvery no we ever heard just sharpened up our knives\nThey said settle, we said never, wrote it on the wall\n\n[Chorus]\nWe are the thunder, hear us roar\nKicking down that closed door\nLouder than they've ever known\nTonight we take the throne",
  },
  {
    title: N("Love song"),
    body: "[Verse]\nYou found me in the noise of an ordinary day\nQuiet as a Sunday, you just took my breath away\nNo grand parade, no fireworks, no scene\nJust your hand in mine and everything between\n\n[Chorus]\nAnd I'd choose you, I'd choose you again\nEvery version of this world I'm ever in\nCome the highs, come the lows, come whatever's true\nI would still, I would always choose you",
  },
  {
    title: N("Breakup song"),
    body: "[Verse]\nI still take the long way, drive right past your street\nLeft your hoodie in the closet, couldn't fold it up so neat\nEverybody says that time is gonna set me free\nBut the clocks all move so slow when you're not here with me\n\n[Chorus]\nSo I'm learning how to miss you and let you go\nTwo things I never thought that I could hold\nYou were half of every plan I ever made\nNow I'm building something new out of the shade",
  },
  {
    title: N("Party anthem"),
    body: "[Verse]\nLights down low, the whole room starting to move\nBassline hitting like it's got something to prove\nNo worries left outside on the floor\nHands to the ceiling, then we ask for more\n\n[Chorus]\nTurn it up, turn it up, let the whole night ring\nWe came here to dance, we came here to sing\nNobody's tired, nobody's slow\nTonight we let it all go",
  },
];
const languages = [
  [N("Auto"), "unknown"],
  [N("English"), "en"],
  [N("Spanish"), "es"],
  [N("French"), "fr"],
  [N("German"), "de"],
  [N("Italian"), "it"],
  [N("Portuguese"), "pt"],
  [N("Japanese"), "ja"],
  [N("Korean"), "ko"],
  [N("Chinese"), "zh"],
  [N("Cantonese"), "yue"],
  [N("Russian"), "ru"],
  [N("Hindi"), "hi"],
  [N("Arabic"), "ar"],
  [N("Dutch"), "nl"],
  [N("Polish"), "pl"],
  [N("Turkish"), "tr"],
  [N("Vietnamese"), "vi"],
  [N("Swedish"), "sv"],
];
const soundPrompts = [
  N(
    "Futuristic laser blast, sharp energy pulse, stereo movement, arcade style",
  ),
  N("Dog barking next to a waterfall"),
  N("Sparkling fantasy energy swirl, mystical shimmer, rising magical burst"),
  N(
    "Running footsteps on pavement, fast pace, urban street environment, energetic motion sound",
  ),
  N("Chugging train coming into station with horn"),
];
const keys = [
  "C",
  "C#",
  "D",
  "Eb",
  "E",
  "F",
  "F#",
  "G",
  "Ab",
  "A",
  "Bb",
  "B",
].flatMap((n) => [n + " major", n + " minor"]);
// KokoroVoiceCatalog.swift's language order and presentation labels.
const voiceLanguages = [
  ["a", N("American English")],
  ["b", N("British English")],
  ["e", N("Spanish")],
  ["f", N("French")],
  ["h", N("Hindi")],
  ["i", N("Italian")],
  ["j", N("Japanese")],
  ["p", N("Portuguese")],
  ["z", N("Chinese")],
];
function voiceName(id: string) {
  const name = id.split("_").slice(1).join(" ");
  return name ? name[0].toUpperCase() + name.slice(1) : id;
}
function voiceLabel(id: string) {
  return `${voiceName(id)} — ${t(voiceLanguages.find(([code]) => id[0] === code)?.[1] || id)}, ${id[1] === "f" ? t("female") : t("male")}`;
}
const commonTempos = [
  ["", N("Auto")],
  ["60", N("60 — slow ballad")],
  ["75", N("75 — downtempo")],
  ["85", N("85 — hip-hop")],
  ["95", N("95 — groove")],
  ["105", N("105 — mid-tempo pop")],
  ["120", N("120 — pop / house")],
  ["128", N("128 — EDM")],
  ["140", N("140 — trap / techno")],
  ["160", N("160 — punk / footwork")],
  ["174", N("174 — drum & bass")],
];

export { kokoroVoices, trackClasses, sectionTags, musicStyles, music3Styles, musicLyrics, languages, soundPrompts, keys, voiceLanguages, voiceName, voiceLabel, commonTempos };
