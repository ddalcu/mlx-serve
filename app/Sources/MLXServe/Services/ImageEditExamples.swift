import Foundation

/// Starter prompts for the image pane's Examples menu (same idiom as
/// `VideoGenView.examplePrompts`: click one, it fills the prompt field, you
/// edit from there).
///
/// The edit library is transcribed VERBATIM from Mage-Flow-Edit's own published
/// galleries. That matters more than it looks: an instruction-tuned editor was
/// trained and evaluated on these exact phrasings, and "Generate a grayscale
/// monocular depth map of this image. Represent relative distance at pixel
/// level, with closer regions brighter and farther regions darker." produces a
/// clean depth map where "make a depth map" produces a grey picture of a dog.
/// Rewording these to sound nicer is a quality regression, not a copy edit.
///
/// Everything here is a PROMPT — no new request field, no engine change. The
/// whole repertoire (depth/canny/segmentation/normals/pose maps, restoration,
/// degradation, camera moves, try-on) is the one edit pipeline being asked for
/// different things.
struct ImagePromptExample: Hashable {
    let title: String
    let body: String
}

struct ImagePromptExampleGroup: Hashable {
    let name: String
    let examples: [ImagePromptExample]
}

enum ImagePromptExamples {

    // MARK: Text-to-image

    /// Shown when there's no source image. Short, concrete, and varied enough to
    /// show the model likes plain descriptive prose rather than tag soup.
    static let textToImage: [ImagePromptExampleGroup] = [
        .init(name: "Starters", examples: [
            .init(title: "Photographic portrait",
                  body: "A close-up portrait of an elderly fisherman with a weathered face and white stubble, wearing a knitted navy sweater, soft overcast daylight, sharp focus on the eyes, shallow depth of field."),
            .init(title: "Landscape",
                  body: "A wide photograph of a mountain lake at first light, mist sitting on the water, snow-capped peaks behind, pine forest along the near shore, calm reflections, natural colour."),
            .init(title: "Product shot",
                  body: "A studio product photograph of a copper moka pot on a white marble surface, soft window light from the left, subtle reflections, clean neutral background."),
            .init(title: "Illustration",
                  body: "A children's-book illustration of a fox reading a book under a lamp post at night, warm gouache textures, soft shadows, muted autumn palette."),
            .init(title: "Text on a sign",
                  body: "A photograph of a small bakery storefront at golden hour with a hand-painted wooden sign reading \"MORNING LOAF\", warm light, shallow depth of field."),
        ]),
    ]

    // MARK: Editing — the generic instructions any in-context editor handles

    private static let contentGroup = ImagePromptExampleGroup(name: "Content", examples: [
        .init(title: "Add an object", body: "Add a hot air balloon floating in the sky"),
        .init(title: "Add several", body: "add 4 balloons"),
        .init(title: "Remove an object", body: "Remove the main object in the foreground"),
        .init(title: "Replace the subject", body: "Replace the main animal with a majestic eagle"),
        .init(title: "Cut out the subject",
              body: "Extract the main foreground subject from the image and isolate it on a clean pure white background. Preserve its shape, identity, texture, and fine boundary details."),
        .init(title: "Change the text", body: "Replace the visible text with 'DREAM BIG'"),
    ])

    private static let appearanceGroup = ImagePromptExampleGroup(name: "Appearance", examples: [
        .init(title: "Recolour something", body: "Change the color of the roof to terracotta orange"),
        .init(title: "Change the material", body: "Transform the texture to appear as hand-blown glass"),
        .init(title: "Art style", body: "Apply Studio Ghibli anime style"),
        .init(title: "Time of day", body: "Change the time of day to golden hour sunset"),
        .init(title: "Mood / colour grade", body: "Apply a moody blue hour atmosphere"),
    ])

    private static let sceneGroup = ImagePromptExampleGroup(name: "Scene & camera", examples: [
        .init(title: "Replace the background", body: "Replace the background with a field of sunflowers"),
        .init(title: "Change the pose", body: "Change the pose to a confident power stance"),
        .init(title: "Zoom in",
              body: "Create a closer camera framing centered on the primary subject, as if using optical zoom, without changing the subject or environment."),
        .init(title: "Zoom out",
              body: "Zoom the camera out to reveal a wider view of the same environment around the main subject, preserving subject identity and visual style."),
        .init(title: "Change viewpoint",
              body: "Change the camera to a high-angle three-quarter viewpoint looking down at the same scene, preserving all subjects."),
        .init(title: "Enlarge the subject",
              body: "Increase only the size of the primary subject so it appears noticeably larger, preserving its shape, texture, pose, lighting, and spatial placement."),
        .init(title: "Shrink the subject",
              body: "Reduce only the size of the primary subject so it appears noticeably smaller, preserving shape, texture, pose, lighting, and spatial placement."),
        .init(title: "Several changes at once",
              body: "Add a red fox as a new foreground subject, replace the background with a misty pine forest, and apply a warm golden color grade. Preserve original landmarks and make scale, perspective, illumination, shadows, and color treatment coherent."),
    ])

    private static let peopleGroup = ImagePromptExampleGroup(name: "People", examples: [
        .init(title: "Hair length", body: "Make the person's hair longer and flowing"),
        .init(title: "Hairstyle", body: "Change the hairstyle to a short pixie cut"),
        .init(title: "Add a beard", body: "Add a well-groomed beard"),
        .init(title: "Try on a garment (2 images)", body: "Dress the person naturally in the provided garment."),
        .init(title: "Reaction meme",
              body: "Turn this portrait into a polished reaction meme. Preserve the person's identity and clothing, exaggerate the facial expression into joyful celebration, and add the exact caption \"THE TESTS ARE FINALLY GREEN\" in large bold white uppercase meme lettering with a black outline. Keep the text fully legible and the composition clean."),
    ])

    /// Fix a photo. The counterpart group DEGRADES one — both directions are
    /// trained, and the pair is genuinely useful for making test material.
    private static let restoreGroup = ImagePromptExampleGroup(name: "Restore", examples: [
        .init(title: "Sharpen (deblur)",
              body: "Remove the optical or motion blur and restore a sharp, detailed version of the same image. Recover clean edges, fine textures, and recognizable facial or object details without changing content."),
        .init(title: "Clear the haze",
              body: "Remove the atmospheric haze completely and restore a crisp, clear image with natural contrast, accurate colors, and sharp distant details. Preserve every subject and the original composition."),
        .init(title: "Remove rain",
              body: "Remove all rain streaks, droplets, wet-lens artifacts, and rain-induced haze. Restore a clear dry version of the same scene while preserving all subjects, geometry, and composition."),
        .init(title: "Remove lens flare",
              body: "Remove all lens flare orbs, optical streaks, glare, ghosting, and bloom introduced into the image. Reconstruct natural lighting and hidden scene details without changing any objects."),
        .init(title: "Brighten a dark photo",
              body: "Enhance this low-light image into a clean, properly exposed photograph. Lift dark details, reduce sensor noise, restore natural colors and contrast, and preserve the exact identity and subjects."),
        .init(title: "Colourise a black-and-white photo",
              body: "Colorize this grayscale image with realistic, natural, context-appropriate colors while preserving all structures, identities, textures, lighting, and composition."),
    ])

    private static let simulateGroup = ImagePromptExampleGroup(name: "Simulate", examples: [
        .init(title: "Add rain",
              body: "Add a steady natural rain shower across the image, including subtle water droplets and damp ground, without changing the existing scene content."),
        .init(title: "Add haze",
              body: "Add a realistic layer of atmospheric haze across the scene, reducing contrast and distant clarity while preserving every subject and the original composition."),
        .init(title: "Add lens flare",
              body: "Add realistic camera-lens ghosting, bloom, and a diagonal flare streak from the brightest light source without altering any objects."),
        .init(title: "Defocus blur",
              body: "Introduce a uniform defocus blur over the whole frame, as if the camera focused incorrectly. Do not change scene content."),
        .init(title: "Dim to night",
              body: "Reduce the illumination to a dim nighttime exposure so details become difficult but still faintly visible. Keep all subjects and composition unchanged."),
        .init(title: "Black and white",
              body: "Create a faithful grayscale version of the image, preserving texture, lighting, geometry, and all subjects."),
    ])

    /// The model doubles as its own ControlNet preprocessor: it can EXTRACT a
    /// control map from a photo, and generate a photo FROM one. Both directions
    /// verified live on this port (depth, canny, segmentation).
    private static let mapsGroup = ImagePromptExampleGroup(name: "Control maps", examples: [
        .init(title: "Depth map",
              body: "Generate a grayscale monocular depth map of this image. Represent relative distance at pixel level, with closer regions brighter and farther regions darker."),
        .init(title: "Canny edges",
              body: "Convert this image into a clean black-and-white Canny edge map showing only the important object and scene contours."),
        .init(title: "Soft edges (HED)",
              body: "Generate a holistically-nested edge detection map with smooth black structural boundaries on a clean white background."),
        .init(title: "Segmentation map",
              body: "Generate a semantic segmentation map that assigns clearly different flat colors to the main subject, other foreground objects, and the background regions."),
        .init(title: "Surface normals",
              body: "Generate a colored surface-normal map of this image, encoding the orientation of every visible surface while preserving scene geometry."),
        .init(title: "Pose skeleton", body: "Show the person's pose as a stick figure."),
        .init(title: "Line sketch",
              body: "Convert this image into a clean monochrome line sketch, preserving the composition and recognizable outlines while removing fill colors and shading."),
    ])

    /// The reverse direction: feed a control map as the source image. The
    /// quoted description is the part to replace — the model leans on it hard,
    /// because the map carries geometry and nothing else.
    private static let fromMapsGroup = ImagePromptExampleGroup(name: "Map → photo", examples: [
        .init(title: "From a depth map",
              body: "Use the depth map to generate a realistic image of \"gray fur, black eyes, white whiskers, round arched opening, wood shavings, terracotta shelter, small animal, pet enclosure, pink nose, scattered seeds\" with consistent geometry."),
        .init(title: "From an edge map",
              body: "Generate a realistic photo of \"man, dark gray turtleneck, navy blue trousers, black belt, arms crossed, short dark hair, light stubble, neutral expression, white background, studio portrait\" using this edge map."),
        .init(title: "From a normal map",
              body: "From this normal map, create a realistic photo of \"gray and white fur, long-haired cat, sitting posture, fluffy tail, green eyes, white bathtub, marble-patterned tiles, corner of bathroom, direct gaze\"."),
        .init(title: "From a pose skeleton",
              body: "Generate a realistic photo of \"gray hair, dark suit, white dress shirt, blue patterned tie, arms crossed, gold cufflinks, small red and white pin, serious expression, stone building background, formal attire, close-up portrait\"."),
        .init(title: "Fill a masked area",
              body: "Recover a complete realistic image from the masked input based on \"insect exoskeleton, cicada molt, tree bark, transparent wings, brownish-yellow color, large compound eye, segmented legs, veined wings, natural texture, close-up view\"."),
    ])

    /// Mage-Flow-Edit's full published repertoire.
    static let mageFlowEdit: [ImagePromptExampleGroup] = [
        contentGroup, appearanceGroup, sceneGroup, peopleGroup,
        restoreGroup, simulateGroup, mapsGroup, fromMapsGroup,
    ]

    /// Other in-context editors (FLUX.2-klein) get the generic instruction
    /// groups only. The control-map and restoration repertoires are things
    /// Mage-Flow-Edit documents and we verified on it — offering them for a
    /// model we haven't checked would be advertising, not a feature.
    static let genericEdit: [ImagePromptExampleGroup] = [
        contentGroup, appearanceGroup, sceneGroup, peopleGroup,
    ]
}
