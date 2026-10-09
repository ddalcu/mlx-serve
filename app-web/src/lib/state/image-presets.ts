import { N } from "../i18n/i18n";
// Presets carry no host RAM or download-size claims.
const fluxResolutions = [
  {
    width: 1024,
    height: 1024,
    label: N("1024 × 1024 (square)"),
  },
  {
    width: 512,
    height: 512,
    label: N("512 × 512 (fast, low RAM)"),
  },
  {
    width: 1152,
    height: 896,
    label: N("1152 × 896 (landscape 4:3)"),
  },
  {
    width: 896,
    height: 1152,
    label: N("896 × 1152 (portrait 3:4)"),
  },
  {
    width: 1216,
    height: 832,
    label: N("1216 × 832 (landscape 3:2)"),
  },
  {
    width: 832,
    height: 1216,
    label: N("832 × 1216 (portrait 2:3)"),
  },
  {
    width: 1344,
    height: 768,
    label: N("1344 × 768 (landscape 16:9)"),
  },
  {
    width: 768,
    height: 1344,
    label: N("768 × 1344 (portrait 9:16)"),
  },
  {
    width: 1536,
    height: 640,
    label: N("1536 × 640 (cinematic)"),
  },
];
const kreaResolutions = [
  {
    width: 1024,
    height: 1024,
    label: N("1024 × 1024 (square)"),
  },
  {
    width: 768,
    height: 768,
    label: N("768 × 768 (square, fast)"),
  },
  {
    width: 512,
    height: 512,
    label: N("512 × 512 (fast, low RAM)"),
  },
  {
    width: 1024,
    height: 1536,
    label: N("1024 × 1536 (portrait 2:3)"),
  },
  {
    width: 1536,
    height: 1024,
    label: N("1536 × 1024 (landscape 3:2)"),
  },
  {
    width: 1344,
    height: 768,
    label: N("1344 × 768 (landscape 16:9)"),
  },
  {
    width: 768,
    height: 1344,
    label: N("768 × 1344 (portrait 9:16)"),
  },
];
const mageFlowResolutions = [
  {
    width: 1024,
    height: 1024,
    label: N("1024 × 1024 (square) · ~6s"),
  },
  {
    width: 768,
    height: 768,
    label: N("768 × 768 (square, fast)"),
  },
  {
    width: 512,
    height: 512,
    label: N("512 × 512 (fastest) · ~3s"),
  },
  {
    width: 1024,
    height: 1536,
    label: N("1024 × 1536 (portrait 2:3)"),
  },
  {
    width: 1536,
    height: 1024,
    label: N("1536 × 1024 (landscape 3:2) · ~13s"),
  },
  {
    width: 1344,
    height: 768,
    label: N("1344 × 768 (landscape 16:9)"),
  },
  {
    width: 768,
    height: 1344,
    label: N("768 × 1344 (portrait 9:16)"),
  },
  {
    width: 2048,
    height: 1152,
    label: N("2048 × 1152 (16:9, large)"),
  },
  {
    width: 2048,
    height: 2048,
    label: N("2048 × 2048 (max) · ~50s"),
  },
  {
    width: 2048,
    height: 512,
    label: N("2048 × 512 (panorama 4:1)"),
  },
  {
    width: 512,
    height: 2048,
    label: N("512 × 2048 (tall 1:4)"),
  },
];
const qwenImageResolutions = [
  {
    width: 1024,
    height: 1024,
    label: N("1024 × 1024 (square)"),
  },
  {
    width: 768,
    height: 768,
    label: N("768 × 768 (square, fast)"),
  },
  {
    width: 512,
    height: 512,
    label: N("512 × 512 (fastest)"),
  },
  {
    width: 832,
    height: 1248,
    label: N("832 × 1248 (portrait 2:3)"),
  },
  {
    width: 1248,
    height: 832,
    label: N("1248 × 832 (landscape 3:2)"),
  },
  {
    width: 1344,
    height: 768,
    label: N("1344 × 768 (landscape 16:9)"),
  },
  {
    width: 768,
    height: 1344,
    label: N("768 × 1344 (portrait 9:16)"),
  },
];
const imageTemplates = {
  Starters: [
    {
      title: N("Photographic portrait"),
      body: "A close-up portrait of an elderly fisherman with a weathered face and white stubble, wearing a knitted navy sweater, soft overcast daylight, sharp focus on the eyes, shallow depth of field.",
    },
    {
      title: N("Landscape"),
      body: "A wide photograph of a mountain lake at first light, mist sitting on the water, snow-capped peaks behind, pine forest along the near shore, calm reflections, natural colour.",
    },
    {
      title: N("Product shot"),
      body: "A studio product photograph of a copper moka pot on a white marble surface, soft window light from the left, subtle reflections, clean neutral background.",
    },
    {
      title: N("Illustration"),
      body: "A children's-book illustration of a fox reading a book under a lamp post at night, warm gouache textures, soft shadows, muted autumn palette.",
    },
    {
      title: N("Text on a sign"),
      body: 'A photograph of a small bakery storefront at golden hour with a hand-painted wooden sign reading "MORNING LOAF", warm light, shallow depth of field.',
    },
  ],
  Content: [
    {
      title: N("Add an object"),
      body: "Add a hot air balloon floating in the sky",
    },
    {
      title: N("Add several"),
      body: "add 4 balloons",
    },
    {
      title: N("Remove an object"),
      body: "Remove the main object in the foreground",
    },
    {
      title: N("Replace the subject"),
      body: "Replace the main animal with a majestic eagle",
    },
    {
      title: N("Cut out the subject"),
      body: "Extract the main foreground subject from the image and isolate it on a clean pure white background. Preserve its shape, identity, texture, and fine boundary details.",
    },
    {
      title: N("Change the text"),
      body: "Replace the visible text with 'DREAM BIG'",
    },
  ],
  Appearance: [
    {
      title: N("Recolour something"),
      body: "Change the color of the roof to terracotta orange",
    },
    {
      title: N("Change the material"),
      body: "Transform the texture to appear as hand-blown glass",
    },
    {
      title: N("Art style"),
      body: "Apply Studio Ghibli anime style",
    },
    {
      title: N("Time of day"),
      body: "Change the time of day to golden hour sunset",
    },
    {
      title: N("Mood / colour grade"),
      body: "Apply a moody blue hour atmosphere",
    },
  ],
  "Scene & camera": [
    {
      title: N("Replace the background"),
      body: "Replace the background with a field of sunflowers",
    },
    {
      title: N("Change the pose"),
      body: "Change the pose to a confident power stance",
    },
    {
      title: N("Zoom in"),
      body: "Create a closer camera framing centered on the primary subject, as if using optical zoom, without changing the subject or environment.",
    },
    {
      title: N("Zoom out"),
      body: "Zoom the camera out to reveal a wider view of the same environment around the main subject, preserving subject identity and visual style.",
    },
    {
      title: N("Change viewpoint"),
      body: "Change the camera to a high-angle three-quarter viewpoint looking down at the same scene, preserving all subjects.",
    },
    {
      title: N("Enlarge the subject"),
      body: "Increase only the size of the primary subject so it appears noticeably larger, preserving its shape, texture, pose, lighting, and spatial placement.",
    },
    {
      title: N("Shrink the subject"),
      body: "Reduce only the size of the primary subject so it appears noticeably smaller, preserving shape, texture, pose, lighting, and spatial placement.",
    },
    {
      title: N("Several changes at once"),
      body: "Add a red fox as a new foreground subject, replace the background with a misty pine forest, and apply a warm golden color grade. Preserve original landmarks and make scale, perspective, illumination, shadows, and color treatment coherent.",
    },
  ],
  People: [
    {
      title: N("Hair length"),
      body: "Make the person's hair longer and flowing",
    },
    {
      title: N("Hairstyle"),
      body: "Change the hairstyle to a short pixie cut",
    },
    {
      title: N("Add a beard"),
      body: "Add a well-groomed beard",
    },
    {
      title: N("Try on a garment (2 images)"),
      body: "Dress the person naturally in the provided garment.",
    },
    {
      title: N("Reaction meme"),
      body: 'Turn this portrait into a polished reaction meme. Preserve the person\'s identity and clothing, exaggerate the facial expression into joyful celebration, and add the exact caption "THE TESTS ARE FINALLY GREEN" in large bold white uppercase meme lettering with a black outline. Keep the text fully legible and the composition clean.',
    },
  ],
  Restore: [
    {
      title: N("Sharpen (deblur)"),
      body: "Remove the optical or motion blur and restore a sharp, detailed version of the same image. Recover clean edges, fine textures, and recognizable facial or object details without changing content.",
    },
    {
      title: N("Clear the haze"),
      body: "Remove the atmospheric haze completely and restore a crisp, clear image with natural contrast, accurate colors, and sharp distant details. Preserve every subject and the original composition.",
    },
    {
      title: N("Remove rain"),
      body: "Remove all rain streaks, droplets, wet-lens artifacts, and rain-induced haze. Restore a clear dry version of the same scene while preserving all subjects, geometry, and composition.",
    },
    {
      title: N("Remove lens flare"),
      body: "Remove all lens flare orbs, optical streaks, glare, ghosting, and bloom introduced into the image. Reconstruct natural lighting and hidden scene details without changing any objects.",
    },
    {
      title: N("Brighten a dark photo"),
      body: "Enhance this low-light image into a clean, properly exposed photograph. Lift dark details, reduce sensor noise, restore natural colors and contrast, and preserve the exact identity and subjects.",
    },
    {
      title: N("Colourise a black-and-white photo"),
      body: "Colorize this grayscale image with realistic, natural, context-appropriate colors while preserving all structures, identities, textures, lighting, and composition.",
    },
  ],
  Simulate: [
    {
      title: N("Add rain"),
      body: "Add a steady natural rain shower across the image, including subtle water droplets and damp ground, without changing the existing scene content.",
    },
    {
      title: N("Add haze"),
      body: "Add a realistic layer of atmospheric haze across the scene, reducing contrast and distant clarity while preserving every subject and the original composition.",
    },
    {
      title: N("Add lens flare"),
      body: "Add realistic camera-lens ghosting, bloom, and a diagonal flare streak from the brightest light source without altering any objects.",
    },
    {
      title: N("Defocus blur"),
      body: "Introduce a uniform defocus blur over the whole frame, as if the camera focused incorrectly. Do not change scene content.",
    },
    {
      title: N("Dim to night"),
      body: "Reduce the illumination to a dim nighttime exposure so details become difficult but still faintly visible. Keep all subjects and composition unchanged.",
    },
    {
      title: N("Black and white"),
      body: "Create a faithful grayscale version of the image, preserving texture, lighting, geometry, and all subjects.",
    },
  ],
  "Control maps": [
    {
      title: N("Depth map"),
      body: "Generate a grayscale monocular depth map of this image. Represent relative distance at pixel level, with closer regions brighter and farther regions darker.",
    },
    {
      title: N("Canny edges"),
      body: "Convert this image into a clean black-and-white Canny edge map showing only the important object and scene contours.",
    },
    {
      title: N("Soft edges (HED)"),
      body: "Generate a holistically-nested edge detection map with smooth black structural boundaries on a clean white background.",
    },
    {
      title: N("Segmentation map"),
      body: "Generate a semantic segmentation map that assigns clearly different flat colors to the main subject, other foreground objects, and the background regions.",
    },
    {
      title: N("Surface normals"),
      body: "Generate a colored surface-normal map of this image, encoding the orientation of every visible surface while preserving scene geometry.",
    },
    {
      title: N("Pose skeleton"),
      body: "Show the person's pose as a stick figure.",
    },
    {
      title: N("Line sketch"),
      body: "Convert this image into a clean monochrome line sketch, preserving the composition and recognizable outlines while removing fill colors and shading.",
    },
  ],
  "Map → photo": [
    {
      title: N("From a depth map"),
      body: 'Use the depth map to generate a realistic image of "gray fur, black eyes, white whiskers, round arched opening, wood shavings, terracotta shelter, small animal, pet enclosure, pink nose, scattered seeds" with consistent geometry.',
    },
    {
      title: N("From an edge map"),
      body: 'Generate a realistic photo of "man, dark gray turtleneck, navy blue trousers, black belt, arms crossed, short dark hair, light stubble, neutral expression, white background, studio portrait" using this edge map.',
    },
    {
      title: N("From a normal map"),
      body: 'From this normal map, create a realistic photo of "gray and white fur, long-haired cat, sitting posture, fluffy tail, green eyes, white bathtub, marble-patterned tiles, corner of bathroom, direct gaze".',
    },
    {
      title: N("From a pose skeleton"),
      body: 'Generate a realistic photo of "gray hair, dark suit, white dress shirt, blue patterned tie, arms crossed, gold cufflinks, small red and white pin, serious expression, stone building background, formal attire, close-up portrait".',
    },
    {
      title: N("Fill a masked area"),
      body: 'Recover a complete realistic image from the masked input based on "insect exoskeleton, cicada molt, tree bark, transparent wings, brownish-yellow color, large compound eye, segmented legs, veined wings, natural texture, close-up view".',
    },
  ],
};

export { fluxResolutions, kreaResolutions, mageFlowResolutions, qwenImageResolutions, imageTemplates };
