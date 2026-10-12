# Building a 3D game or scene with mlx-serve assets

Rules from real agent runs (three.js, but the numbers carry to any engine). Each one fixes a bug
agents shipped. Generate assets with `gen.mjs` next to this file (`node gen.mjs help`).

## Assets: `gen.mjs`

Write every asset as a job in one `assets.json`, run `node <skill dir>/gen.mjs batch assets.json`,
and load files through the `manifest.json` it writes. Jobs run one at a time (the server has one
GPU), unchanged jobs are skipped, and the media models are unloaded at the end so your chat model
keeps its memory. Never write your own SSE client for this.

- Textures that repeat: `"tile": true` (seamless) and `"maps": true` (normal, roughness and AO maps
  beside the colour map). Prompt "seamless tileable texture of ..., top-down, flat even lighting".
- Leaves, blossoms, sprites: prompt "... on a plain white background, no shadow", then
  `"cutout": true, "trim": true` (add `"holes": true` when background shows through gaps).
- 3D props: `{"type": "3d", "prompt": "<one object>", "height_m": 1.6, "max_tris": 8000}`
  generates a reference image, the textured GLB, then scales it to metres with the pivot at the
  base centre and cuts it to a triangle budget. Load with `GLTFLoader` and add it as is.
- Post-process files you already have: `gen.mjs maps|tile|cutout|glb` (PNG and GLB, no server).

## Movement and camera

- three.js cameras look down -Z. With `camera.rotation.set(pitch, yaw, 0, "YXZ")`:
  forward = `(-sin yaw, 0, -cos yaw)`, right = `(cos yaw, 0, -sin yaw)`. Positive yaw turns LEFT.
  Using `(+sin yaw, ...)` mirrors WASD as soon as the player turns: the most common bug.
- Mouse look: `yaw -= movementX * sens`, `pitch -= movementY * sens`, pitch clamped to about
  ±1.45 rad, sens about 0.002. Pointer lock on click or Enter, released by Escape.
- Test it against the engine, not against your own formula: for several yaws, build a real
  `PerspectiveCamera`, call `getWorldDirection`, and assert W moves along it and D along its
  right. A test that copies the formula passes a mirrored game.
- Human scale: eye height 1.6-1.7 m, walk 1.4 m/s, run 3.5-5 m/s, gravity 9.81, jump about 4.5 m/s.
- Collide with everything the player can walk into: trunks, rocks, shrubs, lanterns, buildings and
  water edges (circles and segments, slide along them). A shrub without a collider puts the camera
  inside it.
- A cinematic spline must clear every collider and canopy by 1.5 m or more; test the sampled path
  against the collider list. Letterbox bars cover at most 12.5% of the height each (a 2.39:1 frame
  on 16:9), and fade in, never close the frame.

## Foliage

- Leaves are alpha-cutout CARDS, never solid quads: a leaf or blossom texture with transparency
  (`cutout` above, or drawn on a canvas), `alphaTest` 0.4-0.5, `side: DoubleSide`,
  `transparent: false` (no sorting). A 1x1 opaque plane per leaf renders as a pile of squares.
- Size cards as CLUSTERS, not single leaves: 0.3-0.8 m cards each showing 5-15 leaves,
  300-1500 cards per tree, attached at branch tips, random rotation, per-instance tint
  (`InstancedMesh.setColorAt`). Real-size 3-9 cm leaves disappear at walking distance.
- Trees are branching skeletons: a trunk splitting into 2-4 limbs per level for 4-6 levels, each
  child 0.6-0.8x its parent's length with radius tapering about 0.7x, limbs bent with a little
  noise; merge one tree's limbs into one geometry and reuse a few variants. A single cylinder with
  a blob on top is not a tree.
- Grass is blades, densest where the camera is: tapered, slightly curved blades of 5-7 vertices,
  0.15-0.6 m tall, 3-6 cm wide, 50-200 blades per square metre within 20-25 m of the camera
  (chunks that follow the player), thinned and shortened with distance, swaying in the vertex
  shader by height. Paint the ground under it a grass colour so gaps read as grass. Grass receives
  shadows but does not cast them.
- Look at it from eye height before calling it done: up close, every surface shows its geometry.

## Lighting and atmosphere

- Renderer: `ACESFilmicToneMapping`, `outputColorSpace = SRGBColorSpace`, exposure 0.8-1.2 by day.
  Sun `DirectionalLight` 2-4, hemisphere or ambient fill 0.3-0.8; at night the moon 0.1-0.3 and
  exposure up, so lanterns read.
- Bloom (`UnrealBloomPass`): strength 0.2-0.6, radius 0.3-0.5, threshold 0.85-0.95. A low
  threshold plus fog plus high exposure whites out the whole frame.
- Fog: `FogExp2` density 0.004-0.012 for a 100 m scene, coloured from the sky at that time of day.
- Shadows: one directional shadow camera that follows the player, its position SNAPPED to whole
  shadow-map texels (otherwise shadows swim and shimmer); `shadow.bias` about -0.0005,
  `normalBias` 0.02-0.05 (stops shadow acne, the striped and blocky surface artifact).
- Day cycles: drive every light, colour, exposure and fog value from ONE function of time of day,
  blended with smoothstep between keyframes. Test that each value changes by less than a small
  bound between frames at 60 fps; a value that snaps strobes.
- Distance: real geometry in front of any painted backdrop. Two to four rings of noise-displaced
  ridges at 150-400 m, each fading further into the fog colour; the painting goes behind them as
  the sky band only.
