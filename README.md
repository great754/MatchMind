## MatchMind tactical video

Render the SoccerTrack-v2 `118575` match with the original panoramic footage and a synchronized, translucent top-down pitch at the bottom center:

```bash
source .venv/bin/activate
python -m src.vision.visualize_pitch
```

The output is `outputs/match_118575_matchmind.mp4` (4096×1080, 25 FPS, four minutes). Blue and red dots distinguish the two teams. Teams are estimated once from jersey crops sampled across the clip, with a stable assignment per tracking ID. Goalkeepers wear different kits and their two-cluster assignments should be reviewed. A `.teams.json` sidecar records assignments, and a `.jpg` contains the first rendered frame.

For a ten-second preview:

```bash
python -m src.vision.visualize_pitch --max-frames 250 --output outputs/match_118575_preview.mp4
```

Use `--play` for local playback while rendering, `--inset-width 0.25` to enlarge the map, or `--opacity 0.8` for a more opaque pitch. Press `q` to stop playback.

Player foot positions map onto a 105×68 m field through the supplied pitch landmarks and piecewise affine interpolation. This accounts for the panoramic camera's nonlinear distortion better than a single global homography. Points outside the calibrated landmark hull are omitted. The renderer checks that the source video and zero-based MOT tracking cover the same frames and streams frames without loading the entire video into memory.

The reference image's action labels and selected player are not reproduced: the event CSV uses different player IDs and full-match times with no verified mapping to this clip. The ball files likewise contain full-half positions without a verified clip offset. Neither event nor ball annotations are guessed.
