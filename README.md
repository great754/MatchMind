## MatchMind tactical video

Render the SoccerTrack-v2 `118575` match with the original panoramic footage and a synchronized, translucent top-down pitch at the bottom center:

```bash
source .venv/bin/activate
python -m src.vision.visualize_pitch
```

The output is `outputs/match_118575_matchmind.mp4` (4096×1080, 25 FPS, four minutes). Blue and red dots distinguish the two teams. Team colors use synchronized GSR team labels, including goalkeepers. Goalkeepers have the same team color and a white ring with a GK label. When authoritative labels are unavailable, jersey clustering excludes persistent backfield goalkeeper candidates and assigns them to the defending side afterward. A `.teams.json` sidecar records assignments, and a `.jpg` contains the first rendered frame.

For a ten-second preview:

```bash
python -m src.vision.visualize_pitch --max-frames 250 --output outputs/match_118575_preview.mp4
```

Use `--play` for local playback while rendering, `--inset-width 0.25` to enlarge the map, or `--opacity 0.8` for a more opaque pitch. Press `q` to stop playback.

Tactical positions primarily come from GSR pitch coordinates, joined to MOT tracks through `118575_sync.json` and the synchronized half-video frame offset. The renderer uses `PitchCoordinateTransformer` only for missing GSR samples. Finite GSR positions outside the pitch are omitted from drawing rather than replaced with a guessed location. The renderer checks that the source video and zero-based MOT tracking cover the same frames and streams frames without loading the entire video into memory.

The basic render includes footage and team markers. Add `--annotations` as described below to include the synchronized action panel and acting player.

### Action overlay

```bash
python -m src.vision.visualize_pitch --annotations --output outputs/match_118575_ball_actions.mp4
```

This adds a 12-row action panel and the acting player's ID and foot highlight. An action row lights up for one second from its annotated frame; simultaneous actions can light up together. The four-minute clip contains 88 events across eight classes; all twelve classes are supported.

The official BAS and ball files use the clock of each released half-video. `data/soccertrack/mot/118575_sync.json` records the inferred alignment: the clip's first frame corresponds to first-half frame 6000 (one-based), approximately 4:00 into that video. This was determined by matching the 22 players' image foot positions across 21 checkpoints; mean assignment error is 14.24 pixels, with approximately one-frame timing precision. The mapping also connects MOT IDs to event actor IDs and uses authoritative team sides to correct goalkeeper assignments.

Ball coordinates have a center-circle origin and are translated by (+52.5, +34) meters for the inset. The supplied ball path is reconstructed from event anchors and interpolation, not detected in every video frame; no ball height is supplied, so the footage ring represents its ground projection. Unknown-status samples are omitted, and positions outside the calibrated pitch cannot be projected. `.events.json`, `.teams.json`, and `.sync.json` sidecars retain clip event times, player/team mappings, and synchronization provenance.

To reproduce alignment after downloading both official GSR half files:

```bash
python -m src.vision.align_clip
python -m unittest discover -s tests -v
```

The alignment tool streams the large GSR JSON files into compact arrays rather than loading the entire files into memory. Annotation source and clock convention: [SoccerTrack v2](https://huggingface.co/datasets/atomscott/soccertrack-v2) (CC BY 4.0).

### Motion, possession, pass difficulty and shots

```bash
python -m src.analytics.pipeline
open outputs/analytics_118575_clip/report.html
python -m unittest discover -s tests -v
```

This exports per-player speeds and measured distance, ball speeds, per-frame ownership, completed/intercepted transfer candidates, transparent difficulty features, shot candidates and BAS comparison files. The offline report has a scrubber and clickable events to review against the source clip. Analytics use the synchronized GSR metric positions by default; `--player-source mot` enables the clip's existing image-to-pitch estimate instead.

For the first half, which contains Shot annotations absent from this four-minute clip:

```bash
python -m src.analytics.pipeline --scope half --summary-only
```

Read [the implementation and testing guide](docs/analytics.md) for the rules, units, configuration, output schemas and validation limits. The ball is event-interpolated: speeds are estimated 2D values, BAS matching is not independent accuracy, and the difficulty score is not a calibrated probability.


### GSR accuracy comparison and goalkeeper-aware rendering

```bash
python -m src.vision.visualize_pitch --annotations --output outputs/match_118575_gsr.mp4
```

Before rendering, this compares calibrated MOT foot positions with GSR positions at every available player/frame pair. For the current clip, all 132,000 detections use GSR. The mean image-to-pitch disagreement relative to GSR is **11.68 m**, median **10.74 m**, and P95 **24.96 m**, across **6,000 frames**, using **130,782 paired positions** (99.08% comparison coverage; the remaining image foot points were outside the calibration hull). This is a relative comparison, not an independent absolute accuracy measurement; GSR remains quantized and the clip offset is inferred.

The renderer first loads compact GSR arrays; if those are missing but raw GSR JSON exists, it streams the raw annotations. A per-sample fallback uses calibrated projection only when that actor's GSR coordinates are missing. Team sides come from the same GSR actor join; synchronized GSR labels remain available if position data is missing. Invalid mappings or inconsistent team labels fail explicitly.

Goalkeepers are identified from provider role metadata when available. In this clip they are MOT **20 / actor 476352 (red)** and MOT **21 / actor 476367 (blue)**. Their differently colored actual kits do not affect their team assignment. A fallback without authoritative labels uses median backfield depth to identify keeper candidates, excludes their kits from outfield color clustering, and associates them with the side their team defends. That positional fallback is a heuristic and can be ambiguous for unusual play.

Alongside the video, `.comparison.json` records global/per-player errors, data coverage, sources, and goalkeeper assignments; `.comparison.csv` records per-frame disagreements; `.positions.csv` records the exact coordinates and source used for each marker. Add `--max-frames 250` for a preview; the comparison still covers the entire clip.

```bash
python -m unittest discover -s tests -v
```

The tactical tests cover changing GSR array slots, actor-ID/frame-offset joins, missing samples/files, raw JSON fallback, conflicting labels, keeper colors, and outfield clustering that excludes keeper kits.
