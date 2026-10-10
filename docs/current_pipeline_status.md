# MatchMind Current Pipeline Status

Status date: 2026-10-10

This document describes the current MatchMind repository as implemented now. It covers the input data, synchronization, coordinate systems, tactical video renderer, deterministic analytics, new tactical-state engine, intelligence/commentary layer, projection-validity audit, generated outputs, verification numbers, and limitations.

## Executive Summary

MatchMind is an offline soccer-analysis system built around SoccerTrack match `118575`. It has four connected but distinct responsibilities:

1. **Tactical video rendering**: combine panoramic match footage with a synchronized top-down pitch.
2. **Deterministic match analytics**: calculate motion, possession, observable transfers, transfer difficulty, and shot candidates.
3. **Tactical state analysis**: sample team geometry and pressure at 5 Hz, then publish conservative block, shape, formation, and transfer-network evidence.
4. **Grounded intelligence and review**: serialize closed events, build rolling context, optionally select approved commentary sentences, and expose the results through an offline HTML report.

The implementation prefers evidence over invention:

- GSR metric pitch coordinates are primary when available.
- MOT projection is retained as a fallback and comparison path.
- Missing observations remain missing.
- Unknown tactical states are published as unknown instead of guessed.
- Persistence gates prevent one-frame changes from becoming events.
- BAS labels validate or contextualize results but do not create detector predictions.
- Gemini chooses from deterministic sentences instead of writing unrestricted match narration.
- Projection disagreement is measured and documented rather than silently corrected.

## Current Verified Status

### Tests

The current command is:

```bash
.venv/bin/python -m unittest discover -s tests -v
```

Current result:

```text
Ran 117 tests
OK
```

The suite covers the original analytics and vision behavior plus the new projection-evaluation and tactical-engine branches.

### Current clip facts

The documented baseline uses:

- Match: `118575`
- Clip length: `6,000` frames
- Frame rate: `25 FPS`
- Duration: approximately `240 seconds`
- Pitch: `105 m x 68 m`
- Completed observable transfers: `29`
- Opposite-team transfer candidates: `7`
- Unresolved transfer candidates: `2`
- Trajectory shot candidates: `0`
- BAS shot references in the four-minute clip: `0`

The transfer completion rate is defined as completed observable transfers divided by all detected transfer candidates. It is not official pass accuracy.

### Projection validity audit

The current projection audit reproduces the unchanged MOT fallback against synchronized GSR:

| Metric | Result |
| --- | ---: |
| Mean disagreement | `11.6831065 m` |
| Median / P50 | `10.7398110 m` |
| P75 | `18.5953233 m` |
| P90 | `23.4371475 m` |
| P95 | `24.9560484 m` |
| P99 | `26.4028016 m` |
| Maximum | `30.5152851 m` |
| RMSE | `14.1924825 m` |
| Finite paired samples | `130,782` |
| Total detections | `132,000` |
| Coverage | `99.0772727%` |
| Method changed | No |

This is disagreement relative to GSR, not independently surveyed absolute accuracy.

The mean projection-minus-GSR error vector is approximately `(0.1401, -10.8486)` meters. The audit deliberately does not subtract this bias, refit the transform, change the synchronization offset, or replace actor mappings.

## Main Entry Points

### Tactical video

[`src/vision/visualize_pitch.py`](../src/vision/visualize_pitch.py)

```bash
python -m src.vision.visualize_pitch
```

Preview:

```bash
python -m src.vision.visualize_pitch \
  --max-frames 250 \
  --output outputs/match_118575_preview.mp4
```

With action annotations:

```bash
python -m src.vision.visualize_pitch \
  --annotations \
  --output outputs/match_118575_ball_actions.mp4
```

### Clip alignment

[`src/vision/align_clip.py`](../src/vision/align_clip.py)

```bash
python -m src.vision.align_clip
```

### Analytics

[`src/analytics/pipeline.py`](../src/analytics/pipeline.py)

```bash
python -m src.analytics.pipeline
```

### Tactical analytics

Tactics are enabled by default in the expanded pipeline. They can be configured or disabled:

```bash
.venv/bin/python -m src.analytics.pipeline \
  --summary-only \
  --commentary-provider template \
  --tactics-config configs/tactics.default.json \
  --output outputs/analytics_118575_tactics
```

Disable the tactical branch:

```bash
.venv/bin/python -m src.analytics.pipeline --no-tactics
```

### Projection audit

[`src/vision/projection_evaluation.py`](../src/vision/projection_evaluation.py)

```bash
.venv/bin/python -m src.vision.projection_evaluation \
  --output outputs/validity_audit/projection
```

### Intelligence providers

Deterministic local preview:

```bash
.venv/bin/python -m src.analytics.pipeline \
  --summary-only \
  --commentary-provider template \
  --max-commentary-events 8
```

Gemini:

```bash
.venv/bin/python -m src.analytics.pipeline \
  --summary-only \
  --commentary-provider gemini \
  --max-commentary-events 4 \
  --max-provider-calls 12
```

The Gemini key is local configuration in `.env`, which is ignored by Git. The provider loads the repository-root `.env` itself, while preserving an already-exported shell variable.

## Architecture

```text
Video + MOT + GSR + ball + BAS + calibration
                      |
                      v
        Alignment and identity synchronization
                      |
          +-----------+-----------+
          |                       |
          v                       v
   Tactical renderer       Recording normalization
          |                       |
          v                       v
   Video + pitch inset      Motion and quality gates
                                  |
                       +----------+----------+
                       |                     |
                       v                     v
                 Possession              Ball motion
                       |                     |
             +---------+---------+           |
             |                   |           |
             v                   v           v
        Transfers            Difficulty     Shots
             |                   |           |
             +---------+---------+-----------+
                       |
                       v
                 Canonical events
                       |
             +---------+---------+
             |                   |
             v                   v
     Tactical state engine   Rolling context
             |                   |
             +---------+---------+
                       v
           Commentary and HTML report
```

The renderer and analytics share data but are separate products. The renderer can stream video frames, while analytics generally need a prepared synchronized dataset and offline calculations.

## Inputs and Identity

The dataset is rooted at `data/soccertrack` and is described by [`DatasetPaths`](../src/vision/data_loading.py).

Important inputs include:

- `mot/118575.txt`: MOT bounding boxes.
- `mot/clips/118575.mp4`: panoramic source footage.
- `gsr/118575/`: raw and compact GSR annotations.
- `ball/118575_1st_ball.npz`: first-half ball positions.
- `ball/118575_2nd_ball.npz`: second-half ball positions.
- `bas/118575/118575_12_class_events.json`: action annotations.
- `raw/118575/118575_keypoints.json`: image/pitch calibration landmarks.
- `raw/118575/118575_homography.npy`: retained 3x3 homography utility input.
- `raw/118575/118575_tracker_box_metadata.xml`: optional provider role metadata.

The repository keeps several identity namespaces separate:

| Identity | Meaning |
| --- | --- |
| MOT ID | Temporary tracking identity in the short clip. |
| GSR actor ID | Provider identity such as `476371`. |
| Jersey number | Not currently verified from the active MOT file. |

MOT IDs are not automatically jersey numbers. The synchronization file joins MOT tracks to GSR actors.

## Timeline Synchronization

The short clip and the long released-half annotations use different clocks.

For the current clip:

- FPS: `25`
- Local clip frames: `6,000`
- Local frame zero corresponds to first-half source frame `6000` in one-based annotation numbering.
- Zero-based GSR offset: `5999`
- Approximate source location: four minutes into the half.

A BAS frame is converted approximately as:

```text
local_frame = bas_frame - 1 - source_offset
local_seconds = local_frame / fps
```

[`align_clip.py`](../src/vision/align_clip.py) performs the alignment by:

1. Reading MOT frame-zero foot positions.
2. Scanning both GSR halves for similar player-position patterns.
3. Keeping coarse candidate offsets.
4. Comparing candidate offsets at checkpoints.
5. Refining the best offset locally.
6. Assigning GSR actors to MOT tracks with Hungarian matching.
7. Voting over checkpoints for actor and team assignments.
8. Writing a synchronization sidecar.

The alignment reports approximately one-frame timing precision, but the offset is inferred rather than independently verified.

Raw GSR files are streamed into compact arrays to avoid loading very large JSON files into memory.

## Coordinate Systems

The active pitch coordinate system is a metric rectangle:

- Length: `105 m`
- Width: `68 m`
- Origin: top-left pitch corner.
- `x`: pitch length direction.
- `y`: pitch width direction.

GSR player positions are already in this coordinate system.

Ball coordinates use a center-circle origin and are translated by:

```text
x_pitch = x_ball + 52.5
y_pitch = y_ball + 34
```

The ball has no `z` coordinate. Ball motion is therefore ground-plane 2D motion.

## Image-to-Pitch Projection

[`src/vision/coordinate_transform.py`](../src/vision/coordinate_transform.py) implements the active `PitchCoordinateTransformer`.

Although the repository contains a homography loader, a single global homography is not the active projection because the panoramic image has nonlinear distortion.

The active projection is piecewise:

1. Load image/pitch landmarks.
2. Delaunay-triangulate image points.
3. Find the triangle containing the image foot point.
4. Calculate barycentric weights inside that triangle.
5. Apply the same weights to the matching pitch triangle.
6. Return missing coordinates outside the convex hull.

The interpolation relationship is:

```text
image_point = w1*A + w2*B + w3*C
w1 + w2 + w3 = 1
```

The system does not extrapolate beyond the calibration hull.

### GSR-first policy

Tactical tracking chooses positions in this order:

1. Valid GSR pitch coordinate.
2. MOT foot projected through the calibration transform for a missing GSR sample.
3. Unavailable when neither source is usable.

A present GSR coordinate is not replaced merely because it is outside the field. Each output row records `gsr`, `pitch_transform`, or `unavailable` as its position source.

### Projection audit

[`projection_evaluation.py`](../src/vision/projection_evaluation.py) measures the unchanged MOT fallback against synchronized GSR. It produces:

- Global error statistics.
- Per-player statistics.
- Per-frame aggregates.
- Pitch-region heatmaps.
- Image-region heatmaps.
- Boundary-region statistics.
- Nearest-landmark bins.
- Triangle simplex and aspect bins.
- Synchronization shift experiments from -2 through +2 frames.
- Orientation experiments.
- Nearest-actor agreement checks.
- Supplementary image-foot diagnostics.

Important findings:

- Interior pitch detections average about `12.8336 m` disagreement.
- Detections within 5 m of a boundary average about `1.9575 m` disagreement.
- The central image half averages about `12.0828 m`.
- The outer image quarters average about `4.9886 m`, but the regions do not contain identical pitch distributions.
- Shifting the synchronization offset by a few frames does not explain the large spatial error.
- Reversing axes makes the error worse.
- No independent ground contacts are available to separate calibration error, GSR quantization, and foot-point error.
- GSR remains primary for all renderer positions.

The audit diagnoses the method. It does not claim an accuracy improvement.

## Tactical Video Renderer

[`src/vision/visualize_pitch.py`](../src/vision/visualize_pitch.py) streams the source video frame by frame.

For every frame it:

1. Reads the panoramic image.
2. Checks video/MOT FPS and frame coverage.
3. Looks up synchronized tactical positions.
4. Omits invalid drawing positions.
5. Draws team-colored player markers.
6. Draws goalkeeper rings and labels.
7. Blends a top-down pitch into the bottom center.
8. Adds time and team legend text.
9. Optionally adds BAS action rows and acting-player highlights.
10. Writes the result immediately to MP4.

The entire video is not loaded into memory.

Sidecars include:

- Comparison JSON and CSV.
- Exact position-source CSV.
- Team assignments.
- Synchronization metadata.
- Action events.
- First-frame JPEG.

## Team and Goalkeeper Assignment

Authoritative GSR team sides are preferred.

The fallback in [`src/vision/team_classifier.py`](../src/vision/team_classifier.py) samples jersey crops every 50 frames, removes likely grass, converts BGR to LAB, aggregates median colors, and clusters outfield jerseys into two groups.

Goalkeepers are handled separately because goalkeeper kits can differ from outfield kits. Provider metadata wins. Without metadata, persistent pitch depth is used as a heuristic candidate and the goalkeeper kit is excluded from outfield clustering.

This fallback can be ambiguous. Different goalkeeper colors do not override authoritative team identity.

## Motion Processing

[`src/analytics/motion.py`](../src/analytics/motion.py) operates on each player and the ball.

For position `p_i` at FPS `f`:

```text
step_i = ||p_i - p_(i-1)||
raw_speed_i = step_i * f
```

The implementation:

1. Splits finite observations into consecutive segments.
2. Applies a three-frame coordinate median.
3. Applies centered Savitzky-Golay smoothing.
4. Differentiates smoothed coordinates.
5. Calculates velocity and speed.
6. Rejects impossible speed intervals.
7. Accumulates only accepted step distances.

Default speed gates:

- Player: `14 m/s`.
- Ball: `50 m/s`.

Missing intervals are never bridged. Centered smoothing uses future samples, so the motion output is offline and not causal.

## Possession Detection

[`src/analytics/possession.py`](../src/analytics/possession.py) combines proximity, relative motion, and persistence.

Default rules:

- Acquire within `2.5 m`.
- Require `5` consecutive frames.
- Retain within `3.5 m`.
- Allow ball speed up to `8 m/s`, or relative speed up to `4 m/s`.
- Mark contested when the closest two players differ by less than `0.5 m` in distance.
- Clear ownership when the ball is missing, invalid, or outside the pitch.

States:

- `controlled`
- `candidate`
- `free`
- `contested`
- `unknown`
- `out_of_play`

The confidence value is rule-derived proximity confidence, not a calibrated probability.

## Observable Transfers and Passes

[`src/analytics/passes.py`](../src/analytics/passes.py) runs over the selected recording, not only a 60-second window.

It looks for:

```text
controlled sender
 -> independent ball flight
 -> persistent controlled receiver
```

Default transfer gates:

- At least 3 independent flight frames.
- At least 3 m straight-line displacement.
- Peak speed of at least 3 m/s.
- An 8-second pending-flight timeout.

Outcomes:

- `completed`: same-team receiver.
- `intercepted`: opposite-team receiver.
- `out_of_play`: ball leaves the field.
- `unresolved`: missing data, timeout, or recording end prevents resolution.

The detector rejects self-recovery and direct ownership flips without independent flight. It does not claim that every observable transfer was an intentional pass.

### Network window versus detection window

The transfer detector analyzes the full selected recording. The tactical transfer network applies a separate rolling window:

```text
current_time - 60 seconds < confirmed_receipt_time <= current_time
```

The network aggregates completed observable transfers by sender and receiver. Its whole-recording mode intentionally includes later transfers. Network nodes use mean observed transfer endpoints, not current player positions.

## Transfer Difficulty

[`src/analytics/quality.py`](../src/analytics/quality.py) calculates an inspectable 0-100 retrospective score.

| Component | Maximum contribution |
| --- | ---: |
| Distance | 35 |
| Positive progression | 10 |
| Defenders in interior lane | 25 |
| Nearest target defender | 20 |
| Sender pressure | 10 |

The endpoint is the realized transfer endpoint, so the score is retrospective. The inverse score is not a completion probability. Missing defender positions produce unavailable quality rather than an assumed easy transfer.

## Shot Detection

[`src/analytics/shots.py`](../src/analytics/shots.py) searches for persistent, fast, goal-directed independent ball flight.

Default requirements include:

- Speed at least `8 m/s`.
- Start within `35 m` of the relevant goal.
- Sufficient forward velocity.
- Projected goal-line crossing near the goal mouth.
- Time to goal at most `3 seconds`.
- At least 3 persistent candidate frames.
- Direction consistent with the recent controlling team when known.

The speed window is approximately `1.2 seconds`, truncated by missing data and stopped by confirmed reception.

BAS shot windows are measured separately. BAS labels do not create trajectory candidates.

## Validation

[`src/analytics/validation.py`](../src/analytics/validation.py) uses one-to-one matching with Hungarian assignment.

Default timing tolerance is `1 second`.

Pass references include BAS `PASS`, `HIGH PASS`, `CROSS`, and `THROW IN`, with sender agreement required.

Shot references use BAS `SHOT` timing only. They do not independently verify shooter identity.

Validation reports matches, false positives, false negatives, precision, recall, and F1 where references exist. It explicitly marks validation as not independent because the ball reconstruction uses BAS anchors.

## Tactical State Engine

The new tactical branch is under [`src/tactics`](../src/tactics) and is described in [`docs/tactics.md`](tactics.md).

It does not alter the existing renderer, synchronization, centered analytics smoothing, possession detector, transfer outcomes, difficulty values, or shot detector.

Default tactical sampling is `5 Hz`, meaning every fifth frame of the 25 FPS source. The configured rate must divide the source FPS exactly.

### Shape geometry

For usable outfield points `p_i`:

```text
centroid = mean(p_i)
width = max(y) - min(y)
depth = max(x) - min(x)
compactness = sqrt(mean(||p_i - centroid||^2))
spread = mean(pairwise Euclidean distance)
```

Important: tactical `depth` is front-to-back team spread, not average advancement toward goal. Orientation-aware advancement is stored as longitudinal position:

```text
longitudinal = centroid_x              when attacking +x
longitudinal = 105 - centroid_x        when attacking -x
```

Goalkeepers are excluded from shape, formation, and ball-relative counts. They remain included in local pressure.

### Block bands

A defending team's outfield centroid is classified by distance from its own goal:

- Low: at most `35 m`.
- Mid: between the low and high thresholds.
- High: at least `70 m`.

A block candidate needs 3 seconds of persistence. Missing evidence or unknown ownership resets the state.

### Pressure

Pressure counts observed opponents within:

- `3 m` near radius.
- `5 m` local radius.
- `8 m` outer radius.

Closing velocity uses a causal backward one-second displacement history, not centered future-looking velocity. A pressure candidate needs at least two local defenders, at least one observed closing defender moving inward at `0.5 m/s` or more, and enough observed defenders. Two seconds of same-carrier evidence supports sustained pressure.

Static proximity is not automatically pressing. Missing defenders make counts lower-bound observations.

### Ball-relative counts

For attacking direction `d`:

```text
signed_position = (player_x - ball_x) * d
```

Values beyond `+/-0.5 m` are ahead or behind; values inside that tolerance are level. Outfield goal-side defenders are counted separately.

### Formation estimates

Formation is a conservative longitudinal line-count fit, not verified lineup information.

Supported templates:

- `4-3-3`
- `4-4-2`
- `4-2-3-1`
- `3-5-2`

The fit uses a trailing 15-second window and requires:

- Ten identity-stable outfield actors.
- Known goalkeeper exclusion.
- At least 80% usable history coverage.
- Adjacent line gaps of at least `8 m`.
- Maximum within-line depth of `12 m`.
- Best-fit RMSE no greater than `4.5 m`.
- Best-fit margin of at least `1 m` where a runner-up exists.
- At least `80%` membership stability.
- Five seconds of accepted persistence.

Otherwise the formation remains unknown. No player roles are invented.

### Phase comparisons

Possession-versus-defending comparisons need at least 50 usable samples in both phases. A wider-in-possession flag requires a mean width difference of at least `5 m`. Repeated frames are observations, not independent statistical samples.

### Transfer network

The network aggregates completed observable transfers into directed edges. It records:

- Transfer count.
- Mean distance.
- Total and mean progression.
- Mean heuristic difficulty.
- Long-transfer count.
- BAS release/sender support.
- Incoming and outgoing node counts.

A repeated edge can generate a network-pattern event after 3 repeated transfers persist for 3 seconds. This is not network centrality and not a causal tactical conclusion.

### Pre-event evidence

[`src/tactics/evidence.py`](../src/tactics/evidence.py) links important events to the latest tactical state strictly before their original release/start frame.

This separates:

- Prior observed tactical geometry.
- Later confirmation time.
- Retrospective realized-endpoint transfer geometry.

It avoids attaching later outcomes to earlier tactical evidence, while still documenting that upstream ownership and ball data are offline inputs.

## Intelligence and Commentary

The intelligence layer is under [`src/intelligence`](../src/intelligence).

It converts existing deterministic results into closed Pydantic schemas. Events have:

- Stable content-derived IDs.
- Frame and recording-local timestamp.
- Source detector.
- Team and identity namespace.
- Significance.
- Validation state.
- Metrics.
- Optional tactical metrics.

New tactical event types include:

- `TACTICAL_BLOCK_CHANGED`
- `TEAM_WIDTH_INCREASED`
- `TEAM_COMPACTNESS_CHANGED`
- `HIGH_PRESSURE_SEQUENCE`
- `TRANSFER_NETWORK_PATTERN`
- `FORMATION_ESTIMATE_CHANGED`

Context windows support 15, 30, and 60 seconds. Context can include tactical state and recent tactical events in addition to possession, transfers, motion, shots, BAS actions, and limitations.

Gemini is optional. It does not calculate possession, transfers, shots, formations, pressure, or calibration. It receives closed facts and a deterministic sentence catalogue and selects one to three sentence IDs. Unknown fields, extra fields, unsupported identities, and free-form prose are rejected.

The default provider is `none`. The local `template` provider is deterministic and makes no network calls.

## Outputs

Standard analytics outputs include:

- `player_summary.csv`
- `player_frames.csv`
- `ball_frames.csv`
- `possession.csv`
- `passes.csv`
- `transfers.csv`
- `shots.csv`
- `bas_shot_windows.csv`
- `pass_validation.csv`
- `shot_validation.csv`
- `reference_events.csv`
- `measurement_definitions.json`
- `summary.json`
- `motion.png`
- `passes.png`
- `report.html`

Intelligence outputs include:

- `events.json`
- `rolling_context.json`
- `commentary.json`
- `commentary_cache/`

Tactical outputs include:

- `tactical_states.csv`
- `tactical_states.json`
- `tactical_events.json`
- `tactical_summary.json`
- `transfer_network.csv`
- `transfer_network.json`
- `pre_event_evidence.json`
- `tactical_shape.png`
- `tactical_pressure.png`
- `tactical_profile.json`

Projection-audit outputs include:

- `projection_evaluation.json`
- `projection_pairs.csv`
- `projection_error_by_frame.csv`
- `error_distribution.png`
- `error_over_time.png`
- `pitch_error_heatmap.png`
- `image_error_heatmap.png`

## Interactive Review Report

The generated report is an offline HTML review page. The current tactical report is:

[outputs/analytics_118575_tactics/report.html](../outputs/analytics_118575_tactics/report.html)

It supports:

- Source video and pitch scrubbing.
- Player markers and possession highlights.
- Player speeds and ball speed labels.
- Tactical centroids.
- Shape metric charts for width, depth, and compactness.
- Current pressure text.
- Formation estimates and quality reasons.
- Tactical-event timeline.
- Trailing or whole-recording transfer network.
- Commentary modes.
- Speed and distance callouts.
- Links to detailed CSV/JSON artifacts.

The tactical report's network graph uses completed observable transfers and a trailing 60-second window by default. It does not continuously redraw players' current positions; node positions are mean observed transfer endpoints.

The repository explanation page is:

[docs/current_pipeline_explorer.html](current_pipeline_explorer.html)

That page explains the implementation interactively, while the generated output report shows an actual run's decisions.

## Correctness Evidence

The current test suite provides synthetic known-answer checks for:

- Constant-speed motion and distance.
- Noise reduction and isolated spikes.
- Missing segments and teleport rejection.
- Possession persistence, contesting, flybys, and relative dribbling.
- Completed transfers, interceptions, self-recovery, timeouts, and missing gaps.
- Difficulty geometry and attack direction.
- Shot direction, speed, persistence, and partial windows.
- One-to-one validation and false negatives.
- BAS clock and one-based frame alignment.
- GSR actor joins despite compact-slot changes.
- Raw-GSR fallback and missing-GSR fallback.
- Conflicting mappings and team labels.
- Goalkeeper metadata and jersey fallback.
- Projection audit statistics and truthful headers.
- Tactical width, depth, compactness, spread, orientation, and goalkeeper exclusion.
- Tactical block, pressure, closing velocity, and missing-data rules.
- Formation fits, ambiguity, persistence, and no-future-data leakage.
- Transfer network aggregation and window boundaries.
- Closed tactical intelligence schemas and prompt restrictions.
- Commentary caching, budgets, failure handling, and report interaction.

The passing tests support implementation behavior. They do not independently validate the upstream GSR annotation quality, ball interpolation quality, or absolute projection accuracy.

## Limitations

The current system does not provide:

- Ball height or 3D ball speed.
- Measured strike speed.
- Verified jersey numbers from the active MOT file.
- Verified roster identities for all players.
- Independently surveyed pitch coordinates.
- Causal/live motion metrics.
- Guaranteed intentional-pass classification.
- Calibrated transfer-completion probability.
- Independent shot accuracy evaluation.
- Guaranteed goalkeeper assignment without authoritative metadata.
- Guaranteed accurate projection in panoramic interior regions.
- Proof that pressure caused a turnover.
- Proof that a formation estimate represents an analyst-verified lineup.
- Official passing-network statistics.

The system is currently offline because centered smoothing uses future frames, ball positions are event-interpolated, and event outcomes often require later persistence confirmation. A future live demo would need causal filters, live ball detection, provisional event states, and delayed confirmation logic.

## Key Implementation Files

### Vision and synchronization

- [`src/vision/data_loading.py`](../src/vision/data_loading.py): dataset paths and low-level loaders.
- [`src/vision/align_clip.py`](../src/vision/align_clip.py): timeline and identity synchronization.
- [`src/vision/coordinate_transform.py`](../src/vision/coordinate_transform.py): Delaunay/barycentric projection.
- [`src/vision/tactical_tracking.py`](../src/vision/tactical_tracking.py): GSR-first tactical joins.
- [`src/vision/visualize_pitch.py`](../src/vision/visualize_pitch.py): video rendering.
- [`src/vision/projection_evaluation.py`](../src/vision/projection_evaluation.py): projection validity audit.

### Analytics

- [`src/analytics/data.py`](../src/analytics/data.py): recording normalization.
- [`src/analytics/motion.py`](../src/analytics/motion.py): motion and quality gates.
- [`src/analytics/possession.py`](../src/analytics/possession.py): ownership rules.
- [`src/analytics/passes.py`](../src/analytics/passes.py): observable transfers.
- [`src/analytics/quality.py`](../src/analytics/quality.py): transfer difficulty.
- [`src/analytics/shots.py`](../src/analytics/shots.py): shot candidates.
- [`src/analytics/validation.py`](../src/analytics/validation.py): reference matching.
- [`src/analytics/pipeline.py`](../src/analytics/pipeline.py): orchestration and exports.
- [`src/analytics/report.py`](../src/analytics/report.py): offline report generation.

### Tactics

- [`src/tactics/config.py`](../src/tactics/config.py): thresholds and attack-direction changes.
- [`src/tactics/shape.py`](../src/tactics/shape.py): team geometry and ball-relative counts.
- [`src/tactics/pressure.py`](../src/tactics/pressure.py): causal closing pressure.
- [`src/tactics/formations.py`](../src/tactics/formations.py): conservative formation fits.
- [`src/tactics/engine.py`](../src/tactics/engine.py): sampled tactical states and events.
- [`src/tactics/passing_network.py`](../src/tactics/passing_network.py): directed transfer networks.
- [`src/tactics/evidence.py`](../src/tactics/evidence.py): strictly pre-event evidence.
- [`src/tactics/output.py`](../src/tactics/output.py): tactical exports and plots.
- [`src/tactics/report.py`](../src/tactics/report.py): synchronized tactical widgets.

### Intelligence

- [`src/intelligence/schema.py`](../src/intelligence/schema.py): closed event/context schemas.
- [`src/intelligence/events.py`](../src/intelligence/events.py): canonical event construction.
- [`src/intelligence/context.py`](../src/intelligence/context.py): rolling facts.
- [`src/intelligence/pipeline.py`](../src/intelligence/pipeline.py): intelligence/tactics integration.
- [`src/intelligence/commentary/service.py`](../src/intelligence/commentary/service.py): selection, validation, caching.
- [`src/intelligence/commentary/gemini_provider.py`](../src/intelligence/commentary/gemini_provider.py): bounded Gemini REST adapter.

## Bottom Line

MatchMind currently provides a reproducible offline path from SoccerTrack inputs to synchronized tactical video, metric-space motion, conservative possession and transfer inference, shot candidates, tactical geometry, pressure and formation evidence, transfer networks, projection diagnostics, closed intelligence events, optional bounded Gemini commentary, and an interactive review report.

The strongest current design choices are:

- GSR-first metric coordinates.
- Actor-ID joins instead of compact-slot assumptions.
- Explicit missingness and unknown states.
- Persistence-gated event publication.
- Causal backward histories inside the tactical branch.
- Strictly pre-event evidence links.
- Closed Pydantic schemas.
- Deterministic sentence grounding.
- Source hashes and measurement definitions.
- Projection auditing without unsupported calibration claims.

The main remaining uncertainty is upstream data quality and interpretation: synchronization, calibration, GSR quantization, event-anchored ball reconstruction, tracker identities, and the gap between geometric evidence and soccer intent.
