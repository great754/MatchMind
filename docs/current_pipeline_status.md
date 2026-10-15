# MatchMind Current Pipeline Status

Status date: 2026-10-14

Current repository revision: `89cd0fe` (`main`, pushed to `origin/main`). The current worktree contains a larger uncommitted milestone on top of that commit. It includes updates to the README, analytics/report/evaluation/tactics/intelligence/query tests and configuration, plus new service/deployment files, provider adapters, diagnostics, shot experiments, Azure architecture documentation, and the pipeline explorer/status documents. The local `.env`, datasets, caches, and generated outputs remain ignored.

This document describes the current MatchMind repository as implemented now. It covers the input data, synchronization, coordinate systems, tactical video renderer, deterministic analytics, new tactical-state engine, intelligence/commentary layer, projection-validity audit, generated outputs, verification numbers, and limitations.

## Executive Summary

MatchMind combines an offline soccer-analysis engine and an artifact-only online serving system built around SoccerTrack match `118575`. It has five connected responsibilities:

1. **Tactical video rendering**: combine panoramic match footage with a synchronized top-down pitch.
2. **Deterministic match analytics**: calculate motion, possession, observable transfers, transfer difficulty, and shot candidates.
3. **Tactical state analysis**: sample team geometry and pressure at 5 Hz, then publish conservative block, shape, formation, and transfer-network evidence.
4. **Grounded intelligence and review**: serialize closed events, build rolling context, optionally select approved commentary sentences, and expose the results through an offline HTML report.

5. **Online Ask and serving**: load validated precomputed bundles, expose grounded API answers and follow-ups, and serve the existing report with optional Microsoft/Gemini presentation and local fallback.

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

The canonical command is:

```bash
.venv/bin/python -m unittest discover -s tests -t . -v
```

Verified 2026-10-14: **216 discovered, 216 passed, 0 failed, 0 skipped**. This milestone began with 176 passing tests and added 40 tests. Shared fixtures live in `tests/helpers.py`; evaluation/query modules import correctly in package and discovery contexts. The earlier 117-test claim is stale and superseded by this measured run.

See [the current Microsoft demo milestone](microsoft_demo.md) for complete metrics, diagnostic/experiment artifacts, held-out availability, API commands and actual versus planned Microsoft integration. The original baseline remains unchanged. Both halves of 118575 are now correctly labeled development; held-out metrics remain unavailable.

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

## Reproducible Evaluation Layer

The new [`src/evaluation`](../src/evaluation) package evaluates multiple recordings or halves from a manifest. It is separate from detector implementation and does not fit thresholds.

Run it with:

```bash
.venv/bin/python -m src.evaluation \
  --manifest configs/evaluation.118575.json \
  --output outputs/evaluation
open outputs/evaluation/evaluation_report.html
```

The manifest supports:

- Match ID.
- `clip` or `half` scope.
- Half number.
- `development` or `evaluation` split.
- Shared analytics configuration.
- Shared tactics configuration.

The evaluator rejects duplicate recordings and overlapping time intervals so the same events cannot be counted twice. Development recordings remain visible in the report but are excluded from aggregate evaluation metrics. Both downloaded halves of match `118575` are additional recordings of one development match, not held-out matches, so no cross-match generalization claim is possible.

It measures:

- Transfer matching with sender agreement.
- Shot matching.
- Transfer timing errors.
- Possession coverage and episode diagnostics.
- Goal references.
- Tactical threshold sensitivity.
- Per-recording and aggregate provenance.

The transfer evaluation currently reports:

| Scope | Candidates | BAS references | Matched | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Half 1 | 432 | 556 | 344 | 79.6% | 61.9% | 69.6% |
| Half 2 | 435 | 538 | 362 | 83.2% | 67.3% | 74.4% |
| Development aggregate | 867 | 1,094 | 706 | 81.4% | 64.5% | 72.0% |

The shot evaluation currently reports:

| Scope | Candidates | BAS references | Matched | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Half 1 | 25 | 10 | 3 | 12.0% | 30.0% | 17.1% |
| Half 2 | 21 | 11 | 2 | 9.5% | 18.2% | 12.5% |
| Development aggregate | 46 | 21 | 5 | 10.9% | 23.8% | 14.9% |

Aggregate transfer timing error is approximately `0.1037 s` mean, `0.08 s` median, and `0.40 s` P95. These are BAS consistency checks, not independent accuracy, because ball reconstruction uses BAS anchors.

Sensitivity variants change pressure and formation outputs, which exposes threshold dependence. No variant is promoted to the default because it looks better. The second-half formation estimate remains unknown under the current fixed-roster requirement because substitutions create more than ten actor identities across the recording.

## Ask MatchMind Query Layer

The current worktree adds a local grounded Q&A layer under [`src/query`](../src/query). Its deterministic core reads structured exports without raw tracking, SQL or network access. The optional provider orchestration around that core can call Microsoft or Gemini for bounded tool/statement selection; it does not perform unrestricted natural-language arithmetic.

Example:

```bash
.venv/bin/python -m src.query \
  --recording outputs/analytics_118575_tactics \
  --question "Who was the fastest player?" \
  --mode analyst
```

The report-side integration in [`src/analytics/report.py`](../src/analytics/report.py) adds:

- An Ask MatchMind question panel.
- Suggested grounded questions.
- Beginner, Fan, and Analyst presentation modes.
- Explain buttons keyed to canonical event IDs.
- Timestamp seeking through the existing report controls.
- Source and fact provenance.

Supported tool categories include:

- Match, player, and possession summaries.
- Filtered canonical events.
- Transfer lookup and difficulty ranking.
- Tactical state at or before a timestamp.
- Strictly pre-event evidence.
- Pressure sequences.
- Formation transitions.
- Transfer networks.
- Fastest players and peak timestamps.
- Event validation.

Every query argument is validated against a closed schema. Facts carry stable IDs, record IDs, source files, field names, event IDs, and timestamps where available. Unsupported questions about fatigue, coaching instructions, player intent, names, scores, or unsupported causality return an explicit unknown answer.

Explain answers keep three kinds of evidence separate:

1. What the canonical event says happened.
2. What was observed strictly before the original event release/start.
3. What retrospective geometry or validation says afterward.

No query answer claims that pressure caused a turnover or that a player intended a pass or shot. Gemini commentary remains available; Gemini query presentation now shares the canonical fact and closed-statement boundary with Microsoft and local answers.

The new Q&A tests cover closed dispatch, known answers, unknown questions, modes sharing facts, fact-field traceability, safe HTML text, Explain IDs, and timestamp seeking. These tests are included in the canonical 216-test suite; shared fixtures no longer import another test module.

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

Evaluation outputs include:

- `evaluation_summary.json`
- `evaluation_report.html`
- `per_match/<match>_<scope>_h<half>/evaluation.json`
- Per-recording analytics and sensitivity artifacts.

Query/report outputs include:

- `ask_matchmind.json`
- Fact records embedded in the generated report.
- Explain-button event references and timestamp links.

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
- Ask MatchMind questions.
- Beginner, Fan, and Analyst answer modes.
- Explain buttons linked to canonical event IDs.
- Fact/source provenance for grounded answers.
- Links to detailed CSV/JSON artifacts.

The evaluation report is a separate standalone page with aggregate metrics, per-recording measurements, split policy, sensitivity results, possession diagnostics, and documented failure modes.

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

## Current Evaluation and Serving Milestone

The current uncommitted milestone adds several layers on top of the committed offline pipeline:

### Diagnostics and experiments

- `src/analytics/shot_evidence.py` exposes per-frame shot-gate evidence and direction provenance.
- `src/evaluation/shot_diagnostics.py` explains matched shots, false positives, and false negatives without changing detection.
- `src/evaluation/transfer_diagnostics.py` audits all transfer false positives and false negatives without changing transfer rules.
- `src/evaluation/shot_experiments.py` runs ten one-factor shot ablations plus a baseline reproduction.
- `src/evaluation/diagnostics.py` validates source hashes and exported inputs before creating diagnostic artifacts.
- Shot production remains the original baseline: `5 TP`, `41 FP`, `16 FN`, precision `10.8696%`, recall `23.8095%`, F1 `14.9254%`.
- The best-looking development ablation is not promoted. The project explicitly refuses to tune production thresholds against dependent BAS references from this single match.

### Generalization policy

- Both downloaded halves of match `118575` are development recordings.
- The evaluation aggregate reserved for held-out recordings is currently empty.
- `configs/evaluation.heldout.example.json` is preparation only; no held-out match is installed.
- Cross-match accuracy and generalization are therefore unknown.
- Future evaluation must freeze detector configurations before inspecting held-out results.

### Current provider, artifact and service architecture

Offline: video/data → synchronized GSR-first analytics → tactics → canonical evidence → precomputed artifacts. Online: Local/Azure Blob ArtifactStore → deterministic tools → local/Microsoft/Gemini/optional Agent Framework → structured answers → report/API. Providers select tools/statements; deterministic Python remains authoritative.

`src/providers/agent.py` emits an Answer plus an evidence list (fact_id, event_id, timestamp, source, field, value, category), safe provider/fallback status, selected tools, provider latency and small structured conversation references. Categories distinguish observed identity/metadata, inferred measurements, BAS support where present, heuristic labels and unknown values. BAS support is not independent validation. All providers receive the same canonical facts for a question; unrelated tool plans, invented fields/numbers and invalid selections fall back. Unsupported questions bypass providers.

`src/providers/conversation.py` supports event explanation, “why?”, “take me there”, and player follow-ups. References are recording-scoped and must resolve in the current artifacts; a caller-supplied ID/time is not new evidence. The report includes provider selection, suggestions, modes, source disclosure, limits, event cards, timestamp seeking and actor highlighting. Local HTML is network-free. Service-served HTML uses a same-origin API; offline facts/commentary remain available during outages. No model credentials enter HTML.

`src/artifacts/store.py` defines ArtifactStore, LocalArtifactStore and AzureBlobArtifactStore. Local bundles are validated at startup. Blob snapshots use allowlisted filenames, per-file hashes, startup download/cache and the same schema validation. Missing/corrupt startup artifacts fail readiness rather than becoming fabricated measurements. Bundle export excludes raw data/video, secrets and caches. IDs include scope/half; clip/half overlap must not be aggregated twice.

### Service routes and observability

| Route | Behavior |
|---|---|
| GET /health | Liveness; precomputed serving status |
| GET /ready | Readiness; empty registry returns 503 |
| GET /matches/{recording_id}/summary | Grounded summary facts |
| GET /matches/{recording_id}/events | Event facts, optional team/type filters |
| GET /matches/{recording_id}/players/{actor_id} | Actor namespace summary; unknown actor stays unknown |
| POST /matches/{recording_id}/ask | Closed request: question, mode, timestamp, optional event/provider/conversation |
| GET /matches/{recording_id}/events/{event_id}/explain | Grounded event evidence; absent event returns 404 |
| GET /matches/{recording_id}/report | Existing report with public same-origin API configuration |
| GET /matches/{recording_id}/{asset} | Allowlisted PNG; optional configured clip video.mp4 |

Startup loads all registered artifacts once. Requests never rerun analytics. Correlation IDs and structured logs record route template, status, request/provider latency, provider/query/tool selection, fallback and unknown counts. Questions, prompts, headers and secrets are excluded. Uvicorn uses --no-access-log to avoid a second raw-URL log. POST limit: 16 KiB; question limit: 2,000 characters. Optional Bearer auth protects match routes. Hosted browser gateway/login, distributed rate limiting and cloud media hosting remain unfinished.

### Explicit verification status — 2026-10-14

| Component | Status | Actual verification |
|---|---|---|
| Local API and demo | IMPLEMENTED AND VERIFIED | Actual loopback HTTP requests and eight-turn artifact demo |
| Docker service | IMPLEMENTED AND VERIFIED | Image built and ran; real container HTTP smoke passed |
| Microsoft REST provider | IMPLEMENTED BUT MOCK-VERIFIED ONLY | Request/schema/timeout/fallback tests; no configured resource |
| Agent Framework runtime adapter | IMPLEMENTED BUT MOCK-VERIFIED ONLY | Real SDK invocation code exists; fake-SDK unit test; SDK not installed and real agent did not run |
| Gemini query adapter | IMPLEMENTED BUT MOCK-VERIFIED ONLY for hosted calls this milestone | Fact parity/fallback tests; existing commentary remains supported |
| Local ArtifactStore and bundle packaging | IMPLEMENTED AND VERIFIED | Real bundles and corrupt/missing/traversal tests |
| Azure Blob adapter | IMPLEMENTED BUT MOCK-VERIFIED ONLY | SDK client injection, hash/cache/error tests; no upload/download against Azure |
| Container Apps Bicep/script | PREPARED BUT NOT DEPLOYED | Static checks only; no Azure CLI/subscription configured |
| Fabric CSV analytical export | IMPLEMENTED AND VERIFIED locally | Six typed/keyed tables with deterministic row counts and hashes |
| Fabric Delta import | PREPARED BUT NOT DEPLOYED | Notebook importer exists; Fabric received no data |
| Azure MCP | PREPARED boundary; real integration NOT IMPLEMENTED | Read-only operator policy; no MCP server ran |

No live Microsoft request succeeded. No Azure deployment occurred. No real Agent Framework execution, Fabric upload or MCP execution is claimed. Official Microsoft docs were checked for current resource OpenAI-v1 structured outputs and the project-backed Agent/FoundryChatClient distinction; setup references are in deploy/azure/README.md.

Docker local measurement: image matchmind-service **94,649,434 bytes**, UID **10001**, port **8000/tcp**, readiness about **4.99 seconds**. Three artifact directories were mounted read-only. The optional local clip-media route was verified with an actual HTTP byte-range request; configure MATCHMIND_MEDIA_ROOT for API video playback. The image copies only online code and service dependencies; no raw dataset, outputs, caches, .env or vision packages. Azure Blob dependencies require WITH_AZURE=true; the default verified image is local. Cloud size/startup and load behavior are unmeasured.

### Configuration and exact commands

| Variable | Meaning/default |
|---|---|
| MATCHMIND_PROVIDER | local (default), gemini, microsoft, microsoft-framework |
| MATCHMIND_REGISTRY | configs/serving/local.json; trusted recording aliases to local bundles |
| MATCHMIND_ARTIFACT_STORE | local (default) or azure |
| MATCHMIND_API_TOKEN | Optional server-side match-route Bearer auth; never included in report |
| MATCHMIND_MEDIA_ROOT | Optional trusted directory for clip MP4 files; unset by default |
| AZURE_OPENAI_ENDPOINT / AZURE_OPENAI_DEPLOYMENT / AZURE_OPENAI_API_KEY | Optional resource REST endpoint, model deployment and server secret |
| FOUNDRY_PROJECT_ENDPOINT / FOUNDRY_MODEL | Optional project-backed framework runtime; also requires SDK/Entra login |
| GEMINI_API_KEY | Existing optional server-side Gemini key |
| MATCHMIND_BLOB_ACCOUNT_URL / MATCHMIND_BLOB_CONTAINER / MATCHMIND_RECORDINGS | Blob account/container and JSON recording-ID list |
| MATCHMIND_CACHE | /tmp/matchmind-artifacts, startup Blob snapshot cache |

```bash
.venv/bin/python -m unittest discover -s tests -t . -v
.venv/bin/python -m uvicorn src.serving.app:from_environment --factory --host 127.0.0.1 --port 8000 --no-access-log
# Browser: http://127.0.0.1:8000/matches/118575_clip/report
.venv/bin/python scripts/verify_service.py
.venv/bin/python -m scripts.demo_matchmind
.venv/bin/python -m src.query --recording outputs/analytics_118575_tactics --question 'Who was the fastest player?' --mode analyst

docker build -f Dockerfile.service -t matchmind-service .
docker run --rm -p 127.0.0.1:8000:8000 -v "$PWD/outputs:/app/outputs:ro" matchmind-service
.venv/bin/python scripts/verify_container.py

# Microsoft: configure AZURE_OPENAI_ENDPOINT/DEPLOYMENT/API_KEY in server environment first.
MATCHMIND_PROVIDER=microsoft .venv/bin/python -m uvicorn src.serving.app:from_environment --factory --host 127.0.0.1 --port 8000 --no-access-log
.venv/bin/python -m scripts.demo_matchmind --provider microsoft

.venv/bin/python -m scripts.refresh_report_ui
.venv/bin/python -m src.artifacts
.venv/bin/python -m src.fabric
```

### Analytical dataset, tests and remaining work

Fabric-ready CSV counts: Match **3**, Player **73**, Event **11,707**, Transfer **905**, TacticalState **60,510**, Evaluation **4**. schema.json defines types, primary keys, nullability and hashes. Match records retain source/configuration/limitations. Actor IDs and tracker IDs remain separate. Timestamps are recording-local seconds, never wall-clock timestamps. Evaluation rows remain development only. The importer verifies counts/keys and creates new Delta tables; it has not run against Fabric.

The 40 new product tests cover store loading/corruption/path traversal/hash caching, deterministic Fabric schemas/identity/timing/counts, provider fact parity and outages, optional framework routing with a fake SDK, evidence propagation and rejected fabrication, recording-scoped follow-ups, readiness, body limits, sanitized logs and static deployment/container boundaries. Existing report tests execute offline and API browser-script paths in Node, including provider selection, contextual seeking and source-safe rendering.

Frozen development results are unchanged: transfers 706 TP / 161 FP / 388 FN, precision 81.430219%, recall 64.533821%, F1 72.004080%; shots 5 TP / 41 FP / 16 FN, precision 10.869565%, recall 23.809524%, F1 14.925373%. This product milestone reads preserved baseline artifacts and does not tune/recompute detectors. No second match is installed; cross-match generalization remains unknown. GSR remains primary; fallback projection disagreement remains 11.6831065 m mean, 10.7398110 m median, 24.9560484 m P95.

Remaining work: actual Foundry request, optional SDK version resolution/pinning and live framework run; real Blob/Key Vault/Container Apps deployment, browser ingress authentication, media hosting and load/rate-limit verification; Fabric upload/import; authorized MCP connection. Scientific limits remain BAS-dependent 2D ball reconstruction, absent height/verified identities, heuristic possession/tactics/difficulty, full-half formation roster limitations and no independent causal evidence. See docs/microsoft_demo.md and deploy/azure/README.md for implementation choices and setup. No planned cloud capability is counted as live integration.
