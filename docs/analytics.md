# Understanding and testing MatchMind analytics

The pipeline calculates motion, infers possession, finds observable ball transfers, scores transfer difficulty, and finds goal-directed shot candidates. It writes CSVs plus an offline HTML review page. Run it from the repository root:

```bash
source .venv/bin/activate
python -m src.analytics.pipeline
open outputs/analytics_118575_clip/report.html
```

The clip report includes the source video, a scrubber, a top-down view with player speeds, ownership highlights, motion plots, event tables, and validation results. Click an event timestamp to jump to that moment. The review canvas samples every fifth frame for a smaller HTML file; the calculations and CSVs use every frame at 25 FPS. No server or internet connection is needed to run the calculations or open the generated page once the dataset is downloaded.

The existing video renderer is separate. These analytics are exported for inspection and future overlays; running this command does not overwrite the existing rendered videos.

## Read the code in this order

| File | Responsibility |
| --- | --- |
| `src/analytics/config.py` | Defaults and configurable thresholds, in meters, seconds, and m/s |
| `src/analytics/data.py` | Align ball, player and BAS timelines; normalize coordinates and real player IDs |
| `src/analytics/motion.py` | Raw speed, smoothed speed, velocity, cumulative measured distance and quality flags |
| `src/analytics/possession.py` | Frame-by-frame proximity, movement and persistence rules |
| `src/analytics/passes.py` | Controlled player → independent ball → controlled receiver transitions |
| `src/analytics/quality.py` | Defender geometry and transparent difficulty components |
| `src/analytics/shots.py` | Trajectory candidates and separate BAS shot speed windows |
| `src/analytics/validation.py` | One-to-one event matching and explicit misses |
| `src/analytics/pipeline.py` | Connect the stages and export metrics and provenance |
| `src/analytics/report.py` | Plots, playback review, and clickable event tables |

`analyze_recording(recording, config)` is the entry point for using the algorithms from another Python module. The individual detectors accept arrays/dataframes and can be tested without a video or downloaded match.

## Coordinate and time choices

The default player source is the compact GSR arrays produced earlier by `python -m src.vision.align_clip`. GSR player positions and the supplied ball positions share the provider's metric coordinate convention. That makes them preferable for possession calculations to mixing the ball's coordinates with positions estimated through a separate image calibration.

The compact `pitch` arrays already have a corner origin. Ball coordinates have a center-circle origin, so the loader adds **52.5 m to x and 34 m to y**. The result uses x from 0–105 m, increasing to the right in the camera view, and y from 0–68 m, increasing toward the near touchline. Players outside the field can still have motion metrics; an off-field ball cannot have an owner.

All analytics use the real player IDs, such as `476371`. `mot_id` is retained for connecting them to the 1–22 IDs in the clip. Teams use the provider's `left`/`right` side labels, with the existing side-to-color mapping used in the review canvas. For the current first half, `left` attacks increasing x and `right` attacks decreasing x. Set `left_attack_direction` to `-1` when that orientation reverses; team names/colors alone do not determine attack direction.

The synchronized four-minute clip starts at first-half index **5999**, or frame **6000** in one-based annotations. Local analytics frame 0 is that source frame. A source BAS event on frame `k` becomes:

```python
local_frame = k - 1 - 5999
local_seconds = local_frame / 25
```

The alignment is inferred with approximately one-frame precision. `half_video_seconds` in the exports lets you audit the original half-video clock. Files are checked for consistent FPS, one-based consecutive ball frames, unique actors and sufficient timeline coverage. Missing event actors are left blank rather than invented.

If compact GSR files are unavailable, the clip can use the existing MOT-to-pitch calibration explicitly:

```bash
python -m src.analytics.pipeline --player-source mot --output outputs/analytics_118575_mot
```

This is a different coordinate estimate and can change proximity-based results. It cannot analyze the full half because the MOT data only covers the clip.

## 1. Motion

At 25 FPS, `Δt = 0.04 seconds`. The raw speed for frame i is:

```python
step = norm(position[i] - position[i - 1])
speed_mps = step * 25
speed_kmh = speed_mps * 3.6
```

A one-meter quantization step already produces a raw speed of 25 m/s, so raw speed alone is misleading on this dataset. The implementation first applies a three-frame coordinate median to remove isolated spikes, then a degree-two Savitzky–Golay position filter. Default windows are 15 frames for players and 11 for the ball. Differentiating the smoothed positions gives the velocity and smoothed speed; summing their accepted step lengths gives cumulative measured distance.

Smoothing runs separately on each consecutive observed segment. It never fills a missing ball position, adds distance across a gap, or substitutes zero speed for unavailable motion. The first sample of each segment has no speed because there is no previous observed sample. Distance accumulates only accepted intervals and holds its value during gaps; it is the distance covered by usable evidence, not a guarantee of total distance traveled.

Raw values over a configurable limit get `raw_speed_flag`. Smoothed intervals over 14 m/s for players or 50 m/s for the ball are rejected, along with the derivative touching the rejected position. Their speeds and step distances become blank, rather than being clipped to the limit. These limits are quality gates, not claims about a universal physical maximum. Both the raw estimates and rejection flags remain available for debugging.

The filters are centered and use future samples: this is an **offline** analysis, not a live implementation. Smoothing can shift event boundaries and suppress short peaks.

## 2. Possession

For every frame, the detector computes the ball's distance and relative velocity to every usable player. Its defaults are:

- Acquire within **2.5 m** after **five consecutive frames** of evidence (about 0.2 s).
- Retain an existing owner within **3.5 m**, providing hysteresis against tiny distance fluctuations.
- Require ball speed at most **8 m/s**, or relative ball/player speed at most **4 m/s**. Relative motion allows a fast dribble without treating every high-speed flyby as possession.
- If the nearest two players are within **0.5 m of each other in distance to the ball**, report contested ownership rather than forcing a choice.
- Clear ownership when ball position or motion is unavailable, or the ball is outside the pitch.

The frame state is `controlled`, `candidate`, `free`, `contested`, `unknown`, or `out_of_play`. `possessing_player_id` and `possessing_team` are filled only for controlled frames. `candidate_player_id` records unconfirmed proximity. `control_start_frame` records the beginning of the evidence streak once ownership is confirmed, so acquisition delay does not artificially lengthen the reported pass.

`control_confidence` is a rule-derived proximity score, not a calibrated probability. Team possession summaries include both fraction of **all frames** and fraction of **controlled frames**; unknown/free time is not silently credited to a team. This is observed control, not a complete official possession statistic.

## 3. Transfers and passes

The pass state machine remembers the last controlled player, waits for an independent ball interval, and looks for a persistent controlled receiver. At least three independent frames, three meters of displacement, and a peak speed of three m/s are required by default.

A same-team reception becomes `completed`; an opposite-team reception becomes `intercepted`. A ball leaving the pitch becomes `out_of_play`, and a timeout, missing segment or recording boundary can produce `unresolved`. A player recovering their own ball is not a pass. A direct ownership flip without independent travel is not a pass either. A missing interval terminates the attempt; the detector never invents a completed pass across that gap.

`passes.csv` contains sender, actual receiver, team, local start/end frames and times, coordinates, straight distance, measured path distance, duration, independent-frame count, and mean/median/peak ball speeds. The mean is calculated over observed motion intervals. Missing receivers remain blank.

These are **observable transfer candidates**. Geometry cannot always distinguish an intended pass from a clearance, loose ball or tackle. The opponent reception is known; the intended receiver is not. BAS labels do not create these detections.

## 4. Difficulty

The score is deliberately inspectable. Every component is exported:

| Component | Maximum contribution |
| --- | ---: |
| Distance: `min(distance / 50, 1)` | 35 points |
| Positive forward progression: `min(progression / 30, 1)` | 10 points |
| Defenders within two meters of the interior passing lane: `min(count / 3, 1)` | 25 points |
| Closest defender to the target at launch: `max(0, 1 - distance / 5)` | 20 points |
| Defenders within five meters of the sender: `min(count / 3, 1)` | 10 points |

The lane calculation projects each defender onto the sender-to-target line, measures perpendicular distance, and counts projections between 5% and 95% of that segment. Attack direction determines forward progression. Sender, lane and target pressure use defender positions at launch. Pressure at arrival is exported separately and does not enter the score. Missing all opponent coordinates makes the difficulty score unavailable.

`difficulty_score_0_100` is the sum of the components. `heuristic_completion_score_0_1` is `1 - difficulty / 100`, **not a completion probability**. A value of 0.25 does not mean a 25% chance. The target is the realized reception/end position, so this is a retrospective description; a true pre-pass predictor needs an intended target available at release.

For a future probability model, first review and label the transfer candidates, identify intended receivers including unsuccessful attempts, and train on release-time features with match-level train/validation splits. Calibrate held-out predictions and assess calibration/Brier score as well as discrimination. Do not train on receiver pressure at arrival or final outcome, and do not present this heuristic as a trained model.

## 5. Shots and validation

A shot candidate is a persistent, fast independent ball trajectory toward a nearby goal. The detector requires at least three frames, speed at least 8 m/s, a start within 35 m of the goal, sufficient forward velocity, a projected goal-line crossing within the mouth plus a three-meter margin, and time-to-goal at most three seconds. If a recently controlled player's team is known, its attack direction must agree.

The shot's peak and median smoothed ball speeds come from a short 1.2 s trajectory window. Missing positions truncate the window, and confirmed receptions stop it. `valid_speed_samples`, `window_requested_frames`, `window_truncated` and `window_coverage_fraction` show how much evidence supported the number.

Separately, `bas_shot_windows.csv` measures windows around the official BAS Shot timestamps, including shots the trajectory detector missed. A blank speed is an unavailable measurement, not a zero-speed shot. The ball path is a 2D, event-interpolated reconstruction with no height information; these are estimated ground-plane speeds, not measured impact speeds or radar-equivalent shot speeds.

Validation matches detections to references **one-to-one** within one second. Duplicate predictions cannot claim the same annotation. Pass validation uses BAS `PASS`, `HIGH PASS`, `CROSS`, and `THROW IN` and requires sender ID agreement. Shot validation uses BAS `SHOT` and timing only, so its scores do not establish shooter-ID accuracy. Every unmatched detection and missed annotation is exported.

The clip contains no BAS Shots. Its shot precision/recall are therefore null, not perfect scores. To validate against ten actual first-half Shot annotations:

```bash
python -m src.analytics.pipeline --scope half --summary-only
open outputs/analytics_118575_half/report.html
```

`--summary-only` skips the large per-player frame CSV but still computes the full pipeline, ball frames, possession, events, player summaries, validation and report. Omit that flag if you want full-half `player_frames.csv` too.

BAS is also the source of anchors used to reconstruct the supplied ball path. Comparing detections with it is an alignment/consistency check, **not independent accuracy** on a measured ball track. The current baseline shot detector has substantial false positives and misses; see its exported confusion counts before using it for conclusions.

## How to test and inspect results

Run known-answer synthetic tests and the existing annotation tests:

```bash
python -m unittest discover -s tests -v
```

Tests cover constant five-m/s movement and exact distance; noise and isolated spikes; missing segments and teleport rejection; persistence, contested control and fast flybys; completed transfers, interceptions, self-recovery, timeouts and gap cancellation; defender geometry and score direction; goal-directed versus slow/wide/away trajectories; partial shot-window coverage; duplicate matching and missing reference data. Synthetic tests do not reuse the match's BAS labels as the expected detector outputs.

For a visual check, open the clip report and click several completed and intercepted transfer times. Watch whether the ball leaves the reported sender, travels independently, and reaches the receiver. Scrub an unknown interval to verify that neither ownership nor ball speed is fabricated. The yellow ring indicates the controlled player on the review map. Compare player speed changes with the footage rather than accepting raw quantized peaks.

For a numerical check:

```python
import pandas as pd

root = 'outputs/analytics_118575_clip/'
players = pd.read_csv(root + 'player_frames.csv')
ball = pd.read_csv(root + 'ball_frames.csv')
passes = pd.read_csv(root + 'passes.csv')

assert len(players) == 6000 * 22
assert len(ball) == 6000
assert (players.smoothed_speed_mps.dropna() <= 14).all()
assert (ball.smoothed_speed_mps.dropna() <= 50).all()
assert players.groupby('player_id').distance_m.apply(
    lambda x: (x.diff().dropna() >= 0).all()
).all()
assert (passes.duration_seconds > 0).all()
print(passes[['sender_id', 'receiver_id', 'outcome', 'distance_m']].head())
```

To audit one player's distance, sum its nonblank `step_distance_m` values and compare with the last `distance_m`. To audit a pass, inspect the corresponding frame range in `possession.csv` and `ball_frames.csv`. `pass_validation.csv`/`shot_validation.csv` list exact missed and unmatched events for review.

Tune rules through an explicit config file, saving a separate output folder:

```json
{
  "player_smoothing_frames": 21,
  "acquisition_radius_m": 3.0,
  "retention_radius_m": 4.0,
  "possession_frames": 7
}
```

```bash
python -m src.analytics.pipeline --config my_analytics_config.json --output outputs/analytics_experiment
```

Larger smoothing windows reduce noise but blur quick movements. Larger control radii improve coverage but risk false owners; more persistence reduces flicker but misses brief touches. Review both detections and misses after changing them, and keep separate matches for tuning and evaluation.

`summary.json` records the full effective configuration, input paths and SHA-256 fingerprints, frame offset, coverage, counts and caveats. Outputs are deterministic for the same inputs/configuration. Regenerate reports after code or threshold changes rather than mixing files from different runs.
