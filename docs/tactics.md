# Tactical State Engine

The engine adds deterministic geometric evidence to the existing analytics. It does not infer tactics from video with Gemini and does not alter the GSR-first renderer, synchronization, centered analytics smoothing, ownership rules, transfer outcomes, difficulty values or shot detection. Read [the validity audit](validity_audit.md) first: the MOT fallback still disagrees with GSR by 11.6831 m on average, and no calibration improvement is claimed.

## Run and review

```bash
.venv/bin/python -m src.analytics.pipeline --summary-only \
  --commentary-provider template --output outputs/analytics_118575_tactics
open outputs/analytics_118575_tactics/report.html
.venv/bin/python -m unittest discover -s tests -v
```

`template` is an explicitly labeled local preview, with no hosted requests. `none` is still the default. Existing `--commentary-provider gemini`, key configuration, cache and budgets also support tactical facts. Use `--no-tactics` to omit tactical calculations. Use `--tactics-config configs/tactics.default.json` or a JSON file with selected overrides to reproduce or adjust rules. A tactical rate must divide the source FPS exactly; the default is every fifth 25 FPS frame, giving 5 Hz.

The report retains the existing video, pitch, commentary modes, plots, tables and limitations. New controls show centroids, shape metric histories, current pressure, ball-relative counts, estimated formations and a tactical-event timeline. Click the shape chart or a tactical event to seek the same video/pitch controls. The Transfer Network defaults to completed transfers in the trailing window at the current video time. Its explicit whole-recording option includes later transfers; possession/defending summary comparisons are also labeled whole-recording statistics. Network nodes use mean observed transfer endpoints, not current player positions.

## Code map

| Module | Purpose |
| --- | --- |
| `src/tactics/config.py` | Effective thresholds, sampling and scheduled attack-direction reversals |
| `schema.py` | Closed, finite tactical facts and explicit unknown reasons |
| `shape.py` | Outfield geometry, orientation, defensive-band candidates, ball-relative counts |
| `pressure.py` | Observed proximity and backward relative closing velocity |
| `formations.py` | Trailing longitudinal line-count fit and deterministic quality gates |
| `events.py` | Persistence, cooldown, one-shot latches and canonical tactical events |
| `passing_network.py` | Directed completed observable-transfer aggregation at any requested window |
| `engine.py` | Sampled states, goalkeeper-role lookup, phase summaries and event generation |
| `evidence.py` | Strictly earlier snapshots for original event releases/starts |
| `output.py` | CSV/JSON, source hashes, definitions, plots and compact report data |
| `report.py` | Synchronized review widgets using the existing `seek()` / `draw()` functions |

The existing intelligence pipeline merges tactical canonical events chronologically, puts the latest sample at or before the requested timestamp in rolling context, and supplies only closed facts to commentary. Gemini selects sentence IDs from supported statements; it cannot introduce free-form tactical intent or causality. Extra tactical fields are rejected. Cached requests include the structured tactical context and updated prompt version.

## Formulas and interpretation

**Usable coordinates** are finite observations inside the 105×68 field. Geometry uses raw recording metric positions to avoid borrowing later coordinate samples through centered smoothing. Quantization remains visible; persistence and formation histories reduce interpretation of brief changes. Player identities are GSR actors joined to MOT tracker labels. Roles come from provider XML actor metadata or explicit caller-supplied `recording.goalkeeper_ids`. A team needs exactly one identified roster goalkeeper to support outfield shape/formation. An unknown/ambiguous keeper withholds shape rather than guessing a formation from kit color. The separate renderer's existing goalkeeper fallback remains unchanged.

For usable outfield points `p_i`, centroid `c = mean(p_i)`, width `max(y)-min(y)`, and depth `max(x)-min(x)`. Compactness is `sqrt(mean(||p_i-c||²))`; lower values indicate a tighter observed distribution. Spread is mean Euclidean distance over distinct player pairs. Longitudinal position is centroid x when attacking +x, or `105-centroid_x` when attacking -x; lateral position is centroid y. Centroid separation is Euclidean distance between both usable outfield centroids. All these measurements are in meters. Goalkeepers are excluded; shape requires at least the configured outfield count, and more than ten observed outfield players is ambiguous rather than an accepted extra-wide shape.

**Block height** is a heuristic band of the defending team's outfield centroid distance from its own goal. Default boundaries divide pitch length into thirds: low ≤35 m, high ≥70 m, mid in between. It is evaluated only while the other team has confirmed ownership and enough outfield observations exist. Startup is unknown until three seconds of consistent sampled evidence. A confirmed band is retained during a different pending band, then changes once that band persists. Missing evidence or unknown ownership immediately resets to unknown; no label is guessed through a gap. This is not a conventional analyst-verified defensive-line height model.

**Local pressure** counts observed opponents within three/five/eight meters of the confirmed carrier and reports the nearest distance, available/expected roster opponents, and nearby teammates including the carrier. Counts include goalkeepers because they can challenge a carrier. Local numerical balance is friendly count including carrier minus opponent count within the local radius. Missing opponents make counts lower-bound observations, not guaranteed full-team counts.

Closing velocity uses backward one-second displacement for the carrier and each defender. Any missing position in that interval invalidates that player's velocity; implausible individual average displacement velocities above the configured cap are unavailable. For relative position `r = defender - carrier`, closing speed is `-(v_defender-v_carrier) dot (r/||r||)`, positive toward the carrier. Coincident points cannot establish a velocity direction. A press candidate needs at least two opponents within five meters, at least one observed closing opponent within eight meters moving inward at ≥0.5 m/s, and sufficient available defenders. Only two seconds of continuous candidate evidence with the same carrier supports `sustained_press`. Static proximity is never pressing. The estimate does not establish collective intent or that pressure caused any turnover.

**Players relative to the ball** use `(player_x-ball_x)*attacking_direction`. Values beyond ±0.5 m are ahead/behind; those inside the tolerance are level. Goal-side defenders lie toward their own defended goal in the attacking team's positive direction. `defenders_behind_ball_to_own_goal` is an explicit synonym for that count. Both teams' identified goalkeepers are excluded, and availability counts are retained. These are observed positions, not inferred tactical roles. `AnalyticsConfig.left_attack_direction` establishes initial orientation; scheduled `attack_direction_changes` reset persistent/formation histories so opposite orientations cannot be averaged together.

**Possession versus defending** compares sample means for width, depth, compactness, spread and centroid position. Each phase needs at least 50 usable observations (ten sampled seconds at 5 Hz). A wider-in-possession flag requires a mean width difference ≥5 m. The report shows both sample counts and means. These are descriptions of observed samples, not independent statistical observations, stable season tendencies, or evidence of a coach's instruction. Variable player coverage can bias width estimates.

**Formation** means a conservative longitudinal line-count estimate, not verified lineup information. Supported templates are 4-3-3, 4-4-2, 4-2-3-1 and 3-5-2; all other or ambiguous geometry returns unknown. Each one-second update uses a trailing 15-second window with ten identity-stable outfield actors, known goalkeeper exclusion and sufficient complete samples. Players' mean directed x positions are sorted and partitioned according to each template. Candidate fit RMSE is square-root mean longitudinal squared residual around assigned line centers. Adjacent centers must be at least eight meters apart; a line cannot span more than 12 m in depth. Best-fit RMSE must be ≤4.5 m and exceed the runner-up by at least 1 m when a second fit survives. At least 80% average player/line membership must agree across the window. The accepted estimate must then persist for five seconds. An unavailable or changed fit returns unknown while new evidence accumulates; fit RMSE, margin, sample count/window and membership stability remain available. `clear` means these gates passed, not a probability of tactical correctness. No winger/defender roles are invented. Averaging can hide rotations; GSR quantization and longitudinal-only line structure limit interpretation.

**Transfer Network** contains completed observable transfers, not necessarily intentional passes. Directed edges aggregate counts, available distance/progression/difficulty means, total progression, long-transfer counts and BAS-supported pass counts. Field availability counts distinguish a missing metric from zero. Nodes retain tracker labels and incoming/outgoing transfer counts. BAS support refers only to release timing/sender agreement, not independent intent or outcome validation. The full network, full per-team networks and configurable trailing snapshots are exported. The Python API supports arbitrary finite ordered windows:

```python
from src.tactics.passing_network import transfer_network
network = transfer_network(events, start_seconds=60, end_seconds=120,
                           team='left', long_transfer_m=25)
```

Only receipts confirmed at or before the window endpoint enter the network. A three-transfer directed edge in a trailing 60-second window can create a network-pattern event after three seconds of persistence. No network centrality is presented as a tactical cause.

## Tactical events and prior evidence

Canonical events include `TACTICAL_BLOCK_CHANGED`, `TEAM_WIDTH_INCREASED`, `TEAM_COMPACTNESS_CHANGED`, `HIGH_PRESSURE_SEQUENCE`, `TRANSFER_NETWORK_PATTERN` and `FORMATION_ESTIMATE_CHANGED`. Their source is `tactical_state_engine`, observation kind is `tactical_state`, and nested `tactical` metrics carry the supporting measurements. Timestamps mark confirmation, with an earlier evidence start where meaningful. Shape changes compare two trailing ten-second windows with adequate coverage, then require three seconds of a significant difference. Persistent triggers emit once per continuous episode and rearm after evidence resets, subject to a ten-second cooldown. Block and formation events describe confirmed state transitions rather than repeated per-frame triggers.

`pre_event_evidence.json` links completed/difficult/progressive transfers, opposite-team transfers, ownership changes and shot candidates to the latest tactical sample **strictly before the original release/start**, not before their later publication time. Startup events have explicit unavailable evidence. Existing lane/target difficulty features use the realized endpoint, so they are stored separately under `retrospective_geometry`, with `uses_realized_endpoint=true`. They must not be mistaken for a causal feature known at release.

There is no later tactical coordinate sample or later transfer outcome in an earlier tactical window. This is nevertheless an **offline** system: upstream ownership uses centered smoothing, and the supplied ball reconstruction interpolates between event anchors. Strict earlier timestamp selection does not turn those source measurements into a physically causal live feed. No event says pressure forced a turnover or caused a shot.

## All configurable thresholds

See `configs/tactics.default.json` for a complete machine-readable configuration. CLI JSON overrides merge with defaults; rates and thresholds are heuristic, not calibrated.

| Configuration | Default | Meaning |
| --- | ---: | --- |
| `sample_hz` | 5 | Tactical sampling frequency; must divide source FPS |
| `min_outfield_players` | 8 | Minimum usable outfield observations for shape |
| `block_low_max_m` | 35 | Low-band maximum own-goal centroid distance |
| `block_high_min_m` | 70 | High-band minimum own-goal centroid distance |
| `block_persistence_seconds` | 3 | Continuous band evidence before confirmation |
| `pressure_near_radius_m` | 3 | Near-opponent count radius |
| `pressure_local_radius_m` | 5 | Local pressure/numerical balance radius |
| `pressure_outer_radius_m` | 8 | Outer count and closing-opponent radius |
| `pressure_min_defenders` | 2 | Local opponents required for candidate |
| `pressure_min_observed_defenders` | 8 | Observed opponents required for candidate |
| `closing_min_mps` | 0.5 | Minimum positive relative closing speed |
| `closing_max_mps` | 14 | Maximum accepted individual backward displacement velocity |
| `closing_lookback_seconds` | 1 | Backward displacement interval |
| `press_persistence_seconds` | 2 | Same-carrier candidate evidence duration |
| `ball_line_tolerance_m` | 0.5 | Ahead/behind level-band half-width |
| `shape_window_seconds` | 10 | Each of two trailing shape-comparison windows |
| `shape_window_min_fraction` | 0.8 | Minimum usable fraction in both shape windows |
| `width_change_m` | 5 | Minimum positive mean-width change |
| `compactness_change_m` | 4 | Minimum absolute mean RMS-compactness change |
| `shape_persistence_seconds` | 3 | Significant shape-change duration |
| `event_cooldown_seconds` | 10 | Shape/press/network trigger cooldown |
| `comparison_min_samples` | 50 | Minimum usable samples in each ownership phase |
| `comparison_width_margin_m` | 5 | Wider-in-possession mean-width margin |
| `formation_window_seconds` | 15 | Trailing identity-stable formation window |
| `formation_update_seconds` | 1 | Formation fit cadence, aligned to tactical samples |
| `formation_min_fraction` | 0.8 | Complete-roster sample coverage |
| `formation_min_line_gap_m` | 8 | Minimum adjacent line-center separation |
| `formation_max_line_depth_m` | 12 | Maximum within-line longitudinal span |
| `formation_max_rmse_m` | 4.5 | Maximum accepted best-fit residual |
| `formation_min_fit_margin_m` | 1 | Minimum best-versus-runner-up RMSE gap |
| `formation_min_membership_stability` | 0.8 | Minimum mean player/line agreement |
| `formation_persistence_seconds` | 5 | Accepted fit duration before event |
| `network_window_seconds` | 60 | Trailing network/pattern window |
| `network_long_transfer_m` | 25 | Edge long-transfer distance threshold |
| `network_pattern_min_transfers` | 3 | Directed edge repeat count |
| `network_pattern_persistence_seconds` | 3 | Repeated-edge pattern duration |
| `attack_direction_changes` | `[]` | Increasing `[local_frame, left_direction]` pairs; direction ±1 |

For a second-half reversal use initial `left_attack_direction=-1` in the analytics config, or schedule a change when analyzing a combined timeline. For example `"attack_direction_changes": [[3000,-1]]` reverses direction at local frame 3000 and restarts orientation-dependent evidence.

## Outputs and verification

New outputs sit beside the existing CSV/JSON/PNG/report files:

- `tactical_states.csv` / `tactical_states.json`: sampled geometry, availability, phase, block, pressure, relative counts and formation quality.
- `tactical_events.json`: persistent events also merged into `events.json`.
- `transfer_network.json` / `.csv`: full/per-team/window networks and directed edge statistics.
- `tactical_summary.json`: effective configuration, units, definitions, thresholds, missing-data policy, source hashes, role/sync provenance and observed summaries.
- `pre_event_evidence.json`: strictly earlier state links and separately marked retrospective features.
- `tactical_shape.png` / `tactical_pressure.png`: actual observed shape and pressure series.
- `tactical_profile.json`: measured engine/formation/network computation times; wall-clock profiles are not deterministic model input.

The synthetic tests use rectangles with exact centroids/width/depth/RMS radius, mirrored orientations, known pressure distances and velocities, stationary nearby opponents, sustained/missing evidence, canonical synthetic formations and mathematically known edge aggregation. They mutate future coordinates to verify earlier states/events remain unchanged, verify pre-event sampling strictly precedes release, reject unsupported Gemini tactical fields, and execute report JavaScript in a Node DOM/video stub to verify existing and tactical seeking/mode/network controls. Node is optional for the JavaScript portion; the Python tests remain offline.

The initial measured clip run took about 0.60 s for tactical calculations (formation fits about 0.28 s). The 71,850-frame first-half run took about 8.03 s (formation 3.54 s). Timings depend on hardware/load. Formation is fitted at 1 Hz, sampled history is bounded, and network pattern checks consider the trailing confirmed transfers. Existing export/plot costs are separate. No optional Voronoi/territory layer was added; the core evidence and uncertainty take priority.
