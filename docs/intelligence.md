# Match Intelligence implementation and testing

This layer consumes `analyze_recording()` results. It does not change smoothing, possession rules, transfer detection, difficulty scoring, shot detection, GSR/MOT mapping or rendered videos. Existing CSV paths and report sections remain available. The existing explainer HTML is untouched.

## Data flow and implementation choices

1. `src/intelligence/events.py` adapts possession transitions, transfer outcomes and quality features, trajectory candidates, BAS annotations and accepted motion metrics into closed Pydantic schemas in `schema.py`. Each event has a content-derived ID, detector group, recording-local frame/time, source, explicit identifier namespace, significance and uncertainty/validation state. MOT IDs are display identifiers; `actor_id` retains the GSR identity for joins. If an actor has no MOT mapping, its namespace is `gsr_actor`.
2. `context.py` exposes `ContextBuilder.at(timestamp, window)`, for 15/30/60 seconds. It reads existing motion and possession results and aggregates canonical events; it does not estimate raw soccer actions again. Exports contain periodic snapshots plus commentary times, while the Python API supports any timestamp within the recording.
3. `commentary/service.py` filters by significance/time range, groups nearby events, selects one representative per detector group, applies cooldown and limits, then builds requests for each mode. Related classifications of one transfer share a group. Urgent BAS shots/goals bypass the ordinary cooldown. All three modes are generated in advance; the UI switches without a network request.
4. `commentary/catalogue.py` renders supported sentences with estimates and caveats intact. `commentary/base.py` defines the replaceable provider protocol. `gemini_provider.py` uses Google's [generateContent API](https://ai.google.dev/api/generate-content) with structured sentence-ID selection. Model settings are configurable, and authentication is a header rather than a URL parameter.
5. The service accepts only 1–3 unique IDs from the exact supplied catalogue. Unsupported free-form prose, unknown IDs and extra response fields are rejected. Cached entries are revalidated. Gemini makes editorial choices; final language comes from verified deterministic statements. This is a deliberate safety boundary, with less stylistic freedom than open-ended LLM narration.
6. `src/analytics/pipeline.py` writes the new exports beside the old outputs. `report.py` extends the existing `seek()` and `draw()` behavior. Clicking commentary seeks the source video and pitch together; video scrubbing highlights the nearest entry for the chosen mode. Performance callouts appear for three seconds after their event and can be disabled. Embedded JSON escapes HTML-sensitive characters and commentary is rendered using `textContent`.

## Event and timing semantics

`events.json` is a versioned envelope containing `events`, `player_labels`, configuration, FPS and source offset. Event metrics are nested under `metrics`; unsupported values are omitted. Events sort by `(frame, event_id)`. To recover the half-video clock, add `source_offset_frames / fps` to recording-local timestamps. All coordinates are corner-origin meters on the existing 105×68 pitch.

Completed/intercepted transfers publish once the receiving ownership is confirmed, rather than when the ball is released. The existing detector backdates contact after persistence confirmation; canonical `frame` can therefore be later than `metrics.end_frame`. Start/end coordinates, duration and speed remain the original detector values. Unresolved transfers terminated by missing data publish at the missing frame. This avoids including later outcomes in earlier event windows. Offline centered smoothing still uses later coordinate samples, so this is not a causal live system.

A trajectory shot candidate publishes at the end of its completed speed window. BAS reference shots publish at their annotation frame without borrowing future speed-window values. Their speeds remain available in the unchanged `bas_shot_windows.csv` and existing report table.

`validation_state` distinguishes:

- `estimated`: heuristic or trajectory/motion event.
- `bas_reference`: the supplied BAS annotation itself.
- `bas_time_actor_match`: transfer release timing and sender agree with a BAS pass-family annotation. This does **not** independently verify completion, receiver, difficulty or ball speed.
- `bas_time_match`: a trajectory candidate agrees in time with a BAS shot, without actor verification. It remains a candidate; timing alone must not verify a shooter's identity.

BAS agreement is not independent accuracy: the reconstructed ball path uses BAS event anchors. Difficulty is retrospective geometry using the realized endpoint, on a 0–100 scale, not a calibrated probability. Progression uses the configured attack direction and is not a tactical claim such as a line-breaking pass or dangerous attack. An opposite-team transfer may be an interception, tackle or clearance. A BAS goal can be reported, but no cumulative match score is invented.

## Rolling context definitions

For a timestamp `t`, the last sample is `floor(t * fps)`; a complete window contains frame samples in `(t - window, t]`. Startup windows truncate at frame zero and report their actual `sampled_seconds`. Possession percentages use **all** available frame samples as the denominator; unknown ownership is reported separately, rather than allocating it to a team.

Context includes current ownership/confidence and confirmed-owner streak, completed/intercepted/other unsuccessful transfer counts, recent contiguous completed-transfer sequence, fastest players' accepted peak smoothed speed, observed ball displacement, summed completed-transfer progression, BAS actions, shot candidates and notable events. The streak can begin before the rolling window. Failed transfers, team changes or disconnected sender/receiver links reset the pass sequence. Counts cover the full window; recent transfer/action lists are capped at 12 and notable lists at 20. `ball_displacement_m` is endpoint displacement, not total path length; it is unavailable across missing segments. Progression summed across passes is not net ball displacement. Nothing asserts ball height or measured strike speed.

Query any timestamp with the Python API:

```python
from src.analytics.config import AnalyticsConfig
from src.analytics.data import load_recording
from src.analytics.pipeline import analyze_recording
from src.vision.data_loading import DatasetPaths
from src.intelligence.events import build_events
from src.intelligence.context import ContextBuilder

recording = load_recording(DatasetPaths('118575'), 'clip', 'gsr')
result = analyze_recording(recording, AnalyticsConfig())
events = build_events(recording, result)
context = ContextBuilder(recording, result, events).at(90.0, 30)
print(context['current_possession'], context['completed_passes'])
```

## Labels, thresholds and caching

`PlayerLabel.display` uses a verified name, then verified jersey, then `Player <tracker_id>`. Verified identity requires an explicit verification source. Current recordings supply no verified roster, so labels remain tracker IDs, including goalkeeper IDs. The GSR team/goalkeeper handling is preserved; different goalkeeper kits do not justify changing team identity. Original CSV actor IDs remain unchanged for compatibility.

Defaults are 25 km/h player speed, 80 km/h estimated ball speed, three consecutive above-threshold frames, 15 seconds between speed triggers, and 100-meter measured-distance milestones. Sustained speed produces one event; after dropping below the threshold it can rearm, subject to persistence and cooldown. Distances omit missing/rejected steps. Events feed both the visual callouts and commentary selection; distance milestones have significance 35, so lower the default significance threshold 40 if they should produce commentary.

The cache hashes the complete structured request, mode, prompt/schema versions and provider/model configuration using SHA-256. Only validated sentence IDs are persisted, not raw responses or credentials. Corrupt cache entries are regenerated within the call budget. Failed attempts count toward the budget and produce safe error metadata; they do not abort the deterministic report. The model receives structured facts and the catalogue, not video frames, coordinate streams, logs or environment variables. API retries are disabled to keep calls bounded.

To add another provider, implement `configuration` (public, non-secret model settings) and `generate(request) -> str` according to `CommentaryProvider`, returning JSON sentence selection. Pass that provider to `write_outputs(..., commentary_provider=provider)` or `generate_commentary()`. Keep model-specific transport outside analytics.

## Verify it

Run `python -m unittest discover -s tests -v`. The intelligence tests cover event JSON/ordering, windows and missing-data denominators, pass sequences, labels, sustained/crossing speed events, cooldown, distance milestones, grouping, requests, unsupported fields, mocked Gemini transport/errors, missing credentials, cache stability and reuse, disabled generation and budgets. The report test executes its generated JavaScript in a Node VM with DOM/video stubs to test seeking, scrubbing, highlighting and mode switching. It omits JavaScript execution if Node is unavailable; Python export checks still run.

For a visual check, generate the local template preview using the README command and open `report.html`. Switch modes, click a commentary entry, scrub the video, toggle callouts and labels, and check that old plots/tables still work. The four-minute clip has no BAS Shot annotations; use `--scope half --summary-only` to exercise BAS shot exports. Half scope has no synchronized source video in the existing report architecture.

For Gemini, set `GEMINI_API_KEY` in an ignored `.env` or your environment and run a small bounded request. Check `commentary.json` for `provider_calls`, `cache_hits`, `errors`, sentence IDs and event references. Repeat the same command/output directory: successful cached selections should cause zero new calls. A changed configuration or prompt invalidates the relevant cache. Real API availability and quality require a configured key and network/model access; unit tests make no hosted requests.

## Tactical facts

The deterministic Tactical State Engine now extends canonical events and rolling contexts by default. Its closed `TacticalState` schema contains observed shape, role/position provenance, pressure, ball-relative counts and estimated formation fit metadata. Commentary can select only deterministic statements built from those fields. Unsupported tactical intent, roles and causality remain forbidden. Use `--no-tactics` for the original deterministic-only analytics layer, and see [the tactical guide](tactics.md) for timing, configuration and quality rules. Historical `PASS_*` event enum values remain compatibility names for observable transfers; `measurement_definitions.json` records their interpretation.
