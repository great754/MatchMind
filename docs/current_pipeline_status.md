# TactiqAI Current Pipeline Status

Status date: 2026-10-17

Current repository revision inspected: `6e1c02f`. This worktree also adds the React/TypeScript/Vite Match Intelligence frontend, startup-loaded review projection and browser verification, alongside the existing Microsoft hardening milestone. Existing MATCHMIND_* variables and source/module names remain unchanged. Generated output.txt is removed from tracking while its local copy is preserved; .env, datasets and outputs remain ignored. No deployment was performed.

This document describes the current TactiqAI repository as implemented now. It covers the input data, synchronization, coordinate systems, tactical video renderer, deterministic analytics, new tactical-state engine, intelligence/commentary layer, projection-validity audit, generated outputs, verification numbers, and limitations.


## Local source video cleanup — verified 2026-10-17

Azure mode does not require local source MP4 downloads for playback, tactical review or grounded Ask. After confirming all ten Azure source videos existed with the same byte sizes and served valid MP4 ranges, the ten downloaded `data/soccertrack/mot/clips/*.mp4` files were deleted at the user's request. Azure copies and validated analytics bundles were preserved. Post-deletion verification passed: ten playable Azure videos, ten artifact-ready recordings, `/ready` ready with 10 recordings, and grounded Local Ask 200. Local ignored evidence: `outputs/local_video_cleanup_plan.json`, `outputs/local_video_cleanup_result.json`, and `outputs/local_video_cleanup_smoke.json`.

Source videos would be needed again only to rerun offline processing/real-video checks; download them again when doing that work. Generated rendered videos under outputs were retained. Videos, caches, outputs, credentials and new raw datasets are excluded from the GitHub publication; additional downloaded tracking/annotation paths are ignored. Existing historical tracked dataset files are unchanged.

## Frontend startup repaired — 2026-10-17

The long Vite startup was traced to offloaded dependency files in the cloud-backed Desktop project: `frontend/node_modules/vite/dist/node/cli.js` was marked `compressed,dataless`. The pasted logs showed repeated config/source change events, optimizer rescans and server restarts. The old dependency tree occupied only 31 MB of physical disk despite its installed files; reads required file-provider hydration. The backend remained healthy independently.

`frontend/scripts/runtime.mjs` now installs the unchanged package-lock dependencies into `~/Library/Caches/TactiqAI/frontend/<lock-platform-node-hash>/` on macOS (a user cache directory on other systems), then links `frontend/node_modules` to that local runtime. Installation is staged and published only on success; first preparation installed 111 packages in 2 seconds. The previous dependency directory is preserved as ignored `frontend/node_modules.cloud-backup-<timestamp>`. No source files, .env or credentials are copied. Ordinary `npm run dev`, `npm test`, `npm run build` and `npm run preview` use this helper; a changed lockfile/Node platform gets a fresh runtime automatically. The helper keeps child commands on the caller's Node version. `frontend/.nvmrc` pins the verified Node 22.22.2.

Vite's optimizer cache also lives outside cloud storage. Dependency scanning is limited to index.html with explicit app dependency entries; native file-provider change events are avoided with 500 ms polling while retaining HMR. Test configuration moved to `frontend/vitest.config.ts`, so starting Vite no longer loads Vitest configuration. The previous offloaded dependencies and new symlink are ignored by Git.

Verified results with Node 22.22.2:

- `npm run dev`: Vite ready in **1,099 ms**; frontend left running at **http://127.0.0.1:5173/**.
- `npm test -- --pool=threads --maxWorkers=1`: **35 passed**, five files, **8.28 seconds**; workers started in about 342 ms each.
- `npm run build`: TypeScript checks and production build passed; Vite transformed 2,024 modules and bundled in **776 ms**.
- Actual Chrome: API connected in **581 ms**, all ten recording options present, video readyState=4, playback advancing, no video/page errors, and one navigation during a 15-second reload-stability check. Evidence: `frontend/test-results/frontend-startup.json` and `frontend/test-results/frontend-startup.png`.

These results supersede the earlier frontend worker timeouts, killed type-check processes and stalled production builds. The Azure backend and all ten published recording bundles were not changed by this startup repair. Start normally with `cd frontend && npm run dev`; no Azure configuration change is needed.

## Ten clips fully Azure-ready — approved publication verified 2026-10-17

The user explicitly approved the eight prepared analytics bundles. All eight were published with create-only writes to `tactiqaistorage/tactiq-artifacts/matches/`: **176 canonical artifact files plus eight manifests, 167,490,828 bytes total**. No existing artifact blobs were overwritten; existing videos were not uploaded again. The earlier approval block is resolved.

Each newly published recording then passed the real `verify` workflow: a fresh Azure download, manifest/SHA256/schema validation, private-container check, no unexpected blobs, and real in-process application readiness/view/grounded Local Ask checks. Evidence: `outputs/azure_clip_publication_results.json` and `outputs/azure_clip_readback_results.json`; the exact approved manifests and hashes remain in `outputs/azure_clip_publication_plan.json` (ignored local evidence).

The backend was restarted on **127.0.0.1:8000** with the saved ten-recording configuration and existing private Azure media container `soccertrack-v2`. Live HTTP `/health`, `/ready`, and `/recordings` returned 200. `/ready` is `{"status":"ready","recordings":10}`; all ten catalog rows have status=available, artifact_ready=true and media_available=true. Every clip returned a tactical view/replay with 200, a valid MP4 byte range with 206, and a grounded Local Ask answer with 200/status=ok, match-specific conversation identity, source evidence and no fallback. Evidence: `outputs/azure_all_ten_analysis_verification.json`.

| Recording | Completed transfers | View / Ask / video range |
| --- | ---: | --- |
| 117092_clip | 28 | 200 / 200 / 206 |
| 117093_clip | 28 | 200 / 200 / 206 |
| 118575_clip | 29 | 200 / 200 / 206 |
| 118576_clip | 19 | 200 / 200 / 206 |
| 118577_clip | 18 | 200 / 200 / 206 |
| 118578_clip | 24 | 200 / 200 / 206 |
| 128057_clip | 16 | 200 / 200 / 206 |
| 128058_clip | 26 | 200 / 200 / 206 |
| 132831_clip | 16 | 200 / 200 / 206 |
| 132877_clip | 46 | 200 / 200 / 206 |

The verified Ask payload was `{"question":"How many passes were there altogether?","provider":"local"}`. Counts above are existing canonical completed observable transfers, not new detector runs or accuracy claims. No analytics, possession, GSR alignment or grounding semantics changed. No new external model call, role assignment or Container Apps deployment occurred.

All ten videos and analyses are now available; the former eight video-only entries no longer show analysis pending after refreshing discovery. The video-only state remains for future missing/unpublished analytics and does not grant query access to invalid bundles. No further configuration change is needed for this local setup: refresh the frontend tab. The known 50-test backend regression result and frontend unit/build limitations from the implementation stage remain recorded below.

Final post-publication Chrome integration passed in **1.7 minutes**, with `TACTIQAI_AZURE_REQUIRE_ANALYSIS=1`: all ten entries required status=available/artifact_ready=true; every clip selected, loaded its video, played past 0.2 seconds and sought to 20 seconds; all ten returned grounded Ask 200; no page errors. Saved browser results confirm artifact_ready=true, ask_status=200 and video error=null for every recording. Evidence: `frontend/test-results/azure-clips-results.json`; final full-analysis screenshot: `frontend/test-results/azure-clips.png` (visually inspected). This supersedes the earlier two-analysis/eight-video-only browser result.

Implementation-stage verification remains **50 focused backend tests passed**, including private media ranges, authorization, missing media, and video-only analytics gating. Frontend unit verification timed out starting workers; final type checking was killed with exit 137 and bundling stalled. No new successful unit suite or production build is claimed. The earlier ten-video Chrome pass is retained as historical evidence; final publication verification is recorded above.

## Earlier Azure startup-isolation smoke — 2026-10-17

Azure startup now validates configured recordings independently. Only canonical manifest/hash/schema-validated snapshots enter the serving tools. Missing/invalid recordings remain visible with unavailable/error status and fixed diagnostics; no local fallback or relaxed validation. Discovery distinguishes known/configured, processed, bundle availability and validated readiness. The frontend accepts and disables unavailable/error choices. LocalArtifactStore keeps strict startup behavior.

Real DefaultAzureCredential → AzureCliCredential startup against `tactiqaistorage/tactiq-artifacts` succeeded with `118575_clip`, `128058_clip`, and missing `118575_h1` configured together. Both clips are available/ready. The half is unavailable (`azure_bundle_missing`, artifact_available=false, artifact_ready=false; processing unknown), without aborting startup. Eight other locally registered clips remain unconfigured; their Azure availability is unknown.

Actual HTTP checks: `/health` 200 (2 recordings), `/ready` 200 (`ready`, 2 recordings), `/recordings` 200, both clip `/view` routes 200, missing-half `/view` 503. Local grounded Ask for 118575_clip returned 200/status=ok/fallback=false: 29 completed observable transfers, split 16/13, with transfers.csv source-field evidence. Final live recheck used a fresh `/tmp/matchmind-azure-startup-cache` after an observed Desktop-cache PNG read stall; all HTTP assertions passed again. Saved local evidence: `outputs/azure_startup_verification.json` (ignored). No uploads, analytics changes, external model call or cloud deployment occurred in this fix. This supersedes the earlier 403/read-unverified status. [Status contract, exact startup command and verification details](azure_artifacts.md).

Regression verification: the final focused backend suite passed **95 tests** (Azure workflow, recording setup, frontend review API, providers service and Microsoft hardening), including the safe access-error regression. Full discovery completed 331 tests: 330 passed, one existing real-video alignment test errored because OpenCV could not read `data/soccertrack/mot/clips/118575.mp4` (`moov atom not found`). No analytics/video changes were made to conceal this failure. The access-error regression was added after that run.

Frontend TypeScript checking passed during the elevated build retry. Frontend tests/build did not complete: the initial test run timed out starting workers, and the initial TypeScript launcher failed with ETIMEDOUT. A one-thread test retry and production bundling retry remained stalled after four minutes (the bundler had open lucide-react dependency files); they were stopped. No frontend test or production-build pass is claimed.

## Second independent recording — locally verified 2026-10-15

**128058_clip is now processed, registered and locally tested.** The installed CLI accepts a single pattern per `--include`; the original command accidentally skipped its first filename. `scripts.prepare_recording` now generates explicit positional filenames. The seven other dependencies were already installed; the remaining matching MOT file was downloaded successfully. Preflight now reports ready with no missing files.

A complete independent alignment/analytics/tactics/canonical-artifact run succeeded into outputs/analytics_128058_tactics. First-half alignment uses zero-based offset **5950**, mean image assignment disagreement **17.61 px**, 22 unique mapped actors and minimum 18/21 mapping vote agreement. These are inferred alignment diagnostics, not independent ground-truth accuracy. No 118575 calibration, synchronization, actors, ball or annotation data was reused, and no thresholds changed.

The new clip has 5,981 video frames at 25 FPS (239.24 seconds); audio/container length is 239.998 seconds. Outputs contain **26 completed observable transfers, 13 opposite-team receipt candidates, 3 unresolved transfers, 1 heuristic shot candidate and 0 BAS shot references**. Candidate shots remain heuristic/unvalidated. The serving registry preserves all old entries and adds 128058_clip. The recording selector can now select it after API restart/frontend reload.

Verification: **304 backend tests, 32 frontend tests and production build passed**. Actual artifact API smoke checked both clips, independent Ask totals (118575: 29; 128058: 26), recording discovery and MP4 range responses. A dedicated Local-only Chrome check passed actual 128058 selection, playback, grounded Ask and switching back to 118575, with no console/page errors. The seven preparation-era Chrome regressions also passed; the new real-recording test is opt-in. No new external model call or deployment occurred. [Correct CLI syntax, processing and exact browser command](second_recording.md).

## Grounded question paraphrases — verified 2026-10-15

Ask now performs bounded semantic retrieval before declining familiar football questions. `src/query/planning.py` maps wording to approved closed tools; raw analytics/detectors are unchanged. Completed/successful passes and ball transfers use the same canonical completed-transfer records as overall pass-share possession. Counts, both-team comparisons, totals, count differences and team ranks are computed deterministically in the tool, with source-field provenance. Totals and comparisons appear early in the answer preview. Ambiguous “passes” is explicitly interpreted as team-level confirmed completed observable transfers, not every ball movement.

All five requested regressions resolve to **16 vs 13 completed transfers; 29 altogether**, from the same canonical fact set: “How many passes did both teams make?”, “How many completed passes did each side have?”, “Who passed more?”, “Which team completed the most passes?”, and “How many passes were there altogether?”. Highest speed/fastest, pressed/under pressure, formation/shape (unless the question explicitly requests changing geometry), and possession/pass-share wording are supported. Explicit team filters, both-side pressure retrieval, per-team peak-player speeds and bounded top-N speed rankings are covered. Supported combinations retrieve multiple approved tools even in Local mode. Unsupported free-form wording can still use bounded provider planning; known deterministic evidence survives provider failure. If transfer CSV is missing, canonical completed-transfer events are checked before declining. Only absence of both sources is unknown; valid zero counts and ties are answered accurately.

Verified commands: `.venv/bin/python -m unittest discover -s tests -t .` — **297 passed**; `npm --prefix frontend test` — **28 passed**; `npm --prefix frontend run build` — passed. The Local-only Chrome suite passed **6 checks**, including all five typed pass questions against real clip artifacts. No new external Foundry live call was made; prior post-hardening live status is preserved, and new semantic/provider behavior is offline-tested. This does not claim universal natural-language coverage or unsupported intent/causal facts.

## Possession clarification — 2026-10-15

**Overall possession** uses each team's confirmed completed passes divided by both teams' confirmed completed passes (16/29 and 13/29 here: 55.2% / 44.8%). Unconfirmed control does not enter this percentage. `get_possession_summary` and general possession questions use this pass-based proxy; the UI labels its basis explicitly.

**Current possession**, shown below the pitch, is the team/player currently attributed the ball. It uses established control and continuity during flight/short gaps, not the overall percentage. Current/now questions use `get_current_possession`. Unknown is shown only when attribution is unavailable. This clarification supersedes the earlier description of attributed time as the headline possession statistic; attribution remains available as an internal diagnostic.

## Current demo improvements — verified 2026-10-15

This section supersedes older possession-display and frontend descriptions below. **284 backend tests, 27 frontend tests and production build pass**; five Local-only Chrome checks cover the core demo, missing evidence, fallback, responsive layout and saved identity/color/name-question flow. Screenshots were visually inspected; no separate human manual rehearsal or new external model call is claimed.

- Match setup stores validated, match-local team names/colors and player names/jerseys/affiliations in browser localStorage. Internal IDs remain unchanged. Neutral player labels replace long IDs outside setup; user labels resolve throughout the React interface.
- API/React evidence now uses natural English, including Analyst. Structured IDs/coordinates/enums stay in internal/API evidence and the explicitly technical legacy report.
- Current possession attributes the most recent established controller during flight, occlusion and ambiguity for at most 20 seconds since control. New established control switches it; timeout/out-of-play clears it. No original detector input/output is retuned. The clip attributes 5,032/6,000 frames (83.9%); team time shares are 53.2% and 30.6%, with 968 unknown frames. Completed transfers remain a separate 16/13 count and 55.2%/44.8% share.
- Microsoft planning now admits arbitrary supported evidence questions and compound multi-topic questions, with up to three closed readonly tools and validated English-catalogue selection. Empty/invalid selection cannot expose unrelated retrieved claims. Known single-intent grounding and conservative unsupported-intent guards remain. This extension is offline/mock-verified; the prior post-hardening Microsoft REST live verification remains valid historical verification, not a new multi-tool live test.
- Formation is a dominant estimate across fixed 240-second offline windows (50% clear-fit coverage, 60% known-fit dominance). Both actual clip teams remain unknown: only 24.1%/7.9% clear-fit coverage. No formation is forced; causal pre-event tactical snapshots remain unchanged.
- Low/moderate/high pressure uses cached geometry around the attributed possessor, separately from the unchanged strict confirmed-carrier sustained-pressure events. During ball flight it describes the previous possessor, not the flying ball. Available pressure levels increase from 291 to 743 of 1,200 snapshots; missing geometry remains unavailable.
- New sidecars: possession_attribution.json, formation_windows.json and pressure_levels.json. Existing bundles derive them at startup from cached canonical artifacts. The view API adds ownership, pass_share, formation_windows, pressure_levels and pressure_context. New approved tools retrieve players/distance leaders, sampled positions, current possession, pass share, slow formations and pressure levels. No request runs vision/detectors.
- Shots are unchanged: zero BAS shot references and zero heuristic candidates are acceptable. Foundry retains TLS enabled, automatic certifi, no required SSL_CERT_FILE and the supplied post-hardening microsoft_foundry/grounded_selection/fallback-false live result. Agent Framework, Fabric, Azure MCP and Container Apps are not claimed live.

See [exact implementation, thresholds, files, tests and limits](demo_improvements.md). Remaining limitations: user-entered local identities, a heuristic 20-second continuity timeout, unavailable formation evidence, offline/2D ball inputs, bounded free-form answers and unverified new multi-tool live behavior. Recommended next step is a configured Foundry judge rehearsal; hosted auth/ingress remains separate work.

## Executive Summary

TactiqAI combines an offline soccer-analysis engine and an artifact-only online serving system built around SoccerTrack match `118575`. It has five connected responsibilities:

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

Verified 2026-10-15: **304 discovered, 304 passed, 0 failed, 0 skipped**. The hardening milestone began with 216 passing tests and added 35 offline tests. The focused provider/service suite passed 92 tests. Actual local-only HTTP smoke also passed without external model calls. Shared fixtures live in `tests/helpers.py`; evaluation/query modules import correctly in package and discovery contexts. The earlier 117-test claim is stale and superseded by this measured run.

See [the current Microsoft hardening guide](microsoft_hardening.md) for complete metrics, diagnostic/experiment artifacts, held-out availability, API commands and actual versus planned Microsoft integration. The original baseline remains unchanged. Both halves of 118575 are now correctly labeled development; held-out metrics remain unavailable.

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

## Ask TactiqAI Query Layer

The current worktree adds a local grounded Q&A layer under [`src/query`](../src/query). Its deterministic core reads structured exports without raw tracking, SQL or network access. The optional provider orchestration around that core can call Microsoft or Gemini for bounded tool/statement selection; it does not perform unrestricted natural-language arithmetic.

Example:

```bash
.venv/bin/python -m src.query \
  --recording outputs/analytics_118575_tactics \
  --question "Who was the fastest player?" \
  --mode analyst
```

The report-side integration in [`src/analytics/report.py`](../src/analytics/report.py) adds:

- An Ask TactiqAI question panel.
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

The new Q&A tests cover closed dispatch, known answers, unknown questions, modes sharing facts, fact-field traceability, safe HTML text, Explain IDs, and timestamp seeking. These tests are included in the canonical 283-test suite; shared fixtures no longer import another test module.

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
- Ask TactiqAI questions.
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

TactiqAI currently provides a reproducible offline path from SoccerTrack inputs to synchronized tactical video, metric-space motion, conservative possession and transfer inference, shot candidates, tactical geometry, pressure and formation evidence, transfer networks, projection diagnostics, closed intelligence events, optional bounded Gemini commentary, and an interactive review report.

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

### Microsoft TLS, failure diagnostics and live routing status

Microsoft requests now pass verify=certifi.where() explicitly, preserving certificate/hostname verification without a manual SSL_CERT_FILE export or hard-coded developer path. Session injection, no redirects, bounded requests and existing timeouts are preserved. Certifi is already pinned in both dependency sets. Existing .env contents are untouched; a legacy SSL_CERT_FILE entry is not needed by this provider.

Sanitized fixed-code diagnostics distinguish configuration, TLS, network, timeout, HTTP 400/401/403/404/429/5xx, other HTTP statuses, malformed responses and invalid model plans/selections. JSON logs omit raw exceptions, bodies, prompts, tool arguments, URLs, headers and credentials. Agent logs distinguish tool-plan, argument, evidence-overlap and selection rejection. External API failures still retain the deterministic Answer and generic fallback status.

The remaining saved demo fallback is the formation query selecting get_match_summary. That tool does not support the formation answer. Offline replay using valid empty arguments against actual clip artifacts found 740 formation facts and zero matching summary facts, logged unrelated_tool_plan, and never called the selector. This is a tool-plan/grounding failure, not intentional unsupported behavior or a post-plan Microsoft transport failure. Saved arguments were unavailable, so the earliest historical argument-versus-overlap gate cannot be proven. Wrong model routing remains safely rejected; no model usage is forced.

Intentional player/event follow-ups remain local_followup with fallback false. Unsupported coaching questions remain local with fallback false. A metadata bug marking intentional local paths as fallback when the provider is absent was fixed and regression-tested. No statistics or evidence calculations changed.

```bash
# No network; does not prove authentication or model access.
.venv/bin/python -m scripts.verify_microsoft --without-ssl-cert-file
# Actual local-only HTTP checks; no hosted provider invocation.
MATCHMIND_ARTIFACT_STORE=local MATCHMIND_PROVIDER=local MATCHMIND_API_TOKEN= .venv/bin/python scripts/verify_service.py --local-only
# Run only after approval to send the bounded query/schemas and fastest-player facts to Microsoft:
.venv/bin/python -m scripts.verify_microsoft --live --without-ssl-cert-file
```

The optional verifier prints only configuration validity, CA policy, actual provider/fallback flags and sanitized diagnostic codes. It never prints secrets, endpoint values or fact payloads. Exit codes distinguish configuration/setup failure (2), unverified live response (1), and valid config or grounded live success according to mode (0). The user subsequently supplied a successful post-hardening live verifier result with SSL_CERT_FILE absent, certifi verification enabled, microsoft_foundry / grounded_selection, fallback false, fastest_players / get_fastest_players and empty diagnostics. The frontend task made no new external model calls. See [hardening details](microsoft_hardening.md).

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
| Docker service | VERIFIED IN PREVIOUS MILESTONE | Earlier image built/ran; hardening image not rebuilt or runtime-rechecked |
| Microsoft REST provider | POST-HARDENING LIVE-VERIFIED | Saved fastest/difficulty/pressure demo results show microsoft_foundry / fallback false / grounded_selection; user-supplied post-hardening verifier succeeds without SSL_CERT_FILE, using certifi |
| Agent Framework runtime adapter | IMPLEMENTED BUT MOCK-VERIFIED ONLY | Real SDK invocation code exists; fake-SDK unit test; SDK not installed and real agent did not run |
| Gemini query adapter | IMPLEMENTED BUT MOCK-VERIFIED ONLY for hosted calls this milestone | Fact parity/fallback tests; existing commentary remains supported |
| Local ArtifactStore and bundle packaging | IMPLEMENTED AND VERIFIED | Real bundles and corrupt/missing/traversal tests |
| Azure Blob adapter | IMPLEMENTED BUT MOCK-VERIFIED ONLY | SDK client injection, hash/cache/error tests; no upload/download against Azure |
| Container Apps Bicep/script | PREPARED BUT NOT DEPLOYED | Static checks only; no Azure deployment performed or verified in this task |
| Fabric CSV analytical export | IMPLEMENTED AND VERIFIED locally | Six typed/keyed tables with deterministic row counts and hashes |
| Fabric Delta import | PREPARED BUT NOT DEPLOYED | Notebook importer exists; Fabric received no data |
| Azure MCP | PREPARED boundary; real integration NOT IMPLEMENTED | Read-only operator policy; no MCP server ran |

Microsoft REST is POST-HARDENING LIVE-VERIFIED per the user-supplied verifier result: configuration_valid true, tls_verification true, certifi, SSL_CERT_FILE absent/not required, live_verified true, microsoft_foundry, fallback false, grounded_selection, fastest_players / get_fastest_players and diagnostics []. This supersedes the earlier blocked-check status. Frontend verification uses Local only; no new model call or deployment occurred. Agent Framework, Fabric, Azure MCP and Container Apps remain unverified as live integrations.

Historical Docker local measurement (prior milestone; not rebuilt here): image matchmind-service **94,649,434 bytes**, UID **10001**, port **8000/tcp**, readiness about **4.99 seconds**. Three artifact directories were mounted read-only. The optional local clip-media route was verified with an actual HTTP byte-range request; configure MATCHMIND_MEDIA_ROOT for API video playback. The image copies only online code and service dependencies; no raw dataset, outputs, caches, .env or vision packages. Azure Blob dependencies require WITH_AZURE=true; the default verified image is local. Cloud size/startup and load behavior are unmeasured.

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
.venv/bin/python scripts/verify_service.py --local-only
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

The previous product milestone added 40 tests covering store loading/corruption/path traversal/hash caching, deterministic Fabric schemas/identity/timing/counts, provider fact parity and outages, optional framework routing with a fake SDK, evidence propagation and rejected fabrication, recording-scoped follow-ups, readiness, body limits, sanitized logs and static deployment/container boundaries. Existing report tests execute offline and API browser-script paths in Node, including provider selection, contextual seeking and source-safe rendering.

Frozen development results are unchanged: transfers 706 TP / 161 FP / 388 FN, precision 81.430219%, recall 64.533821%, F1 72.004080%; shots 5 TP / 41 FP / 16 FN, precision 10.869565%, recall 23.809524%, F1 14.925373%. This product milestone reads preserved baseline artifacts and does not tune/recompute detectors. No second match is installed; cross-match generalization remains unknown. GSR remains primary; fallback projection disagreement remains 11.6831065 m mean, 10.7398110 m median, 24.9560484 m P95.

Remaining work: judge rehearsal of the frontend with the configured provider, optional SDK version resolution/pinning and live framework run; real Blob/Key Vault/Container Apps deployment, browser ingress authentication, media hosting and load/rate-limit verification; Fabric upload/import; authorized MCP connection. Scientific limits remain BAS-dependent 2D ball reconstruction, absent height/verified identities, heuristic possession/tactics/difficulty, full-half formation roster limitations and no independent causal evidence. See docs/microsoft_demo.md and deploy/azure/README.md for implementation choices and setup. No planned cloud capability is counted as live integration.

The current hardening adds 35 tests for explicit certifi verification (including absent/invalid SSL environment settings and injected sessions), categorized HTTP/TLS/network/timeout errors, malformed responses and secret-safe logs, deterministic fallback, replayed formation routing, intentional local paths and verification CLI behavior. output.txt is generated output, ignored and untracked with its local copy preserved; .env remains ignored/untracked. No Agent Framework live run, Container Apps deployment, Fabric cloud ingestion or MCP execution is newly claimed.

## Match Intelligence frontend — current implemented milestone

React/TypeScript/Vite now provides one clean analysis screen for `118575_clip`, using `apple-design.md` for system typography, restrained color, grouping, press feedback and accessibility. Video, team/ball pitch, possession owner, selected actor, timeline, tactical state and Ask jumps share one current timestamp. Samples use the preceding precomputed observation at 5 Hz without interpolation. Formation/pressure remain estimates, actor IDs remain identifiers, missing evidence stays unknown. Event selection includes source/ID and backend-categorized explanations. Ask supports the existing modes, providers, follow-ups, evidence, limitations, fallback state and jump-to-moment.

Architecture remains offline analytics → artifacts → FastAPI/closed tools → optional provider → frontend. New `GET /matches/{match_id}/view` returns a closed readonly presentation projection; event explain optionally accepts `with_evidence=true` while retaining its default contract. Existing Ask/player/media/report APIs are reused. Startup loads `review_frames.json` or safely parses precomputed JSON from older report bundles; no raw tracking/detection runs during requests. Future reports export the sample artifact and artifact stores allowlist it. The clip provides 1,200 samples, 22 actors, 481 canonical events and 1,200 tactical states.

Verification: **263/263 Python tests**, including 12 new review-contract tests; **20/20 frontend tests**; production build; **4/4 Chrome browser checks** against real local artifacts/media, with no console/page errors in the core demo. Browser tests cover missing media/state, simulated fallback, modes, unsupported requests and laptop/mobile layout. All browser model requests use Local. No secrets are sent to browser storage/bundles, and the dev proxy avoids broad CORS. Existing credentials/env names and deterministic analytics are preserved. No deployment/auth flow was added. See [frontend implementation, exact run/test commands and manual demo](frontend.md). Hosted authentication/media ingress and a Foundry-backed judge rehearsal remain next steps; scientific limitations above remain unchanged.

### Possession display update — 2026-10-15

Headline possession now uses confirmed completed-pass share, not frame-control time: team completed observable transfers / combined team completed observable transfers. The API reads canonical transfers.csv (works with older summaries); failed/intercepted/unresolved transfers and unconfirmed control are excluded. Zero completed transfers yields unavailable shares. Clip counts are 16 vs 13, displayed as 55.2% vs 44.8%. Instantaneous pitch ownership is unchanged. Verification: 265 Python tests, 21 frontend tests and production build passed for this update.

## One-command short-clip batch — 2026-10-15

Run `.venv/bin/python -m scripts.prepare_clips --run` from the repository root. It prepares 117092, 117093, 118575, 118576, 118577, 118578, 128057, 128058, 132831 and 132877 sequentially. Already registered, validated clip bundles with existing media are skipped. Missing dependencies are downloaded by exact filename with huggingface_hub; videos come exclusively from mot/clips, never full-half panorama video paths. Full-half GSR/ball annotation data is still required for synchronization. Each newly completed clip is registered only after processing/validation succeeds. Failures are recorded in outputs/clip_batch_status.json and can be retried by rerunning; known ready clips are preserved.

118575_h1 and 118575_h2 were removed from the default serving registry (their disk artifacts are retained). Restart the API and refresh React to remove them from the visible selector and discover newly registered clips. Pending clip configurations stay disabled. The batch has not been executed on all ten recordings during this change; no unprocessed clip is claimed working. Preview/config creation and existing-bundle checks were run without downloads. Verification: 308 backend tests and 32 frontend tests passed, including mocked exact-file download/processing/skip/failure tests. Existing analytics thresholds are unchanged.
