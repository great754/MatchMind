# CV Milestone 8 — metric camera motion and landmark relocalization

Verified 2026-11-01. **IMPLEMENTED / LOCALLY VERIFIED / EXPERIMENTAL.** All three existing CV recordings now use separate `*_m8_release` snapshots. M7 snapshots and original videos are retained. **NOT REFERENCE-EVALUATED:** independent real-video metric ground truth is unavailable. Coverage measures accepted geometry, not accuracy.

## Measured M7 → M8 results

|Source|Camera M7 → M8|Current player placement M7 → M8|New accepted frames|Cached replay runtime|
|---|---|---|---|---|
|test_match|182/541 (33.64%) → 269/541 (**49.72%**)|1646/5108 (32.22%) → 2179/5108 (**42.66%**)|87|90.547s|
|team_goal|455/500 (91.00%) → 456/500 (**91.20%**)|4856/5515 (88.05%) → 4861/5515 (**88.14%**)|1|45.863s|
|southampton|509/660 (77.12%) → 600/660 (**90.91%**)|6114/8725 (70.07%) → 7419/8725 (**85.03%**)|91|173.229s|

The test_match ≥50% engineering target was **NOT reached**:269/541=49.7227%;271 would be needed. No acceptance thresholds were relaxed to reach a target. Every previously accepted M7 transform is numerically unchanged; zero accepted frames lost. All detections, anonymous identities, histories, kit features and team beliefs/labels are frozen. Cached replay runtimes exclude original neural inference and are not full fresh-video processing times.

|Source|DIRECT|LANDMARK_RELOCALIZED|MOTION_PROPAGATED|MOTION_LANDMARK_FUSED|PARTIAL|UNKNOWN|Longest accepted|Longest Unknown|
|---|---|---|---|---|---|---|---|---|
|test_match|12|52|188|2|15|272|111frames/4.630s|156frames/6.506s|
|team_goal|69|0|362|4|21|44|160frames/6.400s|38frames/1.520s|
|southampton|13|84|477|11|15|60|295frames/9.843s|33frames/1.101s|

M8 state names expose the actual projection provenance; legacy camera timeline states remain compatible. New center-circle accepted certificates:52 test_match,0 team_goal,84 Southampton. They are per-frame current-geometry validations, not counts of independent camera shots or unique landmarks. No new real-video penalty-line relocalization was accepted; the penalty solver is implemented and synthetic-tested. Keyframe recovery records: test_match285/298/299;team_goal32;Southampton none. Same-epoch orientation lineage is explicit. All three videos contain zero detected cuts; cut reset is synthetic-tested.

## How image motion becomes metric viewing motion

Let H_t map image pixels into pitch metres and M_t map the same stationary field pixels from image t into image t+1. Then **H_(t+1)=H_t M_t^(-1)**. Robust field tracks, held-out residuals, spatial coverage, current named semantics, visible-region plausibility and independent-origin age validate this prediction. Players, officials, graphics, crowd and non-pitch pixels are excluded from static-field tracking. A direct remeasurement to a nearby accepted keyframe can recover a broken adjacent optical-flow chain, using the actual elapsed interval and the same residual/overlap gates.

For a fixed image reference ray q=(width/2,height/2), the framing shift is H_(t+1)(q)−H_t(q), evaluated after homogeneous division. Incremental shifts accumulate only inside a contiguous accepted epoch/segment. This is the **ground intersection of a viewing ray**, not a solved physical 3D camera centre. Local homography Jacobians provide perspective-dependent metric scale; there is no global pixels-to-metres constant. Relative projective derivatives separately expose pan/framing, zoom singular values and rotation. Static matched pixels also retain their before/current world coordinates and metric residuals.

Motion predicts; current landmarks correct. A center-circle fit uses the known radius9.15m/centre(52.5,34), halfway x=52.5 and other named geometry in a full projective optimization. An ellipse centre is not assumed to be the projected physical circle centre. Train/held-out semantic residuals retain the existing mean≤10/P90≤20 source-pixel gates. Conic arc support must be≥120° with robust fit support; named current constraints and safe projective horizon must pass. Multiple candidates, their transforms/scores and rejection reasons are recorded; passing starts must agree within1mP90. Center circle alone remains underconstrained. Symmetric midfield geometry retains a reflection ambiguity without same-epoch independent handedness or an asymmetric current landmark. Handedness does not import an old metric pose: an asymmetric synthetic landmark case solves fully cold; real midfield recovery uses the recorded epoch orientation.

The penalty-line solver uses canonical goal/penalty/six-yard line geometry, independent line families and safe residual/horizon/orientation gates. It does not invent spots or goal observations that were not detected. Existing M7 absolute-window estimates are preserved. Origin proofs resolve the actual independently anchored transform, rather than relabelling a temporally selected transform at that frame as independent.

For each current visible player, bottom-centre image footpoint p is projected through the **current accepted H** and accepted visible region. Numerical recomputation agrees within1e-6m on every saved placement (actual maxima in comparison JSON). Unplaced/off-camera players retain UNKNOWN projection provenance. Historical player memory never supplies a current coordinate. Player jumps are diagnostics and cannot optimize the camera to make players appear smooth.

## Original long UNKNOWN interval and frame216

All305frames151–455 were diagnosed before changes and exported after replay. Among all359 M7 Unknown test_matchframes:184 had circle pixels,69 halfway pixels,69 circle+halfway+top,220 accepted measured field-motion estimates,zero accepted world-field fits. These counts describe detections, not sufficient poses. The full cold midfield combination occurs at229–297. M7 rejected this span because no valid continuing world trajectory/adequately constrained old window survived; M8 can form new current metric hypotheses.

Frame216 contains circle and top-touchline support but **no detected halfway line**. It remains Unknown: a plausible-looking partial conic is not enough. The current field chain also cannot safely carry an old transform through the intervening rejected motion/world fits. Recovery reaches nearby frames with validated current evidence but does not bridge this gap blindly. Fresh keyframe measurements extend test_match through299;300–455 remains Unknown for156frames/6.5065s. Earlier sparse/adjacent-motion trials recovered fewer frames; bounded fresh measurements produced the final269 without changing gates.

Complete frame-level evidence (including raw source-pixel circle/halfway/other masks, conic, motion vectors, previous contiguous accepted H, measured-motion predicted H when available, candidate transforms/scores/reasons, current H, footprints, uncertainty and player feet/coordinates):

- `outputs/world_model/m8_verification/test_match/before/frames.jsonl`
- `outputs/world_model/m8_verification/test_match/release/center_sequence_evidence.jsonl` (all151–455)
- All three `release/timeline.jsonl`, `comparison.json`, `metric_statistics.json`
- All three `images/contact_sheet.jpg`, selected full comparison JPEGs; test229/250/269/295 and Southampton356/378/400/430/454/466 are useful recovered examples.

Manual source-overlay inspection of selected before/after images shows plausible line/circle alignment in the new midfield recovery; pixel offsets remain. Team_goal's prior supported geometry remains unchanged, and350 remains Unknown. Southampton new circle recovery aligns plausibly while323 remains Unknown. This is qualitative inspection, not a ground-truth accuracy score.

## Uncertainty, continuity and player-jump diagnostics

Uncertainty combines accumulated source-pixel motion budget and current semantic residual, mapped through local projective scale. It is an **uncalibrated proxy**, not a statistical confidence interval. Unknown/cut closes the segment; a newly independently validated landmark resets relative lineage. Current/previous/predicted footprints and reasons remain inspectable. Offline bidirectional estimates may reference a future independent anchor; signed legacy anchor age indicates direction and must not be interpreted as causal online latency.

|Source|Consecutive MOTION_PROPAGATED maximum|Uncertainty proxy median/P90/max (m)|State changes|Accepted→Unknown|New conic held-out P90 median (sourcepx)|
|---|---|---|---|---|---|
|test_match|50frames|1.425/1.765/1.990|69|9|14.413|
|team_goal|89frames|1.756/2.498/3.563|124|5|none|
|southampton|104frames|3.114/5.480/6.965|77|5|9.925|

New propagation gates retain bounded motion budget/independent age; M7 accepted solutions keep their original gates (therefore their combined telemetry budget is not necessarily≤20px). Matched pre-existing adjacent player-pair displacement distributions are **identical** before/after: no claim of smoothing or accuracy improvement on unchanged frames.

|Source|Matched P90 before=after (m)|All-pair P90 M7→M8 (m)|All jump-screen flags M7→M8|
|---|---|---|---|
|test_match|0.520|0.520→0.553|41→62|
|team_goal|0.744|0.744→0.744|257→258|
|southampton|0.534|0.534→0.590|182→284|

Newly observable pairs change the denominator and add jump flags. Southampton all-pair maximum4.617m and test5.267m still need review; lower Unknown coverage does not establish better metric stability. No independent real-video metric accuracy, physical camera XYZ, ball/shot/goal inference or persistent verified player ReID is implemented.

## Team evidence retained for the next milestone

No team-classifier algorithm changed. Preserved25 test_match,24 team_goal and33 Southampton representative crops (82total), actual selected foreground masks, per-pixel LAB/HSV/BGR NPZ, cluster centres, original decisions, beliefs/votes, temporal switches/events and source hashes in each `team_evidence/evidence.json`. The source rule is the existing saturation/value/background-contrast foreground filter. Southampton representative crops include dark jerseys, red jerseys and a bright official; its existing clustering can mix the two actual teams and separate the official. The visible frontend warning is preserved. Evidence is diagnostic, not independently labelled team accuracy. A major team redesign is **PLANNED**, outside M8.

## Frontend and verification

Metric Camera Debug shows current H/state/provenance, per-frame/accumulated viewing shift, local uncertainty, zoom/rotation, current/prior/predicted regions through the existing motion overlay, measured field tracks, observed conic, landmark candidates and rejection reasons. The tactical pitch optionally shows the recent contiguous viewing path and accepted footprint. Selected players show projection provenance. A detected conic remains visible when it does not authorize a pose. Existing decoded-frame playback/buffering remains unchanged.

- Full backend:559run,558passed,one expected optional-media skip (`/tmp/tactiq-m8-backend-tests.log`). Includes30new M8 geometry tests, M7/SoccerTrack/Azure/Ask regressions.
- Frontend:82unit tests across14files passed;TypeScript and production build passed.
- All three full replay/schema/hash/origin/footpoint/team-freeze evaluations passed.
- Live health200 (10analytic sources),ready200 (13usable sources),catalog200/all13available. Saved real responses:`outputs/world_model/m8_verification/live_http.json`.
- **All9Chrome suites passed (1.5m)** against the real Azure-backed backend after readiness. All10Azure clips:play,seek,metadata,byte-range206,tactical/event review and recording-scoped grounded Local Ask200 with nonempty evidence. All3CV sources:playback/seeking/stepping/person debug/recording isolation,video206,CV football Ask503. New M8 UI:all3metadata m8;circle recovery250/400;test216Unknown/no current player coordinates;zero page errors.
- Actual overlay rates in2.5s Chrome samples:test_match23.85/sec,team_goal24.66/sec,Southampton29.83/sec;zero missing overlays,max lag1frame. No claim of model interpolation.
- Saved proof:`outputs/world_model/m8_verification/browser/azure-clips-results.json`, `cv-smooth-results.json`, `southampton-results.json`, all3`*-m8.png`;test logs copied under`m8_verification/`.
- First browser attempt began before the API finished restoring its Azure cache and received502;after waiting for real readiness the full nine-suite run passed. No server validation/fallback was bypassed.

## Reproduction and startup

The temporary Python environment had been cleared, and the repo .venv contained cloud-offloaded files that stalled imports. Restored `/tmp/tactiq-cv-m1` with Python3.12 and standard service/scientific dependencies. No checkpoint/model download or training was performed for M8. Neural outputs were reused only from each video’s own hash-linked cache. The temporary runtime is not a permanent installation guarantee.

```sh
# Start backend from repository root; existing .env keeps Azure configuration.
AZURE_TOKEN_CREDENTIALS=AzureCliCredential MATCHMIND_PROVIDER=local \
 MATCHMIND_CACHE=/tmp/matchmind-azure-startup-cache \
 TACTIQAI_BACKEND_PYTHON=/tmp/tactiq-cv-m1/bin/python scripts/start_backend.sh
# Start frontend separately.
cd frontend
npm run dev -- --host 127.0.0.1
```

```sh
# Example full M8 cached replay (baseline is immutable M7).
/tmp/tactiq-cv-m1/bin/python -m src.vision.world.temporal_review \
 --video data/raw/test_match.mp4 \
 --baseline outputs/world_model/test_match_m7_current \
 --poses outputs/world_model/test_match_m2_pose \
 --output outputs/world_model/test_match_m8_release --camera-model m8
/tmp/tactiq-cv-m1/bin/python -m scripts.evaluate_metric_camera \
 --before outputs/world_model/test_match_m7_current \
 --after outputs/world_model/test_match_m8_release \
 --output outputs/world_model/m8_verification/test_match/release
/tmp/tactiq-cv-m1/bin/python -m scripts.export_metric_camera_evidence
/tmp/tactiq-cv-m1/bin/python -m scripts.summarize_metric_camera
/tmp/tactiq-cv-m1/bin/python -m unittest discover -s tests
cd frontend
npm test -- --run
npm run build
TACTIQAI_AZURE_BROWSER=1 TACTIQAI_AZURE_REQUIRE_ANALYSIS=1 \
 ./node_modules/.bin/playwright test --config playwright.m8.config.ts
```

For team_goal use baseline`outputs/world_model/team_goal_m7_current`,poses`outputs/world_model/team_goal/poses`;for Southampton baseline`outputs/world_model/southampton/current`,poses`outputs/world_model/southampton/poses`. Use their own source MP4 and separate `*_m8_release` output. `scripts/process_cv_video.py` fresh-video default remains M7; M8 is the explicit cached review stage above. Registry`configs/serving/cv.json` selects the three validated M8 releases. Do not delete local CV videos: they are still the playback sources. No Azure upload, provider, analytics, possession, grounding, GSR or environment-variable renaming occurred.

**Remaining work / PLANNED:** independent metric annotations, robustness on unresolved circle/reflection/texture gaps, additional real penalty-landmark evidence, calibrated uncertainty, and the separate team-classification milestone. Stop here; ball tracking and new ReID are NOT IMPLEMENTED.

## Implementation files and comparison detail

Core: `metric_motion.py`, `metric_landmarks.py`, `metric_camera.py` under `src/vision/world/`; existing `static_geometry.py`, `temporal_review.py`, world/observation schemas. UI: `MetricCameraDebug.tsx`, `CameraDebug.tsx`, `TemporalDebug.tsx`, `CVReview.tsx`. Registry: `configs/serving/cv.json`. New M8 test/evaluation/evidence scripts and `playwright.m8.config.ts` accompany the implementation. Mandatory status/explorer/report and frontend/Azure/README notes are updated. Prior uncommitted M7/Southampton/playback work remains intact. Generated snapshots, proof and original media remain local; nothing was pushed or uploaded.

|Source|Longest accepted M7→M8(frames)|Longest Unknown M7→M8(frames)|
|---|---|---|
|test_match|110→111|305→156|
|team_goal|140→160|38→38|
|southampton|295→295|69→33|

Original151–455 interval: 197frames had accepted M7 measured motion;M8 recovers68and retains237Unknown. Motion acceptance alone cannot independently anchor the field. Full rejection-count breakdown:`release/long_gap_summary.json`.

Direct final Azure check:`118575_clip/view`200;grounded Local Ask200,answer status`ok`,recording-scoped14evidence rows. Saved`outputs/world_model/m8_verification/azure_118575_live.json`.
