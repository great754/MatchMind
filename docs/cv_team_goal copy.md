# TactiqAI — team_goal CV evaluation

> Historical M5 baseline/source evaluation. Current temporal camera and team-belief authority: [Milestone 6](cv_world_model_m6.md). Original measured results below are preserved.

**LOCALLY VERIFIED execution / EXPERIMENTAL geometry.** Fresh run on 2026-10-25 using the existing M4 people/team and M5 persistent-camera pipeline. This is a separate source evaluation; it does not replace or change the original test_match results. No thresholds, soccer analytics, models or grounding were changed to improve these numbers.

## Measured output

| Metric | Result |
|---|---:|
| Source duration | 20.00 s |
| Frames / FPS | 500 / 25 |
| Resolution | 640×360 |
| Person observations | 5,515 |
| Mean visible observations per frame | 11.03 |
| Minimum / maximum per frame | 4 / 15 |
| Temporary track IDs | 56 |
| Anonymous entity hypotheses | 40 |
| Camera frames accepted by model | **475/500 — 95.0%** |
| Person observations projected by model | **5,116/5,515 — 92.77%** |
| No camera geometry | 278 observations |
| Outside calibration region | 121 observations |
| Current metric position unknown | 399 observations |
| Longest accepted camera run | 142 frames, 180–321 / 5.68 s |
| Longest unknown camera gap | 6 frames, 157–162 / 0.24 s |
| Detected hard cuts | 0 |
| Processing wall time | 473.180 s — about 7 min 53 s |

**These are model-accepted coverage figures, not 95%/92.77% accuracy.** A person observation is a candidate seen in one frame, not a unique physical player. Track fragmentation and entity hypotheses are not independently verified identities.

## Camera states and projection provenance

| Camera state | Frames |
|---|---:|
| DIRECT | 279 |
| RELOCALIZED | 41 |
| PARTIAL_CONSTRAINED | 155 |
| TEMPORAL_TRACKED | 0 |
| UNKNOWN | 25 |

Projected observations: **3,588 direct-calibrated**, including relocalized frames; **1,528 partial-constrained**. Partial coordinates carry propagated canonical provenance and are excluded from the existing direct-only analytics bridge. All 500 world and canonical frames were validated; all 5,515 detector observations remain accounted for. All nonvisible/current-unknown coordinates remain null. **11,635 nonmetric entity rows** plus unresolved player/team/image evidence and roster compare unchanged across the final camera replay.

Unknown-camera reasons: 2 NO_PRIOR, 5 VISIBLE_REGION_INVALID, 17 FIELD_MOTION_VALIDATION_FAILED, 1 INSUFFICIENT_FIELD_FEATURES. Every frame retains its own source, epoch, landmarks, direct age, current support, fit/flow diagnostics and projection reason.

## Team/person output

Visible observations: Team A **2,793**, Team B **2,320**, unknown **402**. These are anonymous clusters learned from this video; they are not asserted to be named clubs or defending sides. Three observed persistent-team switches. Names, verified player roles, exact physical player count and identity/team accuracy remain unverified. No roster of 22 is imposed. The 40 entities and 56 track IDs can include duplicate identities, fragmentation, officials and detection errors.

## What was actually run

Fresh inference on the original `data/raw/team_goal.mp4`, SHA-256 `726e8592c8cac7b12251957ad4a9dff07afbf1d526b0e4e18387f03237805a0d`, 1,753,330 bytes. No detections, team prototypes, masks or poses were borrowed from test_match.

1. Existing hash-checked YOLOX-S COCO person detector, confidence0.4/NMS0.5.
2. Source-specific temporary person seed, without duplicate annotated video generation.
3. Existing hash-checked TVCalib segmentation checkpoint, then 2,000-step absolute fitting with three perturbed starts. 485/500 frames met pose-attempt eligibility.
4. Existing conservative camera baseline and M4 team/kit/player memory.
5. Existing M5 measured field motion, bounded current semantic constraints, current foot projection and canonical provenance.
6. Full snapshot/source/hash/clock validation before application registration.

Stage wall times: person detection **35.302s**; source seed **1.034s**; fresh segmentation + absolute fitting **407.234s**; baseline camera replay **7.711s**; player/team replay **4.118s**; persistent M5 camera replay **17.780s**. The measured pipeline wall is **473.180s**; API/browser verification is additional. This is offline processing, not a real-time inference claim.

## Visual inspection and accuracy limits

Fourteen rendered samples were inspected: 0,2,3,4,50,100,150,200,250,300,350,400,450,499, plus a source contact sheet spanning the clip. Box/arc/touchline/midfield portions often align plausibly, but offsets and extrapolation errors remain. **Frame350 is a concrete failure concern:** the small center-circle arc visible at the right edge does not adequately constrain the whole reconstructed circle, which extends too far into the image despite passing the existing semantic residual gates. The high acceptance rate therefore overstates independently established geometric quality.

This run reports that limitation rather than retuning the model or presenting dependent residuals as truth. No independent metric camera/player reference exists for this video. The frontend continues to label positions experimental, exposes the source/canonical overlay and camera evidence, and leaves rejected frames/feet unknown. Use the source image to review accepted geometry. Model refinement is separate work; no claim is made that every accepted coordinate is correct.

[Rendered contact sheet](../outputs/world_model/team_goal/verification/contact_sheet.jpg) · [frame350 evidence](../outputs/world_model/team_goal/verification/frame_350.jpg) · [verified frontend](../outputs/world_model/team_goal/verification/team-goal-review.png).

## Frontend and live API verification

`configs/serving/cv.json` registers **team_goal → outputs/world_model/team_goal/final**, alongside unchanged test_match. Open **http://127.0.0.1:5173**, refresh, and choose **team_goal · CV Experimental**.

**Chrome: 1 test passed in 3.9s** (test2.8s), checking real playback and640×360 media metadata,500-frame timeline, seeking/frame synchronization, current state/reason, source canonical overlays, null markers on unknown frames, person/camera/kit inspection, frame stepping, byte range206 and switching team_goal↔test_match without state leakage. No page/media errors.

Fresh backend suite: **470 run / 469 pass / one expected optional local-video skip**. Frontend TypeScript/production build passed. Live health200/ten analytics recordings; **ready200/12 usable recordings**, consisting of ten Azure analytics recordings and two local CV recordings. Catalog exposes team_goal as available with CV/video capability, without analytics/Ask capability. team_goal,test_match and118575 video ranges returned206/16bytes. Existing118575 view200/481events and grounded Local Ask200/statusok/14 evidence entries remain verified. team_goal football Ask remains503; no CV goals/events are claimed merely from the filename.

## Storage, cloud and commands

Original team_goal.mp4 is unchanged and is required for local CV playback. **Keep it.** No source video or annotated MP4 was duplicated. No models or datasets were downloaded. No Azure data was uploaded/overwritten; existing Azure configuration, artifacts and MATCHMIND names remain unchanged. All generated data remains under `outputs/world_model/team_goal/`, ignored by Git, as are source videos/models.

Both servers are running. Backend command from repo root remains:

```sh
MATCHMIND_PROVIDER=local MATCHMIND_CACHE=/tmp/matchmind-azure-startup-cache \
TACTIQAI_BACKEND_PYTHON=/tmp/tactiq-cv-m1/bin/python scripts/start_backend.sh
```

Frontend: Node22, `cd frontend && npm run dev`. No Azure setting changes are required.

Reusable fresh-source processing command, using a new output directory:

```sh
/tmp/tactiq-cv-m1/bin/python -X pycache_prefix=/tmp/tactiq-cv-pycache \
  -m scripts.process_cv_video --video data/raw/team_goal.mp4 \
  --detector /tmp/tactiq-cv-models/yolox-s.onnx \
  --semantic-model /tmp/tactiq-cv-models/tvcalib-train59.pt \
  --semantic-sha256 ec551dc181692e1758c953e9bffde718664cebd19f39b674918f7a4588ca9066 \
  --output /tmp/tactiq-team-goal-reproduction
```

Fresh browser verification from `frontend/`:

```sh
./node_modules/.bin/playwright test --config playwright.team-goal.config.ts
```

Files added: `scripts/process_cv_video.py`, `frontend/playwright.team-goal.config.ts`, `frontend/tests/team-goal.spec.ts`, this report. Updated: CV registry, README, current status/explorer. The pipeline composes the existing model implementations; it does not modify them. Full summary and verification evidence: `outputs/world_model/team_goal/summary.json` and `verification/`.
