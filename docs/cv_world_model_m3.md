# TactiqAI CV Milestone 3 — camera diagnosis and conservative tracking

2026-10-24. **IMPLEMENTED / LOCALLY VERIFIED / EXPERIMENTAL.** Historical M3 camera baseline, preserved unchanged by [Milestone 4 player/team memory](cv_world_model_m4.md). The system is more inspectable and rejects additional unsupported motion; it does **not** provide reliable continuous full-clip localization. Lower coverage does not prove higher accuracy. At this snapshot completion, Milestone 4 had not begun.

## Architecture inspected and retained

Inspected the status/explorer, Milestone 2 report, world/observation contracts, semantic perception, TVCalib fitting, optical flow, world association, SoccerTrack piecewise coordinate transformer, serving index, React review and regression/browser tests. No AGENTS.md was found in the project/parent search. Existing source boundaries remain:

`original video → semantic evidence → absolute pose → measured camera motion → fixed 105×68 m pitch → anonymous world entities → canonical observations → existing gated analytics`

Before this change, three independently initialized TVCalib hypotheses used held-out line/curve pixel residuals, metric hypothesis agreement and visible-region validation. LK flow plus RANSAC partial affine motion propagated accepted geometry for at most 0.25 s, requiring fresh semantic line agreement. Camera source provenance already prevented propagated positions from updating reliable history or authorizing cross-track identity association. The main problems were coarse rejection diagnostics, affine motion without a separate held-out projective check, and insufficiently explicit camera transitions in the review.

## Diagnosis first

Before modifying camera algorithms/thresholds, generated a per-frame diagnosis from the immutable Milestone 2 cached frames/optimizer evidence. The baseline report accounted for **all 541 frames**. It is retained under `outputs/world_model/m3_verification/before/`; Milestone 2 source snapshots are preserved. The report was later enriched with class counts/percentages without changing the original camera decisions.

| Absolute camera outcome | Frames | All-frame percentage |
|---|---:|---:|
| Insufficient distinct semantic classes | 205 | 37.89% |
| Too few hypotheses passing held-out residual gates | 244 | 45.10% |
| Invalid projected pitch/visible region | 62 | 11.46% |
| Metric hypotheses disagree | 9 | 1.66% |
| Direct accepted | 21 | 3.88% |

There were **336 actual solver attempts**. For the other 205 frames, the semantic class/axis/curve eligibility gate prevented optimization. Raw retained classes and robust segment classes are recorded separately. The original optimizer did not retain all intermediate parameter trajectories; diagnosis does not invent optimization exceptions or classify the clip as out of distribution.

The baseline had **21 direct + 39 propagated = 60/541 frames (11.09%)**, 481 unknown frames, 15 availability acquisitions and 14 losses. The longest localized run was frames 69–79, 11 samples / 0.45879 s. The longest unavailable run was frames 138–458, **321 samples / 13.38838 s**. These durations use sample count / actual FPS; endpoints are also recorded. Ten initial tracking losses had insufficient affine inlier ratio; four reached propagation expiry, with overlapping additional checks on some frames. After loss, no active anchor remained. The absolute gates explain why it could not simply reacquire through the long central gap.

`frame_outcomes.jsonl` gives one inspectable row per frame, absolute diagnostics, solver attempt, class/constraint axes, tracking blockers and actual measured motion. `summary.json` includes count/percentage tables, availability runs and gap lengths. `timeline.html` provides a hoverable frame timeline. Absolute outcomes are mutually exclusive; overlapping tracking blockers must not be added as a disjoint partition.

## Changes

- Added a projective inter-frame estimator using image features, forward/backward LK consistency, RANSAC and a deterministic disjoint held-out feature sample. It measures motion rather than retaining an old transform.
- Retained every existing affine quality gate and the **0.25 s maximum age**. Added projective inlier ratio ≥0.8, inlier hull ≥15% of the small image, held-out median ≤1 px / P90 ≤2 px, finite nonfolding image geometry and local singular scales within 0.98–1.02 per frame. These are conservative experimental gates, not validated error probabilities.
- Predicted `H_current = H_previous × inverse(image_motion)`, converting the measured 320×180 image transform into original pixels. Fresh semantic ground lines still require mean ≤12 px / P90 ≤24 px and at least two classes; the projected fixed pitch must still intersect the image validly.
- Retained all absolute-fit residual, hypothesis-agreement, manifest/hash/schema and analytics gates. Added finite/visible-region revalidation at the camera-state boundary and explicit rejection of epoch changes even when a cut flag is omitted.
- Added `DIRECT`, `TRACKED`, `RELOCALIZED`, `UNKNOWN` status and acquisition/tracking/loss/relocalization/cut-reset transitions under `camera.motion_evidence.localization`. Tracked canonical positions remain `camera_propagated` / quality `propagated`; relocalized absolute fits remain directly calibrated.
- Search for absolute fits on every structurally eligible input frame. Accepted fresh fits replace previous camera state. Dependent predicted-versus-direct correction is exposed only when both estimates exist; it is never called independent drift accuracy.
- Recorded explicit optimizer exception/nonfinite-residual diagnostics for future fresh inference runs. These code paths are not claimed to have occurred in the real clip.
- Preserved per-entity temporary-track history and last reliable camera epoch. Existing appearance, kit, elapsed time, visibility, current source/quality and historical positions remain separate evidence. Association thresholds and the requirement for a direct calibrated movement check are unchanged.
- Added a React Camera State panel with actual source/status, outcome, absolute rejection, last absolute age, epoch, semantic class count, visible pitch bounds and measured flow quality. Added relocalization/tracking transition filters and readable frame-step buttons. Direct markers, outlined tracked markers, dashed tracked regions and optional historical ghosts remain distinct. Pending frame responses clear old coordinates; recording changes mount an isolated review.

Pan moves the camera region over one fixed canonical pitch. Measured small projective scale changes can pass the stated bounds; substantial or unsupported scale changes drop current geometry. A fresh valid absolute fit may reacquire after zoom. Hard cuts/epoch changes invalidate propagation; only new absolute evidence can establish geometry in the new epoch. Real cut/zoom detection accuracy has not been labeled or measured.

## Verified test_match results

Original video SHA256: `e092a252f47fd4f135506bd573ad36eaa25c95b0bbba3246c988827591c6dbac`. All **541 frames**, 1080×1080, 24000/1001 FPS, approximately 22.57 s; **5,108 cached M1 person observations**. No new person inference or source-video copy was needed.

| Measurement | Milestone 2 | Milestone 3 |
|---|---:|---:|
| Direct-fit frames | 21 | 21 |
| Tracked frames | 39 | 12 |
| Camera-region frames | 60 | 33 |
| Camera-region coverage | 11.09% | **6.10%** |
| Unlocalized frames | 481 | **508** |
| Availability acquisitions / losses | 15 / 14 | **19 / 18** |
| Longest localized run | 11 frames / 0.45879 s | **7 frames / 0.29196 s** |
| Longest unlocalized run | 321 frames / 13.38838 s | **321 frames / 13.38838 s** |
| Direct entity position samples | 145 | 145 |
| Tracked entity position samples | 246 | 79 |
| Localized unresolved-identity samples | 98 | 38 |
| Total localized person observations | 489 / 5,108 | **262 / 5,108 (5.13%)** |
| Accepted cross-track identity reassociations | 0 | 0 |

There are **20 direct-fit runs**, **one initial acquisition**, **19 relocalization transitions**, **five tracking-entry transitions**, **18 loss transitions**, and **zero detected cuts**. Availability acquisitions count every unknown→available boundary, including reacquisitions. Relocalizations also include a fresh absolute correction after tracked geometry, so these counts have different meanings. No detected cut is not proof that all actual cuts would be detected.

The new tracker removes 27 previously available frames and adds none. Eleven immediate losses account for these removals: ten projective quality failures and one unsupported local-scale change; the remaining 16 frames lack an active anchor after those losses. The projective gate checks measured image motion and fresh line consistency; it does not establish absolute metric accuracy. The accepted tracked-frame held-out P90 errors range **0.694–1.526 small-image pixels**. There are no valid paired predicted/direct correction measurements in this real replay, so quantitative direct-to-direct drift remains **unavailable**. Correction mechanics are tested synthetically.

Successful examples: frame 34 first direct acquisition; frame 55 tracked geometry after frame 54's absolute fit; frames 69–75 the longest localized run; frame 459 a later right-side absolute relocalization; frames 538–540 tracked geometry after frame 537. Failure examples: frame 35 projective inlier ratio 0.7838 fails 0.8 despite a small held-out residual; frames 138–458 never have defensible absolute geometry; frame 502 preserves the original player/steward counterexample as off-camera with null current position. These are inspection examples, not ground-truth accuracy labels.

All-frame typed world/canonical verification confirms: unavailable or off-camera positions are null; tracked positions do not replace last reliable coordinates/time; original source hash is unchanged; maximum accepted tracking age is **0.20854 s**; old false-identity example remains rejected. There are still 104 provisional entities, 1,524 ambiguous track-frames and 77 unconfirmed track decisions. These are not verified distinct-player or identity-accuracy counts.

## SoccerTrack reference-supported evaluation

Fresh semantic inference and three-start camera optimization ran on the existing real **118575_clip Azure-derived frames at 10, 60 and 120 s**. These are 4096×1080 panoramic frames retained from the approved short-clip workflow. No additional video/full-half download, Azure artifact modification or upload occurred.

**0/3 accepted**, zero projected reference points. All three fail the held-out residual gates. Reference coordinates were excluded from fitting. Thus median/P90/P95 reference disagreement and independent player-foot accuracy remain **unavailable**, not zero. Three sparse frames cannot measure temporal stability, loss/relocalization frequency or drift. The broadcast perspective model cannot be fairly assumed equivalent to SoccerTrack's nonlinear piecewise panorama calibration. Its transformer and existing analytics are unchanged. See `soccertrack_pose_evaluation.json` for actual class evidence and residuals.

## Performance and verification

Cached camera/world replay: **7.933 s / 541 frames (68.19 cached-replay FPS)**. This reuses M1 person tracks and M2 masks/pose hypotheses. It is not real-time end-to-end CV. Historical semantic inference remains 140.11 ms/input frame and pose optimization 318.29 ms/input frame; combining the historical 266.23 s inference/cache wall time with this replay gives approximately **1.97 FPS**, a derived comparison rather than a new full detector benchmark. No new GPU/production throughput claim.

- Full backend: **418 tests ran, 417 passed, one expected skip** for the deliberately absent optional local SoccerTrack source MP4. Azure playback was exercised separately.
- Full frontend: **51 passed** across eight files. TypeScript and production build passed.
- Fresh final real Chrome: **two tests passed in 36.8 s**; all-ten Azure test **32.8 s**, CV review **2.7 s**. Every Azure clip selected, loaded media, played/advanced, sought, loaded tactical views, navigated events with existing lead-in policy, served a 206 range and returned recording-scoped grounded Local Ask. No page/media errors. CV tested play, scrub, preceding-frame synchronization, direct/tracked/lost/relocalized state, region/markers, overlays, selection, frame stepping and transition lead-in.
- Live `/health` 200 (10 analytics recordings); `/ready` 200 (`{"status":"ready","recordings":11}`); `/recordings` 200 with ten Azure-ready recordings plus `test_match` CV-only capabilities. 118575 view 200; video bytes 0–15 returned 206 and 16 bytes. Grounded Local Ask 200/status ok, recording 118575_clip, 14 evidence entries. CV Ask remains unavailable (503), with no invented football analytics.

The ten verified recordings are: **117092_clip, 117093_clip, 118575_clip, 118576_clip, 118577_clip, 118578_clip, 128057_clip, 128058_clip, 132831_clip, 132877_clip**. No analytics thresholds, possession, GSR alignment, existing MATCHMIND variable names, grounding rules or Azure artifacts changed.

## Commands and evidence

Start the backend from the repository root; existing `.env` supplies the Azure account/container/media settings:

```sh
MATCHMIND_PROVIDER=local MATCHMIND_CACHE=/tmp/matchmind-azure-startup-cache \
TACTIQAI_BACKEND_PYTHON=/tmp/tactiq-cv-m1/bin/python scripts/start_backend.sh
```

The temporary Python environment must remain intact, or recreate it from the service/Azure requirements and set `TACTIQAI_BACKEND_PYTHON`. Offline camera experiments additionally need the optional CV requirements and approved local models. The API does not perform inference.

```sh
cd frontend
PATH=/Users/greatabhieyighan/.nvm/versions/node/v22.22.2/bin:$PATH npm run dev
```

Open **http://127.0.0.1:5173**, refresh and select `test_match · CV Experimental`. The original `data/raw/test_match.mp4` remains required for that local review. SoccerTrack playback streams Azure and needs none of the deleted local SoccerTrack MP4s. No Azure configuration change is required.

Reproduce a new replay (output directory must not already exist):

```sh
/tmp/tactiq-cv-m1/bin/python -X pycache_prefix=/tmp/tactiq-cv-pycache \
  -m src.vision.world.pose_replay --video data/raw/test_match.mp4 \
  --previous outputs/world_model/test_match_m1_conservative \
  --poses outputs/world_model/test_match_m2_pose --output /tmp/tactiq-camera-m3-replay

/tmp/tactiq-cv-m1/bin/python -X pycache_prefix=/tmp/tactiq-cv-pycache \
  -m scripts.evaluate_camera_tracking \
  --before outputs/world_model/test_match_m2_temporal \
  --after outputs/world_model/test_match_m3 \
  --poses outputs/world_model/test_match_m2_pose \
  --output outputs/world_model/m3_verification

/tmp/tactiq-cv-m1/bin/python -X pycache_prefix=/tmp/tactiq-cv-pycache -m unittest discover -s tests
cd frontend
PATH=/Users/greatabhieyighan/.nvm/versions/node/v22.22.2/bin:$PATH npm test
PATH=/Users/greatabhieyighan/.nvm/versions/node/v22.22.2/bin:$PATH npm run build
PATH=/Users/greatabhieyighan/.nvm/versions/node/v22.22.2/bin:$PATH \
TACTIQAI_AZURE_BROWSER=1 TACTIQAI_AZURE_REQUIRE_ANALYSIS=1 \
./node_modules/.bin/playwright test --config playwright.m3.config.ts
```

Current serving registry: `configs/serving/cv.json` → `outputs/world_model/test_match_m3/`. Generated outputs/models/videos stay ignored. Removed one redundant intermediate M3 replay, freeing 91,976,003 bytes; its diagnostics and all final/M1/M2 evidence are preserved (`cleanup.json`). Main evidence directory: `outputs/world_model/m3_verification/`:

- `before/` and `after/`: all-frame outcomes, quantitative summaries, hover timelines.
- `comparison.json`, `tracking_quality.json`, `semantic_integrity.json`, `evaluation.json`.
- `soccertrack_pose_evaluation.json`, `live_http.json`, `verification.json`.
- `camera-examples.jpg`, `cv-review-m3.png`, `azure-clips.png`, per-recording browser JSON and actual test/build logs.

## Files changed in this milestone

New: `src/vision/world/diagnosis.py`, `camera_motion.py`; `scripts/evaluate_camera_tracking.py`; `tests/test_cv_camera_m3.py`; `frontend/src/CameraDebug.tsx`, `CameraDebug.test.tsx`; `frontend/playwright.m3.config.ts`; this report.

Updated: `src/vision/world/localization.py`, `pose.py`, `pose_review.py`, `pose_replay.py`, `schema.py`, `engine.py`; `configs/serving/cv.json`; `frontend/src/CVReview.tsx`; `frontend/tests/cv-review.spec.ts`; `README.md`; `docs/current_pipeline_status.md`, `docs/current_pipeline_explorer.html`, `docs/azure_artifacts.md`; the historical banner in `docs/cv_world_model_m2.md`. Earlier uncommitted work is preserved.

## Remaining limits and Milestone 4 recommendation

Absolute geometry is still available on only 21 frames. The very short tracked sequences, unchanged 13.39 s gap and failed panorama references do not support a robust full-match calibration claim. Image features can include moving people, broadcast overlays or off-plane structures; held-out motion checks and semantic lines reduce some unsupported cases but are not independent ground truth. The stricter scale gate can reject legitimate zoom/perspective motion. Cut detection is heuristic and may miss cuts. No real labeled cut/zoom accuracy, camera error, player identity accuracy or full four-minute automatic camera benchmark is claimed.

Recommend **camera calibration validation and model suitability** for Milestone 4: independently labeled pitch points and contiguous pan/zoom/cut segments, physical broadcast/panorama model comparison, confidence calibration and spatial-motion support on the pitch. Establish accuracy/stability evidence before extending tracking age or strengthening reassociation. Ball, possession, passing, OCR, names, roster matching and learned ReID remain later work. Stop here.
