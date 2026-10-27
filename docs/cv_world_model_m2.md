# TactiqAI CV World Model — Milestone 2

2026-10-24. **IMPLEMENTED / LOCALLY VERIFIED / EXPERIMENTAL.** Playback and interactive review work. Automatic camera localization has limited coverage and is not a production-accuracy claim. Historical Milestone 2 snapshot; superseded by [Milestone 3 camera diagnosis and tracking](cv_world_model_m3.md).

While watching `test_match`, the website shows anonymous visible person candidates, temporary tracks, persistent provisional entities, semantic line/curve hypotheses and image-motion evidence. It infers a camera region and positions in **60/541 frames (11.1%)**: **21 direct camera fits, 39 bounded propagated frames**. The other **481 frames remain unlocalized**. Names, ball state and possession remain Unknown. This is a limited experimental result, not reliable full-clip calibration.

## Repository inspection and plan

Inspected the current status/explorer/M1 documentation; observations and world models; detector, short tracker and global association; existing SoccerTrack piecewise transformer; tactical pitch renderer; Azure media, artifact loading and all ten recording configurations; `/view` and allowlisted media routes; React recording discovery, playback policy and browser tests. M1 supplied the conservative world-state boundary and original person observations, but no automatic calibration or integration into React.

Plan followed: reproduce playback → restore the actual Azure media path → add inspectable pitch/camera evidence → preserve geometry rejection and canonical null semantics → provide validated bounded precomputed review endpoints → integrate capability-driven React review → evaluate the supplied video and SoccerTrack samples → run regressions/browser checks → document failures. The final path uses semantic line/curve camera optimization, held-out residuals, perturbed initializations and bounded motion propagation. Unusable frames remain unknown.

## Playback root cause and repair

Before changes, the running API advertised ten available Azure recordings and returned 200 for video HEAD, but a real GET range returned **503** with `Azure video download unavailable`. Chrome displayed “Match video unavailable” and removed the failed video element. A fresh DefaultAzureCredential client downloaded the same blob's bytes with the **same ETag**; a fresh backend on port 8001 returned 206. Thus the failure was in the old running backend's media-read path, not absent blobs, a wrong registry, deleted local MP4s, an ETag change or a frontend requirement to download the videos again.

The old implementation discarded its exception, so the exact internal old-process exception is **unavailable**. An expired token or a particular SDK fault is not asserted as fact. The old process/runtime was replaced with the intact isolated runtime. `scripts/start_backend.sh` now checks runtime dependencies before launch. `media.py` logs exception class/error code without SDK messages or signed URLs; React offers Retry video. Azure media clients now use **1 MiB initial GET and chunk sizes**, avoiding the SDK's default 32 MiB initial buffering on open-ended ranges. This does not alter range semantics, ETag conditions or artifact validation.

Live current verification: `/health` 200 (10 analytics recordings), `/ready` 200 (`recordings:11`, including one CV review), discovery lists ten Azure structured recordings plus experimental `test_match`; 118575 view 200; source bytes 0–15 return 206; grounded Local Ask returns 200/status ok with 14 evidence entries and recording-scoped conversation. Health's recording count remains the existing analytics count; readiness includes usable CV reviews.

### Fresh Chrome results

All results below were verified this turn against the real Azure-backed website, not copied from historical verification. Playback was muted in the automated browser to avoid autoplay restrictions; no source MP4 was saved locally. Native controls remain available to users.

| Recording | Selector/source/metadata | Play/time advances | Seek 20s/no media error | Tactical sync/event lead-in | HTTP range | Grounded Local Ask/isolation |
|---|---|---|---|---|---|---|
| 117092_clip | Pass | Pass | Pass | Pass | 206 | Pass |
| 117093_clip | Pass | Pass | Pass | Pass | 206 | Pass |
| 118575_clip | Pass | Pass | Pass | Pass | 206 | Pass |
| 118576_clip | Pass | Pass | Pass | Pass | 206 | Pass |
| 118577_clip | Pass | Pass | Pass | Pass | 206 | Pass |
| 118578_clip | Pass | Pass | Pass | Pass | 206 | Pass |
| 128057_clip | Pass | Pass | Pass | Pass | 206 | Pass |
| 128058_clip | Pass | Pass | Pass | Pass | 206 | Pass |
| 132831_clip | Pass | Pass | Pass | Pass | 206 | Pass |
| 132877_clip | Pass | Pass | Pass | Pass | 206 | Pass |

The all-ten test passed in **34.9 seconds**. It also checks 132877 insight navigation, Analyst provenance and Beginner mode. Timeline events are grouped by the existing frontend; verification clicks the visible grouped event row rather than using an ungrouped event index. Early test attempts stalled on overlapping marks and were corrected; they are not counted as passes. Machine evidence and screenshot: `outputs/world_model/m2_verification/azure-clips-results.json`, `azure-clips.png`.

## Approach selection and licenses

| Approach | Assessment | Result |
|---|---|---|
| Classical grass-adjacent white lines | Existing OpenCV, no new weights, fast CPU and inspectable. Cannot identify absolute semantic correspondences from anonymous segments. | Implemented baseline, all metric outputs withheld. |
| Learned pitch segmentation | Semantic evidence and curves; larger model/runtime; broadcast training differs from panoramic SoccerTrack and cropped footage. | Approved TVCalib checkpoint, local CPU inference implemented. |
| Keypoint model / PnLCalib | Explicit correspondences, but another network stack/checkpoint and integration; not benchmarked locally. | Evaluated from official implementation; not installed. |
| Full camera-pose/line/curve optimization | Can use semantic curves when intersections are insufficient; requires ambiguity/stability and held-out residual checks. | Implemented offline through selected MIT TVCalib geometry; held-out residual and local pose-stability gates, then bounded propagation. |
| Hybrid semantic + guarded geometry | Reuses existing types and canonical coordinates, exposes exactly why a fit fails. | Current review implementation. |

[TVCalib](https://github.com/MM4SPA/tvcalib) implementation is [MIT](https://github.com/MM4SPA/tvcalib/blob/main/LICENSE), copyright 2022 MM4SPA; notice retained in `src/vision/world/LICENSE.tvcalib.txt`. Official inference specifies DeepLabV3-ResNet101, 29 classes, RGB, short-side resize 256 and ImageNet normalization. Backbone weights are disabled to prevent extra downloads. `torch.load(weights_only=True)` allows only the checkpoint's `argparse.Namespace` metadata class; no general pickle-execution fallback is enabled.

Approved checkpoint: **488,774,369 bytes**, SHA256 `ec551dc181692e1758c953e9bffde718664cebd19f39b674918f7a4588ca9066`, `/tmp/tactiq-cv-models/tvcalib-train59.pt`, linked by the official TVCalib inference instructions. Separate checkpoint/training-data terms were not independently established; repository MIT licensing is not asserted to automatically relicense every training asset. Local experiment only, no checkpoint redistribution.

[OpenCV](https://opencv.org/license/) Apache-2.0, [PyTorch](https://github.com/pytorch/pytorch/blob/main/LICENSE) and [torchvision](https://github.com/pytorch/vision/blob/main/LICENSE) library notices apply. [PnLCalib](https://github.com/mguti97/PnLCalib) declares GPL-2.0; it was not copied or installed. These alternatives were evaluated architecturally, not compared through published benchmark numbers. Apple Silicon **CPU** was measured; MPS is an optional unverified path, no speed claim.

## Camera and geometry

`PitchEvidence` distinguishes partial/ambiguous/unavailable evidence. Semantic model outputs are hypotheses; unidentified lines do not acquire invented landmark IDs. Straight semantic masks are robustly fitted; circle and goal classes are excluded from planar straight-line fitting. Detected segments retain semantic kind and support. Grass coverage is null when not measured by that provider.

`TemporalCamera` tracks frame/time/epoch, optical-feature continuity, image affine motion, HSV histogram discontinuity, calibration status, optional homography, visible supported region, last reliable transform/time and evidence. Histogram distance above .65 plus fewer than 12 consistent features triggers a conservative hard-cut reset. These are provisional heuristics. A continuous pan is not interpreted as metric player motion. Unknown mode remains unknown; no reliable wide/medium/close-up classifier is claimed.

The intersection solver requires at least four independently constraining correspondences. Normalized DLT design rank, spatial support, RANSAC/inlier count, residuals, finite/non-singular transform and pitch bounds are checked. A pilot accepted five near-dependent sets; a determinant alone had been insufficient. Their evidence was preserved and a rank regression added **before those fits were served**. No player position was emitted even by that pilot. Intersection-only accepted transforms: zero. The final camera-pose path below is separate; rejected intersection fits are never promoted into accepted pose results.

The final pose solver uses semantic line and curve pixels with the fixed 105×68 m field, a main sideline-camera prior, centered principal point and no lens distortion. Three genuinely perturbed starts run 2,000 steps. At least two hypotheses must pass held-out mean residual ≤10 source pixels and P90 ≤20 pixels, and their image-grid metric projections must have P90 disagreement ≤4 m. These are provisional engineering gates, not calibrated accuracy/confidence. Pixel samples excluded from optimization supply the held-out residual; this is still dependent model validation, not independent ground truth. Selected upstream files are namespaced in `src/vision/world/tvcalib/`, pinned to commit `1222c5230af2742395d74918ed6f34eb2b9bf7f9`, with its MIT notice. Circle sampling has a local deterministic seed; the parameter container uses plain torch Module without training-framework features.

The pose path clips the projected fixed canonical pitch to the current image, then maps that polygon back to pitch coordinates. It does not expand the field. Footpoints are bottom centers and must lie inside the accepted visible image/pitch region. The original supplied-calibration/line-intersection provider retains its more restrictive landmark support hull.

Camera propagation is limited to **0.25 seconds** from a direct fit, requires continuous image motion, ≥50 feature matches, ≥80% affine inliers, median motion residual ≤1 pixel in the 320×180 motion image, and fresh agreement with ≥2 semantic ground-line classes (mean ≤12 source pixels, P90 ≤24). The transform is warped by measured camera motion; an unmodified old transform is never reused. Cuts, missing landmarks, poor motion or age expiry remove current calibration/positions. Propagation is labeled `camera_propagated`/`quality=propagated`, drawn outlined/dashed, never updates last reliable coordinates and never supplies the calibrated movement check needed for reassociation. It is also excluded from the analytics bridge. Last reliable transforms/coordinates are history only. Identity/localization/detector/team support remain separate; unavailable confidence stays null, never a fabricated probability.

## Serving and website

`CVReview` validates video/artifact hashes and all typed indexed frames on startup. It rejects corrupt/mismatched clocks and recording IDs. Missing/corrupt optional CV data does not crash other recordings. `GET /matches/{id}/cv` exposes metadata without source filesystem paths; `GET /matches/{id}/cv/frame?timestamp=T` returns one frame at `floor(T*fps)` bounded to the recording. Source changes require restart/revalidation. `video.mp4` uses existing FileResponse range delivery. Requests never run perception or analytics.

Discovery advertises source and capabilities for video, analytics, tactical view, CV world model, calibration and Ask. `test_match` is labeled CV Experimental and has no analytics/Ask. Structured recordings retain the existing Match Intelligence app and canonical bundle validation. A recording can also advertise both analytics and CV capabilities: matching recording clocks are checked, the selector offers a review-mode control, and a mismatched optional CV review cannot break validated analytics. Current live `test_match` remains CV-only; dual capability is covered synthetically, not claimed as a new published Azure CV recording. There is no silent Azure-to-local fallback; the CV source is explicitly local and separately registered in `configs/serving/cv.json`.

React provides actual video controls, scrubber, one-frame stepping, source boxes/IDs, pitch-line overlays, a canonical 105×68 pitch, conditional camera region/markers, optional historical ghosts labeled by last localization age, unknown/ambiguous hypotheses, person selection and supported evidence. Overlay changes affect presentation only. Unavailable metric positions display Unknown; directly fitted positions are solid, propagated positions outlined, ambiguous localized identities carry a question mark. Off-camera entities never receive current coordinates from history. Source evidence labels distinguish anonymous person candidates from verified football identities.

Fresh CV Chrome verification passed in **2.3 s**: play/advance, pause, seek 3/9/15/21 seconds, correct preceding samples, presentation-only overlay controls, actual person-box selection, Unknown values, frame stepping and zero page/media errors. Screenshot and JSON: `outputs/world_model/m2_verification/cv-review-m2.png`, `cv-review-results.json`. The final browser test also verifies an actual direct-fit region/current markers, dashed propagated region, calibration loss with no current markers, and camera-transition two-second lead-in.

Five live frame requests returned 28.7–107.8 kB in **3.3–7.4 ms** locally. This is a small measured sample, not a latency SLA. Browser cache holds at most 120 individual frames; no giant JSONL is loaded into the browser.

## Actual experiments and metrics

### External test clip

Original `data/raw/test_match.mp4`: SHA256 `e092a252f47fd4f135506bd573ad36eaa25c95b0bbba3246c988827591c6dbac`, 1080×1080, 541 decoded frames, 24000/1001 FPS, about 22.57 seconds. Source preserved, never duplicated/uploaded. Current final artifacts: `outputs/world_model/test_match_m2_temporal/`.

Person detections/temporary tracks are reused from the validated M1 observations for this exact source. This is an offline camera/world upgrade, not a fresh end-to-end player-detector benchmark. M1 measured detector time remains historical to that run, not newly timed here. The new semantic model is run on every frame.

| Metric | Result | Classification |
|---|---|---|
| Frames / person observations | 541 / 5,108 | MEASURED; person tracks reused from M1 |
| Pose-eligible / accepted direct frames | 336 / 21 | MEASURED / HEURISTIC acceptance |
| Propagated / unavailable frames | 39 / 481 | MEASURED |
| Camera-region availability | 60/541 = 11.1% | MEASURED |
| Calibration acquisitions / losses | 15 / 14 | MEASURED availability transitions |
| Direct-fit refresh runs | 20 | MEASURED, distinct from availability acquisition |
| Detected hard cuts | 0 | HEURISTIC count, not cut-detection accuracy |
| Direct entity / unresolved-identity positions | 145 / 31 | MEASURED samples |
| Propagated entity / unresolved-identity positions | 246 / 67 | MEASURED samples |
| Localized person-observation coverage | 489/5,108 = 9.57% | MEASURED, not accuracy |
| Provisional entities / visible entity-frame samples | 104 / 3,507 | MEASURED, not 104 unique football players |
| Off-camera entity-frame samples | 25,769 | MEASURED, not unique-player count |
| Global association decision rows | 1,705 (104 births, 1,524 ambiguous, 77 unconfirmed) | MEASURED |
| Accepted cross-track reassociations | 0 | MEASURED; no verified real return claimed |
| Independent identity/localization accuracy | unavailable | UNAVAILABLE |
| Semantic pitch inference | 140.11 ms/input frame | MEASURED CPU |
| Pose optimization | 318.29 ms/input frame (336 eligible frames) | MEASURED CPU |
| Initial camera / world update | 2.12 / 1.86 ms/input frame | MEASURED CPU |
| Inference/camera/world upgrade wall time | 266.23 s | MEASURED; reuses M1 person tracks |
| Cached temporal replay | 6.44 s | MEASURED; no neural rerun |
| Combined sequential upgrade rate | 541/(266.23+6.44) = 1.98 FPS | DERIVED from measured wall times; not end-to-end detector FPS |

The canonical field never expands as the camera pans. Reviewed accepted frames include **34 (1.418 s)**, **100 (4.171 s)** and **459 (19.144 s)**. The inferred region changes from approximately x=0–31.9 m at frame 34 to x=0–44.0 m at frame 137, then x=72.7–105 m at frame 459. These are heuristic inferred bounds, not independently verified precise coordinates. Frame 35 is bounded propagation; midfield frame 216 remains unknown. Calibration is intermittent; it does not continuously cover the whole pan.

Projected pitch overlays were manually inspected at frames 34/100/459/504/537; penalty-area alignment is plausible in the left-side examples, and remaining line disagreement is visible in later examples. These images are retained as experimental evidence, not accuracy labels. No separately labeled real zoom or hard-cut sequence was available. Synthetic tests cover zoom identity preservation with null location, cut reset and relocalization. A real cut-detection accuracy claim is unavailable.

The previously reviewed red-player-to-steward false return remains rejected at frame 502; metric position stays null and the original entity remains off-camera. No appearance-only reassociation safeguard was weakened. Unknown/ambiguous identity remains visible even when location can be inferred.

### SoccerTrack reference-supported experiment

Decoded only **three Azure frames** from 118575 at 10/60/120 seconds. No full MP4 was redownloaded. Existing supplied calibration/grid points were excluded from automatic inference. The anonymous baseline, semantic intersection method and final semantic line/curve pose solver all rejected these sparse panoramic samples. The final solver rejected them for insufficient agreeing low-residual camera hypotheses. Thus projected reference-point count is zero and median/P95 metric error is **null**, not zero. Temporal stability/loss frequency cannot be measured from three sparse frames.

A separately labeled dependent pixel-proximity diagnostic compares detected endpoints with supplied calibration-grid segments; some grid segments are not painted pitch lines, so that number is not detector accuracy. SoccerTrack's existing nonlinear piecewise calibration was not replaced by a global guessed homography. Machine evidence: `m2_verification/soccertrack_pose_evaluation.json` (final), with baseline diagnostics also retained.

## Tests, files and reproduction

Backend: **407 tests ran, 406 passed, one expected missing legacy video skip**. Frontend: **46 passed**. TypeScript/production build passed (258 ms Vite phase). New deterministic tests cover projection, null calibration, no stale position reuse, pan-region change in fixed coordinates, cut invalidation, relocalization, canonical visible-region metadata, rank-degenerate fit rejection, semantic geometry without neural inference, unsupported/curve rejection, indexed sampling, recording isolation, capabilities, readiness, hash corruption, source changes and video byte ranges. Existing M1 memory/identity/confidence tests and unchanged SoccerTrack analytics regressions pass.

Files added: `src/vision/world/pitch.py`, `semantic_pitch.py`, `review.py`, `LICENSE.tvcalib.txt`, `pose.py`, `pose_review.py`, `pose_replay.py`, `localization.py` and selected `tvcalib/` source/license; `src/serving/cv.py`; `configs/serving/cv.json`; `frontend/src/CVReview.tsx`, `CVReview.test.ts`; `frontend/tests/cv-review.spec.ts`, `frontend/playwright.m2.config.ts`; `tests/test_cv_pitch_review.py`; `scripts/start_backend.sh`, `scripts/evaluate_cv_pitch.py`, `scripts/evaluate_cv_pose.py`, `scripts/verify_cv_review.py`, `requirements-cv.txt`; this report. Files updated: observation/world schemas and canonical provider; serving app/media; React discovery/API/App/styles; Azure browser verification; current status/explorer and Azure documentation. Earlier milestone work remains intact in the worktree.

Start the verified backend from the repository root:

```sh
MATCHMIND_PROVIDER=local MATCHMIND_CACHE=/tmp/matchmind-azure-startup-cache \
TACTIQAI_BACKEND_PYTHON=/tmp/tactiq-cv-m1/bin/python scripts/start_backend.sh
```

The existing `.env` supplies Azure account/container/recording/media configuration; no secret values belong in frontend files. This command uses the currently verified temporary runtime; if it is later removed, create an intact runtime from the requirements and set `TACTIQAI_BACKEND_PYTHON` accordingly. Existing `MATCHMIND_*` names are preserved. Open **http://127.0.0.1:5173** and select `test_match · CV Experimental`. The frontend runs with `cd frontend && npm run dev` using Node 22.

Reproduce the offline pose/cache stage and temporal review in new output directories:

```sh
/tmp/tactiq-cv-m1/bin/python -X pycache_prefix=/tmp/tactiq-cv-pycache -m src.vision.world.pose_review \
  --video data/raw/test_match.mp4 --previous outputs/world_model/test_match_m1_conservative \
  --output outputs/world_model/NEW_POSE_RUN \
  --model /tmp/tactiq-cv-models/tvcalib-train59.pt \
  --model-sha256 ec551dc181692e1758c953e9bffde718664cebd19f39b674918f7a4588ca9066
/tmp/tactiq-cv-m1/bin/python -X pycache_prefix=/tmp/tactiq-cv-pycache -m src.vision.world.pose_replay \
  --video data/raw/test_match.mp4 --previous outputs/world_model/test_match_m1_conservative \
  --poses outputs/world_model/NEW_POSE_RUN --output outputs/world_model/NEW_TEMPORAL_REVIEW
```

The cache stage contains source-bound semantic masks and per-frame pose hypotheses. The replay stage runs camera/world/canonical conversion without detector or segmentation inference. Point `MATCHMIND_CV_REGISTRY` (default `configs/serving/cv.json`) to the intended local review and restart for hash/schema validation. Optional offline neural dependencies are in `requirements-cv.txt`; they are not service dependencies.

All generated outputs/checkpoints/videos remain outside Git. No Azure artifact was overwritten or CV artifact uploaded. Existing SoccerTrack source MP4s remain intentionally absent. No additional full-resolution annotated video was created.

## Limitations and next milestone

The website, all-ten playback, interactive CV review, real limited direct/propagated regions and positions, unknown semantics and camera tests are verified. Raw person detections can include officials/staff/crowd; anonymous entity births can duplicate people. Bounding-box bottom centers are only approximate ground contact, especially during occlusion/truncation. Camera-prior, semantic-label and extrapolation errors remain possible even after residual gates. Direct-fit labels mean current-frame inferred calibration, not ground-truth measurements; numeric confidence remains Unknown when unsupported. Coverage remains low: **88.9% of frames are unlocalized**. The perspective prior fails on the sampled SoccerTrack panorama. Metric accuracy, robust full-pan tracking, semantic-label accuracy, independent identity accuracy and verified real returning-player association remain unavailable. These limitations prevent a production-readiness claim. No player-name recognition, new football analytics, new LLM reasoning or cloud CV deployment was implemented.

Recommended Milestone 3: improve and independently evaluate semantic/camera coverage on labeled broadcast pan/zoom/cut sequences, then test camera-aware identity association on those calibrated observations. Keep names/OCR/rosters out of scope until localization and identity evidence are reliable. Milestone 3 has not begun.

## Requested final-report index

1. **Repository inspected:** existing React/serving/media/recording architecture, M1 world model, canonical contracts, SoccerTrack transformer and configurations.
2. **Plan:** reproduce playback, add conservative offline pitch/camera processing, integrate bounded review, test real browsers and preserve uncertainty.
3. **Playback cause:** old running backend failed real Azure GET ranges despite advertising media; its internal suppressed exception is unavailable.
4. **Fix:** intact backend runtime/preflight, safe diagnostics, 1 MiB Azure media buffering and Retry video.
5. **Ten-video table:** all ten pass the table above; current machine evidence is saved.
6. **Approaches:** classical, segmentation, keypoints, perspective pose and hybrid geometry evaluated.
7. **Selected:** semantic line/curve evidence + gated camera optimization and bounded propagation.
8. **Licenses:** official MIT TVCalib notice retained, OpenCV Apache-2.0, Torch/torchvision notices; separate weight/data terms not independently established.
9. **Files:** listed above, including optional inference requirements and the selected upstream geometry source.
10. **CameraState:** clock/epoch, continuity, optional transform/region, direct/propagated source, history and inspectable diagnostics; mode can remain Unknown.
11. **Landmarks:** typed semantic line segments and sampled curve evidence with heuristic support, never invented points.
12. **Calibration:** three perturbed starts, 2,000 steps, held-out residual and local metric-agreement gates.
13. **Visible region:** project the fixed pitch, clip to current image, map back; null on failed checks.
14. **Pan:** real inferred bounds move from the left to right side while field dimensions remain fixed.
15. **Zoom/loss:** propagation ≤0.25 s only with fresh evidence; otherwise current position/region null. Real zoom accuracy unmeasured.
16. **Cuts:** feature/histogram discontinuity resets epoch and invalidates propagation; synthetic reset verified, no labeled real-cut accuracy.
17. **World integration:** identity stays separate from position; propagated positions cannot authorize returns or enter direct analytics.
18. **Website:** capability-driven React CV review inside existing TactiqAI, with future dual review modes and checked recording clocks.
19. **Interaction:** actual video, play/pause, scrubbing, previous/next frame, timestamp sampling and transition lead-in.
20. **Selection:** clickable person boxes expose supported entity/track/team/position/evidence; unknown values remain Unknown.
21. **Controls:** boxes, temporary IDs, entity IDs, team labels, support, landmarks, calibration, region, historical ghosts and alternatives; presentation only.
22. **SoccerTrack:** three actual Azure frames; all pose candidates rejected. Reference metric median/P90/P95 are null, not zero.
23. **test_match:** actual 541-frame 22.57 s source processed; no four-minute CV benchmark claimed.
24. **Localization:** 489/5,108 person observations localized, including uncertain identities; 9.57% coverage, not accuracy.
25. **Calibration:** 21 direct + 39 propagated frames, 11.1% region coverage; 481 frames unlocalized.
26. **World:** 104 provisional candidates, 3,507 visible entity-frame samples, 25,769 off-camera samples; not unique player counts.
27. **Returns:** zero accepted cross-track reassociations; 1,524 ambiguous track-frames; no verified real return claimed.
28. **Performance:** 266.23 s pose/cache run + 6.44 s cached temporal replay; derived 1.98 FPS upgrade rate, not end-to-end detector FPS.
29. **Backend:** 407 ran, 406 passed, one expected missing legacy source-video skip.
30. **Frontend:** 46 passed.
31. **Build:** TypeScript and production build passed.
32. **CV browser:** passed, 2.3 s; direct region/markers, propagated region, loss, playback, scrub, selection, overlays and lead-in.
33. **Existing-video browser:** ten clips passed, 34.9 s; combined browser suite passed in 38.4 s.
34. **Successful example:** frame 100 (4.171 s), plausible left penalty-area alignment, visible inferred region and localized person candidates.
35. **Failure example:** midfield frame 216 and all three panoramic reference samples remain rejected/unlocalized; original rank-degenerate candidates are also preserved.
36. **Correct Unknown:** 481 unlocalized frames, off-camera current coordinates, unresolved identities and unsupported confidence; old player-to-steward return still rejected at frame 502.
37. **Limits:** sparse coverage, model priors/domain shift, approximate footpoints, semantic mistakes, no independent metric/identity accuracy, no CV ball/possession/names and no new external-model verification.
38. **Next milestone:** independently label and improve camera coverage on pan/zoom/cut sequences, then evaluate camera-aware identity association. No Milestone 3 work started.

`semantic_integrity.json` additionally validates all 541 typed world/canonical frames, original source hash, missing-state semantics, propagation provenance and unchanged last reliable positions. The maximum measured propagation age was **0.20854 s**, within the 0.25 s cap. The live Azure-mode API rejects football analytics/Ask for the CV-only recording with 503; capabilities explicitly advertise Ask unavailable.
