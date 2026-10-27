# TactiqAI CV Milestone 5 — persistent camera localization and player placement

> Historical M5 baseline/source evaluation. Current temporal camera and team-belief authority: [Milestone 6](cv_world_model_m6.md). Original measured results below are preserved.

Verified locally on 2026-10-25. **IMPLEMENTED / LOCALLY VERIFIED / EXPERIMENTAL.** The 30% engineering target was **not reached**. Placement improves **5.13% → 23.65%**; camera coverage improves **6.10% → 25.32%**. Independent metric accuracy on this source remains **UNAVAILABLE**. These percentages measure accepted pipeline output, not independently proved correct coordinates.

The original `data/raw/test_match.mp4` is unchanged: 541 frames, 24000/1001 FPS, 1080×1080, 5,108 visible person observations. Its SHA-256 is `e092a252f47fd4f135506bd573ad36eaa25c95b0bbba3246c988827591c6dbac`. M4 remains immutable in `outputs/world_model/test_match_m4_verified/`, frames SHA-256 `29b16164047b2bd63e37685ebe6ae17e0e97932d2865bc69df41e3a601b27ac9`. New output: `outputs/world_model/test_match_m5/`; registry: `configs/serving/cv.json`.

## 1. What was wrong with M4 camera localization?

The independent pose solver often rejects sparse or noisy semantics. M4 lost the active transform after weak short-term motion and required another accepted absolute fit to recover. Its general projective motion features mixed non-field evidence and could not exploit low-parameter pan conditioning or current partial constraints. It offered direct/tracked/unknown states, without a bounded current-landmark update or reobserved field-texture anchors.

## 2. Why was camera coverage only 6.10%?

Only 21 fresh absolute solutions and 12 propagated frames survived: 33/541. The long midfield span could not independently solve a pose, and previous measured continuity had already failed. In 490 of the 508 unknown frames there was no active tracking anchor. Simply recognizing a marking did not determine the eight-degree-of-freedom image-to-ground homography. M4's 6.10% is a camera denominator; player placement was **262/5,108 = 5.13%**.

## 3. Failure diagnosis before changing the algorithm

Every one of the 508 M4 unknown frames was diagnosed first. The mutually exclusive primary reasons sum to 508:

| Primary final failure | Frames |
|---|---:|
| Insufficient semantic classes | 205 |
| Pose residual rejection | 224 |
| Invalid visible region | 53 |
| Absolute hypotheses disagree | 8 |
| Temporal tracking failure | 10 |
| Insufficient motion inliers | 6 |
| Insufficient feature matches | 1 |
| Unsupported scale change | 1 |

Overlapping underlying absolute failures: insufficient semantics 205, residual rejection 239, invalid region 56, disagreement 8. Overlapping tracking blockers: no anchor 490; temporal failure 15; low inliers 7; high motion residual 1; low features 2; scale rejection 1. There were **zero eligible full-pose-evidence cases silently skipped**. Actual detected hard cuts: zero. Zoom/close-up truth and global pitch symmetry do not have independently labeled frame counts; they are not guessed from generic residual failures.

Raw before evidence: `m5_verification/before/frame_outcomes.jsonl`, `summary.json`, `gap_diagnosis.json`, `timeline.html`. All before/after localization losses: `m5_verification/dropouts.json`. Every M5 unknown frame is separately listed in `evaluation.json`, with the absolute, motion, anchor and current-state reasons.

## 4. Architecture changed

```text
fixed configured 105 × 68 m pitch
       ↑ independently validated absolute anchor
accepted camera memory + newly measured static-field image motion
       ↑ bounded robust update with current line/circle evidence
current validated camera-to-pitch homography and visible region
       ↑ inverse-free homogeneous projection of image bottom-center foot
current metric player position, or null with a reason
```

`PitchGeometry` provides corners, center, halfway line, center circle, penalty areas, six-yard boxes, penalty spots/arcs and named endpoints/intersections from the existing configured dimensions. No SoccerTrack geometry was changed. Canonical ground lines and conics become image residuals through the candidate homography. The optional source overlay backprojects the same geometry, rather than drawing a new pitch after a pan.

M5 replays the original source and cached M2 segmentation/validated pose attempts against frozen M4 person snapshots. It measures fresh field motion, filters camera hypotheses, and replaces only metric projection/history fields. It does not rerun or modify M4 detection, teams, player identity or association. The comparison verified **24,935 persistent-entity rows** unchanged outside the explicitly listed metric fields, all unresolved-track nonmetric fields, roster, ball, decisions and reconciliations. Previously omitted empty decision arrays are materialized as schema defaults. Last metric velocity is null: this experiment does not infer player velocity from these camera coordinates.

## 5. Camera states

| State | Meaning | Frames | Projected person observations |
|---|---|---:|---:|
| DIRECT | Current independently anchored geometry; acquisition or consecutive direct refresh | 2 | 14 |
| RELOCALIZED | Independently anchored current geometry restores direct support after uncertainty/propagation | 17 | 140 |
| PARTIAL_CONSTRAINED | Measured motion/remeasured field anchor + current semantic constraints + bounded filter | 114 | 1,019 |
| TEMPORAL_TRACKED | Fresh validated field motion, short semantic dropout | 4 | 35 |
| UNKNOWN | Insufficient accepted current evidence | 404 | 0 |

DIRECT + RELOCALIZED = **19 independent-anchor frames**, versus M4's 21. The added current canonical residual check rejects M4 fits at 129 and 131; they were not retained merely to increase coverage. Every accepted partial frame is marked propagated in the canonical observation contract and excluded by the existing direct-only analytics bridge. No analytics readiness is granted to this CV experiment.

## 6. Temporal camera memory

Typed `CameraMemory` holds epoch, last direct estimate, last accepted estimate, last strongly constrained timestamp, up to 12 accepted estimates, 12 failures, up to six candidate fits, pan direction, scale range and uncertainty interpretation. Estimates retain frame/time, transform, region, landmark identities, heuristic support and relative image-error budget. The current `CameraState` retains measured optical motion and semantic validation; `last_direct` never silently becomes the current estimate. Cuts clear the active prior and constraint clock; old direct evidence remains historical with its original epoch.

## 7. How current evidence validates propagation

Field features come from a grass-supported mask, excluding current image boxes and small background components. Persistent LK IDs require forward/backward consistency and at least three observations. Static membership is established on the preceding frame; the current held-out split is fixed by feature ID, avoiding current-fit selection of the validation points.

The smallest supported model is chosen: similarity (4 parameters), affine (6), or projective (8). All require inlier fraction ≥0.8, held-out median ≤1 and P90 ≤2 pixels at the ≤640px working width, finite non-folding transforms, plausible local scales and ≥0.8 field-mask overlap. Similarity needs ≥40 correspondences, including ≥30 training/10 held-out; higher-parameter fits need ≥50. Projective support must cover ≥15% image area. Low-parameter fits instead require the actual design to be well-conditioned (≤100), with held-out-error extrapolation diagnostic ≤2 pixels at viewport queries. This model-specific replacement of a projective area gate is justified by rank: a long narrow set can determine similarity while failing to determine projective perspective. It is not permission to accept high pixel error.

The current homography is predicted by composing the previous ground transform with inverse measured image motion. A robust bounded MAP update balances line/circle evidence and a small correction prior. Three perturbed similarity starts are tried first; a bounded eight-parameter correction handles shear/tilt when needed. Every candidate uses disjoint semantic training/held-out samples. At least two candidates must pass, with a finite visible region, correction P90 ≤12 source pixels and **each supported semantic category mean ≤10 / P90 ≤20 source pixels**. These gates were not loosened.

A newly accepted partial constraint refreshes the strong-support timestamp, so validated continuous motion can exceed 0.25 seconds since DIRECT. Maximum observed direct age on accepted frames: **1.8352 s**. Pure motion/weak-category dropout remains bounded to 0.25 seconds since strong semantic support. The four real TEMPORAL frames (36, 110, 139, 496) retain the rejected category's actual residuals; that category is not labeled validated and does not refresh the strong clock.

Up to eight accepted texture keyframes can support fresh current-to-anchor remeasurement of persistent field IDs. Unknown frames cannot add world anchors. At least 40 reobserved IDs, fixed held-out tests, field/motion gates and agreement between anchors are required. Eight actual partial frames used this recovery route: 38, 65, 67, 101–103, 142, 462. Texture anchors inherit the earlier camera uncertainty; they are not independent absolute landmarks or global visual SLAM.

## 8. How partial landmarks are used

Identified lines constrain their image orientation/offset against named ground lines. Identified circles constrain their projected conics; known circle centers and radii stay fixed in world coordinates. Penalty/goal-area lines constrain a specific end. Constraint count is reported as an **upper bound**, not a measured full numerical rank. A circle contributes up to five conic constraints and a line up to two. Missing global degrees of freedom are supplied only by a measured anchored prior, never an arbitrary pose seeded from recognition alone.

## 9. Center-circle-only evidence

Without an anchored measured prior: UNKNOWN. With a recent prior and validated motion, a single circle can validate a short TEMPORAL state; it cannot refresh strong support indefinitely or independently solve a camera. Recognizing the circle does not prove a player's position near (52.5,34).

## 10. Center circle + line; actual frame 216

Known-geometry regression tests verify circle + halfway line and circle + top sideline can constrain a compatible measured recent prior. These are **REFERENCE-EVALUATED synthetic behavior tests**, not real-video accuracy.

Actual frame **216 / 9.009 s remains UNKNOWN**. It has Circle central + Side line top, a current 44-feature validated pan (held-out P90 0.2522 working pixels), but **no supported current world anchor**. The prior chain was lost at frame 155 (only 11 eligible static features); frame 156 has 15, and no ≥40-feature anchor reobservation passes to recover it. The last accepted partial frame was 154, and the last direct anchor was 137, now 3.295 s old. Current image motion cannot reconnect that broken world chain by itself. Circle + sideline provide at most seven constraints for an eight-parameter homography, leaving at least one continuous degree of freedom plus possible discrete orientation/symmetry ambiguity. No unique full-pose solution is established. Five assigned visible entities and two unresolved tracks retain null current metric positions. No frame-specific production exception exists.

## 11. Penalty-area behavior

The real right-end sequence **459–513** is continuously localized: direct reacquisition at 459, measured partial constraints on the named penalty/goal-line geometry, direct refresh at 504, plus the short supported-category dropout at 496. Frame 470 and 490 overlays plausibly align with the visible box/arc/touchline; small residual offsets remain. Frame 514 correctly rejects field motion (inlier fraction 0.7917, held-out P90 5.222 working pixels). Known-geometry regression tests cover penalty-box + goal-line constraints separately.

## 12. Ambiguous hypotheses

Three perturbed local fits per correction family retain transforms, residuals, acceptance and reasons. At least two must pass; excessive P90 disagreement (>4 m on query points) yields AMBIGUOUS_GEOMETRY. Reobserved texture anchors must also agree within 4 m on current feature queries. Candidate scoring uses current semantics, bounded temporal difference, measured motion, scale, field overlap and visible-region gates. Tests inject distinct passing local candidates and verify rejection. These local hypotheses share a prior; this is not a global symmetry solver. Prior-free ambiguous geometry stays UNKNOWN.

## 13. Cuts and epochs

Existing cut detection and epoch assignments are preserved. A cut/epoch change clears active camera and texture continuity and requires new evidence. Historical transforms cannot cross epochs. There are **zero detected cuts on this clip**; deterministic cut/epoch regressions cover the behavior, not an independently labeled cut accuracy benchmark.

## 14. Zoom, tilt and shake

Local Jacobian singular values must remain within 0.98–1.02 per frame. Large or folding transforms are rejected; independently valid current geometry can reacquire after zoom. Similarity/affine/projective model selection and the bounded projective correction handle supported pan/rotation/minor scale/shear. For multi-frame texture remeasurement, nth-root scale is an interval-average diagnostic; it is not an instantaneous-zoom guarantee, and an explicit current zoom rejection blocks anchor recovery. Synthetic small shake, slight zoom and unsupported large zoom tests pass. The source's late tightening view was visually inspected; there is no independently labeled real zoom benchmark.

## 15. Uncertainty

Semantic residuals, feature holdouts, candidate spread, heuristic support and a relative image-motion error budget are exposed. The budget accumulates held-out motion residuals in quadrature in source pixels, capped at 20; observed median 0.702 / P90 2.499 / maximum 3.485 pixels. It is **not** a probabilistic covariance or absolute field-position error; it omits systematic absolute-anchor and correlated semantic errors. Player localization confidence remains null. The support score is explicitly heuristic.

## 16. Foot-point projection

For each visible box `(x,y,w,h)`, use `(x+w/2,y+h)` in source pixels. The current accepted homography maps that homogeneous image point to fixed pitch meters. Invalid/nonfinite/out-of-image/out-of-calibration/out-of-pitch points are rejected. Every visible observation carries a reason and typed camera evidence: source/state, epoch, support, last-direct age, current semantic support, relative image budget and actual image foot. DIRECT, temporal and partial provenance remain distinct. No player coordinates are used to fit the camera.

## 17. Current and historical positions

A nonvisible player always has null current pitch position. Last reliable direct pitch position/time/epoch remain separate history. Unresolved visible tracks can receive a supported current projection without being assigned a persistent identity. M4 teams, anonymous identity, track histories, roster, one-to-one assignment, image memory and current/historical kit behavior remain unchanged. The team improvement from M4 is preserved, not re-measured as a new independent accuracy result.

## 18–20. Camera and player coverage

| Measured output | M4 | M5 |
|---|---:|---:|
| Camera frames | 33/541 | 137/541 |
| Camera coverage | 6.10% | 25.32% |
| DIRECT + RELOCALIZED camera frames | 21 | 19 |
| Temporal camera frames | 12 | 4 |
| Partial camera frames | 0 | 114 |
| Unknown camera frames | 508 | 404 |
| Visible person observations (unchanged denominator) | 5,108 | 5,108 |
| Successfully projected observations | 262 | 1,208 |
| Player placement | 5.13% | 23.65% |
| No camera geometry | 4,801 | 3,609 |
| Outside calibrated region | 45 | 291 |

M5 has **1,499 person observations on camera-localized frames**; 291 fail the foot/region projection gate, leaving 1,208 valid pipeline projections. Unknown-position observations total 3,900. This counts the unchanged person detections, including unresolved identities, occasional duplicates and official-like actors; it is not ground-truth unique outfield-player recall. More camera coverage exposes more outside-region rejections. They are never counted as successful. Exclusive state/provenance counts are in section 5.

## 21. Longest localized sequence

M5: **55 frames, 459–513, 2.294 s** by frame-count/FPS convention. M4: seven frames, 69–75. M5 also maintains 45 frames at 42–86. No long continuous full-clip claim.

## 22. Longest unknown sequence

M4: 321 frames, 138–458, 13.388 s. M5: **304 frames, 155–458, 12.679 s**. Improvements at the edges do not solve midfield.

## 23. Every localization dropout

All 18 M4 loss-after-accepted transitions are recorded in `dropouts.json`, including prior successful evidence, failed gate, next accepted M4 frame, M5 status/reason and whether the loss frame is now fixed. **15/18 loss frames become supported**. Remaining loss frames: 94 (insufficient field features), 130 and 133 (no recent measured prior).

The three original isolated one-frame unknown gaps were 76, 78, 130. M5 fixes 76 and 78 using measured field motion + current box/arc/line constraints. Frame 130 remains unknown within a longer rejected interval; it is not a successful fix. M5 has one isolated unknown frame, 94, which was already unknown in M4; recovery at 95 shortens its old gap. Representative losses at 35, 38, 101, 138 and 460 are fixed by current motion/constraints or fresh keyframe remeasurement. Frames 146–148 retain high semantic residual rejection despite acceptable field flow. All runs and all 404 unknown-frame reasons are machine-readable in `evaluation.json`.

## 24. Reprojection, continuity and visual diagnostics

Accepted supported categories retain the same mean≤10/P90≤20 source-pixel gates. Across frames, line-category mean residual median **8.543 px**; line P90 residual median **16.079 px**, maximum **19.564 px**. Circle-category mean residual median **8.430 px**; circle P90 residual median **14.756 px**, maximum **19.899 px**. Rejected categories on weak TEMPORAL frames are stored separately and excluded from these supported-category statistics.

Consecutive visible-region IoU: median **0.9830**, P90 **0.9946** (126 pairs), computed by float64 polygon clipping. Consecutive same-entity foot displacements: 779 pairs, median **0.162 m**, P90 **0.507 m**, max **4.840 m**. A screening threshold of `12m/s × frame interval + 0.5m` flags **24 large changes**, not 24 independently established teleports. Decomposition is retained: the largest at 470 has 76.40px image-foot change, 4.776m changed-foot component and 0.073m same-foot camera component. A direct-refresh step and detector/occlusion jitter remain visible. No smoothing or foot adjustment fabricates a player position to hide these cases.

Manual before/after inspection: 18 paired frames in `m5_verification/before_after_contact.jpg` and individual JPEGs. Inspected box/arc/touchline pan frames 34/35/55/76/78/100/137/138/143/150; unresolved gap 130; center-circle 216; halfway-line/long-unknown 300; right penalty end 459/470/490/504; late goal/edge view 537. Canonical cyan lines plausibly align at accepted samples, with nonzero offsets; no geometry is drawn on unknown samples. These are post-hoc plausibility checks, not independent labels. See `manual_inspection.json` for recorded observations. The website adds detected semantics plus an optional cyan dashed canonical overlay, distinct direct/temporal/partial markers, historical ghosts, camera evidence, and a 541-frame hover/click state timeline. Timeline seeking lands 0.001 frame inside a selected frame to avoid Chrome microsecond truncation, while existing football event lead-in stays unchanged.

## 25. Independent/reference-supported accuracy

**REFERENCE-EVALUATED:** deterministic known-ground-geometry tests of pan, circle/line constraints, penalty lines, foot projection and ambiguity gates. **LOCALLY VERIFIED diagnostics:** full real replay, category residuals, visible-region continuity, displacement screening and manual images. **UNAVAILABLE:** independent real-video metric camera/foot accuracy, calibrated confidence, unique-person/official accuracy and long-return identity. Prior SoccerTrack reference calibration limitations remain in the M2 report; unchanged SoccerTrack provider/analytics regressions pass. Reprojection consistency is never called absolute accuracy.

## 26. Runtime/storage

Final replay: **11.774 s** total, including original decoding, output, projection, fresh field-motion **3.977 s**, camera/anchor filtering **4.014 s**. It reuses existing semantic masks and absolute fits; detector/segmentation/absolute-solver inference is excluded. This is not end-to-end real-time performance. No models, libraries or datasets downloaded for M5; no extra MP4 or Azure artifact written. Removed **177,742,615 bytes** of redundant M5 payloads after retaining their diagnostics and confirming the repeated replay was semantically identical; final M4 and M1–M3 evidence, the original source and final M5 remain.

## 27. Tests

**470 backend tests ran: 469 passed, one expected optional deleted-local-video skip. 59 frontend tests passed across 10 files. TypeScript/production build passed.** New backend camera suite: 37 cases. Full backend log: `m5_verification/backend-tests.log`; frontend and build logs are alongside it.

37 new backend cases cover independent fits, continuous measured pan, short and repeated dropout, >0.25s current constraints, circle+sideline, circle+halfway, prior-free ambiguity, penalty/goal lines, touchline weakness, actual candidate-disagreement rejection, semantic rejection, cuts/epochs, direct refresh vs relocalization, zoom reacquisition, reflections/invalid transforms, relative budget expiry, bounded history, no stale geometry, robust correction, direct-only analytics bridge, invalid/outside feet, current/historical separation, configured geometry, polygon IoU, small shake/zoom, model conditioning, low features, flow-failure measurement clearing and reobserved-anchor epoch isolation. Existing M4 one-to-one/team tests, SoccerTrack, analytics, Azure hash/manifest/schema validation, grounding and playback regression coverage remains in the complete suite.

## 28. Browser/live verification

**Four final full Chrome suites passed (reported 1.3 min)**: all ten live Azure clips (reported 1.2 min), camera placement (2.7 s), existing CV review (3.0 s), player/team review (1.6 s). After a final readability-only person-provenance panel change, **all three affected CV suites passed again in 7.6 s** (2.2 / 2.8 / 1.6 s); all-ten Azure results are preserved from the full run. No page or media errors. Tests exercised selection, real playback, seeking, tactical views, event lead-in, grounded recording-scoped Local Ask, timeline frame selection (including its boundary-precision fix), partial marker/region styling, projected canonical overlay, null geometry on unknown frames, foot provenance and M4 team/kit/history controls. Logs, per-recording JSON and screenshots are in `m5_verification/`.

Live HTTP evidence (`live_http.json`): `/health` 200, ten analytics recordings; `/ready` **200 / 11 usable recordings**; `/recordings` 200, all ten Azure rows `available` and `artifact_ready=true`, plus local CV-only `test_match` (`available`, `artifact_ready=false`). `118575_clip/view` 200, 481 events; Azure video byte range 206 / 16 bytes; grounded **Local Ask 200 / status ok / 118575_clip / 14 evidence entries**. CV football Ask remains rejected with 503. These checks do not claim a Microsoft Foundry provider request was executed. The backend revalidated existing Azure artifacts at startup; nothing was uploaded or changed in Azure.

## 29. Failures still present

404 unknown camera frames: **34 NO_PRIOR; 338 NO_RECENT_MEASURED_PRIOR; 17 FIELD_MOTION_VALIDATION_FAILED; 6 INSUFFICIENT_FIELD_FEATURES; 7 LANDMARK_RESIDUAL_TOO_HIGH; 2 VISIBLE_REGION_INVALID**. The full absolute attempt and texture-anchor diagnostics remain attached. Sparse/motion-blurred semantics, loss of stable field texture during pan, underdetermined midfield geometry and limited absolute solutions remain. Marker feet jitter under occlusion. Learned full camera parameters, global symmetry resolution, independently validated metric uncertainty and robust long-gap field-map localization are not implemented.

## 30. Was ≥30% reached honestly?

**No.** Maximum accepted output from the tested implementation: **23.65% player placement / 25.32% camera coverage**. This is the maximum supported result found in these experiments, not a claim of a mathematical optimum. No denominator changed; no stale coordinates, relaxed semantic residual gates, invented poses or corrupted artifacts were counted.

## 31. Exactly what prevents 30%?

At least 1,533 of the 5,108 observations would be required, versus 1,208: **325 additional supported projections missing**. A 30% camera target would require 163 frames, versus 137. The 304-frame world-anchor gap is the largest obstacle. At 155 mature field correspondences collapse; later optical flow recovers local image motion but cannot restore the lost world reference. Current circle+sideline is underdetermined without that anchored prior, absolute fitting fails/does not have enough geometry, and no reobserved accepted texture anchor reconnects the chain. Another 291 feet fail current visible/calibrated-region checks. Counting those or raising residual/age bounds would produce a misleading percentage.

## 32. Recommended Milestone 6

Continue **camera** validation first: independently annotate selected landmarks/camera/foot correspondences; improve midfield semantic correspondences and supported persistent field-map relocalization; quantify absolute/systematic uncertainty and foot jitter; require measured geometry improvement before increasing placement coverage. Do not begin ReID or ball tracking based solely on this result. Neither was started in M5.

## At an arbitrary frame, what evidence says where the camera looks?

Inspect the current camera state and epoch. A DIRECT/RELOCALIZED transform is anchored by accepted independent current geometry. A PARTIAL transform inherits a validated anchor through freshly measured static-field motion (or newly reobserved field IDs), then passes bounded current line/circle constraints, candidate agreement, scale/non-folding and visible-region checks. TEMPORAL uses fresh validated motion over a short semantic lapse, with its age and rejected evidence visible. This transforms the current image footprint into a polygon on the same fixed 105×68m pitch. A visible person's actual bottom-center image foot is then mapped by that accepted current homography only if it lies in the valid region. The resulting metric coordinate retains the full camera provenance. If that evidence chain is broken, camera and current pitch position are Unknown; remembered positions stay historical. This answers where the model hypothesizes the camera is looking, with explicit support limits, rather than asserting independently measured truth.

## Files and reproducible commands

New M5 implementation: `src/vision/world/pitch_geometry.py`, `field_motion.py`, `persistent_camera.py`, `texture_anchors.py`, `camera_review.py`; `scripts/evaluate_camera_placement.py`; `tests/test_cv_camera_m5.py`; `frontend/src/CameraTimeline.tsx`, `CameraTimeline.test.tsx`; `frontend/tests/camera-placement.spec.ts`, `frontend/playwright.m5.config.ts`.

Updated for M5: `src/vision/world/schema.py`, `provider.py`; `src/observations/schema.py`; `frontend/src/CVReview.tsx`, `CVReview.test.ts`, `CameraDebug.tsx`, `CameraDebug.test.tsx`; `configs/serving/cv.json`; README; current status/explorer; this report and the historical M4 banner; Azure artifact and frontend documentation. Other pre-existing uncommitted work is preserved; it is not attributed to M5.

Backend, from repository root, keeping existing Azure `.env`:

```sh
MATCHMIND_PROVIDER=local MATCHMIND_CACHE=/tmp/matchmind-azure-startup-cache \
TACTIQAI_BACKEND_PYTHON=/tmp/tactiq-cv-m1/bin/python scripts/start_backend.sh
```

Frontend (Node 22): `cd frontend && npm run dev`. Open **http://127.0.0.1:5173**, refresh and select **test_match · CV Experimental**. Enable **Projected pitch geometry**, inspect the timeline and click a person. No Azure settings need changing. Keep the original local `test_match.mp4`; the ten SoccerTrack clips continue streaming Azure and need no local MP4s. The `/tmp` Python runtime must remain intact or be recreated with service/Azure and optional offline CV requirements.

Offline replay uses an unused output directory; it refuses to overwrite:

```sh
/tmp/tactiq-cv-m1/bin/python -X pycache_prefix=/tmp/tactiq-cv-pycache \
  -m src.vision.world.camera_review --video data/raw/test_match.mp4 \
  --baseline outputs/world_model/test_match_m4_verified \
  --poses outputs/world_model/test_match_m2_pose --output /tmp/tactiq-m5-reproduction
/tmp/tactiq-cv-m1/bin/python -X pycache_prefix=/tmp/tactiq-cv-pycache \
  -m scripts.evaluate_camera_placement --baseline outputs/world_model/test_match_m4_verified \
  --current /tmp/tactiq-m5-reproduction --output /tmp/tactiq-m5-evaluation \
  --video data/raw/test_match.mp4
/tmp/tactiq-cv-m1/bin/python -X pycache_prefix=/tmp/tactiq-cv-pycache -m unittest discover -s tests
```

From `frontend/` with Node 22 on PATH:

```sh
npm test
npm run build
TACTIQAI_AZURE_BROWSER=1 TACTIQAI_AZURE_REQUIRE_ANALYSIS=1 \
  ./node_modules/.bin/playwright test --config playwright.m5.config.ts
```
