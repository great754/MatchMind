# TactiqAI CV Milestone 7 — persistent camera localization

Verified **2026-10-27**. **IMPLEMENTED / LOCALLY VERIFIED / EXPERIMENTAL.** Synthetic known geometry is **REFERENCE-EVALUATED**. Independent real-video metric accuracy is **UNAVAILABLE**.

**The ≥50% test_match camera target was NOT reached.** The highest defended result from the evaluated routes is **33.64% camera / 32.22% placement**. This improves M6 coverage without changing the detector denominator or pixel residual gates. team_goal improves to **91.00% / 88.05%**. No previously accepted M6 frame is lost. test_match matched-pair jitter slightly worsens; coverage is not accuracy. This milestone does not establish readiness for ball tracking.

## Before / after

| Metric | test_match M6 → M7 | team_goal M6 → M7 |
|---|---:|---:|
| Camera coverage | 116/541 (21.44%) → **182/541 (33.64%)** | 417/500 (83.40%) → **455/500 (91.00%)** |
| Current player placement | 1,017/5,108 (19.91%) → **1,646/5,108 (32.22%)** | 4,522/5,515 (81.99%) → **4,856/5,515 (88.05%)** |
| Newly recovered / previously accepted lost | **66 / 0** | **38 / 0** |
| Longest supported run | 37 frames / 1.543s → **110 frames / 4.588s** | 117 frames / 4.68s → **140 frames / 5.60s** |
| Longest Unknown run | 349 frames / 14.556s → **305 frames / 12.721s** | 38 frames / 1.52s → **38 frames / 1.52s** |
| Matched-pair large-jump flags | 22 → **26** (650 pairs) | 294 → **217** (4,224 pairs) |
| Matched-pair displacement P90 | 0.502 → **0.546m** | 0.796 → **0.726m** |

Exact machine-readable results, full recovered/lost lists and typed/canonical/identity checks: `outputs/world_model/m7_verification/<video>/release/{coverage.json,comparison.json}`. The two fixed denominators remain 5,108 and 5,515. Current metric coordinates are obtained only from observed footpoints and accepted current camera transforms.

## Diagnosis before algorithm changes

A read-only report was written for **every M6 Unknown frame before estimator changes**: 425 test_match and 83 team_goal. `before/` retains that first report; `unknown_diagnosis/` refines category definitions without changing baseline data. The latter distinguishes explicit circle ambiguity from mere circle visibility. Categories overlap and are not additive. Missing metadata is not proof that an absolute solver could never work.

| Recorded failure indicator | test_match | team_goal |
|---|---:|---:|
| insufficient absolute geometry | 422 | 69 |
| insufficient independent geometry | 1 | 13 |
| missing weak semantics | 86 | 0 |
| failed field motion | 158 | 34 |
| lost texture anchors | 421 | 83 |
| visible region failure | 0 | 5 |
| temporal inconsistency | 0 | 1 |
| scale zoom rejection | 0 | 0 |
| line disagreement | 5 | 1 |
| circle ambiguity | 1 | 13 |
| no previous camera | 416 | 66 |
| cannot reconnect after gap | 415 | 66 |

Each Unknown row records previous/next localization, measured image-space motion and its actual gates, named lines, far touchline, observed/derived intersections, center-circle and box visibility, numerical image change, propagation plausibility and the existing rejection evidence. Image-change >15 mean gray levels at 160×90 is explicitly a diagnostic proxy, not a new hard-cut detector. Intersections derived from line fits are correlated measurements.

In test_match, **267/425 Unknown frames still had accepted M6 field motion**, yet retained world knowledge could not reconnect. M6 outcomes were 420 `NO_CURRENT_GEOMETRY`, four residual rejections and one independent-geometry rejection. The long gap was not fixed by an expiry change.

## Implemented architecture

1. **Persistent state and fixed world-field anchors.** `world_field_camera.py` retains camera transform, independent origin, camera epoch, signed origin age, last accepted frame, ten-frame measured motion history, pan and scale history, visible region and an uncertainty budget. Mature masked static feature IDs inherit fixed canonical coordinates from accepted cameras. An Unknown camera never seeds anchors. Features retain ID, first/last frame, image trajectory, observations, world coordinate, origin, epoch and inherited uncertainty. Missing/inconsistent features expire; their world coordinates are never adjusted to fit players.
2. **Two image measurement routes.** Original-luminance tracking remains preferred. A contrast-normalized route uses CLAHE and more candidate features (2,400 rather than 800; denser sampling and a lower candidate LK minimum eigenvalue), with the **same acceptance residuals**. Current boxes aggressively mask people. A confidently fitted observed far touchline also excludes the non-field half-plane before tracking. The grass-supported side determines the interior; Unknown cameras supply no guessed metric mask. These routes are correlated image evidence, not independent metric cameras. A validated contrast route fills only frames unavailable in the preferred route; route disagreement and selection remain visible. No source-specific frame or recording thresholds were added.
3. **Robust multiframe world fitting.** Fixed world coordinates and current tracked pixels fit a camera with RANSAC and held-out feature IDs selected before fitting. Require ≥40 correspondences, ≥30 training / ≥10 held-out, inlier ratio ≥0.8, held-out median ≤1 and P90 ≤2 working pixels. Full projective fitting requires ≥15% image hull. Narrow strips may instead constrain similarity/affine motion against a retained world-camera reference using existing rank, extrapolation, field overlap and per-frame scale gates. This does not declare a narrow strip sufficient for an unrestricted projective pose.
4. **Far touchline, intersections and circle.** Named lines are transported with measured field motion and compared by orientation/normal residual, not endpoint correspondence. The far touchline combines with horizontal motion and zoom. Current canonical line/conic residuals constrain acceptance and bounded correction. Derived intersection motion is tracked from the same line fits; it adds no independent weight. Small arcs can support a previous measured hypothesis but cannot create an absolute pose. There is no separate learned intersection tracker; persistent texture tracks and line constraints carry that information.
5. **Rolling geometry optimization.** `camera_window.py` robustly optimizes one image-to-pitch anchor against fixed lines/circles across measured neighboring images: normally up to five on each side, optionally up to about one second of past evidence for reacquisition. Relative transforms come from validated images. Extended acquisition requires two lines in each canonical direction, at least two agreeing local optimization starts, ≤4m hypothesis spread, safe visible geometry and the existing held-out semantic category gates (mean ≤10 / P90 ≤20 source pixels). Accepted cached pose candidates provide initializations, not permission to bypass validation. Future support and the actual window are recorded. This is a local shared-anchor temporal optimization, **not full-epoch bundle adjustment**, and nearby multi-start agreement does not prove global uniqueness.
6. **Adaptive continuation and bounded uncertainty.** Accepted field evidence, feature survival, held-out error, semantic agreement and inherited uncertainty govern continuation. The existing 20-source-pixel uncertainty budget remains, with a ten-second independent-anchor-age backstop. Fresh world remeasurement bounds drift using fixed-world constraints; new anchors inherit the full uncertainty and require an independent camera or a held-out validated world fit. Adjacent motion alone cannot continually renew anchors. There is no universal 0.25s expiry. Pixel uncertainty is heuristic, not calibrated metric covariance.
7. **Forward/backward offline reconstruction.** Each direction independently tracks the field, fits current cameras, and respects cuts, epochs and unsupported zoom. Opposite valid estimates must agree within **1m P90** to label recovered frames bidirectional. Actual matrices are selected; homography coefficients and player coordinates are never interpolated. A rejected M7 proposal may retain an already validated M6 camera; that distinction remains in diagnostics. Sparse/dense image routes are not labeled bidirectional merely because they agree.
8. **Absolute origin proof.** M6 can have a current partial/corrected camera and a separate genuine independent M5 origin. M7 preserves that exact separate origin rather than promoting the current partial matrix to an absolute fit. Every final origin has a machine-checked proof: preserved M6 independent anchor, M6 current direct anchor, or accepted M7 temporal-window anchor. Future evidence has an explicit signed age.
9. **Downstream people and teams.** Detections, image feet, identity/visibility, entity histories, team beliefs and team memory are frozen from M6. Players are projected after camera finalization. Historical coordinates never count as current placement. Camera fitting does not optimize against player coordinates; displacement decomposition remains a secondary diagnostic. SoccerTrack analytics, possession, transfers, GSR, grounding and event lead-in were not changed.

## Provenance, continuity and placement

| State | test_match M7 frames | team_goal M7 frames |
|---|---:|---:|
| DIRECT | 1 | 0 |
| RELOCALIZED | 11 | 69 |
| TEMPORAL_PREDICTED | 129 | 337 |
| TEMPORAL_CORRECTED | 0 | 19 |
| PARTIAL_CONSTRAINED | 15 | 21 |
| BIDIRECTIONAL_INTERPOLATED | 26 | 9 |
| UNKNOWN | 359 | 45 |

M6 state counts remain in the immutable baseline reports. test_match M7 has 359 Unknown cameras; team_goal has 45. Both clips have **zero detected cuts / one epoch (0)**; cut/zoom reset behavior is reference-tested synthetically. Real newly recovered bidirectional frames: **26 / 9**. A real directional-disagreement rejection occurs once in team_goal's original-luminance route; its contrast route has zero, as do both test_match routes. Rejected route proposals remain available even if another independently gated route supplies a camera.

Relocalized frame counts: test_match **10 →11**, team_goal **69 →69**. Selected independent-window refinements are recorded per anchor; compatible temporal-to-absolute drift corrections are **three** (frames37,93,129; 0.240/0.842/0.378m P90) and **two** (402,436; 0.483/0.726m). These differ from optimizing a raw anchor itself: the largest selected raw-anchor refinement is about **6.896m P90** at team_goal360, after a gap. It is accepted by static geometry/window checks, not by a claim that the baseline was ground truth.

Current unprojected observations: test_match **3,462** (3,132 no camera +330 outside the valid region); team_goal **659** (538 no camera +121 outside). Team A/B/unknown counts stay exactly **2,106/1,619/1,383** and **2,647/2,042/826**. Existing team_goal switches, identity ambiguities and unknown reasons are retained without reclassification.

## Stability proxies — separate from coverage

| Matched current-observation pairs | test_match M6 → M7 (650 pairs) | team_goal M6 → M7 (4,224 pairs) |
|---|---:|---:|
| Median displacement | 0.150753 →0.154030m | 0.285827 →0.277378m |
| P90 displacement | 0.502122 →0.546031m | 0.796254 →0.725799m |
| Maximum displacement | 4.839761 →4.802494m | 7.969877 →8.142252m |
| Large-jump screening flags | 22 →26 | 294 →217 |

Jump threshold remains `12m/s × frame interval +0.5m`. Real player movement and detector/identity errors are not independently distinguished. test_match's slight matched-pair regression is explicit. team_goal's tail maximum also increases despite lower median/P90/flag counts.

Across all eligible adjacent pairs, test_match has 650→1,160 pairs /22→41 flags; team_goal 4,224→4,640 pairs /294→257 flags. Those differing denominators cannot establish an accuracy improvement.

| Other measured proxy | test_match M6 → M7 | team_goal M6 → M7 |
|---|---:|---:|
| Adjacent camera change median / P90 | 0.372/0.590 →0.332/0.627m | 0.404/0.773 →0.363/0.748m |
| Motion-compensated camera change median / P90 | 0.097/0.185 →0.007/0.793m | 0.132/0.315 →0.023/0.767m |
| Visible-region median IoU | 0.9888 →0.9826 | 0.9855 →0.9851 |
| Valid world-fit held-out P90: median / P90 / max | 0.948/1.170/1.316 working px | 0.752/1.343/1.927 working px |
| Accepted-camera line category P90: median / P90 | 15.153/16.571 source px | 5.146/6.489 source px |
| Accepted-camera circle category P90: median / P90 | 14.501/15.898 source px | 5.037/5.870 source px |

The motion-compensated tails and test_match IoU regress; switching between valid camera/absolute hypotheses contributes. No general camera-jitter or independent-accuracy improvement is claimed. Field-fit distributions that include **rejected** attempts can exceed two pixels; only validated fits are used as successful world measurements. Full before/after camera proxies are in `proxy_comparison.json`; rejected attempts remain in the failure timeline.

Active world-anchor maxima are **1,850 /1,070**. Across all frames the median active counts are 0 /63.5 (Unknown intervals are included). Among validated current world fits, the median of mean active-anchor observation ages is **3.84 /3.97 frames**, and its P90 **10.29 /18.12 frames**. These are actual per-fit active-track ages, not a claim about an uncensored whole-population lifetime distribution. Each sampled anchor retains its lifetime and up to ten image positions. Whole-feature-cohort lifetime statistics and independent static-pixel labels remain unavailable. Median selected forward/backward agreement is **0.259 /0.428m**, P90 **0.613 /0.880m**. Full distributions, including feature survival and spatial support, are in the machine reports.

## The long test_match gap: exact failure and attempted reconnection

The M6 unsupported interval110–458 shrinks to **151–455**. The longest supported M7 run is **41–150** (4.588s); backward evidence recovers456 onward. Both final image routes and both directions are retained in `test_match/long_gap_routes.jsonl`, with condensed boundaries in `long_gap_boundaries.json`.

- At150, the contrast forward world fit has **907 anchors**, inlier ratio **0.872781**, held-out P90 **1.050362 working px**, hull **0.198205** and accepted current field motion.
- At151, it still has **1,201 candidate anchors**, but consensus falls to **0.715566**, below the unchanged0.8 gate; held-out P90 is1.523726. Adjacent field motion also fails. Many candidates are not the same as reliable static support. This is the first concrete forward failure for the longest recovered sequence.
- At152, named line disagreement and motion/world rejection persist. New Unknown frames do not replenish world anchors; their tracks eventually disappear.
- Backward tracking from the next independently grounded area succeeds at456 in the original-luminance route: only ten retained world anchors remain, but a new adjacent field-motion measurement is valid. At455 it has only **five anchors** and rejected adjacent motion. The contrast backward route also rejects: **313 anchors**, ratio **0.504167**, P90 **4.367757px**, hull **0.063722**.
- Therefore the trajectories **do not meet**. M6's accepted cached single-frame fits inside this broad interval are scarce; a new multi-frame anchor at129 is validated, but it cannot bridge the later static-feature break. Many midfield frames show only the top line plus a partial circle, or a single line. That cannot independently restore full pitch geometry. This was not fixed by increasing expiry or changing residual thresholds.

A measured semantic-label migration at146–148 is separately handled: about99.09% of the new `Small rect. left top` pixels follow the previously tracked `Big rect. left top`. The conflicting **label** is quarantined under retained field tracking; it contributes no independent absolute evidence. Pixel/image and all-category residual diagnostics remain visible. The pose is not fitted to pretend the small box moved.

To go farther: obtain reliable static correspondences through the151 transition and later pan, or independently validate a well-observed midfield relocalization using stronger line/conic geometry. Higher-resolution/robust feature correspondence and independent geometry references are the next recommended work. Simply accepting more of the present rejected fits would not meet the target honestly.

## Manual inspection and specific cases

Reviewed **67 before/after sampled frames**:29 test_match,30 team_goal, and eight additional goal frames recovered by the final pitch mask. Contact sheets cover beginning/middle/end of recovered runs, box and goal areas, top touchline, midfield/circle, pan/zoom context, weak semantics and Unknown transitions. Enlarged samples include test_match120/146/150 and goal60/250/350. Sampled recovered overlays align plausibly with visible markings; residual offsets remain. No sampled large drifting block was counted without qualification. This is a qualitative sample, not independent real metric ground truth or exhaustive manual labeling.

- **team_goal near0.058s:** the preceding-frame policy selects frame1 at0.040s. M7 preserves its measured supported camera, with origin2/0.080s and signed age−0.040s; future evidence remains explicit. This is still a boundary recovery, not evidence that the original M5 gap was two-sided.
- **team_goal350:** remains **Unknown**. No canonical overlay/current pitch player coordinate is published; image-space people and team memory remain inspectable. The tiny-arc wrong-circle behavior is not restored. The whole322–359 gap remains unsupported.
- **test_match216:** remains Unknown despite visible circle/top-line evidence. No historical point is counted as a current placement.

Artifacts: `release_images/contact_sheet.jpg`, per-frame `frame_NNN_comparison.jpg`, goal `mask_recoveries/contact_sheet.jpg`, and final browser screenshots under `m7_verification/browser/`.

## React review and live regression

Refresh **http://127.0.0.1:5173**. Select either CV Experimental video. Enable **Camera motion debug**, **Projected pitch geometry** and **Team classification debug**. Persistent orange feature trajectories have fixed-world-coordinate titles. The panel exposes actual active world support, held-out error, spatial hull/cells, track ages, survival, uncertainty, route, top-line/circle residuals, direction/agreements, origin and explicit quarantined labels. Existing marker provenance and player jump decomposition remain. Missing evidence stays unavailable. A world-anchor correction can exist without an adjacent-frame predicted region; that overlay is shown only when the measurement exists.

- **529 backend tests:528 passed, one expected optional-local-media skip.** Includes23 new M7 field/camera regressions and the existing guard against starting neural inference during API serving.
- **74 frontend tests /12 files passed. TypeScript and production build passed.**
- **Final full Chrome: six suites passed (1.4m).** All ten Azure clips individually select, play, seek, serve ranges and load tactical views, event lead-in and recording-scoped grounded Local Ask. Both CV recordings pass playback, seeking, stepping, camera/team/player debugging, recording isolation and CV Ask503. The affected prediction/team panel also passed separately (6.7s). After a final display-only fix for negative ages, both affected CV suites passed again (two passed/7.4s); all74 frontend tests and build were rerun successfully. Future anchors now read “future evidence in …” in both the camera and player panels.
- Earlier runs passed five suites but exposed an M6 test assumption: the first corrected frame need not have adjacent prediction evidence or a born persistent entity. The final integration test selects a frame with both actual measurements; application behavior was not weakened to satisfy it.
- Live HTTP: health200/ten analytics; ready200/**12 usable recordings**; catalog all ten Azure artifacts validated/ready plus two local CV-only sources. CV sources are available but are not Azure artifact-ready or football Ask-ready. Both CV and118575 video ranges206/16bytes.118575 view200/**481 events**; Local Ask200/status`ok`/**14 grounded evidence entries**. Both CV Ask requests503. Foundry was not exercised.

Existing Azure `.env`, artifacts and containers are unchanged; strict manifest/hash/schema validation remains. The local Azure SDK chain initially waited on the unavailable managed-identity endpoint; selecting its existing Azure CLI credential via standard `AZURE_TOKEN_CREDENTIALS` resolves that local startup wait. No MATCHMIND variable was renamed. No Azure objects were written; no new models or local video copies were downloaded. Existing Azure artifacts and video ranges were read during regression checks.

## Measured offline runtime

Both image routes are included in the final run; neural detection and semantic inference are **excluded** because M6 detections and cached semantic/pose data are reused. The legacy `team_filter_seconds` key measures serialization/copy of **frozen M6 team evidence**, not new team filtering.

| Video / route | Preprocessing | Static forward+backward motion | Camera/window processing | Frozen team evidence copy | Person replay/projection (includes team copy) | Route wall |
|---|---:|---:|---:|---:|---:|---:|
| test_match / sparse | 4.457s | 15.321s | 7.828s | 0.0439s | 0.505s | 30.761s |
| test_match / dense | 4.442s | 22.904s | 15.640s | 0.0335s | 0.397s | 45.948s |
| team_goal / sparse | 5.305s | 15.244s | 35.935s | 0.0523s | 0.349s | 58.482s |
| team_goal / dense | 4.935s | 22.445s | 17.860s | 0.0496s | 0.325s | 47.192s |

Final total measured wall time, including both routes, loading/serialization and selection: **83.657s test_match /109.715s team_goal**. Concurrent runs and local load affect these times. No end-to-end real-time claim is made.

## Files, proof and commands

New core: `world_field_camera.py`, `camera_window.py`, `coverage_review.py` under `src/vision/world/`. Updated `temporal_review.py` reuses offline replay and freezes M6 team evidence; `field_motion.py` adds optional candidate-density parameters with unchanged M6 defaults; `static_geometry.py` adds observed-boundary exclusion and semantic ambiguity diagnostics. Serving registry selects `*_m7_current`.

New scripts: `scripts/diagnose_cv_unknown.py`, `scripts/evaluate_camera_coverage.py`; existing temporal evaluator and renderer now support M7 origin proof and correct before/after version labels. Frontend changes: `TemporalDebug.tsx`, `CameraDebug.tsx` and its future-age regression, `PersistentField.test.tsx`, `camera-coverage.spec.ts`, M7 Playwright config and evidence-aware temporal integration test. New backend tests: `tests/test_cv_camera_m7.py`. Current status/explorer, README, frontend/Azure docs and historical M6 banner are updated at completion.

Final snapshots include both independently gated image routes for audit: `outputs/world_model/{test_match,team_goal}_m7_current/`. Final authority is `m7_verification/<video>/release/`, not earlier pilot comparisons. M4/M5/both M6 baseline hashes remain unchanged (`baseline_hashes.json`). Original video hashes remain matched to each manifest. **4,660,183,542 bytes of redundant generated M7 pilots removed**; manifests/diagnostics retained under `pilots/`. Both original CV videos, all model files, baselines and final snapshots remain. No source videos were duplicated.

Backend (repo root; current intact temporary Python runtime):

```sh
AZURE_TOKEN_CREDENTIALS=AzureCliCredential \
MATCHMIND_PROVIDER=local MATCHMIND_CACHE=/tmp/matchmind-azure-startup-cache \
TACTIQAI_BACKEND_PYTHON=/tmp/tactiq-cv-m1/bin/python scripts/start_backend.sh
```

Frontend:

```sh
export PATH="$HOME/.nvm/versions/node/v22.22.2/bin:$PATH"
cd frontend
npm run dev -- --host 127.0.0.1
```

Offline reproduction (choose fresh output directories; keep current snapshots immutable):

```sh
/tmp/tactiq-cv-m1/bin/python -X pycache_prefix=/tmp/tactiq-cv-pycache \
  -m src.vision.world.coverage_review --video data/raw/test_match.mp4 \
  --baseline outputs/world_model/test_match_m6_current \
  --poses outputs/world_model/test_match_m2_pose --output /tmp/tactiq-m7-test-reproduction
/tmp/tactiq-cv-m1/bin/python -X pycache_prefix=/tmp/tactiq-cv-pycache \
  -m src.vision.world.coverage_review --video data/raw/team_goal.mp4 \
  --baseline outputs/world_model/team_goal_m6_current \
  --poses outputs/world_model/team_goal/poses --output /tmp/tactiq-m7-goal-reproduction
/tmp/tactiq-cv-m1/bin/python -m scripts.evaluate_camera_coverage \
  --before outputs/world_model/test_match_m6_current \
  --after /tmp/tactiq-m7-test-reproduction --output /tmp/tactiq-m7-test-evaluation
/tmp/tactiq-cv-m1/bin/python -X pycache_prefix=/tmp/tactiq-cv-pycache -m unittest discover -s tests
```

Frontend directory, Node22:

```sh
npm test
npm run build
TACTIQAI_AZURE_BROWSER=1 TACTIQAI_AZURE_REQUIRE_ANALYSIS=1 \
  ./node_modules/.bin/playwright test --config playwright.m7.config.ts
```

Keep the intact temporary Python runtime, or recreate it with service/Azure and offline CV requirements. No Azure setting change is required for the recordings; keep the two original local CV videos. SoccerTrack videos remain streamed from Azure.

## Remaining work and stop

**PLANNED:** independently evaluate camera/identity geometry, recover reliable correspondences through151, validate midfield reacquisition and reduce hypothesis-switch jitter. **NOT IMPLEMENTED:** full-epoch trajectory bundle adjustment, independent metric ground truth, calibrated camera covariance, an uncensored population-level feature lifetime study, new learned intersection tracking, ReID, ball tracking or new CV football inference.

Stopped after camera coverage, diagnostics/UI and regression verification. No ball, possession, passing, shots, OCR, names, roster matching, commentary or new football Ask was implemented. The next milestone should address independent relocalization/remaining camera failures; the present evidence does not justify claiming the camera/player world model is globally ready for a ball milestone.
