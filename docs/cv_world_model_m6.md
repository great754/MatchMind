# TactiqAI CV Milestone 6 — temporal camera stability and persistent team belief

Verified **2026-10-26**. **IMPLEMENTED / LOCALLY VERIFIED / EXPERIMENTAL.** Known-geometry tests are **REFERENCE-EVALUATED (synthetic)**. Independent metric camera, team and identity accuracy remain **UNAVAILABLE**.

M6 adds measured camera prediction/correction, named-line validation, bounded offline forward/backward checks, explicit team-belief evidence and readable review diagnostics. It is more conservative: both videos lose accepted coverage; unknown-team assignments increase. The strongest measured improvement is lower matched-pair position jitter in team_goal. The frame near0.058s is recovered from measured future evidence; the questionable frame350 geometry is suppressed. These are not claims of full-video geometric accuracy.

## Baselines and inspection before implementation

Read current status/explorer, M5 and team_goal reports, camera/pose/motion/world/kit/schema/provider code, serving, React review, existing tests and actual snapshots. Ran a diagnostic over **every transition of both immutable M5 snapshots before changing algorithms**. Later supplementary diagnostics add all confidence fields and analytically derived intersection movements. No source/baseline file was overwritten.

- `test_match`: `outputs/world_model/test_match_m5/`,541frames/5,108observations; frames SHA-256 `e93c379ea1240c13b56e9a779349e5e5add43c2909dbcb61639231d4e264223a`.
- `team_goal`: `outputs/world_model/team_goal/final/`,500frames/5,515observations; frames SHA-256 `47279325cfddd241001df38e3f3e9ebb0eb67a39af0d0c06009f8986659d4962`.
- M4 remains unchanged: `outputs/world_model/test_match_m4_verified/`, frames SHA-256 `29b16164047b2bd63e37685ebe6ae17e0e97932d2865bc69df41e3a601b27ac9`.

Machine-readable before evidence: `m6_verification/<video>/before/` and expanded `baseline_detailed/`. Every transition records previous/current/next states, epoch, semantic names, retained motion/feature statistics, line correspondences/normal-angle changes, derived observed intersections, transform/region changes, per-entity foot displacement decomposition, team/confidence changes and camera support. Unavailable original measurements remain unavailable; no feature coordinates were invented for M5 records that did not retain them.

Before findings:

- Isolated supported→unknown→supported gaps: test_match **1** (94); team_goal **13** (15,18,60,66,119,155,168,170,173,179,322,462,472).
- Persistent-entity team A→unknown/opposite→A patterns: **zero in both videos**. M4 already held established entity teams through weak observations. Combined track-display unknown flickers: test_match **15**, team_goal0.
- Current-crop evidence flickers still existed: test_match **61 unknown +1 opposite**; team_goal **108 unknown +18 opposite**. Current kit evidence and displayed persistent belief are different measurements.
- Overall large-displacement screening flags: test_match24; team_goal749, using `12m/s × frame interval +0.5m`. These are not independently labeled teleports.
- At0.058s, preceding-frame sampling selects frame1/0.040s, not a frame at0.058. It is a boundary case: frame0 and1 were both Unknown, frame2 accepted. M5 had no prior and zero mature field tracks at1 despite nine named semantic classes. It was not originally a two-sided isolated hole.
- team_goal frame349 had a **17.913m P90 motion-compensated transform change** on diagnostic query points. Frame350's adjacent delta was only0.803m. Looking only at350's immediate change misses the earlier break and stable wrong geometry.

## Camera architecture and evidence

The canonical pitch stays105×68m. Original semantic inference and absolute-fit caches are reused; no new neural model, inference training or video was downloaded. Person detections, image footpoints, anonymous identity/association history and visibility remain frozen. Only team beliefs, current camera/projection evidence, metric history and diagnostics change.

The offline chain is:

```text
accepted independent anchor / accepted camera history
  → newly measured static-field point and named-line motion
  → inverse image-motion composition predicts current image-to-pitch H
  → bounded robust current line/conic correction
  → temporal/observability/visible-region checks
  → accepted current H, or Unknown
  → original observed bottom-center foot projection, or null
```

**Static field:** reuse persistent mature LK feature IDs with preceding-frame static membership, people-excluded grass support and fixed held-out-ID splits. Measure forward and reverse separately. A fresh point route is available when it passes the same guards. RANSAC chooses the smallest supported similarity/affine/projective model. Existing gates remain: ≥40 points for similarity with≥30train/10held-out;≥50for higher-parameter models; inlier fraction≥0.8; held-out median≤1/P90≤2working pixels; per-frame Jacobian scales0.98–1.02; field overlap≥0.8; non-folding/conditioning/spatial-support guards. Working width is≤640; source scaling is retained. Moving people are masked using image boxes, not world coordinates.

**Named lines:** transport fitted infinite ground lines through measured image motion and compare current fitted-line position/orientation. Variable visible segment endpoints are not assumed to be the same physical points. Fitted centerlines remove line-thickness bias from this motion check. Median normal error≤2/P90≤4working pixels and rotation discrepancy≤2° label a line tracked; majority contradiction across multiple current lines rejects the motion. The far/top touchline contributes explicitly. Accepted adjacent field motion also validates the top touchline on **345 test_match /374 team_goal** records. A valid top line never supplies absolute pose by itself.

Detected circle/arc semantics constrain the current pose through canonical conic residuals; field-marking features contribute to measured flow. Observed line-intersection movements are derived conservatively from nonparallel supported segments. They are shown/logged as correlated line evidence, not counted as extra independent measurements. This is not a new full conic-motion solver or globally relocalizing field map.

**Prediction/correction:** `H_pred = H_previous × inverse(H_image_motion)`. Current semantic correction uses the existing robust bounded optimizer. Supported semantic categories retain mean≤10/P90≤20source-pixel gates; the correction remains within the existing12source-pixel limit. There is no per-player coordinate smoothing.

**Temporal sanity:** compare a fresh absolute candidate with motion-compensated accepted geometry; >2m P90 disagreement flags TEMPORAL_GEOMETRY_INCONSISTENT. Current validated correction can support a different result; contradictory raw fits do not override it automatically. Baseline raw-neighbor disagreements are separately retained even after current M6 continuity has already been lost. This compensates for measured fast pan before flagging disagreement; large raw coefficient change alone is not rejection.

**Independent reacquisition:** requires a genuine M5 independent current fit plus richer current observability: two named ground-line families with≥2lines each, or broad center-circle support(≥120°) with halfway and touchline geometry. This is a heuristic observability safeguard, not proof of absolute accuracy. Propagated/partial input snapshots cannot become independent anchors; a dedicated regression enforces that boundary.

**Remeasured texture recovery:** M6 keeps accepted static-field feature keyframes using the existing strict TextureAnchors machinery. Unknown frames do not add anchors. Fresh current-to-keyframe correspondences, holdouts, field agreement and current semantics can restore continuity: **9 test_match /27 team_goal** accepted frames use this route. This preserves measured camera history without relabeling an old propagated pose as independent.

**Memory/uncertainty:** separate typed last-independent and last-accepted transforms, epoch, accepted/failure history, current landmarks, motion/correction evidence and relative image-error budget. RSS motion budgets remain capped at20source pixels. They are heuristic relative diagnostics, not absolute metric covariance; systematic/correlated errors remain. Current player localization confidence stays null. A future offline anchor is explicitly tagged with timestamp and signed age; it is not called a previous direct observation.

## Bounded offline forward/backward handling

An entire unsupported block must be≤**two frames and0.10s**. This minimally covers the measured two-frame clip prefix and isolated one-frame holes; it does not extend M5's0.25s reuse lifetime. At25FPS,two frames are0.08s; at24000/1001FPS,0.0834s. Each step requires a new accepted field measurement, finite region, same epoch and no cut/unsupported zoom. Budgets are accumulated from actual held-out errors.

For internal gaps, forward and backward hypotheses must both pass current geometry where available and agree within **1m P90**. When all semantic markings drop out, two validated field-motion directions can support the short gap; a one-sided boundary estimate still requires current semantic constraints. The output selects a validated hypothesis rather than averaging homography coefficients. Boundary recovery with only one available side is TEMPORAL_PREDICTED, with direction/future anchor explicit.

Measured real-video result: **two backward boundary predictions** in team_goal; **zero real BIDIRECTIONAL_INTERPOLATED frames**, **zero real disagreements with two completed compatible candidates** in either video. Other short gaps did not have enough validated directions/current agreement. They remain Unknown. Synthetic tests exercise successful one/two-frame bidirectional recovery and explicit disagreement rejection; no synthetic counts are presented as real coverage.

Cuts/epochs reset active camera and keyframe continuity and block interpolation. Both videos have one detected epoch(`[0]`) and0detected cuts. Synthetic hard-cut/epoch/zoom tests cover rejection and reset; no independent cut/zoom accuracy benchmark is claimed.

## Exact camera and placement results

| Metric | test_match M5 | test_match M6 | team_goal M5 | team_goal M6 |
|---|---:|---:|---:|---:|
| Frames |541|541|500|500|
| Person observations |5,108|5,108|5,515|5,515|
| Accepted camera frames |137|116|475|417|
| Camera coverage |25.32%|**21.44%**|95.00%|**83.40%**|
| Projected observations |1,208|1,017|5,116|4,522|
| Placement coverage |23.65%|**19.91%**|92.77%|**81.99%**|
| Unknown current positions |3,900|4,091|399|993|
| No camera geometry |3,609|3,821|278|882|
| Outside calibrated region |291|270|121|111|

Coverage decreased under stricter current-evidence and independent-anchor rules. This does not establish that every new rejection was necessary or every remaining projection accurate. test_match loses21previously accepted camera frames and recovers0previous unknowns; team_goal loses65previously accepted frames and recovers7previous unknowns: **0,1,60,462,469,470,472**. Full per-frame lists/reasons are in each final comparison JSON.

| Camera state | test_match M5 | test_match M6 | team_goal M5 | team_goal M6 |
|---|---:|---:|---:|---:|
| DIRECT | 2 | 1 | 279 | 1 |
| RELOCALIZED | 17 | 10 | 41 | 69 |
| TEMPORAL_TRACKED | 4 | 0 | 0 | 0 |
| TEMPORAL_PREDICTED | 0 | 0 | 0 | 2 |
| TEMPORAL_CORRECTED | 0 | 7 | 0 | 223 |
| PARTIAL_CONSTRAINED | 114 | 98 | 155 | 122 |
| BIDIRECTIONAL_INTERPOLATED | 0 | 0 | 0 | 0 |
| UNKNOWN | 404 | 425 | 25 | 83 |

There are **100 test_match /331 team_goal ordinary measured forward-prediction attempts**, including rejected/corrected candidates. State TEMPORAL_PREDICTED counts above specifically identify the two accepted boundary predictions; TEMPORAL_CORRECTED and PARTIAL also use measured priors. Projection counts by state:

- test_match:DIRECT4,RELOCALIZED86,TEMPORAL_CORRECTED54,PARTIAL873; total1,017.
- team_goal:DIRECT5,RELOCALIZED770,TEMPORAL_PREDICTED9,TEMPORAL_CORRECTED2,533,PARTIAL1,205; total4,522.

Canonical source kinds preserve temporal/partial/bidirectional distinctions and camera state retains DIRECT vs RELOCALIZED. All propagated kinds remain quality`propagated` and excluded by the existing direct-only analytics bridge.

### Runs, gaps and flags

| Metric | test_match M5 → M6 | team_goal M5 → M6 |
|---|---|---|
| Longest supported run |55frames/2.294s →37frames/1.543s(459–495)|142frames/5.68s →117frames/4.68s(360–476)|
| Longest unknown run |304frames/12.679s →349frames/14.556s(110–458)|6frames/0.24s →38frames/1.52s(322–359)|
| Isolated unknown gaps |1 →3|13 →10|
| M6 geometry inconsistency flags |0|14|
| Real bidirectional interpolations / disagreement rejections |0/0|0/0|

The longer unknown intervals are a material limitation. In test_match the weak current anchor at137 is not accepted for a cold independent reset; measured field history does not reconnect the lost interval. In team_goal stricter evidence suppresses the322–359interval containing the documented circle problem. Recovery is conservative, not a general full-clip camera solution. Isolated-gap counts alone can change when larger gaps split/merge; the per-frame recovery lists are authoritative.

## Player-coordinate consistency

Current positions use the unchanged observed bounding-box bottom center. Unknown/nonvisible current positions remain null; direct historical coordinates remain separate. The pipeline freezes all image detections, tracks/identity association, identity confidence and visibility. Structural evaluation checks **24,935 test_match /11,635 team_goal entity rows**, unresolved tracks/hypotheses, all1,041world and canonical frames, clocks, source/frame hashes and genuine independent-anchor provenance.

For each consecutive visible entity pair, diagnostics record displacement/speed, same-foot camera component, changed-foot component, image-foot displacement and track change. A large-jump flag uses the unchanged screening threshold`12m/s × dt +0.5m`. The vector decomposition is diagnostic: physical movement vs detection error is not independently known. No output footpoint or coordinate is smoothed to hide a flag.

| All accepted consecutive pairs | test_match M5 → M6 | team_goal M5 → M6 |
|---|---|---|
| Pair count |779 →650|4,796 →4,224|
| Median displacement |0.161518m →0.150753m|0.449325m →0.285827m|
| P90 displacement |0.506741m →0.502122m|1.207400m →0.796254m|
| Maximum displacement |4.839761m →4.839761m|18.109768m →7.969877m|
| Large-jump flags |24 →22|749 →294|

Because coverage changed, the comparable matched-pair analysis is more informative:

| Same frame/entity pairs in both outputs | test_match(650pairs) M5 → M6 | team_goal(4,140pairs) M5 → M6 |
|---|---|---|
| Median |0.158189m →0.150753m|0.448923m →0.286442m|
| P90 |0.513541m →0.502122m|1.155075m →0.795420m|
| Maximum |4.839761m →4.839761m|8.828734m →7.969877m|
| Large-jump flags |24 →22|611 →284|

Thus the team_goal consistency improvement is not solely the result of rejecting more frames. It still does not measure true player speed or absolute metric accuracy. Large foot/identity/detection errors remain.

## Persistent team belief and confidence separation

M6 replays current kit observations chronologically over frozen identity assignments. There is no backward/future team labeling. Entity belief and unresolved-track kit belief are separate; a same-track history can transfer at the existing unambiguous anonymous entity birth, without creating a new identity association.

Typed evidence retains current team/support/source, current raw observation/quality, independent sample count, uncalibrated A/B weights, uncertain count, candidate team/streak, last strong observation, history and switch audit. Identity confidence, team evidence, visibility and position support remain separate in world/canonical/frontend schemas.

Independent strong samples require≥0.08s spacing, torso quality≥0.35, foreground≥0.2, field context≥0.3 and prototype margin≥0.6. Brightness/chroma/field contrast are inherited from the existing torso extractor; no new kit model was trained. Acquisition requires≥3independent samples, weight≥1.2and≥0.8relative support. An established belief switches only after≥6independent strong contradictory observations spanning≥0.4s and the competing weighted support wins. Low-quality crops neither erase an established team nor count toward a switch. These are evidence weights/ratios, not calibrated probabilities.

| Team/stability metric | test_match M5 → M6 | team_goal M5 → M6 |
|---|---|---|
| Team A visible assignments |2,372 →2,106|2,793 →2,647|
| Team B visible assignments |1,652 →1,619|2,320 →2,042|
| Unknown visible assignments |1,084 →1,383|402 →826|
| Persistent-entity isolated unknown/opposite flickers |0 →0|0 →0|
| Combined track-display unknown flickers |15 →0|0 →0|
| Persistent team switches |1 →0|3 →3|
| M6 visible temporal-memory assignments |1,177|1,545|

Unknowns increase because acquisition/current evidence is stricter. Zero persistent flicker was already a baseline property; it is not claimed as a newly measured improvement. Current-crop evidence still has61unknown/1opposite patterns in test_match and108unknown/18opposite in team_goal: the filter exposes those weak observations separately from persistent display. The former test_match switch at503/ENTITY_00064 does not recur under the new evidence window; that is not proof of physical identity correctness.

Unknown/white semantics after M6:

| Reason | test_match | team_goal |
|---|---:|---:|
| TEAM_UNKNOWN_NO_EVIDENCE | 6 | 7 |
| TEAM_UNKNOWN_AMBIGUOUS_COLOR | 211 | 280 |
| TEAM_UNKNOWN_LOW_QUALITY_CROP | 872 | 358 |
| TEAM_UNKNOWN_IDENTITY_UNRESOLVED | 170 | 25 |
| TEAM_UNKNOWN_CONFLICTING_HISTORY | 124 | 156 |

A historically supported entity with a bad crop displays its persistent team, source`temporal_memory`, weak current evidence and separate identity/position support. An unresolved track can have kit-team evidence while its physical identity remains unresolved. No named clubs, defending sides or22-player roster are inferred.

### Audit of the three original team_goal switches

All three remain auditable, rather than being silently removed to reduce a metric. Source crops and input histories are retained in `team_goal/switch_audit.json`, `switch_audit_inputs.json`, and`switch_<frame>_crops.jpg`.

| Original event | Manual evidence/category | M6 event |
|---|---|---|
|136/ENTITY_00011,A→B|**Likely tracker/identity error; subsequent kit correction supported.** TRACK_00014's crop changes from blue at101 to red/white by116. A team filter cannot verify the same physical person survived that track.|148;18supporting independent contradictory samples|
|359/ENTITY_00025,A→B|**Supported kit-belief correction; identity ambiguous.** All sampled324–384crops are red/white while old persistent A was stale. Weak current evidence349 did not erase history.|366;28supporting samples|
|380/ENTITY_00021,B→A|**Supported kit-belief correction; identity ambiguous.** All sampled345–405crops are blue, supporting A rather than the stale B belief.|438;54supporting samples|

These are not verified physical players changing teams. Every M6 switch stores old/new team, frame/time, independent support count, before/after vote weights, sample-span/quality evidence and reason. Stronger ReID/association accuracy is unresolved and was not added.

## Required cases

### team_goal near0.058s

At25FPS the current evidence frame is1/0.040s; next independent anchor is2/0.080s. M5:UNKNOWN/NO_PRIOR and no mature field features; five current observations have no pitch coordinates. M6: **TEMPORAL_PREDICTED**, direction**backward**, based on a fresh2→1field measurement with **227matched field points,159RANSAC training inliers**, held-out median**0.3823px/P90 1.3996px** at640pxworking width, and a validated top touchline. Current semantic correction passes unchanged gates. All five observations project; identity remains provisional/unresolved as appropriate.

The anchor is40ms in the future relative to frame1. Typed provenance records anchor timestamp0.080s, signed age−0.040s, backward direction and measured error budget; the UI explicitly says future evidence. It does not call this a previous DIRECT or a bidirectional interpolation. Frame0 also recovers through two newly measured backward steps. No timestamp-specific production condition exists.

[Frame1 before/after](../outputs/world_model/m6_verification/team_goal/images/frame_001_comparison.jpg).

### team_goal frame350

M5:accepted DIRECT despite visibly questionable extrapolation of a small center-circle arc. M6 first records the349discontinuity as **TEMPORAL_GEOMETRY_INCONSISTENT**. At350 there are current static line/flow measurements, but no reobserved valid world anchor and insufficient independent current geometry. Result: **UNKNOWN / INSUFFICIENT_INDEPENDENT_GEOMETRY**. Current coordinates and canonical overlay are absent; visible people and their team evidence remain inspectable.

The misleading circle is **suppressed**, not proved repaired. The small central arc(~38.5°under the old fit), short penalty arc and limited line families do not justify resetting global geometry. A temporally smooth wrong transform can still pass local residuals; M6 is not a full ambiguity/accuracy solution.

[Frame350 before/after](../outputs/world_model/m6_verification/team_goal/images/frame_350_comparison.jpg) · [frame349](../outputs/world_model/m6_verification/team_goal/images/frame_349_comparison.jpg).

## Review UI and actual visual verification

Visible marker legend: solid D/DIRECT; solid R/RELOCALIZED; outlined T/TEMPORAL; dotted P/PARTIAL; distinct patterned B/BIDIRECTIONAL; faded H/historical;?identity unknown; gray team unknown; no tactical marker for unknown position. Labels/borders accompany color. Region captions preserve actual provenance.

Optional **Camera motion debug** shows measured accepted/rejected field points and vectors, held-out point distinction, current named lines, highlighted far/top touchline and derived intersection movement. Tactical outlines separate previous(gray), predicted(amber) and current accepted regions. Diagnostics expose direction/time, matched points/training inliers, top-line support, pan/scale, agreement when available, current residual/epoch/reason, and reobserved-anchor evidence separately from adjacent motion. Per-player position diagnostics present camera vs foot/track evidence readably. Team debug shows persistent belief/source, weak current crop, independent samples/candidate/last strong observation and switch history.

Manual source/debug inspection:16paired frames per video, including initial recovery, pan, midfield, penalty/goal lines, weak landmarks, loss/recovery and late framing. The problematic350circle is absent under rejection. Accepted samples retain visible residual offsets; this is post-hoc plausibility inspection, not independent reference accuracy. Reconstructed geometry outside observed markings remains uncertain.

[team_goal contact sheet](../outputs/world_model/m6_verification/team_goal/images/contact_sheet.jpg) · [test_match contact sheet](../outputs/world_model/m6_verification/test_match/images/contact_sheet.jpg) · [verified team_goal UI](../outputs/world_model/m6_verification/browser/team_goal-m6.png) · [verified test_match UI](../outputs/world_model/m6_verification/browser/test_match-m6.png).

## Runtime

Cached replay seconds, measured separately for each clip:

| Component | test_match | team_goal |
|---|---:|---:|
| Preprocessing: cached semantics + source/field decode | 3.407553 | 4.296000 |
| Forward/reverse static-field measurements | 12.000733 | 11.659971 |
| Camera filter + offline gap checks | 2.639884 | 7.919121 |
| Team-belief accumulator calls only | 0.009099 | 0.010668 |
| Team/position replay, excluding file serialization | 0.294570 | 0.234209 |
| Total measured M6 replay |20.266775|25.898377|

Total temporal-camera work(static measurement+filter) is **14.641s /19.579s**. Pure team-accumulator timings are not the whole team/position replay cost. Total includes output/hash work, but **excludes original detector/segmentation/absolute-solver inference**, reused from immutable caches. This is not an end-to-end real-time benchmark.

Removed **2,311,311,885bytes** of redundant generated M6 pilots after preserving their diagnostics/metadata. Current snapshots, every M4/M5baseline, original videos and semantic/model evidence remain. No extra annotated MP4, new weights/libraries/dataset downloads or Azure writes occurred.

## Tests and live regression verification

**506backend tests ran:505passed,one expected optional deleted-local-SoccerTrack-video skip. 71frontend tests across11files passed. TypeScript/production build passed.**35new temporal-camera/position/team cases plus a serving guard that patches detector/segmenter/pose constructors to fail and proves precomputed startup/frame/media requests never invoke them. Synthetic tests cover pan/tilt/zoom/cut/epochs, forward/backward prediction, one/two-frame agreement, explicit disagreement rejection, gap/time bounds, no stale/partial-to-direct promotion, top/multiple lines/RANSAC outliers, temporal geometry anomalies, position decomposition, sticky bad-crop team memory, independent sample spacing, sustained switching and confidence separation. Frontend cases cover legend, debug, intersections, future/temporal state, persistent-vs-current team, provenance labels and readable jump/anchor evidence. Existing structured analytics, Azure manifest/hash/schema validation, grounding and event-seeking tests remain in the full suite.

**Final full Chrome:5passed in45.1s.** All ten Azure clips34.9s; existing CV review2.8s; player/team review1.7s; team_goal review1.9s; M6both-video/debug/legend/team review2.7s. After the final explanatory anchor-evidence row, the affected both-video test passed again: **1passed in5.1s**(test4.0s). No page/media errors. Actual selection, playback, seeking, frame stepping/synchronization, new debug controls, video isolation, event lead-in and recording-scoped grounded Local Ask passed.

Live API:health200/ten analytics recordings; **ready200/12usable recordings**; discovery ten Azure-ready plus two local CV-only reviews. Both CVvideo ranges and118575range206/16bytes. Existing118575view200/481events; Local Ask200/statusok/14evidence entries. Both CVfootball Ask requests remain503. No Microsoft Foundry request is claimed.

## Files, commands and artifacts

New: `src/vision/world/static_geometry.py`, `temporal_camera.py`, `team_belief.py`, `temporal_review.py`; `scripts/diagnose_cv_temporal.py`, `evaluate_cv_temporal.py`, `render_cv_temporal.py`; `tests/test_cv_temporal_m6.py`; `frontend/src/TemporalDebug.tsx`, `TemporalDebug.test.tsx`; `frontend/tests/temporal-world.spec.ts`; `frontend/playwright.m6.config.ts`.

Updated for M6: world schema and optional field-motion debug output; canonical schema/provider; CVReview/CameraDebug/CameraTimeline/PlayerWorld and related tests; precomputed serving regression test; existing browser selector precision; CVregistry; README, current status/explorer, frontend/Azure docs and historical report banners. Existing unrelated dirty-worktree changes are preserved and not attributed to this milestone. SoccerTrack analytics, possession, GSR, event timestamps/lead-in, grounding and MATCHMIND variable names were not changed.

Current registry:

- `test_match` → `outputs/world_model/test_match_m6_current/`
- `team_goal` → `outputs/world_model/team_goal_m6_current/`

All proof: `outputs/world_model/m6_verification/`; authoritative matched comparisons: `<video>/final/comparison.json`; complete final transitions:`<video>/final/release_after/transitions.jsonl`; original diagnostic/baseline detail, team audit/crops, before/after images, browser JSON/screenshots and logs are alongside them.

Backend from repository root, existing Azure `.env` retained:

```sh
MATCHMIND_PROVIDER=local MATCHMIND_CACHE=/tmp/matchmind-azure-startup-cache \
TACTIQAI_BACKEND_PYTHON=/tmp/tactiq-cv-m1/bin/python scripts/start_backend.sh
```

Frontend(Node22):`cd frontend && npm run dev`. Refresh **http://127.0.0.1:5173**, select either CVExperimental recording, enable **Camera motion debug** and **Team classification debug**, and scrub/click the timeline. No Azure configuration change is needed. Keep both original local CVvideos. The temporary Python runtime must remain intact or be recreated with service/Azure and optional offline CV requirements.

Offline reproduction uses unused output directories:

```sh
/tmp/tactiq-cv-m1/bin/python -X pycache_prefix=/tmp/tactiq-cv-pycache \
  -m src.vision.world.temporal_review --video data/raw/test_match.mp4 \
  --baseline outputs/world_model/test_match_m5 \
  --poses outputs/world_model/test_match_m2_pose --output /tmp/tactiq-m6-test-reproduction
/tmp/tactiq-cv-m1/bin/python -X pycache_prefix=/tmp/tactiq-cv-pycache \
  -m src.vision.world.temporal_review --video data/raw/team_goal.mp4 \
  --baseline outputs/world_model/team_goal/final \
  --poses outputs/world_model/team_goal/poses --output /tmp/tactiq-m6-team-reproduction
/tmp/tactiq-cv-m1/bin/python -m scripts.evaluate_cv_temporal \
  --before outputs/world_model/team_goal/final --after /tmp/tactiq-m6-team-reproduction \
  --output /tmp/tactiq-m6-team-evaluation
/tmp/tactiq-cv-m1/bin/python -m scripts.render_cv_temporal \
  --before outputs/world_model/team_goal/final --after /tmp/tactiq-m6-team-reproduction \
  --output /tmp/tactiq-m6-team-images --frames 1 349 350
/tmp/tactiq-cv-m1/bin/python -X pycache_prefix=/tmp/tactiq-cv-pycache -m unittest discover -s tests
```

From`frontend/`with Node22:

```sh
npm test
npm run build
TACTIQAI_AZURE_BROWSER=1 TACTIQAI_AZURE_REQUIRE_ANALYSIS=1 \
  ./node_modules/.bin/playwright test --config playwright.m6.config.ts
```

## Remaining limitations and stop condition

Coverage is lower, unknown teams are higher, and several formerly accepted plausible frames are conservatively lost. test_match's long camera gap remains unresolved and is longer under the independent-anchor policy. No real internal gap passed both directions in these clips; bidirectional acceptance/disagreement is reference-tested synthetically, not demonstrated as a real-video success. Current semantics, flow and kit evidence can still be wrong together; temporal smoothness is not accuracy. The original entity11crops expose a likely tracking identity swap; this milestone does not implement ReID. Full-view geometry, true player movement, physical identity/role/team accuracy and calibrated uncertainty need independent references.

**Stopped after M6.** No ball detection/tracking, possession/pass/shot/formation/pressure/tactical inference, names, roster matching, face recognition, commentary or new CVAsk grounding was added. Before the proposed video-only ball milestone, independently evaluate the remaining camera/identity failures and decide whether stability is sufficient; this report does not assert global readiness merely from passing software tests.
