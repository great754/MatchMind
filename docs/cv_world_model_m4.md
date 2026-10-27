# TactiqAI CV Milestone 4 — persistent player/team world memory

> Historical immutable M4 baseline. Current camera/placement authority: [Milestone 5](cv_world_model_m5.md). M4 results below are preserved as verified history.

2026-10-25. **IMPLEMENTED / LOCALLY VERIFIED / EXPERIMENTAL.** Targeted crop inspection is **REFERENCE-EVALUATED**; independent full-video team, role and identity accuracy remain unavailable. This is a player/team upgrade over the unchanged Milestone 3 camera. No ball tracking, possession, passing, OCR, names or named rosters were added.

## Diagnosis before changes

Inspected current status/explorer/M3 documentation, all relevant world and canonical models, detector, team extraction, tracker, global association, camera/projection, serving index, React review and CV tests. Existing architecture was retained. The original path was YOLOX generic people → per-observation torso color → short image tracks → provisional entities → occasional pitch projection. It already retained historical positions and used global Hungarian assignment, but team changes were not temporally modeled and returning image trajectories were not an independent continuity source.

The baseline run accounted for all **541 frames / 5,108 observations** before changing team thresholds. Its torso extraction excluded **every sufficiently saturated pixel with HSV hue 30–95**. That exclusion was also used in the weak identity histogram. In the actual selected yellow/green crops it removed **77–99%** of the torso, sometimes leaving skin/background as kit evidence. The contact sheet visibly shows green/yellow shirts assigned Team A. Gray-shirted official-like people were also assigned Team A. Whole-clip unsupervised prototypes did not solve that bad input representation.

Baseline persistent entities changed teams **26 times**. There were **226 temporary tracks**, **135 spans shorter than 0.5 s**, and 104 provisional entity births. These diagnose instability/fragmentation, not independently measured identity switches.

Before evidence: `outputs/world_model/m4_verification/before_team_diagnosis.json`, `before-team-crops.jpg`; immutable source snapshot `outputs/world_model/test_match_m3/`.

## Architecture and model/library changes

`video + audited person boxes → upper-torso foreground features → multi-frame segment aggregates → two anonymous kit populations → temporary tracking → temporal entity team belief → global continuity/ambiguity → camera-aware memory → canonical observations → existing analytics gates`

**No new model, library, installation or weight download.** Retained the approved OpenCV Zoo YOLOX-S COCO-person model, OpenCV CPU inference, NumPy/SciPy/scikit-learn and the existing Hungarian implementation. YOLOX SHA256 remains `c5c2d13e59ae883e6af3b45daea64af4833a4951c92d116ec270d9ddbe998063`. There is no learned football-role or person-ReID model in this milestone. Inference remains offline; web requests only read validated indexed snapshots.

### Kit extraction and video fitting

- Use the central upper torso: horizontal 22–78%, vertical 18–48% of the observed box, with image clipping. Avoid most head/shorts/legs/background.
- Estimate local field color from a ring below the person. Remove torso pixels only when close to that local LAB field color, rather than excluding all green shirts. Field-ring hue detection does not itself exclude torso hues.
- Use dominant quantized LAB chroma, robust nearby-color medians, foreground fraction/purity, and an HSV histogram. Keep dominant BGR/LAB, torso region, local-background LAB, field context, quality, prototype distances and reason as typed evidence. Quality is heuristic, not a probability.
- Fit **50 multi-frame temporary-segment profiles**, with at least three qualifying samples each; one aggregated profile per segment prevents long tracks from dominating solely by frame count. Require detection, torso and field-context support. Fit two supported chromatic populations with robust median centers. Low-quality/achromatic/background crops and distant/ambiguous prototype matches remain unknown.
- Learned centers for this clip are LAB **(95,171,148.5)** and **(189.375,112,182)**; anonymous labels are assigned by sorted prototypes. In this clip these visually correspond to red and yellow/green. Neither colors nor defending sides are hard-coded. Segment populations were 26 and 24; these are not distinct-player counts.
- This is **offline video-specific fitting**, using multiple frames throughout the clip. It is not a causal online calibration or an independent held-out evaluation. Manual crop labels never enter extraction, fitting, thresholds or association.

### Temporal team identity

`TeamMemory` separates current observation from persistent identity: decayed A/B vote weights, unknown count, usable evidence count, contradiction streak, eight recent typed kit samples and historical LAB. Acquire a team only after multi-frame weighted support. Unknown/hidden torso evidence does not erase a learned team. A stable assignment requires sustained contradictory observations and a new supported vote majority before switching. Every acquisition/change records frame, time, previous/new team and actual counts/weights.

Votes are support ratios, **not calibrated 100% confidence**. The UI explicitly labels them heuristic. Off-camera/currently missing entities have no current kit or current team observation; retained team, last kit, recent/historical features remain memory. One recorded switch remains: ENTITY_00064 at frame 503, A→B, 32 usable kit observations, weights A=1.766/B=7.849, contradiction streak 12. It is inspectable; it is not automatically called a corrected identity or independently verified team decision.

### Detection, NMS and geometric plausibility

Re-ran the real detector on all 541 frames, twice. The repeated audit reproduced **identical boxes and scores on every frame**. Confidence remains **0.4**, NMS **0.5**.

| NMS threshold | Whole-clip observations |
|---|---:|
| 0.3 | 5,066 |
| 0.5 | 5,108 |
| 0.65 | 5,201 |

Five sampled additional proposals at 0.65 (frames 135,315,330,360,435) visually duplicate existing people/body extents. That does not prove every extra proposal is a duplicate or establish detector recall. There was no evidence to justify globally weakening suppression. Kept 0.5. A nested body-part proposal remains in the frame-100 case and is explicitly flagged; no claim of solved merged-player detection.

Added **soft** local size checks: compare image foot-depth neighbors; when calibrated, also use nearby ground positions. This is empirical perspective/context support, **not a 3D human-height estimate from a ground homography**. Preserve the detector score and all observations, and expose a separate geometric support value. No universal pixel-height limit and no automatic removal for being tall.

Observed diagnostic counts (overlapping categories, observation counts):

| Supported diagnostic | Count |
|---|---:|
| Overlapping boxes | 547 |
| Edge truncation | 262 |
| Unusual aspect ratio | 185 |
| Nested-box duplicate candidate | 29 |
| Sudden tracked bbox-size jump | 22 |
| Local size outlier | 7 |

These are geometric warning conditions, not confirmed counts of merged people, heavy occlusions or detector errors. `TWO_PLAYERS_MERGED` is not fabricated without evidence.

### Persistent entity memory and visibility

Typed entity memory now includes first/last seen, current versus last image boxes/feet, last image velocity, current and last kit, accumulated appearance profile/count, team memory, track history, current pitch/source, reliable pitch/time/epoch, role hypothesis and bounded typed history. Current observations remain separate from historical evidence.

Explicit memory states:

- `visible_tracked`: a current detection supports the entity and temporary track.
- `temporarily_occluded`: a short unobserved gap; occlusion is a hypothesis, not independently confirmed.
- `off_camera`: unsupported visibility after grace, or a last reliable area outside the camera region even after conservative movement slack.
- `lost`: camera epoch changed and identity continuity is unsupported.

Entities remain in memory/roster in every state. Missing/off-camera/lost entities have **null current image/pitch observations**. Historical image/kit/pitch evidence persists. No predicted association coordinate is published as a current observation. Team roster state is an anonymous set of hypotheses, with no 11-player cap or forced roles. Per-entity recent history retains up to 64 events; diagnostics retain all team changes from the run.

Role outputs are conservative `team_a_outfield` / `team_b_outfield` **hypotheses** where stable kit plus field context support them; otherwise `unknown_person`. Officials/goalkeepers are typed possible roles, but **automatic official/goalkeeper recognition is NOT IMPLEMENTED**. Gray official-like crops remain unknown rather than being forced into an outfield team. Staff, spectators or colorful keepers can still contaminate generic-person observations; no independent role accuracy claim.

### Tracking, reassociation and ambiguity

Short tracking uses measured image velocity to predict association boxes across brief gaps. It outputs only detections, never missing-frame predicted positions. Strong image overlap remains central; noisy kit observations no longer unconditionally break a new kit-aware temporary track.

Persistent returning tracks are globally assigned one-to-one. The original appearance gate (0.90), acceptance threshold (0.90), row/column margins (0.08), movement bounds and cut/epoch gates remain. Long returns still require direct calibrated movement evidence; shirt color/position alone is insufficient. Propagated camera positions still cannot authorize a metric movement check.

A separate short-image continuity route requires same continuous epoch, gap ≤0.6 s, measured prior image velocity, ≥3 usable kit frames, predicted foot error ≤0.35 previous box heights, bbox consistency ≥0.8, appearance ≥0.95 and the same global margins. Its score combines **0.55 appearance + 0.30 normalized image-trajectory support + 0.15 bbox consistency**. Team compatibility, elapsed time, world movement/trajectory, track history and camera-region evidence remain explicit. Region membership is contextual evidence, not standalone identity proof. Appearance profiles accumulate features rather than claiming learned ReID.

Equal/competing candidates remain unresolved with hypotheses. Roster continuity defers matching new tracks to plausible old entities rather than forcing births or identities. The real final replay accepted **zero cross-track reassociations** and contains no entity with multiple accepted track segments. Known-answer synthetic tests demonstrate supported image-only and metric returns. Real long-return identity continuity remains a limitation; an intermediate pilot's one acceptance is not reported as a final success.

## Before/after measurements

Same original video SHA256 `e092a252f47fd4f135506bd573ad36eaa25c95b0bbba3246c988827591c6dbac`, 541 frames, 1080×1080, 24000/1001 FPS, approximately 22.57 s. Identical frozen M3 camera states and detector boxes permit a controlled player/team comparison.

| Measurement | M3 baseline | M4 final |
|---|---:|---:|
| Person observations / per-frame mean | 5,108 / 9.442 | 5,108 / 9.442 |
| Temporary tracks | 226 | **151** |
| Mean track span, including gaps | 1.046 s | **1.545 s** |
| Track spans <0.5 s | 135 | **77** |
| Persistent entity hypotheses created | 104 | **84** |
| Mean observed entity lifetime | 1.493 s | **2.008 s** |
| Persistent team switches | 26 | **1** |
| Team A / Team B / unknown visible assignments¹ | 3,335 / 1,412 / 361 | **2,372 / 1,652 / 1,084** |
| Same-entity recoveries after missing visibility | 71 | **110** |
| Off-camera transitions | 92 | **72** |
| Ambiguous association track-frames | 1,524 | **1,263** |
| Unconfirmed track-frames | 77 | **72** |
| Accepted cross-track reassociations | 0 | **0** |
| Entities with multiple accepted track segments | 0 | **0** |

¹ Combines persistent team belief for assigned entities with current team evidence for unresolved observations. Raw current-observation assignments in M4 are separately A=2,217 / B=1,616 / unknown=1,275. Persistent visible-entity samples are A=1,658 / B=1,275 / unknown=840. Unknowns rise because early, weak, gray/background or contradictory observations are withheld rather than forced into a team. Different hypotheses/track fragmentation make switch counts diagnostics, not identity accuracy. There were 52,951 ineligible candidate comparisons, not 52,951 unique rejected people.

### Targeted visual reference and manual frames

At **frame 100 / 4.171 s**, the nearest reproducible six-red/four-yellow-green scene, inspect ten selected unique person crops; exclude one nested body-part duplicate. Agent visual crop-label agreement improves **8/10 → 10/10**. This is **targeted, post-hoc visual reference evaluation**, not independently annotated full-video accuracy. Labels were excluded from training, but the frame was not held out from unsupervised video fitting.

There are nine currently assigned entities in that frame (five Team A/four Team B), plus two unresolved tracks. One unresolved track is the sixth red player, with known Team A observation but uncertain persistent identity; the other is the nested Team B proposal. Team evidence does not force identity. The field now colors localized unresolved observations by supported team and marks identity with `?`.

Twelve before/after visual frames cover 0,34,55,100,135,216,281,350,459,480,503,538: clear kits, nearby opponents, overlapping/partial proposals, pans, edge exits/truncation, penalty area, center circle, direct/tracked/unknown calibration and team-change investigation. They do **not** establish a verified long-gap returning-player identity. Contact sheets include Team A, Team B, unknown official-like crops and the baseline misclassified green/yellow crops. Files: `frame-*-comparison.jpg`, `before-team-crops.jpg`, `after-team-crops.jpg`, `nms-sensitivity.jpg`, `manual_team_reference.json`.

### Center circle and every unprojected person

At frame **216 / 9.009 s**, semantic evidence contains **Circle central + Side line top**. That is insufficient distinct ground/curve evidence for the absolute solver, so **it was not attempted**. There is no active tracking anchor. Recognizing a circle is not a valid camera pose. This is not frontend filtering, stale synchronization, or a valid foot point being discarded after an accepted calibration.

Image-space people remain: **five assigned entities + two unresolved tracks**. All seven have `NO_CAMERA_GEOMETRY`; current metric positions remain null. Every visible person now has an explicit projection outcome in the world and canonical contracts. Actual whole-clip outcomes:

- **4,801** `NO_CAMERA_GEOMETRY`.
- **262** `PROJECTED` (145 direct entity + 79 tracked entity + 38 unresolved-identity positions).
- **45** `OUTSIDE_CALIBRATION_REGION`.

Known-answer tests additionally exercise invalid feet, failed/unsupported transforms, accepted center projection and region clipping. No confidence-based rejection path is invented. The M3 camera is byte-equivalent in all 541 typed states: **21 direct + 12 tracked = 33/541 (6.10%)**; 508 unlocalized frames. Camera coverage is not player-tracking accuracy and was not loosened to draw more people.

## React and arbitrary-frame beliefs

World Model panel shows actual entity/team counts, visible/temporarily occluded/off-camera/lost states, unresolved identities, current tracks and persistent hypotheses. Clicking boxes, pitch markers or the entity list shows current/previous tracks, first/last seen, role, persistent team support, projection reason, size warnings and association hypotheses. Team-debug mode displays a decoded-video torso crop, dominant color, LAB summary, cluster distances, current versus accumulated evidence, recent features and recorded changes. Crops clear during seeking and redraw after the decoded frame is ready.

Overlays distinguish temporary TRACK IDs from persistent ENTITY hypotheses. Direct positions are solid; propagated positions outlined; historical ghosts faded/hollow and explicitly historical. Unknown coordinates are never drawn as current. Team labels, IDs, support, detections, ambiguities and historical positions have toggles. Confidence values remain heuristic support, not asserted probabilities.

**Explicit answer to the milestone's final question:** at any indexed video frame, `GET /matches/test_match/cv/frame?timestamp=T` returns the retained anonymous entities, each current supporting track or null, team belief/evidence, visibility/memory reason, current versus historical coordinates, role hypothesis, kit/profile/history and unresolved-track hypotheses. React exposes those same beliefs. The system does not know a verified named or 22-player roster.

For example, frame 216 retains **43 entity hypotheses** (roster A=14/B=10/unknown=19): **five visible**, **two temporarily occluded**, **36 off-camera**, plus **two visible unresolved tracks**. Current entity support:

| Entity | Current track | Team belief | Usable kit evidence frames | Projection |
|---|---|---|---:|---|
| ENTITY_00017 | TRACK_00017 | A | 175 | No camera geometry |
| ENTITY_00035 | TRACK_00046 | B | 96 | No camera geometry |
| ENTITY_00036 | TRACK_00048 | Unknown role/team | 26 | No camera geometry |
| ENTITY_00039 | TRACK_00060 | B | 42 | No camera geometry |
| ENTITY_00041 | TRACK_00040 | A | 24 | No camera geometry |

TRACK_00070 and TRACK_00062 have Team A observations but unresolved persistent identities. Team A/B historical vote ratios for the four known-team visible entities are 1.0 here; that is **not 100% accuracy**. The gray official-like person remains `unknown_person`. Other entity IDs remain queryable in the same snapshot with historical evidence and null current observations.

## Verification, runtime and Azure preservation

**433 backend tests ran: 432 passed, one expected optional local-source-MP4 skip. 53 frontend tests passed across nine files. TypeScript/production build passed.** Added deterministic kit/grass, two-population learning, stable/missing/contradictory team evidence, image-gap continuity, conservative long-return/ambiguity, camera-aware memory, epoch loss, center projection/rejection, soft perspective-size and canonical-reason tests. Existing global assignment, analytics, Azure artifact-validation, camera, grounding and event-lead-in tests still pass.

**Fresh final Chrome: three tests passed in 42.6 s**: all-ten Azure **37.8 s**, existing CV camera review **2.6 s**, new player/team review **1.4 s**. Actual source playback/seeks, preceding-frame synchronization, direct/tracked/lost/relocalized geometry, source overlays when calibration is unavailable, team crops, persistent off-camera teams, null current positions, person inspection and recording isolation passed. No page/media errors. All ten clips select/play/seek, load tactical views, retain event navigation/lead-in and support recording-scoped grounded Local Ask:

`117092_clip,117093_clip,118575_clip,118576_clip,118577_clip,118578_clip,128057_clip,128058_clip,132831_clip,132877_clip`.

Live API: `/health` 200 / ten analytics recordings; `/ready` 200 / **11 usable recordings**; `/recordings` 200 / ten Azure-ready plus CV-only `test_match`; 118575 view 200; video bytes 0–15 returned **206 / 16 bytes**. Grounded Local Ask returned **200/status ok/118575_clip/14 evidence entries**. Experimental CV football Ask remains rejected (503). No SoccerTrack analytics thresholds, possession semantics, GSR alignment, grounding rules, existing MATCHMIND variables, Azure blobs or infrastructure changed. No experimental upload or local-artifact fallback.

Real repeated detector audit: **35.631 s / 541 frames (~15.18 detector-audit FPS)**, including proposal/NMS sensitivity work. Player upgrade over cached detector boxes/frozen camera: feature fitting **4.927 s**, world replay **3.954 s**, measured total **9.462 s (~57.17 cached-upgrade FPS)**. Timings include the actual local run conditions; they are not end-to-end live CV throughput. M3 semantic/pose inference was reused, not rerun or newly benchmarked. No GPU claim.

Final source: `outputs/world_model/test_match_m4_verified/`. Full evidence: `outputs/world_model/m4_verification/`, including `evaluation.json`, `verification.json`, `live_http.json`, detector repeat check/raw proposal cache, manual references, comparisons/contact sheets, browser results/screenshots and test/build logs. Removed redundant M4 pilots/repeated cache after preserving their diagnostics and equality check, freeing **292,443,435 bytes**. Final M4 and all M1/M2/M3 source evidence, models and original video remain intact. No extra annotated MP4 created.

## Files changed

New: `src/vision/world/kits.py`, `player_geometry.py`, `player_review.py`; `scripts/audit_player_detection.py`, `evaluate_player_world.py`; `tests/test_cv_players_m4.py`; `frontend/src/PlayerWorld.tsx`, `PlayerWorld.test.tsx`; `frontend/tests/player-world.spec.ts`; `frontend/playwright.m4.config.ts`; this report.

Updated: `src/vision/world/schema.py`, `detection.py`, `tracking.py`, `engine.py`, `provider.py`; `src/observations/schema.py`; `configs/serving/cv.json`; `frontend/src/CVReview.tsx`; `frontend/tests/cv-review.spec.ts`; README, current status/explorer, Azure artifact documentation and the historical M3 banner. Earlier uncommitted changes are preserved; no model/video/dataset publication was performed.

## Exact commands

Start backend from repository root, retaining existing `.env` Azure settings:

```sh
MATCHMIND_PROVIDER=local MATCHMIND_CACHE=/tmp/matchmind-azure-startup-cache \
TACTIQAI_BACKEND_PYTHON=/tmp/tactiq-cv-m1/bin/python scripts/start_backend.sh
```

Start frontend:

```sh
cd frontend
PATH=/Users/greatabhieyighan/.nvm/versions/node/v22.22.2/bin:$PATH npm run dev
```

Open **http://127.0.0.1:5173**, refresh and select **test_match · CV Experimental**. Enable Team classification debug and click a player. No Azure setting changes are required. Keep `data/raw/test_match.mp4` for this explicitly local experiment. SoccerTrack videos continue streaming Azure; their deleted local MP4s are unnecessary. The temporary Python environment must remain intact or be recreated with the service/Azure and optional offline CV dependencies.

Reproduce a new offline run without overwriting current evidence:

```sh
/tmp/tactiq-cv-m1/bin/python -X pycache_prefix=/tmp/tactiq-cv-pycache \
  -m scripts.audit_player_detection --video data/raw/test_match.mp4 \
  --model /tmp/tactiq-cv-models/yolox-s.onnx --output /tmp/tactiq-m4-audit.json

/tmp/tactiq-cv-m1/bin/python -X pycache_prefix=/tmp/tactiq-cv-pycache \
  -m src.vision.world.player_review --video data/raw/test_match.mp4 \
  --baseline outputs/world_model/test_match_m3 --detections /tmp/tactiq-m4-audit.json \
  --output /tmp/tactiq-m4-review

/tmp/tactiq-cv-m1/bin/python -X pycache_prefix=/tmp/tactiq-cv-pycache \
  -m scripts.evaluate_player_world --before outputs/world_model/test_match_m3 \
  --after outputs/world_model/test_match_m4_verified \
  --cache outputs/world_model/m4_verification/detection_cache.json \
  --output outputs/world_model/m4_verification

/tmp/tactiq-cv-m1/bin/python -X pycache_prefix=/tmp/tactiq-cv-pycache -m unittest discover -s tests
cd frontend
PATH=/Users/greatabhieyighan/.nvm/versions/node/v22.22.2/bin:$PATH npm test
PATH=/Users/greatabhieyighan/.nvm/versions/node/v22.22.2/bin:$PATH npm run build
PATH=/Users/greatabhieyighan/.nvm/versions/node/v22.22.2/bin:$PATH \
TACTIQAI_AZURE_BROWSER=1 TACTIQAI_AZURE_REQUIRE_ANALYSIS=1 \
./node_modules/.bin/playwright test --config playwright.m4.config.ts
```

## Limits and next milestone

Team stability and targeted crop agreement improved, but no independent full-video team/identity/detector/role accuracy is established. Generic-person misses, nested duplicates, overlapping bodies, background people, blurred/occluded kits, fragmented identities and unreliable long returns remain. Unknowns deliberately increase. Eighty-four hypotheses are still far more than a verified on-field roster. Zero final cross-track acceptances is not a solved reassociation system.

Metric localization remains the dominant bottleneck: 6.1% camera coverage and a 13.39 s unknown gap. Recommend **further camera calibration/model validation next**, with independent labeled pitch points/contiguous pan-zoom-cut cases and targeted player continuity evaluation before ball work. **PLANNED**, not implemented. Ball detection/tracking, new possession/passes, OCR, names, named rosters and learned ReID remain **NOT IMPLEMENTED**. Stop after this milestone.
