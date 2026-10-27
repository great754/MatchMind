# CV Milestone 1 — persistent world-model prototype

Date: **2026-10-24**. Status: **IMPLEMENTED / LOCALLY VERIFIED / EXPERIMENTAL**. Not a validated football-player identity or metric-localization product. Historical Milestone 1 snapshot; subsequent work is documented in [Milestone 2](cv_world_model_m2.md).

## Audit and plan

Inspected the existing observations/schema and SoccerTrack adapter, analytics Recording/loader/pipeline, piecewise pitch transformer and projection diagnostics, existing torso-color classifier, tactical tracking/renderers, frontend Pitch and review contracts, canonical artifact store/validation, evaluation/tests, dependencies and both current status documents.

Reusable components: OpenCV/NumPy/SciPy/Pydantic/scikit-learn already belong to the project; `PitchCoordinateTransformer` is reused for supplied SoccerTrack calibration, `extract_jersey_color` for torso evidence and existing pitch dimensions for debugging. Existing observation schemas remain the canonical boundary. The neural detector is separate from deterministic tracking/world logic.

Architectural conflicts resolved explicitly:

- Tracker IDs are temporary; anonymous persistent entity IDs are separate and are never real names or jersey numbers.
- Kit clusters A/B do not imply attacking/defending left/right. The analytics compatibility bridge requires an explicit kit-to-side mapping and half.
- SoccerTrack panoramic calibration is piecewise/nonlinear. It is not flattened into a guessed homography or applied to unrelated footage.
- Existing generic legacy PlayerState required coordinates. The newer canonical observation contract supports nulls and is extended additively, rather than creating a competing canonical format.
- World history may retain a reliable previous coordinate; canonical current observations never borrow it as current.
- Detection and shirt-color scores do not establish individual identity.

Implementation plan used: typed state/contracts → detector and short-term tracker → camera/calibration providers → conservative global association and persistence → canonical adapter → debug viewer/evaluation → synthetic/regression tests and the supplied real video. Production football analytics and event playback were not modified.

## Architecture

```mermaid
flowchart LR
 V[Video] --> D[COCO person perception]
 V --> C[Camera / supplied calibration provider]
 D --> T[Short-term Hungarian tracking]
 T --> R[Inspectable reassociation hypotheses]
 C --> R
 R --> W[Persistent world state]
 W --> O[Existing canonical observations]
 S[SoccerTrack adapter, unchanged] --> O
 O --> G[Readiness / team-side / half requirements]
 G --> A[Existing analytics]
 A --> E[Canonical events / Ask / frontend / Azure]
```

`src/vision/world/` is additive. It does not register this experiment as an API recording, create an Azure artifact bundle, upload blobs, call a language provider or change possession, transfers, shots, pressure, formations, GSR alignment, synchronization, Ask grounding or playback policy.

## Model/library choices and licenses

These alternatives were evaluated at architecture/documentation level, **not benchmarked head-to-head**:

| Component | Selection / alternative | Decision |
|---|---|---|
| Detector | OpenCV Zoo YOLOX-S COCO | Selected: one 35,858,002-byte ONNX model, CPU inference through existing OpenCV; no Torch installation. General person class, not a football-specific model. |
| Detector | RT-DETR | Considered; would add another model/runtime integration without solving the current identity/calibration boundary. [Official Apache-2.0 license](https://github.com/lyuwenyu/RT-DETR/blob/main/LICENSE). |
| Detector | Ultralytics packaged YOLO | Considered; not installed. Its [official AGPL-3.0 license](https://github.com/ultralytics/ultralytics/blob/main/LICENSE) differs from the selected Zoo model. |
| Tracker | Small custom Hungarian IoU + appearance tracker | Selected: deterministic, inspectable, uses existing SciPy; no external tracker/ReID weights. It is weaker than a production camera-aware tracker. |
| Tracker | ByteTrack / BoT-SORT | Considered, not installed. [ByteTrack MIT](https://github.com/ifzhang/ByteTrack/blob/main/LICENSE), [BoT-SORT MIT](https://github.com/NirAharon/BoT-SORT/blob/main/LICENSE). Future integration remains possible. |
| ReID | Torso HSV histogram + existing LAB torso colors | Selected only as weak evidence. No learned individual-person ReID or OCR. |
| ReID | DeepSORT | Considered, not installed; [official GPL-3.0 license](https://github.com/nwojke/deep_sort/blob/master/LICENSE); pretrained-weight terms would also need checking. |

Selected model and wrapper information: [OpenCV Zoo YOLOX](https://github.com/opencv/opencv_zoo/tree/main/models/object_detection_yolox), [directory Apache-2.0 license](https://github.com/opencv/opencv_zoo/blob/main/models/object_detection_yolox/LICENSE). [OpenCV 4.5+ licensing](https://opencv.org/license/) is Apache-2.0. No models from the alternative stacks were downloaded. No neural network is loaded by the world-model unit tests.

Model SHA256: `c5c2d13e59ae883e6af3b45daea64af4833a4951c92d116ec270d9ddbe998063`. The detector checks it before loading. It expects RGB, unnormalized 640×640 letterboxing; source boxes are restored to original image coordinates. Foot point is bottom center, **not box center**. Occluded feet, truncated boxes and perspective remain limitations.

Apple Silicon: actual CPU execution was verified on this Mac (`arm64`). CUDA, Metal/MPS acceleration and learned embeddings were not implemented or benchmarked. The isolated runtime is `/tmp/tactiq-cv-m1`; exact installed packages are recorded in the experiment's `environment.txt`. The existing project environment and requirements pins were not overwritten.

## World state and confidence semantics

`WorldState`, `PlayerEntity`, `TrackObservation`, `IdentityHypothesis`, `AssociationEvidence`, `CameraState` and `BallWorldState` are typed Pydantic structures. Ball state is prepared but remains unknown; no ball detector exists here.

- A visible localized entity has a current coordinate only from valid supplied calibration.
- A visible unlocalized entity can retain its anonymous track/entity association while its current pitch position is null.
- A missing entity immediately loses current coordinates and current image boxes; after the configured grace it becomes `off_camera`.
- `off_camera` means no current observation after grace, **not proof** of physical exit. Occlusion, detector misses and camera movement can all cause it.
- Last image observation time and last reliable metric position/time are distinct. Missing entities are retained; no stationary or invented movement is emitted.
- Unknown tracks have no persistent actor ID. Ambiguous tracks keep alternative entity supports and an explicit unknown alternative.
- Identity support, detection confidence, team-color support, localization confidence and calibration confidence remain separate. Detector confidence is the model score; team support is a LAB-distance margin; identity support is appearance similarity gated by other evidence. Localization/calibration confidence is supplied when available, not invented from geometry. Support scores are **not calibrated probabilities** and hypotheses need not sum to one. Anonymous births do not receive invented numerical identity certainty.
- No exact position is propagated or predicted. The schema reserves those categories for future providers; they are not implemented.

Prototype thresholds are in `configs/vision/world_m1.json`; they are provisional, not empirically validated: image tracker grace .6s; missing-to-off-camera 1s; two hits for an anonymous birth; minimum IoU .05; appearance gate .90; association threshold .90; row/column ambiguity margin .08; movement bound 12m/s plus 3m slack; spatial-prior time scale 3s. Team separation/margin/distance and NMS parameters are exposed without changing their experiment defaults. Position-prior strength decays with elapsed time, but is diagnostic rather than an emitted predicted position.

### Reassociation and global constraints

Available evidence is logged independently: histogram similarity, team compatibility, elapsed time, calibrated displacement/movement feasibility and trajectory residual when available. Appearance provides the support score; known team mismatch, impossible metric movement and camera epoch mismatch reject candidates. Spatial proximity alone never establishes identity. One-to-one Hungarian assignment includes unknown/dummy slots, followed by row/column ambiguity checks.

**Default safeguard:** cross-track reassociation requires both sufficient appearance support **and a feasible calibrated movement check**. With only kit color and no localization, plausible candidates remain ambiguous. `allow_appearance_only_reassociation=true` exists solely to reproduce the documented weak baseline; it is false in the shipped config. Thresholds were not tuned to improve reported metrics.

The safeguard followed a concrete manual failure: the appearance-only prototype linked the red-shirted on-field `ENTITY_00005` at frame 1 to an orange-jacketed off-field person at frame 502 (20.94s), support .9017 after a 20.19s gap. Before/after images, exact evidence and baseline diagnostics/config are retained under `outputs/world_model/appearance_only_failure/`. The conservative output was checked at the same frame/temporary track: assignment is null/ambiguous and the original entity remains off-camera with no current location (`reviewed_failure_correction.json`). This is evidence for rejecting kit-only identity certainty, not evidence that the new system has achieved a higher measured identity accuracy.

Append-only reconciliations record segment start, resolution frame, entity assignment and supporting evidence when later evidence resolves an earlier unknown track segment. Original world snapshots and canonical observations are not rewritten. A retired temporary-track alias cannot claim an already reassociated entity twice; synthetic tests cover that safeguard.

## Camera/calibration boundary

`CameraProvider` accepts video-hash-bound, explicitly bounded homography intervals and optional frame-specific camera signals. It rejects mismatched video metadata, singular homographies and overlapping intervals; points outside the supplied validity region/pitch bounds remain unknown. An explicit cut increments the camera epoch, resets short-term continuity and invalidates old spanning calibration intervals. Automatic cuts, replay detection and optical pan/tilt/zoom estimation are **not implemented**.

`SoccerTrackCameraProvider` reuses the existing Delaunay/barycentric transformer with matching video provenance. No real SoccerTrack CV run was performed in this milestone because the previously removed local four-minute source clips remain absent. Tests verify the piecewise projection adapter with known-answer landmarks and the original SoccerTrack/analytics regression suite remains passing.

Visible pitch region is shown only when a supplied transform and validity polygon support computing it. Nothing is guessed for the supplied video. A formation window or another recording's calibration is never used as this camera's transform.

## Canonical observation integration

The existing `src/observations/schema.py` gains backwards-compatible optional track ID, visibility, detection/team confidence and localization-kind fields, additional provenance categories, and neutral `team_a`/`team_b` cluster labels. Existing SoccerTrack adapter values/array semantics are unchanged.

Directly calibrated metric projections are labeled **inferred** with `localization_kind=direct_calibrated`; they are not raw measured metric truth. Last-known, propagated and predicted coordinates cannot pass the direct-current-position bridge. Unknown calibration always emits null pitch coordinates. The existing BallObservation remains available to SoccerTrack; CV emits unknown ball observations.

The opt-in `analytics_recording` bridge requires explicit team-side mapping and match half, fixed actor rows and aligned frame clocks. It uses NaN for missing coordinates and never fills ambiguous identities or historical positions. A synthetic bridge test runs the unchanged `analyze_recording`: player motion remains measurable while absent ball evidence produces unknown possession and no fabricated passes/shots. This proves input compatibility, **not** production CV artifact/serving readiness.

## Supplied real experiment

Input: `/Users/greatabhieyighan/Desktop/Projects/matchmind/data/raw/test_match.mp4`.

This is the explicitly requested square-cropped broadcast clip: **1080×1080, 541 frames, 23.976 FPS, about 22.57s** (22.564 decoded frame-seconds; container duration 22.574s), 3,375,243 bytes. It is **not** a four-minute SoccerTrack recording. Source SHA256: `e092a252f47fd4f135506bd573ad36eaa25c95b0bbba3246c988827591c6dbac`. The original was reused and preserved; no source video was copied or downloaded.

No matching calibration, roster, player boxes, identity labels or GSR/reference correspondence was supplied. Therefore no real localization error, reference detection coverage or identity-switch accuracy is reported. Calibration and physically feasible metric return checks are unavailable on every frame.

Final conservative output: `outputs/world_model/test_match_m1_conservative/`.

| Diagnostic | Final measured result | Interpretation |
|---|---:|---|
| Person observations | 5108 | Detection events, not distinct players |
| Mean persons/frame | 9.442 | Range 1–17; officials/spectators can be included |
| Heuristic team-label coverage | 84.69% | Not team accuracy |
| Temporary tracks | 226 | Fragmented image tracks |
| Mean / median track span | 1.046s / 0.334s | Not identity continuity accuracy |
| Longest track span | 12.596s | Includes grace gaps |
| Within-track gap segments | 169 | Diagnostic, not MOT fragmentation score |
| Provisional entity births | 104 | Not a count of distinct football players |
| Accepted cross-track reassociations | 0 | Conservative default defers unlocalized kit-only evidence |
| Ambiguous track-frames | 1524 | Repeated observations, not distinct ambiguous people |
| Unknown/unconfirmed track-frames | 77 | Before stable anonymous entity allocation |
| Maximum off-camera candidates | 92 | Retained historical candidates, potentially duplicate physical people |
| Calibrated frames / metric samples | 0 / 0 | Unavailable; no invented geometry |
| Reference detection/ID/localization accuracy | null | Unavailable, never zero-error or perfect |
| Detector mean time | 107.90 ms | OpenCV CPU |
| Tracking mean time | 2.31 ms | Image assignment |
| World update mean time | 4.19 ms | Persistence/association |
| Detector-only / full measured FPS | 9.27 / 7.51 | Single local run, not an isolated hardware benchmark |

These results expose weak short-term continuity and ambiguous identities. The milestone establishes an inspectable uncertainty-preserving architecture; it does **not** establish reliable real-world persistent player identity. No detector comparison, independent precision/recall, HOTA/MOTA/IDF1 or calibrated identity probability is claimed.

### Manual sequence inspection

Source samples and synchronized debug views were inspected at approximately 0, 3, 6, 9, 12, 15, 18 and 21s; Chrome seeks/screenshots specifically cover 3, 9, 15 and 21s.

| Requested case | What the clip/evidence supports |
|---|---|
| Pan / disappearance | During the early pan, observed candidates disappear from detections. Their entity memory persists and current pitch coordinates remain null. Physical off-screen exit versus a detector miss is not individually labeled. |
| New players entering view | Later pan views create new temporary tracks and provisional anonymous entities where no defensible return assignment exists. These may duplicate earlier physical people; birth is not proof of a new roster player. |
| Temporary occlusion | Players overlap in the opening contest. The tracker logs grace-gap segments; these are not independently labeled occlusions. Synthetic tests verify missing-state behavior without attributing every actual gap to occlusion. |
| Return / re-identification | The appearance-only baseline's frame-502 player-to-steward mistake is visually documented. Final kit-only returns stay ambiguous; no successful real identity reassociation is claimed. Calibrated, distinctive synthetic returns do succeed. |
| Zoom / changing scale | The sampled views change scale/coverage, but no camera zoom parameter or verified zoom event is inferred. Synthetic close-up/calibration-loss tests preserve identity continuity without inventing metric location. |
| Calibration-poor sequence | All 541 actual frames are calibration-unavailable. The tactical pitch deliberately has no current player markers or visible-region polygon; image detections and identity hypotheses remain inspectable. |

## Debug outputs and reproduction

- `annotated.mp4`: generated source-plus-pitch debug video (MPEG-4 export; view locally).
- `debug.html`: synchronized original source video, detection/track/entity overlay, canonical pitch, unknown/ambiguous state table and expandable hypotheses.
- `world.jsonl`: typed immutable snapshots.
- `observations.jsonl`: existing canonical observation contract, one record per decoded frame.
- `transitions.jsonl`: visibility/track/localization transitions.
- `associations.jsonl`: inspectable evidence-matrix rows, assignment decisions, margins and insufficiency reasons. Eligible cost is `1 - support`; unknown slots have cost 1.
- `reconciliations.jsonl`: append-only segment-resolution ledger.
- `diagnostics.json`, `frame_diagnostics.json`: measured diagnostics, null unavailable metrics, configuration and limitations.
- `validation.json`: all 541 records schema/invariant checked; every detection accounted for; no invented current metric position or unlocalized histogram-only identity acceptance.
- `environment.txt`, `effective_config.json`: reproducibility information.

Run from the repository root in a suitable environment with the existing project dependencies:

```sh
/tmp/tactiq-cv-m1/bin/python -m src.vision.world \
  --video data/raw/test_match.mp4 \
  --model /tmp/tactiq-cv-models/yolox-s.onnx \
  --config configs/vision/world_m1.json \
  --output outputs/world_model/test_match_new_run
```

The kit prototypes are fitted unsupervised from 12 deterministic frames spanning the clip before the main pass. This is an **offline, noncausal warmup**, not a streaming team classifier or a learned individual ReID model. No football-label accuracy is inferred from that fit.

The source constructor never downloads a model automatically. The experiment used the documented 35.9MB Zoo model, fetched once and hash-verified outside Git. Temporary environments/caches can disappear; recreate an environment from the project dependencies and the recorded package manifest if needed.

Open the debug viewer (already running locally at the time of verification):

```sh
/tmp/tactiq-cv-m1/bin/python -m src.vision.world.serve \
  --output outputs/world_model/test_match_m1_conservative --port 18765
```

Visit **http://127.0.0.1:18765/**. The artifact-only viewer verifies the source video hash and serves native video byte ranges; it does not expose arbitrary repository files or duplicate the input. A generic Python file server was found to reset Chrome seeks to zero, so it was replaced rather than weakening the synchronization check.

No production frontend controls were added. Debug overlay updates follow native video events; the annotated export uses every processed frame. Faded last-known markers are supported when real prior calibration exists; they cannot be demonstrated on this entirely uncalibrated input.

## Tests and verification scope

- Complete backend suite: **381 tests ran; 380 passed, one optional missing local source-video skip**.
- Existing frontend suite: **44 passed**; production TypeScript/Vite build passed.
- Real Chrome debug integration: **1 passed**, including source playback, seeks to 3/9/15/21s, synchronized world-state time, retained missing entities, ambiguous identities and null metric positions.
- Existing SoccerTrack loader, piecewise mapping, observation round-trip, analytical known-answer and playback regression tests pass. The original real-clip integration test remains skipped because its local source MP4 was deleted in the earlier Azure cleanup.
- New deterministic tests cover visibility/off-camera history, strong calibrated returns, ambiguity, impossible movement, global one-to-one assignment, calibration loss/zoom, explicit cuts, unknown identities, source/confidence preservation, append-only reconciliation, downstream unknown-ball behavior, calibration hash/interval validity, reference metric availability and debug byte ranges.
- No new Azure or external language-model call was made. Previous Azure live-verification sections remain historical; Azure availability was not revalidated by this local CV milestone.

The original source video is unchanged. About 219MB of superseded generated pilot output was removed after preserving the appearance-only failure evidence. Model weights, videos, environments and generated experiment outputs remain excluded from Git.

## Files added/changed

Added: `src/vision/world/__init__.py`, `schema.py`, `detection.py`, `tracking.py`, `engine.py`, `camera.py`, `provider.py`, `evaluation.py`, `visualize.py`, `debug.html`, `serve.py`, `__main__.py`; `configs/vision/world_m1.json`; `tests/test_cv_world.py`; `frontend/tests/world-model.spec.ts`; `frontend/playwright.world.config.ts`; this documentation.

Modified: `src/observations/schema.py` (additive optional CV fields/categories only), `docs/current_pipeline_status.md`, `docs/current_pipeline_explorer.html`. Existing production analytics, frontend source, artifact validation and playback-policy source are unchanged.

## What is still unknown / next milestone

For each frame, TactiqAI knows which image person boxes were detected, their detector score, temporary track continuity, optional kit-color cluster, anonymous entity memory or unresolved hypotheses, source/provenance and any valid supplied localization. It knows when it last observed a candidate and separately when it last localized it. It **does not know** a real name/jersey, verified individual identity, exact off-camera movement, metric position without calibration, ball state, exact camera pose, replay status or independent accuracy on this video.

**NOT IMPLEMENTED:** learned individual ReID, automatic pitch perception, automatic camera-cut/replay/zoom understanding, jersey/roster/name recognition, metric propagation and ball perception. **NOT REFERENCE-EVALUATED:** the supplied broadcast video, persistent identity accuracy and real metric localization. **REFERENCE/COMPATIBILITY TESTED SYNTHETICALLY:** known geometry, global assignment and the explicit analytics bridge.

Recommended Milestone 2: obtain one matching SoccerTrack video/calibration/reference set, run a reference-backed detection/tracking/localization baseline, add a defensible player/official/spectator distinction and individual appearance evidence, then evaluate return identities before relaxing the conservative default. No threshold optimization against this unlabeled broadcast and no automatic Milestone 2 implementation.
