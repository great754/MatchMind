# CV Milestone 9 — persistent player world model and team intelligence

Verified 2026-11-01. **IMPLEMENTED / LOCALLY VERIFIED / EXPERIMENTAL.** Registry selects three separate `*_m9_ready` snapshots. Camera model remains **M8**; player/team model is **M9**, algorithm `m9-evidence-weighted-global-v1`. No camera retraining, new ball perception, face recognition, roster names, production analytics or cloud writes.

**Result:** a new persistent anonymous-player graph, five-way team belief, explicit uncertainty/visibility state and dedicated inspector are operational. Actual identity reconnect gains remain modest on test_match and team_goal. Team labels still contain visually apparent errors. This is not a claim that robust ReID/team accuracy is solved or that fragmentation fell substantially on every video.

## Frozen camera and raw-observation baseline

Read the status, explorer, M8 report and all three snapshots before implementation. Actual manifests/frame hashes and original M8 comparisons are frozen in `outputs/world_model/m9_verification/frozen_m8_baseline.json`. Every final frame is checked against its M8 source: remove only the new `player_world` field and the original JSON object is identical. All old cameras, detections, temporary tracks, identities, team evidence, footpoints, canonical coordinates and old ball-unknown states remain intact. Existing `observations.jsonl` is copied byte-for-byte; M9 canonical player observations use a separate file.

|Source|Camera M8 = M9|Current placement M8 = M9|Raw person observations|
|---|---|---|---|
|test_match|269/541 (49.72%)|2179/5108 (42.66%)|5108|
|team_goal|456/500 (91.20%)|4861/5515 (88.14%)|5515|
|southampton|600/660 (90.91%)|7419/8725 (85.03%)|8725|

All1,701frames/19,348observations were evaluated. The separate canonical streams were also fully schema/clock/identity/coordinate/completeness validated:26118/11963/16436records respectively,including explicit historical-state rows;canonical SHA256s are in each evaluation JSON. M8 camera gates and metric results were not changed to help ReID. Test frame216 remains Unknown. Original videos, all M8 snapshots and final M9 snapshots are preserved.

## Architecture and evidence lineage

```text
immutable M8 detections/current camera observations
  → short-track segments (raw tracker ID + epoch + bounded gap)
  → source/model-hash-cached body embedding and foreground kit evidence
  → offline tracklet candidate graph
  → global one-to-one assignment with alternative-assignment margin
  → persistent PLAYER_NNN anonymous hypotheses
  → forward persistent team belief + visibility/history state
  → authoritative player_world + separate VIDEO_CV observation stream
```

`WorldState.player_world` is the new authority; the original M8 association arrays remain inspectable for comparison. A raw tracker ID is never presented as verified player identity. Short gaps within the existing0.6s tracker grace stay in a short track; the same raw ID across a hard camera epoch becomes a separate segment. Segments require at least3actual observations before a persistent entity is created. Shorter fragments remain structured unresolved detections with their original tracks, geometry and uncertainty. Offline confirmation is explicit, so later observations may support an earlier provisional segment.

Persistent IDs are neutral `PLAYER_001`,etc. Team/role may change without renaming identity. They are hypotheses, not a verified count of physical people. No objective seeks22players. Provisional IDs remain explicitly provisional; a successful reconnect changes identity status to ASSOCIATED with separate evidence score. An ambiguous new fragment receives a provisional competing hypothesis, not a forced old identity.

Typed records retain independent identity confidence, team confidence/probabilities, appearance confidence and current-position confidence. These scores are **uncalibrated**. Known identity/team can coexist with unknown current position; a localized observation can have unknown identity/team. Current position always copies the original accepted M8 observation. No identity smoothing, velocity prediction, old player memory or team label modifies that coordinate.

### Appearance representation and cache

The selected pretrained representation is the author's **OSNet x0.25 MSMT17-combineall** checkpoint, evaluated against the existing foreground-colour histogram proxy. The author publishes the architecture and checkpoint in the [Torchreid model zoo](https://kaiyangzhou.github.io/deep-person-reid/MODEL_ZOO); the [architecture source](https://github.com/KaiyangZhou/deep-person-reid/blob/master/torchreid/models/osnet.py) is vendored unmodified with its MIT licence. Generic [Torchvision MobileNet](https://docs.pytorch.org/vision/master/models/generated/torchvision.models.mobilenet_v3_small.html) was considered as a generic classification-feature alternative; it was **not experimentally benchmarked**. No model was trained or fine-tuned here.

Checkpoint SHA256:`cf55163d78fc44c62c82f85ab62d39f10438679b5abe8c698ae08cfa84aa6e18`. Architecture SHA256:`c7c1c29187d6330f859c91da229271531920464c7011aec13842a086b2263cae`. The separate checkpoint is about8.9MiB and remains under ignored `outputs/models/m9/`. Classifier shape4101 is loaded strictly; inference returns512-dimensional features, not those dataset identities.

RGB body crops remove the top18%of the detection, resize128×256, apply ImageNet channel normalization and L2-normalize the network output. This is body appearance, not face identification. Eligible crops require adequate dimensions and foreground quality. Embeddings are sampled every third source frame, pooled over a complete segment, and retained with exact source frame/raw track keys. Future segment evidence is disclosed. Cache keys include original video hash, M8 snapshot hash, checkpoint hash and feature version; file hashes validate both cached features and embedding arrays. No identical neural inference is rerun during iterative debugging. Serving never downloads a checkpoint or imports the neural encoder.

Torso evidence retains dark/achromatic pixels rather than requiring saturation>45. It excludes estimated local grass by LAB contrast, preserves LAB quantiles, HSV histograms, brightness, sharpness, foreground fractions and actual selected pixels. `team_reference/` contains source crops, masks and per-pixel LAB/HSV/BGR NPZ. The torso/embedding representation is a prototype; tiny/blurred crops remain a major limitation. No separate learned shorts/socks model was implemented.

### Identity assignment

Each candidate exposes deep appearance similarity/quality, kit similarity, team compatibility, elapsed time, metric displacement, required average speed, movement allowance, camera uncertainty, trajectory residual and evidence frames. All gated candidates are saved; the inspector shows the best five and the accepted-event explanation.

Physical reachability is a conservative search gate:

`distance ≤ 12m/s × elapsed + 3m slack + 2 × combined camera-uncertainty proxy`.

These are explicit configurable prototype thresholds, not exact physiological limits. Current observations are not moved to satisfy this gate. Reliable geometry can reject an80m/short-gap identity match. If either endpoint's metric uncertainty exceeds3m, metric evidence is downweighted; a weak-metric reconnect also requires stronger appearance, multiple crop samples, bounded uncertainty and compatible search range. Unknown geometry permits only bounded short image continuity, not a long appearance-only identity jump. Velocity is a bounded internal search cue; it never becomes a canonical off-camera coordinate. Team disagreement can hard-gate a match only with substantial aggregate evidence: tiny evidence is shrunk toward uncertainty. Model team errors can nevertheless cause false rejections.

Default ReID appearance gate0.82, candidate score threshold0.78 and **global alternative-assignment margin0.06** remain explicit. Gaps beyond3s receive exponential score decay with3s time constant; search ends at8s. A long high-colour-similarity candidate is not automatically the same player. Simultaneously visible tracks cannot join the same identity.

Hungarian matching solves the complete eligible predecessor/successor graph with unmatched dummy options. An edge is accepted only when removing it worsens the best global assignment by the required margin. This avoids treating sequential reappearances as simultaneous competitors and avoids greedy duplicates. Accepted edges form time-ordered, nonoverlapping paths. Multiple candidates remain available on ambiguity. Scores are not calibrated identity probabilities.

Offline resolution uses complete track-fragment appearance and the global graph, then assigns the accepted identity back to the fragment's earlier observations. Accepted events retain disappearance, reappearance, resolved-at frame and evidence-frame lineage. This is recorded-video smoothing, not a causal live-tracking claim or a separate learned backward ReID network. Hard camera epochs cannot connect.

### Persistent team and role intelligence

Per-video multi-frame segment aggregates learn dominant LAB/brightness populations. Two dominant populations are anonymous TEAM_A/TEAM_B; additional appearance modes can attach to a compatible team. Minority kits remain OTHER, or OFFICIAL hypotheses when isolation, colour separation and central-field context support that interpretation. Official/other is not a third soccer team. These heuristics cannot establish referee identity or guaranteed goalkeeper/team membership.

Each entity accumulates five-way normalized evidence for TEAM_A,TEAM_B,OFFICIAL,OTHER,UNKNOWN, with independent samples separated by≥0.12s. At least3strong samples and sufficient support establish a label. Weak/unavailable current crops preserve the established label and expose `PERSISTENT_TEAM_HISTORY` separately from current visual evidence. Prototype adaptation is conservative: high-quality consistent evidence only, bounded small updates, with final centres/update count saved. Actual bounded prototype updates:56test_match,28team_goal,225Southampton.

A team switch needs sustained contradictory observations:≥8independent samples across≥0.8s within a2s window, crop quality≥0.45 and evidence strength≥0.6. Candidates, rejections and accepted model-belief corrections are logged. An actual player is not assumed to change team. First-pass moderate contradictions caused too many switches; the final pass requires this stricter evidence. Wrong early assignments can still persist, which the reference review shows.

### Visibility and canonical observations

M8 footprint evidence distinguishes last observed area outside the current view (OFF_CAMERA) from a missing detection inside/with unknown view (OCCLUDED hypothesis). Epoch changes or a long unobserved gap become LOST. Ambiguous identity remains distinct from known team/position. The footprint check does not pretend to know where an unseen player has moved.

Every disappearance sets current position/confidence/provenance to unknown. Last observed supported position, its timestamp, last velocity and uncertainty are historical/search evidence only. `video_player_observations.jsonl` stores source VIDEO_CV, entity or null unresolved identity, raw track, timestamp, team/role confidence, visibility, x/y or null, position confidence/provenance and identity confidence. It is **not connected to production SoccerTrack analytics**. World-state validators prevent duplicate identities/tracks, mismatched clocks and coordinates when the current camera is unknown; the full evaluator also verifies every coordinate against its exact M8 observation.

## Fragmentation and identity consistency

|Source|Raw temporary tracks|Confirmed fragments|Unconfirmed fragments|Persistent hypotheses|Accepted reconnects|Ambiguous reconnects|Mean confirmed fragments/entity|
|---|---|---|---|---|---|---|---|
|test_match|151|99|52|98|1|1|1.010|
|team_goal|56|43|13|42|1|4|1.024|
|southampton|57|48|9|42|6|1|1.143|

Do **not** interpret raw151→98 as53successful test_match reconnects:52short fragments remain unconfirmed and only1link is accepted. The actual reductions are99→98,43→42 and48→42 confirmed paths. Broad fragmentation reduction is **not demonstrated** on the first two clips. M8's older entity totals84/40/47 used different provisional/unassigned rules; M9 totals98/42/42 are not directly an identity-accuracy comparison. M9 does not silently re-use those old assignments as truth.

|Source|Temporary tracks/min|Reassociation target attempts|Rejected target decisions|Rejected candidate pairs|Duplicate simultaneous identities|Impossible accepted metric links|
|---|---|---|---|---|---|---|
|test_match|401.52|87|85|1915|0|0|
|team_goal|168.00|35|30|279|0|0|
|southampton|155.30|35|28|374|0|0|

Attempt/ambiguous/rejected target counts differ from pair counts. Zero duplicate/impossible accepted links are measured internal invariants, **not identity accuracy**. Independent identity-switch counts, identity accuracy and team accuracy are unavailable. Unseen identity mistakes can exist despite these checks. Raw detector errors and short-lived false person proposals remain preserved.

## White/unknown problem and team consistency

Same raw-observation denominators are used before/after. Known below includes explicit official/other role labels; it does not mean every observation belongs confidently to one of the two outfield teams.

|Source|Team/role-known M8→M9|Unknown observations M8→M9|Reduction in unknown|M9outfield A/B|M9official|M9other|
|---|---|---|---|---|---|---|
|test_match|3725→4001|1383→1107|19.96%|3587|414|0|
|team_goal|4689→5104|826→411|50.24%|4770|0|334|
|southampton|7349→7456|1376→1269|7.78%|6751|705|0|

|Source|Established→Unknown M8→M9|One-observation unknown flicker M8→M9|All one-observation unknown runs M8→M9|All2–5unknown runs M8→M9|Switch candidates/rejected/accepted M9|
|---|---|---|---|---|---|
|test_match|0→0|0→0|42→42|44→33|1/0/0|
|team_goal|0→0|0→0|13→12|14→9|5/1/2|
|southampton|0→0|0→0|7→6|8→8|3/0/1|

M8 already had sticky belief for established tracks: established→unknown and true interior one-frame flickers were already zero, and remain zero. Do not claim M9 fixed a nonzero baseline flicker rate. All unknown runs include startup/unconfirmed evidence; they count observed samples within the grouped track/entity, not unobserved elapsed video time. Grouping changes from M8 temporary tracks to M9 persistent entities. Final accepted label changes are0/2/1; some candidates remain pending at clip end. Unknown reduction is useful but **does not establish better classification accuracy**.

## Pretrained appearance evaluation

Proxy positives are separated observations of the same raw temporary track; proxy negatives are co-visible, spatially separated tracks. They are not roster/identity ground truth and can contain tracker/detector errors. Histogram and deep features overlap across teammates, so neither can establish identity alone.

|Source|OSNet same-track median|OSNet co-visible negative median/P90|Histogram same-track median|Histogram negative P90|
|---|---|---|---|---|
|test_match|0.858|0.532/0.786|0.940|0.846|
|team_goal|0.847|0.557/0.797|0.901|0.848|
|southampton|0.872|0.637/0.790|0.907|0.727|

These distributions support using the features with motion/global constraints; they are not comparative ReID benchmark accuracy or proof that OSNet is optimal for small football crops.

## Qualitative reference review — REFERENCE-EVALUATED

Versioned annotations: [reference cases](evidence/cv_m9_reference.json).26source-context cases (20candidate/continuity cases plus3measured framing-change and3long-gap cases) and7team/role examples were visually inspected. This is an assistant source-image development-set review, **not an independent human-labelled or held-out identity benchmark**. Source hashes, frame/track IDs, image paths, decisions and APPEARS_SAME/APPEARS_DIFFERENT/INDETERMINATE labels are explicit. Uncertainty was retained rather than inventing ground truth.

Examples:

- test_match516→521,TRACK_00106→TRACK_00144:short-gap yellow/green person appears continuous; accepted model link. Early similar red teammates and overlap remain rejected/ambiguous;18→33 is a potential missed continuity with insufficient reliable geometry.
- team_goal233→256,TRACK_00013→TRACK_00037:model accepts a boundary reconnect, but both endpoint crops are too truncated for this reference inspection to establish identity. Similar jerseys, crossing/context changes and boundary crops remain failures/uncertainties.
- Southampton411→417 and568→572:red-player short-gap continuity appears plausible; the latter changes body shape while falling/rising.2.27/2.67s reappearances among similar kits remain visually indeterminate. Longer6+s similar-kit examples stay unresolved rather than appearance-only merges.
- Framing/scale-change windows258→270test,357→369goal,183→195Southampton show plausible continuous people. These are selected by the measured camera scale diagnostic, not independent optical-zoom labels.
- test frame100 TRACK_00017 preserves a red-team belief despite conflicting current crop; TRACK_00043 remains genuinely unestablished. Goal frame100 TRACK_00012 is a visually apparent wrong blue/red population assignment. Southampton dark/red mixing persists in representative samples, while the bright official hypothesis is plausible. Its existing team-quality warning is retained.

Among8model-accepted reconnect cases,4appear qualitatively plausible and4remain indeterminate in the small endpoint reference review. This is **not4/8identity accuracy**: no definite independent identity labels or continuous full-match reference exists. Coverage/consistency alone does not validate the other four. Annotated failures are retained for further work, not hidden by summary metrics.

Optional Tesseract5.5.2 number recognition attempted76test_match and10Southampton sufficiently large/sharp torso crops;team_goal had0eligible crops. No numeric candidate passed the threshold, no jersey number was established and none influenced identity. Rear-view identity is not automatically established; the evidence accumulator requires repeated high-quality support and is tested separately. Crop/attempt records are in each final `jersey_ocr/`; no OCR dependency is required to serve snapshots. A learned rear-view/jersey detector is **NOT IMPLEMENTED**.

## Runtime and cache

|Source|First feature extraction(s)|First OSNet embedding(s)|Current segment build(s)|Current team model/classification(s)|ReID candidate(s)|Hungarian/global margin(s)|World construction(s)|Cached replay wall(s)|
|---|---|---|---|---|---|---|---|---|
|test_match|1.850|14.594|0.035|1.477|0.055|0.000234|5.581|19.292|
|team_goal|0.895|21.855|0.039|1.174|0.014|0.000189|4.459|7.534|
|southampton|1.855|30.999|0.058|1.142|0.011|0.000233|6.031|12.211|

First extraction timings are saved from the original non-cached run; final replays hit validated caches and perform zero additional embedding inference. Wall time also includes source/hash/cache validation, JSON I/O and optional OCR. Team belief/prototype updates occur inside world construction; the separate team phase measures offline population fitting/classification, not the entire forward belief cost. **Detection and original short-term tracking were reused from M8 and not rerun/timed as zero**. This is not a full fresh-detector benchmark or real-time claim. CPU PyTorch2.14.1/Torchvision0.29.1,4encoder threads; runs can contend with other local work.

## Player World Model Inspector and regression verification

Select a CV source at http://127.0.0.1:5173. Click a video person box or choose a Persistent player in the new inspector. Inspect independent confidence channels, current versus historical position, current visual versus persistent team belief, source crop/features, track/visibility timeline, five-way belief graph, candidate decomposition and accepted/delayed-resolution events. Timeline segments seek the source frame. All entity timelines are available on demand. Official/other have separate colours. The old M8 association panel is explicitly retained as comparison evidence; it is not the M9 authority.

- **608backend tests run:607passed,one expected optional-media skip.**49new M9 tests include global assignment/ambiguity, physical/uncertain movement, weak/sticky/switch team evidence, official population learning, cut/off-camera/no-history-position invariants, delayed resolution, OCR accumulation/bad OCR, duplicate prevention and checkpoint/input lineage rejection. Existing M8/SoccerTrack/Azure/Ask tests remain passing.
- **86frontend tests across15files passed;TypeScript and production build passed.** Inspector tests separate confidence channels, seek history and preserve older-snapshot compatibility.
- **All10real Chrome suites passed(1.9m)** on final M9 snapshots. All10Azure clips select/play/seek/range206/tactical review/grounded Local Ask200. All3CV sources play/seek/step/debug/switch sources,retain M8 camera controls and expose M9 identity evidence;CV football Ask503.
- Final live health200/10analytic sources;ready200/13usable sources;catalog200/all13available. All3CV metadata200(camera m8/player m9),16-byte video ranges206. Direct118575_clip view200;grounded Local Ask200/statusok/14evidence rows.
- Final Chrome overlay samples:23.94updates/sec test_match,25.13team_goal,29.87Southampton;zero missing samples,max observed lag1/1/0frames over~2.5s. Sampling jitter can slightly exceed nominal FPS; this is not generated/interpolated observation rate.
- Earlier browser selector failures came from the new inspector and additional note matching old broad locators; selectors now explicitly target the legacy panel/warning. Final complete run passes,without bypassing assertions.

Saved proof:`outputs/world_model/m9_verification/`,including each`evaluation.json`,`live_http.json`,`browser/*-m9.png`,`browser/player-intelligence-results.json`,`browser/azure-clips-results.json`,`browser/cv-smooth-results.json` and copied test logs. Machine-readable portable results:[M9results](evidence/cv_m9_results.json). Full candidate/segment/decision/event/timeline files are in each final snapshot. Model/cache/source hashes remain explicit.

## Commands and file map

From repository root,using the existing intact Python3.12runtime:

```sh
# Explicit OFFLINE download only; hash-checked,create-only. Never a startup step.
/tmp/tactiq-cv-m1/bin/python -m scripts.download_player_reid
# Example replay: choose a NEW output path; existing snapshots are never overwritten.
/tmp/tactiq-cv-m1/bin/python -m src.vision.world.player_replay \
 --recording test_match --baseline outputs/world_model/test_match_m8_release \
 --output /tmp/tactiq-m9-test-match-review \
 --checkpoint outputs/models/m9/osnet_x0_25_msmt17.pth
/tmp/tactiq-cv-m1/bin/python -m scripts.evaluate_player_world_m9 \
 --recording test_match --directory outputs/world_model/test_match_m9_ready \
 --output outputs/world_model/m9_verification/test_match
# Use the same command with team_goal or southampton and their own M8 baseline.
# --config accepts a validated PlayerWorldConfig JSON; --no-ocr disables optional OCR.
/tmp/tactiq-cv-m1/bin/python -m unittest discover -s tests
```

```sh
# Start backend. Existing .env keeps Azure configuration; no MATCHMIND names changed.
AZURE_TOKEN_CREDENTIALS=AzureCliCredential MATCHMIND_PROVIDER=local \
 MATCHMIND_CACHE=/tmp/matchmind-azure-startup-cache \
 TACTIQAI_BACKEND_PYTHON=/tmp/tactiq-cv-m1/bin/python scripts/start_backend.sh
# In another terminal,Node22:
cd frontend
npm run dev -- --host 127.0.0.1
# Verification:
npm test -- --run
npm run build
TACTIQAI_AZURE_BROWSER=1 TACTIQAI_AZURE_REQUIRE_ANALYSIS=1 \
 ./node_modules/.bin/playwright test --config playwright.m9.config.ts
```

`--baseline` makes the player stage reusable for other validated M8 recordings; the three current names are shortcuts. New video detection/camera processing remains a prerequisite. The fresh-video orchestrator still defaults to M7; M8→M9 is an explicit offline stage, not silently enabled for every future upload. Requirements already pin the encoder runtime in`requirements-cv.txt`;service-only startup does not require PyTorch. The temporary runtime was restored locally; no permanent replacement of the user's environment is implied.

Core files:`player_state.py`,`player_appearance.py`,`player_teams.py`,`player_identity.py`,`player_replay.py`,typed world schema and vendored OSNet/licence. UI:`PlayerWorldInspector.tsx`,CVReview integration and inspector tests. New M9 download/evaluation/reference/render scripts,backend tests,Chrome suite/config and two compatibility selectors. Registry selects the three validated final snapshots. Mandatory status/explorer and this report are updated;older uncommitted camera/playback work and user `copy.md` documents were preserved.

6,411,138,054logical bytes of redundant M9 pilot snapshots were removed after preserving reports/manifests/decisions. Baselines,original MP4s,final snapshots,model and appearance caches remain. **Keep local CV videos:**they are their playback sources. No Azure artifact/video upload,deletion,provider redesign,analytics change,Git commit or push occurred.

## Remaining work and architectural answer

**PLANNED:**independent labelled identity/team references,long-gap reconnection,better small-crop/official/goalkeeper classification,correcting strong-but-wrong initial team beliefs,calibrated uncertainty and broader fresh-video orchestration. Substantial fragmentation reduction on test_match/team_goal is not established. Ball perception,face recognition,names without rosters and new video-derived soccer analytics are **NOT IMPLEMENTED**.

When a player disappears,M9 keeps the anonymous identity/team evidence and last observed position/time/velocity,but current position becomes unknown. A new track is compared against inactive hypotheses using multi-frame body/kit appearance,team compatibility,elapsed time,uncertainty-aware metric reachability and trajectory cues. Global one-to-one assignment must beat competing assignments by a margin;otherwise candidates remain ambiguous or provisional. Accepted offline links can resolve earlier fragment frames and retain exact future-evidence provenance. Persistent team evidence survives weak crops and switches only under sustained strong contradiction. Identity,team and position keep separate confidence channels:an identity/team can remain known while position is unknown,and a current M8-localized footpoint can remain associated with an uncertain identity. Nothing fabricates off-camera coordinates.

Stopped after M9. No ball milestone was begun.
