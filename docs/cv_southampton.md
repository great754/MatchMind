# Southampton — current CV pipeline verification

Verified **2026-10-27**. **LOCALLY VERIFIED / EXPERIMENTAL.** The original `data/raw/southampton.mp4` was processed with fresh detector, semantic pitch and absolute camera inference, followed by person/team/world replay, persistent camera, M6 temporal/team belief and current M7 world-field reconstruction. It is registered as **southampton · CV Experimental**. No existing video outputs were substituted for this source.

## Measured results

| Metric | Result |
|---|---:|
| Original source |1280×720,29.97002997FPS,660frames,22.022s|
| Source size |4,720,872bytes|
| SHA-256 |`8c3d8f2ca442bd21fa90ce66e948b844c2cdebb8af02eed1a39fcab3f2ee49f6`|
| Person observations |8,725 candidates;9–19 per frame;mean13.22|
| Temporary tracks / anonymous entity hypotheses |57 /47|
| Accepted camera geometry |**509/660 —77.12%**|
| Current projected observations |**6,114/8,725 —70.07%**|
| Current position unknown |2,611:2,294 no camera +317 outside valid region|
| Longest accepted run |frames0–294;295frames;9.843s|
| Longest unknown gap |378–446;69frames;2.302s|
| Detected cuts / epochs |0 /one(0)|
| Measured browser overlay updates |**29.91/sec**;source29.97FPS|

The smoothness sample covers2.508s after warmup:76 distinct displayed observations,no missing overlay samples,maximum observed lag one frame and maximum update gap34.4ms. It is a local Chrome sample rather than a guarantee for every frame/device/debug setting. Camera and placement percentages are model-accepted coverage, **not independently measured accuracy**. Observations and anonymous hypotheses are not verified physical-player counts.

## Camera stages and stability

M5 baseline:480/660 camera frames(72.73%),5,766/8,725 projected(66.09%). M6:452/660(68.48%),5,386/8,725 projected(61.73%). M7:509/660(77.12%),6,114/8,725 projected(70.07%). Relative to M6,57 unknown frames recover and zero accepted frames are lost;the detector/image/identity denominator and M6 team evidence remain frozen.

M7 states:TEMPORAL_PREDICTED465,DIRECT1,RELOCALIZED12,PARTIAL_CONSTRAINED15,TEMPORAL_CORRECTED12,BIDIRECTIONAL_INTERPOLATED4,UNKNOWN151. No homography/player-coordinate interpolation or relaxed validation thresholds were used. Visible people remain inspectable when current pitch coordinates are unknown,including sampled midfield frame396.

On5,133 matched adjacent-entity pairs,M6→M7 displacement median0.183879→0.168170m,P900.487057→0.496204m,maximum3.471751→3.638431m;large-jump screening flags85→126. Stability is mixed and the increased tail/flag count is explicit. Independent camera,identity and physical-motion truth are unavailable.

## Important team-colour limitation

**Team A/B is unreliable on this recording.** Manual review shows red and dark kits being mixed by the existing chromatic torso model. The provisional profile populations are7/34,with LAB centres[96.5,128.5,145.0]and[245.0,101.0,216.5];these are not two verified outfield teams. The chroma-based extractor poorly represents this dark-kit case,while bright official clothing can influence clustering. Current assignments6,640A/709B/1,376unknown are recorded diagnostics,not accepted team accuracy.

This finding is prominently displayed through recording-specific `review_notes` in the CV page. No colours,names,sides or team ground truth were manually fabricated. Team/player observations are kept separate from static-field camera fitting. Correct dark/achromatic kit separation and official exclusion need further model work;that work is not claimed complete here.

## Actual verification

- Source/hash/index validation and typed world/canonical validation across all660frames.17,103 identity/image entity rows verified frozen relative to M6. Current/off-camera/unknown semantics and independent camera-origin proof passed.
- Manually inspected17 before/after frames(0,30,60,100,170,250,300,330,360,396,450,500,528,550,600,630,659),the original source contact sheet,and enlarged100/396/600. Sampled accepted overlays align plausibly with pitch markings;offsets remain. Unsupported midfield/end frames do not get a canonical overlay. This is sample inspection,not exhaustive independent accuracy evaluation.
- Backend:529tests ran,528passed/one expected optional-media skip. Frontend:79tests/13files,TypeScript and production build passed.
- **All eight Chrome suites passed (1.4m):** all ten Azure clips;existing test_match/team_goal;Southampton selection,playback,seeking,frame stepping,source dimensions/clock,visible-box/person inspection,team warning/debug,camera evidence,recording isolation and video range. Actual overlay cadence was measured for all three CV videos.
- Live `/health`200(ten structured analytics recordings),`/ready`200(**13 usable recordings**),`/recordings`200:ten Azure sources plus three local CV sources available. Southampton video206/16bytes,metadata/frame endpoints200;unknown frame396 has no current metric coordinates. CV football Ask503. Existing118575view200 and grounded Local Ask200 remain working.

Southampton is a processed **local CV source**,not an Azure artifact-ready football-analytics bundle. Its capability flags preserve that distinction:video and CV world model available;structured analytics/football Ask unavailable. No Azure object was uploaded/overwritten,and no models or extra video copies were downloaded. Keep the original local MP4 for playback.

## Pipeline integration and execution time

`scripts/process_cv_video.py` now defaults to the current M7 completion path for fresh sources,retaining M5/M6 snapshots and producing `current/`. `--camera-model m5` retains explicit baseline behavior;`--resume-temporal` validates the completed source-specific M5 run before continuing. This source began with the existing baseline runner and then used that validated resume path;its real reports and hashes are preserved.

Fresh processing plus temporal replay/validation took **806.528s (~13m27s)** of measured processing time. Fresh semantic inference and absolute fitting took509.372s;detector43.573s;M6 replay43.349s;M7 completion159.855s. This offline work is not frontend/server startup latency and is not rerun by playback. No real-time inference claim is made.

Current registry:`configs/serving/cv.json`→`outputs/world_model/southampton/current/`. Detailed results:`summary.json`;baseline:`summary_m5.json`;camera comparison/typed proof:`verification/current/{coverage.json,comparison.json}`;manual images:`verification/images/`;live proof:`verification/live_http.json`;Chrome results/screenshots/logs:`verification/browser/`. Original source and all source-specific intermediate stages remain intact.

## Use and reproduce

Refresh **http://127.0.0.1:5173** and choose **southampton · CV Experimental**. Camera motion debug,projected geometry,player inspection and team debug are available. The current backend already serves the recording;no Azure settings need changing. Keep `data/raw/southampton.mp4`.

A new fresh run uses a new unused output directory:

```sh
/tmp/tactiq-cv-m1/bin/python -X pycache_prefix=/tmp/tactiq-cv-pycache \
  -m scripts.process_cv_video --video data/raw/southampton.mp4 \
  --detector /tmp/tactiq-cv-models/yolox-s.onnx \
  --semantic-model /tmp/tactiq-cv-models/tvcalib-train59.pt \
  --semantic-sha256 ec551dc181692e1758c953e9bffde718664cebd19f39b674918f7a4588ca9066 \
  --output /tmp/tactiq-southampton-reproduction
```

Backend from repository root,existing Azure `.env` retained:

```sh
AZURE_TOKEN_CREDENTIALS=AzureCliCredential \
MATCHMIND_PROVIDER=local MATCHMIND_CACHE=/tmp/matchmind-azure-startup-cache \
TACTIQAI_BACKEND_PYTHON=/tmp/tactiq-cv-m1/bin/python scripts/start_backend.sh
```

Frontend Node22:`cd frontend && npm run dev`. Full live browser command from frontend:

```sh
TACTIQAI_AZURE_BROWSER=1 TACTIQAI_AZURE_REQUIRE_ANALYSIS=1 \
  ./node_modules/.bin/playwright test --config playwright.southampton.config.ts
```

No ball tracking,goal inference,OCR,names,roster matching or new CV football grounding was added. Existing test_match/team_goal and SoccerTrack artifacts/analytics are unchanged. Next work should address achromatic/dark-kit team separation and remaining camera/jitter errors,with independent evaluation.
