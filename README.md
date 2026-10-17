## TactiqAI tactical video

Render the SoccerTrack-v2 `118575` match with the original panoramic footage and a synchronized, translucent top-down pitch at the bottom center:

```bash
source .venv/bin/activate
python -m src.vision.visualize_pitch
```

The output is `outputs/match_118575_matchmind.mp4` (4096×1080, 25 FPS, four minutes). Blue and red dots distinguish the two teams. Team colors use synchronized GSR team labels, including goalkeepers. Goalkeepers have the same team color and a white ring with a GK label. When authoritative labels are unavailable, jersey clustering excludes persistent backfield goalkeeper candidates and assigns them to the defending side afterward. A `.teams.json` sidecar records assignments, and a `.jpg` contains the first rendered frame.

For a ten-second preview:

```bash
python -m src.vision.visualize_pitch --max-frames 250 --output outputs/match_118575_preview.mp4
```

Use `--play` for local playback while rendering, `--inset-width 0.25` to enlarge the map, or `--opacity 0.8` for a more opaque pitch. Press `q` to stop playback.

Tactical positions primarily come from GSR pitch coordinates, joined to MOT tracks through `118575_sync.json` and the synchronized half-video frame offset. The renderer uses `PitchCoordinateTransformer` only for missing GSR samples. Finite GSR positions outside the pitch are omitted from drawing rather than replaced with a guessed location. The renderer checks that the source video and zero-based MOT tracking cover the same frames and streams frames without loading the entire video into memory.

The basic render includes footage and team markers. Add `--annotations` as described below to include the synchronized action panel and acting player.

### Action overlay

```bash
python -m src.vision.visualize_pitch --annotations --output outputs/match_118575_ball_actions.mp4
```

This adds a 12-row action panel and the acting player's ID and foot highlight. An action row lights up for one second from its annotated frame; simultaneous actions can light up together. The four-minute clip contains 88 events across eight classes; all twelve classes are supported.

The official BAS and ball files use the clock of each released half-video. `data/soccertrack/mot/118575_sync.json` records the inferred alignment: the clip's first frame corresponds to first-half frame 6000 (one-based), approximately 4:00 into that video. This was determined by matching the 22 players' image foot positions across 21 checkpoints; mean assignment error is 14.24 pixels, with approximately one-frame timing precision. The mapping also connects MOT IDs to event actor IDs and uses authoritative team sides to correct goalkeeper assignments.

Ball coordinates have a center-circle origin and are translated by (+52.5, +34) meters for the inset. The supplied ball path is reconstructed from event anchors and interpolation, not detected in every video frame; no ball height is supplied, so the footage ring represents its ground projection. Unknown-status samples are omitted, and positions outside the calibrated pitch cannot be projected. `.events.json`, `.teams.json`, and `.sync.json` sidecars retain clip event times, player/team mappings, and synchronization provenance.

To reproduce alignment after downloading both official GSR half files:

```bash
python -m src.vision.align_clip
.venv/bin/python -m unittest discover -s tests -t . -v
```

The alignment tool streams the large GSR JSON files into compact arrays rather than loading the entire files into memory. Annotation source and clock convention: [SoccerTrack v2](https://huggingface.co/datasets/atomscott/soccertrack-v2) (CC BY 4.0).

### Motion, possession, transfer difficulty and shots

```bash
python -m src.analytics.pipeline
open outputs/analytics_118575_clip/report.html
.venv/bin/python -m unittest discover -s tests -t . -v
```

This exports per-player speeds and measured distance, ball speeds, per-frame ownership, completed/intercepted transfer candidates, transparent difficulty features, shot candidates and BAS comparison files. The offline report has a scrubber and clickable events to review against the source clip. Analytics use the synchronized GSR metric positions by default; `--player-source mot` enables the clip's existing image-to-pitch estimate instead.

For the first half, which contains Shot annotations absent from this four-minute clip:

```bash
python -m src.analytics.pipeline --scope half --summary-only
```

Read [the implementation and testing guide](docs/analytics.md) for the rules, units, configuration, output schemas and validation limits. The ball is event-interpolated: speeds are estimated 2D values, BAS matching is not independent accuracy, and the difficulty score is not a calibrated probability.


### GSR accuracy comparison and goalkeeper-aware rendering

```bash
python -m src.vision.visualize_pitch --annotations --output outputs/match_118575_gsr.mp4
```

Before rendering, this compares calibrated MOT foot positions with GSR positions at every available player/frame pair. For the current clip, all 132,000 detections use GSR. The mean image-to-pitch disagreement relative to GSR is **11.68 m**, median **10.74 m**, and P95 **24.96 m**, across **6,000 frames**, using **130,782 paired positions** (99.08% comparison coverage; the remaining image foot points were outside the calibration hull). This is a relative comparison, not an independent absolute accuracy measurement; GSR remains quantized and the clip offset is inferred.

The renderer first loads compact GSR arrays; if those are missing but raw GSR JSON exists, it streams the raw annotations. A per-sample fallback uses calibrated projection only when that actor's GSR coordinates are missing. Team sides come from the same GSR actor join; synchronized GSR labels remain available if position data is missing. Invalid mappings or inconsistent team labels fail explicitly.

Goalkeepers are identified from provider role metadata when available. In this clip they are MOT **20 / actor 476352 (red)** and MOT **21 / actor 476367 (blue)**. Their differently colored actual kits do not affect their team assignment. A fallback without authoritative labels uses median backfield depth to identify keeper candidates, excludes their kits from outfield color clustering, and associates them with the side their team defends. That positional fallback is a heuristic and can be ambiguous for unusual play.

Alongside the video, `.comparison.json` records global/per-player errors, data coverage, sources, and goalkeeper assignments; `.comparison.csv` records per-frame disagreements; `.positions.csv` records the exact coordinates and source used for each marker. Add `--max-frames 250` for a preview; the comparison still covers the entire clip.

```bash
.venv/bin/python -m unittest discover -s tests -t . -v
```

The tactical tests cover changing GSR array slots, actor-ID/frame-offset joins, missing samples/files, raw JSON fallback, conflicting labels, keeper colors, and outfield clustering that excludes keeper kits.

### AI Match Intelligence

The existing analytics command now also exports `events.json`, `rolling_context.json`, and `commentary.json`. Commentary defaults to `none`, so normal runs and tests make no external calls. The report retains its video, pitch, plots, tables and validation sections, and adds a synchronized commentary timeline, three local style switches, and optional speed/distance callouts.

Try the local preview first (explicitly labeled as deterministic, not AI):

```bash
.venv/bin/python -m src.analytics.pipeline --summary-only \
  --commentary-provider template --max-commentary-events 8 \
  --output outputs/analytics_118575_intelligence_demo
open outputs/analytics_118575_intelligence_demo/report.html
```

For Gemini, copy `.env.example` to `.env` and set `GEMINI_API_KEY` there, or set that environment variable in your shell. `.env` is ignored by Git. Environment values take precedence over `.env`. A missing key fails before analytics runs; credentials never enter prompts, exports or report HTML. No new dependencies are needed: the REST adapter uses the existing `requests`, `python-dotenv` and `pydantic` dependencies.

```bash
.venv/bin/python -m src.analytics.pipeline --summary-only \
  --commentary-provider gemini --max-commentary-events 4 --max-provider-calls 12 \
  --commentary-start 0 --commentary-end 120
```

Each selected moment can cost three calls, one per mode. `--max-provider-calls` caps actual attempted calls, including failures; cache hits do not count. `--gemini-model` changes the default `gemini-3.5-flash-lite` model. `--significance-threshold` filters events before selection. `--intelligence-config path.json` overrides speed/distance thresholds, persistence, cooldowns and other `IntelligenceConfig` fields; explicit CLI options override that file. For example:

```json
{
  "player_speed_kmh": 25,
  "ball_speed_kmh": 80,
  "distance_milestone_m": 100,
  "threshold_persistence_frames": 3,
  "commentary_cooldown_seconds": 8,
  "significance_threshold": 40
}
```

Gemini chooses and orders supported sentences; it cannot inject arbitrary prose into the report. A strict response validator rejects unknown sentence IDs or extra fields. This intentionally favors factual grounding over unrestricted narration. Cached, validated selections live in `<output>/commentary_cache/`; delete that folder to regenerate. Provider/model, prompt version, event facts, window context and mode all affect the cache hash.

```bash
.venv/bin/python -m unittest discover -s tests -t . -v
```

Read [the intelligence implementation and testing guide](docs/intelligence.md) for event semantics, context queries, provider extension and verification. Gemini is optional; the automated tests mock hosted calls. The generated HTML uses the source clip when available; half-scope reports have a tactical scrubber but no clip video. Existing uncertainty remains: interpolated 2D ball data, missing height, tracker identities, team/goalkeeper assignment, calibration, alignment, and heuristic ownership/transfer/shot detection.

### Measurement validity audit

```bash
.venv/bin/python -m src.vision.projection_evaluation --output outputs/validity_audit/projection
```

This reproduces the unchanged projection baseline and exports pair-level errors, time aggregates, region/landmark/triangle diagnostics, offset/orientation experiments, four actual-data plots, and before/after metrics. See [the audit findings](docs/validity_audit.md). GSR remains primary; no calibration improvement is claimed. Prefer `transfers.csv` for observable transfers, and see `measurement_definitions.json` for compatibility field semantics and 2D estimated ball-speed definitions.

### Tactical State Engine

```bash
.venv/bin/python -m src.analytics.pipeline --summary-only \
  --commentary-provider template --output outputs/analytics_118575_tactics
open outputs/analytics_118575_tactics/report.html
```

The deterministic 5 Hz layer adds goalkeeper-excluded team shape, persistent defensive bands, local and sustained closing-pressure evidence, players relative to the ball, conservative formation estimates, observable-transfer networks and strictly earlier tactical evidence for important events. Unknown states remain explicit. The report adds synchronized shape review, centroids, pressure/formation details, a transfer network at the current video time and clickable tactical events. Existing Gemini grounding and caching also accept these closed tactical facts.

Use `--tactics-config configs/tactics.default.json` for explicit thresholds, or `--no-tactics` to omit the layer. New outputs include `tactical_states.csv`, `tactical_events.json`, `transfer_network.json`, `tactical_summary.json`, plots and `pre_event_evidence.json`. Read [the tactical implementation guide](docs/tactics.md) for every formula, threshold, unit, quality gate, test and limitation. This remains offline tracking evidence: upstream centered smoothing and interpolated 2D ball positions do not support causal claims about intent, pressing traps or why a turnover occurred.

## Evaluation and Ask MatchMind

Run frozen-config evaluation over both available halves and open the measured results:

```bash
.venv/bin/python -m src.evaluation --manifest configs/evaluation.118575.json --output outputs/evaluation
open outputs/evaluation/evaluation_report.html
```

Analytics now accepts `--match-id`, `--scope`, `--half`, `--data-root`, and `--output`. Reports include offline Ask MatchMind, grounded event explanations and timestamp links. See [evaluation](docs/evaluation.md) for required files, download/run commands, threshold sensitivity and interpretation, and [Ask MatchMind](docs/ask_matchmind.md) for queries, closed tool schemas, grounding and future provider integration. Only one match is installed; two halves do not establish cross-match generalization. Shot consistency results are poor and reported openly; no accuracy improvement is claimed.

Evaluation diagnostics, controlled shot experiments, held-out preparation and the artifact API are documented in [the evaluation/Azure milestone](docs/milestone_evaluation_azure.md). Current verified suite: 283 discovered/passed, zero failed/skipped. Microsoft integration status and planned cloud roles: [Azure architecture](docs/azure_architecture.md).

The Microsoft/product milestone now includes provider selection, structured evidence and follow-ups, artifact stores, Fabric CSV exports and verified local API/Docker smoke checks. Start with [the current demo guide](docs/microsoft_demo.md) and [Azure preparation](deploy/azure/README.md). Microsoft REST is post-hardening live-verified per the user-supplied verifier result; frontend verification uses Local only. Agent Framework live execution, Azure deployment, Fabric upload and MCP execution remain unverified. See [TactiqAI Microsoft hardening](docs/microsoft_hardening.md).

## Match Intelligence frontend

The polished React/TypeScript/Vite analysis screen pairs the source video with a synchronized tactical pitch, possession and formation estimates, canonical event review, and grounded Ask → evidence → jump interactions. Start the API and Vite using the exact commands in [the frontend guide](docs/frontend.md). Design follows [apple-design.md](docs/apple-design.md).

Current verification: **283 Python tests**, **27 frontend tests**, production build and **5 Chrome browser checks**. Microsoft REST is **post-hardening live-verified per the supplied verifier result**, with SSL_CERT_FILE absent, certifi, grounded selection and no fallback. Frontend checks use Local only.

Terminal 1 (repository root; restart an older API to load the new view route):

```bash
MATCHMIND_MEDIA_ROOT=data/soccertrack/mot/clips MATCHMIND_PROVIDER=microsoft \
.venv/bin/python -m uvicorn src.serving.app:from_environment --factory --host 127.0.0.1 --port 8000
```

Terminal 2 (Node 22.22.2+, 24.15+ or 26+):

```bash
cd frontend
# Locked dependencies are prepared outside cloud storage automatically.
npm run dev
```

Open **http://127.0.0.1:5173**. Use `MATCHMIND_PROVIDER=local` for a network-free demo. Frontend checks: `npm test`, `npm run build`, `npm run test:browser` (installed Chrome and prepared artifacts/media required).

### Match setup and current possession

The first visit now prompts for team names/colors and player names/jerseys; labels persist locally per match. Current possession retains the last established controller through short flights/gaps and is separate from confirmed-transfer share. Formation is summarized over four-minute windows; pressure levels are separate from sustained-pressure events. English evidence and bounded multi-tool Foundry planning replace raw field summaries. See [rules, artifacts, API changes and limitations](docs/demo_improvements.md). Shot thresholds remain unchanged: this clip has zero BAS shot annotations and zero heuristic candidates.

## Possession clarification — 2026-10-15

**Overall possession** uses each team's confirmed completed passes divided by both teams' confirmed completed passes (16/29 and 13/29 here: 55.2% / 44.8%). Unconfirmed control does not enter this percentage. `get_possession_summary` and general possession questions use this pass-based proxy; the UI labels its basis explicitly.

**Current possession**, shown below the pitch, is the team/player currently attributed the ball. It uses established control and continuity during flight/short gaps, not the overall percentage. Current/now questions use `get_current_possession`. Unknown is shown only when attribution is unavailable. This clarification supersedes the earlier description of attributed time as the headline possession statistic; attribution remains available as an internal diagnostic.

Ask now accepts pass-count/total/comparison paraphrases, highest-speed and pressed/formation synonyms, and supported multi-tool combinations without preset wording. For this clip, completed passes are **16 vs 13, 29 total**. Latest verification: **297 backend tests, 28 frontend tests, production build and 6 local Chrome checks**. See [current query behavior](docs/ask_matchmind.md).

## Second recording: 128058

Now independently processed and registered. The missing MOT download was fixed by using explicit filenames instead of a multi-value --include argument. See [the exact download command and independent processing workflow](docs/second_recording.md). Run `.venv/bin/python -m scripts.prepare_recording --config configs/recordings/128058_clip.json` for preflight, then add `--run` after installing dependencies. Successful processing registers a separate recording; restart the API/reload React to select it. The 128058 clip is now available after restarting the API and reloading React.

## One-command short-clip batch — 2026-10-15

Run `.venv/bin/python -m scripts.prepare_clips --run` from the repository root. It prepares 117092, 117093, 118575, 118576, 118577, 118578, 128057, 128058, 132831 and 132877 sequentially. Already registered, validated clip bundles with existing media are skipped. Missing dependencies are downloaded by exact filename with huggingface_hub; videos come exclusively from mot/clips, never full-half panorama video paths. Full-half GSR/ball annotation data is still required for synchronization. Each newly completed clip is registered only after processing/validation succeeds. Failures are recorded in outputs/clip_batch_status.json and can be retried by rerunning; known ready clips are preserved.

118575_h1 and 118575_h2 were removed from the default serving registry (their disk artifacts are retained). Restart the API and refresh React to remove them from the visible selector and discover newly registered clips. Pending clip configurations stay disabled. The batch has not been executed on all ten recordings during this change; no unprocessed clip is claimed working. Preview/config creation and existing-bundle checks were run without downloads. Verification: 308 backend tests and 32 frontend tests passed, including mocked exact-file download/processing/skip/failure tests. Existing analytics thresholds are unchanged.

### Canonical Azure artifact workflow

Use `python -m src.artifacts build|upload|verify --recording 118575_clip` (select one command). [Exact setup, safe upload rules and live status](docs/azure_artifacts.md). Local build and 327 backend tests pass; the provisioned private container is reachable, but Blob data access returned AuthorizationPermissionMismatch before writes. Upload/read-back/application integration are not live-verified. No Container Apps deployment was performed.

### Azure playback and local video storage

With `MATCHMIND_ARTIFACT_STORE=azure` and `MATCHMIND_MEDIA_BLOB_CONTAINER=soccertrack-v2`, the app streams the private Azure clips and loads validated Azure analytics bundles. Local source MP4 downloads are not required for serving or grounded Ask. The ten local downloaded clips were removed after verifying Azure copies and post-deletion playback/readiness. Download clips again only when rerunning offline processing or real-video tests. Keep videos, .env, generated outputs and downloaded datasets out of Git.
