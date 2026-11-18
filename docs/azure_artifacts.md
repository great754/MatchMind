# TactiqAI canonical Azure artifacts

## Local CV pass sidecars — separate from Azure artifact readiness (2026-11-13)

The three existing short CV clips now have locally validated experimental pass sidecars and structured JSON/CSV exports. M14 results: **0 qualified completed-pass estimates; 5/1/2 possible transfers** for test_match/team_goal/southampton. Candidates remain excluded from completed-pass counts and generated review videos visibly require review. Goal/shot/attempt/accuracy values stay unknown. No SoccerTrack annotation, pass/possession calculation, GSR alignment or canonical bundle validation was changed.

The pass loader pins the source MP4, world stream, ball stream/model, event/evidence/config hashes and source frame clock. Missing/corrupt event sidecars do not crash the API or remove usable CV playback. `/recordings` exposes `experimental_events_available`/`assistant_available` separately: CV `artifact_ready:false` and canonical `analytics_available:false` remain intact. Ordinary CV `/ask` stays 503; the unified assistant reads the validated experimental source. These local events were not uploaded or silently treated as Azure-ready analytics.

Final Azure-backed health 200,ready 200/15, all three CV event endpoints 200;existing 118575_clip view/grounded Local Ask 200 and video 206.949 backend tests run / 948 pass / one expected skip; 122 frontend tests/build and five Chrome cases pass. No Azure write or environment-variable rename occurred. The uploaded 51:39 external video remains raw media; its long processing run was deferred in favor of the short clips. [Verified pass report](cv_pass_detection_v1.md) · [Portable evidence](evidence/cv_passes_v1_results.json).


## External video source upload (2026-11-13)

The raw Tottenham vs Arsenal video is uploaded at `soccertrack-v2/external_matches/vibuxer_z4ny0atm95s8/full_match.mp4`, **759,278,077 bytes**, content type `video/mp4`. This object is source media, not a canonical artifact bundle; it does not add a processed recording or make a new source analytics-ready. Existing `tactiq-artifacts` bundles, validation, recordings and media were not changed.

`scripts/stream_video_to_azure.py` uses CLI authentication and 8-MiB memory-bounded blocks, per-block MD5 checking, a source-stream SHA-256 in metadata, final size verification and create-only commit. It accepts `VIDEO_URL` in the same terminal, `.env` or `--url`; the signed source link is not committed. Completed-stream checkpoints under ignored `outputs/azure_video_uploads/` support finalization retries without repeating a complete transfer. Eight focused tests pass. The initial redundant-etag commit error was corrected before the successful retry. No full video was downloaded locally. The first and last 4,096 source/Azure bytes match; the source-stream SHA-256 matches blob metadata. [Guide, exact command and verification scope](stream_video_to_azure.md) · [Verified upload evidence](evidence/external_video_upload_2026_11_13.json).


## Player markers and shorter full-half videos (2026-11-11)

Real Azure 118575_h1 rendering now verifies an automatically marked Player D pass and fastest measured run, with 90/120 marked lead-in frames respectively. Full-half markers use original image-foot coordinates from the compact GSR source whose SHA-256 must match validated bundle metadata. They retain exact actor IDs and source offsets; missing/ambiguous/off-screen points suppress marks. No guessed body boxes, pitch reprojection or unrelated local artifact substitution. Output video hashes and overlay/source-aware cache keys remain enforced. Full-half portrait crops still need actual body boxes.

Requested goal buildup now yields 20 seconds before/five after (default 15/five), with an actual three-goal output of 75.021333 seconds. The bounded ETag range path below is preserved. The service remains ready with 15 usable sources. 920 backend tests run / 919 pass / one skip; 119 frontend tests/build and four Chrome cases pass. No Azure upload, account/container change, video deletion or new hosted model request occurred. [Current verified report](player_focus_update.md) · [Portable results](evidence/player_focus_results.json). Earlier longer-duration examples below describe previous defaults.


## Current full-half rendering and service state (2026-11-11)

The updated workspace and assistant are live-verified with **12 Azure analytics recordings plus three local CV reviews; /health 200, /ready 200 with 15 usable recordings**. Full118575 score is recorded 0–3. The three-goal compilation plays/seeks in Chrome and includes33.96seconds before each goal. Both registered half videos retain private Azure playback and byte-range support. [Current report and exact commands](match_experience_update.md).

Full-half highlight rendering now uses an ephemeral localhost gateway restricted to one registered blob and random URL token. Every Azure range is pinned to the original ETag; changed objects fail. The per-job source-read budget is512 MiB, and cancellation, source-clock and output-size/duration validation remain enforced. The verified render read 196,608,887 bytes from the 3,550,087,937-byte source, rather than downloading the full original as a local file. The gateway closes after rendering. It does not accept arbitrary account/container/blob paths from clients.

For this streaming path, highlight manifests record `source_sha256: null` with an ETag-bound `source_identity_sha256` and explicit `registered_azure_stream` provenance. The cross-field validator requires that identity evidence; it never claims a whole-source hash was measured. Cached/local sources still use their actual source SHA-256, and rendered outputs retain actual SHA-256 validation. **Canonical Azure artifact bundle manifests, hashes and schemas are unchanged and remain mandatory.** No unrelated local artifact fallback was added.

No Azure upload, source-video replacement, storage setting change or renamed MATCHMIND variable was needed for this update. Compute remains on the Mac; Azure supplies storage/media. Full-half player photos remain disabled without checked source boxes. Earlier milestone counts and initial rendering limits below are historical.


## Full match 118575 — active and verified (2026-11-11)

`118575_h1` and `118575_h2` now have validated canonical bundles in `tactiq-artifacts/matches/`, while private media streams from `soccertrack-v2/videos/118575/118575_panorama_1st_half.mp4` and `118575_panorama_2nd_half.mp4`. Paths are registered from validated half metadata, never arbitrary request strings. Existing clip paths, ETags, authorization and byte ranges are preserved. The source videos were not downloaded or uploaded again.

Both halves passed the existing real Azure verification workflow (23 blobs each, hashes/schemas, fresh download, application readiness/view/grounded Ask). Initial uploads failed before commit manifests; retries using4-MiB blocks checked existing hashes and uploaded only nine remaining files per half. No overwrite or validation relaxation. Both halves are in the local registry and existing ignored MATCHMIND_RECORDINGS setting. Final API readiness is15 and Chrome verified both streams/seeking/analysis; all earlier recordings remain available.

The existing compact full-half GSR cache was reused; local BAS/ball inputs matched the uploaded Azure bytes. Full-half generated highlights are enabled by the current bounded streaming renderer above; player photos still require validated source boxes. Original footage, timelines, tactical replay and grounded questions remain available. [Verified results, limits and exact startup command](full_match_118575.md).


## M18 — unified assistant and registered Azure media (verified 2026-11-11)

The current assistant reuses validated Azure artifacts and registered media. `/health`, `/ready`, `/recordings`, 118575 view and existing grounded Local Ask return 200; readiness has 13 usable sources (ten Azure analytics clips and three local CV recordings). All ten Azure source videos passed Chrome playback, seeking and ranged media checks. 118575_clip and 128058_clip report artifact-ready. No 118575_h1 entry is present in the current catalogue; no availability is claimed for it.

Real 118575 goal, sustained-run overlay, difficult/longest pass and recap videos were generated. Source bytes are explicitly cached for rendering/photos under `MATCHMIND_HIGHLIGHT_CACHE`; there is no fallback to unrelated local analytics. M18 uses `outputs/highlights_m18` to preserve existing M17 examples. Manifests, source clocks and output hashes remain checked. **132877_clip has 5,958 decoded video frames versus 6,000 expected analytics frames; highlight rendering refuses that source.** Original Azure playback and analytics remain available; no video or bundle was changed to hide the mismatch.

Player boxes require checked local sync/MOT sources, while sprint estimates require hash-matching metric GSR inputs. Missing optional raw inputs disable the affected photo/overlay/sprint capability; they do not make an unvalidated Azure bundle ready. Canonical actor metadata still comes from the validated bundle. Session-scoped manual identities live separately in `MATCHMIND_ASSISTANT_CACHE` and never rewrite Azure/M9 observations.

No Azure upload, account/container change, model training or environment-variable rename was required for M18. Keep local CV originals. See [M18 exact startup command, verified numbers and remaining scientific limits](unified_match_assistant_m18.md).


## Southampton local CV addition — verified 2026-10-27

All ten existing Azure clips passed live browser playback/seek/range/tactical/event-lead-in/grounded Local Ask checks in the full eight-suite run. Ready200/13available sources:ten Azure artifacts plus test_match,team_goal,southampton local CV reviews. Southampton remains `artifact_ready:false`,CV/video available,football analytics/Ask unavailable;its video 206,metadata/frame200,CV Ask 503. Existing118575view/Local Ask 200. No Azure writes,validation changes or renamed MATCHMIND variables. [Source results and commands](cv_southampton.md).

## Current M7 regression — 2026-10-27

**LOCALLY VERIFIED / EXPERIMENTAL CV.** All ten existing Azure recordings passed final Chrome selection/play/seek/range/tactical/event-lead-in/grounded Local Ask. Health200;ready 200/12;all ten Azure artifacts validated/ready plus two local CV-only sources.118575 view200/481events,range206,Local Ask 200/statusok/14evidence. CV Ask 503. No Azure writes,manifest/hash/schema loosening,unrelated local fallback or renamed MATCHMIND variables. Current CVregistry selects local *_m7_current snapshots. On this Mac,standard AZURE_TOKEN_CREDENTIALS=AzureCliCredential avoids a stalled managed-identity discovery probe;the working credential and existing Azure .env are retained. [Full M7 report and exact commands](cv_world_model_m7.md). Earlier milestone sections are historical.

## Historical M6 regression verification — 2026-10-26

**LOCALLY VERIFIED / EXPERIMENTAL CV.** Fresh final Chrome verified all ten Azure clips in34.9s:selection, playback, seek/range, tactical view, event lead-in and grounded Local Ask. Five full browser suites passed in45.1s;both M6CVreviews and final panel recheck passed. Health200/ten analytics;ready 200/12usable recordings;discovery ten Azure-ready plus two local CV-only sources. Fresh118575view200/481events,range206/16bytes,Local Ask 200/statusok/14evidence entries. Both CVAsk requests503. No experimental upload, Azure overwrite/infrastructure, local artifact fallback, loosened manifest/hash/schema validation or renamed MATCHMIND variable. CVregistry selects local *_m6_current snapshots. Source CVvideos remain local;SoccerTrack media streams Azure. Existing `.env`/start command unchanged. [Full M6 report and commands](cv_world_model_m6.md). Earlier milestone sections are historical.

## Historical Milestone 5 regression — 2026-10-25

**LOCALLY VERIFIED.** Existing Azure-backed startup succeeded; health 200/ten analytics recordings; ready 200/eleven usable recordings; discovery ten validated Azure recordings plus local CV-only test_match. All ten Azure clips passed Chrome selection/play/seek/range/tactical-view/event-lead-in/grounded Local Ask checks. Full four-suite Chrome run passed (reported1.3min); three affected CV suites passed again after final inspection formatting. Fresh 118575 view200, video 206/16bytes, Local Ask 200/statusok/14 evidence entries. CV Ask remains503. Evidence: `outputs/world_model/m5_verification/`.

The local CV registry selects `outputs/world_model/test_match_m5/`: camera137/541=25.32%, player placement1,208/5,108=23.65%. The30% target was not reached; independent metric accuracy is unavailable. This is a local experimental observation source; no CV bundle was published, no Azure object changed, no local fallback added, no manifest/hash/schema gate loosened and no MATCHMIND variable renamed. Original local test_match.mp4 remains required; the ten SoccerTrack videos stream Azure. Existing `.env` and backend command are unchanged. [Current M5 report and exact commands](cv_world_model_m5.md). Earlier regression sections are historical.

## Historical Milestone 4 regression verification — 2026-10-25

Fresh final Chrome checks passed for all ten Azure recordings in **37.8 s**: selection, metadata, play/advancing time, seeking, tactical views, existing event lead-in, 206 ranges and recording-scoped grounded Local Ask. The CV camera review (2.6 s) and new player/team memory review (1.4 s) also passed; **three tests passed in 42.6 s**. Evidence: `outputs/world_model/m4_verification/`. Live health/readiness/discovery returned 200; readiness counts eleven usable recordings (ten Azure analytics plus local CV-only test_match). One fresh 118575 Local Ask returned status ok with 14 evidence entries.

The local CV registry now selects `outputs/world_model/test_match_m4_verified/`. No Azure blobs, artifact validation, analytics thresholds, GSR, possession/grounding rules or existing MATCHMIND names changed. No experimental upload or video download occurred. Keep the original local test_match source for CV playback; SoccerTrack source videos remain streamed from Azure. The backend command below is unchanged. See [current player/team report](cv_world_model_m4.md). Earlier verification sections are historical.

## Historical Milestone 3 regression verification — 2026-10-24

Fresh final Chrome verification passed for **all ten Azure recordings in 32.8 s**: selection, video metadata, playback/time advances, seeking, tactical views, grouped event navigation with the existing lead-in, 206 ranges and recording-scoped grounded Local Ask. CV review also passed (2.7 s); two browser tests passed in 36.8 s. Evidence is in `outputs/world_model/m3_verification/`. Health is 200 with ten analytics recordings; readiness is 200 with eleven usable recordings including the separate local CV review. No Azure blobs, analytics thresholds, credentials, container settings or existing MATCHMIND variable names changed. No experimental uploads occurred.

`configs/serving/cv.json` now selects the local Milestone 3 camera review. It does not make experimental CV artifacts Azure-backed or grant football Ask readiness. Current CV region coverage is 6.1%, with honest Unknown states; see [current report](cv_world_model_m3.md). Keep the original local `test_match.mp4`; the ten Azure SoccerTrack source MP4s remain unnecessary locally. The backend start command below is unchanged. Earlier playback results below are historical.

## Historical Milestone 2 local playback verification — 2026-10-24

All ten existing Azure videos passed a **fresh Chrome run this turn**: selector, metadata, play/advancing time, seek to 20 seconds, no media error, tactical synchronization, grouped event lead-in, 206 range and recording-scoped grounded Local Ask. The test passed in **34.9 s**; results/screenshot are under `outputs/world_model/m2_verification/`. This supersedes historical playback evidence below.

The old running API advertised videos/returned HEAD 200 but failed actual range GET with 503. Fresh credential/blob reads and a fresh backend succeeded on the same blob/ETag. The suppressed internal exception is unavailable; no exact expired-token/SDK cause is asserted. The API was replaced with an intact runtime, safe exception diagnostics and 1 MiB initial/chunk media downloads. `scripts/start_backend.sh` checks runtime dependencies; React has Retry video. No video redownload or Azure write was needed.

The final API is ready with **ten validated Azure analytics recordings plus one separate local experimental CV review**. `/health` keeps its existing analytics count of ten; `/ready` reports eleven usable recordings. CV capabilities do not grant analytics/Ask readiness and experimental artifacts were not uploaded. Structured video URLs remain the same private API streaming route; no local fallback occurs.

Verified start command from the repository root (existing `.env` retains account/container/media/recording settings):

```sh
MATCHMIND_PROVIDER=local MATCHMIND_CACHE=/tmp/matchmind-azure-startup-cache \
TACTIQAI_BACKEND_PYTHON=/tmp/tactiq-cv-m1/bin/python scripts/start_backend.sh
```

The temporary runtime must remain intact; recreate an environment from the service/Azure requirements and set `TACTIQAI_BACKEND_PYTHON` if it is removed. `test_match.mp4` is a separate, explicitly local CV source and must remain for that review. The ten SoccerTrack source MP4s are still unnecessary for Azure playback and remain deleted. See [current CV implementation and limitations](cv_world_model_m3.md).


## Runtime intelligence: no new Azure artifacts — 2026-10-17

This milestone derives review insights at runtime from already validated canonical artifacts. The bundle schema, allowlist, hashes and all ten existing Azure bundles are unchanged. Cross-clip quality outputs are generated locally; no uploads or overwrites are proposed. Missing/corrupt bundle readiness semantics remain unchanged. See [verified milestone](intelligence_milestone.md).

## Local source video cleanup — verified 2026-10-17

Azure mode does not require local source MP4 downloads for playback, tactical review or grounded Ask. After confirming all ten Azure source videos existed with the same byte sizes and served valid MP4 ranges, the ten downloaded `data/soccertrack/mot/clips/*.mp4` files were deleted at the user's request. Azure copies and validated analytics bundles were preserved. Post-deletion verification passed: ten playable Azure videos, ten artifact-ready recordings, `/ready` ready with 10 recordings, and grounded Local Ask 200. Local ignored evidence: `outputs/local_video_cleanup_plan.json`, `outputs/local_video_cleanup_result.json`, and `outputs/local_video_cleanup_smoke.json`.

Source videos would be needed again only to rerun offline processing/real-video checks; download them again when doing that work. Generated rendered videos under outputs were retained. Videos, caches, outputs, credentials and new raw datasets are excluded from the GitHub publication; additional downloaded tracking/annotation paths are ignored. Existing historical tracked dataset files are unchanged.

## Frontend verification limitations resolved — 2026-10-17

The later frontend startup repair identified offloaded Desktop dependency files and repeated native file-provider events. Dependencies/optimizer cache now live outside cloud storage and the watcher polls; the normal npm commands prepare this automatically. Final verification passed: 35 frontend unit tests, TypeScript checking and production build (776 ms Vite bundling), plus actual Chrome connection in 581 ms with all ten recordings and working video. Vite startup was 1,099 ms. Earlier worker/build failures below are historical and superseded. The Azure backend, uploaded bundles and media configuration were not changed. See [frontend startup evidence](frontend.md).

## Ten clips fully Azure-ready — approved publication verified 2026-10-17

The user explicitly approved the eight prepared analytics bundles. All eight were published with create-only writes to `tactiqaistorage/tactiq-artifacts/matches/`: **176 canonical artifact files plus eight manifests, 167,490,828 bytes total**. No existing artifact blobs were overwritten; existing videos were not uploaded again. The earlier approval block is resolved.

Each newly published recording then passed the real `verify` workflow: a fresh Azure download, manifest/SHA256/schema validation, private-container check, no unexpected blobs, and real in-process application readiness/view/grounded Local Ask checks. Evidence: `outputs/azure_clip_publication_results.json` and `outputs/azure_clip_readback_results.json`; the exact approved manifests and hashes remain in `outputs/azure_clip_publication_plan.json` (ignored local evidence).

The backend was restarted on **127.0.0.1:8000** with the saved ten-recording configuration and existing private Azure media container `soccertrack-v2`. Live HTTP `/health`, `/ready`, and `/recordings` returned 200. `/ready` is `{"status":"ready","recordings":10}`; all ten catalog rows have status=available, artifact_ready=true and media_available=true. Every clip returned a tactical view/replay with 200, a valid MP4 byte range with 206, and a grounded Local Ask answer with 200/status=ok, match-specific conversation identity, source evidence and no fallback. Evidence: `outputs/azure_all_ten_analysis_verification.json`.

| Recording | Completed transfers | View / Ask / video range |
| --- | ---: | --- |
| 117092_clip | 28 | 200 / 200 / 206 |
| 117093_clip | 28 | 200 / 200 / 206 |
| 118575_clip | 29 | 200 / 200 / 206 |
| 118576_clip | 19 | 200 / 200 / 206 |
| 118577_clip | 18 | 200 / 200 / 206 |
| 118578_clip | 24 | 200 / 200 / 206 |
| 128057_clip | 16 | 200 / 200 / 206 |
| 128058_clip | 26 | 200 / 200 / 206 |
| 132831_clip | 16 | 200 / 200 / 206 |
| 132877_clip | 46 | 200 / 200 / 206 |

The verified Ask payload was `{"question":"How many passes were there altogether?","provider":"local"}`. Counts above are existing canonical completed observable transfers, not new detector runs or accuracy claims. No analytics, possession, GSR alignment or grounding semantics changed. No new external model call, role assignment or Container Apps deployment occurred.

All ten videos and analyses are now available; the former eight video-only entries no longer show analysis pending after refreshing discovery. The video-only state remains for future missing/unpublished analytics and does not grant query access to invalid bundles. No further configuration change is needed for this local setup: refresh the frontend tab. The known 50-test backend regression result and frontend unit/build limitations from the implementation stage remain recorded below.

Final post-publication Chrome integration passed in **1.7 minutes**, with `TACTIQAI_AZURE_REQUIRE_ANALYSIS=1`: all ten entries required status=available/artifact_ready=true; every clip selected, loaded its video, played past 0.2 seconds and sought to 20 seconds; all ten returned grounded Ask 200; no page errors. Saved browser results confirm artifact_ready=true, ask_status=200 and video error=null for every recording. Evidence: `frontend/test-results/azure-clips-results.json`; final full-analysis screenshot: `frontend/test-results/azure-clips.png` (visually inspected). This supersedes the earlier two-analysis/eight-video-only browser result.

## Private Azure video delivery

All ten source clips already exist under private `soccertrack-v2/mot/clips/<match_id>.mp4`. Optional `MATCHMIND_MEDIA_BLOB_CONTAINER=soccertrack-v2` uses the existing account URL and DefaultAzureCredential. `src/serving/media.py` streams clips through the authenticated API with HEAD, single HTTP byte ranges (including suffix/open-ended ranges), Content-Length/Content-Range/Accept-Ranges and ETag-pinned downloads. Credentials/SAS are never sent to the frontend. Videos remain separate from canonical analytics bundles. Explicit Azure media does not silently fall back to local disk.

Local ignored `.env` configures all ten IDs, the Azure media container, and `/tmp/matchmind-azure-startup-cache`; `.env.example` documents the settings. Existing variable names and provider/token settings are preserved. Video availability (`media_available`, `media_url`) is independent of validated analytics (`artifact_ready`). Video-only recordings can play without enabling tactical/Ask routes. All ten current recordings now have both video and validated analysis.

Backend media/artifact/review/recording regression suite: **50 passed**. Earlier live Chrome verification covered ten selectable/playable/seekable videos while eight analytics bundles were still pending. After the approved upload, all ten live HTTP analytics/Ask checks passed as recorded above. Frontend unit workers timed out; final type checking was killed with exit 137; production bundling stalled and was stopped. No final frontend unit-suite/type-check/production-build pass is claimed.

Run the live browser integration with backend 8000 and Vite 5173, requiring all ten analytics bundles to be ready:

```bash
cd frontend
TACTIQAI_AZURE_BROWSER=1 TACTIQAI_AZURE_REQUIRE_ANALYSIS=1 npx playwright test --config playwright.azure.config.ts
```

## Per-recording startup and discovery

`MATCHMIND_RECORDINGS` is the Azure probe list, not a promise that all entries are ready. `configs/recordings/*.json` describes known recordings; `MATCHMIND_REGISTRY` records local processing/registration. Neither local registry membership nor a cached snapshot establishes Azure readiness. Every configured ID must read its canonical `matches/<id>/bundle.json` on startup and pass the existing manifest, SHA256 and schema checks before entering MatchMindTools and the query routes. No local artifact fallback occurs.

`GET /recordings` returns the union of configured IDs and catalog/registry IDs, including:

- `known`, `configured`: discovery and active-store configuration, independently.
- `processed`: true for registry-registered or validated recordings, false for catalog recordings known to await processing, null when processing is unknown. This does not certify Azure availability.
- `artifact_store`, `artifact_available`: active store and whether its canonical bundle was found. Availability is null for unprobed or indeterminate failures, false for a missing manifest, and true when a manifest was downloaded, even if invalid.
- `artifact_ready`: true only after validation and loading succeed.
- `status`: `available`, `unavailable` (missing/unconfigured), `error` (download/manifest/hash/schema failure), or `not_processed` (catalog awaiting processing).
- `diagnostic`: fixed safe code, also logged per failed recording; SDK exceptions/credentials are not returned. Missing canonical manifests report `azure_bundle_missing`; validation failures identify manifest, artifact download/hash, or schema stages. A 404 for a manifest-listed artifact is an error, not an unpublished recording.

Known but unusable Azure analytics queries return 503; unknown IDs return 404. `/health` remains live with zero usable recordings, while `/ready` returns 503 when none loaded and 200 when at least one validated recording loaded. There is no separate required/default recording policy in this API; clients choose among available recordings. Optional missing/error recordings do not veto readiness. Snapshots/status are fixed for the lifetime of the process; restart after publishing a new recording. LocalArtifactStore validation and query failure behavior remain strict and unchanged.

Historical startup-isolation HTTP verification on port 8017, before ten-clip publication, returned:

| Request | Verified result |
| --- | --- |
| `/health` | 200, `ok`, 2 recordings |
| `/ready` | 200, `ready`, 2 recordings |
| `/recordings` | 200; both clips available/ready; 118575_h1 unavailable, artifact_available=false, artifact_ready=false; eight other locally processed clips unconfigured with Azure availability unknown |
| `/matches/118575_clip/view` | 200, typed recording review |
| `/matches/128058_clip/view` | 200, typed recording review |
| `/matches/118575_h1/view` | 503, azure_bundle_missing |
| `/matches/118575_clip/ask` | 200, Local, status=ok, fallback=false; 29 completed transfers, 16/13, with transfers.csv field evidence |

Ask payload: `{"question":"How many passes were there altogether?","provider":"local"}`. Full local HTTP response evidence is saved in ignored `outputs/azure_startup_verification.json`. This checks real Azure downloads and local HTTP serving, not hosted ingress or an external model.

## Build, upload, verify

Run from the repository root, after installing the normal application dependencies:

```bash
source .venv/bin/activate
python -m pip install -r requirements-azure.txt
export MATCHMIND_BLOB_ACCOUNT_URL=https://tactiqaistorage.blob.core.windows.net
export MATCHMIND_BLOB_CONTAINER=tactiq-artifacts
python -m src.artifacts build --recording 118575_clip
python -m src.artifacts upload --recording 118575_clip
python -m src.artifacts verify --recording 118575_clip
```

`build` selects a processed recording in `configs/serving/local.json`; override with `--registry`. It creates `outputs/serving_bundle/matches/<recording>/bundle.json` and canonical files; override the export root with `--output`. Legacy `python -m src.artifacts --registry ... --output ...` still exports every registered recording.

`upload` validates the local manifest, hashes and existing closed schemas before authentication/network access. Only manifest-listed names from the canonical allowlist are uploaded. Credentials, .env, source videos, raw datasets, caches, registry.json and unrelated files are excluded. Artifact symlinks, unknown names, duplicate manifest keys, missing required files and mismatched hashes are rejected. The existing architecture's prefix is retained: `matches/<recording>/bundle.json`, rather than introducing a second manifest format.

Uploads require a private container, preflight conflicting remote bytes, use create-only writes and publish bundle.json last. Identical bytes are skipped on retry. Conflicts fail rather than overwrite an active snapshot; use a separate configured container for a revised immutable snapshot. A failed first upload can leave unpublished canonical files; rerunning completes matching files safely. Do not run concurrent writers to the same prefix. Recordings have independent prefixes.

`verify` lists the selected prefix, checks required blobs, forces a fresh download through AzureBlobArtifactStore, validates the manifest/SHA256/JSON and closed application schemas, and exercises real in-process application `/ready`, recording `/view` and a Local grounded `/ask` pass-count request. It then retains a validated cache under `outputs/azure_artifact_cache/recordings/<recording>` (`--cache` overrides the root). Unexpected blobs are reported and never downloaded. This does not test hosted HTTP ingress, video delivery or an external model. No football statistics are recomputed.

## Required Azure permission

An authorized RBAC administrator can grant the current developer container-scoped **Storage Blob Data Contributor**. Management-plane Contributor/Owner alone does not provide Blob data access. These commands have NOT been executed here:

```bash
TACTIQ_STORAGE_SCOPE=$(az storage account show --name tactiqaistorage --resource-group rg-greatabhieyighan-4489 --query id -o tsv)
TACTIQ_PRINCIPAL_ID=$(az ad signed-in-user show --query id -o tsv)
az role assignment create --assignee-object-id "$TACTIQ_PRINCIPAL_ID" --assignee-principal-type User --role "Storage Blob Data Contributor" --scope "$TACTIQ_STORAGE_SCOPE/blobServices/default/containers/tactiq-artifacts"
```

If another administrator runs this, supply the developer's object ID, rather than that administrator's own ID. Allow RBAC propagation, then retry upload and verify. Existing deny assignments can also restrict effective access. No account keys, connection strings or SAS tokens are accepted by this workflow. Configuration uses DefaultAzureCredential, which successfully selected the local Azure CLI identity during this test. [Microsoft authorization guidance](https://learn.microsoft.com/en-us/azure/storage/blobs/authorize-access-azure-active-directory).

## Exact first bundle and hierarchy

Local build and manifest/hash/schema validation succeeded: **21 canonical files, 20,623,663 bytes**, plus bundle.json. The hierarchy below describes the first canonical bundle; live download of both clips is now verified above:

```text
tactiq-artifacts/
  matches/
    118575_clip/
      bundle.json
      commentary.json
      events.json
      formation_windows.json
      measurement_definitions.json
      motion.png
      passes.png
      player_summary.csv
      possession.csv
      possession_attribution.json
      pre_event_evidence.json
      pressure_levels.json
      report.html
      rolling_context.json
      summary.json
      tactical_events.json
      tactical_pressure.png
      tactical_shape.png
      tactical_states.json
      tactical_summary.json
      transfer_network.json
      transfers.csv
```

Optional review_frames.json is allowlisted but absent in this source bundle. PNG/HTML/CSV bytes are hash-checked; JSON and existing typed application structures use the project's validators. This does not invent new schemas for historical CSV/image/report formats.

Only after 118575_clip succeeds, run:

```bash
python -m src.artifacts build --recording 128058_clip
python -m src.artifacts upload --recording 128058_clip
python -m src.artifacts verify --recording 128058_clip
```

Its separate prefix is `matches/128058_clip/`; live Azure read and application validation succeeded on 2026-10-17.

To serve a successfully verified recording with the existing application:

```bash
export MATCHMIND_ARTIFACT_STORE=azure
export MATCHMIND_RECORDINGS='["118575_clip","128058_clip","132831_clip","117092_clip","117093_clip","118576_clip","118577_clip","118578_clip","128057_clip","132877_clip"]'
export MATCHMIND_MEDIA_BLOB_CONTAINER=soccertrack-v2
export MATCHMIND_CACHE=/tmp/matchmind-azure-startup-cache
export MATCHMIND_PROVIDER=local
/tmp/tactiqai-azure-verify/bin/python -m uvicorn src.serving.app:from_environment --factory --host 127.0.0.1 --port 8000 --no-access-log
```

Account URL/container exports above must remain set. The final live check used a fresh `/tmp/matchmind-azure-startup-cache`; an attempted restart using the Desktop `outputs/azure_runtime_cache` stalled in a local PNG read, so the command avoids that filesystem wait. The tested interpreter is `/tmp/tactiqai-azure-verify/bin/python`; use a project virtualenv with the same requirements when available. `MATCHMIND_RECORDINGS` explicitly overrides the older list in .env; optional missing IDs may remain configured. Set `MATCHMIND_PROVIDER=local` for the verified deterministic Ask path. Videos remain separately configured using the existing media architecture; none are uploaded by this command.

## Startup-fix regression verification (2026-10-17)

The final focused backend suite passed **95 tests**: `tests.test_artifact_azure_workflow`, `tests.test_recording_setup`, `tests.test_frontend_review`, `tests.test_providers_service`, and `tests.test_microsoft_hardening`. This includes the additional safe access-error diagnostic regression. Coverage includes two valid Azure recordings plus one missing, usable queries and grounded Ask, unavailable query rejection, zero-ready 503, stale-cache rejection, invalid manifests, hash mismatch, missing manifest-listed artifacts, matching-hash schema failure and strict local startup.

Full backend discovery completed 331 tests: 330 passed and one unrelated existing real-video test errored: `tests.test_analytics.PipelineTests.test_real_clip_alignment_and_actor_coverage`. OpenCV reported `moov atom not found` for `data/soccertrack/mot/clips/118575.mp4`; `load_recording` raised `Clip must be readable and match metadata FPS`. No video, alignment or analytics code was changed to mask this failure. The final access-error regression was added after that discovery run.

Frontend TypeScript checking passed during the elevated build retry. Frontend tests/build did not complete: the initial test run timed out starting workers, and the initial TypeScript launcher failed with ETIMEDOUT. A one-thread test retry and production bundling retry remained stalled after four minutes (the bundler had open lucide-react dependency files); they were stopped. No frontend test or production-build pass is claimed.

## Historical verification evidence (2026-10-15)

- Local canonical build: passed for 118575_clip.
- Focused artifact workflow suite: **19 passed**.
- Full backend discovery: **327 passed**, zero failures.
- Cases cover legacy/local regression, CLI, allowlist, prefix isolation, immutable retry/conflict behavior, manifest/hash/missing/schema failures, symlinks and mocked Azure download/cache/application requests.
- Real Azure: authentication and private-container metadata passed; manifest read denied with AuthorizationPermissionMismatch; upload wrote no files; read-back and application integration remain unverified.
- Tests used `/tmp/tactiqai-azure-verify/bin/python` with requirements.txt, requirements-service.txt and requirements-azure.txt because some project venv files were macOS cloud placeholders. Tested Azure SDKs: azure-identity 1.26.0, azure-storage-blob 12.31.0. Dependency ranges remain unchanged.

```bash
python -m unittest tests.test_artifact_azure_workflow -v
python -m unittest discover -s tests -t .
```

No frontend changes/tests, Docker rebuild, cloud deployment or external model invocation were part of this verification. Mocked Azure success does not establish live Azure success.


M8 verification (2026-11-01):all three CV sources now use validated separate M8 snapshots;all ten Azure clips retain playback and grounded Local Ask. Health/ready/catalog200,13usable sources,9Chrome suites passed. CV football Ask stays unavailable. Keep original local CV videos;no Azure upload/configuration change. [Current metric camera report and startup command](cv_world_model_m8.md).

M14 local ball review (2026-11-02): optional `configs/serving/cv_ball.json` attaches validated VIDEO_CV ball sidecars to the three existing local CV recordings. Azure artifact bundles and SoccerTrack ball analytics are unchanged; keep original CV videos for local playback. Missing/corrupt optional ball data reports diagnostics while retaining usable M9 review. Live Azure view200, video range206 and grounded Local Ask 200/statusok/6evidence rows were reverified; ready 200/13. M14 has no verified metric ball positions and does not feed Ask. [Experimental ball report](cv_ball_tracking_m14.md).


M15 verification (2026-11-02): Azure bundles/media and SoccerTrack/Ask semantics remain unchanged. Optional local `configs/serving/cv_ball_m15.json` supplies a separately validated ball review selected explicitly with `?ball_pipeline=m15`;default staysM14. Missing/corrupt M15 returns503 for explicit requests while preserving usable M9/default review. Health/ready/catalog200,ready13;Azure 118575view200,video 206/1024bytes,grounded Local Ask 200/statusok/6evidence. All10Azure clips passed live Chrome playback/analysis regression. M15 does not achieve improved overall detection or independently validated pitch coordinates. Keep local CV videos;no upload or environment change required. [M15 experiment and commands](cv_ball_tracking_m15.md).

M16 verification (2026-11-03): optional local `configs/serving/cv_ball_m16.json` adds a separately validated `ball_pipeline=m16` review for the three existing local CV sources. M14 remains default; M15 remains queryable. M16 performs no cloud writes, local artifact fallback, or changes to Azure manifest/hash/schema validation, SoccerTrack analytics or Ask grounding. Final live health/ready/catalog return 200 with 13 usable recordings; Azure 118575 view returns 200, video range returns 206 / 1,024 bytes and grounded Local Ask returns 200 / status ok / six evidence rows. All ten Azure clips retain browser-verified playback and analysis availability. Keep local CV videos; no Azure or environment changes required. M16's accuracy targets are not achieved. [Full M16 evidence](cv_ball_tracking_m16.md).


M17 verification (2026-11-08): the Highlight Creator reads existing validated Azure analytics and authorized media. It performs no Azure writes or research-footage/model uploads. Media is conditionally downloaded by ETag into an ignored private, size-bounded source cache, hashed and checked against the registered recording rate/count/duration. Clip-relative/source/period clocks and their inherited synchronization uncertainty remain explicit; missing half offsets are not guessed. Real MP4s stay local and retain source/event provenance. Downloads are rights-disabled by default. Live health/ready/catalog200,ready13,all ten Azure playback regressions pass, and existing grounded Local Ask remains statusok/six evidence rows. Existing manifest/hash/schema validation and MATCHMIND variables are unchanged. [M17 report and persistent-runtime startup command](ai_highlight_creator_m17.md).


## M19 optional CV review (2026-11-15)

M19 is a local, versioned CV sidecar experiment, selected through **Ball review → M19 association · opt-in**. Azure canonical bundles, media settings and existing MATCHMIND_* names are unchanged. Missing/invalid optional M19 sources do not remove valid baseline recordings; explicit M19 queries return unavailable instead of silently using another source. The pass panel, profile and unified assistant use the same selected version and reset prior question/video context on changes. Keep the local test_match/team_goal/southampton originals; current CV review and independent annotation need them.

A model-blind independent source-review page is available at `/?recording=test_match&review=m19` (also team_goal/southampton). There are 60 sealed tasks and currently zero human labels; no accuracy claim is made. Labels are separate from model candidates and manual player identities, exported through `/cv/review/export`, and stored under ignored outputs. Do not commit labels, videos, checkpoints, caches or credentials. M19 performed no Azure uploads or long-video processing. Actual live health/readiness and all three clip playback ranges were verified; readiness has 15 usable recordings. [Measured M19 results, test evidence and reproduction commands](cv_tracking_passes_m19.md).


## M20 opt-in trajectory tracking (2026-11-18)

For an existing local CV clip, select **Ball review → M20 Trajectory Tracking · opt-in**. Current-frame crop inference, IMM image-motion predictions, covariance, competing/rejected candidates and loss/recovery diagnostics are synchronized to the original source video. M19 comparison, player focus and the unified assistant share the selected validated event version; predictions never become football facts. M14 remains the default, and existing M14/M16/M19 modes remain available.

M20 adds trusted local `cv_ball_m20.json` / `cv_events_m20.json` registries and a protected `/matches/{id}/cv/trajectory` endpoint. Invalid optional outputs return unavailable for explicit M20 requests without removing valid baseline readiness. Existing Azure artifact/media/authentication settings and MATCHMIND_* variable names are unchanged. Current real readiness is 200 / 15 usable recordings. The four saved human annotations and original videos/checkpoints/artifacts were preserved. Independent accuracy is pending human review; this engineering run needs no labels. No Azure upload, hosted LLM call or long-video CV run occurred. [M20 architecture, actual per-clip results, regressions and commands](cv_trajectory_tracking_m20.md).
