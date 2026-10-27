# TactiqAI canonical Azure artifacts

## Current M6 regression verification — 2026-10-26

**LOCALLY VERIFIED / EXPERIMENTAL CV.** Fresh final Chrome verified all ten Azure clips in34.9s:selection, playback, seek/range, tactical view, event lead-in and grounded Local Ask. Five full browser suites passed in45.1s;both M6CVreviews and final panel recheck passed. Health200/ten analytics;ready200/12usable recordings;discovery ten Azure-ready plus two local CV-only sources. Fresh118575view200/481events,range206/16bytes,Local Ask200/statusok/14evidence entries. Both CVAsk requests503. No experimental upload, Azure overwrite/infrastructure, local artifact fallback, loosened manifest/hash/schema validation or renamed MATCHMIND variable. CVregistry selects local *_m6_current snapshots. Source CVvideos remain local;SoccerTrack media streams Azure. Existing `.env`/start command unchanged. [Full M6 report and commands](cv_world_model_m6.md). Earlier milestone sections are historical.

## Historical Milestone 5 regression — 2026-10-25

**LOCALLY VERIFIED.** Existing Azure-backed startup succeeded; health200/ten analytics recordings; ready200/eleven usable recordings; discovery ten validated Azure recordings plus local CV-only test_match. All ten Azure clips passed Chrome selection/play/seek/range/tactical-view/event-lead-in/grounded Local Ask checks. Full four-suite Chrome run passed (reported1.3min); three affected CV suites passed again after final inspection formatting. Fresh 118575 view200, video206/16bytes, Local Ask200/statusok/14 evidence entries. CV Ask remains503. Evidence: `outputs/world_model/m5_verification/`.

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
