# 118575 full match — Azure processing

Prepared 2026-11-11 from the existing SoccerTrack full-half pitch tracking, event and ball annotations. This is the existing deterministic analytics pipeline, not a new detector-training run. Previously processed short clips and M8–M18 sources are retained.

|Recording|Video time|Frames at 25 FPS|Tracked identities across half|Recorded goals|Recorded shot-category entries|Completed pass estimates|
|---|---|---:|---:|---:|---:|---:|
|118575_h1|47:54|71,850|22|3|10|301|
|118575_h2|48:57|73,425|29|0|11|297|

The combined footage is 96:51. Identities across a half include different appearances/substitutions; 29 does not mean 29 simultaneous players. Goals and shot-category annotations are separate categories. Passes are the existing tracking-derived estimates, not independently verified complete-match pass counts. Tactical/possession semantics are unchanged.

Both original videos remain in `soccertrack-v2/videos/118575/`. FFprobe through an authenticated local range proxy verified each video's 4096×1080 dimensions, 25 FPS and exact frame count. Container audio lasts 0.015/0.018 seconds longer than the video. Neither multi-gigabyte video was downloaded onto the Mac. Full raw GSR JSONs were not downloaded again: the existing compact full-half GSR cache was used, with its hash retained in analytics provenance. The three local BAS/ball inputs matched Azure byte for byte. No new independent tracking accuracy is claimed.

Outputs: `outputs/full_match_118575/h1` and `h2`; canonical manifests under `outputs/full_match_118575/bundles/matches/118575_h1` and `118575_h2`. Exports are 217,067,345 and 218,851,236 bytes including manifests, with 22 artifact files plus each manifest. Video files are excluded. Existing 100 MB per-file limits, schema validation and SHA-256 checks remain unchanged.

Media routing explicitly registers validated half metadata to `videos/<match>/<match>_panorama_1st_half.mp4` or `2nd_half.mp4`. Existing clip paths, private API authorization, ETag conditions, HEAD and ranged seeking are preserved. Arbitrary client blob paths are not accepted. The canonical player roster remains available for full halves, but short-clip boxes are never reused on the full-half clock. Temporary player labels now continue after Z as AA, AB and AC.

**Current follow-up:** full-half highlights are now enabled through bounded ETag-conditioned Azure ranges. The three recorded goals produce a real120.021333-second video with 33.96 seconds before each goal, verified in Chrome from either half. The workspace shows recorded score 0–3, complete player cards, observed heatmaps and pass endpoint maps. Current checks: 905 backend run / 904 pass / one skip, 116 frontend tests/build, three Chrome cases plus a focused click recheck; /ready 200/15. [Full follow-up report, source integrity and limits](match_experience_update.md).

Full-half player photo cards still require validated boxes; short-clip boxes are never reused on a full-half clock. Missing boxes remain unknown rather than guessed. Earlier verification counts below describe the initial full-match processing.

Reproduce analytics locally:

```sh
PYTHON=/Users/greatabhieyighan/Library/Caches/TactiqAI/backend-m17/bin/python
"$PYTHON" -m src.analytics.pipeline --match-id 118575 --scope half --half 1 \
  --player-source gsr --summary-only --commentary-provider template \
  --output outputs/full_match_118575/h1
"$PYTHON" -m src.analytics.pipeline --match-id 118575 --scope half --half 2 \
  --player-source gsr --summary-only --commentary-provider template \
  --output outputs/full_match_118575/h2
```

Use the existing backend startup script and frontend development server. Select each half separately; playback and timeline timestamps are local to that half, not a guessed continuous match clock. Existing `MATCHMIND_*` variables retain their names. Raw videos and generated artifacts remain excluded from Git.

## Azure publication and validation

Both canonical prefixes were published create-only in `tactiq-artifacts`: `matches/118575_h1/` and `matches/118575_h2/`. The first upload attempts failed before either commit manifest was visible. Retrying with 4-MiB upload blocks succeeded: 13 previously uploaded files per half were checked for matching hashes and nine remaining files were uploaded, followed by each manifest. Existing blobs were not overwritten. The SDK transfer setting changed; manifest/hash/schema/file-size requirements did not.

The existing `src.artifacts verify` workflow succeeded separately for both halves: 23 listed blobs each, zero unexpected blobs, fresh-download/hash/schema verification, real application readiness, view and grounded Ask. Completed-transfer results were 301 and 297. Only after successful store read-back were both halves added to `configs/serving/local.json` and the existing ignored `.env` `MATCHMIND_RECORDINGS` list. Source video paths remain in `soccertrack-v2`, while analytics are in `tactiq-artifacts`.

Verification commands:

```sh
AZURE_TOKEN_CREDENTIALS=AzureCliCredential "$PYTHON" -m src.artifacts verify \
  --recording 118575_h1 --cache outputs/full_match_118575/verify_cache
AZURE_TOKEN_CREDENTIALS=AzureCliCredential "$PYTHON" -m src.artifacts verify \
  --recording 118575_h2 --cache outputs/full_match_118575/verify_cache
cd frontend
./node_modules/.bin/playwright test --config=playwright.full-match.config.ts
```

The initial backend suite ran 889 tests: 888 passed and one expected skip. After adding transient-timeout regression coverage, the final full suite ran 892 tests: 891 passed and one expected skip. The focused artifact/media/highlight suite passed 67 tests; 113 frontend unit tests and the typed production build passed. Existing M18 work remains in place. No source video was deleted, retimed or re-uploaded. Raw GSR JSON was not downloaded again and no Azure compute job was launched: analytics and the application still run on the Mac.

## Final running application

The final `/ready` response is **200 / 15 usable recordings**: twelve Azure analytics recordings (the existing ten clips plus these two halves) and all three existing CV recordings. Both full-half views return 200 with their private API video URLs and correct frame/player counts. Ordinary Local Ask is grounded with 14 evidence rows on each half; ranged video requests return 206. Chrome verified both halves play, seek to 1,500 seconds, update the tactical pitch and answer 301/297 pass-count questions without page errors. The final full-half browser case passed in 8.1 seconds (9.1 seconds including runner setup). The existing team_goal browser regression also passed (5.6 seconds).

One restart exposed a transient local `TimeoutError` while reading the existing team_goal CV files. Direct complete source validation passed afterward. Startup now retries complete CV validation once for that specific error; corrupt schemas/hashes are not retried, and repeated timeouts remain unavailable. Regression fixtures cover all three paths. The final server restored team_goal, so adding the full match does not remove an older recording.

Refresh http://127.0.0.1:5173 and select **118575_h1** or **118575_h2**. No manual Azure configuration change is needed. Start the backend again with:

```sh
AZURE_TOKEN_CREDENTIALS=AzureCliCredential MATCHMIND_ARTIFACT_STORE=azure \
MATCHMIND_PROVIDER=local MATCHMIND_CACHE=outputs/cache/azure_m17 \
MATCHMIND_HIGHLIGHT_CACHE=outputs/highlights_m18 \
TACTIQAI_BACKEND_PYTHON=/Users/greatabhieyighan/Library/Caches/TactiqAI/backend-m17/bin/python \
scripts/start_backend.sh
```

Frontend: `npm --prefix frontend run dev`.

[Portable measured results](evidence/full_match_118575_results.json) include actual source hashes, summaries, video probes, both Azure verification responses and final API checks. Private credentials and sessions are excluded.
