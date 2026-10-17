# Independent recording 128058_clip

Inspected 2026-10-15. **Processed, registered and locally verified as a second independent recording.** The currently installed clip is `data/soccertrack/mot/clips/128058.mp4`: readable, 5,981 frames, 25 FPS, 4096×1080, 239.24 seconds. The missing `mot/128058.txt` was downloaded successfully with an explicit filename argument. The browser-added `.txt.txt` / `.txt.mp4` aliases are supported when present; canonical filenames take precedence.

The live filename listing from `HfApi.list_repo_files('atomscott/soccertrack-v2', repo_type='dataset')` confirmed the following exact matching files. No 118575 calibration, synchronization, actors, ball or annotations are reused. Dataset IDs identify separate bundles; no proof of a shared underlying match was found or assumed. Source: [official SoccerTrack-v2 dataset](https://huggingface.co/datasets/atomscott/soccertrack-v2/tree/main).

| Required file (now installed) | Why |
| --- | --- |
| `mot/128058.txt` | Match the excerpt's player boxes to the full-half GSR clock. |
| `gsr/128058/128058_1st.json` | First-half authoritative pitch positions, actors, teams and image boxes. |
| `gsr/128058/128058_2nd.json` | Second-half counterpart; alignment must search both halves instead of guessing. |
| `ball/128058_1st_ball.npz` | First-half ball timeline and half length. |
| `ball/128058_2nd_ball.npz` | Second-half counterpart; both are used to discover the clip's alignment. |
| `bas/128058/128058_12_class_events.json` | Match-specific annotations, released-half clock/FPS and actor references. |
| `raw/128058/128058_keypoints.json` | This match's pitch projection fallback and GSR-versus-projection comparison. Not necessary for pure GSR coordinates, but required by this prepared full workflow. |
| `raw/128058/128058_tracker_box_metadata.xml` | Provider goalkeeper roles for outfield shape and formation. Missing roles must not be guessed from 118575. |

`mot/128058_sync.json`, `gsr/128058/1st_compact.npz` and `2nd_compact.npz` are **generated locally**, not downloads. The ball half, clip offset and MOT-to-GSR mapping are inferred for 128058.

Other available raw files (`homography.npy`, camera intrinsics, padding, player nodes, mapx/mapy and tracker_box_data.xml) are not read by the current GSR-first analytics or Delaunay keypoint fallback. The duplicated raw BAS file is unnecessary because the canonical BAS path above is used. Full-half panorama videos are unnecessary when the matching MOT excerpt is already installed. Do not download/reuse these from another ID.

## Correct explicit-filename download syntax

All dependencies are now installed; preflight reports `ready: true`, no missing files and no download command. For a fresh setup, use explicit filenames from the repository root:

```bash
source .venv/bin/activate
hf download atomscott/soccertrack-v2 --repo-type dataset \
  'mot/128058.txt' \
  'gsr/128058/128058_1st.json' \
  'gsr/128058/128058_2nd.json' \
  'ball/128058_1st_ball.npz' \
  'ball/128058_2nd_ball.npz' \
  'bas/128058/128058_12_class_events.json' \
  'raw/128058/128058_keypoints.json' \
  'raw/128058/128058_tracker_box_metadata.xml' \
  --local-dir data/soccertrack
```

The existing 128058 MP4 is deliberately excluded. Use your existing Hugging Face authentication/dataset access. The user downloaded the seven non-MOT dependencies. The original multi-value `--include` command was incorrect for the installed CLI: it consumed the first filename as a pattern, then ignored it when positional filenames were present. The helper now emits explicit filenames without `--include`. The remaining MOT file was downloaded and verified.

## Process, register and select

```bash
# Read-only preflight; prints a command for only files still missing.
.venv/bin/python -m scripts.prepare_recording --config configs/recordings/128058_clip.json

# After installing the dependencies:
.venv/bin/python -m scripts.prepare_recording --config configs/recordings/128058_clip.json --run
```

The runner validates the independent output directory, runs `src.vision.align_clip --match-id 128058 --data-root data/soccertrack`, then the unchanged-threshold GSR analytics/tactics/intelligence pipeline into `outputs/analytics_128058_tactics`. It uses deterministic template commentary and no model requests. Registration preserves existing entries and occurs only after processing succeeds and the bundle validates with matching match ID/scope. A different match's output directory cannot be overwritten. The default `118575` CLI remains compatible; video rendering now also accepts explicit `--match-id` and `--data-root`.

The full processing run succeeded and the serving registry now contains both clip IDs. Actual synchronization: first half, zero-based source offset 5950, mean assignment disagreement 17.61 pixels across checkpoints; 22 unique actor mappings, minimum vote agreement 18/21. This is inferred alignment, not surveyed ground truth. Generated analytics contain 26 completed observable transfers, 13 opposite-team receipt candidates, 3 unresolved transfers, one heuristic shot candidate and zero BAS shot references. No thresholds were tuned. The video stream is 239.24 seconds; its audio/container is 239.998 seconds, so browser duration slightly exceeds the analytic frame timeline.

Restart FastAPI after successful registration and reload React. `GET /recordings` discovers processed registry entries plus configured pending records. The selector enables only available recordings; 128058 is now available in the production registry. Pending configurations remain disabled in other registries. Each selection remounts the review so video time, events, selected player, answers and conversation state reset. API paths/report/media and identity storage are scoped to the selected recording. A response carrying another recording's ID is rejected. Existing 118575 identities retain their original storage key.

## Verification

Synthetic tests cover isolated bundles, registry preservation, pending status, output protection, filename aliases, match-ID validation, recording-scoped APIs/identities and selector behavior. Synthetic tests establish isolation; actual-bundle API/media/Ask smoke and a dedicated Chrome flow now establish local second-recording functionality. Actual-clip browser regression covers 118575 playback/Ask and the disabled pending-record option. Final verification: **304 backend tests, 32 frontend tests and production build passed**. The seven existing Local-only Chrome checks passed during preparation; a separate actual two-recording Chrome check now passes selection, playback, grounded Ask and switching back to 118575, with no console/page errors. API smoke independently verifies both clips: 29 vs 26 completed transfers and working MP4 byte ranges. Exact final results are recorded in current_pipeline_status.md.

No analytics thresholds or existing 118575 data/artifacts were changed. Existing Microsoft credentials/TLS behavior and live-verification status remain unchanged; no external model call or deployment occurred.

Repeat the real second-recording browser check with `TACTIQAI_BROWSER_REGISTRY=../configs/serving/browser-two-recordings.json npm --prefix frontend run test:browser -- --grep "actual second recording"`. The dedicated check is opt-in because it requires both processed bundles.

## One-command short-clip batch — 2026-10-15

Run `.venv/bin/python -m scripts.prepare_clips --run` from the repository root. It prepares 117092, 117093, 118575, 118576, 118577, 118578, 128057, 128058, 132831 and 132877 sequentially. Already registered, validated clip bundles with existing media are skipped. Missing dependencies are downloaded by exact filename with huggingface_hub; videos come exclusively from mot/clips, never full-half panorama video paths. Full-half GSR/ball annotation data is still required for synchronization. Each newly completed clip is registered only after processing/validation succeeds. Failures are recorded in outputs/clip_batch_status.json and can be retried by rerunning; known ready clips are preserved.

118575_h1 and 118575_h2 were removed from the default serving registry (their disk artifacts are retained). Restart the API and refresh React to remove them from the visible selector and discover newly registered clips. Pending clip configurations stay disabled. The batch has not been executed on all ten recordings during this change; no unprocessed clip is claimed working. Preview/config creation and existing-bundle checks were run without downloads. Verification: 308 backend tests and 32 frontend tests passed, including mocked exact-file download/processing/skip/failure tests. Existing analytics thresholds are unchanged.
