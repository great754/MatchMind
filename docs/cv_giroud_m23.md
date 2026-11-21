# Giroud — existing M23 pipeline run

The separate [40-question Giroud ball review](cv_giroud_review.md) is now ready, with automatic save/next and an isolated label store. [Open the review](http://127.0.0.1:5173/?review=m23&recording=giroud). Preparing it did not change the model outputs or create human labels; the run results below remain the frozen baseline.

Verified **2026-11-21**. All **450 frames** of the user-selected `data/raw/giroud.mp4` were processed. The clip is available in the frontend with source playback, anonymous players, experimental pitch positions, M23 ball evidence, a fresh same-source M21 comparison and grounded Local Ask.

**Ball performance is poor on this clip.** The model selected a ball in only **67/450 frames (14.9%)**, and visual inspection found several selections on advertising, crowd and player details. **Zero completed-pass estimates and zero transfer candidates** passed the unchanged rules. These outputs do not establish that the footage contains no passes. Goals, shots, tackles, pass attempts and pass accuracy remain unavailable from this detector. No accuracy improvement or model promotion is claimed.

[Open Giroud with M23 selected](http://127.0.0.1:5173/?recording=giroud&ball_pipeline=m23) · [Portable measured evidence](evidence/cv_giroud_m23_results.json) · [Inspected diagnostic frames](evidence/cv_giroud_m23_samples.jpg) · [Verified frontend](evidence/cv_giroud_m23_workspace.jpg).

## Source and processing

| Property | Verified value |
| --- | --- |
| Source size | 964,751 bytes |
| Duration / clock | 18.000 seconds / 25 fps / 450 frames |
| Dimensions | 640 × 360 |
| Source SHA-256 | `66fb0421464ba8a7a91a117a8625d4ea6f4870eb88e69ac0732fbaf3c7752e5c` |
| M9 world frames SHA-256 | `715ac2357d7ff536864d272693cc31cc0c4d4970479481710e105fdf9d6194db` |
| Active output | `outputs/m23/giroud_v1/analysis/giroud/full_m23/` |
| Active source/player world | `outputs/m23/giroud_v1/world/` |

Source previews were inspected before processing and show continuous football action. Processing ran locally on the Mac. A byte-identical source copy and generated outputs live in `~/Library/Caches/TactiqAI/m23/giroud_v1`, reached through the existing ignored `outputs/m23` symlink. Azure blobs were unchanged.

The source-specific pipeline ran fresh YOLOX people detection, TVCalib semantics and absolute pose fitting, the existing M7/M8 camera stages and M9 anonymous player/team replay. Ball processing ran fresh soccer-global/M20 crop measurements, M21 current-pixel and past-frame proposals, protected M23 causal memory/search/verification, existing M22 complementary inference and offline DAG association. M22 proposals can fill bounded primary-anchored gaps; they never feed the protected causal memory. Existing visual thresholds, manifest/hash/schema checks and M19 pass gates were retained. There was no neural training.

The separate `scripts/run_cv_source_m23.py` runner supports an explicitly selected new source without editing the three-source frozen evaluation protocol or reading human labels. It merges installed registry entries after strict validation. The original three clips, labels, baseline artifacts and development scores remain separate.

The complementary network uses the fixed existing **test_match-excluded M22 checkpoint**, SHA-256 `e2382fb45171f9660612db8f87c7be4c3c38124a57dd55d18c716cf06547fe0d`, chosen before this clip's inference. Generic primary SHA-256 is `85a76fe86dd8afe384648546b56a7a78580c7cb7b404fc595f97969322d502d5`; soccer detector SHA-256 is `638dc0baeed14f0cb4cb50f55454840a7614928405f5886381f70793439db4df`. Checkpoint paths, config hashes, source-pixel hashes and implementation hashes are retained in the perception report. Original-match training overlap has not been independently audited, and this source has no human reference labels. Precision, recall, F1 and identity accuracy remain undefined.

The older YOLOX and TVCalib files had disappeared from temporary storage. The same previously approved weights were restored into the persistent local model cache and their existing hashes verified. TVCalib: 488,774,369 bytes, SHA-256 `ec551dc181692e1758c953e9bffde718664cebd19f39b674918f7a4588ca9066`, from the [official inference instructions](https://github.com/MM4SPA/tvcalib#inference). YOLOX: 35,858,002 bytes, SHA-256 `c5c2d13e59ae883e6af3b45daea64af4833a4951c92d116ec270d9ddbe998063`. No weights are redistributed in Git.

## Measured outputs and limits

| Output | M23 result |
| --- | ---: |
| Model-selected current ball frames | 67 / 450 (14.9%) |
| Predicted-only ball frames | 14 |
| Ambiguous ball frames | 137 |
| Unknown ball frames | 232 |
| Longest directly selected ball run | 7 frames |
| Completed-pass estimates / transfer candidates | 0 / 0 |
| Stable lower-body contact runs | 1 |
| Person observations across frames | 5,763 |
| Anonymous persistent identity hypotheses | 36 |
| Reconnection hypotheses | 8 |
| Accepted camera frames | 442 / 450 (98.2%) |
| Person observations projected onto pitch | 5,542 / 5,763 (96.2%) |

The fresh M21 control selected a ball in 60 frames, with 12 predicted-only frames and zero qualified passes. Seven more M23 selections do not establish seven more correct detections. The inspected samples include obvious false selections at frames 29, 303, 386 and 432, as shown in the diagnostic contact sheet. This was a post-hoc visual spot-check; it created no reference annotations or accuracy scores.

Camera coverage measures internal acceptance, with **425 frames using propagated temporal camera geometry**. It does not establish metric accuracy. Person observations include potential officials, duplicates and detection errors; 36 identities are anonymous hypotheses, not a verified roster. The filename supplies no player identity. Player heatmaps and speeds inherit the existing geometry and identity limitations.

Ball pitch coordinates remain unavailable under the existing height/ground-contact rules. Predicted, ambiguous, reconstructed and historical positions remain separate from current model detections and cannot become completed-pass evidence by themselves. Green markers indicate selected neural outputs; false detections remain possible even when artifact validation succeeds.

| Recorded processing clock | Seconds |
| --- | ---: |
| Fresh people/semantics/M1–M7 stages | 508.806 |
| M8 camera replay | 44.990 |
| M9 player replay | 16.424 |
| Fresh ball inference loop | 244.949 |

Ball inference throughput was **1.84 processed frames/second**, with 1,286 additional primary search crops. Playback uses the original 25 fps source clock. These separate clocks exclude downloads and service startup; the ball loop excludes initial model setup and final association/export. They are not a real-time throughput claim.

## Frontend and serving verification

The direct link selects M23 explicitly; the ordinary application default remains M14. Giroud has a dedicated **unlabelled inference** display: state counts and diagnostics are shown, with no invented accuracy table or borrowed scores from another recording. Its comparison is a fresh M21 algorithm control on Giroud pixels, serialized in the strict M23 evidence container. The original clips retain their frozen M21 references and labelled development tables.

Verified on the normal Azure-backed backend, port 8000, and frontend, port 5173:

- `/health`: 200; 12 canonical analytics recordings.
- `/ready`: 200, `{"status":"ready","recordings":16}`.
- `/recordings`: 16 sources: 12 validated Azure analytics recordings plus four local CV recordings. Giroud is a local CV source and does not gain canonical Azure `artifact_ready` status.
- Giroud M23 metadata, world frame, intelligent detail and same-source reference: 200 with matching frame/model/ball revisions.
- Giroud source-video range: 206, 16 requested bytes.
- Grounded Local Ask, “How many passes were detected?”: 200, M23 revision attached, zero event entries. The response explicitly qualifies the incomplete experimental pass estimate.
- Existing `test_match`, `team_goal` and `southampton`: M23 metadata/detail/reference, source range and grounded Local Ask pass live checks.
- Azure `118575_clip`: view 200 with 481 events, grounded Local Ask 200 with 14 evidence items, video range 206 with 16 bytes.
- Chrome: Giroud selection, actual video playback, frame navigation, synchronized overlays/comparison, unlabelled warning, absent accuracy table and Local Ask all pass. No page errors were recorded.

Regression checks: **116 targeted backend tests passed**, **182 frontend tests passed**, final wording/frame checks **11 passed**, and the typed production build passed. The build retains its existing large-chunk advisory. The focused Giroud Chrome case passed again after final UI wording changes. Cached association/export reproduced **eight core artifact hashes exactly** across M23 and its fresh M21 control, with no new neural inference.

An optional reproduction process initially stalled reading an obsolete iCloud file descriptor; a bounded retry completed the eight-hash check. Only twelve completed intermediate JSONL files created by this run were losslessly archived, verifying decompressed SHA-256 before reclaiming 358,086,269 bytes. `intermediate_archives.json` records the retained archives. The active world, ball records, source video and earlier milestone data remain available.

## Commands and retained files

From the repository root, start the installed service:

```sh
scripts/start_backend_m23.sh
```

In another terminal:

```sh
npm --prefix frontend run dev
```

Open `http://127.0.0.1:5173/?recording=giroud&ball_pipeline=m23`. For a fresh full run, choose a new output directory and use the existing intact Python runtime and hash-checked model cache:

```sh
"$HOME/Library/Caches/TactiqAI/backend-m17/bin/python" \
  -X pycache_prefix=/tmp/tactiq-giroud-pycache \
  -m scripts.run_cv_source_m23 \
  --video data/raw/giroud.mp4 \
  --output outputs/m23/giroud_run_v2
```

The runner provides separate `--stage world`, `--stage ball` and `--stage install` steps; the default runs all three. Completed analysis directories are immutable. Restore an archived intermediate only if reproducing that earlier stage; serving needs the active validated world and sidecars.

Live checks and focused browser check:

```sh
"$HOME/Library/Caches/TactiqAI/backend-m17/bin/python" \
  -m scripts.verify_cv_m23_serving --recording giroud \
  --output /tmp/tactiq-giroud-live-recheck.json
frontend/node_modules/.bin/playwright test -c frontend/playwright.giroud.config.ts
```

Runtime evidence: `outputs/m23/giroud_v1/{perception/report.json,analysis/report.json,installation.json,live_verification.json,reproduction.json,intermediate_archives.json}`. Registries: `configs/serving/cv.json`, `configs/serving/cv_m23.json` and ignored `outputs/m23/runtime_cv_registry.json`.

No Azure configuration change is required. Keep the local CV cache for playback and review. Videos, weights, generated world/ball artifacts and annotation stores remain outside Git. Remaining work is stronger small-ball/generalization and scene/player false-positive rejection, followed by suitable independent references; M24 has not begun.
