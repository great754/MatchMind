# TactiqAI M23 — Intelligent Ball Reacquisition and Unified Football Tracking

Update 2026-11-21: the explicitly requested new `giroud.mp4` has a separate [existing-model inference run](cv_giroud_m23.md), frontend entry and unlabelled results. Ball quality is poor (67/450 selected frames; zero qualified passes). The original three-clip campaign and measured development results below are unchanged. No new neural training or M24 work occurred.

Development measurements frozen 2026-11-20; final service/test verification 2026-11-21 local time. M23 only; M24 was not started.

## Result and acceptance decision

The selected integration finds **107 of 145 visible balls**, with **18 false detections and 38 misses** on the 150 new reviews: **85.6% precision, 73.8% recall, F1 0.793**. Frozen M21 temporal/offline scores 95/14/50 and F1 0.748. M23 adds 12 correct detections and four false detections. It preserves **one of seven** reference pass matches, with **zero false completed passes in 18.012 exhaustively reviewed seconds**.

The ≥75% recall, ≥109 correct and ≥0.80 F1 targets were **not met**. Precision ≥85% was met. On the separate original sample, correct balls stay at 17/31 while false detections rise from nine to eleven; F1 regresses from 0.596 to 0.576. This is a measured development improvement on the new review set with a regression on the original sample, not a general breakthrough. **No promotion: M14 stays the ordinary default, and earlier modes remain selectable.**

No new training, neural architecture, weights download, paid compute, Azure upload or full-match CV was performed. Only test_match (541), team_goal (500) and southampton (660), total **1,701 frames**, were used for CV. Existing user labels and frozen earlier outputs were retained.

## One selected architecture

M21 current-pixel temporal primary → M20 IMM memory → bounded original-resolution crop inference → joint visual verification → optional M22 current-verified gap proposals → M21 DAG association with protected primary observations → unchanged M9 people and M19 pass gates.

`src/vision/intelligent23/` integrates existing components rather than introducing another tracker. M21 `advanced21/detection.py`, `temporal.py` and `camera.py` provide tiled current inference, past-frame proposals with current-image re-verification, blur/flow features and relative camera motion. Existing M20 `ball20/tracker.py` and `motion.py` supply IMM state, uncertainty, matching and appearance. Existing M16 generic/soccer weights remain fixed. The same source-excluded M22 three-frame ResNet-18/FPN/CenterNet checkpoints provide optional additional visual measurements. `advanced21/offline.py` retains its default arithmetic; optional finite node/edge adjustments leave frozen callers unchanged.

The final pipeline protects the primary path: M22 never replaces an accepted primary observation and never feeds a competing M22 object back into primary causal memory. Its actual dual-verified candidates can connect only a short gap bracketed by primary observations, passing camera-relative motion, size, appearance and existing repeated-visual/DAG requirements. No gap coordinate or interpolation is turned into a neural detection.

## Memory, uncertainty and camera compensation

Each causal frame records last directly observed position/frame/time/box, missed age, velocity/acceleration, normalized IMM mode probabilities, positive covariance, appearance template, bounded active/archived hypotheses, camera warp, planned regions, state and decision reason. The unchanged ordinary matcher expires at 0.35 seconds; search memory may survive up to 1.2 seconds, then expires. Four hypotheses and three trajectory regions are bounded. A single tentative neural hit may request one confirmation crop for up to 0.08 seconds, but cannot itself become an observed ball.

Uncertainty grows through misses. Motion-mode regions are retained only above the shared 0.05 mixture probability and are bounded/deduplicated. Accepted M8/M21 relative camera warps move image predictions and covariance; camera cuts, singular/incompatible transformations and epoch changes reset unsuitable memory. Image velocity is not physical football speed or height.

## Active visual search and verification

Regions are planned before current-frame ball candidates. Original-resolution generic detector crops use 96–384-pixel bounded regions with uncertainty and size margins; identical requests across ablations share inference. Up to two independent 192-pixel discovery regions rotate global tiles/player lower-body context when tracking is uncertain and periodically otherwise. Full M21 global discovery continues every frame. Nearby people guide discovery only; proximity does not prove possession.

The fixed neural threshold remains 0.35. Joint verification uses innovation against the motion mixture, appearance, scale, relative movement, current/previous patch agreement and competing evidence. A crop cannot veto an independently eligible primary view after NMS. A fresh blurred streak can survive an inconclusive line/flow check only with actual neural and compatible motion/appearance evidence. Persistent scene-locked line evidence is rejected; football colour and perfect circularity are not required. Neural scores and uncertainty are uncalibrated.

M22 uses both temporal and separately repeated actual-current-image inference, each ≥0.35. The final proposal cache preserves raw boxes/confidences from the hash-pinned global and freshly measured ROI pools, including real balls that an earlier merge suppressed merely because an unselected primary candidate overlapped. These remain proposals until two-sided primary association accepts them. Low-score proposals, including the 0.344 candidate near the known receipt, remain rejected. No threshold was changed to recover the pass.

## Offline association and provenance

Existing M21 DAG skip edges, camera epochs, confidence rewards, restart costs and repeated visual support remain the basis. Lightweight appearance/memory terms adjust association, not detector confidence. The short complementary connector retains each anchor's adjacent primary support; cropping it away caused a tested boundary bug and was corrected. Primary observations stay fixed. Camera-compensated two-sided interpolation requires valid same-epoch warps and a maximum 0.24-second missing interval; unsupported gaps remain unknown/ambiguous.

Every served frame separates OBSERVED, PREDICTED, AMBIGUOUS and UNKNOWN, inferred reconstruction, historical camera-compensated points, raw candidate provenance and causal search memory from the final offline path. Future evidence may choose an earlier **actual visual candidate**; it cannot invent an earlier observation. Strict serving checks source/world clocks, byte indices, frame/canonical/debug/record/checkpoint hashes, original coordinates, selected candidate confidence and model/config identity. Modified/corrupt optional M23 returns 503; it does not enter Ask or disable ordinary recordings. Listing a configured M23 mode is separate from validation; validation is performed on selection and its cached state is reported explicitly.

## Controlled comparisons — new 150 reviews

145 visible, five absent, 372 explicit negative regions. One-to-one IoU ≥0.25, exactly as frozen. No pooling with the original references.

| System | TP / FP / FN | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| m20_control | 41 / 3 / 104 | 93.2% | 28.3% | 0.434 |
| frozen M21 offline DAG | 49 / 0 / 96 | 100.0% | 33.8% | 0.505 |
| m21_temporal_offline | 95 / 14 / 50 | 87.2% | 65.5% | 0.748 |
| m22_reviewed | 64 / 27 / 81 | 70.3% | 44.1% | 0.542 |
| guided_search | 97 / 14 / 48 | 87.4% | 66.9% | 0.758 |
| guided_verified | 97 / 15 / 48 | 86.6% | 66.9% | 0.755 |
| full_m23 | 107 / 18 / 38 | 85.6% | 73.8% | 0.793 |

The four controlled variants are frozen M21 temporal/offline, guided search, guided+verification, and full M23. Frozen M21 was additionally reproduced byte-for-byte for all three ball artifacts, and the fresh source-pixel hashes match all 1,701 original reference frames. The fresh M21 control gives exactly the frozen aggregate result. A separate cached final rerun reproduced all **48 ball/canonical/diagnostic/event artifact hashes** across the twelve clip/variant combinations.

## Original references — separate sample

31 visible balls, 34 known ball frames and two uncertain-visibility frames. Unknown visibility is excluded, not a negative label. Original 60 task annotations are unchanged.

| System | TP / FP / FN | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| m20_control | 10 / 4 / 21 | 71.4% | 32.3% | 0.444 |
| m21_temporal_offline | 17 / 9 / 14 | 65.4% | 54.8% | 0.596 |
| m22_reviewed | 14 / 7 / 17 | 66.7% | 45.2% | 0.538 |
| full M23 | 17 / 11 / 14 | 60.7% | 54.8% | 0.576 |

## Per-clip ball results

New review (50 reviewed frames per clip; team_goal includes five absent frames):

| System | TP / FP / FN | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| test_match | 44 / 0 / 6 | 100.0% | 88.0% | 0.936 |
| team_goal | 29 / 10 / 16 | 74.4% | 64.4% | 0.690 |
| southampton | 34 / 8 / 16 | 81.0% | 68.0% | 0.739 |

Original references:

| System | TP / FP / FN | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| test_match | 9 / 2 / 2 | 81.8% | 81.8% | 0.818 |
| team_goal | 4 / 5 / 7 | 44.4% | 36.4% | 0.400 |
| southampton | 4 / 4 / 5 | 50.0% | 44.4% | 0.471 |

## team_goal: candidate generation versus selection

Full M23 improves new visible-ball matches from frozen M21's 26/45 to **29/45**, with ten false detections rather than eight. Retrained M22 alone had three correct new balls and zero of the original eleven; M23 finds four of those original eleven. The remaining errors do not justify a claim of cross-clip generalization.

Original visible-frame stage counts: `{'MATCHED': 4, 'SEARCH_REGION_MISSED_REFERENCE': 1, 'VISUAL_VERIFICATION_OR_THRESHOLD': 2, 'NO_MATCHING_VISUAL_CANDIDATE_INSIDE_SEARCH': 3, 'ASSOCIATION_OR_REPEAT_SUPPORT': 1}`. New visible-frame stage counts: `{'MATCHED': 29, 'VISUAL_VERIFICATION_OR_THRESHOLD': 8, 'ASSOCIATION_OR_REPEAT_SUPPORT': 5, 'NO_MATCHING_VISUAL_CANDIDATE_INSIDE_SEARCH': 3}`. The new sample has 29 matched, eight visual-threshold/verification failures, five association/repeat-support failures and three cases without a matching generated candidate inside search. The original top-five M22 availability and per-frame stage traces are below and in the private analysis report; these candidate-pool matches are not accepted detections.

| Original visible frame | Final stage | M22 top-five contains reference |
|---|---|---|
| 20 | MATCHED | True |
| 62 | MATCHED | True |
| 104 | SEARCH_REGION_MISSED_REFERENCE | True |
| 187 | VISUAL_VERIFICATION_OR_THRESHOLD | True |
| 229 | NO_MATCHING_VISUAL_CANDIDATE_INSIDE_SEARCH | True |
| 270 | NO_MATCHING_VISUAL_CANDIDATE_INSIDE_SEARCH | False |
| 312 | MATCHED | True |
| 354 | MATCHED | False |
| 395 | ASSOCIATION_OR_REPEAT_SUPPORT | True |
| 437 | NO_MATCHING_VISUAL_CANDIDATE_INSIDE_SEARCH | False |
| 479 | VISUAL_VERIFICATION_OR_THRESHOLD | False |


Boots at later goal frames and mistaken absent-frame detections remain visually visible in the source error sheets. Southampton still misses tiny/occluded balls, sometimes places regions away from the reference, and can select background/advertising elements. Human boxes were not edited to accommodate model extents.

## Corrections and rejected integrations

All experiment roots are retained. Initial v1 full integration was 94/14/51 (F1 0.743) and lost the known pass. Traces showed crop NMS erasing independent primary eligibility; preserving that support and enabling tentative/discovery crops fixed this stage. Unrestricted complementary feedback in v2 was **88/43/57**, F1 0.638: M22 could reinforce shirts/competing objects. Its apparent one-pass endpoint match followed a yellow shirt at receipt and was explicitly rejected after source-image inspection.

Protecting the primary path and restricting M22 to two-sided gaps raised the result to 105/18/40, but the known pass still failed. The original global/ROI pools contained real receiver-area candidates that the old merge/memory gates had suppressed. Restoring those unchanged visual proposals and retaining adjacent anchor support yields the final **107/18/38** and one genuine evidence-qualified endpoint match. These are systematic corrections within the requested architecture, not a threshold sweep or detector-family search. Remaining original-sample false-positive regression is disclosed; promotion is rejected.

## Passes, players and all seven references

M9 identities/teams and M19 release, flight, contact, coverage, separation and team thresholds are unchanged. A predicted or reconstructed ball never qualifies as a touch. At ±0.25-second one-to-one endpoint matching, M23 matches **1/7**, preserving M20/retrained-M22's known reference. The aspirational three matches were not reached. Six exhaustive windows total 18.012 seconds; no false completed pass appears there. The full clips contain three test_match and one team_goal qualified pass **estimates**, plus uncertain candidates; events outside reviewed windows are not independently validated. Goals, shots, tackles, interceptions, total attempts and pass accuracy remain unavailable from this pass detector.

| Recording | Human release → receipt | Ball near release / receipt | Observed flight frames | Guided frames | Receiver in M9 | Matched | Remaining model diagnostic |
|---|---|---|---:|---:|---|---|---|
| test_match | 64 → 84 | True / True | 17 | 8 | True | True | Evidence-qualified match |
| test_match | 442 → 455 | True / True | 11 | 3 | True | False | NO_EVENT_WITH_BOTH_ENDPOINTS_INSIDE_FIXED_MATCH_TOLERANCE; CHECK_FLIGHT_CONTINUITY_TEAM_AND_CONTACT_GATES |
| team_goal | 66 → 89 | True / False | 2 | 2 | False | False | NO_DIRECT_BALL_NEAR_HUMAN_RECEIPT; NO_UNAMBIGUOUS_M9_LOWER_BODY_CONTACT_NEAR_RECEIPT |
| team_goal | 181 → 216 | True / True | 17 | 4 | True | False | NO_UNAMBIGUOUS_M9_LOWER_BODY_CONTACT_NEAR_RECEIPT |
| team_goal | 150 → 163 | True / True | 11 | 3 | True | False | NO_EVENT_WITH_BOTH_ENDPOINTS_INSIDE_FIXED_MATCH_TOLERANCE; CHECK_FLIGHT_CONTINUITY_TEAM_AND_CONTACT_GATES |
| southampton | 367 → 404 | True / True | 32 | 0 | True | False | NO_EVENT_WITH_BOTH_ENDPOINTS_INSIDE_FIXED_MATCH_TOLERANCE; CHECK_FLIGHT_CONTINUITY_TEAM_AND_CONTACT_GATES |
| southampton | 434 → 442 | True / True | 7 | 1 | True | False | NO_UNAMBIGUOUS_M9_LOWER_BODY_CONTACT_NEAR_RECEIPT |


Sender/receiver evidence is in each private `pass_reference_traces.json`, including visible M9 segment/team support and source-frame proximity candidates. It is model evidence, not independently verified player/team truth. Sparse original identity pairs cannot establish true ball reacquisition identity, switch rate or fragmentation. Those accuracy fields stay null; navigation labels such as “correct on model reacquisition frame” mean pointwise box agreement only.

## Whole-clip states — coverage is not accuracy

| Recording | OBSERVED | PREDICTED | AMBIGUOUS | UNKNOWN | Track IDs (proxy) |
|---|---:|---:|---:|---:|---:|
| test_match | 456 | 24 | 61 | 0 | 3 |
| team_goal | 301 | 76 | 21 | 102 | 7 |
| southampton | 358 | 43 | 118 | 141 | 8 |


Track IDs/changes are engineering proxies. More observed frames are not automatically correct footballs. Historical trail points, search hypotheses and orange reconstructions remain separate from current green observations.

## Runtime and resource cost

Apple Silicon/MPS, existing Python/Torch runtime and checkpoints. Final proposals reuse the fresh v2 M21/ROI measurements and hash-pinned M22 global outputs. The final cache step re-decodes actual source pixels, checks their hashes, recomputes lightweight verification features and removes complementary feedback; it does **not** claim a new neural run. All four controlled analyses use the same measurement campaign.

| Recording | Shared neural campaign + cache derivation (s) | FPS | Primary NN (s) | Extra crop NN (s) | M22 ROI NN (s) | Extra crops | Peak RSS (GiB) | Offline four-way analysis (s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| test_match | 337.19 | 1.60 | 183.54 | 75.37 | 37.19 | 3669 | 0.99 | 25.29 |
| team_goal | 233.02 | 2.15 | 88.69 | 67.55 | 47.61 | 2532 | 1.18 | 13.34 |
| southampton | 535.47 | 1.23 | 287.47 | 104.31 | 69.84 | 3641 | 1.26 | 21.40 |


The campaign includes 9,842 extra generic crop views and 6,704 M22 temporal/current ROI views. Frozen M20 soccer measurements and M22 global neural outputs are cached; their original neural cost is excluded. The shared campaign also includes rejected full-feedback proposal crops, so these timings are not isolated optimized M23 deployment throughput. No real-time claim is made at the videos' 24/25 FPS. Extra generic+M22 ROI neural time divided by twelve net additional correct reviewed detections is a sparse engineering cost ratio, not independent recovery efficiency. Actual component times are in the evidence JSON. Peak RSS is the cumulative process peak at each clip, not isolated per-clip RAM or GPU peak. Per-stage standalone cold runtimes and true correct-trajectory fragmentation need a separate benchmark.

## Frontend and grounded Ask

Open [test_match](http://127.0.0.1:5173/?recording=test_match), [team_goal](http://127.0.0.1:5173/?recording=team_goal) or [southampton](http://127.0.0.1:5173/?recording=southampton). Select **Ball review → M23 — Intelligent Ball Tracking**. Source-video controls synchronize M21 temporal/offline versus M23 canvases. The dashboard switches separate original/new per-clip comparisons and exposes causal memory, uncertainty/search regions, current observations, history/reconstruction, reasons, interaction evidence and navigation to successful/incorrect model reacquisitions, misses, false detections, disappearances and all seven pass cases. Enable Ball candidates for unselected/rejected boxes. Source frame/model/evidence revision mismatches are refused.

Ask uses only the selected version's validated M19-qualified events. Checkpoint/model-set, ball and event hashes separate recording/version caches and conversation context; switching modes clears previous responses. Player focus, M9 pitch overlays, video seeking and Azure analytics remain separate.

Actual three-clip M23 Chrome verification passed. The legacy M21 test now requests source playback before expecting decoded data from a metadata-preload player; all original frame/clock/grounding assertions remain. Broader browser status: `{'status': '11 distinct Chrome cases passed targeted checks; latest complete isolated-browser run: 7 passed and 4 timed out; combined suite is not all-green', 'distinct_targeted_passed': 11, 'm23_three_clip_targeted_passed': True, 'final_combined': {'passed': 7, 'timed_out': 4, 'affected': ['M23 detail API wait', 'M19 source/interaction wait', 'full-match media readiness', 'match-experience cross-half goal rendering']}, 'earlier_combined_runs': [{'passed': 4, 'failed': 7}, {'passed': 2, 'failed': 9}, {'passed': 7, 'failed': 4}], 'test_integrity': 'No evidence/clock/grounding assertion removed; M21 explicitly requests playback before requiring decoded data', 'runtime_mitigations': 'Fresh frontend, isolated browser projects, opt-in tracing and idle-sleep prevention did not eliminate every long-batch timeout', 'baseline_probe': 'Unchanged frontend probe also timed out before page setup; this does not establish the cause of all batch failures', 'hosted_llm_calls': 0, 'isolated_previous': {'passed': 7, 'timed_out': 4, 'affected': ['M23 detail API wait', 'M20 diagnostic API wait', 'full-match media readiness', 'match-experience cross-half goal rendering']}, 'frontend_restart': 'Final frontend restarted successfully on 5173; direct /api/ready and backend /ready returned 200 with 15 recordings'}`. Screenshot evidence, once copied after visual inspection: [workspace](evidence/cv_m23_workspace.jpg). No raw video, checkpoint or human annotation file belongs in Git.

## Tests and live Azure

Backend: **1,122 run, 1,121 passed, one existing skip** in the complete fresh-process suite; 30 focused M23 tests also passed. Frontend: **181 passed** with two workers; TypeScript `tsc --noEmit` and production build passed. The first highly concurrent frontend run had a legacy five-second timeout; bounded workers passed without changing that test's assertion. Camera pan/cut/reset, covariance growth/expiry, tentative confirmation, ambiguity, blur, wrong-object rejection, dual M22 current checks, crop bounds/deduplication, DAG defaults/anchor support, reconstruction provenance, optional corrupt-mode handling and Ask revision isolation are covered.

Live verification record: `{"verification_started_at_utc": "2026-11-21T14:42:12.957989+00:00", "backend": "http://127.0.0.1:8000", "provider": "local", "/health": {"status_code": 200, "body": {"status": "ok", "recordings": 12, "processing": "precomputed_artifacts"}}, "/ready": {"status_code": 200, "body": {"status": "ready", "recordings": 15}}, "recordings": {"total": 15, "ready_azure_analytics": 12, "local_cv": 3}, "cv": [{"recording_id": "test_match", "metadata_frame_reference_status": 200, "source_frame": 351, "video_status": 206, "video_bytes": 16, "grounded_ask_status": 200, "event_count": 3, "cv_model_revision": "1ac32c2f5d4506ba81e403f5949c6797795f4c5bb4d2833e12365a021504694c:ba980ef2f6293275b617c858066629a33c539e707f7ec1509791c8c4035d4fcd:4635a6001864c275e1a7d507326afbd55a1c9d902ff550a071054df0c54d7947", "new_development_metrics": {"tp": 44, "fp": 0, "fn": 6, "precision": 1.0, "recall": 0.88, "f1": 0.9361702127659575, "precision_wilson95": [0.9197016822179859, 0.9999999999999999], "recall_wilson95": [0.7619492111337642, 0.9438247628937535], "tn": 0, "reviewed_frames": 50, "unknown_visibility_frames": 0}}, {"recording_id": "team_goal", "metadata_frame_reference_status": 200, "source_frame": 8, "video_status": 206, "video_bytes": 16, "grounded_ask_status": 200, "event_count": 1, "cv_model_revision": "d2f2c7ef42982e198d0d778002872edd0e381b86145264046c3a33d41b8bf21e:a2b7224f67cf8b150094848f7ef9c50509f572c2276160ec6baf380213fa5c08:50c4d1d2546b693c1b4c05d4e7af65d1521cbcef081e7d1ab6163eba3f31495f", "new_development_metrics": {"tp": 29, "fp": 10, "fn": 16, "precision": 0.7435897435897436, "recall": 0.6444444444444445, "f1": 0.6904761904761905, "precision_wilson95": [0.589180432321585, 0.854313741565478], "recall_wilson95": [0.49835546011952203, 0.7678110864309596], "tn": 2, "reviewed_frames": 50, "unknown_visibility_frames": 0}}, {"recording_id": "southampton", "metadata_frame_reference_status": 200, "source_frame": 16, "video_status": 206, "video_bytes": 16, "grounded_ask_status": 200, "event_count": 0, "cv_model_revision": "5cab3e521444fe45b7306bab76df05aa2289fcb526619ca64bd19770d65f0db8:e14ebf32525905c78697cef3a239119ab31c77038efebaa6531bf54c53468ff4:57f290deb92b5413ef9ac8eaa49fa4c451088bc4143476551c73c1dde5eb09dc", "new_development_metrics": {"tp": 34, "fp": 8, "fn": 16, "precision": 0.8095238095238095, "recall": 0.68, "f1": 0.7391304347826086, "precision_wilson95": [0.6669893264500356, 0.9001811039014357], "recall_wilson95": [0.5418944286372394, 0.792419559060006], "tn": 0, "reviewed_frames": 50, "unknown_visibility_frames": 0}}], "azure_118575_clip": {"view_status": 200, "events": 481, "ask_status": 200, "evidence_items": 14, "video_status": 206, "video_bytes": 16}, "verified_at_utc": "2026-11-21T14:42:36.147542+00:00"}`

## Immutable roots, hashes and source preservation

Input cache: `outputs/m23/inputs_v2`; fresh neural campaign: `outputs/m23/perception_v2`; primary-protected proposal cache: `outputs/m23/perception_primary_v4`; final analysis: `outputs/m23/analysis_v5`; diagnostics: `outputs/m23/diagnostics_final_v1`. `outputs/m23` is a local ignored link to `~/Library/Caches/TactiqAI/m23`, outside iCloud. Original M14–M22 outputs remain unchanged. To recover from disk exhaustion during the folder-wide iCloud download, 198 rejected M23 intermediate JSONL files were losslessly gzip-archived, with uncompressed SHA-256 checked before removing redundant copies (933,091,813 bytes reclaimed). Their reports and compressed original bytes remain; see `outputs/m23/rejected_intermediate_archives.json` and decompress before inspecting those rejected roots. The active v2 neural cache, v4 protected cache and final v5 analysis remain uncompressed and unchanged. Installation creates hash-equivalent cached M9/video paths and changes only a new runtime manifest's explicit `source_video` path, preserving video/frame bytes and original manifests. This is not fallback to unrelated artifacts.

Final analysis SHA-256: `92826c317c6dd4045db5867248aad8a37a4eb35890a5575bc5cb5230f5c5b89f`. Original M19 labels: `c4ac71548269fd1fe4e601684dffa09a706e8a215ed978ff9e0c8feb2966991e`. New frozen 150 labels: `0b4c470a5f116f2a8c4811f7a5b7d66f11b017534d0f4d5146aae7a1312a51b4`. The withdrawn 20-label snapshot remains `ca678291e33d97f4fdec774c396df5d7a159a57c17ee283928fa6fe7f869cd4b`. Component/checkpoint/source/config hashes and all three byte-identical M21 reproduction checks are in [aggregate evidence](evidence/cv_m23_results.json) and the private manifests. Original files were not relabelled; iCloud hydration failures were diagnosed rather than weakening hash/schema checks.

## Reproduction and startup commands

Run from the repository root with the existing intact runtime. Preparation copies only the existing approved inputs. Use new roots for new experiments; completed artifacts are immutable. A first run can omit `--base-inputs`; the current prepared v2 snapshot already includes complementary global outputs.

```sh
PYTHON="$HOME/Library/Caches/TactiqAI/backend-m17/bin/python"
"$PYTHON" -m scripts.prepare_cv_m23 --output outputs/m23/inputs_reproduction --base-inputs outputs/m23/inputs_v2
"$PYTHON" -m scripts.run_cv_m23 --inputs outputs/m23/inputs_reproduction --output outputs/m23/perception_reproduction
"$PYTHON" -m scripts.cache_cv_m23_primary --inputs outputs/m23/inputs_reproduction --parent outputs/m23/perception_reproduction --output outputs/m23/perception_primary_reproduction
"$PYTHON" -m scripts.analyze_cv_m23 --inputs outputs/m23/inputs_reproduction --perception outputs/m23/perception_primary_reproduction --output outputs/m23/analysis_reproduction
"$PYTHON" -m scripts.install_cv_m23 --analysis outputs/m23/analysis_v5
scripts/start_backend_m23.sh
```

The startup profile uses the existing `MATCHMIND_*` names, AzureCliCredential and stable artifact/highlight/assistant caches. It explicitly selects `outputs/m23/runtime_cv_registry.json`. The project's existing Azure `.env` account/container/recording configuration is still required. Overrides already exported by the user take precedence. Default port is 8000. Frontend: `npm --prefix frontend run dev` (5173).

```sh
"$PYTHON" -m unittest discover -s tests -v
npm --prefix frontend test -- --run --maxWorkers=2
(cd frontend && node_modules/.bin/tsc --noEmit)
npm --prefix frontend run build
frontend/node_modules/.bin/playwright test --config frontend/playwright.m23.config.ts
"$PYTHON" -m scripts.verify_cv_m23_serving --output outputs/m23/live_verification_final.json
```

For an exact cached association check, run `analyze_cv_m23` against `perception_primary_v4` with a new output directory and compare the twelve ball/canonical/diagnostic/pass hashes. Fresh MPS neural outputs are measured anew; exact cache reproduction is a different claim from deterministic new inference.

## Independent evaluation workflow and remaining work

**No genuinely unseen recording was approved. Independent generalization remains unverified.** The withdrawn M21 Azure review tasks and Tottenham–Arsenal full match were not used for M23 CV or evaluation. Legacy regression tests only preserve/inspect the withdrawn review's existing export and blocking behaviour; they do not create labels or restore its evaluation role.

`scripts.prepare_cv_m23_independent` first requires an explicit approval record binding a local video's SHA-256, original-match ID and approval reference. Its preview stage decodes twelve source-only frames without model suggestions. A person must verify relevant football players/ball and original-match non-overlap against the complete training/development source inventory, including weak-source match IDs. Without that audit, it cannot freeze a source/protocol. Its freeze stage records source/approval/review/analysis/code/config hashes and the fixed IoU/pass policy, while explicitly refusing promotion and marking checkpoint selection and final labels incomplete. Select and hash one existing checkpoint before inference, then freeze source-only review windows before final labels; do not tune on those labels. No new source was processed by this workflow.

Remaining within the measured limitations: improve tiny/blurred candidate availability, reject boots/boards in bounded gaps, recover correct contact/identity/team evidence for the other six passes, measure continuous reacquisition identity on independent references, and benchmark one final cold pipeline rather than a shared diagnostic campaign. Do not retune labels or relax event/manifest gates. **M24 was not begun.**

## CourtVision guidance and attribution

Revisited [CourtVision](https://github.com/aiden-guan/courtvision/tree/581f67e82c9e271751baffb62046cee4e1e94902) at commit `581f67e82c9e271751baffb62046cee4e1e94902`: [ball.py](https://github.com/aiden-guan/courtvision/blob/581f67e82c9e271751baffb62046cee4e1e94902/courtvision/perception/ball.py), [student.py](https://github.com/aiden-guan/courtvision/blob/581f67e82c9e271751baffb62046cee4e1e94902/courtvision/perception/student.py), [NBA analysis](https://github.com/aiden-guan/courtvision/blob/581f67e82c9e271751baffb62046cee4e1e94902/courtvision/nba/analyze.py), [ball labels](https://github.com/aiden-guan/courtvision/blob/581f67e82c9e271751baffb62046cee4e1e94902/courtvision/training/ball_labels.py), [pseudo labels](https://github.com/aiden-guan/courtvision/blob/581f67e82c9e271751baffb62046cee4e1e94902/courtvision/training/pseudo_labels.py). Camera-relative association, separate coasts/observations, bounded gaps/restarts and confidence-aware linking informed this football adaptation. Basketball colour/gravity/3D assumptions and pseudo-labels were not imported as football truth. No new CourtVision weights or verbatim upstream implementation was added. Existing [MIT attribution](third_party/courtvision_m22.md) remains.
