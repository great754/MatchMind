# M22 — Trained temporal football detector

Verified implementation and development experiment, 2026-11-20. **Opt-in research; no accuracy promotion.** M14 remains the ordinary default, M20 the frozen experimental control, and M21 diagnostics remain available. This report follows the latest three-frame architecture brief, which supersedes the earlier broader M22 proposal. M23 has not started.

## Architecture and source research

CourtVision was studied at commit `581f67e82c9e271751baffb62046cee4e1e94902`: [student detector](https://github.com/aiden-guan/courtvision/blob/581f67e82c9e271751baffb62046cee4e1e94902/courtvision/perception/student.py), [training](https://github.com/aiden-guan/courtvision/blob/581f67e82c9e271751baffb62046cee4e1e94902/courtvision/training/train_student.py), [ball labels](https://github.com/aiden-guan/courtvision/blob/581f67e82c9e271751baffb62046cee4e1e94902/courtvision/training/ball_labels.py), [pseudo-labels](https://github.com/aiden-guan/courtvision/blob/581f67e82c9e271751baffb62046cee4e1e94902/courtvision/training/pseudo_labels.py), and [offline analysis](https://github.com/aiden-guan/courtvision/blob/581f67e82c9e271751baffb62046cee4e1e94902/courtvision/nba/analyze.py). The useful ideas are three chronological RGB inputs, a pretrained small backbone, high-resolution center prediction, confidence-aware weak supervision, ignoring unknown labels, and offline linking. Basketball weights, court/color assumptions, pseudo-labels, player heads and event rules were not imported. One detector architecture is implemented:

`t−2 → t−1 → t RGB → 9-channel ResNet-18 → 64-channel FPN → stride-4 heatmap / center-offset / log-box-size heads`.

The RGB stem weights are repeated and divided by three. The stem, final backbone block and heads train; earlier blocks and batch-normalization statistics are frozen. FPN refinements use group normalization. The box-size head supports the existing IoU evaluation; there is no unvalidated visibility or physical-height head. Model code lives in `src/vision/football22/`.

Temporal spacing approximates 25 Hz using the actual source FPS. Context is causal and resets at frozen M9 cut epochs; missing beginning-of-shot context duplicates the earliest available frame. Whole-source letterboxing preserves all pixels. Additional overlapping 320-source-pixel tiles improve small-ball resolution. Rounded horizontal/vertical resize factors, crop origin and padding are tracked separately and inverted into source pixels. No reference box determines inference crops.

Every frame receives a full-source search plus tiles. Candidate confidence remains the network output. A direct candidate needs temporal confidence ≥0.35 and a same-network current-frame-only check ≥0.35 with spatial agreement; that check does **not** establish human truth or calibrated probability. Rejected candidates remain diagnostic. The fixed M21 skip-edge DAG links eligible candidates, followed by the existing reconstruction. Future evidence may influence **offline association**, never neural visual input. Observed, inferred, ambiguous and unknown states remain separate. Reconstructed points cannot supply observed release/flight/receipt or verified pitch height.

## Data, splits and original-label integrity

Only `test_match` (541 frames), `team_goal` (500), and `southampton` (660) were processed. Twelve model-free source samples per clip and all six selected review sheets were visually inspected for players, pitch and useful football footage. Color/variance gates are content proxies, not semantic action/replay recognition or ball truth. The withdrawn M21 set and the Azure Tottenham–Arsenal full match are excluded.

The original 60 reviews remain development evidence. Their SHA-256 is `c4ac71548269fd1fe4e601684dffa09a706e8a215ed978ff9e0c8feb2966991e`. They contain **34 known ball frames: 31 visible, three not visible**. Two uncertain ball frames are excluded. Seven completed passes occur in six exhaustive windows totaling 18.012 seconds; six other pass windows remain excluded. Twelve same-person identity pairs offer only three model-matched pairs, nine unmatched pairs and no different-person negatives.

The pre-training audit is frozen in `outputs/m22/data_v1/baseline_error_breakdown.json`. It separates missing neural candidates, incorrect selected boxes, below-threshold reference candidates and association failures. For example, M20 has no matching candidate on all 11 reviewed visible `team_goal` frames. Some centers fall inside a reference box but fail IoU; those remain failures. Ball-size evidence is measured from existing boxes. Blur, occlusion and specific lookalike classes have not received exhaustive human cause labels; no numerical cause breakdown is invented for them.

One small-data strong-only campaign was trained and fully evaluated: **0 correct / 1 false / 31 missed balls after association, zero of seven passes**. It was rejected as an upgrade. The same architecture was then improved using **120 existing SoccerNet temporal training triplets**, explicitly weak organizer positives, weight **0.15**. Their unknown surroundings receive no background-negative loss. No existing external validation/test images enter training or evaluation truth. Whole-original-match overlap between those weak sources and the approved clips is not established; these experiments cannot establish independent cross-match generalization.

The final campaign uses three source-excluded fits of this single architecture:

| Excluded inference source | Strong training source / frames | Strong validation source / frames | Weak training triplets |
|---|---|---|---:|
| test_match | team_goal / 11 | southampton / 11 | 120 |
| team_goal | test_match / 12 | southampton / 11 | 120 |
| southampton | test_match / 12 | team_goal / 11 | 120 |

The target clip's human labels are excluded from its fitting and early stopping. Source folds avoid neighboring-frame train/validation leakage. All three clips, however, already influenced development and data decisions. **There is no untouched final test**, and aggregate results combine three source-excluded checkpoints, not one all-data checkpoint.

Strong visible/absent frames carry positive/negative supervision. Uncertain/unannotated pixels have zero loss. A rejected proposal supervises only the explicitly checked region; rejecting one region does not declare the entire frame ball-absent. Original labels, frozen upstream artifacts, source videos and prior failed experiments are retained.

## Training and resources

The user approved the official ImageNet [ResNet-18 checkpoint](https://download.pytorch.org/models/resnet18-f37072fd.pth), saved only under ignored `outputs/m22/pretrained/`: **46,830,571 bytes**, SHA-256 `f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec`. No basketball checkpoint or new dataset was downloaded. Training uses the existing runtime: Torch **2.15.0**, torchvision **0.29.2**, OpenCV **5.0.0.93**, Apple MPS, with CPU fallback. Those installed versions differ from the older research requirement pins; no clean installation of those older pins was tested or changed. Float32 is deliberate; mixed precision is disabled.

The final campaign ran eight epochs per fold, batch two, AdamW (heads 0.001, trainable backbone 0.0001; weight decay 0.0005), cosine scheduling, gradient clipping at five and early-stopping patience six. Human examples repeat six times per epoch; weak examples once. Seed is 20261119 with explicit fold seeds. Augmentations share geometry/exposure across all three frames: consistent flip, ball-preserving source crop, JPEG degradation, short linear blur, and occasional duplicated-current history. Strong crops are 192/256/320/384 source pixels; weak 1080p crops are 384/640/960/1080 to reduce their network ball sizes.

Checkpoint selection uses actual whole-source/tiled/current-checked **validation detector F1, then precision, then loss**. This corrected a misleading whole-frame-loss selection mismatch found during development. Association and pass gates were not searched. Last checkpoints include optimizer/scheduler, RNG, loader state and provenance for resumption. Resuming the completed campaign verified all three model hashes and skipped fitting. Interrupted MPS training is supported but bit-identical GPU resumption has not been established.

Private experiment roots are immutable versions: `temporal_data_weak_v1`, `temporal_training_final_v1`, `perception_final_v1`, and `analysis_final_v1` under `outputs/m22/`. Earlier interrupted/preliminary roots are diagnostics, not additional reported systems. Artifacts, model binaries and raw human labels are not committed.

## Measured development result — regression, not promotion

The final model **does not meet any aggregate research target**. The weak-data/scaling/validation changes improve the initial strong-only fit, but the resulting system remains worse than M20. All results below use the fixed IoU ≥0.25 rule on the original development references.

| System | Ball TP / FP / FN | Precision | Recall | F1 | Completed passes matched |
|---|---:|---:|---:|---:|---:|
| Frozen M20 | 10 / 4 / 21 | 71.4% | 32.3% | 0.444 | 1 / 7 |
| M21 offline DAG | 11 / 2 / 20 | 84.6% | 35.5% | 0.500 | 0 / 7 |
| M21 temporal + offline | 17 / 9 / 14 | 65.4% | 54.8% | 0.596 | 0 / 7 |
| Initial M22 strong-only fit + fixed DAG | 0 / 1 / 31 | 0% | 0% | 0 | 0 / 7 |
| Final M22 + fixed DAG | **8 / 22 / 23** | **26.7%** | **25.8%** | **0.262** | **0 / 7** |

| Final M22 recording | Ball TP / FP / FN | Precision | Recall | F1 | Pass matches | Direct model observations across clip |
|---|---:|---:|---:|---:|---:|---:|
| test_match | 8 / 4 / 3 | 66.7% | 72.7% | 0.696 | 0 / 2 | 457 / 541 |
| team_goal | 0 / 10 / 11 | 0% | 0% | 0 | 0 / 3 | 464 / 500 |
| southampton | 0 / 8 / 9 | 0% | 0% | 0 | 0 / 2 | 552 / 660 |

The high direct-observation coverage is **not accuracy**: the model repeatedly accepts incorrect visual objects on two sources. Current-only agreement and temporal consistency fail to reject these errors. Before association, the top eligible neural candidate gives **7 TP / 25 FP / 24 FN**, precision 21.9%, recall 22.6%, F1 0.222. Linking helps slightly but cannot recover absent correct candidates. At candidate threshold 0.05, top-1/3/5 candidate recall is **8/31, 10/31, 11/31**; these are diagnostic pools, not accepted observations. Median/mean/p90 source-pixel center error is **6.31 / 6.59 / 9.42 pixels**, only on the eight IoU-matched observations, across differing source resolutions.

M22 precision's Wilson 95% interval is 14.2–44.4%; recall's is 13.7–43.2%. One reviewer and 31 visible balls remain a very small, development-influenced sample. The seven-pass comparison has zero matches and zero false estimates **inside the 18.012 exhaustively reviewed seconds**; pass precision is undefined, recall 0%. The whole `test_match` clip produces two evidence-qualified, unreviewed estimates and six possible transfers. Those estimates are outside the matched-reference result and do not establish two true completed passes. The other clips yield none. All seven release/flight/receipt/player/team failure traces are in `outputs/m22/analysis_final_v1/<recording>/pass_reference_traces.json`; missing observations/clear contacts and unchanged continuity/team gates explain rejections, without inventing human labels for every model frame.

| Source-excluded checkpoint | Selected epoch | SHA-256 |
|---|---:|---|
| test_match | 4 | `adff759a7ddafe23ee84411cb68eb26eb8639ef5c5b6fff279efe6b9e0c77ab9` |
| team_goal | 6 | `d4c4c1770f00c93bb3c6930f011a225d08c69b3dc0f3cfff47c39aceed885368` |
| southampton | 5 | `ebb1cebfe6ed47fde7c7877e4a92fef47789b7837674770359a76b603086a3ee` |

Measured training time across the three final fits was **720.9 seconds**, peak process RSS **1,009,401,856 bytes**. Best checkpoint sizes are **47,528,147 / 47,528,531 / 47,528,339 bytes** (test_match/team_goal/southampton). Inference processed all 1,701 source frames / **46,514 neural views**, including current-only checks. Per-clip processing rates were **0.883 / 1.919 / 0.777 FPS**, summed measured inference **1,723.4 seconds**, peak inference process RSS **740,671,488 bytes**. These are local MPS measurements with other workloads, not real-time throughput or a clean hardware benchmark. Detailed counts, localization, confidence intervals, source folds, runtime and checkpoint hashes are in [portable aggregate evidence](evidence/cv_m22_results.json); private raw source labels are excluded.

The limiting evidence is clear: **small-data cross-source generalization and false-positive rejection remain unresolved**. Verified source-specific hard negatives, tighter diverse visible-ball supervision, and independent unseen football action are needed before another accuracy claim. No threshold, IoU criterion, association configuration or pass gate was loosened to improve this table.

## Review workflow

Open [M22 football training review](http://127.0.0.1:5173/?review=m22&recording=team_goal). There are **150 new frames, 50 per approved clip**, arranged in short sequences across ten temporal bins per clip. Selection prioritizes M20/M21 disagreements, uncertain proposals and missing candidates. Original reviewed frames are excluded. All 150 selected source frames were inspected on six model-free sheets before publication. Proposals are never pre-saved as labels.

1. Check the clip name and current source frame. Your reviewer identity is remembered; settings are collapsed.
2. Inspect the enlarged proposal previews. Click **Accept proposal** for the actual ball and **Not the ball** for incorrect boxes. Accept overlapping proposals too if they show the same ball. A different non-overlapping ball choice requires reviewing the previous choice again.
3. If no proposal fits, draw/correct a tight ball box. **Ball not visible** explicitly marks absence; **Can't tell** records uncertainty. Rejecting all proposals alone never becomes a whole-frame absence label.
4. Once every proposal is reviewed and the ball decision is complete, the page **automatically saves and moves to the next frame**, after positive server acknowledgement. There is no Save button or per-frame confirmation checkbox. Your choices confirm that you checked the source. The last frame stays available with a completion message.
5. Use **Previous frame** to fix a saved decision; corrections save automatically too. Incomplete choices remain browser drafts when navigating. A failed/timeout/unacknowledged save keeps the current frame and offers a deliberate retry; it never enters an automatic retry loop. Optional notes pause saving while being edited. Notes, context images, examples, provenance and exports are under expandable sections.

New labels go to `outputs/m22/human_review/labels.json`, separately from the original 60 and withdrawn M21 labels. They are model-assisted **development** labels. A new data revision importer validates them and never overwrites original frames. The former M21 page now shows a withdrawal notice; its POST returns 410 while historical exports remain preserved.

## Review simplification verification (2026-11-20)

The updated review UI passed **171 frontend tests**, **21 focused backend tests**, the typed production build and **two Chrome cases in 15.0 seconds**. Chrome checked automatic progression and back/correction on all three approved clips, desktop/mobile layouts, and failed-save recovery. Six captured automatic-save payloads passed the real backend validator without being written. All **six existing M22 labels**, the original **60 M19 reviews**, and **20 saved withdrawn M21 records** remain byte-identical. Opening a frame never accepts a proposal automatically. The original 150 tasks and model results are unchanged; no retraining or new source selection occurred. [Verification](evidence/cv_m22_review_simple.json) · [Simplified UI](evidence/cv_m22_review_simple.jpg). The implementation verification below retains its original milestone counts.

## Frontend and evidence boundaries

At [the short-clip workspace](http://127.0.0.1:5173/?recording=team_goal), select **Ball review → M22 — Temporal Football Detector**. The panel provides synchronized M20/M22 source playback, source-aligned activation maps, observed/inferred state, candidate rejection reasons, per-recording comparisons, checkpoint information and review links. The heatmap is diagnostic activation, not calibrated confidence. Existing player tracking, camera/pitch display and source seeking remain in place.

Every seven-pass reference has a local release/flight/receipt/player/team evidence trace. Existing `detect_m19` and `PassConfig` are unchanged. Possible proximity is not a touch or possession; goals, shots, tackles, interceptions, pass attempts/accuracy and possession percentages remain unavailable when unsupported. The assistant uses the selected version's qualified structured evidence. Source/model/event revisions isolate its indexes and context; changing versions clears the old answer/video.

Optional M22 files are strictly checked for hash, schema, source clock, selected-candidate agreement and checkpoint provenance before registration. Runtime changes invalidate them. Missing/corrupt M22 never silently falls back to another ball model or removes an otherwise usable recording. Canonical Azure manifests and analytics readiness are unchanged. These three CV source videos and M22 outputs remain local and are needed for their playback/review; existing canonical SoccerTrack media remains Azure-backed.

## Final verification

- **Backend:** 1,082 tests run, 1,081 passed, one skip. The final run includes checkpoint-only assistant revision isolation and withdrawn-review POST protection.
- **Frontend:** all 164 tests passed with two workers / 30-second typing-test budget. TypeScript and production build passed (main entry 499.48 kB; lazy M22 detector panel 7.92 kB; review panel 9.73 kB).
- **Chrome:** **all 11 cases passed in one final 2.4-minute run**, including all three M22 playback/seek/comparison/heatmap/Local Ask checks, all three explicit-save review checks (writes mocked), M14/M19/M20/M21 regression, Azure full halves, player focus/markers, goal compilations, score, player stats and maps. No real human annotations were written by browser checks. [Verified M22 UI](evidence/cv_m22_workspace.jpg).
- **Reproduction:** all three M22 sidecars passed strict validation; a fresh cached-analysis replay matched **12 primary ball/canonical/event/evidence SHA-256 values**. Completed-training resume verified the three frozen checkpoint hashes. This establishes deterministic cached analysis, not bit-identical GPU retraining.
- **Live API:** Azure-backed health/ready/catalog 200, readiness 15 usable recordings; 118575_clip view 200 / 481 events, grounded Local Ask 200 / 14 facts, video range 206. A full-half 1-MiB range returned 206. M22 metadata/frame/Local Ask return 200 for each approved clip, with correct floor-based source clocks and checkpoint/ball/event version isolation.

Earlier overlapping runs hit unit-worker, API, playback and browser-launch timeouts. Existing configured artifact files and small portrait/highlight caches were offloaded by iCloud and were restored; the final run followed inference completion, restoration, a clean backend restart and keeping the Mac awake. Those early failures are not recorded as passes. No analytics validation, source clock or event evidence gate was relaxed to obtain the final results.

## Reproduction commands

Use an intact runtime with existing vision dependencies; these commands neither install dependencies nor download models automatically. Output roots must be fresh, or use the explicit resume option for the same campaign.

```sh
export TACTIQAI_BACKEND_PYTHON=/Users/greatabhieyighan/Library/Caches/TactiqAI/backend-m17/bin/python

# Existing frozen data and explicitly weak local supervision:
"$TACTIQAI_BACKEND_PYTHON" -m scripts.prepare_cv_m22 --temporal-from outputs/m22/data_v1 --output outputs/m22/temporal_data_repro --weak-local
"$TACTIQAI_BACKEND_PYTHON" -m scripts.train_cv_m22 --data outputs/m22/temporal_data_weak_v1 --output outputs/m22/temporal_training_final_v1 --epochs 8 --resume
"$TACTIQAI_BACKEND_PYTHON" -m scripts.run_cv_m22 --data outputs/m22/temporal_data_weak_v1 --training outputs/m22/temporal_training_final_v1 --output outputs/m22/perception_final_v1 --resume
"$TACTIQAI_BACKEND_PYTHON" -m scripts.analyze_cv_m22 --perception outputs/m22/perception_final_v1 --training outputs/m22/temporal_training_final_v1 --output outputs/m22/analysis_repro --no-register

# After real additional review saves, freeze new supervision without replacing v1:
"$TACTIQAI_BACKEND_PYTHON" -m scripts.import_cv_m22_review --base outputs/m22/data_v1 --output outputs/m22/data_v2
"$TACTIQAI_BACKEND_PYTHON" -m scripts.prepare_cv_m22 --temporal-from outputs/m22/data_v2 --output outputs/m22/temporal_data_v2 --weak-local
```

Review creation uses `scripts.prepare_cv_m22_review`: inspect the draft's source-only sheets, then publish with `--publish-draft`, `--curator` and one `--inspected-sheet` for **each** actual sheet. Existing published tasks cannot be overwritten. Curation is an explicit recorded judgement, not automatically fabricated source truth.

Start the Azure-backed application from the repository root:

```sh
AZURE_TOKEN_CREDENTIALS=AzureCliCredential MATCHMIND_ARTIFACT_STORE=azure MATCHMIND_PROVIDER=local MATCHMIND_CACHE=outputs/cache/azure_m17 MATCHMIND_HIGHLIGHT_CACHE=outputs/highlights_m18 TACTIQAI_BACKEND_PYTHON=/Users/greatabhieyighan/Library/Caches/TactiqAI/backend-m17/bin/python scripts/start_backend.sh
# Separate terminal:
npm --prefix frontend run dev
```

Checks:

```sh
"$TACTIQAI_BACKEND_PYTHON" -m unittest discover -s tests
npm --prefix frontend test -- --maxWorkers=2 --testTimeout=30000
npm --prefix frontend run build
npm --prefix frontend run test:browser -- --config playwright.m22.config.ts
```

## Licensing and remaining work

Architecture adaptation credits CourtVision (MIT, Copyright 2026 Aiden Guan); see [third-party notice](third_party/courtvision_m22.md). CourtVision research weights and sports video datasets have separate terms and were not reused. ImageNet initialization comes from the approved official PyTorch distribution. Existing SoccerNet training material is research footage subject to [SoccerNet access/rights terms](https://www.soccer-net.org/data); local availability is not a blanket redistribution or commercial license. No raw footage, external image collection, annotations or model checkpoint is published to GitHub. SoccerTrack image-space/GSR, reconstructed ball NPZ and BAS data were audited as structurally different evidence and not silently converted into certified visible-ball training truth.

Next work stays within M22: finish human corrections and representative hard negatives, retrain the same architecture using versioned labels, obtain explicitly user-selected useful unseen footage and check training overlap, then freeze checkpoint/protocol before its independent labels. Short-clip development measurements cannot establish full-match generalization, reliable continuous identity, metric 3D trajectories or accurate possession totals. Additional compute or large data acquisition requires explicit authorization.
