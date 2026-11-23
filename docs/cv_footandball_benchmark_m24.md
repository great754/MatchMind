# M24 — FootAndBall pretrained detector benchmark

Verified 2026-11-23 on this Mac. **Reject the official pretrained checkpoint as an upgrade on these clips.** It is fast, but finds **0/31 reviewed visible Giroud balls**, with only **2/31 top-five candidate matches**. Keep the current pipeline. No fine-tuning, tracker integration, M25 work, new annotation campaign or full-match inference was performed.

**Storage update later that day:** at the user's request, all four source videos were archived to Azure and their 16 local copies deleted after full verification. The M24 page now reads the same pinned bytes from Azure; benchmark results and annotations are unchanged. Earlier run paths below are historical. See [current media setup and offline reproduction recovery](cv_azure_media.md).

[Open the read-only benchmark](http://127.0.0.1:5173/?benchmark=m24&recording=giroud) · [Aggregate measurements](evidence/cv_m24_results.json) · [Verified browser view](evidence/cv_m24_workspace.jpg).

## What was inspected and implemented

The [official repository](https://github.com/jac99/FootAndBall/tree/01261092a93de55026745e532a2e12188ad86801) was cloned and its README, model, FPN, NMS, loss, inference script, training script, ISSIA reader, data reader, augmentation and checkpoint inspected before implementation. Official commit: `01261092a93de55026745e532a2e12188ad86801`.

The actual `fb1` checkpoint has **198,840 parameters**, 74 state-dictionary entries and **825,503 bytes**. SHA-256: `42fbc1f457fc34d804b403e26138524fc40facde303ccd7b4be73121e8da4cd5`. It loads strictly with `torch.load(..., weights_only=True, map_location='cpu')`. Its name is `models/model_20201019_1416_final.pth`; no training dataset was downloaded. No global dependency or model environment was changed.

Implementation findings from the pinned source:

- Nine 3×3 convolution/BatchNorm/ReLU layers with channels 16,32,32,32,32,64,64,64,64, separated into five pooled levels. The FPN uses 32-channel lateral projections and bilinear top-down additions. Returned feature maps have strides four and sixteen.
- Ball and player classification heads each have 32→32→2 convolutions and two-class softmax. Player regression uses 32→32→4. **There is no ball extent regressor.** The upstream 40-pixel ball box is a display convention; this adapter never returns it as a predicted box.
- The original loss mines background negatives at 3:1, with ball/player/location weights 5/1/0.01. Checked-in training marks a 3×3 region for both classes. Its configuration says Adam, learning rate 0.001, 20 epochs and batch size 12; these settings do not establish the released checkpoint's actual training history. Training crops are 1280×720; validation uses a 1920×1080 center crop. Augmentation includes color jitter, ±5° affine rotation, 0.8–1.2 scaling and horizontal flips.
- README provenance describes ISSIA cameras 1–4 plus SoccerPlayerDetection set 1; the checked-in training configuration differs. We cannot independently reconstruct checkpoint training from that configuration. The ball supervision is from ISSIA; the second dataset supplies player labels.
- Upstream defaults in `run_detector.py` are 0.7 for both branches. The builder accepts `test`, but `forward` recognizes `eval`; parity checking explicitly uses evaluation behavior rather than relying on that old API inconsistency. Detect-mode dtype/allocation and old data-reader dependencies are avoided. All three neural heads are retained unchanged.

### Research context and limits

The [paper](https://arxiv.org/html/1912.05445v2) ([published DOI](https://doi.org/10.5220/0008916000470056)) combines fine spatial features with broader context to recognize small balls. It reports ISSIA ball AP 0.909 and 37 FPS at 1920×1080 on GTX 1060, with roughly 199k parameters in its comparison table. ISSIA uses six synchronized long-shot cameras; SoccerPlayerDetection includes broadcast views but lacks ball labels. The paper identifies occlusions, white clothing and stadium clutter as failure modes. These results concern its own data split, AP definition and hardware; they do not transfer to our center-localization scores. Its efficiency and small-object design motivated this experiment, but our broadcast clips demonstrate a substantial generalization failure. The particular contribution of scale, compression, camera viewpoint or appearance cannot be isolated by this benchmark alone.

## Frozen protocol and input integrity

`configs/vision/cv_m24.json` was frozen **before any scored FootAndBall result**. Protocol SHA-256: `01f7c77cc4d880f68af868e1cd3c906d377568936e0c72908e4238219e0ba503`.

All **2,151 native frames** were processed, plus exactly one predefined diagnostic: **all 450 Giroud frames at 2× resolution**. Each frame has its source clock, decoded-pixel hash, float32 confidence heatmap, ranked candidates, selected center, mapping, timing and heatmap hash. No M20 prediction, M21 association, M22 temporal input, M23 proposal or label enters FootAndBall inference.

Native preprocessing is BGR→RGB, float32/255, ImageNet mean `[.485,.456,.406]` and std `[.229,.224,.225]`, followed by explicit **right/bottom padding to multiples of 32**. Padding is zero in normalized space. There is no native resize, crop or aspect distortion. Giroud/team_goal become 640×384; test_match 1088×1088; southampton 1280×736. The single 2× diagnostic is bilinear Giroud 1280×720, padded to 1280×736. A cell `(x,y)` maps to `((4x+1.5)/scale,(4y+1.5)/scale)` in source pixels. Padding centers are rejected.

Postprocessing reproduces strict greater-than-all-eight-neighbors NMS, including replicated-border ties and plateau suppression. Valid positive peaks are ordered by descending confidence, then y/x for deterministic ties. Top 100 are retained, including scores below 0.7. Primary selection is **rank one only when ≥0.7**. We also score the official-style all-peaks-at-0.7 output separately; its reviewed counts are identical here. Neither scores nor thresholds were tuned after seeing labels.

Primary matching is **one-to-one maximum cardinality, then minimum center distance**, within **1% of the source diagonal**: 7.343 px for Giroud/team_goal, 15.274 px for test_match, 14.686 px for southampton. Sensitivities at 0.5% and 1.5% remain separate. A wrong selected center on a visible frame contributes both FP and FN. Predictions and interpolated tracker positions are excluded from observed detections. Candidate recall@1/3/5 uses unthresholded ranked pools; it is an oracle availability diagnostic, not deployed recall. Earlier pipelines have different retained-pool policies, so their candidate recalls are descriptive rather than equal-capacity detector comparisons.

Native IoU ≥0.25 is retained for M21/M22/M23 meaningful boxes; **FootAndBall native IoU is unavailable**. Its synthetic display boxes were never scored. Sample counts and protocols are separate:

- **Giroud:** all 40 completed answers validated against source/proposals, then frozen: 31 visible, nine uncertain, no absent. Includes 32 source-spread tasks and eight post-hoc diagnostic tasks. Original full labels retain 16 accepted proposals, 15 manual visible boxes and 49 rejected regions. Annotation SHA-256 remains `eceb6c2a46fb5c96ebdbe40d39fdc2f535d93034fab2b754d8b2e344d0e6519c`.
- **Original M19:** all original 60 labels preserved; the ball subset is 36 tasks: 31 visible, three not visible, two uncertain. Independent source review at acquisition, subsequently used for development.
- **Completed M22:** 150 assisted development tasks, 145 visible and five not visible. Originals, tasks and rejected regions are preserved. The withdrawn M21 review is excluded.

Uncertain frames never become negative examples. Accepted proposals, manual corrections and rejected regions preserve their original meanings; a rejected region is not an absent-ball label. All frozen original input files were hash-verified. These are **one-reviewer, temporally correlated development measurements**, with proposal and frame-selection bias. Giroud is not an untouched final test. Wilson intervals below do not remove those limitations.

Frozen M21 and M23 predictions use identical source clocks and exact reviewed frames. M22 on the three older clips uses its frozen offline outputs. **Giroud M22 is a newly run, fixed raw temporal secondary reference**, using the existing test_match-excluded checkpoint, original global-plus-tile search, context offsets −6/−3/0 and original dual 0.35 gates, without M23 regions. It is not an earlier frozen M22 offline Giroud result. No checkpoint was retrained.

## Same-frame results

### Giroud Completed

| System | Precision | Recall | F1 | Correct | False | Missed | FP/scored frame |
|---|---:|---:|---:|---:|---:|---:|---:|
| M21 | 28.6% | 6.5% | 0.105 | 2 | 5 | 29 | 0.161 |
| M22 | 46.2% | 19.4% | 0.273 | 6 | 7 | 25 | 0.226 |
| M23 | 28.6% | 6.5% | 0.105 | 2 | 5 | 29 | 0.161 |
| FootAndBall | — | 0.0% | 0.000 | 0 | 0 | 31 | 0.000 |

| System | Precision Wilson95 | Recall Wilson95 | Matched center error mean / median / p95 (px) | Candidate @1 / @3 / @5 |
|---|---|---|---|---|
| M21 | 8.2%–64.1% | 1.8%–20.7% | 1.427 / 1.427 / 2.711 | 4/31 / 4/31 / 4/31 |
| M22 | 23.2%–70.9% | 9.2%–36.3% | 0.825 / 0.727 / 1.621 | 7/31 / 11/31 / 11/31 |
| M23 | 8.2%–64.1% | 1.8%–20.7% | 1.427 / 1.427 / 2.711 | 14/31 / 15/31 / 15/31 |
| FOOTANDBALL | — | 0.0%–11.0% | — (no correct selections) | 0/31 / 1/31 / 2/31 |

### Original M19

| System | Precision | Recall | F1 | Correct | False | Missed | FP/scored frame |
|---|---:|---:|---:|---:|---:|---:|---:|
| M21 | 80.8% | 67.7% | 0.737 | 21 | 5 | 10 | 0.147 |
| M22 | 71.4% | 48.4% | 0.577 | 15 | 6 | 16 | 0.176 |
| M23 | 78.6% | 71.0% | 0.746 | 22 | 6 | 9 | 0.176 |
| FootAndBall | 100.0% | 3.2% | 0.062 | 1 | 0 | 30 | 0.000 |

| System | Precision Wilson95 | Recall Wilson95 | Matched center error mean / median / p95 (px) | Candidate @1 / @3 / @5 |
|---|---|---|---|---|
| M21 | 62.1%–91.5% | 50.1%–81.4% | 2.788 / 2.512 / 4.351 | 21/31 / 23/31 / 23/31 |
| M22 | 50.0%–86.2% | 32.0%–65.2% | 4.446 / 4.303 / 7.158 | 13/31 / 15/31 / 16/31 |
| M23 | 60.5%–89.8% | 53.4%–83.9% | 2.813 / 2.486 / 4.527 | 17/31 / 22/31 / 23/31 |
| FOOTANDBALL | 20.7%–100.0% | 0.6%–16.2% | 2.182 / 2.182 / 2.182 | 4/31 / 8/31 / 9/31 |

### Completed M22

| System | Precision | Recall | F1 | Correct | False | Missed | FP/scored frame |
|---|---:|---:|---:|---:|---:|---:|---:|
| M21 | 87.2% | 65.5% | 0.748 | 95 | 14 | 50 | 0.093 |
| M22 | 80.2% | 50.3% | 0.619 | 73 | 18 | 72 | 0.120 |
| M23 | 88.0% | 75.9% | 0.815 | 110 | 15 | 35 | 0.100 |
| FootAndBall | 100.0% | 1.4% | 0.027 | 2 | 0 | 143 | 0.000 |

| System | Precision Wilson95 | Recall Wilson95 | Matched center error mean / median / p95 (px) | Candidate @1 / @3 / @5 |
|---|---|---|---|---|
| M21 | 79.6%–92.2% | 57.5%–72.8% | 0.600 / 0.000 / 1.860 | 90/145 / 100/145 / 100/145 |
| M22 | 70.9%–87.1% | 42.3%–58.4% | 2.243 / 1.794 / 5.500 | 69/145 / 80/145 / 80/145 |
| M23 | 81.1%–92.6% | 68.3%–82.1% | 0.854 / 0.607 / 2.355 | 94/145 / 117/145 / 121/145 |
| FOOTANDBALL | 34.2%–100.0% | 0.4%–4.9% | 4.068 / 4.068 / 4.164 | 20/145 / 27/145 / 31/145 |

FootAndBall Giroud precision is undefined because it selects nothing; zero false detections is not evidence of adequate recognition. The nominal 100% precisions on other sets are based on only one/two selections (very wide intervals). Localization errors above condition on correct matches; full selected-visible errors including wrong selections are separately recorded in the aggregate JSON.

### Original native box criterion (IoU ≥0.25)

| Reference | System | TP / FP / FN | Precision | Recall | F1 |
|---|---|---|---:|---:|---:|
| original_m19 | M21 | 17 / 9 / 14 | 65.4% | 54.8% | 0.596 |
| original_m19 | M22 | 14 / 7 / 17 | 66.7% | 45.2% | 0.538 |
| original_m19 | M23 | 17 / 11 / 14 | 60.7% | 54.8% | 0.576 |
| completed_m22 | M21 | 95 / 14 / 50 | 87.2% | 65.5% | 0.748 |
| completed_m22 | M22 | 64 / 27 / 81 | 70.3% | 44.1% | 0.542 |
| completed_m22 | M23 | 107 / 18 / 38 | 85.6% | 73.8% | 0.793 |
| giroud_completed | M21 | 1 / 6 / 30 | 14.3% | 3.2% | 0.053 |
| giroud_completed | M22 | 6 / 7 / 25 | 46.2% | 19.4% | 0.273 |
| giroud_completed | M23 | 1 / 6 / 30 | 14.3% | 3.2% | 0.053 |

### Frozen center-tolerance sensitivity

Values are TP / FP / FN; no threshold or scale was selected from this table.

| Reference | System | 0.5% diagonal | 1% primary | 1.5% diagonal |
|---|---|---|---|---|
| original_m19 | M21 | 20 / 6 / 11 | 21 / 5 / 10 | 21 / 5 / 10 |
| original_m19 | M22 | 14 / 7 / 17 | 15 / 6 / 16 | 15 / 6 / 16 |
| original_m19 | M23 | 21 / 7 / 10 | 22 / 6 / 9 | 22 / 6 / 9 |
| original_m19 | FOOTANDBALL | 1 / 0 / 30 | 1 / 0 / 30 | 1 / 0 / 30 |
| completed_m22 | M21 | 95 / 14 / 50 | 95 / 14 / 50 | 95 / 14 / 50 |
| completed_m22 | M22 | 72 / 19 / 73 | 73 / 18 / 72 | 73 / 18 / 72 |
| completed_m22 | M23 | 110 / 15 / 35 | 110 / 15 / 35 | 110 / 15 / 35 |
| completed_m22 | FOOTANDBALL | 2 / 0 / 143 | 2 / 0 / 143 | 2 / 0 / 143 |
| giroud_completed | M21 | 2 / 5 / 29 | 2 / 5 / 29 | 2 / 5 / 29 |
| giroud_completed | M22 | 6 / 7 / 25 | 6 / 7 / 25 | 6 / 7 / 25 |
| giroud_completed | M23 | 2 / 5 / 29 | 2 / 5 / 29 | 2 / 5 / 29 |
| giroud_completed | FOOTANDBALL | 0 / 0 / 31 | 0 / 0 / 31 | 0 / 0 / 31 |

### FootAndBall by source (same reference scopes)

| Reference | Recording | TP / FP / FN | Candidate @1 / @3 / @5 |
|---|---|---|---|
| original_m19 | southampton | 0 / 0 / 9 | 1/9 / 2/9 / 2/9 |
| original_m19 | team_goal | 0 / 0 / 11 | 0/11 / 0/11 / 0/11 |
| original_m19 | test_match | 1 / 0 / 10 | 3/11 / 6/11 / 7/11 |
| completed_m22 | southampton | 0 / 0 / 50 | 4/50 / 7/50 / 7/50 |
| completed_m22 | team_goal | 0 / 0 / 45 | 5/45 / 5/45 / 7/45 |
| completed_m22 | test_match | 2 / 0 / 48 | 11/50 / 15/50 / 17/50 |
| giroud_completed | giroud | 0 / 0 / 31 | 0/31 / 1/31 / 2/31 |

## Candidate availability, sizes and visual evidence

Every missed visible reference receives a full-precision heatmap/ranking audit, including all strict peaks before the top-100 cap, the best nearby peak, rank/confidence, maximum response within tolerance and padding rejection count. These checks are saved under ignored `outputs/m24/evaluation_v2/missed_visible_diagnostics.json`; no original reference is changed.

| Reference | Misses | No strict nearby peak | Weak nearby peak | Nearby peak outside top 100 |
|---|---:|---:|---:|---:|
| original_m19 | 30 | 5 | 23 | 2 |
| completed_m22 | 143 | 22 | 104 | 17 |
| giroud_completed | 31 | 14 | 16 | 1 |

“No strict peak” means no local maximum inside the scoring radius, **not zero neural activation**. Tiny nonzero softmax responses do not establish ball recognition. Low confidence and higher-ranked distractors can coexist. All nearby Giroud candidates are below 0.7; there are no missed reviewed balls with an above-threshold near-ball peak outranked by another above-threshold object. The top-100 limit accounts for some missing retained candidates, but removing it would not create a valid detection at the frozen threshold. Strict-NMS suppression and weak responses are reported numerically rather than guessed from appearance.

Ball size is **human box width/height and sqrt(area)**, not a verified physical diameter. Relative sqrt(area)/source diagonal, source resolution, confidence, selection and correctness are recorded per visible reference. Box tightness varies, particularly between older manual boxes and proposal-assisted boxes; bins are descriptive.

| Reference | sqrt(area) source px | Visible | FootAndBall recall | M21 recall | M23 recall |
|---|---|---:|---:|---:|---:|
| original_m19 | ≤6 | 0 | — | — | — |
| original_m19 | >6–10 | 2 | 0.0% | 50.0% | 50.0% |
| original_m19 | >10 | 29 | 3.4% | 69.0% | 72.4% |
| completed_m22 | ≤6 | 7 | 0.0% | 71.4% | 85.7% |
| completed_m22 | >6–10 | 38 | 0.0% | 71.1% | 78.9% |
| completed_m22 | >10 | 100 | 2.0% | 63.0% | 74.0% |
| giroud_completed | ≤6 | 5 | 0.0% | 0.0% | 0.0% |
| giroud_completed | >6–10 | 22 | 0.0% | 4.5% | 4.5% |
| giroud_completed | >10 | 4 | 0.0% | 25.0% | 25.0% |

The 2× Giroud diagnostic still yields **0 TP / 0 FP / 31 FN**, zero top-five matches, and 28.36 full-processing FPS versus 71.10 FPS native. A single upscale does not recover recognition on this sample. No additional scales or tiles were tried.

Nine actual source-linked example figures were inspected. Green outlines are human references; orange is rank one **even when below threshold**, pink is observed M23. The right crop is nearest-neighbor 4×, and the heatmap uses absolute [0,1] intensity; very low responses appear black. These figures support qualitative descriptions, not additional human labels:

| Source frame | Inspected observation |
|---|---|
| [Giroud 60](evidence/cv_m24_examples/giroud_0060.jpg) · 2.40 s | A small, soft-edged bright ball is visible in open pitch; the highest candidate is near the goal/advertising area, confidence 0.01325. No accepted detection. |
| [Giroud 127](evidence/cv_m24_examples/giroud_0127.jpg) · 5.08 s | Blurred ball adjacent to white clothing; M23's selected center lies outside the human scoring radius. The precise cause of FootAndBall's weak response is unknown. |
| [test_match 471](evidence/cv_m24_examples/test_match_0471.jpg) · 19.64 s | A larger visible ball is correctly selected at 0.7805. |
| [test_match 9](evidence/cv_m24_examples/test_match_0009.jpg) · 0.38 s | A visibly elongated ball is missed; rank one is at a player's boot, confidence 0.00856. This is a below-threshold distractor, not an accepted FP. |
| [test_match 202](evidence/cv_m24_examples/test_match_0202.jpg) · 8.43 s | Human reference says not visible; M23 nevertheless selects a location. No new visibility judgment is substituted for that reference. |
| [team_goal 4](evidence/cv_m24_examples/team_goal_0004.jpg) · 0.16 s | Small bright ball near the goalkeeper is missed; rank one lies in the goal/stadium region at 0.00137. |
| [team_goal 163](evidence/cv_m24_examples/team_goal_0163.jpg) · 6.52 s | Ball beside one player's foot; rank one is at a different player's foot and M23 is also on a different location. |
| [southampton 14](evidence/cv_m24_examples/southampton_0014.jpg) · 0.47 s | Ball by the goalkeeper is missed; the top candidate is above the pitch near graphics/crowd. Exact distractor identity is uncertain. |
| [southampton 87](evidence/cv_m24_examples/southampton_0087.jpg) · 2.90 s | Bright ball/foot region is difficult in the source crop; the highest FootAndBall candidate and M23 center lie near different stadium regions. Occlusion cause is not established. |

There are **zero accepted FootAndBall false positives in these scored sets**; a claimed accepted-FP example would be invented. Examples show real misses, correct selections, lower-ranked-object confusion and M23 false positives. Uninspected visual causes remain explicitly uncertain. Possible scale/domain mismatch is an inference, not a demonstrated causal explanation.

## Runtime on this Mac

Apple **M4, 16 GiB RAM**, Torch 2.15.0, OpenCV 5.0.0. Float32, four Torch CPU threads and two OpenCV threads. MPS ran all frames without fallback. Safe initialization of the first full-video model took **76.68 ms**. Below, full processing includes decoding, normalization, all three neural heads, synchronization, finite checks, candidate decoding, hashes, compressed heatmap writes and JSONL writes; it excludes initial model loading and human-reference evaluation. RSS is the cumulative process high-water mark, not dedicated GPU memory.

| Source/mode | Frames | Full wall s | Full FPS | All-in inference mean / median / p95 ms | Pre / neural / post mean ms | Peak RSS MiB |
|---|---:|---:|---:|---|---|---:|
| giroud/native | 450 | 6.329 | 71.10 | 11.83 / 8.33 / 16.63 | 1.30 / 8.71 / 1.21 | 490.3 |
| test_match/native | 541 | 23.169 | 23.35 | 32.59 / 32.44 / 33.53 | 6.15 / 23.94 / 1.52 | 604.8 |
| team_goal/native | 500 | 7.348 | 68.04 | 12.03 / 11.54 / 19.68 | 1.47 / 8.65 / 1.21 | 604.8 |
| southampton/native | 660 | 23.087 | 28.59 | 27.09 / 26.79 / 28.37 | 4.94 / 19.26 / 1.99 | 629.5 |
| giroud/upscale2 | 450 | 15.865 | 28.36 | 27.51 / 27.46 / 29.16 | 5.45 / 19.25 / 1.89 | 629.5 |

Paired profile: frozen Giroud frames 0/225/449, one warmup per frame and ten timed repetitions each/device. CPU mean all-in inference **20.04 ms** (median 19.71, p95 22.17), neural mean 18.52 ms. MPS mean all-in **11.30 ms** (median 10.79, p95 15.27), neural mean 7.53 ms. CPU/MPS warm-profile initialization was 6.38/27.17 ms. Profile process peak RSS was 705.14 MiB, including both devices and the official compatibility model. Paired preprocessing/postprocessing means were CPU 1.24/0.27 ms, MPS 1.82/1.41 ms; host→device transfer and synchronization overhead also appear in the all-in total.

The adapter exactly matches **all three official CPU outputs**, max absolute error 0. CPU/MPS heatmap max absolute difference is **5.588×10⁻⁹**. Official parity is on the same padded tensor, not a claim that padding has no border effects. MPS unsupported-operation fallback records its reason and reruns on CPU; actual fallback was not needed on this Mac.

The earlier **1.84 FPS M23 Giroud rate includes people/semantic perception, guided detector work and associated full-pipeline stages**. FootAndBall's 71.10 FPS on Giroud is detector-video processing only; it does not replace those tasks. Some short-clip detector rates exceed source FPS, while test_match's 23.35 FPS is below its 23.976 FPS source. This is not a real-time whole-TactiqAI or full-match performance claim.

## Read-only comparison interface

Open `http://127.0.0.1:5173/?benchmark=m24&recording=giroud`, or choose **M24 detector benchmark** from a short-clip workspace. Choose any of the four recordings and a reference set. Find correct detections, misses, false selections, FootAndBall wins, M23 wins or both-fail cases. Empty categories stay empty. Frame numbers are source frame numbers. Both panels show the **same actual frame**; left has FootAndBall ranked centers/confidences, optional heatmap and human reference, right has the frozen observed M23 center. The optional original video supplies context.

Heatmaps are mapped from stride-four cell centers to original pixels and cropped before display; padding is never stretched over the image. Below-threshold candidates are labelled separately from selected observations. The interface has **no annotation/save route or controls**. It does not alter player tracking, pitch displays, event rules, Azure readiness or Ask grounding. Missing/corrupt M24 configuration or pinned evidence returns an optional benchmark diagnostic, while ordinary recordings remain usable. Served source pixels and heatmaps are hash-checked.

## Validation and live application

- Backend: **1,156 tests run, 1,155 passed, one skip** in the complete suite; **13 focused M24 tests** also passed after adding explicit stale-evidence and unsupported-MPS fallback coverage. The first restricted run failed three local-HTTP fixture binds; rerun with localhost sockets permitted passed. No implementation failure was hidden.
- Frontend: **189 tests passed across 30 files**, including three new benchmark cases. Strict TypeScript and final production build passed. Existing Vite chunk-size advisory remains.
- Actual Chrome: M24 comparison passed in 8.3 s, checking all four clips, source frames, mapped heatmap, case navigation, correct-case display, original-video playback and **zero annotation writes**; the earlier attempt found an ambiguous test locator for the two images, corrected to exact matching. Existing completed Giroud/M23 browser check passed in 7.9 s. No full legacy browser-suite success is claimed.
- Live Azure-backed backend: `/health` **200**, `/ready` **200 / ready / 16 recordings**, catalog **12 Azure-ready canonical analytics plus four local CV recordings**. All four legacy M23 metadata/frame/reference/grounded Local Ask paths **200**, video ranges **206**. Azure `118575_clip` view **200 / 481 events**, grounded Local Ask **200 / 14 evidence facts**, video range **206**. No hosted Gemini payload was sent. Both servers remain available on 8000/5173.
- Official checkpoint loads strictly; all three heads match the upstream CPU forward exactly; real MPS/CPU results are finite and almost identical. Padding/resizing, NMS/candidate ordering, one-to-one assignment, uncertain exclusion, observed-only baseline scoring, missing/stale optional evidence and fallback diagnostics are covered.


The original M19, withdrawn M21, completed M22 and Giroud label files remain byte-identical, as do active M21/M22/M23 outputs and original sources. GPU inference writes only the new ignored M24 paths. No Azure writes, environment-variable renames, media deletions, annotation submissions, new training or hosted LLM payloads occurred. All CV sources/checkpoints/sidecars remain local; canonical SoccerTrack media/analytics remain Azure-backed. The M24 benchmark reads pinned local copies of the approved short clips.

## Reproduction and storage

Use the existing runtime, `~/Library/Caches/TactiqAI/backend-m17/bin/python`. The actual run paths below are already populated; use **fresh versioned output paths** to rerun, preserving frozen evidence. `outputs/m24` is an ignored symlink into `~/Library/Caches/TactiqAI/m24`, outside iCloud. Models, original videos, raw annotations and full heatmaps are not committed. Git contains code, protocol, optional registry, aggregate evidence and selected small figures only.

```sh
git clone https://github.com/jac99/FootAndBall /tmp/tactiq-footandball-upstream
git -C /tmp/tactiq-footandball-upstream checkout 01261092a93de55026745e532a2e12188ad86801

# Run from the repository root with the existing backend Python:
python -m scripts.prepare_cv_m24 --output outputs/m24/inputs_v1 --upstream /tmp/tactiq-footandball-upstream
python -m scripts.run_cv_m24 --inputs outputs/m24/inputs_v1 --output outputs/m24/perception_v1
python -m scripts.reference_cv_m24 --inputs outputs/m24/inputs_v1 --output outputs/m24/m22_reference_v1
python -m scripts.profile_cv_m24 --inputs outputs/m24/inputs_v1 --output outputs/m24/runtime_v1.json --upstream /tmp/tactiq-footandball-upstream
python -m scripts.evaluate_cv_m24 --inputs outputs/m24/inputs_v1 --perception outputs/m24/perception_v1 --m22 outputs/m24/m22_reference_v1 --output outputs/m24/evaluation_v2
python -m scripts.install_cv_m24 --inputs outputs/m24/inputs_v1 --evaluation outputs/m24/evaluation_v2
python -m scripts.evidence_cv_m24 --inputs outputs/m24/inputs_v1 --evaluation outputs/m24/evaluation_v2 --output docs/evidence/cv_m24_examples

scripts/start_backend_m23.sh
# Separate terminal:
npm --prefix frontend run dev

python -m unittest discover -s tests
npm --prefix frontend test -- --run
npm --prefix frontend exec -- tsc --project frontend/tsconfig.json --noEmit
npm --prefix frontend run build
npm --prefix frontend exec -- playwright test --config frontend/playwright.m24.config.ts
npm --prefix frontend exec -- playwright test --config frontend/playwright.giroud-completed.config.ts
python -m scripts.verify_cv_m23_serving --recording giroud --recording test_match --recording team_goal --recording southampton --output outputs/m24/serving_verification_v1.json
```

These short-clip pipelines require their actual local source caches. Cloning GitHub alone does not recreate ignored evidence; missing M24 artifacts disable only this benchmark. Existing Azure settings and credentials are sufficient for the unchanged main application. **No Azure setting change or additional review is needed to inspect M24 here.**

## Recommendation and bounded follow-up

**Scenario D for deployment, with Scenario C motivating a possible future data experiment.** Reject this pretrained checkpoint as a replacement or useful general Giroud candidate source. Its Giroud top-five recall is too low to justify immediate M23 integration; developer-frame top-five recall of 21–29% is also insufficient. Do not lower thresholds or cherry-pick the handful of test_match successes to describe an improvement. Raw M22 does better on Giroud, but remains poor and is only a secondary temporal reference.

If another milestone is authorized, a bounded FootAndBall fine-tuning feasibility experiment is reasonable because initialization is cheap and some test_match balls are recognized. It is a hypothesis, not the current recommended production model. Proposed scope:

1. Preserve this official checkpoint/raw benchmark as control. **Exclude all Giroud frames and labels from fitting, early stopping and threshold selection** if Giroud remains a frozen evaluation recording. Giroud has already informed model choice, so future claims still need a separately frozen, genuinely unseen source for independent final testing.
2. Reuse only approved, rights-appropriate existing material: original M19 visible/absent references and completed M22 development annotations with their explicitly rejected regions. The 150-frame M22 set has 145 positives, five absent frames and 372 rejected regions; do not turn unknown surroundings into negative examples. Keep original labels unchanged and version any derived conversion. Train center targets, not artificial 40-pixel extent labels.
3. Audit the existing 120 approved SoccerNet weak source rows and their temporal contexts before using any frames. The fixed checkpoint metadata identifies `SN_game6`, weight 0.15, no human-confirmed ball supervision and no established original-match overlap audit. Organizer positives are weak evidence; they cannot certify unreviewed background or independent ground truth. SoccerTrack reconstructed NPZ/GSR points are not automatically visible-ball image labels. Local access to footage is not a redistribution or commercial license. No additional dataset is currently downloaded or approved by this milestone.
4. Split by original match and source, with neighboring frames kept together. Use broadcast small-ball scale/blur augmentation and known boot/graphic hard negatives only in training folds. Match ball-size distributions; do not tune resize modes on Giroud. Establish a small fixed compute budget and compare the same frozen center criterion and source-level validation to the unchanged official baseline.
5. Proceed to candidate integration only if that separate experiment improves independent candidate availability and false-positive control. More precise temporal/3D modeling cannot recover a ball that the visual model never recognizes. No fine-tuning or tracker change has begun.

### Attribution and licensing

FootAndBall code is MIT, copyright 2020 Sport Algorithmics and Gaming. The unchanged official FPN, adapted model hierarchy, [license](../src/vision/footandball24/LICENSE.FootAndBall.txt) and [notice](../src/vision/footandball24/NOTICE.md) are included. Strict peak suppression is independently implemented with equivalent behavior, rather than copying upstream's Kornia-derived NMS. The official checkpoint stays ignored and is hash-pinned; training data/footage terms are separate and were not independently licensed by this experiment. No dataset, full video, raw human-label store or checkpoint is published.

**M24 stops here.** The deliverable is verified experimental evidence, including a negative result; M25 and fine-tuning require a subsequent request.
