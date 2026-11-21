# Giroud — completed human review and frozen-model evaluation

Verified **2026-11-21**. **All 40 answers were received**, validated and copied byte-for-byte into a separate immutable evaluation snapshot. There are **31 visible-ball references, nine uncertain frames, zero absent-ball references, 16 accepted proposals, 15 manually drawn visible-ball boxes and 49 rejected proposal regions**. No questions are missing. Original labels and model outputs are unchanged.

[Open Giroud results](http://127.0.0.1:5173/?recording=giroud&ball_pipeline=m23) · [Review/correct your answers](http://127.0.0.1:5173/?review=m23&recording=giroud) · [Aggregate evidence](evidence/cv_giroud_completed_results.json) · [Verified results display](evidence/cv_giroud_completed_workspace.jpg).

## Measured results

Matching uses the **existing IoU ≥ 0.25 rule** on the selected current visual box. Predictions and thresholds were frozen before this review. Uncertain frames are excluded from known-visibility metrics. A wrong selected box on a visible-ball frame contributes both one false detection and one missed ball.

| System | Correct / false / missed | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: |
| M23 | 1 / 6 / 30 | 14.3% | 3.2% | 0.053 |
| Fresh same-source M21 control | 1 / 6 / 30 | 14.3% | 3.2% | 0.053 |

The denominator for recall is **31 visible-ball frames**. Precision uses the seven selected model boxes in the known-visibility sample. These are sparse, temporally correlated, **model-assisted development measurements**; there is no independent final-test claim. The nine uncertain answers remain uncertain. No specificity or absent-ball false-positive rate can be measured from this review because no frame was labelled absent.

| Selection subset | Tasks / visible / uncertain | M23 correct / false / missed |
| --- | ---: | ---: |
| Source-spread four-frame sequences | 32 / 24 / 8 | 0 / 2 / 24 |
| Post-hoc diagnostic examples | 8 / 7 / 1 | 1 / 4 / 6 |

M21 has the same subset counts. A local diagnostic spot-check inspected frames 60, 65, 102, 127, 179, 192, 256, 303, 322, 382, 386 and 432. No human box was edited. Small-box localization affects matching: frame 102's selected box has **IoU 0.217**, below the fixed criterion. The matching threshold was retained.

The ball-only review supplies no completed-pass/non-pass references. The existing pipeline still has zero qualified pass estimates on Giroud; pass precision/recall, goals, shots and pass accuracy cannot be inferred from these ball labels.

## Where the 30 misses occur

| Failure category | Visible-ball frames |
| --- | ---: |
| No matching box in the retained raw M23 proposal pool | 12 |
| Matching proposal exists below the fixed visual threshold | 4 |
| Eligible matches exist only as complementary proposals; gap policy does not admit one | 12 |
| Eligible primary proposal exists but offline association does not select it | 2 |

The saved pool contains a matching proposal in **19/31** visible frames, with an eligible matching proposal in **15/31**. One is selected correctly and fourteen are missed. These are diagnostic pool counts, not an oracle detector result or a deployable label-guided selection algorithm. The retained pool follows the existing candidate merging, retention and complementary current/temporal gates; it is not every raw neural activation.

The twelve complement-only cases identify a concrete integration limit. M23 permits complementary evidence to bridge short gaps between compatible primary anchors; it cannot independently establish a new track. Two more cases have primary evidence that the offline path does not retain. Improving only the detector would leave these association losses unresolved.

Next development should test **temporally verified complementary-track acquisition**, while preserving current-pixel confidence checks, ambiguity handling, appearance/scale checks and scene/player false-positive rejection. It also needs better small-ball proposal coverage on the twelve missing-pool cases and stronger visual evidence for the four low-score cases. Candidate admission must be evaluated on the original clips and Giroud before any promotion. More manual review is not required to establish these current development priorities.

## Preserved provenance and frontend behaviour

Evaluation root: `outputs/m23/giroud_v1/review_evaluation_v1/`.

- `snapshot/tasks.json` and `snapshot/labels.json`: byte-identical copies of the actual published tasks and saved human labels, retained locally with read-only file permissions.
- `report.json`: complete aggregate comparisons, source/model/annotation hashes and separate selection subsets.
- `failure_traces.json`: per-reviewed-frame diagnostic categories, retained locally. Raw annotation stores remain outside Git.
- `live_verification.json`: actual service checks after installing the score display.

The exact evaluation report path and SHA-256 are pinned in the Giroud entry of `configs/serving/cv_m23.json`. The original inference report and its hash are unchanged. Serving validates the evaluation's source, prediction revisions and annotation snapshot, and recomputes both systems' metrics from that snapshot before exposing scores. A corrupt optional evaluation is diagnosed without disabling valid model playback. Later live-label corrections mark the displayed frozen scores as stale until a fresh evaluation is prepared.

The frontend now shows **Results from your completed ball review**, with 40/40 receipt, the 31/9 visibility split, both score rows and expandable failure categories. It replaces the earlier unlabelled-results message. The original three clips retain their existing development tables. The source video, ordinary M14 default, experimental pass rules and Ask grounding are unchanged.

Verified: **26 targeted backend tests**, **186 frontend tests**, typed production build and a Chrome check of real 40/40 resume, measured table, source playback and unchanged annotations. Backend health/ready/catalog return 200; readiness remains **16** sources. Giroud and all three original M23 clips pass metadata/detail/reference, source range and grounded Local Ask checks. Azure `118575_clip` view/Local Ask/video remain 200/200/206, with 481 view events and 14 Ask evidence items. The build retains its existing large-chunk advisory.

No training, new neural inference, threshold relaxation, Azure upload or model-output rewrite occurred. Earlier M19/M21/M22 labels remain byte-identical. M24 has not begun.

## Commands

For a fresh evaluation after correcting labels, use a **new** output directory:

```sh
"$HOME/Library/Caches/TactiqAI/backend-m17/bin/python" \
  -m scripts.evaluate_cv_source_review \
  --output outputs/m23/giroud_v1/review_evaluation_v2
```

The evaluator defaults to the Giroud source, tasks and live label store. It refuses incomplete reviews, unadjudicated multiple reviewers and reuse of an existing evaluation directory. A new report must be validated and its exact SHA-256 registered before serving it; it never replaces the active model or silently publishes newly calculated scores.

```sh
"$HOME/Library/Caches/TactiqAI/backend-m17/bin/python" -m unittest \
  tests.test_cv_source_evaluation tests.test_cv_source_review \
  tests.test_cv_m23_serving tests.test_cv_source_m23
frontend/node_modules/.bin/playwright test \
  -c frontend/playwright.giroud-completed.config.ts
```

The normal service command remains `scripts/start_backend_m23.sh`; frontend is `npm --prefix frontend run dev` in a separate terminal. Both are running. Keep the local CV cache for source playback and annotation snapshots.
