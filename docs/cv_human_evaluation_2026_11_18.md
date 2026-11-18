# CV human-reference evaluation — 2026-11-18

All **60 user reviews are saved: 20 each for test_match, team_goal and southampton**. The frozen M14/M16/M19/M20 outputs have now been evaluated against an unchanged private snapshot. M20 remains an opt-in experiment: this sample shows a small recall gain, more false detections and no pass-detection improvement over M19. It does not establish accuracy acceptance or generalization.

## Reference receipt and eligibility

| Recording | Saved / sealed | Known ball frames (visible / not visible) | Uncertain ball frames | Eligible / total pass windows | Completed reference passes | Identity pairs |
|---|---:|---:|---:|---:|---:|---:|
| test_match | 20 / 20 | 12 (11 / 1) | 0 | 2 / 4 | 2 | 4 |
| team_goal | 20 / 20 | 11 (11 / 0) | 1 | 2 / 4 | 3 | 4 |
| southampton | 20 / 20 | 11 (9 / 2) | 1 | 2 / 4 | 2 | 4 |
| Total | 60 / 60 | 34 (31 / 3) | 2 | 6 / 12 | 7 | 12 |

The six eligible pass windows cover **18.012 seconds**. Five uncertain windows and one reviewed window without the exhaustive attestation are excluded. Saved reviews remain useful even when uncertainty prevents a metric. No uncertain label was converted to a negative, corrected by a model, or discarded to improve scores. This is one reviewer's uniformly timed development-clip sample; the independence attestations are validated, not externally audited.

## Measured ball detection

Selected **direct observations** must overlap the human ball box at IoU ≥0.25. A wrong location on a visible-ball frame contributes both a false positive and a miss. Prediction-only positions cannot match. Precision is correct detections divided by all evaluated detections; recall is correct detections divided by 31 reviewed visible balls. F1 combines precision and recall.

| Frozen version | Correct detections | False detections | Missed visible balls | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|---:|
| Original M14 | 6 | 3 | 25 | 66.7% | 19.4% | 0.300 |
| Original M16 | 9 | 2 | 22 | 81.8% | 29.0% | 0.429 |
| M19 with M16 detector | 9 | 2 | 22 | 81.8% | 29.0% | 0.429 |
| M20 | 10 | 4 | 21 | 71.4% | 32.3% | 0.444 |

M20 gains **one correct detection and two false detections** over M19. Approximate Wilson 95% intervals are wide: M19 precision **52.3–94.9%**, recall **16.1–46.6%**; M20 precision **45.4–88.3%**, recall **18.6–49.9%**. The sample does not demonstrate a statistically reliable improvement. These metrics evaluate the frozen operating points; they do not isolate every architectural difference between historical detector versions.

| Recording | M19 correct / false / missed | M20 correct / false / missed | M20 ball recall | M20 pass matches / reference passes |
|---|---:|---:|---:|---:|
| test_match | 8 / 2 / 3 | 8 / 2 / 3 | 72.7% | 1 / 2 |
| team_goal | 0 / 0 / 11 | 0 / 1 / 11 | 0% | 0 / 3 |
| southampton | 1 / 0 / 8 | 2 / 1 / 7 | 22.2% | 0 / 2 |

**team_goal remains the principal failure on the reviewed frames**: increased engineering coverage did not translate into correct detections at those timestamps. Southampton gains one match with one additional false detection. Test_match's sampled result is unchanged.

Matched-box median center error is **2.39 pixels on nine matches for M19** and **2.38 pixels on ten matches for M20**. This conditional localization result excludes missed/wrong balls and must not be presented as overall ball-position accuracy.

## Passes and player continuity

Original M14 and M16 have **zero qualified predictions in the eligible windows**, so pass precision is undefined and recall/F1 are zero. M19 and M20 each match **one of seven completed human-reference passes**: one true positive, zero false positives and six misses; recall **14.3%**, F1 **0.250**. The one matched event is the existing test_match estimate at source frames **66–83**, not an additional M20 pass. Pass matching uses one-to-one release/receipt endpoints within **0.25 seconds**, only for qualified estimates entirely inside exhaustively reviewed windows. Candidates remain excluded. This temporal match does not independently verify actor identity or team attribution.

The arithmetic precision is 1/1, but **one prediction is insufficient to claim reliable precision**; its approximate 95% interval is **20.7–100%**. Zero false positives in 18.012 reviewed seconds does not establish a low false-positive rate across the videos. Six missed passes remain the practical limitation; no gates were relaxed to increase totals.

All **12 continuity references identify the same person across their two frames**. Only **three pairs** match unique visible M9 player boxes in both frames (one test_match, two team_goal); those three agree. **Nine unmatched/ambiguous pairs are excluded**, including all four Southampton pairs. Conditional agreement 3/3 (approximate interval **43.8–100%**) is not overall tracking accuracy. There are no different-person negative pairs to measure false merges. M19/M20 share the same M9 player snapshots, so this establishes no player-ID improvement.

Sparse ball boxes do not supply continuous ball identity or reacquisition truth. Those metrics remain null. No shot/goal/tackle totals, pass-attempt accuracy or possession truth were inferred from these labels.

## Reproduction and preservation

The sealed task fingerprint is unchanged. Private references and full evaluation outputs remain under ignored `outputs/m20/human_evaluation_v1/`; raw labels, reviewer identity, notes and boxes are not published. The live label file and private reference snapshot have identical SHA-256 after evaluation. Original inference outputs and historical no-label evaluation reports were preserved. No model training, threshold tuning, video upload, canonical artifact change or long-match run occurred.

```sh
eval_python=/Users/greatabhieyighan/Library/Caches/TactiqAI/backend-m17/bin/python
# Set REVIEWER_ID to the reviewer used in your local form.
# Use fresh output paths: explicit existing targets are rejected.
"$eval_python" -X pycache_prefix=/tmp/tactiq-human-eval-pycache \
  -m scripts.evaluate_cv_m19 --root outputs/m19/run_v3 \
  --labels outputs/m20/human_evaluation_v1/reference_snapshot.json \
  --reviewer "$REVIEWER_ID" --output outputs/m20/human_evaluation_v2/baseline_results.json
"$eval_python" -X pycache_prefix=/tmp/tactiq-human-eval-pycache \
  -m scripts.evaluate_cv_m20 --root outputs/m20/run_v1 \
  --labels outputs/m20/human_evaluation_v1/reference_snapshot.json \
  --reviewer "$REVIEWER_ID" --output outputs/m20/human_evaluation_v2/m19_m20_results.json
```

The M19 script also reports M19 association on M14 detections separately from original M14; that variant is retained in the evidence. M20 reporting now distinguishes completed sample evaluation from unestablished accuracy acceptance and includes per-recording results. `--output` preserves historical measurements in both scripts.

**82 focused backend tests passed**, including independent metric rules, M19/M20 regressions, no-label null results, completed-sample reporting and refusing existing explicit outputs. Live checks: frontend **200**, backend `/health` **200**, `/ready` **200 / 15 usable recordings**, `/recordings` **200**. All three clips also return 200 for M20 metadata/events/trajectory and saved-review tasks, with source-video ranges returning 206. No frontend code or Azure settings changed for this evaluation. Keep the three local CV videos for playback. M14 remains the default; M20 is selectable under Ball review.

The next research priority is stronger small-ball visual verification and player-box coverage, especially on team_goal. These completed development labels should not be treated as an unseen validation set after future tuning. [Aggregate evidence, intervals and input fingerprints](evidence/cv_human_evaluation_2026_11_18.json).
