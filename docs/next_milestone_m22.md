# TactiqAI — Next milestone proposal: M22

Updated 2026-11-19. **Proposed scope only; M22 implementation has not started.**

## Decision and correction

**Do not complete the current 48 M21 review tasks.** The user reports that the selected excerpts have no relevant player/ball action and rejects the use of that source for this review. They came from the Azure Tottenham–Arsenal full-match upload, which the user had not selected for M21 review. Those tasks are withdrawn from the evaluation plan.

The selection error was choosing fixed offsets before checking source suitability and the intended source scope. A valid source hash, working video player and functioning review form do not establish that the footage contains the action needed for evaluation. Earlier browser checks established transport and interface behavior only. The user's report is the evidence for rejecting the excerpts; this correction does not claim a separate visual audit.

Keep the old task manifest, excerpts, acquisition record and any saved M21 labels as an audit trail. Do not reinterpret blank/non-action frames as a useful positive-action benchmark, or include withdrawn tasks in final acceptance. The review URL may still be accessible; this documentation update does not disable it or generate replacement tasks. The original 60 short-clip reviews remain intact and useful as development evidence.

## What M21 actually established

M21 implemented and ran 20 configurations on all 1,701 frames of the three short clips. It did **not** establish a major independently validated accuracy improvement.

| Development configuration | Correct balls / false detections / misses | Precision | Recall | Pass matches |
|---|---:|---:|---:|---:|
| M20 control | 10 / 4 / 21 | 71.4% | 32.3% | 1 / 7 |
| Offline candidate graph | 11 / 2 / 20 | 84.6% | 35.5% | 0 / 7 |
| Multi-frame + matched IMM | 15 / 8 / 16 | 65.2% | 48.4% | 0 / 7 |
| Temporal + offline graph | 17 / 9 / 14 | 65.4% | 54.8% | 0 / 7 |

These figures use 34 known ball frames, including 31 visible balls, and seven completed passes in six exhaustively reviewed windows. The 60 original annotations are development data, not an untouched final test. The withdrawn excerpts were not used to calculate this table, so their rejection does not invalidate these development measurements.

The higher-recall configuration finds four of the 11 reviewed visible balls in `team_goal`, with five false detections. It still loses the sole matched pass. M21 therefore retains M20 as its experiment control; the ordinary application default remains M14. Camera smoothing worsened internal residuals. Player alternatives did not beat M9 on the three matched continuity pairs; nine pairs were unmatched, with no different-person negatives. Real 3D geometry remains unverified.

The previous implementation verification recorded 1,061 backend tests passed with one skip, 159 frontend tests, a production build and nine browser cases passed. Those are engineering checks, not evidence of perception accuracy or review-source relevance. They were not rerun for this documentation correction.

## Recommended M22: Reliable small-ball detection and pass recovery

The priority is to detect the actual ball more reliably across the existing short clips, reject lookalikes, and preserve enough real observations to recover completed passes. Additional trajectory complexity and full-match processing should follow evidence of improvement at this basic stage.

### 1. Keep the intended source scope

Use only the existing `test_match`, `team_goal` and `southampton` videos for initial development. Keep their source hashes and original clocks. The Azure full match is outside this milestone's processing and review scope unless the user explicitly chooses it later. Do not substitute another video to fill an evaluation quota.

Any additional source for final evaluation must be explicitly selected by the user. Further samples from the three existing development videos can strengthen diagnosis, but must not be described as independent cross-match validation.

### 2. Check footage suitability before creating review work

First inspect model-free source footage within the allowed scope. Produce a small contact sheet and short-window inventory showing the source name, exact timestamps, source hash and visible content. Check for active play, actual players and potentially labelable ball visibility before generating positive-action tasks. Include a separate, deliberate sample of valid negative cases; do not discard negatives because a detector dislikes them.

Freeze selection criteria and exclusions before comparing model outputs. Keep a ledger of skipped intro/replay/title/blank/non-action windows and the reason for each. Do not select final-test frames from a model's successes or failures. Record contact-sheet inspection as source curation, not ground truth or accuracy evaluation.

The acceptance check for this stage is a source inventory that explains what the proposed review will measure. No new human-review workload is claimed ready until that check passes.

### 3. Audit the existing 60 annotations and detector misses

Preserve every original label. For the three clips, distinguish failures caused by missing neural candidates, inaccurate boxes, lookalike objects, association rejection and insufficient release/flight/receipt evidence. Keep uncertain labels excluded and reported. Use the existing `team_goal` failures as a development diagnostic rather than inventing a new final test from them.

Check whether the ball's source-pixel size, resize policy, crop coverage, blur and domain differences explain missed candidates. Keep the matching rule fixed; a box-center diagnostic cannot replace the existing IoU rule to manufacture improvement.

### 4. Improve visual detection before adding more trajectory models

Compare a small, controlled set of football-specific detector/crop options against the frozen M20 control and the strongest M21 candidates. Inspect current-pixel evidence and hard negatives such as socks, shirts, pitch markings and boards. Preserve coordinate mapping and original confidence values. Predictions and fused-frame proposals remain distinct from directly observed detections.

Fine-tune a detector only if the available permitted data supports it. Document dataset/checkpoint provenance, visibility labels, source-match exclusions, class balance, training commands and compute costs. Do not train a large model on the current sparse boxes and claim generalization. Obtain explicit authorization before downloading a large model/dataset or using paid compute.

Use source-group development splits. Report a detector-only comparison and matched association ablations so that a change in tracking cannot conceal a detector regression. Keep thresholds fixed for the comparison or report separately declared development-only operating-point studies.

### 5. Recover passes without relaxing evidence gates

Trace each of the seven reviewed completed passes through release, flight, receiver contact, player identity and team support. Explain the missing evidence for each failure. Compare the same detector inputs through M20 and M21 association, then evaluate the extra verifier separately.

Keep completed-pass estimates, possible transfers and unsupported attempts separate. Require actual observations for the existing pass gates. Do not add goals, shots, tackles, interceptions, pass accuracy or possession totals from unsupported ball tracks. Do not restore a pass by inserting an old event anchor into a new trajectory.

### 6. Request only useful additional annotations

After the source audit, prepare a compact review set where each task has a stated purpose: tightly boxed visible balls and clear absence/uncertainty; consecutive ball frames for association; same-person and different-person pairs; and complete pass/non-pass windows with enough context to judge release and receipt. Show the source names and time ranges before asking the user to review.

Preserve all previous labels and save new versions separately. Do not require the user to redo completed work. Add clear task instructions, explicit save acknowledgement and saved-task progress. A source suitability check must precede publishing a review link.

For final acceptance, acquire an explicitly chosen unseen source group with training-match overlap checked where possible. Freeze the candidate and protocol before inspecting its new labels. If labels guide tuning, classify that set as development and obtain another final test. Do not claim independent acceptance if no suitable new source has been selected.

### 7. Keep the interface and analytics grounded

Continue supporting synchronized source-video comparison, direct/inferred overlays, source seeking, player focus and experiment-isolated Ask. Keep the M20 experiment control until a replacement is justified. Preserve ordinary defaults and canonical SoccerTrack/Azure behavior unless a separately documented decision changes them.

Do not let a review-page transport test substitute for checking source suitability. Clearly mark withdrawn review sets in any future review-workflow change, preserve their saved labels and prevent accidental use in acceptance reports.

## Acceptance and deliverables

Before implementation, freeze a comparison protocol and explicit non-regression requirements. At minimum:

- Report per-clip and aggregate ball TP/FP/FN, precision, recall, F1 and sample sizes, plus uncertainty intervals and source grouping.
- Show where improved recall increases false detections; do not select by recall alone.
- Preserve M20's matched completed-pass result and ball precision before recommending a replacement; improvement should add correctly supported detections or passes without concealing another regression.
- Measure pass recall/precision only in exhaustively reviewed windows; report false passes per reviewed minute with the limited duration.
- Measure continuous association only where consecutive human references exist. Keep unmatched identity pairs and unmeasurable geometry visible.
- Do not promote on development metrics alone. When independent final evidence is unavailable or inconclusive, leave the control unchanged and state why.
- Run appropriate backend/frontend/build/browser regressions, original-label/hash checks, all three short-clip checks and existing Azure playback/grounded assistant checks after code changes.

Deliver a source-suitability inventory, immutable label/version manifest, reproducible detector/training/inference commands where applicable, controlled ablations, per-pass failure analysis, an honest comparison report and a concrete promotion decision. Update `current_pipeline_status.md` and `current_pipeline_explorer.html` with measured results and remaining work.

## Deferred work

Full-match CV, complex 3D trajectories, physical ball height/shot speed, broad player ReID changes and new unsupported football statistics are deferred. M21 already showed that more algorithms do not guarantee useful detection gains. The next milestone should resolve visual evidence and pass recovery first.

## Current handoff

No additional processing, training, task generation, label changes, Azure writes or M22 implementation occurred in this documentation update. The current 48-task M21 review is withdrawn from the evaluation plan. No replacement source or independent final test is claimed ready. Existing short-clip outputs and the original 60 development reviews remain available.

[M21 measurements and implementation](cv_advanced_perception_m21.md) · [Original human evaluation](cv_human_evaluation_2026_11_18.md) · [Current pipeline status](current_pipeline_status.md) · [Pipeline explorer](current_pipeline_explorer.html)
