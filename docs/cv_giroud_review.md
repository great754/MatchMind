# Giroud ball review — ready

Prepared and verified **2026-11-21**. [Open the 40-question review](http://127.0.0.1:5173/?review=m23&recording=giroud), or click **Review the ball** from Giroud's analysis page. Both servers are running. No Azure setting change is needed.

## What to do

1. Look at the ball in the original frame. Use **Zoom** if it is tiny. **More context & optional notes** includes the original video and neighbouring frames.
2. For each blue proposal, choose **Accept proposal** if it tightly covers the actual football, or **Not the ball** if it covers something else. There are at most two proposals per question.
3. If neither proposal is correct but you can see the ball elsewhere, drag a tight box around it in the main image. Review the remaining proposals too.
4. If the ball is hidden, choose **Ball not visible**. If you cannot determine its position or visibility, choose **Can't tell**. Rejecting both proposals alone does not mean the ball is absent.
5. A completed decision saves automatically and advances after the server confirms it. **There is no Save step.** Use **Previous frame** to correct an answer. An incomplete browser draft is retained; closing and reopening resumes at the first unanswered question for your reviewer ID.

At **40/40**, the page says **“Saved. This clip is complete. You can go back to make corrections.”** The final Next button is disabled because there are no more questions. A failed save keeps the current question and offers **Retry automatic save**.

Draw around **the ball**, not a player. Check each frame independently; you do not need to identify a player or label passes in this review. The filename does not identify a player for the model.

## What was selected

The review uses only the user-approved **Giroud** clip: 18 seconds, 450 frames at 25 fps, 640×360; source SHA-256 `66fb0421464ba8a7a91a117a8625d4ea6f4870eb88e69ac0732fbaf3c7752e5c`.

- **32 frames:** eight four-frame sequences spread across the entire source, including its beginning and end.
- **Eight diagnostic frames:** 29, 60, 102, 179, 240, 303, 386 and 432, selected after inspecting the existing model run.
- **76 proposals total**, capped at two per question. These are actual saved M23 neural candidates, with unchanged boxes and confidence. Where available, the selected candidate is shown first. Proposals are suggestions and never become labels merely by opening a page.
- Both source-only preview sheets were inspected before publication. The selected images contain football players, pitch and active match footage. This inspection created no human ball labels.

This is **model-assisted development evidence**. The diagnostic examples and model proposals influence selection, and original-match training overlap has not been independently audited. These labels can support development and measurement of this clip's failures; they are not an untouched final test set. Existing Giroud precision/recall/F1 remain unmeasured until actual human reviews are received and evaluated.

## Storage and isolation

Published tasks: `configs/training/cv_giroud_review.json`, fingerprint `ee1aa6425a483a9b348f32211cd7044b1fe3616f884ae1d3049989b5916ed089`.

Human labels will save to `outputs/m23/giroud_v1/human_review/labels.json`, in the existing persistent local cache outside iCloud. The file is created only on a real completed save. Browser reviewer IDs and drafts use their own `tactiq.m23` namespace. The old M22 tasks, labels, reviewer setting and drafts retain their original scope. Opening the new review does not prefill Giroud with earlier answers.

The source-specific store requires explicit approved-source configuration and its own versioned manifest. Task fingerprint, source hash, clock, dimensions, box bounds, proposal IDs, reviewed-proposal completeness and accepted-box consistency are checked. Correcting an answer replaces only the same reviewer's answer to that task. Other reviewers remain separate. Rejected proposal regions are local hard negatives and never automatically imply whole-frame ball absence.

No neural inference, model retraining, pipeline threshold change, model artifact rewrite or Azure upload was performed to prepare this review. The previous Giroud model results remain unchanged. Original M19/M21/M22 annotations were SHA-256 checked before and after verification and are byte-identical.

## Verified behaviour

- Review tasks, all **40 actual source JPEGs**, both source-only preview sheets and export return 200. Original video context plays.
- The analysis-page **Review the ball** link opens the Giroud review. Opening either page sends no annotation saves.
- Chrome checks exercise all 40 questions, proposal acceptance/rejection, automatic save/next, back/correction, failed-save retry, reload/resume, the final completion message and a mobile layout without horizontal overflow.
- Browser annotation writes were mocked. The live Giroud store remained empty throughout preparation. Test progress shown during automation is not human annotation data.
- Both existing M22 review browser cases passed on a clean retry with separate temporary artifacts. The initial attempt reported missing trace files during teardown.
- **45 targeted backend tests**, **183 frontend tests** and the typed production build pass. The build retains its existing large-chunk advisory.
- Backend health and readiness return 200; readiness remains **16** usable recordings. The source remains local CV; Azure analytics configuration is unchanged.

[Verified review screenshot](evidence/cv_giroud_review_workspace.jpg) · [Machine-readable verification](evidence/cv_giroud_review_results.json) · [Giroud model results and limitations](cv_giroud_m23.md).

## Commands

The published review is already installed. Restart if necessary:

```sh
scripts/start_backend_m23.sh
npm --prefix frontend run dev
```

Run these in separate terminals. Review: `http://127.0.0.1:5173/?review=m23&recording=giroud`.

Focused checks, using the installed runtime:

```sh
"$HOME/Library/Caches/TactiqAI/backend-m17/bin/python" -m unittest \
  tests.test_cv_source_review tests.test_cv_m22 tests.test_cv_source_m23
frontend/node_modules/.bin/playwright test \
  -c frontend/playwright.giroud-review.config.ts \
  --output=/tmp/tactiq-giroud-review-browser-artifacts
```

Task preparation is implemented in `scripts/prepare_cv_source_review.py`. Preparation creates a draft and source-only preview sheets; publication requires explicitly identifying both inspected sheets and a curator. Published task manifests cannot be overwritten. The existing source and model artifacts are required to recreate a fresh draft.

After the human review is complete, evaluate the actual saved labels against the frozen Giroud predictions and use the measured missed-ball/false-positive cases to prioritize further development. This preparation did not start M24 or retrain a model.
