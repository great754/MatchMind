# Existing-system intelligence milestone — 2026-10-17

Baseline: `fed136e`; this milestone extends the existing query/evidence, review, evaluation and React components. No CV implementation, production detector changes, Azure writes, infrastructure deployment or external model calls.

## Implemented and locally verified

`src/query/intelligence.py` derives a deterministic post-match catalogue from existing grounded tools: completed observable transfers/pass share, attributed control, peak tracked speed, measured distance, strongest repeated connection, difficult completed transfer, geometric pressure and trajectory-candidate counts when supported. No artificial timestamp is assigned to whole-clip aggregates. Missing/unknown evidence is omitted. Formation claims are omitted because these clips' four-minute window estimates are unknown. Shot candidates are explicitly not confirmed shots. The existing timeline still exposes canonical tactical events and states.

Each insight includes closed scalar facts, canonical source locators, recording-scoped IDs, classification, optional timestamp/actor/event and three closed Answer presentations with identical underlying facts. Existing language renderers and display-name mapping remain authoritative. Beginner/Fan/Analyst retain their existing evidence density and limitation text; the underlying metrics never change. Providers can retrieve the same deterministic catalogue through `get_match_intelligence`; `Summarize this match` routes through the closed tool registry. No new model call is needed.

The additive `intelligence` field on `/matches/{id}/view` feeds **Key Match Insights**. Expanding a card exposes grounded sentences; Review and existing evidence timestamp controls reuse the video/pitch seek path and supported actor/event selection. Analyst shows canonical source locators. Older view fixtures can omit this additive field. No browser football analytics were added.

## Ten-clip frozen evaluation

Reproduce after existing Azure snapshots have been validated/downloaded:

```sh
/tmp/tactiqai-azure-verify/bin/python -m src.evaluation.artifact_quality \
  --bundles /tmp/matchmind-azure-startup-cache \
  --exports outputs --output outputs/intelligence_quality
```

Outputs: `quality.json`, `quality.html` (expandable shot evidence) and `demo.md`. These are local generated reports, excluded from source publication. The CLI reuses manifest/hash/schema validation, existing Hungarian one-to-one event matching and micro aggregation. It rejects duplicate or overlapping same-match/half frame intervals. It does not average per-clip F1. Optional diagnostic CSVs are accepted only beside summary/events bytes identical to the canonical bundle; diagnostic hashes are recorded and recomputed counts must agree with the frozen canonical baseline. No source video or detector rerun is required.

All reference agreement is **DEPENDENT VALIDATION**: BAS anchors contribute to ball reconstruction. OBSERVED coverage means supplied tracking availability, not independent visual accuracy. INFERRED control/transfers, HEURISTIC pressure/formation/difficulty, REFERENCE-SUPPORTED timing/sender agreement, DEPENDENT VALIDATION metrics and UNAVAILABLE projection/intent evidence remain distinct. Four zero-shot-reference clips have null shot precision/recall/F1 and are excluded from eligible shot precision counts.

| Clip | Supplied ball coverage | Dependent transfer F1 | BAS shot refs | Dependent shot F1 | Alignment residual px |
|---|---:|---:|---:|---:|---:|
| 117092_clip | 88.9% | 0.688 | 2 | 0.333 | 57.25 |
| 117093_clip | 69.5% | 0.680 | 3 | 0.000 | 18.35 |
| 118575_clip | 71.5% | 0.719 | 0 | unavailable | 14.24 |
| 118576_clip | 64.1% | 0.641 | 3 | 0.000 | 14.42 |
| 118577_clip | 55.0% | 0.750 | 2 | 0.571 | 15.58 |
| 118578_clip | 77.4% | 0.600 | 2 | 0.000 | 25.24 |
| 128057_clip | 64.6% | 0.658 | 1 | 0.000 | 17.44 |
| 128058_clip | 76.3% | 0.624 | 0 | unavailable | 17.61 |
| 132831_clip | 100.0% | 0.304 | 0 | unavailable | 18.82 |
| 132877_clip | 100.0% | 0.773 | 0 | unavailable | 14.54 |

Micro transfer results: 391 detections, 516 references, 293 matches, 98 FP, 223 FN; precision **0.749361**, recall **0.567829**, F1 **0.646086**. Absolute matched timing mean **0.133106 s**, median **0.08 s**, p95 **0.52 s**.

Shot results: 17 candidates overall; 14 candidates in six reference-containing clips, 13 references, 3 matches, 11 FP, 10 FN; precision **0.214286**, recall **0.230769**, F1 **0.222222**. The other three candidates occur in four zero-reference clips and remain unvalidated. All ten clips' long-window formation estimates are unknown; no forced formation insight. Alignment residuals use local sync files only when hashes match canonical source provenance; these pixel residuals are not independent pitch-coordinate accuracy. Independent projection error is unavailable.

## Frozen shot diagnostics

All 21 eligible FP/FN errors have timestamped saved candidate/reference-window/ball context and explicit unavailable evidence. Four have nearby PASS/HIGH PASS context; one has CROSS context. Four reference windows have zero usable ball coverage. All 21 have reconstructed-ball context. These labels overlap and are hypotheses, not verified error causes or independent shot-intent labels. Original unrounded persistence/gate traces, independent intent, verified attacking-direction and camera/calibration errors are unavailable from these frozen exports. No production threshold or detector behavior changed; no proposed correction is justified by this evidence alone.

## Canonical observation boundary

`src/observations/schema.py` defines closed, finite frame/fps/time-aligned recording observations with source offset, nullable half and explicit corner-origin metre convention; stable actor/track IDs distinct from person/jersey identity; nullable metric pitch/image player and ball positions; optional team/role; separate position/identity quality/confidence/provenance; and optional time-bounded calibration with landmarks, transform and validity region. Unknown coordinates, identities, teams and calibration remain null. Duplicate frame actor IDs, invalid clocks/confidence/nonfinite coordinates and reversed/out-of-range calibration are rejected.

`src/observations/soccertrack.py` streams the existing analytics Recording into this boundary; `analytics_arrays` proves the fixed-array conceptual input round-trip including NaN gaps. SoccerTrack ball coordinates remain labeled reconstructed; unknown identity/calibration confidence is never invented. Production loading is intentionally unchanged. A future CV provider should emit these same observations, including nulls when calibration/ball/identity is unavailable, followed by an explicitly tested conversion into existing Recording inputs. No provider/model was implemented here.

## Evidence-ranked 90-second demo

The computed ranking chooses **132877_clip**, rather than a hardcoded default. Score: seekable insight categories, capped completed-transfer count, supplied ball coverage and dependent transfer F1. This is a demo evidence-density ranking, not an accuracy score. It has 46 completed transfers, 100% supplied ball-position coverage and transfer F1 0.772727; it has **zero BAS shot references**, so do not demonstrate confirmed-shot accuracy.

1. **0–10 s:** select `132877_clip`; choose **Local evidence**.
2. **10–25 s:** play synchronized video/pitch; inspect player movement.
3. **25–40 s:** expand Fastest tracked player; Review **03:29.00** (actor 510766, displayed using the existing tracker/name mapping).
4. **40–55 s:** ask **Which player was fastest?**; switch to Analyst and inspect facts.
5. **55–65 s:** switch to Beginner; the same facts remain grounded.
6. **65–80 s:** expand Difficult completed transfer; Review **02:21.28**, actor 518309, canonical event `evt_a5c23863f3b25a231073`.
7. **80–90 s:** open canonical provenance and limitations. Optional geometric-pressure moment: **02:12.60**.

## Verification and boundaries

Real Azure API: `/health` 200, `/ready` 200 with 10 recordings, `/recordings` ten available artifact-ready recordings, all ten view projections with mode-invariant insight facts, grounded Local summary Ask 200 for 132877 with 97 facts. Evidence: `outputs/intelligence_live_verification.json`; browser evidence: `frontend/test-results/azure-clips-results.json` and `azure-clips.png`. This verifies the new runtime projection against existing Azure bundles; no new artifact upload occurred.

Complete backend suite: initial run exposed the already-deleted optional local source video dependency. The real-video test now explicitly skips when that external MP4 is absent; it still runs/fails normally when present. Final complete suite: **354 tests ran, 353 passed, one optional local-video skip**. Actual Chrome live-Azure integration: **1 test passed in 2.5 minutes**, covering all ten video/select/play/seek/Local Ask checks plus the 132877 insight seeks (209.00 and 141.28 seconds), synchronized pitch, Analyst provenance and Beginner mode; no page errors. The local-registry browser config was not rerun; live Azure was used. Frontend: 36 tests passed; TypeScript and production build passed.

Microsoft/Gemini presentation remains the existing closed grounded selection architecture, mock-tested by the backend suite; no new live provider call is claimed. Future CV accuracy, independent shot truth, camera transforms and cross-camera identity remain unavailable. No Azure artifact changes are proposed or awaiting approval. Next step: review this contract, then define an independently labeled ordinary-video observation benchmark before choosing a CV detector.

## Files changed in this milestone

- Query/review: `src/query/intelligence.py`, `src/query/tools.py`, `src/query/dispatch.py`, `src/query/planning.py`, `src/serving/review.py`.
- Evaluation: `src/evaluation/artifact_quality.py`, `src/evaluation/shot_diagnostics.py`.
- Observation boundary: `src/observations/__init__.py`, `src/observations/schema.py`, `src/observations/soccertrack.py`.
- Frontend: `frontend/src/App.tsx`, `frontend/src/contracts.ts`, `frontend/src/styles.css`, `frontend/src/App.test.tsx`, `frontend/tests/azure-clips.spec.ts`.
- Backend tests: `tests/test_match_intelligence.py`, `tests/test_artifact_quality.py`, `tests/test_observations.py`, `tests/test_analytics.py`.
- Documentation: this document, `docs/current_pipeline_status.md`, `docs/current_pipeline_explorer.html`, `docs/evaluation.md`, `docs/frontend.md`, `docs/azure_artifacts.md`, `docs/azure_architecture.md`, `docs/intelligence.md`.

## Local startup commands

From the repository root, with the existing `.env` and Azure CLI login:

```sh
/tmp/tactiqai-azure-verify/bin/python -X pycache_prefix=/tmp/tactiqai-python-cache -m uvicorn src.serving.app:from_environment --factory --host 127.0.0.1 --port 8000 --no-access-log
```

In a second terminal:

```sh
cd frontend
nvm use 22.22.2
npm run dev
```

These commands use the environment already verified on this machine. The temporary Python environment is not a portable deployment dependency; recreate a virtual environment from requirements.txt if it is removed. No MATCHMIND_* name or Azure configuration change is required for the current application. Select **Local evidence** for the documented demo.

The complete quality JSON was regenerated and matched byte-for-byte: SHA256 `b84dd831d0624e65827ed9a02bdc15f8eaec38b09d844df874b40b7abb7da179`.

For repeatable local backend checks on this cloud-managed Desktop folder, use `python -X pycache_prefix=/tmp/tactiqai-milestone-pycache -m unittest discover -s tests`. A repeated verification run was observed waiting on a Desktop `__pycache__` read; placing generated bytecode outside Desktop avoids that cache dependency. This changes no application configuration.
