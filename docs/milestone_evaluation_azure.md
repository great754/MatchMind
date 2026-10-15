# Evaluation and Azure preparation milestone

Verified 2026-10-11. Existing GSR-first coordinates, goalkeeper assignment, deterministic analytics, report possession/formation display, Ask tools and Gemini support were preserved. No Azure deployment or new hosted-model call was performed.

## Reproducibility

Canonical command: `.venv/bin/python -m unittest discover -s tests -t . -v`. Phase 0 reproduced **142 discovered, 142 passed, 0 failed, 0 skipped** after moving fixtures to `tests/helpers.py`. Final suite: **176 discovered, 176 passed, 0 failed, 0 skipped**. Direct package imports also work. Tests establish implementation behavior, not football accuracy.

## Shot baseline and experiments

Both halves of 118575 are development data. Original full-precision evaluation was rerun: 46 candidates, 21 BAS references, **5 TP / 41 FP / 16 FN**, precision **0.10869565217391304**, recall **0.23809523809523808**, F1 **0.1492537313432836**. Half 1: 3/22/7, P 12.0%, R 30.0%, F1 17.1429%. Half 2: 2/19/9, P 9.5238%, R 18.1818%, F1 12.5%. The original aggregate snapshot remains in `outputs/evaluation/baseline_118575.json`; baseline configuration is `configs/shots/baseline.json`.

Ten one-factor experiments were run after diagnostics existed. Each uses tracking-only prediction inputs, then BAS evaluation. Here are all experiments; percentages are rounded for display, exact values/configs/source hashes/candidate starts are in `outputs/evaluation/shot_experiments.json`.

| Experiment | TP | FP | FN | Precision % | Recall % | F1 % | Change |
|---|---:|---:|---:|---:|---:|---:|---|
| acceleration_1 | 5 | 40 | 16 | 11.111 | 23.810 | 15.152 | Require estimated change in velocity; centered smoothing/interpolation limit interpretation. |
| baseline | 5 | 41 | 16 | 10.870 | 23.810 | 14.925 | Original production thresholds and velocity fallback; retained for comparison. |
| distance_45 | 6 | 41 | 15 | 12.766 | 28.571 | 17.647 | Expand distance gate only; may also admit long transfers. |
| forward_04 | 5 | 47 | 16 | 9.615 | 23.810 | 13.699 | Relax longitudinal velocity fraction only; inspect lateral trajectories. |
| goal_margin_0 | 5 | 23 | 16 | 17.857 | 23.810 | 20.408 | Restrict projection to the physical goal mouth; tests wider-margin false intersections. |
| no_goal_intersection | 8 | 135 | 13 | 5.594 | 38.095 | 9.756 | Ablation of goal-mouth intersection; not proposed for production. |
| persistence_1 | 8 | 93 | 13 | 7.921 | 38.095 | 13.115 | Single-frame ablation of persistence; exposes quantization sensitivity. |
| recent_required | 3 | 10 | 18 | 23.077 | 14.286 | 17.647 | Require recent team evidence instead of choosing the attacked goal solely from velocity. |
| speed_6 | 5 | 45 | 16 | 10.000 | 23.810 | 14.085 | Lower minimum estimated 2D speed only; tests whether slow reconstructed trajectories miss shots. |
| time_5 | 5 | 61 | 16 | 7.576 | 23.810 | 11.494 | Relax straight-line time-to-goal only. |
| velocity_only | 7 | 44 | 14 | 13.725 | 33.333 | 19.444 | Ablate possession direction check; diagnostic only. |

Selected production configuration: **original baseline, unchanged**. Therefore production before/after precision, recall and F1 are identical: 10.869565%, 23.809524%, 14.925373%. The narrower goal-mouth margin has better development F1, but could reject legitimate off-target shots. Requiring recent possession removes many unsupported candidates but also loses two matches. A velocity-only direction rule gains matches while discarding team direction evidence. None is selected from this single development match. This is a defensible refusal to tune to dependent reference data, not a claim that the baseline is good.

Cross/long-pass separation was investigated through post-hoc BAS context and independent direction/trajectory gate ablations. There is insufficient 2D evidence to add an intentional-shot classifier: loosening goal intersection yields 135 false positives. No BAS-class filter was added to prediction.

## Error evidence

All **62 shot diagnostic rows** (5 matched, 41 FP, 16 FN) include timing, geometry, velocity, recent possession, attack-direction source, quality, independent gate evidence and nearby actors. JSON/CSV: `outputs/evaluation/shot_error_analysis.*`. The evaluation HTML contains expandable evidence, a pitch debug view and timestamp links into each half report. Full-half video is unavailable; the pitch still seeks. FN attribution is the first failed gate on the frame with the fewest failed gates within the matching window, breaking ties by reference proximity. Exact reference-frame gates and all window failure counts are also retained. This describes rejection evidence, not a proven causal explanation.

FP hypotheses: **30 direction unconfirmed** (velocity fallback without recent team possession), **9 forward-pass context**, **1 long-ball context**, **1 unknown**. These are not established soccer intent; direction-unconfirmed does not prove wrong direction. Some true positives also use velocity fallback.

FN gate attribution: speed too low **5**; insufficient persistence **3**; no independent flight **2**; goal-mouth projection **2**; wrong direction **2**; distance gate **1**; forward velocity fraction **1**. There was no missing-ball primary attribution in these actual references; the missing-data and unknown-direction cases are covered synthetically. A single reference can fail multiple gates.

Transfer audit covers all **161 FP and 388 FN**. Overall baseline remains 867 candidates / 1,094 references / 706 matched, P **81.430219%**, R **64.533821%**, F1 **72.004080%**. Absolute release timing error mean **0.1037393768 s**, median **0.08 s**, P95 **0.40 s**.

| BAS action | Matched / references | FN | Recall |
|---|---:|---:|---:|
| PASS | 598 / 907 | 309 | 65.9316% |
| HIGH PASS | 87 / 111 | 24 | 78.3784% |
| CROSS | 10 / 26 | 16 | 38.4615% |
| THROW IN | 11 / 50 | 39 | 22.0000% |

Class precision, FP and F1 are **undefined** because candidates have no BAS class labels. Most misses numerically are PASS, while THROW IN has the weakest class recall. FN diagnostic categories: possession never acquired **218**, sender mismatch **65**, distance below threshold **58**, no released independent flight in window **19**, self recovery **14**, flight too short **13**, one-to-one competition **1**. These categories describe heuristic/reference disagreement and do not prove that acquisition failure caused a real missed pass. `transfer_error_analysis.json/csv` includes state counts, nearest action and flight audit. No transfer thresholds changed.

## Generalization and Microsoft status

Only match **118575** is downloaded. No true held-out result exists. The normal manifest labels both halves development; `development_aggregate` contains their metrics, and evaluation `aggregate` has zero recordings and null precision/recall/F1. Another-match file inventory, placement/download commands, and split procedure are in [evaluation guide](evaluation.md). `configs/evaluation.heldout.example.json` is preparation, not measured evidence.

Implemented: provider protocols, Microsoft OpenAI-v1 REST tool/statement and commentary adapters, closed deterministic dispatcher reuse, Agent Framework-compatible callable/schema boundary, provenance-preserving agent selection, safe local fallback, artifact-only FastAPI API, lightweight service requirements and non-root Dockerfile. Provider HTTP calls are mocked in tests; **no live Foundry request or SDK-hosted agent was validated**. Gemini tests and cached/local commentary remain functional. The existing offline UI does not depend on the API.

Remaining: real Foundry deployment/authentication test, optional Agent Framework SDK lifecycle, cloud media hosting, Container Apps provisioning/authentication/limits/monitoring, and optional Fabric/MCP integration. [Azure architecture](azure_architecture.md) distinguishes each role. Docker daemon was unavailable, so image build/run was not verified. API routes were tested in process; startup loads real local artifacts without importing cv2/scipy/matplotlib or the offline analytics pipeline.

## Exact local commands

Run from the repository root with the existing Python 3.14 environment. Install the full pinned dependencies for offline processing/tests; the service-only requirements omit vision/plotting/scientific-processing packages.

```bash
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m unittest discover -s tests -t . -v

# Analytics/report: separate output avoids overwriting an existing hosted commentary demo.
.venv/bin/python -m src.analytics.pipeline --match-id 118575 --scope clip --half 1 --summary-only --commentary-provider template --output outputs/analytics_118575_local_demo
open outputs/analytics_118575_local_demo/report.html

# Evaluation, then diagnostics and component experiments.
.venv/bin/python -m src.evaluation --manifest configs/evaluation.118575.json --output outputs/evaluation --reuse-sensitivity
.venv/bin/python -m src.evaluation.diagnostics --output outputs/evaluation
.venv/bin/python -m src.evaluation.shot_experiments --evaluation outputs/evaluation --configs configs/shots --output outputs/evaluation
open outputs/evaluation/evaluation_report.html
open outputs/evaluation/shot_experiments.html

# Existing deterministic Ask MatchMind CLI and report UI.
.venv/bin/python -m src.query --recording outputs/analytics_118575_tactics --question 'Which team had more possession?' --mode analyst
open outputs/analytics_118575_tactics/report.html

# Local API. Registry is administrator-controlled; aliases identify recordings, including half.
.venv/bin/python -m uvicorn src.serving.app:from_environment --factory --host 127.0.0.1 --port 8000
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/matches/118575_h1/summary
curl -X POST http://127.0.0.1:8000/matches/118575_h1/ask -H 'Content-Type: application/json' -d '{"question":"Which team had more possession?","mode":"analyst"}'
```

Microsoft is optional. Set `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_DEPLOYMENT`, and `AZURE_OPENAI_API_KEY` in the server process environment using your secret manager. `.env.example` contains placeholders only. Then:

```bash
MATCHMIND_PROVIDER=microsoft .venv/bin/python -m uvicorn src.serving.app:from_environment --factory --host 127.0.0.1 --port 8000
# Microsoft commentary uses the same closed sentence catalogue as Gemini.
.venv/bin/python -m src.analytics.pipeline --match-id 118575 --scope clip --half 1 --summary-only --commentary-provider microsoft --output outputs/analytics_118575_microsoft_demo
# Existing Gemini path is retained; credentials are read server-side from .env/environment.
.venv/bin/python -m src.analytics.pipeline --match-id 118575 --scope clip --half 1 --summary-only --commentary-provider gemini --output outputs/analytics_118575_gemini_demo
```

For a service-only environment install `requirements-service.txt`. When Docker is running:

```bash
docker build -f Dockerfile.service -t matchmind-service .
docker run --rm -p 127.0.0.1:8000:8000 -v "$PWD/outputs:/app/outputs:ro" matchmind-service
```

Default provider is local. Optional `MATCHMIND_API_TOKEN` protects match routes with a Bearer token; `/health` remains public. Do not expose a public deployment until its planned ingress authentication is configured. Serving an existing report does not regenerate analytics or commentary. Video files are not served by this API; the local offline report retains its relative source clip link. Cloud media hosting remains a separate task.

## How to understand and verify the implementation

Start with `shot_evidence.py`: each frame has independent gate failures and explicit direction provenance. `shots.py` segments eligible frames, applies persistence and truncates speed windows at receptions. `shot_diagnostics.py` joins frozen predictions to reference assignments only after detection. `shot_experiments.py` loads JSON overrides and checks original baseline start-frame reproduction before reporting variants. `passes.py` adds an optional audit collector without changing detection rules; `transfer_diagnostics.py` uses that collector post hoc.

For serving, follow `src/query/dispatch.py` → `src/query/tools.py` → `src/providers/agent.py`. The provider receives closed tool results and a deterministic statement catalogue. Final text selects catalogue entries; model prose cannot create facts. `src/serving/app.py` loads only allowlisted precomputed artifacts at startup. Unit tests use synthetic known-answer trajectories and temporary artifacts, plus mocked HTTP and an executed report browser-script harness. Real data reruns reproduce the baseline; tests alone do not validate football interpretations.

Remaining scientific limits: BAS-anchored interpolated 2D ball paths make BAS checks dependent; no ball height or measured strike speed; straight-line goal projection cannot establish intent, crosses, saves or scoring; ownership/formation/pressure/difficulty remain heuristics; full-half roster substitutions may leave formations unknown; tracking, identity, calibration and timing errors remain. Projection fallback mean disagreement stays **11.6831065 m** (median 10.7398110 m, P95 24.9560484 m). GSR remains primary. No independent tactical or possession ground truth and no held-out match accuracy are claimed.
