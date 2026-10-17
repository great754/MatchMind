> Historical product-milestone verification. Current TactiqAI status: 251 tests pass, configured Microsoft REST has saved live success, and TLS hardening is unit-tested with a new live recheck pending explicit approval. See [current hardening](microsoft_hardening.md).

# Microsoft-native demo verification — 2026-10-14

The milestone began with 176 passing tests. Current suite: **216 discovered / 216 passed / 0 failed / 0 skipped**. All previous dirty work was preserved. Production detector thresholds were not changed. Both required current-pipeline documents describe the resulting implementation.

| Capability | Status | Evidence |
|---|---|---|
| Artifact FastAPI service | IMPLEMENTED AND VERIFIED | Real loopback HTTP smoke script, plus route/schema tests |
| Local Ask and eight-turn demo | IMPLEMENTED AND VERIFIED | `outputs/demo_script.json`, synthetic tests and report script execution |
| Container | IMPLEMENTED AND VERIFIED | Built and ran non-root image; actual HTTP endpoint smoke passed |
| Microsoft resource REST provider | IMPLEMENTED BUT MOCK-VERIFIED ONLY | Closed schemas, timeouts, malformed-response and fallback tests; no Azure configuration |
| Optional Agent Framework runtime adapter | IMPLEMENTED BUT MOCK-VERIFIED ONLY | Code invokes Agent/FoundryChatClient and existing dispatcher; test uses a fake SDK, real SDK absent |
| Gemini query and commentary | IMPLEMENTED BUT MOCK-VERIFIED ONLY for hosted calls in this milestone | Same canonical facts/closed selector and failure tests; no new Gemini network calls |
| Local ArtifactStore/bundle export | IMPLEMENTED AND VERIFIED | Missing/corrupt/path-traversal/symlink tests, real artifact packaging |
| Azure Blob store | IMPLEMENTED BUT MOCK-VERIFIED ONLY | Hash verification, cache, corrupt manifest and network-failure tests |
| Container Apps | PREPARED BUT NOT DEPLOYED | Bicep/script/env references; no Azure CLI/subscription configured |
| Fabric analytical CSVs | IMPLEMENTED AND VERIFIED locally | Match 3, Player 73, Event 11,707, Transfer 905, TacticalState 60,510, Evaluation 4 |
| Fabric workspace import | PREPARED BUT NOT DEPLOYED | Explicit-schema Delta importer; no workspace access |
| Azure MCP | PREPARED boundary only; real integration NOT IMPLEMENTED | Operator-only read policy; no server connected/executed |

Docker verification used `matchmind-service`, reported **94,649,434 bytes**, UID **10001**, port **8000/tcp**, and approximately **4.99 seconds** until readiness on this machine. Image size/startup are local measurements, not cloud benchmarks. The local image has no Azure SDK; use the optional `WITH_AZURE=true` build for Blob deployment. Artifacts were mounted read-only and the verification container was stopped afterward.

## Architecture and implementation choices

Offline video/data processing still produces deterministic artifacts. Online serving loads those artifacts at startup; neither HTTP requests nor models rerun video analytics. `ArtifactStore` separates local bundles from hash-verified Azure Blob snapshots. Only allowlisted serving files are packaged; raw tracking, video, credentials and commentary caches are excluded.

`ask()` calculates the existing canonical answer and validates any agent-selected tools. An unrelated tool plan falls back. All presentation providers receive the same canonical facts, and can select only existing statement indices. Model-authored prose and new numeric values are rejected. Responses include structured evidence, source fields, event/timestamp references, uncertainty categories, limitations, actual provider and fallback state.

Follow-ups carry only recording/event/player/time/team/query references. References must resolve in the current bundle; cross-recording and missing references produce unknown. Caller state is never treated as new football evidence. “Take me there” returns the verified time. The report seeks the pitch/video and highlights an actor for player answers. Offline HTML uses its embedded catalogue; service-served HTML connects to the same-origin API and permits provider selection. API outages retain offline catalogue answers. Explain cards and cached commentary remain local.

Observability uses server-generated request IDs, route templates, selected provider/query/tools, request/provider latency, fallback and unknown counts. It does not log question text, prompts, headers or keys. Run Uvicorn with `--no-access-log` to avoid raw URLs in its separate access log. POST bodies are bounded to 16 KiB and questions to 2,000 characters. Optional service Bearer authentication protects match routes. Hosted browser authentication and distributed rate limits remain deployment work.

## Commands and demo

```bash
.venv/bin/python -m unittest discover -s tests -t . -v
.venv/bin/python -m uvicorn src.serving.app:from_environment --factory --host 127.0.0.1 --port 8000 --no-access-log
# Browser: http://127.0.0.1:8000/matches/118575_clip/report
.venv/bin/python scripts/verify_service.py
.venv/bin/python -m scripts.demo_matchmind
.venv/bin/python -m src.query --recording outputs/analytics_118575_tactics --question 'Who was the fastest player?' --mode analyst

docker build -f Dockerfile.service -t matchmind-service .
docker run --rm -p 127.0.0.1:8000:8000 -v "$PWD/outputs:/app/outputs:ro" matchmind-service
.venv/bin/python scripts/verify_container.py

.venv/bin/python -m scripts.refresh_report_ui
.venv/bin/python -m src.artifacts
.venv/bin/python -m src.fabric
```

Microsoft: configure server-side `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_DEPLOYMENT`, `AZURE_OPENAI_API_KEY`, then:

```bash
MATCHMIND_PROVIDER=microsoft .venv/bin/python -m uvicorn src.serving.app:from_environment --factory --host 127.0.0.1 --port 8000 --no-access-log
.venv/bin/python -m scripts.demo_matchmind --provider microsoft
```

No live Microsoft request succeeded because no resource configuration/credentials were available. Agent Framework did not actually run. Setup, optional SDK, artifact upload, Bicep deployment, Fabric import and MCP boundaries are detailed in [deployment guide](../deploy/azure/README.md). A future operator with credentials should run one minimal grounded Microsoft request before claiming live verification.

The configured local clip media route was also verified with an actual HTTP byte-range request. Set MATCHMIND_MEDIA_ROOT=data/soccertrack/mot/clips when launching the local service for video playback. The eight-turn local demo is verified: fastest actor at **171.28 s**, player follow-up; most difficult completed transfer at **78.4 s**, difficulty explanation and seeking; Team 1 pressure peak at **21.0 s**; estimated formation timeline; truthful unknown for coaching instructions. These are deterministic artifact answers, not verified jersey identities or tactical intent.

## Scientific status and next milestone

Preserved development metrics: transfers **706 TP / 161 FP / 388 FN**, precision **81.430219%**, recall **64.533821%**, F1 **72.004080%**. Shots **5 TP / 41 FP / 16 FN**, precision **10.869565%**, recall **23.809524%**, F1 **14.925373%**. Metrics were read from the existing frozen evaluation artifacts; this product milestone did not rerun or tune detectors. Projection disagreement remains mean 11.6831065 m, median 10.7398110 m, P95 24.9560484 m; GSR stays primary.

No additional match is installed: cross-match generalization remains unknown. BAS-anchored interpolated 2D ball data prevents independent validation; possession, formations, pressure and difficulty remain heuristic/inferred; full-half substitutions can leave formation unknown. No ball height, verified identities, causal tactical claims or authoritative score are inferred.

Next milestone: configure a real Foundry deployment and verify the grounded request; resolve/pin and run the optional Agent Framework SDK; deploy the verified service with real Blob/Key Vault/ingress authentication and media hosting; import/verify the Fabric snapshot. Keep independent held-out match acquisition separate and freeze detector configurations first.
