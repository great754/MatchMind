# TactiqAI Microsoft REST hardening

Verified local state: 2026-10-14. Existing MATCHMIND_* environment variables and module/file names are preserved. No new product feature, detector threshold change or Azure deployment is part of this milestone.

## What is verified

- Baseline: 216 tests passed. Final canonical suite: **251 discovered, 251 passed, 0 failed, 0 skipped**; 35 new tests.
- Focused provider/service suite: **92 passed, 0 failed, 0 skipped**.
- Actual local-only HTTP service smoke passed: health/readiness, known/missing match, known/unknown player and event, summary/events/explain/report, grounded/unsupported questions, malformed requests and response fact consistency. No external provider was contacted by this smoke check.
- Configuration check succeeded with SSL_CERT_FILE absent in the verification process. It performs no network request and does not establish authentication/model access.
- The Microsoft REST path is **LIVE-VERIFIED before these changes**, supported by the user's report and inspected local output.txt: fastest-player, difficult-transfer and pressure queries returned microsoft_foundry / fallback false / grounded_selection.
- **Subsequent user-supplied status: POST-HARDENING LIVE-VERIFIED.** The verifier succeeded with SSL_CERT_FILE absent, explicit certifi verification, fastest_players / get_fastest_players, microsoft_foundry / grounded_selection, fallback false and diagnostics []. This supersedes the earlier blocked check. The frontend task made no new model calls.
- Agent Framework is **NOT LIVE-VERIFIED**. Azure Container Apps is **NOT DEPLOYED**. Fabric cloud ingestion has **NOT RUN**. Azure MCP has **NOT RUN / NOT BEEN INTEGRATED**. Earlier Docker verification is historical; the image was not rebuilt during this hardening milestone.

## Automatic secure TLS

MicrosoftFoundryProvider passes `verify=certifi.where()` explicitly on each injected/default Session.post call. The path is resolved from the installed package at runtime; no developer-specific path is stored. Certificate and hostname verification remain enabled. There is no insecure retry or warning suppression. The existing pinned requirements already contain certifi for local and service/container installs.

SSL_CERT_FILE is not required by the provider. Explicit verification also prevents a false Session.verify or Requests CA environment setting from silently disabling/changing this provider's trust configuration. The current policy deliberately uses certifi's CA bundle; private enterprise roots require a separately reviewed trust-policy change, never disabled verification. Requests documents CA-bundle paths through its verify argument and certifi support. [Requests TLS documentation](https://requests.readthedocs.io/en/latest/user/advanced/#ssl-cert-verification).

Existing .env contents were not changed. If a legacy SSL_CERT_FILE entry remains there, it is unused by this provider; verification can remove it from the checking process only. No user shell/environment file is rewritten.

The API remains the existing resource REST surface: configure AZURE_OPENAI_ENDPOINT as the supported resource root or /openai/v1 base. The provider constructs /openai/v1/chat/completions. Do not set the endpoint to /openai/v1/responses. AZURE_OPENAI_DEPLOYMENT and AZURE_OPENAI_API_KEY remain server-side configuration. [Microsoft structured outputs](https://learn.microsoft.com/en-us/azure/foundry/openai/how-to/structured-outputs).

## Sanitized diagnostics

`src/providers/diagnostics.py` defines a closed MicrosoftFailure enum, fixed public-safe messages and MicrosoftProviderError metadata. The tactiqai.providers logger emits JSON containing only event, provider, code, stage and optional HTTP status. It never includes upstream response bodies, exception text, URLs/deployment names, prompts, tool arguments, headers, keys or tokens.

| Code | Meaning / next investigation |
|---|---|
| configuration_error | Missing/invalid endpoint, deployment or authentication configuration |
| tls_failure | Certificate verification failed; investigate trusted CAs/network interception |
| network_failure | Connection/transport failure |
| timeout | Connect/read deadline exceeded |
| http_400 | Request/model/API incompatibility |
| http_401 | Authentication failure |
| http_403 | Permission/access denial |
| http_404 | Endpoint/deployment/resource not found |
| http_429 | Rate/quota limit |
| http_5xx | Microsoft service failure |
| http_error | Other non-200 status; redirects remain disabled |
| malformed_response | Unexpected envelope, JSON, refusal or oversized response |
| invalid_tool_plan | Malformed function-call plan |
| invalid_selection | Invalid statement/commentary selection schema |
| request_invalid | Local request serialization/size failure |
| internal_error | Other unexpected provider failure, without raw details |

Agent fallback logs additionally distinguish tool_plan_invalid, tool_arguments_invalid, unrelated_tool_plan and selection_invalid. Only allowlisted tool names enter the response/service metrics; unknown model-supplied names are not echoed. The external Answer/fallback contract stays unchanged. Microsoft failures retain deterministic local facts and the existing generic fallback message. No retry, model-generated statistic, ungrounded prose or silent tool repair was introduced.

## The observed formation fallback

The saved output.txt contains eight demo turns. Three completed Microsoft grounded selection. Player, difficulty and seeking follow-ups remained local_followup with fallback false. Coaching instructions stayed unsupported/local with fallback false. The **formation query** selected **get_match_summary** and returned local/fallback true.

This is category **D: tool-plan/grounding validation failure**, not intentional unsupported behavior and not evidence of a TLS/network outage. get_match_summary does not provide formation timeline facts. Offline replay against the actual clip artifacts with its valid empty argument schema found **740 canonical formation facts and zero matching fact IDs** in the summary result. The agent emitted unrelated_tool_plan at tool_evidence and did not call the model selector. Evidence is recorded in ignored outputs/microsoft_hardening_routing.json.

The old output records selected tool names but not arguments or the exception. Consequently the exact earliest historical gate—invalid arguments versus the subsequent evidence-overlap check—cannot be proven retrospectively. Either way, this selected tool cannot support the formation answer; a selection HTTP failure cannot occur after a successful summary dispatch because the overlap check rejects first. New logs make the distinction explicit on future runs. Model routing has not been forced or silently repaired. The model may still choose an unrelated tool; safe fallback is the intended result in that case.

One separate metadata bug was fixed: when a hosted provider is absent, intentional contextual/unsupported local answers previously could be marked fallback true merely because Microsoft was requested. They now remain intentional local responses with fallback false. Actual provider failures on supported non-contextual questions still mark fallback true. Regression tests cover both cases.

## Verification commands

Run from the repository root:

```bash
.venv/bin/python -m unittest discover -s tests -t . -v
.venv/bin/python -m unittest tests.test_microsoft_hardening tests.test_providers_service tests.test_product_milestone -v

# Safe configuration-only check; no Microsoft request, no secrets printed.
.venv/bin/python -m scripts.verify_microsoft --without-ssl-cert-file

# Actual local HTTP smoke; explicitly avoids hosted providers.
MATCHMIND_ARTIFACT_STORE=local MATCHMIND_PROVIDER=local MATCHMIND_API_TOKEN= .venv/bin/python scripts/verify_service.py --local-only
```

After explicit approval to send the bounded query/tool schemas and canonical fastest-player facts to the configured Microsoft endpoint:

```bash
.venv/bin/python -m scripts.verify_microsoft --live --without-ssl-cert-file
```

The optional verifier prints configuration validity, CA policy, whether SSL_CERT_FILE is present, actual provider/fallback status and sanitized diagnostics. It does not print endpoint/deployment/key values, question text or football fact values. Exit 0 means configuration valid in configuration-only mode, or grounded Microsoft success in live mode; 1 means no verified live grounded selection; 2 means setup/configuration failure. It is not required for application startup. `--without-ssl-cert-file` removes only that variable in the verifier process after loading .env, leaving the file and parent shell unchanged.

Normal local startup still works with the existing variables:

```bash
MATCHMIND_PROVIDER=local .venv/bin/python -m uvicorn src.serving.app:from_environment --factory --host 127.0.0.1 --port 8000 --no-access-log
# Microsoft startup after configuring the existing AZURE_OPENAI_* variables:
MATCHMIND_PROVIDER=microsoft .venv/bin/python -m uvicorn src.serving.app:from_environment --factory --host 127.0.0.1 --port 8000 --no-access-log
```

## Generated files, scope and remaining limits

.env remains ignored and untracked. output.txt is now ignored and removed from Git's index; its modified local copy is preserved. No .env values were modified or printed, and no deployment was attempted. The generated verification JSON files remain under ignored outputs/.

Files changed: Microsoft provider, agent diagnostics/status handling, smoke-service scripts, .gitignore and current pipeline documents. New files: provider diagnostics module, optional verification CLI, 35 offline hardening tests and this guide. Certifi was already pinned; no dependency upgrade was necessary.

No detector tuning occurred: preserved development transfer precision/recall/F1 are 81.430219% / 64.533821% / 72.004080%; shots remain 5 TP / 41 FP / 16 FN and 10.869565% / 23.809524% / 14.925373%. Both halves of 118575 are development data, not independent held-out matches. Cross-match generalization remains unknown. BAS-dependent 2D ball reconstruction, tracking/identity uncertainty and heuristic possession/tactics remain unchanged.

Remaining Microsoft work: frontend judge rehearsal with the configured provider, live exercise of real failure conditions where practical, model-routing reliability assessment, environment-specific CA requirements and later deployment/load verification. Mock HTTP-status tests do not establish live handling of actual quota/authentication incidents. No automatic retries or rate-limit backoff policy was added in this scoped hardening task.
