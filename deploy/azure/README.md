# Azure preparation and verification

Historical milestone status 2026-10-14: local FastAPI is verified; Docker HTTP verification was performed in the prior milestone. The configured Microsoft REST resource has since been live-verified, corroborated by saved demo output. Automatic TLS/error hardening is unit-tested; the new live recheck awaits explicit approval. See [TactiqAI hardening](../../docs/microsoft_hardening.md). **PREPARED BUT NOT DEPLOYED**: Bicep, deployment script, Blob adapter, Fabric import and MCP policy. Storage account/container are now provisioned externally; see the current Blob verification status below. No agents, role assignments, successful storage uploads or Fabric tables were created by this workflow.

## Local and container demo

```bash
.venv/bin/python -m uvicorn src.serving.app:from_environment --factory --host 127.0.0.1 --port 8000 --no-access-log
# Open http://127.0.0.1:8000/matches/118575_clip/report
.venv/bin/python scripts/smoke_service.py --recording 118575_clip
.venv/bin/python scripts/verify_service.py

docker build -f Dockerfile.service -t matchmind-service .
docker run --rm -p 127.0.0.1:8000:8000 -v "$PWD/outputs:/app/outputs:ro" matchmind-service
.venv/bin/python scripts/verify_container.py
```

The local image runs as UID 10001, exposes 8000, loads three artifact directories from `configs/serving/local.json`, and does not contain datasets, outputs, `.env`, caches or video-processing libraries. The default Docker image does not include Azure or Agent Framework SDKs. Container image size/startup measurements are recorded in the current pipeline status. Local startup fails if any registered bundle is malformed; an empty registry has `/health` but returns 503 from `/ready`.

Actual local MP4 byte-range playback requests were verified. For local video through the API set `MATCHMIND_MEDIA_ROOT=data/soccertrack/mot/clips`. This separately allowlists `<match_id>.mp4` for clip recordings. Half reports still have pitch seeking without full-half media. Optional Bearer authentication requires `Authorization: Bearer <MATCHMIND_API_TOKEN>` on match routes. The browser does not receive that token or model credentials. For a hosted browser UI, configure an authenticated gateway that supplies service authorization; the prepared Bicep protects the API with a secret reference but does not implement the gateway login flow.

## Microsoft configuration

`MATCHMIND_PROVIDER=local|gemini|microsoft|microsoft-framework` selects the server default; the report/API can select these presentation providers. Local uses no network. Set server-side `AZURE_OPENAI_ENDPOINT` (resource root or `/openai/v1/`), `AZURE_OPENAI_DEPLOYMENT`, `AZURE_OPENAI_API_KEY`; use a deployed model supporting tools and structured outputs. No endpoint, deployment or credential is hard-coded. The existing REST integration follows current documented Azure OpenAI-v1 structured outputs, with safe timeout/malformed-response fallback. [Microsoft structured outputs](https://learn.microsoft.com/en-us/azure/foundry/openai/how-to/structured-outputs).

```bash
MATCHMIND_PROVIDER=microsoft .venv/bin/python -m uvicorn src.serving.app:from_environment --factory --host 127.0.0.1 --port 8000 --no-access-log
.venv/bin/python -m scripts.demo_matchmind --provider microsoft
```

For the optional real framework path, install `requirements-agent-framework.txt` in an isolated environment, resolve/pin its versions, set `FOUNDRY_PROJECT_ENDPOINT` and `FOUNDRY_MODEL`, authenticate `az login`, then select `microsoft-framework`. `FrameworkProvider` instantiates `Agent` and `FoundryChatClient`, invokes existing dispatcher functions, discards SDK prose and runs the REST closed-statement selector. That final selector also requires the resource endpoint/deployment/auth configuration above. Use `MATCHMIND_PROVIDER=microsoft-framework .venv/bin/python -m uvicorn src.serving.app:from_environment --factory --host 127.0.0.1 --port 8000 --no-access-log` after configuring both project and resource settings. Framework runtime is **MOCK-VERIFIED ONLY**; the SDK is absent and no real agent ran here. Current Microsoft guidance distinguishes project-backed Foundry clients from standalone Azure OpenAI resource clients. [Agent Framework Foundry guidance](https://learn.microsoft.com/en-us/agent-framework/integrations/by-component/model-providers/microsoft-foundry).

## Blob artifact strategy

Current 2026-10-15 workflow: [canonical artifact build/upload/verify guide](../../docs/azure_artifacts.md). Use the safe manifest-based CLI instead of uploading arbitrary output directories:

```bash
python -m src.artifacts build --recording 118575_clip
python -m src.artifacts upload --recording 118575_clip
python -m src.artifacts verify --recording 118575_clip
```

Install requirements-azure.txt; set MATCHMIND_BLOB_ACCOUNT_URL and MATCHMIND_BLOB_CONTAINER. DefaultAzureCredential uses Entra authentication. Existing MATCHMIND_ARTIFACT_STORE, MATCHMIND_RECORDINGS and MATCHMIND_CACHE continue to select Azure serving. Prefixes remain matches/<recording>/bundle.json. Uploads are create-only, allowlisted, hash/schema validated, and manifest-last.

Verified 2026-10-17: all ten configured clips have canonical Azure analytics bundles and private Azure video. Following explicit approval, eight additional bundles were published create-only (184 files including manifests; 167,490,828 bytes), then freshly downloaded and manifest/hash/schema/application-validated. Existing blobs were not overwritten and no videos were reuploaded. The restarted local API reports ten ready recordings. Health/readiness/discovery, all ten tactical views, video byte ranges and grounded Local Ask requests passed live HTTP checks. Missing/corrupt analytics still fail per recording; no local artifact fallback occurs. No role assignment or Container Apps deployment was performed. See the artifact guide for exact evidence and frontend build/unit-check limitations.

## Container Apps resources

Provision a resource group, Container Apps managed environment, registry, Blob container, Key Vault and user-assigned identity separately. The identity needs AcrPull on the registry, Storage Blob Data Reader on the artifact container and Key Vault Secrets User on the referenced vault secrets. The deployment script does not create these permissions. Model API-key and service Bearer-token values stay in Key Vault, referenced by URLs in configuration. Do not put secret values in parameter files.

```bash
docker build --build-arg WITH_AZURE=true -f Dockerfile.service -t "$MATCHMIND_IMAGE" .
# Authenticate/push to your own registry, then set values from env.example.
docker push "$MATCHMIND_IMAGE"
bash deploy/azure/deploy.sh
```

`container-app.bicep` configures HTTPS ingress to port 8000, 0.5 CPU/1 GiB, one warm demo replica, maximum two replicas, HTTP concurrency target 10, startup/readiness `/ready` and liveness `/health`. Blob downloads may require a larger startup budget for bigger datasets. Scale and memory are assumptions, not cloud load-test results. Bicep has static tests; Azure CLI validation/deployment was not available. Container Apps is sufficient for this HTTP service; AKS has no demonstrated requirement. [Template specification](https://learn.microsoft.com/en-us/azure/container-apps/azure-resource-manager-api-spec).

## Fabric analytical purpose and import

```bash
.venv/bin/python -m src.fabric
```

Upload the six CSVs and `schema.json` from `outputs/fabric` to a Fabric Lakehouse Files folder, e.g. `Files/matchmind/snapshot_20261014`. Create a notebook attached to that lakehouse. Load `schema.json`, import/copy `fabric_import.py`, then call:

```python
import json
manifest=json.load(open('/lakehouse/default/Files/matchmind/snapshot_20261014/schema.json'))
counts=import_snapshot(spark,'Files/matchmind/snapshot_20261014',manifest)
print(counts)
```

The importer uses explicit types, checks rows/primary keys, and writes new Delta tables with `errorifexists`; it never overwrites an existing analytical snapshot silently. Verify the returned counts against the manifest, then join on recording_id and actor/event IDs. Fabric supports CSV ingestion and Delta tables. [Lakehouse formats](https://learn.microsoft.com/en-us/fabric/data-engineering/lakehouse-and-delta-tables).

Tables: Match, Player, Event, Transfer, TacticalState, Evaluation. The schema defines columns/types, keys, nullability, hashes and counts. Match rows retain source hashes/configuration/limitations. Actor and tracker IDs stay separate. Timestamps are recording-local seconds, not UTC. Clip and half recordings overlap: filter scope before aggregating player totals or events. Evaluation rows are development only. Fabric’s purpose is cross-recording dashboards, player metrics, tactical trends and detector evaluation; no model/data agent computes authoritative match facts. No Fabric workspace was available, so upload/import was not executed.

## MCP operator boundary

`mcp-policy.json` specifies read-only service/revision metadata, artifact hashes and sanitized observability. It forbids secrets, mutation, code execution and football calculations. Configure an operator Entra identity, read-only RBAC and a host tool allowlist before connecting Azure MCP. This policy file does not enforce permissions by itself. MCP stays outside the public Ask tool registry. No MCP server/credential was available, so none ran. [Azure MCP scope](https://learn.microsoft.com/en-us/azure/developer/azure-mcp-server/overview).

## Existing Azure clip videos

Set optional `MATCHMIND_MEDIA_BLOB_CONTAINER=soccertrack-v2` with the existing account URL/Entra credential to stream private clips via the API (HEAD and byte-range seeking). The browser receives no Blob credential or SAS. All ten existing clips now have video and validated canonical analytics bundles, following the approved eight-bundle create-only upload and fresh Azure read-back verification. The restarted backend reports ten ready recordings; views, video ranges and grounded Local Ask passed live HTTP checks for each. Future video-only recordings remain explicitly pending for analysis until their bundles validate. Local ignored `.env` is configured for the ten clips. See [current playback and publication status](../../docs/azure_artifacts.md).
