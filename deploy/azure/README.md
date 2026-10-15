# Azure preparation and verification

Status 2026-10-14: local FastAPI and Docker HTTP verification performed. No Azure subscription/CLI/credentials are configured. **PREPARED BUT NOT DEPLOYED**: Bicep, deployment script, Blob adapter, Fabric import and MCP policy. No resources, agents, role assignments, storage uploads or Fabric tables were created.

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

```bash
.venv/bin/python -m src.artifacts
# Produces outputs/serving_bundle/matches/<recording_id>/bundle.json plus allowlisted files.
# Authenticate with your operator account before uploading a frozen snapshot:
az storage blob upload-batch --auth-mode login --account-name "$MATCHMIND_STORAGE_ACCOUNT" --destination "$MATCHMIND_BLOB_CONTAINER" --source outputs/serving_bundle --pattern 'matches/*'
```

`bundle.json` contains SHA256 hashes. Prefixes include recording identity, so overlapping clip and half scopes cannot overwrite one another. Use a versioned container or versioned deployment snapshot; do not mutate the active snapshot during a demo. `AzureBlobArtifactStore` downloads only known bundle names at startup, verifies hashes and closed schemas, then serves from a local cache. Failure to load startup data fails readiness rather than fabricating empty statistics. Azure availability is not required per question.

Local defaults: `MATCHMIND_ARTIFACT_STORE=local`, `MATCHMIND_REGISTRY=configs/serving/local.json`. Azure configuration: `MATCHMIND_ARTIFACT_STORE=azure`, `MATCHMIND_BLOB_ACCOUNT_URL`, `MATCHMIND_BLOB_CONTAINER`, `MATCHMIND_RECORDINGS` (JSON list), `MATCHMIND_CACHE` (default `/tmp/matchmind-artifacts`). Install `requirements-azure.txt`; the factory uses Azure Blob `ContainerClient` with `DefaultAzureCredential`. [Official Python Blob guidance](https://learn.microsoft.com/en-us/azure/storage/blobs/storage-quickstart-blobs-python).

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
