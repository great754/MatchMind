#!/bin/sh
# Explicit local M23 review profile; canonical analytics/media remain in Azure.
set -eu
cd "$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)"
tactiq_runtime_root="${TACTIQAI_RUNTIME_ROOT:-$HOME/Library/Caches/TactiqAI}"
export AZURE_TOKEN_CREDENTIALS="${AZURE_TOKEN_CREDENTIALS:-AzureCliCredential}"
export MATCHMIND_ARTIFACT_STORE="${MATCHMIND_ARTIFACT_STORE:-azure}"
export MATCHMIND_PROVIDER="${MATCHMIND_PROVIDER:-local}"
export MATCHMIND_CACHE="${MATCHMIND_CACHE:-$tactiq_runtime_root/azure_artifacts_m22}"
export MATCHMIND_HIGHLIGHT_CACHE="${MATCHMIND_HIGHLIGHT_CACHE:-$tactiq_runtime_root/highlights_m22}"
export MATCHMIND_ASSISTANT_CACHE="${MATCHMIND_ASSISTANT_CACHE:-$tactiq_runtime_root/assistant_m22}"
export MATCHMIND_CV_REGISTRY="${MATCHMIND_CV_REGISTRY:-outputs/m23/runtime_cv_registry.json}"
export TACTIQAI_BACKEND_PYTHON="${TACTIQAI_BACKEND_PYTHON:-$tactiq_runtime_root/backend-m17/bin/python}"
if [ ! -f "$MATCHMIND_CV_REGISTRY" ]; then
  echo 'M23 runtime registry is missing. Validate and install an M23 analysis with python -m scripts.install_cv_m23 --analysis <analysis-root> first.' >&2
  exit 1
fi
exec scripts/start_backend.sh
