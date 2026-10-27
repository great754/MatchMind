#!/bin/sh
# Use an explicit, intact runtime; never silently launch a different Python.
set -eu
cd "$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)"
backend_python="${TACTIQAI_BACKEND_PYTHON:-.venv/bin/python}"
"$backend_python" -c 'import os,fastapi,uvicorn; from dotenv import load_dotenv; load_dotenv(); exec("import azure.identity,azure.storage.blob" if os.getenv("MATCHMIND_ARTIFACT_STORE","local")=="azure" else "pass")' || {
  echo 'Backend dependencies unavailable. Set TACTIQAI_BACKEND_PYTHON to an intact environment with requirements-service.txt and requirements-azure.txt installed.' >&2
  exit 1
}
exec "$backend_python" -X pycache_prefix="${TMPDIR:-/tmp}/tactiqai-python-cache" -m uvicorn src.serving.app:from_environment --factory --host 127.0.0.1 --port "${TACTIQAI_API_PORT:-8000}" --no-access-log
