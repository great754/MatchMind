"""Stream a direct video URL to Azure using bounded memory and CLI login."""
import argparse
import base64
import hashlib
import json
from pathlib import Path
import os
import uuid

import requests
from azure.core import MatchConditions
from azure.identity import AzureCliCredential
from azure.storage.blob import BlobServiceClient, ContentSettings

ACCOUNT = "tactiqaistorage"
CONTAINER = "soccertrack-v2"
BLOB_NAME = "external_matches/vibuxer_z4ny0atm95s8/full_match.mp4"
CHUNK_SIZE = 8 * 1024 * 1024


def stream_video(url, blob, http=requests, progress=print, checkpoint=None):
    """Publish only a complete stream; never replace an existing video."""
    if blob.exists():
        raise RuntimeError("Destination already exists; it has not been overwritten.")
    source_key = hashlib.sha256(url.encode()).hexdigest()
    if checkpoint and Path(checkpoint).is_file():
        saved = json.loads(Path(checkpoint).read_text())
        if saved.get('source_key') == source_key:
            _, staged = blob.get_block_list(block_list_type='all')
            available = {b.id: b.size for b in staged}
            if saved.get('blocks') and sum(b['size'] for b in saved['blocks']) == saved['bytes'] and all(available.get(b['id']) == b['size'] for b in saved['blocks']):
                progress('Resuming final commit from saved ordered blocks…', flush=True)
                return commit_video(blob, saved, progress)
    blocks = []
    total = 0
    sha256 = hashlib.sha256()
    with http.get(url, headers={"Accept-Encoding": "identity"}, stream=True,
                  timeout=(30, 120)) as response:
        if response.status_code != 200:
            raise RuntimeError(f"Video source returned HTTP {response.status_code}; check that the link is current.")
        if "video" not in response.headers.get("Content-Type", "").lower():
            raise RuntimeError("Source did not return video content; use a direct video link.")
        if response.headers.get("Content-Encoding", "identity").lower() != "identity":
            raise RuntimeError("Source unexpectedly compressed the stream; no video was committed.")
        expected = response.headers.get("Content-Length")
        expected = int(expected) if expected is not None else None
        for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
            if not chunk:
                continue
            if len(blocks) >= 50000:
                raise RuntimeError("Video exceeds the block limit; no video was committed.")
            block_id = base64.b64encode(uuid.uuid4().bytes).decode("ascii")
            blob.stage_block(block_id=block_id, data=chunk, length=len(chunk),
                             validate_content=True)
            blocks.append({"id": block_id, "size": len(chunk)})
            sha256.update(chunk)
            total += len(chunk)
            progress(f"Transferred: {total / 1024**2:.1f} MiB", flush=True)
    if not total or (expected is not None and total != expected):
        raise RuntimeError(f"Incomplete source: received {total} bytes, expected {expected}; no video was committed.")
    saved = {'source_key': source_key, 'blocks': blocks, 'bytes': total, 'sha256': sha256.hexdigest()}
    if checkpoint:
        path = Path(checkpoint)
        path.parent.mkdir(parents=True, exist_ok=True)
        temp = path.with_suffix('.tmp')
        temp.write_text(json.dumps(saved))
        os.replace(temp, path)
    return commit_video(blob, saved, progress)


def commit_video(blob, saved, progress):
    progress("Committing uploaded blocks…", flush=True)
    # IfMissing sets If-None-Match:* itself. Passing etag as well leaks an
    # unconsumed keyword to this SDK's HTTP transport and raises TypeError.
    blob.commit_block_list([b['id'] for b in saved['blocks']], match_condition=MatchConditions.IfMissing,
                           metadata={"sha256": saved['sha256']},
                           content_settings=ContentSettings(content_type="video/mp4"))
    actual = blob.get_blob_properties().size
    if actual != saved["bytes"]:
        raise RuntimeError(f"Upload verification failed: {actual} != {saved['bytes']}")
    return {"bytes": actual, "sha256": saved["sha256"]}


def main(argv=None):
    from dotenv import load_dotenv
    load_dotenv()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default=os.environ.get("VIDEO_URL"),
                        help="Direct video URL; defaults to exported VIDEO_URL or .env.")
    args = parser.parse_args(argv)
    if not args.url:
        parser.error("VIDEO_URL is missing. Export it in this terminal or pass --url. Shell exports are not shared between terminals.")
    credential = AzureCliCredential()
    service = BlobServiceClient(f"https://{ACCOUNT}.blob.core.windows.net", credential=credential)
    try:
        print(f"Streaming to {ACCOUNT}/{CONTAINER}/{BLOB_NAME}", flush=True)
        checkpoint = Path(__file__).resolve().parents[1] / 'outputs' / 'azure_video_uploads' / (hashlib.sha256(f'{ACCOUNT}/{CONTAINER}/{BLOB_NAME}'.encode()).hexdigest() + '.json')
        result = stream_video(args.url, service.get_blob_client(container=CONTAINER, blob=BLOB_NAME), checkpoint=checkpoint)
        print(f"UPLOAD VERIFIED: {result['bytes']:,} bytes; SHA-256 {result['sha256']}", flush=True)
        return 0
    except RuntimeError as error:
        print(f"Upload stopped: {error}", flush=True)
        return 1
    except Exception as error:
        # Signed source URLs must not leak through requests exception messages.
        code = getattr(error, "error_code", None)
        print(f"Upload stopped: {type(error).__name__}" + (f" ({code})" if code else "")
              + ". No automatic retry or overwrite was attempted.", flush=True)
        return 1
    finally:
        service.close()
        credential.close()


if __name__ == "__main__":
    raise SystemExit(main())
