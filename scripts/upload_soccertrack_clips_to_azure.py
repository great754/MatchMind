import base64
import os
import sys
import requests

from azure.identity import DefaultAzureCredential
from azure.storage.blob import BlobServiceClient
from huggingface_hub import hf_hub_url, get_token


REPO_ID = "atomscott/soccertrack-v2"
REPO_TYPE = "dataset"

AZURE_ACCOUNT_URL = "https://tactiqaistorage.blob.core.windows.net"
AZURE_CONTAINER = "soccertrack-v2"

# ONLY the short ~4-minute MOT clips.
CLIPS = [
    "mot/clips/117092.mp4",
    "mot/clips/117093.mp4",
    "mot/clips/118575.mp4",
    "mot/clips/118576.mp4",
    "mot/clips/118577.mp4",
    "mot/clips/118578.mp4",
    "mot/clips/128057.mp4",
    "mot/clips/128058.mp4",
    "mot/clips/132831.mp4",
    "mot/clips/132877.mp4",
]

# Only this much data is held in memory at once.
CHUNK_SIZE = 8 * 1024 * 1024  # 8 MB


def human_size(num_bytes):
    gb = num_bytes / (1024 ** 3)
    if gb >= 1:
        return f"{gb:.2f} GB"

    mb = num_bytes / (1024 ** 2)
    return f"{mb:.1f} MB"


def upload_stream(blob_client, response, total_size):
    block_ids = []
    transferred = 0
    block_number = 0

    for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
        if not chunk:
            continue

        block_id = base64.b64encode(
            f"{block_number:08d}".encode()
        ).decode()

        blob_client.stage_block(
            block_id=block_id,
            data=chunk,
        )

        block_ids.append(block_id)
        transferred += len(chunk)
        block_number += 1

        if total_size:
            percent = transferred / total_size * 100
            print(
                f"\r    {human_size(transferred)} / "
                f"{human_size(total_size)} "
                f"({percent:5.1f}%)",
                end="",
                flush=True,
            )
        else:
            print(
                f"\r    {human_size(transferred)} uploaded",
                end="",
                flush=True,
            )

    print()

    blob_client.commit_block_list(block_ids)

    return transferred


def main():
    print("TactiqAI SoccerTrack-v2 → Azure uploader")
    print("=" * 60)
    print("Mode: direct streaming")
    print("Local video storage required: ~0 GB")
    print(f"Destination: {AZURE_CONTAINER}")
    print()

    token = get_token()

    if not token:
        print(
            "ERROR: No Hugging Face token found.\n"
            "Run:\n\n"
            "    hf auth login\n"
        )
        sys.exit(1)

    credential = DefaultAzureCredential()

    blob_service = BlobServiceClient(
        account_url=AZURE_ACCOUNT_URL,
        credential=credential,
    )

    container_client = blob_service.get_container_client(
        AZURE_CONTAINER
    )

    try:
        container_client.get_container_properties()
    except Exception as exc:
        print("ERROR: Cannot access Azure container:")
        print(exc)
        sys.exit(1)

    headers = {
        "Authorization": f"Bearer {token}"
    }

    completed = []
    skipped = []
    failed = []

    for index, hf_path in enumerate(CLIPS, 1):
        filename = os.path.basename(hf_path)

        # Preserve useful source hierarchy inside Azure.
        blob_name = f"mot/clips/{filename}"

        print()
        print(f"[{index}/{len(CLIPS)}] {filename}")
        print(f"  Hugging Face: {hf_path}")
        print(f"  Azure:        {AZURE_CONTAINER}/{blob_name}")

        blob_client = container_client.get_blob_client(blob_name)

        # Resume support: if it already exists, don't upload it again.
        if blob_client.exists():
            props = blob_client.get_blob_properties()
            print(
                f"  SKIP: already in Azure "
                f"({human_size(props.size)})"
            )
            skipped.append(filename)
            continue

        url = hf_hub_url(
            repo_id=REPO_ID,
            filename=hf_path,
            repo_type=REPO_TYPE,
        )

        try:
            with requests.get(
                url,
                headers=headers,
                stream=True,
                allow_redirects=True,
                timeout=(30, 300),
            ) as response:

                response.raise_for_status()

                total_size = int(
                    response.headers.get("Content-Length", 0)
                )

                print(
                    "  Source size:   "
                    + (
                        human_size(total_size)
                        if total_size
                        else "unknown"
                    )
                )

                uploaded_size = upload_stream(
                    blob_client,
                    response,
                    total_size,
                )

            # Verify the committed Azure object.
            props = blob_client.get_blob_properties()

            if props.size != uploaded_size:
                raise RuntimeError(
                    f"Azure size mismatch: uploaded "
                    f"{uploaded_size} bytes but Blob reports "
                    f"{props.size} bytes"
                )

            if total_size and props.size != total_size:
                raise RuntimeError(
                    f"Source/Azure size mismatch: source "
                    f"{total_size}, Azure {props.size}"
                )

            print(
                f"  VERIFIED:     {human_size(props.size)}"
            )

            completed.append(filename)

        except Exception as exc:
            print(f"  FAILED: {exc}")
            failed.append((filename, str(exc)))

    print()
    print("=" * 60)
    print("UPLOAD SUMMARY")
    print("=" * 60)
    print(f"Uploaded: {len(completed)}")
    print(f"Skipped:  {len(skipped)}")
    print(f"Failed:   {len(failed)}")

    if completed:
        print("\nUploaded:")
        for name in completed:
            print(f"  ✓ {name}")

    if skipped:
        print("\nAlready present:")
        for name in skipped:
            print(f"  - {name}")

    if failed:
        print("\nFailed:")
        for name, error in failed:
            print(f"  ✗ {name}")
            print(f"    {error}")

        sys.exit(1)

    print("\nAll available 4-minute clips are in Azure.")


if __name__ == "__main__":
    main()
