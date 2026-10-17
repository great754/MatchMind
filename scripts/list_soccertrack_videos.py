from huggingface_hub import HfApi, get_hf_file_metadata, hf_hub_url
from pathlib import Path
import csv

REPO_ID = "atomscott/soccertrack-v2"
REPO_TYPE = "dataset"

api = HfApi()

print(f"Scanning {REPO_ID}...\n")

# Uses your existing `hf auth login` credentials.
files = api.list_repo_files(
    repo_id=REPO_ID,
    repo_type=REPO_TYPE,
)

video_extensions = {
    ".mp4",
    ".mov",
    ".avi",
    ".mkv",
    ".webm",
    ".m4v",
}

videos = [
    f for f in files
    if Path(f).suffix.lower() in video_extensions
]

print(f"Found {len(videos)} video files in repository metadata.")
print("Checking which ones are accessible...\n")

accessible = []
blocked = []

for i, path in enumerate(videos, start=1):
    try:
        url = hf_hub_url(
            repo_id=REPO_ID,
            filename=path,
            repo_type=REPO_TYPE,
        )

        metadata = get_hf_file_metadata(url)

        size = metadata.size or 0

        accessible.append({
            "path": path,
            "size_bytes": size,
            "size_mb": size / (1024 ** 2),
            "size_gb": size / (1024 ** 3),
        })

        print(
            f"[{i}/{len(videos)}] OK  "
            f"{size / (1024 ** 3):7.2f} GB  {path}"
        )

    except Exception as exc:
        blocked.append((path, str(exc)))
        print(f"[{i}/{len(videos)}] BLOCKED  {path}")


# Plain-text file for easy copy/paste back into ChatGPT.
txt_path = Path("downloadable_videos.txt")

with txt_path.open("w") as f:
    for item in accessible:
        f.write(f"{item['path']}\n")


# CSV also includes sizes.
csv_path = Path("downloadable_videos.csv")

with csv_path.open("w", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=["path", "size_bytes", "size_mb", "size_gb"],
    )
    writer.writeheader()
    writer.writerows(accessible)


total_bytes = sum(item["size_bytes"] for item in accessible)

print("\n" + "=" * 70)
print("RESULT")
print("=" * 70)
print(f"Accessible videos: {len(accessible)}")
print(f"Blocked videos:    {len(blocked)}")
print(f"Total size:        {total_bytes / (1024 ** 3):.2f} GB")
print()
print(f"Paths saved to:    {txt_path}")
print(f"Details saved to:  {csv_path}")

if blocked:
    print("\nBlocked/unavailable:")
    for path, _ in blocked:
        print(f"  {path}")
