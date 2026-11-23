# Short CV clips moved to Azure — 2026-11-23

The four original clips are stored privately in **tactiqaistorage / soccertrack-v2**, under `cv/clips/<recording>/<SHA-256>.mp4`. The user's `giround` refers to the existing file/recording named **giroud**. Uploads were create-only; no existing blob was overwritten.

| Recording | Original bytes | SHA-256 |
|---|---:|---|
| giroud | 964,751 | `66fb0421464ba8a7a91a117a8625d4ea6f4870eb88e69ac0732fbaf3c7752e5c` |
| southampton | 4,720,872 | `8c3d8f2ca442bd21fa90ce66e948b844c2cdebb8af02eed1a39fcab3f2ee49f6` |
| team_goal | 1,753,330 | `726e8592c8cac7b12251957ad4a9dff07afbf1d526b0e4e18387f03237805a0d` |
| test_match | 3,375,243 | `e092a252f47fd4f135506bd573ad36eaa25c95b0bbba3246c988827591c6dbac` |

Azure contains **10,814,196 original bytes**. Each upload was downloaded and fully SHA-256 verified, then all four were reverified immediately before deletion. **16 matching local source copies** were removed from `data/raw/`, Downloads and the M23/M24 research caches: 47,012,905 logical bytes. Some copies were hard links; actual freed storage is smaller and APFS/iCloud-dependent. The inventory identified 13 distinct inodes with 37,199,872 allocated bytes before deletion. No model, annotation, analysis sidecar, manifest or unrelated video was deleted.

## Current app behavior

`configs/serving/cv_media.json` explicitly pins each blob's location, full SHA-256, size and ETag. Startup revalidates the original bytes against the unchanged CV manifests. Browser video uses authenticated Azure byte-range streaming. The precomputed analyses, models, labels and score reports remain local.

Source-frame decoding for M19/M22/Giroud review, the M24 comparison and highlight generation uses a private temporary file from the verified Azure source; the file is removed when the operation finishes, including exceptions. The backend retains a bounded in-memory copy of these small sources. **It does not recreate persistent original videos in the project or research caches.** Stream reads use the pinned ETag; changed/corrupt cloud data is rejected, with no unrelated local fallback. Metadata and catalog entries explicitly expose `media_storage: azure` for these clips.

The media migration preserves source hashes, clocks, human labels, M23 selections, pass/stat evidence rules and M24 benchmark numbers. It does not create canonical analytics bundles or grant these clips Azure `artifact_ready`; they remain CV recordings with Azure-hosted media. Canonical SoccerTrack artifact/media configuration is unchanged.

## Verified after local deletion and a fresh backend restart

- Health, readiness and catalog: **200**, with **16 usable recordings**. All four clip entries explicitly report Azure media.
- All four original-video ranges: **206**, with ETags matching the pinned Azure registrations. M24 metadata, source JPEGs, heatmaps and video ranges all work; saved M19/M22/Giroud review source images return **200**. Source-frame pixel hashes still match M24 inference inputs.
- All four existing M23 frame/reference/grounded Local Ask paths pass. Canonical Azure `118575_clip` view remains **200 / 481 events**, grounded Local Ask **200 / 14 evidence facts**, video range **206**.
- Actual Chrome: completed Giroud review and source playback **passed (22.3 s)**; four-source M24 benchmark and playback **passed (14.1 s)**. Annotation writes were zero and original label hashes remain unchanged.
- Backend regressions: **1,165 run, 1,164 passed, one skip**, including nine new tests covering missing local originals, full SHA/ETag rejection, cloud ranges, private temporary-file cleanup and deletion preflight. Legacy source-adapter compatibility issues found by the first runs were corrected; final suite passes. Initial post-restart checks ran before the listening socket opened and were repeated after startup; the verifier also corrected HTTP-header case handling.
- Every deleted source path is still absent after these requests. Both servers remain running. [Aggregate evidence](evidence/cv_azure_media_results.json).


## Commands and recovery

The normal backend command remains **`scripts/start_backend_m23.sh`**; frontend remains **`npm --prefix frontend run dev`**. Existing Azure login/container settings suffice. No environment-variable rename or extra manual setting is required on this machine. Internet access and Azure credentials are now required to open these source videos. Existing research caches containing models, labels and results must remain available.

The completed operation used the existing backend Python (`~/Library/Caches/TactiqAI/backend-m17/bin/python`):

```sh
python -m scripts.migrate_cv_media_to_azure
# Only after the app is using the verified Azure registration:
python -m scripts.migrate_cv_media_to_azure --delete-verified-copies
python -m scripts.verify_cv_azure_media --output outputs/m24/azure_media_live_verification_v1.json
```

The complete inventory/receipt are `outputs/m24/azure_media_inventory_complete_v1.json` and `outputs/m24/azure_media_deletion_complete_v1.json`; the latter preserves both original receipts. A final filename check found the byte-identical Downloads/southampton.mp4, which was separately hash-verified and deleted with an expanded inventory and `--include-downloads`. Verification against the complete receipt used `python -m scripts.verify_cv_azure_media --receipt outputs/m24/azure_media_deletion_complete_v1.json --output outputs/m24/azure_media_live_verification_complete_v1.json`. These files are private ignored JSON under `outputs/m24/azure_media_*_v1.json`. The migration tool refuses changed local files, unmatched inventory/archive identities, symlinks or paths outside the project/cache roots (Downloads additionally requires the explicit flag and an inventoried matching entry). It fully checks all remote sources and all local candidates before unlinking the first file. Repeating the upload phase needs a fresh inventory filename; existing content-addressed blobs are verified and reused, never overwritten.

For an older offline pipeline that requires a physical MP4, explicitly download the pinned source into a temporary working location and remove it after processing. For example:

```sh
az storage blob download --account-name tactiqaistorage --container-name soccertrack-v2 \
  --auth-mode login \
  --name cv/clips/giroud/66fb0421464ba8a7a91a117a8625d4ea6f4870eb88e69ac0732fbaf3c7752e5c.mp4 \
  --file /tmp/giroud.mp4
```

Verify the downloaded SHA before use. Reproducing historical snapshots that hard-code deleted local video paths requires explicitly restoring those exact pinned bytes to the expected paths first. The frozen manifest/reference files themselves were preserved; original video bytes are now archived in Azure. Normal frontend playback, saved analyses and source-frame reviews do not need that manual restoration.
