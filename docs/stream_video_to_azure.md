# Stream a direct video to Azure

`scripts/stream_video_to_azure.py` reads a direct video URL and stages 8-MiB blocks in Azure using the current Azure CLI login. It does not save the full video locally. Its existing destination is:

```text
tactiqaistorage / soccertrack-v2
external_matches/vibuxer_z4ny0atm95s8/full_match.mp4
```

Export `VIDEO_URL` in the **same terminal** that runs the script. A variable exported in another terminal is not shared with the agent's command shell. The script also supports `.env` and `--url`; do not commit signed URLs or add them to documentation.

```sh
cd /Users/greatabhieyighan/Desktop/Projects/matchmind
# Set VIDEO_URL to the current direct video link in this terminal.
/Users/greatabhieyighan/Library/Caches/TactiqAI/backend-m17/bin/python \
  scripts/stream_video_to_azure.py
```

The working runtime already contains Requests, python-dotenv and the Azure SDK. Azure CLI container access was verified on 2026-11-13. The script reports a clear error if the URL is missing. A missing, expired or non-video source is rejected. Streams with a declared Content-Length must match it; empty/incomplete streams are not committed. Unknown-length video streams are supported.

Each staged block uses Azure content-MD5 validation. The script computes SHA-256 while reading the source, stores that digest as blob metadata, commits atomically with a create-only condition and checks the final blob size. It refuses an existing destination and protects against another upload creating that destination during the stream. Exceptions do not automatically overwrite or restart the transfer. Before committing a complete stream, the script saves the source-URL fingerprint (not the signed URL), ordered block IDs, sizes and stream SHA-256 under ignored `outputs/azure_video_uploads/`. A rerun with the same URL can retry finalization without retransferring video when Azure still has every saved block. This is final-commit recovery, not partial-download resume.

Eight regression tests cover complete streaming/conditional commit, existing destinations, incomplete/empty sources, HTTP/non-video rejection, unknown-length input, actual installed SDK condition serialization and commit recovery:

```sh
/Users/greatabhieyighan/Library/Caches/TactiqAI/backend-m17/bin/python \
  -m unittest tests.test_stream_video_to_azure
```

This transfers the original video only. It does not run soccer analytics, create a canonical artifact bundle or register a new frontend recording. No existing source videos or analytics are replaced.

The installed Azure SDK translates `match_condition=MatchConditions.IfMissing` into `If-None-Match: *`. Do not additionally pass `etag="*"` with that condition: this SDK leaves the redundant argument for the HTTP transport and raises `TypeError`. The SDK serialization regression checks both the conditional header and absence of leaked transport arguments.

## Actual upload — 2026-11-13

The Tottenham vs Arsenal source uploaded successfully as **759,278,077 bytes** in 91 ordered blocks, with `video/mp4` content type. The full video was never saved locally. The first attempt staged the bytes but its redundant-etag commit failed; its original order had not been checkpointed, so the stream was sent again after the SDK-condition fix. The corrected run committed successfully and verified the final size. No existing media or analytics was overwritten.

The committed blob’s size and SHA-256 metadata match the completed stream checkpoint. The first and last 4,096 bytes were independently fetched from both the original source and ETag-pinned Azure ranges; both pairs are identical. The entire blob was not independently rehashed after upload. [Portable verified evidence](evidence/external_video_upload_2026_11_13.json).
