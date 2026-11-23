# M25 — dataset pilot, including the long Azure broadcast

Status: **The 20-image M25 pilot is ready for human review, including real Gemini proposals. No training or M26 work.**

[Open the review workspace](http://127.0.0.1:5173/?review=m25). It reuses the existing M22/Giroud annotation editor, including accept/reject, native-coordinate drawing, visibility/uncertainty, automatic save/next and back/correction. The page opens at the first unreviewed frame and displays all available CV/Gemini proposals separately.

## Sources and selected images

No original source videos were restored persistently. `data/raw` is empty. The four archived short sources and the existing long Azure broadcast were accessed with pinned ETag-conditioned ranges. The selected collection is recorded in `configs/training/cv_m25_sources.json`; adding an Azure source there does not require changing source-discovery code. Other canonical SoccerTrack recordings are unchanged and were not processed by this pilot.

| Source | Native format | Dataset candidates | Pilot images |
|---|---|---:|---:|
| giroud | 640×360, 25 FPS, 18 s | 2 | 2 |
| southampton | 1280×720, 29.97 FPS, 22.022 s | 2 | 2 |
| team_goal | 640×360, 25 FPS, 20 s | 2 | 2 |
| test_match | 1080×1080, 23.976 FPS, 22.574 s | 2 | 2 |
| Tottenham–Arsenal long broadcast | 1280×720, **50 FPS, 3,099.08 s (51:39.08)** | 30 | **12** |

The requested roughly hour-long source is `soccertrack-v2/external_matches/vibuxer_z4ny0atm95s8/full_match.mp4`, 759,278,077 bytes, pinned ETag `"0x8DF41BE9BE376A5"`. Its archive SHA is `2c6310d984110604bf753f69336916c41d3d03886f60a80d1123e8cbd32675e8`. This run verifies Azure ETag/size and bounded ranges; it does **not** claim a new whole-video SHA readback. Thirty source times span 120–2,979.08 s. Selected long-source times: **120.00, 317.18, 514.36, 711.54, 908.72, 1,105.88, 1,303.06, 1,598.84, 1,894.60, 2,190.36, 2,781.90 and 2,979.08 seconds**. These cover separate moments across the broadcast, not one goal sequence.

The initial experiment has **20 annotation frames**, even though preparation extracted 38 candidates and separate native temporal-context frames. No other candidate was sent for ball annotation. Source sampling uses seed 25 and time diversity, not detector success. Stable frame IDs hash archive source SHA and native frame index. Original resolution/aspect are preserved; there is no export crop or artificial ball-size box.

## Source preparation and filtering

Unannotated source images were extracted before ball inference. Contrast/blank checks, pitch coverage and Hough line features provide an explainable camera screen. **Green coverage alone never grants eligibility**: survivors initially enter `uncertain`, requiring source inspection. All four source-only contact sheets were actually inspected before selecting the pilot. Clearly unsuitable candidates include celebrations, manager/player close-ups, substitution scenes, goal close-ups and heavily zoomed views. Twenty inspected wide broadcast-camera candidates enter the pilot; the other 18 remain outside annotation. Twelve have explicit exclusion/uncertainty notes and six additional wide-view candidates are simply not selected.

Perceptual-DCT and exact source/image identity diagnostics are retained; **zero near-duplicate flags** were found among the 38 images at Hamming distance ≤3. Selection keeps widely separated long-source times. This is a prototype screen plus actual preparation inspection, **not a validated automated seven-class shot classifier or perfect replay detector**. Score-overlay progression and absence of an identifiable replay marker support the selected views, but hidden wide-angle replays remain a limitation. Ball reviewers may mark uncertain frames and explain suitability in notes; all seven shot types can be corrected in the review UI, while a robust automatic replay/context classifier remains further work. Do not infer original-match identity solely from a visually similar frame.

## Existing references and CV proposals

**Eight existing verified-visible human annotations are reused**, two per short source. The exact decoded pixels were checked against the frozen M24 source-frame hashes. An FFmpeg/OpenCV decode mismatch was caught before reuse; the original extraction was preserved, and these eight images were decoded through temporary SHA-verified Azure sources using OpenCV and required to match the original pixel hashes exactly. No matching tolerance was relaxed. The original M19/M22/Giroud annotation stores are not edited.

M16 YOLO11s was run on the 12 unlabelled long-source images at a recorded 0.05 candidate floor, `imgsz=1280`; 11 images have retained M16 proposals. Full native outputs are stored separately from the bounded UI subset. Existing M22 supplies proposals on **all 12** using its fixed test_match-excluded checkpoint and original global-plus-tile inference. Actual neighboring Azure source frames are supplied with the model's source-FPS-aware spacing: **t−4, t−2, t at 50 FPS**. Earlier 6/3-context diagnostic results were preserved separately and are not the displayed final proposal set. Temporal cut detection remains limited; these are review proposals, not observations certified by a tracker.

Available M21/M22/M23 candidate pools on the eight short-source frames are reused from frozen output files. M23/M21 are not rerun across the long match. **19/20 pilot images have at least one CV proposal**, including cached short-source proposals; the remaining image retains its trusted human box. Per-model native boxes, confidence, source frame, configuration and checkpoint/output provenance are retained separately. New-output timestamps explicitly mean local response/cache recording time; unavailable historical prediction clocks stay null rather than being invented. IoU-based overlap groups do not merge or approve predictions, and scores from different models are not treated as comparable probabilities. The current frontend shows source/model names and per-model overlay toggles. Rejected regions never imply whole-frame absence.

## Gemini: verified real responses and cost

The configured **`gemini-3.5-flash-lite`** model was checked through its metadata endpoint and actually processed the **12 unverified full-resolution long-source PNGs**. All **12 responses pass the strict JSON/coordinate checks and contain a box proposal**. The eight exact pre-existing human labels were never sent. These are proposed boxes, **not twelve proven-correct ball detections**; quality remains for the user to establish.

Google's [image documentation](https://ai.google.dev/gemini-api/docs/image-understanding) defines boxes as **[ymin, xmin, ymax, xmax], normalized 0–1000**. The prompt explicitly rejects boots, socks, advertising and guessed trajectories. The parser validates booleans, schema, finite coordinates, ordering, bounds and uncertain/null cases before mapping into native xywh. Actual returned model version is `gemini-3.5-flash-lite`. No crop, size approximation or fabricated confidence is applied. Full raw replies, prompt/model/image cache keys, parsed values and usage are private and retained.

Recorded successful-response usage: **14,268 input tokens, 540 output tokens, zero reported thought tokens**, standard service tier. At [official rates](https://ai.google.dev/gemini-api/docs/pricing) of $0.30/M input and $2.50/M output, the estimate is **$0.0056304**, against a **$1 pilot maximum**. This is a token-based estimate, not an invoice. Two earlier HTTP 429 requests hit a monthly project cap; the user fixed it. One later connection reset interrupted the third frame, so the job resumed from its two completed caches. Those unsuccessful/interrupted calls supplied no usage metadata and their billing cannot be independently established here.

There are **12 distinct images sent**, **12 successful replies**, **two historical HTTP quota errors** and **one transport reset**. The successful first two images were not resent during recovery. A complete rerun with outbound requests explicitly disabled reused all 12 real caches and made **zero network calls**. No other candidate, context frame or verified-visible/absent image was sent. Invalid/uncertain-response handling is tested; actual real replies happened to be schema-valid nonempty boxes, so no real absent-ball success case is claimed.

Only explicit human review can convert a new proposal into a label. Original labels and CV proposals survive the Gemini-only resume. Overlap groups retain each model's separate prediction; no cross-model confidence ranking approves boxes automatically.

## Storage, review and split safety

Ignored data lives under `~/Library/Caches/TactiqAI/m25`, reached through `outputs/m25`. It includes `videos.json`, `frames.json`, source sheets, image PNGs, exact-reference PNGs, deduplication audit, native M22 context/output caches, M16 raw outputs, Gemini request/cache evidence and **`pilot_20.json`**. Full videos, raw annotations, dataset images and model binaries are excluded from GitHub.

The review page filters the 20 images by video, annotation/review state and source exclusion. A separate append-only shot-type history supports broadcast gameplay, close-up, replay, celebration, crowd/non-gameplay, transition and uncertain. Excluded images retain all evidence but the server blocks new ball labels until eligibility is corrected. Existing trusted boxes display as the current human label; AI boxes remain unreviewed. Manual box corrections use original coordinates, never screen coordinates. New explicit human decisions append isolated **`human_revisions.json`** records with image/batch hashes and prior-reference provenance. Atomic replacement and fsync preserve revision history; GET requests never save or approve. Automatic navigation occurs only after the server acknowledges a completed decision. Original evidence stays unchanged; new review answers are not backfilled into original M19/M22/Giroud stores.

All split assignments remain **unassigned**, except Giroud's protected evaluation role. Original-match overlap between short clips, organizer training footage and the long broadcast is not fully established. Group by original match before later training/validation/test assignment; do not split neighboring frames randomly. Giroud remains excluded from any future fitting. No YOLO/heatmap training export, fine-tuning, annotation-accuracy score or model promotion is claimed. Native box/source/provenance records support future exports after human review and split audit.

## Verified checks and commands

- **1,175 backend tests run / 1,174 passed / one skip**, including ten new M25 coordinate, bounds, integrity, optional-route and append-only revision tests. The final fsync change also passed all ten focused tests.
- **189 frontend tests passed**; TypeScript and production build passed. Existing bundle-size advisory remains.
- Actual M25 Chrome check **passed (7.1 s)**: real pilot, long-video source selection, image display, Gemini overlay toggles, reversible shot exclusion, explicit-action auto-save and back navigation. Saves were intercepted; **no real human label was created**. Initial browser checks used an ambiguous native select label; explicit accessible labels fixed the locator.
- Rerunning the annotation command returned the cached pilot and made **no model/API calls**. No original source MP4 was persistently restored and no model was trained.

Run from the repository root using `~/Library/Caches/TactiqAI/backend-m17/bin/python`:

```sh
python -m scripts.prepare_cv_m25
python -m scripts.annotate_cv_m25
python -m scripts.propose_cv_m25_temporal
python -m unittest tests.test_cv_m25
npm --prefix frontend exec -- playwright test --config frontend/playwright.m25.config.ts
scripts/start_backend_m23.sh
# separate terminal
npm --prefix frontend run dev
```

The populated source/image roots are versioned evidence. Do not overwrite them to expand the experiment. `python -m scripts.resume_cv_m25_gemini --max-cost-usd 1` is the explicit Gemini-only phase; it now reuses the completed response caches and makes no new calls. The combined-proposal view is verified. **Stop here for the user to inspect/correct the 20 images.** After review, evaluate annotation quality and original-match overlap before deciding on expansion. **No training, extra Gemini batch or M26.**
