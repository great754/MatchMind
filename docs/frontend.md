# TactiqAI Match Intelligence frontend

## Short-clip CV pass analysis (2026-11-13)

Refresh http://127.0.0.1:5173 and select **test_match**, **team_goal** or **southampton**. The new **Pass analysis** card separates completed-pass estimates from possible transfers. Current results are 0 qualified passes and 5/1/2 possible transfers respectively. This is not a statement that the footage has no passes; reliable ball/identity/team evidence is insufficient. Goals, shots and pass accuracy remain unavailable.

Expand **Review possible transfers** to watch source lead-in footage. Choose a player in **CV player focus** or click a source box/pitch marker for their report and available heatmap. Focus persists across missing frames. **Create a candidate review video** submits the request and produces a playable, visibly labelled review video with pre-event player marks. The assistant supports pass counts/profiles and explicit candidate review; ordinary canonical CV Ask stays unavailable. Developer mode retains detailed per-event evidence without cluttering normal video answers.

122 frontend tests/build and five real Chrome cases pass, including all three CV clips, marked review videos and prior SoccerTrack/full-half flows. Readiness remains 15. Existing Azure settings are unchanged; keep the original local CV videos. [Actual results, images, structured exports and limits](cv_pass_detection_v1.md).


## Player focus and one answer at a time (2026-11-11)

Click a player on the mini-map to focus the page and assistant on that player. The heading/banner confirm the selection; **Ask about this player** uses the current card. Ask without repeating their name. Selecting someone else clears the old answer and context. **Back to match** exits player focus. Each new question replaces the previous answer/video and cancels unfinished old rendering.

Video answers show the answer and video. Enable **Developer mode** to inspect per-event timestamps/source links, detailed metrics, diagrams and raw evidence. Stats answers retain the player report. Player videos automatically add a labelled marker from verified visible source observations before the action. Goal clips now request 15 seconds of lead-in normally, 20 for explicit buildup; the real three-goal compilation lasts 75.021333 seconds. Fastest-run requests show an honestly labelled measured run when no sustained sprint qualifies.

Verified: 119 frontend tests/build, four Chrome cases, live Azure pass/run/goals and /ready 200 with 15 usable sources. No setting change is needed; refresh localhost:5173. [Actual results, marker images and limits](player_focus_update.md). Earlier UI instructions/durations below are historical.


## Current workspace — score, player cards and goal videos (2026-11-11)

Refresh http://127.0.0.1:5173. The focused match screen shows the recorded score, original video, live pitch and one Ask input. Click a player on the pitch or choose one in **Player report** for shots, goals, completed/attempted passes, estimated pass accuracy, tackles, interceptions, receptions, crosses, distance and peak speed. **Heatmap** shows observed position occupancy; **Pass map** filters endpoint arrows and opens their original footage. Unavailable measurements stay unavailable; no fabricated rating or portrait is added. Names/jerseys need explicit confirmation in Match setup.

The loaded full match displays **Team A 0–3 Team B**. **Key moments → Goals** starts 30 seconds before a goal when available. Ask “Show me all the goals with buildup” for a playable compilation: the verified three-goal video has33.96 seconds of lead-in per goal and lasts120.021333 seconds. Requests from the second half correctly use the first-half goal source. The configured LLM interprets requests first; deterministic evidence supplies facts, and model failure is visibly identified as fallback.

Verified: 116 frontend tests, TypeScript/build, three real Chrome cases plus a final focused pitch-click/goal-lead-in check. Score, complete profile/stat chat, heatmap, pass filters, goal video playback/seeking, correct source navigation and mobile overflow passed. The current main API is ready with 15 usable recordings. The workspace requests a compact view (~80.45% smaller for 118575_h1), while full evidence remains available. No Azure configuration change is needed. [Actual results, limitations and exact start commands](match_experience_update.md).

Earlier sections below retain their milestone-specific verification results; this section supersedes their UI/highlight restrictions.


## Full match 118575 — select either half (2026-11-11)

Refresh localhost:5173 and choose **118575_h1** or **118575_h2**. Full Azure playback, seeking to25 minutes, tactical replay and grounded pass-count answers301/297 passed real Chrome verification. Both halves are already registered; no manual Azure setting change is needed. The final server has15 usable recordings, including all previous clips/CV sources. Half-local playback clocks remain explicit. Temporary labels continue A–Z, AA, AB, AC for the second-half roster.

Full-half generated compilations are now supported through bounded Azure ranges (see the current update above). Full-half player photo cards still require checked boxes; clip-only boxes are not reused. Keep local CV originals. [Full-match report, counts, startup and limitations](full_match_118575.md).


## M18 — one Ask TactiqAI conversation (2026-11-11)

The current production interface has one **Ask TactiqAI** input for questions, statistics, explanations and highlight requests. The older separate Highlight Creator instructions below are historical. Responses show readable answers, supported metrics, playable videos, source moments and follow-ups; raw IDs and JSON stay in the collapsed evidence inspector. Provider selection is under **Assistant settings**. Existing detailed evidence modes are under **Evidence display**.

**Match setup** now offers source player photos, three alternative views, enlargement, shirt swatches, jersey/name/team/role confirmation and undo. Tracker IDs are never assumed to be jersey numbers. Old browser labels need explicit server confirmation. CV labels apply only to the exact supported appearance segment, not uncertain merges. Many panoramic crops are small; no independent 80% jersey-readability result is claimed.

Verified: 113 frontend tests and typed build; 24 current Chrome cases across the full run and focused rechecks. Real goal/sprint/pass videos, original playback/seek, photo setup and recording isolation passed. Azure-backed readiness is 13. 132877_clip original playback works, but generated clips are safely rejected because its video frame count differs from the analytics. No source retiming or validation relaxation.

Run `npm --prefix frontend run dev` and open http://127.0.0.1:5173. Keep the existing Azure settings and local CV videos. [M18 architecture, exact backend command, evaluations, failures/rechecks and limitations](unified_match_assistant_m18.md).


## M17 — Highlight Creator (2026-11-08)

Use **AI Highlight Creator**, or type an explicit video request into **Ask TactiqAI**. The existing Ask form routes video requests to the creator and scrolls to it. Real MP4s support play/pause/seek, individual moments, original-footage links, more/shorter context and session-scoped follow-ups. Try goals or the longest pass on 118575_clip; annotated shots on 117093_clip. CV-only videos explain that verified football events are unavailable. Downloads require the explicit server rights setting and remain disabled by default. Current 23-test Chrome suite, 109 unit tests and typed build pass. Fresh language score 18/20 remains below 95%; see the documented supported phrasings. [M17 capability matrix, measured videos, limitations and startup](ai_highlight_creator_m17.md).


## Southampton — new current CV source (2026-10-27)

**LOCALLY VERIFIED / EXPERIMENTAL.** Fresh processing of `data/raw/southampton.mp4`:660frames,29.97FPS,1280×720,22.022s. Current M7 camera coverage **509/660(77.12%)**;current placement **6,114/8,725(70.07%)**.57 frames recovered relative to M6;zero accepted M6frames lost. Longest support9.843s;longest Unknown gap2.302s. Matched jump flags85→126:coverage does not establish accuracy.

**Known limitation:** dark-kit team separation is unreliable;the page visibly warns that A/B labels may mix teams. This is not verified team accuracy. Person/camera evidence remains separate. CV football Ask stays 503.

Select **southampton · CV Experimental** at http://127.0.0.1:5173. Playback/seeking/stepping/person and camera/team debug verified;actual Chrome overlay updates **29.91/sec**,no missing samples and at most one-frame lag in the2.5s sample.79frontend tests,TypeScript/build and **all eight Chrome suites passed(1.4m)**;backend528passed/one expected skip. All ten Azure clips and both earlier CV sources remain working. Health200;ready 200/**13 usable sources**;video 206;existing grounded Local Ask 200.

The fresh-video runner now completes M7 by default. Registry:`outputs/world_model/southampton/current/`. Keep the original local MP4;no Azure changes or uploads. Full fresh processing took806.528s;web playback serves precomputed data. [Full results,commands,team limitation and proof](cv_southampton.md). Earlier sections below retain their original verified counts.

## Playback smoothness follow-up — current frontend (2026-10-27)

**IMPLEMENTED / LOCALLY VERIFIED.** CV overlays now follow decoded video frames rather than low-frequency timeupdate events. Buffered look-ahead and a memoized camera timeline keep existing observations synchronized. Chrome measured **23.94 updates/sec test_match** and **24.76 team_goal**, with no missing overlays in the2.5s samples and maximum observed lag one frame. Model coordinates, M7 coverage and Azure artifacts are unchanged.

**79 frontend tests,TypeScript/build and all six affected CV browser suites passed (29.2s).** Exact seeking, stepping, camera/team/player inspection and recording isolation passed. Refresh http://127.0.0.1:5173;no reprocessing/configuration change is required. [Measured proof and implementation](cv_playback_smoothness.md). Earlier M7 camera verification below remains the geometry authority.

## Current M7 persistent camera review — 2026-10-27

**LOCALLY VERIFIED / EXPERIMENTAL CV.** Both CV recordings now serve M7. Camera motion debug adds orange persistent fixed-world feature histories,active anchors,held-out errors,hull/lifetime/survival,uncertainty,measurement route,top-line/circle residuals and explicit ambiguous semantic labels. Current source/provenance,team/identity/visibility and historical position remain separate. Coverage camera/placement:test_match33.64%/32.22%,team_goal91.00%/88.05%;test_match50% target not met.74 frontend tests,TypeScript/build and all six final Chrome suites pass;all ten Azure clips/playback/seeking/tactical/event lead-in/grounded Local Ask remain working. CV football Ask 503. [Full M7 report and exact commands](cv_world_model_m7.md). Earlier milestone sections are historical.

## Historical M6 temporal review — 2026-10-26

**LOCALLY VERIFIED / EXPERIMENTAL CV.** Both CVsources now serve M6snapshots. Visible marker provenance legend, optional Camera motion debug(static points/vectors, accepted/rejected lines, top touchline and derived intersections), previous/predicted/current regions, future/reobserved-anchor diagnostics, readable player jump components and persistent-vs-current team evidence are locally/browser verified. Frame1near0.058s in team_goal is recovered backward;350is rejected. Current coverage is21.44%/19.91%camera/placement for test_match and83.40%/81.99%for team_goal;independent accuracy remains unavailable.71frontend tests/ build passed;five full Chrome suites45.1s, final affected panel recheck5.1s. Existing football lead-in, Azure playback and grounded Local Ask remain working. CVfootball Ask remains unavailable. [Full M6 report and commands](cv_world_model_m6.md). Earlier milestone sections are historical.

## Historical M5 CV camera review — locally verified 2026-10-25

Select `test_match · CV Experimental`. The registry serves `outputs/world_model/test_match_m5/`. Inspect the541-frame camera timeline, hover current reasons/landmarks/support/direct age, click to seek inside the selected frame, enable **Projected pitch geometry**, and click a person for readable camera/foot provenance. Direct solid, temporal hollow, partial dashed and historical ghost positions remain distinct; UNKNOWN has no current metric markers or canonical overlay. M4 team/kit/image memory controls remain available.

Camera25.32% / player placement23.65%; ≥30% not reached. Actual midfield frame216 stays Unknown. Partial provenance is propagated and does not grant CV football analytics or Ask readiness. Diagnostic frame-boundary precision is separate from unchanged football event lead-in. Full59 frontend tests/build pass; four full Chrome suites pass, followed by all three affected CV suites after final formatting. [M5 report, evidence, commands and limitations](cv_world_model_m5.md). Earlier milestone notes are historical.

## Event playback lead-in — 2026-10-17

Moment navigation now uses the shared two-second playback policy while displayed evidence timestamps stay unchanged. Manual scrubbing remains exact. See [timestamp policy and real examples](playback_seeking.md); this supersedes earlier exact-landing demo timestamps.

## Key Match Insights — 2026-10-17

The existing React review now includes deterministic Key Match Insights from the additive view.intelligence projection. Cards reuse Evidence, identity mapping and synchronized seek; Analyst exposes canonical source locators. All modes use identical facts. No additional provider request is needed to render insight modes. See [milestone](intelligence_milestone.md) for tested demo timestamps and verification scope.

## Frontend startup repaired — 2026-10-17

The long Vite startup was traced to offloaded dependency files in the cloud-backed Desktop project: `frontend/node_modules/vite/dist/node/cli.js` was marked `compressed,dataless`. The pasted logs showed repeated config/source change events, optimizer rescans and server restarts. The old dependency tree occupied only 31 MB of physical disk despite its installed files; reads required file-provider hydration. The backend remained healthy independently.

`frontend/scripts/runtime.mjs` now installs the unchanged package-lock dependencies into `~/Library/Caches/TactiqAI/frontend/<lock-platform-node-hash>/` on macOS (a user cache directory on other systems), then links `frontend/node_modules` to that local runtime. Installation is staged and published only on success; first preparation installed 111 packages in 2 seconds. The previous dependency directory is preserved as ignored `frontend/node_modules.cloud-backup-<timestamp>`. No source files, .env or credentials are copied. Ordinary `npm run dev`, `npm test`, `npm run build` and `npm run preview` use this helper; a changed lockfile/Node platform gets a fresh runtime automatically. The helper keeps child commands on the caller's Node version. `frontend/.nvmrc` pins the verified Node 22.22.2.

Vite's optimizer cache also lives outside cloud storage. Dependency scanning is limited to index.html with explicit app dependency entries; native file-provider change events are avoided with 500 ms polling while retaining HMR. Test configuration moved to `frontend/vitest.config.ts`, so starting Vite no longer loads Vitest configuration. The previous offloaded dependencies and new symlink are ignored by Git.

Verified results with Node 22.22.2:

- `npm run dev`: Vite ready in **1,099 ms**; frontend left running at **http://127.0.0.1:5173/**.
- `npm test -- --pool=threads --maxWorkers=1`: **35 passed**, five files, **8.28 seconds**; workers started in about 342 ms each.
- `npm run build`: TypeScript checks and production build passed; Vite transformed 2,024 modules and bundled in **776 ms**.
- Actual Chrome: API connected in **581 ms**, all ten recording options present, video readyState=4, playback advancing, no video/page errors, and one navigation during a 15-second reload-stability check. Evidence: `frontend/test-results/frontend-startup.json` and `frontend/test-results/frontend-startup.png`.

These results supersede the earlier frontend worker timeouts, killed type-check processes and stalled production builds. The Azure backend and all ten published recording bundles were not changed by this startup repair. Start normally with `cd frontend && npm run dev`; no Azure configuration change is needed.


The React/TypeScript/Vite app lives in `frontend/`. It presents recording `118575_clip` using existing deterministic artifacts. `docs/apple-design.md` informed system typography, restrained color, clear grouping, immediate button feedback, accessible focus and reduced-motion behavior. No new football detector or browser analytics was introduced.

## Run locally

Use Node 22.22.2+, 24.15+ or 26+. From the repository root, start the API:

```bash
MATCHMIND_MEDIA_ROOT=data/soccertrack/mot/clips MATCHMIND_PROVIDER=local \
  .venv/bin/python -m uvicorn src.serving.app:from_environment --factory \
  --host 127.0.0.1 --port 8000 --no-access-log
```

In another terminal:

```bash
cd frontend
# Locked dependencies are prepared outside cloud storage automatically.
npm run dev
```

Open http://127.0.0.1:5173. Restart an older API process to load the new route. The prepared registry and `outputs/analytics_118575_tactics` must exist; video uses `data/soccertrack/mot/clips/118575.mp4`. Missing media does not prevent evidence review. To use the already configured Foundry provider, set `MATCHMIND_PROVIDER=microsoft` on the server or select Microsoft Foundry in the UI. Existing server `.env` names remain unchanged. No secrets belong in frontend environment files.

## Architecture and contracts

`SoccerTrack → offline analytics → canonical artifacts → FastAPI/closed tools → optional provider → React` remains the architecture. The browser only formats units/percentages, selects already computed samples, filters canonical events and draws geometry.

| Route | Purpose |
| --- | --- |
| `GET /matches/118575_clip/view` | New closed read-only projection: summary, possession, replay samples, canonical events, tactical states, sources, limitations and safe media URL. |
| `POST /matches/118575_clip/ask` | Existing bounded tools and provider selection; question, mode, provider, timestamp and recording-scoped conversation. |
| `GET /matches/118575_clip/events/{id}/explain?mode=analyst&with_evidence=true` | Existing deterministic explanation plus backend evidence categories; omitting the new flag preserves the old response. |
| `GET /matches/118575_clip/players/{id}` | Existing player summary and provenance. |
| `GET /matches/118575_clip/video.mp4` | Existing restricted media route and byte-range playback. |
| `GET /matches/118575_clip/report` | Existing detailed report. |

`review_frames.json` is exported alongside future offline reports and allowlisted for artifact bundles. Older installed bundles safely parse the JSON literal embedded in their precomputed report at **startup**, without executing HTML/JS or reading raw tracking. Missing review samples return explicit unknown. Requests do not run detectors. Zod validates browser responses against the closed API shapes.

One React timestamp drives video, pitch, timeline and tactical context. Native playback/seeking updates it; event/answer jumps set both it and `video.currentTime`. Pitch uses floor(timestamp × FPS / stride), and tactical state uses the latest sample at or before that time. There is no browser interpolation. The prepared clip contains 1,200 5 Hz review samples and 22 actors, using existing GSR-derived team assignments including goalkeepers. White ball, team colors, possession ring and selected-player ring are distinct. Pressure labels identify a supported carrier; defender identities are not fabricated from a count.

Selecting an event seeks, shows its source/ID and requests its explanation. Evidence groups (observed, inferred, validated, heuristic, unknown) come from the backend. Ask chips use the same API as typed questions. Beginner/Fan/Analyst re-request the backend presentation; facts stay canonical. Replies show actual provider/fallback state, expandable facts, provenance, limitations and timestamp jumps. Actor IDs are explicitly not verified jersey numbers. Unsupported questions, missing formations/media/samples, network errors and timeouts remain visible.

Possession now uses continuity-attributed control, retaining the last established possessor during flight or short gaps for up to 20 seconds; unknown time stays unassigned. Completed-transfer share is displayed separately. Four-minute formation windows and geometric low/moderate/high pressure levels are distinct from faster shape snapshots and stricter sustained-pressure events. See [the focused demo improvements](demo_improvements.md) for exact rules, sources and verified limits.

## Verify and understand the code

```bash
.venv/bin/python -m unittest discover -s tests -t . -v
cd frontend
npm test
npm run build
npm run test:browser
```

The browser suite requires installed Google Chrome and the prepared local artifacts/media. It starts isolated Local-only servers on 18000 and 15173, leaves an existing port-8000 server untouched, and sends no model requests. It exercises real artifact loading, fastest-player/pressure Ask, jump-to-video, player evidence, modes, event explanation, unsupported questions, missing media/state, simulated fallback and 1024/390 px overflow checks. Screenshots go to ignored `frontend/test-results/`. A simulated provider fallback checks UI handling; it does not establish a real Microsoft outage.

Read `src/App.tsx` for coordination, `src/time.ts` for sample selection, `src/Pitch.tsx` for drawing, `src/Evidence.tsx` for answer display, and `src/contracts.ts` / `src/api.ts` for API boundaries. Backend projection lives in `src/serving/review.py`; contract tests in `tests/test_frontend_review.py`. Frontend tests live beside components and in `frontend/tests/demo.spec.ts`.

Manual demo: play/pause and scrub; click a pitch actor; filter and select a transfer; inspect explanation and source; ask fastest player and jump to 02:51.28; ask pressure and jump to 00:21.00; switch modes; select Foundry to exercise the configured server provider; ask an unsupported coaching-intent question. Check that video, pitch and displayed timestamp follow each jump. Repeat at narrow width and with media absent.

## Security and limits

The browser calls FastAPI only, through a narrow same-origin Vite proxy. Provider/storage credentials stay server-side; only validated, match-local identity labels in browser storage, no unsafe HTML rendering, broad CORS or new static-file server is used. No auth or deployment was added. `npm run build` produces `frontend/dist`; a hosted demo still needs same-origin API/media proxying and an appropriate authentication boundary. `vite preview` alone is a static-build preview, not the configured development API proxy.

Microsoft REST is **POST-HARDENING LIVE-VERIFIED**, as supplied by the user: SSL_CERT_FILE absent, explicit certifi verification, `fastest_players` / `get_fastest_players`, `microsoft_foundry`, `grounded_selection`, fallback false and empty diagnostics. This frontend task used Local-only verification and made no new external model calls. Agent Framework, Fabric, Azure MCP and Container Apps are not claimed live.

Limits remain offline centered smoothing, event-interpolated 2D ball positions, unknown formations/identities, heuristic possession/pressure/difficulty and no independent held-out match. The next useful step is a judge rehearsal with the configured Foundry provider, followed by authenticated hosting if desired.

## Current setup and English answers

First-visit Match setup accepts team names/colors and player names/jerseys/team assignments keyed to existing tracking references. Only this validated presentation mapping is saved in browser localStorage; canonical IDs remain unchanged. Neutral labels replace long IDs when names are absent. React answers, evidence and selected context use readable sentences and source captions. Analyst retains understandable detail without serializing raw fields. Microsoft free-form planning can retrieve up to three closed tools; Local keeps conservative supported intents. The new behavior and API fields are documented in [demo improvements](demo_improvements.md).

## Possession clarification — 2026-10-15

**Overall possession** uses each team's confirmed completed passes divided by both teams' confirmed completed passes (16/29 and 13/29 here: 55.2% / 44.8%). Unconfirmed control does not enter this percentage. `get_possession_summary` and general possession questions use this pass-based proxy; the UI labels its basis explicitly.

**Current possession**, shown below the pitch, is the team/player currently attributed the ball. It uses established control and continuity during flight/short gaps, not the overall percentage. Current/now questions use `get_current_possession`. Unknown is shown only when attribution is unavailable. This clarification supersedes the earlier description of attributed time as the headline possession statistic; attribution remains available as an internal diagnostic.

## Grounded question paraphrases — verified 2026-10-15

Ask now performs bounded semantic retrieval before declining familiar football questions. `src/query/planning.py` maps wording to approved closed tools; raw analytics/detectors are unchanged. Completed/successful passes and ball transfers use the same canonical completed-transfer records as overall pass-share possession. Counts, both-team comparisons, totals, count differences and team ranks are computed deterministically in the tool, with source-field provenance. Totals and comparisons appear early in the answer preview. Ambiguous “passes” is explicitly interpreted as team-level confirmed completed observable transfers, not every ball movement.

All five requested regressions resolve to **16 vs 13 completed transfers; 29 altogether**, from the same canonical fact set: “How many passes did both teams make?”, “How many completed passes did each side have?”, “Who passed more?”, “Which team completed the most passes?”, and “How many passes were there altogether?”. Highest speed/fastest, pressed/under pressure, formation/shape (unless the question explicitly requests changing geometry), and possession/pass-share wording are supported. Explicit team filters, both-side pressure retrieval, per-team peak-player speeds and bounded top-N speed rankings are covered. Supported combinations retrieve multiple approved tools even in Local mode. Unsupported free-form wording can still use bounded provider planning; known deterministic evidence survives provider failure. If transfer CSV is missing, canonical completed-transfer events are checked before declining. Only absence of both sources is unknown; valid zero counts and ties are answered accurately.

Verified commands: `.venv/bin/python -m unittest discover -s tests -t .` — **297 passed**; `npm --prefix frontend test` — **28 passed**; `npm --prefix frontend run build` — passed. The Local-only Chrome suite passed **6 checks**, including all five typed pass questions against real clip artifacts. No new external Foundry live call was made; prior post-hardening live status is preserved, and new semantic/provider behavior is offline-tested. This does not claim universal natural-language coverage or unsupported intent/causal facts.

## Recording selector

React now discovers `GET /recordings`, enables processed recordings and disables pending configurations. Selection remounts the review, resets conversation/time/context, scopes API/report paths and uses a match-specific identity storage key. `118575_clip` remains the default; `128058_clip` is now processed and registered; restart the API/reload React to discover it. See [second-recording requirements](second_recording.md).


M8 verification (2026-11-01):all three CV sources now use validated separate M8 snapshots;all ten Azure clips retain playback and grounded Local Ask. Health/ready/catalog200,13usable sources,9Chrome suites passed. CV football Ask stays unavailable. Keep original local CV videos;no Azure upload/configuration change. [Current metric camera report and startup command](cv_world_model_m8.md).


M9 (2026-11-01):all three CV sources now expose the Player World Model Inspector with persistent anonymous hypotheses,team belief,visibility timelines and global ReID evidence. M8 camera/position observations remain unchanged. Identity gains are modest and team errors remain;these are experimental scores. 607backend tests pass/one expected skip,86frontend tests and10Chrome suites pass. [Current M9 report,commands and limitations](cv_player_world_model_m9.md).


M15 experimental ball comparison:choose a CV recording,then **Ball review → M15 experimental**. M14 remains default. Video observations,distinct expiring predictions,validation-derived image uncertainty regions,candidate scores and explicit pitch unavailability remain synchronized through seeking. Learned verifier/local crops are disabled after validation and their displayed scores are diagnostic. No unsupported pitch markers are drawn.102frontend tests/build and11Chrome suites plus M14 regression passed. [Full measured M15 status and limitations](cv_ball_tracking_m15.md).

## M16 comparison review — 2026-11-03

For `test_match`, `team_goal` or `southampton`, **Ball review** now offers M14, M15 and **M16 experimental**. M14 remains the default. The M16 selector is enabled only when its separate sidecar has passed source/hash/schema/clock validation; explicit invalid M16 requests return 503 instead of silently changing models. Observations, image-only predictions, uncertainty proxies and unavailable pitch coordinates remain distinct and synchronized to source frames.

M16 improves fresh organizer-box TEST F1 to 54.04% (M14 28.43%, M15 18.84% on identical samples), but does not reach the 85% target and regresses on team_goal and southampton. It is an optional comparison, not a production replacement. The complete validated frame now serializes directly to JSON to reduce playback latency without dropping evidence. Local CV videos remain needed. Start the existing backend and `npm --prefix frontend run dev`; no Azure configuration change is required. [M16 measurements and commands](cv_ball_tracking_m16.md).


## M19 optional CV review (2026-11-15)

M19 is a local, versioned CV sidecar experiment, selected through **Ball review → M19 association · opt-in**. Azure canonical bundles, media settings and existing MATCHMIND_* names are unchanged. Missing/invalid optional M19 sources do not remove valid baseline recordings; explicit M19 queries return unavailable instead of silently using another source. The pass panel, profile and unified assistant use the same selected version and reset prior question/video context on changes. Keep the local test_match/team_goal/southampton originals; current CV review and independent annotation need them.

A model-blind independent source-review page is available at `/?recording=test_match&review=m19` (also team_goal/southampton). There are 60 sealed tasks and currently zero human labels; no accuracy claim is made. Labels are separate from model candidates and manual player identities, exported through `/cv/review/export`, and stored under ignored outputs. Do not commit labels, videos, checkpoints, caches or credentials. M19 performed no Azure uploads or long-video processing. Actual live health/readiness and all three clip playback ranges were verified; readiness has 15 usable recordings. [Measured M19 results, test evidence and reproduction commands](cv_tracking_passes_m19.md).


## M20 opt-in trajectory tracking (2026-11-18)

For an existing local CV clip, select **Ball review → M20 Trajectory Tracking · opt-in**. Current-frame crop inference, IMM image-motion predictions, covariance, competing/rejected candidates and loss/recovery diagnostics are synchronized to the original source video. M19 comparison, player focus and the unified assistant share the selected validated event version; predictions never become football facts. M14 remains the default, and existing M14/M16/M19 modes remain available.

M20 adds trusted local `cv_ball_m20.json` / `cv_events_m20.json` registries and a protected `/matches/{id}/cv/trajectory` endpoint. Invalid optional outputs return unavailable for explicit M20 requests without removing valid baseline readiness. Existing Azure artifact/media/authentication settings and MATCHMIND_* variable names are unchanged. Current real readiness is 200 / 15 usable recordings. The four saved human annotations and original videos/checkpoints/artifacts were preserved. Independent accuracy is pending human review; this engineering run needs no labels. No Azure upload, hosted LLM call or long-video CV run occurred. [M20 architecture, actual per-clip results, regressions and commands](cv_trajectory_tracking_m20.md).
