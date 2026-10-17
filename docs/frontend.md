# Match Intelligence frontend

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
