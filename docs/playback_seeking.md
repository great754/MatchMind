# Event playback lead-in — 2026-10-17

This is presentation/navigation behavior. Canonical analytics, event confirmation rules, evidence values, source offsets and Azure artifacts are unchanged.

## Timestamp audit

| Concept | Existing meaning |
|---|---|
| Observation/start | Source-dependent observed boundary; not automatically publication time. |
| Release | Transfers' `metrics.start_frame / fps`; `transfers.csv.start_seconds` already exposes this to Ask. |
| Receipt/end | Transfers' `metrics.end_frame / fps`; can precede established receiver ownership. |
| Canonical event time | Immutable `event.timestamp`, derived from recording-local publication frame. |
| Confirmation/publication | Transfer events publish when receiver ownership is established (or unresolved termination is known). Shot candidates publish at candidate window end. Persistent tactical transitions publish at confirmation. Separate confirmation times are not universally stored. |
| Display/evidence time | Existing marker/answer/insight timestamp. An Ask transfer fact may already display release, while its canonical event marker displays publication. Neither is rewritten. |
| Playback seek time | The chosen supported action anchor minus pre-roll, clamped to zero and usable video/replay duration. |

Previously React `selectEvent` passed `event.timestamp` directly into the exact video seek. Evidence and insights also called that exact seek, sometimes with a release timestamp and sometimes publication. Technical report commentary, tactical buttons, Ask follow-ups and URL timestamps likewise sought their displayed time without a shared lead-in. React “Take me there” returned grounded evidence but did not automatically navigate.

## One policy, two navigation intents

`src/serving/static/playback.js` is the shared policy imported by React and embedded into technical reports. Its default is `DEFAULT_EVENT_PREROLL_SECONDS = 2`. The helper accepts an explicit override bounded to 0–10 seconds. No per-event threshold tuning or new environment variable is introduced.

`getPlaybackAnchorTime` uses:

- Transfer outcomes and derived pass classifications: recorded `metrics.start_frame / fps`.
- Trajectory shot candidates: recorded candidate `metrics.start_frame / fps`.
- Sustained pressure and tactical changes: recorded `tactical.evidence_start_frame / fps`.
- Possession transitions, motion milestones, BAS references and events without a valid recorded start: supplied canonical/evidence timestamp.

Start frames must be finite, nonnegative integers, use a known positive recording FPS and not follow publication. Formation fitting-window starts are not transition onsets, and durations are not subtracted to invent starts. Recording-local frame clocks are used directly; no half-video source offset is added.

`getPlaybackSeekTime = clamp(anchor - pre_roll, 0, min(replay_duration, known_video_duration))`.

React `seekMoment` handles timeline marks/rows, insights, Ask/Explain evidence timestamps, explicit Explain navigation and verified conversation follow-ups (“Take me there”, “Go there”, “Show that moment”). Evidence/markers/conversation retain their original timestamps. The pitch and playhead follow playback time; selected event/actor remain selected. Recording selection remounts review state and resets time to zero. Playback preserves its existing paused/playing state; press Play to watch a paused selection unfold.

Raw scrubbing, native video controls, Return to start and the report's continuous tactical-chart scrub remain exact. They do not apply pre-roll. `Evidence` itself does not modify its Answer or Fact objects.

The React Zod event contract now preserves `tactical.evidence_start_frame`, which it previously discarded. `/view`, event query responses and conversation schemas retain their canonical meaning; no playback time is stored as grounded conversation evidence.

## Technical reports and deep links

`src/serving/playback.py` installs the same JavaScript policy into HTML responses in memory, adapting commentary, tactical events, table time buttons, Ask timestamp/follow-up navigation and Explain cards. The saved Azure HTML bytes are never rewritten. Newly generated offline reports include the same policy. The installer is idempotent and script-escapes serialized evidence.

Use `?t=<canonical_seconds>&event_id=<canonical_event_id>` for an unambiguous event deep link. A timestamp-only legacy link still receives pre-roll; event metadata is used only when its timestamp identifies one canonical event. Ambiguous timestamps fall back to the supplied timestamp rather than guessing an action. Table times already represent source starts and receive lead-in as presentation timestamps.

## Actual artifact examples

All examples below are from the existing validated `132877_clip` bundle at 25 FPS. All five examples were artifact-verified and browser-tested against real Azure video.

| Type | Start/release s | Canonical/publication s | Receipt/window end s | Playback seek s |
|---|---:|---:|---:|---:|
| Completed transfer | 4.64 | 6.48 | 6.32 | 2.64 |
| Completed transfer | 10.44 | 12.12 | 11.96 | 8.44 |
| Completed transfer | 14.92 | 15.76 | 15.60 | 12.92 |
| Team width increased | 18.00 | 21.00 | unavailable; publication confirms persistence | 16.00 |
| Trajectory shot candidate | 165.84 | 166.72 | 166.72 | 163.84 |

These are heuristic/inferred events, not verified official football labels. No sustained pressure sequence exists in these ten bundles; pressure-start behavior is covered synthetically and a real tactical width transition supplies the required actual tactical example.

## Verification

Tests cover release/candidate/evidence starts, fallback clocks, early/invalid/future starts, configurable lead-in and duration limits; unchanged marker/evidence timestamps; synchronized pitch; timeline/insight/Ask navigation; closed conversation context; report adaptation/escaping/idempotence, generated browser execution and legacy deep links.

Complete backend suite: 356 tests, 355 passed and one optional deleted source-video skip. Frontend suite: 44 tests passed. TypeScript/Vite production build passed. Real Azure Chrome test passed: **1 test, 41.9 seconds**, covering three transfers, one tactical width change, one shot candidate, playing past each published time, Ask timestamp/follow-up navigation, recording reset within one frame, and an event-specific technical-report deep link. Before/after video frames were reviewed; this verifies playback lead-in/progression, not independent football-label accuracy. The browser test uses actual category-filtered timeline clicks because all-category markers can overlap at the same timestamp; it captures before/after screenshots and plays through each event. No new external model call or Azure write is used.

Reproduce browser verification with the Azure backend on 8000 and frontend on 5173:

```sh
cd frontend
nvm use 22.22.2
npx playwright test --config playwright.playback.config.ts
```

Real event output: `frontend/test-results/playback-real-events.json`; screenshots: `playback-<event_id>-before.png` and `playback-<event_id>-after.png`. These remain local generated evidence.

Health, readiness and recordings returned 200 after the fix; readiness reports ten usable recordings. All ten local Azure snapshots passed their original manifest hashes and schemas, including saved HTML. Evidence: `outputs/playback_artifact_verification.json`. No Azure upload/overwrite occurred.

## Files changed for playback

- Shared policy: `src/serving/static/playback.js` and its TypeScript declaration `playback.d.ts`.
- Report presentation: `src/serving/playback.py`, `src/serving/app.py`, `src/analytics/report.py` (HTML navigation only).
- React integration: `frontend/src/App.tsx`, `frontend/src/contracts.ts`, `frontend/vite.config.ts` (allows only the shared static directory alongside the existing frontend root).
- Regression/integration tests: `frontend/src/playback.test.ts`, `frontend/src/App.test.tsx`, `frontend/tests/playback-seek.spec.ts`, `frontend/playwright.playback.config.ts`, `frontend/tests/azure-clips.spec.ts`, `tests/test_playback.py`, `tests/test_intelligence.py`.
- Documentation: this page, `docs/current_pipeline_status.md`, `docs/current_pipeline_explorer.html`, `docs/frontend.md`.

The earlier demo evidence times remain correct; their playback landing changes intentionally. For example the speed evidence at 03:29.00 lands at 03:27.00, and the difficult-transfer insight at 02:21.28 lands at 02:19.28. No evidence fact or identifier was changed.
