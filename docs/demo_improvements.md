# Match setup and grounded demo improvements

Implemented 2026-10-15. Start the API and frontend using [frontend.md](frontend.md). Restart an older API process to load these changes. No shots or existing control/pass/pressure event thresholds were retuned.

## Names and colors

On first visit, Match setup prompts for team names/colors. Expand “Name your players” to associate tracking references with player names, jersey numbers and a displayed team. Blank player names receive stable roster-order labels such as Player 4. Tracking references appear only in setup so the user can identify the right track. Reopen setup from the header at any time.

Mappings live in browser localStorage under the match-scoped key `tactiqai:118575_clip:identity:v1`, with schema validation, bounded text, hex colors and numeric jersey strings. Invalid stored data falls back safely. No secrets or match evidence are stored there. Labels remain user-provided, not verified identities. Canonical IDs and team analytics are unchanged; player team edits affect presentation/marker colors, not the underlying team-specific measurements. Team names/colors resolve throughout the React pitch, timeline, selected context, possession, formation and evidence. Names in questions are translated to internal references before calling the API. Another browser will need its own setup.

## Current possession and pass share

`attribute_possession` consumes the unchanged per-frame control output. Established control assigns player/team. Free flight, candidate changes, contested frames and missing observations keep the previous established controller for at most **20 seconds since last established control**. A newly controlled player switches attribution immediately; out-of-play clears it. Candidates do not extend the timeout. Before the first controller, after timeout, or after out-of-play, attribution is unknown until new control is established.

This is deterministic attribution, not proof of continuous physical contact. The passer retains attribution during a pass; receiver control closes the flight. An interception changes the team only at opponent control. It never feeds back into pass/shot detectors or the stricter tactical event engine. The pitch ring and current possession label use the attributed owner. Recording-wide possession uses attributed team-frame share; unknown time remains unassigned. Confirmed completed-transfer share remains separate and explicitly labeled.

The prepared clip has **5,032/6,000 frames (83.9%) attributed**, with 968 unknown frames under this policy. Confirmed transfer counts remain **16 vs 13**, with transfer shares **55.2% / 44.8%**. Neither figure is official football possession/pass accuracy.

## Formation and pressure

Formation display/Ask use fixed **240-second offline windows** of the existing quality-gated formation fits. At least 50% of a window's samples must contain a clear known fit; at least 60% of those known fits must agree on one formation. Otherwise the window stays unknown. The dominant label remains constant within that window. Future observations within the same offline window contribute to this descriptive summary; it is not causal pre-event evidence. Position, shape, control and pressure sample rates are unchanged.

For this clip, clear-fit coverage is only **24.1% / 7.9%**, so both formation summaries remain **unknown**. Stability comes from aggregation; no evidence threshold was loosened to force a formation.

Pressure display derives low/moderate/high from cached metric review positions around the attributed possessor, requiring at least eight observed defenders. This includes retained-possession intervals: it describes pressure around the last possessor, not around an independently flying ball. High means at least two defenders within 3 m, or two within 5 m plus a closing defender within 8 m. Moderate means one within 5 m, or a closing defender within 8 m. Otherwise it is low. Missing carrier/geometry or insufficient defenders means unavailable. Closing uses the existing backward velocity evidence. The conservative sustained-pressure candidate, persistence, event timestamps and cooldown remain unchanged. High pressure level alone is not a sustained event. Missing ball/possessor/defender geometry can legitimately make pressure unavailable even when possession attribution is retained. The clip has 743 available level samples (61.9%), compared with 291 confirmed-carrier pressure samples.

## Free-form Ask and English

Questions outside the finite Local router and compound questions spanning multiple evidence topics can now reach provider planning. Microsoft can plan **one to three allowlisted readonly tools**. Arguments pass the existing closed schemas; tool outputs supply facts and a deterministic English catalogue. A second bounded selection chooses only relevant supported sentences, possibly across datasets. It may decline with an empty selection. Invalid plans/arguments/selections return a natural unsupported/fallback answer; retrieved unrelated claims are never exposed after failed selection. Known questions keep their existing provider-independent fact grounding; explicit unsupported intent/causality questions are rejected conservatively.

New tools: `get_players`, `get_distance_leaders`, `get_positions`, `get_current_possession`, `get_confirmed_pass_share`, `get_pressure_level`, `get_formation_windows`. Existing `get_formation_timeline` now returns slow windows. Existing tools cover transfers, events, motion summaries, networks, tactical states, validation and earlier evidence. The current recording timestamp is supplied with free-form planning. The model cannot execute arbitrary queries, calculate new statistics, supply new sentences or invent identities. This is bounded evidence retrieval/composition, not unrestricted generative football analysis. Local mode remains network-free and has a smaller supported question vocabulary.

`src/query/language.py` renders supported facts as English sentences; `frontend/src/language.ts` renders evidence disclosures. Raw fields, coordinates, enum labels, fact IDs and serialized booleans stay in API structures, not normal React prose. All modes use readable language. Analyst opens supporting evidence; source captions remain understandable. User-provided labels replace internal actor placeholders before display. Neutral labels remain when identities are unknown.

## Artifacts and API

New exports: `possession_attribution.json`, `formation_windows.json`, `pressure_levels.json`. Future pipeline runs write them without changing detectors. Existing bundles can derive continuity/windows from cached canonical CSV/JSON at startup; no request runs detection. To prepare sidecars explicitly:

```bash
.venv/bin/python -m scripts.prepare_demo_context outputs/analytics_118575_tactics
```

`GET /matches/{id}/view` adds `ownership`, `pass_share`, `formation_windows` and `pressure_levels`. Existing Ask, player, explanation and media routes remain. Sources identify persisted sidecars or their source-artifact derivations. No identity endpoint is needed: identity is local presentation state.

## Verification and limits

Run the canonical Python suite, `npm --prefix frontend test`, `npm --prefix frontend run build`, and `npm --prefix frontend run test:browser`. Browser tests require installed Google Chrome, artifacts and media; they use isolated Local-only ports. Tests cover control/flight/interception/occlusion/noise/timeouts, formation windows, pressure levels, bounded multi-tool planning, unsupported and invalid selections, English formatting, stored identities, neutral labels and marker colors. Verified final results: **283 backend tests, 27 frontend tests, production build and 5 Chrome checks passed**. Core browser flow produced no console/page errors. Actual-artifact typed API/current-possession smoke and screenshot inspection also passed. Full current results are recorded in current_pipeline_status.md.

Microsoft REST retains its user-supplied **post-hardening live-verified** status: TLS enabled, explicit certifi, SSL_CERT_FILE absent/not required, microsoft_foundry, grounded_selection, fallback false. This task does not repeat an external live call. Newly enabled multi-tool planning is offline/mock-verified until a user-run live rehearsal. Agent Framework, Fabric, Azure MCP and Container Apps remain not live-verified/deployed.

Zero BAS shot references occur in this clip; zero heuristic candidates are acceptable. Remaining limits include offline smoothing/interpolated 2D ball, heuristic attribution and pressure, insufficient formation evidence, local-only user identities and bounded question coverage. Recommended next step: a judge rehearsal with the configured Foundry provider and user-entered roster, checking multi-tool answers and explicit unsupported cases.

## Possession clarification — 2026-10-15

**Overall possession** uses each team's confirmed completed passes divided by both teams' confirmed completed passes (16/29 and 13/29 here: 55.2% / 44.8%). Unconfirmed control does not enter this percentage. `get_possession_summary` and general possession questions use this pass-based proxy; the UI labels its basis explicitly.

**Current possession**, shown below the pitch, is the team/player currently attributed the ball. It uses established control and continuity during flight/short gaps, not the overall percentage. Current/now questions use `get_current_possession`. Unknown is shown only when attribution is unavailable. This clarification supersedes the earlier description of attributed time as the headline possession statistic; attribution remains available as an internal diagnostic.
