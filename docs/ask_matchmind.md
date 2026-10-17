# Ask MatchMind

The report includes suggestions, Beginner/Fan/Analyst modes, evidence disclosure, Explain cards, contextual follow-ups, timestamp seeking and actor highlighting. Offline HTML uses embedded deterministic answers without network calls. Service-served reports connect to the same-origin Ask API with local, Microsoft, Gemini or optional Agent Framework presentation. Hosted answers preserve canonical facts and select only supported statements; outages retain deterministic fallback. No model/API credentials enter the browser. Current verification and setup are in [Microsoft demo](microsoft_demo.md).

```bash
.venv/bin/python -m src.analytics.pipeline --summary-only --commentary-provider template --output outputs/analytics_118575_tactics
open outputs/analytics_118575_tactics/report.html
.venv/bin/python -m src.query --recording outputs/analytics_118575_tactics --question 'Who was the fastest player?' --mode analyst
.venv/bin/python -m unittest discover -s tests -t . -v
```

Try “Which team had more possession?”, “Show the most difficult completed transfer”, “When was Team 1 under the most pressure?”, “Which player connected with the most teammates?”, “How did Team 1 shape change in possession?”, “When did Team 2 use a high block?”, “What formation was detected?”, and “Explain this event.” A time such as `2:14` selects a nearby relevant event within two seconds; an unavailable turnover/shot returns unknown. Without an explicit time, event explanation selects the nearest eligible event to the scrubber. Unsupported questions such as fatigue, intentions, coaching instructions, names or scores return “MatchMind cannot determine that from the available tracking data.” The router is a conservative finite set of intents, not general natural-language understanding. Use left/right or Team 1/2; color aliases are declined because mappings differ by recording.

## Tool contract

`src/query/tools.py` reads only structured exports: summaries, events, transfers, tactical states, network inputs and prior evidence. It never queries raw tracking or invokes a provider. Actor IDs are the Python player-summary API namespace; tracker labels are explicitly distinguished from jersey numbers. Temporal network queries use the existing deterministic directed completed-transfer aggregator and confirmed-receipt windows. Peak player timestamps are exported alongside the existing maximum speed, using the same accepted smoothed speed samples. Missing peak times in older exports remain null.

All tool responses validate against closed `QueryResult`/`Fact` schemas. Scalar facts carry stable IDs, record IDs, exact field names, file provenance, event IDs and timestamps where available. Source selectors describe records/fields rather than pretending every CSV is a JSON pointer; transfer records are linked from canonical release frame/actor to `transfer_id`. Structured event and tactical inputs are also validated with existing closed models. Nonfinite numbers and unexpected response fields are rejected. Query argument schemas reject unlisted fields and enforce bounds; tool dispatch is an explicit allowlist with no file/SQL/network access.

```python
from src.query import MatchMindTools, dispatch, tool_specifications

tools = MatchMindTools('outputs/analytics_118575_tactics')
result = dispatch(tools, 'get_difficult_transfers', {'team': 'left', 'limit': 5})
print(result.model_dump(mode='json'))
# Future Foundry / Agent Framework adapters can publish these schemas:
specs = tool_specifications()
```

The interface includes match/player/possession summaries, filtered events, transfer lookup/ranking, tactical snapshot at or before a time, strictly earlier event evidence, sustained pressure sequences, formation transitions, transfer networks, fastest players and event validation. Natural questions about peak pressure use ranked observed proximity snapshots, separately labeled from sustained closing-pressure sequences. Network “most connected” counts distinct incoming/outgoing teammates in completed observable transfers, with deterministic tie ordering. Formation unknown reasons and sample timestamps are preserved. Shape comparisons use the existing whole-recording phase summary and are labeled accordingly.

## Grounding and explanation

`answers.py` formats only returned fields. Each statement identifies its supporting facts; CLI answers include the facts themselves. The report deduplicates them into `ask_matchmind.json`'s `facts` store. Analyst mode exposes fact IDs and file/record/field selectors. Modes use identical supporting facts; Beginner changes terminology and explanations, Fan retains football language, Analyst exposes provenance and limitations. A future provider can implement `StatementSelector.select` and return only a closed list of catalogue statement IDs. Extra free text, invalid IDs and unsupported fields are rejected. This retains the existing grounded sentence-selection philosophy instead of permitting unverifiable free-form numeric claims.

Explain combines what happened, transfer geometry/difficulty, strictly earlier tactical snapshots, and BAS validation state. Retrospective difficulty/lane/realized-target geometry is clearly distinguished from earlier pressure/shape. No explanation claims that pressure forced an action or that a player intended a pass/shot. Upstream centered smoothing and BAS-anchored interpolation remain offline inputs even though tactical sample timestamps strictly precede event release. Geometry cannot establish causal intent.

The browser uses `textContent` for generated statements and escapes `<`, `>` and `&` in embedded JSON. Tests inject script-closing text and a random API-key sentinel, execute generated JavaScript with a DOM/video stub, verify correct Explain IDs and timestamp seeks, and check unsupported answers and mode behavior. API credentials are neither read nor serialized by query tools. Future live provider integrations must stay server-side; this milestone does not expose a network service or add an unauthenticated API.

## Limits

Only the downloaded 118575 recordings can be evaluated today. A successful grounded answer does not make its inferred source statistic accurate. Coarse GSR, missing coverage, uncertain identity/role assignment, event-interpolated 2D ball speeds and poor shot detection remain visible limitations. Difficulty is retrospective and heuristic, not a calibrated completion probability. BAS sender/time support does not validate pass intent or outcome. No authoritative possession/formation ground truth is available. Single-player maxima can be influenced by tracking noise. Offline half reports embed more data than clip reports and may take longer to open; no hosted service is required.

“Before”/“sequence” questions also select the ten latest eligible events within a trailing 15-second window strictly before the original release/start; later receipt/outcome events are excluded. The Python `get_event_sequence(event_id, lookback_seconds=15)` tool supports bounded windows up to 60 seconds. Offline sequences use the report's representative event groups to avoid duplicate descriptions of the same transfer.

## Current interactive demo update — 2026-10-15

The React demo resolves user-provided team/player identities locally and translates names to stable references before API calls. API answers now use deterministic English catalogues. Questions outside Local intents, and compound multi-topic questions, can reach Microsoft planning for up to three closed tools; selection can decline unsupported evidence. Known single-intent grounding/fallback checks remain. Current possession uses a separate 20-second continuity attribution; completed transfer share uses `get_confirmed_pass_share`. Formation timelines summarize fixed 240-second windows; `get_pressure_level` separates geometric pressure from strict sustained events. [Full implementation and limitations](demo_improvements.md). Multi-tool live behavior has not been reverified externally during this update.

## Grounded question paraphrases — verified 2026-10-15

Ask now performs bounded semantic retrieval before declining familiar football questions. `src/query/planning.py` maps wording to approved closed tools; raw analytics/detectors are unchanged. Completed/successful passes and ball transfers use the same canonical completed-transfer records as overall pass-share possession. Counts, both-team comparisons, totals, count differences and team ranks are computed deterministically in the tool, with source-field provenance. Totals and comparisons appear early in the answer preview. Ambiguous “passes” is explicitly interpreted as team-level confirmed completed observable transfers, not every ball movement.

All five requested regressions resolve to **16 vs 13 completed transfers; 29 altogether**, from the same canonical fact set: “How many passes did both teams make?”, “How many completed passes did each side have?”, “Who passed more?”, “Which team completed the most passes?”, and “How many passes were there altogether?”. Highest speed/fastest, pressed/under pressure, formation/shape (unless the question explicitly requests changing geometry), and possession/pass-share wording are supported. Explicit team filters, both-side pressure retrieval, per-team peak-player speeds and bounded top-N speed rankings are covered. Supported combinations retrieve multiple approved tools even in Local mode. Unsupported free-form wording can still use bounded provider planning; known deterministic evidence survives provider failure. If transfer CSV is missing, canonical completed-transfer events are checked before declining. Only absence of both sources is unknown; valid zero counts and ties are answered accurately.

Verified commands: `.venv/bin/python -m unittest discover -s tests -t .` — **297 passed**; `npm --prefix frontend test` — **28 passed**; `npm --prefix frontend run build` — passed. The Local-only Chrome suite passed **6 checks**, including all five typed pass questions against real clip artifacts. No new external Foundry live call was made; prior post-hardening live status is preserved, and new semantic/provider behavior is offline-tested. This does not claim universal natural-language coverage or unsupported intent/causal facts.
