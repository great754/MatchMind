# Measured evaluation: 118575

Both downloaded halves were evaluated with unchanged detector thresholds and a one-second matching tolerance. Transfer matching requires sender agreement. These are BAS consistency checks, not independent accuracy; ball reconstruction uses BAS anchors. Neither half is a held-out match.

| Recording | Detector | Candidates | BAS refs | TP | FP | FN | Precision | Recall | F1 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Half 1 | transfers | 432 | 556 | 344 | 88 | 212 | 79.6% | 61.9% | 69.6% |
| Half 1 | shots | 25 | 10 | 3 | 22 | 7 | 12.0% | 30.0% | 17.1% |
| Half 2 | transfers | 435 | 538 | 362 | 73 | 176 | 83.2% | 67.3% | 74.4% |
| Half 2 | shots | 21 | 11 | 2 | 19 | 9 | 9.5% | 18.2% | 12.5% |
| Aggregate | transfers | 867 | 1094 | 706 | 161 | 388 | 81.4% | 64.5% | 72.0% |
| Aggregate | shots | 46 | 21 | 5 | 41 | 16 | 10.9% | 23.8% | 14.9% |

Absolute transfer timing error: aggregate mean 0.1037 s, median 0.08 s, P95 0.40 s. First-half median/P95 are 0.04/0.32 s; second-half 0.08/0.44 s. Three first-half BAS GOAL references were inspected/exported; there is no independent goal detector evaluation. Shot precision and recall are weak, and no improvement is claimed.

Possession sanity (not accuracy): first-half controlled 27.90%, unknown 27.22%; second-half controlled 27.88%, unknown 29.76%. The halves contain 697/750 ownership episodes, including 122/149 shorter than 0.2 s. Other state percentages, ownership changes, sender agreement and diagnostic thresholds are in the JSON. Absence of a diagnostic warning does not establish correctness.

Each half includes 12 one-at-a-time tactical sensitivity variants. The baseline detects no sustained press; reducing persistence from 2 s to 1 s yields 12/16 sustained-pressure samples in the two halves. Increasing local radius from 5 m to 6 m yields 4 sustained samples in half two. These differences expose threshold dependence; no variant was chosen or promoted to the default. Half-one accepted formation team-samples change from 2,167 at baseline to 3,373/1,433 with 3/7-second persistence; these are sampled estimates, not verified formations.

Second-half formation estimates are unknown throughout. The current formation engine requires exactly ten outfield identities in its recording roster; the second half includes 15 left-team and 14 right-team actor identities across substitutions. It reports `incomplete_roster` even when an individual sample has ten observed outfield players. Active-lineup/substitution handling is a remaining limitation, not hidden or patched during frozen evaluation.

The four-minute development clip still reproduces 29 completed, 7 opposite-team and 2 unresolved transfer candidates, with zero BAS/trajectory shots. Its separate development evaluation is under `outputs/evaluation_clip/`, so overlapping clip/half events are never double counted. No-reference shot precision/recall remain null.

Projection audit remains 11.6831065 m mean fallback disagreement; GSR-first policy is unchanged. Only 118575 is installed; cross-match performance is unknown. See [evaluation instructions](evaluation.md) for file preparation and manifests, [Ask MatchMind](ask_matchmind.md) for grounded query contracts, and generated `outputs/evaluation/evaluation_report.html` for every per-recording result.

Validation: 142 tests pass, including the original 117, synthetic geometric gates, evaluation counts/timing, multi-match path/output isolation, closed query dispatch, grounded known answers, unknown questions, mode consistency, safe HTML text, Explain IDs and timestamp seeking. Generated clip JavaScript was also executed against its real embedded data in a Node DOM/video harness. Headless Chrome exited without rendering, so a visual browser preview was unavailable.
