# Evaluation and recording selection

The detectors retain their original thresholds. BAS release-time checks are consistency checks, not independent accuracy: the supplied ball path uses BAS anchors. Tracking identities are not verified jersey numbers. The projection audit remains unchanged: 11.6831065 m mean fallback disagreement with GSR, not surveyed ground-contact error. See [validity audit](validity_audit.md).

Run the two downloaded, disjoint halves:

```bash
.venv/bin/python -m src.evaluation --manifest configs/evaluation.118575.json --output outputs/evaluation
open outputs/evaluation/evaluation_report.html
```

The manifest accepts `recordings` with `match_id`, `scope` (`clip`/`half`), `half` (1/2), and `split` (`development`/`evaluation`). Global `analytics_config` and `tactics_config` overrides are exported with every result. No thresholds are fitted. Duplicate and overlapping recordings are rejected to avoid double counting. Development recordings are visible but excluded from aggregate results. Both halves of 118575 are additional recordings of the development match, **not held-out matches**; no cross-match generalization or independent validation is claimed. Evaluate the overlapping development clip separately with `configs/evaluation.clip.json`.

`--skip-sensitivity` skips the expensive one-at-a-time variants. Otherwise each recording runs ± variants of local pressure radius, pressure persistence, both block boundaries, formation RMSE, and formation persistence; full effective configs and output counts are retained. No variant is selected for nicer results. Geometric correctness and persistence are tested synthetically in `test_tactics.py`; sensitivity is descriptive, not tactical label accuracy.

Individual analytics runs now support:

```bash
.venv/bin/python -m src.analytics.pipeline --match-id 118575 --scope half --half 2 --summary-only --output outputs/analytics_118575_half_h2
```

Half two defaults to reversed attack direction. Override `left_attack_direction` in analytics JSON if the provider's orientation differs. Explicitly scheduled tactical direction changes remain available. A requested clip half must match its synchronization metadata. Custom dataset roots propagate to the report video link.

## Required files for another match

Only 118575 is installed. The resolver requires these paths under `--data-root`:

- `mot/<match>_sync.json`: match ID, FPS, clip half, zero-based clip offset, MOT-to-GSR actor/team map, side-to-color map, method/provenance.
- `bas/<match>/<match>_12_class_events.json`: released-half clock, matching FPS and action frame/actor labels.
- `ball/<match>_1st_ball.npz`, `ball/<match>_2nd_ball.npz`: consecutive one-based frames, centered metric x/y and status.
- `gsr/<match>/1st_compact.npz`, `2nd_compact.npz`: pitch, actor IDs and team sides; compact conversion also saves image foot coordinates.
- For clip alignment/rendering: `mot/<match>.txt`, `mot/clips/<match>.mp4`, and `raw/<match>/<match>_keypoints.json` for the fallback. Goalkeeper roles require `<match>_tracker_box_metadata.xml` or explicit `recording.goalkeeper_ids`; absent roles intentionally withhold outfield shape/formation.

The official source is [SoccerTrack v2](https://huggingface.co/datasets/atomscott/soccertrack-v2/tree/main), CC BY 4.0. It is gated: accept its terms and authenticate locally before downloading. Existing organizer clip download details and MOT normalization requirements are in `data/soccertrack/mot/README.md`. Do not treat the full-half panorama as the MOT excerpt. Example download commands (replace the selected ID):

```bash
hf auth login
hf download atomscott/soccertrack-v2 --repo-type dataset --include 'bas/118576/*' 'ball/118576*' 'gsr/118576/*' 'raw/118576/*' --local-dir data/soccertrack
# Place the paired organizer excerpt at mot/clips/118576.mp4 and zero-based tracks at mot/118576.txt.
.venv/bin/python -m src.vision.align_clip --match-id 118576 --data-root data/soccertrack
.venv/bin/python -m src.analytics.pipeline --match-id 118576 --scope half --half 1 --summary-only
```

Compact conversion derives half length from ball arrays and match paths from arguments. Synchronization is inferred from image-foot assignment; inspect its diagnostics rather than accepting arbitrary metadata. The alignment tool needs both halves, the clip, MOT, and fallback metadata used by team classification. Downloads are not automatically performed, and absent data are never fabricated.

## Interpretation

`evaluation_summary.json` includes every recording, sources/hashes/configs, matches with signed timing offsets, absolute timing mean/median/P95, possession diagnostics, goal references and threshold sensitivity. Per-recording analytics and `evaluation.json` live under `per_match/<match>_<scope>_h<half>/`.

Transfer checks use all observable transfer candidates and global one-to-one BAS pass-like matching with sender agreement. Class-level recall and matched/reference counts are identifiable. Class precision/FP/F1 are null because detector predictions do not carry PASS/HIGH PASS/CROSS/THROW IN labels; assigning all unmatched transfers to each class would fabricate class precision. Zero-reference recordings retain detections but no FP/precision/recall claim. Aggregate micro metrics use only reference-containing evaluation recordings and disclose exclusions. Possession has coverage/stability and pre-BAS sender agreement diagnostics, never ground-truth precision/recall. Candidate frames are counted separately. Ownership episode durations break at missing ownership; changes across such gaps are separately labeled. Warnings use explicit exported diagnostic thresholds.

For later report regeneration, `--reuse-sensitivity` reuses a prior sweep only when source hashes, analytics configuration, tactical configuration and deterministic implementation fingerprint agree. Otherwise the sweep reruns. An unavailable reference class never receives invented class precision; class timing summaries and recall are reported both per recording and in aggregates.
