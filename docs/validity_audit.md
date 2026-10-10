# Measurement validity audit — 2026-10-10

Reproduce from the repository root:

```bash
.venv/bin/python -m src.vision.projection_evaluation --output outputs/validity_audit/projection
.venv/bin/python -m src.analytics.pipeline --summary-only --commentary-provider none --output outputs/validity_audit/baseline_analytics
.venv/bin/python -m unittest discover -s tests -v
```

## Projection findings

The existing baseline reproduces exactly, using all detections and the existing actor map, frame offset and piecewise Delaunay/barycentric projection. No transform, offset or actor-map changes were deployed.

| Metric | Before | After (unchanged method) |
| --- | ---: | ---: |
| Mean disagreement | 11.6831065 m | 11.6831065 m |
| Median / P50 | 10.7398110 m | 10.7398110 m |
| P75 | 18.5953233 m | 18.5953233 m |
| P90 | 23.4371475 m | 23.4371475 m |
| P95 | 24.9560484 m | 24.9560484 m |
| P99 | 26.4028016 m | 26.4028016 m |
| Maximum | 30.5152851 m | 30.5152851 m |
| RMSE | 14.1924825 m | 14.1924825 m |
| Finite pairs / detections | 130,782 / 132,000 | 130,782 / 132,000 |
| Coverage | 99.0772727% | 99.0772727% |

The main error is a **systematic negative y disagreement**: mean projection-minus-GSR vector `(0.1401, -10.8486)` meters. Interior pitch detections average 12.8336 m disagreement; detections within five meters of a boundary average 1.9575 m. The image center half averages 12.0828 m; outer quarters average 4.9886 m. These samples occupy different pitch regions, so this does not establish that the panorama edges are intrinsically more accurate.

Nearest-landmark distances below 100 pixels average 9.3829 m disagreement, versus 20.4825 m at 100–200 pixels and 20.5135 m at 200–400 pixels. Of 65 supplied landmarks, 42 lie on the two touchlines. This pattern is consistent with inadequate interior interpolation for a curved panorama, but does not isolate distortion, annotation contact errors or reference bias. Triangle aspect bins do not show a simple monotonic relationship with error. All associations are observational, not identified causes.

Shifting synchronization by -2/-1/0/+1/+2 frames yields means 11.68168/11.68170/11.68311/11.68597/11.69023 m with identical coverage. Such small changes do not explain the large spatial discrepancy or justify changing the existing image-based synchronization. Reversing x, y or both worsens mean disagreement to 41.5891/32.2027/55.0651 m; this is not a simple orientation reversal. The existing image-foot matching reports 14.24185 px mean assignment error over 21 checkpoints. A projected nearest-actor check agrees with the assigned actor only 39.38% of the time, but cannot adjudicate identity when projection error exceeds player spacing. The actor mapping was retained.

A supplementary matched-image check uses all 132,000 provider image foot positions: mean MOT/GSR image-foot disagreement is 13.6715 px, with mean image vector (−6.0910, +0.8034) px. At one-second checkpoints, the mapped actor is also the nearest image-foot actor 93.2008% of the time, supporting the existing map while leaving crowded ambiguities. Projecting the provider’s own image feet through the same unchanged transform does not remove the large metric discrepancy: on the identical 127,572 common finite pairs, MOT feet average 11.9189 m versus provider feet 12.1810 m. Native-foot projection has only 96.8720% overall coverage, so replacing feet would also lose coverage. This is an input diagnostic, not a calibration change or accuracy improvement, and argues against MOT foot estimation alone explaining the error.

No offset was fitted to the paired observations. Subtracting the mean error would optimize in-sample agreement and conceal the spatially varying bias. No independent ground contacts are available here to separate foot-point estimation, GSR quantization and projection errors. **No independent accuracy improvement is claimed.**

GSR remains primary for all 132,000 renderer positions. The fallback is particularly unreliable in sampled pitch-interior/central-panorama regions, and is unavailable outside its landmark hull (1,218 detections). Even near-boundary paired errors are not an absolute accuracy guarantee. Full per-player, triangle, image/boundary/landmark-bin and shift statistics are in `projection_evaluation.json`; exact pairs and temporal aggregates are in CSV. The four PNGs use actual finite paired detections; empty heatmap bins remain blank. Heatmap pitch bins cover the field; outside-pitch observations are retained in the separate boundary aggregates.

## Terminology, quality and compatibility

The original ownership and transfer calculations and numeric difficulty values are unchanged. The reproduced clip still has 6,000 frames at 25 FPS, 29 completed, seven opposite-team, two unresolved transfers and no trajectory/BAS shots. Opposite-team receipt is not proof of an intentional interception. `transfer_completion_rate` explicitly divides completed observable transfers by all detected candidates; it is not official pass accuracy.

`transfers.csv` is the preferred export, adding observation kind, BAS pass support, source and validation scope. `passes.csv`, old field names and old canonical PASS enum values remain compatible; `measurement_definitions.json` documents their observable-transfer meaning. BAS support verifies release-time/sender consistency only, not intent, completion or receiver. UI event labels say transfer, and grounded commentary already says estimated transfer.

`inverse_geometric_difficulty_0_1` is the preferred alias for the unchanged inverse heuristic. The legacy `heuristic_completion_score_0_1` is retained and explicitly deprecated as a name, not transformed into a probability. Report headers label the original 0–100 score retrospective heuristic difficulty. Estimated ball/shot-window speed headers explicitly say **2D ball speed**, and limitations continue to reject measured strike velocity or inferred ball height.

Player frame exports distinguish observed GSR / pitch-transform / unavailable coordinates and actor identity mapping. Canonical events distinguish observable transfers, BAS actions, inferred control and estimated motion; validation states and existing ownership confidence remain intact. The confidence is rule-derived, not probabilistic. Synchronization provenance remains in the summary and measurement definitions.

Phase 1 modified `src/analytics/pipeline.py`, `quality.py`, `report.py`, `src/intelligence/schema.py`, `events.py`, `commentary/service.py`, and the missing-key test isolation. It added `src/vision/projection_evaluation.py`, `tests/test_projection_evaluation.py` and this audit. All 88 tests passed after Phase 1, including the original 83. The original missing-key test had failed because provider initialization now loads a local `.env`; the test mocks that load instead of changing any credentials.
