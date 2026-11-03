# TactiqAI CV Milestone 12 — Dataset expansion audit

Verified 2026-11-02. **IMPLEMENTED / LOCALLY VERIFIED / EXPERIMENTAL.** Added a bounded, provenance-rich SoccerNet Tracking research sample to frozen SoccerTrack-v2 training data. No production model is activated. **Worldwide physical-player disjointness is NOT independently certified.** Match/source-provided identity/hash checks and conservative repeated-club exclusions are verified; they are not a global player registry. The full strict physical-player separation acceptance requirement remains unverified rather than being silently weakened or claimed complete.

## Sources and permitted scope

|Source|Verified access and suitability|Decision|
|---|---|---|
|SoccerNet Tracking2023|Public official HF metadata and exact HTTP206 member reads; original GT boxes/track IDs and game/team metadata; moving main-camera footage,1080p/25fps|Used locally for research only|
|SoccerNet ReID2023|Public official archive metadata; manual action/replay correspondences. Author explicitly scopes identities to an action, not a global person. Train archive about12.11GB; central-directory read exceeded the32MB audit cap and was refused|Not downloaded as image data; not treated as globally identified people|
|SoccerNet GSR2024|Correct official repository `SN-GSR-2024` returns200/public metadata; author documents team/role/jersey tracking labels and200clips|Not downloaded; overlapping parent footage and large archives require further audit|
|SportsMOT|Official source documents persistent player GT within clips, broadcast football, CC BY-NC4.0 plus competition conditions and redistribution restriction; official access through Codalab enrollment|No enrollment/terms acceptance or mirror download performed|
|SoccerTrack-v2|Existing pinned ten-MOT-clip inventory, official CC BY4.0 attribution and M10 provenance;51training identities already anchored conservatively|Only frozen TRAIN reused; oldVAL/TEST never reallocated|

Sources: [SoccerNet official access documentation](https://www.soccer-net.org/data), [research/commercial-use FAQ](https://www.soccer-net.org/faq), [Tracking developer kit](https://github.com/SoccerNet/sn-tracking), [ReID identity semantics](https://github.com/SoccerNet/sn-reid), [GSR developer kit](https://github.com/SoccerNet/sn-gamestate), [SportsMOT official terms](https://github.com/MCG-NJU/SportsMOT#terms), [SoccerTrack-v2 official dataset](https://huggingface.co/datasets/atomscott/soccertrack-v2).

The SoccerNet FAQ permits research training and excludes intended commercial use. Public metadata access is not a blanket redistribution/commercial license. These models/crops stay local and experimental; no videos, crops or trained weights were published, uploaded to Azure or activated in the product. Source-code licences such as MIT do not replace dataset permissions. The separately gated88.7GBraw tracking pack was not accessed or bypassed. No NDA or competition terms were accepted on the user's behalf. This audit does not assert commercial rights to derived checkpoints.

## Availability checked before payload

Pinned official tracking revision:`3777f456d908570e10b19e7354744640e77c351e`. Train ZIP9,575,523,385bytes; test ZIP8,711,601,474bytes. Neither was downloaded in full. Read train ZIP directory (~4.63MB), then small game/clock/GT files. Actual train archive contains57clips across games4,6,9; actual test archive54clips across games7,8,11. This is the inspected payload, not a claim that the entire advertised12-game dataset was used.

One small frame per candidate game was inspected after checking access/use terms. Original GT overlays verified image/annotation alignment, team-side tags and camera domain. Member extraction requires the pinned public URL, exact byte-range response, expected archive length, ZIP name/length and CRC32. Saved source frames/crops receive SHA256s. Whole-archive LFS hashes are recorded as published provenance, **not claimed cryptographically reverified without downloading the archive**. A32MB metadata budget prevented an accidental12GBReID download.

Selected frames are read by bounded ranges, decoded in memory and discarded after crop extraction. Original full videos are unnecessary for this adapter. No arbitrary ZIP paths are unpacked.14candidate source clips were sampled (8train-source,3validation-source,3test-source); one best-supported anchor clip per identity is retained in the curated manifest. Noncurated research samples/provisional audit evidence remain ignored under outputs. No existing CV source video was removed.

## Identity and split semantics

Organizer MOT GT IDs represent annotated identity within each30-second clip. They are not generated TactiqAI tracker IDs. `gameID`, original team side and numeric organizer shirt metadata guard same-game aliases; those tags are never OCR output or neural model inputs. Alphabetical/unknown aliases, referees, ball/staff and new-source goalkeeper roles are excluded. The GT boxes themselves remain unchanged. No physical player name or face identity is inferred.

One anchor clip per same-game/team/numeric alias prevents treating the same known alias in another clip as a negative class. Positives are only original persistent GT observations **inside that anchor clip**; no cross-clip positive identity labels are invented. Source/recording/match/annotation identity, original GT track, timestamps, boxes, quality, masks and hashes remain separate fields. These labels support within-match research; the75-class count below is not independent worldwide biometric identity verification.

The small sample audit exposed the same white/blue club in games4 and9. All game4 left/white-blue players were excluded from validation. Final split:

- TRAIN: frozen SoccerTrack training matches plus SoccerNet game6, both audited kit groups; first-half clips097–104 only.
- New VALIDATION: game4 **right/green group only**, from clips060–062.
- New final TEST: game9 both groups, clips151–153. It was never in M10/M11 model selection or evaluation.

All source-provided identity keys, matches, team scopes, retained crop hashes and original source-frame/video hashes are disjoint across splits. Same-game repeated aliases never straddle splits. First-half scope avoids unverified halftime side changes. Current kit overlap is conservatively audited, but official metadata lacks registered global player IDs and full rosters; transfers and hidden cross-game identity overlap cannot be certified. **Namespace disjointness alone is not proof of physical-player disjointness.** This missing guarantee limits acceptance and blocks a stronger deployment/generalization claim.

|Split|Crops|Annotated identities|Sources|
|---|---|---|---|
|train|6,592|75|{'soccernet_tracking': 1231, 'soccertrack_v2': 5361}|
|validation|535|11|{'soccernet_tracking': 535}|
|test|966|21|{'soccernet_tracking': 966}|

Training expands51→75annotated identities (**+47.1%**) and5,361→6,592crops (**+23.0%**), adding24same-match GT identity anchors/1,231broadcast crops. This is useful but modest: only one additional training game. Curated validation has535crops/11identities; TEST966/21. Fixed early/late retrieval retains8validation and20test identities (24and60crop queries respectively). Some identities lack enough supported observations in both windows; the same protocol excludes them for every model before inference.

## Quality audit and near duplicates

New-source sampling uses original1-based MOT frames,25fps and12-frame spacing (0.48s). Per-GT-track perceptual-hash Hamming distance≥3 versus the previously retained crop rejects near duplicates. Crop dimensions, sharpness, brightness/contrast, clipping and overlap-proxy filters apply uniformly. Exact crop/frame hashes are checked across splits; source match separation prevents adjacent/event overlap. No claim of exhaustive near-duplicate proof across all public datasets or pretraining corpora is made.

The first provisional extraction produced an empty validation set: the inherited HSV “non-green foreground” rule rejected green uniforms as grass. That data was archived before training. The final new-source pipeline excludes that hue-based fraction from eligibility for **all** new-source splits, retaining it as metadata; size/sharpness/clipping/overlap checks remain. Anchor choice uses the clip with most supported observations, with deterministic lexical tie breaking. No model metrics or TEST retrieval influenced this correction. Original M10 crops/quality scores/manifests remain frozen.

Each saved crop masks the top18% before storage. Metadata records original pixel dimensions, sharpness, brightness/contrast/saturation, overlap proxy, bounded clipping, source annotations/hashes/CRC, pixel box, timestamp, known role/team provenance, unknown actual visibility/view direction and face-mask policy. Quality proxies are not independent occlusion/pose truth. No face model, jersey OCR or changed production filter was introduced.

## Unified pipeline and reproducibility

`ml/reid13/data.py` adapts both sources to a consistent manifest. Original GT/metadata remain under `source_metadata`; `dataset.json` is sealed before training. `dataset_summary.json`, `source_audit.json`, pinned ZIP inventories and extraction reports document every retained source/row. Frozen M8/M9 snapshots and M10/M11 models/data/results/reference cases are hash-preserved.

```sh
# From repository root; isolated existing CV runtime, retained M10/M11 assets required.
# New output root for an exact protocol replay, not a newly unseen TEST set.
export TACTIQAI_REID13_RUN_ROOT=/tmp/tactiq-m12-m13-reproduction
/tmp/tactiq-cv-m1/bin/python -m ml.reid13.setup --download
/tmp/tactiq-cv-m1/bin/python -m ml.reid13.data
```

Setup downloads only bounded public directory/annotation data and the small author OSNetx1.0 initializer. Crop extraction performs the selected bounded frame reads. No application-startup download, large archive extraction or cloud storage write. Existing optional `requirements-reid-research.txt` and CV/backend runtime dependencies suffice.

**IMPLEMENTED:** multi-source adapter, provenance/hash/CRC, quality correction, explicit GT identity scope, conservative match/alias guards, fresh validation/TEST separation and sealed dataset. **LOCALLY VERIFIED:** actual counts/ranges/samples/split hashes and original baseline preservation. **EXPERIMENTAL / NOT CERTIFIED:** global physical-player separation, only one added training game and one fresh TEST game. **PLANNED:** independently registered/roster-linked physical identities, more disjoint games/teams and a larger prospective broadcast evaluation. No additional tracker pseudo-labels manufactured.

[Machine-readable source audit](evidence/cv_m12_source_audit.json) · [M13 training/evaluation report](cv_soccer_reid_m13_training.md). The combined run stops at M12–M13.
