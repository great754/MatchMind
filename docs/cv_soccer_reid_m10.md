# TactiqAI CV Milestone 10 — Soccer-ReID v1

Verified 2026-11-02. **IMPLEMENTED / LOCALLY VERIFIED / EXPERIMENTAL.** First dedicated local ML training experiment completed. **SOCCER-ReID v1 BEATS GENERIC OSNET BUT INTEGRATION IS NOT YET JUSTIFIED.**

The representation passes its preregistered held-out retrieval gate against both original M9 OSNet and the matching generic crop baseline. It is not activated in TactiqAI. The existing M8 camera system, M9 player snapshots/team evidence/fusion, generic checkpoint, production analytics, provider/grounding and all working MATCHMIND variables remain unchanged. Research explorer: **http://127.0.0.1:5174/retrieval_explorer.html**. Active application remains **http://127.0.0.1:5173**, using M9.

## Final measured comparison

All percentages below use the same held-out samples/protocol. “Tracklet” here is a deterministic GT-identity temporal window, not an independently detected/recovered physical identity.

|Held-out metric|Generic M9 OSNet|Soccer-ReID v1|
|---|---|---|
|Crop Rank-1|28.68%|37.21%|
|Crop Rank-5|65.12%|77.52%|
|Crop mAP|31.50%|42.23%|
|Same-team crop Rank-1|28.68%|37.98%|
|Same-team crop Rank-5|68.22%|79.84%|
|Same-team crop mAP|33.71%|43.88%|
|Same-team mean-tracklet Rank-1|58.14%|88.37%|
|Same-team mean-tracklet mAP|69.04%|92.19%|
|Same-team quality-weighted tracklet Rank-1|60.47%|90.70%|
|Same-team quality-weighted tracklet Rank-5|86.05%|95.35%|
|Same-team quality-weighted tracklet mAP|71.13%|93.69%|
|Same-team pair ROC-AUC|65.34%|74.94%|
|Same-team pair false-match rate at fixed0.82|18.97%|1.90%|
|Same-team pair false-nonmatch rate at fixed0.82|63.31%|87.08%|

The false-nonmatch regression at the frozen0.82 gate matters: better ranking is not a calibrated drop-in similarity score. These are crop-pair rates, not the percentage of actual M9 reconnects lost. M9 uses tracklet medians and additional gates. No threshold was changed to improve the result.

TEST contains2unseen matches,44eligible annotated identities/4,867crops. Fixed retrieval windows provide129queries across43identities; one identity lacks sufficient early/late observations and is excluded by the predeclared protocol for every model. This is a small annotation-defined benchmark, not an independent human-verified global player-identity accuracy estimate.

|M9 deterministic graph ablation|Generic OSNet|Soccer-ReID shadow only|
|---|---|---|
|test_match:accepted / ambiguous / rejected targets|1 / 1 / 85|1 / 3 / 83|
|test_match:persistent hypotheses / fragments per entity|98 / 1.010|98 / 1.010|
|team_goal:accepted / ambiguous / rejected targets|1 / 4 / 30|2 / 2 / 31|
|team_goal:persistent hypotheses / fragments per entity|42 / 1.024|41 / 1.049|
|southampton:accepted / ambiguous / rejected targets|6 / 1 / 28|8 / 0 / 27|
|southampton:persistent hypotheses / fragments per entity|42 / 1.143|40 / 1.200|
|Simultaneous duplicate violations|0|0|
|Impossible accepted metric links|0|0|
|Existing reference linked cases:qualitatively plausible|4|4|
|Existing reference linked cases:indeterminate|4|4|
|Existing reference linked cases:apparent failures after documented amendment|0|0|

Shadow totals are8→11accepted reconnects,not deployed M10 player snapshots. The fixed existing reference set covers only part of the changed edges. No independent identity-switch accuracy is available. One plausible existing test_match516→521link becomes ambiguous. New links include plausible goalkeeper/official continuity,but several remain boundary or longer-gap hypotheses. The small qualitative set also contained an erroneous M9 label,amended transparently below. These results are promising,but do not establish safe application-level identity improvement at unchanged thresholds. **M9 remains active.**

## 1. Baseline freeze and boundaries

Before implementation,read current status/explorer,M9 report,evaluations,candidate decisions,caches and all three actual snapshots. `frozen_m9.json` records snapshot/manifest/candidate/cache/checkpoint/registry hashes. `baseline_preservation.json` verifies the final snapshots,deterministic player/team/reachability/global-assignment code,analytics code,registry and generic initializer are unchanged.

M8 camera/placement counts remain:

- test_match269/541camera;2,179/5,108current placements.
- team_goal456/500camera;4,861/5,515current placements.
- Southampton600/660camera;7,419/8,725current placements.

M9 accepted reconnects remain1/1/6in the actual app. The three professional broadcast videos are never part of supervised training,validation or TEST. Their source hashes are disjoint from the official SoccerTrack clips. They are **development/reference demonstration footage**,not quantitatively labelled held-out ReID ground truth.

No ball perception,face recognition,roster-name identification,team retraining,camera retraining,new tactical analytics or learned candidate scorer was started. Production startup does not import the ML training package.

## 2. What supervision actually provides

Audited all10available organizer MOT files,original roster XML,existing hash-validated artifact MOT→GSR mappings,resident compact GSR arrays and bounded original GSR JSON samples. Dataset provenance is pinned to Hugging Face revision`325bd4c57ee8e135ab652f3921d274526ada496b`. [Official dataset](https://atomscott.github.io/SoccerTrack-v2/) and [MOT format documentation](https://github.com/AtomScott/SoccerTrack-v2/blob/main/docs/format-mot.md) describe sequence-persistent annotation IDs; actual file contents take precedence over generic format examples.

|Identifier|Meaning and use|
|---|---|
|M9 TRACK_NNN|Generated temporary visual track;never supervised identity truth|
|Released MOT id|Organizer ground-truth persistent sequence ID;target scoped as match:MOT:id within that released clip|
|GSR track_id|GSR sequence track key;not interchangeable with the same integer MOT id|
|GSR attributes.player_id|Registered player key that joins roster metadata;used for mapping/leakage checks,not an assumed crop-to-player join|
|GSR annotation id / image_id|Annotation-instance/image keys,not player identity or directly a clip timestamp|
|XML player id/teamId/shirtNumber|Registered roster key,club/team key and jersey metadata;no names used|
|GSR attributes.team=left/right|Pitch-side metadata;not a universal club ID or shirt colour|
|MOT→GSR artifact mapping|Existing inferred spatial synchronization,with retained provenance;not independently annotated physical identity truth|

A concrete inspected example:118575 GSR track12has attributes.player_id476367(role goalkeeper,jersey1),while the existing clip mapping associates that player with MOT21. Joining MOT12directly to GSR track12would be wrong. Compact118575arrays contain coords/pitch/ids/sides with71,850half-frames;their ids include registered key476352. Full GSR JSON files are cloud-offloaded multi-gigabyte placeholders. Only bounded8KiBprefix/16KiBtail source samples were read remotely;the full JSON was not hydrated or used as the crop-label source. Existing GSR alignment was not changed.

The released MOT files use actual0-based frames in this pinned copy and placeholder−1last columns;do not fabricate confidence,role or visibility from those fields. The local README documents118578's earlier normalization. The dataset generator verifies decoded frame count/fps against the paired file and records original frame base. Identity training uses the organizer's persistent GT ID only within its clip. It does not assert a globally verified physical identity across every recording. Annotation noise and inferred roster/team mappings remain limitations.

All8used Azure video byte hashes match the official paired MOT videos at the pinned revision,verified against published LFS SHA256s. `official_video_hash_verification.json` records every match,size and hash.2,649,891,856bytes were read through existing private Azure access. Downloads were read-only and transient;largest single video604,313,569bytes. Temporary MP4s were removed after extraction. Existing local CV source videos and Azure blobs were not deleted or changed.

## 3. Dataset generator,split and leakage prevention

`ml/reid/audit.py`,`dataset.py`,`protocol.py` retain each sample's match/recording,frame,timestamp,GT identity,registered mapping/team,source bbox,quality,crop hash,source video hash,annotation hash and row/mapping provenance. The original raw detection/annotation sources are preserved.

Fixed nominal groups:

- TRAIN:117092,117093,118575,118576,118577,118578.
- VALIDATION:128058,132831.
- TEST:128057,132877.

Because teams/registered players recur across matches,reserve whole TEST teams/registered IDs first,then VALIDATION,then TRAIN.117092/118578have no eligible training players after this conservative guard. The actual training footage is117093/118575/118576/118577. Within TRAIN,retain only the first match per registered-player or team+jersey alias,avoiding cross-match false negatives without manufacturing cross-match positive labels. Require≥30usable samples per identity;one five-sample validation identity is removed. Versioned split:`configs/ml/soccer-reid-v1-split.json`. Exact curation manifest/hash is retained,not a random image split.

|Split|Matches used|Identities|Crops|Small / medium / large crops|
|---|---|---|---|---|
|train|117093, 118575, 118576, 118577|51|5361|2347 / 2755 / 259|
|validation|128058, 132831|32|3612|1776 / 1589 / 247|
|test|128057, 132877|44|4867|1571 / 2833 / 463|

There is no train/validation/test overlap in match,annotation identity,registered ID,team or source video hash. Team/jersey alias exclusion is conservative;it is not biometric proof that no person ever changes registration or shirt number across seasons. Training positives come only from understood within-sequence GT,not M9 tracker output.

Sampling is every50frames/2s per GT identity,with perceptual-hash near-duplicate rejection. Retain width/height/area,sharpness,foreground fraction,brightness,contrast,saturation,boundary clipping,overlap/occlusion proxy and quality score. Filter very small/weak/clipped/high-overlap crops. True visibility and view direction are not available;record null/unknown rather than inventing labels. Raw annotations cannot prove every desired front/side/back/occlusion transition.

Training uses P=8identities,K=4crops,one crop from each temporal quartile. Seven identities come from one team where possible,with one other-team identity;same-match teammates are preferred. Each augmentation run samples338,688same-team negatives versus112,896different-team negatives:75%of negative pairs are same-team;281,888are also same-match.48,384positive pairs per run span temporal variation. Batch-hard mining emphasizes the closest same-team negative;easy negatives remain present. Body-scale/view/lighting variety is encouraged by footage and sampling,not claimed as independently labelled pose categories.

### Quality audit limitation

The frozen v1manifest contains5,789slightly negative raster clipping-proxy values(min−0.09481),because integer raster crop area can exceed fractional source-box area. This is a metadata quantization artifact,not negative physical clipping. Quality scores were already bounded[0,1],and both models used exactly the same recorded pooling weights. The measured experiment,crops,selection and TEST results were **not rewritten** after discovering it. Future generator output clamps the reported clipping fraction only;the executed original generator is archived in`training_source/dataset.py`. The difference and regression test are explicit in`quality_metadata_audit.json`. This prototype issue is another reason to retain full provenance and qualify quality-proxy claims.

## 4. Architecture,loss and crop/augmentation ablations

Start from the untouched M9 OSNet x0.25 MSMT17-combineall initializer;do not train from random weights. Fine-tune conv4,conv5,fc and a new51-class identity head. Earlier layers and their batch-normalization statistics remain frozen. Output is a normalized512Ddescriptor. The identity head is training-only,not a roster-name recognizer.

Loss is identity cross-entropy plus same-team-prioritized batch-hard cosine triplet(margin0.3),a conventional metric-learning baseline rather than a novel loss. [OSNet paper](https://arxiv.org/abs/1905.00953) · [Triplet-loss reference](https://arxiv.org/abs/1703.07737).

Crop comparisons on VALIDATION before fine-tuning:

|Generic OSNet preprocessing|Same-team crop mAP|Same-team weighted tracklet mAP|Selection score|
|---|---|---|---|
|m9_body|29.92%|56.02%|0.48193|
|full_masked|30.92%|56.54%|0.48853|
|multi_region|28.89%|52.47%|0.45395|

A reproduces M9 body below the top18%;B preserves full-body layout with the head region neutral-masked;C averages shared-backbone full/upper/lower-region embeddings. Saved crop assets mask the top18%before training. No face-recognition model or facial identity evidence is used. All strategies preserve torso/shorts/socks information where visible;C is deterministic averaging,not a Transformer.

B was selected using generic validation score alone. On TEST,the generic B baseline is worse than generic A:crop mAP29.28%versus31.50%,same-team mAP31.66%versus33.71%. The fine-tuned B model beats **both**,so the result is not merely choosing a favourable crop after inspecting TEST.

Two runs use identical initialization,seed and six-epoch maximum:augmentation off versus moderate brightness/contrast/saturation,mild blur,resolution degradation,limited horizontal flip and small partial occlusion. No arbitrary rotations or destructive jersey transforms. AdamW,learning rate0.0003,weight decay0.0005,batch32,168steps/epoch,patience3. Python/NumPy/PyTorch/data sampling seed20261102. Twelve actual epochs/checkpoints were retained.

Selection score=0.7VALIDATION same-team quality-weighted tracklet mAP+0.3VALIDATION same-team crop mAP. Moderate augmentation epoch6wins:validation same-team crop mAP36.39%,weighted tracklet mAP65.94%. Unaugmented training reaches lower training loss but does not win the selected validation criterion. Training/validation curves are generated from actual epoch history. Validation loss is fixed-sample triplet loss,not meaningless classification loss over unseen validation IDs.

The generic benchmark is measured on VALIDATION before training. TEST remains sealed until the selected checkpoint/config/data hashes are locked. Generic A,generic matching B and the selected trained B are then evaluated once in the final held-out comparison,following the stronger isolation requirement. Later strata/failure analysis reuse the cached TEST vectors;no model,epoch or threshold is selected from those diagnostics.

## 5. Held-out protocol and generalization

Crop queries are three temporally spaced observations per eligible identity from10–70s;galleries are three observations from≥170s. Galleries are conditioned on the same match,then restricted to the same team for the critical protocol. No adjacent-frame positive retrieval. Tracklet windows hold10–30observations each,mean versus quality-weighted pooling. Actual query/gallery tracklet separation is102–112s. Tracklets are built from GT temporal windows and therefore do not measure detector fragmentation or real occlusion recovery by themselves.

Predeclared representation gate:≥3percentage-point gains in same-team crop Rank-1,same-team crop mAP and same-team quality-weighted tracklet mAP,with overall crop mAP regression≤0.5point. The trained model passes against both original generic A and matching generic B. Gate/config hashes are retained. Test identities and held-out teams are unseen by training and selection under available metadata.

Same-team size breakdown:

|Query size|Queries|Generic Rank-1 / mAP|Soccer Rank-1 / mAP|
|---|---|---|---|
|small|43|27.91% / 34.31%|32.56% / 43.90%|
|medium|74|28.38% / 32.76%|41.89% / 44.49%|
|large|12|33.33% / 37.42%|33.33% / 40.05%|

Small<45pixels high,medium45–84,large≥85. Gains do not come solely from large crops. Only12large query crops exist;small strata have uncertainty. The overlap-proxy group has10queries:Rank-1remains10%for both,although mAP improves. The sampled retrieval queries contain no sharpness<50cases,so no heavy-blur subgroup accuracy is claimed. View direction remains unlabelled;front/back observations are qualitative examples only.

Post-hoc outfield-only diagnostic(removing roster-labelled GKs,not used for selection):same-team Rank-123.33%→33.33%,mAP29.05%→39.85%,weighted tracklet mAP68.97%→93.21%.120queries/40identities. The gain is not explained only by visually distinctive goalkeepers.

Per-match diagnostics show gains on both held-out matches;132877has the largest tracklet gain. With only two test matches,no broad population confidence interval or professional-broadcast generalization claim is justified. Training data is fixed full-pitch university footage;M9 demonstrations are moving-camera professional broadcast footage with predicted/noisy boxes.

## 6. Hard negatives,false negatives and research UI

`hard_negative_analysis.json`,`failures_0.jpg`,`failures_1.jpg` show highest-scoring wrong same-team candidates and weak true positives. `false_negative_analysis.json` and dedicated sheets show the lowest-scoring true pairs. Source crop review finds tiny/blurred detail,very similar uniforms,pose/scale and lighting changes. Back-print/front appearance changes are qualitative,not a labelled view benchmark. High similarity/low true-pair similarity alone does not prove a source-label error;none was independently established in these sheets.

The standalone retrieval explorer has129selectable queries and top10candidates for generic/trained models,with similarity,same/different annotated identity,match,team and measured quality. Crops remain head-masked. Chrome verified query changes,two model columns,20loaded candidate images,no page errors and bounded page width. Curves,similarity distributions and result/decision JSON are linked. Research-only status is explicit. It is not an activated production model selector.

## 7. M9 shadow ablation and reference audit

Only appearance embeddings change. Reuse identical eligible source observations,raw detections/boxes,segmentation,feature-quality/histogram evidence,stored M9 segment team labels/strengths,geometry,reachability,time limits,thresholds and global assignment code. Graph aggregation stays M9's median,not the new benchmark's weighted pooling. Generic graph replay exactly reproduces the original accepted-edge lists before comparison. No M10 canonical player snapshot is generated or activated because the integration decision is withheld.

`shadow_comparison.json` includes full metrics,candidate margins and every existing reference case. `reference_endpoint_comparison.json` adds actual endpoint-crop similarities,so zoom windows do not rely on trivial same-tracklet cosine1.0. The full26case generic/trained crop+tracklet similarity table is retained;do not infer identity certainty from a high score.

Specific results:

- test_match18→33:tracklet similarity rises roughly0.931→0.955,but neither model links it;missing independent continuity is not repaired by appearance alone.
- test_match516→521:generic link becomes AMBIGUOUS despite stronger learned appearance,owing to global competing-candidate margins. This is a visible regression,not hidden by the total count.
- team_goal boundary link233→256:LINKED→AMBIGUOUS;the original qualitative identity remained indeterminate.
- Southampton short-gap red-player links and2–3squalitatively indeterminate links remain. The learned graph adds long boundary/official-like continuity and another red-player link;these are not independent identity truth.
- Long-gap unresolved examples remain unlinked. Hard physical impossibility still rejects even high similarity. Global one-to-one consistency and uncertainty are preserved.

A closer source-context review corrected the old`team_goal_01`qualitative label:frames69/92show the upper-field player plausibly continuous;the M9 bottom-boundary interpretation was mistaken. Original M9 evidence is untouched. `docs/evidence/cv_m10_reference_amendments.json` and `reference_amendments.json` preserve original/amended interpretations. It must **not** be counted as an apparent new false match. Expanded Southampton crop review also shows yellow official-like kit at both endpoints of116→213,not an official/outfield-person crossing. Reference labels are qualitative development annotations,not independently verified ground truth.

The final integration decision is conservative:ranking improves substantially,but the frozen0.82 crop-pair threshold is not calibrated for the learned score distribution,a plausible existing application link is lost,and stronger broadcast identity references are needed. This is not a claim that the learned model is worse at retrieval. No threshold retuning against TEST,no learned scorer and no second training search were used to force deployment.

## 8. Runtime,registry and reproducibility

Apple M4 MacBook Air,16GBunified memory,8-core GPU;MPSavailable outside sandbox,CUDAunavailable,CPUfallback supported. Warm-up training step4.67s;steady synthetic batch32step≈0.0423s. Before the bounded run,estimated5–15minutes,168batches/epoch,up to6epochs/run,two runs,below1GBworking-memory target and≤605MBtransient video. Peak training memory was not measured,so the target is not a measured peak claim.

Actual model selection wall interval≈319.79s(5m20s):174.37straining steps plus95.17svalidation inference,with remaining time in generic baselines/I/O/checkpoint work. Dataset extraction/download is separate and each match retains measured extraction time. Final artifact logical size≈241MBbefore final reports;all temporary videos were deleted. No cloud GPU training or storage write was performed.

OSNet outputs512D;selected checkpoint:

`outputs/ml/soccer_reid_m10/run_moderate/epoch_06.pth`

SHA256:`584568ee5b04237fdf4893b20d8e1eb7f3de2ff2086e10d6f0993981a0ab47cb`

Training manifest SHA256:`6cb957ed439cb580af5fa826cc3726881f15aac4af73304d59e194c0f8d74eb3`

Training configuration SHA256:`c1097c1af82ee299dda54282737d5683d7c27a2929a4f681fe2080a18f3531a3`

Initializer SHA256:`cf55163d78fc44c62c82f85ab62d39f10438679b5abe8c698ae08cfa84aa6e18`.

`model_registry.json` records model/version/architecture,date,dataset/config/checkpoint hashes,evaluation status and version fingerprint. `candidate_models.json` retains all12candidate versions with validation results;only the selected model was evaluated on TEST. Checkpoints include weights,optimizer state and Torch RNG state;seeds,packages,selection lock,history and executed training-source copies are retained. Torch2.14.1,Torchvision0.29.1,NumPy2.5.4 and OpenCV5.0.0were measured. MPS numerical nondeterminism may prevent bitwise-identical retraining;CPU inference determinism is tested. The original generic checkpoint remains unchanged.

## 9. Tests and live regressions

- **642backend tests run:641passed,one expected optional-media skip.**34new ML tests cover identity/split/source/team leakage,temporal positives,same-team PK negatives,quality filtering/metadata,bounded crops,hash/version integrity,embedding normalization/deterministic inference,mean/weighted pooling,held-out isolation,integration gating and unchanged physical/global constraints. Existing M8/M9/SoccerTrack/Azure/Ask regressions pass.
- **86frontend tests across15files passed;TypeScript/production build passed.**
- **Nine application CV browser suites passed**,and the Azure suite passed all10clips on its isolated retry. The first attempt timed out waiting for first-clip video readiness;minimal real range check returned206and the unchanged Azure-only retry passed(1.3m). No code/timeout/validation was weakened to bypass it.
- **Research explorer Chrome suite passed**:query selection,loaded evidence,both models and width/no-error checks. This is additional to the10application suites verified across initial/retry runs.
- Live health/ready/catalog200;13available sources;all3CV metadata remain camera m8/player m9. Existing Azure view200,grounded Local Ask200/statusok/14evidence rows. CV football Ask remains503through the unchanged application regression. Existing Azure storage,provider architecture and grounding remain unchanged.

Logs/screenshots/JSON:`outputs/ml/soccer_reid_m10/verification/`,`live_regression.json`,`research_explorer_browser.png`,`baseline_preservation.json`. Machine-readable portable results:[M10results](evidence/cv_m10_results.json).

## 10. Commands and files

Training code lives only in`ml/reid/`:audit,dataset/protocol,sampling,model,losses,train,evaluate,tracklet,registry,finalize,shadow and research visualizations. No production inference file was edited. Versioned configs are under`configs/ml/`;reference amendments and summary under`docs/evidence/`. Heavy crops/weights/vectors remain ignored under`outputs/`;no videos or trained weights were added to Git.

Current research explorer command:

```sh
/tmp/tactiq-cv-m1/bin/python -m http.server 5174 --bind 127.0.0.1 \
 --directory outputs/ml/soccer_reid_m10
# http://127.0.0.1:5174/retrieval_explorer.html
```

Explicit offline pipeline(does not run at application startup):

```sh
# Requires existing authorized dataset/Azure access and hash-validated artifact mappings.
# Use a NEW root for a new experiment;the completed selection/test is sealed.
export TACTIQAI_REID_RUN_ROOT=/tmp/tactiq-soccer-reid-reproduction
/tmp/tactiq-cv-m1/bin/python -m ml.reid.audit --download
/tmp/tactiq-cv-m1/bin/python -m ml.reid.dataset
/tmp/tactiq-cv-m1/bin/python -m ml.reid.train --benchmark
/tmp/tactiq-cv-m1/bin/python -m ml.reid.train
/tmp/tactiq-cv-m1/bin/python -m ml.reid.finalize
/tmp/tactiq-cv-m1/bin/python -m ml.reid.shadow
/tmp/tactiq-cv-m1/bin/python -m ml.reid.visualize
```

Exact v1crops/manifests and executed source are retained. Current generator includes the documented clipping-metadata fix;for exact v1data provenance use the retained frozen manifest/crops/source rather than silently regenerate it. To repeat training using the exact retained data,copy`curated_manifest.json`into a clean root;its crop paths still refer to retained source assets. Existing selection locks refuse overwrite/reselection.

Regression commands:

```sh
/tmp/tactiq-cv-m1/bin/python -m unittest discover -s tests
cd frontend
npm test -- --run
npm run build
TACTIQAI_AZURE_BROWSER=1 TACTIQAI_AZURE_REQUIRE_ANALYSIS=1 \
 ./node_modules/.bin/playwright test --config playwright.m9.config.ts
TACTIQAI_REID_RESEARCH=1 \
 ./node_modules/.bin/playwright test --config playwright.reid-research.config.ts
```

Active backend command is unchanged:

```sh
AZURE_TOKEN_CREDENTIALS=AzureCliCredential MATCHMIND_PROVIDER=local \
 MATCHMIND_CACHE=/tmp/matchmind-azure-startup-cache \
 TACTIQAI_BACKEND_PYTHON=/tmp/tactiq-cv-m1/bin/python scripts/start_backend.sh
```

## 11. Limits and stop condition

**IMPLEMENTED:**supervised annotation-scoped dataset,conservative match/team/player separation,metric-learning fine-tuning,crop/augmentation comparisons,held-out same-team/tracklet evaluation,cache/hashes/registry,shadow ablation,retrieval explorer and failure evidence.

**REFERENCE-EVALUATED:**existing M9 source-image cases plus changed-link review,with an explicit annotation amendment. This is development evidence,not independent physical-player identity accuracy.

**NOT INTEGRATED:**Soccer-ReID v1in the production M9 player model. No new M10serving snapshots,team classifier,camera model,ball pipeline or learned reassociation scorer.

**PLANNED:**validation-only calibration of crop/tracklet/global-assignment score distributions,independent broadcast reappearance labels,more unseen matches,better tiny/occluded/view-direction supervision and further quality-proxy validation. Future model selection needs fresh held-out discipline;do not iterate against this TEST result.

The answer to the milestone question is **yes on this held-out annotation-defined SoccerTrack benchmark**:the soccer-specific representation materially improves same-team and long-gap tracklet retrieval. It is **not yet established as a safe drop-in application improvement** under M9's frozen score thresholds/reference quality. Active TactiqAI therefore stays on generic M9.

**SOCCER-ReID v1 BEATS GENERIC OSNET BUT INTEGRATION IS NOT YET JUSTIFIED.** Stopped after M10;no ball milestone was begun.
