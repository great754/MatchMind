"""Offline CV milestone. No Azure registration, artifact upload or football inference."""
import argparse,hashlib,json,platform,time
from collections import Counter,defaultdict
from pathlib import Path
import cv2
import numpy as np
from .schema import Config
from .camera import CameraProvider,SoccerTrackCameraProvider
from .detection import YoloXDetector,TeamClassifier,MODEL_SHA256,MODEL_URL
from .tracking import ShortTermTracker
from .engine import WorldModel
from .provider import canonical_observation
from .evaluation import ReferenceEvaluator
from .visualize import annotated_frame,compact,write_debug

def run(video,model,output,calibration=None,camera_events=None,reference=None,config=None,confidence=.4,soccertrack_calibration=None):
    started=time.perf_counter();video=Path(video).resolve();output=Path(output);output.mkdir(parents=True,exist_ok=True)
    if (output/'diagnostics.json').exists():raise ValueError('Use a new output directory; existing experiment is preserved')
    config=config or Config();cv2.setNumThreads(4)
    cap=cv2.VideoCapture(str(video))
    if not cap.isOpened():raise ValueError('Video could not be opened')
    fps=float(cap.get(cv2.CAP_PROP_FPS));width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH));height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT));expected=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if not np.isfinite(fps) or fps<=0:cap.release();raise ValueError('Valid video FPS required')
    detector=YoloXDetector(model,confidence,config.nms_threshold)
    camera=SoccerTrackCameraProvider(video,soccertrack_calibration) if soccertrack_calibration else CameraProvider(video,calibration,camera_events)
    evaluator=ReferenceEvaluator(video,fps,reference)
    tracker=ShortTermTracker(config);world=WorldModel(video.stem,config)
    # Deterministic unsupervised kit warmup, on this clip only; not labeled evaluation.
    warmup_start=time.perf_counter();colors=[]
    for frame_number in np.unique(np.linspace(0,max(0,expected-1),min(expected,12),dtype=int)):
        cap.set(cv2.CAP_PROP_POS_FRAMES,int(frame_number));ok,frame=cap.read()
        if ok:
            for d in detector.detect(frame):
                color=TeamClassifier.color(frame,d)
                if color is not None:colors.append(color)
    teams=TeamClassifier.fit(colors,config);warmup=time.perf_counter()-warmup_start;cap.set(cv2.CAP_PROP_POS_FRAMES,0)
    writer=cv2.VideoWriter(str(output/'annotated.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),fps,(width+600,height))
    if not writer.isOpened():cap.release();raise ValueError('Annotated video writer unavailable')
    times=defaultdict(list);counts=[];visual=[];tracks_seen=defaultdict(list);decisions=Counter();previous={};index=0;sequence_examples=[]
    files=[(output/n).open('w') for n in ('world.jsonl','observations.jsonl','transitions.jsonl','associations.jsonl','reconciliations.jsonl')]
    try:
        while True:
            ok,frame=cap.read()
            if not ok:break
            now=index/fps;tick=time.perf_counter();detections=detector.detect(frame);times['detector_seconds'].append(time.perf_counter()-tick)
            tick=time.perf_counter();detections=[teams.classify(frame,d) for d in detections];times['team_seconds'].append(time.perf_counter()-tick)
            c=camera.state(index,now);tick=time.perf_counter();tracks=tracker.update(detections,now,(width,height),cut=c.continuity=='cut');times['tracking_seconds'].append(time.perf_counter()-tick)
            localized=[]
            for t in tracks:
                p=camera.project(t.foot,index)
                localized.append(t.model_copy(update={'pitch':p,'localization_confidence':c.calibration_confidence if p is not None else None,'localization_source':c.source if p is not None else 'unknown'}))
                tracks_seen[t.track_id].append(index)
            tick=time.perf_counter();state=world.update(localized,c);times['world_update_seconds'].append(time.perf_counter()-tick)
            evaluator.update(index,localized)
            observation=canonical_observation(state,fps,str(video))
            files[0].write(state.model_dump_json(exclude={'decisions','reconciliations'})+'\n');files[1].write(observation.model_dump_json()+'\n')
            for reconciliation in state.reconciliations:files[4].write(json.dumps(reconciliation,allow_nan=False)+'\n')
            for d in state.decisions:decisions[d['decision']]+=1;files[3].write(json.dumps(d,allow_nan=False)+'\n')
            for e in state.entities:
                signature=(e.visibility,e.track_id,e.position_source)
                if previous.get(e.entity_id)!=signature:
                    entry=dict(frame=index,timestamp=now,entity_id=e.entity_id,visibility=e.visibility,track_id=e.track_id,current_pitch=e.current_pitch,last_reliable_pitch=e.last_reliable_pitch,last_seen=e.last_seen,identity_status=e.identity_status,evidence=e.evidence[-2:])
                    files[2].write(json.dumps(entry,allow_nan=False)+'\n')
                    if len(sequence_examples)<25:sequence_examples.append(entry)
                    previous[e.entity_id]=signature
            counts.append(dict(frame=index,timestamp=now,detections=len(detections),visible=sum(e.visibility=='visible' for e in state.entities),
                off_camera=sum(e.visibility=='off_camera' for e in state.entities),temporarily_missing=sum(e.visibility=='temporarily_missing' for e in state.entities),
                unknown=sum(u.status=='unknown' for u in state.unassigned),ambiguous=sum(u.status=='ambiguous' for u in state.unassigned),
                localized=sum(e.current_pitch is not None for e in state.entities),calibration_valid=c.calibration_valid,
                team_known=sum(t.team is not None for t in localized)))
            visual.append(compact(state));writer.write(annotated_frame(frame,state));index+=1
            if index%100==0:print(f'Processed {index}/{expected} frames',flush=True)
    finally:
        cap.release();writer.release()
        for f in files:f.close()
    if index!=expected:raise ValueError(f'Video decoded {index}/{expected} frames; incomplete experiment')
    scores=[e.identity_confidence for e in world.entities.values() if e.identity_confidence is not None]
    durations=[(max(v)-min(v)+1)/fps for v in tracks_seen.values()]
    summary=dict(status='locally verified experimental person/world-model diagnostics; not football accuracy',
        input=dict(path=str(video),sha256=camera.video_hash,bytes=video.stat().st_size,frames=index,fps=fps,duration_seconds=index/fps,width=width,height=height),
        detector=dict(name='OpenCV Zoo YOLOX-S COCO person',model_sha256=MODEL_SHA256,model_url=MODEL_URL,license='Apache-2.0',confidence_threshold=confidence,backend='OpenCV DNN CPU',architecture=platform.machine()),
        configuration=config.model_dump(),detection=dict(total_person_observations=sum(r['detections'] for r in counts),mean_person_count=float(np.mean([r['detections'] for r in counts])),
            min_person_count=min(r['detections'] for r in counts),max_person_count=max(r['detections'] for r in counts),team_labeled_fraction=sum(r['team_known'] for r in counts)/max(1,sum(r['detections'] for r in counts))),
        tracking=dict(temporary_tracks=len(tracks_seen),mean_track_span_seconds=float(np.mean(durations)),median_track_span_seconds=float(np.median(durations)),max_track_span_seconds=max(durations),
            gap_segments=sum(sum(b>a+1 for a,b in zip(v,v[1:])) for v in tracks_seen.values()),identity_switch_accuracy=None,
            limitation='Track span includes tracker grace gaps. Fragmentation/identity accuracy require matched reference identities.'),
        localization=dict(calibration_valid_frames=sum(r['calibration_valid'] for r in counts),localized_entity_samples=sum(r['localized'] for r in counts),median_error_m=None,p95_error_m=None,
            status='supplied calibration only' if any(r['calibration_valid'] for r in counts) else 'UNAVAILABLE: no matching video calibration'),
        world=dict(anonymous_entities=len(world.entities),decisions=dict(decisions),maximum_off_camera=max(r['off_camera'] for r in counts),unknown_track_samples=sum(r['unknown'] for r in counts),ambiguous_track_samples=sum(r['ambiguous'] for r in counts),
            heuristic_identity_support={'count':len(scores),'min':min(scores) if scores else None,'median':float(np.median(scores)) if scores else None},
            caveat='Accepted reassociation is heuristic acceptance, not independently verified success. Anonymous entity births can duplicate a physical person across cuts or unresolved returns.'),
        reference=evaluator.report(),timing={k:dict(total=sum(v),mean_ms=1000*float(np.mean(v))) for k,v in times.items()},
        warmup_seconds=warmup,total_wall_seconds=time.perf_counter()-started,sequence_examples=sequence_examples,
        limitations=['Input-specific experiment; 22-second broadcast is not a 4-minute SoccerTrack evaluation.',
            'COCO person detector can include officials, staff and spectators; no verified player-role classification.',
            'Two kit clusters are heuristic; no names, jersey OCR, roster/role identification or defending-side inference.',
            'Torso histograms are weak appearance; support scores are not probabilities. Thresholds are provisional.',
            'OFF_CAMERA means unobserved after grace, not a verified physical exit; occlusion and missed detections can cause it.',
            'No automatic hard-cut/replay/zoom detector. Only supplied cut signals reset continuity; unknown cuts remain a failure risk.',
            'No supplied calibration means all metric positions/visible pitch region remain unknown. No borrowed homography.',
            'No current position from last-known memory; propagated/predicted positions are not implemented.',
            'Ball detection is not implemented. No canonical analytics bundle was registered or uploaded; downstream event inference is not run on unknown inputs.'])
    summary['timing']['total_effective_fps']=index/summary['total_wall_seconds']
    summary['timing']['detector_only_fps']=index/sum(times['detector_seconds'])
    (output/'diagnostics.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n');(output/'frame_diagnostics.json').write_text(json.dumps(counts,allow_nan=False))
    write_debug(output,video,visual,summary)
    print(json.dumps({k:summary[k] for k in ('input','detection','tracking','localization','world','timing')},indent=2),flush=True)
    return summary

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--video',required=True,type=Path);p.add_argument('--model',required=True,type=Path);p.add_argument('--output',required=True,type=Path)
    p.add_argument('--calibration',type=Path);p.add_argument('--soccertrack-calibration',type=Path);p.add_argument('--camera-events',type=Path);p.add_argument('--reference',type=Path);p.add_argument('--config',type=Path);p.add_argument('--confidence',type=float,default=.4)
    a=p.parse_args();config=Config.model_validate_json(a.config.read_text()) if a.config else Config()
    run(a.video,a.model,a.output,a.calibration,a.camera_events,a.reference,config,a.confidence,a.soccertrack_calibration)
if __name__=='__main__':main()
