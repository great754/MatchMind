"""Offline M2 review generation, reusing immutable M1 person detections/tracks.

No detector during web requests. Does not publish or create analytics bundles.
"""
import argparse,hashlib,json,time
from pathlib import Path
from collections import Counter
import cv2
from .schema import WorldState,TrackObservation
from .pitch import PitchPerception,TemporalCamera
from .engine import WorldModel
from .provider import canonical_observation

def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()

def generate(video,previous,output,semantic_model=None,model_sha256=None,device='cpu'):
    video=Path(video).resolve();previous=Path(previous);output=Path(output)
    if output.exists():raise ValueError('Use a new output directory')
    meta=json.loads((previous/'diagnostics.json').read_text())
    if digest(video)!=meta['input']['sha256']:raise ValueError('Video does not match original observations')
    output.mkdir(parents=True);cap=cv2.VideoCapture(str(video));fps=meta['input']['fps']
    perception=PitchPerception();camera=TemporalCamera();world=WorldModel(video.stem)
    semantic=None
    if semantic_model:
        from .semantic_pitch import SemanticPitch
        semantic=SemanticPitch(semantic_model,model_sha256,device)
    offsets=[];metrics=Counter();timings=Counter();transitions=[];old_status=None;previous_valid=False;started=time.perf_counter()
    with (previous/'world.jsonl').open() as source,(output/'frames.jsonl').open('wb') as target:
        for index,line in enumerate(source):
            ok,image=cap.read()
            if not ok:raise ValueError('Incomplete source video')
            old=WorldState.model_validate_json(line)
            tick=time.perf_counter();points=None
            if semantic:evidence,points=semantic.detect(image)
            else:evidence=perception.detect(image)
            timings['pitch_seconds']+=time.perf_counter()-tick
            tick=time.perf_counter();c=camera.update(image,index,index/fps,evidence,points,'TVCalib semantic line hypotheses' if semantic else 'automatic anonymous line evidence');timings['camera_seconds']+=time.perf_counter()-tick
            tracks=[]
            for e in old.entities:
                if e.visibility=='visible':tracks.append(TrackObservation(bbox=e.image_bbox,detection_confidence=e.detection_confidence,
                    appearance=e.appearance,team=e.team,team_confidence=e.team_confidence,track_id=e.track_id,hits=2))
            tracks.extend(u.observation for u in old.unassigned)
            tracks=[t.model_copy(update={'pitch':camera.project(t.foot,c),'localization_source':c.source}) for t in tracks]
            tick=time.perf_counter();state=world.update(tracks,c);timings['world_seconds']+=time.perf_counter()-tick
            canonical_observation(state,fps,str(video))  # Validate canonical boundary, including region.
            row=state.model_dump(exclude={'decisions','reconciliations'})
            offsets.append(target.tell());target.write((json.dumps(row,allow_nan=False,separators=(',',':'))+'\n').encode())
            metrics[c.calibration_status]+=1;metrics['lines']+=len(evidence.lines);metrics['cuts']+=c.continuity=='cut'
            metrics['calibration_acquisitions']+=c.calibration_valid and not previous_valid
            metrics['calibration_losses']+=previous_valid and not c.calibration_valid
            metrics['visible_region_frames']+=c.visible_pitch_region is not None
            previous_valid=c.calibration_valid
            metrics['direct_positions']+=sum(e.current_pitch is not None for e in state.entities)
            metrics['ambiguous_track_frames']+=sum(u.status=='ambiguous' for u in state.unassigned)
            metrics['off_camera_entity_frames']+=sum(e.visibility=='off_camera' for e in state.entities)
            metrics['reassociations']+=sum(d['decision']=='reassociated' for d in state.decisions)
            status=(c.calibration_status,c.epoch)
            if status!=old_status:
                transitions.append({'frame':index,'timestamp':index/fps,'label':'Hard cut' if c.continuity=='cut' else 'Calibration '+c.calibration_status});old_status=status
    cap.release()
    manifest={'schema_version':1,'recording_id':video.stem,'source_video':str(video),'video_sha256':digest(video),
        'fps':fps,'frames':len(offsets),'width':meta['input']['width'],'height':meta['input']['height'],
        'frame_offsets':offsets,'frames_sha256':digest(output/'frames.jsonl'),'transitions':transitions,
        'pitch_calibration_available':bool(metrics['valid']),
        'provenance':'M1 precomputed person detections/temporary tracks; M2 semantic line camera fit' if semantic else 'M1 precomputed person detections/temporary tracks; M2 anonymous lines; no semantic anchor',
        'semantic_model_sha256':model_sha256 if semantic else None}
    (output/'manifest.json').write_text(json.dumps(manifest))
    report={'classification':'MEASURED diagnostics, HEURISTIC perception; reference accuracy UNAVAILABLE',
        'metrics':dict(metrics),'frames':len(offsets),'timing':{k: {'total_seconds':v,'mean_ms':1000*v/len(offsets)} for k,v in timings.items()},
        'total_seconds':time.perf_counter()-started,'person_detector':'Reused M1 observations; no new inference timing',
        'automatic_metric_localization':'EXPERIMENTAL: semantic line fits; self residual is dependent validation' if metrics['valid'] else 'UNAVAILABLE: insufficient consistent semantic correspondences',
        'original_detection_run':str(previous),'source_preserved':True}
    (output/'diagnostics.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--video',required=True);p.add_argument('--previous',required=True);p.add_argument('--output',required=True)
    p.add_argument('--semantic-model');p.add_argument('--model-sha256');p.add_argument('--device',default='cpu',choices=['cpu','mps'])
    args=p.parse_args();generate(args.video,args.previous,args.output,args.semantic_model,args.model_sha256,args.device)
