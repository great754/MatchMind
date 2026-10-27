"""Offline full-frame semantic camera experiment. No cloud publication or serving inference."""
import argparse,json,time,gc
from pathlib import Path
from collections import Counter
import cv2,numpy as np
from .review import digest
from .semantic_pitch import SemanticPitch,semantic_geometry
from .pose import PoseFitter
from .pitch import PitchPerception,TemporalCamera
from .schema import WorldState,TrackObservation
from .engine import WorldModel
from .provider import canonical_observation
from src.observations.schema import PitchCalibrationObservation,Provenance

def run(video,previous,output,model,model_sha256,steps=2000,batch_size=12):
    video=Path(video).resolve();previous=Path(previous);output=Path(output)
    if output.exists():raise ValueError('Preserve existing experiments; use a new output')
    meta=json.loads((previous/'diagnostics.json').read_text());sha=digest(video)
    if sha!=meta['input']['sha256']:raise ValueError('Video hash mismatch')
    output.mkdir(parents=True);fps=meta['input']['fps'];width=meta['input']['width'];height=meta['input']['height'];frames=meta['input']['frames']
    cv2.setNumThreads(4);started=time.perf_counter();semantic=SemanticPitch(model,model_sha256);cap=cv2.VideoCapture(str(video));masks=[];confidences=[];groups=[];evidence=[];times=Counter()
    for index in range(frames):
        ok,image=cap.read()
        if not ok:raise ValueError('Incomplete video')
        tick=time.perf_counter();mask,confidence=semantic.masks(image);times['semantic_seconds']+=time.perf_counter()-tick
        masks.append(mask.astype(np.uint8));confidences.append(confidence)
        e,_=semantic_geometry(mask,confidence,width,height);evidence.append(e);groups.append(PoseFitter.groups(mask,confidence))
        if index%100==0:print('Semantic frames',index,flush=True)
    cap.release();del semantic;gc.collect()
    np.savez_compressed(output/'semantic_masks.npz',mask=np.stack(masks),confidence=np.stack(confidences))
    del masks,confidences
    eligible=[i for i,g in enumerate(groups) if PoseFitter.sufficient(g)];fits=[None]*frames;diagnostics=[{'accepted':False,'reason':'Insufficient distinct semantic ground/curve evidence','outcome':'INSUFFICIENT_SEMANTIC_CLASSES','solver_attempted':False} for _ in range(frames)]
    fitter=PoseFitter(width,height,steps);print('Pose-eligible frames',len(eligible),flush=True)
    for start in range(0,len(eligible),batch_size):
        indices=eligible[start:start+batch_size];tick=time.perf_counter()
        try:result=fitter.batch([groups[i] for i in indices])
        except (RuntimeError,ValueError,np.linalg.LinAlgError,cv2.error) as exc:
            result=[(None,{'accepted':False,'outcome':'POSE_OPTIMIZATION_FAILED','solver_attempted':True,'reason':'Camera optimizer raised an exception','exception_class':type(exc).__name__,'source_classes':list(groups[i])}) for i in indices]
        times['pose_seconds']+=time.perf_counter()-tick
        for index,(fit,diagnostic) in zip(indices,result):fits[index]=fit;diagnostics[index]=diagnostic
        print('Pose frames',min(start+batch_size,len(eligible)),'accepted',sum(f is not None for f in fits),flush=True)
    (output/'pose_fits.json').write_text(json.dumps({'video_sha256':sha,'model_sha256':model_sha256,'frames':[{'fit':fit,'diagnostic':d} for fit,d in zip(fits,diagnostics)]},allow_nan=False))
    world=WorldModel(video.stem);camera=TemporalCamera();cap=cv2.VideoCapture(str(video));metrics=Counter();transitions=[];offsets=[];old_valid=False;old_status=None
    with (previous/'world.jsonl').open() as source,(output/'frames.jsonl').open('wb') as target,(output/'observations.jsonl').open('w') as observations,(output/'associations.jsonl').open('w') as associations:
        for i,line in enumerate(source):
            ok,image=cap.read()
            if not ok:raise ValueError('Incomplete video replay')
            old=WorldState.model_validate_json(line);tick=time.perf_counter();c=camera.update(image,i,i/fps,evidence[i]);fit=fits[i]
            if fit:
                h=np.array(fit['homography']);calibration=PitchCalibrationObservation(start_frame=i,end_frame=i,landmarks=[l.start for l in evidence[i].lines],
                    image_to_pitch=tuple(h.flat),validity_region=fit['image_region'],provenance=Provenance(provider='cv-camera-pose-m2',source='TVCalib semantic line/curve optimization; heuristic held-out residual and local stability gates',quality='inferred'))
                camera.last_transform=tuple(h.flat);camera.last_timestamp=i/fps
                c=c.model_copy(update={'calibration':calibration,'calibration_valid':True,'calibration_status':'valid','visible_pitch_region':fit['pitch_region'],
                    'last_reliable_transform':tuple(h.flat),'last_reliable_timestamp':i/fps,'source':calibration.provenance.source})
            elif old_valid:c=c.model_copy(update={'calibration_status':'lost'})
            c=c.model_copy(update={'motion_evidence':{**c.motion_evidence,'pose':diagnostics[i]}});times['camera_seconds']+=time.perf_counter()-tick
            tracks=[TrackObservation(bbox=e.image_bbox,detection_confidence=e.detection_confidence,appearance=e.appearance,team=e.team,team_confidence=e.team_confidence,track_id=e.track_id,hits=2) for e in old.entities if e.visibility=='visible']+[u.observation for u in old.unassigned]
            tracks=[t.model_copy(update={'pitch':camera.project(t.foot,c),'localization_source':c.source,'localization_confidence':None}) for t in tracks]
            tick=time.perf_counter();state=world.update(tracks,c);times['world_seconds']+=time.perf_counter()-tick
            offsets.append(target.tell());target.write((state.model_dump_json(exclude={'decisions','reconciliations'})+'\n').encode());observations.write(canonical_observation(state,fps,str(video)).model_dump_json()+'\n')
            for decision in state.decisions:
                associations.write(json.dumps(decision,allow_nan=False)+'\n');metrics['decision_'+decision['decision']]+=1
            metrics[c.calibration_status]+=1;metrics['acquisitions']+=c.calibration_valid and not old_valid;metrics['losses']+=old_valid and not c.calibration_valid
            metrics['visible_region_frames']+=c.visible_pitch_region is not None;metrics['cut_frames']+=c.continuity=='cut'
            metrics['localized_entity_samples']+=sum(e.current_pitch is not None for e in state.entities)
            metrics['localized_unknown_identity_samples']+=sum(u.observation.pitch is not None for u in state.unassigned)
            metrics['visible_entity_samples']+=sum(e.visibility=='visible' for e in state.entities);metrics['off_camera_entity_samples']+=sum(e.visibility=='off_camera' for e in state.entities)
            signature=(c.calibration_status,c.epoch)
            if old_status!=signature:transitions.append({'frame':i,'timestamp':i/fps,'label':'Hard cut' if c.continuity=='cut' else 'Calibration '+c.calibration_status});old_status=signature
            old_valid=c.calibration_valid
    cap.release()
    manifest={'schema_version':1,'recording_id':video.stem,'source_video':str(video),'video_sha256':sha,'fps':fps,'frames':frames,'width':width,'height':height,
        'frame_offsets':offsets,'frames_sha256':digest(output/'frames.jsonl'),'transitions':transitions,'pitch_calibration_available':bool(metrics['valid']),
        'semantic_model_sha256':model_sha256,'provenance':'Experimental semantic line/curve camera pose; M1 cached person tracks; no football analytics/ball/name inference'}
    (output/'manifest.json').write_text(json.dumps(manifest))
    report={'classification':'MEASURED diagnostics; HEURISTIC camera localization; independent accuracy UNAVAILABLE','video_sha256':sha,'frames':frames,'pose_eligible_frames':len(eligible),
        'metrics':dict(metrics),'entities':len(world.entities),'timing':{k:{'seconds':v,'mean_ms_per_input_frame':v*1000/frames} for k,v in times.items()},
        'wall_seconds':time.perf_counter()-started,'source_person_observations':str(previous),'detector_timing':'Reused M1; not newly benchmarked',
        'pose_assumptions':'Main sideline camera, centered principal point, no lens distortion. Gates are provisional; residual is dependent validation.'}
    (output/'diagnostics.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--video',required=True);p.add_argument('--previous',required=True);p.add_argument('--output',required=True);p.add_argument('--model',required=True);p.add_argument('--model-sha256',required=True);p.add_argument('--steps',type=int,default=2000);p.add_argument('--batch-size',type=int,default=12);a=p.parse_args();run(a.video,a.previous,a.output,a.model,a.model_sha256,a.steps,a.batch_size)
