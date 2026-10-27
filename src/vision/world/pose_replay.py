"""Replay cached semantic masks/pose hypotheses through bounded camera/world logic."""
import argparse,json,time
from pathlib import Path
from collections import Counter
import cv2,numpy as np
from .review import digest
from .schema import WorldState,TrackObservation
from .semantic_pitch import semantic_geometry
from .localization import CameraLocalizer
from .pitch import TemporalCamera
from .engine import WorldModel
from .provider import canonical_observation

def replay(video,previous,poses,output):
    video=Path(video).resolve();previous=Path(previous);poses=Path(poses);output=Path(output)
    if output.exists():raise ValueError('Use a new output directory')
    manifest=json.loads((poses/'manifest.json').read_text());fits=json.loads((poses/'pose_fits.json').read_text());sha=digest(video)
    if sha!=manifest['video_sha256'] or sha!=fits['video_sha256']:raise ValueError('Cached pose/video mismatch')
    frames=manifest['frames'];fps=manifest['fps'];width=manifest['width'];height=manifest['height'];data=np.load(poses/'semantic_masks.npz',allow_pickle=False)
    masks=data['mask'];confidences=data['confidence'];data.close()
    if len(masks)!=frames or len(confidences)!=frames or len(fits['frames'])!=frames:raise ValueError('Cached frame count mismatch')
    output.mkdir(parents=True);started=time.perf_counter();metrics=Counter();offsets=[];transitions=[];old_status=None;old_kind=None;cap=cv2.VideoCapture(str(video));camera=CameraLocalizer(width,height);world=WorldModel(manifest['recording_id']);examples=[]
    with (previous/'world.jsonl').open() as source,(output/'frames.jsonl').open('wb') as target,(output/'observations.jsonl').open('w') as observations,(output/'associations.jsonl').open('w') as associations:
        for index,line in enumerate(source):
            ok,image=cap.read()
            if not ok:raise ValueError('Incomplete source video')
            evidence,_=semantic_geometry(masks[index],confidences[index],width,height)
            pose=fits['frames'][index];c=camera.state(image,index,index/fps,evidence,pose['fit'],pose['diagnostic'])
            old=WorldState.model_validate_json(line);tracks=[TrackObservation(bbox=e.image_bbox,detection_confidence=e.detection_confidence,appearance=e.appearance,team=e.team,team_confidence=e.team_confidence,track_id=e.track_id,hits=2) for e in old.entities if e.visibility=='visible']+[u.observation for u in old.unassigned]
            tracks=[t.model_copy(update={'pitch':TemporalCamera.project(t.foot,c),'localization_source':c.source,'localization_confidence':None}) for t in tracks]
            state=world.update(tracks,c);offsets.append(target.tell());target.write((state.model_dump_json(exclude={'decisions','reconciliations'})+'\n').encode());observations.write(canonical_observation(state,fps,str(video)).model_dump_json()+'\n')
            for decision in state.decisions:associations.write(json.dumps(decision,allow_nan=False)+'\n');metrics['decision_'+decision['decision']]+=1
            metrics['frames_'+('direct' if c.calibration_valid and c.calibration_source_kind=='direct_calibrated' else 'propagated' if c.calibration_valid else 'unavailable')]+=1
            metrics['visible_region_frames']+=c.visible_pitch_region is not None;metrics['cut_frames']+=c.continuity=='cut'
            metrics['visible_entity_samples']+=sum(e.visibility=='visible' for e in state.entities);metrics['off_camera_entity_samples']+=sum(e.visibility=='off_camera' for e in state.entities)
            for e in state.entities:
                metrics['entity_positions_'+e.position_source]+=1
            metrics['localized_unknown_identity_samples']+=sum(u.observation.pitch is not None for u in state.unassigned)
            kind='direct' if c.calibration_valid and c.calibration_source_kind=='direct_calibrated' else 'propagated' if c.calibration_valid else 'unavailable'
            metrics['direct_fit_refreshes']+=kind=='direct' and old_kind!='direct';metrics['calibration_acquisitions']+=kind in ('direct','propagated') and old_kind not in ('direct','propagated');metrics['losses']+=kind=='unavailable' and old_kind in ('direct','propagated')
            event=c.motion_evidence.get('localization',{}).get('transition')
            if event:metrics['transition_'+event]+=1
            signature=(kind,c.epoch,event)
            if signature!=old_status:
                label='Hard cut' if c.continuity=='cut' else 'Calibration valid' if kind=='direct' else 'Camera propagated' if kind=='propagated' else 'Calibration lost' if old_kind in ('direct','propagated') else 'Calibration unavailable'
                transitions.append({'frame':index,'timestamp':index/fps,'label':label,'transition':event,'status':c.motion_evidence.get('localization',{}).get('status')});old_status=signature
            old_kind=kind
            if c.calibration_valid and len(examples)<12:examples.append({'frame':index,'timestamp':index/fps,'source':kind,'region':c.visible_pitch_region,'localized_entities':sum(e.current_pitch is not None for e in state.entities)})
    cap.release();data.close()
    if len(offsets)!=frames:raise ValueError('Incomplete world replay')
    result={k:v for k,v in manifest.items() if k not in ('frame_offsets','frames_sha256','transitions')};result.update(frame_offsets=offsets,frames_sha256=digest(output/'frames.jsonl'),transitions=transitions,
        source_pose_fits_sha256=digest(poses/'pose_fits.json'),source_masks_sha256=digest(poses/'semantic_masks.npz'),pitch_calibration_available=bool(metrics['frames_direct']))
    (output/'manifest.json').write_text(json.dumps(result));report={'classification':'MEASURED experimental diagnostics; HEURISTIC localization; independent accuracy UNAVAILABLE','frames':frames,'metrics':dict(metrics),'entities':len(world.entities),
        'replay_wall_seconds':time.perf_counter()-started,'inference_source':str(poses),'inference_timing':json.loads((poses/'diagnostics.json').read_text())['timing'],
        'max_camera_propagation_seconds':camera.max_age,'propagated_positions_never_update_last_reliable':True,'propagated_positions_never_authorize_reassociation':True,'examples':examples}
    (output/'diagnostics.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--video',required=True);p.add_argument('--previous',required=True);p.add_argument('--poses',required=True);p.add_argument('--output',required=True);a=p.parse_args();replay(a.video,a.previous,a.poses,a.output)
