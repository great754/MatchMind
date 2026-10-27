"""M5 camera experiment over immutable M4 people/team/identity snapshots."""
import argparse,json,time
from pathlib import Path
from collections import Counter
import cv2,numpy as np
from .schema import WorldState
from .field_motion import field_image,FieldMotionTracker
from .pose import PoseFitter
from .persistent_camera import PersistentCamera
from .texture_anchors import TextureAnchors
from .player_geometry import project_player
from .provider import canonical_observation
from .review import digest
from src.observations.schema import CameraProjectionEvidence

def run(video,baseline,poses,output,camera_only=False):
    started=time.perf_counter();video=Path(video).resolve();baseline=Path(baseline);poses=Path(poses);output=Path(output)
    if output.exists():raise ValueError('Preserve experiments; choose a new output directory')
    manifest=json.loads((baseline/'manifest.json').read_text());fits=json.loads((poses/'pose_fits.json').read_text());sha=digest(video)
    if sha!=manifest['video_sha256'] or sha!=fits['video_sha256'] or digest(baseline/'frames.jsonl')!=manifest['frames_sha256']:raise ValueError('Source/baseline/cache hash mismatch')
    cache=np.load(poses/'semantic_masks.npz',allow_pickle=False);masks=cache['mask'];confidence=cache['confidence'];cache.close();frames=manifest['frames'];fps=manifest['fps'];width=manifest['width'];height=manifest['height']
    if len(masks)!=frames or len(fits['frames'])!=frames:raise ValueError('Incomplete cached evidence')
    output.mkdir(parents=True);cap=cv2.VideoCapture(str(video));camera=PersistentCamera(width,height);tracker=FieldMotionTracker();anchors=TextureAnchors();metrics=Counter();timeline=[];transitions=[];offsets=[];last_status=None;history={};times=Counter()
    target=(output/('camera.jsonl' if camera_only else 'frames.jsonl')).open('wb');observations=None if camera_only else (output/'observations.jsonl').open('w')
    try:
        for index,line in enumerate((baseline/'frames.jsonl').open()):
            old=WorldState.model_validate_json(line);ok,image=cap.read()
            if not ok or old.frame!=index or abs(old.timestamp-index/fps)>1e-6:raise ValueError('Misaligned source')
            if old.camera.continuity=='cut':tracker=FieldMotionTracker();anchors=TextureAnchors()
            boxes=[e.image_bbox for e in old.entities if e.visibility=='visible']+[u.observation.bbox for u in old.unassigned];tick=time.perf_counter();field=field_image(image,boxes);motion=tracker.update(field);times['field_motion_seconds']+=time.perf_counter()-tick
            groups=PoseFitter.groups(masks[index],confidence[index]);p=fits['frames'][index];tick=time.perf_counter();anchor,anchor_diagnostic=anchors.propose(index,old.camera.epoch,field,tracker.measurements,motion,width,height)
            c=camera.update(old.camera,groups,p['fit'],p['diagnostic'],motion,field,anchor,anchor_diagnostic);
            if c.calibration_valid:anchors.add(index,c.epoch,c.memory.last_accepted,field,tracker.measurements,c.calibration_source_kind=='direct_calibrated')
            times['camera_filter_seconds']+=time.perf_counter()-tick
            info=c.motion_evidence['localization'];kind=info['status'];metrics['camera_'+kind]+=1
            if c.calibration_valid:metrics['camera_accepted']+=1
            metrics['cuts']+=c.continuity=='cut';meta=CameraProjectionEvidence(epoch=c.epoch,state=kind,source=c.calibration_source_kind if c.calibration_valid else 'unknown',support=c.memory.last_accepted.support if c.calibration_valid else None,last_direct_age_seconds=info['last_absolute_age_seconds'],current_semantic_support=info['current_semantic_support'],relative_motion_error_budget_px=info['relative_motion_error_budget_px'])
            entities=[]
            for e in old.entities:
                point,reason=project_player(e.image_foot,c) if e.visibility=='visible' else (None,'NOT_CURRENTLY_VISIBLE')
                if point is not None:
                    metrics['players_projected']+=1;metrics['players_'+kind]+=1
                    if c.calibration_source_kind=='direct_calibrated':history[e.entity_id]=(point,c.timestamp,c.epoch)
                if e.visibility=='visible':metrics['players_eligible']+=1;metrics['projection_'+reason]+=1
                reliable=history.get(e.entity_id)
                entities.append(e.model_copy(update={'current_pitch':point,'position_source':c.calibration_source_kind if point is not None else 'unknown','localization_confidence':None,'projection_reason':reason,'projection_camera':meta.model_copy(update={'image_foot':e.image_foot}) if e.visibility=='visible' else None,'last_reliable_pitch':reliable[0] if reliable else None,'last_localized_at':reliable[1] if reliable else None,'last_reliable_camera_epoch':reliable[2] if reliable else None,'last_velocity':None}))
            unassigned=[]
            for u in old.unassigned:
                point,reason=project_player(u.observation.foot,c);t=u.observation.model_copy(update={'pitch':point,'localization_source':c.source,'localization_confidence':None,'projection_reason':reason,'projection_camera':meta.model_copy(update={'image_foot':u.observation.foot})});unassigned.append(u.model_copy(update={'observation':t}));metrics['players_eligible']+=1;metrics['projection_'+reason]+=1
                if point is not None:metrics['players_projected']+=1;metrics['players_'+kind]+=1
            state=old.model_copy(update={'camera':c,'entities':entities,'unassigned':unassigned});offsets.append(target.tell());target.write(((c if camera_only else state).model_dump_json()+'\n').encode())
            if observations:observations.write(canonical_observation(state,fps,str(video)).model_dump_json()+'\n')
            entry={'frame':index,'timestamp':index/fps,'state':kind,'cut':c.continuity=='cut','reason':info['outcome'],'last_direct_age_seconds':info['last_absolute_age_seconds'],'support':meta.support,'landmarks':info['semantic_classes']};timeline.append(entry)
            if kind!=last_status:
                label='Hard cut' if c.continuity=='cut' else 'Calibration valid' if kind in ('DIRECT','RELOCALIZED') else 'Camera propagated' if kind=='TEMPORAL_TRACKED' else 'Camera constrained' if kind=='PARTIAL_CONSTRAINED' else 'Calibration lost' if last_status not in (None,'UNKNOWN') else 'Calibration unavailable'
                transitions.append({'frame':index,'timestamp':index/fps,'label':label,'transition':info['transition'],'status':kind});last_status=kind
            if index%100==0:print('Camera frames',index,'accepted',metrics['camera_accepted'],'projected',metrics['players_projected'],flush=True)
    finally:
        target.close();cap.release()
        if observations:observations.close()
    if len(offsets)!=frames:raise ValueError('Incomplete camera replay')
    report={'classification':'MEASURED experimental camera/placement diagnostics; independent accuracy unavailable','frames':frames,'metrics':dict(metrics),'camera_coverage_percent':100*metrics['camera_accepted']/frames,'player_placement_percent':100*metrics['players_projected']/metrics['players_eligible'],'timing':dict(times),'wall_seconds':time.perf_counter()-started,'m4_player_team_identity_frozen':True,'source_m4_frames_sha256':manifest['frames_sha256'],'source_video_sha256':sha,'camera_only_pilot':camera_only}
    (output/'diagnostics.json').write_text(json.dumps(report,indent=2));(output/'timeline.json').write_text(json.dumps(timeline))
    if not camera_only:
        m={k:v for k,v in manifest.items() if k not in ('frame_offsets','frames_sha256','transitions')};m.update(frame_offsets=offsets,frames_sha256=digest(output/'frames.jsonl'),transitions=transitions,camera_timeline=timeline,source_m4_frames_sha256=manifest['frames_sha256'],camera_model_version='m5',provenance='M5 motion/partial camera constraints over frozen M4 people; no analytics or cloud publication');(output/'manifest.json').write_text(json.dumps(m))
    print(json.dumps(report,indent=2),flush=True);return report
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--video',required=True);p.add_argument('--baseline',required=True);p.add_argument('--poses',required=True);p.add_argument('--output',required=True);p.add_argument('--camera-only',action='store_true');a=p.parse_args();run(a.video,a.baseline,a.poses,a.output,a.camera_only)
