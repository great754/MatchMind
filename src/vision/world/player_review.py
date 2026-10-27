"""Offline player/team upgrade over frozen M3 camera states and audited detector boxes."""
import argparse,json,time
from pathlib import Path
from collections import defaultdict,Counter
import cv2,numpy as np
from .schema import Detection,WorldState,Config
from .tracking import ShortTermTracker
from .kits import torso_features,VideoTeams
from .engine import WorldModel
from .player_geometry import project_player,assess_sizes
from .provider import canonical_observation
from .review import digest

def run(video,baseline,detections,output):
    started=time.perf_counter();video=Path(video).resolve();baseline=Path(baseline);output=Path(output)
    if output.exists():raise ValueError('Use a new output directory; preserve previous evidence')
    manifest=json.loads((baseline/'manifest.json').read_text());cache=json.loads(Path(detections).read_text())
    if digest(video)!=manifest['video_sha256'] or cache['source_sha256']!=manifest['video_sha256']:raise ValueError('Source/cache hash mismatch')
    if digest(baseline/'frames.jsonl')!=manifest['frames_sha256']:raise ValueError('Baseline camera snapshot hash mismatch')
    fps=manifest['fps'];width=manifest['width'];height=manifest['height'];cameras=[]
    for i,line in enumerate((baseline/'frames.jsonl').open()):
        state=WorldState.model_validate_json(line)
        if state.frame!=i or state.recording_id!=manifest['recording_id']:raise ValueError('Camera clock/recording mismatch')
        cameras.append(state.camera)
    if len(cameras)!=manifest['frames'] or len(cache['frames'])!=len(cameras):raise ValueError('Incomplete inputs')
    cap=cv2.VideoCapture(str(video));frames=[];profiles=defaultdict(list);warmup=ShortTermTracker();extract_started=time.perf_counter()
    for index,row in enumerate(cache['frames']):
        ok,image=cap.read()
        if not ok or row['frame']!=index:raise ValueError('Incomplete or misaligned video/detector input')
        found=[]
        for raw in row['detections']:
            d=Detection.model_validate(raw);kit,appearance=torso_features(image,d.bbox)
            found.append(d.model_copy(update={'team':None,'team_confidence':None,'kit':kit,'appearance':appearance}))
        frames.append(found)
        for t in warmup.update(found,index/fps,(width,height),cut=cameras[index].continuity=='cut'):
            if t.kit and t.kit.lab and t.kit.quality>=.35 and t.kit.field_context>=.3 and t.detection_confidence>=.5 and index%3==0:profiles[t.track_id].append(t.kit.lab)
    cap.release();teams=VideoTeams.fit(profiles);feature_seconds=time.perf_counter()-extract_started
    output.mkdir(parents=True);world=WorldModel(manifest['recording_id']);tracker=ShortTermTracker();offsets=[];metrics=Counter();track_frames=defaultdict(list);entity_first={};changes=[];previous={};replay_started=time.perf_counter()
    with (output/'frames.jsonl').open('wb') as target,(output/'observations.jsonl').open('w') as observations,(output/'associations.jsonl').open('w') as associations,(output/'frame_diagnostics.jsonl').open('w') as diagnostic:
        for index,found in enumerate(frames):
            c=cameras[index];now=index/fps;found=assess_sizes([teams.classify(d) for d in found],c,width,height)
            tracks=tracker.update(found,now,(width,height),cut=c.continuity=='cut');localized=[]
            for t in tracks:
                p,reason=project_player(t.foot,c);localized.append(t.model_copy(update={'pitch':p,'projection_reason':reason,'localization_source':c.source if p is not None else 'unknown'}));track_frames[t.track_id].append(index)
                metrics['projection_'+reason]+=1;metrics['observation_'+str(t.team)]+=1
                for flag in t.size_evidence.flags:metrics['geometry_'+flag]+=1
            state=world.update(localized,c);offsets.append(target.tell());target.write((state.model_dump_json(exclude={'decisions','reconciliations'})+'\n').encode());observations.write(canonical_observation(state,fps,str(video)).model_dump_json()+'\n')
            for d in state.decisions:associations.write(json.dumps({'frame':index,**d},allow_nan=False)+'\n');metrics['decision_'+d['decision']]+=1
            for e in state.entities:
                entity_first.setdefault(e.entity_id,index)
                if e.visibility=='visible':metrics['visible_team_'+str(e.team)]+=1;metrics['visible_role_'+e.role]+=1
                for event in e.history:
                    if event.frame==index:
                        metrics['event_'+event.kind]+=1
                        if event.kind in ('team_changed','team_acquired'):changes.append({'entity_id':e.entity_id,**event.model_dump()})
                if e.memory_state!=previous.get(e.entity_id):metrics['memory_entry_'+e.memory_state]+=1
                previous[e.entity_id]=e.memory_state
            diagnostic.write(json.dumps({'frame':index,'detected':len(found),'visible_entities':sum(e.visibility=='visible' for e in state.entities),'ambiguous_tracks':sum(u.status=='ambiguous' for u in state.unassigned),'projection_reasons':dict(Counter(t.projection_reason for t in localized))},allow_nan=False)+'\n')
    durations=[(v[-1]-v[0]+1)/fps for v in track_frames.values()]
    result={k:v for k,v in manifest.items() if k not in ('frame_offsets','frames_sha256')};result.update(frame_offsets=offsets,frames_sha256=digest(output/'frames.jsonl'),player_model_version='m4',team_model=teams.diagnostic,source_camera_frames_sha256=manifest['frames_sha256'],source_detector_cache_sha256=digest(detections),provenance='Experimental persistent player/team memory; frozen M3 camera; no new football analytics')
    (output/'manifest.json').write_text(json.dumps(result,allow_nan=False))
    report={'classification':'MEASURED diagnostics / EXPERIMENTAL hypotheses; independent identity accuracy unavailable','frames':len(frames),'person_observations':sum(map(len,frames)),'metrics':dict(metrics),'entities':len(world.entities),'entities_with_multiple_track_segments':sum(len(e.track_history)>1 for e in world.entities.values()),'temporary_tracks':len(track_frames),'mean_track_span_seconds':float(np.mean(durations)),'short_tracks_under_0_5_seconds':sum(t<.5 for t in durations),'mean_entity_observed_lifetime_seconds':float(np.mean([e.last_seen-(e.first_seen or 0)+1/fps for e in world.entities.values()])),'team_model':teams.diagnostic,'team_changes':changes,'timing':{'features_warmup_seconds':feature_seconds,'world_replay_seconds':time.perf_counter()-replay_started,'wall_seconds':time.perf_counter()-started},'independent_team_accuracy':None,'independent_identity_accuracy':None,'camera_changed':False,'nms_threshold':.5}
    (output/'diagnostics.json').write_text(json.dumps(report,indent=2,allow_nan=False));print(json.dumps({k:v for k,v in report.items() if k!='team_changes'},indent=2),flush=True)
    return report
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--video',required=True);p.add_argument('--baseline',required=True);p.add_argument('--detections',required=True);p.add_argument('--output',required=True);a=p.parse_args();run(a.video,a.baseline,a.detections,a.output)
