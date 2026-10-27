"""M6 offline stability replay over immutable M5 snapshots and cached semantics."""
import argparse,json,time
from pathlib import Path
from collections import Counter
import cv2,numpy as np
from .schema import WorldState,TeamVotes,EntityEvent
from .review import digest
from .field_motion import field_image,FieldMotionTracker
from .static_geometry import measure,with_lines
from .pose import PoseFitter
from .temporal_camera import TemporalCameraFilter
from .team_belief import TeamBelief,kit_margin
from .player_geometry import project_player
from .provider import canonical_observation
from src.observations.schema import CameraProjectionEvidence,TeamBeliefEvidence


def position_diagnostic(previous,entity,point,camera,fps):
    if previous is None or point is None or previous.current_pitch is None or previous.image_foot is None or entity.image_foot is None:return None
    d=float(np.linalg.norm(np.array(point)-previous.current_pitch));h=np.array(camera.calibration.image_to_pitch).reshape(3,3);samefoot=cv2.perspectiveTransform(np.array([[previous.image_foot]],float),h)[0,0];cam=float(np.linalg.norm(samefoot-previous.current_pitch));foot=float(np.linalg.norm(samefoot-point));flag=d>12/fps+.5
    return {'distance_m':d,'speed_mps':d*fps,'large_jump':flag,'screening_threshold_m':12/fps+.5,'camera_component_m':cam,'changed_foot_component_m':foot,'image_foot_delta_px':float(np.linalg.norm(np.array(entity.image_foot)-previous.image_foot)),'identity_track_changed':entity.track_id!=previous.track_id,'interpretation':'identity track changed' if entity.track_id!=previous.track_id else 'camera change dominates' if cam>foot*1.5 else 'image foot movement/detection dominates' if foot>cam*1.5 else 'mixed camera and image-foot change','accuracy':'diagnostic decomposition; physical movement and detection error not independently distinguished'}


def run(video,baseline,poses,output):
    started=time.perf_counter();baseline=Path(baseline);poses=Path(poses);output=Path(output);video=Path(video).resolve()
    if output.exists():raise ValueError('Preserve baselines; choose a fresh output directory')
    m=json.loads((baseline/'manifest.json').read_text());sha=digest(video)
    if sha!=m['video_sha256'] or digest(baseline/'frames.jsonl')!=m['frames_sha256']:raise ValueError('Baseline/source hash mismatch')
    rows=[WorldState.model_validate_json(l) for l in (baseline/'frames.jsonl').open()];fps=m['fps'];width=m['width'];height=m['height'];n=len(rows)
    if n!=m['frames'] or any(r.frame!=i or r.recording_id!=m['recording_id'] for i,r in enumerate(rows)):raise ValueError('Incomplete/misaligned baseline')
    output.mkdir(parents=True);times={};cv2.setNumThreads(4);tick=time.perf_counter();cache=np.load(poses/'semantic_masks.npz',allow_pickle=False);masks=cache['mask'];confidence=cache['confidence'];groups=[PoseFitter.groups(masks[i],confidence[i]) for i in range(n)];del masks,confidence;cache.close()
    fields=[];cap=cv2.VideoCapture(str(video))
    for row in rows:
        ok,image=cap.read()
        if not ok:raise ValueError('Incomplete source video')
        fields.append(field_image(image,[e.image_bbox for e in row.entities if e.visibility=='visible']+[u.observation.bbox for u in row.unassigned]))
    cap.release();times['preprocessing_seconds']=time.perf_counter()-tick;forward=[{'accepted':False,'reason':'FIRST_FRAME'} for _ in rows];backward=[{'accepted':False,'reason':'LAST_FRAME'} for _ in rows];tick=time.perf_counter();feature_measurements=[None]*n
    for direction,indices,target in (('forward',range(n),forward),('backward',range(n-1,-1,-1),backward)):
        tracker=FieldMotionTracker(debug=True);previous_index=None
        for i in indices:
            cut=previous_index is not None and (rows[i].camera.epoch!=rows[previous_index].camera.epoch or rows[max(i,previous_index)].camera.continuity=='cut')
            if cut:tracker=FieldMotionTracker(debug=True)
            measured=tracker.update(fields[i])
            if direction=='forward' and tracker.measurements is not None:feature_measurements[i]={'ids':tracker.measurements['ids'].copy(),'points':tracker.measurements['points'].copy()}
            if previous_index is not None and not cut:
                measured=with_lines(measured,groups[previous_index],groups[i],width,height)
                if not measured.get('accepted'):
                    alternative=measure(fields[previous_index],fields[i],groups[previous_index],groups[i],width,height)
                    if alternative.get('accepted'):measured={**alternative,'route':'fresh field points with named-line validation'}
            target[i]=measured if not cut else {'accepted':False,'reason':'HARD_CUT'};previous_index=i
            if i%100==0:print('Measured static field',direction,i,'/',n,flush=True)
    times['static_motion_seconds']=time.perf_counter()-tick;tick=time.perf_counter();cameras,filter_diagnostic=TemporalCameraFilter(width,height,fps).run([r.camera for r in rows],groups,forward,backward,fields,feature_measurements);del fields;times['camera_filter_seconds']=time.perf_counter()-tick
    metrics=Counter();beliefs={};events=[];history={};previous={};timeline=[];transitions=[];offsets=[];last_status=None;team_time=0.;pure_team_time=0.;jump_rows=[];displacements=[]
    with (output/'frames.jsonl').open('wb') as target,(output/'observations.jsonl').open('w') as obs:
        for i,(old,c) in enumerate(zip(rows,cameras)):
            loc=c.motion_evidence['localization'];debug=c.motion_evidence['temporal_camera'];state=loc['status'];metrics['camera_'+state]+=1;metrics['cuts']+=c.continuity=='cut'
            if c.calibration_valid:metrics['camera_accepted']+=1
            meta=CameraProjectionEvidence(epoch=c.epoch,state=state,source=c.calibration_source_kind if c.calibration_valid else 'unknown',support=loc.get('support_score'),last_direct_age_seconds=loc['last_absolute_age_seconds'],current_semantic_support=loc['current_semantic_support'],relative_motion_error_budget_px=debug.get('relative_motion_error_budget_px'),absolute_anchor_timestamp=debug.get('absolute_anchor_timestamp'),anchor_age_seconds=debug.get('anchor_age_seconds'),temporal_support=state not in ('DIRECT','RELOCALIZED','UNKNOWN'),temporal_direction=debug['direction'],forward_backward_agreement_p90_m=debug.get('forward_backward_agreement_p90_m'))
            entities=[];tick=time.perf_counter()
            for e in old.entities:
                visible=e.visibility=='visible'
                key=('entity',e.entity_id)
                if key not in beliefs:
                    prior=beliefs.pop(('track',e.track_id,c.epoch),None) if visible else None
                    beliefs[key]=prior or TeamBelief()
                    if prior is not None:metrics['team_track_history_transfers']+=1
                belief=beliefs[key];kit=e.current_kit;raw_observed=e.team_memory.current_observation
                if visible:
                    team_tick=time.perf_counter();team,event=belief.update(raw_observed,kit_margin(kit),kit,i,old.timestamp);pure_team_time+=time.perf_counter()-team_tick
                    if event:events.append({'entity_id':e.entity_id,**event})
                    metrics['team_'+str(team['current_team'])]+=1;metrics['team_source_'+team['source']]+=1
                else:
                    team={'current_team':belief.team,'support':(belief.a if belief.team=='team_a' else belief.b)/max(belief.a+belief.b,1e-9) if belief.team else None,'source':'temporal_memory' if belief.team else 'unknown','current_observation':None,'current_quality':None,'current_strong':False,'independent_sample':False,'unknown_reason':None if belief.team else 'TEAM_UNKNOWN_NO_EVIDENCE','evidence_a':belief.a,'evidence_b':belief.b,'independent_samples':belief.samples,'uncertain_observations':belief.unknown,'candidate_team':belief.candidate,'candidate_streak':belief.streak,'last_strong_observation':belief.last_strong,'identity_resolved':True,'history':list(belief.history),'latest_switch':None}
                point,reason=project_player(e.image_foot,c) if visible else (None,'NOT_CURRENTLY_VISIBLE')
                if visible:metrics['players_eligible']+=1;metrics['projection_'+reason]+=1
                if point is not None:
                    metrics['players_projected']+=1;metrics['players_'+state]+=1
                    if c.calibration_source_kind=='direct_calibrated':history[e.entity_id]=(point,c.timestamp,c.epoch)
                reliable=history.get(e.entity_id);pd=position_diagnostic(previous.get(e.entity_id),e,point,c,fps) if visible else None
                if pd:
                    displacements.append(pd['distance_m'])
                    if pd['large_jump']:jump_rows.append({'frame':i,'entity':e.entity_id,**pd})
                changes=[EntityEvent(frame=v['frame'],timestamp=v['timestamp'],kind=v['kind'],previous=v['old'],current=v['new'],reason=v['reason']) for v in belief.switches]
                memory=e.team_memory.model_copy(update={'votes':TeamVotes(team_a=belief.a,team_b=belief.b,unknown=belief.unknown),'current_observation':raw_observed if visible else None,'stable':belief.team is not None,'evidence_count':belief.samples,'contradictory_streak':belief.streak})
                entities.append(e.model_copy(update={'team':team['current_team'],'team_confidence':team['support'],'team_belief':TeamBeliefEvidence.model_validate(team),'team_memory':memory,'history':([h for h in e.history if not h.kind.startswith('team_')]+changes)[-64:],'current_pitch':point,'position_source':c.calibration_source_kind if point is not None else 'unknown','localization_confidence':None,'projection_reason':reason,'projection_camera':meta.model_copy(update={'image_foot':e.image_foot}) if visible else None,'position_diagnostic':pd,'last_reliable_pitch':reliable[0] if reliable else None,'last_localized_at':reliable[1] if reliable else None,'last_reliable_camera_epoch':reliable[2] if reliable else None,'last_velocity':None}))
            unassigned=[]
            for u in old.unassigned:
                t=u.observation;belief=beliefs.setdefault(('track',t.track_id,c.epoch),TeamBelief());team_tick=time.perf_counter();team,event=belief.update(t.team,t.team_confidence,t.kit,i,old.timestamp,False);pure_team_time+=time.perf_counter()-team_tick;point,reason=project_player(t.foot,c);metrics['players_eligible']+=1;metrics['projection_'+reason]+=1;metrics['team_'+str(team['current_team'])]+=1;metrics['team_source_'+team['source']]+=1
                if point is not None:metrics['players_projected']+=1;metrics['players_'+state]+=1
                unassigned.append(u.model_copy(update={'observation':t.model_copy(update={'team':team['current_team'],'team_confidence':team['support'],'team_belief':TeamBeliefEvidence.model_validate(team),'pitch':point,'projection_reason':reason,'projection_camera':meta.model_copy(update={'image_foot':t.foot}),'localization_confidence':None,'localization_source':c.source})}))
            team_time+=time.perf_counter()-tick;previous={e.entity_id:e for e in entities if e.visibility=='visible'}
            roster=old.roster.model_copy(update={'team_a':[e.entity_id for e in entities if e.team=='team_a'],'team_b':[e.entity_id for e in entities if e.team=='team_b'],'unknown':[e.entity_id for e in entities if e.team is None]});current=old.model_copy(update={'camera':c,'entities':entities,'unassigned':unassigned,'roster':roster});offsets.append(target.tell());target.write((current.model_dump_json()+'\n').encode());obs.write(canonical_observation(current,fps,str(video)).model_dump_json()+'\n')
            timeline.append({'frame':i,'timestamp':i/fps,'state':state,'cut':c.continuity=='cut','reason':loc['outcome'],'last_direct_age_seconds':loc['last_absolute_age_seconds'],'support':loc.get('support_score'),'landmarks':loc['semantic_classes']})
            if state!=last_status:
                transitions.append({'frame':i,'timestamp':i/fps,'label':'Calibration valid' if state in ('DIRECT','RELOCALIZED') else 'Camera propagated' if state in ('TEMPORAL_PREDICTED','TEMPORAL_CORRECTED') else 'Camera constrained' if state in ('PARTIAL_CONSTRAINED','BIDIRECTIONAL_INTERPOLATED') else 'Calibration lost','transition':loc['transition'],'status':state});last_status=state
    times['team_filter_seconds']=pure_team_time;times['team_projection_replay_seconds']=team_time;report={'classification':'EXPERIMENTAL temporal consistency; independent metric accuracy unavailable','frames':n,'metrics':dict(metrics),'camera_coverage_percent':100*metrics['camera_accepted']/n,'player_placement_percent':100*metrics['players_projected']/metrics['players_eligible'],'temporal_camera':filter_diagnostic,'team_events':events,'team_memory_usage':metrics['team_source_temporal_memory'],'position_large_jumps':jump_rows,'position_displacements':{'count':len(displacements),'median_m':float(np.median(displacements)) if displacements else None,'p90_m':float(np.quantile(displacements,.9)) if displacements else None,'max_m':max(displacements) if displacements else None,'large_flags':len(jump_rows)},'timing':times,'wall_seconds':time.perf_counter()-started,'source_baseline_frames_sha256':m['frames_sha256'],'source_video_sha256':sha,'identity_and_detections_frozen':True,'inference':'cached semantics and M5 people; no neural inference or cloud writes'}
    result={k:v for k,v in m.items() if k not in ('frame_offsets','frames_sha256','transitions','camera_timeline')};result.update(frame_offsets=offsets,frames_sha256=digest(output/'frames.jsonl'),transitions=transitions,camera_timeline=timeline,camera_model_version='m6',team_model_version='m6',source_baseline_frames_sha256=m['frames_sha256'],provenance='Experimental M6 measured static geometry, bounded offline smoothing and persistent team belief; frozen identities; no football analytics')
    (output/'manifest.json').write_text(json.dumps(result));(output/'diagnostics.json').write_text(json.dumps(report,indent=2,allow_nan=False));print(json.dumps({k:v for k,v in report.items() if k not in ('position_large_jumps','team_events','temporal_camera')},indent=2),flush=True);return report

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--video',required=True);p.add_argument('--baseline',required=True);p.add_argument('--poses',required=True);p.add_argument('--output',required=True);a=p.parse_args();run(a.video,a.baseline,a.poses,a.output)
