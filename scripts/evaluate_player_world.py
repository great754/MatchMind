"""Before/after player diagnostics and small, explicitly visual reference checks."""
import argparse,json
from pathlib import Path
from collections import Counter,defaultdict
import cv2,numpy as np
from src.serving.cv import CVReview
from src.vision.world.schema import WorldState
from src.observations.schema import RecordingObservation
from src.vision.world.player_geometry import project_player

def load(root):
    return [WorldState.model_validate_json(l) for l in (Path(root)/'frames.jsonl').open()]
def people(state):
    return [dict(bbox=e.image_bbox,team=e.team,entity=e.entity_id,track=e.track_id,kit=e.current_kit) for e in state.entities if e.visibility=='visible']+[dict(bbox=u.observation.bbox,team=u.observation.team,entity=None,track=u.observation.track_id,kit=u.observation.kit) for u in state.unassigned]
def metrics(rows,fps):
    track_frames=defaultdict(list);first={};last={};previous={};switches=0;recovered=0;off=0;teams=Counter();visible_teams=Counter()
    for s in rows:
        for p in people(s):track_frames[p['track']].append(s.frame);teams[str(p['team'])]+=1
        for e in s.entities:
            if e.visibility=='visible':first.setdefault(e.entity_id,s.frame);last[e.entity_id]=s.frame;visible_teams[str(e.team)]+=1
            old=previous.get(e.entity_id)
            if old:
                switches+=e.team is not None and old.team is not None and e.team!=old.team
                recovered+=e.visibility=='visible' and old.visibility!='visible'
                off+=e.visibility=='off_camera' and old.visibility!='off_camera'
            previous[e.entity_id]=e
    spans=[(v[-1]-v[0]+1)/fps for v in track_frames.values()]
    return {'detections':sum(len(people(s)) for s in rows),'detections_per_frame':sum(len(people(s)) for s in rows)/len(rows),'temporary_tracks':len(track_frames),'mean_track_span_seconds':float(np.mean(spans)),'short_tracks_under_0_5_seconds':sum(t<.5 for t in spans),'all_visible_team_assignments':dict(teams),'visible_persistent_team_assignments':dict(visible_teams),'team_switches':switches,'entities':len(rows[-1].entities),'entities_with_multiple_track_segments':sum(len(e.track_history)>1 for e in rows[-1].entities),'mean_entity_observed_lifetime_seconds':float(np.mean([(last[k]-v+1)/fps for k,v in first.items()])),'same_entity_recoveries':recovered,'off_camera_transitions':off,'camera_direct_frames':sum(s.camera.calibration_valid and s.camera.calibration_source_kind=='direct_calibrated' for s in rows),'camera_tracked_frames':sum(s.camera.calibration_valid and s.camera.calibration_source_kind=='camera_propagated' for s in rows)}

def evaluate(before,after,cache,output):
    before=Path(before);after=Path(after);output=Path(output);output.mkdir(parents=True,exist_ok=True);oldreview=CVReview(before);review=CVReview(after);a=load(before);b=load(after);fps=review.manifest['fps']
    assert oldreview.manifest['video_sha256']==review.manifest['video_sha256'];assert len(a)==len(b)==541
    previous={};reasons=Counter();max_payload=0
    canonical=[RecordingObservation.model_validate_json(l) for l in (after/'observations.jsonl').open()]
    assert len(canonical)==len(b)
    for old,new,observation in zip(a,b,canonical,strict=True):
        assert old.camera==new.camera and new.frame==observation.frame
        current=[e.track_id for e in new.entities if e.visibility=='visible'];assert len(current)==len(set(current))
        for e in new.entities:
            if e.visibility!='visible':assert e.image_bbox is None and e.current_pitch is None and e.current_kit is None and e.team_memory.current_observation is None
            if not new.camera.calibration_valid:assert e.current_pitch is None
            if e.visibility=='visible':
                p,reason=project_player(e.image_foot,new.camera);assert e.projection_reason==reason;assert (p is None)==(e.current_pitch is None);reasons[reason]+=1
            if e.position_source=='camera_propagated' and e.entity_id in previous:assert e.last_reliable_pitch==previous[e.entity_id].last_reliable_pitch and e.last_localized_at==previous[e.entity_id].last_localized_at
        for u in new.unassigned:
            p,reason=project_player(u.observation.foot,new.camera);assert reason==u.observation.projection_reason;assert (p is None)==(u.observation.pitch is None);reasons[reason]+=1
        for p in observation.players:
            if p.visibility=='visible':assert p.projection_reason not in (None,'UNKNOWN')
            if p.localization_kind=='camera_propagated':assert p.position.quality=='propagated'
        previous={e.entity_id:e for e in new.entities};max_payload=max(max_payload,len(new.model_dump_json()))
    proposals=json.loads(Path(cache).read_text());nms=proposals['nms_sensitivity_totals'];samples=[];cap=cv2.VideoCapture(str(review.video));tiles=[];crop_tiles=[]
    frame_tags={0:['clear kits','nearby opponents'],34:['pan','penalty area','direct calibration'],55:['tracked calibration'],100:['six-red-four-yellow-green reproducible scene','nearby opponents'],135:['overlap/partial-player proposal inspection'],216:['center circle','calibration unavailable'],281:['partial occlusion','nearby opponents'],350:['pan','edge truncation'],459:['right-side relocalization'],480:['team-change investigation'],503:['recorded team-change investigation'],538:['late tracked calibration','historical memory']}
    for index,tags in frame_tags.items():
        cap.set(cv2.CAP_PROP_POS_FRAMES,index);ok,image=cap.read();assert ok
        sides=[]
        for name,state in [('BEFORE',a[index]),('AFTER',b[index])]:
            annotated=image.copy()
            for p in people(state):
                x,y,w,h=map(int,p['bbox']);color=(100,120,255) if p['team']=='team_a' else (110,230,210) if p['team']=='team_b' else (200,200,200)
                cv2.rectangle(annotated,(x,y),(x+w,y+h),color,2);text=f'{p["entity"] or "ID?"} {p["track"]} {p["team"] or "Unknown"}'
                cv2.putText(annotated,text,(max(0,x),max(14,y)),cv2.FONT_HERSHEY_SIMPLEX,.36,color,1)
            annotated=cv2.resize(annotated,(540,540));cv2.putText(annotated,f'{name} frame {index} ({index/fps:.3f}s)',(8,22),cv2.FONT_HERSHEY_SIMPLEX,.6,(255,255,255),2);sides.append(annotated)
        comparison=np.hstack(sides);cv2.imwrite(str(output/f'frame-{index}-comparison.jpg'),comparison)
        samples.append({'frame':index,'timestamp':index/fps,'tags':tags,'before_team_counts':dict(Counter(str(p['team']) for p in people(a[index]))),'after_team_counts':dict(Counter(str(p['team']) for p in people(b[index]))),'camera_outcome':b[index].camera.motion_evidence.get('localization'),'visible_projection_reasons':dict(Counter(e.projection_reason for e in b[index].entities if e.visibility=='visible'))})
        if index in (100,216,503):
            for p in people(b[index]):
                kit=p['kit'];x,y,w,h=kit.torso_bbox if kit and kit.torso_bbox else p['bbox'];crop=image[max(0,int(y)):min(image.shape[0],int(y+h)),max(0,int(x)):min(image.shape[1],int(x+w))]
                if not crop.size:continue
                tile=np.zeros((160,135,3),np.uint8);tile[:110]=cv2.resize(crop,(135,110));cv2.putText(tile,f'{index} {p["track"][-4:]}',(2,124),cv2.FONT_HERSHEY_SIMPLEX,.36,(255,255,255),1);cv2.putText(tile,str(p['team']),(2,139),cv2.FONT_HERSHEY_SIMPLEX,.36,(255,255,255),1);cv2.putText(tile,('q %.2f'%kit.quality) if kit else 'no kit',(2,154),cv2.FONT_HERSHEY_SIMPLEX,.36,(255,255,255),1);crop_tiles.append(tile)
    cap.release()
    while len(crop_tiles)%6:crop_tiles.append(np.zeros_like(crop_tiles[0]))
    cv2.imwrite(str(output/'after-team-crops.jpg'),np.vstack([np.hstack(crop_tiles[i:i+6]) for i in range(0,len(crop_tiles),6)]))
    # Ten visually reviewed detector crops in the requested nearest scene. The
    # extra nested body-part proposal is excluded, never used to train the model.
    # These are agent visual reference labels, not an independent held-out set.
    reference=['team_a','team_a','team_b','team_a','team_a','team_a','team_a','team_b',None,'team_b','team_b']
    raw=proposals['frames'][100]['detections'];manual=[]
    for d,label in zip(raw,reference,strict=True):
        if label is None:continue
        def matched(state):return next(p['team'] for p in people(state) if np.allclose(p['bbox'],d['bbox']))
        manual.append({'frame':100,'bbox':d['bbox'],'visual_kit_reference':label,'before':matched(a[100]),'after':matched(b[100])})
    reference_report={'classification':'REFERENCE-EVALUATED targeted agent visual crop inspection, not independent/representative team accuracy','frame':100,'evaluated_unique_person_crops':len(manual),'before_agreement':sum(r['before']==r['visual_kit_reference'] for r in manual),'after_agreement':sum(r['after']==r['visual_kit_reference'] for r in manual),'source_sha256':review.manifest['video_sha256'],'labels_excluded_from_training':True,'image_not_held_out_from_unsupervised_video_fitting':True,'excluded_nested_proposal_count':1,'rows':manual}
    (output/'manual_team_reference.json').write_text(json.dumps(reference_report,indent=2))
    baseline=metrics(a,fps);current=metrics(b,fps);decisions=Counter();rejected=0;accepted=[]
    for l in (after/'associations.jsonl').open():
        d=json.loads(l);decisions[d['decision']]+=1;rejected+=sum(not e['eligible'] for e in d['evidence'])
        if d['decision']=='reassociated':accepted.append(d)
    report={'before':baseline,'after':current,'nms_sensitivity_totals':nms,'projection_reasons':dict(reasons),'camera_states_identical':True,'all_frame_integrity_checked':len(b),'maximum_serialized_world_frame_bytes':max_payload,'decision_counts':dict(decisions),'rejected_candidate_comparisons':rejected,'accepted_reassociations':accepted,'reference':reference_report,'manual_frames':samples,'independent_player_identity_accuracy':None,'independent_full_video_team_accuracy':None}
    (output/'evaluation.json').write_text(json.dumps(report,indent=2,allow_nan=False));print(json.dumps({k:v for k,v in report.items() if k not in ('reference','manual_frames','accepted_reassociations')},indent=2))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--before',required=True);p.add_argument('--after',required=True);p.add_argument('--cache',required=True);p.add_argument('--output',required=True);a=p.parse_args();evaluate(a.before,a.after,a.cache,a.output)
