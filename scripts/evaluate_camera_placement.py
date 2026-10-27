"""Evaluate M5 coverage and dependent diagnostics against immutable M4 snapshots.

No independent metric truth exists for this clip. Thresholded displacements below
are screening diagnostics, not measured teleports or accuracy.
"""
import argparse,json,itertools,hashlib
from pathlib import Path
from collections import Counter
import cv2,numpy as np
from src.vision.world.pitch_geometry import PitchGeometry
from src.vision.world.review import digest
from src.observations.schema import RecordingObservation

METRIC={'current_pitch','position_source','localization_confidence','projection_reason','projection_camera','last_reliable_pitch','last_localized_at','last_reliable_camera_epoch','last_velocity'}
TRACK_METRIC={'pitch','localization_source','localization_confidence','projection_reason','projection_camera'}
def runs(flags):
    result=[];start=0
    for flag,g in itertools.groupby(flags):
        n=len(list(g));result.append({'localized':bool(flag),'start':start,'end':start+n-1,'frames':n});start+=n
    return result

def polygon_area(points):
    if len(points)<3:return 0.
    p=np.asarray(points,float);return abs(float(np.sum(p[:,0]*np.roll(p[:,1],-1)-p[:,1]*np.roll(p[:,0],-1)))/2)

def polygon_iou(first,second):
    # Float64 clipping avoids OpenCV float32 near-collinear intersection errors.
    subject=list(map(lambda p:np.array(p,float),first));clip=np.asarray(second,float)
    orientation=np.sign(np.sum(clip[:,0]*np.roll(clip[:,1],-1)-clip[:,1]*np.roll(clip[:,0],-1)))
    cross=lambda a,b:float(a[0]*b[1]-a[1]*b[0])
    for a,b in zip(clip,np.roll(clip,-1,axis=0)):
        previous=subject;subject=[]
        if not previous:break
        for start,end in zip(previous,previous[1:]+previous[:1]):
            ds=orientation*cross(b-a,start-a);de=orientation*cross(b-a,end-a)
            if (ds>=0)!=(de>=0):subject.append(start+(end-start)*ds/(ds-de))
            if de>=0:subject.append(end)
    inter=polygon_area(subject);union=polygon_area(first)+polygon_area(second)-inter
    return inter/union if union>0 else 0.

def evaluate(baseline,current,output,video):
    baseline,current,output=map(Path,(baseline,current,output));output.mkdir(parents=True,exist_ok=True)
    before=[json.loads(s) for s in (baseline/'frames.jsonl').open()];after=[json.loads(s) for s in (current/'frames.jsonl').open()];manifest=json.loads((current/'manifest.json').read_text());fps=manifest['fps'];assert len(before)==len(after)==manifest['frames']
    assert digest(baseline/'frames.jsonl')==manifest['source_m4_frames_sha256'];assert digest(current/'frames.jsonl')==manifest['frames_sha256'];assert digest(video)==manifest['video_sha256']
    errors=[];losses=[];motion=Counter();unknown=Counter();residuals={'lines':[],'circles':[]};budgets=[];jumps=[];player_displacements=[];region_iou=[];anchors=[];semantic_dropouts=[];frozen_count=0
    def visible(row):return [e for e in row['entities'] if e['visibility']=='visible']
    def freeze(e):return {k:v for k,v in e.items() if k not in METRIC}
    def brief(c):
        loc=c['motion_evidence'].get('localization',{});return {'status':loc.get('status'),'reason':loc.get('outcome'),'absolute':loc.get('absolute_outcome'),'age_direct':loc.get('last_absolute_age_seconds'),'landmarks':loc.get('semantic_classes'),'flow':c['motion_evidence'].get('field_motion',{}).get('reason'),'anchor':c['motion_evidence'].get('anchor_measurement',{}).get('reason')}
    for i,(a,b) in enumerate(zip(before,after)):
        assert [freeze(e) for e in a['entities']]==[freeze(e) for e in b['entities']],f'People changed at {i}';frozen_count+=len(a['entities'])
        assert {k:v for k,v in {**{'decisions':[],'reconciliations':[]},**a}.items() if k not in ('entities','unassigned','camera')}=={k:v for k,v in b.items() if k not in ('entities','unassigned','camera')}
        assert len(a['unassigned'])==len(b['unassigned'])
        for u,v in zip(a['unassigned'],b['unassigned']):
            assert {k:x for k,x in u.items() if k!='observation'}=={k:x for k,x in v.items() if k!='observation'}
            assert {k:x for k,x in u['observation'].items() if k not in TRACK_METRIC}=={k:x for k,x in v['observation'].items() if k not in TRACK_METRIC}
        assert all(e['current_pitch'] is None for e in b['entities'] if e['visibility']!='visible')
        c=b['camera'];info=c['motion_evidence']['localization'];flow=c['motion_evidence']['field_motion'];motion[flow.get('reason','unreported')]+=1
        if not c['calibration_valid']:unknown[info['outcome']]+=1;assert c['calibration'] is None and c['visible_pitch_region'] is None
        else:
            for kind,stats in info.get('landmark_validation',{}).get('types',{}).items():
                residuals[kind].append({'frame':i,**stats,'rejected_weak_category':kind in info.get('landmark_validation',{}).get('rejected_categories',[])})
            budgets.append(info['relative_motion_error_budget_px'])
            if info['outcome']=='REOBSERVED_ANCHOR_PARTIAL_ACCEPTED':anchors.append(i)
            if info['status']=='TEMPORAL_TRACKED':semantic_dropouts.append({'frame':i,**brief(c),'validation':info.get('landmark_validation')})
        if i and before[i-1]['camera']['calibration_valid'] and not a['camera']['calibration_valid']:
            next_good=next((j for j in range(i+1,len(before)) if before[j]['camera']['calibration_valid']),None)
            losses.append({'frame':i,'previous_frame':i-1,'before_previous':brief(before[i-1]['camera']),'before_loss':brief(a['camera']),'next_before_localized':next_good,'m5':brief(c),'fixed_now':c['calibration_valid']})
        if i and c['calibration_valid'] and after[i-1]['camera']['calibration_valid']:
            old={e['entity_id']:e for e in visible(after[i-1]) if e['current_pitch'] is not None}
            for e in visible(b):
                if e['current_pitch'] is None or e['entity_id'] not in old:continue
                prev=old[e['entity_id']];d=float(np.linalg.norm(np.array(e['current_pitch'])-prev['current_pitch']));player_displacements.append(d)
                if d>12/fps+.5:
                    h=np.array(c['calibration']['image_to_pitch']).reshape(3,3);oldfoot=cv2.perspectiveTransform(np.array([[prev['image_foot']]],float),h)[0,0]
                    jumps.append({'frame':i,'entity':e['entity_id'],'meters':d,'screening_threshold_m':12/fps+.5,'before':prev['position_source'],'after':e['position_source'],'same_foot_camera_component_m':float(np.linalg.norm(oldfoot-prev['current_pitch'])),'changed_foot_component_m':float(np.linalg.norm(oldfoot-e['current_pitch'])),'image_foot_change_px':float(np.linalg.norm(np.array(e['image_foot'])-prev['image_foot']))})
            region_iou.append(polygon_iou(c['visible_pitch_region'],after[i-1]['camera']['visible_pitch_region']))
    canonical=Counter()
    for line in (current/'observations.jsonl').open():
        r=RecordingObservation.model_validate_json(line)
        for p in r.players:
            if p.pitch is not None:canonical[p.localization_kind]+=1;assert p.camera is not None and p.image_foot==p.camera.image_foot
            if p.localization_kind in ('partial_constrained','camera_propagated'):assert p.position.quality=='propagated'
            if p.visibility!='visible':assert p.pitch is None
    rb=runs([s['camera']['calibration_valid'] for s in before]);ra=runs([s['camera']['calibration_valid'] for s in after]);single=lambda r:[v['start'] for v in r if not v['localized'] and v['frames']==1 and v['start']>0 and v['end']<len(after)-1]
    statistics=lambda a:{'count':len(a),'median':float(np.median(a)) if len(a) else None,'p90':float(np.quantile(a,.9)) if len(a) else None,'max':float(max(a)) if len(a) else None}
    real=json.loads((current/'diagnostics.json').read_text())
    summary={'classification':'LOCALLY VERIFIED coverage / dependent diagnostics; independent video camera accuracy UNAVAILABLE','baseline':{'camera_frames':33,'camera_percent':100*33/541,'projected_players':262,'player_percent':100*262/5108},'current':real,'nonmetric_entity_rows_identical':frozen_count,'canonical_projected_sources':dict(canonical),'unknown_reasons':dict(unknown),'field_motion_outcomes':dict(motion),'runs':ra,'longest_localized':max((r for r in ra if r['localized']),key=lambda r:r['frames']),'longest_unknown':max((r for r in ra if not r['localized']),key=lambda r:r['frames']),'before_longest_unknown':max((r for r in rb if not r['localized']),key=lambda r:r['frames']),'one_frame_unknown_before':single(rb),'one_frame_unknown_after':single(ra),'all_m4_localization_losses':losses,'all_unknown_frames':[{'frame':i,**brief(r['camera'])} for i,r in enumerate(after) if not r['camera']['calibration_valid']],'center216':{'camera':brief(after[216]['camera']),'current_people':len(visible(after[216]))+len(after[216]['unassigned']),'semantic_constraint_upper_bound':after[216]['camera']['motion_evidence']['localization']['semantic_constraint_upper_bound']},'partial_reobserved_anchor_frames':anchors,'short_semantic_bridges':semantic_dropouts,'accepted_landmark_residuals':residuals,'relative_image_budget_source_pixels':statistics(budgets),'consecutive_visible_region_iou':statistics(region_iou),'projected_player_displacement_m':statistics(player_displacements),'large_displacement_screening_count':len(jumps),'large_displacements':jumps,'independent_metric_accuracy':'UNAVAILABLE: no labeled test_match camera/foot metric ground truth','timing_note':'Cached semantics and absolute fits; original decoding, fresh field motion, camera refinement, projection and output included; model inference excluded'}
    (output/'evaluation.json').write_text(json.dumps(summary,indent=2));(output/'dropouts.json').write_text(json.dumps(losses,indent=2))
    selected=[34,35,55,76,78,100,130,137,138,143,150,216,300,459,470,490,504,537]
    cap=cv2.VideoCapture(str(video));tiles=[];geo=PitchGeometry()
    for index in selected:
        cap.set(cv2.CAP_PROP_POS_FRAMES,index);ok,image=cap.read();assert ok;panels=[]
        for label,row in [('M4',before[index]),('M5',after[index])]:
            panel=image.copy();c=row['camera']
            if c['calibration_valid']:
                h=np.array(c['calibration']['image_to_pitch']).reshape(3,3)
                for seg in geo.overlay(h,manifest['width'],manifest['height']):cv2.line(panel,tuple(seg['start']),tuple(seg['end']),(255,220,40),2,cv2.LINE_AA)
                for e in visible(row):
                    if e['current_pitch'] is not None:cv2.circle(panel,tuple(map(lambda x:int(round(x)),e['image_foot'])),6,(80,30,255),2)
                for u in row['unassigned']:
                    if u['observation']['pitch'] is not None:
                        x,y,w,h=u['observation']['bbox'];cv2.circle(panel,(round(x+w/2),round(y+h)),6,(80,30,255),2)
            panel=cv2.resize(panel,(440,440));header=np.full((55,440,3),20,np.uint8);text=f"{label} frame {index} {brief(c)['status']}";cv2.putText(header,text,(8,20),cv2.FONT_HERSHEY_SIMPLEX,.48,(240,240,240),1);cv2.putText(header,'Cyan: projected pitch; red: projectable feet',(8,43),cv2.FONT_HERSHEY_SIMPLEX,.4,(240,240,240),1);panels.append(np.vstack([header,panel]))
        pair=np.hstack(panels);cv2.imwrite(str(output/f'frame_{index:03d}.jpg'),pair,[cv2.IMWRITE_JPEG_QUALITY,88]);tiles.append(cv2.resize(pair,(440,248)))
    cap.release();sheet=np.vstack([np.hstack(tiles[i:i+3]) for i in range(0,len(tiles),3)]);cv2.imwrite(str(output/'before_after_contact.jpg'),sheet,[cv2.IMWRITE_JPEG_QUALITY,90]);print(json.dumps({k:v for k,v in summary.items() if k in ('baseline','longest_localized','longest_unknown','unknown_reasons','one_frame_unknown_before','one_frame_unknown_after','center216','large_displacement_screening_count','canonical_projected_sources','nonmetric_entity_rows_identical')},indent=2))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--baseline',required=True);p.add_argument('--current',required=True);p.add_argument('--output',required=True);p.add_argument('--video',required=True);a=p.parse_args();evaluate(a.baseline,a.current,a.output,a.video)
