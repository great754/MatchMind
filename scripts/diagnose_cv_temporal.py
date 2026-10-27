"""Measure every adjacent-frame transition before changing temporal algorithms."""
import argparse,json
from pathlib import Path
from collections import Counter
import cv2,numpy as np
from src.vision.world.review import digest
from scripts.evaluate_camera_placement import polygon_iou

def h_of(c):return np.array(c['calibration']['image_to_pitch']).reshape(3,3) if c['calibration_valid'] else None

def line_deltas(a,b,width,height):
    before={l['kind']:l for l in a['pitch_evidence'].get('lines',[]) if l.get('kind')};result=[]
    for line in b['pitch_evidence'].get('lines',[]):
        name=line.get('kind')
        if name not in before:continue
        old=before[name];p=np.array([old['start'],old['end']],float);q=np.array([line['start'],line['end']],float);u=p[1]-p[0];v=q[1]-q[0];u/=max(np.linalg.norm(u),1e-9);v/=max(np.linalg.norm(v),1e-9)
        result.append({'kind':name,'previous':p.tolist(),'current':q.tolist(),'midpoint_delta_px':(q.mean(axis=0)-p.mean(axis=0)).tolist(),'orientation_delta_degrees':float(np.degrees(np.arccos(np.clip(abs(u@v),0,1))))})
    return result

def line_intersections(lines):
    """Derived from line fits; never counted as independent motion evidence."""
    result=[]
    def intersect(a,b):
        a,b=np.asarray(a,float),np.asarray(b,float);u=a[1]-a[0];v=b[1]-b[0];d=b[0]-a[0];cross=lambda p,q:p[0]*q[1]-p[1]*q[0];den=cross(u,v)
        if abs(den)/max(1,np.linalg.norm(u)*np.linalg.norm(v))<.1:return None
        t,r=cross(d,v)/den,cross(d,u)/den
        if not -.1<=t<=1.1 or not -.1<=r<=1.1:return None
        return a[0]+t*u
    for i,a in enumerate(lines):
        for b in lines[i+1:]:
            previous=intersect(a['previous'],b['previous']);current=intersect(a['current'],b['current'])
            if previous is not None and current is not None:result.append({'features':[a['kind'],b['kind']],'previous':previous.tolist(),'current':current.tolist(),'delta_source_px':(current-previous).tolist(),'source':'derived intersection of observed line fits; correlated with line evidence'})
    return result

def diagnose(directory,output):
    root=Path(directory);output=Path(output);output.mkdir(parents=True,exist_ok=False);m=json.loads((root/'manifest.json').read_text());assert digest(root/'frames.jsonl')==m['frames_sha256'];rows=[json.loads(l) for l in (root/'frames.jsonl').open()];fps=m['fps'];width,height=m['width'],m['height'];transitions=[];jumps=[];team_patterns=[];kit_patterns=[];switches=[];states=Counter();teams=Counter();current_teams=Counter();history=[];eligible=0;projected=0
    visible=lambda r:{e['entity_id']:e for e in r['entities'] if e['visibility']=='visible'}
    for i,row in enumerate(rows):
        c=row['camera'];loc=c['motion_evidence']['localization'];states[loc['status']]+=1;now=visible(row);prev=visible(rows[i-1]) if i else {};nxt=visible(rows[i+1]) if i+1<len(rows) else {};changes=[];confidence_changes=[];h=h_of(c);oldh=h_of(rows[i-1]['camera']) if i else None
        for key,e in now.items():
            teams[e['team'] or 'unknown']+=1;current_teams[e['team_memory']['current_observation'] or 'unknown']+=1;eligible+=1;projected+=e['current_pitch'] is not None
            if key in prev:
                old=prev[key]
                confidence_changes.append({'entity':key,'identity_before':old['identity_confidence'],'identity_current':e['identity_confidence'],'team_before':old['team_confidence'],'team_current':e['team_confidence'],'position_before':old['localization_confidence'],'position_current':e['localization_confidence'],'visibility':e['visibility']})
                if key in nxt:
                    aa=old['team_memory']['current_observation'];bb=e['team_memory']['current_observation'];cc=nxt[key]['team_memory']['current_observation']
                    if aa and aa==cc and bb!=aa:kit_patterns.append({'frame':i,'entity':key,'before':aa,'current':bb,'after':cc,'kind':'unknown' if bb is None else 'opposite'})
                if old['team']!=e['team']:changes.append({'entity':key,'old':old['team'],'new':e['team'],'old_support':old['team_confidence'],'new_support':e['team_confidence']})
                if key in nxt and old['team'] is not None and old['team']==nxt[key]['team'] and e['team']!=old['team']:team_patterns.append({'frame':i,'entity':key,'before':old['team'],'current':e['team'],'after':nxt[key]['team'],'kind':'unknown_flicker' if e['team'] is None else 'opposite_flicker'})
                if oldh is not None and h is not None and old['current_pitch'] is not None and e['current_pitch'] is not None:
                    d=float(np.linalg.norm(np.array(e['current_pitch'])-old['current_pitch']));atold=cv2.perspectiveTransform(np.array([[old['image_foot']]],float),h)[0,0];entry={'frame':i,'entity':key,'meters':d,'speed_mps':d*fps,'same_foot_camera_component_m':float(np.linalg.norm(atold-old['current_pitch'])),'changed_foot_component_m':float(np.linalg.norm(atold-e['current_pitch'])),'image_foot_change_px':float(np.linalg.norm(np.array(e['image_foot'])-old['image_foot'])),'track_changed':old['track_id']!=e['track_id']}
                    history.append(entry)
                    if d>12/fps+.5:jumps.append(entry)
            for event in e['history']:
                if event['frame']==i and event['kind']=='team_changed':switches.append({'entity':key,**event,'team_memory':e['team_memory'],'current_kit':e['current_kit'],'track_id':e['track_id']})
        for u in row['unassigned']:teams[u['observation']['team'] or 'unknown']+=1;current_teams[u['observation']['team'] or 'unknown']+=1;eligible+=1;projected+=u['observation']['pitch'] is not None
        flow=c['motion_evidence'].get('temporal_camera',{}).get('field_motion',c['motion_evidence'].get('field_motion',{}));delta=None;transport=None;iou=None
        if oldh is not None and h is not None:
            grid=np.array([[(x*width,y*height) for x in (.2,.5,.8) for y in (.2,.5,.8)]],float);delta=float(np.quantile(np.linalg.norm(cv2.perspectiveTransform(grid,h)-cv2.perspectiveTransform(grid,oldh),axis=2),.9));iou=polygon_iou(c['visible_pitch_region'],rows[i-1]['camera']['visible_pitch_region'])
            if flow.get('accepted'):
                prediction=oldh@np.linalg.inv(np.array(flow['homography']));transport=float(np.quantile(np.linalg.norm(cv2.perspectiveTransform(grid,h)-cv2.perspectiveTransform(grid,prediction),axis=2),.9))
        transitions.append({'frame':i,'timestamp':row['timestamp'],'previous_state':rows[i-1]['camera']['motion_evidence']['localization']['status'] if i else None,'current_state':loc['status'],'next_state':rows[i+1]['camera']['motion_evidence']['localization']['status'] if i+1<len(rows) else None,'epoch':c['epoch'],'reason':loc['outcome'],'landmarks':loc['semantic_classes'],'field_motion':flow,'line_correspondences':line_deltas(rows[i-1]['camera'],c,width,height) if i else [],'line_intersection_movements':line_intersections(line_deltas(rows[i-1]['camera'],c,width,height)) if i else [],'camera_transform_difference_p90_m':delta,'motion_compensated_difference_p90_m':transport,'region_iou':iou,'player_displacements':[p for p in history if p['frame']==i],'team_changes':changes,'confidence_changes':confidence_changes,'previous_camera_support':rows[i-1]['camera']['motion_evidence']['localization'].get('support_score') if i else None,'support':loc.get('support_score')})
    isolated=[r['frame'] for r in transitions[1:-1] if r['current_state']=='UNKNOWN' and r['previous_state']!='UNKNOWN' and r['next_state']!='UNKNOWN'];values=[p['meters'] for p in history];summary={'recording_id':m['recording_id'],'baseline_frames_sha256':m['frames_sha256'],'source_video_sha256':m['video_sha256'],'frames':len(rows),'person_observations':eligible,'projected':projected,'camera_states':dict(states),'team_assignments':dict(teams),'current_kit_observations':dict(current_teams),'isolated_camera_gaps':isolated,'team_patterns':team_patterns,'current_kit_patterns':kit_patterns,'current_kit_pattern_counts':dict(Counter(p['kind'] for p in kit_patterns)),'team_pattern_counts':dict(Counter(p['kind'] for p in team_patterns)),'team_switches':switches,'large_jumps':jumps,'displacement_summary':{'pairs':len(values),'median_m':float(np.median(values)) if values else None,'p90_m':float(np.quantile(values,.9)) if values else None,'max_m':max(values) if values else None,'large_jump_screening_count':len(jumps)},'required_0058_case':transitions[min(len(rows)-1,int(.058*fps))],'frame350':transitions[min(350,len(rows)-1)]}
    with (output/'transitions.jsonl').open('w') as f:
        for r in transitions:f.write(json.dumps(r,allow_nan=False)+'\n')
    (output/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False));print(json.dumps({k:summary[k] for k in ('recording_id','frames','camera_states','team_assignments','isolated_camera_gaps','team_pattern_counts','displacement_summary')},indent=2));return summary
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--directory',required=True);p.add_argument('--output',required=True);a=p.parse_args();diagnose(a.directory,a.output)
