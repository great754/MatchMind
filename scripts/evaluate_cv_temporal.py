"""Full-frame M6 provenance checks and matched-denominator temporal comparison."""
import argparse,json,itertools
from pathlib import Path
from collections import Counter
import numpy as np
from src.serving.cv import CVReview,digest
from src.vision.world.schema import WorldState
from src.observations.schema import RecordingObservation
from scripts.diagnose_cv_temporal import diagnose


def run_stats(rows,fps):
    runs=[];start=0
    for flag,g in itertools.groupby(r.camera.calibration_valid for r in rows):
        n=len(list(g));runs.append({'localized':flag,'start':start,'end':start+n-1,'frames':n,'seconds':n/fps});start+=n
    return {'runs':runs,'longest_supported':max((r for r in runs if r['localized']),key=lambda r:r['frames'],default=None),'longest_unknown':max((r for r in runs if not r['localized']),key=lambda r:r['frames'],default=None)}

def track_patterns(rows):
    displays=[]
    for r in rows:displays.append({**{e.track_id:e.team for e in r.entities if e.visibility=='visible'},**{u.observation.track_id:u.observation.team for u in r.unassigned}})
    patterns=[]
    for i in range(1,len(rows)-1):
        for track in displays[i-1].keys() & displays[i].keys() & displays[i+1].keys():
            a,b,c=displays[i-1][track],displays[i][track],displays[i+1][track]
            if a and a==c and b!=a:patterns.append({'frame':i,'track':track,'before':a,'current':b,'after':c,'kind':'unknown' if b is None else 'opposite'})
    return {'counts':dict(Counter(p['kind'] for p in patterns)),'patterns':patterns}

def pair_displacements(rows):
    result={};previous={}
    for r in rows:
        current={e.entity_id:e for e in r.entities if e.visibility=='visible' and e.current_pitch is not None}
        for key,e in current.items():
            if key in previous:result[(r.frame,key)]=float(np.linalg.norm(np.array(e.current_pitch)-previous[key].current_pitch))
        previous=current
    return result

def numbers(values,threshold):
    return {'pairs':len(values),'median_m':float(np.median(values)) if values else None,'p90_m':float(np.quantile(values,.9)) if values else None,'max_m':max(values) if values else None,'large_jump_screening_count':sum(v>threshold for v in values)}

def evaluate(before,after,output):
    before,after,output=map(Path,(before,after,output));output.mkdir(parents=True,exist_ok=True);old_review=CVReview(before);new_review=CVReview(after);m=new_review.manifest
    if m['source_baseline_frames_sha256']!=old_review.manifest['frames_sha256'] or m['video_sha256']!=old_review.manifest['video_sha256']:raise ValueError('Baseline/source identity changed')
    a=[WorldState.model_validate_json(l) for l in (before/'frames.jsonl').open()];b=[WorldState.model_validate_json(l) for l in (after/'frames.jsonl').open()];assert len(a)==len(b)==m['frames'];eligible=0;projected=0;unknown_teams=Counter();provenance=Counter();identity_rows=0
    frozen=('entity_id','track_id','last_track_id','track_history','visibility','identity_status','identity_confidence','image_bbox','image_foot','detection_confidence','appearance','appearance_evidence_count')
    with (after/'observations.jsonl').open() as observations:
        for index,(old,new,line) in enumerate(zip(a,b,observations,strict=True)):
            canonical=RecordingObservation.model_validate_json(line);assert canonical.frame==new.frame==index and canonical.recording_id==old.recording_id==new.recording_id
            assert len(old.entities)==len(new.entities);assert len(old.unassigned)==len(new.unassigned)
            for e,f in zip(old.entities,new.entities,strict=True):
                assert all(getattr(e,k)==getattr(f,k) for k in frozen);identity_rows+=1
                if f.visibility!='visible':assert f.current_pitch is None
            for u,v in zip(old.unassigned,new.unassigned,strict=True):assert u.observation.track_id==v.observation.track_id and u.observation.bbox==v.observation.bbox and u.hypotheses==v.hypotheses
            if new.camera.calibration_valid and new.camera.calibration_source_kind=='direct_calibrated':
                assert old.camera.calibration_valid and old.camera.calibration_source_kind=='direct_calibrated';np.testing.assert_allclose(new.camera.calibration.image_to_pitch,old.camera.calibration.image_to_pitch)
            if new.camera.memory and new.camera.memory.last_direct:
                anchor=a[new.camera.memory.last_direct.frame].camera;assert anchor.calibration_valid and anchor.calibration_source_kind=='direct_calibrated';assert anchor.epoch==new.camera.epoch
            for p in canonical.players:
                if p.visibility=='visible':
                    eligible+=1
                    if p.team is None:unknown_teams[p.team_evidence.unknown_reason]+=1
                else:assert p.pitch is None
                if p.pitch is not None:
                    projected+=1;provenance[p.localization_kind]+=1;assert new.camera.calibration_valid and p.camera is not None;assert p.image_foot==p.camera.image_foot
                    if p.localization_kind!='direct_calibrated':assert p.position.quality=='propagated'
                if not new.camera.calibration_valid:assert p.pitch is None
    old_pairs=pair_displacements(a);new_pairs=pair_displacements(b);common=sorted(old_pairs.keys()&new_pairs.keys());threshold=12/m['fps']+.5
    previous_unknown=[i for i,r in enumerate(a) if not r.camera.calibration_valid];recovered=[i for i in previous_unknown if b[i].camera.calibration_valid];newly_rejected=[i for i,r in enumerate(a) if r.camera.calibration_valid and not b[i].camera.calibration_valid]
    report={'classification':'LOCALLY VERIFIED structural/provenance checks and dependent temporal diagnostics; independent metric accuracy unavailable','recording_id':m['recording_id'],'frames':len(b),'person_observations':eligible,'projected':projected,'projected_provenance':dict(provenance),'frozen_identity_and_image_rows':identity_rows,'before_frames_sha256':old_review.manifest['frames_sha256'],'after_frames_sha256':m['frames_sha256'],'before_runs':run_stats(a,m['fps']),'after_runs':run_stats(b,m['fps']),'all_pairs_before':numbers(list(old_pairs.values()),threshold),'all_pairs_after':numbers(list(new_pairs.values()),threshold),'matched_pairs_before':numbers([old_pairs[k] for k in common],threshold),'matched_pairs_after':numbers([new_pairs[k] for k in common],threshold),'previous_unknown_frames':previous_unknown,'previous_unknown_recovered':recovered,'previous_unknown_still_unknown':[i for i in previous_unknown if i not in recovered],'previously_accepted_now_rejected':newly_rejected,'track_display_flickers_before':track_patterns(a),'track_display_flickers_after':track_patterns(b),'unknown_team_reasons_after':dict(unknown_teams),'epochs_before':sorted({r.camera.epoch for r in a}),'epochs_after':sorted({r.camera.epoch for r in b}),'important_cases':{str(i):{'timestamp':b[i].timestamp,'before':a[i].camera.motion_evidence['localization'],'after':b[i].camera.motion_evidence['localization'],'temporal':b[i].camera.motion_evidence['temporal_camera'],'projected_after':sum(e.current_pitch is not None for e in b[i].entities)+sum(u.observation.pitch is not None for u in b[i].unassigned)} for i in (1,349,350) if i<len(b)}}
    (output/'comparison.json').write_text(json.dumps(report,indent=2,allow_nan=False))
    diagnose(after,output/'release_after')
    print(json.dumps({k:v for k,v in report.items() if k in ('recording_id','person_observations','projected','matched_pairs_before','matched_pairs_after','previous_unknown_recovered','unknown_team_reasons_after')},indent=2));return report

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--before',required=True);p.add_argument('--after',required=True);p.add_argument('--output',required=True);a=p.parse_args();evaluate(a.before,a.after,a.output)
