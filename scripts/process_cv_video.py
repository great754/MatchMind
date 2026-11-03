"""Run fresh people/semantic inference and the current M7 offline CV pipeline.

No annotated-video duplication, cloud writes, training or serving-time inference.
All intermediate stages belong to this source and retain hashes for validation.
"""
import argparse,json,time
from pathlib import Path
from collections import Counter
import cv2,numpy as np
from scripts.audit_player_detection import audit
from src.vision.world.review import digest
from src.vision.world.schema import CameraState,Detection
from src.vision.world.tracking import ShortTermTracker
from src.vision.world.engine import WorldModel
from src.vision.world.pose_review import run as fit_poses
from src.vision.world.pose_replay import replay as replay_camera
from src.vision.world.player_review import run as replay_people
from src.vision.world.camera_review import run as replay_persistent_camera
from src.serving.cv import CVReview
from src.vision.world.temporal_review import run as replay_temporal
from src.vision.world.coverage_review import process as replay_world_field


def seed_people(video,cache,output):
    """Minimal source-specific person seed for the existing offline pose runner."""
    source=json.loads(Path(cache).read_text());cap=cv2.VideoCapture(str(video))
    fps=cap.get(cv2.CAP_PROP_FPS);frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH));height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT));cap.release()
    if fps<=0 or len(source['frames'])!=frames or source['source_sha256']!=digest(video):raise ValueError('Detector/source clock or hash mismatch')
    output=Path(output);output.mkdir();tracker=ShortTermTracker();world=WorldModel(video.stem)
    with (output/'world.jsonl').open('w') as target:
        for index,row in enumerate(source['frames']):
            if row['frame']!=index:raise ValueError('Detector clock mismatch')
            tracks=tracker.update([Detection.model_validate(d) for d in row['detections']],index/fps,(width,height))
            camera=CameraState(frame=index,timestamp=index/fps)
            target.write(world.update(tracks,camera).model_dump_json()+'\n')
    (output/'diagnostics.json').write_text(json.dumps({'input':{'sha256':source['source_sha256'],'fps':fps,'frames':frames,'width':width,'height':height}}))


def summarize(directory,output,stages,wall):
    summary_started=time.perf_counter();review=CVReview(directory);manifest=review.manifest;fps=manifest['fps'];camera=Counter();projection=Counter();team=Counter();role=Counter();tracks=set();entities=set();max_people=0;minimum=None;eligible=0;projected=0;runs=[];unknown=Counter();switches=0;previous_teams={};cuts=0
    for line in (Path(directory)/'frames.jsonl').open():
        row=json.loads(line);c=row['camera'];loc=c['motion_evidence']['localization'];status=loc['status'];camera[status]+=1
        good=c['calibration_valid'];frame=row['frame'];cuts+=c['continuity']=='cut'
        if not good:unknown[loc['outcome']]+=1
        if not runs or runs[-1]['localized']!=good:runs.append({'localized':good,'start':frame,'end':frame,'frames':1})
        else:runs[-1]['end']=frame;runs[-1]['frames']+=1
        people=0
        for e in row['entities']:
            entities.add(e['entity_id'])
            if e['visibility']!='visible':
                if e['current_pitch'] is not None:raise ValueError('Historical position leaked into current')
                continue
            people+=1;eligible+=1;team[e['team'] or 'unknown']+=1;role[e['role']]+=1
            if e['track_id']:tracks.add(e['track_id'])
            projection[e['projection_reason']]+=1
            if e['current_pitch'] is not None:projected+=1
            if e['entity_id'] in previous_teams and e['team'] is not None and previous_teams[e['entity_id']] is not None and e['team']!=previous_teams[e['entity_id']]:switches+=1
            previous_teams[e['entity_id']]=e['team']
        for u in row['unassigned']:
            t=u['observation'];people+=1;eligible+=1;tracks.add(t['track_id']);team[t['team'] or 'unknown']+=1;projection[t['projection_reason']]+=1
            if t['pitch'] is not None:projected+=1
        max_people=max(max_people,people);minimum=people if minimum is None else min(minimum,people)
    source=Path(manifest['source_video']);final=json.loads((Path(directory)/'diagnostics.json').read_text());players=json.loads((Path(directory).parent/'players'/'diagnostics.json').read_text());poses=json.loads((Path(directory).parent/'poses'/'diagnostics.json').read_text())
    best=lambda value:max((r for r in runs if r['localized']==value),key=lambda r:r['frames'],default=None)
    summary={'status':'LOCALLY VERIFIED experimental pipeline output; independent metric/player/team accuracy unavailable','recording_id':manifest['recording_id'],'source':{'path':str(source),'sha256':manifest['video_sha256'],'bytes':source.stat().st_size,'fps':fps,'frames':manifest['frames'],'duration_seconds':manifest['frames']/fps,'width':manifest['width'],'height':manifest['height']},'person_observations':eligible,'mean_people_per_frame':eligible/manifest['frames'],'min_people_per_frame':minimum,'max_people_per_frame':max_people,'temporary_tracks':len(tracks),'anonymous_entity_hypotheses':len(entities),'observed_team_assignments':dict(team),'visible_assigned_role_hypotheses':dict(role),'persistent_team_switches':switches,'camera_states':dict(camera),'camera_localized_frames':manifest['frames']-camera['UNKNOWN'],'camera_coverage_percent':100*(manifest['frames']-camera['UNKNOWN'])/manifest['frames'],'players_projected':projected,'player_placement_percent':100*projected/max(1,eligible),'projection_reasons':dict(projection),'unknown_camera_reasons':dict(unknown),'longest_localized_run':best(True),'longest_unknown_run':best(False),'runs':runs,'cuts_detected':cuts,'camera_player_provenance_counts':final['metrics'],'team_model':players['team_model'],'absolute_pose_eligible_frames':poses['pose_eligible_frames'],'timing':{'stage_wall_seconds':stages,'end_to_end_wall_seconds':wall+time.perf_counter()-summary_started,'includes':'Fresh detector, segmentation, absolute fits, people/team replay, camera stages listed above, validation; no training or uploads'},'limitations':['COCO person candidates can include officials and duplicate/merged boxes; these are not verified unique-player counts.','Team/identity/roles are heuristic anonymous hypotheses, not verified names or accuracy.','Camera acceptance and coverage are diagnostic; no independent metric camera/foot ground truth.','No ball tracking, verified goal/event inference, football analytics or grounded CV Ask.','The original local source is required for this local CV review; Azure data remains unchanged.']}
    Path(output).write_text(json.dumps(summary,indent=2,allow_nan=False));return summary


def temporal_stages(video,output,stage):
    """Preserve the M5 snapshot; finalize current camera and frozen team evidence."""
    stage('m6_temporal_camera_and_team',lambda:replay_temporal(video,output/'final',output/'poses',output/'m6'))
    stage('m7_world_field_camera',lambda:replay_world_field(video,output/'m6',output/'poses',output/'current'))
    return output/'current'


def upgrade(video,output):
    """Resume only from a fully validated source-specific M5 run."""
    video=Path(video).resolve();output=Path(output);baseline=CVReview(output/'final')
    if digest(video)!=baseline.manifest['video_sha256']:raise ValueError('Resume source hash mismatch')
    old=json.loads((output/'summary.json').read_text());(output/'summary_m5.json').write_text(json.dumps(old,indent=2));stages=dict(old['timing']['stage_wall_seconds']);started=time.perf_counter()
    def stage(name,fn):
        print('START',name,flush=True);tick=time.perf_counter();fn();stages[name]=time.perf_counter()-tick;print('COMPLETE',name,round(stages[name],3),'seconds',flush=True)
    directory=temporal_stages(video,output,stage)
    report=summarize(directory,output/'summary.json',stages,old['timing']['end_to_end_wall_seconds']+time.perf_counter()-started)
    print('CURRENT SUMMARY',json.dumps(report,indent=2),flush=True);return report


def run(video,detector,semantic,semantic_sha256,output,camera_model="m7"):
    video=Path(video).resolve();output=Path(output)
    if not video.is_file():raise ValueError('Source video missing')
    if output.exists():raise ValueError('Choose a fresh output directory; existing runs are preserved')
    output.mkdir(parents=True);started=time.perf_counter();stages={}
    def stage(name,fn):
        print('START',name,flush=True);tick=time.perf_counter();fn();stages[name]=time.perf_counter()-tick;print('COMPLETE',name,round(stages[name],3),'seconds',flush=True)
    cache=output/'detections.json'
    stage('person_detection',lambda:audit(str(video),str(detector),str(cache)))
    stage('person_seed',lambda:seed_people(video,cache,output/'person_seed'))
    stage('semantic_and_absolute_camera',lambda:fit_poses(video,output/'person_seed',output/'poses',str(semantic),semantic_sha256))
    stage('camera_baseline',lambda:replay_camera(video,output/'person_seed',output/'poses',output/'camera_baseline'))
    stage('player_team_world',lambda:replay_people(video,output/'camera_baseline',cache,output/'players'))
    stage('persistent_camera',lambda:replay_persistent_camera(video,output/'players',output/'poses',output/'final'))
    directory=output/'final'
    if camera_model=='m7':directory=temporal_stages(video,output,stage)
    report=summarize(directory,output/'summary.json',stages,time.perf_counter()-started)
    print('FINAL SUMMARY',json.dumps({k:v for k,v in report.items() if k not in ('runs','team_model','camera_player_provenance_counts')},indent=2),flush=True)
    return report

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--video',required=True);p.add_argument('--detector',required=True);p.add_argument('--semantic-model',required=True);p.add_argument('--semantic-sha256',required=True);p.add_argument('--output',required=True);p.add_argument('--camera-model',choices=['m5','m7'],default='m7');p.add_argument('--resume-temporal',action='store_true');a=p.parse_args();upgrade(a.video,a.output) if a.resume_temporal else run(a.video,a.detector,a.semantic_model,a.semantic_sha256,a.output,a.camera_model)
