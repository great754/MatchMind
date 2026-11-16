"""Run the new video pass stage after validating the unchanged player/camera/ball stages."""
import argparse,csv,gc,json,time
from pathlib import Path
from src.serving.cv import CVReview,digest
from src.vision.events.passes import PassConfig,detect_passes,signature


def lean_world(path):
    rows=[]
    with Path(path).open('rb') as stream:
        for line in stream:
            w=json.loads(line)
            rows.append({'frame':w['frame'],'timestamp':w['timestamp'],'camera':{'epoch':w['camera']['epoch'],'continuity':w['camera']['continuity']},'entities':[{'track_id':e['track_id'],'image_bbox':e['image_bbox'],'visibility':e['visibility']} for e in w['entities'] if e['visibility']=='visible'],'unassigned':[{'observation':{'track_id':u['observation']['track_id'],'bbox':u['observation']['bbox']}} for u in w['unassigned']],'player_world':{'players':[p for p in w['player_world']['players'] if p['visibility_state']=='VISIBLE']}})
    return rows


def export_tables(name,output,events,rows,fps):
    from src.assistant.catalog import letter
    groups={}
    for row in rows:
        for ref,person in row['people'].items():
            g=groups.setdefault(ref,{'first':row['frame'],'entity':person['entity'],'visible':0,'ball_observed':0})
            g['visible']+=1;g['ball_observed']+=row['observed']
    people=[]
    for ref,g in sorted(groups.items(),key=lambda pair:pair[1]['first']):
        if g['visible']<max(3,int(fps*.3)):continue
        people.append({'recording_id':name,'player_ref':ref,'temporary_label':'Player '+letter(len(people)),'entity_id':g['entity'],'visible_frames':g['visible'],'ball_observed_when_visible_frames':g['ball_observed'],'completed_pass_estimates':sum(e['status']=='estimated' and e['sender_ref']==ref for e in events),'received_pass_estimates':sum(e['status']=='estimated' and e['receiver_ref']==ref for e in events),'possible_transfers_for_review':sum(e['status']=='candidate' and e['sender_ref']==ref for e in events),'pass_attempts':None,'pass_accuracy':None,'goals':None,'shots':None,'tackles':None})
    fields=['recording_id','player_ref','temporary_label','entity_id','visible_frames','ball_observed_when_visible_frames','completed_pass_estimates','received_pass_estimates','possible_transfers_for_review','pass_attempts','pass_accuracy','goals','shots','tackles']
    with (output/'player_summary.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(people)
    with (output/'events.csv').open('w',newline='') as f:
        fields=['recording_id','event_id','event_type','status','start','end','sender_ref','receiver_ref','team','ball_observed_fraction','reasons'];writer=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');writer.writeheader();writer.writerows([{**e,'reasons':'|'.join(e['reasons'])} for e in events])
    (output/'player_summary.json').write_text(json.dumps(people,indent=2)+'\n')
    return {file:digest(output/file) for file in ['events.csv','player_summary.csv','player_summary.json']}


def process(name,base,ball_directory,output):
    started=time.monotonic();output=Path(output)
    if output.exists():raise ValueError('Use a fresh output directory; existing experiments are retained')
    print('VALIDATE',name,'all source, world and ball frames',flush=True)
    review=CVReview(base);review.attach_ball(ball_directory)
    if review.ball_review is None:raise ValueError(review.ball_diagnostic)
    metadata=review.manifest;ball=review.ball_review;cfg=PassConfig()
    world=lean_world(review.data);balls=[json.loads(line) for line in ball.path.open()]
    events,summary,rows=detect_passes(name,world,balls,metadata['fps'],cfg)
    output.mkdir(parents=True)
    (output/'events.json').write_text(json.dumps(events,indent=2,allow_nan=False)+'\n')
    (output/'evidence_frames.jsonl').write_text(''.join(json.dumps(r,allow_nan=False,separators=(',',':'))+'\n' for r in rows))
    exports=export_tables(name,output,events,rows,metadata['fps'])
    manifest={'exports_sha256':exports,'schema_version':'cv-passes-v1','recording_id':name,'frames':metadata['frames'],'fps':metadata['fps'],'width':metadata['width'],'height':metadata['height'],'video_sha256':metadata['video_sha256'],'world_frames_sha256':metadata['frames_sha256'],'ball_version':ball.manifest.get('version','m14'),'ball_frames_sha256':ball.manifest['ball_frames_sha256'],'ball_detector_sha256':ball.manifest['detector_sha256'],'events_sha256':digest(output/'events.json'),'evidence_sha256':digest(output/'evidence_frames.jsonl'),'config':cfg.model_dump(),'config_sha256':signature(cfg.model_dump()),'statistics':summary,'elapsed_seconds':time.monotonic()-started,'upstream_execution':'Validated complete existing camera/player/ball snapshots reused; no fresh neural inference or model training.','provenance':'Experimental image-space lower-body proximity, observed release/flight, stable receiver and same-team hypotheses; not independently verified passes.','unavailable':['goals','shots','tackles','pass_attempts','pass_accuracy','verified_possession']}
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2,allow_nan=False)+'\n')
    print('COMPLETE',name,json.dumps(summary),flush=True)
    del world,balls,rows,review;gc.collect()
    return manifest


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--recordings',nargs='+',default=['test_match','team_goal','southampton']);p.add_argument('--output',default='outputs/world_model/cv_passes_v1');p.add_argument('--ball-version',choices=['m14','m15','m16'],default='m14');a=p.parse_args()
    registry=json.loads(Path('configs/serving/cv.json').read_text());report={}
    for name in a.recordings:report[name]=process(name,registry[name],Path('outputs/world_model')/('ball_'+a.ball_version)/name,Path(a.output)/name)
    Path(a.output,'run_report.json').write_text(json.dumps(report,indent=2)+'\n')
if __name__=='__main__':main()
