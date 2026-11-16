"""Reproducible frozen-input M19 association experiments, never default promotion."""
import argparse,json,time,gc
from pathlib import Path
from collections import Counter
import numpy as np
from src.serving.cv import digest
from src.vision.ball.schema import canonical
from src.vision.ball19.tracker import VisualBallTracker,AssociationConfig
from src.vision.events.passes_m19 import detect_m19
from src.vision.events.m19_players import short_links
from scripts.process_cv_passes import export_tables


def compact_worlds(path):
    worlds=[]
    for line in Path(path).open('rb'):
        w=json.loads(line);camera=w['camera'];cal=camera.get('calibration')
        worlds.append({'frame':w['frame'],'timestamp':w['timestamp'],'camera':{'epoch':camera['epoch'],'continuity':camera['continuity'],'calibration_valid':camera['calibration_valid'],'calibration':cal,'motion_evidence':{'field_motion':camera.get('motion_evidence',{}).get('field_motion',{})}},'entities':[{'track_id':e['track_id'],'image_bbox':e['image_bbox'],'visibility':e['visibility']} for e in w['entities'] if e['visibility']=='visible'],'unassigned':[{'observation':{'track_id':u['observation']['track_id'],'bbox':u['observation']['bbox']}} for u in w['unassigned']],'player_world':{'players':[{k:p.get(k) for k in ['entity_id','current_segment_id','current_track_id','visibility_state','identity_status','team','current_position','position_provenance','position_uncertainty_m']} for p in w['player_world']['players'] if p['visibility_state']=='VISIBLE']}})
    return worlds


def ball_metrics(rows):
    run=longest=changes=0;last=None;ids=set()
    for r in rows:
        if r['directly_observed']:
            run+=1;longest=max(longest,run);ids.add(r['track_id'])
            if last is not None and last!=r['track_id']:changes+=1
            last=r['track_id']
        else:run=0
    return {'frames':len(rows),'direct_observed_frames':sum(r['directly_observed'] for r in rows),'direct_observed_fraction':sum(r['directly_observed'] for r in rows)/len(rows),'predicted_only_frames':sum(r['predicted_image_position'] is not None for r in rows),'track_ids':len(ids),'observed_track_changes':changes,'longest_direct_run_frames':longest,'states':dict(Counter(r['tracking_status'] for r in rows)),'independent_precision':None,'independent_recall':None}


def run(output,prediction_seconds=None):
    out=Path(output)
    if out.exists():raise ValueError('Use a fresh output path; preserve experiments')
    out.mkdir(parents=True);config=json.loads(Path('configs/vision/cv_m19.json').read_text())
    if prediction_seconds is not None:config['ball']['prediction_seconds']=prediction_seconds
    report={'version':'cv-m19-v1','config':config,'clips':{},'independent_human_labels':0,'default_promotion':False}
    for name in ['test_match','team_goal','southampton']:
        base=Path('outputs/world_model')/(name+'_m9_ready');m=json.loads((base/'manifest.json').read_text());print('M19 VALIDATE',name,flush=True)
        if digest(base/'frames.jsonl')!=m['frames_sha256'] or digest(Path(m['source_video']))!=m['video_sha256']:raise ValueError('Frozen source changed')
        worlds=compact_worlds(base/'frames.jsonl');clip={'players':None,'ball':{}}
        cache=Path('outputs/world_model/m9_cache')/name;meta=json.loads((cache/'manifest.json').read_text())
        if meta['key']['source_sha256']!=m['video_sha256'] or digest(cache/'features.json')!=meta['features_sha256'] or digest(cache/'embeddings.npz')!=meta['embeddings_sha256']:raise ValueError('Appearance source mismatch')
        features=json.loads((cache/'features.json').read_text())
        with np.load(cache/'embeddings.npz',allow_pickle=False) as data:embeddings={k:data[k] for k in data.files}
        clip['players']=short_links(features,embeddings,worlds,m['fps'],config['player_links']);(out/name).mkdir();(out/name/'player_links.json').write_text(json.dumps(clip['players'],indent=2)+'\n');del features,embeddings;gc.collect()
        for version in ['m14','m16']:
            source=Path('outputs/world_model')/('ball_'+version)/name;bm=json.loads((source/'manifest.json').read_text())
            if digest(source/'ball_frames.jsonl')!=bm['ball_frames_sha256'] or bm['video_sha256']!=m['video_sha256'] or bm['source_m9_frames_sha256']!=m['frames_sha256']:raise ValueError('Ball source changed')
            raw=[json.loads(line) for line in (source/'ball_frames.jsonl').open()]
            threshold=bm.get('tracker_config',{}).get('minimum_confidence')
            if threshold is None:threshold=json.loads(Path('outputs/ml/ball_m16/selection.json').read_text())['detector']['threshold']
            tracker=VisualBallTracker(m['width'],m['height'],threshold,AssociationConfig(**config['ball']));directory=out/name/('ball_'+version);directory.mkdir();offsets=[];rows=[];tick=time.monotonic()
            with (directory/'ball_frames.jsonl').open('wb') as target,(directory/'video_ball_observations.jsonl').open('w') as canonical_file:
                for i,(world,b) in enumerate(zip(worlds,raw,strict=True)):
                    if b['frame']!=i or abs(b['timestamp']-i/m['fps'])>1e-6:raise ValueError('Input clock mismatch')
                    people=[{'bbox':e['image_bbox']} for e in world['entities']]+[{'bbox':e['observation']['bbox']} for e in world['unassigned']]
                    state=tracker.step(i,i/m['fps'],b['candidates'],world['camera'],people);rows.append(state.model_dump());offsets.append(target.tell());target.write((state.model_dump_json()+'\n').encode());canonical_file.write(canonical(name,m['fps'],state,bm['detector_sha256']).model_dump_json()+'\n')
            elapsed=time.monotonic()-tick;stats=ball_metrics(rows)
            manifest={'version':'m19','recording_id':name,'frames':m['frames'],'fps':m['fps'],'width':m['width'],'height':m['height'],'video_sha256':m['video_sha256'],'source_m9_frames_sha256':m['frames_sha256'],'ball_frames_sha256':digest(directory/'ball_frames.jsonl'),'canonical_sha256':digest(directory/'video_ball_observations.jsonl'),'detector_sha256':bm['detector_sha256'],'source_ball_version':version,'source_ball_frames_sha256':bm['ball_frames_sha256'],'frame_offsets':offsets,'statistics':stats,'config':config['ball'],'detector_confidence_threshold_unchanged':threshold,'inference_reused':True,'association_wall_seconds':elapsed,'interpolation_enabled':False}
            (directory/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
            events,summary,evidence=detect_m19(name,worlds,rows,m['fps']);p=out/name/('passes_'+version);p.mkdir();(p/'events.json').write_text(json.dumps(events,indent=2)+'\n');(p/'evidence_frames.jsonl').write_text(''.join(json.dumps(r,separators=(',',':'))+'\n' for r in evidence));export_tables(name,p,events,evidence,m['fps']);clip['ball'][version]={'statistics':stats,'passes':summary,'association_wall_seconds':elapsed,'input_model_sha256':bm['detector_sha256'],'input_ball_frames_sha256':bm['ball_frames_sha256'],'operating_threshold':threshold};print('M19 RESULT',name,version,json.dumps(clip['ball'][version]),flush=True)
        report['clips'][name]=clip;(out/'report.json').write_text(json.dumps(report,indent=2)+'\n');del worlds,raw,rows;gc.collect()
    return report

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',default='outputs/m19/run_v3');p.add_argument('--prediction-seconds',type=float);a=p.parse_args();run(a.output,a.prediction_seconds)
