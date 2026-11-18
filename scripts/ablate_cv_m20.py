"""Isolate motion-only versus added crop candidates on identical verified globals."""
import argparse,json
from pathlib import Path
import cv2
from src.serving.cv import digest
from src.vision.events.passes import signature
from src.vision.events.passes_m19 import detect_m19
from src.vision.ball20.tracker import TrajectoryBallTracker
from src.vision.ball20.detector import merge_candidates
from scripts.run_cv_m19 import compact_worlds,ball_metrics


def run(source,output):
    source=Path(source);output=Path(output)
    if output.exists():raise ValueError('Use a fresh ablation directory')
    output.mkdir(parents=True);cfg=json.loads(Path('configs/vision/cv_m20.json').read_text());report={'source':str(source),'config_sha256':signature(cfg),'detector_inference':'reused_verified_global_detections_only','local_crop_candidates_enabled':False,'independent_accuracy':None,'clips':{}}
    for name in ['test_match','team_goal','southampton']:
        base=Path('outputs/world_model')/(name+'_m9_ready');manifest=json.loads((base/'manifest.json').read_text());ball_manifest=json.loads((source/name/'ball_m20/manifest.json').read_text())
        if digest(Path(manifest['source_video']))!=manifest['video_sha256'] or digest(base/'frames.jsonl')!=manifest['frames_sha256'] or digest(source/name/'detector_inputs.jsonl')!=ball_manifest['detector_inputs_sha256'] or ball_manifest['config_sha256']!=signature(cfg):raise ValueError('Ablation source changed')
        worlds=compact_worlds(base/'frames.jsonl');rows=[];t=TrajectoryBallTracker(manifest['width'],manifest['height'],cfg);cap=cv2.VideoCapture(manifest['source_video'])
        try:
            with (source/name/'detector_inputs.jsonl').open() as records:
                for world,line in zip(worlds,records,strict=True):
                    record=json.loads(line);ok,image=cap.read()
                    if not ok:raise ValueError('Source missing')
                    people=[{'bbox':e['image_bbox']} for e in world['entities']]+[{'bbox':u['observation']['bbox']} for u in world['unassigned']]
                    t.prepare(world['frame'],record['timestamp'],world['camera'],people);t.plan['regions']=[]
                    ball,debug=t.consume(image,merge_candidates(record['global_candidates'],[]));rows.append(ball.model_dump())
        finally:cap.release()
        events,summary,evidence=detect_m19(name,worlds,rows,manifest['fps']);directory=output/name;directory.mkdir();(directory/'ball_frames.jsonl').write_text(''.join(json.dumps(r,separators=(',',':'))+'\n' for r in rows));(directory/'events.json').write_text(json.dumps(events,indent=2)+'\n')
        report['clips'][name]={'ball':ball_metrics(rows),'passes':summary};print(name,report['clips'][name]['ball']['direct_observed_frames'],summary['completed_pass_estimates'],flush=True)
    (output/'report.json').write_text(json.dumps(report,indent=2)+'\n');return report

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--output',required=True);a=p.parse_args();run(a.source,a.output)
