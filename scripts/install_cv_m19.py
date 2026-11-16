"""Validate and register an explicitly opt-in M19 sidecar without replacing defaults."""
import argparse,json
from pathlib import Path
from src.serving.cv import CVReview,digest
from src.serving.cv_events import CVEventReview
from src.vision.events.passes import PassConfig,signature


def install(root):
    root=Path(root);ball_registry={};events_registry={};report={}
    for name in ['test_match','team_goal','southampton']:
        base=Path(f'outputs/world_model/{name}_m9_ready');m=json.loads((base/'manifest.json').read_text());ball=root/name/'ball_m16';bm=json.loads((ball/'manifest.json').read_text());events=root/name/'passes_m16';rows=json.loads((events/'events.json').read_text());cfg=PassConfig();summary=json.loads((root/'report.json').read_text())['clips'][name]['ball']['m16']['passes']
        manifest={'schema_version':'cv-passes-v1','algorithm_version':'cv-pass-m19-v1','independent_reference_status':'awaiting_independent_human_labels',**{k:m[k] for k in ['recording_id','frames','fps','width','height','video_sha256']},'world_frames_sha256':m['frames_sha256'],'ball_version':'m19','ball_frames_sha256':bm['ball_frames_sha256'],'ball_detector_sha256':bm['detector_sha256'],'events_sha256':digest(events/'events.json'),'evidence_sha256':digest(events/'evidence_frames.jsonl'),'config':cfg.model_dump(),'config_sha256':signature(cfg.model_dump()),'statistics':summary,'unavailable':['goals','shots','tackles','pass_attempts','pass_accuracy','verified_possession']}
        (events/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');review=CVReview(base);review.attach_ball(ball)
        if review.ball_review is None:raise ValueError('M19 ball review rejected')
        layer=CVEventReview(events,review);report[name]={'events_validated':len(layer.events),'qualified_estimates':summary['completed_pass_estimates'],'candidate_transfers':summary['transfer_candidates'],'default_changed':False};ball_registry[name]=str(ball);events_registry[name]=str(events)
    Path('configs/serving/cv_ball_m19.json').write_text(json.dumps(ball_registry,indent=2)+'\n');Path('configs/serving/cv_events_m19.json').write_text(json.dumps(events_registry,indent=2)+'\n');(root/'installation.json').write_text(json.dumps(report,indent=2)+'\n');print(report,flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',default='outputs/m19/run_v3');a=p.parse_args();install(a.root)
