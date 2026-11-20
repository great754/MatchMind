"""Compare frozen selected-ball outputs on the completed assisted development review."""
import argparse,json
from pathlib import Path
from src.serving.cv import digest
from src.serving.ball import BallReview
from src.vision.football22.review import DevelopmentReviewStore,DevelopmentLabel
from src.vision.football22.data import NAMES,write
from src.evaluation.cv_m22_development import ball_metrics


def run(snapshot,output,candidate=None):
    root=Path(output)
    if root.exists():raise ValueError('Use a fresh development-evaluation directory')
    snapshot=Path(snapshot);store=DevelopmentReviewStore(snapshot/'tasks.json',snapshot);rows=store.labels()
    if len({r['reviewer_id'] for r in rows})!=1:raise ValueError('Choose/adjudicate one reviewer explicitly')
    for row in rows:
        item=DevelopmentLabel.model_validate({k:v for k,v in row.items() if k in DevelopmentLabel.model_fields})
        store.validate(store.tasks[item.task_id]['recording_id'],item)
    layers={'m20_control':lambda n:Path('outputs/m20/run_v1')/n/'ball_m20',
        'm21_offline_dag':lambda n:Path('outputs/m21/analysis_final_v1')/n/'offline_dag',
        'm21_temporal_offline':lambda n:Path('outputs/m21/analysis_final_v1')/n/'temporal_offline',
        'm22_initial':lambda n:Path('outputs/m22/analysis_final_v1')/n}
    if candidate:layers['m22_reviewed']=lambda n:Path(candidate)/n
    report={'reference_role':'development_only','model_assisted':True,'independent_final_test':False,
        'annotation_snapshot_sha256':digest(snapshot/'labels.json'),'task_manifest_sha256':digest(snapshot/'tasks.json'),
        'frame_count':len(rows),'visible_ball_frames':sum(r['state']=='visible' for r in rows),'systems':{}}
    tasks=list(store.tasks.values());sources=store.dataset['sources']
    for version,directory in layers.items():
        balls={};hashes={}
        for name in NAMES:
            meta=json.loads((Path('outputs/world_model')/(name+'_m9_ready')/'manifest.json').read_text())
            if any(meta[k]!=sources[name][k] for k in ['video_sha256','frames','fps','width','height']):raise ValueError('Review prediction source mismatch')
            review=BallReview(directory(name),meta)
            balls[name]=[review.frame(i).model_dump() for i in range(meta['frames'])]
            hashes[name]=review.manifest['ball_frames_sha256']
        result=ball_metrics(tasks,rows,balls,sources)
        result['per_recording']={name:ball_metrics([t for t in tasks if t['recording_id']==name],
            [r for r in rows if store.tasks[r['task_id']]['recording_id']==name],balls,sources) for name in NAMES}
        result['ball_frames_sha256']=hashes;report['systems'][version]=result
        print('M22 COMPLETED REVIEW',version,result['ball'],flush=True)
    root.mkdir(parents=True);write(root/'report.json',report)
    return report

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--snapshot',required=True);p.add_argument('--output',required=True);p.add_argument('--candidate-analysis');a=p.parse_args();run(a.snapshot,a.output,a.candidate_analysis)
