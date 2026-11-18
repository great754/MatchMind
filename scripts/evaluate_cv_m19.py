"""Evaluate only independent source-reviewed labels; zero labels yields undefined accuracy."""
import argparse,json
from pathlib import Path
from src.evaluation.cv_m19 import evaluate
from src.serving.cv_review_labels import HumanReviewStore,HumanAnnotation


def run(root,labels_path=None,reviewer=None,output=None):
    target=Path(output) if output else Path(root)/'independent_evaluation.json'
    if output and target.exists():raise ValueError('Use a fresh evaluation output; preserve prior measurements')
    store=HumanReviewStore();dataset=store.dataset
    protocol=dict(dataset['protocol'])
    protocol['sources']=dataset['sources']
    annotations=[]
    if labels_path:
        document=json.loads(Path(labels_path).read_text())
        if document['dataset_fingerprint']!=dataset['fingerprint']:raise ValueError('Sealed task fingerprint mismatch')
        for row in document['annotations']:
            label=HumanAnnotation.model_validate({k:v for k,v in row.items() if k!='source_video_sha256'})
            store.validate(store.tasks[label.task_id]['recording_id'],label)
            if reviewer is None or row['reviewer_id']==reviewer:annotations.append(row)
    from src.serving.cv import digest
    for name,source in dataset['sources'].items():
        base=Path('outputs/world_model')/(name+'_m9_ready');manifest=json.loads((base/'manifest.json').read_text())
        if digest(Path(manifest['source_video']))!=source['video_sha256'] or digest(base/'frames.jsonl')!=manifest['frames_sha256']:raise ValueError('Evaluation source hash mismatch')
        for variant in ['m14','m16']:
            for directory in [Path(root)/name/('ball_'+variant),Path('outputs/world_model')/('ball_'+variant)/name]:
                metadata=json.loads((directory/'manifest.json').read_text())
                if digest(directory/'ball_frames.jsonl')!=metadata['ball_frames_sha256'] or metadata['video_sha256']!=source['video_sha256']:raise ValueError('Evaluation ball hash mismatch')
    players={}
    for name in dataset['sources']:
        frames={t[k] for t in dataset['tasks'] if t['recording_id']==name and t['kind']=='identity' for k in ['frame','other_frame']}
        people={}
        if annotations:
            from scripts.run_cv_m19 import compact_worlds
            from src.vision.events.passes import visible_players
            worlds=compact_worlds(Path('outputs/world_model')/(name+'_m9_ready')/'frames.jsonl')
            for frame in frames:people[frame]=list(visible_players(worlds[frame]).values())
        players[name]=people
    report={'dataset_fingerprint':dataset['fingerprint'],'sealed_tasks':len(dataset['tasks']),'reviewer':reviewer,'variants':{}}
    for variant in ['m14','m16']:
        balls={name:[json.loads(line) for line in (Path(root)/name/('ball_'+variant)/'ball_frames.jsonl').open()] for name in dataset['sources']}
        events={name:json.loads((Path(root)/name/('passes_'+variant)/'events.json').read_text()) for name in dataset['sources']}
        report['variants'][variant]=evaluate(dataset['tasks'],annotations,balls,events,protocol,players)
    for variant in ['m14','m16']:
        balls={name:[json.loads(line) for line in (Path('outputs/world_model')/('ball_'+variant)/name/'ball_frames.jsonl').open()] for name in dataset['sources']}
        from scripts.run_cv_m19 import compact_worlds
        from src.vision.events.passes import detect_passes
        events={name:detect_passes(name,compact_worlds(Path('outputs/world_model')/(name+'_m9_ready')/'frames.jsonl'),balls[name],dataset['sources'][name]['fps'])[0] for name in dataset['sources']}
        report['variants']['baseline_'+variant]=evaluate(dataset['tasks'],annotations,balls,events,protocol,players)
    target.parent.mkdir(parents=True,exist_ok=True);target.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',default='outputs/m19/run_v3');parser.add_argument('--labels');parser.add_argument('--reviewer');parser.add_argument('--output');args=parser.parse_args();run(args.root,args.labels,args.reviewer,args.output)
