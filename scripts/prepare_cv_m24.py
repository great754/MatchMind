"""Freeze M24's protocol, real labels, source clocks and prior predictions before inference."""
import argparse,json,shutil
from datetime import datetime,timezone
from pathlib import Path
from src.serving.cv import digest
from src.vision.intelligent23.inputs import Inputs,write
from src.vision.intelligent23.evaluation import references
from src.vision.football22.review import DevelopmentReviewStore,DevelopmentLabel
from src.vision.events.passes import signature
from src.vision.ball.schema import BallFrame,VideoBallObservation,canonical


def prepare(output,upstream):
    root=Path(output);upstream=Path(upstream)
    if root.exists():raise ValueError('Use a new immutable M24 snapshot')
    root.mkdir(parents=True);files={}
    def copy(source,key):
        source=Path(source);target=root/key;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(source,target)
        if digest(source)!=digest(target):raise ValueError('Input changed during snapshot')
        target.chmod(0o400);files[key]={'path':str(target),'sha256':digest(target),'bytes':target.stat().st_size}
        return target
    config=copy('configs/vision/cv_m24.json','protocol.json');protocol=json.loads(config.read_text())
    inputs=Inputs('outputs/m23/inputs_v2');original,old_labels,new_tasks,new_labels=references(inputs)
    for group,tasks_path,labels_path in [
        ('original_m19',inputs.path('configs/evaluation/cv_m19_tasks.json'),inputs.path('outputs/m19/human_review/labels.json')),
        ('completed_m22',inputs.path('outputs/m22/completed_review_snapshot_v1/tasks.json'),inputs.path('outputs/m22/completed_review_snapshot_v1/labels.json')),
        ('giroud_completed',Path('configs/training/cv_giroud_review.json'),Path('outputs/m23/giroud_v1/human_review/labels.json'))]:
        copy(tasks_path,f'references/{group}/tasks.json');copy(labels_path,f'references/{group}/labels.json')
    g=DevelopmentReviewStore(root/'references/giroud_completed/tasks.json',root/'references/giroud_completed',approved_sources={'giroud'})
    giroud=g.labels()
    if len(giroud)!=40 or len({r['reviewer_id'] for r in giroud})!=1 or {r['task_id'] for r in giroud}!=set(g.tasks):raise ValueError('Expected all forty actual Giroud reviews from one reviewer')
    for r in giroud:
        valid=g.validate('giroud',DevelopmentLabel.model_validate({k:v for k,v in r.items() if k in DevelopmentLabel.model_fields}))
        if signature(valid)!=signature(r):raise ValueError('Giroud annotation provenance mismatch')
    groups={}
    for name,dataset,labels in [('original_m19',original,old_labels),('completed_m22',{'tasks':new_tasks},new_labels),('giroud_completed',g.dataset,giroud)]:
        lookup={t['task_id']:t for t in dataset['tasks']};rows=[]
        for row in labels:
            task=lookup[row['task_id']]
            if task.get('kind','ball')!='ball':continue
            rows.append({'task_id':row['task_id'],'recording_id':task['recording_id'],'frame':task['frame'],
                'state':row['state'],'bbox':row['bbox'],'model_assisted':name!='original_m19',
                'selection_kind':task.get('selection_kind','assisted_development' if name!='original_m19' else 'source_time_selection')})
        groups[name]=rows
    sources={};registry=json.loads(Path('configs/serving/cv_m23.json').read_text())
    for name in protocol['primary_recordings']:
        if name=='giroud':meta=json.loads(Path('outputs/m23/giroud_v1/world/manifest.json').read_text());video=Path(meta['source_video'])
        else:
            meta=inputs.json(f'outputs/world_model/{name}_m9_ready/manifest.json');video=inputs.path(meta['source_video'])
        if meta['frames']!=protocol['expected_frames'][name] or digest(video)!=meta['video_sha256']:raise ValueError('Approved video changed')
        cached=copy(video,f'videos/{name}.mp4');sources[name]={k:meta[k] for k in ['recording_id','frames','fps','width','height','video_sha256','frames_sha256']}
        sources[name]['video']=str(cached)
        for model in ['m21','m23','m22']:
            if name=='giroud' and model=='m22':continue
            directory=(Path(registry[name]['root'])/name/('full_m23' if model=='m23' else 'm21_temporal_offline') if name=='giroud' else
                Path(registry[name]['root'])/name/'full_m23' if model=='m23' else
                inputs.path(Path('outputs/m21/analysis_final_v1')/name/'temporal_offline/manifest.json').parent if model=='m21' else
                inputs.path(Path('outputs/m22/analysis_review_v1')/name/'manifest.json').parent)
            m=json.loads((directory/'manifest.json').read_text())
            if any(meta[k]!=m[k] for k in ['recording_id','frames','fps','width','height','video_sha256']) or m['source_m9_frames_sha256']!=meta['frames_sha256']:raise ValueError('Prior output source clock mismatch')
            ball=directory/'ball_frames.jsonl';canon=directory/'video_ball_observations.jsonl'
            if digest(ball)!=m['ball_frames_sha256'] or digest(canon)!=m['canonical_sha256']:raise ValueError('Prior output hash mismatch')
            with ball.open('rb') as stream,canon.open() as observed:
                for i in range(meta['frames']):
                    if stream.tell()!=m['frame_offsets'][i]:raise ValueError('Prior output byte index mismatch')
                    b=BallFrame.model_validate_json(stream.readline());c=VideoBallObservation.model_validate_json(observed.readline())
                    if b.frame!=i or abs(b.timestamp-i/meta['fps'])>1e-6 or canonical(name,meta['fps'],b,m['detector_sha256'])!=c:raise ValueError('Prior canonical output mismatch')
                if stream.read(1) or observed.read(1):raise ValueError('Unindexed prior output')
            copy(ball,f'baselines/{name}/{model}.jsonl');copy(directory/'manifest.json',f'baselines/{name}/{model}_manifest.json')
    weight=copy(upstream/'models/model_20201019_1416_final.pth','model/official.pth')
    if digest(weight)!=protocol['checkpoint_sha256']:raise ValueError('Official checkpoint mismatch')
    for path in ['README.md','LICENSE','network/footandball.py','network/fpn.py','network/nms.py','network/ssd_loss.py','run_detector.py','train_detector.py','data/issia_dataset.py','data/data_reader.py','data/augmentation.py','config.txt']:
        copy(upstream/path,'upstream/'+path)
    trained=inputs.manifest['complement_models']['test_match'];copy(inputs.path(trained['checkpoint']),'model/m22_giroud.pth')
    copy(inputs.path('configs/vision/cv_m22.json'),'model/m22_config.json')
    result={'schema_version':'cv-m24-inputs-v1','frozen_at_utc':datetime.now(timezone.utc).isoformat(),
        'protocol_sha256':digest(config),'upstream_commit':protocol['upstream_commit'],'files':files,'sources':sources,
        'references':groups,'reference_counts':{k:len(v) for k,v in groups.items()},'giroud_labels_sha256':files['references/giroud_completed/labels.json']['sha256'],
        'm22_giroud_checkpoint_sha256':trained['checkpoint_sha256'],'withdrawn_m21_set_used':False,
        'independent_final_test':False,'new_training':False,'no_scored_footandball_results_seen_before_freeze':True}
    write(root/'manifest.json',result);print(json.dumps({'sources':{n:s['frames'] for n,s in sources.items()},'references':result['reference_counts'],'protocol_sha256':result['protocol_sha256']}),flush=True)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True);p.add_argument('--upstream',required=True);a=p.parse_args();prepare(a.output,a.upstream)
