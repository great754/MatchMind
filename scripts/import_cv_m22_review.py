"""Freeze a NEW training-data revision from explicitly verified M22 development labels."""
import argparse,json
from pathlib import Path
import cv2
from src.vision.football22.data import write,frame
from src.vision.football22.review import DevelopmentLabel,DevelopmentReviewStore
from src.serving.cv import digest
from src.vision.events.passes import signature


def run(base_root,output,reviewer=None,task_manifest='configs/training/cv_m22_tasks.json',labels_root='outputs/m22/human_review'):
    base=Path(base_root)/'dataset.json';dataset=json.loads(base.read_text());root=Path(output)
    if root.exists():raise ValueError('Use a fresh training-data revision')
    store=DevelopmentReviewStore(task_manifest,labels_root);annotations=store.labels()
    if not annotations:raise ValueError('No human M22 labels saved; model proposals cannot be imported as truth')
    reviewers={r['reviewer_id'] for r in annotations}
    if reviewer is None and len(reviewers)!=1:raise ValueError('Select one reviewer or adjudicate labels explicitly')
    reviewer=reviewer or next(iter(reviewers));chosen=[r for r in annotations if r['reviewer_id']==reviewer]
    if not chosen:raise ValueError('No annotations for the requested reviewer')
    root.mkdir(parents=True);rows=list(dataset['rows']);seen={(r['recording_id'],r['frame']) for r in rows};excluded=[];added=[]
    for annotation in chosen:
        body=DevelopmentLabel.model_validate({k:v for k,v in annotation.items() if k in DevelopmentLabel.model_fields})
        task=store.tasks[body.task_id];name=task['recording_id'];source=dataset['sources'][name]
        checked=store.validate(name,body)
        if annotation['source_video_sha256']!=source['video_sha256'] or checked['frame']!=annotation['frame']:
            raise ValueError('Saved annotation/source mismatch')
        key=(name,task['frame'])
        if key in seen:raise ValueError('Do not replace original labels; explicitly version/adjudicate conflicting supervision')
        negatives=[r['bbox'] for r in checked['hard_negative_regions']]
        if body.state=='uncertain' and not negatives:
            excluded.append({'task_id':body.task_id,'reason':'uncertain visibility without a verified negative region'});continue
        path=root/'verified_frames'/name/(str(task['frame'])+'.png');path.parent.mkdir(parents=True,exist_ok=True)
        if not cv2.imwrite(str(path),frame(source,task['frame'])):raise ValueError('Source frame write failed')
        rows.append({'recording_id':name,'frame':task['frame'],'timestamp':task['frame']/source['fps'],
            'source_video_sha256':source['video_sha256'],'path':str(path),'image_sha256':digest(path),
            'state':body.state,'bbox':body.bbox,'origin':'human_confirmed_model_assisted_development',
            'model_assisted':True,'reference_role':'development_only','negative_regions':negatives})
        seen.add(key);added.append(body.task_id)
    result={**{k:v for k,v in dataset.items() if k!='fingerprint'},'rows':rows,
        'parent_dataset_sha256':digest(base),'m22_task_manifest_sha256':digest(task_manifest),
        'm22_annotation_snapshot_sha256':digest(Path(labels_root)/'labels.json'),
        'additional_human_confirmed_frames':len(added),'excluded_uncertain':excluded,
        'hard_negative_policy':'Only explicitly rejected source regions supervised; no surrounding unknown pixels made negative',
        'independent_test_available':False,'withdrawn_m21_set_used':False}
    result['fingerprint']=signature(result);write(root/'dataset.json',result)
    print('New development-data revision:',len(added),'human-confirmed frames;',len(excluded),'uncertain tasks excluded. Original labels preserved.')
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--base',required=True);p.add_argument('--output',required=True);p.add_argument('--reviewer');a=p.parse_args();run(a.base,a.output,a.reviewer)
