"""Add a frozen external-training visual-pair ablation without rerunning detector inference."""
import argparse
import copy
import json
from pathlib import Path
import time
import cv2
import numpy as np
import torch
from ml.ball16.temporal import VisualPair, crop, pair
from src.serving.cv import digest
from src.vision.advanced21.experiment import NAMES, fresh_directory, write_json
from src.vision.advanced21.temporal import center


def run(source, output):
    source = Path(source)
    report = json.loads((source/'report.json').read_text())
    root = fresh_directory(output)
    checkpoint = Path('outputs/ml/ball_m16/temporal_visual.pth')
    provenance = json.loads(Path('outputs/ml/ball_m16/temporal_result.json').read_text())
    if digest(checkpoint) != provenance['checkpoint_sha256']:
        raise ValueError('Existing visual-pair checkpoint changed')
    model = VisualPair()
    model.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=True)['model'])
    model.eval();torch.set_num_threads(4);cv2.setNumThreads(2)
    config = json.loads((source/'config.json').read_text())
    config['external_visual_verifier'] = {'checkpoint_sha256': digest(checkpoint),
        'architecture': 'Frozen M16 shared CNN visual-pair model',
        'training_positive_pairs': provenance['positive_pairs'],
        'training_negative_regions': provenance['reviewed_negative_regions'],
        'training_negative_origin': 'Prior Codex source-only review; not independent human evaluation truth',
        'source_training_group': 'SoccerNet SN_game6; organizer positives with unverified visibility',
        'calibrated': False, 'development_labels_used_to_train_CNN': False}
    write_json(root/'config.json',config)
    report['config_sha256'] = digest(root/'config.json')
    report['external_visual_verifier'] = config['external_visual_verifier']
    report['input_parent_sha256'] = digest(source/'report.json')
    for name in NAMES:
        input_path=source/name/'inputs.jsonl'
        if digest(input_path)!=report['clips'][name]['inputs_sha256']:
            raise ValueError('Parent inference cache changed')
        directory=root/name;directory.mkdir()
        video=cv2.VideoCapture('data/raw/'+name+'.mp4');previous=None;previous_epoch=None;started=time.perf_counter();scored=0
        try:
            with input_path.open() as original,(directory/'inputs.jsonl').open('w') as target:
                for line in original:
                    row=json.loads(line);ok,image=video.read()
                    if not ok:raise ValueError('Source frame unavailable')
                    if row['camera_epoch']!=previous_epoch:previous=None
                    previous_epoch=row['camera_epoch']
                    candidates={}
                    for group in row['candidate_sets'].values():
                        for c in group:
                            if c['confidence']>=.35:candidates[(tuple(c['bbox']),c['confidence'])]=c
                    keys=list(candidates);scores={key:None for key in keys}
                    if previous is not None and keys:
                        pgray,cgray=cv2.cvtColor(previous,cv2.COLOR_BGR2GRAY),cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
                        points=np.float32([center(candidates[key]['bbox']) for key in keys]).reshape(-1,1,2)
                        old,status,_=cv2.calcOpticalFlowPyrLK(cgray,pgray,points,None,winSize=(19,19),maxLevel=2)
                        back,reverse,_=cv2.calcOpticalFlowPyrLK(pgray,cgray,old,None,winSize=(19,19),maxLevel=2) if old is not None else (None,None,None)
                        batch,valid=[],[]
                        if back is not None and reverse is not None:
                            for j,key in enumerate(keys):
                                if not status[j,0] or not reverse[j,0] or not np.isfinite(old[j]).all() or np.linalg.norm(back[j]-points[j])>4:continue
                                x,y=old[j,0];box=candidates[key]['bbox'];w,h=box[2:]
                                if not 0<=x<previous.shape[1] or not 0<=y<previous.shape[0]:continue
                                a=crop(previous,[float(x-w/2),float(y-h/2),w,h]);b=crop(image,box)
                                batch.append(pair(a,b));valid.append(key)
                            if batch:
                                with torch.inference_mode():predictions=model(torch.stack(batch)).sigmoid().tolist()
                                scores.update(zip(valid,predictions,strict=True));scored+=len(valid)
                    for group in row['candidate_sets'].values():
                        for c in group:c['temporal_features']['external_temporal_score']=scores.get((tuple(c['bbox']),c['confidence']))
                    target.write(json.dumps(row,separators=(',',':'),allow_nan=False)+'\n')
                    previous=image
        finally:video.release()
        report['clips'][name]['inputs_sha256']=digest(directory/'inputs.jsonl')
        report['clips'][name]['external_verifier_seconds']=time.perf_counter()-started
        report['clips'][name]['external_verifier_scored_candidates']=scored
        write_json(root/'report.json',report)
        print('M21 EXTERNAL PAIR',name,'scored',scored,'seconds',round(time.perf_counter()-started,2),flush=True)
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--source',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();run(args.source,args.output)
