"""Independent current-pixel cross-model re-verification at the unchanged .35 visual gate."""
import argparse
import json
from pathlib import Path
import time
import cv2
import torch
from ultralytics import YOLO
from src.serving.cv import digest
from src.evaluation.cv_m19 import iou
from src.vision.advanced21.detection import DetectorBackend, bounded_region
from src.vision.advanced21.experiment import NAMES, fresh_directory, write_json


def run(source, output):
    source=Path(source);root=fresh_directory(output)
    report=json.loads((source/'report.json').read_text());config=json.loads((source/'config.json').read_text())
    device='mps' if torch.backends.mps.is_available() else 'cpu';torch.set_num_threads(4);cv2.setNumThreads(2)
    backends={name:DetectorBackend(YOLO(config['detectors'][name]),digest(config['detectors'][name]),ball_class=class_id,device=device)
              for name,class_id in [('soccer',0),('generic',32)]}
    config['cross_detector_verification']={'visual_threshold':.35,'minimum_box_iou':.1,
        'crop_size':192,'changes_original_confidence':False,
        'checkpoints':{name:b.checkpoint_sha256 for name,b in backends.items()}}
    write_json(root/'config.json',config)
    report['config_sha256']=digest(root/'config.json');report['cross_detector_verification']=config['cross_detector_verification']
    report['input_parent_sha256']=digest(source/'report.json')
    for name in NAMES:
        p=source/name/'inputs.jsonl'
        if digest(p)!=report['clips'][name]['inputs_sha256']:raise ValueError('Parent perception cache changed')
        directory=root/name;directory.mkdir();capture=cv2.VideoCapture('data/raw/'+name+'.mp4');started=time.perf_counter();views=0;inference_seconds=0
        try:
            with p.open() as stream,(directory/'inputs.jsonl').open('w') as target:
                for line in stream:
                    row=json.loads(line);ok,image=capture.read()
                    if not ok:raise ValueError('Source frame unavailable')
                    unique={(tuple(c['bbox']),c['confidence']):c for group in row['candidate_sets'].values() for c in group if c['confidence']>=.35}
                    regions={'soccer':[],'generic':[]}
                    for c in unique.values():
                        other='soccer' if c['inference_view'].startswith('current_generic') or c['inference_view']=='current_pixels_after_temporal_proposal' else 'generic'
                        x,y,w,h=c['bbox'];regions[other].append(bounded_region((x+w/2,y+h/2),image.shape[1],image.shape[0],192))
                    neural={}
                    for backend,boxes in regions.items():
                        neural[backend],timing=backends[backend].infer(image,boxes,'current_cross_verification_'+backend)
                        views+=timing['views'];inference_seconds+=timing['seconds']
                    checked={}
                    for key,c in unique.items():
                        other='soccer' if c['inference_view'].startswith('current_generic') or c['inference_view']=='current_pixels_after_temporal_proposal' else 'generic'
                        matches=[d for d in neural[other] if iou(c['bbox'],d['bbox'])>=.1]
                        best=max(matches,key=lambda d:d['confidence']) if matches else None
                        checked[key]={'other_backend':other,'visual_confidence':best['confidence'] if best else 0.,
                                      'matched_bbox':best['bbox'] if best else None,'accepted':best is not None and best['confidence']>=.35}
                    for group in row['candidate_sets'].values():
                        for c in group:c['cross_detector']=checked.get((tuple(c['bbox']),c['confidence']))
                    row['cross_detector_candidates']=neural
                    target.write(json.dumps(row,separators=(',',':'),allow_nan=False)+'\n')
        finally:capture.release()
        report['clips'][name]['inputs_sha256']=digest(directory/'inputs.jsonl')
        report['clips'][name]['cross_detector_seconds']=time.perf_counter()-started
        report['clips'][name]['cross_detector_inference_seconds']=inference_seconds
        report['clips'][name]['cross_detector_views']=views
        write_json(root/'report.json',report)
        print('M21 CROSS VERIFIER',name,'views',views,'seconds',round(time.perf_counter()-started,2),flush=True)
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--source',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();run(args.source,args.output)
