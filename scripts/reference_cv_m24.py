"""Fixed M22 secondary reference on Giroud, independent of M23 search regions."""
import argparse,json,time
from pathlib import Path
from collections import deque
import cv2,torch
from src.serving.cv import digest
from src.vision.football22.inference import FootballDetector
from src.vision.footandball24.evaluation import center
from src.vision.intelligent23.inputs import write


def run(inputs,output):
    inputs=Path(inputs);root=Path(output)
    if root.exists():raise ValueError('Preserve completed references; use a fresh directory')
    root.mkdir(parents=True);m=json.loads((inputs/'manifest.json').read_text());config=json.loads((inputs/'model/m22_config.json').read_text())
    torch.set_num_threads(4);cv2.setNumThreads(2)
    detector=FootballDetector(inputs/'model/m22_giroud.pth',m['m22_giroud_checkpoint_sha256'],config)
    source=m['sources']['giroud'];cap=cv2.VideoCapture(source['video']);history=deque(maxlen=7);started=time.perf_counter()
    try:
        with (root/'giroud.jsonl').open('w') as stream:
            for frame in range(source['frames']):
                ok,image=cap.read()
                if not ok:raise ValueError('Source decode failed')
                history.append(image);images=[history[max(0,len(history)-1-d)] for d in [6,3,0]]
                raw=detector.infer(images,frame);eligible=[c for c in raw['candidates'] if c['observation_eligible']]
                def convert(c):return {'center':center(c['bbox']),'confidence':c['confidence'],'bbox':c['bbox']}
                row={'frame':frame,'selected':convert(eligible[0]) if eligible else None,'candidates':[convert(c) for c in raw['candidates']]}
                stream.write(json.dumps(row,allow_nan=False)+'\n')
                if frame%100==0:print('M24 M22 reference',frame,flush=True)
    finally:cap.release()
    write(root/'report.json',{'frames':source['frames'],'seconds':time.perf_counter()-started,'device':detector.device,
        'source_sha256':source['video_sha256'],'checkpoint_sha256':m['m22_giroud_checkpoint_sha256'],
        'output_sha256':digest(root/'giroud.jsonl'),'config_sha256':digest(inputs/'model/m22_config.json'),
        'method':'fresh fixed M22 raw temporal top1; global native tiles; original dual 0.35 gates; context -6,-3,0; no M23 regions; not the earlier frozen M22 offline pipeline',
        'assumption':'No Giroud camera cuts in the frozen source clock; no tracker state used.'})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--inputs',required=True);p.add_argument('--output',required=True);a=p.parse_args();run(a.inputs,a.output)
