"""Paired CPU/MPS runtime and official-forward compatibility, no accuracy tuning."""
import argparse,json,sys,time,resource
from pathlib import Path
import cv2,numpy as np,torch
from src.serving.cv import digest
from src.vision.footandball24.inference import Detector,preprocess
from src.vision.intelligent23.inputs import write


def run(inputs,output,upstream):
    root=Path(output)
    if root.exists():raise ValueError('Preserve original runtime evidence')
    m=json.loads((Path(inputs)/'manifest.json').read_text());p=Path(inputs)/'model/official.pth';source=m['sources']['giroud'];cap=cv2.VideoCapture(source['video']);images=[]
    try:
        for frame in [0,225,449]:
            cap.set(cv2.CAP_PROP_POS_FRAMES,frame);ok,image=cap.read()
            if not ok:raise ValueError('Profile decode failed')
            images.append(image)
    finally:cap.release()
    torch.set_num_threads(4);cv2.setNumThreads(2);report={'frames':[0,225,449],'repetitions_per_frame':10,'warmup_per_frame':1,'devices':{},'torch':torch.__version__,'source_video_sha256':source['video_sha256']};outputs={}
    for device in ['cpu','mps']:
        if device=='mps' and not torch.backends.mps.is_available():report['devices']['mps']={'available':False};continue
        detector=Detector(p,device);times=[];outputs[device]=[]
        for image in images:
            detector.infer(image)
            for _ in range(10):
                value=detector.infer(image);times.append(value['timing'])
            outputs[device].append(value['heatmap'])
        report['devices'][device]={'available':True,'initialization_seconds':detector.initialization_seconds,'fallbacks':detector.fallbacks,'parameters':detector.parameters,
            'latency':{k:{'mean':float(np.mean([t[k] for t in times])),'median':float(np.median([t[k] for t in times])),'p95':float(np.quantile([t[k] for t in times],.95))} for k in times[0]}}
    if len(outputs)==2:report['cpu_mps_heatmap_max_abs_difference']=max(float(np.max(abs(a-b))) for a,b in zip(outputs['cpu'],outputs['mps']))
    sys.path.insert(0,str(Path(upstream).resolve()))
    from network.footandball import model_factory
    official=model_factory('fb1','train');official.phase='eval';official.eval();official.load_state_dict(torch.load(p,map_location='cpu',weights_only=True))
    adapter=Detector(p,'cpu');tensor,mapping=preprocess(images[0])
    with torch.inference_mode():
        old=official(tensor);new=adapter.model(tensor)
    report['official_cpu_forward_max_abs_difference']={key:float(torch.max(abs(old[i].permute(0,3,1,2)-new[key]))) for i,key in enumerate(['player_bbox','player','ball'])}
    if any(v>1e-6 for v in report['official_cpu_forward_max_abs_difference'].values()):raise ValueError('Official forward parity failed')
    report['peak_rss_bytes_macos']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss;report['checkpoint_sha256']=digest(p);report['checkpoint_bytes']=p.stat().st_size
    write(root,report);print(json.dumps(report,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--inputs',required=True);p.add_argument('--output',required=True);p.add_argument('--upstream',required=True);a=p.parse_args();run(a.inputs,a.output,a.upstream)
