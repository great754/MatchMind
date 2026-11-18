"""Matched 640/960 crop inference on sealed uniform-time frames; no accuracy truth."""
import argparse,json,time
from pathlib import Path
import cv2,torch,ultralytics
from src.serving.cv import digest
from src.vision.ball20.detector import YoloCropAdapter
from ml.ball16.detector import make_model


def run(root):
    root=Path(root);cfg=json.loads(Path('configs/vision/cv_m20.json').read_text());tasks=json.loads(Path('configs/evaluation/cv_m19_tasks.json').read_text())
    if digest(Path(cfg['detector']['checkpoint']))!=cfg['detector']['sha256']:raise ValueError('Checkpoint changed')
    torch.set_num_threads(4);cv2.setNumThreads(4);device='mps' if torch.backends.mps.is_available() else 'cpu';model=make_model(cfg['detector']['checkpoint'])
    result={'checkpoint_sha256':cfg['detector']['sha256'],'ultralytics_version':ultralytics.__version__,'device':device,'task_fingerprint':tasks['fingerprint'],'independent_accuracy':None,'false_positive_rate':None,'selection':'sealed uniform-time ball frames; identical existing M20 planned crops at both resolutions','results':{}}
    samples=[]
    for name in tasks['sources']:
        chosen={t['frame'] for t in tasks['tasks'] if t['recording_id']==name and t['kind']=='ball'}
        cache=[json.loads(line) for line in (root/name/'detector_inputs.jsonl').open()]
        manifest=json.loads(Path(f'outputs/world_model/{name}_m9_ready/manifest.json').read_text());cap=cv2.VideoCapture(manifest['source_video'])
        try:
            for frame in sorted(chosen):
                if not cache[frame]['regions']:continue
                cap.set(cv2.CAP_PROP_POS_FRAMES,frame);ok,image=cap.read()
                if not ok:raise ValueError('Source frame unavailable')
                samples.append((name,frame,image,cache[frame]['regions']))
        finally:cap.release()
    for resolution in [640,960]:
        adapter=YoloCropAdapter(model,resolution,device);rows=[]
        for name,frame,image,regions in samples:
            candidates,timing=adapter.infer(image,regions);rows.append({'recording_id':name,'frame':frame,'regions':len(regions),'seconds':timing['seconds'],'candidates':len(candidates),'above_frozen_visual_threshold':sum(c['confidence']>=cfg['detector']['threshold'] for c in candidates)})
            print('CROP BENCHMARK',resolution,name,frame,round(timing['seconds'],3),flush=True)
        result['results'][str(resolution)]={'sampled_frames_with_regions':len(rows),'crop_views':sum(r['regions'] for r in rows),'seconds':sum(r['seconds'] for r in rows),'above_threshold_candidates':sum(r['above_frozen_visual_threshold'] for r in rows),'rows':rows}
    (root/'crop_resolution_benchmark.json').write_text(json.dumps(result,indent=2)+'\n');return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);a=p.parse_args();run(a.root)
