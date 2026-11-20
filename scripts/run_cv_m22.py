"""Run the trained primary architecture on approved clips, with resumable source-pinned caches."""
import argparse,json,resource,time
from collections import deque
from pathlib import Path
import cv2
import torch
from src.vision.football22.data import NAMES,write
from src.vision.football22.inference import FootballDetector
from src.vision.football22.model import context_indices
from src.serving.cv import digest


def run(data_root,training_root,output,resume=False):
    torch.set_num_threads(4);cv2.setNumThreads(2)
    data_path=Path(data_root)/'dataset.json';dataset=json.loads(data_path.read_text());config_path=Path('configs/vision/cv_m22.json');config=json.loads(config_path.read_text());root=Path(output)
    if root.exists() and not resume:raise ValueError('Use a fresh inference directory or explicit --resume')
    root.mkdir(parents=True,exist_ok=True)
    provenance={'schema_version':'cv-m22-perception-v1','dataset_sha256':digest(data_path),'config_sha256':digest(config_path),
        'training_manifest_sha256':digest(Path(training_root)/'manifest.json'),'reference_role':'development_only','sources':dataset['sources'],
        'config':config,'torch':torch.__version__,'opencv':cv2.__version__,'causal_visual_input':True,'clips':{}}
    report=root/'report.json'
    if report.exists():
        previous=json.loads(report.read_text())
        if any(previous[k]!=provenance[k] for k in provenance if k!='clips'):raise ValueError('Resume source/config/training mismatch')
        provenance=previous
    write(report,provenance)
    for name in NAMES:
        meta=dataset['sources'][name];result=json.loads((Path(training_root)/name/'result.json').read_text())
        directory=root/name;directory.mkdir(exist_ok=True);path=directory/'detections.jsonl'
        if digest(meta['source_video'])!=meta['video_sha256']:raise ValueError('Source video changed')
        if name in provenance['clips']:
            if digest(path)!=provenance['clips'][name]['detections_sha256']:raise ValueError('Completed perception cache changed')
            print('M22 inference verified completed',name,flush=True);continue
        detector=FootballDetector(result['checkpoint'],result['checkpoint_sha256'],config)
        if detector.metadata['split']['excluded_development_source']!=name:raise ValueError('Target source influenced this checkpoint')
        done=0
        if path.exists():
            # Keep complete records only after interrupted writes. Every record
            # is checked against its exact frame/checkpoint and replayed pixels.
            good=[]
            for line in path.read_bytes().splitlines(keepends=True):
                try:row=json.loads(line)
                except ValueError:break
                if not line.endswith(b'\n') or row['frame']!=done or row['checkpoint_sha256']!=detector.checkpoint_sha256:raise ValueError('Invalid resumable inference index')
                good.append(line);done+=1
            path.write_bytes(b''.join(good))
        capture=cv2.VideoCapture(meta['source_video']);history={};epochs=dataset['source_cut_epochs'][name];started=time.perf_counter()
        try:
            with path.open('a') as output_stream:
                for index in range(meta['frames']):
                    ok,image=capture.read()
                    if not ok:raise ValueError('Video ended before its frozen clock')
                    history[index]=image
                    for old in list(history):
                        if old<index-6:del history[old]
                    if index<done:continue
                    ids=context_indices(index,meta['fps'],epochs);images=[history[i] for i in ids]
                    row=detector.infer(images,index);row.update(recording_id=name,timestamp=index/meta['fps'],context_frames=ids,camera_epoch=epochs[index])
                    output_stream.write(json.dumps(row,separators=(',',':'),allow_nan=False)+'\n');output_stream.flush()
                    if index%100==0:print('M22 INFER',name,index,'candidates',len(row['candidates']),flush=True)
        finally:capture.release()
        rows=[json.loads(line) for line in path.open()]
        if len(rows)!=meta['frames']:raise ValueError('Incomplete temporal inference')
        seconds=sum(r['seconds'] for r in rows)
        provenance['clips'][name]={'frames':len(rows),'detections_sha256':digest(path),'checkpoint_sha256':detector.checkpoint_sha256,
            'model':detector.metadata,'inference_seconds':seconds,'processing_fps':len(rows)/seconds,'wall_seconds_this_process':time.perf_counter()-started,
            'inference_views':sum(r['inference_views'] for r in rows),'peak_rss_bytes_macos':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            'mps_allocated_bytes':torch.mps.current_allocated_memory() if detector.device=='mps' else None,'device':detector.device}
        write(report,provenance);print('M22 INFER complete',name,provenance['clips'][name],flush=True)
        del detector
        if torch.backends.mps.is_available():torch.mps.empty_cache()
    return provenance

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',required=True);p.add_argument('--training',required=True);p.add_argument('--output',required=True);p.add_argument('--resume',action='store_true');a=p.parse_args();run(a.data,a.training,a.output,a.resume)
