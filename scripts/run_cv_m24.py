"""Run the frozen official single-frame detector on exactly the approved short clips."""
import argparse,json,resource,time,hashlib
from pathlib import Path
import cv2,numpy as np,torch
from src.serving.cv import digest
from src.vision.intelligent23.inputs import write
from src.vision.footandball24.inference import Detector


def run(inputs,output,resume=False):
    inputs,root=Path(inputs),Path(output);manifest=json.loads((inputs/'manifest.json').read_text())
    protocol=json.loads((inputs/'protocol.json').read_text())
    if digest(inputs/'protocol.json')!=manifest['protocol_sha256']:raise ValueError('Frozen benchmark protocol changed')
    if root.exists() and not resume:raise ValueError('Use a fresh M24 inference root')
    root.mkdir(parents=True,exist_ok=True);torch.set_num_threads(4);cv2.setNumThreads(2)
    detector=Detector(inputs/'model/official.pth')
    report={'schema_version':'cv-m24-perception-v1','inputs_manifest_sha256':digest(inputs/'manifest.json'),
        'protocol_sha256':manifest['protocol_sha256'],'model_sha256':digest(inputs/'model/official.pth'),
        'parameters':detector.parameters,'checkpoint_bytes':(inputs/'model/official.pth').stat().st_size,
        'initialization_seconds':detector.initialization_seconds,'torch':torch.__version__,'opencv':cv2.__version__,
        'device':detector.device,'labels_read_during_inference':False,'no_temporal_inputs':True,'clips':{},
        'source_sha256':{str(p):digest(p) for p in [Path(__file__),*Path('src/vision/footandball24').glob('*.py')]}}
    if (root/'report.json').exists():
        previous=json.loads((root/'report.json').read_text())
        for key in ['inputs_manifest_sha256','protocol_sha256','model_sha256','source_sha256']:
            if previous[key]!=report[key]:raise ValueError('M24 resume provenance changed')
        report=previous
    write(root/'report.json',report)
    for name,mode,scale in [(n,'native',1) for n in protocol['primary_recordings']]+[('giroud','upscale2',2)]:
        key=name+'/'+mode;source=manifest['sources'][name];video=Path(source['video'])
        if digest(video)!=source['video_sha256']:raise ValueError('Pinned source changed')
        directory=root/name/mode;directory.mkdir(parents=True,exist_ok=True);path=directory/'frames.jsonl'
        if key in report['clips']:
            if digest(path)!=report['clips'][key]['frames_sha256']:raise ValueError('Completed inference changed')
            continue
        if path.exists():raise ValueError('Partial inference preserved; choose a fresh run directory')
        capture=cv2.VideoCapture(str(video));started=time.perf_counter();timings=[];offsets=[];write_seconds=0.
        try:
            with path.open('wb') as stream:
                for frame in range(source['frames']):
                    ok,image=capture.read()
                    if not ok or image.shape[:2]!=(source['height'],source['width']):raise ValueError('Source decode/shape mismatch')
                    values=detector.infer(image,scale);heat=values.pop('heatmap');tick=time.perf_counter()
                    heat_path=directory/'heatmaps'/f'{frame:06d}.npz';heat_path.parent.mkdir(exist_ok=True)
                    np.savez_compressed(heat_path,ball=heat)
                    row={'schema_version':'cv-m24-frame-v1','recording_id':name,'frame':frame,'timestamp':frame/source['fps'],
                        'source_pixels_sha256':hashlib.sha256(image.tobytes()).hexdigest(),'heatmap_file':str(heat_path),
                        'heatmap_sha256':digest(heat_path),'heatmap_shape':list(heat.shape),'synthetic_box_is_prediction':False,**values}
                    offsets.append(stream.tell());stream.write(json.dumps(row,separators=(',',':'),allow_nan=False).encode()+b'\n');stream.flush()
                    write_seconds+=time.perf_counter()-tick;timings.append(row['timing'])
                    if frame%100==0:print('M24 INFER',key,frame,'/',source['frames'],flush=True)
                if capture.read()[0]:raise ValueError('Unindexed source frames')
        finally:capture.release()
        wall=time.perf_counter()-started
        report['clips'][key]={'frames':source['frames'],'fps':source['fps'],'source_video_sha256':source['video_sha256'],
            'frames_sha256':digest(path),'frame_offsets':offsets,'wall_seconds':wall,'full_video_processing_fps':source['frames']/wall,
            'serialization_and_heatmap_write_seconds':write_seconds,'peak_rss_bytes_macos':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            'device':detector.device,'fallbacks':list(detector.fallbacks),
            'timing':{k:{'total':sum(t[k] for t in timings),'mean':float(np.mean([t[k] for t in timings])),
                'median':float(np.median([t[k] for t in timings])),'p95':float(np.quantile([t[k] for t in timings],.95))} for k in timings[0]}}
        write(root/'report.json',report);print('M24 COMPLETE',key,round(wall,2),'s',round(source['frames']/wall,2),'FPS',flush=True)
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--inputs',required=True);p.add_argument('--output',required=True);p.add_argument('--resume',action='store_true');a=p.parse_args();run(a.inputs,a.output,a.resume)
