"""Seal model-blind uniformly timed human review tasks; no labels are generated."""
import json,hashlib
from pathlib import Path
from src.vision.events.passes import signature


def prepare():
    protocol=json.loads(Path('configs/evaluation/cv_m19_protocol.json').read_text());tasks=[];sources={}
    for name in protocol['recordings']:
        m=json.loads(Path(f'outputs/world_model/{name}_m9_ready/manifest.json').read_text());n=m['frames'];fps=m['fps'];sources[name]={k:m[k] for k in ['video_sha256','frames','fps','width','height']}
        for i in range(protocol['ball_frames_per_clip']):
            frame=min(n-1,int((i+.5)*n/protocol['ball_frames_per_clip']));tasks.append({'kind':'ball','recording_id':name,'frame':frame,'timestamp':frame/fps})
        for i in range(protocol['pass_windows_per_clip']):
            center=(i+.5)*n/protocol['pass_windows_per_clip'];half=protocol['pass_window_seconds']*fps/2;start=max(0,int(center-half));end=min(n-1,int(center+half));tasks.append({'kind':'passes','recording_id':name,'start_frame':start,'end_frame':end,'start':start/fps,'end':end/fps})
        for i in range(protocol['identity_pairs_per_clip']):
            a=min(n-2,int((i+.5)*n/protocol['identity_pairs_per_clip']));b=min(n-1,a+max(1,round(protocol['identity_gap_seconds']*fps)));tasks.append({'kind':'identity','recording_id':name,'frame':a,'other_frame':b,'timestamp':a/fps,'other_timestamp':b/fps})
    for task in tasks:task['task_id']='m19_'+signature([protocol,sources[task['recording_id']],task])[:24]
    result={'schema_version':'cv-m19-review-tasks-v1','protocol':protocol,'sources':sources,'tasks':tasks,'selection_uses_model_candidates':False,'human_labels_created':0};result['fingerprint']=signature(result)
    p=Path('configs/evaluation/cv_m19_tasks.json')
    if p.exists() and json.loads(p.read_text())!=result:raise ValueError('Sealed review tasks changed; create a new protocol version')
    p.write_text(json.dumps(result,indent=2)+'\n');print('SEALED',len(tasks),'unlabelled source tasks; zero human labels')
    return result
if __name__=='__main__':prepare()
