"""Fresh inference on sealed model-blind review frames; never accuracy without human labels."""
import gc,json,time
from pathlib import Path
import cv2,torch
from src.serving.cv import digest
from ml.ball14.model import make_model as model14,infer as infer14
from ml.ball16.detector import make_model as model16,infer as infer16

def run():
    torch.set_num_threads(4);cv2.setNumThreads(4);dataset=json.loads(Path('configs/evaluation/cv_m19_tasks.json').read_text());selection14=json.loads(Path('outputs/ml/ball_m14/campaign_selection.json').read_text())['selected'];selection16=json.loads(Path('outputs/ml/ball_m16/selection.json').read_text());weights=json.loads(Path('outputs/ml/ball_m16/training_result.json').read_text())
    if digest(Path(selection14['checkpoint']))!=selection14['checkpoint_sha256'] or digest(Path(weights['checkpoint']))!=weights['checkpoint_sha256']:raise ValueError('Frozen checkpoint changed')
    result={'task_fingerprint':dataset['fingerprint'],'frames_selected_without_model_outputs':True,'training':False,'independent_accuracy':None,'variants':{}}
    for version in ['m14','m16']:
        device='cpu' if version=='m14' else 'mps' if torch.backends.mps.is_available() else 'cpu';started=time.monotonic();model=model14(False,None,device).eval() if version=='m14' else model16(weights['checkpoint']);load=time.monotonic()-started;records=[]
        for name in dataset['sources']:
            meta=json.loads(Path(f'outputs/world_model/{name}_m9_ready/manifest.json').read_text());cap=cv2.VideoCapture(meta['source_video'])
            try:
                for task in [t for t in dataset['tasks'] if t['recording_id']==name and t['kind']=='ball']:
                    cap.set(cv2.CAP_PROP_POS_FRAMES,task['frame']);ok,image=cap.read()
                    if not ok:raise ValueError('Source frame missing')
                    cs,timing=infer14(model,image,False,selection14['tiled'],device,selection14['tile_size']) if version=='m14' else infer16(model,image,selection16['detector']['resolution'])
                    records.append({'task_id':task['task_id'],'recording_id':name,'frame':task['frame'],'candidates':cs,**timing});print('BENCHMARK',version,name,task['frame'],round(timing['seconds'],3),flush=True)
            finally:cap.release()
        result['variants'][version]={'device':device,'model_sha256':selection14['checkpoint_sha256'] if version=='m14' else weights['checkpoint_sha256'],'load_seconds':load,'inference_seconds':sum(r['seconds'] for r in records),'frames':len(records),'records':records};Path('outputs/m19/detector_benchmark.json').write_text(json.dumps(result,indent=2)+'\n');del model;gc.collect()
    return result
if __name__=='__main__':run()
