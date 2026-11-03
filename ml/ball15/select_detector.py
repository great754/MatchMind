"""Model/tiling selection reads only the locked training/validation splits."""
import json,time
from pathlib import Path
import cv2,numpy as np
from sklearn.isotonic import IsotonicRegression
from ml.ball14.evaluate import detection_metrics,iou
from .detector import make_model,infer
from .common import ROOT,load,save,sha

def predictions(model,rows,name,tile,generic):
 path=ROOT/'cache'/f'{name}.json';path.parent.mkdir(exist_ok=True)
 if path.exists():
  out=json.loads(path.read_text());assert [p['sample_id'] for p in out]==[r['sample_id'] for r in rows];return out
 out=[]
 for i,row in enumerate(rows):
  cs,timing=infer(model,cv2.imread(row['path']),tile,generic=generic);out.append({'sample_id':row['sample_id'],'candidates':cs,**timing})
  if i%40==0:print('INFER',name,i,flush=True)
 path.write_text(json.dumps(out));return out

def run():
 if (ROOT/'detector_lock.json').exists():raise ValueError('Detector already sealed')
 rows=[r for r in load('dataset.json')['samples'] if r['split']=='validation'][::4];tr=load('detector_training.json');reports=[]
 for kind,path in [('generic',str(ROOT/'yolo11n.pt')),('soccer',tr['best'])]:
  model=make_model(path)
  for tile in [320,640]:
   ps=predictions(model,rows,f'val_{kind}_{tile}',tile,kind=='generic')
   for threshold in [.1,.2,.35,.5,.65]:reports.append({'kind':kind,'checkpoint':path,'checkpoint_sha256':sha(path),'tile':tile,'threshold':threshold,'metrics':detection_metrics(rows,[p['candidates'] for p in ps],threshold,.25),'inference_seconds':sum(p['seconds'] for p in ps)})
 chosen=max(reports,key=lambda c:(c['metrics']['f1'],c['metrics']['precision'],-c['inference_seconds']));save('detector_lock.json',{'selected':chosen,'validation_candidates':reports,'dataset_sha256':sha(ROOT/'dataset.json'),'demo_sha256':load('demo_seal.json')['sha256'],'final_test_seen':False});print('DETECTOR_LOCK',chosen,flush=True)
 # Calibration and verifier training caches, kept distinct from final TEST.
 model=make_model(chosen['checkpoint'])
 for split in ['train','validation']:
  rs=[r for r in load('dataset.json')['samples'] if r['split']==split];predictions(model,rs,split+'_selected',chosen['tile'],chosen['kind']=='generic')
if __name__=='__main__':run()
