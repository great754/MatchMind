"""Fresh sequence-disjoint image GT and trajectories; original M14 TEST is historical only."""
import json,configparser,hashlib,concurrent.futures
from pathlib import Path
import numpy as np,cv2
from ml.reid13.remote import member_bytes
from .setup import remote
from .common import ROOT,save,load,sha
PLAN={'train':{'game':'6','clips':['SNMOT-109','SNMOT-110'],'frames':list(range(101,221))},'validation':{'game':'4','clips':['SNMOT-070','SNMOT-071'],'frames':list(range(201,281))},'test':{'game':'9','clips':['SNMOT-159','SNMOT-160'],'frames':list(range(401,501))}}
def build():
 if (ROOT/'dataset.json').exists():raise ValueError('Dataset sealed')
 url,size,entries,rev=remote();samples=[];network=0
 for split,plan in PLAN.items():
  for seq in plan['clips']:
   dest=ROOT/'source_metadata'/seq;dest.mkdir(exist_ok=True,parents=True)
   for suffix in ['gt/gt.txt','seqinfo.ini']:
    p=dest/suffix;p.parent.mkdir(exist_ok=True)
    if not p.exists():b,n=member_bytes(url,size,entries[f'train/{seq}/{suffix}']);p.write_bytes(b);network+=n
   meta=configparser.ConfigParser();meta.read(dest/'gameinfo.ini');meta=meta['Sequence'];assert meta['gameid']==plan['game'];ids={int(k.split('_')[1]) for k,v in meta.items() if k.startswith('trackletid_') and v.startswith('ball;')};gt=np.loadtxt(dest/'gt/gt.txt',delimiter=',');frame_dir=ROOT/'frames'/seq;frame_dir.mkdir(exist_ok=True,parents=True)
   def fetch(frame):
    p=frame_dir/f'{frame:06d}.jpg';entry=entries[f'train/{seq}/img1/{frame:06d}.jpg']
    if p.exists():b=p.read_bytes();n=0
    else:b,n=member_bytes(url,size,entry);p.write_bytes(b)
    im=cv2.imdecode(np.frombuffer(b,np.uint8),cv2.IMREAD_COLOR);h,w=im.shape[:2];rs=gt[(gt[:,0]==frame)&np.isin(gt[:,1],list(ids))];balls=[{'track_id':int(r[1]),'bbox':r[2:6].tolist(),'visibility':'NOT_ANNOTATED'} for r in rs if r[4]>0 and r[5]>0];people=[{'bbox':r[2:6].tolist()} for r in gt[(gt[:,0]==frame)&~np.isin(gt[:,1],list(ids))]]
    return {'sample_id':f'{seq}:{frame}','split':split,'match_id':'SN_game'+plan['game'],'recording_id':seq,'frame':frame-1,'timestamp':(frame-1)/25,'fps':25,'path':str(p),'width':w,'height':h,'balls':balls,'people':people,'source_sha256':hashlib.sha256(b).hexdigest(),'annotation_sha256':sha(dest/'gt/gt.txt'),'clipstart_ms':meta['clipstart'],'provenance':{'source':'original organizer image GT','revision':rev,'archive_member':entry['path'],'crc32':entry['crc']}},n
   with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
    for r,n in pool.map(fetch,plan['frames']):samples.append(r);network+=n
   print('DATA',split,seq,'network',network,flush=True)
 old=json.loads(Path('outputs/ml/ball_m14/dataset.json').read_text());assert not {r['sample_id'] for r in samples}&{r['sample_id'] for r in old['samples']}
 save('dataset.json',{'samples':samples,'plan':PLAN,'network_bytes':network,'visibility_airborne_occlusion_gt':'unavailable','split':'match disjoint within M15; games same as M14 but new nonadjacent clips/frames, no old exposed TEST tuning','demo_videos':'never used for training or selection'})
 save('protocol_seal.json',{'dataset_sha256':sha(ROOT/'dataset.json'),'plan':PLAN,'manual_demo_annotations_sha256':sha(ROOT/'demo_annotations.json') if (ROOT/'demo_annotations.json').exists() else None,'detector_grid':{'models':['generic_yolo11n','soccer_yolo11n'],'tile':[320,640],'threshold':[.1,.2,.35,.5,.65]},'training':'12 epochs batch8 crop320 imgsz640, bounded; validation only selection','motion':'GRU24 vs CV/CA, image-normalized deltas/confidence/dt; horizons1,3,5,10','association':'interpretable vs logistic, labeled only by organizer GT IoU, validation selection','selection':'maximum validation F1, then precision; components disabled unless validation improves; TEST once after lock','old_m14_test':'historical reproduction from immutable cache only'})
 print('SEALED',len(samples),flush=True)
if __name__=='__main__':build()
