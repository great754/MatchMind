"""Bounded match-disjoint positive-box corpus; unknown absence is never a negative label."""
import json,configparser,hashlib,time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np,cv2
from ml.reid13.remote import member_bytes
from .common import ROOT,save,load,sha
OLD=Path('outputs/ml/soccer_reid_m13')
def build():
 if (ROOT/'dataset.json').exists():raise ValueError('Dataset sealed')
 audit=load('source_audit.json');revision=audit['revision'];train_entries=json.loads((OLD/'SN-Tracking-2023_zip_index.json').read_text());test_entries=load('test_zip_index.json');maps={'train':{e['path']:e for e in train_entries},'test':{e['path']:e for e in test_entries}};sizes={'train':9575523385,'test':audit['archive_size']};metadata={}
 for p in list(Path('outputs/ml/ball_m15/source_metadata').glob('*/gameinfo.ini'))+list((ROOT/'source_metadata').rglob('gameinfo.ini')):
  c=configparser.ConfigParser();c.read(p);metadata[p.parent.name]=(dict(c['Sequence']),'train' if 'ball_m15' in str(p) else 'test')
 plan={'train':{'games':['6','8'],'frames_per_clip':80},'validation':{'games':['4'],'frames_per_clip':36},'test':{'games':['7','11'],'frames_per_clip':100}}
 oldtest=set()
 for m in ['ball_m14','ball_m15']:
  for r in json.loads(Path(f'outputs/ml/{m}/dataset.json').read_text())['samples']:
   if r['split']=='test':oldtest.add(r['sample_id'])
 tasks=[];network=0
 for seq,(meta,archive) in sorted(metadata.items()):
  split=next((s for s,p in plan.items() if meta['gameid'] in p['games']),None)
  if split is None:continue
  if split=='validation' and seq not in [f'SNMOT-{i:03d}' for i in range(60,68)]:continue
  if split=='test' and seq not in [f'SNMOT-{i:03d}' for i in list(range(116,122))+list(range(187,192))]:continue
  base=ROOT/'source_metadata'/archive/seq;base.mkdir(parents=True,exist_ok=True);gtpath=base/'gt/gt.txt';gtpath.parent.mkdir(exist_ok=True);key=f'{archive}/{seq}/gt/gt.txt';url=f'https://huggingface.co/datasets/SoccerNet/SN-Tracking-2023/resolve/{revision}/{archive}.zip'
  if not gtpath.exists():
   existing=OLD/'source_metadata/train'/seq/'gt/gt.txt'
   if archive=='train' and existing.exists():gtpath.write_bytes(existing.read_bytes())
   else:b,n=member_bytes(url,sizes[archive],maps[archive][key]);gtpath.write_bytes(b);network+=n
  gt=np.loadtxt(gtpath,delimiter=',');ball_ids={int(k.split('_')[1]) for k,v in meta.items() if k.startswith('trackletid_') and v.startswith('ball;')};positive=sorted({int(r[0]) for r in gt if int(r[1]) in ball_ids and r[4]>=(3 if split=='train' else .01) and r[5]>=(3 if split=='train' else .01) and r[2]>=0 and r[3]>=0 and f'{seq}:{int(r[0])}' not in oldtest})
  if not positive:continue
  selected=([positive[i] for i in np.unique(np.linspace(0,len(positive)-1,min(len(positive),plan[split]['frames_per_clip']),dtype=int))] if split=='train' else [f for f in range(321,357) if f in positive] if split=='validation' else [f for f in range(301,401) if f in positive])
  for frame in selected:tasks.append((seq,meta,archive,split,frame,gt,ball_ids,str(gtpath),url))
 def fetch(task):
  seq,meta,archive,split,frame,gt,ids,gtpath,url=task;p=ROOT/'frames'/seq/f'{frame:06d}.jpg';p.parent.mkdir(parents=True,exist_ok=True);e=maps[archive][f'{archive}/{seq}/img1/{frame:06d}.jpg']
  if p.exists():b=p.read_bytes();n=0
  else:
   for attempt in range(6):
    try:b,n=member_bytes(url,sizes[archive],e);break
    except (ValueError,TimeoutError) as error:
     if '429' not in str(error) or attempt==5:raise
     time.sleep(min(45,5*2**attempt))
   p.write_bytes(b)
  im=cv2.imdecode(np.frombuffer(b,np.uint8),cv2.IMREAD_COLOR);h,w=im.shape[:2];balls=[]
  for r in gt[(gt[:,0]==frame)&np.isin(gt[:,1],list(ids))]:
   x,y,bw,bh=map(float,r[2:6])
   if bw>0 and bh>0 and x>=0 and y>=0 and x+bw<=w and y+bh<=h:balls.append({'track_id':int(r[1]),'bbox':[x,y,bw,bh],'visibility':'UNVERIFIED_ORGANIZER_BOX','occlusion':'UNKNOWN','height':'UNKNOWN','confidence':'ORGANIZER_ANNOTATION_NOT_VISIBILITY_CERTIFICATE'})
  if not balls:return None,n
  return {'sample_id':f'{seq}:{frame}','recording_id':seq,'match_id':'SN_game'+meta['gameid'],'camera':'main_broadcast','split':split,'frame':frame-1,'timestamp':(frame-1)/25,'fps':25,'width':w,'height':h,'path':str(p),'balls':balls,'source_sha256':hashlib.sha256(b).hexdigest(),'annotation_sha256':sha(gtpath),'provenance':{'revision':revision,'archive':archive,'member':e['path'],'crc32':e['crc']},'visibility':'UNVERIFIED_ORGANIZER_BOX','verified_negative':False},n
 rows=[]
 with ThreadPoolExecutor(max_workers=2) as pool:
  for i,(r,n) in enumerate(pool.map(fetch,tasks)):
   network+=n
   if r:rows.append(r)
   if i%200==0:print('DATA',i,len(tasks),'bytes',network,flush=True)
 sets={s:{r['match_id'] for r in rows if r['split']==s} for s in plan}
 if any(sets[a]&sets[b] for a,b in [('train','validation'),('train','test'),('validation','test')]):raise ValueError('Match leakage')
 if {r['sample_id'] for r in rows if r['split']=='train'}&oldtest:raise ValueError('Historical TEST leakage')
 hashes=[r['source_sha256'] for r in rows]
 if len(set(hashes))!=len(hashes):raise ValueError('Exact duplicate source frames')
 summary={s:{'frames':len(rs:=[r for r in rows if r['split']==s]),'organizer_positive_boxes':sum(len(r['balls']) for r in rs),'games':sorted(sets[s]),'independently_visibility_verified_frames':0,'verified_no_visible_ball_frames':0} for s in plan};save('dataset.json',{'samples':rows,'plan':plan,'summary':summary,'network_bytes':network,'fresh_test_matches':['SN_game7','SN_game11'],'limits':'Organizer positive boxes only; cannot claim3000verified visible positives or1000visibility-labelledTEST. No unknown absence turned into a negative.'});save('protocol.json',{'dataset_sha256':sha(ROOT/'dataset.json'),'train_visibility_status':'organizer labels only, manually reviewed subset reported separately','test':'sealed before any final TEST detector inference; matches7+11 never used in M14/M15 training or VAL','detector':'bounded soccer YOLO11s fine tune, max4epochs early stopping; compare640/960/1280 image inference on identical VAL sample','threshold_grid':[.1,.2,.35,.5,.65],'selection':'VAL F1 then correct center coverage, reject learned components if VAL regresses; M14 default retained','metric_gate':'No authoritative ground coordinates without independently verified ground/metric references'});print('SEALED',summary,'network',network,flush=True)
if __name__=='__main__':build()
