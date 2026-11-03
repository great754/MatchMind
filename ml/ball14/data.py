"""Original image-space ball GT; pitch-coordinate trajectories are never detector labels."""
import json,configparser,hashlib
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import cv2,numpy as np
from ml.reid13.remote import member_bytes
from .common import ROOT,save,load,sha
SOURCE=Path('outputs/ml/soccer_reid_m13')
PLAN={'train':{'game':'6','clips':['SNMOT-097','SNMOT-098','SNMOT-099','SNMOT-100'],'frames':list(range(13,738,24))},'validation':{'game':'4','clips':['SNMOT-060','SNMOT-061'],'frames':list(range(200,280))},'test':{'game':'9','clips':['SNMOT-151','SNMOT-152'],'frames':list(range(300,400))}}

def annotations(seq,frame):
 p=SOURCE/'source_metadata/train'/seq;meta=configparser.ConfigParser();meta.read(p/'gameinfo.ini');s=meta['Sequence'];ids={int(k.split('_')[1]) for k,v in s.items() if k.startswith('trackletid_') and v.startswith('ball;')};raw=np.loadtxt(p/'gt/gt.txt',delimiter=',');rs=raw[(raw[:,0]==frame)&np.isin(raw[:,1],list(ids))];return [{'track_id':int(r[1]),'bbox':[float(x) for x in r[2:6]],'visibility':'NOT_ANNOTATED','identity_scope':'organizer GT ball object within this clip, not verified in-play ball'} for r in rs if r[4]>0 and r[5]>0],s['gameid'],sha(p/'gt/gt.txt')

def build():
 if (ROOT/'dataset.json').exists():raise ValueError('Dataset sealed; use a new root/version')
 inv=json.loads((SOURCE/'soccernet_inventory.json').read_text());rev=inv['SN-Tracking-2023']['sha'];tree=json.loads((SOURCE/'SN-Tracking-2023_tree.json').read_text());size=next(x['size'] for x in tree if x['path']=='train.zip');url=f'https://huggingface.co/datasets/SoccerNet/SN-Tracking-2023/resolve/{rev}/train.zip';entries={e['path']:e for e in json.loads((SOURCE/'SN-Tracking-2023_zip_index.json').read_text())};samples=[];network=0
 for split,p in PLAN.items():
  for seq in p['clips']:
   directory=ROOT/'frames'/seq;directory.mkdir(parents=True,exist_ok=True)
   def fetch(frame):
    name=f'train/{seq}/img1/{frame:06d}.jpg';e=entries[name];path=directory/f'{frame:06d}.jpg'
    if path.exists():b=path.read_bytes();n=0
    else:b,n=member_bytes(url,size,e,budget=2000000);path.write_bytes(b)
    balls,game,h=annotations(seq,frame)
    if game!=p['game']:raise ValueError('Match clock/provenance mismatch')
    image=cv2.imdecode(np.frombuffer(b,np.uint8),cv2.IMREAD_COLOR)
    if image is None or image.shape[:2]!=(1080,1920):raise ValueError('Invalid source image')
    for ball in balls:
     x,y,w,hb=ball['bbox'];crop=image[max(0,int(y)):min(1080,int(y+hb)),max(0,int(x)):min(1920,int(x+w))];ball['sharpness_proxy']=float(cv2.Laplacian(cv2.cvtColor(crop,cv2.COLOR_BGR2GRAY),cv2.CV_64F).var()) if crop.size else None
    return {'sample_id':seq+':'+str(frame),'split':split,'match_id':'SN_game'+game,'recording_id':seq,'frame':frame-1,'timestamp':(frame-1)/25,'fps':25,'path':str(path),'source_sha256':hashlib.sha256(b).hexdigest(),'width':1920,'height':1080,'balls':balls,'annotation_sha256':h,'provenance':{'source':'original SoccerNet Tracking GT','revision':rev,'archive_member':name,'crc32':e['crc']},'no_annotation_is_not_verified_invisibility':True},n
   with ThreadPoolExecutor(max_workers=8) as pool:
    for r,n in pool.map(fetch,p['frames']):samples.append(r);network+=n
   print(split,seq,len(p['frames']),'range_bytes',network,flush=True)
 sets={s:{r['match_id'] for r in samples if r['split']==s} for s in PLAN}
 if sets['train']&sets['validation'] or sets['train']&sets['test'] or sets['validation']&sets['test']:raise ValueError('Match leakage')
 summary={s:{'frames':len(a:=[r for r in samples if r['split']==s]),'annotated_ball_boxes':sum(len(r['balls']) for r in a),'frames_without_ball_annotation':sum(not r['balls'] for r in a),'matches':sorted(sets[s]),'clips':PLAN[s]['clips']} for s in PLAN};save('dataset.json',{'samples':samples,'plan':PLAN,'source':'public SoccerNet Tracking research-use data; same original GT, no pitch NPZ converted to image labels','annotation_quality':'organizer interpolated/tracked GT; visibility/airborne/occlusion states absent; may include multiple or reserve balls','final_demo_videos_never_train':['test_match','team_goal','southampton']});save('dataset_summary.json',{'splits':summary,'network_bytes':network,'full_archive_downloaded':False,'match_disjoint':True,'source_terms':'SoccerNet FAQ permits research training, not intended commercial use; artifacts remain local','source_uri':'https://www.soccer-net.org/faq'});print(summary,flush=True)
if __name__=='__main__':build()
