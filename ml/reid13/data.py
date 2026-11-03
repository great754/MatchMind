"""Multi-source GT crop adapter with provenance, match guards and sealed fresh evaluation."""
import argparse,configparser,hashlib,json,re,io
from pathlib import Path
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import cv2,numpy as np
from ml.reid.dataset import quality,perceptual_hash
from .common import ROOT,M10,load,save,sha
from .remote import member_bytes
# Audited game/kit scopes: game4 left and game9 left share the white-blue club.
# Reserve only game4 right (green) for VAL; game9 entirely TEST. No shared kit scope.
PLAN={'train':{'game':'6','sequences':[f'SNMOT-{i:03d}' for i in range(97,105)],'sides':['left','right']},'validation':{'game':'4','sequences':['SNMOT-060','SNMOT-061','SNMOT-062'],'sides':['right']},'test':{'game':'9','sequences':['SNMOT-151','SNMOT-152','SNMOT-153'],'sides':['left','right']}}

def metadata(seq):
 c=configparser.ConfigParser();c.read(ROOT/'source_metadata/train'/seq/'gameinfo.ini');return c['Sequence']
def allowed(meta,sides):
 result={}
 for key,value in meta.items():
  if not key.startswith('trackletid_'):continue
  role,number=[p.strip() for p in value.split(';')]
  if not role.startswith('player team ') or not number.isdecimal():continue
  side=role.rsplit(' ',1)[-1]
  if side in sides:result[int(key.split('_')[1])]={'side':side,'number':number,'identity':f'SN:game{meta["gameid"]}:{side}:{number}'}
 return result

def download_extract(split):
 plan=PLAN[split];repo='SN-Tracking-2023';revision=load('soccernet_inventory.json')[repo]['sha'];size=next(x['size'] for x in load(repo+'_tree.json') if x['path']=='train.zip');url=f'https://huggingface.co/datasets/SoccerNet/{repo}/resolve/{revision}/train.zip';entries={e['path']:e for e in load(repo+'_zip_index.json')};samples=[];totalbytes=0
 for seq in plan['sequences']:
  done=ROOT/'dataset'/seq/'samples.json'
  if done.exists():samples+=json.loads(done.read_text());continue
  meta=metadata(seq)
  if meta['gameid']!=plan['game'] or not meta['gametimestart'].startswith('1 -'):raise ValueError('Game/half scope differs from audited plan')
  people=allowed(meta,plan['sides']);gtpath=ROOT/'source_metadata/train'/seq/'gt/gt.txt';gt=np.loadtxt(gtpath,delimiter=',');seqpath=ROOT/'source_metadata/train'/seq/'seqinfo.ini';c=configparser.ConfigParser();c.read(seqpath);fps=int(c['Sequence']['framerate']);length=int(c['Sequence']['seqlength']);frames=range(1,length+1,12);directory=done.parent;directory.mkdir(parents=True,exist_ok=True);last={};out=[];skips=Counter()
  def fetch(frame):
   path=f'train/{seq}/img1/{frame:06d}.jpg';e=entries[path];data,n=member_bytes(url,size,e,budget=2000000);return frame,data,n,e
  # Frames discarded after verified crop extraction; never unpack arbitrary ZIP paths.
  with ThreadPoolExecutor(max_workers=8) as pool:
   for frame,data,n,e in pool.map(fetch,frames):
    totalbytes+=n;image=cv2.imdecode(np.frombuffer(data,np.uint8),cv2.IMREAD_COLOR)
    if image is None or image.shape[:2]!=(1080,1920):raise ValueError('Unexpected original frame')
    boxes=gt[gt[:,0]==frame]
    for row in boxes:
     tid=int(row[1])
     if tid not in people:continue
     q,crop=quality(image,row[2:6],[r[2:6] for r in boxes if r[1]!=tid and r[4]>8 and r[5]>24])
     if q is None or q['width']<8 or q['height']<24 or q['sharpness']<12 or q['edge_clipping_fraction']>.15 or q['occlusion_proxy']>.45 or q['brightness']<15 or q['contrast']<8:skips['quality']+=1;continue
     ph=perceptual_hash(crop)
     if tid in last and sum(a!=b for a,b in zip(ph,last[tid]))<3:skips['near_duplicate']+=1;continue
     last[tid]=ph;p=people[tid];crop[:int(.18*crop.shape[0])]=127;path=directory/f'{frame:06d}_{tid}.jpg';cv2.imwrite(str(path),crop,[cv2.IMWRITE_JPEG_QUALITY,95]);sid=f'SN:{seq}:{frame}:{tid}';out.append({'sample_id':sid,'source':'soccernet_tracking','match_id':'SN_game'+meta['gameid'],'recording_id':seq,'frame':frame,'timestamp':(frame-1)/fps,'absolute_game_seconds':int(meta['clipstart'])/1000+(frame-1)/fps,'fps':fps,'player_identity':p['identity'],'registered_player_id':None,'team_identity':f'SN_game{meta["gameid"]}_{p["side"]}','mot_ground_truth_id':tid,'number_metadata_for_leakage_only':p['number'],'source_bbox':row[2:6].tolist(),'quality':q,'crop_path':str(path),'crop_sha256':sha(path),'perceptual_hash':ph,'annotation_sha256':sha(gtpath),'metadata_sha256':sha(ROOT/'source_metadata/train'/seq/'gameinfo.ini'),'source_sha256':hashlib.sha256(data).hexdigest(),'source_provenance':{'repository':f'SoccerNet/{repo}','revision':revision,'archive':'train.zip','member':e['path'],'crc32':e['crc'],'raw_frame_sha256':hashlib.sha256(data).hexdigest()},'identity_scope':'original organizer GT within clip; numeric organizer team+jersey metadata guards same-match aliases, never OCR or arbitrary tracker IDs','face_policy':'top18percent masked before saving','split':split})
  done.write_text(json.dumps(out));(directory/'extraction.json').write_text(json.dumps({'samples':len(out),'skips':dict(skips),'annotation_sha256':sha(gtpath),'original_frame_base':1,'frame_step':12,'fps':fps,'source_member_crc_checked':True}));samples+=out;print(split,seq,'crops',len(out),'downloaded_bytes',totalbytes,flush=True)
 save(f'{split}_source_samples.json',samples);save(f'{split}_download.json',{'bytes_this_execution':totalbytes,'full_archive_downloaded':False});return samples

def validate(rows):
 sets={s:{'matches':set(),'people':set(),'teams':set(),'crops':set(),'frames':set()} for s in ['train','validation','test']}
 for r in rows:
  s=sets[r['split']];s['matches'].add(r['match_id']);s['people'].add(r['player_identity']);s['teams'].add(r['team_identity']);s['crops'].add(r['crop_sha256']);s['frames'].add(r['source_sha256'])
 for a,b in [('train','validation'),('train','test'),('validation','test')]:
  for key in sets[a]:
   if sets[a][key]&sets[b][key]:raise ValueError(f'Leakage in {key}: {a}/{b}')
 return {s:{k:len(v) for k,v in t.items()} for s,t in sets.items()}

def build():
 if (ROOT/'dataset.json').exists():raise ValueError('Dataset sealed; new version required')
 rows=[]
 for split in PLAN:rows+=download_extract(split)
 old=[{**r,'source':'soccertrack_v2'} for r in json.loads((M10/'curated_manifest.json').read_text())['samples'] if r['split']=='train']
 rows+=old;counts=Counter(r['player_identity'] for r in rows);rows=[r for r in rows if counts[r['player_identity']]>=16]
 # One anchor sequence per same-match identity; no inferred cross-clip positive labels.
 anchors={}
 pairs=Counter((r['player_identity'],r['recording_id']) for r in rows)
 for identity in sorted({r['player_identity'] for r in rows}):
  options=[(count,recording) for (person,recording),count in pairs.items() if person==identity];anchors[identity]=sorted(options,key=lambda x:(-x[0],x[1]))[0][1]
 rows=[r for r in rows if r['recording_id']==anchors[r['player_identity']]]
 counts=Counter(r['player_identity'] for r in rows);rows=[r for r in rows if counts[r['player_identity']]>=16]
 integrity=validate(rows)
 for r in rows:
  if sha(r['crop_path'])!=r['crop_sha256']:raise ValueError('Crop hash mismatch')
 save('dataset.json',{'version':'soccer-reid-v2-dataset','samples':rows,'plan':PLAN,'m10_train_manifest_sha256':sha(M10/'curated_manifest.json'),'identity_policy':'one anchor clip per numeric same-game/team alias; within-clip GT positives only; unknown aliases/goalkeepers/officials excluded','global_physical_identity_registry_available':False,'cross_game_guard':'match separation plus visually audited nonrepeated club/kit scopes; global transfers/rosters not independently certified'})
 save('dataset_summary.json',{'splits':{s:{'crops':len(t:=[r for r in rows if r['split']==s]),'annotation_identities':len({r['player_identity'] for r in t}),'sources':dict(Counter(r['source'] for r in t)),'matches':sorted({r['match_id'] for r in t})} for s in PLAN},'integrity':integrity,'new_test_never_in_m10_m11':True,'quality_audit':'Original non-green HSV fraction retained as metadata but excluded as a filter: green uniforms were mistaken for grass. Applied uniformly to all new-source splits before any training or model evaluation; original provisional extraction archived.', 'sampling':'0.48s between original SN observations; pHash Hamming>=3 vs previous retained same-track crop; one anchor clip per identity','limitations':'Global physical-player registry absent. Match/annotation-identity/source hashes are checked; club/kit overlap audited conservatively, not proof against unobserved transfers. No broad global identity-disjoint claim.'});print(load('dataset_summary.json'),flush=True)
if __name__=='__main__':build()
