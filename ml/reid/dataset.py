"""Reproducible GT crop extraction with full provenance and match-level separation."""
import argparse,json,hashlib,time,os
from pathlib import Path
from collections import Counter
import cv2,numpy as np
from .audit import ROOT
from src.vision.world.player_appearance import digest

STRATEGIES=['m9_body','full_masked','multi_region']
def perceptual_hash(crop):
 gray=cv2.resize(cv2.cvtColor(crop,cv2.COLOR_BGR2GRAY),(32,32));d=cv2.dct(gray.astype(np.float32))[:8,:8];return ''.join('1' if x>np.median(d) else '0' for x in d.ravel())
def quality(image,box,others):
 x,y,w,h=box;H,W=image.shape[:2];x1=max(0,int(x));y1=max(0,int(y));x2=min(W,int(x+w));y2=min(H,int(y+h));crop=image[y1:y2,x1:x2]
 if not crop.size:return None,None
 gray=cv2.cvtColor(crop,cv2.COLOR_BGR2GRAY);hsv=cv2.cvtColor(crop,cv2.COLOR_BGR2HSV);sharp=float(cv2.Laplacian(gray,cv2.CV_64F).var());sat=float(np.mean(hsv[:,:,1]));brightness=float(gray.mean());contrast=float(gray.std());foreground=float(np.mean(~((hsv[:,:,0]>=25)&(hsv[:,:,0]<=95)&(hsv[:,:,1]>50))));overlap=0.
 for a,b,c,d in others:
  intersection=max(0,min(x+w,a+c)-max(x,a))*max(0,min(y+h,b+d)-max(y,b));overlap=max(overlap,intersection/max(w*h,1))
 clipped=1-((x2-x1)*(y2-y1))/max(w*h,1);score=float(np.clip(min(1,h/70)*min(1,sharp/100)*min(1,contrast/35)*(1-min(overlap,.95))*(1-clipped),0,1));q={'width':float(w),'height':float(h),'area':float(w*h),'sharpness':sharp,'foreground_fraction':foreground,'occlusion_proxy':overlap,'annotation_visibility':None,'brightness':brightness,'contrast':contrast,'saturation':sat,'edge_clipping_fraction':float(np.clip(clipped,0,1)),'boundary_truncated':bool(x<1 or y<1 or x+w>W-1 or y+h>H-1),'score':score,'size_bucket':'small' if h<45 else 'medium' if h<85 else 'large','view_direction':'unknown'}
 return q,crop

def crop_regions(image,strategy):
 h=image.shape[0]
 if strategy=='m9_body':return [image[int(.18*h):]]
 masked=image.copy();masked[:int(.18*h)]=127
 if strategy=='full_masked':return [masked]
 if strategy=='multi_region':return [masked,image[int(.18*h):int(.60*h)],image[int(.55*h):]]
 raise ValueError('Unknown crop strategy')

def extract_match(n,audit,source,source_provenance,output,step=50):
 out=Path(output)/n;out.mkdir(parents=True,exist_ok=True);done=out/'samples.json';annotation=ROOT/'annotations/mot'/f'{n}.txt';allowed=audit['allowed_players'][n]
 if done.exists():return json.loads(done.read_text())
 raw=np.loadtxt(annotation,delimiter=',');base=int(raw[:,0].min());raw[:,0]-=base;cap=cv2.VideoCapture(str(source));fps=cap.get(cv2.CAP_PROP_FPS);frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH));height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
 if abs(fps-25)>.01 or frames!=int(raw[:,0].max())+1:raise ValueError('Video/annotation frame clock mismatch')
 sha=digest(source)
 official=ROOT/'official_video_metadata.json'
 if official.exists() and json.loads(official.read_text())[n]['sha256']!=sha:raise ValueError('Video differs from official paired annotation clip')
 rows=[];skipped=Counter();last_hash={};last_time={};started=time.perf_counter()
 for i in range(frames):
  if i%step:
   if not cap.grab():raise ValueError('Incomplete video')
   continue
  ok,image=cap.read()
  if not ok:raise ValueError('Incomplete sampled frame')
  boxes=raw[raw[:,0]==i]
  for r in boxes:
   tid=str(int(r[1]))
   if tid not in allowed:continue
   bbox=r[2:6];q,crop=quality(image,bbox,[v[2:6] for v in boxes if v[1]!=r[1]])
   if q is None or q['width']<8 or q['height']<24 or q['sharpness']<12 or q['foreground_fraction']<.12 or q['edge_clipping_fraction']>.15 or q['occlusion_proxy']>.45 or q['brightness']<15 or q['contrast']<8:skipped['quality']+=1;continue
   ph=perceptual_hash(crop)
   if tid in last_hash and sum(a!=b for a,b in zip(ph,last_hash[tid]))<3:skipped['near_duplicate']+=1;continue
   if tid in last_time and i/fps-last_time[tid]<step/fps-1e-6:raise ValueError('Temporal sampler violated')
   last_hash[tid]=ph;last_time[tid]=i/fps;filename=f'{i:06d}_{tid}.jpg';path=out/filename
   # Never retain facial pixels in dataset assets, including the full-layout strategy.
   crop[:int(.18*crop.shape[0])]=127;cv2.imwrite(str(path),crop,[cv2.IMWRITE_JPEG_QUALITY,95]);p=allowed[tid];rows.append({'sample_id':f'{n}:{i}:{tid}','match_id':n,'recording_id':n+'_clip','frame':i,'timestamp':i/fps,'fps':fps,'player_identity':p['annotation_identity'],'mot_ground_truth_id':int(tid),'registered_player_id':p['roster_player_id'],'team_identity':p['team_id'],'source_bbox':bbox.tolist(),'quality':q,'source_sha256':sha,'source_provenance':source_provenance,'annotation_sha256':audit['annotations'][n]['annotation_sha256'],'annotation_row_provenance':f'{n}.txt frame={int(r[0])+base}, id={tid}; original frame base={base}','identity_scope':'organizer persistent MOT ground truth within match clip; not generated tracker output','team_mapping_provenance':p['mapping_source'],'crop_path':str(path),'crop_sha256':digest(path),'perceptual_hash':ph,'face_policy':'top18percent masked before saving','split':next(s for s,v in audit['groups'].items() if n in v)})
 cap.release();done.write_text(json.dumps(rows));(out/'extraction.json').write_text(json.dumps({'samples':len(rows),'skipped':dict(skipped),'source_sha256':sha,'fps':fps,'frames':frames,'width':width,'height':height,'wall_seconds':time.perf_counter()-started,'min_sample_separation_seconds':step/fps},indent=2));print(n,'samples',len(rows),'skips',dict(skipped),flush=True);return rows

def build(download=True):
 audit=json.loads((ROOT/'identity_audit.json').read_text());all_rows=[]
 if download:
  from dotenv import load_dotenv
  from azure.identity import AzureCliCredential
  from azure.storage.blob import ContainerClient
  from azure.core import MatchConditions
  load_dotenv('.env');container=ContainerClient(account_url=os.environ['MATCHMIND_BLOB_ACCOUNT_URL'],container_name=os.environ['MATCHMIND_MEDIA_BLOB_CONTAINER'],credential=AzureCliCredential(),connection_timeout=10,read_timeout=60,retry_total=2)
 for n,players in audit['allowed_players'].items():
  if not players:continue
  done=ROOT/'dataset'/n/'samples.json'
  if done.exists():all_rows+=json.loads(done.read_text());continue
  temp=ROOT/'temporary_video';temp.mkdir(exist_ok=True);video=temp/(n+'.mp4');blob=container.get_blob_client(f'mot/clips/{n}.mp4');prop=blob.get_blob_properties();print(n,'download bytes',prop.size,flush=True)
  with video.open('xb') as f:blob.download_blob(etag=prop.etag,match_condition=MatchConditions.IfNotModified,max_concurrency=4).readinto(f)
  try:all_rows+=extract_match(n,audit,video,{'provider':'existing_private_azure_media','blob':f'mot/clips/{n}.mp4','etag':prop.etag,'size':prop.size},ROOT/'dataset')
  finally:video.unlink(missing_ok=True)
 manifest={'version':'soccer-reid-dataset-v1','split_manifest_sha256':digest('configs/ml/soccer-reid-v1-split.json'),'samples':all_rows,'min_separation_seconds':2,'provenance':'Original organizer sequence identity GT; inferred team mapping kept separate; no M9 evaluation videos'};path=ROOT/'dataset_manifest.json';path.write_text(json.dumps(manifest));summary={s:{'samples':sum(r['split']==s for r in all_rows),'identities':len({r['player_identity'] for r in all_rows if r['split']==s}),'matches':sorted({r['match_id'] for r in all_rows if r['split']==s})} for s in ['train','validation','test']};(ROOT/'dataset_summary.json').write_text(json.dumps({'splits':summary,'manifest_sha256':digest(path)},indent=2));print(summary,flush=True)
 return manifest
if __name__=='__main__':build()
