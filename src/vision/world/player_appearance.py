"""Hash-cached, head-excluded OSNet + foreground kit evidence. No face recognition."""
import hashlib,json,time
from pathlib import Path
import cv2,numpy as np

MODEL_SHA256='cf55163d78fc44c62c82f85ab62d39f10438679b5abe8c698ae08cfa84aa6e18'
MODEL_URL='https://drive.google.com/uc?export=download&id=1Kkx2zW89jq_NETu4u42CFZTMVD5Hwm6e'
FEATURE_VERSION='m9-body-osnet-msmt17-foreground-v1'
def digest(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for chunk in iter(lambda:f.read(1048576),b''):h.update(chunk)
 return h.hexdigest()
def unit(v):
 v=np.asarray(v,float);return v/max(float(np.linalg.norm(v)),1e-12)
def cosine(a,b):return float(np.clip(np.dot(unit(a),unit(b)),0,1)) if a is not None and b is not None else None

def foreground(image,box):
 x,y,w,h=box;ih,iw=image.shape[:2];x1=max(0,int(x+.18*w));x2=min(iw,int(x+.82*w));y1=max(0,int(y+.20*h));y2=min(ih,int(y+.53*h));crop=image[y1:y2,x1:x2]
 if crop.size==0 or min(crop.shape[:2])<3:return {'quality':0.,'reason':'TORSO_TOO_SMALL','torso_bbox':[x1,y1,0,0]},None
 lab=cv2.cvtColor(crop,cv2.COLOR_BGR2LAB);hsv=cv2.cvtColor(crop,cv2.COLOR_BGR2HSV)
 ring=image[max(0,int(y+h)):min(ih,int(y+1.20*h)+1),max(0,int(x-.4*w)):min(iw,int(x+1.4*w))];background=None
 if ring.size:
  rh=cv2.cvtColor(ring,cv2.COLOR_BGR2HSV);grass=(rh[:,:,0]>25)&(rh[:,:,0]<95)&(rh[:,:,1]>40)
  if grass.sum()>=6:background=np.median(cv2.cvtColor(ring,cv2.COLOR_BGR2LAB)[grass],axis=0)
 # Retain dark and achromatic kit pixels. Saturation is evidence, not an exclusion.
 mask=hsv[:,:,2]>12
 if background is not None:mask &= np.linalg.norm(lab.astype(float)-background,axis=2)>20
 if mask.sum()<6:return {'quality':0.,'reason':'INSUFFICIENT_FOREGROUND','torso_bbox':[x1,y1,x2-x1,y2-y1]},mask
 values=lab[mask];col=np.median(values,axis=0);purity=float(np.mean(np.linalg.norm(values.astype(float)-col,axis=1)<42));sharp=float(cv2.Laplacian(cv2.cvtColor(crop,cv2.COLOR_BGR2GRAY),cv2.CV_64F).var());quality=float(np.clip(mask.mean()*purity*min(1,h/55)*min(1,.35+sharp/80),0,1));hist=cv2.calcHist([hsv],[0,1,2],mask.astype(np.uint8),[12,3,3],[0,180,0,256,0,256]).ravel()
 return {'quality':quality,'reason':'DARK_AND_ACHROMATIC_FOREGROUND_RETAINED','torso_bbox':[x1,y1,x2-x1,y2-y1],'lab':col.tolist(),'lab_quantiles':np.quantile(values,[.2,.5,.8],axis=0).tolist(),'hsv':np.median(hsv[mask],axis=0).tolist(),'foreground_fraction':float(mask.mean()),'pixel_count':int(mask.sum()),'sharpness':sharp,'background_lab':background.tolist() if background is not None else None,'histogram':unit(hist).tolist()},mask

def raw_people(row):
 people=[]
 for e in row['entities']:
  if e['visibility']!='visible':continue
  people.append({'track_id':e['track_id'],'bbox':e['image_bbox'],'foot':e['image_foot'],'pitch':e['current_pitch'],'projection':e.get('projection_camera'),'detection_confidence':e['detection_confidence'],'legacy_team':e['team'],'legacy_entity_id':e['entity_id']})
 for u in row['unassigned']:
  t=u['observation'];people.append({'track_id':t['track_id'],'bbox':t['bbox'],'foot':list((t['bbox'][0]+t['bbox'][2]/2,t['bbox'][1]+t['bbox'][3])),'pitch':t['pitch'],'projection':t.get('projection_camera'),'detection_confidence':t['detection_confidence'],'legacy_team':t['team'],'legacy_entity_id':None})
 if len({p['track_id'] for p in people})!=len(people):raise ValueError('Duplicate raw temporary track')
 return people

class BodyEncoder:
 def __init__(self,checkpoint):
  if digest(checkpoint)!=MODEL_SHA256:raise ValueError('OSNet checkpoint hash mismatch')
  import torch
  from .vendor.osnet import osnet_x0_25
  self.torch=torch;torch.set_num_threads(4);self.model=osnet_x0_25(num_classes=4101,pretrained=False);self.model.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=True),strict=True);self.model.eval()
 def encode(self,crops):
  torch=self.torch;a=np.stack([cv2.resize(cv2.cvtColor(c,cv2.COLOR_BGR2RGB),(128,256)).transpose(2,0,1) for c in crops]).astype(np.float32)/255;a=(a-np.array([.485,.456,.406],np.float32)[None,:,None,None])/np.array([.229,.224,.225],np.float32)[None,:,None,None]
  with torch.inference_mode():v=self.model(torch.from_numpy(a)).numpy()
  return [unit(x).astype(np.float32) for x in v]

def extract(video,rows,manifest,cache,checkpoint):
 cache=Path(cache);cache.mkdir(parents=True,exist_ok=True);key={'source_sha256':manifest['video_sha256'],'baseline_sha256':manifest['frames_sha256'],'model_sha256':MODEL_SHA256,'feature_version':FEATURE_VERSION};meta=cache/'manifest.json'
 if meta.exists():
  m=json.loads(meta.read_text())
  if m['key']!=key or digest(cache/'features.json')!=m['features_sha256'] or digest(cache/'embeddings.npz')!=m['embeddings_sha256']:raise ValueError('Appearance cache source/model/hash mismatch')
  z=np.load(cache/'embeddings.npz');return json.loads((cache/'features.json').read_text()),{k:z[k] for k in z.files},{'cached_extraction_timing':m['timing'],'foreground_features_seconds':0.,'appearance_embedding_seconds':0.,'extraction_wall_seconds':0.,'cache_hit':True}
 encoder=BodyEncoder(checkpoint);cap=cv2.VideoCapture(str(video));features=[];embeddings={};batch=[];keys=[];feature_time=embedding_time=0.;started=time.perf_counter()
 def flush():
  nonlocal embedding_time
  if not batch:return
  t=time.perf_counter();vectors=encoder.encode(batch);embedding_time+=time.perf_counter()-t
  embeddings.update(zip(keys,vectors));batch.clear();keys.clear()
 for index,row in enumerate(rows):
  ok,image=cap.read()
  if not ok:raise ValueError('Incomplete source video')
  tick=time.perf_counter();people=raw_people(row)
  for p in people:
   f,mask=foreground(image,p['bbox']);p.update(feature=f,observation_id=f'{index}:{p["track_id"]}',frame=index,epoch=row['camera']['epoch']);x,y,w,h=p['bbox'];body=image[max(0,int(y+.18*h)):min(image.shape[0],int(y+h)),max(0,int(x)):min(image.shape[1],int(x+w))]
   if index%3==0 and f['quality']>=.12 and h>=24 and body.size and body.shape[1]>=7:batch.append(body.copy());keys.append(p['observation_id']);p['embedding_key']=p['observation_id']
  feature_time+=time.perf_counter()-tick;features.append(people)
  if len(batch)>=48:flush()
 flush();cap.release();timing={'foreground_features_seconds':feature_time,'appearance_embedding_seconds':embedding_time,'extraction_wall_seconds':time.perf_counter()-started,'cache_hit':False};(cache/'features.json').write_text(json.dumps(features));np.savez_compressed(cache/'embeddings.npz',**embeddings);meta.write_text(json.dumps({'key':key,'features_sha256':digest(cache/'features.json'),'embeddings_sha256':digest(cache/'embeddings.npz'),'timing':timing,'embedded_observations':len(embeddings)},indent=2));return features,embeddings,timing
