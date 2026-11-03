"""Tracklets from frozen annotations; no tracker pseudo-labels or fresh TEST tuning."""
import json,math
from collections import Counter
from pathlib import Path
import numpy as np
from ml.reid.protocol import validate_splits
from .common import ROOT,M10,save,sha

def diverse(rows,n=12):
 """Time-stratified quality/diversity selection; no pose/orientation labels invented."""
 rows=sorted(rows,key=lambda r:r['timestamp'])
 if len(rows)<=n:return rows
 features=np.array([[math.log(max(r['quality']['height'],1)),r['quality']['brightness']/128,r['quality']['contrast']/64,int(r['perceptual_hash'][:8],2)/255] for r in rows])
 chosen=[]
 for bucket in np.array_split(np.arange(len(rows)),n):
  def score(i):return rows[i]['quality']['score']+.2*(min(np.linalg.norm(features[i]-features[j]) for j in chosen) if chosen else 0)
  chosen.append(max(bucket,key=score))
 return [rows[i] for i in chosen]

def tracks(rows,training=False):
 out=[]
 for identity in sorted({r['player_identity'] for r in rows}):
  items=[r for r in rows if r['player_identity']==identity]
  windows=[(a,a+40) for a in range(0,240,40)] if training else [(10,70),(170,250)]
  for k,(a,b) in enumerate(windows):
   candidates=[r for r in items if a<=r['timestamp']<b]
   if len(candidates)<8:continue
   selected=diverse(candidates,12)
   out.append({'tracklet_id':f'{identity}:window{k}','player_identity':identity,'match_id':selected[0]['match_id'],'team_identity':selected[0]['team_identity'],'window':k,'samples':[r['sample_id'] for r in selected],'start':selected[0]['timestamp'],'end':selected[-1]['timestamp'],'quality':float(np.mean([r['quality']['score'] for r in selected]))})
 return out

def prepare():
 manifest=json.loads((M10/'curated_manifest.json').read_text());rows=manifest['samples'];validate_splits(rows)
 # Original quality is frozen. Correct only the new documented display field.
 for r in rows:r['quality']['edge_clipping_fraction']=float(np.clip(r['quality']['edge_clipping_fraction'],0,1))
 for r in rows:
  if sha(r['crop_path'])!=r['crop_sha256']:raise ValueError('Frozen crop hash mismatch')
 result={'m10_manifest_sha256':sha(M10/'curated_manifest.json'),'samples':rows,'tracklets':{s:tracks([r for r in rows if r['split']==s],s=='train') for s in ['train','validation','test']}}
 save('dataset.json',result)
 raw=json.loads((M10/'dataset_manifest.json').read_text())['samples'];audit=json.loads((M10/'identity_audit.json').read_text())
 summary={'splits':{s:{'crops':len(t:=[r for r in rows if r['split']==s]),'identities':len({r['player_identity'] for r in t}),'tracklets':len(result['tracklets'][s]),'matches':sorted({r['match_id'] for r in t})} for s in result['tracklets']},'official_mot_sequences':sorted(audit['annotations']),'training_raw_sequence_identities':len({r['player_identity'] for r in raw if r['split']=='train'}),'training_unique_registered_players':len({r['registered_player_id'] for r in raw if r['split']=='train'}),'expansion_result':'All ten official MOT clips audited. No additional disjoint match/team identities beyond frozen training anchors can be asserted without weakening M10 leakage controls. Full-half videos do not by themselves supply audited paired box identity labels. Keep 51; more identities unavailable under these constraints.','sequence_selection':'8–12 observations, temporal bins and quality/scale/brightness/contrast/phash diversity; true pose/orientation unknown','geometry':'existing supplied per-match nonlinear pitch-keypoint interpolation; proxy position, not independent ground truth; uncertainty remains unknown'}
 save('dataset_audit.json',summary);return result
if __name__=='__main__':print(prepare()['tracklets'].keys())
