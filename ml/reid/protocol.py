"""Leakage guards, fixed retrieval protocol and predeclared integration gate."""
import json
from pathlib import Path
import numpy as np
from src.vision.world.player_appearance import digest
from .audit import ROOT

def curate():
 raw=json.loads((ROOT/'dataset_manifest.json').read_text());rows=raw['samples'];audit=json.loads((ROOT/'identity_audit.json').read_text());anchors={};people=set();jerseys=set()
 for match in audit['groups']['train']:
  for tid,p in sorted(audit['allowed_players'][match].items()):
   jersey=(p['team_id'],p['jersey_metadata']);pid=p['roster_player_id']
   if pid in people or jersey in jerseys:continue
   anchors[p['annotation_identity']]=match;people.add(pid);jerseys.add(jersey)
 rows=[r for r in rows if r['split']!='train' or r['player_identity'] in anchors];counts={}
 for r in rows:counts[r['player_identity']]=counts.get(r['player_identity'],0)+1
 rows=[r for r in rows if counts[r['player_identity']]>=30]
 for r in rows:
  if digest(r['crop_path'])!=r['crop_sha256']:raise ValueError('Crop content/hash mismatch')
 validate_splits(rows)
 manifest={'version':'soccer-reid-v1-curated','raw_manifest_sha256':digest(ROOT/'dataset_manifest.json'),'split_sha256':digest('configs/ml/soccer-reid-v1-split.json'),'training_anchor_policy':'one match per registered player/team+jersey alias; avoid cross-match false negatives rather than assume cross-match positives','samples':rows}
 (ROOT/'curated_manifest.json').write_text(json.dumps(manifest));return manifest

def validate_splits(rows):
 sets={s:{'matches':set(),'identities':set(),'players':set(),'teams':set(),'hashes':set()} for s in ['train','validation','test']}
 for r in rows:
  s=sets[r['split']];s['matches'].add(r['match_id']);s['identities'].add(r['player_identity']);s['players'].add(r['registered_player_id']);s['teams'].add(r['team_identity']);s['hashes'].add(r['source_sha256'])
  if r['recording_id'] in ['test_match','team_goal','southampton']:raise ValueError('M9 evaluation video entered supervised data')
 for a,b in [('train','validation'),('train','test'),('validation','test')]:
  for key in sets[a]:
   if sets[a][key]&sets[b][key]:raise ValueError(f'Leakage {a}/{b}: {key}')
 return {s:{k:sorted(v) for k,v in d.items()} for s,d in sets.items()}

def retrieval_sets(rows,tracklet=False):
 q=[];g=[]
 for identity in sorted({r['player_identity'] for r in rows}):
  items=[r for r in rows if r['player_identity']==identity];early=[r for r in items if 10<=r['timestamp']<=70];late=[r for r in items if r['timestamp']>=170]
  if len(early)<10 or len(late)<10:continue
  if tracklet:q.append(early[:30]);g.append(late[-30:])
  else:
   q.extend([[early[i]] for i in np.linspace(0,len(early)-1,3,dtype=int)]);g.extend([[late[i]] for i in np.linspace(0,len(late)-1,3,dtype=int)])
 return q,g

def integration_gate(generic,soccer):
 changes={k:soccer[k]-generic[k] for k in ['same_team_rank1','same_team_map','tracklet_map','rank1','map']}
 passed=changes['same_team_rank1']>=.03 and changes['same_team_map']>=.03 and changes['tracklet_map']>=.03 and changes['map']>=-.005
 return {'passed':passed,'deltas':changes,'predeclared_requirements':{'same_team_rank1_gain':.03,'same_team_map_gain':.03,'tracklet_map_gain':.03,'overall_map_max_regression':.005},'threshold_tuned_on_test':False}
