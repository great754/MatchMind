"""Offline-only M11 evidence ablations over frozen M9 observations and gates."""
import json,time,pickle,math
from pathlib import Path
import cv2,numpy as np,torch
from scipy.optimize import linear_sum_assignment
from src.vision.world.player_identity import build_segments,globally_associate,candidate,uncertainty
from src.vision.world.player_state import PlayerWorldConfig
from .common import ROOT,M10,load,save,sha
from .model import Visual,image_array
from .evaluate import load_models
from .train import sequence_arrays
from .scorer import cues,physical_gate

def describe(s,fps):
 a=s.observations[0];b=s.observations[-1];velocity=None
 if len(s.observations)>3:
  early=s.observations[max(0,len(s.observations)-7)]
  if early.get('pitch') is not None and b.get('pitch') is not None and b['frame']>early['frame']:
   v=(np.asarray(b['pitch'])-early['pitch'])/((b['frame']-early['frame'])/fps);v*=min(1,12/max(np.linalg.norm(v),1e-8));velocity=v.tolist()
 u=uncertainty(a);v=uncertainty(b)
 return {'tracklet_id':s.key,'team_identity':s.team_label,'team_strength':s.team_strength,'epoch':s.epoch,'start':s.start/fps,'end':s.end/fps,'start_pitch':a.get('pitch'),'end_pitch':b.get('pitch'),'end_velocity':velocity,'uncertainty':max(u,v) if u is not None and v is not None else None,'visibility':None,'quality':s.quality,'samples':s.evidence_frames}

def learned_candidates(segments,fps,cfg,scorer):
 evidence={};modified=cfg.model_copy(update={'minimum_appearance':0.})
 for current in segments:
  rows=[]
  for previous in segments:
   if previous.end>=current.start or previous.epoch!=current.epoch or (current.start-previous.end)/fps>cfg.max_reconnect_seconds:continue
   e=candidate(previous,current,fps,modified);a=describe(previous,fps);b=describe(current,fps);gate,reason=physical_gate(a,b,max_speed=cfg.max_speed_mps,slack=cfg.movement_slack_m,max_gap=cfg.max_reconnect_seconds);x=cues(a,b,e.appearance_similarity if e.appearance_similarity is not None else np.nan);prob=float(scorer.predict_proba(x[None])[0,1]);e=e.model_copy(update={'score':prob,'eligible':bool(e.eligible and gate),'reason':e.reason if gate else reason});rows.append(e)
  evidence[current.key]=sorted(rows,key=lambda e:-e.score)
 return evidence

def assign(segments,evidence,threshold,margin):
 """Same Hungarian/dummy/global-alternative semantics as M9; no greedy joins."""
 n=len(segments);cost=np.full((n,2*n),1-threshold);lookup={s.key:i for i,s in enumerate(segments)}
 for j,s in enumerate(segments):
  for e in evidence.get(s.key,[]):
   if e.eligible and e.score>=threshold:cost[lookup[e.segment_id],j]=1-e.score
 rr,cc=linear_sum_assignment(cost);optimum=float(cost[rr,cc].sum());edges=[];ambiguous=set();predecessor={}
 for i,j in zip(rr,cc):
  if j>=n or cost[i,j]>=1-threshold:continue
  alternative=cost.copy();alternative[i,j]=1-threshold;ar,ac=linear_sum_assignment(alternative);delta=float(alternative[ar,ac].sum()-optimum)
  if delta<margin:ambiguous.add(segments[j].key);continue
  e=next(e for e in evidence[segments[j].key] if e.segment_id==segments[i].key);edges.append({'from_segment':segments[i].key,'to_segment':segments[j].key,'global_assignment_margin':delta,'evidence':e.model_dump()});predecessor[segments[j].key]=segments[i].key
 mapping={}
 for s in sorted(segments,key=lambda s:(s.start,s.key)):mapping[s.key]=mapping[predecessor[s.key]] if s.key in predecessor else s.key
 for i,a in enumerate(segments):
  for b in segments[i+1:]:
   if mapping[a.key]==mapping[b.key] and max(a.start,b.start)<=min(a.end,b.end):raise ValueError('Duplicate simultaneous player')
 return mapping,evidence,ambiguous,edges,{}

def stats(result,segments):
 mapping,evidence,ambiguous,events,_=result
 return {'accepted_reconnects':len(events),'ambiguous_reconnects':len(ambiguous),'rejected_targets':sum(bool(x) for x in evidence.values())-len(events)-len(ambiguous),'persistent_entities':len(set(mapping.values())),'duplicate_violations':sum(mapping[a.key]==mapping[b.key] and max(a.start,b.start)<=min(a.end,b.end) for i,a in enumerate(segments) for b in segments[i+1:]),'impossible_metric_links':sum(e['evidence']['position_weight']>0 and e['evidence']['distance_m'] is not None and e['evidence']['distance_m']>e['evidence']['allowed_distance_m'] for e in events),'accepted_edges':[[e['from_segment'],e['to_segment']] for e in events]}

def temporal_segments(segments,features,fine,temporal,fps,device):
 weights={}
 for s in segments:
  observations=[o for o in s.observations if o.get('embedding_key') in fine]
  if not observations:continue
  # One useful observation per time stratum; quality selects within each stratum.
  selected=[max(chunk,key=lambda o:o['feature']['quality']) for chunk in np.array_split(np.asarray(observations,dtype=object),min(12,len(observations)))]
  x=torch.tensor(np.array([[fine[o['embedding_key']] for o in selected]]),device=device);q=torch.tensor([[o['feature']['quality'] for o in selected]],device=device);t=torch.tensor([[o['frame']/fps for o in selected]],device=device)
  with torch.inference_mode():_,z,w=temporal(x,q,t)
  s.embedding=z[0].cpu().numpy();weights[s.key]={'frames':[o['frame'] for o in selected],'weights':w[0].cpu().tolist(),'below_training_minimum':len(selected)<8}
 return weights

def run():
 start=time.perf_counter();fine_model,temporal,device=load_models();pretrained=Visual().to(device).eval();summary={};lock=load('selection_lock.json')
 with (ROOT/'reassociation.pkl').open('rb') as f:scorer=pickle.load(f)
 refs=json.loads(Path('docs/evidence/cv_m9_reference.json').read_text());cases=refs['cases']+refs['supplementary_cases'];amendments=json.loads(Path('docs/evidence/cv_m10_reference_amendments.json').read_text())
 for name in ['test_match','team_goal','southampton']:
  source=Path('outputs/world_model')/(name+'_m9_ready');manifest=json.loads((source/'manifest.json').read_text());rows=[json.loads(x) for x in (source/'frames.jsonl').open()];cache=Path('outputs/world_model/m9_cache')/name;features=json.loads((cache/'features.json').read_text());z=np.load(cache/'embeddings.npz');generic={k:z[k] for k in z.files};z=np.load(M10/'m9_shadow'/name/'embeddings.npz');m10={k:z[k] for k in z.files};directory=ROOT/'shadow'/name;directory.mkdir(parents=True,exist_ok=True);p=directory/'visual_embeddings.npz'
  if p.exists():
   meta=json.loads((directory/'cache.json').read_text())
   if sha(p)!=meta['sha256'] or meta['visual_sha256']!=lock['visual_sha256'] or meta['video_sha256']!=manifest['video_sha256']:raise ValueError('Shadow cache mismatch')
   z=np.load(p);pre={k[4:]:z[k] for k in z.files if k.startswith('pre:')};fine={k[5:]:z[k] for k in z.files if k.startswith('fine:')}
  else:
   if sha(manifest['source_video'])!=manifest['video_sha256']:raise ValueError('Source video hash changed')
   cap=cv2.VideoCapture(manifest['source_video']);pre={};fine={};images=[];keys=[]
   def flush():
    if not images:return
    x=torch.tensor(np.stack(images),device=device)
    with torch.inference_mode():tokens=pretrained.prefix(x);a=pretrained.finish(tokens).cpu().numpy();b=fine_model.finish(tokens).cpu().numpy()
    pre.update(zip(keys,a));fine.update(zip(keys,b));images.clear();keys.clear()
   for frame,people in enumerate(features):
    ok,image=cap.read()
    if not ok:raise ValueError('Incomplete shadow source')
    for person in people:
     if person.get('embedding_key') not in generic:continue
     x,y,w,h=person['bbox'];crop=image[max(0,int(y)):min(image.shape[0],int(y+h)),max(0,int(x)):min(image.shape[1],int(x+w))]
     if not crop.size:raise ValueError('Invalid old eligible crop')
     images.append(image_array(crop));keys.append(person['embedding_key'])
    if len(images)>=32:flush()
   flush();cap.release();np.savez_compressed(p,**{'pre:'+k:v for k,v in pre.items()},**{'fine:'+k:v for k,v in fine.items()});(directory/'cache.json').write_text(json.dumps({'sha256':sha(p),'visual_sha256':lock['visual_sha256'],'video_sha256':manifest['video_sha256'],'identical_eligible_observations':set(pre)==set(fine)==set(generic)}))
  frozen={s['segment_id']:s for s in json.loads((source/'segments.json').read_text())};cfg=PlayerWorldConfig.model_validate(manifest['player_world_config']);results={};segmentsets={};lookup=None
  for kind,embeddings in [('m9',generic),('m10',m10),('vit_pretrained',pre),('vit_finetuned',fine),('temporal',fine)]:
   segments,l=build_segments(features,rows,embeddings,manifest['fps']);segments=[s for s in segments if s.key in frozen]
   for s in segments:s.team_label=frozen[s.key]['team_label'];s.team_strength=frozen[s.key]['team_strength']
   if kind=='temporal':
    weights=temporal_segments(segments,features,fine,temporal,manifest['fps'],device);(directory/'attention.json').write_text(json.dumps(weights,indent=2))
   result=globally_associate(segments,manifest['fps'],cfg);results[kind]=result;segmentsets[kind]=segments;lookup=l
  actual=json.loads((source/'reassociation_events.json').read_text());assert results['m9'][3] and [(e['from_segment'],e['to_segment']) for e in results['m9'][3]]==[(e['from_segment'],e['to_segment']) for e in actual]
  # Verify new assignment implementation exactly replays M9 given the same evidence.
  replay=assign(segmentsets['m9'],results['m9'][1],cfg.minimum_score,cfg.ambiguity_margin);assert replay[0].keys()==results['m9'][0].keys();assert [[e['from_segment'],e['to_segment']] for e in replay[3]]==[[e['from_segment'],e['to_segment']] for e in results['m9'][3]]
  evidence=learned_candidates(segmentsets['temporal'],manifest['fps'],cfg,scorer);threshold=max(cfg.minimum_score,lock['probability_threshold']);results['temporal_scorer']=assign(segmentsets['temporal'],evidence,threshold,cfg.ambiguity_margin);segmentsets['temporal_scorer']=segmentsets['temporal'];comparison={kind:stats(result,segmentsets[kind]) for kind,result in results.items()};reference=[]
  for c in cases:
   if c['recording_id']!=name:continue
   a,b=c['frames'];ta,tb=c['tracks'];ka=lookup.get(f'{a}:{ta}');kb=lookup.get(f'{b}:{tb}');decisions={}
   for kind,result in results.items():
    mapping,ev,amb,events,_=result;decisions[kind]='UNRESOLVED' if ka not in mapping or kb not in mapping else 'CONTINUOUS_RAW_TRACK' if ka==kb else 'LINKED' if mapping[ka]==mapping[kb] else 'AMBIGUOUS' if kb in amb else 'NOT_LINKED'
   reference.append({**c,'assessment':amendments.get(c.get('case_id'),{}).get('amended_assessment',c.get('assessment')),'decisions':decisions})
  comparison['references']=reference;comparison['probability_threshold']=threshold;comparison['same_frozen_physical_team_camera_evidence']=True
  for kind,result in results.items():(directory/f'{kind}_events.json').write_text(json.dumps(result[3],indent=2))
  (directory/'scorer_candidates.json').write_text(json.dumps({k:[e.model_dump() for e in v] for k,v in evidence.items()},indent=2));summary[name]=comparison;print('SHADOW',name,{k:(v['accepted_reconnects'],v['ambiguous_reconnects']) for k,v in comparison.items() if isinstance(v,dict) and 'accepted_reconnects' in v},flush=True)
 save('shadow_results.json',{'sources':summary,'wall_seconds':time.perf_counter()-start,'production_activated':False,'independent_identity_accuracy':None});return summary
if __name__=='__main__':run()
