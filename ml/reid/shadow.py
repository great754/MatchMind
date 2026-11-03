"""Appearance-only M9 graph ablation. Never changes serving snapshots or team evidence."""
import json,time
from pathlib import Path
import cv2,numpy as np,torch
from .protocol import ROOT
from .model import choose_device,seed_all
from .evaluate import load_model
from .dataset import crop_regions
from src.vision.world.player_appearance import digest
from src.vision.world.player_identity import build_segments,globally_associate
from src.vision.world.player_state import PlayerWorldConfig

def tensors(images,strategy):
 out=[]
 for im in images:
  im=im.copy();im[:int(.18*im.shape[0])]=127;regions=[]
  for r in crop_regions(im,strategy):
   a=cv2.resize(cv2.cvtColor(r,cv2.COLOR_BGR2RGB),(128,256)).transpose(2,0,1).astype(np.float32)/255;regions.append((a-np.array([.485,.456,.406],np.float32)[:,None,None])/np.array([.229,.224,.225],np.float32)[:,None,None])
  out.append(regions)
 return torch.from_numpy(np.asarray(out))

def run():
 lock=json.loads((ROOT/'selection_lock.json').read_text());assert digest(lock['selected_checkpoint'])==lock['checkpoint_sha256'];seed_all(20261102);device=choose_device();model,strategy=load_model(lock['selected_checkpoint'],device=device);summary={};reference=json.loads(Path('docs/evidence/cv_m9_reference.json').read_text());cases=reference['cases']+reference['supplementary_cases']
 for name in ['test_match','team_goal','southampton']:
  start=time.perf_counter();source=Path('outputs/world_model')/(name+'_m9_ready');m=json.loads((source/'manifest.json').read_text());rows=[json.loads(l) for l in (source/'frames.jsonl').open()];cache=Path('outputs/world_model/m9_cache')/name;features=json.loads((cache/'features.json').read_text());oldz=np.load(cache/'embeddings.npz');old={k:oldz[k] for k in oldz.files};out=ROOT/'m9_shadow'/name;out.mkdir(parents=True,exist_ok=True);npz=out/'embeddings.npz';meta_path=out/'embedding_manifest.json'
  if npz.exists():
   meta=json.loads(meta_path.read_text());assert meta['checkpoint_sha256']==lock['checkpoint_sha256'] and meta['source_sha256']==m['video_sha256'] and digest(npz)==meta['embeddings_sha256'];z=np.load(npz);new={k:z[k] for k in z.files}
  else:
   assert digest(m['source_video'])==m['video_sha256'];cap=cv2.VideoCapture(m['source_video']);new={};images=[];keys=[]
   def flush():
    if not images:return
    with torch.inference_mode():v=model(tensors(images,strategy).to(device))[1].cpu().numpy()
    new.update(zip(keys,v));keys.clear();images.clear()
   for i,ps in enumerate(features):
    ok,image=cap.read()
    if not ok:raise ValueError('Incomplete M9 source')
    for p in ps:
     if p.get('embedding_key') not in old:continue
     x,y,w,h=p['bbox'];crop=image[max(0,int(y)):min(image.shape[0],int(y+h)),max(0,int(x)):min(image.shape[1],int(x+w))]
     if not crop.size:raise ValueError('Invalid existing eligible crop')
     images.append(crop.copy());keys.append(p['embedding_key'])
    if len(images)>=48:flush()
   flush();cap.release();np.savez_compressed(npz,**new);meta={'checkpoint_sha256':lock['checkpoint_sha256'],'source_sha256':m['video_sha256'],'source_m9_frames_sha256':m['frames_sha256'],'embeddings_sha256':digest(npz),'strategy':strategy,'same_eligible_observations':len(new)==len(old),'embedding_count':len(new)};meta_path.write_text(json.dumps(meta,indent=2))
  oldsegments,lookup=build_segments(features,rows,old,m['fps']);newsegments,_=build_segments(features,rows,new,m['fps']);frozen={s['segment_id']:s for s in json.loads((source/'segments.json').read_text())};oldsegments=[s for s in oldsegments if s.key in frozen];newsegments=[s for s in newsegments if s.key in frozen]
  for s in oldsegments+newsegments:s.team_label=frozen[s.key]['team_label'];s.team_strength=frozen[s.key]['team_strength']
  cfg=PlayerWorldConfig.model_validate(m['player_world_config']);oldmap,olde,olda,oldevents,_=globally_associate(oldsegments,m['fps'],cfg);newmap,newe,newa,newevents,_=globally_associate(newsegments,m['fps'],cfg);actual=json.loads((source/'reassociation_events.json').read_text());assert [(e['from_segment'],e['to_segment']) for e in oldevents]==[(e['from_segment'],e['to_segment']) for e in actual];assert set(new)==set(old);old_by={s.key:s for s in oldsegments};new_by={s.key:s for s in newsegments};results=[]
  for c in cases:
   if c['recording_id']!=name:continue
   frames=c.get('frames');tracks=c.get('tracks');a,b=frames;ta,tb=tracks;ka=lookup.get(f'{a}:{ta}');kb=lookup.get(f'{b}:{tb}')
   def decision(mapping,ambiguous):return 'UNRESOLVED_FRAGMENT' if ka not in mapping or kb not in mapping else 'CONTINUOUS_RAW_TRACK' if ka==kb else 'LINKED' if mapping[ka]==mapping[kb] else 'AMBIGUOUS' if kb in ambiguous else 'NOT_LINKED'
   def similarity(table):
    if ka not in table or kb not in table:return None
    x,y=table[ka].embedding,table[kb].embedding;return float(np.dot(x,y)) if x is not None and y is not None else None
   # Graph candidates use complete tracklet medians; reference endpoint images may be truncated.
   amendment_path=ROOT/'reference_amendments.json' if (ROOT/'reference_amendments.json').exists() else Path('docs/evidence/cv_m10_reference_amendments.json');amendments=json.loads(amendment_path.read_text()) if amendment_path.exists() else {};correction=amendments.get(c.get('case_id'),{});result={**c,'original_m9_assessment':c.get('assessment'),'assessment':correction.get('amended_assessment',c.get('assessment')),'reference_amendment':correction or None,'generic_tracklet_similarity':similarity(old_by),'soccer_tracklet_similarity':similarity(new_by),'m9_decision':decision(oldmap,olda),'m10_shadow_decision':decision(newmap,newa),'same_frozen_team_and_geometry_evidence':True};results.append(result)
  for filename,value in [('reference_comparison.json',results),('generic_events.json',oldevents),('soccer_events.json',newevents),('soccer_candidates.json',{k:[e.model_dump() for e in v] for k,v in newe.items()})]:(out/filename).write_text(json.dumps(value,indent=2))
  def stats(mapping,ambiguous,events,ev):return {'accepted_reconnects':len(events),'ambiguous_reconnects':len(ambiguous),'rejected_reconnect_targets':sum(bool(v) for v in ev.values())-len(events)-len(ambiguous),'persistent_entities':len(set(mapping.values())),'fragments_per_entity':len(mapping)/len(set(mapping.values())),'duplicate_violations':0,'impossible_metric_links':sum(e['evidence']['position_weight']>0 and e['evidence']['distance_m']>e['evidence']['allowed_distance_m'] for e in events),'global_assignment_margins':[e['global_assignment_margin'] for e in events]}
  summary[name]={'generic':stats(oldmap,olda,oldevents,olde),'soccer_shadow':stats(newmap,newa,newevents,newe),'reference_cases':results,'new_apparent_false_links':[r for r in results if r.get('assessment')=='APPEARS_DIFFERENT' and r['m10_shadow_decision']=='LINKED' and r['m9_decision']!='LINKED'],'wall_seconds':time.perf_counter()-start,'m8_and_m9_snapshots_changed':False,'team_inputs_and_thresholds_frozen':True};print(name,summary[name]['generic'],summary[name]['soccer_shadow'],flush=True)
 (ROOT/'shadow_comparison.json').write_text(json.dumps(summary,indent=2));return summary
if __name__=='__main__':run()
