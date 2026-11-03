"""Read-only representation/pooling shadow; frozen M9 physical/global decisions."""
import json,time
from pathlib import Path
import cv2,numpy as np,torch
from ml.reid.shadow import tensors
from ml.reid.tracklet import pool
from ml.reid.model import choose_device
from src.vision.world.player_identity import build_segments,globally_associate
from src.vision.world.player_state import PlayerWorldConfig
from .common import ROOT,M10,load,save,sha
from .train import load_model
from .model import Pooler

def run():
 lock=load('selection_lock.json');device=choose_device();model=load_model(lock['visual_checkpoint'],device);pooler=Pooler().to(device);pooler.load_state_dict(torch.load(lock['pool_checkpoint'],map_location=device,weights_only=True)['model']);pooler.eval();summary={};tick=time.perf_counter();ref=json.loads(Path('docs/evidence/cv_m9_reference.json').read_text());cases=ref['cases']+ref['supplementary_cases'];amend=json.loads(Path('docs/evidence/cv_m10_reference_amendments.json').read_text())
 for name in ['test_match','team_goal','southampton']:
  base=Path('outputs/world_model')/(name+'_m9_ready');manifest=json.loads((base/'manifest.json').read_text());rows=[json.loads(x) for x in (base/'frames.jsonl').open()];cache=Path('outputs/world_model/m9_cache')/name;features=json.loads((cache/'features.json').read_text());z=np.load(cache/'embeddings.npz');old={k:z[k] for k in z.files};z=np.load(M10/'m9_shadow'/name/'embeddings.npz');m10={k:z[k] for k in z.files};out=ROOT/'shadow'/name;out.mkdir(exist_ok=True,parents=True);p=out/'embeddings.npz'
  if p.exists():
   meta=json.loads((out/'cache.json').read_text())
   if sha(p)!=meta['sha256'] or meta['checkpoint_sha256']!=lock['visual_sha256']:raise ValueError('Shadow cache hash mismatch')
   z=np.load(p);new={k:z[k] for k in z.files}
  else:
   if sha(manifest['source_video'])!=manifest['video_sha256']:raise ValueError('Shadow source changed')
   cap=cv2.VideoCapture(manifest['source_video']);new={};images=[];keys=[]
   def flush():
    if not images:return
    with torch.inference_mode():v=model(tensors(images,'full_masked').to(device))[1].cpu().numpy()
    new.update(zip(keys,v));images.clear();keys.clear()
   for frame,people in enumerate(features):
    ok,image=cap.read()
    if not ok:raise ValueError('Incomplete shadow video')
    for person in people:
     if person.get('embedding_key') not in old:continue
     x,y,w,h=person['bbox'];crop=image[max(0,int(y)):min(image.shape[0],int(y+h)),max(0,int(x)):min(image.shape[1],int(x+w))]
     if not crop.size:raise ValueError('Invalid old eligible crop')
     images.append(crop);keys.append(person['embedding_key'])
    if len(images)>=48:flush()
   flush();cap.release();np.savez_compressed(p,**new);(out/'cache.json').write_text(json.dumps({'sha256':sha(p),'checkpoint_sha256':lock['visual_sha256'],'source_sha256':manifest['video_sha256'],'same_eligible_observations':set(new)==set(old)}))
  assert set(new)==set(old);frozen={s['segment_id']:s for s in json.loads((base/'segments.json').read_text())};cfg=PlayerWorldConfig.model_validate(manifest['player_world_config']);results={};sets={};attention={}
  for kind,embeddings in [('m9',old),('m10',m10),('m13_median',new),('m13_selected_pool',new)]:
   segments,lookup=build_segments(features,rows,embeddings,manifest['fps']);segments=[s for s in segments if s.key in frozen]
   for s in segments:
    s.team_label=frozen[s.key]['team_label'];s.team_strength=frozen[s.key]['team_strength']
    if kind!='m13_selected_pool':continue
    obs=[o for o in s.observations if o.get('embedding_key') in new]
    if not obs:continue
    selected=[max(chunk,key=lambda o:o['feature']['quality']) for chunk in np.array_split(np.array(obs,dtype=object),min(8,len(obs)))];v=np.array([new[o['embedding_key']] for o in selected]);q=[o['feature']['quality'] for o in selected];method=lock['selected_aggregation']
    if method=='learned':
     with torch.inference_mode():a,w=pooler(torch.tensor(v[None],device=device),torch.tensor([q],device=device))
     s.embedding=a[0].cpu().numpy();weights=w[0].cpu().tolist()
    else:s.embedding=pool(v,q if method=='quality_weighted' else None);weights=None
    attention[s.key]={'frames':[o['frame'] for o in selected],'weights':weights,'method':method,'shorter_than4':len(selected)<4}
   results[kind]=globally_associate(segments,manifest['fps'],cfg);sets[kind]=segments
  actual=json.loads((base/'reassociation_events.json').read_text());assert [(e['from_segment'],e['to_segment']) for e in results['m9'][3]]==[(e['from_segment'],e['to_segment']) for e in actual]
  def stats(kind):
   mapping,ev,amb,events,timing=results[kind];segments=sets[kind];duplicate=sum(mapping[a.key]==mapping[b.key] and max(a.start,b.start)<=min(a.end,b.end) for i,a in enumerate(segments) for b in segments[i+1:]);return {'accepted':len(events),'ambiguous':len(amb),'rejected_targets':sum(bool(v) for v in ev.values())-len(events)-len(amb),'persistent_entities':len(set(mapping.values())),'fragments_per_entity':len(mapping)/len(set(mapping.values())),'duplicate_violations':duplicate,'impossible_metric_links':sum(e['evidence']['position_weight']>0 and e['evidence']['distance_m']>e['evidence']['allowed_distance_m'] for e in events),'global_assignment_margins':[e['global_assignment_margin'] for e in events],'edges':[[e['from_segment'],e['to_segment']] for e in events]}
  reference=[]
  for c in cases:
   if c['recording_id']!=name:continue
   a,b=c['frames'];ta,tb=c['tracks'];ka=lookup.get(f'{a}:{ta}');kb=lookup.get(f'{b}:{tb}');decisions={}
   for kind,result in results.items():
    mapping,ev,amb,events,_=result;decisions[kind]='UNRESOLVED' if ka not in mapping or kb not in mapping else 'CONTINUOUS_RAW_TRACK' if ka==kb else 'LINKED' if mapping[ka]==mapping[kb] else 'AMBIGUOUS' if kb in amb else 'NOT_LINKED'
   reference.append({**c,'assessment':amend.get(c.get('case_id'),{}).get('amended_assessment',c.get('assessment')),'decisions':decisions})
  summary[name]={k:stats(k) for k in results};summary[name]['references']=reference;summary[name]['physical_team_camera_and_global_config_unchanged']=True
  for kind,result in results.items():
   (out/f'{kind}_events.json').write_text(json.dumps(result[3],indent=2));(out/f'{kind}_candidates.json').write_text(json.dumps({k:[e.model_dump() for e in v] for k,v in result[1].items()},indent=2))
  (out/'aggregation.json').write_text(json.dumps(attention,indent=2));print('SHADOW',name,{k:(v['accepted'],v['ambiguous']) for k,v in summary[name].items() if isinstance(v,dict) and 'accepted' in v},flush=True)
 save('shadow_results.json',{'sources':summary,'wall_seconds':time.perf_counter()-tick,'production_activated':False,'threshold_policy':'original M9 appearance0.82 / score0.78 / global margin0.06 for all variants; crop-only validation thresholds not misapplied as calibrated segment/fusion scores','independent_identity_accuracy':None});return summary
if __name__=='__main__':run()
