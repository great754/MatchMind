"""Full-frame M8 invariants and M9 internal consistency; never identity accuracy."""
import argparse,json
from pathlib import Path
from collections import Counter,defaultdict
import numpy as np
from src.vision.world.schema import WorldState
from src.vision.world.player_state import VideoPlayerObservation
from src.vision.world.player_appearance import raw_people,digest,cosine
from src.serving.cv import CVReview

def streaks(values,unknown=None):
 runs=[];start=None
 for i,v in enumerate(list(values)+[False]):
  missing=v is None if unknown is None else v==unknown
  if i==len(values):missing=False
  if missing and start is None:start=i
  elif not missing and start is not None:runs.append(i-start);start=None
 return runs

def team_metrics(series,unknown):
 count=Counter({k:0 for k in ['known_observations','unknown_observations','established_to_unknown_events','label_changes','one_frame_unknown_flickers','short_established_unknown_runs_2_to_5']});unknown_runs=[]
 for values in series.values():
  established=False
  for i,v in enumerate(values):
   missing=v==unknown;count['known_observations']+=not missing;count['unknown_observations']+=missing
   if established and missing and i and values[i-1]!=unknown:count['established_to_unknown_events']+=1
   if not missing:established=True
   if i and v!=unknown and values[i-1]!=unknown and v!=values[i-1]:count['label_changes']+=1
  runs=streaks(values,unknown);unknown_runs+=runs
  i=0
  while i<len(values):
   if values[i]!=unknown:i+=1;continue
   start=i
   while i<len(values) and values[i]==unknown:i+=1
   if start>0 and i<len(values):
    count['one_frame_unknown_flickers']+=i-start==1;count['short_established_unknown_runs_2_to_5']+=2<=i-start<=5
 count['one_frame_unknown_runs']=sum(x==1 for x in unknown_runs);count['short_unknown_runs_2_to_5']=sum(2<=x<=5 for x in unknown_runs);return dict(count)

def evaluate(name,directory,output):
 directory=Path(directory);output=Path(output);output.mkdir(parents=True,exist_ok=True);base=Path('outputs/world_model')/(name+'_m8_release');old=CVReview(base);new=CVReview(directory);m=new.manifest;assert old.manifest['frames_sha256']==m['source_m8_frames_sha256'];a=[json.loads(l) for l in (base/'frames.jsonl').open()];b=[json.loads(l) for l in (directory/'frames.jsonl').open()];before=defaultdict(list);after=defaultdict(list);raw=placed=unknown_cam=0;states=Counter();identity=Counter();duplicates=0
 for x,y in zip(a,b,strict=True):
  # No camera, detector, old player, short-track or existing canonical analytics changes.
  copy={k:v for k,v in y.items() if k!='player_world'};assert copy==x;state=WorldState.model_validate(y);people=raw_people(x);bytrack={p['track_id']:p for p in people};world=state.player_world;assert world is not None and len(world.players)+len(world.unresolved_detections)>=len(people);visible=[p for p in world.players if p.current_track_id is not None];assert len(visible)+len(world.unresolved_detections)==len(people);assert len({p.entity_id for p in visible})==len(visible);raw+=len(people);placed+=sum(p['pitch'] is not None for p in people);unknown_cam+=not x['camera']['calibration_valid']
  for p in world.unresolved_detections:
   source=bytrack[p.track_id];assert p.current_position==(tuple(source['pitch']) if source['pitch'] is not None else None);after['UNRESOLVED:'+p.track_id].append('UNKNOWN')
  for p in people:before[p['track_id']].append(p['legacy_team'])
  for p in world.players:
   states[p.visibility_state]+=1
   if p.current_track_id is not None:
    source=bytrack[p.current_track_id];assert p.current_position==(tuple(source['pitch']) if source['pitch'] is not None else None);after[p.entity_id].append(p.team.label);identity[p.identity_status]+=1
    if p.current_position is None:assert p.position_confidence is None and p.position_provenance=='UNKNOWN'
    else:assert p.position_confidence is not None and p.position_uncertainty_m is not None
   else:assert p.current_position is None and p.position_confidence is None and p.position_provenance=='UNKNOWN'
 assert digest(base/'observations.jsonl')==digest(directory/'observations.jsonl');assert raw==sum(len(raw_people(x)) for x in a);events=json.loads((directory/'reassociation_events.json').read_text());diag=json.loads((directory/'diagnostics.json').read_text());impossible=[e for e in events if not e['evidence']['eligible'] or (e['evidence']['position_weight']>0 and e['evidence']['distance_m']>e['evidence']['allowed_distance_m'])];assert not impossible;series=json.loads((directory/'player_timelines.json').read_text());switches=Counter(e['kind'] for v in diag['team_events'].values() for e in v);candidate=json.loads((directory/'reid_candidates.json').read_text());reasons=Counter(c['reason'] for vs in candidate.values() for c in vs)
 # Validate the separate future-facing canonical stream against authoritative states.
 canonical_count=0;keys=set();lookups=[({p['entity_id']:p for p in r['player_world']['players']},{p['track_id']:p for p in r['player_world']['unresolved_detections']}) for r in b]
 with (directory/'video_player_observations.jsonl').open() as source:
  for line in source:
   c=VideoPlayerObservation.model_validate_json(line);assert 0<=c.frame<len(b) and abs(c.timestamp-c.frame/m['fps'])<1e-6;players,unresolved=lookups[c.frame];key=(c.frame,c.entity_id or 'UNRESOLVED:'+str(c.track_id));assert key not in keys;keys.add(key)
   if c.entity_id is None:p=unresolved[c.track_id];assert c.visible and c.identity_confidence==0
   else:p=players[c.entity_id];assert c.track_id==p['current_track_id'] and c.visible==(p['current_track_id'] is not None) and c.team==p['team']['label']
   assert (c.x,c.y)==(tuple(p['current_position']) if p['current_position'] is not None else (None,None));assert c.position_provenance==p['position_provenance'] and c.position_confidence==p['position_confidence'];canonical_count+=1
 assert canonical_count==sum(len(r['player_world']['players'])+len(r['player_world']['unresolved_detections']) for r in b)
 # Same temporary-track segments provide a proxy for embedding consistency, not identity labels.
 cache=Path('outputs/world_model/m9_cache')/name;features=json.loads((cache/'features.json').read_text());z=np.load(cache/'embeddings.npz');positive=[];negative=[];histpos=[];histneg=[];last={}
 for i,people in enumerate(features):
  current=[p for p in people if p.get('embedding_key') in z]
  for p in current:
   tid=p['track_id'];v=z[p['embedding_key']]
   if tid in last and i-last[tid][0]>=3:positive.append(cosine(last[tid][1],v));histpos.append(cosine(last[tid][2],p['feature'].get('histogram')))
   last[tid]=(i,v,p['feature'].get('histogram'))
  for j,p in enumerate(current):
   for q in current[j+1:]:
    if np.linalg.norm(np.array(p['foot'])-q['foot'])>max(p['bbox'][3],q['bbox'][3]):negative.append(cosine(z[p['embedding_key']],z[q['embedding_key']]));histneg.append(cosine(p['feature'].get('histogram'),q['feature'].get('histogram')))
 def distribution(v):
  v=[x for x in v if x is not None];return {'count':len(v),'median':float(np.median(v)) if v else None,'p10':float(np.quantile(v,.1)) if v else None,'p90':float(np.quantile(v,.9)) if v else None}
 report={'recording_id':name,'classification':'LOCALLY VERIFIED INTERNAL CONSISTENCY; NOT INDEPENDENT IDENTITY ACCURACY','m8_frames_sha256':old.manifest['frames_sha256'],'m9_frames_sha256':m['frames_sha256'],'frames':len(a),'camera_accepted':len(a)-unknown_cam,'camera_unchanged':True,'raw_observations':raw,'current_placed':placed,'all_m8_raw_layers_identical':True,'video_player_canonical_sha256':digest(directory/'video_player_observations.jsonl'),'video_player_canonical_records_validated':canonical_count,'temporary_tracks':diag['temporary_track_ids'],'temporary_tracks_per_minute':diag['temporary_track_ids']/(len(a)/m['fps']/60),'track_fragments':diag['track_fragments'],'confirmed_track_fragments':diag['confirmed_track_fragments'],'unconfirmed_track_fragments':diag['unconfirmed_track_fragments'],'persistent_entities':diag['persistent_entities'],'mean_fragments_per_entity':diag['mean_fragments_per_entity'],'reassociation_attempts':sum(bool(v) for v in candidate.values()),'successful_reconnects':len(events),'rejected_reconnect_targets':sum(bool(v) for v in candidate.values())-len(events)-diag['ambiguous_reconnects'],'ambiguous_reconnects':diag['ambiguous_reconnects'],'rejected_candidate_pairs':sum(not c['eligible'] for vs in candidate.values() for c in vs),'candidate_rejection_reasons':dict(reasons),'simultaneous_duplicate_identity_violations':duplicates,'physically_impossible_accepted_reassociations':0,'independent_identity_switches':None,'identity_status_observations':dict(identity),'visibility_state_observations':dict(states),'team_before_m8_temporary_track_grouping':team_metrics(before,None),'team_after_m9_persistent_entity_grouping':team_metrics(after,'UNKNOWN'),'team_event_counts':dict(switches),'team_role_counts':diag['counts'],'embedding_evaluation':{'label':'RAW_TRACK_CONTINUITY_AND_COVISIBLE_NEGATIVE_PROXIES; NOT IDENTITY ACCURACY','osnet_same_track':distribution(positive),'osnet_covisible_separated_tracks':distribution(negative),'histogram_same_track':distribution(histpos),'histogram_covisible_separated_tracks':distribution(histneg)},'timing':diag['timing'],'independent_team_accuracy':None,'independent_identity_accuracy':None};(output/'evaluation.json').write_text(json.dumps(report,indent=2));print(name,report['team_before_m8_temporary_track_grouping'],report['team_after_m9_persistent_entity_grouping'],len(events),flush=True)
 return report
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--recording',required=True);p.add_argument('--directory',required=True);p.add_argument('--output',required=True);a=p.parse_args();evaluate(a.recording,a.directory,a.output)
