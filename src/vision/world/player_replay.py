"""M9 offline persistent player graph over immutable M8 camera/observation inputs."""
import argparse,json,time,shutil,subprocess,csv,io
from pathlib import Path
from collections import defaultdict,Counter
import cv2,numpy as np
from .schema import WorldState
from .player_state import PlayerWorldConfig,PersistentPerson,PlayerWorldFrame,VideoPlayerObservation,UnresolvedPlayerDetection
from .player_appearance import digest,extract,raw_people,cosine
from .player_identity import build_segments,globally_associate,uncertainty,absence_state
from .player_teams import TeamAppearance,PersistentTeam,JerseyBelief

SOURCES={'test_match':'outputs/world_model/test_match_m8_release','team_goal':'outputs/world_model/team_goal_m8_release','southampton':'outputs/world_model/southampton_m8_release'}
def run(name,output,checkpoint='outputs/models/m9/osnet_x0_25_msmt17.pth',config=None,ocr=True,baseline=None):
 started=time.perf_counter();cfg=config or PlayerWorldConfig()
 if baseline is None and name not in SOURCES:raise ValueError('--baseline is required for a new recording')
 baseline=Path(baseline or SOURCES[name]);output=Path(output)
 if output.exists():raise ValueError('Use a new directory to preserve prior evidence')
 m=json.loads((baseline/'manifest.json').read_text())
 if m['recording_id']!=name or m['camera_model_version']!='m8' or digest(baseline/'frames.jsonl')!=m['frames_sha256'] or digest(m['source_video'])!=m['video_sha256']:raise ValueError('M9 requires matching recording, M8 camera snapshot and source hashes')
 rows=[json.loads(l) for l in (baseline/'frames.jsonl').open()];features,embeddings,timing=extract(m['source_video'],rows,m,Path('outputs/world_model/m9_cache')/name,checkpoint);tick=time.perf_counter();segments,lookup=build_segments(features,rows,embeddings,m['fps']);timing['short_track_segmentation_seconds']=time.perf_counter()-tick;tick=time.perf_counter();teams=TeamAppearance.fit(segments)
 for people in features:
  for p in people:p['visual_team']=teams.classify(p['feature'])
 for s in segments:s.team_summary()
 timing['team_classification_seconds']=time.perf_counter()-tick;all_segments=segments;segments=[s for s in all_segments if len(s.observations)>=cfg.min_track_observations];confirmed_keys={s.key for s in segments};mapping,evidence,ambiguous,events,reidtime=globally_associate(segments,m['fps'],cfg);timing.update(reidtime);output.mkdir(parents=True);entity_segments=defaultdict(list)
 for s in segments:entity_segments[mapping[s.key]].append(s)
 for e in events:e['entity_id']=mapping[e['to_segment']]
 (output/'unconfirmed_fragments.json').write_text(json.dumps([{'segment_id':s.key,'track_id':s.track_id,'frames':[o['frame'] for o in s.observations],'reason':'INSUFFICIENT_REPEATED_OBSERVATIONS_FOR_PERSISTENT_ENTITY'} for s in all_segments if s.key not in confirmed_keys],indent=2))
 # Persist every candidate, including gates; UI includes a bounded best-candidate list.
 (output/'reid_candidates.json').write_text(json.dumps({k:[x.model_dump() for x in v] for k,v in evidence.items()},indent=2));(output/'segments.json').write_text(json.dumps([{'segment_id':s.key,'track_id':s.track_id,'epoch':s.epoch,'start':s.start,'end':s.end,'entity_id':mapping[s.key],'appearance_quality':s.quality,'embedding_frames':s.evidence_frames,'team_label':s.team_label,'team_strength':s.team_strength} for s in segments],indent=2));(output/'reassociation_events.json').write_text(json.dumps(events,indent=2));(output/'identity_decisions.json').write_text(json.dumps({s.key:{'entity_id':mapping[s.key],'decision':'REASSOCIATED' if any(e['to_segment']==s.key for e in events) else 'IDENTITY_AMBIGUOUS' if s.key in ambiguous else 'NEW_PROVISIONAL_ENTITY','retained_candidates':len(evidence[s.key]),'reason':'GLOBAL_ONE_TO_ONE_WITH_ALTERNATIVE_ASSIGNMENT_MARGIN; WHOLE_TRACK_FUTURE_APPEARANCE'} for s in segments},indent=2));(output/'team_model.json').write_text(json.dumps(teams.diagnostic,indent=2))
 ocr_records=run_ocr(m,segments,output) if ocr else {};jerseys={key:JerseyBelief() for key in entity_segments};beliefs={key:PersistentTeam(cfg) for key in entity_segments};last={};timeline=defaultdict(list);offsets=[];counters=Counter();team_histories=defaultdict(list);tick=time.perf_counter()
 with (output/'frames.jsonl').open('wb') as target,(output/'video_player_observations.jsonl').open('w') as canonical:
  for i,row in enumerate(rows):
   camera=row['camera'];now=row['timestamp'];current={mapping[lookup[p['observation_id']]]:p for p in features[i] if lookup[p['observation_id']] in confirmed_keys};players=[];unresolved=[]
   if len(current)!=sum(lookup[p['observation_id']] in confirmed_keys for p in features[i]):raise ValueError('Two visible tracks assigned one player')
   for p in features[i]:
    if lookup[p['observation_id']] in confirmed_keys:continue
    v=teams.classify(p['feature']);team=PersistentTeam(cfg).snapshot();team=team.model_copy(update={'current_visual_label':v['label'],'current_visual_quality':v['quality']});pos=p['pitch'];u=uncertainty(p);conf=p['detection_confidence']/(1+(u if u is not None else 10)) if pos is not None else None;prov=(p.get('projection') or {}).get('placement_provenance') or 'UNKNOWN' if pos is not None else 'UNKNOWN';unresolved.append(UnresolvedPlayerDetection(observation_id=p['observation_id'],track_id=p['track_id'],bbox=p['bbox'],team=team,current_position=pos,position_confidence=conf,position_provenance=prov));counters['visible_observations']+=1;counters['projected_observations']+=pos is not None;counters['team_UNKNOWN']+=1;canonical.write(VideoPlayerObservation(frame=i,timestamp=now,entity_id=None,track_id=p['track_id'],team='UNKNOWN',team_confidence=0,visible=True,visibility_state='UNRESOLVED_DETECTION',x=pos[0] if pos is not None else None,y=pos[1] if pos is not None else None,position_confidence=conf,position_provenance=prov,identity_confidence=0).model_dump_json()+'\n')
   for key,ss in entity_segments.items():
    if i<min(s.start for s in ss):continue
    p=current.get(key);old=last.get(key);relevant=[e for e in events if e['entity_id']==key and e['reappeared_at_frame']<=i];state='VISIBLE';reason='CURRENT_IMAGE_OBSERVATION';position=None;position_conf=None;provenance='UNKNOWN';u=None;velocity=old.last_seen_velocity if old else None;disappeared=old.last_disappearance_time if old else None;reappeared=old.last_reappearance_time if old else None
    if p:
     segment=next(s for s in ss if s.key==lookup[p['observation_id']]);current_visual=teams.classify(p['feature']);belief=beliefs[key].update(current_visual,i,now);teams.adapt(p['feature'],current_visual,belief.label,now,p['track_id']);position=p['pitch'];u=uncertainty(p);position_conf=float(p['detection_confidence']/(1+(u if u is not None else 10))) if position is not None else None;provenance=(p.get('projection') or {}).get('placement_provenance') or 'UNKNOWN' if position is not None else 'UNKNOWN';candidates=evidence[segment.key];identity='ASSOCIATED' if relevant else 'AMBIGUOUS' if segment.key in ambiguous else 'PROVISIONAL';identity_conf=relevant[-1]['evidence']['score'] if relevant else .35 if identity=='AMBIGUOUS' else .6
     if identity=='AMBIGUOUS':state='IDENTITY_AMBIGUOUS'
     if old and old.current_track_id is None:reappeared=now
     if position is not None and old and old.last_reliable_position is not None and old.last_reliable_timestamp is not None and u is not None and u<=cfg.max_metric_uncertainty_m and old.position_uncertainty_m is not None and old.position_uncertainty_m<=cfg.max_metric_uncertainty_m:
      dt=now-old.last_reliable_timestamp
      if dt>0:
       v=(np.array(position)-old.last_reliable_position)/dt
       if np.linalg.norm(v)<=cfg.max_speed_mps:velocity=tuple(v)
     if p['observation_id'] in ocr_records and ocr_records[p['observation_id']]['rear_view_verified']:
      r=ocr_records[p['observation_id']];jerseys[key].update(r['candidate'],r['confidence'],r['quality'],i,r['crop'])
     appearance=[{'frame':f,'source':'OSNET_MSMT17_HEAD_EXCLUDED','cache_key':f'{f}:{segment.track_id}','quality':segment.quality} for f in segment.evidence_frames[:8]]
    else:
     if old is None:continue
     segment=next(s for s in ss if s.key==old.current_segment_id) if old.current_segment_id else next(s for s in reversed(ss) if s.start<=i);belief=beliefs[key].snapshot(False);identity=old.identity_status;identity_conf=old.identity_confidence;candidates=evidence[segment.key];appearance=old.appearance_evidence
     if old.current_track_id is not None:disappeared=now
     state,reason=absence_state(camera,old.last_reliable_position,old.position_uncertainty_m,old.last_seen_time,old.camera_epoch,now,cfg)
    latest_position=position if position is not None else old.last_reliable_position if old else None;latest_time=now if position is not None else old.last_reliable_timestamp if old else None
    converted=[]
    for c in candidates[:5]:converted.append(c.model_copy(update={'entity_id':mapping[c.segment_id]}))
    person=PersistentPerson(entity_id=key,current_track_id=p['track_id'] if p else None,current_segment_id=segment.key,visibility_state=state,identity_status=identity,identity_confidence=identity_conf,team=belief,current_position=position,position_confidence=position_conf,position_uncertainty_m=u if p else old.position_uncertainty_m,position_provenance=provenance,camera_epoch=camera['epoch'] if p else old.camera_epoch,camera_state=camera['motion_evidence']['metric_camera']['state'],camera_source=camera['source'],last_reliable_position=latest_position,last_reliable_timestamp=latest_time,last_seen_time=now if p else old.last_seen_time,last_seen_velocity=velocity,last_disappearance_time=disappeared,last_reappearance_time=reappeared,visibility_reason=reason,appearance_confidence=segment.quality,current_appearance=p['feature'] if p else None,appearance_evidence=appearance,track_ids=list(dict.fromkeys(s.track_id for s in ss if s.start<=i)),segment_ids=[s.key for s in ss if s.start<=i],reid_candidates=converted,reassociation_events=relevant[-8:],jersey_number=jerseys[key].snapshot());players.append(person);last[key]=person
    timeline[key].append({'frame':i,'timestamp':now,'visibility':state,'track_id':person.current_track_id,'identity_confidence':identity_conf,'identity_status':identity,'team':belief.label,'probabilities':belief.probabilities,'team_source':belief.source,'visual_team':belief.current_visual_label,'current_position':position,'position_confidence':position_conf,'position_provenance':provenance,'camera_epoch':camera['epoch'],'appearance_confidence':segment.quality})
    counters['visible_observations']+=p is not None;counters['projected_observations']+=position is not None
    if p:counters['team_'+belief.label]+=1
    canonical.write(VideoPlayerObservation(frame=i,timestamp=now,entity_id=key,track_id=person.current_track_id,team=belief.label,team_confidence=belief.confidence,visible=p is not None,visibility_state=state,x=position[0] if position is not None else None,y=position[1] if position is not None else None,position_confidence=position_conf,position_provenance=provenance,identity_confidence=identity_conf).model_dump_json()+'\n')
   payload=PlayerWorldFrame(frame=i,timestamp=now,players=players,unresolved_detections=unresolved);row['player_world']=payload.model_dump();WorldState.model_validate(row);offsets.append(target.tell());target.write((json.dumps(row,separators=(',',':'))+'\n').encode())
 timing['world_state_construction_seconds']=time.perf_counter()-tick;timing['detection_seconds']=None;timing['short_term_tracking_seconds']=None;timing['detection_tracking_note']='Reused M8 raw observations; not rerun or timed as zero';timing['wall_seconds']=time.perf_counter()-started
 index={}
 for key,values in timeline.items():
  runs=[]
  for v in values:
   signature=(v['visibility'],v['track_id'],v['identity_status'])
   if runs and runs[-1]['signature']==list(signature):runs[-1]['end_frame']=v['frame']
   else:runs.append({'start_frame':v['frame'],'end_frame':v['frame'],'signature':list(signature)})
  index[key]={'runs':runs,'team_history':[v for j,v in enumerate(values) if j%3==0 or j==len(values)-1],'events':[e for e in events if e['entity_id']==key]}
 teams.diagnostic.update(final_centers_lab=teams.centers.tolist() if teams.centers is not None else None,conservative_prototype_updates=teams.adaptation_count);(output/'team_model.json').write_text(json.dumps(teams.diagnostic,indent=2));(output/'player_timelines.json').write_text(json.dumps(index));manifest={**m,'frame_offsets':offsets,'frames_sha256':digest(output/'frames.jsonl'),'player_model_version':'m9','player_algorithm_version':'m9-evidence-weighted-global-v1','team_model_version':'m9','source_m8_frames_sha256':m['frames_sha256'],'player_world_index':index,'player_world_config':cfg.model_dump(),'player_world_provenance':'Offline anonymous hypotheses; M8 cameras/raw observations retained; not independent identity accuracy'};manifest['review_notes']=m.get('review_notes',[])+['M9 identity and team scores are experimental, uncalibrated hypotheses. Officials are role hypotheses; no roster names or ball perception.']
 (output/'manifest.json').write_text(json.dumps(manifest));shutil.copyfile(baseline/'observations.jsonl',output/'observations.jsonl');(output/'diagnostics.json').write_text(json.dumps({'recording_id':name,'counts':dict(counters),'temporary_track_ids':len({s.track_id for s in all_segments}),'track_fragments':len(all_segments),'confirmed_track_fragments':len(segments),'unconfirmed_track_fragments':len(all_segments)-len(segments),'persistent_entities':len(entity_segments),'successful_reconnects':len(events),'ambiguous_reconnects':len(ambiguous),'candidate_counts':reidtime['candidate_counts'],'mean_fragments_per_entity':len(segments)/len(entity_segments),'team_model':teams.diagnostic,'team_events':{key:b.events for key,b in beliefs.items()},'timing':timing,'independent_identity_accuracy':None,'independent_team_accuracy':None,'camera_changed':False,'config':cfg.model_dump()},indent=2));print(name,dict(counters),'segments',len(segments),'entities',len(entity_segments),'reconnects',len(events),flush=True)
 return manifest

def run_ocr(m,segments,output):
 records={};engine=shutil.which('tesseract');cap=cv2.VideoCapture(m['source_video']);directory=output/'jersey_ocr';directory.mkdir()
 if not engine:(directory/'status.json').write_text(json.dumps({'status':'NOT_AVAILABLE','attempts':0}));return records
 for s in segments:
  eligible=[o for o in s.observations if o['bbox'][3]>=100 and o['feature'].get('sharpness',0)>=80 and o['feature']['quality']>=.6]
  for o in sorted(eligible,key=lambda p:-p['feature']['quality'])[:2]:
   cap.set(cv2.CAP_PROP_POS_FRAMES,o['frame']);ok,image=cap.read()
   if not ok:continue
   x,y,w,h=o['bbox'];crop=image[max(0,int(y+.22*h)):min(image.shape[0],int(y+.58*h)),max(0,int(x+.2*w)):min(image.shape[1],int(x+.8*w))]
   if not crop.size:continue
   path=directory/f'{o["frame"]}_{s.track_id}.png';cv2.imwrite(str(path),cv2.resize(crop,None,fx=4,fy=4,interpolation=cv2.INTER_CUBIC));r=subprocess.run([engine,str(path),'stdout','--psm','7','-c','tessedit_char_whitelist=0123456789','tsv'],capture_output=True,text=True,timeout=10);entries=list(csv.DictReader(io.StringIO(r.stdout),delimiter='\t'));valid=[e for e in entries if e.get('text','').strip().isdigit() and float(e.get('conf','-1'))>=75];best=max(valid,key=lambda e:float(e['conf'])) if valid else None;records[o['observation_id']]={'candidate':best['text'].strip() if best else None,'confidence':float(best['conf'])/100 if best else 0,'quality':o['feature']['quality'],'frame':o['frame'],'crop':str(path),'rear_view_verified':False,'reason':'NUMBER_CANDIDATE_ONLY; LOGOS/FRONT/BACK_AMBIGUOUS; NEVER_PRIMARY_IDENTITY'}
 cap.release();(directory/'attempts.json').write_text(json.dumps(records,indent=2));return records

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--recording',required=True);p.add_argument('--baseline');p.add_argument('--config');p.add_argument('--output',required=True);p.add_argument('--checkpoint',default='outputs/models/m9/osnet_x0_25_msmt17.pth');p.add_argument('--no-ocr',action='store_true');a=p.parse_args();run(a.recording,a.output,a.checkpoint,config=PlayerWorldConfig.model_validate_json(Path(a.config).read_text()) if a.config else None,ocr=not a.no_ocr,baseline=a.baseline)
