"""Offline tracklet graph with uncertainty-aware evidence and global path assignment."""
from dataclasses import dataclass,field
import math,time
import numpy as np
from scipy.optimize import linear_sum_assignment
from .player_appearance import cosine,unit
from .player_state import PlayerWorldConfig,ReIDCandidate

@dataclass
class ShortTrack:
 key:str
 track_id:str
 epoch:int
 observations:list=field(default_factory=list)
 embedding:object=None
 histogram:object=None
 quality:float=0
 team_label:str='UNKNOWN'
 team_strength:float=0
 evidence_frames:list=field(default_factory=list)
 @property
 def start(self):return self.observations[0]['frame']
 @property
 def end(self):return self.observations[-1]['frame']
 def summarize(self,embeddings):
  samples=[o for o in self.observations if o.get('embedding_key') in embeddings];self.embedding=unit(np.median([embeddings[o['embedding_key']] for o in samples],axis=0)) if samples else None;self.quality=float(np.mean([o['feature']['quality'] for o in samples])) if samples else 0;self.evidence_frames=[o['frame'] for o in samples]
  hist=[o['feature']['histogram'] for o in self.observations if o['feature'].get('histogram') and o['feature']['quality']>=.2];self.histogram=unit(np.median(hist,axis=0)) if hist else None
 def team_summary(self):
  weights={}
  for o in self.observations:
   v=o['visual_team']
   if v['label']!='UNKNOWN':weights[v['label']]=weights.get(v['label'],0)+v['quality']*v['strength']
  if weights:self.team_label=max(weights,key=weights.get);self.team_strength=weights[self.team_label]/(sum(weights.values())+1.)

def build_segments(features,rows,embeddings,fps=None):
 active={};segments=[];lookup={}
 for i,people in enumerate(features):
  epoch=rows[i]['camera']['epoch']
  for p in people:
   key=(epoch,p['track_id']);s=active.get(key)
   if s is None or i-s.end>(max(2,round(.6*fps)) if fps else 2):
    s=ShortTrack(f'{epoch}:{p["track_id"]}:{i}',p['track_id'],epoch);active[key]=s;segments.append(s)
   s.observations.append(p);lookup[p['observation_id']]=s.key
 for s in segments:s.summarize(embeddings)
 return segments,lookup

def uncertainty(o):
 p=o.get('projection') or {};value=p.get('uncertainty_pitch_m_proxy');return float(value) if value is not None and math.isfinite(value) else None

def candidate(previous,current,fps,config=None):
 cfg=config or PlayerWorldConfig();elapsed=(current.start-previous.end)/fps;app=cosine(previous.embedding,current.embedding);hist=cosine(previous.histogram,current.histogram);compatible=not(previous.team_label!='UNKNOWN' and current.team_label!='UNKNOWN' and previous.team_label!=current.team_label and min(previous.team_strength,current.team_strength)>=.8);a=previous.observations[-1];b=current.observations[0];u1=uncertainty(a);u2=uncertainty(b);u=math.hypot(u1,u2) if u1 is not None and u2 is not None else None;distance=speed=allowed=trajectory=None;metric=False;weak_metric=False;quality=min(previous.quality,current.quality)
 if elapsed>0 and a.get('pitch') is not None and b.get('pitch') is not None and u is not None:
  distance=float(np.linalg.norm(np.array(a['pitch'])-b['pitch']));speed=distance/elapsed;allowed=cfg.max_speed_mps*elapsed+cfg.movement_slack_m+2*u;metric=max(u1,u2)<=cfg.max_metric_uncertainty_m;weak_metric=not metric and max(u1,u2)<=8 and (app or 0)>=.90 and quality>=.25 and min(len(previous.evidence_frames),len(current.evidence_frames))>=3 and distance<=allowed
  if len(previous.observations)>3:
   early=previous.observations[max(0,len(previous.observations)-7)]
   if early.get('pitch') is not None and a['frame']>early['frame']:
    velocity=(np.array(a['pitch'])-early['pitch'])/((a['frame']-early['frame'])/fps);norm=np.linalg.norm(velocity)
    if norm>cfg.max_speed_mps:velocity*=cfg.max_speed_mps/norm
    trajectory=float(np.linalg.norm(np.array(b['pitch'])-(np.array(a['pitch'])+velocity*elapsed)))
 image_gap=float(np.linalg.norm(np.array(b['foot'])-a['foot'])/max(a['bbox'][3],1));shape=float(np.exp(-np.abs(np.log(np.array(b['bbox'][2:])/a['bbox'][2:])).mean()));image_continuity=0<elapsed<=.25 and image_gap<=.5 and shape>=.85
 spatial=math.exp(-distance/max(allowed or 1,1)) if metric or weak_metric else 0.;vel=math.exp(-trajectory/max(allowed or 1,1)) if (metric or weak_metric) and trajectory is not None else spatial
 position_weight=.35 if metric else .35/(1+(u/cfg.max_metric_uncertainty_m)**2) if weak_metric else 0
 if metric or weak_metric:score=(.8-position_weight)*(app or 0)+.1*(hist or 0)+position_weight*spatial+.1*vel
 else:score=.7*(app or 0)+.15*(hist or 0)+.15*math.exp(-image_gap)
 score*=min(1,.65+quality)*math.exp(-max(0,elapsed-cfg.long_gap_decay_start_seconds)/cfg.long_gap_decay_seconds);reason='EVIDENCE_COMPATIBLE';eligible=True
 if elapsed<=0:eligible=False;reason='SIMULTANEOUS_OR_OVERLAPPING_TRACKS'
 elif previous.epoch!=current.epoch:eligible=False;reason='CAMERA_EPOCH_DISCONTINUITY'
 elif elapsed>cfg.max_reconnect_seconds:eligible=False;reason='STALE_IDENTITY_SEARCH'
 elif not compatible:eligible=False;reason='STRONG_TEAM_CONFLICT'
 elif app is None or app<cfg.minimum_appearance or quality<.12:eligible=False;reason='WEAK_OR_MISSING_APPEARANCE'
 elif metric and distance>allowed:eligible=False;reason='PHYSICALLY_IMPLAUSIBLE_REASSOCIATION'
 elif not metric and not weak_metric and not image_continuity:eligible=False;reason='INSUFFICIENT_POSITION_OR_IMAGE_CONTINUITY'
 return ReIDCandidate(entity_id=previous.key,segment_id=previous.key,appearance_similarity=app,appearance_quality=quality,team_compatible=compatible,elapsed_seconds=elapsed,distance_m=distance,required_average_speed_mps=speed,allowed_distance_m=allowed,position_uncertainty_m=u,position_weight=position_weight,trajectory_error_m=trajectory,visibility_compatible=True if metric or weak_metric else None,score=float(np.clip(score,0,1)),eligible=eligible,reason=reason,evidence_frames=(previous.evidence_frames[-3:]+current.evidence_frames[:3]))

def globally_associate(segments,fps,config=None):
 cfg=config or PlayerWorldConfig();start=time.perf_counter();n=len(segments);cost=np.full((n,2*n),1-cfg.minimum_score);evidence={};ambiguous=set();candidate_counts={'eligible':0,'rejected':0};bykey={s.key:s for s in segments}
 for j,current in enumerate(segments):
  row=[]
  for i,previous in enumerate(segments):
   if previous.end>=current.start or previous.epoch!=current.epoch or (current.start-previous.end)/fps>cfg.max_reconnect_seconds:continue
   e=candidate(previous,current,fps,cfg);row.append((i,e));candidate_counts['eligible' if e.eligible else 'rejected']+=1
  evidence[current.key]=sorted([e for _,e in row],key=lambda x:-x.score)
  viable=sorted([(i,e) for i,e in row if e.eligible and e.score>=cfg.minimum_score],key=lambda x:-x[1].score)
  for i,e in viable:cost[i,j]=1-e.score
 reid_seconds=time.perf_counter()-start;tick=time.perf_counter();rr,cc=linear_sum_assignment(cost);optimum=float(cost[rr,cc].sum());predecessor={};events=[]
 for i,j in zip(rr,cc):
  if j>=n or cost[i,j]>=1-cfg.minimum_score:continue
  # Confidence is a global alternative-assignment margin, not a greedy row margin.
  alternative=cost.copy();alternative[i,j]=1-cfg.minimum_score;ar,ac=linear_sum_assignment(alternative);global_margin=float(alternative[ar,ac].sum()-optimum)
  if global_margin<cfg.ambiguity_margin:ambiguous.add(segments[j].key);continue
  previous,current=segments[i],segments[j];e=next(x for x in evidence[current.key] if x.segment_id==previous.key);predecessor[current.key]=previous.key;events.append({'kind':'REASSOCIATION_ACCEPTED','from_segment':previous.key,'to_segment':current.key,'source_track':previous.track_id,'target_track':current.track_id,'disappeared_at_frame':previous.end+1,'reappeared_at_frame':current.start,'resolved_at_frame':max(current.start,min(current.evidence_frames[:3][-1] if current.evidence_frames else current.end,current.end)),'offline_future_evidence':True,'global_assignment_margin':global_margin,'evidence':e.model_dump()})
 assignment_seconds=time.perf_counter()-tick;mapping={};next_id=1
 for s in sorted(segments,key=lambda s:(s.start,s.key)):
  previous=predecessor.get(s.key)
  if previous:mapping[s.key]=mapping[previous]
  else:mapping[s.key]=f'PLAYER_{next_id:03d}';next_id+=1
 # Acyclic, strictly nonoverlapping paths imply one visible segment per entity.
 for s in segments:
  for t in segments:
   if s.key<t.key and mapping[s.key]==mapping[t.key] and max(s.start,t.start)<=min(s.end,t.end):raise ValueError('Simultaneous duplicate persistent identity')
 return mapping,evidence,ambiguous,events,{'reid_candidate_seconds':reid_seconds,'global_assignment_seconds':assignment_seconds,'candidate_counts':candidate_counts}

def absence_state(camera,last_position,last_uncertainty,last_seen,last_epoch,now,config=None):
 """Visibility hypothesis from observed geometry; never extrapolate a current point."""
 cfg=config or PlayerWorldConfig()
 if last_epoch!=camera['epoch']:return 'LOST','CAMERA_EPOCH_DISCONTINUITY'
 region=camera.get('visible_pitch_region')
 if camera.get('calibration_valid') and region and len(region)>=3 and last_position is not None:
  import cv2
  distance=cv2.pointPolygonTest(np.asarray(region,np.float32),tuple(last_position),True)
  if distance<-(last_uncertainty or 0)-cfg.movement_slack_m:return 'OFF_CAMERA','LAST_OBSERVED_AREA_OUTSIDE_CURRENT_CAMERA_FOOTPRINT; CURRENT_POSITION_UNKNOWN'
 if now-last_seen>cfg.lost_after_seconds:return 'LOST','LONG_UNOBSERVED_GAP; CURRENT_POSITION_UNKNOWN'
 return 'OCCLUDED','DETECTION_MISSING_INSIDE_OR_WITH_UNKNOWN_CAMERA_FOOTPRINT; OCCLUSION_UNCONFIRMED'
