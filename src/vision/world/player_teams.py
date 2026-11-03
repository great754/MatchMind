"""Video-learned kit populations and sticky five-way entity beliefs (uncalibrated)."""
from dataclasses import dataclass,field
import numpy as np
from .player_state import PlayerWorldConfig,TeamIntelligence

LABELS=['TEAM_A','TEAM_B','OFFICIAL','OTHER','UNKNOWN']
class TeamAppearance:
 def __init__(self,centers=None,labels=None,diagnostic=None):self.centers=centers;self.labels=labels or [];self.diagnostic=diagnostic or {};self.adapted_at={};self.adaptation_count=0
 @classmethod
 def fit(cls,segments):
  from sklearn.cluster import KMeans
  profiles=[];owners=[]
  for s in segments:
   samples=[o for o in s.observations if o['feature'].get('lab') is not None and o['feature']['quality']>=.25]
   if len(samples)>=3:profiles.append(np.median([o['feature']['lab'] for o in samples],axis=0));owners.append(s)
  if len(profiles)<6:return cls(diagnostic={'status':'INSUFFICIENT_INDEPENDENT_SEGMENTS','profiles':len(profiles)})
  p=np.array(profiles);scale=np.array([.7,1.,1.]);k=min(3,len(p));model=KMeans(n_clusters=k,n_init=20,random_state=9).fit(p*scale)
  # Split an extra kit mode only when a third broad population is present.
  supports=[sum(len(s.observations) for s,c in zip(owners,model.labels_) if c==i) for i in range(k)]
  minor=int(np.argmin(supports));concurrent={}
  for s,c in zip(owners,model.labels_):
   if c==minor:
    for o in s.observations:concurrent[o['frame']]=concurrent.get(o['frame'],0)+1
  if len(p)>=12 and concurrent and np.mean(list(concurrent.values()))>=1.5:k=4;model=KMeans(n_clusters=k,n_init=20,random_state=9).fit(p*scale)
  centers=np.array([np.median(p[model.labels_==i],axis=0) for i in range(k)]);support=[sum(len(s.observations) for s,c in zip(owners,model.labels_) if c==i) for i in range(k)];order=sorted(range(k),key=lambda i:-support[i]);main=sorted(order[:2],key=lambda i:tuple(centers[i]));labels=['OTHER']*k
  for i,label in zip(main,['TEAM_A','TEAM_B']):labels[i]=label
  others=[]
  for i in order[2:]:
   assigned=[s for s,c in zip(owners,model.labels_) if c==i];concurrency={};positions=[]
   for s in assigned:
    for o in s.observations:
     concurrency[o['frame']]=concurrency.get(o['frame'],0)+1
     if o.get('pitch') is not None:positions.append(o['pitch'])
   # A rare, isolated kit in central play is an official hypothesis, not referee identity.
   isolated=np.mean(list(concurrency.values()))<1.5;central=bool(positions) and np.mean([(18<q[0]<87 and 7<q[1]<61) for q in positions])>=.6;separation=min(np.linalg.norm((centers[i]-centers[j])*scale) for j in main)
   if isolated and central and separation>=30:labels[i]='OFFICIAL'
   elif not isolated:
    distances=sorted((np.linalg.norm((centers[i]-centers[j])*scale),j) for j in main)
    if distances[0][0]<.65*distances[1][0]:labels[i]=labels[distances[0][1]]
   others.append({'cluster':i,'isolated':bool(isolated),'central_play':bool(central),'separation':float(separation),'role_is_hypothesis':True})
  if np.linalg.norm((centers[main[0]]-centers[main[1]])*scale)<22:return cls(diagnostic={'status':'INSUFFICIENT_DISTINCT_OUTFIELD_POPULATIONS','centers_lab':centers.tolist()})
  return cls(centers,labels,{'status':'OFFLINE_VIDEO_LEARNED_PROVISIONAL_POPULATIONS','profiles':len(profiles),'centers_lab':centers.tolist(),'labels':labels,'observation_support':support,'minority_role_evidence':others,'colors_are_not_fixed':True,'calibrated':False})
 def adapt(self,feature,visual,established,timestamp,track):
  if self.centers is None or established=='UNKNOWN' or visual['label']!=established or visual.get('quality',0)<.65 or visual.get('strength',0)<.6 or timestamp-self.adapted_at.get(track,-100)<.5:return
  candidates=[i for i,label in enumerate(self.labels) if label==established]
  if not candidates:return
  i=min(candidates,key=lambda i:np.linalg.norm(self.centers[i]-feature['lab']));self.centers[i]+=.005*np.clip(np.asarray(feature['lab'])-self.centers[i],-10,10);self.adapted_at[track]=timestamp;self.adaptation_count+=1
 def classify(self,feature):
  q=feature.get('quality',0)
  if self.centers is None or feature.get('lab') is None or q<.18:return {'label':'UNKNOWN','quality':q,'strength':0.,'reason':'WEAK_FOREGROUND_EVIDENCE','distances':[]}
  d=np.linalg.norm((self.centers-feature['lab'])*[.7,1,1],axis=1);order=np.argsort(d);margin=float((d[order[1]]-d[order[0]])/max(d[order[1]],1));label=self.labels[order[0]] if d[order[0]]<=48 and margin>=.18 else 'UNKNOWN';strength=float(np.clip(margin*(.5+.5*q),0,1));return {'label':label,'quality':q,'strength':strength,'reason':'VIDEO_KIT_POPULATION' if label!='UNKNOWN' else 'AMBIGUOUS_KIT_POPULATIONS','distances':d.tolist(),'margin':margin,'lab':feature['lab']}

@dataclass
class PersistentTeam:
 config:PlayerWorldConfig=field(default_factory=PlayerWorldConfig)
 label:str='UNKNOWN'
 votes:dict=field(default_factory=lambda:{k:.2 for k in LABELS})
 samples:int=0
 last_sample:float|None=None
 candidate:str|None=None
 candidate_times:list=field(default_factory=list)
 events:list=field(default_factory=list)
 last_visual:dict=field(default_factory=lambda:{'label':'UNKNOWN','quality':0})
 def update(self,visual,frame,timestamp):
  self.last_visual=visual;observed=visual['label'];strong=observed!='UNKNOWN' and visual.get('quality',0)>=.25 and visual.get('strength',0)>=.30;independent=strong and (self.last_sample is None or timestamp-self.last_sample>=self.config.team_sample_seconds-1e-8)
  if independent:
   self.last_sample=timestamp;self.samples+=1;self.votes[observed]+=max(.4,visual['strength']);total=sum(self.votes.values());winner=max(self.votes,key=self.votes.get)
   if self.label=='UNKNOWN' and self.samples>=self.config.team_min_samples and self.votes[winner]/total>=.65:self.label=winner;self.events.append({'kind':'TEAM_ESTABLISHED','frame':frame,'timestamp':timestamp,'label':winner,'independent_samples':self.samples})
   elif self.label!='UNKNOWN' and observed!=self.label and visual['quality']>=self.config.team_switch_quality and visual['strength']>=self.config.team_switch_strength:
    if self.candidate!=observed:
     if self.candidate:self.events.append({'kind':'TEAM_SWITCH_REJECTED','frame':frame,'timestamp':timestamp,'reason':'CONTRADICTION_CHANGED','candidate':self.candidate})
     self.candidate=observed;self.candidate_times=[];self.events.append({'kind':'TEAM_SWITCH_CANDIDATE','frame':frame,'timestamp':timestamp,'from':self.label,'candidate':observed})
    self.candidate_times=[t for t in self.candidate_times if timestamp-t<=2.]+[timestamp]
    if len(self.candidate_times)>=self.config.team_switch_samples and timestamp-self.candidate_times[0]>=self.config.team_switch_span_seconds:
     old=self.label;self.label=observed;self.votes={k:.2 for k in LABELS};self.votes[observed]=len(self.candidate_times);self.events.append({'kind':'TEAM_SWITCH_ACCEPTED','frame':frame,'timestamp':timestamp,'from':old,'label':observed,'samples':len(self.candidate_times),'reason':'SUSTAINED_INDEPENDENT_STRONG_CONTRADICTION'});self.candidate=None;self.candidate_times=[]
   elif observed==self.label and self.candidate:
    self.events.append({'kind':'TEAM_SWITCH_REJECTED','frame':frame,'timestamp':timestamp,'candidate':self.candidate,'reason':'ESTABLISHED_TEAM_REOBSERVED'});self.candidate=None;self.candidate_times=[]
  return self.snapshot()
 def snapshot(self,visible=True):
  total=sum(self.votes.values());p={k:v/total for k,v in self.votes.items()};confidence=p[self.label] if self.label!='UNKNOWN' else p['UNKNOWN'];v=self.last_visual if visible else {'label':'UNKNOWN','quality':0};source='CURRENT_VISUAL_AND_HISTORY' if self.label==v['label'] and self.label!='UNKNOWN' else 'PERSISTENT_TEAM_HISTORY' if self.label!='UNKNOWN' else 'INSUFFICIENT_TEAM_EVIDENCE'
  return TeamIntelligence(label=self.label,confidence=confidence,probabilities=p,source=source,current_visual_label=v['label'],current_visual_quality=v.get('quality',0),independent_samples=self.samples,switch_candidate=self.candidate,events=self.events[-8:])

@dataclass
class JerseyBelief:
 """Optional OCR evidence accumulator; bad/single crops never establish a number."""
 samples:list=field(default_factory=list)
 def update(self,number,confidence,quality,frame,crop=None):
  if number is None or not str(number).isdigit() or not 1<=int(number)<=99 or confidence<.75 or quality<.6:return self.snapshot()
  if any(x['frame']==frame for x in self.samples):return self.snapshot()
  self.samples.append({'number':str(number),'confidence':confidence,'quality':quality,'frame':frame,'crop':crop});return self.snapshot()
 def snapshot(self):
  counts={}
  for x in self.samples:counts[x['number']]=counts.get(x['number'],0)+1
  winner=max(counts,key=counts.get) if counts else None
  return {'number':winner if winner and counts[winner]>=3 else None,'samples':self.samples,'source':'OPTIONAL_OCR_EVIDENCE_ONLY','established':bool(winner and counts[winner]>=3)}
