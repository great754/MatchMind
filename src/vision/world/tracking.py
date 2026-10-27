"""Small deterministic Hungarian image tracker; no motion prediction is emitted."""
import numpy as np
from scipy.optimize import linear_sum_assignment
from .schema import Config,TrackObservation,SizeEvidence

def similarity(a,b):
    if a is None or b is None:return None
    a=np.asarray(a);b=np.asarray(b)
    denom=np.linalg.norm(a)*np.linalg.norm(b)
    return float(np.clip(np.dot(a,b)/denom,0,1)) if denom>0 and a.shape==b.shape else None

def iou(a,b):
    x=max(a[0],b[0]);y=max(a[1],b[1]);r=min(a[0]+a[2],b[0]+b[2]);d=min(a[1]+a[3],b[1]+b[3])
    intersection=max(0,r-x)*max(0,d-y)
    return intersection/(a[2]*a[3]+b[2]*b[3]-intersection)

class ShortTermTracker:
    def __init__(self,config=None):
        self.config=config or Config();self.active={};self.next_id=1;self.last_timestamp=None
    def reset(self):self.active={}
    def update(self,detections,timestamp,image_size,cut=False):
        if self.last_timestamp is not None and timestamp<=self.last_timestamp:raise ValueError('Tracker clock must increase')
        self.last_timestamp=timestamp
        if cut:self.reset()
        self.active={k:v for k,v in self.active.items() if timestamp-v[1]<=self.config.track_grace_seconds}
        ids=sorted(self.active);cost=np.full((len(detections),len(ids)+len(detections)),1.)
        for i,d in enumerate(detections):
            for j,key in enumerate(ids):
                previous,seen=self.active[key];age=timestamp-seen
                predicted=tuple(np.array(previous.bbox)+[*(np.array(previous.image_velocity)*age),0,0]) if previous.image_velocity is not None else previous.bbox
                overlap=max(iou(d.bbox,previous.bbox),iou(d.bbox,predicted))
                distance=np.linalg.norm(np.array(d.foot)-previous.foot)/np.hypot(*image_size)
                appearance=similarity(d.appearance,previous.appearance)
                if overlap<self.config.min_iou or distance>self.config.max_image_motion_per_second*(timestamp-seen):continue
                if d.team and previous.team and d.team!=previous.team and (d.kit is None or previous.kit is None):continue
                cost[i,j]=.8*(1-overlap)+.2*(1-(appearance if appearance is not None else overlap))
        assignments=dict(zip(*linear_sum_assignment(cost))) if len(detections) else {}
        result=[]
        for i,d in enumerate(detections):
            j=assignments.get(i,len(ids));old=None
            if j<len(ids) and cost[i,j]<1: key=ids[j];old=self.active[key][0]
            else:key=f'TRACK_{self.next_id:05d}';self.next_id+=1
            velocity=tuple((np.array(d.foot)-old.foot)/(timestamp-self.active[key][1])) if old is not None else None
            track=TrackObservation(**d.model_dump(),track_id=key,hits=old.hits+1 if old else 1,image_velocity=velocity)
            if old is not None and (d.bbox[3]/old.bbox[3]>1.8 or d.bbox[3]/old.bbox[3]<.55):
                size=track.size_evidence or SizeEvidence()
                track=track.model_copy(update={'size_evidence':size.model_copy(update={'flags':size.flags+['BBOX_SIZE_JUMP'],'support':min(size.support,.7)})})
            self.active[key]=(track,timestamp);result.append(track)
        return result
