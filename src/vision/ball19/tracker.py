"""One-to-one visual association with one temporal support gate, not two stacked gates."""
import math
from dataclasses import dataclass
import numpy as np
from scipy.optimize import linear_sum_assignment
from src.vision.ball.schema import BallCandidate,BallFrame
from src.vision.ball.geometry import camera_motion,warp_point

@dataclass(frozen=True)
class AssociationConfig:
    minimum_birth_hits:int=2
    maximum_hypotheses:int=4
    prediction_seconds:float=.35
    ambiguity_margin:float=.12
    max_image_diagonals_per_second:float=2.
    movement_slack_px:float=12.

class VisualBallTracker:
    def __init__(self,width,height,threshold,config=None):
        if not 0<threshold<=1:raise ValueError('Use the frozen detector operating threshold')
        self.width=width;self.height=height;self.threshold=threshold;self.cfg=config or AssociationConfig()
        if self.cfg.minimum_birth_hits<2:raise ValueError('Repeated visual birth evidence is mandatory')
        if self.cfg.maximum_hypotheses<1 or not 0<self.cfg.prediction_seconds<=.35 or not 0<=self.cfg.ambiguity_margin<=1 or self.cfg.max_image_diagonals_per_second<=0 or self.cfg.movement_slack_px<0:raise ValueError('Invalid association bounds')
        self.diagonal=math.hypot(width,height);self.hypotheses=[];self.next_id=0
        self.previous_camera=None;self.previous_time=None;self.last_selected=None;self.last_confirmed=None;self.last_confirmed_time=None
    def step(self,frame,timestamp,candidates,camera,people=(),*,association_priors=None,association_cost=None):
        cs=[BallCandidate.model_validate(c) for c in candidates]
        if any(c.bbox[0]+c.bbox[2]>self.width+1e-5 or c.bbox[1]+c.bbox[3]>self.height+1e-5 for c in cs):raise ValueError('Ball box outside source')
        if self.previous_time is not None and timestamp<=self.previous_time:raise ValueError('Ball clock must increase')
        dt=0 if self.previous_time is None else timestamp-self.previous_time
        cut=self.previous_camera is not None and (camera['epoch']!=self.previous_camera['epoch'] or camera.get('continuity')=='cut')
        motion,motion_source=camera_motion(self.previous_camera,camera)
        if cut:self.hypotheses=[];self.last_selected=None;self.last_confirmed=None;self.last_confirmed_time=None
        self.previous_camera=camera;self.previous_time=timestamp
        self.hypotheses=[h for h in self.hypotheses if timestamp-h['observed_at']<=self.cfg.prediction_seconds]
        viable=[c for c in cs if c.confidence>=self.threshold]
        for h in self.hypotheses:
            if motion is not None:
                p=warp_point(motion,h['point']);q=warp_point(motion,h['point']+h['velocity']*max(dt,.001))
                if p is not None and q is not None:
                    h['point']=p;h['velocity']=(q-p)/max(dt,.001)
                    observed=warp_point(motion,h['last_observed_warped'])
                    if observed is not None:h['last_observed_warped']=observed
            h['prior']=h['point'].copy();h['point']=h['point']+h['velocity']*dt
            h['sigma']+=self.diagonal*.025*dt;h['candidate']=None
            # Optional M20 motion priors; the default M19 path is unchanged.
            if association_priors is not None and h['id'] in association_priors:
                point,sigma=association_priors[h['id']]
                if not np.isfinite(point).all() or not math.isfinite(sigma) or sigma<=0:raise ValueError('Invalid external association prior')
                h['point']=np.asarray(point,float);h['sigma']=sigma
        costs=np.full((len(self.hypotheses),len(viable)),1e6)
        for i,h in enumerate(self.hypotheses):
            age=timestamp-h['observed_at'];gate=self.cfg.movement_slack_px+3*h['sigma']+self.diagonal*self.cfg.max_image_diagonals_per_second*max(dt,.001)
            for j,c in enumerate(viable):
                error=float(np.linalg.norm(np.asarray(c.center)-h['point']))
                if error<=gate:
                    cost=error/max(gate,1)+.5*(1-c.confidence)
                    costs[i,j]=cost if association_cost is None else association_cost(h,c,cost)
                    if not math.isfinite(costs[i,j]) or costs[i,j]<0:raise ValueError('Association cost must be finite and nonnegative')
        used=set()
        if costs.size:
            for i,j in zip(*linear_sum_assignment(costs)):
                if costs[i,j]>=1e6:continue
                h=self.hypotheses[int(i)];c=viable[int(j)];z=np.asarray(c.center,float);age=timestamp-h['observed_at']
                # The previous posterior is already camera-warped; velocity uses actual elapsed time.
                measured=(z-h['last_observed_warped'])/max(age,.001) if motion is None else (z-h['prior'])/max(dt,.001)
                limit=self.diagonal*self.cfg.max_image_diagonals_per_second
                if np.linalg.norm(measured)>limit:measured*=limit/np.linalg.norm(measured)
                h['velocity']=.5*h['velocity']+.5*measured
                h['hits']=h['hits']+1 if h['confirmed'] or h['last_frame']==frame-1 else 1
                h['confirmed']=h['confirmed'] or h['hits']>=self.cfg.minimum_birth_hits
                h['weight']=.5*h['weight']+.5*c.confidence;h['point']=z;h['last_observed_warped']=z.copy();h['observed_at']=timestamp;h['last_frame']=frame;h['candidate']=c;h['sigma']=max(4,.8*h['sigma']);used.add(int(j))
        for j,c in enumerate(viable):
            if j in used:continue
            self.next_id+=1;z=np.asarray(c.center,float)
            self.hypotheses.append({'id':self.next_id,'point':z,'last_observed_warped':z.copy(),'velocity':np.zeros(2),'sigma':8.,'observed_at':timestamp,'hits':1,'confirmed':False,'weight':c.confidence,'last_frame':frame,'candidate':c})
        eligible=sorted([h for h in self.hypotheses if h['candidate'] is not None and h['confirmed']],key=lambda h:(-h['weight'],h['id']))
        selected=None;ambiguous=False
        if eligible:
            if len(eligible)==1 or eligible[0]['weight']-eligible[1]['weight']>=self.cfg.ambiguity_margin:selected=eligible[0]
            else:ambiguous=True
        self.hypotheses=sorted(self.hypotheses,key=lambda h:(not h['confirmed'],-h['weight']*math.exp(-(timestamp-h['observed_at'])/.2),h['id']))[:self.cfg.maximum_hypotheses]
        common={'version':'m19','frame':frame,'timestamp':timestamp,'camera_epoch':camera['epoch'],'candidates':cs,'candidate_count':len(cs),'detection_status':'DETECTED' if viable else 'MISSING','last_confirmed_image':self.last_confirmed,'last_confirmed_timestamp':self.last_confirmed_time,'camera_compensation':motion_source,'motion_model':'M19_ONE_TO_ONE_VISUAL_ASSOCIATION','search_stage':'FROZEN_GLOBAL_CANDIDATES','coordinate_state':'METRIC_UNAVAILABLE','coordinate_provenance':'M19_NO_VERIFIED_GROUND_HEIGHT','track_id':self.last_selected}
        if selected is not None:
            c=selected['candidate'];self.last_selected=selected['id'];self.last_confirmed=c.center;self.last_confirmed_time=timestamp
            common.update(track_id=selected['id'],last_confirmed_image=c.center,last_confirmed_timestamp=timestamp)
            return BallFrame(**common,tracking_status='TRACKED',directly_observed=True,image_position=c.center,image_bbox=c.bbox,detection_confidence=c.confidence,tracking_confidence=c.confidence,estimated_velocity=tuple(map(float,selected['velocity'])),velocity_provenance='M19_IMAGE_PX_PER_SECOND; NOT_METRIC',reason='ONE_TO_ONE_ASSIGNMENT_WITH_REPEATED_VISUAL_SUPPORT')
        if ambiguous:return BallFrame(**{**common,'detection_status':'AMBIGUOUS'},tracking_status='AMBIGUOUS',ambiguity_reason='M19_COMPETING_VISUALLY_SUPPORTED_TRAJECTORIES',reason='NO_FORCED_SELECTION')
        active=next((h for h in self.hypotheses if h['id']==self.last_selected),None)
        if active is not None and timestamp-active['observed_at']<=self.cfg.prediction_seconds and 0<=active['point'][0]<self.width and 0<=active['point'][1]<self.height:
            overlap=any((box:=p.get('bbox')) and box[0]<=active['point'][0]<=box[0]+box[2] and box[1]<=active['point'][1]<=box[1]+box[3] for p in people)
            return BallFrame(**common,tracking_status='OCCLUDED' if overlap else 'PREDICTED',predicted_image_position=tuple(map(float,active['point'])),prediction_age_seconds=timestamp-active['observed_at'],prediction_uncertainty_px=active['sigma'],reason='BOUNDED_PREDICTION_ONLY; NOT_A_VISUAL_OBSERVATION')
        return BallFrame(**common,tracking_status='DETECTED' if viable else 'LOST',reason='AWAIT_REPEATED_VISUAL_SUPPORT' if viable else 'NO_CURRENT_VISUAL_EVIDENCE')
