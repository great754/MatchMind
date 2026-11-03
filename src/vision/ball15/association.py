"""Interpretable candidate evidence and optional serialized logistic verifier."""
import math
import numpy as np
from src.vision.ball.schema import BallCandidate

def evidence(candidate,prediction=None,sigma=None,previous=None,people=(),width=1920,height=1080,age=0,competition=0):
 c=candidate if isinstance(candidate,BallCandidate) else BallCandidate.model_validate(candidate);point=np.array(c.center);bw,bh=c.bbox[2:];diagonal=math.hypot(width,height);distance=np.linalg.norm(point-prediction) if prediction is not None else 0.;normalized=distance/max(sigma or 12,4);direction=0.
 if previous is not None and prediction is not None:
  a=np.asarray(prediction)-previous;b=point-previous;direction=float(np.dot(a,b)/max(np.linalg.norm(a)*np.linalg.norm(b),1e-9))
 proximity=1.
 for p in people:
  b=p.get('bbox') or p.get('image_bbox')
  if b:proximity=min(proximity,float(np.linalg.norm(point-[b[0]+b[2]/2,b[1]+b[3]])/max(b[3],1)))
 return [c.confidence,min(normalized,20),direction,math.log(max(bw*bh,1))/math.log(max(width*height,2)),abs(math.log(bw/bh)),min(proximity,1),float(prediction is not None),min(age,1),min(competition,1)]

def logistic(features,model):
 x=(np.asarray(features)-np.asarray(model['mean']))/np.asarray(model['scale']);z=float(x@np.asarray(model['coef'])+model['intercept']);return 1/(1+math.exp(-max(-40,min(40,z))))

def score(features,model=None):
 if model is not None:return logistic(features,model)
 conf,distance,direction,size,aspect,near,history,age,competition=features
 visual=conf*math.exp(-.25*aspect)
 return float(.7*visual+.25*math.exp(-.5*distance**2)*history+.05*max(0,direction)*history)

def rank(candidates,prediction,sigma,previous,people,width,height,age,model=None):
 scores=[]
 for c in candidates:
  f=evidence(c,prediction,sigma,previous,people,width,height,age,max([x['confidence'] for x in candidates if x is not c],default=0));scores.append({'candidate_id':c['candidate_id'],'score':score(f,model),'features':f})
 return sorted(scores,key=lambda s:-s['score'])
